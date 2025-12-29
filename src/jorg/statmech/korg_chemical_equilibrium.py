"""
Complete Korg.jl Chemical Equilibrium Solver
============================================

Direct port of Korg.jl's chemical_equilibrium function from src/statmech.jl:120-343

This is the COMPLETE solver that simultaneously solves:
- Element conservation equations (92 elements)
- Charge neutrality constraint
- Saha ionization equilibrium
- Molecular equilibrium constants

Source: Korg.jl/src/statmech.jl:120-343
"""

import numpy as np
from scipy.optimize import root
from typing import Dict, Tuple, Callable
import warnings

from .species import Species, MAX_ATOMIC_NUMBER
from ..constants import kboltz_eV, kboltz_cgs, me_cgs, hplanck_cgs


def translational_U(mass_cgs: float, temperature: float) -> float:
    """
    Translational partition function contribution for free particles

    Source: Korg.jl/src/statmech.jl:48-52

    Parameters
    ----------
    mass_cgs : float
        Particle mass in grams
    temperature : float
        Temperature in K

    Returns
    -------
    float
        Translational partition function factor
    """
    return (2 * np.pi * mass_cgs * kboltz_cgs * temperature / hplanck_cgs**2)**1.5


def saha_ion_weights(temperature: float, ne: float, atomic_number: int,
                     ionization_energies: Dict[int, Tuple[float, float, float]],
                     partition_funcs: Dict[Species, Callable]) -> Tuple[float, float]:
    """
    Calculate Saha ionization weights for an element

    Returns (wII, wIII) where wII = n(X II)/n(X I) and wIII = n(X III)/n(X I)

    Source: Korg.jl/src/statmech.jl:18-35

    Parameters
    ----------
    temperature : float
        Temperature in K
    ne : float
        Electron number density in cm^-3
    atomic_number : int
        Atomic number Z
    ionization_energies : Dict
        Ionization energies in eV
    partition_funcs : Dict
        Partition function interpolators

    Returns
    -------
    Tuple[float, float]
        (wII, wIII) ionization weight factors
    """
    chi_I, chi_II, chi_III = ionization_energies[atomic_number]

    log_T = np.log(temperature)

    # Get partition functions (interpolators return U, not log(U))
    species_I = Species.from_atomic_number(atomic_number, 0)
    species_II = Species.from_atomic_number(atomic_number, 1)

    U_I = float(partition_funcs[species_I](log_T))
    U_II = float(partition_funcs[species_II](log_T))
    log_U_I = np.log(max(U_I, 1e-300))
    log_U_II = np.log(max(U_II, 1e-300))

    k = kboltz_eV
    trans_U = translational_U(me_cgs, temperature)

    # Saha equation for first ionization (in log space to avoid overflow)
    # wII = (n_II / n_I) = (2 * U_II / U_I) * (trans_U / ne) * exp(-χ_I / kT)
    # log(wII) = log(2) + log_U_II - log_U_I + log(trans_U) - log(ne) - χ_I/(kT)
    log_wII = (np.log(2.0) + log_U_II - log_U_I + np.log(trans_U) -
               np.log(ne) - chi_I / (k * temperature))
    wII = np.exp(log_wII)

    # Second ionization (skip for hydrogen)
    if atomic_number == 1:
        wIII = 0.0
    else:
        species_III = Species.from_atomic_number(atomic_number, 2)
        if species_III in partition_funcs:
            U_III = float(partition_funcs[species_III](log_T))
            log_U_III = np.log(max(U_III, 1e-300))
            # wIII = wII * (2 * U_III / U_II) * (trans_U / ne) * exp(-χ_II / kT)
            log_wIII = (log_wII + np.log(2.0) + log_U_III - log_U_II +
                       np.log(trans_U) - np.log(ne) - chi_II / (k * temperature))
            wIII = np.exp(log_wIII)
        else:
            wIII = 0.0

    return wII, wIII


def setup_chemical_equilibrium_residuals(temperature: float, n_total: float,
                                        absolute_abundances: np.ndarray,
                                        ionization_energies: Dict[int, Tuple],
                                        partition_funcs: Dict[Species, Callable],
                                        log_equilibrium_constants: Dict = None):
    """
    Setup residual function for chemical equilibrium nonlinear system

    Source: Korg.jl/src/statmech.jl:272-343

    The system of equations:
    - x[0:92] = neutral fraction for each element
    - x[92] = electron density / n_total * 1e5 (scaled for numerical stability)

    Residuals:
    - F[0:92] = element conservation equations
    - F[92] = charge neutrality equation

    Parameters
    ----------
    temperature : float
        Temperature in K
    n_total : float
        Total number density in cm^-3
    absolute_abundances : np.ndarray
        Element abundances (92 elements)
    ionization_energies : Dict
        Ionization energies
    partition_funcs : Dict
        Partition functions
    log_equilibrium_constants : Dict, optional
        Molecular equilibrium constants

    Returns
    -------
    Callable
        Residual function for scipy.optimize.root
    """

    # Precompute Saha weights with ne=1 (will scale later)
    wII_ne = np.zeros(MAX_ATOMIC_NUMBER)
    wIII_ne2 = np.zeros(MAX_ATOMIC_NUMBER)

    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        if Z in ionization_energies:
            wII, wIII = saha_ion_weights(temperature, 1.0, Z, ionization_energies, partition_funcs)
            wII_ne[Z-1] = wII
            wIII_ne2[Z-1] = wIII

    def residuals(x):
        """
        Residual function matching Korg.jl/src/statmech.jl:291-342
        """
        # Extract electron density (scaled by 1e-5 for numerical stability)
        ne = abs(x[-1]) * n_total * 1e-5

        # Extract neutral fractions and compute number densities
        neutral_fractions = np.abs(x[:-1])  # abs() prevents negative densities
        atom_number_densities = absolute_abundances * (n_total - ne)
        neutral_number_densities = atom_number_densities * neutral_fractions

        # Initialize residuals
        F = np.zeros_like(x)

        # Element conservation and charge neutrality
        # Source: Korg.jl/src/statmech.jl:303-312
        for Z in range(1, MAX_ATOMIC_NUMBER + 1):
            wII = wII_ne[Z-1] / ne
            wIII = wIII_ne2[Z-1] / (ne * ne)

            # Element conservation: n(X_total) = n(X I) + n(X II) + n(X III)
            # Residual: n(X_total) - (1 + wII + wIII) * n(X I)
            F[Z-1] = atom_number_densities[Z-1] - (1 + wII + wIII) * neutral_number_densities[Z-1]

            # Charge neutrality contribution from this element
            # Each X II contributes 1 electron, each X III contributes 2 electrons
            F[-1] += (wII + 2 * wIII) * neutral_number_densities[Z-1]

        # Charge neutrality: Σ electrons from ions = ne
        # Source: Korg.jl/src/statmech.jl:312
        F[-1] -= ne

        # Add molecular contributions if provided
        # Source: Korg.jl/src/statmech.jl:314-337
        if log_equilibrium_constants is not None:
            # Convert to log10 space for stability (matches Korg.jl)
            log_neutral_densities = np.log10(np.maximum(neutral_number_densities, 1e-100))

            for mol_species, log_K_func in log_equilibrium_constants.items():
                try:
                    log_T = np.log(temperature)
                    log_K_partial_pressure = log_K_func(log_T)  # log10(K_p)

                    # Convert from partial pressure to number density form in log10
                    # log10(K_n) = log10(K_p) - (n_atoms - 1) * log10(kT)
                    n_atoms = len(mol_species.formula.atoms)
                    log_nK = log_K_partial_pressure - (n_atoms - 1) * np.log10(kboltz_cgs * temperature)

                    # Get constituent atoms
                    atoms = list(mol_species.get_atoms())

                    if mol_species.charge == 1:  # Charged diatomic
                        # First atom is ionized, second is neutral
                        Z1, Z2 = atoms[0], atoms[1]
                        wII_1 = wII_ne[Z1-1] / ne

                        # n_mol = n(Z1 II) * n(Z2 I) / K
                        log_n1_II = log_neutral_densities[Z1-1] + np.log10(wII_1)
                        log_n2_I = log_neutral_densities[Z2-1]
                        log_n_mol = log_n1_II + log_n2_I - log_nK
                        n_mol = 10**log_n_mol

                        # Subtract molecules from element conservation
                        F[Z1-1] -= n_mol
                        F[Z2-1] -= n_mol
                        # Add electron from ionized molecule
                        F[-1] += n_mol

                    else:  # Neutral molecule
                        # n_mol = Π n(atoms) / K
                        log_n_mol = sum(log_neutral_densities[Z-1] for Z in atoms) - log_nK
                        n_mol = 10**log_n_mol

                        # Subtract molecules from each constituent element
                        for Z in atoms:
                            F[Z-1] -= n_mol

                except (KeyError, IndexError, ValueError):
                    continue  # Skip problematic molecules

        # Normalize residuals for numerical stability
        # Source: Korg.jl/src/statmech.jl:339-340
        F[:-1] /= np.maximum(atom_number_densities, 1e-100)
        F[-1] /= (ne * 1e-5)

        return F

    return residuals


def chemical_equilibrium(temp: float, nt: float, model_atm_ne: float,
                        absolute_abundances: Dict[int, float],
                        ionization_energies: Dict[int, Tuple],
                        partition_funcs: Dict[Species, Callable],
                        log_equilibrium_constants: Dict = None,
                        electron_number_density_warn_threshold: float = 0.1,
                        **kwargs) -> Tuple[float, Dict[Species, float]]:
    """
    Complete chemical equilibrium solver matching Korg.jl

    Source: Korg.jl/src/statmech.jl:120-165

    Solves the complete system of chemical equilibrium equations including:
    - Element conservation for all 92 elements
    - Charge neutrality constraint
    - Saha ionization equilibrium
    - Molecular equilibrium constants

    Parameters
    ----------
    temperature : float
        Temperature in K
    n_total : float
        Total number density in cm^-3
    model_atm_ne : float
        Model atmosphere electron density (initial guess) in cm^-3
    absolute_abundances : Dict[int, float]
        Element abundances N_X/N_total
    ionization_energies : Dict
        Ionization energies in eV
    partition_funcs : Dict[Species, Callable]
        Partition function interpolators
    log_equilibrium_constants : Dict, optional
        Molecular equilibrium constants
    electron_number_density_warn_threshold : float
        Warning threshold for ne discrepancy

    Returns
    -------
    Tuple[float, Dict[Species, float]]
        (electron_density, species_densities)
    """

    # Rename parameters to match internal variable names
    temperature = temp
    n_total = nt

    # Convert abundances dict to array
    abs_abund_array = np.zeros(MAX_ATOMIC_NUMBER)
    for Z, abund in absolute_abundances.items():
        if 1 <= Z <= MAX_ATOMIC_NUMBER:
            abs_abund_array[Z-1] = abund

    # Compute initial guess by neglecting molecules
    # Source: Korg.jl/src/statmech.jl:124-128
    neutral_fraction_guess = np.zeros(MAX_ATOMIC_NUMBER)
    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        if Z in ionization_energies:
            wII, wIII = saha_ion_weights(temperature, model_atm_ne, Z, ionization_energies, partition_funcs)
            neutral_fraction_guess[Z-1] = 1.0 / (1.0 + wII + wIII)

    # Initial guess: [neutral_fractions, ne/n_total*1e5]
    x0 = np.concatenate([neutral_fraction_guess, [model_atm_ne / n_total * 1e5]])

    # Setup residual function
    residuals_func = setup_chemical_equilibrium_residuals(
        temperature, n_total, abs_abund_array,
        ionization_energies, partition_funcs, log_equilibrium_constants
    )

    # Solve nonlinear system
    # Source: Korg.jl/src/statmech.jl:192-205
    try:
        sol = root(residuals_func, x0, method='hybr', options={'xtol': 1e-8, 'maxfev': 1000})

        if not sol.success:
            # Try again with very small ne guess (Korg.jl fallback)
            x0[-1] = 1e-5
            sol = root(residuals_func, x0, method='hybr', options={'xtol': 1e-8, 'maxfev': 1000})

            if not sol.success:
                raise RuntimeError(f"Chemical equilibrium solver failed: {sol.message}")
    except Exception as e:
        raise RuntimeError(f"Chemical equilibrium solver failed: {e}")

    # Extract solution
    neutral_fractions = np.abs(sol.x[:-1])
    ne = abs(sol.x[-1]) * n_total * 1e-5

    # Check convergence warning
    if ((ne / n_total > 1e-4) and
        (abs((ne - model_atm_ne) / model_atm_ne) > electron_number_density_warn_threshold)):
        warnings.warn(
            f"Electron number density differs from model atmosphere by "
            f"{abs((ne - model_atm_ne) / model_atm_ne)*100:.1f}% "
            f"(calculated ne = {ne:.3e}, model atmosphere ne = {model_atm_ne:.3e})"
        )

    # Build species densities dict
    # Source: Korg.jl/src/statmech.jl:141-162
    species_densities = {}

    # Neutral atomic species
    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        species = Species.from_atomic_number(Z, 0)
        species_densities[species] = (n_total - ne) * abs_abund_array[Z-1] * neutral_fractions[Z-1]

    # Ionized atomic species
    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        if Z in ionization_energies:
            wII, wIII = saha_ion_weights(temperature, ne, Z, ionization_energies, partition_funcs)

            n_neutral = species_densities[Species.from_atomic_number(Z, 0)]
            species_densities[Species.from_atomic_number(Z, 1)] = wII * n_neutral
            species_densities[Species.from_atomic_number(Z, 2)] = wIII * n_neutral

    # Molecular species
    if log_equilibrium_constants is not None:
        log_T = np.log(temperature)

        for mol_species, log_K_func in log_equilibrium_constants.items():
            try:
                log_K_partial = log_K_func(log_T)  # log10(K_p)
                n_atoms = len(mol_species.formula.atoms)
                log_nK = log_K_partial - (n_atoms - 1) * np.log10(kboltz_cgs * temperature)

                atoms = list(mol_species.get_atoms())

                if mol_species.charge == 1:
                    Z1, Z2 = atoms[0], atoms[1]
                    n1_II = species_densities[Species.from_atomic_number(Z1, 1)]
                    n2_I = species_densities[Species.from_atomic_number(Z2, 0)]

                    if n1_II > 0 and n2_I > 0:
                        n_mol = 10**(np.log10(n1_II) + np.log10(n2_I) - log_nK)
                        species_densities[mol_species] = n_mol
                else:
                    element_log_ns = [np.log10(species_densities[Species.from_atomic_number(Z, 0)])
                                     for Z in atoms]
                    if all(np.isfinite(element_log_ns)):
                        n_mol = 10**(sum(element_log_ns) - log_nK)
                        species_densities[mol_species] = n_mol
            except (KeyError, ValueError):
                continue

    return ne, species_densities


__all__ = ['chemical_equilibrium', 'saha_ion_weights', 'translational_U']
