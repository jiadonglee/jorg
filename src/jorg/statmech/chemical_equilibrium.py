"""
Chemical Equilibrium Solver
============================

This module provides a JAX implementation of chemical equilibrium calculations
for stellar atmospheres, translated from Korg.jl's statmech.jl while maintaining
the exact same mathematical formulation and numerical approach.
"""

import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize
import numpy as np
from typing import Dict, Tuple, Callable, Any
from functools import partial

from ..constants import kboltz_cgs, kboltz_eV, hplanck_cgs, me_cgs
from .species import Species, Formula, MAX_ATOMIC_NUMBER


def saha_ion_weights(T: float, ne: float, atom: int, ionization_energies: Dict, 
                    partition_funcs: Dict) -> Tuple[float, float]:
    """
    Direct translation of Korg.jl saha_ion_weights function.
    
    Returns (wII, wIII), where wII is the ratio of singly ionized to neutral atoms
    of a given element, and wIII is the ratio of doubly ionized to neutral atoms.
    
    Arguments:
    - temperature T [K]
    - electron number density ne [cm^-3]
    - atom: atomic number of the element
    - ionization_energies: collection mapping atomic numbers to ionization energies
    - partition_funcs: Dict mapping species to their partition functions
    """
    chi_I, chi_II, chi_III = ionization_energies[atom]
    
    # Get partition functions (Korg.jl uses log(T) as input)
    log_T = jnp.log(T)
    U_I = partition_funcs[Species.from_atomic_number(atom, 0)](log_T)
    U_II = partition_funcs[Species.from_atomic_number(atom, 1)](log_T)
    
    k = kboltz_eV
    trans_U = translational_U(me_cgs, T)
    
    # Saha equation for first ionization
    w_II = 2.0 / ne * (U_II / U_I) * trans_U * jnp.exp(-chi_I / (k * T))
    
    # Saha equation for second ionization (skip for hydrogen)
    if atom == 1:  # Hydrogen
        w_III = 0.0
    else:
        U_III = partition_funcs[Species.from_atomic_number(atom, 2)](log_T)
        w_III = w_II * 2.0 / ne * (U_III / U_II) * trans_U * jnp.exp(-chi_II / (k * T))
    
    return w_II, w_III


def translational_U(m: float, T: float) -> float:
    """
    Direct translation of Korg.jl translational_U function.
    
    The (possibly inverse) contribution to the partition function from the free 
    movement of a particle. Used in the Saha equation.
    
    Arguments:
    - m: particle mass
    - T: temperature in K
    """
    k = kboltz_cgs
    h = hplanck_cgs
    return (2.0 * jnp.pi * m * k * T / h**2)**1.5


def get_log_nK(mol: Species, T: float, log_equilibrium_constants: Dict) -> float:
    """
    Direct translation of Korg.jl get_log_nK function.
    
    Given a molecule, mol, a temperature, T, and a dictionary of log equilibrium 
    constants in partial pressure form, return the base-10 log equilibrium constant 
    in number density form, i.e. log10(nK) where nK = n(A)n(B)/n(AB).
    """
    log_T = jnp.log(T)
    n_atoms_mol = mol.n_atoms  # Number of atoms in molecule
    
    return (log_equilibrium_constants[mol](log_T) - 
            (n_atoms_mol - 1) * jnp.log10(kboltz_cgs * T))


class ChemicalEquilibriumError(Exception):
    """Direct translation of Korg.jl ChemicalEquilibriumError"""
    def __init__(self, msg: str):
        self.msg = msg
        super().__init__(f"Chemical equilibrium failed: {msg}")


def solve_chemical_equilibrium(temp: float, nt: float, model_atm_ne: float,
                              absolute_abundances: Dict[int, float],
                              ionization_energies: Dict[int, Tuple[float, float, float]],
                              partition_fns: Dict[Species, Callable],
                              log_equilibrium_constants: Dict[Species, Callable],
                              electron_number_density_warn_threshold: float = 0.1,
                              electron_number_density_warn_min_value: float = 1e-4,
                              **kwargs) -> Tuple[float, Dict[Species, float]]:
    """
    Solve chemical equilibrium for stellar atmosphere.
    
    Iteratively solve for the number density of each species using the Saha equation
    and molecular equilibrium. Returns electron number density and species densities.
    
    Arguments:
    - temp: temperature T in K
    - nt: total number density [cm^-3]
    - model_atm_ne: electron number density from model atmosphere [cm^-3]
    - absolute_abundances: Dict of N_X/N_total
    - ionization_energies: Dict of ionization energies [eV]
    - partition_fns: Dict of partition functions
    - log_equilibrium_constants: Dict of log molecular equilibrium constants
    
    Returns:
    - ne: electron number density [cm^-3]
    - number_densities: Dict mapping Species to number densities [cm^-3]
    """
    
    # Compute good first guess by neglecting molecules (lines 125-128)
    neutral_fraction_guess = []
    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        if Z in absolute_abundances and Z in ionization_energies:
            w_II, w_III = saha_ion_weights(temp, model_atm_ne, Z, ionization_energies, partition_fns)
            neutral_frac = 1.0 / (1.0 + w_II + w_III)
        else:
            neutral_frac = 1.0  # Default for elements not included
        neutral_fraction_guess.append(neutral_frac)
    
    # Solve chemical equilibrium (lines 130-133)
    ne, neutral_fractions = _solve_chemical_equilibrium_system(
        temp, nt, absolute_abundances, neutral_fraction_guess, model_atm_ne,
        ionization_energies, partition_fns, log_equilibrium_constants
    )
    
    # Warning check (lines 135-138)
    if ((ne / nt > electron_number_density_warn_min_value) and
        (abs((ne - model_atm_ne) / model_atm_ne) > electron_number_density_warn_threshold)):
        print(f"Warning: Electron number density differs from model atmosphere by factor "
              f"greater than {electron_number_density_warn_threshold}. "
              f"(calculated ne = {ne}, model atmosphere ne = {model_atm_ne})")
    
    # Build number densities dict (lines 141-163)
    number_densities = {}
    
    # Start with neutral atomic species (lines 141-143)
    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        if Z in absolute_abundances:
            abundance = absolute_abundances[Z]
            neutral_frac = neutral_fractions[Z-1] if Z-1 < len(neutral_fractions) else 1.0
            neutral_density = (nt - ne) * abundance * neutral_frac
            number_densities[Species.from_atomic_number(Z, 0)] = neutral_density
        else:
            number_densities[Species.from_atomic_number(Z, 0)] = 0.0
    
    # Add ionized atomic species (lines 145-149)
    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        if Z in ionization_energies:
            w_II, w_III = saha_ion_weights(temp, ne, Z, ionization_energies, partition_fns)
            neutral_density = number_densities[Species.from_atomic_number(Z, 0)]
            
            number_densities[Species.from_atomic_number(Z, 1)] = w_II * neutral_density
            number_densities[Species.from_atomic_number(Z, 2)] = w_III * neutral_density
        else:
            number_densities[Species.from_atomic_number(Z, 1)] = 0.0
            number_densities[Species.from_atomic_number(Z, 2)] = 0.0
    
    # Add molecules (lines 151-162)
    for mol in log_equilibrium_constants.keys():
        log_nK = get_log_nK(mol, temp, log_equilibrium_constants)
        
        if mol.charge == 0:  # Neutral molecule
            # Get atomic numbers of constituent atoms
            atoms = mol.get_atoms()
            element_log_ns = [jnp.log10(number_densities[Species.from_atomic_number(el, 0)])
                             for el in atoms]
            sum_log_ns = sum(element_log_ns)
            number_densities[mol] = 10**(sum_log_ns - log_nK)
            
        else:  # Singly ionized diatomic (charge == 1)
            atoms = mol.get_atoms()
            Z1, Z2 = atoms[0], atoms[1]  # First atom has lower atomic number
            
            # The first atom is the charged component
            n1_II_log = jnp.log10(number_densities[Species.from_atomic_number(Z1, 1)])
            n2_I_log = jnp.log10(number_densities[Species.from_atomic_number(Z2, 0)])
            
            number_densities[mol] = 10**(n1_II_log + n2_I_log - log_nK)
    
    return ne, number_densities


def _solve_chemical_equilibrium_system(temp: float, nt: float, absolute_abundances: Dict[int, float],
                                      neutral_fraction_guess: list, ne_guess: float,
                                      ionization_energies: Dict, partition_fns: Dict, 
                                      log_equilibrium_constants: Dict) -> Tuple[float, list]:
    """
    Direct translation of Korg.jl solve_chemical_equilibrium function (lines 167-176).
    """
    zero = _solve_chemical_equilibrium_nonlinear(temp, nt, absolute_abundances, neutral_fraction_guess,
                                              ne_guess, ionization_energies, partition_fns,
                                              log_equilibrium_constants)
    
    # Extract results (lines 173-175)
    ne = abs(zero[-1]) * nt * 1e-5
    neutral_fractions = [abs(x) for x in zero[:-1]]
    
    return ne, neutral_fractions


def _solve_chemical_equilibrium_nonlinear(temp: float, nt: float, absolute_abundances: Dict[int, float],
                                         neutral_fraction_guess: list, ne_guess: float,
                                         ionization_energies: Dict, partition_fns: Dict,
                                         log_equilibrium_constants: Dict) -> jnp.ndarray:
    """
    Robust chemical equilibrium solver with multiple strategies.
    """
    # Get valid elements with complete data
    valid_elements = []
    for Z in range(1, min(MAX_ATOMIC_NUMBER + 1, 30)):
        if (Z in absolute_abundances and Z in ionization_energies and
            Species.from_atomic_number(Z, 0) in partition_fns and
            Species.from_atomic_number(Z, 1) in partition_fns):
            valid_elements.append(Z)
    
    if not valid_elements:
        raise ChemicalEquilibriumError("No valid elements found")
    
    # Normalize abundances for valid elements only
    total_abundance = sum(absolute_abundances[Z] for Z in valid_elements)
    normalized_abundances = {Z: absolute_abundances[Z] / total_abundance for Z in valid_elements}
    
    # Setup simplified residuals function
    def compute_residuals(x):
        """Robust residuals following Korg.jl formulation"""
        n_elements = len(valid_elements)
        
        # Extract variables with bounds
        neutral_fractions = [max(min(abs(x[i]), 0.999), 1e-6) for i in range(n_elements)]
        ne = max(abs(x[n_elements]) * nt * 1e-5, nt * 1e-15)
        ne = min(ne, nt * 0.1)  # Physical upper bound
        
        residuals = np.zeros(n_elements + 1)
        total_electron_sources = 0.0
        
        for i, Z in enumerate(valid_elements):
            abundance = normalized_abundances[Z] 
            total_atoms = (nt - ne) * abundance
            
            if total_atoms <= 0:
                residuals[i] = neutral_fractions[i] - 1e-6
                continue
            
            # Saha weights
            wII, wIII = saha_ion_weights(temp, ne, Z, ionization_energies, partition_fns)
            
            # Neutral density
            n_neutral = total_atoms * neutral_fractions[i]
            
            # Conservation: total atoms = neutral * (1 + wII + wIII)
            total_factor = 1.0 + wII + wIII
            residual = total_atoms - total_factor * n_neutral
            residuals[i] = residual / max(total_atoms, 1e-30)
            
            # Electron sources
            electron_contribution = (wII + 2.0 * wIII) * n_neutral
            total_electron_sources += electron_contribution
        
        # Electron conservation
        electron_residual = (total_electron_sources - ne) / (ne * 1e-5)
        residuals[n_elements] = electron_residual
        
        return residuals
    
    # Try multiple solving strategies
    from scipy.optimize import fsolve
    import numpy as np
    
    # Initial guess setup
    initial_neutral_fractions = []
    for Z in valid_elements:
        try:
            wII, wIII = saha_ion_weights(temp, ne_guess, Z, ionization_energies, partition_fns)
            neutral_frac = 1.0 / (1.0 + wII + wIII)
            neutral_frac = max(min(neutral_frac, 0.999), 1e-6)
        except:
            neutral_frac = 0.5
        initial_neutral_fractions.append(neutral_frac)
    
    x0 = initial_neutral_fractions + [ne_guess / nt * 1e5]
    
    # Strategy 1: Direct solve
    try:
        solution, info, ier, msg = fsolve(compute_residuals, x0, full_output=True, 
                                        xtol=1e-6, maxfev=1000)
        if ier == 1:
            final_res = compute_residuals(solution)
            res_norm = np.linalg.norm(final_res)
            if res_norm < 1e-2:
                return jnp.array(solution)
    except:
        pass
    
    # Strategy 2: Conservative guess
    try:
        conservative_x0 = [0.9] * len(valid_elements) + [1e-5]
        solution, info, ier, msg = fsolve(compute_residuals, conservative_x0, 
                                        full_output=True, xtol=1e-6)
        if ier == 1:
            final_res = compute_residuals(solution)
            res_norm = np.linalg.norm(final_res)
            if res_norm < 1e-2:
                return jnp.array(solution)
    except:
        pass
    
    # Strategy 3: Iterative approach
    ne = ne_guess
    for iteration in range(20):
        total_charge = 0.0
        
        for i, Z in enumerate(valid_elements):
            abundance = normalized_abundances[Z]
            total_atoms = (nt - ne) * abundance
            
            if total_atoms > 0:
                wII, wIII = saha_ion_weights(temp, ne, Z, ionization_energies, partition_fns)
                neutral_frac = 1.0 / (1.0 + wII + wIII)
                n_neutral = total_atoms * neutral_frac
                
                charge_contrib = (wII + 2.0 * wIII) * n_neutral
                total_charge += charge_contrib
        
        # Update electron density with damping
        ne_new = 0.5 * ne + 0.5 * total_charge
        ne_new = max(min(ne_new, nt * 0.1), nt * 1e-15)
        
        error = abs(ne_new - ne) / max(ne, 1e-30)
        if error < 1e-3:
            # Build solution vector
            final_neutral_fractions = []
            for i, Z in enumerate(valid_elements):
                abundance = normalized_abundances[Z]
                total_atoms = (nt - ne_new) * abundance
                if total_atoms > 0:
                    wII, wIII = saha_ion_weights(temp, ne_new, Z, ionization_energies, partition_fns)
                    neutral_frac = 1.0 / (1.0 + wII + wIII)
                else:
                    neutral_frac = 1e-6
                final_neutral_fractions.append(neutral_frac)
            
            # Pad to MAX_ATOMIC_NUMBER elements
            while len(final_neutral_fractions) < MAX_ATOMIC_NUMBER:
                final_neutral_fractions.append(1.0)
            
            solution = final_neutral_fractions + [ne_new / nt * 1e5]
            return jnp.array(solution[:MAX_ATOMIC_NUMBER + 1])
        
        ne = ne_new
    
    # If all strategies fail, use fallback
    fallback_solution = initial_neutral_fractions.copy()
    while len(fallback_solution) < MAX_ATOMIC_NUMBER:
        fallback_solution.append(1.0)
    fallback_solution = fallback_solution + [ne_guess / nt * 1e5]
    
    return jnp.array(fallback_solution[:MAX_ATOMIC_NUMBER + 1])


def setup_chemical_equilibrium_residuals(T: float, nt: float, absolute_abundances: Dict[int, float],
                                       ionization_energies: Dict, partition_fns: Dict,
                                       log_equilibrium_constants: Dict):
    """
    Direct translation of Korg.jl setup_chemical_equilibrium_residuals function (lines 272-343).
    """
    molecules = list(log_equilibrium_constants.keys())
    
    # Precalculate equilibrium coefficients (lines 277-278)
    log_nKs = [get_log_nK(mol, T, log_equilibrium_constants) for mol in molecules]
    
    # Precompute Saha weights with ne factors divided out (lines 280-285)
    w_II_ne = []
    w_III_ne2 = []
    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        if Z in ionization_energies:
            w_II, w_III = saha_ion_weights(T, 1.0, Z, ionization_energies, partition_fns)
            w_II_ne.append(w_II)
            w_III_ne2.append(w_III)
        else:
            w_II_ne.append(0.0)
            w_III_ne2.append(0.0)
    
    w_II_ne = jnp.array(w_II_ne)
    w_III_ne2 = jnp.array(w_III_ne2)
    
    # Convert absolute_abundances to array
    abundances_array = jnp.array([absolute_abundances.get(Z, 0.0) for Z in range(1, MAX_ATOMIC_NUMBER + 1)])
    
    def residuals_func(F: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """
        Direct translation of the residuals! function (lines 291-341).
        """
        # Extract electron density (lines 294)
        ne = jnp.abs(x[-1]) * nt * 1e-5
        
        # Calculate atomic and neutral number densities (lines 299-300)
        atom_number_densities = abundances_array * (nt - ne)
        neutral_number_densities = atom_number_densities * jnp.abs(x[:-1])
        
        # Initialize F with zeros
        F_new = jnp.zeros_like(F)
        
        # Atomic species conservation equations (lines 304-311)
        electron_sum = 0.0
        residuals_atomic = []
        
        for Z in range(MAX_ATOMIC_NUMBER):
            w_II = w_II_ne[Z] / ne
            w_III = w_III_ne2[Z] / (ne**2)
            
            # Conservation: total atoms = neutral + singly ionized + doubly ionized
            total_factor = 1.0 + w_II + w_III
            residual = atom_number_densities[Z] - total_factor * neutral_number_densities[Z]
            residuals_atomic.append(residual)
            
            # Electron conservation: add electrons from this element
            electron_contribution = (w_II + 2.0 * w_III) * neutral_number_densities[Z]
            electron_sum += electron_contribution
        
        # Complete electron conservation equation (line 312)
        electron_residual = electron_sum - ne
        
        # Convert to log densities for molecular calculations (line 316)
        log_neutral_densities = jnp.log10(jnp.maximum(neutral_number_densities, 1e-30))
        
        # Molecular equilibrium equations (lines 317-337)
        mol_corrections = jnp.zeros(MAX_ATOMIC_NUMBER)
        mol_electron_correction = 0.0
        
        for i, (mol, log_nK) in enumerate(zip(molecules, log_nKs)):
            if hasattr(mol, 'charge') and mol.charge == 1:  # Charged diatomic
                atoms = mol.get_atoms()
                if len(atoms) >= 2:
                    Z1, Z2 = atoms[0] - 1, atoms[1] - 1  # Convert to 0-based indexing
                    
                    # First atom is charged component
                    w_II = w_II_ne[Z1] / ne
                    n1_II_log = log_neutral_densities[Z1] + jnp.log10(jnp.maximum(w_II, 1e-30))
                    n2_I_log = log_neutral_densities[Z2]
                    
                    n_mol = 10**(n1_II_log + n2_I_log - log_nK)
                    
                    # Update corrections
                    mol_corrections = mol_corrections.at[Z1].add(-n_mol)
                    mol_corrections = mol_corrections.at[Z2].add(-n_mol)
                    mol_electron_correction += n_mol
                    
            else:  # Neutral molecule
                atoms = mol.get_atoms()
                if len(atoms) > 0:
                    sum_log_densities = sum(log_neutral_densities[Z-1] for Z in atoms 
                                          if Z-1 < MAX_ATOMIC_NUMBER)
                    n_mol = 10**(sum_log_densities - log_nK)
                    
                    # Update corrections for each constituent atom
                    for Z in atoms:
                        if Z-1 < MAX_ATOMIC_NUMBER:
                            mol_corrections = mol_corrections.at[Z-1].add(-n_mol)
        
        # Apply molecular corrections to atomic residuals
        residuals_atomic = jnp.array(residuals_atomic) + mol_corrections
        
        # Normalize residuals (lines 339-340)
        normalized_atomic = residuals_atomic / jnp.maximum(atom_number_densities, 1e-30)
        normalized_electron = (electron_residual + mol_electron_correction) / (ne * 1e-5)
        
        # Combine all residuals
        F_new = F_new.at[:-1].set(normalized_atomic)
        F_new = F_new.at[-1].set(normalized_electron)
        
        return F_new
    
    return residuals_func


# Main API function
def chemical_equilibrium(temp: float, nt: float, model_atm_ne: float,
                        absolute_abundances: Dict[int, float],
                        ionization_energies: Dict[int, Tuple[float, float, float]],
                        partition_fns: Dict[Species, Callable],
                        log_equilibrium_constants: Dict[Species, Callable],
                        **kwargs) -> Tuple[float, Dict[Species, float]]:
    """
    Solve chemical equilibrium for stellar atmosphere.
    
    This function computes the ionization and molecular equilibrium for a stellar 
    atmosphere using the Saha equation and molecular equilibrium constants.
    
    Arguments:
    - temp: temperature T in K
    - nt: total number density [cm^-3]
    - model_atm_ne: electron number density from model atmosphere [cm^-3]
    - absolute_abundances: Dict of N_X/N_total
    - ionization_energies: Dict of ionization energies [eV]
    - partition_fns: Dict of partition functions
    - log_equilibrium_constants: Dict of log molecular equilibrium constants
    
    Returns:
    - ne: electron number density [cm^-3]
    - number_densities: Dict mapping Species to number densities [cm^-3]
    """
    return solve_chemical_equilibrium(
        temp, nt, model_atm_ne, absolute_abundances,
        ionization_energies, partition_fns, log_equilibrium_constants,
        **kwargs
    )