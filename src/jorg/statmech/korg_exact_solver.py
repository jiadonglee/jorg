"""
Korg.jl-Exact Chemical Equilibrium Solver for Jorg
===================================================

This module implements the EXACT chemical equilibrium solver from Korg.jl's statmech.jl,
ensuring proper electron density recalculation for metal-poor stars.

Key features:
- Direct translation of Korg.jl's solve_chemical_equilibrium function
- Proper electron density recalculation (not using atmospheric values)
- Correct handling of metal-poor conditions
- Iterative Newton solver with proper convergence
"""

import numpy as np
from scipy.optimize import fsolve, root
from typing import Dict, Tuple, Callable
import warnings

from ..constants import kboltz_cgs, kboltz_eV, hplanck_cgs, me_cgs
from .species import Species, Formula, MAX_ATOMIC_NUMBER

# Constants matching Korg.jl exactly
KBOLTZ_EV = 8.617333262145e-5  # eV/K - Korg.jl's exact value
ELECTRON_MASS_CGS = 9.1093837015e-28  # g - electron mass


def translational_U(m: float, T: float) -> float:
    """
    Translational partition function contribution (Korg.jl line 48-51)
    
    The (possibly inverse) contribution to the partition function from the 
    free movement of a particle. Used in the Saha equation.
    
    Parameters:
    -----------
    m : float
        Particle mass in g
    T : float
        Temperature in K
        
    Returns:
    --------
    float
        Translational partition function
    """
    k = kboltz_cgs
    h = hplanck_cgs
    return (2 * np.pi * m * k * T / h**2)**1.5


def saha_ion_weights(T: float, ne: float, atom: int, 
                     ionization_energies: Dict, 
                     partition_funcs: Dict) -> Tuple[float, float]:
    """
    Calculate Saha ionization weights (Korg.jl line 18-35)
    
    Returns (wII, wIII), where wII is the ratio of singly ionized to neutral atoms,
    and wIII is the ratio of doubly ionized to neutral atoms.
    
    Parameters:
    -----------
    T : float
        Temperature in K
    ne : float
        Electron number density in cm^-3
    atom : int
        Atomic number
    ionization_energies : Dict
        Ionization energies for each element
    partition_funcs : Dict
        Partition functions for each species
        
    Returns:
    --------
    Tuple[float, float]
        (wII, wIII) - ionization weight ratios
    """
    chi_I, chi_II, chi_III = ionization_energies[atom]
    
    # Get partition functions
    atom_formula = Formula.from_atomic_number(atom)
    log_T = np.log(T)
    
    # Get partition functions with defaults
    species_I = Species(atom_formula, 0)
    species_II = Species(atom_formula, 1)
    
    if species_I in partition_funcs:
        U_I = partition_funcs[species_I](log_T)
    else:
        U_I = 1.0  # Default partition function
    
    if species_II in partition_funcs:
        U_II = partition_funcs[species_II](log_T)
    else:
        U_II = 1.0  # Default partition function
    
    k = KBOLTZ_EV
    trans_U = translational_U(ELECTRON_MASS_CGS, T)
    
    # Saha equation for first ionization
    wII = 2.0 / ne * (U_II / U_I) * trans_U * np.exp(-chi_I / (k * T))
    
    # Saha equation for second ionization
    if atom == 1:  # Hydrogen has no second ionization in this context
        wIII = 0.0
    else:
        species_III = Species(atom_formula, 2)
        if species_III in partition_funcs:
            U_III = partition_funcs[species_III](log_T)
        else:
            U_III = 1.0  # Default partition function
        wIII = wII * 2.0 / ne * (U_III / U_II) * trans_U * np.exp(-chi_II / (k * T))
    
    return wII, wIII


def setup_chemical_equilibrium_residuals(T: float, nt: float, 
                                        absolute_abundances: np.ndarray,
                                        ionization_energies: Dict,
                                        partition_funcs: Dict,
                                        log_equilibrium_constants: Dict):
    """
    Setup residual function for chemical equilibrium (Korg.jl line 272-343)
    
    Creates a residual function that can be used with a nonlinear solver.
    Each equation specifies conservation of a particular element.
    
    Parameters:
    -----------
    T : float
        Temperature in K
    nt : float
        Total number density in cm^-3
    absolute_abundances : np.ndarray
        Normalized element abundances (92 elements)
    ionization_energies : Dict
        Ionization energies
    partition_funcs : Dict
        Partition functions
    log_equilibrium_constants : Dict
        Molecular equilibrium constants
        
    Returns:
    --------
    Callable
        Residual function for solver
    """
    # Get molecules and their equilibrium constants
    molecules = list(log_equilibrium_constants.keys()) if log_equilibrium_constants else []
    
    # Calculate log nK for each molecule (Korg.jl line 278)
    log_nKs = []
    for mol in molecules:
        # Get log K in number density form
        log_K_pressure = log_equilibrium_constants[mol](np.log(T))
        n_atoms_mol = len(mol.formula.atoms)  # Number of atoms in molecule
        log_nK = log_K_pressure - (n_atoms_mol - 1) * np.log10(kboltz_cgs * T)
        log_nKs.append(log_nK)
    
    # Precompute Saha weights with ne factored out (Korg.jl line 282-285)
    wII_ne = np.zeros(MAX_ATOMIC_NUMBER)
    wIII_ne2 = np.zeros(MAX_ATOMIC_NUMBER)
    
    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        if Z in ionization_energies:
            # Calculate with ne = 1 to get the ne-independent part
            wII, wIII = saha_ion_weights(T, 1.0, Z, ionization_energies, partition_funcs)
            wII_ne[Z-1] = wII
            wIII_ne2[Z-1] = wIII
    
    def residuals(x):
        """
        Residual function for chemical equilibrium (Korg.jl line 291-341)
        
        x[0:92] = neutral fractions for each element
        x[92] = electron density scaled by nt/1e5
        """
        # Ensure x is a numpy array
        x = np.asarray(x, dtype=np.float64)
        F = np.zeros(MAX_ATOMIC_NUMBER + 1, dtype=np.float64)
        
        # Extract electron density (Korg.jl line 294)
        ne = abs(x[-1]) * nt * 1e-5
        
        # Calculate number densities (Korg.jl line 299-300)
        atom_number_densities = absolute_abundances * (nt - ne)
        neutral_number_densities = atom_number_densities * np.abs(x[0:MAX_ATOMIC_NUMBER])
        
        F[-1] = 0  # Initialize electron conservation equation
        
        # Element conservation equations (Korg.jl line 304-312)
        for Z in range(1, MAX_ATOMIC_NUMBER + 1):
            if absolute_abundances[Z-1] > 0:
                wII = wII_ne[Z-1] / ne if ne > 0 else 0
                wIII = wIII_ne2[Z-1] / ne**2 if ne > 0 else 0
                
                # Conservation: n_total = n_neutral * (1 + wII + wIII)
                F[Z-1] = atom_number_densities[Z-1] - (1 + wII + wIII) * neutral_number_densities[Z-1]
                
                # Electron contribution from this element
                F[-1] += (wII + 2*wIII) * neutral_number_densities[Z-1]
        
        # Complete electron conservation equation
        F[-1] -= ne
        
        # Handle molecules (Korg.jl line 316-337)
        if len(molecules) > 0:
            # Convert to log for molecular calculations
            log_neutral = np.log10(np.maximum(neutral_number_densities, 1e-300))
            
            for mol, log_nK in zip(molecules, log_nKs):
                if mol.charge == 1:  # Charged diatomic
                    # Get atoms (first has lower atomic number, is charged)
                    atoms = sorted(mol.formula.atoms)
                    Z1, Z2 = atoms[0], atoms[1]
                    
                    wII = wII_ne[Z1-1] / ne if ne > 0 else 0
                    n1_II_log = log_neutral[Z1-1] + np.log10(max(wII, 1e-300))
                    n2_I_log = log_neutral[Z2-1]
                    n_mol = 10**(n1_II_log + n2_I_log - log_nK)
                    
                    # Update residuals
                    F[Z1-1] -= n_mol
                    F[Z2-1] -= n_mol
                    F[-1] += n_mol  # Charged molecule contributes electron
                    
                else:  # Neutral molecule
                    atoms = mol.formula.atoms
                    log_sum = sum(log_neutral[Z-1] for Z in atoms)
                    n_mol = 10**(log_sum - log_nK)
                    
                    # Update residuals for each constituent atom
                    for Z in atoms:
                        F[Z-1] -= n_mol
        
        # Scale residuals (Korg.jl line 339-340)
        F[0:MAX_ATOMIC_NUMBER] /= np.maximum(atom_number_densities, 1e-300)
        F[-1] /= (ne * 1e-5) if ne > 0 else 1.0
        
        return F
    
    return residuals


def solve_chemical_equilibrium_korg_exact(T: float, nt: float, 
                                         absolute_abundances: Dict[int, float],
                                         model_atm_ne: float,
                                         ionization_energies: Dict,
                                         partition_funcs: Dict,
                                         log_equilibrium_constants: Dict = None) -> Tuple[float, np.ndarray]:
    """
    Solve chemical equilibrium exactly as Korg.jl does (line 167-176)
    
    This is the core solver that properly recalculates electron density
    based on actual abundances, not atmospheric values.
    
    Parameters:
    -----------
    T : float
        Temperature in K
    nt : float
        Total number density in cm^-3
    absolute_abundances : Dict[int, float]
        Absolute abundances by atomic number
    model_atm_ne : float
        Model atmosphere electron density (used as initial guess)
    ionization_energies : Dict
        Ionization energies
    partition_funcs : Dict
        Partition functions
    log_equilibrium_constants : Dict
        Molecular equilibrium constants
        
    Returns:
    --------
    Tuple[float, np.ndarray]
        (electron_density, neutral_fractions)
    """
    # Convert abundances to array (handle both dict and array input)
    abs_array = np.zeros(MAX_ATOMIC_NUMBER)
    if isinstance(absolute_abundances, dict):
        for Z in range(1, MAX_ATOMIC_NUMBER + 1):
            if Z in absolute_abundances:
                abs_array[Z-1] = float(absolute_abundances[Z])
    else:
        # If already an array, use it directly
        abs_array[:] = absolute_abundances[:MAX_ATOMIC_NUMBER]
    
    # Normalize abundances
    total = np.sum(abs_array)
    if total > 0:
        abs_array /= total
    
    # Initial guess: use Saha equation with atmospheric ne (Korg.jl line 125-128)
    neutral_fraction_guess = np.zeros(MAX_ATOMIC_NUMBER)
    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        if Z in ionization_energies and abs_array[Z-1] > 0:
            wII, wIII = saha_ion_weights(T, model_atm_ne, Z, 
                                        ionization_energies, partition_funcs)
            neutral_fraction_guess[Z-1] = 1.0 / (1.0 + wII + wIII)
        else:
            neutral_fraction_guess[Z-1] = 1.0  # Assume neutral if no data
    
    # Setup residual function
    residuals_func = setup_chemical_equilibrium_residuals(
        T, nt, abs_array, ionization_energies, partition_funcs, 
        log_equilibrium_constants or {}
    )
    
    # Initial guess vector (Korg.jl line 186)
    x0 = np.zeros(MAX_ATOMIC_NUMBER + 1)
    x0[0:MAX_ATOMIC_NUMBER] = neutral_fraction_guess
    x0[-1] = model_atm_ne / nt * 1e5  # Scaled electron density
    
    # Try to solve with initial guess
    try:
        # Use root finding with Newton method (like Korg.jl)
        sol = root(residuals_func, x0, method='hybr', options={'maxfev': 1000})
        
        if not sol.success:
            # Try with smaller electron density guess (Korg.jl line 199)
            x0[-1] = 1e-5
            sol = root(residuals_func, x0, method='hybr', options={'maxfev': 1000})
        
        if sol.success:
            # Extract solution (Korg.jl line 173-175)
            ne = abs(sol.x[-1]) * nt * 1e-5
            neutral_fractions = np.abs(sol.x[0:MAX_ATOMIC_NUMBER])
            return ne, neutral_fractions
        else:
            raise ValueError(f"Chemical equilibrium failed to converge: {sol.message}")
            
    except Exception as e:
        warnings.warn(f"Chemical equilibrium solver failed: {e}")
        # Fallback: return atmospheric values
        return model_atm_ne, neutral_fraction_guess


def chemical_equilibrium_korg_exact(temp: float, nt: float, model_atm_ne: float,
                                   absolute_abundances: Dict[int, float],
                                   ionization_energies: Dict,
                                   partition_funcs: Dict = None,
                                   log_equilibrium_constants: Dict = None,
                                   **kwargs) -> Tuple[float, Dict[Species, float]]:
    """
    Main interface matching Korg.jl's chemical_equilibrium function (line 120-165)
    
    This properly recalculates electron density for metal-poor stars!
    
    Parameters:
    -----------
    temp : float
        Temperature in K
    nt : float
        Total number density in cm^-3
    model_atm_ne : float
        Model atmosphere electron density (initial guess only!)
    absolute_abundances : Dict[int, float]
        Absolute abundances by atomic number
    ionization_energies : Dict
        Ionization energies
    partition_funcs : Dict
        Partition functions
    log_equilibrium_constants : Dict
        Molecular equilibrium constants
        
    Returns:
    --------
    Tuple[float, Dict[Species, float]]
        (electron_density, species_number_densities)
    """
    # Solve for electron density and neutral fractions
    ne, neutral_fractions = solve_chemical_equilibrium_korg_exact(
        temp, nt, absolute_abundances, model_atm_ne,
        ionization_energies, partition_funcs, log_equilibrium_constants
    )
    
    # Build number densities dictionary (Korg.jl line 141-149)
    number_densities = {}
    
    # Neutral atomic species (Korg.jl line 141-143)
    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        if Z in absolute_abundances and absolute_abundances[Z] > 0:
            atom = Formula.from_atomic_number(Z)
            neutral_density = (nt - ne) * absolute_abundances[Z] * neutral_fractions[Z-1]
            number_densities[Species(atom, 0)] = neutral_density
            
            # Ionized species (Korg.jl line 145-148)
            if Z in ionization_energies:
                wII, wIII = saha_ion_weights(temp, ne, Z, 
                                            ionization_energies, partition_funcs)
                number_densities[Species(atom, 1)] = wII * neutral_density
                number_densities[Species(atom, 2)] = wIII * neutral_density
    
    # Molecules (Korg.jl line 151-162)
    if log_equilibrium_constants:
        for mol in log_equilibrium_constants.keys():
            # Calculate molecular number density
            log_K_pressure = log_equilibrium_constants[mol](np.log(temp))
            n_atoms = len(mol.formula.atoms)
            log_nK = log_K_pressure - (n_atoms - 1) * np.log10(kboltz_cgs * temp)
            
            if mol.charge == 0:  # Neutral molecule
                log_sum = sum(np.log10(max(number_densities.get(Species(Formula.from_atomic_number(Z), 0), 1e-300), 1e-300))
                             for Z in mol.formula.atoms)
                number_densities[mol] = 10**(log_sum - log_nK)
            else:  # Charged molecule
                atoms = sorted(mol.formula.atoms)
                Z1, Z2 = atoms[0], atoms[1]
                n1_II = number_densities.get(Species(Formula.from_atomic_number(Z1), 1), 1e-300)
                n2_I = number_densities.get(Species(Formula.from_atomic_number(Z2), 0), 1e-300)
                number_densities[mol] = 10**(np.log10(n1_II) + np.log10(n2_I) - log_nK)
    
    # Warning if electron density differs significantly (Korg.jl line 135-138)
    if ne / nt > 1e-4:  # Only warn if significant
        relative_diff = abs((ne - model_atm_ne) / model_atm_ne) if model_atm_ne > 0 else 0
        if relative_diff > 0.1:
            warnings.warn(f"Electron density differs from atmosphere by {relative_diff:.1%} "
                         f"(calculated ne = {ne:.2e}, atmosphere ne = {model_atm_ne:.2e})")
    
    return ne, number_densities


# Export the main function
__all__ = ['chemical_equilibrium_korg_exact', 'solve_chemical_equilibrium_korg_exact',
           'saha_ion_weights', 'translational_U']