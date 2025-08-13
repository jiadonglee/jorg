"""
Fast Saha Equation Implementation with JIT and Vectorization
============================================================

Optimized JAX implementation of Saha ionization equilibrium calculations
with comprehensive JIT compilation and vectorization for maximum performance.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from typing import Dict, Tuple, Any, Callable
import numpy as np
from functools import partial

from ..constants import kboltz_cgs, hplanck_cgs, me_cgs, EV_TO_ERG
from .species import Species, Formula, all_atomic_species, MAX_ATOMIC_NUMBER

# Exact Korg.jl constants for perfect compatibility and performance
KORG_KBOLTZ_CGS = 1.380649e-16  # erg/K
KORG_HPLANCK_CGS = 6.62607015e-27  # erg*s  
KORG_ELECTRON_MASS_CGS = 9.1093897e-28  # g
KORG_KBOLTZ_EV = 8.617333262145e-5  # eV/K

# Pre-computed constants for maximum performance
PI_2 = 2.0 * jnp.pi
TRANS_CONST = PI_2 * KORG_ELECTRON_MASS_CGS * KORG_KBOLTZ_CGS / (KORG_HPLANCK_CGS ** 2)
TRANS_POW = 1.5


@jit
def translational_U_fast(T: float) -> float:
    """
    Ultra-fast translational partition function using pre-computed constants.
    
    Computes (2πmkT/h²)^(3/2) for electron mass with minimal operations.
    
    Parameters:
    -----------
    T : float
        Temperature in K
        
    Returns:
    --------
    float
        Translational partition function: (2πmkT/h²)^(3/2)
    """
    return (TRANS_CONST * T) ** TRANS_POW


@jit
def saha_ion_weight_single(T: float, ne: float, chi_I: float, chi_II: float,
                          U_I: float, U_II: float, U_III: float) -> Tuple[float, float]:
    """
    Fast Saha equation for a single element with JIT compilation.
    
    Parameters:
    -----------
    T : float
        Temperature in K
    ne : float
        Electron density in cm^-3
    chi_I, chi_II : float
        First and second ionization energies in eV
    U_I, U_II, U_III : float
        Partition functions for neutral, singly, and doubly ionized states
        
    Returns:
    --------
    Tuple[float, float]
        (wII, wIII) - ionization weight ratios
    """
    k_T = KORG_KBOLTZ_EV * T
    trans_U = translational_U_fast(T)
    factor = 2.0 * trans_U / ne
    
    # First ionization
    wII = factor * (U_II / U_I) * jnp.exp(-chi_I / k_T)
    
    # Second ionization (conditional on valid energy)
    wIII = jnp.where(
        chi_II > 0.0,
        wII * factor * (U_III / U_II) * jnp.exp(-chi_II / k_T),
        0.0
    )
    
    return wII, wIII


@jit
def saha_ion_weights_vectorized(T: float, ne: float, 
                               chi_I: jnp.ndarray, chi_II: jnp.ndarray,
                               U_I: jnp.ndarray, U_II: jnp.ndarray, U_III: jnp.ndarray,
                               atomic_numbers: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Vectorized Saha equation for multiple elements simultaneously.
    
    Parameters:
    -----------
    T : float
        Temperature in K
    ne : float
        Electron density in cm^-3
    chi_I, chi_II : jnp.ndarray
        First and second ionization energies [eV]
    U_I, U_II, U_III : jnp.ndarray
        Partition functions arrays
    atomic_numbers : jnp.ndarray
        Array of atomic numbers for special cases
        
    Returns:
    --------
    Tuple[jnp.ndarray, jnp.ndarray]
        (wII, wIII) - vectorized ionization weights
    """
    k_T = KORG_KBOLTZ_EV * T
    trans_U = translational_U_fast(T)
    factor = 2.0 * trans_U / ne
    
    # Vectorized first ionization
    exp_factor_I = jnp.exp(-chi_I / k_T)
    wII = factor * (U_II / U_I) * exp_factor_I
    
    # Vectorized second ionization with conditionals
    exp_factor_II = jnp.exp(-jnp.where(chi_II > 0, chi_II, 0.0) / k_T)
    wIII_calc = wII * factor * (U_III / U_II) * exp_factor_II
    
    # Apply conditions: valid chi_II and not hydrogen (Z=1)
    wIII = jnp.where((chi_II > 0.0) & (atomic_numbers != 1), wIII_calc, 0.0)
    
    return wII, wIII


@jit 
def saha_batch_temperatures(chi_I: float, chi_II: float,
                           U_I: jnp.ndarray, U_II: jnp.ndarray, U_III: jnp.ndarray,
                           T_array: jnp.ndarray, ne_array: jnp.ndarray,
                           atomic_number: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Vectorized Saha equation across multiple temperatures for a single element.
    
    Parameters:
    -----------
    chi_I, chi_II : float
        Ionization energies for the element
    U_I, U_II, U_III : jnp.ndarray
        Partition functions at different temperatures
    T_array : jnp.ndarray
        Array of temperatures
    ne_array : jnp.ndarray
        Array of electron densities
    atomic_number : int
        Atomic number for special cases
        
    Returns:
    --------
    Tuple[jnp.ndarray, jnp.ndarray]
        (wII, wIII) arrays across all temperatures
    """
    # Vectorize over temperatures
    vectorized_saha = vmap(saha_ion_weight_single, in_axes=(0, 0, None, None, 0, 0, 0))
    
    return vectorized_saha(T_array, ne_array, chi_I, chi_II, U_I, U_II, U_III)


class FastSahaCalculator:
    """
    Fast Saha equation calculator with pre-compiled functions and vectorization.
    """
    
    def __init__(self, ionization_energies: Dict[int, Tuple[float, float, float]]):
        """
        Initialize with ionization energy data.
        
        Parameters:
        -----------
        ionization_energies : Dict[int, Tuple[float, float, float]]
            Ionization energies for all elements
        """
        self.max_elements = MAX_ATOMIC_NUMBER
        
        # Pre-process ionization energies into arrays for vectorization
        self.chi_I = jnp.zeros(self.max_elements)
        self.chi_II = jnp.zeros(self.max_elements)
        self.chi_III = jnp.zeros(self.max_elements)
        self.atomic_numbers = jnp.arange(1, self.max_elements + 1)
        
        for Z in range(1, self.max_elements + 1):
            if Z in ionization_energies:
                chi_I, chi_II, chi_III = ionization_energies[Z]
                self.chi_I = self.chi_I.at[Z-1].set(chi_I)
                self.chi_II = self.chi_II.at[Z-1].set(chi_II if chi_II > 0 else 0.0)
                self.chi_III = self.chi_III.at[Z-1].set(chi_III if chi_III > 0 else 0.0)
        
        # Pre-compile functions
        self._compile_functions()
    
    def _compile_functions(self):
        """Pre-compile all Saha equation functions."""
        
        @jit
        def compute_all_ionization_weights(T: float, ne: float,
                                         U_I: jnp.ndarray, U_II: jnp.ndarray, U_III: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """
            Compute ionization weights for all elements at given conditions.
            """
            return saha_ion_weights_vectorized(T, ne, self.chi_I, self.chi_II, 
                                             U_I, U_II, U_III, self.atomic_numbers)
        
        self._compute_all_weights = compute_all_ionization_weights
        
        @jit
        def compute_ionization_fractions(wII: jnp.ndarray, wIII: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            """
            Compute ionization fractions from weights.
            
            Returns:
            --------
            Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
                (f_I, f_II, f_III) - neutral, singly, doubly ionized fractions
            """
            total = 1.0 + wII + wIII
            f_I = 1.0 / total
            f_II = wII / total
            f_III = wIII / total
            return f_I, f_II, f_III
        
        self._compute_fractions = compute_ionization_fractions
        
        @jit
        def compute_electron_contribution(wII: jnp.ndarray, wIII: jnp.ndarray,
                                        abundances: jnp.ndarray, total_atoms: jnp.ndarray) -> float:
            """
            Compute total electron contribution from ionization.
            
            Parameters:
            -----------
            wII, wIII : jnp.ndarray
                Ionization weights
            abundances : jnp.ndarray
                Element abundances
            total_atoms : jnp.ndarray
                Total atomic densities
                
            Returns:
            --------
            float
                Total electron density from ionization
            """
            neutral_fractions = 1.0 / (1.0 + wII + wIII)
            neutral_densities = total_atoms * neutral_fractions
            electron_contrib = (wII + 2.0 * wIII) * neutral_densities
            return jnp.sum(electron_contrib)
        
        self._compute_electron_contrib = compute_electron_contribution
    
    @partial(jit, static_argnums=(0,))
    def calculate_ionization_equilibrium(self, T: float, ne: float,
                                       U_I: jnp.ndarray, U_II: jnp.ndarray, U_III: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Calculate complete ionization equilibrium for all elements.
        
        Parameters:
        -----------
        T : float
            Temperature in K
        ne : float
            Electron density in cm^-3
        U_I, U_II, U_III : jnp.ndarray
            Partition function arrays for all elements
            
        Returns:
        --------
        Dict[str, jnp.ndarray]
            Dictionary containing ionization weights and fractions
        """
        # Compute ionization weights
        wII, wIII = self._compute_all_weights(T, ne, U_I, U_II, U_III)
        
        # Compute ionization fractions
        f_I, f_II, f_III = self._compute_fractions(wII, wIII)
        
        return {
            'wII': wII,
            'wIII': wIII,
            'f_I': f_I,
            'f_II': f_II,
            'f_III': f_III
        }
    
    @partial(jit, static_argnums=(0,))
    def solve_electron_density(self, T: float, nt: float, abundances: jnp.ndarray,
                             U_I: jnp.ndarray, U_II: jnp.ndarray, U_III: jnp.ndarray,
                             ne_guess: float, max_iter: int = 20) -> float:
        """
        Solve for electron density using charge conservation.
        
        Parameters:
        -----------
        T : float
            Temperature in K
        nt : float
            Total number density in cm^-3
        abundances : jnp.ndarray
            Element abundances (normalized)
        U_I, U_II, U_III : jnp.ndarray
            Partition function arrays
        ne_guess : float
            Initial electron density guess
        max_iter : int
            Maximum iterations
            
        Returns:
        --------
        float
            Converged electron density
        """
        ne = ne_guess
        
        # Fixed-point iteration with vectorized operations
        for _ in range(max_iter):
            # Compute ionization weights
            wII, wIII = self._compute_all_weights(T, ne, U_I, U_II, U_III)
            
            # Compute electron density from charge conservation
            total_atoms = abundances * (nt - ne)
            ne_new = self._compute_electron_contrib(wII, wIII, abundances, total_atoms)
            
            # Check convergence
            rel_error = jnp.abs(ne_new - ne) / jnp.maximum(ne, 1e-30)
            if rel_error < 1e-6:
                return ne_new
            
            # Update with damping for stability
            ne = 0.7 * ne_new + 0.3 * ne
            ne = jnp.clip(ne, nt * 1e-15, nt * 0.1)
        
        return ne


# Vectorized grid calculations
@jit
def saha_weights_temperature_grid(chi_I: float, chi_II: float,
                                 U_I_grid: jnp.ndarray, U_II_grid: jnp.ndarray, U_III_grid: jnp.ndarray,
                                 T_grid: jnp.ndarray, ne_grid: jnp.ndarray,
                                 atomic_number: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute Saha weights on a temperature-density grid for a single element.
    
    Parameters:
    -----------
    chi_I, chi_II : float
        Ionization energies
    U_I_grid, U_II_grid, U_III_grid : jnp.ndarray
        Partition function grids [n_temp, n_density]
    T_grid, ne_grid : jnp.ndarray
        Temperature and density grids
    atomic_number : int
        Atomic number
        
    Returns:
    --------
    Tuple[jnp.ndarray, jnp.ndarray]
        (wII_grid, wIII_grid) on the full grid
    """
    # Double vectorization over temperature and density
    double_vectorized = vmap(vmap(saha_ion_weight_single, in_axes=(0, 0, None, None, 0, 0, 0)),
                           in_axes=(1, 1, None, None, 1, 1, 1))
    
    return double_vectorized(T_grid, ne_grid, chi_I, chi_II, U_I_grid, U_II_grid, U_III_grid)


@jit
def compute_ionization_degree(T: float, ne: float, atomic_number: int,
                            chi_I: float, chi_II: float,
                            U_I: float, U_II: float, U_III: float) -> Tuple[float, float]:
    """
    Compute ionization degree for a single element.
    
    Parameters:
    -----------
    T : float
        Temperature in K
    ne : float
        Electron density in cm^-3
    atomic_number : int
        Atomic number
    chi_I, chi_II : float
        Ionization energies in eV
    U_I, U_II, U_III : float
        Partition functions
        
    Returns:
    --------
    Tuple[float, float]
        (first_ionization_degree, second_ionization_degree)
    """
    wII, wIII = saha_ion_weight_single(T, ne, chi_I, chi_II, U_I, U_II, U_III)
    
    total = 1.0 + wII + wIII
    first_degree = wII / total
    second_degree = wIII / total
    
    return first_degree, second_degree


# High-level API functions
def create_fast_saha_calculator(ionization_energies: Dict[int, Tuple[float, float, float]]) -> FastSahaCalculator:
    """
    Create a pre-compiled fast Saha equation calculator.
    
    Parameters:
    -----------
    ionization_energies : Dict[int, Tuple[float, float, float]]
        Ionization energies for all elements
        
    Returns:
    --------
    FastSahaCalculator
        Pre-compiled calculator instance
    """
    return FastSahaCalculator(ionization_energies)


@jit
def saha_ion_weights_fast(T: float, ne: float, atom: int, 
                         ionization_energies: Dict,
                         partition_funcs: Dict) -> Tuple[float, float]:
    """
    Fast drop-in replacement for the standard saha_ion_weights function.
    
    This is a JIT-compiled version optimized for single-element calculations.
    
    Parameters:
    -----------
    T : float
        Temperature in K
    ne : float
        Electron density in cm^-3
    atom : int
        Atomic number
    ionization_energies : Dict
        Ionization energy dictionary
    partition_funcs : Dict
        Partition function dictionary
        
    Returns:
    --------
    Tuple[float, float]
        (wII, wIII) ionization weights
    """
    # Extract energies
    chi_I, chi_II, chi_III = ionization_energies[atom]
    
    # Get partition functions
    log_T = jnp.log(T)
    species_0 = Species.from_atomic_number(atom, 0)
    species_1 = Species.from_atomic_number(atom, 1)
    species_2 = Species.from_atomic_number(atom, 2)
    
    U_I = partition_funcs[species_0](log_T)
    U_II = partition_funcs[species_1](log_T)
    U_III = partition_funcs[species_2](log_T) if species_2 in partition_funcs else 1.0
    
    return saha_ion_weight_single(T, ne, chi_I, chi_II, U_I, U_II, U_III)


# Performance utilities
def benchmark_saha_calculations(n_elements: int = 30, n_conditions: int = 100) -> Dict[str, float]:
    """
    Benchmark Saha equation calculations.
    
    Parameters:
    -----------
    n_elements : int
        Number of elements to test
    n_conditions : int
        Number of temperature/density conditions
        
    Returns:
    --------
    Dict[str, float]
        Timing results
    """
    import time
    
    # Generate test data
    T_array = jnp.linspace(3000, 8000, n_conditions)
    ne_array = jnp.logspace(10, 14, n_conditions)
    chi_I_array = jnp.linspace(5, 25, n_elements)
    chi_II_array = jnp.linspace(10, 50, n_elements)
    U_arrays = jnp.ones((n_elements, n_conditions))
    atomic_numbers = jnp.arange(1, n_elements + 1)
    
    # Benchmark vectorized calculation
    start_time = time.time()
    for T, ne in zip(T_array, ne_array):
        wII, wIII = saha_ion_weights_vectorized(T, ne, chi_I_array, chi_II_array,
                                               U_arrays[:, 0], U_arrays[:, 0], U_arrays[:, 0],
                                               atomic_numbers)
    vectorized_time = time.time() - start_time
    
    # Benchmark individual calculations
    start_time = time.time()
    for T, ne in zip(T_array, ne_array):
        for i in range(n_elements):
            wII, wIII = saha_ion_weight_single(T, ne, chi_I_array[i], chi_II_array[i],
                                              1.0, 1.0, 1.0)
    individual_time = time.time() - start_time
    
    return {
        'vectorized_time': vectorized_time,
        'individual_time': individual_time,
        'speedup_factor': individual_time / vectorized_time,
        'n_calculations': n_elements * n_conditions
    }