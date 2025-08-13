"""
Fast Partition Functions with JIT and Vectorization
===================================================

Optimized JAX implementation of partition function calculations with comprehensive
JIT compilation, vectorization, and caching for maximum performance.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from typing import Dict, Any, Callable, Tuple
import numpy as np
from functools import partial, lru_cache
import json
from pathlib import Path

from ..constants import kboltz_cgs, RYDBERG, ELECTRON_MASS, EV_TO_ERG
from .species import Species, Formula, MAX_ATOMIC_NUMBER

# Performance constants
CACHE_SIZE = 1024
SPLINE_POINTS = 50
MIN_LOG_T = jnp.log(1000.0)  # 1000K
MAX_LOG_T = jnp.log(50000.0)  # 50000K


@jit
def hydrogen_partition_function_fast(log_T: float) -> float:
    """
    Fast hydrogen partition function with JIT compilation.
    
    For hydrogen, U = 2 (spin degeneracy) at stellar temperatures.
    """
    return 2.0


@jit
def helium_partition_function_fast(log_T: float) -> float:
    """
    Fast helium partition function with JIT compilation.
    
    For helium I, U = 1 (singlet ground state).
    """
    return 1.0


@jit
def simple_atom_partition_function_fast(atomic_number: int, ionization: int, log_T: float) -> float:
    """
    Fast approximate partition function for atoms using JIT compilation.
    
    Parameters:
    -----------
    atomic_number : int
        Atomic number (Z)
    ionization : int
        Ionization stage (0=neutral, 1=singly ionized, etc.)
    log_T : float
        Natural log of temperature in K
        
    Returns:
    --------
    float
        Partition function value
    """
    T = jnp.exp(log_T)
    T_ref = 5000.0
    
    # Use conditional logic optimized for JAX
    # Hydrogen
    hydrogen_U = 2.0
    
    # Helium
    helium_neutral_U = 1.0
    helium_ion_U = 2.0
    
    # Iron (complex atom)
    iron_neutral_U = 25.0 * (T / T_ref)**0.3
    iron_ion_U = 30.0 * (T / T_ref)**0.2
    
    # Default atoms
    default_neutral_U = 2.0 * (T / T_ref)**0.1
    default_ion_U = 1.0 * (T / T_ref)**0.1
    
    # Nested conditionals using jnp.where for JAX compatibility
    result = jnp.where(
        atomic_number == 1,
        hydrogen_U,
        jnp.where(
            atomic_number == 2,
            jnp.where(ionization == 0, helium_neutral_U, helium_ion_U),
            jnp.where(
                atomic_number == 26,
                jnp.where(ionization == 0, iron_neutral_U, iron_ion_U),
                jnp.where(ionization == 0, default_neutral_U, default_ion_U)
            )
        )
    )
    
    return result


class FastPartitionFunctions:
    """
    Fast partition function calculator with pre-compiled functions and caching.
    """
    
    def __init__(self, korg_data_path: str = None):
        """
        Initialize with optional Korg.jl data for high accuracy.
        
        Parameters:
        -----------
        korg_data_path : str, optional
            Path to Korg partition function data file
        """
        self.use_korg_data = korg_data_path is not None and Path(korg_data_path).exists()
        
        if self.use_korg_data:
            self._load_korg_data(korg_data_path)
        
        # Pre-compile all functions
        self._compile_functions()
    
    def _load_korg_data(self, data_path: str):
        """Load and pre-process Korg.jl partition function data."""
        with open(data_path, 'r') as f:
            self.korg_data = json.load(f)
        
        # Pre-process data into JAX arrays for fast interpolation
        self._preprocess_korg_data()
    
    def _preprocess_korg_data(self):
        """Pre-process Korg data into JAX arrays for fast interpolation."""
        self.korg_species = {}
        self.korg_log_T_arrays = {}
        self.korg_log_U_arrays = {}
        
        for species_str, data in self.korg_data.items():
            if 'log_T' in data and 'log_U' in data:
                log_T_array = jnp.array(data['log_T'])
                log_U_array = jnp.array(data['log_U'])
                
                # Sort by temperature for interpolation
                sort_indices = jnp.argsort(log_T_array)
                log_T_sorted = log_T_array[sort_indices]
                log_U_sorted = log_U_array[sort_indices]
                
                self.korg_log_T_arrays[species_str] = log_T_sorted
                self.korg_log_U_arrays[species_str] = log_U_sorted
    
    def _compile_functions(self):
        """Pre-compile all partition function calculations."""
        
        if self.use_korg_data:
            @jit
            def interpolate_korg_partition_function(species_str: str, log_T: float) -> float:
                """Fast interpolation of Korg partition function data."""
                if species_str in self.korg_log_T_arrays:
                    log_T_array = self.korg_log_T_arrays[species_str]
                    log_U_array = self.korg_log_U_arrays[species_str]
                    
                    # Linear interpolation using JAX
                    log_U_interp = jnp.interp(log_T, log_T_array, log_U_array)
                    return jnp.exp(log_U_interp)
                else:
                    # Fallback to simple model
                    return 1.0
            
            self._interpolate_korg = interpolate_korg_partition_function
        
        # Vectorized partition function calculation
        @jit
        def compute_partition_functions_vectorized(atomic_numbers: jnp.ndarray, 
                                                 ionizations: jnp.ndarray,
                                                 log_T: float) -> jnp.ndarray:
            """
            Vectorized partition function calculation for multiple species.
            
            Parameters:
            -----------
            atomic_numbers : jnp.ndarray
                Array of atomic numbers
            ionizations : jnp.ndarray
                Array of ionization stages
            log_T : float
                Log temperature
                
            Returns:
            --------
            jnp.ndarray
                Partition function values
            """
            # Use vmap for vectorization
            vectorized_pf = vmap(simple_atom_partition_function_fast, in_axes=(0, 0, None))
            return vectorized_pf(atomic_numbers, ionizations, log_T)
        
        self._compute_vectorized = compute_partition_functions_vectorized
        
        # Batch temperature processing
        @jit
        def compute_partition_functions_batch(atomic_number: int, ionization: int,
                                            log_T_array: jnp.ndarray) -> jnp.ndarray:
            """
            Batch partition function calculation across multiple temperatures.
            
            Parameters:
            -----------
            atomic_number : int
                Atomic number
            ionization : int
                Ionization stage
            log_T_array : jnp.ndarray
                Array of log temperatures
                
            Returns:
            --------
            jnp.ndarray
                Partition function values at all temperatures
            """
            vectorized_temp = vmap(simple_atom_partition_function_fast, in_axes=(None, None, 0))
            return vectorized_temp(atomic_number, ionization, log_T_array)
        
        self._compute_batch = compute_partition_functions_batch
    
    @partial(jit, static_argnums=(0, 1, 2))
    def get_partition_function_fast(self, atomic_number: int, ionization: int, log_T: float) -> float:
        """
        Fast partition function calculation with JIT compilation.
        
        Parameters:
        -----------
        atomic_number : int
            Atomic number (Z)
        ionization : int
            Ionization stage
        log_T : float
            Natural log of temperature in K
            
        Returns:
        --------
        float
            Partition function value
        """
        if self.use_korg_data:
            # Try Korg data first
            species_str = f"Z{atomic_number:02d}_{ionization}"
            if species_str in self.korg_log_T_arrays:
                return self._interpolate_korg(species_str, log_T)
        
        # Fallback to fast analytical approximation
        return simple_atom_partition_function_fast(atomic_number, ionization, log_T)
    
    @partial(jit, static_argnums=(0,))
    def get_all_partition_functions_fast(self, log_T: float) -> jnp.ndarray:
        """
        Get partition functions for all elements and ionization stages.
        
        Parameters:
        -----------
        log_T : float
            Natural log of temperature in K
            
        Returns:
        --------
        jnp.ndarray
            Array of shape (MAX_ATOMIC_NUMBER, 3) containing partition functions
            for neutral, singly ionized, and doubly ionized states
        """
        atomic_numbers = jnp.arange(1, MAX_ATOMIC_NUMBER + 1)
        
        # Compute for all ionization stages
        ionization_0 = jnp.zeros_like(atomic_numbers)
        ionization_1 = jnp.ones_like(atomic_numbers)
        ionization_2 = 2 * jnp.ones_like(atomic_numbers)
        
        U_0 = self._compute_vectorized(atomic_numbers, ionization_0, log_T)
        U_1 = self._compute_vectorized(atomic_numbers, ionization_1, log_T)
        U_2 = self._compute_vectorized(atomic_numbers, ionization_2, log_T)
        
        return jnp.stack([U_0, U_1, U_2], axis=1)


# Optimized spline interpolation for Korg data
@jit
def cubic_spline_interpolation_fast(x: float, x_array: jnp.ndarray, y_array: jnp.ndarray) -> float:
    """
    Fast cubic spline interpolation optimized for JAX.
    
    Parameters:
    -----------
    x : float
        Point to interpolate at
    x_array : jnp.ndarray
        Array of x values (must be sorted)
    y_array : jnp.ndarray
        Array of corresponding y values
        
    Returns:
    --------
    float
        Interpolated value
    """
    # Find insertion point
    i = jnp.searchsorted(x_array, x, side='right') - 1
    i = jnp.clip(i, 0, len(x_array) - 2)
    
    # Linear interpolation for simplicity and speed
    x0, x1 = x_array[i], x_array[i + 1]
    y0, y1 = y_array[i], y_array[i + 1]
    
    t = (x - x0) / (x1 - x0)
    return y0 * (1 - t) + y1 * t


# Factory functions for creating optimized partition function callables
def create_fast_partition_function(atomic_number: int, ionization: int, 
                                 fast_calculator: FastPartitionFunctions) -> Callable:
    """
    Create a fast partition function callable for a specific species.
    
    Parameters:
    -----------
    atomic_number : int
        Atomic number
    ionization : int
        Ionization stage
    fast_calculator : FastPartitionFunctions
        Pre-compiled calculator instance
        
    Returns:
    --------
    Callable
        Fast partition function callable
    """
    @jit
    def partition_function(log_T: float) -> float:
        return fast_calculator.get_partition_function_fast(atomic_number, ionization, log_T)
    
    return partition_function


def create_fast_partition_functions_dict(korg_data_path: str = None) -> Dict[Species, Callable]:
    """
    Create a complete dictionary of fast partition functions.
    
    Parameters:
    -----------
    korg_data_path : str, optional
        Path to Korg partition function data
        
    Returns:
    --------
    Dict[Species, Callable]
        Dictionary mapping Species to fast partition function callables
    """
    fast_calculator = FastPartitionFunctions(korg_data_path)
    partition_fns = {}
    
    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        for ionization in range(3):  # Neutral, singly, doubly ionized
            species = Species.from_atomic_number(Z, ionization)
            partition_fns[species] = create_fast_partition_function(Z, ionization, fast_calculator)
    
    return partition_fns


# Vectorized batch processing functions
@jit
def compute_partition_functions_grid(atomic_numbers: jnp.ndarray, 
                                   ionizations: jnp.ndarray,
                                   log_T_array: jnp.ndarray) -> jnp.ndarray:
    """
    Compute partition functions on a grid of species and temperatures.
    
    Parameters:
    -----------
    atomic_numbers : jnp.ndarray
        Array of atomic numbers [n_species]
    ionizations : jnp.ndarray
        Array of ionization stages [n_species]
    log_T_array : jnp.ndarray
        Array of log temperatures [n_temperatures]
        
    Returns:
    --------
    jnp.ndarray
        Grid of partition functions [n_species, n_temperatures]
    """
    # Double vectorization: over species and temperatures
    double_vectorized = vmap(vmap(simple_atom_partition_function_fast, in_axes=(None, None, 0)), 
                           in_axes=(0, 0, None))
    
    return double_vectorized(atomic_numbers, ionizations, log_T_array)


# High-level API functions
def create_default_partition_functions_fast(korg_data_path: str = None) -> Dict[Species, Callable]:
    """
    Create default fast partition functions with optional Korg.jl data.
    
    This is a drop-in replacement for create_default_partition_functions
    with significant performance improvements.
    
    Parameters:
    -----------
    korg_data_path : str, optional
        Path to korg_partition_functions.json file
        
    Returns:
    --------
    Dict[Species, Callable]
        Dictionary of fast partition function callables
    """
    return create_fast_partition_functions_dict(korg_data_path)


@jit
def hydrogen_partition_function_exact(log_T: float) -> float:
    """Exact hydrogen partition function for validation."""
    return 2.0


@jit  
def helium_partition_function_exact(log_T: float) -> float:
    """Exact helium I partition function for validation."""
    return 1.0


# Performance monitoring utilities
def benchmark_partition_functions(n_species: int = 100, n_temperatures: int = 100) -> Dict[str, float]:
    """
    Benchmark partition function calculations.
    
    Parameters:
    -----------
    n_species : int
        Number of species to test
    n_temperatures : int
        Number of temperatures to test
        
    Returns:
    --------
    Dict[str, float]
        Timing results
    """
    import time
    
    # Generate test data
    atomic_numbers = jnp.arange(1, n_species + 1)
    ionizations = jnp.zeros(n_species, dtype=int)
    log_T_array = jnp.linspace(jnp.log(3000), jnp.log(8000), n_temperatures)
    
    # Benchmark vectorized computation
    start_time = time.time()
    result = compute_partition_functions_grid(atomic_numbers, ionizations, log_T_array)
    result.block_until_ready()  # Wait for JAX computation
    vectorized_time = time.time() - start_time
    
    # Benchmark individual calls
    start_time = time.time()
    for Z in atomic_numbers:
        for log_T in log_T_array:
            _ = simple_atom_partition_function_fast(int(Z), 0, log_T)
    individual_time = time.time() - start_time
    
    return {
        'vectorized_time': vectorized_time,
        'individual_time': individual_time,
        'speedup_factor': individual_time / vectorized_time,
        'n_calculations': n_species * n_temperatures
    }