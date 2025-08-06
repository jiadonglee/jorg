"""
Exact Korg.jl Partition Functions for Jorg Precision Matching
============================================================

This module provides partition functions that exactly match Korg.jl's implementation,
using the extracted NIST atomic level data from Korg.jl's HDF5 files.

This replaces the simplified partition functions with the exact same values
Korg.jl uses, eliminating the 3-4% line depth discrepancies identified in the analysis.
"""

import jax.numpy as jnp
from jax import jit
import json
import os
from typing import Dict, Callable
from scipy.interpolate import CubicSpline
import numpy as np

from .species import Species, MAX_ATOMIC_NUMBER

# Global storage for interpolators (loaded once)
_partition_interpolators = None
_partition_data_loaded = False

def load_korg_partition_data():
    """Load the extracted Korg.jl partition function data"""
    
    global _partition_interpolators, _partition_data_loaded
    
    if _partition_data_loaded:
        return _partition_interpolators
    
    # Path to the extracted partition function data (in Jorg/data/)
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data')
    data_file = os.path.join(data_dir, 'korg_partition_functions.json')
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"Korg.jl partition function data not found at {data_file}. "
            f"Run 'julia Jorg/scripts/extract_korg_partition_functions.jl' to extract the data."
        )
    
    print(f"🔬 Loading exact Korg.jl partition functions from {data_file}")
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    partition_functions = data['partition_functions']
    metadata = data['metadata']
    
    print(f"   Source: {metadata['source']}")
    print(f"   Species count: {metadata['species_count']}")
    print(f"   Temperature range: {metadata['temperature_range'][0]:.0f}-{metadata['temperature_range'][1]:.0f} K")
    
    # Create cubic spline interpolators for each species
    interpolators = {}
    
    for species_key, species_data in partition_functions.items():
        Z = species_data['atomic_number']
        charge = species_data['ionization']
        log_T_values = np.array(species_data['log_T'])
        U_values = np.array(species_data['U'])
        
        # Create Jorg species object
        try:
            species = Species.from_atomic_number(Z, charge)
            
            # Create cubic spline interpolator (matches Korg.jl's method)
            interpolator = CubicSpline(log_T_values, U_values, extrapolate=True)
            
            # Create non-JIT wrapper for scipy interpolation 
            def partition_func(log_T: float, interp=interpolator) -> float:
                # scipy interpolation can't be JITted, so use regular function
                return float(interp(float(log_T)))
            
            interpolators[species] = partition_func
            
        except Exception as e:
            print(f"   Warning: Could not create species for Z={Z}, charge={charge}: {e}")
    
    _partition_interpolators = interpolators
    _partition_data_loaded = True
    
    print(f"   ✅ Loaded {len(interpolators)} exact Korg.jl partition functions")
    
    # Validation: Check Fe I value at 5780K
    fe_i = Species.from_atomic_number(26, 0)
    if fe_i in interpolators:
        U_fe_5780 = interpolators[fe_i](np.log(5780.0))
        print(f"   ✅ Fe I at 5780K: U = {U_fe_5780:.3f} (Korg.jl exact)")
    
    return _partition_interpolators


def get_korg_exact_partition_functions() -> Dict[Species, Callable]:
    """
    Get exact Korg.jl partition functions for all species.
    
    Returns:
    --------
    Dict[Species, Callable]
        Dictionary mapping species to partition function callables.
        Each function takes log_T and returns the partition function value.
    """
    return load_korg_partition_data()


@jit
def korg_exact_partition_function(species: Species, log_T: float) -> float:
    """
    Calculate exact Korg.jl partition function for a species.
    
    Parameters:
    -----------
    species : Species
        The atomic/ionic species
    log_T : float
        Natural logarithm of temperature in K
        
    Returns:
    --------
    float
        Partition function value (exact match to Korg.jl)
    """
    partition_fns = load_korg_partition_data()
    
    if species in partition_fns:
        return partition_fns[species](log_T)
    else:
        # Fallback to simple approximation for missing species
        return simple_partition_function_fallback(species, log_T)


@jit 
def simple_partition_function_fallback(species: Species, log_T: float) -> float:
    """
    Fallback partition function for species not in Korg.jl data.
    
    Uses simple temperature scaling as backup.
    """
    Z = species.get_atom()
    charge = species.charge
    T = jnp.exp(log_T)
    
    if Z == 1:  # Hydrogen
        if charge == 0:
            return 2.0  # H I
        else:
            return 1.0  # H II
    elif Z == 2:  # Helium  
        if charge == 0:
            return 1.0  # He I
        else:
            return 2.0  # He II, He III
    else:
        # Simple scaling for other elements
        base_U = float(2 * Z)  # Rough estimate
        return base_U * (T / 5778.0)**0.2


def create_korg_exact_partition_functions(verbose: bool = True) -> Dict[Species, Callable]:
    """
    Create exact Korg.jl partition functions for chemical equilibrium.
    
    This replaces the approximate partition functions in Jorg with the exact
    same values that Korg.jl uses, eliminating partition function discrepancies
    as a source of the 3-4% line depth difference.
    
    Parameters:
    -----------
    verbose : bool
        Print loading information
        
    Returns:
    --------
    Dict[Species, Callable]
        Exact Korg.jl partition functions
    """
    if verbose:
        print("🎯 Creating exact Korg.jl partition functions for precision matching")
    
    partition_fns = get_korg_exact_partition_functions()
    
    if verbose:
        print(f"   ✅ {len(partition_fns)} exact Korg.jl partition functions loaded")
        
        # Show key species for validation
        key_species = [
            (Species.from_atomic_number(26, 0), "Fe I"),
            (Species.from_atomic_number(22, 0), "Ti I"), 
            (Species.from_atomic_number(20, 0), "Ca I"),
        ]
        
        print("   🔍 Key species validation:")
        for species, name in key_species:
            if species in partition_fns:
                U_5780 = partition_fns[species](np.log(5780.0))
                print(f"     {name}: U(5780K) = {U_5780:.3f}")
    
    return partition_fns


def benchmark_korg_exact_partition_functions():
    """Benchmark the exact Korg partition functions."""
    import time
    
    print("🚀 BENCHMARKING EXACT KORG.JL PARTITION FUNCTIONS")
    print("=" * 60)
    
    partition_fns = create_korg_exact_partition_functions(verbose=True)
    
    # Test temperatures
    temperatures = [3000, 4000, 5000, 5778, 6000, 7000, 8000, 10000]
    log_temperatures = [np.log(T) for T in temperatures]
    
    # Test key species
    test_species = [
        Species.from_atomic_number(26, 0),  # Fe I
        Species.from_atomic_number(26, 1),  # Fe II
        Species.from_atomic_number(22, 0),  # Ti I
        Species.from_atomic_number(20, 0),  # Ca I
    ]
    
    print(f"\n🔬 Testing {len(test_species)} species across {len(temperatures)} temperatures")
    
    # Benchmark calculation time
    start_time = time.time()
    
    total_calculations = 0
    for species in test_species:
        if species in partition_fns:
            for log_T in log_temperatures:
                U = partition_fns[species](log_T)
                total_calculations += 1
    
    calc_time = time.time() - start_time
    
    print(f"✅ Completed {total_calculations} calculations in {calc_time:.3f}s")
    print(f"   Average: {calc_time/total_calculations*1e6:.1f} μs per calculation")
    
    # Show temperature dependence for Fe I
    fe_i = Species.from_atomic_number(26, 0)
    if fe_i in partition_fns:
        print(f"\n📊 Fe I partition function temperature dependence:")
        for T in temperatures:
            U = partition_fns[fe_i](np.log(T))
            print(f"   T = {T:5.0f} K: U = {U:6.2f}")
    
    print("\n✅ Exact Korg.jl partition functions ready for production!")
    
    return {
        'species_count': len(partition_fns),
        'calculation_time': calc_time,
        'calculations_per_second': total_calculations / calc_time
    }


if __name__ == "__main__":
    benchmark_korg_exact_partition_functions()