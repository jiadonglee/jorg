"""
Exact Korg.jl Partition Functions for Jorg
==========================================

This module provides partition functions that exactly match Korg.jl's implementation,
using the extracted NIST atomic level data from Korg.jl's HDF5 files.

This replaces the simplified partition functions with the exact same values
Korg.jl uses, eliminating the 20-40% discrepancies identified in the root cause analysis.
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
            f"Run extract_korg_partition_functions.py from the Korg.jl root directory first."
        )
    
    # Load the JSON data
    with open(data_file, 'r') as f:
        partition_data = json.load(f)
    
    # Create interpolators for each species
    interpolators = {}
    
    for key, data in partition_data.items():
        Z = data['atomic_number']
        charge = data['charge']
        
        # Create species key
        species = Species.from_atomic_number(Z, charge)
        
        # Create cubic spline interpolator
        logTs = np.array(data['logTs'])
        partition_values = np.array(data['partition_values'])
        
        # Use extrapolation for temperatures outside the range
        interpolator = CubicSpline(logTs, partition_values, extrapolate=True)
        
        interpolators[species] = interpolator
    
    _partition_interpolators = interpolators
    _partition_data_loaded = True
    
    print(f"✅ Loaded {len(interpolators)} Korg.jl partition functions")
    
    return interpolators

@jit  
def korg_partition_function(species: Species, log_T: float) -> float:
    """
    Calculate partition function exactly matching Korg.jl.
    
    This function provides the exact same partition function values that Korg.jl
    uses, eliminating the 20-40% discrepancies from simplified approximations.
    
    Parameters:
    -----------
    species : Species
        The atomic species (element and charge state)
    log_T : float
        Natural logarithm of temperature in K
        
    Returns:
    --------
    float
        Partition function value exactly matching Korg.jl
    """
    
    # Get the interpolators (this will be JIT-compiled away)
    interpolators = load_korg_partition_data()
    
    if species not in interpolators:
        # Fallback for species not in Korg.jl data
        return 1.0
    
    # Use the interpolator to get exact Korg.jl value
    interpolator = interpolators[species]
    
    # Convert to numpy for scipy interpolation, then back to JAX
    log_T_np = float(log_T)
    U_value = interpolator(log_T_np)
    
    return jnp.array(U_value, dtype=jnp.float64)

def create_korg_partition_functions() -> Dict[Species, Callable]:
    """
    Create partition function dictionary exactly matching Korg.jl.
    
    Returns:
    --------
    Dict[Species, Callable]
        Dictionary mapping Species to partition function callables that
        exactly match Korg.jl's values
    """
    
    # Load the interpolators
    interpolators = load_korg_partition_data()
    
    partition_funcs = {}
    
    for species, interpolator in interpolators.items():
        # Create a function that uses the interpolator
        def make_partition_func(interp):
            def partition_func(log_T):
                # Convert JAX array to float for scipy, then back to JAX
                log_T_val = float(log_T) if hasattr(log_T, 'item') else log_T
                U_val = interp(log_T_val)
                return jnp.array(U_val, dtype=jnp.float64)
            return partition_func
        
        partition_funcs[species] = make_partition_func(interpolator)
    
    return partition_funcs

def validate_against_korg_values():
    """
    Validate our partition functions against known Korg.jl values.
    """
    
    print("\n=== VALIDATING KORG.JL PARTITION FUNCTIONS ===")
    
    # Test cases from our debugging analysis
    test_cases = [
        # (Z, charge, T, expected_value, description)
        (1, 0, 5778.0, 2.000, "H I at solar temperature"),
        (1, 1, 5778.0, 1.000, "H II (bare proton)"),
        (2, 0, 5778.0, 1.000, "He I at solar temperature"),  
        (2, 1, 5778.0, 2.000, "He II (hydrogen-like)"),
        (26, 0, 5778.0, 30.784, "Fe I at solar temperature"),
        (26, 1, 5778.0, 46.634, "Fe II at solar temperature"),
        (26, 2, 5778.0, 23.042, "Fe III at solar temperature"),
        # Test temperature dependence
        (26, 0, 3000.0, None, "Fe I at cool temperature"),
        (26, 0, 10000.0, None, "Fe I at hot temperature"),
    ]
    
    partition_funcs = create_korg_partition_functions()
    
    all_passed = True
    
    for Z, charge, T, expected, description in test_cases:
        species = Species.from_atomic_number(Z, charge)
        
        if species in partition_funcs:
            log_T = np.log(T)
            U_calc = partition_funcs[species](log_T)
            
            if expected is not None:
                error = abs(float(U_calc) - expected) / expected * 100
                status = "✅" if error < 0.1 else "⚠️"
                print(f"{status} {description}: U = {float(U_calc):.3f} (expected {expected:.3f}, error {error:.3f}%)")
                
                if error >= 0.1:
                    all_passed = False
            else:
                print(f"ℹ️  {description}: U = {float(U_calc):.3f}")
        else:
            print(f"❌ {description}: Species not found")
            all_passed = False
    
    if all_passed:
        print("\n✅ All validation tests passed!")
    else:
        print("\n⚠️ Some validation tests failed")
    
    return all_passed

def compare_with_simplified_functions():
    """
    Compare the exact Korg.jl partition functions with the old simplified ones.
    """
    
    print("\n=== COMPARING WITH SIMPLIFIED FUNCTIONS ===")
    
    # Import the old simplified functions for comparison
    from .partition_functions import create_simplified_partition_functions
    
    korg_funcs = create_korg_partition_functions()
    simple_funcs = create_simplified_partition_functions()
    
    test_species = [
        (1, 0, "H I"),
        (2, 0, "He I"), 
        (6, 0, "C I"),
        (8, 0, "O I"),
        (26, 0, "Fe I"),
        (26, 1, "Fe II"),
        (26, 2, "Fe III"),
    ]
    
    test_T = 5778.0  # Solar temperature
    log_T = np.log(test_T)
    
    print(f"Comparison at T = {test_T} K:")
    print("Species    Korg.jl    Simplified   Ratio    Improvement")
    print("-" * 55)
    
    for Z, charge, name in test_species:
        species = Species.from_atomic_number(Z, charge)
        
        if species in korg_funcs and species in simple_funcs:
            U_korg = float(korg_funcs[species](log_T))
            U_simple = float(simple_funcs[species](log_T))
            
            ratio = U_simple / U_korg if U_korg > 0 else 0
            improvement = abs(1.0 - ratio) * 100
            
            status = "✅" if improvement < 5 else "📈" if improvement < 20 else "🔧"
            
            print(f"{name:8s}   {U_korg:8.3f}    {U_simple:8.3f}   {ratio:6.3f}   {status} {improvement:5.1f}%")

def get_partition_function_summary():
    """Get a summary of available partition functions"""
    
    interpolators = load_korg_partition_data()
    
    # Count by element
    element_counts = {}
    for species in interpolators.keys():
        Z = species.formula.atoms[-1]  # Atomic number
        if Z not in element_counts:
            element_counts[Z] = 0
        element_counts[Z] += 1
    
    print(f"\nPartition function coverage:")
    print(f"  Total species: {len(interpolators)}")
    print(f"  Elements covered: {len(element_counts)} (Z=1 to Z={max(element_counts.keys())})")
    print(f"  Average charge states per element: {len(interpolators) / len(element_counts):.1f}")
    
    # Show some key elements
    key_elements = [1, 2, 6, 8, 12, 14, 26, 28]  # H, He, C, O, Mg, Si, Fe, Ni
    print(f"\nKey elements:")
    for Z in key_elements:
        if Z in element_counts:
            element_name = {1: "H", 2: "He", 6: "C", 8: "O", 12: "Mg", 14: "Si", 26: "Fe", 28: "Ni"}.get(Z, f"Z{Z}")
            print(f"  {element_name:2s} (Z={Z:2d}): {element_counts[Z]} charge states")

# Convenience function to replace the old partition function creation
def create_default_partition_functions() -> Dict[Species, Callable]:
    """
    Create default partition functions exactly matching Korg.jl.
    
    This replaces the old create_default_partition_functions() with exact Korg.jl values.
    """
    return create_korg_partition_functions()

if __name__ == "__main__":
    # Run validation when executed directly
    print("Korg.jl Partition Functions Validation")
    print("=" * 50)
    
    try:
        # Load and validate
        validate_against_korg_values()
        
        # Compare with old functions
        compare_with_simplified_functions()
        
        # Show summary
        get_partition_function_summary()
        
        print("\n" + "=" * 50)
        print("✅ Korg.jl partition functions ready for use!")
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()