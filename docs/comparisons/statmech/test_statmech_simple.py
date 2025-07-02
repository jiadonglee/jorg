#!/usr/bin/env python3
"""
Simple comparison test between Jorg and Korg.jl statistical mechanics implementations.

Focuses on core functionality with simplified test cases.
"""

import sys
import os
import numpy as np
import time
from pathlib import Path

# Add Jorg to path
jorg_path = Path(__file__).parent / "Jorg" / "src"
sys.path.insert(0, str(jorg_path))

# Test individual components
def test_translational_partition_function():
    """Test translational partition function."""
    print("=== Testing Translational Partition Function ===")
    
    from jorg.statmech.ionization import translational_u
    from jorg.constants import kboltz_cgs, hplanck_cgs, me_cgs
    
    # Test parameters
    masses = [me_cgs, 1.67262e-24]  # electron, proton
    mass_names = ['electron', 'proton']
    temperatures = [3000, 5000, 8000]
    
    for T in temperatures:
        print(f"\nT = {T} K:")
        for mass, name in zip(masses, mass_names):
            # Jorg result
            jorg_result = translational_u(mass, T)
            
            # Reference calculation
            k = kboltz_cgs
            h = hplanck_cgs
            expected = (2.0 * np.pi * mass * k * T / (h * h))**1.5
            
            rel_diff = abs(jorg_result - expected) / expected
            status = "PASS" if rel_diff < 1e-10 else "FAIL"
            
            print(f"  {name:8s}: Jorg={jorg_result:.3e}, Ref={expected:.3e}, "
                  f"RelDiff={rel_diff:.2e} [{status}]")

def test_simple_saha():
    """Test simple Saha equation."""
    print("\n=== Testing Simple Saha Equation ===")
    
    from jorg.statmech.saha_equation import simple_saha_test
    
    # Test conditions
    conditions = [
        {'T': 5778, 'ne': 1e13, 'name': 'Solar photosphere'},
        {'T': 8000, 'ne': 1e14, 'name': 'A star'},
        {'T': 10000, 'ne': 1e15, 'name': 'Hot star'}
    ]
    
    elements = [
        {'Z': 1, 'chi': 13.598, 'name': 'H'},
        {'Z': 2, 'chi': 24.587, 'name': 'He'},
        {'Z': 26, 'chi': 7.902, 'name': 'Fe'}
    ]
    
    for condition in conditions:
        print(f"\n{condition['name']} (T={condition['T']}K, ne={condition['ne']:.0e}):")
        
        for element in elements:
            ratio = simple_saha_test(condition['T'], condition['ne'], 
                                   element['Z'], element['chi'])
            ion_frac = ratio / (1.0 + ratio)
            
            print(f"  {element['name']:2s}: n(X+)/n(X) = {ratio:.3e}, "
                  f"ion_frac = {ion_frac:.3e}")

def test_partition_functions():
    """Test partition functions."""
    print("\n=== Testing Partition Functions ===")
    
    from jorg.statmech.partition_functions import (
        hydrogen_partition_function, simple_partition_function
    )
    
    temperatures = [3000, 5000, 8000]
    elements = [1, 2, 6, 8, 26]  # H, He, C, O, Fe
    element_names = ['H', 'He', 'C', 'O', 'Fe']
    
    for T in temperatures:
        log_T = np.log(T)
        print(f"\nT = {T} K (log_T = {log_T:.3f}):")
        
        # Test hydrogen specifically
        h_partition = hydrogen_partition_function(log_T)
        print(f"  H (specific): {h_partition:.3f}")
        
        # Test general function
        for Z, name in zip(elements, element_names):
            partition = simple_partition_function(Z, log_T)
            print(f"  {name:2s} (Z={Z:2d}): {partition:.3f}")

def test_ionization_energies():
    """Test ionization energies data."""
    print("\n=== Testing Ionization Energies Data ===")
    
    from jorg.statmech.saha_equation import create_default_ionization_energies
    
    ionization_energies = create_default_ionization_energies()
    
    print("Sample ionization energies (eV):")
    print("Element    χI        χII       χIII")
    print("-" * 35)
    
    elements = [1, 2, 6, 7, 8, 11, 12, 13, 14, 16, 20, 26]
    element_names = ['H', 'He', 'C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'S', 'Ca', 'Fe']
    
    for Z, name in zip(elements, element_names):
        if Z in ionization_energies:
            chi_I, chi_II, chi_III = ionization_energies[Z]
            print(f"{name:7s}    {chi_I:7.3f}   {chi_II:7.3f}   {chi_III:7.3f}")

def test_constants():
    """Test physical constants."""
    print("\n=== Testing Physical Constants ===")
    
    from jorg.constants import kboltz_cgs, hplanck_cgs, me_cgs, EV_TO_ERG
    
    # Compare with CODATA 2018 values
    constants = {
        'kboltz_cgs': {'jorg': kboltz_cgs, 'codata': 1.380649e-16},
        'hplanck_cgs': {'jorg': hplanck_cgs, 'codata': 6.62607015e-27},
        'me_cgs': {'jorg': me_cgs, 'codata': 9.1093837015e-28},
        'EV_TO_ERG': {'jorg': EV_TO_ERG, 'codata': 1.602176634e-12}
    }
    
    print("Constant           Jorg Value        CODATA Value      Rel. Diff")
    print("-" * 65)
    
    for name, values in constants.items():
        jorg_val = float(values['jorg'])
        codata_val = values['codata']
        rel_diff = abs(jorg_val - codata_val) / codata_val
        
        print(f"{name:15s}   {jorg_val:.6e}   {codata_val:.6e}   {rel_diff:.2e}")

def compare_with_korg_values():
    """Compare specific values with expected Korg.jl results."""
    print("\n=== Comparing with Expected Korg.jl Values ===")
    
    # These are reference values that should match Korg.jl
    # Based on solar photosphere conditions (T=5778K, ne=1e13)
    reference_tests = [
        {
            'description': 'Hydrogen ionization (solar)',
            'T': 5778.0,
            'ne': 1e13,
            'Z': 1,
            'chi': 13.598,
            'expected_ion_frac': 1.5e-4,  # Approximate
            'tolerance': 0.5  # 50% tolerance
        },
        {
            'description': 'Iron ionization (solar)',
            'T': 5778.0,
            'ne': 1e13,
            'Z': 26,
            'chi': 7.902,
            'expected_ion_frac': 0.9,  # Most Fe should be ionized
            'tolerance': 0.2  # 20% tolerance
        }
    ]
    
    from jorg.statmech.saha_equation import simple_saha_test
    
    for test in reference_tests:
        print(f"\n{test['description']}:")
        
        ratio = simple_saha_test(test['T'], test['ne'], test['Z'], test['chi'])
        ion_frac = ratio / (1.0 + ratio)
        
        expected = test['expected_ion_frac']
        rel_diff = abs(ion_frac - expected) / expected
        
        status = "PASS" if rel_diff < test['tolerance'] else "FAIL"
        
        print(f"  Calculated: {ion_frac:.3e}")
        print(f"  Expected:   {expected:.3e}")
        print(f"  Rel. diff:  {rel_diff:.3f}")
        print(f"  Status:     {status}")

def main():
    """Run all simple tests."""
    print("=" * 60)
    print("SIMPLE JORG STATISTICAL MECHANICS TESTS")
    print("=" * 60)
    
    try:
        test_constants()
        test_ionization_energies()
        test_translational_partition_function()
        test_partition_functions()
        test_simple_saha()
        compare_with_korg_values()
        
        print("\n" + "=" * 60)
        print("ALL SIMPLE TESTS COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running tests: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()