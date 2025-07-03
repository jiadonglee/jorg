#!/usr/bin/env python3
"""Comprehensive test of chemical equilibrium"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Jorg/src'))

from jorg.statmech import chemical_equilibrium
from jorg.statmech.species import Species
from jorg.statmech.saha_equation import create_default_ionization_energies
from jorg.statmech.partition_functions import create_default_partition_functions
from jorg.statmech.molecular import create_default_log_equilibrium_constants
from jorg.abundances import format_A_X

def test_chemical_equilibrium_comprehensive():
    """Comprehensive test with proper error handling"""
    
    print("Comprehensive Chemical Equilibrium Test")
    print("=" * 50)
    
    # Load data
    ionization_energies = create_default_ionization_energies()
    partition_fns = create_default_partition_functions()
    log_equilibrium_constants = create_default_log_equilibrium_constants()
    
    # Get realistic abundances
    A_X = format_A_X()
    absolute_abundances = {}
    total = 0.0
    
    key_elements = [1, 2, 6, 7, 8, 26]  # H, He, C, N, O, Fe
    for Z in key_elements:
        if Z in A_X:
            linear_ab = 10**(A_X[Z] - 12.0)
            absolute_abundances[Z] = linear_ab
            total += linear_ab
    
    for Z in absolute_abundances:
        absolute_abundances[Z] /= total
    
    print(f"Abundances: {[(Z, f'{ab:.2e}') for Z, ab in absolute_abundances.items()]}")
    
    # Test cases
    test_cases = [
        ("Solar", 5778.0, 1e15, 1e12),
        ("Cool", 4000.0, 1e15, 1e9),
        ("Hot", 8000.0, 1e15, 1e13),
    ]
    
    results = []
    
    for case_name, temp, nt, ne_guess in test_cases:
        print(f"\n=== {case_name} Test ===")
        print(f"T={temp}K, nt={nt:.1e}, ne_guess={ne_guess:.1e}")
        
        try:
            ne_calc, densities = chemical_equilibrium(
                temp, nt, ne_guess, absolute_abundances,
                ionization_energies, partition_fns, log_equilibrium_constants
            )
            
            # Extract results
            h1 = densities.get(Species.from_atomic_number(1, 0), 0)
            h2 = densities.get(Species.from_atomic_number(1, 1), 0)
            fe1 = densities.get(Species.from_atomic_number(26, 0), 0)
            fe2 = densities.get(Species.from_atomic_number(26, 1), 0)
            
            h_total = h1 + h2
            h_ion_frac = h2 / h_total if h_total > 0 else 0
            
            # Charge balance check
            total_charge = sum(charge * densities[Species.from_atomic_number(Z, charge)]
                             for Z in range(1, 30) for charge in range(1, 3)
                             if Species.from_atomic_number(Z, charge) in densities)
            
            charge_error = abs(ne_calc - total_charge) / ne_calc * 100 if ne_calc > 0 else 100
            
            print(f"✅ SUCCESS:")
            print(f"  ne: {ne_calc:.3e} cm^-3")
            print(f"  H ionization: {h_ion_frac:.6e}")
            print(f"  H I: {h1:.3e} cm^-3")
            print(f"  H II: {h2:.3e} cm^-3")
            print(f"  Fe I: {fe1:.3e} cm^-3")
            print(f"  Fe II: {fe2:.3e} cm^-3")
            print(f"  Charge error: {charge_error:.1f}%")
            
            results.append({
                'case': case_name,
                'success': True,
                'ne': ne_calc,
                'h_ionization': h_ion_frac,
                'charge_error': charge_error
            })
            
        except Exception as e:
            print(f"❌ FAILED: {e}")
            results.append({
                'case': case_name,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY:")
    successful = sum(1 for r in results if r.get('success', False))
    total_tests = len(results)
    
    print(f"Success rate: {successful}/{total_tests}")
    
    if successful > 0:
        print("\nSuccessful cases:")
        for result in results:
            if result.get('success'):
                case = result['case']
                ne = result['ne']
                h_ion = result['h_ionization']
                charge_err = result['charge_error']
                print(f"  {case:8} ne={ne:.2e}  H_ion={h_ion:.2e}  charge_err={charge_err:.1f}%")
    
    return successful == total_tests

if __name__ == "__main__":
    success = test_chemical_equilibrium_comprehensive()
    if success:
        print("\n🎯 ALL TESTS PASSED!")
    else:
        print("\n⚠️ Some tests failed")
    sys.exit(0 if success else 1)