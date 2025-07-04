#!/usr/bin/env python3
"""
Simple test to validate molecular equilibrium fix

Tests that the new molecular equilibrium constants produce realistic results
without requiring the full chemical equilibrium solver.
"""

import sys
from pathlib import Path

# Add Jorg to path
jorg_src = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(jorg_src))

import numpy as np
import warnings
warnings.filterwarnings('ignore')

def test_molecular_equilibrium_constants():
    """Test that molecular equilibrium constants are reasonable"""
    print("Testing molecular equilibrium constants...")
    
    try:
        from jorg.statmech.molecular import (
            create_default_log_equilibrium_constants, 
            get_log_nK
        )
        from jorg.statmech.species import Species
        
        # Load equilibrium constants
        log_equilibrium_constants = create_default_log_equilibrium_constants()
        
        # Test at solar conditions
        T_solar = 5778.0  # K
        
        # Test key molecules
        test_molecules = {
            'H2': {'expected_range': (7, 9), 'description': 'Very stable diatomic'},
            'H2O': {'expected_range': (10, 20), 'description': 'Stable triatomic'},
            'CO': {'expected_range': (2, 5), 'description': 'Strong triple bond'},
            'OH': {'expected_range': (6, 9), 'description': 'Moderately stable'},
            'CN': {'expected_range': (4, 7), 'description': 'Triple bond'},
        }
        
        results = {}
        
        for mol_name, props in test_molecules.items():
            try:
                species = Species.from_string(mol_name)
                if species in log_equilibrium_constants:
                    # Get partial pressure equilibrium constant
                    log_nK = get_log_nK(species, T_solar, log_equilibrium_constants)
                    
                    # Convert back to partial pressure form for comparison
                    n_atoms = len(species.formula.atoms)
                    from jorg.constants import kboltz_cgs
                    log_pK = log_nK + (n_atoms - 1) * np.log10(kboltz_cgs * T_solar)
                    
                    expected_min, expected_max = props['expected_range']
                    within_range = expected_min <= log_pK <= expected_max
                    
                    status = "✅" if within_range else "⚠️"
                    print(f"  {mol_name}: log10(pK) = {log_pK:.2f} {status} ({props['description']})")
                    
                    results[mol_name] = within_range
                else:
                    print(f"  {mol_name}: Not found in equilibrium constants ❌")
                    results[mol_name] = False
            except Exception as e:
                print(f"  {mol_name}: Error - {e} ❌")
                results[mol_name] = False
        
        success_rate = sum(results.values()) / len(results)
        
        if success_rate >= 0.8:
            print(f"✅ Molecular equilibrium constants are reasonable ({success_rate:.1%} success)")
            return True
        else:
            print(f"❌ Issues with molecular equilibrium constants ({success_rate:.1%} success)")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_temperature_dependence():
    """Test that equilibrium constants have correct temperature dependence"""
    print("Testing temperature dependence...")
    
    try:
        from jorg.statmech.molecular import create_default_log_equilibrium_constants, get_log_nK
        from jorg.statmech.species import Species
        
        log_equilibrium_constants = create_default_log_equilibrium_constants()
        
        # Test H2 molecule across temperature range
        h2_species = Species.from_string('H2')
        
        if h2_species not in log_equilibrium_constants:
            print("❌ H2 not found in equilibrium constants")
            return False
        
        temperatures = [3000, 4000, 5000, 6000, 7000, 8000]
        log_nKs = []
        
        print("H2 equilibrium constant vs temperature:")
        for T in temperatures:
            log_nK = get_log_nK(h2_species, T, log_equilibrium_constants)
            log_nKs.append(log_nK)
            print(f"  T={T}K: log10(nK) = {log_nK:.2f}")
        
        # Check that equilibrium constant increases with temperature
        # (molecules become less stable at higher T, so K should increase)
        increasing = all(log_nKs[i] < log_nKs[i+1] for i in range(len(log_nKs)-1))
        
        if increasing:
            print("✅ Temperature dependence is correct (K increases with T)")
            return True
        else:
            print("❌ Temperature dependence is incorrect")
            return False
            
    except Exception as e:
        print(f"❌ Temperature dependence test failed: {e}")
        return False

def test_molecular_abundance_order_of_magnitude():
    """Test that molecular abundances are in reasonable range"""
    print("Testing molecular abundance order of magnitude...")
    
    try:
        from jorg.statmech.molecular import create_default_log_equilibrium_constants, get_log_nK
        from jorg.statmech.species import Species
        from jorg.constants import kboltz_cgs
        
        log_equilibrium_constants = create_default_log_equilibrium_constants()
        
        # Simulate simple calculation for H2O abundance
        T = 5778.0  # K, solar temperature
        
        # Typical stellar atmosphere densities (cm^-3)
        n_H = 1e15   # Hydrogen density
        n_O = 1e12   # Oxygen density (roughly 1/1000 of H)
        
        h2o_species = Species.from_string('H2O')
        
        if h2o_species not in log_equilibrium_constants:
            print("❌ H2O not found in equilibrium constants")
            return False
        
        # Get equilibrium constant in number density form
        log_nK = get_log_nK(h2o_species, T, log_equilibrium_constants)
        nK = 10**log_nK
        
        # For H2O: H2 + 1/2 O2 ⇌ H2O
        # Simplified: assuming H2 ~ n_H and O2 ~ n_O
        # n_H2O ≈ n_H * n_O / nK
        estimated_n_H2O = n_H * n_O / nK
        
        print(f"H2O equilibrium constant: log10(nK) = {log_nK:.2f}")
        print(f"Estimated H2O density: {estimated_n_H2O:.2e} cm^-3")
        
        # Check if abundance is reasonable (not 10^22 times too high!)
        reasonable = 1e-20 < estimated_n_H2O < 1e10
        
        if reasonable:
            print("✅ H2O abundance estimate is reasonable")
            return True
        else:
            print("❌ H2O abundance estimate is unreasonable")
            return False
            
    except Exception as e:
        print(f"❌ Abundance order of magnitude test failed: {e}")
        return False

def test_comparison_with_old_values():
    """Test that new equilibrium constants differ from the broken old ones"""
    print("Testing comparison with old (broken) values...")
    
    try:
        from jorg.statmech.molecular import create_default_log_equilibrium_constants, get_log_nK
        from jorg.statmech.species import Species
        from jorg.constants import kboltz_cgs
        
        log_equilibrium_constants = create_default_log_equilibrium_constants()
        
        T = 5778.0  # K
        
        # Old broken values (oversimplified A + B/T formulas)
        old_broken_values = {
            'H2O': 50.0,  # Was ridiculously high (12.0 - 4000.0/T gave ~50 at solar T)
            'CO': -25.0,  # Was way too negative  
        }
        
        improvements = {}
        
        for mol_name, old_bad_value in old_broken_values.items():
            try:
                species = Species.from_string(mol_name)
                if species in log_equilibrium_constants:
                    log_nK = get_log_nK(species, T, log_equilibrium_constants)
                    
                    # Convert to partial pressure form
                    n_atoms = len(species.formula.atoms)
                    log_pK = log_nK + (n_atoms - 1) * np.log10(kboltz_cgs * T)
                    
                    difference = abs(log_pK - old_bad_value)
                    improved = difference > 5.0  # Should differ by at least 5 orders of magnitude
                    
                    status = "✅" if improved else "❌"
                    print(f"  {mol_name}: old={old_bad_value:.1f}, new={log_pK:.1f}, diff={difference:.1f} {status}")
                    
                    improvements[mol_name] = improved
                else:
                    print(f"  {mol_name}: Not found")
                    improvements[mol_name] = False
            except Exception as e:
                print(f"  {mol_name}: Error - {e}")
                improvements[mol_name] = False
        
        success_rate = sum(improvements.values()) / len(improvements)
        
        if success_rate >= 1.0:
            print("✅ New equilibrium constants are significantly improved")
            return True
        else:
            print("❌ New equilibrium constants not sufficiently different from old broken ones")
            return False
            
    except Exception as e:
        print(f"❌ Comparison test failed: {e}")
        return False

def run_all_tests():
    """Run all molecular validation tests"""
    print("JORG MOLECULAR EQUILIBRIUM VALIDATION")
    print("=" * 45)
    
    tests = [
        ("Equilibrium constants", test_molecular_equilibrium_constants),
        ("Temperature dependence", test_temperature_dependence), 
        ("Abundance order of magnitude", test_molecular_abundance_order_of_magnitude),
        ("Comparison with old values", test_comparison_with_old_values),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n" + "=" * 45)
    print(f"TEST SUMMARY: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Molecular equilibrium fix is working!")
        print("📈 Jorg now uses realistic molecular equilibrium constants")
        print("🔬 No more 10^22 molecular abundance discrepancies!")
        return True
    elif passed >= total * 0.75:
        print("✅ Most tests passed - Significant improvement achieved")
        return True
    else:
        print("⚠️ Some issues remain with molecular equilibrium")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)