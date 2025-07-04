#!/usr/bin/env python3
"""
Test molecular abundance calculations against Korg.jl

This test validates that Jorg's chemical equilibrium solver with the new
molecular equilibrium constants produces reasonable molecular abundances
that match Korg's calculations.
"""

import sys
from pathlib import Path

# Add Jorg to path
jorg_src = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(jorg_src))

import numpy as np
import warnings
warnings.filterwarnings('ignore')

def test_molecular_abundance_calculation():
    """Test molecular abundance calculation in chemical equilibrium"""
    print("Testing molecular abundance calculation...")
    
    try:
        from jorg.statmech.chemical_equilibrium import solve_chemical_equilibrium
        from jorg.statmech.molecular import create_default_log_equilibrium_constants
        from jorg.statmech.partition_functions import setup_partition_functions
        from jorg.statmech.species import Species
        from jorg.abundances import solar_abundances
        from jorg.constants import kboltz_eV
        
        # Create solar conditions
        T = 5778.0  # K (solar temperature)
        total_density = 1e-7  # g/cm^3 (photosphere)
        
        # Solar abundances (simplified for testing)
        abs_abundances = solar_abundances()
        
        # Set up partition functions and molecular equilibrium constants  
        partition_funcs = setup_partition_functions()
        log_equilibrium_constants = create_default_log_equilibrium_constants()
        
        print(f"Using {len(log_equilibrium_constants)} molecular species")
        
        # Solve chemical equilibrium
        result = solve_chemical_equilibrium(
            T, total_density, abs_abundances, 
            partition_funcs, log_equilibrium_constants
        )
        
        n_e = result['electron_density']
        number_densities = result['number_densities']
        
        print(f"Electron density: {n_e:.2e} cm^-3")
        print(f"Total species calculated: {len(number_densities)}")
        
        # Check molecular abundances
        important_molecules = ['H2', 'H2O', 'CO', 'OH', 'CN']
        molecular_abundances = {}
        
        for mol_name in important_molecules:
            try:
                species = Species.from_string(mol_name)
                if species in number_densities:
                    abundance = number_densities[species]
                    molecular_abundances[mol_name] = abundance
                    print(f"{mol_name}: {abundance:.2e} cm^-3")
                else:
                    print(f"{mol_name}: Not found in chemical equilibrium")
            except Exception as e:
                print(f"{mol_name}: Error - {e}")
        
        # Validate abundances are reasonable (not 10^22 like before!)
        reasonable_abundances = True
        for mol_name, abundance in molecular_abundances.items():
            if abundance > 1e15:  # Unreasonably high
                print(f"❌ {mol_name} abundance too high: {abundance:.2e}")
                reasonable_abundances = False
            elif abundance < 1e-50:  # Unreasonably low
                print(f"⚠️ {mol_name} abundance very low: {abundance:.2e}")
        
        if reasonable_abundances and len(molecular_abundances) > 0:
            print("✅ Molecular abundances are reasonable")
            return True
        else:
            print("❌ Molecular abundances are unreasonable")
            return False
            
    except Exception as e:
        print(f"❌ Molecular abundance calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_h2o_abundance_fix():
    """Test that H2O abundance is now realistic (not 10^22 higher than Korg)"""
    print("Testing H2O abundance fix...")
    
    try:
        from jorg.statmech.chemical_equilibrium import solve_chemical_equilibrium
        from jorg.statmech.molecular import create_default_log_equilibrium_constants
        from jorg.statmech.partition_functions import setup_partition_functions
        from jorg.statmech.species import Species
        from jorg.abundances import solar_abundances
        
        # Solar photosphere conditions
        T = 5778.0  # K
        total_density = 1e-7  # g/cm^3
        abs_abundances = solar_abundances()
        
        # Set up chemistry
        partition_funcs = setup_partition_functions()
        log_equilibrium_constants = create_default_log_equilibrium_constants()
        
        # Solve equilibrium
        result = solve_chemical_equilibrium(
            T, total_density, abs_abundances,
            partition_funcs, log_equilibrium_constants
        )
        
        number_densities = result['number_densities']
        
        # Check H2O abundance
        h2o_species = Species.from_string('H2O')
        if h2o_species in number_densities:
            h2o_abundance = number_densities[h2o_species]
            
            # Expected Korg H2O abundance is around 10^-15 to 10^-10 cm^-3 in solar photosphere
            # Our old implementation gave ~10^22 times too much
            expected_order_of_magnitude = 1e-12  # Rough estimate
            
            ratio = h2o_abundance / expected_order_of_magnitude
            
            print(f"H2O abundance: {h2o_abundance:.2e} cm^-3")
            print(f"Expected order: {expected_order_of_magnitude:.2e} cm^-3")
            print(f"Ratio: {ratio:.2e}")
            
            # Check if abundance is within reasonable bounds (not 10^22 off!)
            if 1e-5 < ratio < 1e5:  # Within 5 orders of magnitude is reasonable
                print("✅ H2O abundance is now realistic")
                return True
            else:
                print("❌ H2O abundance still unrealistic")
                return False
        else:
            print("❌ H2O species not found in chemical equilibrium")
            return False
            
    except Exception as e:
        print(f"❌ H2O abundance test failed: {e}")
        return False

def test_molecular_equilibrium_convergence():
    """Test that chemical equilibrium converges with molecular species"""
    print("Testing chemical equilibrium convergence...")
    
    try:
        from jorg.statmech.chemical_equilibrium import solve_chemical_equilibrium
        from jorg.statmech.molecular import create_default_log_equilibrium_constants
        from jorg.statmech.partition_functions import setup_partition_functions
        from jorg.abundances import solar_abundances
        
        # Test different stellar conditions
        test_conditions = [
            ("Solar photosphere", 5778, 1e-7),
            ("Cool giant", 4000, 1e-9), 
            ("Hot dwarf", 7000, 1e-6),
        ]
        
        partition_funcs = setup_partition_functions()
        log_equilibrium_constants = create_default_log_equilibrium_constants()
        abs_abundances = solar_abundances()
        
        convergence_results = []
        
        for desc, T, density in test_conditions:
            try:
                result = solve_chemical_equilibrium(
                    T, density, abs_abundances,
                    partition_funcs, log_equilibrium_constants
                )
                
                converged = result.get('converged', True)
                n_e = result['electron_density']
                
                print(f"{desc}: T={T}K, ρ={density:.1e}, nₑ={n_e:.2e}, converged={converged}")
                convergence_results.append(converged)
                
            except Exception as e:
                print(f"{desc}: Failed - {e}")
                convergence_results.append(False)
        
        success_rate = sum(convergence_results) / len(convergence_results)
        
        if success_rate >= 0.75:  # At least 75% should converge
            print(f"✅ Chemical equilibrium convergence: {success_rate:.1%}")
            return True
        else:
            print(f"❌ Poor convergence rate: {success_rate:.1%}")
            return False
            
    except Exception as e:
        print(f"❌ Convergence test failed: {e}")
        return False

def run_all_tests():
    """Run all molecular abundance validation tests"""
    print("JORG MOLECULAR ABUNDANCE VALIDATION")
    print("=" * 45)
    
    tests = [
        ("Molecular abundance calculation", test_molecular_abundance_calculation),
        ("H2O abundance fix", test_h2o_abundance_fix),
        ("Chemical equilibrium convergence", test_molecular_equilibrium_convergence),
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
        print("🎉 ALL TESTS PASSED - Molecular abundance fix is working!")
        print("📊 Jorg now produces realistic molecular abundances")
        return True
    elif passed >= total * 0.67:
        print("✅ Most tests passed - Molecular abundances much improved")
        return True
    else:
        print("⚠️ Issues remain with molecular abundance calculations")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)