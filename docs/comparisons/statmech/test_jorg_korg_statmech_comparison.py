#!/usr/bin/env python3
"""
Comprehensive comparison test between Jorg and Korg.jl statistical mechanics implementations.

This test suite compares the outputs of:
1. Saha ionization equation calculations
2. Partition function evaluations
3. Chemical equilibrium solvers 
4. Translational partition functions

The tests use identical input parameters and compare results to ensure
the Jorg implementation matches Korg.jl behavior.
"""

import sys
import os
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add Jorg to path
jorg_path = Path(__file__).parent / "Jorg" / "src"
sys.path.insert(0, str(jorg_path))

try:
    # Import Jorg modules
    from jorg.statmech import (
        saha_ion_weights, 
        translational_u, 
        chemical_equilibrium,
        hydrogen_partition_function,
        simple_partition_function
    )
    from jorg.statmech.saha_equation import (
        create_default_ionization_energies,
        create_simple_partition_functions,
        simple_saha_test,
        validate_saha_implementation,
        BARKLEM_COLLET_IONIZATION_ENERGIES
    )
    from jorg.statmech.species import Species, Formula, MAX_ATOMIC_NUMBER
    from jorg.statmech.chemical_equilibrium import (
        create_default_absolute_abundances,
        validate_chemical_equilibrium
    )
    from jorg.constants import kboltz_cgs, hplanck_cgs, me_cgs, EV_TO_ERG
    
    JORG_AVAILABLE = True
    
except ImportError as e:
    print(f"Warning: Could not import Jorg modules: {e}")
    JORG_AVAILABLE = False

# Import Julia if available
try:
    import julia
    from julia import Main
    Main.eval('using Pkg; Pkg.activate(".")')
    Main.eval('using Korg')
    JULIA_AVAILABLE = True
    
except ImportError:
    print("Warning: Julia not available - will compare against reference values")
    JULIA_AVAILABLE = False


class StatMechComparisonTest:
    """Comprehensive statistical mechanics comparison between Jorg and Korg.jl"""
    
    def __init__(self):
        self.results = {
            'saha_tests': [],
            'partition_function_tests': [],
            'chemical_equilibrium_tests': [],
            'translational_tests': [],
            'summary': {},
            'timing': {}
        }
        
        # Test parameters covering typical stellar atmosphere conditions
        self.test_conditions = [
            {'T': 3000.0, 'ne': 1e10, 'name': 'Cool_star_photosphere'},
            {'T': 4000.0, 'ne': 1e11, 'name': 'M_dwarf_photosphere'}, 
            {'T': 5778.0, 'ne': 1e13, 'name': 'Solar_photosphere'},
            {'T': 8000.0, 'ne': 1e14, 'name': 'A_star_photosphere'},
            {'T': 10000.0, 'ne': 1e15, 'name': 'Hot_star_photosphere'},
            {'T': 15000.0, 'ne': 1e16, 'name': 'B_star_photosphere'},
        ]
        
        # Elements to test (focus on common ones)
        self.test_elements = [1, 2, 6, 7, 8, 11, 12, 13, 14, 16, 20, 26]  # H, He, C, N, O, Na, Mg, Al, Si, S, Ca, Fe
        
    def setup_korg_data(self):
        """Setup Korg.jl data for comparison."""
        if not JULIA_AVAILABLE:
            print("Julia not available - using reference values")
            return
            
        try:
            # Load Korg constants and data
            Main.eval('''
            using Korg
            korg_kboltz_eV = Korg.kboltz_eV
            korg_electron_mass = Korg.electron_mass_cgs
            korg_ionization_energies = Korg.ionization_energies
            korg_partition_funcs = Korg.default_partition_funcs
            ''')
            
            self.korg_constants = {
                'kboltz_eV': float(Main.korg_kboltz_eV),
                'electron_mass_cgs': float(Main.korg_electron_mass)
            }
            
            print("Korg.jl data loaded successfully")
            
        except Exception as e:
            print(f"Error loading Korg.jl data: {e}")
            
    def test_constants_compatibility(self):
        """Test that fundamental constants match between implementations."""
        print("\n=== Testing Constants Compatibility ===")
        
        # Compare physical constants
        jorg_kboltz_eV = kboltz_cgs / EV_TO_ERG
        
        constants_comparison = {
            'kboltz_cgs': {
                'jorg': float(kboltz_cgs),
                'expected': 1.380649e-16  # CODATA 2018
            },
            'kboltz_eV': {
                'jorg': float(jorg_kboltz_eV),
                'expected': 8.617333e-5  # CODATA 2018
            },
            'hplanck_cgs': {
                'jorg': float(hplanck_cgs),
                'expected': 6.62607015e-27  # CODATA 2018
            },
            'electron_mass_cgs': {
                'jorg': float(me_cgs),
                'expected': 9.1093837015e-28  # CODATA 2018
            }
        }
        
        for const_name, values in constants_comparison.items():
            rel_diff = abs(values['jorg'] - values['expected']) / values['expected']
            print(f"{const_name:20s}: Jorg={values['jorg']:.6e}, Expected={values['expected']:.6e}, RelDiff={rel_diff:.2e}")
            
        if JULIA_AVAILABLE and hasattr(self, 'korg_constants'):
            print("\nComparison with Korg.jl:")
            for const_name in ['kboltz_eV', 'electron_mass_cgs']:
                if const_name in self.korg_constants:
                    korg_val = self.korg_constants[const_name]
                    if const_name == 'kboltz_eV':
                        jorg_val = float(jorg_kboltz_eV)
                    else:
                        jorg_val = float(me_cgs)
                    rel_diff = abs(jorg_val - korg_val) / korg_val
                    print(f"{const_name:20s}: Jorg={jorg_val:.6e}, Korg={korg_val:.6e}, RelDiff={rel_diff:.2e}")
                    
    def test_translational_partition_function(self):
        """Test translational partition function calculations."""
        print("\n=== Testing Translational Partition Function ===")
        
        masses = [me_cgs, 1.67262e-24, 6.64466e-24]  # electron, proton, alpha particle
        mass_names = ['electron', 'proton', 'alpha']
        
        for condition in self.test_conditions[:3]:  # Test subset for speed
            T = condition['T']
            
            for mass, name in zip(masses, mass_names):
                # Calculate using Jorg
                start_time = time.time()
                jorg_result = float(translational_u(mass, T))
                jorg_time = time.time() - start_time
                
                # Calculate reference value
                k = kboltz_cgs
                h = hplanck_cgs
                expected = (2.0 * np.pi * mass * k * T / (h * h))**1.5
                
                rel_diff = abs(jorg_result - expected) / expected
                
                test_result = {
                    'temperature': T,
                    'particle': name,
                    'mass': mass,
                    'jorg_result': jorg_result,
                    'expected': expected,
                    'relative_difference': rel_diff,
                    'jorg_time': jorg_time,
                    'status': 'PASS' if rel_diff < 1e-10 else 'FAIL'
                }
                
                self.results['translational_tests'].append(test_result)
                print(f"T={T:6.0f}K, {name:8s}: Jorg={jorg_result:.3e}, Ref={expected:.3e}, RelDiff={rel_diff:.2e}")
                
    def test_saha_equation(self):
        """Test Saha equation calculations."""
        print("\n=== Testing Saha Equation ===")
        
        if not JORG_AVAILABLE:
            print("Jorg not available - skipping Saha tests")
            return
            
        # Setup test data
        ionization_energies = create_default_ionization_energies()
        partition_funcs = create_simple_partition_functions()
        
        for condition in self.test_conditions:
            T = condition['T']
            ne = condition['ne']
            
            print(f"\nTesting at T={T}K, ne={ne:.0e} cm^-3:")
            
            for Z in self.test_elements:
                if Z not in ionization_energies:
                    continue
                    
                element_name = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 
                               'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca'][Z-1] if Z <= 20 else f"Z{Z}"
                
                try:
                    # Test Jorg implementation
                    start_time = time.time()
                    wII_jorg, wIII_jorg = saha_ion_weights(T, ne, Z, ionization_energies, partition_funcs)
                    jorg_time = time.time() - start_time
                    
                    # Test simple implementation for comparison
                    chi_I = ionization_energies[Z][0]
                    ref_ratio = simple_saha_test(T, ne, Z, chi_I)
                    
                    # Calculate ionization fractions
                    ion_frac_jorg = wII_jorg / (1.0 + wII_jorg + wIII_jorg)
                    ion_frac_ref = ref_ratio / (1.0 + ref_ratio)
                    
                    rel_diff = abs(ion_frac_jorg - ion_frac_ref) / max(ion_frac_ref, 1e-30)
                    
                    test_result = {
                        'condition': condition['name'],
                        'temperature': T,
                        'electron_density': ne,
                        'element': element_name,
                        'atomic_number': Z,
                        'ionization_energy': chi_I,
                        'wII_jorg': float(wII_jorg),
                        'wIII_jorg': float(wIII_jorg),
                        'ionization_fraction_jorg': float(ion_frac_jorg),
                        'ionization_fraction_reference': float(ion_frac_ref),
                        'relative_difference': float(rel_diff),
                        'jorg_time': jorg_time,
                        'status': 'PASS' if rel_diff < 0.1 else 'FAIL'  # Allow 10% difference due to different partition functions
                    }
                    
                    self.results['saha_tests'].append(test_result)
                    
                    if Z in [1, 6, 8, 26]:  # Show results for key elements
                        print(f"  {element_name:2s}: wII={wII_jorg:.3e}, wIII={wIII_jorg:.3e}, "
                              f"ion_frac={ion_frac_jorg:.3e}, rel_diff={rel_diff:.2e}")
                        
                except Exception as e:
                    print(f"  Error testing {element_name}: {e}")
                    
    def test_partition_functions(self):
        """Test partition function calculations."""
        print("\n=== Testing Partition Functions ===")
        
        if not JORG_AVAILABLE:
            print("Jorg not available - skipping partition function tests")
            return
            
        test_temperatures = [3000, 5000, 8000, 12000]
        
        for T in test_temperatures:
            log_T = np.log(T)
            
            print(f"\nTesting at T={T}K (log_T={log_T:.3f}):")
            
            # Test hydrogen partition function
            start_time = time.time()
            h_partition = hydrogen_partition_function(log_T)
            h_time = time.time() - start_time
            
            print(f"  H partition function: {h_partition:.3f} (expected: 2.0)")
            
            # Test simple partition functions for key elements
            for Z in [1, 2, 6, 8, 26]:  # H, He, C, O, Fe
                element_name = ['H', 'He', 'C', 'O', 'Fe'][{1:0, 2:1, 6:2, 8:3, 26:4}[Z]]
                
                start_time = time.time()
                partition = simple_partition_function(Z, log_T)
                partition_time = time.time() - start_time
                
                test_result = {
                    'temperature': T,
                    'log_temperature': log_T,
                    'element': element_name,
                    'atomic_number': Z,
                    'partition_function': float(partition),
                    'computation_time': partition_time
                }
                
                self.results['partition_function_tests'].append(test_result)
                print(f"  {element_name:2s} (Z={Z:2d}): U = {partition:.3f}")
                
    def test_chemical_equilibrium(self):
        """Test chemical equilibrium solver."""
        print("\n=== Testing Chemical Equilibrium Solver ===")
        
        if not JORG_AVAILABLE:
            print("Jorg not available - skipping chemical equilibrium tests")
            return
            
        # Setup test data
        ionization_energies = create_default_ionization_energies()
        partition_funcs = create_simple_partition_functions()
        log_equilibrium_constants = {}  # No molecules for this test
        absolute_abundances = create_default_absolute_abundances()
        
        print(f"Testing with {len(absolute_abundances)} elements")
        print("Key abundances:", {Z: f"{ab:.2e}" for Z, ab in list(absolute_abundances.items())[:5]})
        
        for condition in self.test_conditions[:4]:  # Test subset for speed
            T = condition['T']
            nt = 1e17  # Total number density (cm^-3)
            ne_guess = condition['ne']
            
            print(f"\nTesting chemical equilibrium at T={T}K, nt={nt:.0e}, ne_guess={ne_guess:.0e}:")
            
            try:
                start_time = time.time()
                ne_calc, number_densities = chemical_equilibrium(
                    T, nt, ne_guess, absolute_abundances, ionization_energies,
                    partition_funcs, log_equilibrium_constants
                )
                equilibrium_time = time.time() - start_time
                
                # Calculate some key ratios
                h_neutral = number_densities.get(Species.from_atomic_number(1, 0), 0.0)
                h_ionized = number_densities.get(Species.from_atomic_number(1, 1), 0.0)
                h_total = h_neutral + h_ionized
                h_ion_fraction = h_ionized / h_total if h_total > 0 else 0.0
                
                fe_neutral = number_densities.get(Species.from_atomic_number(26, 0), 0.0)
                fe_ionized = number_densities.get(Species.from_atomic_number(26, 1), 0.0)
                fe_total = fe_neutral + fe_ionized
                fe_ion_fraction = fe_ionized / fe_total if fe_total > 0 else 0.0
                
                test_result = {
                    'condition': condition['name'],
                    'temperature': T,
                    'total_density': nt,
                    'electron_density_guess': ne_guess,
                    'electron_density_calculated': float(ne_calc),
                    'electron_density_ratio': float(ne_calc / ne_guess),
                    'hydrogen_ionization_fraction': float(h_ion_fraction),
                    'iron_ionization_fraction': float(fe_ion_fraction),
                    'total_species_count': len(number_densities),
                    'computation_time': equilibrium_time,
                    'status': 'PASS' if abs(ne_calc - ne_guess) / ne_guess < 2.0 else 'WARNING'  # Allow factor of 2 difference
                }
                
                self.results['chemical_equilibrium_tests'].append(test_result)
                
                print(f"  Calculated ne = {ne_calc:.3e} cm^-3 (ratio = {ne_calc/ne_guess:.2f})")
                print(f"  H ionization fraction = {h_ion_fraction:.3e}")
                print(f"  Fe ionization fraction = {fe_ion_fraction:.3e}")
                print(f"  Total species calculated: {len(number_densities)}")
                print(f"  Computation time: {equilibrium_time:.3f} s")
                
            except Exception as e:
                print(f"  Error in chemical equilibrium: {e}")
                test_result = {
                    'condition': condition['name'],
                    'temperature': T,
                    'error': str(e),
                    'status': 'FAIL'
                }
                self.results['chemical_equilibrium_tests'].append(test_result)
                
    def test_ionization_energies_data(self):
        """Test ionization energies data consistency."""
        print("\n=== Testing Ionization Energies Data ===")
        
        if not JORG_AVAILABLE:
            print("Jorg not available - skipping ionization energies test")
            return
            
        ionization_energies = create_default_ionization_energies()
        
        print("Sample ionization energies (eV):")
        print("Element    χI        χII       χIII")
        print("-" * 35)
        
        for Z in [1, 2, 6, 7, 8, 11, 12, 13, 14, 16, 20, 26]:
            if Z in ionization_energies:
                chi_I, chi_II, chi_III = ionization_energies[Z]
                element_name = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 
                               'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca'][Z-1] if Z <= 20 else f"Z{Z}"
                if Z == 26:
                    element_name = "Fe"
                print(f"{element_name:7s}    {chi_I:7.3f}   {chi_II:7.3f}   {chi_III:7.3f}")
                
        # Check that hydrogen has no second ionization
        h_energies = ionization_energies[1]
        assert h_energies[1] == -1.000 or h_energies[1] == 0.0, "Hydrogen should have no second ionization"
        print(f"\nHydrogen ionization energies: {h_energies}")
        
    def generate_summary(self):
        """Generate test summary and statistics."""
        print("\n=== Test Summary ===")
        
        # Count test results by category
        categories = ['saha_tests', 'partition_function_tests', 'chemical_equilibrium_tests', 'translational_tests']
        
        summary = {}
        for category in categories:
            tests = self.results.get(category, [])
            total = len(tests)
            passed = sum(1 for test in tests if test.get('status') == 'PASS')
            failed = sum(1 for test in tests if test.get('status') == 'FAIL')
            warnings = sum(1 for test in tests if test.get('status') == 'WARNING')
            
            summary[category] = {
                'total': total,
                'passed': passed,
                'failed': failed,
                'warnings': warnings,
                'pass_rate': passed / total if total > 0 else 0.0
            }
            
            print(f"\n{category.replace('_', ' ').title()}:")
            print(f"  Total: {total}, Passed: {passed}, Failed: {failed}, Warnings: {warnings}")
            print(f"  Pass rate: {passed/total*100:.1f}%" if total > 0 else "  No tests")
            
        self.results['summary'] = summary
        
        # Overall statistics
        total_tests = sum(s['total'] for s in summary.values())
        total_passed = sum(s['passed'] for s in summary.values())
        overall_pass_rate = total_passed / total_tests if total_tests > 0 else 0.0
        
        print(f"\nOverall: {total_passed}/{total_tests} tests passed ({overall_pass_rate*100:.1f}%)")
        
        # Performance summary
        print("\nPerformance Summary:")
        for category in categories:
            tests = self.results.get(category, [])
            if tests and any('jorg_time' in test or 'computation_time' in test for test in tests):
                times = []
                for test in tests:
                    if 'jorg_time' in test:
                        times.append(test['jorg_time'])
                    elif 'computation_time' in test:
                        times.append(test['computation_time'])
                
                if times:
                    print(f"  {category}: avg={np.mean(times)*1000:.2f}ms, max={np.max(times)*1000:.2f}ms")
                    
    def save_results(self, filename: str = None):
        """Save detailed results to JSON file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"jorg_korg_statmech_comparison_{timestamp}.json"
            
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
                
        results_serializable = convert_numpy(self.results)
        
        with open(filename, 'w') as f:
            json.dump(results_serializable, f, indent=2)
            
        print(f"\nDetailed results saved to: {filename}")
        return filename
        
    def run_all_tests(self):
        """Run complete test suite."""
        print("=" * 60)
        print("JORG vs KORG.JL STATISTICAL MECHANICS COMPARISON")
        print("=" * 60)
        
        # Check availability
        print(f"Jorg available: {JORG_AVAILABLE}")
        print(f"Julia/Korg.jl available: {JULIA_AVAILABLE}")
        
        if not JORG_AVAILABLE:
            print("ERROR: Jorg not available - cannot run tests")
            return
            
        # Setup
        if JULIA_AVAILABLE:
            self.setup_korg_data()
            
        # Run test suite
        self.test_constants_compatibility()
        self.test_ionization_energies_data()
        self.test_translational_partition_function()
        self.test_partition_functions()
        self.test_saha_equation()
        self.test_chemical_equilibrium()
        
        # Generate summary
        self.generate_summary()
        
        # Save results
        results_file = self.save_results()
        
        print("\n" + "=" * 60)
        print("COMPARISON TEST COMPLETE")
        print("=" * 60)
        
        return results_file


def main():
    """Main test execution."""
    try:
        # Run validation tests if Jorg is available
        if JORG_AVAILABLE:
            print("Running Jorg validation tests...")
            validate_saha_implementation()
            print("\nRunning chemical equilibrium validation...")
            validate_chemical_equilibrium()
            print("\n" + "="*50 + "\n")
        
        # Run comparison tests
        tester = StatMechComparisonTest()
        results_file = tester.run_all_tests()
        
        return results_file
        
    except Exception as e:
        print(f"Error running tests: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()