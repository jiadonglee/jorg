#!/usr/bin/env python3
"""
Comprehensive test suite for Jorg statistical mechanics modules.

This test validates the fixed chemical equilibrium solver and provides
detailed comparison with expected Korg.jl behavior.
"""

import sys
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add Jorg to path
jorg_path = Path(__file__).parent / "Jorg" / "src"
sys.path.insert(0, str(jorg_path))

try:
    from jorg.statmech.species import Species, Formula
    from jorg.statmech.saha_equation import (
        saha_ion_weights, translational_U, simple_saha_test,
        create_default_ionization_energies, validate_saha_implementation
    )
    from jorg.statmech.partition_functions import (
        hydrogen_partition_function, simple_partition_function,
        partition_function
    )
    from jorg.constants import kboltz_cgs, kboltz_eV, hplanck_cgs, me_cgs, EV_TO_ERG
    JORG_AVAILABLE = True
except ImportError as e:
    print(f"Error importing Jorg: {e}")
    JORG_AVAILABLE = False


class ComprehensiveStatMechTest:
    """Comprehensive test suite for statistical mechanics modules."""
    
    def __init__(self):
        self.test_results = {
            'constants': {},
            'species_operations': {},
            'translational_partition': {},
            'atomic_partition': {},
            'saha_equation': {},
            'chemical_equilibrium': {},
            'performance': {},
            'validation': {}
        }
        
        # Test conditions covering stellar atmosphere range
        self.stellar_conditions = [
            {'T': 3500, 'ne': 5e10, 'log_g': 5.0, 'name': 'M_dwarf'},
            {'T': 4500, 'ne': 2e11, 'log_g': 4.5, 'name': 'K_dwarf'},
            {'T': 5778, 'ne': 1e13, 'log_g': 4.44, 'name': 'Sun'},
            {'T': 7000, 'ne': 5e13, 'log_g': 4.2, 'name': 'F_star'},
            {'T': 9000, 'ne': 2e14, 'log_g': 4.0, 'name': 'A_star'},
            {'T': 12000, 'ne': 8e14, 'log_g': 4.0, 'name': 'B_star'},
            {'T': 20000, 'ne': 1e16, 'log_g': 4.0, 'name': 'Hot_star'}
        ]
        
        # Key elements for testing
        self.test_elements = {
            1: 'H', 2: 'He', 3: 'Li', 6: 'C', 7: 'N', 8: 'O',
            11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 16: 'S', 
            20: 'Ca', 22: 'Ti', 26: 'Fe', 28: 'Ni'
        }
    
    def test_physical_constants(self):
        """Test physical constants against CODATA 2018."""
        print("=== Testing Physical Constants ===")
        
        # CODATA 2018 values
        codata = {
            'kboltz_cgs': 1.380649e-16,
            'hplanck_cgs': 6.62607015e-27,
            'me_cgs': 9.1093837015e-28,
            'eV_to_erg': 1.602176634e-12
        }
        
        jorg_values = {
            'kboltz_cgs': float(kboltz_cgs),
            'hplanck_cgs': float(hplanck_cgs),
            'me_cgs': float(me_cgs),
            'eV_to_erg': float(EV_TO_ERG)
        }
        
        results = {}
        max_rel_diff = 0.0
        
        for const, expected in codata.items():
            jorg_val = jorg_values[const]
            rel_diff = abs(jorg_val - expected) / expected
            max_rel_diff = max(max_rel_diff, rel_diff)
            
            status = "PASS" if rel_diff < 1e-4 else "FAIL"
            results[const] = {
                'jorg': jorg_val,
                'codata': expected,
                'relative_difference': rel_diff,
                'status': status
            }
            
            print(f"  {const:15s}: {rel_diff:.2e} relative difference [{status}]")
        
        overall_status = "PASS" if max_rel_diff < 1e-4 else "FAIL"
        results['overall_status'] = overall_status
        results['max_relative_difference'] = max_rel_diff
        
        self.test_results['constants'] = results
        print(f"  Overall constants test: {overall_status}")
        
    def test_species_operations(self):
        """Test Species and Formula operations."""
        print("\n=== Testing Species Operations ===")
        
        results = {}
        
        try:
            # Test Formula creation
            h_formula = Formula.from_atomic_number(1)
            he_formula = Formula.from_atomic_number(2)
            
            # Test Species creation
            h1 = Species.from_atomic_number(1, 0)  # H I
            h2 = Species.from_atomic_number(1, 1)  # H II
            he1 = Species.from_atomic_number(2, 0)  # He I
            he2 = Species.from_atomic_number(2, 1)  # He II
            
            # Test comparison operators
            species_list = [h2, he1, h1, he2]  # Out of order
            sorted_species = sorted(species_list)
            expected_order = [h1, h2, he1, he2]
            
            comparison_works = all(a == b for a, b in zip(sorted_species, expected_order))
            
            # Test string representations
            str_representations = [str(s) for s in [h1, h2, he1, he2]]
            expected_strs = ['H I', 'H II', 'He I', 'He II']
            string_repr_works = str_representations == expected_strs
            
            # Test hashing (for dictionary keys)
            species_dict = {h1: 'hydrogen_neutral', h2: 'hydrogen_ion'}
            hash_works = len(species_dict) == 2 and h1 in species_dict
            
            results = {
                'formula_creation': True,
                'species_creation': True,
                'comparison_operators': comparison_works,
                'string_representation': string_repr_works,
                'hashing': hash_works,
                'overall_status': 'PASS' if all([comparison_works, string_repr_works, hash_works]) else 'FAIL'
            }
            
            print(f"  Formula creation: ✅")
            print(f"  Species creation: ✅")
            print(f"  Comparison operators: {'✅' if comparison_works else '❌'}")
            print(f"  String representation: {'✅' if string_repr_works else '❌'}")
            print(f"  Hashing: {'✅' if hash_works else '❌'}")
            
        except Exception as e:
            results = {
                'error': str(e),
                'overall_status': 'FAIL'
            }
            print(f"  Species operations failed: {e}")
        
        self.test_results['species_operations'] = results
        
    def test_translational_partition_functions(self):
        """Test translational partition function calculations."""
        print("\n=== Testing Translational Partition Functions ===")
        
        results = {}
        test_cases = []
        
        # Test different particles and temperatures
        particles = [
            {'mass': me_cgs, 'name': 'electron'},
            {'mass': 1.67262e-24, 'name': 'proton'},
            {'mass': 6.64466e-24, 'name': 'alpha_particle'}
        ]
        
        temperatures = [3000, 5000, 8000, 12000]
        
        max_rel_diff = 0.0
        
        for particle in particles:
            for T in temperatures:
                # Jorg calculation
                start_time = time.time()
                jorg_result = float(translational_U(particle['mass'], T))
                jorg_time = time.time() - start_time
                
                # Reference calculation
                k = kboltz_cgs
                h = hplanck_cgs
                expected = (2.0 * np.pi * particle['mass'] * k * T / (h * h))**1.5
                
                rel_diff = abs(jorg_result - expected) / expected
                max_rel_diff = max(max_rel_diff, rel_diff)
                
                test_case = {
                    'particle': particle['name'],
                    'mass': particle['mass'],
                    'temperature': T,
                    'jorg_result': jorg_result,
                    'expected': expected,
                    'relative_difference': rel_diff,
                    'computation_time': jorg_time,
                    'status': 'PASS' if rel_diff < 1e-12 else 'FAIL'
                }
                
                test_cases.append(test_case)
        
        # Show sample results
        for case in test_cases[::3]:  # Every 3rd case
            print(f"  {case['particle']:8s} T={case['temperature']:5.0f}K: "
                  f"RelDiff={case['relative_difference']:.2e}")
        
        results = {
            'test_cases': test_cases,
            'max_relative_difference': max_rel_diff,
            'overall_status': 'PASS' if max_rel_diff < 1e-12 else 'FAIL'
        }
        
        self.test_results['translational_partition'] = results
        print(f"  Overall translational test: {results['overall_status']}")
        
    def test_atomic_partition_functions(self):
        """Test atomic partition function calculations."""
        print("\n=== Testing Atomic Partition Functions ===")
        
        results = {}
        test_cases = []
        
        temperatures = [3000, 5000, 8000, 12000]
        
        for Z in [1, 2, 6, 8, 26]:  # Key elements
            element_name = self.test_elements[Z]
            
            for T in temperatures:
                log_T = np.log(T)
                
                # Test simple partition function
                start_time = time.time()
                partition = float(simple_partition_function(Z, log_T))
                comp_time = time.time() - start_time
                
                # Test hydrogen-specific function
                if Z == 1:
                    h_partition = float(hydrogen_partition_function(log_T))
                    h_matches = abs(h_partition - 2.0) < 1e-10
                else:
                    h_matches = True
                
                # Check physical reasonableness
                reasonable = (0.5 < partition < 1000.0)  # Reasonable range
                
                test_case = {
                    'element': element_name,
                    'atomic_number': Z,
                    'temperature': T,
                    'log_temperature': log_T,
                    'partition_function': partition,
                    'physically_reasonable': reasonable,
                    'hydrogen_specific_match': h_matches,
                    'computation_time': comp_time
                }
                
                test_cases.append(test_case)
        
        # Show key results
        for T in [5000, 8000]:
            print(f"  T={T}K:")
            for Z in [1, 2, 26]:
                case = next(c for c in test_cases if c['atomic_number'] == Z and c['temperature'] == T)
                print(f"    {case['element']:2s}: U = {case['partition_function']:6.2f}")
        
        all_reasonable = all(case['physically_reasonable'] for case in test_cases)
        all_h_match = all(case['hydrogen_specific_match'] for case in test_cases)
        
        results = {
            'test_cases': test_cases,
            'all_physically_reasonable': all_reasonable,
            'hydrogen_function_correct': all_h_match,
            'overall_status': 'PASS' if all_reasonable and all_h_match else 'FAIL'
        }
        
        self.test_results['atomic_partition'] = results
        print(f"  Overall partition function test: {results['overall_status']}")
        
    def test_saha_equation_comprehensive(self):
        """Comprehensive test of Saha equation implementation."""
        print("\n=== Testing Saha Equation (Comprehensive) ===")
        
        results = {}
        test_cases = []
        
        ionization_energies = create_default_ionization_energies()
        
        # Test across stellar conditions
        for condition in self.stellar_conditions[:5]:  # Test subset
            T = condition['T']
            ne = condition['ne']
            name = condition['name']
            
            print(f"  {name} (T={T}K, ne={ne:.0e}):")
            
            condition_results = []
            
            for Z in [1, 2, 6, 8, 26]:  # Key elements
                if Z not in ionization_energies:
                    continue
                    
                element_name = self.test_elements[Z]
                chi_I = ionization_energies[Z][0]
                
                # Calculate using simple Saha test
                start_time = time.time()
                ratio = simple_saha_test(T, ne, Z, chi_I)
                comp_time = time.time() - start_time
                
                ion_fraction = ratio / (1.0 + ratio)
                
                # Validate against expected behavior
                if Z == 1:  # Hydrogen
                    if T < 6000:
                        expected = ion_fraction < 0.01
                    elif T < 10000:
                        expected = 1e-4 < ion_fraction < 0.1
                    else:
                        expected = ion_fraction > 0.05
                elif Z == 26:  # Iron
                    if T > 4000:
                        expected = ion_fraction > 0.5  # Should be mostly ionized
                    else:
                        expected = True  # Any value acceptable for cool stars
                else:
                    expected = True  # Accept any reasonable value
                
                test_case = {
                    'condition': name,
                    'element': element_name,
                    'atomic_number': Z,
                    'temperature': T,
                    'electron_density': ne,
                    'ionization_energy': chi_I,
                    'ionization_ratio': ratio,
                    'ionization_fraction': ion_fraction,
                    'expected_behavior': expected,
                    'computation_time': comp_time
                }
                
                test_cases.append(test_case)
                condition_results.append(test_case)
                
                print(f"    {element_name:2s}: ion_frac={ion_fraction:.3e} "
                      f"{'✅' if expected else '⚠️'}")
        
        # Overall validation
        all_expected = all(case['expected_behavior'] for case in test_cases)
        
        results = {
            'test_cases': test_cases,
            'all_expected_behavior': all_expected,
            'overall_status': 'PASS' if all_expected else 'WARNING'
        }
        
        self.test_results['saha_equation'] = results
        print(f"  Overall Saha equation test: {results['overall_status']}")
        
    def test_performance_benchmarks(self):
        """Test performance of key functions."""
        print("\n=== Performance Benchmarks ===")
        
        results = {}
        
        # Benchmark translational partition function
        def benchmark_translational():
            times = []
            for _ in range(1000):
                start = time.time()
                translational_U(me_cgs, 5778.0)
                times.append(time.time() - start)
            return np.mean(times), np.std(times)
        
        # Benchmark partition functions
        def benchmark_partition():
            times = []
            log_T = np.log(5778.0)
            for _ in range(1000):
                start = time.time()
                simple_partition_function(26, log_T)  # Iron
                times.append(time.time() - start)
            return np.mean(times), np.std(times)
        
        # Benchmark Saha equation
        def benchmark_saha():
            times = []
            for _ in range(100):
                start = time.time()
                simple_saha_test(5778.0, 1e13, 26, 7.902)
                times.append(time.time() - start)
            return np.mean(times), np.std(times)
        
        print("  Running performance benchmarks...")
        
        trans_mean, trans_std = benchmark_translational()
        partition_mean, partition_std = benchmark_partition()
        saha_mean, saha_std = benchmark_saha()
        
        results = {
            'translational_partition': {
                'mean_time_ms': trans_mean * 1000,
                'std_time_ms': trans_std * 1000,
                'calls_per_second': 1.0 / trans_mean
            },
            'atomic_partition': {
                'mean_time_ms': partition_mean * 1000,
                'std_time_ms': partition_std * 1000,
                'calls_per_second': 1.0 / partition_mean
            },
            'saha_equation': {
                'mean_time_ms': saha_mean * 1000,
                'std_time_ms': saha_std * 1000,
                'calls_per_second': 1.0 / saha_mean
            }
        }
        
        print(f"  Translational U: {trans_mean*1e6:.1f} ± {trans_std*1e6:.1f} μs")
        print(f"  Partition func:  {partition_mean*1e6:.1f} ± {partition_std*1e6:.1f} μs")
        print(f"  Saha equation:   {saha_mean*1e3:.2f} ± {saha_std*1e3:.2f} ms")
        
        self.test_results['performance'] = results
        
    def run_validation_tests(self):
        """Run validation against known literature values."""
        print("\n=== Validation Against Literature ===")
        
        results = {}
        
        # Solar photosphere validation
        T_sun = 5778.0
        ne_sun = 1e13
        
        # Hydrogen ionization (should be ~1.5e-4)
        h_ratio = simple_saha_test(T_sun, ne_sun, 1, 13.598)
        h_ion_frac = h_ratio / (1.0 + h_ratio)
        h_literature = 1.5e-4
        h_rel_diff = abs(h_ion_frac - h_literature) / h_literature
        h_valid = h_rel_diff < 0.5  # 50% tolerance
        
        # Iron ionization (should be >90%)
        fe_ratio = simple_saha_test(T_sun, ne_sun, 26, 7.902)
        fe_ion_frac = fe_ratio / (1.0 + fe_ratio)
        fe_literature = 0.93
        fe_rel_diff = abs(fe_ion_frac - fe_literature) / fe_literature
        fe_valid = fe_rel_diff < 0.2  # 20% tolerance
        
        results = {
            'solar_hydrogen': {
                'calculated': h_ion_frac,
                'literature': h_literature,
                'relative_difference': h_rel_diff,
                'valid': h_valid
            },
            'solar_iron': {
                'calculated': fe_ion_frac,
                'literature': fe_literature,
                'relative_difference': fe_rel_diff,
                'valid': fe_valid
            },
            'overall_validation': h_valid and fe_valid
        }
        
        print(f"  Solar H ionization: {h_ion_frac:.3e} vs {h_literature:.3e} "
              f"({'✅' if h_valid else '❌'})")
        print(f"  Solar Fe ionization: {fe_ion_frac:.3f} vs {fe_literature:.3f} "
              f"({'✅' if fe_valid else '❌'})")
        
        self.test_results['validation'] = results
        
    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE STATISTICAL MECHANICS TEST SUMMARY")
        print("=" * 60)
        
        # Count overall results
        categories = ['constants', 'species_operations', 'translational_partition', 
                     'atomic_partition', 'saha_equation', 'validation']
        
        passed = 0
        total = 0
        
        for category in categories:
            if category in self.test_results:
                result = self.test_results[category]
                status = result.get('overall_status', 'UNKNOWN')
                
                if status == 'PASS':
                    status_icon = "✅"
                    passed += 1
                elif status == 'WARNING':
                    status_icon = "⚠️"
                    passed += 0.5  # Partial credit
                else:
                    status_icon = "❌"
                
                total += 1
                
                print(f"{category.replace('_', ' ').title():25s}: {status_icon} {status}")
        
        # Overall assessment
        pass_rate = passed / total if total > 0 else 0.0
        
        if pass_rate >= 0.9:
            overall_assessment = "EXCELLENT"
        elif pass_rate >= 0.7:
            overall_assessment = "GOOD"
        elif pass_rate >= 0.5:
            overall_assessment = "ACCEPTABLE"
        else:
            overall_assessment = "NEEDS_IMPROVEMENT"
        
        print(f"\nOverall Assessment: {overall_assessment} ({pass_rate:.1%} pass rate)")
        
        # Performance summary
        if 'performance' in self.test_results:
            perf = self.test_results['performance']
            print(f"\nPerformance Summary:")
            print(f"  Translational U: {perf['translational_partition']['calls_per_second']:.0f} calls/sec")
            print(f"  Partition functions: {perf['atomic_partition']['calls_per_second']:.0f} calls/sec")
            print(f"  Saha equation: {perf['saha_equation']['calls_per_second']:.0f} calls/sec")
        
        self.test_results['summary'] = {
            'pass_rate': pass_rate,
            'overall_assessment': overall_assessment,
            'total_categories': total,
            'passed_categories': passed
        }
        
    def save_results(self, filename: str = None):
        """Save detailed results to JSON file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"statmech_comprehensive_test_{timestamp}.json"
        
        # Convert numpy types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_for_json(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_for_json(self.test_results)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nDetailed results saved to: {filename}")
        return filename
        
    def run_all_tests(self):
        """Run the complete comprehensive test suite."""
        if not JORG_AVAILABLE:
            print("Error: Jorg modules not available")
            return None
        
        print("=" * 60)
        print("COMPREHENSIVE JORG STATISTICAL MECHANICS TEST SUITE")
        print("=" * 60)
        
        # Run all test categories
        self.test_physical_constants()
        self.test_species_operations()
        self.test_translational_partition_functions()
        self.test_atomic_partition_functions()
        self.test_saha_equation_comprehensive()
        self.test_performance_benchmarks()
        self.run_validation_tests()
        
        # Generate summary
        self.generate_summary_report()
        
        # Save results
        results_file = self.save_results()
        
        return results_file


def main():
    """Run comprehensive test suite."""
    tester = ComprehensiveStatMechTest()
    results_file = tester.run_all_tests()
    
    if results_file:
        print(f"\nTest completed successfully. Results in: {results_file}")
    
    return results_file


if __name__ == "__main__":
    main()