#!/usr/bin/env python3
"""
Test JAX Implementation with Real MARCS Data
============================================

Test the JAX atmosphere interpolation with actual MARCS grid files
and compare results against Korg subprocess calls.
"""

import sys
import numpy as np
import warnings
from pathlib import Path

# Add Jorg to path
jorg_path = Path(__file__).parent / "Jorg" / "src"
sys.path.insert(0, str(jorg_path))

try:
    import jax
    import jax.numpy as jnp
    from jorg.atmosphere_jax import interpolate_marcs_jax, AtmosphereInterpolationError
    JAX_AVAILABLE = True
except ImportError as e:
    print(f"JAX not available: {e}")
    JAX_AVAILABLE = False

from jorg.atmosphere import call_korg_interpolation

def test_jax_vs_korg(test_cases):
    """Compare JAX implementation against Korg for multiple test cases"""
    
    print("JAX vs KORG ATMOSPHERE INTERPOLATION COMPARISON")
    print("=" * 70)
    
    if not JAX_AVAILABLE:
        print("❌ JAX not available, cannot run comparison")
        return []
    
    results = []
    
    for i, (description, Teff, logg, m_H, alpha_m, C_m) in enumerate(test_cases):
        print(f"\n{i+1}. Testing: {description}")
        print(f"   Parameters: Teff={Teff}K, logg={logg}, [M/H]={m_H}, [α/M]={alpha_m}, [C/M]={C_m}")
        
        # Get Korg reference
        try:
            print("   Getting Korg reference...")
            korg_atm = call_korg_interpolation(Teff, logg, m_H, alpha_m, C_m)
            korg_success = True
            print(f"   ✅ Korg: {len(korg_atm.layers)} layers, spherical={korg_atm.spherical}")
            
        except Exception as e:
            print(f"   ❌ Korg failed: {e}")
            korg_success = False
            korg_atm = None
        
        # Test JAX implementation
        try:
            print("   Testing JAX implementation...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Suppress JAX warnings for cleaner output
                jax_atm = interpolate_marcs_jax(Teff, logg, m_H, alpha_m, C_m)
            jax_success = True
            print(f"   ✅ JAX: {len(jax_atm.layers)} layers, spherical={jax_atm.spherical}")
            
        except Exception as e:
            print(f"   ❌ JAX failed: {e}")
            jax_success = False
            jax_atm = None
        
        # Compare results if both succeeded
        if korg_success and jax_success:
            comparison = compare_atmospheres(korg_atm, jax_atm, description)
            results.append(comparison)
            print(f"   📊 Comparison: {comparison['summary']}")
        else:
            results.append({
                'test': description,
                'korg_success': korg_success,
                'jax_success': jax_success,
                'summary': 'Failed to compare'
            })
    
    return results

def compare_atmospheres(korg_atm, jax_atm, description):
    """Compare two atmospheres and return comparison metrics"""
    
    # Basic structure comparison
    layer_count_match = len(korg_atm.layers) == len(jax_atm.layers)
    spherical_match = korg_atm.spherical == jax_atm.spherical
    
    # Sample a few layers for detailed comparison
    n_layers = min(len(korg_atm.layers), len(jax_atm.layers))
    sample_indices = [n_layers//4, n_layers//2, 3*n_layers//4] if n_layers > 10 else [n_layers//2]
    
    layer_diffs = []
    for idx in sample_indices:
        if idx < n_layers:
            korg_layer = korg_atm.layers[idx]
            jax_layer = jax_atm.layers[idx]
            
            # Calculate relative differences
            temp_diff = abs(korg_layer.temp - jax_layer.temp) / korg_layer.temp * 100
            nt_diff = abs(korg_layer.number_density - jax_layer.number_density) / korg_layer.number_density * 100
            ne_diff = abs(korg_layer.electron_number_density - jax_layer.electron_number_density) / korg_layer.electron_number_density * 100
            tau_diff = abs(korg_layer.tau_5000 - jax_layer.tau_5000) / korg_layer.tau_5000 * 100
            
            max_diff = max(temp_diff, nt_diff, ne_diff, tau_diff)
            layer_diffs.append(max_diff)
    
    # Overall assessment
    avg_diff = np.mean(layer_diffs) if layer_diffs else float('inf')
    max_diff = np.max(layer_diffs) if layer_diffs else float('inf')
    
    if avg_diff < 1.0:
        summary = "Excellent agreement"
    elif avg_diff < 5.0:
        summary = "Good agreement"
    elif avg_diff < 20.0:
        summary = "Moderate agreement"
    else:
        summary = "Poor agreement"
    
    return {
        'test': description,
        'korg_success': True,
        'jax_success': True,
        'layer_count_match': layer_count_match,
        'spherical_match': spherical_match,
        'avg_diff_percent': avg_diff,
        'max_diff_percent': max_diff,
        'summary': summary,
        'layer_diffs': layer_diffs
    }

def test_specific_stellar_types():
    """Test specific stellar types that exercise different interpolation methods"""
    
    test_cases = [
        # Standard SDSS grid cases
        ("Solar G-type", 5777.0, 4.44, 0.0, 0.0, 0.0),
        ("Hot F-type", 6500.0, 4.0, 0.0, 0.0, 0.0),
        ("Cool K-type", 4500.0, 4.5, 0.0, 0.0, 0.0),
        
        # Metallicity variations
        ("Metal-poor solar", 5777.0, 4.44, -1.0, 0.0, 0.0),
        ("Metal-rich solar", 5777.0, 4.44, 0.3, 0.0, 0.0),
        
        # Alpha enhancement
        ("Alpha-enhanced", 5000.0, 4.0, -0.5, 0.4, 0.0),
        
        # Giants (spherical)
        ("K giant", 4000.0, 2.0, 0.0, 0.0, 0.0),
        ("G giant", 5000.0, 2.5, 0.0, 0.0, 0.0),
        
        # Cool dwarfs (should use cubic interpolation)
        ("Cool M dwarf", 3500.0, 4.8, 0.0, 0.0, 0.0),
        ("Cool K dwarf", 3800.0, 4.5, 0.0, 0.0, 0.0),
    ]
    
    # Test low metallicity cases (different grid)
    low_z_cases = [
        ("Very metal-poor", 5000.0, 4.0, -3.0, 0.4, 0.0),
        ("Extremely metal-poor", 6000.0, 4.5, -4.0, 0.4, 0.0),
    ]
    
    print("TESTING STANDARD STELLAR TYPES")
    print("=" * 50)
    standard_results = test_jax_vs_korg(test_cases)
    
    print(f"\n\nTESTING LOW METALLICITY CASES")
    print("=" * 50)
    low_z_results = test_jax_vs_korg(low_z_cases)
    
    return standard_results + low_z_results

def analyze_results(results):
    """Analyze and summarize test results"""
    
    print(f"\n\n{'='*70}")
    print("COMPREHENSIVE RESULTS ANALYSIS")
    print("="*70)
    
    if not results:
        print("❌ No results to analyze")
        return
    
    # Count successes
    total_tests = len(results)
    both_success = sum(1 for r in results if r.get('korg_success', False) and r.get('jax_success', False))
    jax_only_success = sum(1 for r in results if r.get('jax_success', False) and not r.get('korg_success', False))
    korg_only_success = sum(1 for r in results if r.get('korg_success', False) and not r.get('jax_success', False))
    both_failed = sum(1 for r in results if not r.get('korg_success', False) and not r.get('jax_success', False))
    
    print(f"\nSUCCESS RATES:")
    print(f"  Both succeeded:     {both_success}/{total_tests} ({both_success/total_tests*100:.1f}%)")
    print(f"  JAX only succeeded: {jax_only_success}/{total_tests}")
    print(f"  Korg only succeeded: {korg_only_success}/{total_tests}")
    print(f"  Both failed:        {both_failed}/{total_tests}")
    
    # Analyze agreements for successful comparisons
    successful_comparisons = [r for r in results if 'avg_diff_percent' in r]
    
    if successful_comparisons:
        print(f"\nAGREEMENT ANALYSIS ({len(successful_comparisons)} comparisons):")
        
        avg_diffs = [r['avg_diff_percent'] for r in successful_comparisons]
        max_diffs = [r['max_diff_percent'] for r in successful_comparisons]
        
        print(f"  Average difference: {np.mean(avg_diffs):.2f}% ± {np.std(avg_diffs):.2f}%")
        print(f"  Maximum difference: {np.mean(max_diffs):.2f}% ± {np.std(max_diffs):.2f}%")
        print(f"  Best case:         {np.min(avg_diffs):.2f}%")
        print(f"  Worst case:        {np.max(avg_diffs):.2f}%")
        
        # Agreement categories
        excellent = sum(1 for d in avg_diffs if d < 1.0)
        good = sum(1 for d in avg_diffs if 1.0 <= d < 5.0)
        moderate = sum(1 for d in avg_diffs if 5.0 <= d < 20.0)
        poor = sum(1 for d in avg_diffs if d >= 20.0)
        
        print(f"\n  AGREEMENT DISTRIBUTION:")
        print(f"    Excellent (< 1%):   {excellent}/{len(successful_comparisons)}")
        print(f"    Good (1-5%):        {good}/{len(successful_comparisons)}")
        print(f"    Moderate (5-20%):   {moderate}/{len(successful_comparisons)}")
        print(f"    Poor (> 20%):       {poor}/{len(successful_comparisons)}")
        
        # Detailed results
        print(f"\n  DETAILED RESULTS:")
        for r in successful_comparisons:
            print(f"    {r['test']:<20}: {r['avg_diff_percent']:.1f}% avg, {r['max_diff_percent']:.1f}% max - {r['summary']}")
    
    # Overall assessment
    print(f"\n{'='*70}")
    if both_success >= 0.8 * total_tests and (not successful_comparisons or np.mean(avg_diffs) < 10.0):
        print("✅ JAX IMPLEMENTATION EXCELLENT - Ready for production")
    elif both_success >= 0.6 * total_tests and (not successful_comparisons or np.mean(avg_diffs) < 20.0):
        print("✅ JAX IMPLEMENTATION GOOD - Suitable for most applications")
    elif both_success >= 0.4 * total_tests:
        print("⚠️ JAX IMPLEMENTATION MODERATE - Needs refinement")
    else:
        print("❌ JAX IMPLEMENTATION POOR - Requires major fixes")
    
    print("="*70)

def main():
    """Main test function"""
    
    print("TESTING JAX ATMOSPHERE IMPLEMENTATION WITH REAL MARCS DATA")
    print("=" * 80)
    
    # Check if grid files exist
    grid_dir = Path("Jorg/data/marcs_grids")
    required_files = [
        "SDSS_MARCS_atmospheres.h5",
        "MARCS_metal_poor_atmospheres.h5", 
        "resampled_cool_dwarf_atmospheres.h5"
    ]
    
    missing_files = []
    for filename in required_files:
        if not (grid_dir / filename).exists():
            missing_files.append(filename)
    
    if missing_files:
        print(f"❌ Missing MARCS grid files: {missing_files}")
        print(f"   Expected location: {grid_dir}")
        return
    
    print(f"✅ All MARCS grid files found in {grid_dir}")
    print()
    
    # Run comprehensive tests
    results = test_specific_stellar_types()
    
    # Analyze results
    analyze_results(results)
    
    return results

if __name__ == "__main__":
    results = main()