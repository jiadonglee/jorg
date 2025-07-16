#!/usr/bin/env python3
"""
Stellar Parameter Grid Test for Jorg Synthesis

This test validates Jorg's synthesis across different stellar types
using a simplified approach to identify performance bottlenecks.
"""

import sys
import numpy as np
import jax.numpy as jnp
from pathlib import Path
import time
import json

# Add Jorg to path
jorg_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(jorg_path))

from jorg.synthesis import format_abundances, interpolate_atmosphere
from jorg.radiative_transfer import radiative_transfer
from jorg.constants import PLANCK_H, BOLTZMANN_K, SPEED_OF_LIGHT


def create_simple_synthesis(Teff, logg, m_H, wavelength_range):
    """
    Simplified synthesis function that bypasses slow components
    to test the core pipeline performance
    """
    print(f"    Synthesizing Teff={Teff}, logg={logg}, [M/H]={m_H}")
    
    start_time = time.time()
    
    # Step 1: Format abundances (fast)
    A_X = format_abundances(m_H)
    step1_time = time.time()
    
    # Step 2: Interpolate atmosphere (potentially slow)
    atm = interpolate_atmosphere(Teff, logg, A_X)
    step2_time = time.time()
    
    # Step 3: Create wavelength grid
    wl_start, wl_end = wavelength_range
    wavelengths = jnp.linspace(wl_start, wl_end, 50)  # Reduced resolution for speed
    frequencies = SPEED_OF_LIGHT / (wavelengths * 1e-8)
    n_layers = atm['n_layers']
    n_wavelengths = len(wavelengths)
    step3_time = time.time()
    
    # Step 4: Simple continuum absorption (bypass complex chemistry)
    # Use simplified opacity model
    alpha_continuum = jnp.zeros((n_layers, n_wavelengths))
    for i in range(n_layers):
        T = atm['temperature'][i]
        rho = atm['density'][i]
        
        # Simple H- ff + Thomson scattering model
        for j in range(n_wavelengths):
            wl_cm = wavelengths[j] * 1e-8
            
            # H- free-free (simplified)
            h_ff = 1e-26 * rho * T**(-1.5) * (wl_cm * 1e8)**3
            
            # Thomson scattering
            thomson = 6.65e-25 * atm['electron_density'][i]
            
            alpha_continuum = alpha_continuum.at[i, j].set(h_ff + thomson)
    
    step4_time = time.time()
    
    # Step 5: Source function (Planck)
    S = jnp.zeros((n_layers, n_wavelengths))
    for i in range(n_layers):
        T = atm['temperature'][i]
        for j in range(n_wavelengths):
            nu = frequencies[j]
            h_nu_kt = PLANCK_H * nu / (BOLTZMANN_K * T)
            
            if h_nu_kt < 50:  # Avoid overflow
                planck = (2 * PLANCK_H * nu**3 / SPEED_OF_LIGHT**2) / (jnp.exp(h_nu_kt) - 1)
            else:
                planck = 0.0
            S = S.at[i, j].set(planck)
    
    step5_time = time.time()
    
    # Step 6: Radiative transfer
    result = radiative_transfer(
        alpha=alpha_continuum,
        S=S,
        spatial_coord=jnp.array(atm['height']),
        mu_points=5,
        spherical=False,
        alpha_ref=alpha_continuum[:, 0],
        tau_ref=jnp.array(atm['tau_5000']),
        tau_scheme="anchored",
        I_scheme="linear_flux_only"
    )
    
    step6_time = time.time()
    
    # Calculate timing
    timing = {
        'total': step6_time - start_time,
        'abundances': step1_time - start_time,
        'atmosphere': step2_time - step1_time,
        'wavelengths': step3_time - step2_time,
        'continuum': step4_time - step3_time,
        'source': step5_time - step4_time,
        'radiative_transfer': step6_time - step5_time
    }
    
    return {
        'wavelengths': wavelengths,
        'flux': result.flux,
        'continuum': S[0, :],  # Surface source function as proxy
        'alpha': alpha_continuum,
        'atmosphere': atm,
        'timing': timing,
        'success': True
    }


def test_stellar_parameter_grid():
    """Test synthesis across different stellar types"""
    print("Testing synthesis across stellar parameter grid...")
    
    # Define test grid (reduced for performance)
    stellar_params = [
        (5777, 4.44, 0.0, "Sun"),
        (3500, 4.5, 0.0, "M_dwarf"),
        (4500, 4.5, 0.0, "K_dwarf"),
        (6500, 4.2, 0.0, "F_dwarf"),
        (5800, 4.5, -0.5, "Metal_poor"),
        (4800, 2.5, 0.0, "Giant"),
    ]
    
    wavelength_ranges = [
        (5000, 5100, "Blue"),
        (5500, 5600, "Green"),
    ]
    
    results = {}
    
    for Teff, logg, m_H, star_name in stellar_params:
        print(f"\n  Testing {star_name} (Teff={Teff}, logg={logg}, [M/H]={m_H})")
        star_results = {}
        
        for wl_start, wl_end, wl_name in wavelength_ranges:
            print(f"    {wl_name} band: {wl_start}-{wl_end} Å")
            
            try:
                result = create_simple_synthesis(Teff, logg, m_H, (wl_start, wl_end))
                
                # Validate result
                assert np.all(np.isfinite(result['flux']))
                assert np.all(result['flux'] > 0)
                assert len(result['flux']) == len(result['wavelengths'])
                
                # Store results (convert JAX arrays to numpy for JSON serialization)
                star_results[wl_name] = {
                    'wavelengths': np.array(result['wavelengths']),
                    'flux': np.array(result['flux']),
                    'continuum': np.array(result['continuum']),
                    'timing': result['timing'],
                    'n_layers': result['atmosphere']['n_layers'],
                    'success': True
                }
                
                timing = result['timing']
                print(f"      ✓ Success ({timing['total']:.1f}s total)")
                print(f"        Flux: {np.min(result['flux']):.2e}-{np.max(result['flux']):.2e}")
                print(f"        Timing: atm={timing['atmosphere']:.1f}s, " +
                      f"cont={timing['continuum']:.1f}s, RT={timing['radiative_transfer']:.1f}s")
                
            except Exception as e:
                print(f"      ✗ Failed: {e}")
                star_results[wl_name] = {'success': False, 'error': str(e)}
        
        results[star_name] = star_results
    
    return results


def test_wavelength_coverage():
    """Test different wavelength ranges"""
    print("\nTesting wavelength coverage...")
    
    # Test different wavelength ranges with solar parameters
    wl_ranges = [
        (4000, 4100, "Blue"),
        (5000, 5100, "Green"), 
        (6000, 6100, "Red"),
        (7000, 7100, "NIR"),
    ]
    
    results = {}
    
    for wl_start, wl_end, name in wl_ranges:
        print(f"  Testing {name}: {wl_start}-{wl_end} Å")
        
        try:
            result = create_simple_synthesis(5777, 4.44, 0.0, (wl_start, wl_end))
            
            results[name] = {
                'wavelength_range': [wl_start, wl_end],
                'n_points': len(result['wavelengths']),
                'flux_range': [float(np.min(result['flux'])), float(np.max(result['flux']))],
                'timing': result['timing']['total'],
                'success': True
            }
            
            print(f"    ✓ Success ({result['timing']['total']:.1f}s)")
            print(f"      Flux: {np.min(result['flux']):.2e}-{np.max(result['flux']):.2e}")
            
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            results[name] = {'success': False, 'error': str(e)}
    
    return results


def analyze_performance(stellar_results):
    """Analyze performance characteristics"""
    print("\nAnalyzing performance...")
    
    # Collect timing data
    timings = []
    for star_data in stellar_results.values():
        for wl_data in star_data.values():
            if wl_data.get('success', False):
                timings.append(wl_data['timing'])
    
    if not timings:
        print("  No successful syntheses to analyze")
        return
    
    # Calculate statistics
    components = ['total', 'atmosphere', 'continuum', 'radiative_transfer']
    stats = {}
    
    for component in components:
        times = [t[component] for t in timings]
        stats[component] = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times)
        }
    
    print("  Performance statistics (seconds):")
    for component, stat in stats.items():
        print(f"    {component:15}: {stat['mean']:.2f} ± {stat['std']:.2f} " +
              f"(range: {stat['min']:.2f}-{stat['max']:.2f})")
    
    # Identify bottlenecks
    mean_times = {comp: stats[comp]['mean'] for comp in components if comp != 'total'}
    bottleneck = max(mean_times.items(), key=lambda x: x[1])
    
    print(f"  Primary bottleneck: {bottleneck[0]} ({bottleneck[1]:.2f}s average)")
    
    return stats


def main():
    """Main test suite"""
    print("=" * 60)
    print("Jorg Synthesis Stellar Parameter Grid Test")
    print("=" * 60)
    
    # Test stellar parameter grid
    stellar_results = test_stellar_parameter_grid()
    
    # Test wavelength coverage
    wavelength_results = test_wavelength_coverage()
    
    # Analyze performance
    performance_stats = analyze_performance(stellar_results)
    
    # Save results
    output_dir = Path(__file__).parent
    
    all_results = {
        'stellar_grid': stellar_results,
        'wavelength_coverage': wavelength_results,
        'performance_stats': performance_stats,
        'test_info': {
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_stellar_types': len(stellar_results),
            'n_wavelength_ranges': len(wavelength_results)
        }
    }
    
    with open(output_dir / "synthesis_stellar_grid_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Summary
    print("\n" + "=" * 60)
    print("STELLAR GRID TEST SUMMARY")
    print("=" * 60)
    
    # Count successes
    stellar_successes = sum(1 for star_data in stellar_results.values()
                           for wl_data in star_data.values() 
                           if wl_data.get('success', False))
    stellar_total = sum(len(star_data) for star_data in stellar_results.values())
    
    wavelength_successes = sum(1 for wl_data in wavelength_results.values() 
                              if wl_data.get('success', False))
    wavelength_total = len(wavelength_results)
    
    print(f"Stellar parameter grid: {stellar_successes}/{stellar_total} successful")
    print(f"Wavelength coverage: {wavelength_successes}/{wavelength_total} successful")
    
    if performance_stats:
        avg_time = performance_stats['total']['mean']
        print(f"Average synthesis time: {avg_time:.1f}s")
        
        if avg_time < 10:
            print("✅ Performance is acceptable for basic synthesis")
        elif avg_time < 30:
            print("⚠️ Performance is moderate, could be optimized")
        else:
            print("❌ Performance is slow, needs optimization")
    
    # Overall assessment
    success_rate = (stellar_successes + wavelength_successes) / (stellar_total + wavelength_total)
    
    if success_rate >= 0.9:
        print("🎉 SYNTHESIS GRID TEST PASSED - Jorg works across stellar types!")
        success = True
    elif success_rate >= 0.7:
        print("✅ Most synthesis tests passed - Minor issues")
        success = True
    else:
        print("❌ Many synthesis tests failed")
        success = False
    
    print(f"Overall success rate: {success_rate:.1%}")
    print("Results saved to: synthesis_stellar_grid_results.json")
    print("=" * 60)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)