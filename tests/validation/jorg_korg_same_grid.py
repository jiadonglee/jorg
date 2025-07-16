#!/usr/bin/env python3
"""
Jorg vs Korg Comparison with Same Wavelength Grid

This script tests Jorg using the exact same wavelength grid as Korg
for a fair performance and accuracy comparison.
"""

import sys
import time
import json
import numpy as np
from pathlib import Path

# Add Jorg to path
jorg_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(jorg_path))

def load_korg_reference():
    """Load Korg reference data"""
    try:
        with open('korg_reference_final.json') as f:
            korg_data = json.load(f)
        
        print("📊 Loaded Korg reference data:")
        print(f"   Time: {korg_data['timing']:.1f}s")
        print(f"   Points: {korg_data['n_points']}")
        print(f"   Wavelength range: {min(korg_data['wavelengths']):.1f}-{max(korg_data['wavelengths']):.1f} Å")
        print(f"   Flux mean: {korg_data['flux_stats']['mean']:.3f}")
        print(f"   Continuum mean: {korg_data['continuum_stats']['mean']:.2e}")
        
        return korg_data
        
    except FileNotFoundError:
        print("❌ Korg reference data not found")
        return None
    except Exception as e:
        print(f"❌ Error loading Korg data: {e}")
        return None


def test_jorg_same_grid(korg_wavelengths):
    """Test Jorg with exact same wavelength grid as Korg"""
    print(f"\n🚀 Testing Jorg with Same Wavelength Grid")
    print("=" * 50)
    
    try:
        from jorg.synthesis_optimized import synthesize_fast
        from jorg.synthesis import format_abundances, interpolate_atmosphere
        
        # Same parameters as Korg test
        Teff, logg, m_H = 5777, 4.44, 0.0
        vmic = 1.0
        
        print(f"Parameters: Teff={Teff}K, logg={logg}, [M/H]={m_H}")
        print(f"Wavelengths: {len(korg_wavelengths)} points (same as Korg)")
        print(f"Range: {min(korg_wavelengths):.1f}-{max(korg_wavelengths):.1f} Å")
        print(f"Microturbulence: {vmic} km/s")
        
        # Convert to JAX array
        import jax.numpy as jnp
        wavelengths = jnp.array(korg_wavelengths)
        
        print(f"\nStep 1: Format abundances...")
        A_X = format_abundances(m_H)
        
        print(f"Step 2: Interpolate atmosphere...")
        atm = interpolate_atmosphere(Teff, logg, A_X)
        
        print(f"Step 3: Run synthesis with {len(wavelengths)} wavelengths...")
        start_time = time.time()
        
        # Use optimized synthesis with same wavelength grid
        result = synthesize_fast(
            atm=atm,
            linelist=None,  # No lines for speed
            A_X=A_X,
            wavelengths=wavelengths,
            vmic=vmic,
            hydrogen_lines=False,  # Disable for speed
            mu_values=5,  # Reasonable number of mu points
            verbose=True
        )
        
        elapsed = time.time() - start_time
        
        print(f"\n✅ SUCCESS! Jorg synthesis completed")
        print(f"   Time: {elapsed:.1f}s")
        print(f"   Wavelengths: {len(result.wavelengths)} points")
        print(f"   Flux range: {np.min(result.flux):.3f} - {np.max(result.flux):.3f}")
        print(f"   Flux mean: {np.mean(result.flux):.3f}")
        print(f"   Continuum mean: {np.mean(result.cntm):.2e}")
        
        # Validate outputs
        assert len(result.wavelengths) == len(korg_wavelengths), "Wavelength mismatch"
        assert np.all(np.isfinite(result.flux)), "Non-finite flux values"
        assert np.all(result.flux > 0), "Negative flux values"
        assert np.all(np.isfinite(result.cntm)), "Non-finite continuum"
        
        print(f"   ✓ All validation checks passed")
        
        # Apply rectification to match Korg
        flux_rectified = result.flux / result.cntm
        
        return {
            'wavelengths': np.array(result.wavelengths),
            'flux': np.array(flux_rectified),  # Rectified like Korg
            'continuum': np.array(result.cntm),
            'timing': elapsed,
            'n_points': len(result.wavelengths),
            'flux_stats': {
                'min': float(np.min(flux_rectified)),
                'max': float(np.max(flux_rectified)),
                'mean': float(np.mean(flux_rectified)),
                'std': float(np.std(flux_rectified))
            },
            'continuum_stats': {
                'min': float(np.min(result.cntm)),
                'max': float(np.max(result.cntm)),
                'mean': float(np.mean(result.cntm))
            },
            'success': True
        }
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def compare_results_detailed(jorg_result, korg_result):
    """Detailed comparison of Jorg vs Korg results"""
    print(f"\n📊 Detailed Jorg vs Korg Comparison")
    print("=" * 50)
    
    if not jorg_result.get('success', False):
        print("❌ Jorg failed, cannot compare")
        return
    
    if not korg_result.get('success', False):
        print("❌ Korg failed, cannot compare")
        return
    
    # Extract data
    wl_jorg = jorg_result['wavelengths']
    flux_jorg = jorg_result['flux']
    cont_jorg = jorg_result['continuum']
    time_jorg = jorg_result['timing']
    
    wl_korg = np.array(korg_result['wavelengths'])
    flux_korg = np.array(korg_result['flux'])
    cont_korg = np.array(korg_result['continuum'])
    time_korg = korg_result['timing']
    
    print(f"Grid comparison:")
    print(f"   Jorg points: {len(wl_jorg)}")
    print(f"   Korg points: {len(wl_korg)}")
    print(f"   Grid match: {'✅ YES' if len(wl_jorg) == len(wl_korg) else '❌ NO'}")
    
    # Wavelength grid comparison
    if len(wl_jorg) == len(wl_korg):
        wl_diff = np.abs(wl_jorg - wl_korg)
        print(f"   Wavelength difference: max {np.max(wl_diff):.3e} Å")
        
        # Point-by-point flux comparison
        flux_diff = np.abs(flux_jorg - flux_korg)
        rel_diff = flux_diff / np.abs(flux_korg)
        
        print(f"\nFlux comparison (point-by-point):")
        print(f"   Jorg flux: {np.min(flux_jorg):.3f} - {np.max(flux_jorg):.3f} (mean: {np.mean(flux_jorg):.3f})")
        print(f"   Korg flux: {np.min(flux_korg):.3f} - {np.max(flux_korg):.3f} (mean: {np.mean(flux_korg):.3f})")
        print(f"   Absolute difference: max {np.max(flux_diff):.3f}, mean {np.mean(flux_diff):.3f}")
        print(f"   Relative difference: max {np.max(rel_diff):.1%}, mean {np.mean(rel_diff):.1%}")
        print(f"   RMS difference: {np.sqrt(np.mean(flux_diff**2)):.3f}")
        
        # Agreement assessment
        if np.max(rel_diff) < 0.01:
            agreement = "Excellent (< 1%)"
        elif np.max(rel_diff) < 0.05:
            agreement = "Good (< 5%)"
        elif np.max(rel_diff) < 0.1:
            agreement = "Acceptable (< 10%)"
        else:
            agreement = "Poor (> 10%)"
        
        print(f"   Agreement: {agreement}")
        
        # Continuum comparison
        cont_diff = np.abs(cont_jorg - cont_korg) / np.abs(cont_korg)
        print(f"\nContinuum comparison:")
        print(f"   Jorg: {np.mean(cont_jorg):.2e}")
        print(f"   Korg: {np.mean(cont_korg):.2e}")
        print(f"   Relative difference: {np.mean(cont_diff):.1%}")
        
    else:
        print(f"   ⚠ Different grids, interpolating for comparison...")
        
        # Interpolate to common grid
        wl_common = wl_korg  # Use Korg grid as reference
        flux_jorg_interp = np.interp(wl_common, wl_jorg, flux_jorg)
        
        flux_diff = np.abs(flux_jorg_interp - flux_korg)
        rel_diff = flux_diff / np.abs(flux_korg)
        
        print(f"   Interpolated comparison:")
        print(f"   Max relative difference: {np.max(rel_diff):.1%}")
        print(f"   Mean relative difference: {np.mean(rel_diff):.1%}")
    
    # Performance comparison
    speedup = time_korg / time_jorg if time_jorg > 0 else float('inf')
    print(f"\nPerformance comparison:")
    print(f"   Jorg time: {time_jorg:.1f}s")
    print(f"   Korg time: {time_korg:.1f}s")
    print(f"   Speedup: {speedup:.1f}x")
    
    # Calculate points per second
    jorg_pps = len(wl_jorg) / time_jorg if time_jorg > 0 else 0
    korg_pps = len(wl_korg) / time_korg if time_korg > 0 else 0
    
    print(f"   Jorg throughput: {jorg_pps:.0f} points/second")
    print(f"   Korg throughput: {korg_pps:.0f} points/second")
    print(f"   Throughput ratio: {jorg_pps/korg_pps:.1f}x" if korg_pps > 0 else "")
    
    return {
        'agreement': agreement if len(wl_jorg) == len(wl_korg) else "Different grids",
        'max_rel_diff': np.max(rel_diff) if len(wl_jorg) == len(wl_korg) else None,
        'speedup': speedup,
        'same_grid': len(wl_jorg) == len(wl_korg)
    }


def create_comparison_plot(jorg_result, korg_result):
    """Create detailed comparison plot"""
    print(f"\n📈 Creating Detailed Comparison Plot")
    print("=" * 50)
    
    try:
        import matplotlib.pyplot as plt
        
        if not (jorg_result.get('success', False) and korg_result.get('success', False)):
            print("Cannot create plot - missing results")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Jorg vs Korg: Same Wavelength Grid Comparison', fontsize=16)
        
        wl_jorg = jorg_result['wavelengths']
        flux_jorg = jorg_result['flux']
        wl_korg = np.array(korg_result['wavelengths'])
        flux_korg = np.array(korg_result['flux'])
        
        # Plot 1: Spectrum overlay
        axes[0, 0].plot(wl_korg, flux_korg, 'r-', linewidth=1, alpha=0.8, label='Korg.jl')
        axes[0, 0].plot(wl_jorg, flux_jorg, 'b-', linewidth=1.5, alpha=0.9, label='Jorg (optimized)')
        axes[0, 0].set_xlabel('Wavelength (Å)')
        axes[0, 0].set_ylabel('Rectified Flux')
        axes[0, 0].set_title('Spectrum Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Difference plot
        if len(wl_jorg) == len(wl_korg):
            diff = flux_jorg - flux_korg
            axes[0, 1].plot(wl_jorg, diff * 100, 'g-', linewidth=1)
            axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            axes[0, 1].set_xlabel('Wavelength (Å)')
            axes[0, 1].set_ylabel('Flux Difference (%)')
            axes[0, 1].set_title('Jorg - Korg Difference')
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'Different\nWavelength\nGrids', 
                           transform=axes[0, 1].transAxes, ha='center', va='center')
            axes[0, 1].set_title('Grid Mismatch')
        
        # Plot 3: Performance bars
        times = [jorg_result['timing'], korg_result['timing']]
        labels = ['Jorg\n(optimized)', 'Korg\n(reference)']
        colors = ['blue', 'red']
        
        bars = axes[1, 0].bar(labels, times, color=colors, alpha=0.7)
        axes[1, 0].set_ylabel('Time (s)')
        axes[1, 0].set_title('Performance Comparison')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add speedup annotation
        speedup = korg_result['timing'] / jorg_result['timing']
        axes[1, 0].text(0.5, max(times) * 0.8, f'{speedup:.1f}x\nspeedup', 
                       ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Plot 4: Flux statistics
        stats = ['Mean', 'Min', 'Max', 'Std']
        jorg_stats = [jorg_result['flux_stats']['mean'], jorg_result['flux_stats']['min'],
                     jorg_result['flux_stats']['max'], jorg_result['flux_stats']['std']]
        korg_stats = [korg_result['flux_stats']['mean'], korg_result['flux_stats']['min'],
                     korg_result['flux_stats']['max'], korg_result['flux_stats']['std']]
        
        x = np.arange(len(stats))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, jorg_stats, width, label='Jorg', alpha=0.7, color='blue')
        axes[1, 1].bar(x + width/2, korg_stats, width, label='Korg', alpha=0.7, color='red')
        axes[1, 1].set_xlabel('Statistic')
        axes[1, 1].set_ylabel('Flux Value')
        axes[1, 1].set_title('Flux Statistics')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(stats)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_file = Path(__file__).parent / "jorg_korg_same_grid_comparison.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Comparison plot saved to {output_file}")
        
        try:
            plt.show()
            print("✓ Plot displayed")
        except:
            print("⚠ Cannot display plot (no GUI)")
        
        plt.close()
        
    except ImportError:
        print("⚠ Matplotlib not available, skipping plot")
    except Exception as e:
        print(f"✗ Plotting failed: {e}")


def main():
    """Main comparison with same wavelength grid"""
    print("🌟 Jorg vs Korg: Same Wavelength Grid Test")
    print("=" * 60)
    
    # Load Korg reference
    korg_data = load_korg_reference()
    if not korg_data:
        print("❌ Cannot proceed without Korg reference data")
        return False
    
    # Test Jorg with same wavelength grid
    jorg_result = test_jorg_same_grid(korg_data['wavelengths'])
    
    # Compare results
    if jorg_result.get('success', False):
        comparison = compare_results_detailed(jorg_result, korg_data)
        create_comparison_plot(jorg_result, korg_data)
    
    # Final summary
    print(f"\n" + "=" * 60)
    print("SAME GRID COMPARISON SUMMARY")
    print("=" * 60)
    
    if jorg_result.get('success', False):
        print(f"✅ SUCCESS: Jorg synthesis with same grid as Korg")
        print(f"")
        print(f"📊 GRID COMPARISON:")
        print(f"   • Korg points: {korg_data['n_points']}")
        print(f"   • Jorg points: {jorg_result['n_points']}")
        print(f"   • Grid match: {'✅ YES' if jorg_result['n_points'] == korg_data['n_points'] else '❌ NO'}")
        
        print(f"")
        print(f"⚡ PERFORMANCE:")
        speedup = korg_data['timing'] / jorg_result['timing']
        print(f"   • Korg time: {korg_data['timing']:.1f}s")
        print(f"   • Jorg time: {jorg_result['timing']:.1f}s")
        print(f"   • Speedup: {speedup:.1f}x")
        
        print(f"")
        print(f"🎯 ACCURACY:")
        print(f"   • Korg flux mean: {korg_data['flux_stats']['mean']:.3f}")
        print(f"   • Jorg flux mean: {jorg_result['flux_stats']['mean']:.3f}")
        
        if 'comparison' in locals() and comparison:
            print(f"   • Agreement: {comparison.get('agreement', 'Unknown')}")
            if comparison.get('max_rel_diff'):
                print(f"   • Max difference: {comparison['max_rel_diff']:.1%}")
        
        # Overall assessment
        if jorg_result['timing'] < korg_data['timing'] and jorg_result['n_points'] == korg_data['n_points']:
            print(f"\n🏆 EXCELLENT: Jorg matches Korg grid with {speedup:.1f}x speedup!")
        elif jorg_result['n_points'] == korg_data['n_points']:
            print(f"\n✅ GOOD: Jorg matches Korg grid")
        else:
            print(f"\n⚠ PARTIAL: Different wavelength grids")
        
        success = True
    else:
        print(f"❌ FAILED: Jorg synthesis with same grid failed")
        print(f"   Error: {jorg_result.get('error', 'Unknown')}")
        success = False
    
    print("=" * 60)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)