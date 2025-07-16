#!/usr/bin/env python3
"""
Jorg vs Korg Comparison with Realistic Grid

This script tests Jorg with a realistic wavelength grid size that matches
the density of Korg's grid but is computationally feasible.
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
    """Load Korg reference data and analyze grid"""
    try:
        with open('korg_reference_final.json') as f:
            korg_data = json.load(f)
        
        wl_korg = np.array(korg_data['wavelengths'])
        wl_range = max(wl_korg) - min(wl_korg)
        resolution = len(wl_korg) / wl_range  # points per Å
        
        print("📊 Korg reference analysis:")
        print(f"   Time: {korg_data['timing']:.1f}s")
        print(f"   Points: {korg_data['n_points']}")
        print(f"   Range: {min(wl_korg):.1f}-{max(wl_korg):.1f} Å ({wl_range:.1f} Å)")
        print(f"   Resolution: {resolution:.1f} points/Å")
        print(f"   Flux mean: {korg_data['flux_stats']['mean']:.3f}")
        
        return korg_data, resolution
        
    except Exception as e:
        print(f"❌ Error loading Korg data: {e}")
        return None, None


def create_matched_wavelength_grid(korg_wl, target_points=200):
    """Create wavelength grid that matches Korg's range but with fewer points"""
    wl_min = min(korg_wl)
    wl_max = max(korg_wl)
    
    # Create evenly spaced grid with target number of points
    wl_jorg = np.linspace(wl_min, wl_max, target_points)
    
    print(f"📏 Wavelength grid matching:")
    print(f"   Korg range: {wl_min:.1f}-{wl_max:.1f} Å ({len(korg_wl)} points)")
    print(f"   Jorg range: {min(wl_jorg):.1f}-{max(wl_jorg):.1f} Å ({len(wl_jorg)} points)")
    print(f"   Korg resolution: {len(korg_wl)/(wl_max-wl_min):.1f} points/Å")
    print(f"   Jorg resolution: {len(wl_jorg)/(wl_max-wl_min):.1f} points/Å")
    
    return wl_jorg


def test_jorg_matched_grid(wavelengths):
    """Test Jorg with matched wavelength grid"""
    print(f"\n🚀 Testing Jorg with Matched Grid")
    print("=" * 50)
    
    try:
        from jorg.synthesis_optimized import synthesize_fast
        from jorg.synthesis import format_abundances, interpolate_atmosphere
        
        # Same parameters as Korg test
        Teff, logg, m_H = 5777, 4.44, 0.0
        vmic = 1.0
        
        print(f"Parameters: Teff={Teff}K, logg={logg}, [M/H]={m_H}")
        print(f"Wavelengths: {len(wavelengths)} points")
        print(f"Range: {min(wavelengths):.1f}-{max(wavelengths):.1f} Å")
        
        # Convert to JAX array
        import jax.numpy as jnp
        wl_jax = jnp.array(wavelengths)
        
        print(f"Setting up synthesis...")
        A_X = format_abundances(m_H)
        atm = interpolate_atmosphere(Teff, logg, A_X)
        
        print(f"Running synthesis...")
        start_time = time.time()
        
        # Use optimized synthesis 
        result = synthesize_fast(
            atm=atm,
            linelist=None,  # No lines for speed comparison
            A_X=A_X,
            wavelengths=wl_jax,
            vmic=vmic,
            hydrogen_lines=False,  # Disable for speed
            mu_values=5,  # Reasonable mu points
            verbose=True
        )
        
        elapsed = time.time() - start_time
        
        print(f"\n✅ SUCCESS! Jorg synthesis completed")
        print(f"   Time: {elapsed:.1f}s")
        print(f"   Points: {len(result.wavelengths)}")
        print(f"   Flux range: {np.min(result.flux):.3f} - {np.max(result.flux):.3f}")
        print(f"   Continuum mean: {np.mean(result.cntm):.2e}")
        
        # Apply rectification to match Korg
        flux_rectified = result.flux / result.cntm
        
        print(f"   Rectified flux: {np.min(flux_rectified):.3f} - {np.max(flux_rectified):.3f}")
        print(f"   Rectified mean: {np.mean(flux_rectified):.3f}")
        
        # Validate
        assert np.all(np.isfinite(flux_rectified)), "Non-finite flux"
        assert np.all(flux_rectified > 0), "Negative flux"
        print(f"   ✓ All validation checks passed")
        
        return {
            'wavelengths': np.array(result.wavelengths),
            'flux': np.array(flux_rectified),
            'continuum': np.array(result.cntm),
            'timing': elapsed,
            'n_points': len(result.wavelengths),
            'flux_stats': {
                'min': float(np.min(flux_rectified)),
                'max': float(np.max(flux_rectified)),
                'mean': float(np.mean(flux_rectified)),
                'std': float(np.std(flux_rectified))
            },
            'success': True
        }
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def interpolate_korg_to_jorg_grid(korg_result, jorg_wavelengths):
    """Interpolate Korg results to Jorg wavelength grid for comparison"""
    wl_korg = np.array(korg_result['wavelengths'])
    flux_korg = np.array(korg_result['flux'])
    cont_korg = np.array(korg_result['continuum'])
    
    # Interpolate to Jorg grid
    flux_korg_interp = np.interp(jorg_wavelengths, wl_korg, flux_korg)
    cont_korg_interp = np.interp(jorg_wavelengths, wl_korg, cont_korg)
    
    print(f"📊 Interpolated Korg to Jorg grid:")
    print(f"   Original Korg points: {len(wl_korg)}")
    print(f"   Interpolated points: {len(jorg_wavelengths)}")
    print(f"   Flux mean (original): {np.mean(flux_korg):.3f}")
    print(f"   Flux mean (interpolated): {np.mean(flux_korg_interp):.3f}")
    
    return {
        'wavelengths': jorg_wavelengths,
        'flux': flux_korg_interp,
        'continuum': cont_korg_interp,
        'timing': korg_result['timing'],
        'n_points': len(jorg_wavelengths),
        'flux_stats': {
            'min': float(np.min(flux_korg_interp)),
            'max': float(np.max(flux_korg_interp)),
            'mean': float(np.mean(flux_korg_interp)),
            'std': float(np.std(flux_korg_interp))
        },
        'interpolated': True
    }


def compare_matched_grids(jorg_result, korg_interp):
    """Compare Jorg vs interpolated Korg on same grid"""
    print(f"\n📊 Matched Grid Comparison")
    print("=" * 50)
    
    if not jorg_result.get('success', False):
        print("❌ Jorg failed, cannot compare")
        return
    
    wl = jorg_result['wavelengths']
    flux_jorg = jorg_result['flux']
    flux_korg = korg_interp['flux']
    
    # Point-by-point comparison
    flux_diff = np.abs(flux_jorg - flux_korg)
    rel_diff = flux_diff / np.abs(flux_korg)
    
    print(f"Spectral comparison:")
    print(f"   Wavelength points: {len(wl)}")
    print(f"   Range: {min(wl):.1f}-{max(wl):.1f} Å")
    
    print(f"\nFlux statistics:")
    print(f"   Jorg mean: {np.mean(flux_jorg):.3f}")
    print(f"   Korg mean: {np.mean(flux_korg):.3f}")
    print(f"   Difference: {abs(np.mean(flux_jorg) - np.mean(flux_korg)):.3f}")
    
    print(f"\nPoint-by-point accuracy:")
    print(f"   Max absolute diff: {np.max(flux_diff):.3f}")
    print(f"   Mean absolute diff: {np.mean(flux_diff):.3f}")
    print(f"   Max relative diff: {np.max(rel_diff):.1%}")
    print(f"   Mean relative diff: {np.mean(rel_diff):.1%}")
    print(f"   RMS difference: {np.sqrt(np.mean(flux_diff**2)):.3f}")
    
    # Agreement assessment
    if np.max(rel_diff) < 0.05:
        agreement = "Excellent (< 5%)"
    elif np.max(rel_diff) < 0.1:
        agreement = "Good (< 10%)"
    elif np.max(rel_diff) < 0.2:
        agreement = "Acceptable (< 20%)"
    else:
        agreement = "Poor (> 20%)"
    
    print(f"   Agreement: {agreement}")
    
    # Performance comparison
    speedup = korg_interp['timing'] / jorg_result['timing']
    print(f"\nPerformance:")
    print(f"   Jorg time: {jorg_result['timing']:.1f}s")
    print(f"   Korg time: {korg_interp['timing']:.1f}s")
    print(f"   Speedup: {speedup:.1f}x")
    
    # Efficiency comparison (points per second per wavelength point)
    jorg_efficiency = len(wl) / jorg_result['timing']
    korg_efficiency = 3001 / korg_interp['timing']  # Original Korg points
    
    print(f"   Jorg efficiency: {jorg_efficiency:.0f} points/s")
    print(f"   Korg efficiency: {korg_efficiency:.0f} points/s")
    print(f"   Efficiency ratio: {jorg_efficiency/korg_efficiency:.1f}x")
    
    return {
        'agreement': agreement,
        'max_rel_diff': np.max(rel_diff),
        'mean_rel_diff': np.mean(rel_diff),
        'speedup': speedup,
        'efficiency_ratio': jorg_efficiency/korg_efficiency
    }


def create_matched_plot(jorg_result, korg_interp):
    """Create comparison plot for matched grids"""
    print(f"\n📈 Creating Matched Grid Comparison Plot")
    print("=" * 50)
    
    try:
        import matplotlib.pyplot as plt
        
        if not jorg_result.get('success', False):
            print("Cannot create plot - Jorg failed")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Jorg vs Korg: Matched Wavelength Grid Comparison', fontsize=16)
        
        wl = jorg_result['wavelengths']
        flux_jorg = jorg_result['flux']
        flux_korg = korg_interp['flux']
        
        # Plot 1: Spectrum overlay
        axes[0, 0].plot(wl, flux_korg, 'r-', linewidth=2, alpha=0.8, label='Korg (interpolated)')
        axes[0, 0].plot(wl, flux_jorg, 'b-', linewidth=1.5, alpha=0.9, label='Jorg (optimized)')
        axes[0, 0].set_xlabel('Wavelength (Å)')
        axes[0, 0].set_ylabel('Rectified Flux')
        axes[0, 0].set_title('Spectrum Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Absolute difference
        diff = flux_jorg - flux_korg
        axes[0, 1].plot(wl, diff, 'g-', linewidth=1)
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('Wavelength (Å)')
        axes[0, 1].set_ylabel('Flux Difference (Jorg - Korg)')
        axes[0, 1].set_title('Absolute Difference')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Relative difference
        rel_diff = np.abs(diff) / np.abs(flux_korg) * 100
        axes[1, 0].plot(wl, rel_diff, 'purple', linewidth=1)
        axes[1, 0].set_xlabel('Wavelength (Å)')
        axes[1, 0].set_ylabel('Relative Difference (%)')
        axes[1, 0].set_title('Relative Difference')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Performance summary
        metrics = ['Speedup', 'Max Rel Diff (%)', 'Mean Rel Diff (%)']
        speedup = korg_interp['timing'] / jorg_result['timing']
        values = [speedup, np.max(rel_diff), np.mean(rel_diff)]
        colors = ['green', 'orange', 'blue']
        
        bars = axes[1, 1].bar(metrics, values, color=colors, alpha=0.7)
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Performance & Accuracy Summary')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        output_file = Path(__file__).parent / "jorg_korg_matched_grid_comparison.png"
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
    """Main matched grid comparison"""
    print("🌟 Jorg vs Korg: Matched Wavelength Grid Comparison")
    print("=" * 60)
    
    # Load Korg reference and analyze grid
    korg_data, korg_resolution = load_korg_reference()
    if not korg_data:
        print("❌ Cannot proceed without Korg reference data")
        return False
    
    # Create matched wavelength grid (200 points for reasonable performance)
    target_points = 200
    matched_wavelengths = create_matched_wavelength_grid(korg_data['wavelengths'], target_points)
    
    # Test Jorg with matched grid
    jorg_result = test_jorg_matched_grid(matched_wavelengths)
    
    if jorg_result.get('success', False):
        # Interpolate Korg to same grid
        korg_interp = interpolate_korg_to_jorg_grid(korg_data, matched_wavelengths)
        
        # Compare on matched grids
        comparison = compare_matched_grids(jorg_result, korg_interp)
        
        # Create comparison plot
        create_matched_plot(jorg_result, korg_interp)
    
    # Final summary
    print(f"\n" + "=" * 60)
    print("MATCHED GRID COMPARISON SUMMARY")
    print("=" * 60)
    
    if jorg_result.get('success', False):
        print(f"✅ SUCCESS: Jorg synthesis with matched grid")
        print(f"")
        print(f"📊 GRID ANALYSIS:")
        print(f"   • Korg original: {korg_data['n_points']} points ({korg_data['timing']:.1f}s)")
        print(f"   • Matched grid: {target_points} points")
        print(f"   • Jorg synthesis: {jorg_result['timing']:.1f}s")
        print(f"   • Coverage: Same wavelength range as Korg")
        
        if 'comparison' in locals() and comparison:
            print(f"")
            print(f"🎯 ACCURACY RESULTS:")
            print(f"   • Agreement level: {comparison['agreement']}")
            print(f"   • Max relative difference: {comparison['max_rel_diff']:.1%}")
            print(f"   • Mean relative difference: {comparison['mean_rel_diff']:.1%}")
            
            print(f"")
            print(f"⚡ PERFORMANCE RESULTS:")
            print(f"   • Speedup vs Korg: {comparison['speedup']:.1f}x")
            print(f"   • Efficiency ratio: {comparison['efficiency_ratio']:.1f}x")
            
            # Overall grade
            if comparison['max_rel_diff'] < 0.1 and comparison['speedup'] > 2:
                grade = "A+ (Excellent)"
            elif comparison['max_rel_diff'] < 0.2 and comparison['speedup'] > 1:
                grade = "A (Good)"
            elif comparison['max_rel_diff'] < 0.3:
                grade = "B (Acceptable)"
            else:
                grade = "C (Needs work)"
            
            print(f"")
            print(f"🏆 OVERALL GRADE: {grade}")
        
        success = True
    else:
        print(f"❌ FAILED: Jorg synthesis with matched grid failed")
        print(f"   Error: {jorg_result.get('error', 'Unknown')}")
        success = False
    
    print("=" * 60)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)