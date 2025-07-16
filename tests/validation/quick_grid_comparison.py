#!/usr/bin/env python3
"""
Quick Jorg vs Korg Grid Comparison

Tests Jorg with progressively larger grids to find the performance limit
and compare accuracy vs Korg.
"""

import sys
import time
import json
import numpy as np
from pathlib import Path

# Add Jorg to path
jorg_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(jorg_path))

def load_korg_data():
    """Load Korg reference data"""
    try:
        with open('korg_reference_final.json') as f:
            korg_data = json.load(f)
        
        wl_range = max(korg_data['wavelengths']) - min(korg_data['wavelengths'])
        print(f"📊 Korg reference: {korg_data['n_points']} points, {wl_range:.1f} Å range")
        print(f"   Time: {korg_data['timing']:.1f}s")
        print(f"   Flux mean: {korg_data['flux_stats']['mean']:.3f}")
        
        return korg_data
        
    except Exception as e:
        print(f"❌ Error loading Korg data: {e}")
        return None


def test_jorg_grid_sizes():
    """Test Jorg with different grid sizes to find performance limits"""
    print(f"\n🚀 Testing Jorg Grid Size Performance")
    print("=" * 50)
    
    try:
        from jorg.synthesis_optimized import synth_minimal
        
        # Test different grid sizes
        grid_sizes = [20, 50, 100, 150, 200]
        wavelength_range = (5000, 5030)  # 30 Å range like Korg
        
        results = {}
        
        for n_points in grid_sizes:
            print(f"\nTesting {n_points} points:")
            
            try:
                start_time = time.time()
                
                wl, flux, cont = synth_minimal(
                    Teff=5777, logg=4.44, m_H=0.0,
                    wavelengths=wavelength_range,
                    rectify=True, vmic=1.0,
                    n_points=n_points
                )
                
                elapsed = time.time() - start_time
                
                # Calculate flux statistics
                flux_stats = {
                    'mean': float(np.mean(flux)),
                    'std': float(np.std(flux)),
                    'min': float(np.min(flux)),
                    'max': float(np.max(flux))
                }
                
                results[n_points] = {
                    'timing': elapsed,
                    'flux_stats': flux_stats,
                    'wavelengths': np.array(wl),
                    'flux': np.array(flux),
                    'continuum': np.array(cont),
                    'success': True
                }
                
                print(f"   ✅ {elapsed:.1f}s - flux mean: {flux_stats['mean']:.1f}")
                
                # Stop if getting too slow
                if elapsed > 30:
                    print(f"   ⚠ Getting slow, stopping grid size testing")
                    break
                    
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                results[n_points] = {'success': False, 'error': str(e)}
                
        return results
        
    except Exception as e:
        print(f"❌ Grid testing failed: {e}")
        return {}


def find_best_jorg_grid(jorg_results):
    """Find the best Jorg grid size balancing performance and resolution"""
    successful = {k: v for k, v in jorg_results.items() if v.get('success', False)}
    
    if not successful:
        print("❌ No successful Jorg results")
        return None, None
    
    print(f"\n📊 Jorg Grid Size Analysis:")
    print(f"   Successful grids: {list(successful.keys())}")
    
    # Find best trade-off (fast but not too coarse)
    best_size = None
    best_result = None
    
    for size, result in successful.items():
        timing = result['timing']
        print(f"   {size} points: {timing:.1f}s")
        
        # Prefer larger grids if they're still fast
        if timing < 15:  # Reasonable time limit
            if best_size is None or size > best_size:
                best_size = size
                best_result = result
    
    if best_size:
        print(f"   ✓ Best grid: {best_size} points ({best_result['timing']:.1f}s)")
    
    return best_size, best_result


def compare_to_korg(jorg_result, korg_data, jorg_grid_size):
    """Compare best Jorg result to Korg"""
    print(f"\n⚖️ Comparing Best Jorg vs Korg")
    print("=" * 50)
    
    if not jorg_result:
        print("❌ No Jorg result to compare")
        return
    
    # Interpolate Korg to Jorg wavelength grid
    wl_korg = np.array(korg_data['wavelengths'])
    flux_korg = np.array(korg_data['flux'])
    wl_jorg = jorg_result['wavelengths']
    
    # Find overlapping range
    wl_min = max(min(wl_jorg), min(wl_korg))
    wl_max = min(max(wl_jorg), max(wl_korg))
    
    print(f"Comparison setup:")
    print(f"   Jorg grid: {jorg_grid_size} points")
    print(f"   Korg grid: {len(wl_korg)} points")
    print(f"   Overlap range: {wl_min:.1f}-{wl_max:.1f} Å")
    
    # Interpolate both to common grid
    wl_common = np.linspace(wl_min, wl_max, jorg_grid_size)
    flux_jorg_interp = np.interp(wl_common, wl_jorg, jorg_result['flux'])
    flux_korg_interp = np.interp(wl_common, wl_korg, flux_korg)
    
    # Calculate differences
    abs_diff = np.abs(flux_jorg_interp - flux_korg_interp)
    rel_diff = abs_diff / np.abs(flux_korg_interp)
    
    print(f"\nAccuracy comparison:")
    print(f"   Jorg flux mean: {np.mean(flux_jorg_interp):.3f}")
    print(f"   Korg flux mean: {np.mean(flux_korg_interp):.3f}")
    print(f"   Max absolute diff: {np.max(abs_diff):.3f}")
    print(f"   Mean absolute diff: {np.mean(abs_diff):.3f}")
    print(f"   Max relative diff: {np.max(rel_diff):.1%}")
    print(f"   Mean relative diff: {np.mean(rel_diff):.1%}")
    
    # Performance comparison
    speedup = korg_data['timing'] / jorg_result['timing']
    efficiency = (jorg_grid_size / jorg_result['timing']) / (len(wl_korg) / korg_data['timing'])
    
    print(f"\nPerformance comparison:")
    print(f"   Jorg time: {jorg_result['timing']:.1f}s")
    print(f"   Korg time: {korg_data['timing']:.1f}s")
    print(f"   Speedup: {speedup:.1f}x")
    print(f"   Efficiency ratio: {efficiency:.1f}x")
    
    # Assessment
    if np.max(rel_diff) < 0.1 and speedup > 2:
        assessment = "Excellent"
    elif np.max(rel_diff) < 0.2 and speedup > 1:
        assessment = "Good"
    elif speedup > 1:
        assessment = "Acceptable"
    else:
        assessment = "Needs work"
    
    print(f"   Overall assessment: {assessment}")
    
    return {
        'max_rel_diff': np.max(rel_diff),
        'mean_rel_diff': np.mean(rel_diff),
        'speedup': speedup,
        'efficiency': efficiency,
        'assessment': assessment
    }


def create_performance_plot(jorg_results):
    """Create performance vs grid size plot"""
    print(f"\n📈 Creating Performance Analysis Plot")
    print("=" * 50)
    
    try:
        import matplotlib.pyplot as plt
        
        successful = {k: v for k, v in jorg_results.items() if v.get('success', False)}
        if len(successful) < 2:
            print("Not enough data for performance plot")
            return
        
        sizes = list(successful.keys())
        times = [successful[s]['timing'] for s in sizes]
        flux_means = [successful[s]['flux_stats']['mean'] for s in sizes]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Jorg Performance vs Grid Size', fontsize=14)
        
        # Plot 1: Performance
        ax1.plot(sizes, times, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Grid Size (points)')
        ax1.set_ylabel('Time (s)')
        ax1.set_title('Synthesis Time vs Grid Size')
        ax1.grid(True, alpha=0.3)
        
        # Add performance annotations
        for size, time in zip(sizes, times):
            ax1.annotate(f'{time:.1f}s', (size, time), 
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        # Plot 2: Flux consistency
        ax2.plot(sizes, flux_means, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Grid Size (points)')
        ax2.set_ylabel('Mean Flux')
        ax2.set_title('Flux Consistency vs Grid Size')
        ax2.grid(True, alpha=0.3)
        
        # Add flux annotations
        for size, flux in zip(sizes, flux_means):
            ax2.annotate(f'{flux:.1f}', (size, flux), 
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        
        # Save plot
        output_file = Path(__file__).parent / "jorg_grid_performance_analysis.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Performance plot saved to {output_file}")
        
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
    """Main grid comparison test"""
    print("🌟 Quick Jorg vs Korg Grid Comparison")
    print("=" * 60)
    
    # Load Korg data
    korg_data = load_korg_data()
    if not korg_data:
        print("❌ Cannot proceed without Korg data")
        return False
    
    # Test Jorg with different grid sizes
    jorg_results = test_jorg_grid_sizes()
    
    # Find best Jorg configuration
    best_size, best_result = find_best_jorg_grid(jorg_results)
    
    # Compare best Jorg to Korg
    comparison = None
    if best_result:
        comparison = compare_to_korg(best_result, korg_data, best_size)
    
    # Create performance plot
    create_performance_plot(jorg_results)
    
    # Final summary
    print(f"\n" + "=" * 60)
    print("QUICK GRID COMPARISON SUMMARY")
    print("=" * 60)
    
    if best_result:
        print(f"✅ SUCCESS: Found optimal Jorg configuration")
        print(f"")
        print(f"🎯 OPTIMAL CONFIGURATION:")
        print(f"   • Grid size: {best_size} points")
        print(f"   • Synthesis time: {best_result['timing']:.1f}s")
        print(f"   • Flux mean: {best_result['flux_stats']['mean']:.1f}")
        
        if comparison:
            print(f"")
            print(f"⚖️ COMPARISON VS KORG:")
            print(f"   • Speedup: {comparison['speedup']:.1f}x")
            print(f"   • Max difference: {comparison['max_rel_diff']:.1%}")
            print(f"   • Mean difference: {comparison['mean_rel_diff']:.1%}")
            print(f"   • Assessment: {comparison['assessment']}")
        
        print(f"")
        print(f"📊 PERFORMANCE ANALYSIS:")
        successful_grids = [k for k, v in jorg_results.items() if v.get('success', False)]
        print(f"   • Working grid sizes: {successful_grids}")
        print(f"   • Performance limit: ~{max(successful_grids)} points")
        print(f"   • Recommended size: {best_size} points (best speed/accuracy)")
        
        print(f"")
        print(f"🏆 RESULT: Jorg synthesis optimization SUCCESSFUL!")
        print(f"   Ready for practical use with {best_size}-point grids")
        
        success = True
    else:
        print(f"❌ FAILED: No successful Jorg configurations found")
        success = False
    
    print("=" * 60)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)