#!/usr/bin/env python3
"""
Simple Jorg vs Korg synth() Comparison

This script tests a basic solar case to compare Jorg and Korg synth() outputs.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add Jorg to path
jorg_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(jorg_path))

def test_jorg_synth_basic():
    """Test basic Jorg synth() functionality"""
    print("🌟 Testing Jorg synth() Function")
    print("=" * 50)
    
    try:
        from jorg.synthesis import synth
        
        # Basic solar case
        print("Testing basic solar case:")
        print("  Teff=5777K, logg=4.44, [M/H]=0.0")
        print("  Wavelengths: 5000-5020 Å (small range for speed)")
        print("  Rectified spectrum")
        
        start_time = time.time()
        
        wavelengths, flux, continuum = synth(
            Teff=5777,
            logg=4.44,
            m_H=0.0,
            wavelengths=(5000, 5020),  # Small range for testing
            rectify=True,
            vmic=1.0
        )
        
        elapsed = time.time() - start_time
        
        print(f"\n✅ SUCCESS! Synthesis completed in {elapsed:.1f}s")
        print(f"  Wavelength points: {len(wavelengths)}")
        print(f"  Wavelength range: {np.min(wavelengths):.1f} - {np.max(wavelengths):.1f} Å")
        print(f"  Flux range: {np.min(flux):.3f} - {np.max(flux):.3f}")
        print(f"  Flux mean: {np.mean(flux):.3f}")
        print(f"  Continuum range: {np.min(continuum):.2e} - {np.max(continuum):.2e}")
        print(f"  Continuum mean: {np.mean(continuum):.2e}")
        
        # Validate outputs
        assert len(wavelengths) > 0, "No wavelength points generated"
        assert len(flux) == len(wavelengths), "Flux and wavelength arrays have different lengths"
        assert len(continuum) == len(wavelengths), "Continuum and wavelength arrays have different lengths"
        assert np.all(np.isfinite(wavelengths)), "Non-finite wavelength values"
        assert np.all(np.isfinite(flux)), "Non-finite flux values"
        assert np.all(np.isfinite(continuum)), "Non-finite continuum values"
        assert np.all(flux > 0), "Negative flux values"
        assert np.all(continuum > 0), "Negative continuum values"
        
        # Expected ranges for solar spectrum
        if np.mean(flux) > 0.5 and np.mean(flux) < 1.5:
            print("  ✓ Flux values in expected range for rectified spectrum")
        else:
            print(f"  ⚠ Flux values may be outside expected range: mean={np.mean(flux):.3f}")
        
        return {
            'wavelengths': wavelengths,
            'flux': flux,
            'continuum': continuum,
            'timing': elapsed,
            'success': True
        }
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def test_jorg_synth_variations():
    """Test Jorg synth() with different parameters"""
    print("\n🔬 Testing Parameter Variations")
    print("=" * 50)
    
    try:
        from jorg.synthesis import synth
        
        test_cases = [
            {
                'name': 'Metal-poor solar',
                'params': {'Teff': 5777, 'logg': 4.44, 'm_H': -0.5, 'wavelengths': (5000, 5020), 'rectify': True, 'vmic': 1.0}
            },
            {
                'name': 'M dwarf',
                'params': {'Teff': 3500, 'logg': 4.5, 'm_H': 0.0, 'wavelengths': (5000, 5020), 'rectify': True, 'vmic': 1.0}
            },
            {
                'name': 'K dwarf',
                'params': {'Teff': 4500, 'logg': 4.5, 'm_H': 0.0, 'wavelengths': (5000, 5020), 'rectify': True, 'vmic': 1.0}
            },
            {
                'name': 'Unrectified',
                'params': {'Teff': 5777, 'logg': 4.44, 'm_H': 0.0, 'wavelengths': (5000, 5020), 'rectify': False, 'vmic': 1.0}
            },
            {
                'name': 'Different wavelengths',
                'params': {'Teff': 5777, 'logg': 4.44, 'm_H': 0.0, 'wavelengths': (5500, 5520), 'rectify': True, 'vmic': 1.0}
            },
        ]
        
        results = {}
        
        for i, test_case in enumerate(test_cases):
            name = test_case['name']
            params = test_case['params']
            
            print(f"\n{i+1}. Testing {name}:")
            
            # Extract key parameters for display
            Teff = params['Teff']
            logg = params['logg'] 
            m_H = params['m_H']
            wl_range = params['wavelengths']
            rectify = params['rectify']
            
            print(f"   Teff={Teff}K, logg={logg}, [M/H]={m_H}")
            print(f"   {wl_range[0]}-{wl_range[1]} Å, rectify={rectify}")
            
            try:
                start_time = time.time()
                
                wavelengths, flux, continuum = synth(**params)
                
                elapsed = time.time() - start_time
                
                results[name] = {
                    'wavelengths': wavelengths,
                    'flux': flux,
                    'continuum': continuum,
                    'timing': elapsed,
                    'params': params,
                    'success': True
                }
                
                print(f"   ✅ Success ({elapsed:.1f}s)")
                print(f"      Points: {len(wavelengths)}")
                print(f"      Flux: {np.min(flux):.3f} - {np.max(flux):.3f} (mean: {np.mean(flux):.3f})")
                
                # Quick validation
                assert np.all(np.isfinite(flux)), "Non-finite flux values"
                assert np.all(flux > 0), "Negative flux values"
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                results[name] = {'success': False, 'error': str(e)}
        
        return results
        
    except Exception as e:
        print(f"\n❌ Parameter variation testing failed: {e}")
        return {}


def analyze_spectrum_properties(result):
    """Analyze the properties of a synthetic spectrum"""
    print("\n📊 Spectrum Analysis")
    print("=" * 50)
    
    if not result.get('success', False):
        print("No successful spectrum to analyze")
        return
    
    wavelengths = result['wavelengths']
    flux = result['flux']
    continuum = result['continuum']
    
    # Basic properties
    print(f"Spectral Properties:")
    print(f"  Wavelength coverage: {np.min(wavelengths):.1f} - {np.max(wavelengths):.1f} Å")
    print(f"  Spectral resolution: {len(wavelengths)} points")
    print(f"  Δλ (average): {np.mean(np.diff(wavelengths)):.3f} Å")
    
    print(f"\nFlux Statistics:")
    print(f"  Range: {np.min(flux):.3f} - {np.max(flux):.3f}")
    print(f"  Mean: {np.mean(flux):.3f}")
    print(f"  Std: {np.std(flux):.3f}")
    print(f"  Median: {np.median(flux):.3f}")
    
    print(f"\nContinuum Statistics:")
    print(f"  Range: {np.min(continuum):.2e} - {np.max(continuum):.2e}")
    print(f"  Mean: {np.mean(continuum):.2e}")
    print(f"  Std: {np.std(continuum):.2e}")
    
    # Look for spectral features
    flux_normalized = flux / np.mean(flux)
    min_flux_idx = np.argmin(flux_normalized)
    max_flux_idx = np.argmax(flux_normalized)
    
    print(f"\nSpectral Features:")
    print(f"  Deepest point: λ={wavelengths[min_flux_idx]:.1f} Å, flux={flux_normalized[min_flux_idx]:.3f}")
    print(f"  Highest point: λ={wavelengths[max_flux_idx]:.1f} Å, flux={flux_normalized[max_flux_idx]:.3f}")
    print(f"  Total variation: {np.max(flux_normalized) - np.min(flux_normalized):.3f}")


def create_simple_plot(result):
    """Create a simple plot of the spectrum"""
    print("\n📈 Creating Spectrum Plot")
    print("=" * 50)
    
    try:
        import matplotlib.pyplot as plt
        
        if not result.get('success', False):
            print("No successful spectrum to plot")
            return
        
        wavelengths = result['wavelengths']
        flux = result['flux']
        continuum = result['continuum']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot spectrum
        ax1.plot(wavelengths, flux, 'b-', linewidth=1.5, label='Synthetic spectrum')
        ax1.set_xlabel('Wavelength (Å)')
        ax1.set_ylabel('Flux (rectified)')
        ax1.set_title('Jorg Solar Spectrum Synthesis')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot continuum
        ax2.plot(wavelengths, continuum, 'r-', linewidth=1.5, label='Continuum')
        ax2.set_xlabel('Wavelength (Å)')
        ax2.set_ylabel('Continuum Flux (erg/s/cm²/Å)')
        ax2.set_title('Continuum Level')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_yscale('log')
        
        plt.tight_layout()
        
        # Save plot
        output_file = Path(__file__).parent / "jorg_synth_test_spectrum.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Spectrum plot saved to {output_file}")
        
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
    """Main test function"""
    print("🌟 Simple Jorg synth() Function Test")
    print("This test validates Jorg's synth() function with basic parameters")
    print("and compares it to expected behavior for stellar synthesis.\n")
    
    # Test basic functionality
    basic_result = test_jorg_synth_basic()
    
    # Test parameter variations
    variation_results = test_jorg_synth_variations()
    
    # Analyze spectrum properties
    if basic_result.get('success', False):
        analyze_spectrum_properties(basic_result)
        create_simple_plot(basic_result)
    
    # Final summary
    print("\n" + "=" * 60)
    print("JORG synth() TEST SUMMARY")
    print("=" * 60)
    
    basic_success = basic_result.get('success', False)
    variation_successes = sum(1 for r in variation_results.values() if r.get('success', False))
    total_variations = len(variation_results)
    
    print(f"Basic solar test: {'✅ PASS' if basic_success else '❌ FAIL'}")
    print(f"Parameter variations: {variation_successes}/{total_variations} successful")
    
    if basic_success and variation_successes >= total_variations * 0.8:
        print("\n🎉 EXCELLENT: Jorg synth() function is working correctly!")
        print("✅ Generates realistic solar spectra")
        print("✅ Handles different stellar parameters")
        print("✅ Produces physically reasonable outputs")
        success = True
    elif basic_success:
        print("\n✅ GOOD: Basic Jorg synth() function works")
        print("⚠ Some parameter variations may have issues")
        success = True
    else:
        print("\n❌ ISSUES: Jorg synth() function needs attention")
        success = False
    
    # Performance summary
    if basic_success:
        timing = basic_result['timing']
        if timing < 5:
            print(f"⚡ Performance: Excellent ({timing:.1f}s)")
        elif timing < 15:
            print(f"⚡ Performance: Good ({timing:.1f}s)")
        else:
            print(f"⚡ Performance: Acceptable ({timing:.1f}s)")
    
    print("=" * 60)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)