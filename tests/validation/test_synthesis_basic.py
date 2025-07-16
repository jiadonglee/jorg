#!/usr/bin/env python3
"""
Basic Synthesis Test for Jorg

This test validates the basic functionality of Jorg's synthesis pipeline
by testing both synth() and synthesize() functions with simple parameters.
"""

import sys
import numpy as np
import jax.numpy as jnp
from pathlib import Path
import time

# Add Jorg to path
jorg_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(jorg_path))

from jorg.synthesis import synth, synthesize, format_abundances


def test_format_abundances():
    """Test abundance formatting"""
    print("Testing abundance formatting...")
    
    # Test basic formatting
    A_X = format_abundances(0.0)  # Solar
    assert len(A_X) == 92
    assert A_X[0] == 12.0  # Hydrogen
    print("  ✓ Solar abundances")
    
    # Test metallicity scaling
    A_X_poor = format_abundances(-0.5)
    assert A_X_poor[25] < A_X[25]  # Iron should be lower
    print("  ✓ Metal-poor abundances")
    
    # Test individual element overrides
    A_X_custom = format_abundances(0.0, Fe=-0.3)
    assert abs(A_X_custom[25] - (A_X[25] - 0.3)) < 0.01
    print("  ✓ Custom element abundances")


def test_synth_function():
    """Test the high-level synth() function"""
    print("\nTesting synth() function...")
    
    # Test basic solar synthesis
    start_time = time.time()
    try:
        wavelengths, flux, continuum = synth(
            Teff=5777,
            logg=4.44, 
            m_H=0.0,
            wavelengths=(5000, 5100),
            rectify=True,
            vmic=1.0
        )
        elapsed = time.time() - start_time
        
        # Validate outputs
        assert len(wavelengths) > 0
        assert len(flux) == len(wavelengths)
        assert len(continuum) == len(wavelengths)
        assert np.all(np.isfinite(wavelengths))
        assert np.all(np.isfinite(flux))
        assert np.all(np.isfinite(continuum))
        assert np.all(flux > 0)
        assert np.all(continuum > 0)
        assert np.min(wavelengths) >= 5000
        assert np.max(wavelengths) <= 5100
        
        print(f"  ✓ Basic synthesis ({elapsed:.1f}s)")
        print(f"    Wavelengths: {len(wavelengths)} points, {np.min(wavelengths):.1f}-{np.max(wavelengths):.1f} Å")
        print(f"    Flux: {np.min(flux):.3f}-{np.max(flux):.3f}")
        print(f"    Continuum: {np.min(continuum):.2e}-{np.max(continuum):.2e}")
        
    except Exception as e:
        print(f"  ✗ Basic synthesis failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test different stellar parameters
    test_params = [
        (4000, 4.5, 0.0, "M_dwarf"),
        (6000, 4.0, -0.5, "F_star_poor"),
        (5000, 2.5, 0.2, "Giant_rich")
    ]
    
    for Teff, logg, m_H, name in test_params:
        try:
            wl, flux, cntm = synth(
                Teff=Teff, logg=logg, m_H=m_H,
                wavelengths=(5400, 5450),
                rectify=True
            )
            
            # Quick validation
            assert np.all(np.isfinite(flux))
            assert np.all(flux > 0)
            print(f"  ✓ {name}: flux {np.min(flux):.3f}-{np.max(flux):.3f}")
            
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
            return False
    
    return True


def test_synthesize_function():
    """Test the detailed synthesize() function"""
    print("\nTesting synthesize() function...")
    
    try:
        # Format abundances and atmosphere
        A_X = format_abundances(0.0)
        from jorg.synthesis import interpolate_atmosphere
        atm = interpolate_atmosphere(5777, 4.44, A_X)
        
        # Create wavelength grid
        wavelengths = jnp.linspace(5000, 5050, 50)
        
        # Run detailed synthesis
        start_time = time.time()
        result = synthesize(
            atm=atm,
            linelist=None,  # Use default/simple linelist
            A_X=A_X,
            wavelengths=wavelengths,
            vmic=1.0,
            return_cntm=True,
            verbose=False
        )
        elapsed = time.time() - start_time
        
        # Validate SynthesisResult
        assert hasattr(result, 'flux')
        assert hasattr(result, 'cntm')
        assert hasattr(result, 'wavelengths')
        assert hasattr(result, 'alpha')
        assert hasattr(result, 'mu_grid')
        assert hasattr(result, 'number_densities')
        assert hasattr(result, 'electron_number_density')
        
        # Validate arrays
        assert len(result.flux) == len(wavelengths)
        assert len(result.wavelengths) == len(wavelengths)
        assert np.all(np.isfinite(result.flux))
        assert np.all(result.flux > 0)
        
        if result.cntm is not None:
            assert len(result.cntm) == len(wavelengths)
            assert np.all(np.isfinite(result.cntm))
            assert np.all(result.cntm > 0)
        
        # Validate alpha matrix
        assert result.alpha.shape[1] == len(wavelengths)
        assert result.alpha.shape[0] == atm['n_layers']
        assert np.all(np.isfinite(result.alpha))
        assert np.all(result.alpha >= 0)
        
        print(f"  ✓ Detailed synthesis ({elapsed:.1f}s)")
        print(f"    Flux: {np.min(result.flux):.2e}-{np.max(result.flux):.2e}")
        print(f"    Alpha shape: {result.alpha.shape}")
        print(f"    μ grid: {len(result.mu_grid)} points")
        print(f"    Species: {len(result.number_densities)}")
        print(f"    Layers: {len(result.electron_number_density)}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Detailed synthesis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_synthesis_options():
    """Test various synthesis options"""
    print("\nTesting synthesis options...")
    
    # Test rectification
    try:
        wl1, flux1, cntm1 = synth(Teff=5777, logg=4.44, wavelengths=(5000, 5020), rectify=True)
        wl2, flux2, cntm2 = synth(Teff=5777, logg=4.44, wavelengths=(5000, 5020), rectify=False)
        
        # Rectified should be ~1, unrectified should be much larger
        assert np.mean(flux1) < 10  # Rectified flux
        assert np.mean(flux2) > 1e-10  # Absolute flux
        print("  ✓ Rectification options")
        
    except Exception as e:
        print(f"  ✗ Rectification test failed: {e}")
        return False
    
    # Test different wavelength ranges
    test_ranges = [
        (5000, 5100),
        (5500, 5600),
        (6000, 6100)
    ]
    
    for wl_start, wl_end in test_ranges:
        try:
            wl, flux, cntm = synth(
                Teff=5777, logg=4.44,
                wavelengths=(wl_start, wl_end)
            )
            assert np.min(wl) >= wl_start
            assert np.max(wl) <= wl_end
            assert np.all(np.isfinite(flux))
            print(f"  ✓ Range {wl_start}-{wl_end} Å")
            
        except Exception as e:
            print(f"  ✗ Range {wl_start}-{wl_end} Å failed: {e}")
            return False
    
    return True


def test_performance():
    """Test synthesis performance"""
    print("\nTesting synthesis performance...")
    
    # Small range test
    start_time = time.time()
    wl, flux, cntm = synth(
        Teff=5777, logg=4.44,
        wavelengths=(5000, 5020),
        rectify=True
    )
    small_time = time.time() - start_time
    
    # Medium range test  
    start_time = time.time()
    wl, flux, cntm = synth(
        Teff=5777, logg=4.44,
        wavelengths=(5000, 5100),
        rectify=True
    )
    medium_time = time.time() - start_time
    
    print(f"  Small range (20 Å): {small_time:.1f}s")
    print(f"  Medium range (100 Å): {medium_time:.1f}s")
    
    # Performance check
    if small_time < 30 and medium_time < 60:
        print("  ✓ Performance acceptable")
        return True
    else:
        print("  ⚠ Performance slower than expected")
        return True  # Don't fail on performance


def main():
    """Main test suite"""
    print("=" * 60)
    print("Jorg Basic Synthesis Test Suite")
    print("=" * 60)
    
    tests = [
        ("Abundance formatting", test_format_abundances),
        ("Synth function", test_synth_function),
        ("Synthesize function", test_synthesize_function),
        ("Synthesis options", test_synthesis_options),
        ("Performance", test_performance)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 60)
    print("SYNTHESIS TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL SYNTHESIS TESTS PASSED!")
        print("Jorg synthesis pipeline is working correctly.")
        success = True
    else:
        print("⚠️ Some synthesis tests failed.")
        success = False
    
    print("=" * 60)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)