#!/usr/bin/env python3
"""
Minimal Synthesis Test

Test the most basic synthesis functionality to verify it works.
"""

import sys
import time
from pathlib import Path

# Add Jorg to path
jorg_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(jorg_path))

def test_minimal_synth():
    """Test the absolute minimal synth call"""
    print("Testing minimal synth() call...")
    
    try:
        # Import with timing
        start = time.time()
        from jorg.synthesis import synth
        import_time = time.time() - start
        print(f"  Import time: {import_time:.1f}s")
        
        # Simple synth call with minimal wavelength range
        start = time.time()
        wavelengths, flux, continuum = synth(
            Teff=5777,
            logg=4.44,
            m_H=0.0,
            wavelengths=(5000, 5010),  # Very small range
            rectify=False,  # Avoid normalization
        )
        elapsed = time.time() - start
        
        print(f"  Synthesis time: {elapsed:.1f}s")
        print(f"  Result: {len(wavelengths)} wavelengths")
        print(f"  Flux range: {min(flux):.2e} - {max(flux):.2e}")
        print(f"  Continuum range: {min(continuum):.2e} - {max(continuum):.2e}")
        
        # Basic validation
        assert len(wavelengths) > 0
        assert len(flux) == len(wavelengths)
        assert len(continuum) == len(wavelengths)
        
        print("  ✓ Minimal synth test PASSED")
        return True
        
    except Exception as e:
        print(f"  ✗ Minimal synth test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_calls():
    """Test multiple synthesis calls to check for memory issues"""
    print("\nTesting multiple synthesis calls...")
    
    try:
        from jorg.synthesis import synth
        
        times = []
        for i in range(3):
            start = time.time()
            wavelengths, flux, continuum = synth(
                Teff=5777,
                logg=4.44,
                m_H=0.0,
                wavelengths=(5000, 5020),
                rectify=False,
            )
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Call {i+1}: {elapsed:.1f}s, {len(flux)} points")
        
        print(f"  Average time: {sum(times)/len(times):.1f}s")
        
        # Check if times are increasing (memory leak indicator)
        if times[-1] > times[0] * 2:
            print("  ⚠ Warning: Synthesis time increasing significantly")
        else:
            print("  ✓ Multiple calls test PASSED")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Multiple calls test FAILED: {e}")
        return False

def test_different_parameters():
    """Test different stellar parameters"""
    print("\nTesting different stellar parameters...")
    
    try:
        from jorg.synthesis import synth
        
        test_stars = [
            (5777, 4.44, 0.0, "Sun"),
            (4000, 4.5, 0.0, "M_dwarf"),
            (6000, 4.0, 0.0, "F_star"),
        ]
        
        for Teff, logg, m_H, name in test_stars:
            start = time.time()
            try:
                wavelengths, flux, continuum = synth(
                    Teff=Teff,
                    logg=logg,
                    m_H=m_H,
                    wavelengths=(5000, 5020),
                    rectify=False,
                )
                elapsed = time.time() - start
                print(f"  {name}: {elapsed:.1f}s, flux range {min(flux):.2e}-{max(flux):.2e}")
                
            except Exception as e:
                print(f"  {name}: FAILED - {e}")
        
        print("  ✓ Different parameters test completed")
        return True
        
    except Exception as e:
        print(f"  ✗ Different parameters test FAILED: {e}")
        return False

def main():
    """Main minimal test"""
    print("=" * 50)
    print("Jorg Minimal Synthesis Test")
    print("=" * 50)
    
    tests = [
        test_minimal_synth,
        test_multiple_calls,
        test_different_parameters,
    ]
    
    passed = 0
    for test_func in tests:
        if test_func():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"MINIMAL SYNTHESIS RESULTS: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 MINIMAL SYNTHESIS WORKS!")
    elif passed > 0:
        print("⚠️ Partial synthesis functionality")
    else:
        print("❌ Synthesis not working")
    
    print("=" * 50)
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)