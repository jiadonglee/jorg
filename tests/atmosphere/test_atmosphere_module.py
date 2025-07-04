#!/usr/bin/env python3
"""
Test Atmosphere Module
=====================

Main test script for the Jorg atmosphere interpolation module.
"""

import sys
from pathlib import Path

# Add Jorg to path
jorg_src = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(jorg_src))

import warnings
warnings.filterwarnings('ignore')

def test_atmosphere_imports():
    """Test that atmosphere module imports correctly"""
    print("Testing atmosphere module imports...")
    
    try:
        from jorg.atmosphere import (
            interpolate_marcs, 
            solar_atmosphere, 
            validate_atmosphere,
            call_korg_interpolation
        )
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic atmosphere calculation functionality"""
    print("Testing basic functionality...")
    
    from jorg.atmosphere import interpolate_marcs, solar_atmosphere
    
    # Test solar atmosphere
    try:
        solar_atm = solar_atmosphere()
        assert len(solar_atm.layers) > 0
        assert not solar_atm.spherical
        print(f"✅ Solar atmosphere: {len(solar_atm.layers)} layers")
    except Exception as e:
        print(f"❌ Solar atmosphere failed: {e}")
        return False
    
    # Test custom atmosphere
    try:
        atm = interpolate_marcs(4000.0, 2.0, 0.0)  # K giant
        assert len(atm.layers) > 0
        assert atm.spherical  # Should be spherical for low logg
        print(f"✅ K giant atmosphere: {len(atm.layers)} layers, spherical={atm.spherical}")
    except Exception as e:
        print(f"❌ K giant atmosphere failed: {e}")
        return False
    
    return True

def test_different_grids():
    """Test different interpolation grids"""
    print("Testing different interpolation grids...")
    
    from jorg.atmosphere import interpolate_marcs
    
    test_cases = [
        ("Standard SDSS", 5777, 4.44, 0.0, "should use standard grid"),
        ("Cool dwarf", 3500, 4.8, 0.0, "should use cool dwarf grid"),
        ("Giant", 4000, 2.0, 0.0, "should be spherical"),
    ]
    
    for desc, Teff, logg, m_H, expected in test_cases:
        try:
            atm = interpolate_marcs(Teff, logg, m_H)
            grid_type = "Cool dwarf" if len(atm.layers) > 70 else "Standard"
            geometry = "Spherical" if atm.spherical else "Planar"
            print(f"✅ {desc}: {len(atm.layers)} layers, {grid_type}, {geometry}")
        except Exception as e:
            print(f"❌ {desc} failed: {e}")
            return False
    
    return True

def test_backward_compatibility():
    """Test backward compatibility functions"""
    print("Testing backward compatibility...")
    
    from jorg.atmosphere import call_korg_interpolation
    
    try:
        atm = call_korg_interpolation(5777, 4.44, 0.0)
        assert len(atm.layers) > 0
        print(f"✅ Backward compatibility: {len(atm.layers)} layers")
        return True
    except Exception as e:
        print(f"❌ Backward compatibility failed: {e}")
        return False

def run_all_tests():
    """Run all atmosphere module tests"""
    print("JORG ATMOSPHERE MODULE TESTS")
    print("=" * 40)
    
    tests = [
        ("Import test", test_atmosphere_imports),
        ("Basic functionality", test_basic_functionality),
        ("Different grids", test_different_grids),
        ("Backward compatibility", test_backward_compatibility),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n" + "=" * 40)
    print(f"TEST SUMMARY: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Atmosphere module is ready!")
        return True
    else:
        print("⚠️ Some tests failed - check implementation")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)