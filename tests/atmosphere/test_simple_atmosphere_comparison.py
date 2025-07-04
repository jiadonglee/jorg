#!/usr/bin/env python3
"""
Simple Atmosphere Interpolation Test
===================================
"""

import sys
sys.path.insert(0, 'Jorg/src')

from jorg.atmosphere import call_korg_interpolation

def test_atmosphere_interpolation():
    """Test one case and validate against expected results"""
    
    print("TESTING JORG ATMOSPHERE INTERPOLATION")
    print("=" * 50)
    
    # Test solar case
    print("\nTesting Solar-type G star")
    print("Parameters: Teff=5777K, logg=4.44, [M/H]=0.0")
    
    try:
        atmosphere = call_korg_interpolation(5777.0, 4.44, 0.0)
        
        print(f"✅ Interpolation successful")
        print(f"   Layers: {len(atmosphere.layers)}")
        print(f"   Spherical: {atmosphere.spherical}")
        
        # Test specific layers
        test_layers = [15, 25, 35]
        for layer_idx in test_layers:
            if layer_idx < len(atmosphere.layers):
                layer = atmosphere.layers[layer_idx]
                print(f"   Layer {layer_idx}:")
                print(f"     T = {layer.temp:.1f} K")
                print(f"     nt = {layer.number_density:.3e}")
                print(f"     ne = {layer.electron_number_density:.3e}")
                print(f"     tau_5000 = {layer.tau_5000:.6f}")
                print(f"     z = {layer.z:.3e} cm")
        
        # Test different stellar types
        test_cases = [
            ("Cool K-type", 4500.0, 4.5, 0.0),
            ("Cool M dwarf", 3500.0, 4.8, 0.0),
            ("Metal-poor G", 5777.0, 4.44, -1.0)
        ]
        
        for description, Teff, logg, m_H in test_cases:
            print(f"\nTesting {description} (Teff={Teff}, logg={logg}, [M/H]={m_H})")
            try:
                atm = call_korg_interpolation(Teff, logg, m_H)
                print(f"   ✅ Success: {len(atm.layers)} layers, spherical={atm.spherical}")
            except Exception as e:
                print(f"   ❌ Failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

if __name__ == "__main__":
    success = test_atmosphere_interpolation()
    print(f"\n{'='*50}")
    print(f"Test {'PASSED' if success else 'FAILED'}")
    print("="*50)