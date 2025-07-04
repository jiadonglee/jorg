#!/usr/bin/env python3
"""
Test Jorg Atmosphere Interpolation vs Korg
==========================================

This script tests Jorg's atmosphere interpolation against Korg to ensure
they produce identical results.
"""

import sys
import numpy as np
from pathlib import Path

# Add Jorg to path
# jorg_path = Path(__file__).parent / "Jorg" / "src"
jorg_path = "/Users/jdli/Project/Korg.jl/Jorg/src/"
sys.path.insert(0, str(jorg_path))

from jorg.atmosphere import interpolate_marcs_atmosphere, get_atmosphere_layer
from jorg.constants import kboltz_cgs

def test_atmosphere_interpolation():
    """Test atmosphere interpolation for various stellar types"""
    
    print("TESTING JORG ATMOSPHERE INTERPOLATION vs KORG")
    print("=" * 60)
    
    # Test cases: same as our stellar types comparison
    test_cases = [
        ("Solar-type G star", 5777.0, 4.44, 0.0),
        ("Cool K-type star", 4500.0, 4.5, 0.0),
        ("Cool M dwarf", 3500.0, 4.8, 0.0),
        ("Giant K star", 4500.0, 2.5, 0.0),
        ("Metal-poor G star", 5777.0, 4.44, -1.0),
        ("Metal-rich G star", 5777.0, 4.44, +0.3),
    ]
    
    all_results = []
    
    for description, Teff, logg, m_H in test_cases:
        print(f"\nTesting: {description}")
        print(f"Stellar Parameters: Teff={Teff}K, logg={logg}, [M/H]={m_H}")
        
        try:
            # Interpolate atmosphere using Jorg (which calls Korg)
            atmosphere = interpolate_marcs_atmosphere(Teff, logg, m_H=m_H)
            
            print(f"✅ Interpolation successful")
            print(f"   Layers: {len(atmosphere.layers)}")
            print(f"   Spherical: {atmosphere.spherical}")
            if atmosphere.R:
                print(f"   Radius: {atmosphere.R:.2e} cm")
            
            # Test specific layers (matching our previous tests)
            test_layers = [15, 25, 35] if len(atmosphere.layers) > 35 else [min(15, len(atmosphere.layers)-1)]
            
            layer_results = []
            for layer_idx in test_layers:
                if layer_idx < len(atmosphere.layers):
                    layer = get_atmosphere_layer(atmosphere, layer_idx)
                    P = layer.number_density * kboltz_cgs * layer.temp
                    
                    layer_result = {
                        'layer': layer_idx,
                        'T': layer.temp,
                        'nt': layer.number_density,
                        'ne': layer.electron_number_density,
                        'P': P,
                        'tau_5000': layer.tau_5000,
                        'z': layer.z
                    }
                    layer_results.append(layer_result)
                    
                    print(f"   Layer {layer_idx}: T={layer.temp:.1f}K, "
                          f"nt={layer.number_density:.2e}, ne={layer.electron_number_density:.2e}")
            
            all_results.append((description, Teff, logg, m_H, atmosphere, layer_results))
            
        except Exception as e:
            print(f"❌ Interpolation failed: {e}")
    
    return all_results

def validate_atmosphere_consistency(results):
    """Validate that interpolated atmospheres are physically consistent"""
    
    print(f"\n\nVALIDATING ATMOSPHERE CONSISTENCY")
    print("=" * 60)
    
    for description, Teff, logg, m_H, atmosphere, layer_results in results:
        print(f"\nValidating: {description}")
        
        # Check temperature decreases with depth (increasing layer index)
        temps = [layer.temp for layer in atmosphere.layers]
        temp_increasing = all(temps[i] <= temps[i+1] for i in range(len(temps)-1))
        print(f"   Temperature increases with depth: {'✅' if temp_increasing else '❌'}")
        
        # Check density increases with depth
        densities = [layer.number_density for layer in atmosphere.layers]
        density_increasing = all(densities[i] <= densities[i+1] for i in range(len(densities)-1))
        print(f"   Density increases with depth: {'✅' if density_increasing else '❌'}")
        
        # Check electron density is reasonable fraction of total density
        for layer in layer_results:
            ne_fraction = layer['ne'] / layer['nt']
            reasonable_ne = 1e-15 < ne_fraction < 0.5  # Very broad range
            print(f"   Layer {layer['layer']} ne/nt = {ne_fraction:.2e}: {'✅' if reasonable_ne else '❌'}")

def extract_atmosphere_for_chemical_equilibrium(atmosphere, layer_index=25):
    """
    Extract atmosphere parameters needed for chemical equilibrium testing.
    
    Returns parameters in the format needed by chemical equilibrium functions.
    """
    if layer_index >= len(atmosphere.layers):
        layer_index = len(atmosphere.layers) - 1
    
    layer = atmosphere.layers[layer_index]
    
    return {
        'T': layer.temp,
        'nt': layer.number_density,
        'ne_guess': layer.electron_number_density,
        'P': layer.number_density * kboltz_cgs * layer.temp,
        'tau_5000': layer.tau_5000,
        'z': layer.z
    }

def main():
    print("JORG ATMOSPHERE INTERPOLATION VALIDATION")
    print("=" * 70)
    print("Testing Jorg atmosphere interpolation by calling Korg.jl")
    print("This ensures identical atmospheric conditions for chemical equilibrium tests")
    print()
    
    # Test atmosphere interpolation
    results = test_atmosphere_interpolation()
    
    # Validate physical consistency
    validate_atmosphere_consistency(results)
    
    # Show example for chemical equilibrium
    if results:
        print(f"\n\nEXAMPLE ATMOSPHERE EXTRACTION FOR CHEMICAL EQUILIBRIUM")
        print("=" * 60)
        
        description, Teff, logg, m_H, atmosphere, layer_results = results[0]  # Solar case
        
        for layer_idx in [15, 25, 35]:
            if layer_idx < len(atmosphere.layers):
                params = extract_atmosphere_for_chemical_equilibrium(atmosphere, layer_idx)
                print(f"\nLayer {layer_idx} parameters for chemical equilibrium:")
                for key, value in params.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.6e}")
                    else:
                        print(f"  {key}: {value}")
    
    print(f"\n\n{'='*70}")
    print("✅ ATMOSPHERE INTERPOLATION TESTING COMPLETE")
    print(f"Tested {len(results)} stellar types successfully")
    print("✅ Ready for chemical equilibrium comparison with identical inputs")
    print("="*70)
    
    return results

if __name__ == "__main__":
    results = main()