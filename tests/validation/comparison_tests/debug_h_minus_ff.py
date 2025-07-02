#!/usr/bin/env python3
"""
Debug H^- free-free implementation differences between Korg and Jorg
"""
import numpy as np
import json
import sys
import os

# Add Jorg to path
sys.path.insert(0, '/Users/jdli/Project/Korg.jl/Jorg')

from jorg.continuum.hydrogen import h_minus_ff_absorption

def debug_h_minus_ff():
    """Debug H^- free-free calculation"""
    
    print("=== H^- FREE-FREE DEBUG ===")
    
    # Load Korg reference
    with open('/Users/jdli/Project/Korg.jl/korg_detailed_reference.json', 'r') as f:
        ref = json.load(f)
    
    frequency = ref['frequency']
    temperature = ref['temperature']
    electron_density = ref['electron_density']
    U_H_I = ref['partition_functions']['U_H_I']
    
    nH_I = ref['h_i_density']
    nH_I_div_U = nH_I / U_H_I
    
    korg_h_minus_ff = ref['korg_components']['h_minus_ff']
    
    print(f"Test conditions:")
    print(f"  Frequency: {frequency:.3e} Hz")
    print(f"  Temperature: {temperature} K")
    print(f"  Electron density: {electron_density:.3e} cm^-3")
    print(f"  n(H I)/U(H I): {nH_I_div_U:.3e} cm^-3")
    print()
    
    print(f"Korg H^- free-free: {korg_h_minus_ff:.3e} cm^-1")
    
    # Calculate Jorg H^- free-free
    freq_array = np.array([frequency])
    jorg_h_minus_ff = h_minus_ff_absorption(
        freq_array, temperature, nH_I_div_U, electron_density
    )[0]
    
    print(f"Jorg H^- free-free: {jorg_h_minus_ff:.3e} cm^-1")
    print()
    
    # Compare
    ratio = jorg_h_minus_ff / korg_h_minus_ff
    percent_diff = 100 * (jorg_h_minus_ff - korg_h_minus_ff) / korg_h_minus_ff
    
    print(f"Comparison:")
    print(f"  Jorg/Korg ratio: {ratio:.3f}")
    print(f"  Percent difference: {percent_diff:.1f}%")
    print()
    
    if abs(percent_diff) > 10:
        print("*** SIGNIFICANT DIFFERENCE FOUND ***")
        print("This H^- free-free difference could explain the 4% total error!")
        print()
        
        # The issue is likely in the Bell & Berrington table implementation
        print("Probable cause: Simplified H^- free-free implementation in Jorg")
        print("- Korg uses exact Bell & Berrington (1987) interpolation table")
        print("- Jorg uses simplified functional approximation")
        print("- Need to implement accurate Bell & Berrington interpolation")
        
        # Estimate the correction factor needed
        correction_factor = korg_h_minus_ff / jorg_h_minus_ff
        print(f"\\nCorrection factor needed: {correction_factor:.3f}")
        
        return correction_factor
    else:
        print("H^- free-free implementation is accurate")
        return 1.0

if __name__ == "__main__":
    debug_h_minus_ff()