#!/usr/bin/env python3
"""
Debug H I bound-free implementation in Jorg
"""
import numpy as np
import sys
import os

# Add Jorg to path
sys.path.insert(0, '/Users/jdli/Project/Korg.jl/Jorg')

from jorg.continuum.hydrogen import h_i_bf_absorption
import jax.numpy as jnp

def debug_h_i_bf():
    """Debug H I bound-free calculation"""
    
    print("=== H I BOUND-FREE DEBUG ===")
    
    # Test conditions
    T = 5778.0  # K
    ne = 1e15   # cm^-3
    nH_I = 1e16 # cm^-3
    U_H_I = 2.000000011513405  # Exact from Korg
    
    # Create test frequencies (5000-6000 Å)
    wavelengths = np.linspace(5000, 6000, 51)  # Å
    c_cgs = 2.99792458e10
    frequencies = c_cgs * 1e8 / wavelengths  # Hz
    frequencies = frequencies[::-1]  # Ascending for Jorg
    
    print(f"Test conditions:")
    print(f"  Temperature: {T} K")
    print(f"  Electron density: {ne:.3e} cm^-3") 
    print(f"  H I density: {nH_I:.3e} cm^-3")
    print(f"  U(H I): {U_H_I}")
    print(f"  n(H I)/U(H I): {nH_I/U_H_I:.3e} cm^-3")
    print()
    
    # Debug inputs to H I bound-free function
    nH_I_div_u = nH_I / U_H_I
    nHe_I = 0.0  # No He in this test
    inv_U_H = 1.0 / U_H_I
    
    print(f"Function inputs:")
    print(f"  frequencies: {len(frequencies)} points from {frequencies[0]:.3e} to {frequencies[-1]:.3e} Hz")
    print(f"  temperature: {T}")
    print(f"  nH_I_div_partition: {nH_I_div_u:.3e}")
    print(f"  nHe_I: {nHe_I}")
    print(f"  electron_density: {ne:.3e}")
    print(f"  inv_U_H: {inv_U_H}")
    print()
    
    # Call H I bound-free function
    try:
        alpha_h_i_bf = h_i_bf_absorption(
            frequencies, T, nH_I_div_u, nHe_I, ne, inv_U_H
        )
        
        print(f"H I bound-free results:")
        print(f"  Output shape: {alpha_h_i_bf.shape}")
        print(f"  Output dtype: {alpha_h_i_bf.dtype}")
        print(f"  Range: {np.min(alpha_h_i_bf):.3e} to {np.max(alpha_h_i_bf):.3e} cm^-1")
        print(f"  Mean: {np.mean(alpha_h_i_bf):.3e} cm^-1")
        print(f"  Non-zero values: {np.count_nonzero(alpha_h_i_bf)}/{len(alpha_h_i_bf)}")
        print()
        
        # Check a few specific values
        mid_idx = len(alpha_h_i_bf) // 2
        print(f"Sample values:")
        print(f"  At {wavelengths[-(mid_idx+1)]:.0f} Å: {alpha_h_i_bf[mid_idx]:.3e} cm^-1")
        print(f"  At {wavelengths[-1]:.0f} Å: {alpha_h_i_bf[0]:.3e} cm^-1")
        print(f"  At {wavelengths[0]:.0f} Å: {alpha_h_i_bf[-1]:.3e} cm^-1")
        print()
        
        # Compare to expected order of magnitude
        # For solar conditions, H I bf should be comparable to H^- bf
        print("Expected vs actual:")
        expected_magnitude = 1e-8  # Rough estimate
        print(f"  Expected magnitude: ~{expected_magnitude:.0e} cm^-1")
        print(f"  Actual magnitude: ~{np.mean(alpha_h_i_bf):.0e} cm^-1")
        print(f"  Ratio: {np.mean(alpha_h_i_bf)/expected_magnitude:.1f}")
        
        if np.max(alpha_h_i_bf) == 0:
            print("\\n*** WARNING: All H I bound-free values are zero! ***")
            print("This explains the missing 4% in total opacity.")
            print("Need to investigate H I bound-free implementation.")
            
    except Exception as e:
        print(f"Error calling H I bound-free function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_h_i_bf()