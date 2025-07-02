#!/usr/bin/env python3
"""Debug symmetry issue in line profiles"""

import numpy as np
import jax.numpy as jnp
import sys
from pathlib import Path

# Add jorg to path
sys.path.insert(0, str(Path(__file__).parent / '..' / 'src'))

from jorg.lines.profiles import voigt_hjerting, line_profile
from jorg.lines.broadening import doppler_width
from jorg.constants import ATOMIC_MASS_UNIT

def debug_line_profile_symmetry():
    """Debug the symmetry issue"""
    
    print("=== Debugging Line Profile Symmetry ===")
    
    # Simple test parameters
    lambda_0 = 5500e-8  # cm
    temperature = 5778.0  # K
    mass_amu = 55.845  # Fe atomic mass
    xi = 1.5e5  # 1.5 km/s microturbulence
    
    # Calculate Doppler width
    mass_grams = mass_amu * ATOMIC_MASS_UNIT
    sigma = doppler_width(lambda_0, temperature, mass_grams, xi)
    
    # Very small gamma for testing
    gamma = 1e-13
    amplitude = 1.0
    
    print(f"Parameters:")
    print(f"  λ₀ = {lambda_0*1e8:.1f} Å")
    print(f"  σ = {sigma*1e8:.6f} Å")
    print(f"  γ = {gamma*1e8:.6f} Å")
    print(f"  γ/σ = {gamma/sigma:.2e}")
    
    # Create symmetric wavelength grid
    n_points = 201  # Odd number for exact center
    wl_range = 5 * sigma  # ±5σ range
    center_idx = n_points // 2
    
    # Manual symmetric grid construction
    delta_wl = wl_range / (n_points // 2)
    offsets = jnp.arange(-(n_points//2), (n_points//2) + 1) * delta_wl
    wavelengths = lambda_0 + offsets
    
    print(f"\nGrid info:")
    print(f"  Grid points: {n_points}")
    print(f"  Range: ±{wl_range*1e8:.3f} Å")
    print(f"  Spacing: {delta_wl*1e8:.6f} Å")
    print(f"  Center index: {center_idx}")
    
    # Calculate profile
    profile = line_profile(lambda_0, sigma, gamma, amplitude, wavelengths)
    
    print(f"\nProfile diagnostics:")
    print(f"  Center wavelength: {wavelengths[center_idx]*1e8:.6f} Å")
    print(f"  Center value: {profile[center_idx]:.6e}")
    print(f"  Peak index: {jnp.argmax(profile)}")
    print(f"  Peak value: {jnp.max(profile):.6e}")
    
    # Test exact symmetry pairs
    print(f"\nSymmetry test (offset from center):")
    print("Offset (Å)    Left Value       Right Value      Rel Error")
    print("-" * 60)
    
    max_error = 0.0
    for i in range(1, min(11, center_idx + 1)):
        left_idx = center_idx - i
        right_idx = center_idx + i
        
        left_wl = wavelengths[left_idx]
        right_wl = wavelengths[right_idx]
        left_val = profile[left_idx]
        right_val = profile[right_idx]
        
        # Check wavelength symmetry first
        wl_offset_left = abs(left_wl - lambda_0)
        wl_offset_right = abs(right_wl - lambda_0)
        wl_sym_error = abs(wl_offset_left - wl_offset_right) / max(wl_offset_left, wl_offset_right)
        
        # Check profile value symmetry
        if max(left_val, right_val) > 0:
            val_sym_error = abs(left_val - right_val) / max(left_val, right_val)
        else:
            val_sym_error = 0.0
        
        max_error = max(max_error, val_sym_error)
        
        offset_ang = (right_wl - lambda_0) * 1e8
        print(f"{offset_ang:8.3f}     {left_val:12.6e}   {right_val:12.6e}   {val_sym_error:9.2e}")
        
        if wl_sym_error > 1e-14:
            print(f"  WARNING: Wavelength grid not symmetric: {wl_sym_error:.2e}")
    
    print(f"\nMaximum symmetry error: {max_error:.2e}")
    
    # Test direct calculation with exact symmetric points
    print(f"\nDirect calculation test:")
    test_offsets = jnp.array([-2*sigma, -sigma, 0, sigma, 2*sigma])
    test_wavelengths = lambda_0 + test_offsets
    test_profile = line_profile(lambda_0, sigma, gamma, amplitude, test_wavelengths)
    
    print("Offset (σ)     Wavelength (Å)   Profile Value")
    print("-" * 45)
    for i, (offset, wl, val) in enumerate(zip(test_offsets/sigma, test_wavelengths, test_profile)):
        print(f"{offset:8.1f}     {wl*1e8:12.6f}    {val:12.6e}")
    
    # Check symmetry of direct calculation
    left_vals = test_profile[:2][::-1]  # Reverse [-2σ, -σ]
    right_vals = test_profile[3:]       # [+σ, +2σ]
    
    print(f"\nDirect symmetry check:")
    for i, (left, right) in enumerate(zip(left_vals, right_vals)):
        if max(left, right) > 0:
            error = abs(left - right) / max(left, right)
            sigma_offset = (i + 1)
            print(f"  ±{sigma_offset}σ: {error:.2e}")
    
    return max_error

if __name__ == "__main__":
    error = debug_line_profile_symmetry()