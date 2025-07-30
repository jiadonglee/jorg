#!/usr/bin/env python3
"""
Quick Line Opacity Test
=======================

Minimal test to verify line opacity functionality.
"""

import sys
import os
sys.path.insert(0, "/Users/jdli/Project/Korg.jl/Jorg/src")

import jax.numpy as jnp

def quick_test():
    """Quick line opacity test"""
    
    print("🚀 QUICK LINE OPACITY TEST")
    print("=" * 28)
    
    from jorg.lines.opacity import calculate_line_opacity_korg_method
    
    # Test the optimized Fe I line at 5001.52 Å
    fe_params = {
        'wavelengths': jnp.array([5001.52]),
        'line_wavelength': 5001.6058,
        'excitation_potential': 4.6382,
        'log_gf': -1.045,
        'temperature': 5014.7,
        'electron_density': 1.94e12,
        'hydrogen_density': 1.31e17,
        'abundance': 3.236e-8,
        'atomic_mass': 55.845,
        'gamma_rad': 8.30e7,
        'gamma_stark': 0.0,
        'vald_vdw_param': None,
        'microturbulence': 2.0,
        'partition_function': 27.844
    }
    
    # Test optimized parameter
    opacity_optimized = calculate_line_opacity_korg_method(
        log_gamma_vdw=None,  # Use species-specific -7.820
        species_name='Fe I',
        **fe_params
    )
    
    optimized_val = float(opacity_optimized[0])
    korg_target = 1.284e-11
    error = abs(optimized_val - korg_target) / korg_target * 100
    
    print(f"Fe I wing opacity test at 5001.52 Å:")
    print(f"  Korg target:    {korg_target:.6e} cm⁻¹")
    print(f"  Jorg optimized: {optimized_val:.6e} cm⁻¹")
    print(f"  Error:          {error:.3f}%")
    
    if error < 1.0:
        status = "✅ EXCELLENT"
    elif error < 5.0:
        status = "✅ GOOD"
    else:
        status = "❌ POOR"
    
    print(f"  Status: {status}")
    
    # Test multiple wavelengths
    test_wavelengths = jnp.array([5001.0, 5001.52, 5002.0])
    
    fe_params_multi = {
        **fe_params,
        'wavelengths': test_wavelengths
    }
    
    opacity_multi = calculate_line_opacity_korg_method(
        log_gamma_vdw=None,
        species_name='Fe I',
        **fe_params_multi
    )
    
    print(f"\nMulti-wavelength test:")
    print(f"{'Wavelength':>12s} {'Opacity':>15s}")
    print("-" * 30)
    
    for i, wl in enumerate(test_wavelengths):
        opacity_val = float(opacity_multi[i])
        print(f"{float(wl):12.2f} {opacity_val:15.6e}")
    
    return error, status

if __name__ == "__main__":
    error, status = quick_test()
    
    print(f"\n📊 QUICK TEST SUMMARY")
    print("=" * 19)
    print(f"Optimized Fe I vdW parameter: -7.820")
    print(f"5001.52 Å accuracy: {error:.3f}% ({status})")
    print(f"")
    print(f"Line opacity system: {'✅ WORKING' if error < 5 else '❌ ISSUES'}")
    print(f"Species optimization: {'✅ ACTIVE' if error < 1 else '⚠️ CHECK'}")