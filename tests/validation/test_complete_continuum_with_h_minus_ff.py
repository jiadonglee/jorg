#!/usr/bin/env python3
"""
Test complete continuum with proper H⁻ free-free implementation
"""

import sys
from pathlib import Path
jorg_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(jorg_src))

import numpy as np
import jax.numpy as jnp
from jorg.continuum.complete_continuum import (
    h_minus_ff_absorption_coefficient, total_continuum_absorption_jorg
)

print("COMPLETE CONTINUUM WITH H⁻ FREE-FREE TEST")
print("=" * 42)

# Test conditions
T = 4838.3  # K
ne = 2.28e12  # cm⁻³
n_HI = 2.5e16  # cm⁻³
n_HII = 6.0e10  # cm⁻³
n_HeI = 2.0e15  # cm⁻³
n_HeII = 1.0e11  # cm⁻³
n_FeI = 9.0e10  # cm⁻³
n_FeII = 3.0e10  # cm⁻³
n_H2 = 1.0e13  # cm⁻³

frequency = 5.451e14  # Hz (5500 Å)
korg_reference = 3.5e-9  # cm⁻¹

print(f"Test frequency: {frequency:.2e} Hz (5500 Å)")
print(f"Temperature: {T} K")
print(f"Korg reference: {korg_reference:.2e} cm⁻¹")
print()

# Test H⁻ free-free function directly
h_i_ground = n_HI  # Approximate ground state density
alpha_h_minus_ff = h_minus_ff_absorption_coefficient(frequency, T, h_i_ground, ne)

print(f"DIRECT H⁻ FREE-FREE TEST:")
print(f"  H I ground density: {h_i_ground:.2e} cm⁻³")
print(f"  H⁻ ff coefficient: {float(alpha_h_minus_ff):.2e} cm⁻¹")
print(f"  Fraction of Korg: {float(alpha_h_minus_ff) / korg_reference:.1f}")
print()

# Test complete opacity
alpha_total = total_continuum_absorption_jorg(
    jnp.array([frequency]), T, ne, n_HI, n_HII, n_HeI, n_HeII, n_FeI, n_FeII, n_H2
)[0]

print(f"COMPLETE CONTINUUM OPACITY:")
print(f"  Jorg total: {float(alpha_total):.2e} cm⁻¹")
print(f"  Korg reference: {korg_reference:.2e} cm⁻¹")
print(f"  Jorg/Korg ratio: {float(alpha_total) / korg_reference:.2f}")
print()

# Compare with and without H⁻ ff
# Test old version by temporarily modifying code would be complex, so just compare magnitudes
if abs(float(alpha_total) / korg_reference - 1.0) < 0.3:
    print("🎉 EXCELLENT: Within 30% of Korg! H⁻ ff implementation successful!")
elif abs(float(alpha_total) / korg_reference - 1.0) < 0.5:
    print("✅ VERY GOOD: Within 50% of Korg!")
elif abs(float(alpha_total) / korg_reference - 1.0) < 1.0:
    print("✅ GOOD: Within factor of 2 of Korg!")
else:
    print("⚠️  Need more work: Still significant difference")
    
# Test wavelength dependence  
wavelengths = [4000, 5500, 7000]  # Å
frequencies = [2.998e18 / wl for wl in wavelengths]

print()
print("WAVELENGTH DEPENDENCE:")
for wl, freq in zip(wavelengths, frequencies):
    alpha = total_continuum_absorption_jorg(
        jnp.array([freq]), T, ne, n_HI, n_HII, n_HeI, n_HeII, n_FeI, n_FeII, n_H2
    )[0]
    print(f"  {wl:4.0f} Å: {float(alpha):.2e} cm⁻¹")

# Check blue vs red
alpha_blue = total_continuum_absorption_jorg(
    jnp.array([frequencies[0]]), T, ne, n_HI, n_HII, n_HeI, n_HeII, n_FeI, n_FeII, n_H2
)[0]
alpha_red = total_continuum_absorption_jorg(
    jnp.array([frequencies[2]]), T, ne, n_HI, n_HII, n_HeI, n_HeII, n_FeI, n_FeII, n_H2
)[0]

blue_red_ratio = float(alpha_blue) / float(alpha_red)
print()
print(f"Blue/Red ratio: {blue_red_ratio:.2f}")
if blue_red_ratio > 1.0:
    print("✅ Correct wavelength dependence (blue > red)")
else:
    print("❌ Incorrect wavelength dependence")

print()
print("CONCLUSION:")
if float(alpha_total) > korg_reference * 0.7 and float(alpha_total) < korg_reference * 1.5:
    print("🎯 SUCCESS: H⁻ free-free implementation brings Jorg very close to Korg!")
    print("   The remaining small difference is likely from approximations")
    print("   or missing minor continuum sources")
else:
    print("🔧 PROGRESS: H⁻ free-free helps but may need fine-tuning")
    print("   Check Bell & Berrington coefficient and wavelength scaling")