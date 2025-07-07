#!/usr/bin/env python3
"""
Debug Bell & Berrington scaling to understand the correct implementation
"""

import sys
from pathlib import Path
jorg_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(jorg_src))

import numpy as np
import jax.numpy as jnp
from jorg.continuum.complete_continuum import h_minus_ff_absorption_coefficient

print("BELL & BERRINGTON SCALING DEBUG")
print("=" * 35)

# Test conditions
T = 4838.3  # K
ne = 2.28e12  # cm⁻³
n_HI = 2.5e16  # cm⁻³
frequency = 5.451e14  # Hz (5500 Å)

# Calculate θ for our temperature
theta = 5040.0 / T
wavelength_angstrom = 2.998e18 / frequency

print(f"Temperature: {T} K")
print(f"θ = 5040/T: {theta:.2f}")
print(f"Wavelength: {wavelength_angstrom:.0f} Å")
print()

# Bell & Berrington table reference points
print("BELL & BERRINGTON TABLE REFERENCE:")
print("θ = 1.0 (T = 5040 K):")
print("  λ = 3038 Å: K = 0.0789")
print("  λ = 5063 Å: K = 0.132") 
print("  λ = 7595 Å: K = 0.243")
print()

# What our temperature corresponds to in the table
print(f"Our temperature θ = {theta:.2f} vs table θ = 1.0")
print(f"Temperature ratio: {theta:.2f}")
print()

# Manual calculation using Korg's approach
print("MANUAL CALCULATION (following Korg exactly):")

# K value interpolation (simplified)
if wavelength_angstrom <= 5063.0:
    k_base = 0.0789 + (0.132 - 0.0789) * (wavelength_angstrom - 3038.0) / (5063.0 - 3038.0)
else:
    k_base = 0.132 + (0.243 - 0.132) * (wavelength_angstrom - 5063.0) / (7595.0 - 5063.0)

print(f"Base K value (θ=1.0): {k_base:.4f}")

# Temperature scaling
theta_factor = (theta / 1.0)**1.5
print(f"Temperature factor θ^1.5: {theta_factor:.4f}")

# Final K with proper scaling
K = k_base * 1e-26 * theta_factor
print(f"Final K: {K:.2e} cm⁴/dyn")

# Electron pressure
P_e = ne * 1.381e-16 * T  # dyn/cm²
print(f"Electron pressure: {P_e:.2e} dyn/cm²")

# Ground state H I density (from Korg: 2 * nH_I_div_partition)
# For partition function ≈ 2: nH_I_div_partition ≈ nH_I / 2
n_h_i_ground = 2.0 * (n_HI / 2.0)  # ≈ n_HI
print(f"H I ground density: {n_h_i_ground:.2e} cm⁻³")

# Final absorption coefficient
alpha_manual = K * P_e * n_h_i_ground
print(f"Manual α_ff(H⁻): {alpha_manual:.2e} cm⁻¹")
print()

# Compare with our function
alpha_function = h_minus_ff_absorption_coefficient(frequency, T, n_h_i_ground, ne)
print(f"Function α_ff(H⁻): {float(alpha_function):.2e} cm⁻¹")
print(f"Manual/Function ratio: {alpha_manual / float(alpha_function):.1f}")
print()

# Check what we need to match Korg
korg_total = 3.5e-9  # cm⁻¹
current_other_sources = 2.7e-10  # Our current non-ff sources 
needed_h_minus_ff = korg_total - current_other_sources
print(f"KORG COMPARISON:")
print(f"Korg total: {korg_total:.2e} cm⁻¹")
print(f"Our other sources: {current_other_sources:.2e} cm⁻¹")
print(f"Needed H⁻ ff: {needed_h_minus_ff:.2e} cm⁻¹")
print(f"Current H⁻ ff: {alpha_manual:.2e} cm⁻¹")
print(f"Scaling factor needed: {needed_h_minus_ff / alpha_manual:.1f}")
print()

if needed_h_minus_ff / alpha_manual > 10:
    print("🔧 H⁻ ff is too small by factor >10")
    print("   Likely issue with K values or pressure calculation")
elif needed_h_minus_ff / alpha_manual > 2:
    print("⚠️  H⁻ ff is too small by factor 2-10")
    print("   May need to adjust temperature or wavelength scaling") 
else:
    print("✅ H⁻ ff magnitude is reasonable")
    print("   Small adjustments should achieve agreement")

# Test wavelength dependence  
print()
print("WAVELENGTH DEPENDENCE CHECK:")
wavelengths = [4000, 5500, 7000]
for wl in wavelengths:
    freq = 2.998e18 / wl
    alpha = h_minus_ff_absorption_coefficient(freq, T, n_h_i_ground, ne)
    print(f"  {wl:4.0f} Å: {float(alpha):.2e} cm⁻¹")

alpha_4000 = h_minus_ff_absorption_coefficient(2.998e18/4000, T, n_h_i_ground, ne)
alpha_7000 = h_minus_ff_absorption_coefficient(2.998e18/7000, T, n_h_i_ground, ne)
ratio = float(alpha_4000) / float(alpha_7000)
print(f"4000/7000 ratio: {ratio:.2f}")

if ratio < 1.0:
    print("✅ Correct: H⁻ ff decreases toward blue (expected from B&B table)")
else:
    print("❌ Incorrect: H⁻ ff increases toward blue")