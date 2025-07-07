#!/usr/bin/env python3
"""
Test the corrected H⁻ bound-free implementation
"""

import sys
from pathlib import Path
jorg_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(jorg_src))

import numpy as np
import jax.numpy as jnp
from jorg.continuum.complete_continuum import (
    h_minus_bf_cross_section, total_continuum_absorption_jorg
)

print("CORRECTED H⁻ BOUND-FREE TEST")
print("=" * 32)

# Test conditions
T = 4838.3  # K
ne = 2.28e12  # cm⁻³
n_HI = 2.5e16  # cm⁻³
wavelengths = [4000, 5500, 7000]  # Å
frequencies = [2.998e18 / wl for wl in wavelengths]

print(f"Temperature: {T} K")
print(f"Electron density: {ne:.2e} cm⁻³")
print(f"H I density: {n_HI:.2e} cm⁻³")
print()

# Test new H⁻ density calculation (Korg exact formula)
h_minus_binding_eV = 0.754204
kboltz_eV = 8.617333262145e-5
coef = 3.31283018e-22  # cm³*eV^1.5
beta = 1.0 / (kboltz_eV * T)

h_minus_density = 0.25 * n_HI * ne * coef * (beta**1.5) * np.exp(h_minus_binding_eV * beta)

print(f"CORRECTED H⁻ DENSITY:")
print(f"  Korg exact formula: {h_minus_density:.2e} cm⁻³")
print(f"  H⁻ fraction of H I: {h_minus_density / n_HI:.2e}")
print()

# Test new cross-sections
print(f"CORRECTED H⁻ CROSS-SECTIONS:")
for wl, freq in zip(wavelengths, frequencies):
    sigma = h_minus_bf_cross_section(freq)
    print(f"  {wl:4.0f} Å: {float(sigma):.2e} cm²")

print()

# Test total opacity with corrected H⁻
print(f"TOTAL OPACITY WITH CORRECTED H⁻:")
for wl, freq in zip(wavelengths, frequencies):
    alpha_total = total_continuum_absorption_jorg(
        jnp.array([freq]), T, ne, n_HI, 6.0e10, 2.0e15, 1.0e11, 9.0e10, 3.0e10, 1.0e13
    )
    print(f"  {wl:4.0f} Å: {float(alpha_total[0]):.2e} cm⁻¹")

print()

# Compare to previous values and Korg
print(f"COMPARISON:")
freq_5500 = 2.998e18 / 5500  # Hz
alpha_5500 = total_continuum_absorption_jorg(
    jnp.array([freq_5500]), T, ne, n_HI, 6.0e10, 2.0e15, 1.0e11, 9.0e10, 3.0e10, 1.0e13
)[0]

korg_5500 = 3.5e-9  # cm⁻¹
jorg_5500 = float(alpha_5500)

print(f"  Korg 5500 Å: {korg_5500:.2e} cm⁻¹")
print(f"  Jorg 5500 Å: {jorg_5500:.2e} cm⁻¹")
print(f"  Ratio (Jorg/Korg): {jorg_5500 / korg_5500:.2f}")
print()

if abs(jorg_5500 / korg_5500 - 1) < 0.2:
    print("🎉 EXCELLENT: Corrected H⁻ gives perfect agreement!")
elif abs(jorg_5500 / korg_5500 - 1) < 0.5:
    print("✅ VERY GOOD: Corrected H⁻ within 50% of Korg")
elif abs(jorg_5500 / korg_5500 - 1) < 1.0:
    print("✅ GOOD: Corrected H⁻ within factor of 2 of Korg")
else:
    print("⚠️  IMPROVEMENT NEEDED: Still significant difference")

# Check wavelength dependence
alpha_4000 = total_continuum_absorption_jorg(
    jnp.array([frequencies[0]]), T, ne, n_HI, 6.0e10, 2.0e15, 1.0e11, 9.0e10, 3.0e10, 1.0e13
)[0]
alpha_7000 = total_continuum_absorption_jorg(
    jnp.array([frequencies[2]]), T, ne, n_HI, 6.0e10, 2.0e15, 1.0e11, 9.0e10, 3.0e10, 1.0e13
)[0]

blue_red_ratio = float(alpha_4000) / float(alpha_7000)
print()
print(f"WAVELENGTH DEPENDENCE:")
print(f"  4000 Å: {float(alpha_4000):.2e} cm⁻¹")
print(f"  7000 Å: {float(alpha_7000):.2e} cm⁻¹")
print(f"  Blue/Red ratio: {blue_red_ratio:.2f}")

if blue_red_ratio > 1.0:
    print("✅ CORRECT: Blue opacity > Red opacity (expected)")
else:
    print("❌ INCORRECT: Red opacity > Blue opacity (unexpected)")