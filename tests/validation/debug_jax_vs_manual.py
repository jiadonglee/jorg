#!/usr/bin/env python3
"""
Debug the discrepancy between JAX and manual calculations
"""

import sys
from pathlib import Path
jorg_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(jorg_src))

import numpy as np
import jax.numpy as jnp
from jorg.continuum.complete_continuum import (
    h_minus_bf_cross_section, h_minus_ff_cross_section,
    rayleigh_scattering_korg_style, thomson_scattering_cross_section,
    h_i_bf_cross_section, he_i_bf_cross_section, metal_bf_cross_section
)

print("DEBUG JAX vs MANUAL CALCULATION")
print("=" * 35)

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

print(f"Test frequency: {frequency:.2e} Hz (5500 Å)")
print(f"Temperature: {T} K")
print()

# Manual calculation of each component
print("MANUAL COMPONENT CALCULATION:")

# Stimulated emission factor
stim_factor = 1.0 - np.exp(-6.626e-27 * frequency / (1.381e-16 * T))
print(f"Stimulated emission factor: {stim_factor:.6f}")
print()

# 1. H⁻ bound-free
h_minus_binding_eV = 0.754204
kboltz_eV = 8.617333262145e-5
coef = 3.31283018e-22
beta = 1.0 / (kboltz_eV * T)
h_minus_density = 0.25 * n_HI * ne * coef * (beta**1.5) * np.exp(h_minus_binding_eV * beta)
sigma_h_minus_bf = h_minus_bf_cross_section(frequency)
alpha_h_minus_bf = h_minus_density * float(sigma_h_minus_bf) * stim_factor
print(f"1. H⁻ bf: n={h_minus_density:.2e}, σ={float(sigma_h_minus_bf):.2e}, α={alpha_h_minus_bf:.2e}")

# 2. H⁻ free-free
sigma_h_minus_ff = h_minus_ff_cross_section(frequency, T)
alpha_h_minus_ff = n_HI * ne * float(sigma_h_minus_ff) * stim_factor * 1e-15  # Scale factor
print(f"2. H⁻ ff: σ={float(sigma_h_minus_ff):.2e}, α={alpha_h_minus_ff:.2e}")

# 3. H I bound-free
sigma_h_i_bf = h_i_bf_cross_section(frequency, 1)
alpha_h_i_bf = n_HI * float(sigma_h_i_bf) * stim_factor
print(f"3. H I bf: σ={float(sigma_h_i_bf):.2e}, α={alpha_h_i_bf:.2e}")

# 4. He I bound-free
sigma_he_i_bf = he_i_bf_cross_section(frequency)
alpha_he_i_bf = n_HeI * float(sigma_he_i_bf) * stim_factor
print(f"4. He I bf: σ={float(sigma_he_i_bf):.2e}, α={alpha_he_i_bf:.2e}")

# 5. Thomson scattering
sigma_thomson = thomson_scattering_cross_section()
alpha_thomson = ne * float(sigma_thomson)
print(f"5. Thomson: σ={float(sigma_thomson):.2e}, α={alpha_thomson:.2e}")

# 6. Rayleigh scattering
alpha_rayleigh = rayleigh_scattering_korg_style(frequency, n_HI, n_HeI, n_H2)
print(f"6. Rayleigh: α={float(alpha_rayleigh):.2e}")

# 7. Fe I bound-free
sigma_fe_bf = metal_bf_cross_section(frequency, 26, 0)
alpha_fe_bf = n_FeI * float(sigma_fe_bf) * stim_factor
print(f"7. Fe I bf: σ={float(sigma_fe_bf):.2e}, α={alpha_fe_bf:.2e}")

# Manual total
alpha_manual_total = (alpha_h_minus_bf + alpha_h_minus_ff + alpha_h_i_bf + 
                     alpha_he_i_bf + alpha_thomson + float(alpha_rayleigh) + alpha_fe_bf)

print()
print(f"MANUAL TOTAL: {alpha_manual_total:.2e} cm⁻¹")
print()

# Now call JAX function and compare
from jorg.continuum.complete_continuum import total_continuum_absorption_jorg

alpha_jax = total_continuum_absorption_jorg(
    jnp.array([frequency]), T, ne, n_HI, n_HII, n_HeI, n_HeII, n_FeI, n_FeII, n_H2
)[0]

print(f"JAX TOTAL: {float(alpha_jax):.2e} cm⁻¹")
print(f"JAX/Manual ratio: {float(alpha_jax) / alpha_manual_total:.1f}")
print()

if float(alpha_jax) / alpha_manual_total > 2:
    print("❌ MAJOR DISCREPANCY: JAX function has bug or missing component")
    print("   Need to check JAX implementation line by line")
elif float(alpha_jax) / alpha_manual_total > 1.2:
    print("⚠️  MODERATE DISCREPANCY: Small bug or approximation difference")
else:
    print("✅ GOOD AGREEMENT: Manual and JAX calculations match")

# Check if any component is suspiciously large
max_component = max(alpha_h_minus_bf, alpha_h_minus_ff, alpha_h_i_bf, 
                   alpha_he_i_bf, alpha_thomson, float(alpha_rayleigh), alpha_fe_bf)

print()
print("COMPONENT ANALYSIS:")
components = [
    ("H⁻ bf", alpha_h_minus_bf),
    ("H⁻ ff", alpha_h_minus_ff), 
    ("H I bf", alpha_h_i_bf),
    ("He I bf", alpha_he_i_bf),
    ("Thomson", alpha_thomson),
    ("Rayleigh", float(alpha_rayleigh)),
    ("Fe I bf", alpha_fe_bf)
]

for name, value in components:
    percentage = 100 * value / alpha_manual_total
    marker = "🔥" if value == max_component else "  "
    print(f"{marker} {name:8}: {value:.2e} cm⁻¹ ({percentage:5.1f}%)")

print()
print("Expected dominance: H⁻ bf should be largest, followed by Rayleigh or Thomson")