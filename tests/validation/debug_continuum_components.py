#!/usr/bin/env python3
"""
Debug each continuum component individually
"""

import sys
from pathlib import Path
jorg_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(jorg_src))

import numpy as np
import jax.numpy as jnp
from jorg.continuum.complete_continuum import (
    h_minus_bf_cross_section, h_minus_ff_cross_section, 
    h_i_bf_cross_section, he_i_bf_cross_section,
    thomson_scattering_cross_section, rayleigh_scattering_cross_section,
    metal_bf_cross_section
)

print("CONTINUUM COMPONENTS DEBUG")
print("=" * 30)

# Test conditions
T = 4838.3  # K
ne = 2.28e12  # cm⁻³ 
frequency = 5.451e14  # Hz (5500 Å)

# Species densities
n_HI = 2.5e16
n_HII = 6.0e10
n_HeI = 2.0e15  
n_HeII = 1.0e11
n_FeI = 9.0e10
n_FeII = 3.0e10

print(f"Test frequency: {frequency:.2e} Hz (5500 Å)")
print(f"Temperature: {T} K")
print()

# Test each cross-section
components = {}

# 1. H⁻ bound-free
sigma_h_minus_bf = h_minus_bf_cross_section(frequency)
print(f"H⁻ bf cross-section: {float(sigma_h_minus_bf):.2e} cm²")
components['H⁻ bf'] = float(sigma_h_minus_bf)

# 2. H⁻ free-free  
sigma_h_minus_ff = h_minus_ff_cross_section(frequency, T)
print(f"H⁻ ff cross-section: {float(sigma_h_minus_ff):.2e} cm²")
components['H⁻ ff'] = float(sigma_h_minus_ff)

# 3. H I bound-free
sigma_h_i_bf = h_i_bf_cross_section(frequency, 1)
print(f"H I bf cross-section (n=1): {float(sigma_h_i_bf):.2e} cm²")
components['H I bf'] = float(sigma_h_i_bf)

# 4. He I bound-free
sigma_he_i_bf = he_i_bf_cross_section(frequency)
print(f"He I bf cross-section: {float(sigma_he_i_bf):.2e} cm²")
components['He I bf'] = float(sigma_he_i_bf)

# 5. Thomson scattering
sigma_thomson = thomson_scattering_cross_section()
print(f"Thomson cross-section: {float(sigma_thomson):.2e} cm²")
components['Thomson'] = float(sigma_thomson)

# 6. Rayleigh scattering
sigma_rayleigh = rayleigh_scattering_cross_section(frequency)
print(f"Rayleigh cross-section: {float(sigma_rayleigh):.2e} cm²")
components['Rayleigh'] = float(sigma_rayleigh)

# 7. Fe I bound-free
sigma_fe_bf = metal_bf_cross_section(frequency, 26, 0)
print(f"Fe I bf cross-section: {float(sigma_fe_bf):.2e} cm²")
components['Fe I bf'] = float(sigma_fe_bf)

print()
print("ABSORPTION COEFFICIENT CONTRIBUTIONS:")
print("=" * 40)

# Calculate contributions to α (cm⁻¹)
stim_factor = 1.0 - np.exp(-6.626e-27 * frequency / (1.381e-16 * T))
print(f"Stimulated emission factor: {stim_factor:.4f}")
print()

total_alpha = 0.0

# H⁻ contributions (simplified densities)
h_minus_density = 1e-12 * n_HI * ne / T  # Simplified Saha
alpha_h_minus_bf = h_minus_density * components['H⁻ bf'] * stim_factor
alpha_h_minus_ff = n_HI * ne * components['H⁻ ff'] * stim_factor * 1e-15
print(f"H⁻ density: {h_minus_density:.2e} cm⁻³")
print(f"H⁻ bf contribution: {alpha_h_minus_bf:.2e} cm⁻¹")
print(f"H⁻ ff contribution: {alpha_h_minus_ff:.2e} cm⁻¹")
total_alpha += alpha_h_minus_bf + alpha_h_minus_ff

# H I bf
alpha_h_i_bf = n_HI * components['H I bf'] * stim_factor
print(f"H I bf contribution: {alpha_h_i_bf:.2e} cm⁻¹")
total_alpha += alpha_h_i_bf

# He I bf
alpha_he_i_bf = n_HeI * components['He I bf'] * stim_factor
print(f"He I bf contribution: {alpha_he_i_bf:.2e} cm⁻¹")
total_alpha += alpha_he_i_bf

# Thomson scattering
alpha_thomson = ne * components['Thomson']
print(f"Thomson contribution: {alpha_thomson:.2e} cm⁻¹")
total_alpha += alpha_thomson

# Rayleigh scattering
alpha_rayleigh = n_HI * components['Rayleigh']
print(f"Rayleigh contribution: {alpha_rayleigh:.2e} cm⁻¹")
total_alpha += alpha_rayleigh

# Fe I bf
alpha_fe_bf = n_FeI * components['Fe I bf'] * stim_factor
print(f"Fe I bf contribution: {alpha_fe_bf:.2e} cm⁻¹")
total_alpha += alpha_fe_bf

print()
print(f"TOTAL α: {total_alpha:.2e} cm⁻¹")
print(f"Expected (Korg): {1.7e-16:.2e} cm⁻¹")
print(f"Ratio: {total_alpha / 1.7e-16:.2e}")

# Find the dominant contributor
max_contrib = max([alpha_h_minus_bf, alpha_h_minus_ff, alpha_h_i_bf, alpha_he_i_bf, 
                   alpha_thomson, alpha_rayleigh, alpha_fe_bf])
print()
print("DOMINANT CONTRIBUTOR:")
if max_contrib == alpha_h_i_bf:
    print(f"❌ H I bf: {alpha_h_i_bf:.2e} cm⁻¹ (SUSPICIOUS!)")
elif max_contrib == alpha_rayleigh:
    print(f"❌ Rayleigh: {alpha_rayleigh:.2e} cm⁻¹ (CHECK WAVELENGTH DEPENDENCE!)")
else:
    print(f"✓ Expected component is dominant")