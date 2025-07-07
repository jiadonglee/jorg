#!/usr/bin/env python3
"""
Debug the unexpected wavelength dependence
"""

import sys
from pathlib import Path
jorg_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(jorg_src))

import numpy as np
import jax.numpy as jnp
from jorg.continuum.complete_continuum import (
    rayleigh_scattering_korg_style, h_minus_bf_cross_section,
    thomson_scattering_cross_section, total_continuum_absorption_jorg
)

print("DEBUG WAVELENGTH DEPENDENCE")
print("=" * 30)

# Test three wavelengths
wavelengths = np.array([4000, 5500, 7000])  # Å
frequencies = 2.998e18 / wavelengths  # Hz

# Test conditions
T = 4838.3  # K
ne = 2.28e12  # cm⁻³
n_HI = 2.5e16  # cm⁻³
n_HeI = 2.0e15  # cm⁻³
n_H2 = 1.0e13  # cm⁻³

print("Testing individual components at different wavelengths:")
print()

for i, (wl, freq) in enumerate(zip(wavelengths, frequencies)):
    print(f"WAVELENGTH: {wl:.0f} Å ({freq:.2e} Hz)")
    
    # 1. Thomson scattering (wavelength independent)
    sigma_thomson = thomson_scattering_cross_section()
    alpha_thomson = ne * float(sigma_thomson)
    
    # 2. Rayleigh scattering (should be ∝ λ⁻⁴)
    alpha_rayleigh = rayleigh_scattering_korg_style(freq, n_HI, n_HeI, n_H2)
    
    # 3. H⁻ bound-free (wavelength dependent)
    sigma_h_minus_bf = h_minus_bf_cross_section(freq)
    # Estimate H⁻ density
    h_minus_binding_eV = 0.754
    thermal_energy_eV = 8.617e-5 * T
    saha_factor = (ne / 2.0) * (6.626e-27**2 / (2 * np.pi * 9.109e-28 * 1.381e-16 * T))**(1.5)
    saha_factor *= np.exp(h_minus_binding_eV / thermal_energy_eV)
    h_minus_density = min(n_HI * saha_factor, n_HI * 1e-8)
    stim_factor = 1.0 - np.exp(-6.626e-27 * freq / (1.381e-16 * T))
    alpha_h_minus_bf = h_minus_density * float(sigma_h_minus_bf) * stim_factor
    
    # 4. Total from JAX function
    alpha_total_jax = total_continuum_absorption_jorg(
        jnp.array([freq]), T, ne, n_HI, 6.0e10, n_HeI, 1.0e11, 9.0e10, 3.0e10, n_H2
    )
    
    print(f"  Thomson:     {alpha_thomson:.2e} cm⁻¹")
    print(f"  Rayleigh:    {float(alpha_rayleigh):.2e} cm⁻¹")
    print(f"  H⁻ bf:       {alpha_h_minus_bf:.2e} cm⁻¹")
    print(f"  Manual sum:  {alpha_thomson + float(alpha_rayleigh) + alpha_h_minus_bf:.2e} cm⁻¹")
    print(f"  JAX total:   {float(alpha_total_jax[0]):.2e} cm⁻¹")
    print()

# Check Rayleigh scaling
print("RAYLEIGH SCATTERING SCALING:")
rayleigh_4000 = rayleigh_scattering_korg_style(frequencies[0], n_HI, n_HeI, n_H2)
rayleigh_7000 = rayleigh_scattering_korg_style(frequencies[2], n_HI, n_HeI, n_H2)
rayleigh_ratio = float(rayleigh_4000) / float(rayleigh_7000)
expected_ratio = (7000/4000)**4  # λ⁻⁴ scaling

print(f"  4000 Å: {float(rayleigh_4000):.2e} cm⁻¹")
print(f"  7000 Å: {float(rayleigh_7000):.2e} cm⁻¹")
print(f"  Ratio (4000/7000): {rayleigh_ratio:.2f}")
print(f"  Expected λ⁻⁴ ratio: {expected_ratio:.2f}")
print(f"  Scaling check: {'✓' if abs(rayleigh_ratio/expected_ratio - 1) < 0.1 else '❌'}")
print()

# Check H⁻ bf scaling
print("H⁻ BOUND-FREE SCALING:")
h_minus_4000 = h_minus_bf_cross_section(frequencies[0])
h_minus_7000 = h_minus_bf_cross_section(frequencies[2])
print(f"  4000 Å cross-section: {float(h_minus_4000):.2e} cm²")
print(f"  7000 Å cross-section: {float(h_minus_7000):.2e} cm²")
print(f"  Ratio (4000/7000): {float(h_minus_4000)/float(h_minus_7000):.2f}")

# Check total scaling
print()
print("TOTAL OPACITY SCALING:")
alpha_4000 = total_continuum_absorption_jorg(
    jnp.array([frequencies[0]]), T, ne, n_HI, 6.0e10, n_HeI, 1.0e11, 9.0e10, 3.0e10, n_H2
)[0]
alpha_7000 = total_continuum_absorption_jorg(
    jnp.array([frequencies[2]]), T, ne, n_HI, 6.0e10, n_HeI, 1.0e11, 9.0e10, 3.0e10, n_H2
)[0]

print(f"  4000 Å total: {float(alpha_4000):.2e} cm⁻¹")
print(f"  7000 Å total: {float(alpha_7000):.2e} cm⁻¹")
print(f"  Ratio (4000/7000): {float(alpha_4000)/float(alpha_7000):.2f}")

if float(alpha_4000) > float(alpha_7000):
    print("  ✓ Blue opacity > Red opacity (as expected)")
    print("  ? Previous test may have had incorrect wavelength ordering")
else:
    print("  ❌ Red opacity > Blue opacity (unexpected)")
    print("  Need to investigate dominant opacity source")