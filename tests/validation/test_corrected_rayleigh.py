#!/usr/bin/env python3
"""
Test the corrected Rayleigh scattering implementation
"""

import sys
from pathlib import Path
jorg_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(jorg_src))

import numpy as np
import jax.numpy as jnp
from jorg.continuum.complete_continuum import rayleigh_scattering_korg_style, total_continuum_absorption_jorg

print("CORRECTED RAYLEIGH SCATTERING TEST")
print("=" * 35)

# Test conditions
T = 4838.3  # K
ne = 2.28e12  # cm⁻³ 
frequency = 5.451e14  # Hz (5500 Å)

# Species densities
n_HI = 2.5e16
n_HeI = 2.0e15
n_H2 = 1.0e13

print(f"Test frequency: {frequency:.2e} Hz (5500 Å)")
print(f"H I density: {n_HI:.2e} cm⁻³")
print(f"He I density: {n_HeI:.2e} cm⁻³")
print(f"H2 density: {n_H2:.2e} cm⁻³")
print()

# Test corrected Rayleigh scattering
alpha_rayleigh_new = rayleigh_scattering_korg_style(frequency, n_HI, n_HeI, n_H2)
print(f"Corrected Rayleigh scattering: {float(alpha_rayleigh_new):.2e} cm⁻¹")

# Compare to old simple formula (for reference)
wavelength_angstrom = 2.998e18 / frequency
sigma_rayleigh_old = 5.8e-24 * (5000.0 / wavelength_angstrom)**4
alpha_rayleigh_old = n_HI * sigma_rayleigh_old
print(f"Old simple Rayleigh: {alpha_rayleigh_old:.2e} cm⁻¹")
print(f"Improvement factor: {alpha_rayleigh_old / float(alpha_rayleigh_new):.1f}")
print()

# Test full continuum calculation with corrected Rayleigh
frequencies = jnp.array([frequency])
alpha_total = total_continuum_absorption_jorg(
    frequencies, T, ne, n_HI, 6.0e10, n_HeI, 1.0e11, 9.0e10, 3.0e10, n_H2
)

print(f"Total continuum (corrected): {float(alpha_total[0]):.2e} cm⁻¹")
print(f"Expected Korg value: {3.5e-9:.2e} cm⁻¹")
print(f"Ratio (Jorg/Korg): {float(alpha_total[0]) / 3.5e-9:.1f}")
print()

if abs(float(alpha_total[0]) / 3.5e-9 - 1) < 0.5:
    print("🎉 EXCELLENT: Jorg matches Korg within 50%!")
elif abs(float(alpha_total[0]) / 3.5e-9 - 1) < 2.0:
    print("✅ GOOD: Jorg matches Korg within factor of 2")
elif abs(float(alpha_total[0]) / 3.5e-9 - 1) < 5.0:
    print("⚠️  REASONABLE: Jorg matches Korg within factor of 5")
else:
    print("❌ POOR: Still significant disagreement")
    
# Break down the contributions again
print()
print("COMPONENT BREAKDOWN:")
thomson = ne * 6.652e-25
h_minus_saha = min(n_HI * (ne/2) * (6.626e-27**2/(2*np.pi*9.109e-28*1.381e-16*T))**(1.5) * np.exp(0.754/(8.617e-5*T)), n_HI*1e-8)
h_minus_contrib = h_minus_saha * 3.9e-17 * 0.996  # rough estimate

print(f"Thomson: {thomson:.2e} cm⁻¹")
print(f"Rayleigh (corrected): {float(alpha_rayleigh_new):.2e} cm⁻¹")
print(f"H⁻ estimate: {h_minus_contrib:.2e} cm⁻¹")
total_manual = thomson + float(alpha_rayleigh_new) + h_minus_contrib
print(f"Manual total: {total_manual:.2e} cm⁻¹")
print(f"JAX function: {float(alpha_total[0]):.2e} cm⁻¹")