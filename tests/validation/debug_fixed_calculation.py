#!/usr/bin/env python3
"""
Debug the fixed continuum calculation using the actual JAX function
"""

import sys
from pathlib import Path
jorg_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(jorg_src))

import numpy as np
import jax.numpy as jnp
from jorg.continuum.complete_continuum import total_continuum_absorption_jorg

print("FIXED CONTINUUM CALCULATION DEBUG")
print("=" * 35)

# Test conditions
T = 4838.3  # K
ne = 2.28e12  # cm⁻³ 
frequencies = jnp.array([5.451e14])  # Hz (5500 Å)

# Species densities
n_HI = 2.5e16
n_HII = 6.0e10
n_HeI = 2.0e15  
n_HeII = 1.0e11
n_FeI = 9.0e10
n_FeII = 3.0e10

print(f"Test frequency: {frequencies[0]:.2e} Hz (5500 Å)")
print(f"Temperature: {T} K")
print(f"Electron density: {ne:.2e} cm⁻³")
print()

# Call the JAX function directly  
alpha_total = total_continuum_absorption_jorg(
    frequencies, T, ne, n_HI, n_HII, n_HeI, n_HeII, n_FeI, n_FeII
)

print(f"Total α from JAX function: {float(alpha_total[0]):.2e} cm⁻¹")
print(f"Expected (Korg): {1.7e-16:.2e} cm⁻¹")
print(f"Ratio: {float(alpha_total[0]) / 1.7e-16:.2e}")
print()

# Now check what components are included by manually testing key physics

# H⁻ binding energy = 0.754 eV
h_minus_binding_eV = 0.754
thermal_energy_eV = 8.617e-5 * T  # kT in eV

# Corrected Saha factor for H⁻ formation
saha_factor = (ne / 2.0) * (6.626e-27**2 / (2 * np.pi * 9.109e-28 * 1.381e-16 * T))**(1.5)
saha_factor *= np.exp(h_minus_binding_eV / thermal_energy_eV)
h_minus_density = n_HI * saha_factor

# Apply realistic cutoff
h_minus_density = min(h_minus_density, n_HI * 1e-8)

print(f"Corrected H⁻ density: {h_minus_density:.2e} cm⁻³")
print(f"H⁻ fraction of H I: {h_minus_density / n_HI:.2e}")

# If this is still too big, the issue is elsewhere
if h_minus_density > n_HI * 1e-6:
    print("❌ H⁻ density still suspiciously large")
else:
    print("✓ H⁻ density seems reasonable")
    
print()
print("If H⁻ density is reasonable but total opacity is still too large,")
print("the error might be in other cross-sections or wavelength thresholds.")