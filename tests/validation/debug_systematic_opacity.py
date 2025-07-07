#!/usr/bin/env python3
"""
Systematic debugging of opacity calculation issues
"""

import sys
from pathlib import Path
jorg_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(jorg_src))

import numpy as np
import jax.numpy as jnp
from jorg.continuum.complete_continuum import (
    h_minus_bf_cross_section, rayleigh_scattering_cross_section,
    thomson_scattering_cross_section
)

print("SYSTEMATIC OPACITY DEBUGGING")
print("=" * 30)

# Test conditions - exactly matching Korg
T = 4838.3  # K
ne = 2.28e12  # cm⁻³ 
wavelength_angstrom = 5500.0  # Å
frequency = 2.99792458e18 / wavelength_angstrom  # Hz

print(f"Wavelength: {wavelength_angstrom} Å")
print(f"Frequency: {frequency:.2e} Hz")
print(f"Photon energy: {6.626e-27 * frequency / 1.602e-12:.2f} eV")
print()

# Species densities (matching test)
n_HI = 2.5e16
n_HeI = 2.0e15

print("CROSS-SECTION ANALYSIS:")
print("=" * 25)

# 1. Check H⁻ bound-free threshold
h_minus_binding_eV = 0.754
threshold_wavelength = 6.626e-27 * 2.998e10 / (h_minus_binding_eV * 1.602e-12) * 1e8  # Å
print(f"H⁻ bf threshold: {threshold_wavelength:.0f} Å")
print(f"Test wavelength: {wavelength_angstrom} Å")

if wavelength_angstrom < threshold_wavelength:
    print("✓ H⁻ bf absorption possible")
    sigma_h_minus_bf = h_minus_bf_cross_section(frequency)
    print(f"H⁻ bf cross-section: {float(sigma_h_minus_bf):.2e} cm²")
else:
    print("❌ H⁻ bf absorption impossible at this wavelength!")
    sigma_h_minus_bf = 0.0

# 2. Check Rayleigh scattering (λ^-4 dependence)
sigma_rayleigh = rayleigh_scattering_cross_section(frequency)
print(f"Rayleigh cross-section: {float(sigma_rayleigh):.2e} cm²")

# Expected Rayleigh at 5500 Å
sigma_rayleigh_expected = 5.8e-24 * (5000.0 / wavelength_angstrom)**4
print(f"Expected Rayleigh: {sigma_rayleigh_expected:.2e} cm²")
print(f"Rayleigh ratio: {float(sigma_rayleigh) / sigma_rayleigh_expected:.2f}")

# 3. Thomson scattering
sigma_thomson = thomson_scattering_cross_section()
print(f"Thomson cross-section: {float(sigma_thomson):.2e} cm²")
print()

print("ABSORPTION COEFFICIENT ESTIMATES:")
print("=" * 35)

# Thomson contribution (should dominate)
alpha_thomson = ne * float(sigma_thomson)
print(f"Thomson: ne × σ = {ne:.2e} × {float(sigma_thomson):.2e} = {alpha_thomson:.2e} cm⁻¹")

# Rayleigh contribution  
alpha_rayleigh = n_HI * float(sigma_rayleigh)
print(f"Rayleigh: nHI × σ = {n_HI:.2e} × {float(sigma_rayleigh):.2e} = {alpha_rayleigh:.2e} cm⁻¹")

# H⁻ contribution (if any)
if sigma_h_minus_bf > 0:
    # Use corrected H⁻ density calculation
    h_minus_binding_eV = 0.754
    thermal_energy_eV = 8.617e-5 * T
    saha_factor = (ne / 2.0) * (6.626e-27**2 / (2 * np.pi * 9.109e-28 * 1.381e-16 * T))**(1.5)
    saha_factor *= np.exp(h_minus_binding_eV / thermal_energy_eV)
    h_minus_density = min(n_HI * saha_factor, n_HI * 1e-8)
    
    stim_factor = 1.0 - np.exp(-6.626e-27 * frequency / (1.381e-16 * T))
    alpha_h_minus = h_minus_density * float(sigma_h_minus_bf) * stim_factor
    print(f"H⁻ bf: nH⁻ × σ × stim = {h_minus_density:.2e} × {float(sigma_h_minus_bf):.2e} × {stim_factor:.3f} = {alpha_h_minus:.2e} cm⁻¹")
else:
    alpha_h_minus = 0.0
    print(f"H⁻ bf: Not possible at this wavelength")

total_alpha_manual = alpha_thomson + alpha_rayleigh + alpha_h_minus
print()
print(f"Total α (manual): {total_alpha_manual:.2e} cm⁻¹")
print(f"Expected (Korg): {1.7e-16:.2e} cm⁻¹") 
print(f"Ratio: {total_alpha_manual / 1.7e-16:.2e}")
print()

# Identify the problem
if alpha_rayleigh > 10 * alpha_thomson:
    print("❌ PROBLEM: Rayleigh scattering is abnormally large!")
    print("   Check wavelength dependence and reference values")
elif alpha_h_minus > 10 * alpha_thomson:
    print("❌ PROBLEM: H⁻ absorption is abnormally large!")
    print("   Check cross-section or density calculation")
elif total_alpha_manual > 1000 * 1.7e-16:
    print("❌ PROBLEM: All contributions are too large!")
    print("   Check fundamental constants or unit conversions")
else:
    print("✓ Individual contributions seem reasonable")
    print("   Problem might be in JAX function implementation")