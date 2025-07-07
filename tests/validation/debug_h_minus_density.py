#!/usr/bin/env python3
"""
Debug H⁻ density calculation to see if it's reasonable
"""

import sys
from pathlib import Path
jorg_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(jorg_src))

import numpy as np

print("H⁻ DENSITY DEBUGGING")
print("=" * 25)

# Test conditions
T = 4838.3  # K
ne = 2.28e12  # cm⁻³
n_HI = 2.5e16  # cm⁻³

print(f"Temperature: {T} K")
print(f"Electron density: {ne:.2e} cm⁻³") 
print(f"H I density: {n_HI:.2e} cm⁻³")
print()

# My current H⁻ Saha calculation
h_minus_binding_eV = 0.754
thermal_energy_eV = 8.617e-5 * T  # kT in eV

print(f"H⁻ binding energy: {h_minus_binding_eV} eV")
print(f"Thermal energy (kT): {thermal_energy_eV:.3f} eV")
print(f"Binding/thermal ratio: {h_minus_binding_eV / thermal_energy_eV:.1f}")
print()

# Saha equation for H⁻ formation: H + e⁻ → H⁻
# n(H⁻)/n(H I) = (ne/2) * (h²/2πmkT)^(3/2) * exp(χ/kT)
saha_factor = (ne / 2.0) * (6.626e-27**2 / (2 * np.pi * 9.109e-28 * 1.381e-16 * T))**(1.5)
saha_factor *= np.exp(h_minus_binding_eV / thermal_energy_eV)

h_minus_density_raw = n_HI * saha_factor
h_minus_density_limited = min(h_minus_density_raw, n_HI * 1e-8)

print(f"SAHA CALCULATION:")
print(f"  Saha factor: {saha_factor:.2e}")
print(f"  Raw H⁻ density: {h_minus_density_raw:.2e} cm⁻³")
print(f"  Limited H⁻ density: {h_minus_density_limited:.2e} cm⁻³")
print(f"  H⁻ fraction of H I: {h_minus_density_limited / n_HI:.2e}")
print()

# Compare to typical values in stellar atmospheres
# H⁻ is usually a very small fraction of total H
print(f"PHYSICAL REASONABLENESS:")
if h_minus_density_limited / n_HI < 1e-6:
    print(f"✓ H⁻ fraction < 10⁻⁶ (reasonable for stellar atmospheres)")
elif h_minus_density_limited / n_HI < 1e-4:
    print(f"⚠ H⁻ fraction ~ 10⁻⁴ to 10⁻⁶ (might be high)")
else:
    print(f"❌ H⁻ fraction > 10⁻⁴ (unreasonably high)")

# Check contribution to opacity at different wavelengths
wavelengths = [4000, 5500, 7000]  # Å
print()
print(f"H⁻ OPACITY CONTRIBUTIONS:")

for wl in wavelengths:
    frequency = 2.998e18 / wl  # Hz
    
    # My current cross-section (corrected version)
    wavelength_cm = 2.998e10 / frequency
    wavelength_angstrom = wavelength_cm * 1e8
    h_minus_binding_eV = 0.754
    threshold_wavelength = 6.626e-27 * 2.998e10 / (h_minus_binding_eV * 1.602e-12) * 1e8
    
    if wavelength_angstrom < threshold_wavelength and wavelength_angstrom > 1000:
        # Use simple cross-section for testing
        sigma = 4.0e-17 * (wavelength_angstrom / 5000.0)**1.5
    else:
        sigma = 0.0
    
    stim_factor = 1.0 - np.exp(-6.626e-27 * frequency / (1.381e-16 * T))
    alpha_h_minus = h_minus_density_limited * sigma * stim_factor
    
    print(f"  {wl:4.0f} Å: σ={sigma:.2e} cm², α={alpha_h_minus:.2e} cm⁻¹")

print()
print(f"CONCLUSION:")
print(f"If H⁻ opacity is dominating, either:")
print(f"1. H⁻ density is too high (check Saha equation)")
print(f"2. H⁻ cross-section is too high (check Stilley & Callaway)")
print(f"3. Other opacity sources are too low (check Thomson, Rayleigh)")

# Quick check of Thomson vs H⁻ 
sigma_thomson = 6.652e-25  # cm²
alpha_thomson = ne * sigma_thomson
alpha_h_minus_typical = h_minus_density_limited * 3e-17 * 0.996  # Typical values

print(f"")
print(f"OPACITY COMPARISON:")
print(f"  Thomson scattering: {alpha_thomson:.2e} cm⁻¹")
print(f"  H⁻ bf (typical): {alpha_h_minus_typical:.2e} cm⁻¹")
print(f"  Ratio (H⁻/Thomson): {alpha_h_minus_typical / alpha_thomson:.1f}")

if alpha_h_minus_typical > 10 * alpha_thomson:
    print(f"❌ H⁻ opacity is suspiciously dominant")
else:
    print(f"✓ H⁻ and Thomson opacities are comparable")