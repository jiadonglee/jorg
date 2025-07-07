#!/usr/bin/env python3
"""
Debug wavelength dependence with corrected H⁻ implementation
"""

import sys
from pathlib import Path
jorg_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(jorg_src))

import numpy as np
import jax.numpy as jnp
from jorg.continuum.complete_continuum import (
    h_minus_bf_cross_section, rayleigh_scattering_korg_style,
    thomson_scattering_cross_section, total_continuum_absorption_jorg
)

print("DEBUG WAVELENGTH DEPENDENCE (CORRECTED)")
print("=" * 40)

# Test conditions
T = 4838.3  # K
ne = 2.28e12  # cm⁻³
n_HI = 2.5e16  # cm⁻³
n_HeI = 2.0e15  # cm⁻³
n_H2 = 1.0e13  # cm⁻³

wavelengths = [4000, 5500, 7000]  # Å
frequencies = [2.998e18 / wl for wl in wavelengths]

print("COMPONENT ANALYSIS:")
print()

for wl, freq in zip(wavelengths, frequencies):
    print(f"WAVELENGTH: {wl} Å")
    
    # 1. Thomson scattering (constant)
    sigma_thomson = thomson_scattering_cross_section()
    alpha_thomson = ne * float(sigma_thomson)
    
    # 2. Rayleigh scattering 
    alpha_rayleigh = rayleigh_scattering_korg_style(freq, n_HI, n_HeI, n_H2)
    
    # 3. H⁻ bound-free (corrected)
    sigma_h_minus = h_minus_bf_cross_section(freq)
    
    # Corrected H⁻ density
    h_minus_binding_eV = 0.754204
    kboltz_eV = 8.617333262145e-5
    coef = 3.31283018e-22
    beta = 1.0 / (kboltz_eV * T)
    h_minus_density = 0.25 * n_HI * ne * coef * (beta**1.5) * np.exp(h_minus_binding_eV * beta)
    
    stim_factor = 1.0 - np.exp(-6.626e-27 * freq / (1.381e-16 * T))
    alpha_h_minus = h_minus_density * float(sigma_h_minus) * stim_factor
    
    # 4. Total
    alpha_total_manual = alpha_thomson + float(alpha_rayleigh) + alpha_h_minus
    
    # 5. JAX function
    alpha_total_jax = total_continuum_absorption_jorg(
        jnp.array([freq]), T, ne, n_HI, 6.0e10, n_HeI, 1.0e11, 9.0e10, 3.0e10, n_H2
    )[0]
    
    print(f"  Thomson:        {alpha_thomson:.2e} cm⁻¹")
    print(f"  Rayleigh:       {float(alpha_rayleigh):.2e} cm⁻¹")
    print(f"  H⁻ bf:          {alpha_h_minus:.2e} cm⁻¹")
    print(f"  Manual total:   {alpha_total_manual:.2e} cm⁻¹")
    print(f"  JAX total:      {float(alpha_total_jax):.2e} cm⁻¹")
    print(f"  Dominant:       {'H⁻' if alpha_h_minus > max(alpha_thomson, float(alpha_rayleigh)) else 'Rayleigh' if float(alpha_rayleigh) > alpha_thomson else 'Thomson'}")
    print()

# Check individual scaling
print("SCALING ANALYSIS:")
print()

# Rayleigh scaling
rayleigh_4000 = rayleigh_scattering_korg_style(frequencies[0], n_HI, n_HeI, n_H2)
rayleigh_7000 = rayleigh_scattering_korg_style(frequencies[2], n_HI, n_HeI, n_H2)
rayleigh_ratio = float(rayleigh_4000) / float(rayleigh_7000)

print(f"RAYLEIGH SCALING:")
print(f"  4000/7000 ratio: {rayleigh_ratio:.2f}")
print(f"  Expected λ⁻⁴: {(7000/4000)**4:.2f}")
print(f"  Scaling OK: {'✅' if abs(rayleigh_ratio / (7000/4000)**4 - 1) < 0.2 else '❌'}")
print()

# H⁻ scaling
h_minus_4000 = h_minus_bf_cross_section(frequencies[0])
h_minus_7000 = h_minus_bf_cross_section(frequencies[2])
h_minus_ratio = float(h_minus_4000) / float(h_minus_7000)

print(f"H⁻ CROSS-SECTION SCALING:")
print(f"  4000 Å: {float(h_minus_4000):.2e} cm²")
print(f"  7000 Å: {float(h_minus_7000):.2e} cm²")
print(f"  4000/7000 ratio: {h_minus_ratio:.2f}")
print()

# Check if there's a bug in my frequency/wavelength conversion
print("FREQUENCY CHECK:")
for wl, freq in zip(wavelengths, frequencies):
    check_wl = 2.998e18 / freq
    print(f"  {wl} Å → {freq:.2e} Hz → {check_wl:.1f} Å ({'✅' if abs(check_wl - wl) < 1 else '❌'})")

print()
print("CONCLUSION:")
if rayleigh_ratio > 5:  # Should dominate for proper λ⁻⁴ scaling
    print("✅ Rayleigh shows correct λ⁻⁴ scaling")
    if h_minus_ratio < 1:
        print("⚠️  H⁻ cross-section decreases toward blue (unexpected)")
        print("   This suggests either incorrect H⁻ cross-section formula")
        print("   or missing physics in other components")
    else:
        print("✅ H⁻ cross-section increases toward blue (expected)")
else:
    print("❌ Rayleigh scaling is incorrect")