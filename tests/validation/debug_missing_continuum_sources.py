#!/usr/bin/env python3
"""
Debug missing continuum sources to find remaining factor of 13 difference
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
    thomson_scattering_cross_section, rayleigh_scattering_korg_style,
    metal_bf_cross_section, positive_ion_ff_cross_section,
    total_continuum_absorption_jorg
)

print("MISSING CONTINUUM SOURCES DEBUG")
print("=" * 35)

# Test conditions (same as before)
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

# Current Jorg opacity
alpha_jorg = total_continuum_absorption_jorg(
    jnp.array([frequency]), T, ne, n_HI, n_HII, n_HeI, n_HeII, n_FeI, n_FeII, n_H2
)[0]

# Korg reference
alpha_korg = 3.5e-9  # cm⁻¹

print(f"CURRENT STATUS:")
print(f"  Jorg opacity: {float(alpha_jorg):.2e} cm⁻¹")
print(f"  Korg opacity: {alpha_korg:.2e} cm⁻¹")
print(f"  Missing factor: {alpha_korg / float(alpha_jorg):.1f}")
print()

# Let's manually calculate ALL possible continuum sources
print("DETAILED CONTINUUM SOURCE ANALYSIS:")
print()

# Stimulated emission factor
stim_factor = 1.0 - np.exp(-6.626e-27 * frequency / (1.381e-16 * T))

# 1. H⁻ bound-free (current implementation)
h_minus_binding_eV = 0.754204
kboltz_eV = 8.617333262145e-5
coef = 3.31283018e-22
beta = 1.0 / (kboltz_eV * T)
h_minus_density = 0.25 * n_HI * ne * coef * (beta**1.5) * np.exp(h_minus_binding_eV * beta)
sigma_h_minus_bf = h_minus_bf_cross_section(frequency)
alpha_h_minus_bf = h_minus_density * float(sigma_h_minus_bf) * stim_factor
print(f"1. H⁻ bf:          {alpha_h_minus_bf:.2e} cm⁻¹  ({100*alpha_h_minus_bf/alpha_korg:.1f}% of Korg)")

# 2. H⁻ free-free (currently disabled)
sigma_h_minus_ff = h_minus_ff_cross_section(frequency, T)
alpha_h_minus_ff_approx = n_HI * ne * float(sigma_h_minus_ff) * stim_factor * 1e-15
print(f"2. H⁻ ff (approx): {alpha_h_minus_ff_approx:.2e} cm⁻¹  ({100*alpha_h_minus_ff_approx/alpha_korg:.1f}% of Korg)")

# 3. H I bound-free (ground state only)
sigma_h_i_bf = h_i_bf_cross_section(frequency, 1)
alpha_h_i_bf = n_HI * float(sigma_h_i_bf) * stim_factor
print(f"3. H I bf (n=1):   {alpha_h_i_bf:.2e} cm⁻¹  ({100*alpha_h_i_bf/alpha_korg:.1f}% of Korg)")

# 4. H I bound-free (higher levels n=2,3,4,5,6)
alpha_h_i_bf_higher = 0.0
for n in range(2, 7):
    sigma_n = h_i_bf_cross_section(frequency, n)
    # Approximate population (Boltzmann factor)
    E_n = 13.6 * (1 - 1/n**2)  # eV
    boltz_factor = np.exp(-E_n / (kboltz_eV * T))
    n_level = n_HI * (2 * n**2) * boltz_factor / 2.0  # Rough estimate
    alpha_n = n_level * float(sigma_n) * stim_factor
    alpha_h_i_bf_higher += alpha_n
print(f"4. H I bf (n>1):   {alpha_h_i_bf_higher:.2e} cm⁻¹  ({100*alpha_h_i_bf_higher/alpha_korg:.1f}% of Korg)")

# 5. He I bound-free
sigma_he_i_bf = he_i_bf_cross_section(frequency)
alpha_he_i_bf = n_HeI * float(sigma_he_i_bf) * stim_factor
print(f"5. He I bf:        {alpha_he_i_bf:.2e} cm⁻¹  ({100*alpha_he_i_bf/alpha_korg:.1f}% of Korg)")

# 6. Thomson scattering
sigma_thomson = thomson_scattering_cross_section()
alpha_thomson = ne * float(sigma_thomson)
print(f"6. Thomson:        {alpha_thomson:.2e} cm⁻¹  ({100*alpha_thomson/alpha_korg:.1f}% of Korg)")

# 7. Rayleigh scattering
alpha_rayleigh = rayleigh_scattering_korg_style(frequency, n_HI, n_HeI, n_H2)
print(f"7. Rayleigh:       {float(alpha_rayleigh):.2e} cm⁻¹  ({100*float(alpha_rayleigh)/alpha_korg:.1f}% of Korg)")

# 8. Metal bound-free (Fe, Mg, Si, Ca, Na, Al)
alpha_metals = 0.0
elements = [(26, n_FeI, n_FeII), (12, 3e10, 1e10), (14, 4e10, 2e10), 
           (20, 2e10, 5e9), (11, 2e10, 1e9), (13, 3e10, 1e10)]
for Z, n_I, n_II in elements:
    for charge, density in [(0, n_I), (1, n_II)]:
        sigma_metal = metal_bf_cross_section(frequency, Z, charge)
        alpha_metal = density * float(sigma_metal) * stim_factor
        alpha_metals += alpha_metal
print(f"8. Metal bf:       {alpha_metals:.2e} cm⁻¹  ({100*alpha_metals/alpha_korg:.1f}% of Korg)")

# 9. Free-free absorption by positive ions (H II, He II, metals)
alpha_ff_positive = 0.0
# H II free-free
sigma_h_ii_ff = positive_ion_ff_cross_section(frequency, T, 1)
alpha_h_ii_ff = n_HII * ne * float(sigma_h_ii_ff) * stim_factor
alpha_ff_positive += alpha_h_ii_ff

# He II free-free  
sigma_he_ii_ff = positive_ion_ff_cross_section(frequency, T, 2)
alpha_he_ii_ff = n_HeII * ne * float(sigma_he_ii_ff) * stim_factor
alpha_ff_positive += alpha_he_ii_ff

print(f"9. Positive ion ff: {alpha_ff_positive:.2e} cm⁻¹  ({100*alpha_ff_positive/alpha_korg:.1f}% of Korg)")

# Calculate totals
alpha_current_sources = (alpha_h_minus_bf + alpha_h_i_bf + alpha_he_i_bf + 
                        alpha_thomson + float(alpha_rayleigh) + alpha_metals)

alpha_all_sources = (alpha_h_minus_bf + alpha_h_minus_ff_approx + alpha_h_i_bf + 
                    alpha_h_i_bf_higher + alpha_he_i_bf + alpha_thomson + 
                    float(alpha_rayleigh) + alpha_metals + alpha_ff_positive)

print()
print("SUMMARY:")
print(f"Current sources:   {alpha_current_sources:.2e} cm⁻¹  ({100*alpha_current_sources/alpha_korg:.1f}% of Korg)")
print(f"All sources:       {alpha_all_sources:.2e} cm⁻¹  ({100*alpha_all_sources/alpha_korg:.1f}% of Korg)")
print(f"Still missing:     {alpha_korg - alpha_all_sources:.2e} cm⁻¹  ({100*(alpha_korg - alpha_all_sources)/alpha_korg:.1f}% of Korg)")
print()

# Identify largest missing components
missing_sources = [
    ("H⁻ ff", alpha_h_minus_ff_approx),
    ("H I higher levels", alpha_h_i_bf_higher),
    ("Positive ion ff", alpha_ff_positive),
    ("Unknown/unmodeled", alpha_korg - alpha_all_sources)
]

print("LARGEST MISSING COMPONENTS:")
for name, value in sorted(missing_sources, key=lambda x: x[1], reverse=True):
    if value > 0:
        print(f"  {name}: {value:.2e} cm⁻¹ ({100*value/alpha_korg:.1f}% of Korg)")

print()
if alpha_h_minus_ff_approx > alpha_korg * 0.1:
    print("🎯 H⁻ free-free is likely the main missing component!")
    print("   Need to implement proper Bell & Berrington 1987 tables")
elif alpha_h_i_bf_higher > alpha_korg * 0.1:
    print("🎯 Higher H I levels (n>1) are significant!")
    print("   Need to include excited state populations")
elif (alpha_korg - alpha_all_sources) > alpha_korg * 0.2:
    print("🔍 Still missing major unidentified continuum source")
    print("   May need to check partition functions or other physics")
else:
    print("✅ Most continuum sources identified!")
    print("   Small remaining difference likely from approximations")