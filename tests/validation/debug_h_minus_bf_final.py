#!/usr/bin/env python3
"""
Final debug of H⁻ bound-free - the main missing component
"""

import sys
from pathlib import Path
jorg_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(jorg_src))

import numpy as np
import jax.numpy as jnp
from jorg.continuum.complete_continuum import h_minus_bf_cross_section

print("H⁻ BOUND-FREE FINAL DEBUG")
print("=" * 29)

# Test conditions
T = 4838.3  # K
ne = 2.28e12  # cm⁻³
n_HI = 2.5e16  # cm⁻³
frequency = 5.451e14  # Hz (5500 Å)

# Korg reference values
korg_h_minus_bf = 3.384e-9  # cm⁻¹
korg_total = 3.527e-9  # cm⁻¹

print(f"Test conditions:")
print(f"  Temperature: {T} K")
print(f"  Electron density: {ne:.2e} cm⁻³")
print(f"  H I density: {n_HI:.2e} cm⁻³")
print(f"  Frequency: {frequency:.2e} Hz (5500 Å)")
print()
print(f"Korg reference:")
print(f"  H⁻ bf: {korg_h_minus_bf:.2e} cm⁻¹ (96% of total)")
print(f"  Total: {korg_total:.2e} cm⁻¹")
print()

# Calculate our H⁻ density (Korg exact formula)
h_minus_binding_eV = 0.754204
kboltz_eV = 8.617333262145e-5
coef = 3.31283018e-22
beta = 1.0 / (kboltz_eV * T)

# Ground state H I density: 2 * nH_I_div_partition
# With partition function = 2: nH_I_div_partition = nH_I / 2
nH_I_div_partition = n_HI / 2.0
nHI_groundstate = 2 * nH_I_div_partition  # = n_HI

# H⁻ density
h_minus_density = 0.25 * nHI_groundstate * ne * coef * (beta**1.5) * np.exp(h_minus_binding_eV * beta)

print(f"H⁻ DENSITY CALCULATION:")
print(f"  nH_I_div_partition: {nH_I_div_partition:.2e} cm⁻³")
print(f"  nHI_groundstate: {nHI_groundstate:.2e} cm⁻³")
print(f"  H⁻ density: {h_minus_density:.2e} cm⁻³")
print()

# Calculate our H⁻ bf cross-section
sigma_h_minus_bf = h_minus_bf_cross_section(frequency)
print(f"H⁻ BF CROSS-SECTION:")
print(f"  Our σ(H⁻ bf): {float(sigma_h_minus_bf):.2e} cm²")

# Stimulated emission factor
stim_factor = 1.0 - np.exp(-6.626e-27 * frequency / (1.381e-16 * T))
print(f"  Stimulated emission factor: {stim_factor:.6f}")

# Our H⁻ bf absorption coefficient
alpha_our_h_minus_bf = h_minus_density * float(sigma_h_minus_bf) * stim_factor
print(f"  Our α(H⁻ bf): {alpha_our_h_minus_bf:.2e} cm⁻¹")
print()

# Comparison
print(f"COMPARISON:")
print(f"  Korg H⁻ bf: {korg_h_minus_bf:.2e} cm⁻¹")
print(f"  Our H⁻ bf:  {alpha_our_h_minus_bf:.2e} cm⁻¹")
print(f"  Ratio (Korg/Our): {korg_h_minus_bf / alpha_our_h_minus_bf:.1f}")
print()

# Determine the issue
ratio = korg_h_minus_bf / alpha_our_h_minus_bf
if ratio > 10:
    print(f"🔥 MAJOR ISSUE: Factor {ratio:.0f} too small!")
    print("   Likely causes:")
    print("   1. H⁻ cross-section too small")
    print("   2. H⁻ density calculation error")
    print("   3. Missing physics or units conversion")
elif ratio > 3:
    print(f"⚠️  SIGNIFICANT ISSUE: Factor {ratio:.1f} too small")
    print("   Need to check cross-section implementation")
elif ratio > 1.5:
    print(f"✅ CLOSE: Factor {ratio:.1f} difference")
    print("   Minor correction needed")
else:
    print("🎉 EXCELLENT AGREEMENT!")

# Check what cross-section we'd need
needed_sigma = korg_h_minus_bf / (h_minus_density * stim_factor)
print()
print(f"ANALYSIS:")
print(f"  Needed σ for perfect agreement: {needed_sigma:.2e} cm²")
print(f"  Our current σ: {float(sigma_h_minus_bf):.2e} cm²")
print(f"  Cross-section ratio needed: {needed_sigma / float(sigma_h_minus_bf):.1f}")

if needed_sigma / float(sigma_h_minus_bf) > 5:
    print("   🔧 Cross-section is the main issue")
    print("   Check McLaughlin+ 2017 implementation")
elif h_minus_density < 1e6:
    print("   🔧 H⁻ density might be too small")
    print("   Check Saha equation implementation")
else:
    print("   🔧 Complex issue - check all components")

# Test if this explains our total opacity discrepancy
if abs(korg_h_minus_bf - alpha_our_h_minus_bf) > 1e-9:
    missing_opacity = korg_h_minus_bf - alpha_our_h_minus_bf
    our_total_estimated = 3.31e-10 + missing_opacity  # Our current + missing H⁻ bf
    print()
    print(f"TOTAL OPACITY PROJECTION:")
    print(f"  Our current total: 3.31e-10 cm⁻¹")
    print(f"  Missing H⁻ bf: {missing_opacity:.2e} cm⁻¹")
    print(f"  Projected total: {our_total_estimated:.2e} cm⁻¹")
    print(f"  Korg total: {korg_total:.2e} cm⁻¹")
    print(f"  Projected agreement: {100 * our_total_estimated / korg_total:.1f}%")
    
    if our_total_estimated / korg_total > 0.9:
        print("   🎯 PERFECT! Fixing H⁻ bf will achieve excellent agreement!")
    elif our_total_estimated / korg_total > 0.8:
        print("   ✅ VERY GOOD! Fixing H⁻ bf will achieve good agreement!")
    else:
        print("   ⚠️  More work needed beyond H⁻ bf")