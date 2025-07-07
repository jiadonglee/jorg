#!/usr/bin/env python3
"""
Debug opacity scaling issue by comparing manual calculations
"""

import numpy as np

# Constants (CGS)
THOMSON_CROSS_SECTION = 6.652e-25  # cm²
AMU_CGS = 1.6605402e-24  # g

# Atmospheric conditions (from Korg)
T = 4838.3  # K
ne = 2.28e12  # cm⁻³ electron density 
nt = 2.89e16  # cm⁻³ total number density

# Fixed species densities
n_HI = 2.5e16  # cm⁻³
n_HII = 6.0e10  # cm⁻³  
n_HeI = 2.0e15  # cm⁻³
n_HeII = 1.0e11  # cm⁻³
n_H2 = 1.0e13  # cm⁻³

print("MANUAL OPACITY CALCULATION DEBUG")
print("=" * 40)

print(f"Temperature: {T:.1f} K")
print(f"Electron density: {ne:.2e} cm⁻³")
print(f"Total density: {nt:.2e} cm⁻³")
print()

# Calculate Thomson scattering manually
print("THOMSON SCATTERING:")
alpha_thomson = ne * THOMSON_CROSS_SECTION  # cm⁻¹

# Convert to opacity per unit mass
mean_molecular_weight = 1.3  # AMU
mass_density = nt * mean_molecular_weight * AMU_CGS  # g/cm³
opacity_thomson = alpha_thomson / mass_density  # cm²/g

print(f"  Absorption coeff: {alpha_thomson:.2e} cm⁻¹")
print(f"  Mass density: {mass_density:.2e} g/cm³")
print(f"  Opacity per mass: {opacity_thomson:.2e} cm²/g")
print()

# Compare to Korg result
korg_opacity = 3.5e-9  # cm²/g from Korg
print(f"COMPARISON:")
print(f"  Korg total opacity: {korg_opacity:.2e} cm²/g")
print(f"  Thomson only: {opacity_thomson:.2e} cm²/g")
print(f"  Ratio (Thomson/Korg): {opacity_thomson/korg_opacity:.1f}")
print()

# This shows Thomson scattering alone is ~10,000x larger than Korg's total!
# This means either:
# 1. Korg uses different units
# 2. Korg's mass density calculation is different  
# 3. There's a systematic error in the calculation

print("ANALYSIS:")
if opacity_thomson > 1000 * korg_opacity:
    print("❌ PROBLEM: Thomson scattering alone is much larger than Korg total!")
    print("   This suggests a fundamental scaling error.")
    print("   Need to investigate Korg's units and mass density calculation.")
else:
    print("✅ Thomson scattering is reasonable compared to Korg total")
    
print()
print("POSSIBLE ISSUES:")
print("1. Different mass density calculation in Korg")
print("2. Different units (per unit volume vs per unit mass)")
print("3. Different definition of opacity")
print("4. Error in Thomson cross-section or constants")