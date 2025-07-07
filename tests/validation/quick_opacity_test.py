#!/usr/bin/env python3
"""
Quick opacity test to check if scaling is now correct
"""

import sys
from pathlib import Path
jorg_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(jorg_src))

import numpy as np
from jorg.statmech.species import Species, Formula
from jorg.continuum.complete_continuum import calculate_total_continuum_opacity

print("QUICK OPACITY SCALING TEST")
print("=" * 30)

# Test conditions (from Korg test)
T = 4838.3  # K
ne = 2.28e12  # cm⁻³ electron density
frequencies = np.array([5.451e14])  # Single frequency at 5500 Å

# Fixed species densities (same as Korg test)
number_densities = {}
number_densities[Species(Formula([1]), 0)] = 2.5e16    # H I
number_densities[Species(Formula([1]), 1)] = 6.0e10    # H II
number_densities[Species(Formula([2]), 0)] = 2.0e15    # He I
number_densities[Species(Formula([2]), 1)] = 1.0e11    # He II
number_densities[Species(Formula([26]), 0)] = 9.0e10   # Fe I
number_densities[Species(Formula([26]), 1)] = 3.0e10   # Fe II
number_densities[Species(Formula([1, 1]), 0)] = 1.0e13  # H2

print(f"Temperature: {T} K")
print(f"Electron density: {ne:.2e} cm⁻³")
print(f"H I density: {number_densities[Species(Formula([1]), 0)]:.2e} cm⁻³")
print()

try:
    # Calculate Jorg absorption coefficient
    α_jorg = calculate_total_continuum_opacity(frequencies, T, ne, number_densities)
    
    # Convert to opacity per mass
    total_density = sum(number_densities.values())
    mean_molecular_weight = 1.4  # AMU
    mass_density = total_density * mean_molecular_weight * 1.66054e-24  # g/cm³
    opacity_jorg = α_jorg[0] / mass_density
    
    print(f"Jorg absorption coefficient: {α_jorg[0]:.2e} cm⁻¹")
    print(f"Jorg opacity per mass: {opacity_jorg:.2e} cm²/g")
    print(f"Mass density: {mass_density:.2e} g/cm³")
    print()
    
    # Expected results from Korg
    korg_absorption_expected = 1.7e-16  # From previous test
    korg_opacity_expected = 3.5e-9  # cm²/g
    
    print(f"Expected Korg absorption: {korg_absorption_expected:.2e} cm⁻¹")
    print(f"Expected Korg opacity: {korg_opacity_expected:.2e} cm²/g")
    print()
    
    # Compare
    absorption_ratio = α_jorg[0] / korg_absorption_expected if korg_absorption_expected > 0 else float('inf')
    opacity_ratio = opacity_jorg / korg_opacity_expected
    
    print(f"Absorption ratio (Jorg/Korg): {absorption_ratio:.2f}")
    print(f"Opacity ratio (Jorg/Korg): {opacity_ratio:.2f}")
    print()
    
    if 0.1 < opacity_ratio < 10:
        print("✅ GOOD: Opacity scaling is reasonable!")
    elif 0.01 < opacity_ratio < 100:
        print("⚠️  MODERATE: Opacity scaling needs adjustment but in right ballpark")
    else:
        print("❌ BAD: Opacity scaling still wrong by orders of magnitude")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()