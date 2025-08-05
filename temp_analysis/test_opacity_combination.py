#!/usr/bin/env python3
"""
Test how continuum and line opacity are combined
"""

import sys
import numpy as np
sys.path.insert(0, 'src')

from jorg.opacity.layer_processor import LayerProcessor
from jorg.atmosphere import interpolate_marcs
from jorg.lines.linelist import read_linelist

print("🔍 TESTING OPACITY COMBINATION")
print("=" * 50)

# Create physics data first
from jorg.statmech import (
    create_default_ionization_energies,
    create_default_partition_functions,
    create_default_log_equilibrium_constants
)

ionization_energies = create_default_ionization_energies()
partition_funcs = create_default_partition_functions()
log_equilibrium_constants = create_default_log_equilibrium_constants()

# Create layer processor
processor = LayerProcessor(
    ionization_energies=ionization_energies,
    partition_funcs=partition_funcs,
    log_equilibrium_constants=log_equilibrium_constants,
    verbose=True
)

# Test atmosphere
atm = interpolate_marcs(Teff=5780, logg=4.44, m_H=0.0)
test_layer = 30  # Middle layer

# Extract layer conditions
T = atm.layers[test_layer].temp
ne = atm.layers[test_layer].electron_number_density
nt = atm.layers[test_layer].number_density

print(f"\nTest layer {test_layer}:")
print(f"  T = {T:.1f} K")
print(f"  ne = {ne:.2e} cm⁻³")
print(f"  nt = {nt:.2e} cm⁻³")

# Test wavelengths
wl_array = np.linspace(5000, 5001, 101)  # Å

# Create simple abundances
abs_abundances = {}
A_X = np.array([12.0] + [7.5]*91)  # Solar-like
for Z in range(1, 93):
    abs_abundances[Z] = 10**(A_X[Z-1] - 12) / np.sum(10**(A_X - 12))

# Calculate chemical equilibrium for this layer
from jorg.statmech import chemical_equilibrium_working_optimized as chemical_equilibrium

print("\nCalculating chemical equilibrium...")
ne_solution, number_densities = chemical_equilibrium(
    T, ne, nt, abs_abundances, ionization_energies
)

print(f"Chemical equilibrium: {len(number_densities)} species")

# 1. Test continuum only
print("\n" + "-"*50)
print("TEST 1: Continuum opacity only")

continuum_opacity = processor._calculate_continuum_opacity(
    wl_array, T, ne_solution, number_densities
)

print(f"Continuum opacity stats:")
print(f"  Min: {np.min(continuum_opacity):.3e} cm⁻¹")
print(f"  Max: {np.max(continuum_opacity):.3e} cm⁻¹")
print(f"  Mean: {np.mean(continuum_opacity):.3e} cm⁻¹")
print(f"  Negative values: {np.sum(continuum_opacity < 0)}")

# 2. Test with a single line
print("\n" + "-"*50)
print("TEST 2: Single line opacity")

# Load VALD and get one line
vald_path = "/Users/jdli/Project/Korg.jl/data/linelists/vald_extract_stellar_solar_threshold001.vald"
linelist = read_linelist(vald_path, format='vald')

# Get one line in our range
wl_min = 5000 * 1e-8
wl_max = 5001 * 1e-8
test_lines = [line for line in linelist if wl_min <= line.wavelength <= wl_max][:1]

if test_lines:
    print(f"Test line: λ={test_lines[0].wavelength*1e8:.3f} Å")
    
    line_opacity = processor._calculate_line_opacity(
        wl_array, T, ne_solution, number_densities,
        test_lines, line_buffer=1.0, hydrogen_lines=False,
        vmic=1.0, log_g=4.44, continuum_opacity=continuum_opacity
    )
    
    print(f"\nLine opacity stats:")
    print(f"  Min: {np.min(line_opacity):.3e} cm⁻¹")
    print(f"  Max: {np.max(line_opacity):.3e} cm⁻¹")
    print(f"  Mean: {np.mean(line_opacity):.3e} cm⁻¹")
    print(f"  Negative values: {np.sum(line_opacity < 0)}")
    
    # 3. Test total opacity
    print("\n" + "-"*50)
    print("TEST 3: Total opacity (continuum + line)")
    
    total_opacity = continuum_opacity + line_opacity
    
    print(f"Total opacity stats:")
    print(f"  Min: {np.min(total_opacity):.3e} cm⁻¹")
    print(f"  Max: {np.max(total_opacity):.3e} cm⁻¹")
    print(f"  Mean: {np.mean(total_opacity):.3e} cm⁻¹")
    print(f"  Negative values: {np.sum(total_opacity < 0)}")
    
    # Check where negative values occur
    if np.any(total_opacity < 0):
        neg_idx = np.where(total_opacity < 0)[0]
        print(f"\n❌ NEGATIVE TOTAL OPACITY at {len(neg_idx)} points!")
        for idx in neg_idx[:5]:
            print(f"  λ={wl_array[idx]:.3f} Å:")
            print(f"    continuum={continuum_opacity[idx]:.3e}")
            print(f"    line={line_opacity[idx]:.3e}")
            print(f"    total={total_opacity[idx]:.3e}")

print("\n✅ Test complete")