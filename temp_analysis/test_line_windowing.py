#!/usr/bin/env python3
"""
Test script to verify line windowing fixes for Jorg line opacity

This script tests the improvements to line opacity calculation,
specifically the proper application of cutoff thresholds to reduce
the ~11× overestimate compared to Korg.jl.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add Jorg source to path
sys.path.append("/Users/jdli/Project/Korg.jl/Jorg/src/")

# Import Jorg modules
from jorg.synthesis import synthesize, interpolate_atmosphere
from jorg.abundances import format_A_X
from jorg.lines.linelist import read_linelist

print("=" * 70)
print("JORG LINE WINDOWING TEST")
print("Testing fixes for 11× line opacity overestimate")
print("=" * 70)

# Load VALD line list
linelist_path = "/Users/jdli/Project/Korg.jl/data/linelists/vald_extract_stellar_solar_threshold001.vald"
print(f"\n1. Loading linelist: {Path(linelist_path).name}")
linelist = read_linelist(str(linelist_path))
print(f"   Total lines loaded: {len(linelist)}")

# Filter to strong lines only for comparison
strong_lines = [line for line in linelist if line.log_gf > -2.0]
print(f"   Strong lines (log_gf > -2.0): {len(strong_lines)}")

# Set up solar atmosphere
print("\n2. Setting up solar atmosphere")
A_X_dict = format_A_X()  # Solar abundances dictionary
A_X = np.full(92, -50.0)  # Initialize with very low abundances
A_X[0] = 12.0  # H = 12.0 (required)
for Z, abundance in A_X_dict.items():
    if 1 <= Z <= 92:
        A_X[Z-1] = abundance  # Convert to 0-based indexing
        
atm = interpolate_atmosphere(Teff=5780., logg=4.44, m_H=0)
print(f"   Teff = 5780 K, log g = 4.44, [M/H] = 0.0")

# Test wavelength range
wavelength_range = (5000, 5020)
print(f"\n3. Synthesis wavelength range: {wavelength_range[0]}-{wavelength_range[1]} Å")

# Run synthesis with full linelist (verbose mode)
print("\n4. Running synthesis with FULL linelist (verbose=True)")
print("-" * 50)
result_full = synthesize(
    atm, linelist, A_X, 
    wavelengths=wavelength_range,
    verbose=True,
    vmic=1.0
)
print("-" * 50)

# Extract statistics
alpha_mean_full = np.mean(result_full.alpha, axis=1)
alpha_max_full = np.max(result_full.alpha, axis=1)
print(f"\nFull linelist results:")
print(f"   Mean alpha: {np.mean(alpha_mean_full):.3e} cm⁻¹")
print(f"   Max alpha:  {np.mean(alpha_max_full):.3e} cm⁻¹")

# Run synthesis with strong lines only
print("\n5. Running synthesis with STRONG lines only")
print("-" * 50)
result_strong = synthesize(
    atm, strong_lines, A_X,
    wavelengths=wavelength_range,
    verbose=True,
    vmic=1.0
)
print("-" * 50)

# Extract statistics
alpha_mean_strong = np.mean(result_strong.alpha, axis=1)
alpha_max_strong = np.max(result_strong.alpha, axis=1)
print(f"\nStrong lines only results:")
print(f"   Mean alpha: {np.mean(alpha_mean_strong):.3e} cm⁻¹")
print(f"   Max alpha:  {np.mean(alpha_max_strong):.3e} cm⁻¹")

# Compare with expected Korg.jl values
print("\n6. Comparison with Korg.jl reference")
print("=" * 50)
korg_mean_alpha = 1.514e-6  # From notebook
reduction_factor = np.mean(alpha_mean_full) / korg_mean_alpha

print(f"Korg.jl mean alpha: {korg_mean_alpha:.3e} cm⁻¹")
print(f"Jorg mean alpha:    {np.mean(alpha_mean_full):.3e} cm⁻¹")
print(f"Ratio (Jorg/Korg):  {reduction_factor:.1f}×")

if reduction_factor < 2.0:
    print("\n✅ SUCCESS! Line opacity is now within 2× of Korg.jl")
elif reduction_factor < 5.0:
    print("\n⚠️  PARTIAL SUCCESS: Reduced from 11× to {:.1f}×".format(reduction_factor))
else:
    print("\n❌ ISSUE REMAINS: Still {:.1f}× too high".format(reduction_factor))

# Calculate effective line density
wl_range = wavelength_range[1] - wavelength_range[0]
line_density_full = len(linelist) / wl_range
print(f"\n7. Line density analysis:")
print(f"   Input line density: {line_density_full:.1f} lines/Å")
print(f"   Expected density:   ~10-20 lines/Å (solar photosphere)")

# Save results for further analysis
print("\n8. Saving results...")
np.savez("line_windowing_test_results.npz",
         wavelengths=result_full.wavelengths,
         flux_full=result_full.flux,
         flux_strong=result_strong.flux,
         alpha_full=result_full.alpha,
         alpha_strong=result_strong.alpha)
print("   Results saved to: line_windowing_test_results.npz")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)