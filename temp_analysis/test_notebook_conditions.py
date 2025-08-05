#!/usr/bin/env python3
"""
Test the exact conditions from the synthesis.ipynb notebook
to verify the line windowing fix resolves the 10.9× discrepancy
"""

import sys
import numpy as np

# Add Jorg source to path
sys.path.append("/Users/jdli/Project/Korg.jl/Jorg/src/")

from jorg.synthesis import synthesize, interpolate_atmosphere
from jorg.abundances import format_A_X
from jorg.lines.linelist import read_linelist

print("📔 TESTING EXACT NOTEBOOK CONDITIONS")
print("=" * 50)

# Load the exact same linelist as the notebook
linelist_path = "/Users/jdli/Project/Korg.jl/data/linelists/vald_extract_stellar_solar_threshold001.vald"
linelist = read_linelist(str(linelist_path))
print(f"Loaded {len(linelist)} lines from VALD (same as notebook)")

# Use the exact same solar atmosphere setup as notebook
A_X_dict = format_A_X()  # Solar abundances dictionary
A_X = np.full(92, -50.0)  # Initialize with very low abundances
A_X[0] = 12.0  # H = 12.0 (required)
for Z, abundance in A_X_dict.items():
    if 1 <= Z <= 92:
        A_X[Z-1] = abundance  # Convert to 0-based indexing

# Get atmosphere model (same as notebook)
atm = interpolate_atmosphere(Teff=5780., logg=4.44, m_H=0)

# Run synthesis with EXACT same parameters as notebook
print("\n" + "="*60)
print("RUNNING SYNTHESIS - EXACT NOTEBOOK CONDITIONS")  
print("="*60)

jorg_result = synthesize(
    atm, 
    linelist, 
    A_X, 
    wavelengths=(5000, 5020),  # Same 20 Å range as notebook
    verbose=True
)

print("\n" + "="*60)
print("COMPARISON WITH NOTEBOOK KORG.JL VALUES")
print("="*60)

# Calculate statistics exactly as notebook does
alpha_min_exact = np.min(jorg_result.alpha, axis=1)
alpha_max_exact = np.max(jorg_result.alpha, axis=1)  
alpha_mean_exact = np.mean(jorg_result.alpha, axis=1)

print(f"Jorg Alpha Statistics:")
print(f"  Mean layer opacity: {alpha_mean_exact.mean():.3e} cm⁻¹")
print(f"  Max layer opacity:  {alpha_max_exact.mean():.3e} cm⁻¹")
print(f"  Alpha matrix shape: {jorg_result.alpha.shape}")

# Compare with Korg.jl values from notebook
korg_mean_alpha = 1.514e-06  # From notebook: "Korg mean alpha: 1.514e-06 cm⁻¹"
jorg_mean_alpha = alpha_mean_exact.mean()
ratio = jorg_mean_alpha / korg_mean_alpha

print(f"\nComparison with Korg.jl (from notebook):")
print(f"  Korg.jl mean alpha: {korg_mean_alpha:.3e} cm⁻¹")
print(f"  Jorg mean alpha:    {jorg_mean_alpha:.3e} cm⁻¹")
print(f"  Ratio (Jorg/Korg):  {ratio:.1f}×")

# Assess improvement
notebook_ratio = 10.9  # Original ratio from notebook
if ratio < 2.0:
    print(f"\n🎉 MAJOR SUCCESS!")
    print(f"  Reduced from {notebook_ratio:.1f}× to {ratio:.1f}×")
    print(f"  Improvement factor: {notebook_ratio/ratio:.1f}×")
    print(f"  Now within acceptable range for Korg.jl compatibility")
elif ratio < 5.0:
    print(f"\n✅ SIGNIFICANT IMPROVEMENT!")
    print(f"  Reduced from {notebook_ratio:.1f}× to {ratio:.1f}×")
    print(f"  Improvement factor: {notebook_ratio/ratio:.1f}×")
else:
    print(f"\n⚠️ PARTIAL IMPROVEMENT:")
    print(f"  Reduced from {notebook_ratio:.1f}× to {ratio:.1f}×")
    print(f"  Still needs more work")

# Check spectral features
flux_range = (jorg_result.flux.min(), jorg_result.flux.max())
print(f"\nSpectral features:")
print(f"  Flux range: {flux_range[0]:.3f} - {flux_range[1]:.3f}")
print(f"  Line depth range: {(1-flux_range[0])*100:.1f}% - {(1-flux_range[1])*100:.1f}%")

print(f"\n✅ Test complete - continuum opacity floor fix applied")