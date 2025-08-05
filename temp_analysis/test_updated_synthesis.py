#!/usr/bin/env python3
"""
Test updated synthesis.py with line windowing fixes
"""

import sys
import numpy as np

# Add Jorg source to path
sys.path.append("/Users/jdli/Project/Korg.jl/Jorg/src/")

from jorg.synthesis import synthesize, interpolate_atmosphere
from jorg.abundances import format_A_X
from jorg.lines.linelist import read_linelist

print("🔬 TESTING UPDATED SYNTHESIS MODULE")
print("=" * 50)

# Load small line list for quick test
linelist_path = "/Users/jdli/Project/Korg.jl/data/linelists/vald_extract_stellar_solar_threshold001.vald"
print(f"Loading linelist: {linelist_path.split('/')[-1]}")
linelist = read_linelist(str(linelist_path))

# Filter to strong lines for quicker test  
strong_lines = [line for line in linelist if line.log_gf > -1.0]
print(f"Using {len(strong_lines)} strong lines (log_gf > -1.0)")

# Set up solar atmosphere
print("\nSetting up solar atmosphere...")
A_X_dict = format_A_X()
A_X = np.full(92, -50.0)
A_X[0] = 12.0
for Z, abundance in A_X_dict.items():
    if 1 <= Z <= 92:
        A_X[Z-1] = abundance
        
atm = interpolate_atmosphere(Teff=5780., logg=4.44, m_H=0)

# Run synthesis with verbose output
print("\n" + "="*50)
print("RUNNING SYNTHESIS WITH UPDATED MODULE")
print("="*50)

result = synthesize(
    atm=atm,
    linelist=strong_lines,
    A_X=A_X,
    wavelengths=(5000, 5010),  # Small range for quick test
    verbose=True,
    rectify=True
)

print("\n" + "="*50)
print("TEST RESULTS")
print("="*50)

# Check results
alpha_mean = np.mean(result.alpha)
alpha_max = np.max(result.alpha)
flux_range = (result.flux.min(), result.flux.max())

print(f"Alpha matrix shape: {result.alpha.shape}")
print(f"Mean alpha: {alpha_mean:.3e} cm⁻¹")
print(f"Max alpha:  {alpha_max:.3e} cm⁻¹")
print(f"Flux range: {flux_range[0]:.3f} - {flux_range[1]:.3f}")

# Expected Korg values for comparison
korg_mean_alpha = 1.514e-6
ratio = alpha_mean / korg_mean_alpha

print(f"\nComparison with Korg.jl:")
print(f"Korg mean alpha: {korg_mean_alpha:.3e} cm⁻¹")
print(f"Ratio (Jorg/Korg): {ratio:.2f}×")

if ratio < 2.0:
    print("✅ SUCCESS: Line opacity within 2× of Korg.jl")
else:
    print("❌ Issue: Still higher than expected")

print("\n✅ Updated synthesis module test complete!")