#!/usr/bin/env python3
"""
Test the improved line windowing fix with continuum opacity floor
"""

import sys
import numpy as np

# Add Jorg source to path
sys.path.append("/Users/jdli/Project/Korg.jl/Jorg/src/")

from jorg.synthesis import synthesize, interpolate_atmosphere
from jorg.abundances import format_A_X
from jorg.lines.linelist import read_linelist

print("🔧 TESTING IMPROVED LINE WINDOWING FIX")
print("=" * 50)

# Load linelist (use smaller subset for testing)
linelist_path = "/Users/jdli/Project/Korg.jl/data/linelists/vald_extract_stellar_solar_threshold001.vald"
linelist = read_linelist(str(linelist_path))

# Filter to reasonable number of lines for testing
test_lines = [line for line in linelist if -1.0 <= line.log_gf <= 0.0]
print(f"Using {len(test_lines)} test lines (-1.0 <= log_gf <= 0.0)")

# Set up solar atmosphere
A_X_dict = format_A_X()
A_X = np.full(92, -50.0)
A_X[0] = 12.0
for Z, abundance in A_X_dict.items():
    if 1 <= Z <= 92:
        A_X[Z-1] = abundance

atm = interpolate_atmosphere(Teff=5780., logg=4.44, m_H=0)

# Run synthesis with verbose output to see windowing statistics
print("\n" + "="*50)
print("RUNNING SYNTHESIS WITH CONTINUUM OPACITY FLOOR")
print("="*50)

result = synthesize(
    atm=atm,
    linelist=test_lines,
    A_X=A_X,
    wavelengths=(5000, 5010),  # Small range for quick test
    verbose=True,
    rectify=True
)

print("\n" + "="*50)
print("RESULTS ANALYSIS")
print("="*50)

# Analyze results
alpha_mean = np.mean(result.alpha)
alpha_max = np.max(result.alpha)
line_depth_range = (result.flux.min(), result.flux.max())

print(f"Alpha matrix shape: {result.alpha.shape}")
print(f"Mean alpha: {alpha_mean:.3e} cm⁻¹")
print(f"Max alpha:  {alpha_max:.3e} cm⁻¹")
print(f"Flux range: {line_depth_range[0]:.3f} - {line_depth_range[1]:.3f}")

# Compare with Korg.jl reference
korg_mean_alpha = 1.514e-6
ratio = alpha_mean / korg_mean_alpha

print(f"\nComparison with Korg.jl:")
print(f"Korg mean alpha: {korg_mean_alpha:.3e} cm⁻¹")
print(f"Ratio (Jorg/Korg): {ratio:.2f}×")

if ratio < 2.0:
    print("✅ SUCCESS: Line opacity within 2× of Korg.jl")
    if ratio < 1.5:
        print("🎉 EXCELLENT: Line opacity within 1.5× of Korg.jl")
elif ratio < 5.0:
    print("⚠️  IMPROVEMENT: Reduced overestimate")
else:
    print("❌ Still problematic: Need further fixes")

print(f"\n✅ Test complete!")