#!/usr/bin/env python3
"""
Debug Thomson scattering contribution specifically
"""

import sys
from pathlib import Path
jorg_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(jorg_src))

import numpy as np
import jax.numpy as jnp
from jorg.continuum.complete_continuum import thomson_scattering_cross_section

print("THOMSON SCATTERING DEBUG")
print("=" * 25)

# Test conditions
ne = 2.28e12  # cm⁻³ electron density

# Calculate Thomson cross-section
sigma_thomson = thomson_scattering_cross_section()
print(f"Thomson cross-section: {sigma_thomson:.2e} cm²")

# Standard value
sigma_thomson_standard = 6.652e-25  # cm²
print(f"Standard Thomson cross-section: {sigma_thomson_standard:.2e} cm²")
print(f"Ratio: {float(sigma_thomson) / sigma_thomson_standard:.6f}")
print()

# Calculate absorption coefficient
alpha_thomson = ne * float(sigma_thomson)  # cm⁻¹
alpha_thomson_standard = ne * sigma_thomson_standard  # cm⁻¹

print(f"Thomson absorption coefficient:")
print(f"  Jorg: {alpha_thomson:.2e} cm⁻¹")
print(f"  Standard: {alpha_thomson_standard:.2e} cm⁻¹")
print(f"  Ratio: {alpha_thomson / alpha_thomson_standard:.6f}")
print()

# Expected from manual calculation
alpha_expected = 1.52e-12  # cm⁻¹ from earlier calculation
print(f"Expected (manual): {alpha_expected:.2e} cm⁻¹")
print(f"Jorg/Expected: {alpha_thomson / alpha_expected:.2e}")

if abs(alpha_thomson / alpha_expected - 1) < 0.1:
    print("✅ Thomson scattering calculation is correct")
else:
    print("❌ Thomson scattering calculation has errors")
    
print()
print("This suggests the error is not in Thomson scattering alone...")
print("Need to check other continuum sources!")