#!/usr/bin/env python3
"""
Test Voigt profile to ensure it's always positive
"""

import sys
import numpy as np
sys.path.insert(0, 'src')

from jorg.opacity.korg_line_processor import KorgLineProcessor

print("🧪 TESTING VOIGT PROFILE SIGN")
print("=" * 40)

processor = KorgLineProcessor()

# Test parameters
test_cases = [
    {"alpha": 0.1, "v": 0.0, "name": "Line center, small damping"},
    {"alpha": 0.1, "v": 1.0, "name": "Near line, small damping"},
    {"alpha": 0.1, "v": 5.0, "name": "Far from line, small damping"},
    {"alpha": 1.0, "v": 0.0, "name": "Line center, medium damping"},
    {"alpha": 1.0, "v": 10.0, "name": "Far from line, medium damping"},
    {"alpha": 5.0, "v": 0.0, "name": "Line center, large damping"},
]

print("Testing Voigt-Hjerting function:")
for case in test_cases:
    alpha = case["alpha"]
    v = case["v"]
    result = processor._voigt_hjerting(alpha, v)
    print(f"{case['name']:35} α={alpha:3.1f}, v={v:4.1f} → H={result:10.6e}")
    if result < 0:
        print("  ❌ NEGATIVE VALUE DETECTED!")

# Test full line profile
print("\n" + "=" * 40)
print("Testing full line profile calculation:")

# Line parameters
lambda_0 = 5000e-8  # 5000 Å in cm
sigma = 1e-10       # Doppler width in cm
gamma = 5e-11       # Lorentz width in cm  
amplitude = 1e-15   # Line strength

# Test wavelengths around line center
test_wavelengths = np.linspace(lambda_0 - 10*sigma, lambda_0 + 10*sigma, 21)

print(f"\nLine parameters:")
print(f"  λ₀ = {lambda_0*1e8:.1f} Å")
print(f"  σ = {sigma*1e8:.3f} mÅ")
print(f"  γ = {gamma*1e8:.3f} mÅ")
print(f"  amplitude = {amplitude:.2e}")

print("\nProfile values:")
for i, wl in enumerate(test_wavelengths):
    profile = processor._line_profile(lambda_0, sigma, gamma, amplitude, wl)
    offset = (wl - lambda_0) * 1e8  # in mÅ
    print(f"  Δλ = {offset:6.2f} mÅ: {profile:10.3e} cm⁻¹", end="")
    if profile < 0:
        print(" ❌ NEGATIVE!")
    else:
        print()

# Test with very large offsets
print("\n" + "=" * 40)
print("Testing extreme wavelength offsets:")

extreme_offsets = [1e-7, 1e-6, 1e-5, 1e-4]  # cm
for offset in extreme_offsets:
    wl = lambda_0 + offset
    profile = processor._line_profile(lambda_0, sigma, gamma, amplitude, wl)
    print(f"  Δλ = {offset*1e8:.0f} Å: {profile:10.3e} cm⁻¹", end="")
    if profile < 0:
        print(" ❌ NEGATIVE!")
    else:
        print()

print("\n✅ Test complete")