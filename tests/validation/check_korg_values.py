#!/usr/bin/env python3
"""
Check the actual Korg values we should be comparing against
"""

print("KORG VALUE VERIFICATION")
print("=" * 25)

# From the Julia output above
korg_absorption_actual = 3.4991477662292695e-9  # cm⁻¹
jorg_absorption = 1.07e-7  # cm⁻¹ from manual calculation

print(f"Korg absorption coefficient: {korg_absorption_actual:.2e} cm⁻¹")
print(f"Jorg absorption coefficient: {jorg_absorption:.2e} cm⁻¹")
print(f"Ratio (Jorg/Korg): {jorg_absorption / korg_absorption_actual:.1f}")
print()

# This is much more reasonable! ~30x difference instead of 10^8

# Let's break down my calculation vs expected
print("COMPONENT ANALYSIS:")
print("=" * 20)

thomson_contrib = 1.52e-12  # cm⁻¹
rayleigh_contrib = 9.90e-08  # cm⁻¹ 
h_minus_contrib = 8.31e-09   # cm⁻¹

print(f"Thomson scattering:  {thomson_contrib:.2e} cm⁻¹")
print(f"Rayleigh scattering: {rayleigh_contrib:.2e} cm⁻¹")
print(f"H⁻ bound-free:      {h_minus_contrib:.2e} cm⁻¹")
print()

total_jorg = thomson_contrib + rayleigh_contrib + h_minus_contrib
print(f"Total Jorg:     {total_jorg:.2e} cm⁻¹")
print(f"Total Korg:     {korg_absorption_actual:.2e} cm⁻¹")
print(f"Difference:     {(total_jorg - korg_absorption_actual) / korg_absorption_actual * 100:.1f}%")
print()

if abs(total_jorg / korg_absorption_actual - 1) < 0.5:
    print("✅ EXCELLENT: Jorg and Korg agree within 50%!")
elif abs(total_jorg / korg_absorption_actual - 1) < 2.0:
    print("✅ GOOD: Jorg and Korg agree within factor of 2")
elif abs(total_jorg / korg_absorption_actual - 1) < 10.0:
    print("⚠️  REASONABLE: Jorg and Korg agree within order of magnitude")
else:
    print("❌ POOR: Jorg and Korg disagree significantly")

print()
print("CONCLUSION: The opacity comparison was using wrong expected values!")
print("Need to update test with correct Korg absorption coefficient values.")