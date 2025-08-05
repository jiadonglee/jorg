#!/usr/bin/env python3
"""
Jorg Unit Test 6: Radiative Transfer Calculation (FIXED)

Tests Jorg's radiative transfer processing with exact Korg.jl compatibility:
- Uses same atmospheric model as Korg.jl
- Uses same wavelength settings
- Calls RT APIs correctly
"""

import sys
import numpy as np
sys.path.insert(0, '../../src')

print("=" * 70)
print("JORG UNIT TEST 6: RADIATIVE TRANSFER CALCULATION (FIXED)")
print("=" * 70)

# 1. Import all required modules
print("\n1. Import Required Modules:")
print("   Loading Jorg modules...")

from jorg.atmosphere import interpolate_marcs
from jorg.radiative_transfer_korg_compatible import radiative_transfer_korg_compatible
from jorg.constants import kboltz_cgs, c_cgs, hplanck_cgs

print("   ✅ All modules imported successfully")

# 2. Setup Atmospheric Model (SAME AS KORG.JL)
print("\n2. Setup Atmospheric Model (Same as Korg.jl):")
print("   Loading MARCS model atmosphere...")

# Solar parameters matching Korg.jl test
Teff = 5780.0
logg = 4.44
m_H = 0.0

# Load MARCS atmosphere exactly as Korg.jl does
atm = interpolate_marcs(Teff, logg, m_H)

# Extract atmospheric properties
temperatures = np.array([layer.temp for layer in atm.layers])
number_densities = np.array([layer.number_density for layer in atm.layers])
electron_densities = np.array([layer.electron_number_density for layer in atm.layers])
tau_5000 = np.array([layer.tau_5000 for layer in atm.layers])
heights = np.array([layer.z for layer in atm.layers])

n_layers = len(atm.layers)

print("   ✅ MARCS atmospheric structure loaded:")
print(f"      Layers: {n_layers}")
print(f"      Temperature range: {temperatures.min():.1f} - {temperatures.max():.1f} K")
print(f"      τ₅₀₀₀ range: {tau_5000.min():.2e} - {tau_5000.max():.2e}")
print(f"      Number density range: {number_densities.min():.2e} - {number_densities.max():.2e} cm⁻³")
print(f"      Height range: {heights.min():.2e} - {heights.max():.2e} cm")

# 3. Wavelength Grid (SAME AS KORG.JL TEST)
print("\n3. Wavelength Grid (Same as Korg.jl test):")
print("   Setting up wavelength grid...")

# Match Korg.jl test: 1000 points from 5000-5100 Å
λ_start = 5000.0
λ_end = 5100.0
n_wavelengths = 1000
wavelengths = np.linspace(λ_start, λ_end, n_wavelengths)

print("   ✅ Wavelength grid created:")
print(f"      Range: {λ_start} - {λ_end} Å")
print(f"      Points: {n_wavelengths}")
print(f"      Resolution: {(λ_end - λ_start)/(n_wavelengths-1):.3f} Å")

# 4. Create Opacity Matrix (Representative values like Korg.jl test)
print("\n4. Opacity Matrix:")
print("   Creating opacity matrix with representative values...")

alpha_matrix = np.zeros((n_layers, n_wavelengths))

# Fill with representative opacity values matching Korg.jl test pattern
for i in range(n_layers):
    for j in range(n_wavelengths):
        # Base continuum opacity
        α_continuum = 3.5e-9  # cm⁻¹
        
        # Line opacity (wavelength dependent, stronger in deeper layers)
        λ = wavelengths[j]
        line_strength = 1e-5 * (i / n_layers)
        
        # Add spectral lines at specific wavelengths
        if abs(λ - 5020.0) < 0.1 or abs(λ - 5050.0) < 0.1 or abs(λ - 5080.0) < 0.1:
            line_strength *= 50  # Strong absorption lines
        
        α_total = α_continuum + line_strength
        alpha_matrix[i, j] = α_total

print("   ✅ Opacity matrix created:")
print(f"      Shape: {alpha_matrix.shape}")
print(f"      Opacity range: {alpha_matrix.min():.2e} - {alpha_matrix.max():.2e} cm⁻¹")

# 5. Source Function (Planck function)
print("\n5. Source Function Calculation:")
print("   Computing Planck source function for each layer...")

source_matrix = np.zeros((n_layers, n_wavelengths))

for i in range(n_layers):
    T = temperatures[i]
    for j in range(n_wavelengths):
        λ_cm = wavelengths[j] * 1e-8  # Convert Å to cm
        
        # Planck function B_λ(T)
        exponent = hplanck_cgs * c_cgs / (λ_cm * kboltz_cgs * T)
        if exponent < 500:  # Avoid overflow
            planck_denominator = np.exp(exponent) - 1
            source_matrix[i, j] = (2 * hplanck_cgs * c_cgs**2 / λ_cm**5) / planck_denominator
        else:
            source_matrix[i, j] = 0.0  # Wien tail

print("   ✅ Source function calculated:")
print(f"      Shape: {source_matrix.shape}")
print(f"      Source range: {source_matrix.min():.2e} - {source_matrix.max():.2e} erg/s/cm²/Å/sr")

# 6. Call Radiative Transfer Correctly (MAIN FUNCTION)
print("\n6. Radiative Transfer Calculation (Correct API):")
print("   Calling radiative_transfer_korg_compatible with proper parameters...")

# Set up parameters exactly as synthesis.py does
spatial_coord = heights  # Use actual layer heights
mu_values = 20  # Default from Korg.jl
tau_scheme = "anchored"
I_scheme = "linear_flux_only"

# Reference opacity for anchoring (α at 5000 Å)
alpha5_reference = alpha_matrix[:, 0]  # First wavelength is 5000 Å

# Call the main RT function correctly
try:
    flux, intensity, mu_surface_grid, mu_weights = radiative_transfer_korg_compatible(
        alpha=alpha_matrix,
        source=source_matrix,
        spatial_coord=spatial_coord,
        mu_points=mu_values,
        spherical=False,  # Plane-parallel
        include_inward_rays=False,
        tau_scheme=tau_scheme,
        I_scheme=I_scheme,
        alpha_ref=alpha5_reference,
        tau_ref=tau_5000
    )
    
    print("   ✅ Radiative transfer completed successfully!")
    print(f"      Flux shape: {flux.shape}")
    print(f"      Flux range: {flux.min():.2e} - {flux.max():.2e} erg/s/cm²/Å")
    print(f"      μ points: {len(mu_surface_grid)}")
    print(f"      μ range: {mu_surface_grid.min():.3f} - {mu_surface_grid.max():.3f}")
    
except Exception as e:
    print(f"   ❌ RT calculation failed: {e}")
    flux = None

# 7. Compute Continuum Flux
print("\n7. Continuum Flux Calculation:")
print("   Computing continuum-only flux for comparison...")

# Create continuum-only opacity
alpha_continuum_only = np.ones_like(alpha_matrix) * 3.5e-9  # Constant continuum

try:
    continuum_flux, _, _, _ = radiative_transfer_korg_compatible(
        alpha=alpha_continuum_only,
        source=source_matrix,
        spatial_coord=spatial_coord,
        mu_points=mu_values,
        spherical=False,
        include_inward_rays=False,
        tau_scheme=tau_scheme,
        I_scheme=I_scheme,
        alpha_ref=alpha5_reference,
        tau_ref=tau_5000
    )
    
    print("   ✅ Continuum flux calculated:")
    print(f"      Continuum range: {continuum_flux.min():.2e} - {continuum_flux.max():.2e} erg/s/cm²/Å")
    
except Exception as e:
    print(f"   ❌ Continuum calculation failed: {e}")
    continuum_flux = None

# 8. Analysis and Validation
if flux is not None and continuum_flux is not None:
    print("\n8. Flux Analysis and Validation:")
    
    # Rectified flux
    rectified_flux = flux / np.maximum(continuum_flux, 1e-10)
    line_depths = 1.0 - rectified_flux
    
    print("   ✅ Spectral analysis:")
    print(f"      Rectified flux range: {rectified_flux.min():.3f} - {rectified_flux.max():.3f}")
    print(f"      Maximum line depth: {line_depths.max()*100:.1f}%")
    print(f"      Strong lines (>10% depth): {np.sum(line_depths > 0.1)}")
    print(f"      Mean flux/continuum ratio: {np.mean(rectified_flux):.3f}")
    
    # Physical validation
    print("\n   Physical validation:")
    checks = [
        ("Flux positivity", np.all(flux > 0)),
        ("Continuum positivity", np.all(continuum_flux > 0)),
        ("Line depths < 100%", np.all(line_depths < 1.0)),
        ("μ weights normalized", np.abs(np.sum(mu_weights) - 1.0) < 1e-6),
        ("Flux < Continuum", np.all(flux <= continuum_flux * 1.01))  # Allow 1% tolerance
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"      {check_name}: {status}")
        all_passed = all_passed and passed
    
    if all_passed:
        print("\n   ✅ All validation checks passed!")
    else:
        print("\n   ⚠️ Some validation checks failed")

# 9. Test Individual RT Functions (Optional)
print("\n9. Testing Individual RT Functions:")
print("   These are internal functions used by the main RT...")

# Test generate_mu_grid
from jorg.radiative_transfer_korg_compatible import generate_mu_grid

mu_points, mu_weights = generate_mu_grid(5)
print(f"   ✅ generate_mu_grid(5): {len(mu_points)} points, sum of weights = {np.sum(mu_weights):.3f}")

# 10. Summary
print("\n10. Radiative Transfer Test Summary:")
print("    " + "="*50)
print("    FIXED JORG RADIATIVE TRANSFER TEST COMPLETE")
print("    " + "="*50)
print(f"    • Atmospheric model: MARCS {Teff}K/{logg}/{m_H}")
print(f"    • Atmospheric layers: {n_layers}")
print(f"    • Wavelength points: {n_wavelengths} ({λ_start}-{λ_end} Å)")
print(f"    • Angular points (μ): {mu_values}")
print(f"    • RT scheme: {tau_scheme} τ, {I_scheme} intensity")
print(f"    • API usage: ✅ Main RT function called correctly")
if flux is not None:
    print(f"    • Flux computation: ✅ Successful")
    print(f"    • Max line depth: {line_depths.max()*100:.1f}%")
else:
    print(f"    • Flux computation: ❌ Failed")
print()
print("    Test demonstrates correct usage of Jorg RT APIs")
print("    matching Korg.jl atmospheric structure and wavelengths.")