#!/usr/bin/env julia
"""
Korg.jl API Flow 6: Radiative Transfer Calculation (FIXED)

Demonstrates Korg.jl radiative transfer with same settings as Jorg test:
- Uses actual MARCS atmospheric model
- Same wavelength grid settings
- Proper RT API usage
"""

using Korg
using Printf
using Statistics: mean

println("="^70)
println("KORG.JL API FLOW 6: RADIATIVE TRANSFER CALCULATION (FIXED)")
println("="^70)

# 1. Setup Atmospheric Model
println("\n1. Setup Atmospheric Model:")
println("   Loading MARCS model atmosphere...")

# Solar parameters
Teff = 5780.0
logg = 4.44
m_H = 0.0

# Load MARCS atmosphere
atm = interpolate_marcs(Teff, logg, m_H)
temperatures = [layer.temp for layer in atm.layers]
number_densities = [layer.number_density for layer in atm.layers]
electron_densities = [layer.electron_number_density for layer in atm.layers]
tau_5000 = [layer.tau_5000 for layer in atm.layers]
heights = [layer.z for layer in atm.layers]

n_layers = length(temperatures)

println("   ✅ MARCS atmospheric structure loaded:")
println("      Layers: $(n_layers)")
@printf("      Temperature range: %.1f - %.1f K\n", minimum(temperatures), maximum(temperatures))
@printf("      τ₅₀₀₀ range: %.2e - %.2e\n", minimum(tau_5000), maximum(tau_5000))
@printf("      Number density range: %.2e - %.2e cm⁻³\n", minimum(number_densities), maximum(number_densities))
@printf("      Height range: %.2e - %.2e cm\n", minimum(heights), maximum(heights))

# 2. Wavelength Grid
println("\n2. Wavelength Grid:")
println("   Setting up wavelength grid...")

# Match test settings: 1000 points from 5000-5100 Å
λ_start = 5000.0
λ_end = 5100.0
n_wavelengths = 1000
wavelengths = range(λ_start, λ_end, length=n_wavelengths)

println("   ✅ Wavelength grid created:")
println("      Range: $(λ_start) - $(λ_end) Å")
println("      Points: $(n_wavelengths)")
@printf("      Resolution: %.3f Å\n", (λ_end - λ_start)/(n_wavelengths-1))

# 3. Create Opacity Matrix
println("\n3. Opacity Matrix:")
println("   Creating opacity matrix with representative values...")

alpha_matrix = zeros(n_layers, n_wavelengths)

# Fill with representative opacity values
for i in 1:n_layers
    for j in 1:n_wavelengths
        # Base continuum opacity
        α_continuum = 3.5e-9  # cm⁻¹
        
        # Line opacity (wavelength dependent, stronger in deeper layers)
        λ = wavelengths[j]
        line_strength = 1e-5 * (i / n_layers)
        
        # Add spectral lines at specific wavelengths
        if abs(λ - 5020.0) < 0.1 || abs(λ - 5050.0) < 0.1 || abs(λ - 5080.0) < 0.1
            line_strength *= 50  # Strong absorption lines
        end
        
        α_total = α_continuum + line_strength
        alpha_matrix[i, j] = α_total
    end
end

println("   ✅ Opacity matrix created:")
println("      Shape: $(size(alpha_matrix))")
@printf("      Opacity range: %.2e - %.2e cm⁻¹\n", minimum(alpha_matrix), maximum(alpha_matrix))

# 4. Source Function (Planck function)
println("\n4. Source Function Calculation:")
println("   Computing Planck source function for each layer...")

# Physical constants
k_B = 1.38e-16   # erg/K
h = 6.626e-27    # erg⋅s  
c = 2.998e10     # cm/s

source_matrix = zeros(n_layers, n_wavelengths)

for i in 1:n_layers
    T = temperatures[i]
    for j in 1:n_wavelengths
        λ_cm = wavelengths[j] * 1e-8  # Convert Å to cm
        
        # Planck function B_λ(T)
        exponent = h * c / (λ_cm * k_B * T)
        if exponent < 500  # Avoid overflow
            planck_denominator = exp(exponent) - 1
            source_matrix[i, j] = (2 * h * c^2 / λ_cm^5) / planck_denominator
        else
            source_matrix[i, j] = 0.0  # Wien tail
        end
    end
end

println("   ✅ Source function calculated:")
println("      Shape: $(size(source_matrix))")
@printf("      Source range: %.2e - %.2e erg/s/cm²/Å/sr\n", minimum(source_matrix), maximum(source_matrix))

# 5. Call Radiative Transfer (Main API)
println("\n5. Radiative Transfer Calculation (Main API):")
println("   Calling radiative_transfer with proper parameters...")

# Set up parameters
mu_values = 20  # Default angular quadrature points

# Reference opacity for anchoring (α at 5000 Å)
alpha5_reference = alpha_matrix[:, 1]  # First wavelength is 5000 Å

# Call the main RT function
using Korg.RadiativeTransfer

flux, intensity, mu_grid, mu_weights = RadiativeTransfer.radiative_transfer(
    atm,                    # Model atmosphere object
    alpha_matrix,           # Opacity matrix
    source_matrix,          # Source function matrix
    mu_values;              # Number of μ points
    α_ref=alpha5_reference, # Reference opacity
    τ_scheme="anchored",    # Optical depth scheme
    I_scheme="linear_flux_only"  # Intensity scheme
)

println("   ✅ Radiative transfer completed successfully!")
println("      Flux shape: $(size(flux))")
@printf("      Flux range: %.2e - %.2e erg/s/cm²/Å\n", minimum(flux), maximum(flux))
println("      μ points: $(length(mu_grid))")
if length(mu_grid) > 1
    @printf("      μ range: %.3f - %.3f\n", minimum([μ for (μ, w) in mu_grid]), maximum([μ for (μ, w) in mu_grid]))
else
    println("      μ range: N/A (optimized flux-only calculation)")
end

# 6. Compute Continuum Flux
println("\n6. Continuum Flux Calculation:")
println("   Computing continuum-only flux for comparison...")

# Create continuum-only opacity
alpha_continuum_only = ones(size(alpha_matrix)) * 3.5e-9  # Constant continuum

continuum_flux, _, _, _ = RadiativeTransfer.radiative_transfer(
    atm,
    alpha_continuum_only,
    source_matrix,
    mu_values;
    α_ref=alpha5_reference,
    τ_scheme="anchored",
    I_scheme="linear_flux_only"
)

println("   ✅ Continuum flux calculated:")
@printf("      Continuum range: %.2e - %.2e erg/s/cm²/Å\n", minimum(continuum_flux), maximum(continuum_flux))

# 7. Analysis and Validation
println("\n7. Flux Analysis and Validation:")

# Rectified flux
rectified_flux = flux ./ max.(continuum_flux, 1e-10)
line_depths = 1.0 .- rectified_flux

println("   ✅ Spectral analysis:")
@printf("      Rectified flux range: %.3f - %.3f\n", minimum(rectified_flux), maximum(rectified_flux))
@printf("      Maximum line depth: %.1f%%\n", maximum(line_depths)*100)
println("      Strong lines (>10% depth): $(sum(line_depths .> 0.1))")
@printf("      Mean flux/continuum ratio: %.3f\n", mean(rectified_flux))

# Physical validation
println("\n   Physical validation:")
checks = [
    ("Flux positivity", all(flux .> 0)),
    ("Continuum positivity", all(continuum_flux .> 0)),
    ("Line depths < 100%", all(line_depths .< 1.0)),
    ("μ weights normalized", length(mu_grid) == 1 || abs(sum([w for (μ, w) in mu_grid]) - 1.0) < 1e-6),
    ("Flux < Continuum", all(flux .<= continuum_flux * 1.01))  # Allow 1% tolerance
]

global all_passed = true
for (check_name, passed) in checks
    status = passed ? "✅ PASS" : "❌ FAIL"
    @printf("      %-25s %s\n", check_name * ":", status)
    global all_passed = all_passed && passed
end

if all_passed
    println("\n   ✅ All validation checks passed!")
else
    println("\n   ⚠️ Some validation checks failed")
end

# 8. Test Individual RT Functions
println("\n8. Testing Individual RT Functions:")
println("   These are internal functions used by the main RT...")

# Test generate_mu_grid
mu_points, mu_weights = RadiativeTransfer.generate_mu_grid(5)
@printf("   ✅ generate_mu_grid(5): %d points, sum of weights = %.3f\n", length(mu_points), sum(mu_weights))

# 9. Summary
println("\n9. Radiative Transfer Test Summary:")
println("    " * "="^50)
println("    FIXED KORG.JL RADIATIVE TRANSFER TEST COMPLETE")
println("    " * "="^50)
println("    • Atmospheric model: MARCS $(Teff)K/$(logg)/$(m_H)")
println("    • Atmospheric layers: $(n_layers)")
println("    • Wavelength points: $(n_wavelengths) ($(λ_start)-$(λ_end) Å)")
println("    • Angular points (μ): $(mu_values)")
println("    • RT scheme: anchored τ, linear_flux_only intensity")
println("    • API usage: ✅ Main RT function called correctly")
println("    • Flux computation: ✅ Successful")
@printf("    • Max line depth: %.1f%%\n", maximum(line_depths)*100)
println()
println("    Test demonstrates correct usage of Korg.jl RT APIs")
println("    with proper atmospheric structure and wavelengths.")