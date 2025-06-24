using Korg, Statistics, Printf, JSON
using Korg.ContinuumAbsorption
using Korg: @species_str

println("Generating Korg reference data...")

# Simple test conditions to avoid bounds issues
T = 5778.0  # Temperature (K)
wavelengths = collect(5000:20:6000)  # Wavelength range (Å)
frequencies = [Korg.c_cgs / (λ * 1e-8) for λ in wavelengths]  # Convert to frequencies (Hz)

# Simplified number densities (avoid He to prevent bounds errors)
number_densities = Dict(
    species"H_I" => 1e16,
    species"H_II" => 1e12,
    species"He_I" => 0.0,  # Set to zero to avoid bounds issues
    species"H2" => 1e10
)

# Simple partition functions
partition_funcs = Dict(
    species"H_I" => (log_T -> 2.0),
    species"He_I" => (log_T -> 1.0)
)

# Electron density
ne = 1e15

println("Test conditions:")
@printf("  Temperature: %.1f K\n", T)
@printf("  Electron density: %.2e cm⁻³\n", ne)
@printf("  H I density: %.2e cm⁻³\n", number_densities[species"H_I"])
@printf("  Wavelength range: %d-%d Å\n", wavelengths[1], wavelengths[end])

# Calculate continuum absorption
println("\nCalculating continuum absorption...")

α_continuum = total_continuum_absorption(
    reverse(frequencies),  # Korg expects high to low frequency
    T, ne, number_densities, partition_funcs
)

println("Continuum absorption calculated successfully")
@printf("Alpha range: %.2e to %.2e cm⁻¹\n", minimum(α_continuum), maximum(α_continuum))

# Thomson scattering
α_thomson = ne * 6.6524e-25

# Prepare results
results = Dict(
    "wavelengths_angstrom" => wavelengths,
    "frequencies" => frequencies,
    "alpha_total" => reverse(α_continuum),  # Reverse to match ascending frequency order
    "alpha_thomson" => α_thomson,
    "temperature" => T,
    "electron_density" => ne,
    "number_densities" => Dict(
        "H_I" => number_densities[species"H_I"],
        "H_II" => number_densities[species"H_II"],
        "He_I" => number_densities[species"He_I"],
        "H2" => number_densities[species"H2"]
    )
)

# Save to JSON
open("Jorg/korg_reference_data.json", "w") do f
    JSON.print(f, results)
end

println("\nKorg reference data saved to Jorg/korg_reference_data.json")
println("Data includes ", length(wavelengths), " wavelength points from ", wavelengths[1], " to ", wavelengths[end], " Å")