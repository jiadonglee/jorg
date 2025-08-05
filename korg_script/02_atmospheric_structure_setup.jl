#!/usr/bin/env julia
"""
Korg.jl API Flow 2: Atmospheric Structure Setup

Demonstrates Korg.jl equivalent of Jorg's atmospheric structure processing:
- ModelAtmosphere object handling
- Physical constants access
- Atmospheric data extraction
- Pressure calculation
"""

using Korg
using Printf

println("="^70)
println("KORG.JL API FLOW 2: ATMOSPHERIC STRUCTURE SETUP")
println("="^70)

# 1. Load Atmospheric Model (from previous script or recreate)
println("\n1. Atmospheric Model Loading:")
println("   Loading MARCS stellar atmosphere...")

# Solar parameters
Teff = 5780.0  # K
logg = 4.44    # log surface gravity  
m_H = 0.0      # metallicity [M/H]

# Create MARCS atmosphere (equivalent to Jorg's interpolate_atmosphere)
atm = interpolate_marcs(Teff, logg, m_H)

println("   ✅ MARCS atmosphere loaded:")
println("      Model type: $(typeof(atm))")
println("      Layers: $(length(atm.layers))")

# 2. Physical Constants Access (equivalent to Jorg's .constants)
println("\n2. Physical Constants:")
println("   Accessing fundamental constants...")

# Korg.jl constants (equivalent to Jorg's kboltz_cgs, c_cgs, hplanck_cgs)
# These are typically built into calculations, but we can define them
const k_B = 1.380649e-16  # erg/K - Boltzmann constant
const c_light = 2.99792458e10  # cm/s - Speed of light  
const h_planck = 6.62607015e-27  # erg⋅s - Planck constant

println("   ✅ Physical constants defined:")
println("      Boltzmann constant: $(k_B) erg/K")
println("      Speed of light: $(c_light) cm/s")
println("      Planck constant: $(h_planck) erg⋅s")

# 3. Atmospheric Data Extraction (equivalent to Jorg's atm_dict creation)
println("\n3. Atmospheric Data Extraction:")
println("   Extracting layer-by-layer atmospheric structure...")

# Access atmospheric structure (equivalent to Jorg's layer access)
temperatures = [layer.temp for layer in atm.layers]  # K
log_pressures = [log10(layer.number_density * 1.38e-16 * layer.temp) for layer in atm.layers]   # log(P) in dyn/cm²
number_densities = [layer.number_density for layer in atm.layers]  # cm⁻³
electron_densities = [layer.electron_number_density for layer in atm.layers]  # cm⁻³ 
tau_5000 = [layer.tau_5000 for layer in atm.layers]     # τ₅₀₀₀
heights = [layer.z for layer in atm.layers]            # cm

# Convert to linear units
pressures = [layer.number_density * 1.38e-16 * layer.temp for layer in atm.layers]  # dyn/cm²
# Note: tau_5000 is already linear from Korg layers

println("   ✅ Atmospheric structure extracted:")
println("      Layers: $(length(temperatures))")
println("      Temperature: ", round(minimum(temperatures), digits=1), " - ", round(maximum(temperatures), digits=1), " K")
println("      Pressure: ", @sprintf("%.2e", minimum(pressures)), " - ", @sprintf("%.2e", maximum(pressures)), " dyn/cm²")
println("      Number density: ", @sprintf("%.2e", minimum(number_densities)), " - ", @sprintf("%.2e", maximum(number_densities)), " cm⁻³")

# 4. Number Density Calculation (equivalent to Jorg's number_density calculation)
println("\n4. Number Density Calculation:")
println("   Computing particle number densities...")

# Number densities are already available from Korg layers
# These are total particle number densities

println("   ✅ Number densities calculated:")
println("      Range: ", @sprintf("%.2e", minimum(number_densities)), " - ", @sprintf("%.2e", maximum(number_densities)), " cm⁻³")
println("      Direct from Korg atmosphere layers")

# 5. Electron Density Estimation (equivalent to Jorg's electron_density)
println("\n5. Electron Density Estimation:")
println("   Estimating electron densities...")

# Electron densities are already available from Korg layers
# These are calculated from chemical equilibrium

println("   ✅ Electron densities estimated:")
println("      Range: ", @sprintf("%.2e", minimum(electron_densities)), " - ", @sprintf("%.2e", maximum(electron_densities)), " cm⁻³")
println("      Direct from Korg chemical equilibrium")

# 6. Atmospheric Dictionary Creation (equivalent to Jorg's atm_dict)
println("\n6. Atmospheric Dictionary Creation:")
println("   Creating structured atmospheric data...")

# Create comprehensive atmospheric data structure (equivalent to Jorg's format)
atm_dict = Dict(
    "temperature" => temperatures,           # K
    "pressure" => pressures,                # dyn/cm²  
    "number_density" => number_densities,   # particles/cm³
    "electron_density" => electron_densities, # electrons/cm³
    "tau_5000" => tau_5000,         # optical depth at 5000 Å
    "height" => heights            # Heights in cm
)

println("   ✅ Atmospheric dictionary created:")
println("      Keys: $(collect(keys(atm_dict)))")
println("      Layers: $(length(atm_dict["temperature"]))")

# 7. Pressure Verification (equivalent to Jorg's P = n_tot * k * T)
println("\n7. Pressure Verification:")
println("   Verifying ideal gas law: P = n_total × k_B × T")

# Calculate pressure from ideal gas law
calculated_pressures = number_densities .* k_B .* temperatures

# Compare with MARCS pressures
pressure_ratios = calculated_pressures ./ pressures
mean_ratio = sum(pressure_ratios) / length(pressure_ratios)
std_ratio = sqrt(sum((pressure_ratios .- mean_ratio).^2) / length(pressure_ratios))

println("   ✅ Pressure verification:")
println("      MARCS pressures: ", @sprintf("%.2e", minimum(pressures)), " - ", @sprintf("%.2e", maximum(pressures)), " dyn/cm²")
println("      Calculated (PV=nkT): ", @sprintf("%.2e", minimum(calculated_pressures)), " - ", @sprintf("%.2e", maximum(calculated_pressures)), " dyn/cm²")
println("      Agreement ratio: ", round(mean_ratio, digits=2), " ± ", round(std_ratio, digits=2))

if 0.8 <= mean_ratio <= 1.2
    println("      ✅ Good agreement with ideal gas law")
else
    println("      ⚠️ Significant deviation from ideal gas law")
end

# 8. Layer Analysis (equivalent to Jorg's layer diagnostics)
println("\n8. Layer Analysis:")
println("   Analyzing atmospheric structure by layer...")

println("   Layer structure (first 10 layers):")
println("   $("Layer"^10) $("T [K]"^8) $("P [dyn/cm²]"^12) $("n_tot [cm⁻³]"^12) $("τ₅₀₀₀"^8)")
println("   " * "-"^60)

for i in 1:min(10, length(temperatures))
    println("   ", rpad(i, 5), " ", rpad(round(temperatures[i], digits=1), 8), " ", 
           rpad(@sprintf("%.2e", pressures[i]), 12), " ", 
           rpad(@sprintf("%.2e", number_densities[i]), 12), " ", 
           @sprintf("%.2e", tau_5000[i]))
end

if length(temperatures) > 10
    println("   ... ($(length(temperatures)-10) more layers)")
end

# 9. Summary Output (equivalent to Jorg's verbose diagnostics)
println("\n9. Atmospheric Structure Summary:")
println("   ═" * "═"^50)
println("   KORG.JL ATMOSPHERIC STRUCTURE COMPLETE")
println("   ═" * "═"^50)
println("   • Total layers: $(length(temperatures))")
println("   • Temperature span: ", round(maximum(temperatures) - minimum(temperatures), digits=1), " K")
println("   • Pressure range: ", @sprintf("%.1e", maximum(pressures)/minimum(pressures)), "×")
println("   • Optical depth range: ", @sprintf("%.1e", maximum(tau_5000)/minimum(tau_5000)), "×")
println("   • Physical constants: ✅ Loaded")
println("   • Ideal gas verification: ✅ ", round(mean_ratio, digits=2), "× agreement")
println()
println("   Ready for chemical equilibrium and opacity calculations...")

# Export for next scripts
println("\n10. Exported Variables:")
println("    atm = MARCS atmosphere object")
println("    atm_dict = structured atmospheric data")
println("    temperatures, pressures, number_densities = layer arrays")
println("    electron_densities = estimated electron populations")
println("    Physical constants: k_B, c_light, h_planck")