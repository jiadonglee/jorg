#!/usr/bin/env julia
"""
Korg.jl Synthesis Reference Data Generator

This script generates reference synthesis data for validation against Jorg.
Tests both synth() and synthesize() functions across stellar parameter grid.
"""

# Load Korg module
include("../../../src/Korg.jl")
using .Korg
using CSV, DataFrames, JSON

println("=" ^ 60)
println("Korg.jl Synthesis Reference Generator")
println("=" ^ 60)

# Test stellar parameter grid
stellar_parameters = [
    # Solar-type stars
    (5777, 4.44, 0.0, "Sun"),
    (5800, 4.5, 0.0, "Solar_analog"),
    (5800, 4.5, -0.5, "Metal_poor_solar"),
    (5800, 4.5, 0.3, "Metal_rich_solar"),
    
    # M dwarfs
    (3500, 4.5, 0.0, "M_dwarf"),
    (3800, 4.8, -0.3, "M_dwarf_poor"),
    
    # K dwarfs  
    (4500, 4.5, 0.0, "K_dwarf"),
    (5200, 4.6, 0.2, "K_dwarf_rich"),
    
    # G dwarfs
    (5500, 4.4, -0.2, "G_dwarf"),
    (6000, 4.3, 0.1, "G_dwarf_rich"),
    
    # F dwarfs
    (6500, 4.2, 0.0, "F_dwarf"),
    (7000, 4.0, -0.1, "F_dwarf_poor"),
    
    # Giants
    (4800, 2.5, 0.0, "K_giant"),
    (5200, 3.0, -0.5, "G_giant_poor"),
]

# Wavelength ranges to test
wavelength_ranges = [
    (5000, 5100, "Blue_green"),
    (5400, 5500, "Green"), 
    (6000, 6100, "Red"),
    (6500, 6600, "Deep_red"),
]

results = Dict()
errors = []

println("\nGenerating reference data for $(length(stellar_parameters)) stars...")

for (i, (Teff, logg, m_H, name)) in enumerate(stellar_parameters)
    println("\n$(i)/$(length(stellar_parameters)): $name (Teff=$Teff, logg=$logg, [M/H]=$m_H)")
    
    try
        # Format abundances
        A_X = Korg.format_A_X(m_H)
        
        # Interpolate atmosphere  
        atm = Korg.interpolate_marcs(Teff, logg, A_X)
        
        star_results = Dict()
        
        for (λ_start, λ_end, wl_name) in wavelength_ranges
            println("  Testing $wl_name: $λ_start-$λ_end Å")
            
            try
                # Test synth() function
                wl_synth, flux_synth, cntm_synth = Korg.synth(
                    Teff=Teff, logg=logg, m_H=m_H,
                    wavelengths=(λ_start, λ_end),
                    rectify=true
                )
                
                # Test synthesize() function  
                result_detailed = Korg.synthesize(
                    atm, Korg.get_VALD_solar_linelist(), A_X, 
                    λ_start, λ_end;
                    vmic=1.0
                )
                
                # Store results
                range_key = "$(name)_$(wl_name)"
                star_results[range_key] = Dict(
                    "wavelengths_synth" => collect(wl_synth),
                    "flux_synth" => collect(flux_synth),
                    "continuum_synth" => collect(cntm_synth),
                    "wavelengths_detailed" => collect(result_detailed.wavelengths),
                    "flux_detailed" => collect(result_detailed.flux),
                    "continuum_detailed" => result_detailed.cntm !== nothing ? collect(result_detailed.cntm) : nothing,
                    "alpha_shape" => size(result_detailed.alpha),
                    "mu_grid_length" => length(result_detailed.mu_grid),
                    "n_species" => length(result_detailed.number_densities),
                    "stellar_params" => [Teff, logg, m_H],
                    "wavelength_range" => [λ_start, λ_end]
                )
                
                println("    ✓ Success: flux range $(minimum(flux_synth):.2e) - $(maximum(flux_synth):.2e)")
                
            catch e
                println("    ✗ Failed: $e")
                push!(errors, "$name $wl_name: $e")
            end
        end
        
        results[name] = star_results
        
    catch e
        println("  ✗ Star failed: $e")
        push!(errors, "$name: $e")
    end
end

# Save results
println("\nSaving reference data...")

# Save as JSON
open("korg_synthesis_reference.json", "w") do f
    JSON.print(f, results, 2)
end

# Save stellar parameters as CSV
params_df = DataFrame(
    Name = [p[4] for p in stellar_parameters],
    Teff = [p[1] for p in stellar_parameters], 
    logg = [p[2] for p in stellar_parameters],
    m_H = [p[3] for p in stellar_parameters]
)
CSV.write("korg_stellar_parameters.csv", params_df)

# Save wavelength ranges
wl_df = DataFrame(
    Name = [w[3] for w in wavelength_ranges],
    Lambda_start = [w[1] for w in wavelength_ranges],
    Lambda_end = [w[2] for w in wavelength_ranges]
)
CSV.write("korg_wavelength_ranges.csv", wl_df)

# Save errors if any
if !isempty(errors)
    println("\nErrors encountered:")
    for error in errors
        println("  - $error")
    end
    
    open("korg_synthesis_errors.txt", "w") do f
        for error in errors
            println(f, error)
        end
    end
end

# Summary
successful_results = sum(length(v) for v in values(results))
println("\n" * "=" ^ 60)
println("Reference data generation complete!")
println("Successful combinations: $successful_results")
println("Errors: $(length(errors))")
println("Files saved:")
println("  - korg_synthesis_reference.json")
println("  - korg_stellar_parameters.csv") 
println("  - korg_wavelength_ranges.csv")
if !isempty(errors)
    println("  - korg_synthesis_errors.txt")
end
println("=" ^ 60)
