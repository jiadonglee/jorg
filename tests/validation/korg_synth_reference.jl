#!/usr/bin/env julia
"""
Korg.jl synth() Reference Data Generator

This script generates reference data using Korg.jl's synth() function
for comparison with Jorg's implementation.
"""

# Load Korg module
include("../../../src/Korg.jl")
using .Korg
using CSV, DataFrames, JSON

println("=" ^ 60)
println("Korg.jl synth() Reference Generator")
println("=" ^ 60)

# Test parameters for synth() function
test_cases = [
    # Basic solar case
    Dict(
        "name" => "solar_basic",
        "Teff" => 5777,
        "logg" => 4.44,
        "m_H" => 0.0,
        "wavelengths" => (5000, 5100),
        "rectify" => true,
        "vmic" => 1.0
    ),
    
    # Metal-poor solar
    Dict(
        "name" => "solar_metal_poor",
        "Teff" => 5777,
        "logg" => 4.44,
        "m_H" => -0.5,
        "wavelengths" => (5000, 5100),
        "rectify" => true,
        "vmic" => 1.0
    ),
    
    # Different stellar types
    Dict(
        "name" => "m_dwarf",
        "Teff" => 3500,
        "logg" => 4.5,
        "m_H" => 0.0,
        "wavelengths" => (5000, 5100),
        "rectify" => true,
        "vmic" => 1.0
    ),
    
    Dict(
        "name" => "k_dwarf",
        "Teff" => 4500,
        "logg" => 4.5,
        "m_H" => 0.0,
        "wavelengths" => (5000, 5100),
        "rectify" => true,
        "vmic" => 1.0
    ),
    
    Dict(
        "name" => "f_dwarf",
        "Teff" => 6500,
        "logg" => 4.2,
        "m_H" => 0.0,
        "wavelengths" => (5000, 5100),
        "rectify" => true,
        "vmic" => 1.0
    ),
    
    # Different wavelength ranges
    Dict(
        "name" => "solar_blue",
        "Teff" => 5777,
        "logg" => 4.44,
        "m_H" => 0.0,
        "wavelengths" => (4500, 4600),
        "rectify" => true,
        "vmic" => 1.0
    ),
    
    Dict(
        "name" => "solar_red",
        "Teff" => 5777,
        "logg" => 4.44,
        "m_H" => 0.0,
        "wavelengths" => (6000, 6100),
        "rectify" => true,
        "vmic" => 1.0
    ),
    
    # Unrectified spectrum
    Dict(
        "name" => "solar_unrectified",
        "Teff" => 5777,
        "logg" => 4.44,
        "m_H" => 0.0,
        "wavelengths" => (5000, 5100),
        "rectify" => false,
        "vmic" => 1.0
    ),
    
    # Different microturbulence
    Dict(
        "name" => "solar_vmic_2",
        "Teff" => 5777,
        "logg" => 4.44,
        "m_H" => 0.0,
        "wavelengths" => (5000, 5100),
        "rectify" => true,
        "vmic" => 2.0
    ),
    
    # Individual element variation
    Dict(
        "name" => "solar_fe_poor",
        "Teff" => 5777,
        "logg" => 4.44,
        "m_H" => 0.0,
        "Fe" => -0.3,
        "wavelengths" => (5000, 5100),
        "rectify" => true,
        "vmic" => 1.0
    )
]

results = Dict()
errors = []

println("\nGenerating reference data for $(length(test_cases)) test cases...")

for (i, test_case) in enumerate(test_cases)
    name = test_case["name"]
    println("\n$(i)/$(length(test_cases)): Testing $name")
    
    try
        # Extract parameters
        Teff = test_case["Teff"]
        logg = test_case["logg"]
        m_H = test_case["m_H"]
        wavelengths = test_case["wavelengths"]
        rectify = test_case["rectify"]
        vmic = test_case["vmic"]
        
        # Handle individual element abundances
        abundances = Dict()
        for key in keys(test_case)
            if key in ["Fe", "C", "O", "Mg", "Si", "Ca", "Ti", "Cr", "Ni"]
                abundances[Symbol(key)] = test_case[key]
            end
        end
        
        println("  Parameters: Teff=$Teff, logg=$logg, [M/H]=$m_H")
        println("  Wavelengths: $(wavelengths[1])-$(wavelengths[2]) Å")
        println("  Rectify: $rectify, vmic: $vmic km/s")
        if !isempty(abundances)
            println("  Individual abundances: $abundances")
        end
        
        # Run Korg synth
        start_time = time()
        
        if isempty(abundances)
            wavelengths_out, flux, continuum = Korg.synth(
                Teff=Teff,
                logg=logg,
                m_H=m_H,
                wavelengths=wavelengths,
                rectify=rectify,
                vmic=vmic
            )
        else
            wavelengths_out, flux, continuum = Korg.synth(
                Teff=Teff,
                logg=logg,
                m_H=m_H,
                wavelengths=wavelengths,
                rectify=rectify,
                vmic=vmic;
                abundances...
            )
        end
        
        elapsed = time() - start_time
        
        # Store results
        results[name] = Dict(
            "parameters" => test_case,
            "wavelengths" => collect(wavelengths_out),
            "flux" => collect(flux),
            "continuum" => collect(continuum),
            "timing" => elapsed,
            "n_points" => length(wavelengths_out),
            "flux_stats" => Dict(
                "min" => minimum(flux),
                "max" => maximum(flux),
                "mean" => sum(flux) / length(flux),
                "std" => sqrt(sum((flux .- sum(flux)/length(flux)).^2) / length(flux))
            ),
            "continuum_stats" => Dict(
                "min" => minimum(continuum),
                "max" => maximum(continuum),
                "mean" => sum(continuum) / length(continuum)
            ),
            "success" => true
        )
        
        println("  ✓ Success in $(elapsed:.1f)s")
        println("    Points: $(length(wavelengths_out))")
        println("    Flux range: $(minimum(flux):.3e) - $(maximum(flux):.3e)")
        println("    Continuum range: $(minimum(continuum):.3e) - $(maximum(continuum):.3e)")
        
        # Save individual spectrum
        spectrum_df = DataFrame(
            wavelength = collect(wavelengths_out),
            flux = collect(flux),
            continuum = collect(continuum)
        )
        CSV.write("korg_synth_$(name).csv", spectrum_df)
        
    catch e
        println("  ✗ Failed: $e")
        push!(errors, "$name: $e")
        results[name] = Dict("success" => false, "error" => string(e))
    end
end

# Save comprehensive results
println("\nSaving reference data...")

# Save results as JSON
open("korg_synth_reference.json", "w") do f
    JSON.print(f, results, 2)
end

# Save test parameters
test_params = DataFrame(
    name = [tc["name"] for tc in test_cases],
    Teff = [tc["Teff"] for tc in test_cases],
    logg = [tc["logg"] for tc in test_cases],
    m_H = [tc["m_H"] for tc in test_cases],
    wl_start = [tc["wavelengths"][1] for tc in test_cases],
    wl_end = [tc["wavelengths"][2] for tc in test_cases],
    rectify = [tc["rectify"] for tc in test_cases],
    vmic = [tc["vmic"] for tc in test_cases]
)
CSV.write("korg_synth_test_parameters.csv", test_params)

# Save errors if any
if !isempty(errors)
    println("\nErrors encountered:")
    for error in errors
        println("  - $error")
    end
    
    open("korg_synth_errors.txt", "w") do f
        for error in errors
            println(f, error)
        end
    end
end

# Summary
successful_cases = sum(get(result, "success", false) for result in values(results))
println("\n" * "=" ^ 60)
println("Korg.jl synth() reference generation complete!")
println("Successful cases: $successful_cases/$(length(test_cases))")
println("Errors: $(length(errors))")
println("Files generated:")
println("  - korg_synth_reference.json (comprehensive results)")
println("  - korg_synth_test_parameters.csv (test parameters)")
println("  - korg_synth_[name].csv (individual spectra)")
if !isempty(errors)
    println("  - korg_synth_errors.txt (error log)")
end
println("=" ^ 60)