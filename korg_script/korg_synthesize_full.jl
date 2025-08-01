#!/usr/bin/env julia
"""
Korg.jl synthesize() Function with Complete Result Dictionary
===========================================================

This script uses Korg.jl's synthesize() function (not synth()) to get
the complete SynthesisResult with all diagnostic information and saves
the result dictionary to files.
"""

using Korg
using Printf
using Statistics
using Dates
using JLD2  # For saving Julia data structures
using JSON3  # For JSON export

println("🔬 Korg.jl synthesize() - Complete Analysis")
println("="^60)

# Synthesis parameters
Teff = 5780      # Effective temperature (K)
logg = 4.44      # Surface gravity
m_H = 0.0        # Metallicity [M/H]
wavelengths = (5000, 5200)  # Wavelength range (Å)
vmic = 1.0       # Microturbulence (km/s)

println("Parameters:")
println("  Teff: $Teff K")
println("  logg: $logg")
println("  [M/H]: $m_H")
println("  Wavelengths: $wavelengths Å")
println("  vmic: $vmic km/s")
println("  Using synthesize() for complete diagnostics")
println()

try
    # Create atmosphere and abundance array (following Korg.jl workflow)
    println("🌍 Creating model atmosphere...")
    A_X = format_A_X(m_H)  # Create abundance array
    atm = interpolate_marcs(Teff, logg, A_X)  # Interpolate atmosphere
    
    println("✅ Atmosphere created:")
    println("   Layers: $(length(atm.layers))")
    println("   Temperature range: $(round(minimum([l.temp for l in atm.layers]), digits=1)) - $(round(maximum([l.temp for l in atm.layers]), digits=1)) K")
    println("   Pressure range: $(round(minimum([l.pressure for l in atm.layers]), sigdigits=3)) - $(round(maximum([l.pressure for l in atm.layers]), sigdigits=3)) dyn/cm²")
    println()
    
    # Load linelist (using default VALD)
    println("📋 Loading linelist...")
    # Try different linelist loading methods
    try
        linelist = get_VALD_solar_linelist()
        println("✅ VALD solar linelist loaded: $(length(linelist)) lines")
    catch
        try
            # Alternative: read from file if get_VALD_solar_linelist() doesn't exist
            linelist_files = filter(f -> endswith(f, ".vald"), readdir("."))
            if !isempty(linelist_files)
                linelist = read_linelist(linelist_files[1])
                println("✅ VALD linelist from file: $(length(linelist)) lines")
            else
                # Use empty linelist for continuum-only synthesis
                linelist = []
                println("⚠️  No linelist found - running continuum-only synthesis")
            end
        catch
            linelist = []
            println("⚠️  Using empty linelist - continuum-only synthesis")
        end
    end
    println()
    
    # Run full synthesis using synthesize()
    println("⚗️  Running complete stellar synthesis...")
    start_time = time()
    
    result = synthesize(
        atm,                    # Model atmosphere
        linelist,              # Line list
        A_X,                   # Abundance array
        wavelengths,           # Wavelength range
        vmic=vmic,             # Microturbulent velocity
        return_cntm=true,      # Return continuum
        verbose=false          # Suppress detailed output
    )
    
    elapsed_time = time() - start_time
    
    println("✅ Full synthesis completed!")
    println("   Time: $(round(elapsed_time, digits=2)) seconds")
    println()
    
    # Analyze the complete result structure
    println("📊 SYNTHESIS RESULT ANALYSIS")
    println("-" * 50)
    
    # Basic spectrum info
    println("Spectrum data:")
    println("  Wavelength points: $(length(result.wavelengths))")
    println("  Wavelength range: $(round(minimum(result.wavelengths), digits=2)) - $(round(maximum(result.wavelengths), digits=2)) Å")
    println("  Flux range: $(round(minimum(result.flux), digits=4)) - $(round(maximum(result.flux), digits=4))")
    
    if result.cntm !== nothing
        println("  Continuum range: $(round(minimum(result.cntm), sigdigits=4)) - $(round(maximum(result.cntm), sigdigits=4))")
        # Calculate normalized flux
        normalized_flux = result.flux ./ result.cntm
        min_norm = minimum(normalized_flux)
        max_absorption = (1.0 - min_norm) * 100
        println("  Normalized flux range: $(round(min_norm, digits=3)) - $(round(maximum(normalized_flux), digits=3))")
        println("  Deepest absorption: $(round(max_absorption, digits=1))%")
    end
    println()
    
    # Opacity matrix analysis
    println("Opacity matrix (alpha):")
    println("  Shape: $(size(result.alpha)) (layers × wavelengths)")
    println("  Opacity range: $(round(minimum(result.alpha), sigdigits=4)) - $(round(maximum(result.alpha), sigdigits=4)) cm⁻¹")
    println("  Mean opacity: $(round(mean(result.alpha), sigdigits=4)) cm⁻¹")
    println()
    
    # Intensity analysis
    println("Intensity array:")
    println("  Shape: $(size(result.intensity))")
    println("  Intensity range: $(round(minimum(result.intensity), sigdigits=4)) - $(round(maximum(result.intensity), sigdigits=4))")
    println()
    
    # Atmospheric structure
    println("Atmospheric structure:")
    println("  Layers: $(length(result.electron_number_density))")
    println("  Electron density range: $(round(minimum(result.electron_number_density), sigdigits=3)) - $(round(maximum(result.electron_number_density), sigdigits=3)) cm⁻³")
    println()
    
    # Number densities
    println("Species number densities:")
    println("  Species count: $(length(result.number_densities))")
    for (i, (species, densities)) in enumerate(result.number_densities)
        if i <= 5  # Show first 5 species
            println("  $(species): $(round(minimum(densities), sigdigits=3)) - $(round(maximum(densities), sigdigits=3)) cm⁻³")
        elseif i == 6
            println("  ... ($(length(result.number_densities)-5) more species)")
            break
        end
    end
    println()
    
    # Mu grid
    println("Radiative transfer μ grid:")
    println("  μ points: $(length(result.mu_grid))")
    println("  μ values: $(round.([μ[1] for μ in result.mu_grid[1:min(5, end)]], digits=3))$(length(result.mu_grid) > 5 ? "..." : "")")
    println()
    
    # Subspectra
    println("Wavelength subspectra:")
    println("  Ranges: $(length(result.subspectra))")
    for (i, subspec) in enumerate(result.subspectra)
        println("  Range $i: indices $(subspec)")
    end
    println()
    
    # Save complete result to JLD2 (Julia binary format)
    println("💾 Saving complete synthesis result...")
    
    # Save as JLD2 (preserves Julia data types)
    jld2_filename = "korg_synthesis_result.jld2"
    @save jld2_filename result
    println("✅ Complete result saved to: $jld2_filename")
    
    # Create a simplified dictionary for JSON export
    result_dict = Dict(
        "metadata" => Dict(
            "generated_on" => string(now()),
            "parameters" => Dict(
                "Teff" => Teff,
                "logg" => logg,
                "m_H" => m_H,
                "wavelengths" => wavelengths,
                "vmic" => vmic
            ),
            "synthesis_time" => elapsed_time,
            "atmosphere_layers" => length(result.electron_number_density),
            "linelist_size" => length(linelist),
            "wavelength_points" => length(result.wavelengths)
        ),
        "spectrum" => Dict(
            "wavelengths" => collect(result.wavelengths),
            "flux" => collect(result.flux),
            "continuum" => result.cntm !== nothing ? collect(result.cntm) : nothing
        ),
        "opacity" => Dict(
            "alpha_matrix_shape" => size(result.alpha),
            "alpha_min" => minimum(result.alpha),
            "alpha_max" => maximum(result.alpha),
            "alpha_mean" => mean(result.alpha)
        ),
        "atmosphere" => Dict(
            "electron_density" => collect(result.electron_number_density),
            "number_densities_species" => [string(s) for s in keys(result.number_densities)]
        ),
        "radiative_transfer" => Dict(
            "mu_grid_size" => length(result.mu_grid),
            "mu_values" => [μ[1] for μ in result.mu_grid],
            "mu_weights" => [μ[2] for μ in result.mu_grid],
            "intensity_shape" => size(result.intensity)
        )
    )
    
    # Save as JSON
    json_filename = "korg_synthesis_result.json"
    open(json_filename, "w") do f
        JSON3.pretty(f, result_dict)
    end
    println("✅ Summary result saved to: $json_filename")
    
    # Save detailed spectrum data to text file  
    txt_filename = "korg_synthesis_spectrum.txt"
    open(txt_filename, "w") do file
        println(file, "# Korg.jl Complete Synthesis Result")
        println(file, "# Generated on: $(now())")
        println(file, "# Parameters: Teff=$Teff K, logg=$logg, [M/H]=$m_H, vmic=$vmic km/s")
        println(file, "# Atmosphere layers: $(length(result.electron_number_density))")
        println(file, "# Linelist size: $(length(linelist)) lines")
        println(file, "# Synthesis time: $(round(elapsed_time, digits=2)) seconds")
        println(file, "#")
        println(file, "# Column 1: Wavelength (Å)")
        println(file, "# Column 2: Flux")
        if result.cntm !== nothing
            println(file, "# Column 3: Continuum (erg/s/cm²/Å)")
            println(file, "# Column 4: Normalized Flux")
            println(file, "#")
            println(file, "# Wavelength    Flux           Continuum       NormalizedFlux")
            normalized_flux = result.flux ./ result.cntm
            for i in 1:length(result.wavelengths)
                @printf(file, "%10.4f  %15.6e  %15.6e  %12.6f\n", 
                       result.wavelengths[i], result.flux[i], result.cntm[i], normalized_flux[i])
            end
        else
            println(file, "#")
            println(file, "# Wavelength    Flux")
            for i in 1:length(result.wavelengths)
                @printf(file, "%10.4f  %15.6e\n", result.wavelengths[i], result.flux[i])
            end
        end
    end
    println("✅ Spectrum data saved to: $txt_filename")
    
    # Save opacity matrix to binary file (due to large size)
    opacity_filename = "korg_opacity_matrix.jld2"
    @save opacity_filename alpha=result.alpha
    println("✅ Opacity matrix saved to: $opacity_filename")
    
    println()
    println("📁 FILES CREATED:")
    println("   1. $jld2_filename - Complete Julia result object")
    println("   2. $json_filename - Summary in JSON format")  
    println("   3. $txt_filename - Spectrum data (wavelength, flux, continuum)")
    println("   4. $opacity_filename - Opacity matrix (layers × wavelengths)")
    println()
    
    # Show file sizes
    files = [jld2_filename, json_filename, txt_filename, opacity_filename]
    println("📊 File sizes:")
    for file in files
        if isfile(file)
            size_mb = filesize(file) / 1024 / 1024
            println("   $(file): $(round(size_mb, digits=2)) MB")
        end
    end
    
    println()
    println("🎉 SUCCESS! Complete synthesis result with full diagnostics saved.")
    println("   Use: result = load(\"$jld2_filename\")[\"result\"] to reload in Julia")
    
catch e
    println("❌ Synthesis failed: $e")
    println()
    println("Common solutions:")
    println("- Install required packages: Pkg.add([\"JLD2\", \"JSON3\"])")
    println("- Check Korg.jl installation and project environment")
    println("- Verify model atmosphere data is available")
    rethrow(e)
end

println()
println("🏁 Korg.jl synthesize() complete analysis finished!")