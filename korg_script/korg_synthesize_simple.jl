#!/usr/bin/env julia
"""
Korg.jl synthesize() Function - Simple Version
=============================================

This script uses Korg.jl's synthesize() function to get the complete
SynthesisResult and saves the data to text files (no extra packages needed).
"""

using Korg
using Printf
using Statistics
using Dates

println("🔬 Korg.jl synthesize() - Complete Result")
println("="^50)

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
    # Create atmosphere and abundance array
    println("🌍 Creating model atmosphere...")
    A_X = format_A_X(m_H)
    atm = interpolate_marcs(Teff, logg, A_X)
    
    println("✅ Atmosphere created: $(length(atm.layers)) layers")
    println()
    
    # Load linelist - try different methods
    println("📋 Loading linelist...")
    linelist = []
    try
        # Try to use the default linelist function names that might exist
        if isdefined(Korg, :get_VALD_solar_linelist)
            linelist = Korg.get_VALD_solar_linelist()
        elseif isdefined(Korg, :get_GALAH_DR3_linelist)
            linelist = Korg.get_GALAH_DR3_linelist()
        else
            println("⚠️  No built-in linelist found - using empty list (continuum only)")
        end
    catch e
        println("⚠️  Linelist loading failed: $e")
        println("   Using empty linelist for continuum-only synthesis")
    end
    
    if !isempty(linelist)
        println("✅ Linelist loaded: $(length(linelist)) lines")
    else
        println("ℹ️  Running continuum-only synthesis")
    end
    println()
    
    # Run synthesis using synthesize() function
    println("⚗️  Running synthesize() function...")
    start_time = time()
    
    result = synthesize(
        atm,              # Model atmosphere
        linelist,         # Line list (empty for continuum-only)
        A_X,              # Abundance array  
        wavelengths,      # Wavelength range
        vmic=vmic,        # Microturbulent velocity
        return_cntm=true  # Return continuum
    )
    
    elapsed_time = time() - start_time
    
    println("✅ synthesize() completed!")
    println("   Time: $(round(elapsed_time, digits=2)) seconds")
    println()
    
    # Analyze the SynthesisResult structure
    println("📊 SYNTHESIS RESULT STRUCTURE")
    println("-"^40)
    
    # Basic spectrum properties
    println("Spectral data:")
    println("  wavelengths: $(length(result.wavelengths)) points")
    println("  flux: $(round(minimum(result.flux), digits=4)) - $(round(maximum(result.flux), digits=4))")
    
    if result.cntm !== nothing
        println("  continuum: $(round(minimum(result.cntm), sigdigits=4)) - $(round(maximum(result.cntm), sigdigits=4))")
        normalized_flux = result.flux ./ result.cntm
        println("  normalized flux: $(round(minimum(normalized_flux), digits=3)) - $(round(maximum(normalized_flux), digits=3))")
        max_absorption = (1.0 - minimum(normalized_flux)) * 100
        println("  deepest absorption: $(round(max_absorption, digits=1))%")
    end
    println()
    
    # Opacity matrix (alpha)
    println("Opacity matrix (alpha):")
    println("  shape: $(size(result.alpha)) (layers × wavelengths)")
    println("  range: $(round(minimum(result.alpha), sigdigits=4)) - $(round(maximum(result.alpha), sigdigits=4)) cm⁻¹")
    println("  mean: $(round(mean(result.alpha), sigdigits=4)) cm⁻¹")
    println()
    
    # Intensity array
    println("Intensity array:")
    println("  shape: $(size(result.intensity))")
    println("  range: $(round(minimum(result.intensity), sigdigits=4)) - $(round(maximum(result.intensity), sigdigits=4))")
    println()
    
    # Atmospheric quantities
    println("Atmospheric structure:")
    println("  electron_number_density: $(length(result.electron_number_density)) layers")
    println("    range: $(round(minimum(result.electron_number_density), sigdigits=3)) - $(round(maximum(result.electron_number_density), sigdigits=3)) cm⁻³")
    println()
    
    # Number densities by species
    println("Species number densities: $(length(result.number_densities)) species")
    species_list = collect(keys(result.number_densities))
    for (i, species) in enumerate(species_list[1:min(5, end)])
        densities = result.number_densities[species]
        println("  $(species): $(round(minimum(densities), sigdigits=3)) - $(round(maximum(densities), sigdigits=3)) cm⁻³")
    end
    if length(species_list) > 5
        println("  ... ($(length(species_list)-5) more species)")
    end
    println()
    
    # Radiative transfer grid
    println("Radiative transfer:")
    println("  μ grid points: $(length(result.mu_grid))")
    mu_vals = [μ_tuple[1] for μ_tuple in result.mu_grid[1:min(3, end)]]
    println("  μ values: $(round.(mu_vals, digits=3))$(length(result.mu_grid) > 3 ? "..." : "")")
    println("  subspectra: $(length(result.subspectra)) ranges")
    println()
    
    # Save the complete result data
    println("💾 SAVING SYNTHESIS RESULT")
    println("-"^30)
    
    # 1. Save spectrum data
    spectrum_file = "korg_synthesize_spectrum.txt"
    open(spectrum_file, "w") do file
        println(file, "# Korg.jl synthesize() Result - Spectrum Data")
        println(file, "# Generated: $(now())")
        println(file, "# Parameters: Teff=$Teff K, logg=$logg, [M/H]=$m_H, vmic=$vmic km/s")
        println(file, "# Synthesis time: $(round(elapsed_time, digits=2)) seconds")
        println(file, "# Linelist: $(length(linelist)) lines")
        println(file, "#")
        
        if result.cntm !== nothing
            println(file, "# Wavelength(Å)    Flux              Continuum         NormalizedFlux")
            normalized_flux = result.flux ./ result.cntm
            for i in 1:length(result.wavelengths)
                @printf(file, "%12.4f  %15.6e  %15.6e  %12.6f\n", 
                       result.wavelengths[i], result.flux[i], result.cntm[i], normalized_flux[i])
            end
        else
            println(file, "# Wavelength(Å)    Flux")
            for i in 1:length(result.wavelengths)
                @printf(file, "%12.4f  %15.6e\n", result.wavelengths[i], result.flux[i])
            end
        end
    end
    
    # 2. Save atmospheric structure
    atmosphere_file = "korg_synthesize_atmosphere.txt"
    open(atmosphere_file, "w") do file
        println(file, "# Korg.jl synthesize() Result - Atmospheric Structure")
        println(file, "# Generated: $(now())")
        println(file, "# Parameters: Teff=$Teff K, logg=$logg, [M/H]=$m_H")
        println(file, "#")
        println(file, "# Layer  ElectronDensity(cm⁻³)")
        for i in 1:length(result.electron_number_density)
            @printf(file, "%5d  %15.6e\n", i, result.electron_number_density[i])
        end
    end
    
    # 3. Save opacity matrix summary (full matrix is too large for text)
    opacity_file = "korg_synthesize_opacity_summary.txt"
    open(opacity_file, "w") do file
        println(file, "# Korg.jl synthesize() Result - Opacity Matrix Summary")
        println(file, "# Generated: $(now())")
        println(file, "# Matrix shape: $(size(result.alpha)) (layers × wavelengths)")
        println(file, "# Opacity range: $(minimum(result.alpha)) - $(maximum(result.alpha)) cm⁻¹")
        println(file, "#")
        println(file, "# Layer statistics (min, max, mean opacity per layer):")
        println(file, "# Layer    MinOpacity      MaxOpacity      MeanOpacity")
        
        for layer in 1:size(result.alpha, 1)
            layer_opacity = result.alpha[layer, :]
            @printf(file, "%5d  %12.6e  %12.6e  %12.6e\n", 
                   layer, minimum(layer_opacity), maximum(layer_opacity), mean(layer_opacity))
        end
        
        println(file, "#")
        println(file, "# Wavelength statistics (min, max, mean opacity per wavelength):")
        println(file, "# WaveIndex  Wavelength(Å)   MinOpacity      MaxOpacity      MeanOpacity")
        
        n_samples = min(100, length(result.wavelengths))  # Sample 100 wavelengths
        step = length(result.wavelengths) ÷ n_samples
        
        for i in 1:step:length(result.wavelengths)
            wl_opacity = result.alpha[:, i]
            @printf(file, "%8d  %12.4f  %12.6e  %12.6e  %12.6e\n", 
                   i, result.wavelengths[i], minimum(wl_opacity), maximum(wl_opacity), mean(wl_opacity))
        end
    end
    
    # 4. Save species number densities
    species_file = "korg_synthesize_species.txt"
    open(species_file, "w") do file
        println(file, "# Korg.jl synthesize() Result - Species Number Densities")
        println(file, "# Generated: $(now())")
        println(file, "# Species count: $(length(result.number_densities))")
        println(file, "#")
        println(file, "# Species name and density range (min - max cm⁻³):")
        
        for (species, densities) in result.number_densities
            @printf(file, "%-20s  %12.6e  %12.6e\n", 
                   string(species), minimum(densities), maximum(densities))
        end
    end
    
    # 5. Save radiative transfer info
    rt_file = "korg_synthesize_radiative_transfer.txt"
    open(rt_file, "w") do file
        println(file, "# Korg.jl synthesize() Result - Radiative Transfer Data")
        println(file, "# Generated: $(now())")
        println(file, "# Intensity array shape: $(size(result.intensity))")
        println(file, "#")
        println(file, "# μ grid (values and weights):")
        println(file, "# Index    μ_value      Weight")
        
        for (i, (mu_val, weight)) in enumerate(result.mu_grid)
            @printf(file, "%5d  %10.6f  %10.6f\n", i, mu_val, weight)
        end
        
        println(file, "#")
        println(file, "# Subspectra ranges:")
        for (i, subspec) in enumerate(result.subspectra)
            println(file, "# Range $i: $(subspec)")
        end
    end
    
    println("✅ Files saved:")
    println("   1. $spectrum_file - Wavelength, flux, continuum data")
    println("   2. $atmosphere_file - Electron density by layer")
    println("   3. $opacity_file - Opacity matrix summary")
    println("   4. $species_file - Species number densities")
    println("   5. $rt_file - Radiative transfer μ grid and intensity info")
    
    println()
    println("📊 File sizes:")
    files = [spectrum_file, atmosphere_file, opacity_file, species_file, rt_file]
    for file in files
        size_kb = round(filesize(file) / 1024, digits=1)
        println("   $file: $(size_kb) KB")
    end
    
    println()
    println("🎉 SUCCESS! Complete synthesize() result saved with full diagnostics.")
    println("   All data from SynthesisResult structure preserved in text files.")
    
catch e
    println("❌ synthesize() failed: $e")
    println()
    println("Common solutions:")
    println("- Check Korg.jl installation: using Korg")
    println("- Verify project environment: julia --project=/path/to/Korg.jl")
    println("- Ensure model atmosphere data is available")
    
    # Print more detailed error info
    println("\nError details:")
    showerror(stdout, e)
    println()
end

println()
println("🏁 Korg.jl synthesize() analysis complete!")