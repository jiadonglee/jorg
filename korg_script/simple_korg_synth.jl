#!/usr/bin/env julia
"""
Simple Julia Script to Run Korg.jl synth() with VALD Linelist
============================================================

This script demonstrates basic usage of Korg.jl stellar synthesis.
"""

using Korg
using Printf
using Dates

println("🔬 Korg.jl Stellar Synthesis")
println("=" ^ 50)

# CONFIGURATION
# =============
# Stellar type selection (change this to switch stellar types)
stellar_type = "metal_poor_k_giant"  # Options: "solar", "arcturus", "metal_poor_k_giant", "custom"

# Predefined stellar types
stellar_params = Dict(
    "solar" => (Teff=5771, logg=4.44, m_H=0.0),
    "arcturus" => (Teff=4250, logg=1.4, m_H=-0.5),
    "metal_poor_k_giant" => (Teff=4500, logg=1.5, m_H=-2.5),
    "custom" => (Teff=5780, logg=4.44, m_H=0.0)  # Edit these for custom parameters
)

# Get parameters for selected stellar type
params = stellar_params[stellar_type]
Teff = params.Teff      # Effective temperature (K)
logg = params.logg      # Surface gravity
m_H = params.m_H        # Metallicity [M/H]

# Other synthesis parameters
wavelengths = (5000, 5200)  # Wavelength range (Å)

println("Stellar type: $stellar_type")
println("Parameters:")
println("  Teff: $Teff K")
println("  logg: $logg")
println("  [M/H]: $m_H")
println("  Wavelengths: $wavelengths Å")
println()

try
    # Load VALD linelist
    println("📋 Loading VALD linelist...")
    linelist = []
    try
        # Try to find VALD linelist in common locations
        vald_paths = [
            "/Users/jdli/Project/Korg.jl/data/linelists/vald_extract_stellar_solar_threshold001.vald",
            "/Users/jdli/Project/Korg.jl/misc/Tutorial notebooks/basics/linelist.vald",
            joinpath(dirname(dirname(@__DIR__)), "data", "linelists", "vald_extract_stellar_solar_threshold001.vald"),
            joinpath(dirname(dirname(@__DIR__)), "misc", "Tutorial notebooks", "basics", "linelist.vald")
        ]
        
        loaded = false
        for vald_path in vald_paths
            if isfile(vald_path)
                linelist = read_linelist(vald_path)
                println("✅ Loaded VALD linelist from: $vald_path")
                println("   Lines loaded: $(length(linelist))")
                loaded = true
                break
            end
        end
        
        if !loaded
            println("⚠️  VALD linelist file not found in standard locations")
            println("   Using empty linelist for continuum-only synthesis")
        end
    catch e
        println("⚠️  VALD linelist loading failed: $e")
        println("   Using empty linelist for continuum-only synthesis")
    end
    
    # Create atmosphere and abundance array
    println("🌍 Creating model atmosphere...")
    A_X = format_A_X(m_H)
    atm = interpolate_marcs(Teff, logg, A_X)
    println("✅ Atmosphere created: $(length(atm.layers)) layers")
    
    # Run synthesis with VALD linelist
    println("⚗️  Running stellar synthesis with VALD linelist...")
    start_time = time()
    
    result = synthesize(
        atm,              # Model atmosphere
        linelist,         # VALD line list
        A_X,              # Abundance array  
        wavelengths,      # Wavelength range
        vmic=1.0,         # Microturbulent velocity
        return_cntm=true  # Return continuum
    )
    
    # Extract wavelengths and flux
    wl = result.wavelengths
    flux = result.flux
    continuum = result.cntm
    
    # Normalize flux by continuum if available
    if continuum !== nothing
        flux = flux ./ continuum
    end
    
    elapsed_time = time() - start_time
    
    println("✅ Synthesis completed!")
    println("   Time: $(round(elapsed_time, digits=2)) seconds")
    println("   Wavelength points: $(length(wl))")
    println("   Flux range: $(round(minimum(flux), digits=3)) - $(round(maximum(flux), digits=3))")
    
    # Calculate line statistics
    min_flux = minimum(flux)
    max_line_depth = (1.0 - min_flux) * 100
    n_strong_lines = sum(flux .< 0.9)
    
    println("   Deepest absorption: $(round(max_line_depth, digits=1))%")
    println("   Strong lines (>10% depth): $n_strong_lines")
    
    if continuum !== nothing
        println("   Continuum range: $(minimum(continuum)) - $(maximum(continuum))")
    end
    
    println()
    
    # Save spectrum to text file
    println("� Saving spectrum to file...")
    
    # Capitalize stellar type for display
    stellar_display = replace(stellar_type, "_" => " ") |> titlecase
    
    # Save spectrum data
    spectrum_filename = "korg_$(stellar_type)_spectrum.txt"
    open(spectrum_filename, "w") do file
        println(file, "# Korg.jl $stellar_display Spectrum (VALD Linelist)")
        println(file, "# Teff=$(Teff)K, logg=$(logg), [M/H]=$m_H")
        println(file, "# Generated: $(now())")
        println(file, "# VALD lines used: $(length(linelist))")
        println(file, "# Synthesis time: $(round(elapsed_time, digits=2)) seconds")
        println(file, "# Strong lines (>10% depth): $n_strong_lines")
        println(file, "# Deepest absorption: $(round(max_line_depth, digits=1))%")
        println(file, "#")
        println(file, "# Wavelength(Å)    NormalizedFlux")
        
        for i in 1:length(wl)
            @printf(file, "%12.4f  %12.6f\n", wl[i], flux[i])
        end
    end
    
    println("✅ Spectrum saved as: $spectrum_filename")
    
    # Create a simple ASCII plot preview (first 20 points)
    println("\n📊 Spectrum preview (first 20 wavelength points):")
    println("Wavelength(Å)  Flux     ASCII Plot")
    println("-" ^ 45)
    
    for i in 1:min(20, length(wl))
        # Create simple ASCII bar (scaled to 40 characters)
        bar_length = round(Int, flux[i] * 40)
        bar = "█" ^ bar_length * "░" ^ (40 - bar_length)
        @printf("%10.2f    %.3f   %s\n", wl[i], flux[i], bar)
    end
    
    if length(wl) > 20
        println("... ($(length(wl)-20) more points in file)")
    end
    
    println()
    println("🎉 VALD synthesis complete! Spectrum shows $(n_strong_lines) strong absorption lines from $(length(linelist)) VALD lines.")
    
catch e
    println("❌ Synthesis failed: $e")
    println()
    println("Common issues:")
    println("- Make sure Korg.jl is properly installed")
    println("- Check that you're in the correct project environment")
    println("- Verify model atmosphere files are available")
end

println()
println("🏁 Done!")