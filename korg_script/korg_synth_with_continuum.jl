#!/usr/bin/env julia
"""
Korg.jl Synthesis with Wavelength, Flux, and Continuum Output
============================================================

This script runs Korg.jl synthesis and saves the results to a text file
with wavelength, normalized flux, and continuum data.
"""

using Korg
using Printf
using Statistics
using Dates

println("🔬 Korg.jl Stellar Synthesis with Continuum Data")
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
println()

try
    # Run synthesis with rectify=false to get continuum
    println("⚗️  Running stellar synthesis (with continuum)...")
    start_time = time()
    
    wl, flux, continuum = synth(
        Teff=Teff,
        logg=logg,
        m_H=m_H,
        wavelengths=wavelengths,
        rectify=false,   # Get absolute flux and continuum
        vmic=vmic
    )
    
    elapsed_time = time() - start_time
    
    println("✅ Synthesis completed!")
    println("   Time: $(round(elapsed_time, digits=2)) seconds")
    println("   Wavelength points: $(length(wl))")
    println("   Absolute flux range: $(round(minimum(flux), sigdigits=4)) - $(round(maximum(flux), sigdigits=4))")
    println("   Continuum range: $(round(minimum(continuum), sigdigits=4)) - $(round(maximum(continuum), sigdigits=4))")
    
    # Calculate normalized flux
    normalized_flux = flux ./ continuum
    
    println("   Normalized flux range: $(round(minimum(normalized_flux), digits=3)) - $(round(maximum(normalized_flux), digits=3))")
    
    # Calculate line statistics from normalized flux
    min_norm_flux = minimum(normalized_flux)
    max_line_depth = (1.0 - min_norm_flux) * 100
    n_strong_lines = sum(normalized_flux .< 0.9)
    n_medium_lines = sum(0.9 .<= normalized_flux .< 0.98)
    
    println("   Deepest absorption: $(round(max_line_depth, digits=1))%")
    println("   Strong lines (>10% depth): $n_strong_lines")
    println("   Medium lines (2-10% depth): $n_medium_lines")
    println("   Mean continuum: $(round(mean(continuum), sigdigits=4))")
    println()
    
    # Find the deepest line
    deepest_idx = argmin(normalized_flux)
    deepest_wl = wl[deepest_idx]
    deepest_norm_flux = normalized_flux[deepest_idx]
    deepest_abs_flux = flux[deepest_idx]
    deepest_continuum = continuum[deepest_idx]
    deepest_depth = (1.0 - deepest_norm_flux) * 100
    
    println("🌟 Deepest absorption line:")
    println("   Wavelength: $(round(deepest_wl, digits=2)) Å")
    println("   Normalized flux: $(round(deepest_norm_flux, digits=3))")
    println("   Absolute flux: $(round(deepest_abs_flux, sigdigits=4))")
    println("   Continuum: $(round(deepest_continuum, sigdigits=4))")
    println("   Depth: $(round(deepest_depth, digits=1))%")
    println()
    
    # Save complete data to text file
    println("💾 Saving complete spectrum data...")
    
    open("korg_spectrum_complete.txt", "w") do file
        println(file, "# Korg.jl Solar Spectrum - Complete Data")
        println(file, "# Generated on: $(now())")
        println(file, "# Parameters: Teff=$Teff K, logg=$logg, [M/H]=$m_H, vmic=$vmic km/s")
        println(file, "# Wavelength range: $wavelengths Å")
        println(file, "# Synthesis time: $(round(elapsed_time, digits=2)) seconds")
        println(file, "# Total wavelength points: $(length(wl))")
        println(file, "# Strong absorption lines (>10% depth): $n_strong_lines")
        println(file, "# Deepest line: $(round(deepest_wl, digits=2)) Å, $(round(deepest_depth, digits=1))% depth")
        println(file, "#")
        println(file, "# Column 1: Wavelength (Å)")
        println(file, "# Column 2: Absolute Flux (erg/s/cm²/Å)")
        println(file, "# Column 3: Continuum Flux (erg/s/cm²/Å)")
        println(file, "# Column 4: Normalized Flux (dimensionless)")
        println(file, "#")
        println(file, "# Wavelength    AbsoluteFlux      ContinuumFlux     NormalizedFlux")
        
        for i in 1:length(wl)
            @printf(file, "%10.4f  %15.6e  %15.6e  %12.6f\n", 
                   wl[i], flux[i], continuum[i], normalized_flux[i])
        end
    end
    
    println("✅ Complete spectrum saved to: korg_spectrum_complete.txt")
    
    # Also save a simple 3-column version
    open("korg_spectrum_simple.txt", "w") do file
        println(file, "# Korg.jl Solar Spectrum - Simple Format")
        println(file, "# Teff=$Teff K, logg=$logg, [M/H]=$m_H")
        println(file, "# Wavelength(Å)  NormalizedFlux  Continuum(erg/s/cm²/Å)")
        for i in 1:length(wl)
            @printf(file, "%10.4f  %12.6f  %15.6e\n", 
                   wl[i], normalized_flux[i], continuum[i])
        end
    end
    
    println("✅ Simple format saved to: korg_spectrum_simple.txt")
    println()
    
    # Show sample data
    println("📊 Sample data (first 10 points):")
    println("   Wavelength    Normalized    Continuum")
    println("       (Å)         Flux      (erg/s/cm²/Å)")
    println("   " * "-"^45)
    for i in 1:10:min(100, length(wl))
        @printf("   %8.2f      %6.3f      %8.2e\n", 
               wl[i], normalized_flux[i], continuum[i])
    end
    
    println()
    println("🎉 Success! Generated solar spectrum with complete flux and continuum data.")
    println("   Files created:")
    println("   - korg_spectrum_complete.txt (4 columns: wavelength, abs flux, continuum, norm flux)")
    println("   - korg_spectrum_simple.txt (3 columns: wavelength, norm flux, continuum)")
    
catch e
    println("❌ Synthesis failed: $e")
    println()
    println("Common solutions:")
    println("- Run: julia --project=/path/to/Korg.jl")
    println("- Check Korg.jl installation")
    println("- Verify model atmosphere data is available")
end

println()
println("🏁 Korg.jl synthesis with continuum data complete!")