#!/usr/bin/env julia
"""
Basic Julia Script to Run Korg.jl synth() 
=========================================

Simple demonstration of Korg.jl stellar synthesis without additional dependencies.
"""

using Korg
using Printf
using Statistics

println("🔬 Korg.jl Stellar Synthesis")
println("="^50)

# Synthesis parameters
Teff = 5780      # Effective temperature (K)
logg = 4.44      # Surface gravity
m_H = 0.0        # Metallicity [M/H]
wavelengths = (5000, 5200)  # Wavelength range (Å)

println("Parameters:")
println("  Teff: $Teff K")
println("  logg: $logg")
println("  [M/H]: $m_H")
println("  Wavelengths: $wavelengths Å")
println()

try
    # Run synthesis
    println("⚗️  Running stellar synthesis...")
    start_time = time()
    
    rectify = true  # Store rectify setting
    
    wl, flux, continuum = synth(
        Teff=Teff,
        logg=logg,
        m_H=m_H,
        wavelengths=wavelengths,
        rectify=rectify, # Continuum normalize
        vmic=1.0         # Microturbulence (km/s)
    )
    
    elapsed_time = time() - start_time
    
    println("✅ Synthesis completed!")
    println("   Time: $(round(elapsed_time, digits=2)) seconds")
    println("   Wavelength points: $(length(wl))")
    println("   Flux range: $(round(minimum(flux), digits=3)) - $(round(maximum(flux), digits=3))")
    
    # Calculate line statistics
    min_flux = minimum(flux)
    max_line_depth = (1.0 - min_flux) * 100
    n_strong_lines = sum(flux .< 0.9)
    n_medium_lines = sum(0.9 .<= flux .< 0.98)
    
    println("   Deepest absorption: $(round(max_line_depth, digits=1))%")
    println("   Strong lines (>10% depth): $n_strong_lines")
    println("   Medium lines (2-10% depth): $n_medium_lines")
    
    if continuum !== nothing
        println("   Mean continuum: $(round(mean(continuum), sigdigits=4))")
    end
    
    println()
    
    # Show some example wavelengths and fluxes
    println("📊 Sample spectrum points:")
    println("   Wavelength (Å)  |  Normalized Flux")
    println("   " * "-"^37)
    
    # Show every 1000th point
    step = max(1, length(wl) ÷ 10)
    for i in 1:step:length(wl)
        @printf "   %8.2f        |    %6.3f\n" wl[i] flux[i]
    end
    
    println()
    
    # Find the deepest line
    deepest_idx = argmin(flux)
    deepest_wl = wl[deepest_idx]
    deepest_flux = flux[deepest_idx]
    deepest_depth = (1.0 - deepest_flux) * 100
    
    println("🌟 Deepest absorption line:")
    println("   Wavelength: $(round(deepest_wl, digits=2)) Å")
    println("   Flux: $(round(deepest_flux, digits=3))")
    println("   Depth: $(round(deepest_depth, digits=1))%")
    
    # Save data to text file
    println()
    println("💾 Saving spectrum data...")
    
    open("korg_spectrum.txt", "w") do file
        println(file, "# Korg.jl Solar Spectrum")
        println(file, "# Teff=$Teff K, logg=$logg, [M/H]=$m_H")
        println(file, "# Rectify=$rectify, vmic=1.0 km/s")
        if continuum !== nothing
            println(file, "# Wavelength(Å)  NormalizedFlux  Continuum(erg/s/cm²/Å)")
            for i in 1:length(wl)
                @printf(file, "%10.4f  %12.6f  %15.6e\n", wl[i], flux[i], continuum[i])
            end
        else
            println(file, "# Wavelength(Å)  NormalizedFlux")
            for i in 1:length(wl)
                @printf(file, "%10.4f  %12.6f\n", wl[i], flux[i])
            end
        end
    end
    
    println("✅ Spectrum saved to: korg_spectrum.txt")
    println()
    println("🎉 Success! Generated solar spectrum with $n_strong_lines strong absorption lines.")
    println("   Deepest line at $(round(deepest_wl, digits=1)) Å with $(round(deepest_depth, digits=1))% absorption.")
    
catch e
    println("❌ Synthesis failed: $e")
    println()
    println("Common solutions:")
    println("- Run: julia --project=/path/to/Korg.jl")
    println("- Check Korg.jl installation with: using Korg")
    println("- Verify model atmosphere data is available")
end

println()
println("🏁 Basic Korg.jl synthesis complete!")