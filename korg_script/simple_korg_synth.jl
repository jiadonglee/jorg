#!/usr/bin/env julia
"""
Simple Julia Script to Run Korg.jl synth() with VALD Linelist
============================================================

This script demonstrates basic usage of Korg.jl stellar synthesis.
"""

using Korg
using Plots

println("🔬 Korg.jl Stellar Synthesis")
println("=" * 50)

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
    # Run synthesis with built-in linelist
    println("⚗️  Running stellar synthesis...")
    start_time = time()
    
    wl, flux, continuum = synth(
        Teff=Teff,
        logg=logg,
        m_H=m_H,
        wavelengths=wavelengths,
        rectify=true,    # Continuum normalize
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
    
    println("   Deepest absorption: $(round(max_line_depth, digits=1))%")
    println("   Strong lines (>10% depth): $n_strong_lines")
    
    if continuum !== nothing
        println("   Continuum range: $(minimum(continuum)) - $(maximum(continuum))")
    end
    
    println()
    
    # Create plot
    println("📊 Creating spectrum plot...")
    
    p = plot(wl, flux, 
             linewidth=1.0,
             color=:blue,
             alpha=0.8,
             xlabel="Wavelength (Å)",
             ylabel="Normalized Flux",
             title="Korg.jl Solar Spectrum\nTeff=$(Teff)K, logg=$(logg), [M/H]=$m_H",
             grid=true,
             gridwidth=1,
             gridalpha=0.3,
             size=(800, 600))
    
    # Set y-axis limits for normalized spectrum
    ylims!(p, (0, 1.1))
    
    # Save plot
    savefig(p, "korg_solar_spectrum.png")
    println("✅ Plot saved as: korg_solar_spectrum.png")
    
    # Display plot (if in interactive environment)
    display(p)
    
    println()
    println("🎉 Synthesis complete! Spectrum shows $(n_strong_lines) strong absorption lines.")
    
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