#!/usr/bin/env julia
"""
Detailed component-by-component analysis of Korg continuum absorption
to identify sources of remaining 4% difference with Jorg
"""

using Pkg
Pkg.activate(".")

using Korg
using JSON
using Statistics

function detailed_continuum_analysis()
    println("=== DETAILED KORG CONTINUUM COMPONENT ANALYSIS ===")
    
    # Test conditions (same as Jorg comparison)
    T = 5778.0  # K
    ne = 1e15   # cm^-3
    nH_I = 1e16 # cm^-3
    
    # Wavelength range
    wavelengths = 5000:20:6000  # Å
    frequencies = reverse(Korg.c_cgs * 1e8 ./ wavelengths)  # Hz (high to low for Korg)
    
    println("Test conditions:")
    println("  Temperature: $T K")
    println("  Electron density: $(ne) cm⁻³") 
    println("  H I density: $(nH_I) cm⁻³")
    println("  Wavelength range: $(minimum(wavelengths))-$(maximum(wavelengths)) Å")
    println()
    
    # Number densities for calculation
    number_densities = Dict(
        Korg.species"H I" => nH_I,
        Korg.species"H II" => 1e12,
        Korg.species"He I" => 0.0,
        Korg.species"H2" => 1e10
    )
    
    # Get partition functions (exact values from Korg)
    partition_funcs = Korg.default_partition_funcs
    U_H_I = partition_funcs[Korg.species"H I"](log(T))
    U_He_I = partition_funcs[Korg.species"He I"](log(T))
    
    println("Partition function values:")
    println("  U(H I) = $U_H_I")
    println("  U(He I) = $U_He_I")
    println()
    
    # Calculate individual continuum components
    println("Calculating individual continuum components...")
    
    # 1. H I bound-free
    alpha_h_i_bf = Korg.ContinuumAbsorption.H_I_bf(
        frequencies, T, nH_I / U_H_I, 0.0, ne, 1.0 / U_H_I
    )
    
    # 2. H^- bound-free  
    alpha_h_minus_bf = Korg.ContinuumAbsorption.Hminus_bf(
        frequencies, T, nH_I / U_H_I, ne
    )
    
    # 3. H^- free-free
    alpha_h_minus_ff = Korg.ContinuumAbsorption.Hminus_ff(
        frequencies, T, nH_I / U_H_I, ne
    )
    
    # 4. Thomson scattering
    σ_thomson = 6.65246e-25  # cm^2 (from Korg source)
    alpha_thomson = ne * σ_thomson
    
    # 5. Rayleigh scattering (H I)
    alpha_rayleigh_h = Korg.ContinuumAbsorption.rayleigh_H_I(
        frequencies, T, nH_I / U_H_I, 1.0 / U_H_I
    )
    
    # Total continuum
    alpha_total = alpha_h_i_bf .+ alpha_h_minus_bf .+ alpha_h_minus_ff .+ 
                  alpha_thomson .+ alpha_rayleigh_h
    
    # Calculate contributions as percentages
    contrib_h_i_bf = 100 * mean(alpha_h_i_bf ./ alpha_total)
    contrib_h_minus_bf = 100 * mean(alpha_h_minus_bf ./ alpha_total)
    contrib_h_minus_ff = 100 * mean(alpha_h_minus_ff ./ alpha_total)
    contrib_thomson = 100 * mean(alpha_thomson ./ alpha_total)
    contrib_rayleigh = 100 * mean(alpha_rayleigh_h ./ alpha_total)
    
    println("Component contributions (% of total):")
    println("  H I bound-free:     $(round(contrib_h_i_bf, digits=2))%")
    println("  H^- bound-free:     $(round(contrib_h_minus_bf, digits=2))%")
    println("  H^- free-free:      $(round(contrib_h_minus_ff, digits=2))%")
    println("  Thomson scattering: $(round(contrib_thomson, digits=2))%")
    println("  Rayleigh scattering: $(round(contrib_rayleigh, digits=2))%")
    println()
    
    # Create detailed output for Jorg comparison
    detailed_data = Dict(
        "wavelengths_angstrom" => collect(wavelengths),
        "frequencies" => collect(frequencies),
        "temperature" => T,
        "electron_density" => ne,
        "h_i_density" => nH_I, 
        "partition_functions" => Dict(
            "U_H_I" => U_H_I,
            "U_He_I" => U_He_I
        ),
        "components" => Dict(
            "alpha_h_i_bf" => alpha_h_i_bf,
            "alpha_h_minus_bf" => alpha_h_minus_bf,
            "alpha_h_minus_ff" => alpha_h_minus_ff,
            "alpha_thomson" => alpha_thomson,
            "alpha_rayleigh_h" => alpha_rayleigh_h,
            "alpha_total" => alpha_total
        ),
        "contributions_percent" => Dict(
            "h_i_bf" => contrib_h_i_bf,
            "h_minus_bf" => contrib_h_minus_bf,
            "h_minus_ff" => contrib_h_minus_ff,
            "thomson" => contrib_thomson,
            "rayleigh" => contrib_rayleigh
        )
    )
    
    # Save detailed data
    open("detailed_korg_components.json", "w") do f
        JSON.print(f, detailed_data, 2)
    end
    
    println("Detailed component data saved to detailed_korg_components.json")
    
    # Print some key values for debugging
    println("\nKey values at 5500 Å:")
    mid_idx = length(wavelengths) ÷ 2
    println("  H I bf:     $(alpha_h_i_bf[mid_idx]:.3e) cm⁻¹")
    println("  H^- bf:     $(alpha_h_minus_bf[mid_idx]:.3e) cm⁻¹") 
    println("  H^- ff:     $(alpha_h_minus_ff[mid_idx]:.3e) cm⁻¹")
    println("  Thomson:    $(alpha_thomson:.3e) cm⁻¹")
    println("  Rayleigh:   $(alpha_rayleigh_h[mid_idx]:.3e) cm⁻¹")
    println("  Total:      $(alpha_total[mid_idx]:.3e) cm⁻¹")
    
    return detailed_data
end

if abspath(PROGRAM_FILE) == @__FILE__
    detailed_continuum_analysis()
end