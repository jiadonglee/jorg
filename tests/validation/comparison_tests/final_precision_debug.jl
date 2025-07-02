#!/usr/bin/env julia
"""
Final precision debug to get Korg-Jorg difference below 1%
Focus on the largest components: H^- bound-free and free-free
"""

using Pkg
Pkg.activate(".")

using Korg
using JSON

function detailed_component_comparison()
    println("=== FINAL PRECISION DEBUG ===")
    
    # Test conditions
    T = 5778.0  # K
    ne = 1e15   # cm^-3
    nH_I = 1e16 # cm^-3
    
    # Single wavelength for detailed analysis
    wavelength = 5500.0  # Å
    frequency = Korg.c_cgs * 1e8 / wavelength  # Hz
    frequencies = [frequency]
    
    println("Test conditions:")
    println("  Temperature: $T K")
    println("  Electron density: $(ne) cm⁻³") 
    println("  H I density: $(nH_I) cm⁻³")
    println("  Wavelength: $wavelength Å")
    println()
    
    # Get exact partition functions
    partition_funcs = Korg.default_partition_funcs
    U_H_I = partition_funcs[Korg.species"H I"](log(T))
    nH_I_div_U = nH_I / U_H_I
    inv_U_H = 1.0 / U_H_I
    
    println("Exact parameters:")
    println("  U(H I): $U_H_I")
    println("  n(H I)/U(H I): $(nH_I_div_U) cm⁻³")
    println("  1/U(H I): $inv_U_H")
    println()
    
    # Calculate individual Korg components
    println("Korg individual components:")
    
    # 1. H^- bound-free
    α_h_minus_bf_korg = Korg.ContinuumAbsorption.Hminus_bf(
        frequencies, T, nH_I_div_U, ne
    )[1]
    
    # 2. H^- free-free
    α_h_minus_ff_korg = Korg.ContinuumAbsorption.Hminus_ff(
        frequencies, T, nH_I_div_U, ne
    )[1]
    
    # 3. H I bound-free
    α_h_i_bf_korg = Korg.ContinuumAbsorption.H_I_bf(
        frequencies, T, nH_I, 0.0, ne, inv_U_H
    )[1]
    
    # 4. Thomson scattering
    σ_thomson = 6.65246e-25  # cm^2
    α_thomson_korg = ne * σ_thomson
    
    # Total
    α_total_korg = α_h_minus_bf_korg + α_h_minus_ff_korg + α_h_i_bf_korg + α_thomson_korg
    
    println("  H^- bound-free: $(α_h_minus_bf_korg) cm⁻¹")
    println("  H^- free-free:  $(α_h_minus_ff_korg) cm⁻¹")
    println("  H I bound-free: $(α_h_i_bf_korg) cm⁻¹")
    println("  Thomson:        $(α_thomson_korg) cm⁻¹")
    println("  Total (sum):    $(α_total_korg) cm⁻¹")
    println()
    
    # Component fractions
    frac_h_minus_bf = 100 * α_h_minus_bf_korg / α_total_korg
    frac_h_minus_ff = 100 * α_h_minus_ff_korg / α_total_korg
    frac_h_i_bf = 100 * α_h_i_bf_korg / α_total_korg
    frac_thomson = 100 * α_thomson_korg / α_total_korg
    
    println("Korg component fractions:")
    println("  H^- bound-free: $(round(frac_h_minus_bf, digits=2))%")
    println("  H^- free-free:  $(round(frac_h_minus_ff, digits=2))%")
    println("  H I bound-free: $(round(frac_h_i_bf, digits=2))%")
    println("  Thomson:        $(round(frac_thomson, digits=2))%")
    println()
    
    # Compare with Korg total_continuum_absorption
    number_densities = Dict(
        Korg.species"H I" => nH_I,
        Korg.species"H II" => 1e12,
        Korg.species"He I" => 0.0,
        Korg.species"H2" => 1e10
    )
    
    # Use the internal total_continuum_absorption
    α_total_func_korg = Korg.ContinuumAbsorption.total_continuum_absorption(
        frequencies, T, ne, number_densities, partition_funcs
    )[1]
    
    println("Korg total comparison:")
    println("  Component sum:     $(α_total_korg) cm⁻¹")
    println("  total_continuum:   $(α_total_func_korg) cm⁻¹")
    println("  Difference:        $(100*(α_total_func_korg - α_total_korg)/α_total_func_korg)%")
    println()
    
    # Create reference data for Jorg debugging
    reference_data = Dict(
        "wavelength" => wavelength,
        "frequency" => frequency,
        "temperature" => T,
        "electron_density" => ne,
        "h_i_density" => nH_I,
        "partition_functions" => Dict(
            "U_H_I" => U_H_I
        ),
        "korg_components" => Dict(
            "h_minus_bf" => α_h_minus_bf_korg,
            "h_minus_ff" => α_h_minus_ff_korg,
            "h_i_bf" => α_h_i_bf_korg,
            "thomson" => α_thomson_korg,
            "total_sum" => α_total_korg,
            "total_function" => α_total_func_korg
        ),
        "component_fractions" => Dict(
            "h_minus_bf" => frac_h_minus_bf,
            "h_minus_ff" => frac_h_minus_ff,
            "h_i_bf" => frac_h_i_bf,
            "thomson" => frac_thomson
        )
    )
    
    # Save for Jorg comparison
    open("korg_detailed_reference.json", "w") do f
        JSON.print(f, reference_data, 2)
    end
    
    println("Detailed reference data saved to korg_detailed_reference.json")
    
    return reference_data
end

if abspath(PROGRAM_FILE) == @__FILE__
    detailed_component_comparison()
end