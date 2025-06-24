#!/usr/bin/env julia
"""
Get Korg's H I bound-free calculation for comparison
"""

using Pkg
Pkg.activate(".")

using Korg
using JSON

function get_korg_h_i_bf()
    println("=== KORG H I BOUND-FREE REFERENCE ===")
    
    # Test conditions (same as Jorg)
    T = 5778.0  # K
    ne = 1e15   # cm^-3
    nH_I = 1e16 # cm^-3
    
    # Single wavelength test
    wavelength = 5500.0  # Å
    frequency = Korg.c_cgs * 1e8 / wavelength  # Hz
    frequencies = [frequency]
    
    println("Test conditions:")
    println("  Temperature: $T K")
    println("  Electron density: $(ne) cm⁻³") 
    println("  H I density: $(nH_I) cm⁻³")
    println("  Wavelength: $wavelength Å")
    println("  Frequency: $(frequency) Hz")
    println()
    
    # Get partition functions
    partition_funcs = Korg.default_partition_funcs
    U_H_I = partition_funcs[Korg.species"H I"](log(T))
    
    println("Partition function:")
    println("  U(H I) = $U_H_I")
    println()
    
    # Calculate H I bound-free directly
    α_h_i_bf = Korg.ContinuumAbsorption.H_I_bf(
        frequencies, T, nH_I, 0.0, ne, 1.0 / U_H_I
    )
    
    println("Korg H I bound-free results:")
    println("  Alpha: $(α_h_i_bf[1]) cm⁻¹")
    println()
    
    # Compare with total continuum
    number_densities = Dict(
        Korg.species"H I" => nH_I,
        Korg.species"H II" => 1e12,
        Korg.species"He I" => 0.0,
        Korg.species"H2" => 1e10
    )
    
    α_total = total_continuum_absorption(
        frequencies, T, ne, number_densities, partition_funcs
    )
    
    println("Total continuum comparison:")
    println("  H I bf: $(α_h_i_bf[1]) cm⁻¹")
    println("  Total:  $(α_total[1]) cm⁻¹")
    println("  H I bf fraction: $(100 * α_h_i_bf[1] / α_total[1])%")
    
    return α_h_i_bf[1], α_total[1]
end

if abspath(PROGRAM_FILE) == @__FILE__
    get_korg_h_i_bf()
end