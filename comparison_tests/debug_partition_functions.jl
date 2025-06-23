#!/usr/bin/env julia
"""
Debug partition function differences between Korg and Jorg
"""

using Pkg
Pkg.activate(".")

using Korg
using JSON

function debug_partition_functions()
    println("=== PARTITION FUNCTION DEBUG ===")
    
    T = 5778.0  # K
    
    # Get Korg's exact partition functions
    partition_funcs = Korg.default_partition_funcs
    
    # Calculate exact values for key species
    log_T = log(T)
    U_H_I = partition_funcs[Korg.species"H I"](log_T)
    U_H_II = partition_funcs[Korg.species"H II"](log_T)
    U_He_I = partition_funcs[Korg.species"He I"](log_T)
    
    println("Korg partition functions at T = $T K:")
    println("  U(H I)  = $U_H_I")
    println("  U(H II) = $U_H_II") 
    println("  U(He I) = $U_He_I")
    println()
    
    # Compare with simple approximations used in Jorg
    println("Simple approximations used in Jorg:")
    println("  U(H I)  = 2.0")
    println("  U(H II) = 1.0")
    println("  U(He I) = 1.0")
    println()
    
    # Calculate errors
    err_H_I = 100 * abs(U_H_I - 2.0) / U_H_I
    err_H_II = 100 * abs(U_H_II - 1.0) / U_H_II
    err_He_I = 100 * abs(U_He_I - 1.0) / U_He_I
    
    println("Errors in simple approximations:")
    println("  H I error:  $(round(err_H_I, digits=6))%")
    println("  H II error: $(round(err_H_II, digits=6))%")
    println("  He I error: $(round(err_He_I, digits=6))%")
    
    # Create accurate partition function data for Jorg
    partition_data = Dict(
        "temperature" => T,
        "log_temperature" => log_T,
        "exact_values" => Dict(
            "U_H_I" => U_H_I,
            "U_H_II" => U_H_II,
            "U_He_I" => U_He_I
        ),
        "approximations" => Dict(
            "U_H_I" => 2.0,
            "U_H_II" => 1.0,
            "U_He_I" => 1.0
        ),
        "errors_percent" => Dict(
            "H_I" => err_H_I,
            "H_II" => err_H_II,
            "He_I" => err_He_I
        )
    )
    
    # Save data
    open("partition_function_debug.json", "w") do f
        JSON.print(f, partition_data, 2)
    end
    
    println("\nPartition function data saved to partition_function_debug.json")
    
    return U_H_I, U_H_II, U_He_I
end

if abspath(PROGRAM_FILE) == @__FILE__
    debug_partition_functions()
end