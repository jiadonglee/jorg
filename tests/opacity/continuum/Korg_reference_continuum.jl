using Korg
using Printf

println("KORG.JL REFERENCE CALCULATION")
println("="^40)

# Test parameters (matching Jorg exactly)
frequencies_hz = [4.e+14, 5.e+14, 6.e+14, 7.e+14, 8.e+14, 1.e+15, 2.e+15, 3.e+15, 4.e+15, 5.e+15, 6.e+15]
temperature = 5780.0
electron_density = 4280000000000.0

# Species densities
n_h_i = 1.5e+16
n_h_ii = 4280000000000.0
n_he_i = 1000000000000000.0
n_he_ii = 10000000000000.0
n_fe_i = 3000000000000.0
n_fe_ii = 1000000000000.0

println("Parameters:")
println("  Temperature: ", temperature, " K")
println("  Electron density: ", electron_density, " cm⁻³")
println("  H I density: ", n_h_i, " cm⁻³")
println()

# Calculate individual components for comparison
println("Calculating individual components...")

# H I partition function
U_H_I = Korg.default_partition_funcs[Korg.species"H I"](log(temperature))
n_h_i_div_u = n_h_i / U_H_I

try
    # H⁻ bound-free (McLaughlin+ 2017)
    alpha_hminus_bf = Korg.ContinuumAbsorption.Hminus_bf(
        frequencies_hz, temperature, n_h_i_div_u, electron_density
    )
    
    # H⁻ free-free (Bell & Berrington 1987)
    alpha_hminus_ff = Korg.ContinuumAbsorption.Hminus_ff(
        frequencies_hz, temperature, n_h_i_div_u, electron_density
    )
    
    # H I bound-free (Nahar 2021)
    alpha_h_i_bf = Korg.ContinuumAbsorption.H_I_bf(
        frequencies_hz, temperature, n_h_i, n_he_i, electron_density, 1.0/U_H_I
    )
    
    # Calculate total
    alpha_total = alpha_hminus_bf .+ alpha_hminus_ff .+ alpha_h_i_bf
    
    println("✅ Korg.jl calculation successful!")
    println()
    println("COMPONENT PEAKS:")
    println("  H⁻ bf: ", maximum(alpha_hminus_bf), " cm⁻¹")
    println("  H⁻ ff: ", maximum(alpha_hminus_ff), " cm⁻¹")
    println("  H I bf: ", maximum(alpha_h_i_bf), " cm⁻¹")
    println("  TOTAL: ", maximum(alpha_total), " cm⁻¹")
    println()
    
    println("KORG.JL RESULTS:")
    println("Frequency (Hz)      α_korg (cm⁻¹)")
    println("-" ^ 40)
    
    for (i, freq) in enumerate(frequencies_hz)
        @printf "%.1e        %.6e\n" freq alpha_total[i]
    end
    
    # Save alpha_total results to a file
    open("korg_alpha_total.csv", "w") do file
        println(file, "frequency_hz,alpha_total_cm^-1")
        for (i, freq) in enumerate(frequencies_hz)
            println(file, "$freq,$(alpha_total[i])")
        end
    end
    println()
    println("✅ alpha_total results saved to korg_alpha_total.csv")
    
    # Save component breakdown for detailed analysis
    open("korg_components.csv", "w") do file
        println(file, "frequency_hz,hminus_bf,hminus_ff,h_i_bf,total")
        for (i, freq) in enumerate(frequencies_hz)
            println(file, "$freq,$(alpha_hminus_bf[i]),$(alpha_hminus_ff[i]),$(alpha_h_i_bf[i]),$(alpha_total[i])")
        end
    end
    println("✅ Component breakdown saved to korg_components.csv")
    
catch e
    println("❌ Error in Korg.jl calculation: ", e)
end

println()
println("✅ Korg.jl reference completed!")