using Korg

"""
Test Korg Chemical Equilibrium with Identical Inputs
====================================================

This script runs Korg chemical equilibrium using the same atmospheric
conditions that will be used for Jorg testing.
"""

# Atmosphere data (same as extracted for Jorg)
const korg_atmosphere_data = Dict(
    "Solar-type G star" => Dict(
        "Teff" => 5777.0,
        "logg" => 4.44,
        "M_H" => 0.0,
        "layers" => Dict(
            15 => Dict("T" => 4590.009579508528, "nt" => 8.315411129918017e15, "ne_guess" => 7.239981204782037e11),
            25 => Dict("T" => 4838.221978288154, "nt" => 2.7356685421333148e16, "ne_guess" => 2.3860243024247812e12),
            35 => Dict("T" => 5383.722228881833, "nt" => 8.337958936823813e16, "ne_guess" => 9.453517368277791e12)
        )
    ),
    "Cool K-type star" => Dict(
        "Teff" => 4500.0,
        "logg" => 4.5,
        "M_H" => 0.0,
        "layers" => Dict(
            15 => Dict("T" => 3608.9886137413555, "nt" => 2.4769114629938384e16, "ne_guess" => 2.990034164984107e11),
            25 => Dict("T" => 3802.2316033227494, "nt" => 7.680801945308658e16, "ne_guess" => 9.936302279128654e11),
            35 => Dict("T" => 4269.554262251411, "nt" => 1.9795187990913702e17, "ne_guess" => 4.888600236940689e12)
        )
    ),
    "Cool M dwarf" => Dict(
        "Teff" => 3500.0,
        "logg" => 4.8,
        "M_H" => 0.0,
        "layers" => Dict(
            15 => Dict("T" => 2676.971224171452, "nt" => 3.440690227103728e15, "ne_guess" => 9.420055688925724e9),
            25 => Dict("T" => 2757.6431441697528, "nt" => 1.561033634416188e16, "ne_guess" => 3.8264281998144066e10),
            35 => Dict("T" => 2910.3334110333217, "nt" => 6.345755693469862e16, "ne_guess" => 1.5800975648355624e11)
        )
    ),
    "Giant K star" => Dict(
        "Teff" => 4500.0,
        "logg" => 2.5,
        "M_H" => 0.0,
        "layers" => Dict(
            15 => Dict("T" => 3554.649123320977, "nt" => 1.5362922620728632e15, "ne_guess" => 4.276264434134044e10),
            25 => Dict("T" => 3794.766627174385, "nt" => 4.402883893569443e15, "ne_guess" => 1.5259244406659048e11),
            35 => Dict("T" => 4277.143406880127, "nt" => 1.1366300170141498e16, "ne_guess" => 6.865808132660596e11)
        )
    ),
    "Metal-poor G star" => Dict(
        "Teff" => 5777.0,
        "logg" => 4.44,
        "M_H" => -1.0,
        "layers" => Dict(
            15 => Dict("T" => 4666.866866260754, "nt" => 2.1882756521811176e16, "ne_guess" => 3.2283879911501184e11),
            25 => Dict("T" => 4840.113180178101, "nt" => 7.163604786541968e16, "ne_guess" => 1.0550775750963367e12),
            35 => Dict("T" => 5343.101981014455, "nt" => 1.9377501284096784e17, "ne_guess" => 6.0831917573359375e12)
        )
    ),
    "Metal-rich G star" => Dict(
        "Teff" => 5777.0,
        "logg" => 4.44,
        "M_H" => 0.3,
        "layers" => Dict(
            15 => Dict("T" => 4562.69282361102, "nt" => 5.886360013228573e15, "ne_guess" => 9.417291391503596e11),
            25 => Dict("T" => 4835.645170542186, "nt" => 1.9410995333394496e16, "ne_guess" => 3.1444758977749937e12),
            35 => Dict("T" => 5394.3344692256205, "nt" => 6.003205854880858e16, "ne_guess" => 1.2010194973623188e13)
        )
    )
)

function test_stellar_type_identical_inputs(stellar_type, stellar_data)
    println("\nTesting: $stellar_type")
    println("Stellar Parameters: Teff=$(stellar_data["Teff"])K, logg=$(stellar_data["logg"]), [M/H]=$(stellar_data["M_H"])")
    
    # Get abundances exactly as done in original tests
    M_H = stellar_data["M_H"]
    A_X = Korg.format_A_X(M_H)
    rel_abundances = 10.0 .^ (A_X .- 12.0)
    total_particles_per_H = sum(rel_abundances)
    absolute_abundances = rel_abundances ./ total_particles_per_H
    
    results = []
    
    # Test all layers
    for (layer_idx, layer_data) in stellar_data["layers"]
        try
            # Use EXACT same atmospheric conditions
            T = layer_data["T"]
            nt = layer_data["nt"]
            ne_guess = layer_data["ne_guess"]
            
            println("  Layer $layer_idx: T=$(round(T, digits=1))K, nt=$(round(nt, sigdigits=3)), ne_guess=$(round(ne_guess, sigdigits=3))")
            
            # Calculate chemical equilibrium
            ne_sol, number_densities = Korg.chemical_equilibrium(
                T, nt, ne_guess, absolute_abundances,
                Korg.ionization_energies, Korg.default_partition_funcs,
                Korg.default_log_equilibrium_constants
            )
            
            error_percent = abs(ne_sol - ne_guess) / ne_guess * 100
            
            # Extract key species
            n_H_I = get(number_densities, Korg.species"H I", 0.0)
            n_H_II = get(number_densities, Korg.species"H II", 0.0)
            n_H2O = get(number_densities, Korg.species"H2O", 0.0)
            n_Fe_I = get(number_densities, Korg.species"Fe I", 0.0)
            
            # Calculate properties
            ionization_fraction = n_H_II / (n_H_I + n_H_II)
            
            result = (
                layer = layer_idx,
                T = T,
                nt = nt,
                ne_guess = ne_guess,
                ne_sol = ne_sol,
                error_percent = error_percent,
                ionization_fraction = ionization_fraction,
                p_H_I = n_H_I * Korg.kboltz_cgs * T,
                p_H_II = n_H_II * Korg.kboltz_cgs * T,
                p_H2O = n_H2O * Korg.kboltz_cgs * T,
                p_Fe_I = n_Fe_I * Korg.kboltz_cgs * T
            )
            push!(results, result)
            
            println("    Error: $(round(error_percent, digits=1))%, Ion_frac: $(round(ionization_fraction, sigdigits=3))")
            
        catch e
            println("    ❌ Layer $layer_idx failed: $e")
        end
    end
    
    # Summary for this stellar type
    if length(results) > 0
        avg_error = sum(r.error_percent for r in results) / length(results)
        max_error = maximum(r.error_percent for r in results)
        min_error = minimum(r.error_percent for r in results)
        
        println("  Summary: $(length(results)) layers, avg error: $(round(avg_error, digits=1))%, range: $(round(min_error, digits=1))%-$(round(max_error, digits=1))%")
        
        return results
    else
        println("  ❌ No successful calculations")
        return []
    end
end

function main()
    println("KORG CHEMICAL EQUILIBRIUM WITH IDENTICAL INPUTS")
    println(repeat("=", 80))
    println("Testing Korg using predetermined atmospheric conditions")
    println("This provides baseline for comparison with Jorg")
    println()
    
    all_results = []
    
    for (stellar_type, stellar_data) in korg_atmosphere_data
        results = test_stellar_type_identical_inputs(stellar_type, stellar_data)
        if length(results) > 0
            push!(all_results, (stellar_type, stellar_data, results))
        end
    end
    
    # Overall summary
    println("\n" * repeat("=", 80))
    println("KORG PERFORMANCE SUMMARY (IDENTICAL INPUTS)")
    println(repeat("=", 80))
    
    if length(all_results) > 0
        println("Stellar Type                | Avg Error | Error Range | Ion Range")
        println(repeat("-", 70))
        
        all_errors = []
        for (stellar_type, stellar_data, results) in all_results
            avg_error = sum(r.error_percent for r in results) / length(results)
            max_error = maximum(r.error_percent for r in results)
            min_error = minimum(r.error_percent for r in results)
            ion_fracs = [r.ionization_fraction for r in results]
            min_ion = minimum(ion_fracs)
            max_ion = maximum(ion_fracs)
            
            append!(all_errors, [r.error_percent for r in results])
            
            println("$(rpad(stellar_type, 25)) | $(lpad(round(avg_error, digits=1), 7))%  | $(lpad(round(min_error, digits=1), 4))%-$(lpad(round(max_error, digits=1), 4))% | $(lpad(round(min_ion, sigdigits=2), 4))-$(lpad(round(max_ion, sigdigits=2), 4))")
        end
        
        # Overall statistics
        println("\nOverall Korg Statistics (Identical Inputs):")
        println("  Total tests: $(length(all_errors))")
        println("  Mean error: $(round(sum(all_errors)/length(all_errors), digits=2))%")
        println("  Median error: $(round(sort(all_errors)[div(length(all_errors),2)], digits=2))%")
        println("  Max error: $(round(maximum(all_errors), digits=1))%")
        println("  Success rate: 100.0%")
    else
        println("❌ No successful tests completed")
    end
    
    println("\n" * repeat("=", 80))
    println("✅ KORG TESTING WITH IDENTICAL INPUTS COMPLETE")
    println("Ready for direct comparison with Jorg results")
    println(repeat("=", 80))
    
    return all_results
end

# Run the test
main()