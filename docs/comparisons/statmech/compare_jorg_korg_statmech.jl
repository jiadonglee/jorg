#!/usr/bin/env julia
"""
Comprehensive comparison between Jorg (Python) and Korg.jl statistical mechanics modules.

This script runs identical calculations in both Jorg and Korg.jl and compares:
1. Physical constants
2. Translational partition functions  
3. Saha equation calculations
4. Chemical equilibrium solver
5. Performance benchmarks
"""

using Korg
using JSON
using Statistics
using Printf
using Dates

# Test conditions covering stellar atmosphere parameter space
const STELLAR_CONDITIONS = [
    (T=3500.0, ne=5e10, log_g=5.0, name="M_dwarf"),
    (T=4500.0, ne=2e11, log_g=4.5, name="K_dwarf"), 
    (T=5778.0, ne=1e13, log_g=4.44, name="Sun"),
    (T=7000.0, ne=5e13, log_g=4.2, name="F_star"),
    (T=9000.0, ne=2e14, log_g=4.0, name="A_star"),
]

# Key elements for testing
const TEST_ELEMENTS = [1, 2, 6, 8, 11, 12, 13, 14, 16, 20, 22, 26, 28]

function test_physical_constants()
    """Test physical constants against Korg.jl values."""
    println("=== Comparing Physical Constants ===")
    
    korg_constants = (
        kboltz_cgs = Korg.kboltz_cgs,
        kboltz_eV = Korg.kboltz_eV,
        hplanck_cgs = Korg.hplanck_cgs,
        electron_mass_cgs = Korg.electron_mass_cgs,
        eV_to_cgs = Korg.eV_to_cgs
    )
    
    # Read Jorg constants from Python output
    jorg_output = read(`python -c "
import sys
sys.path.insert(0, 'Jorg/src')
from jorg.constants import kboltz_cgs, kboltz_eV, hplanck_cgs, me_cgs, EV_TO_ERG
from jorg.statmech.saha_equation import KORG_KBOLTZ_CGS, KORG_KBOLTZ_EV, KORG_HPLANCK_CGS, KORG_ELECTRON_MASS_CGS

print('kboltz_cgs:', float(kboltz_cgs))
print('kboltz_eV:', float(kboltz_eV)) 
print('hplanck_cgs:', float(hplanck_cgs))
print('electron_mass_cgs:', float(me_cgs))
print('eV_to_cgs:', float(EV_TO_ERG))
print('korg_kboltz_cgs:', float(KORG_KBOLTZ_CGS))
print('korg_kboltz_eV:', float(KORG_KBOLTZ_EV))
print('korg_hplanck_cgs:', float(KORG_HPLANCK_CGS))
print('korg_electron_mass_cgs:', float(KORG_ELECTRON_MASS_CGS))
"`, String)
    
    jorg_constants = Dict{String, Float64}()
    for line in split(jorg_output, '\n')
        if ':' in line
            key, val = split(line, ':')
            jorg_constants[strip(key)] = parse(Float64, strip(val))
        end
    end
    
    results = Dict()
    max_rel_diff = 0.0
    
    comparisons = [
        ("kboltz_cgs", korg_constants.kboltz_cgs, "korg_kboltz_cgs"),
        ("kboltz_eV", korg_constants.kboltz_eV, "korg_kboltz_eV"), 
        ("hplanck_cgs", korg_constants.hplanck_cgs, "korg_hplanck_cgs"),
        ("electron_mass_cgs", korg_constants.electron_mass_cgs, "korg_electron_mass_cgs"),
    ]
    
    for (name, korg_val, jorg_key) in comparisons
        jorg_val = jorg_constants[jorg_key]
        rel_diff = abs(jorg_val - korg_val) / korg_val
        max_rel_diff = max(max_rel_diff, rel_diff)
        
        status = rel_diff < 1e-12 ? "PASS" : "FAIL"
        results[name] = (korg=korg_val, jorg=jorg_val, rel_diff=rel_diff, status=status)
        
        @printf "  %-20s: %.2e relative difference [%s]\n" name rel_diff status
    end
    
    println("  Overall constants: $(max_rel_diff < 1e-12 ? "PASS" : "FAIL")")
    return results
end

function test_translational_partition_functions()
    """Test translational partition function calculations."""
    println("\n=== Comparing Translational Partition Functions ===")
    
    # Test parameters
    masses = [Korg.electron_mass_cgs, 1.67262e-24, 6.64466e-24]  # electron, proton, alpha
    mass_names = ["electron", "proton", "alpha"]
    temperatures = [3000.0, 5000.0, 8000.0, 12000.0]
    
    results = []
    max_rel_diff = 0.0
    
    for (i, (mass, name)) in enumerate(zip(masses, mass_names))
        for T in temperatures
            # Korg.jl calculation
            korg_result = Korg.translational_U(mass, T)
            
            # Jorg calculation via Python
            jorg_output = read(`python -c "
import sys
sys.path.insert(0, 'Jorg/src')
from jorg.statmech.saha_equation import translational_U
result = float(translational_U($mass, $T))
print(result)
"`, String)
            jorg_result = parse(Float64, strip(jorg_output))
            
            rel_diff = abs(jorg_result - korg_result) / korg_result
            max_rel_diff = max(max_rel_diff, rel_diff)
            
            test_case = (
                particle=name, mass=mass, temperature=T,
                korg_result=korg_result, jorg_result=jorg_result, 
                rel_diff=rel_diff
            )
            push!(results, test_case)
        end
    end
    
    # Show sample results
    for case in results[1:3:end]  # Every 3rd case
        @printf "  %-8s T=%5.0fK: RelDiff=%.2e\n" case.particle case.temperature case.rel_diff
    end
    
    println("  Overall translational: $(max_rel_diff < 1e-12 ? "PASS" : "FAIL")")
    return results
end

function test_saha_equation_comprehensive()
    """Comprehensive comparison of Saha equation calculations."""
    println("\n=== Comparing Saha Equation Calculations ===")
    
    results = []
    
    for condition in STELLAR_CONDITIONS
        T, ne, name = condition.T, condition.ne, condition.name
        @printf "  %s (T=%.0fK, ne=%.0e):\n" name T ne
        
        condition_results = []
        
        for Z in [1, 2, 26]  # H, He, Fe as key test cases
            element_name = Z == 1 ? "H" : Z == 2 ? "He" : "Fe"
            
            # Get ionization energies
            chi_I = Korg.ionization_energies[Z][1]
            
            # Korg.jl calculation 
            wII_korg, wIII_korg = Korg.saha_ion_weights(T, ne, Z, Korg.ionization_energies, Korg.default_partition_funcs)
            ion_frac_korg = wII_korg / (1.0 + wII_korg + wIII_korg)
            
            # Jorg calculation via Python
            jorg_output = read(`python -c "
import sys
sys.path.insert(0, 'Jorg/src')
from jorg.statmech.saha_equation import simple_saha_test
ratio = float(simple_saha_test($T, $ne, $Z, $chi_I))
ion_frac = ratio / (1.0 + ratio)
print(f'{ratio} {ion_frac}')
"`, String)
            
            jorg_vals = split(strip(jorg_output))
            ratio_jorg = parse(Float64, jorg_vals[1])
            ion_frac_jorg = parse(Float64, jorg_vals[2])
            
            # Calculate relative differences
            rel_diff_ratio = abs(wII_korg - ratio_jorg) / wII_korg
            rel_diff_frac = abs(ion_frac_korg - ion_frac_jorg) / ion_frac_korg
            
            test_case = (
                condition=name, element=element_name, atomic_number=Z,
                temperature=T, electron_density=ne, ionization_energy=chi_I,
                korg_ratio=wII_korg, jorg_ratio=ratio_jorg,
                korg_ion_frac=ion_frac_korg, jorg_ion_frac=ion_frac_jorg,
                rel_diff_ratio=rel_diff_ratio, rel_diff_frac=rel_diff_frac
            )
            
            push!(condition_results, test_case)
            push!(results, test_case)
            
            status = rel_diff_frac < 1e-3 ? "✅" : rel_diff_frac < 1e-2 ? "⚠️" : "❌"
            @printf "    %s: ion_frac_diff=%.3e %s\n" element_name rel_diff_frac status
        end
    end
    
    # Overall assessment
    all_good = all(case.rel_diff_frac < 1e-2 for case in results)
    println("  Overall Saha equation: $(all_good ? "PASS" : "WARNING")")
    
    return results
end

function test_species_representations()
    """Test Species and Formula representations."""
    println("\n=== Comparing Species Representations ===")
    
    # Test species creation and string representations
    test_species = [
        (Z=1, charge=0, expected="H I"),
        (Z=1, charge=1, expected="H II"), 
        (Z=2, charge=0, expected="He I"),
        (Z=26, charge=0, expected="Fe I"),
        (Z=26, charge=1, expected="Fe II"),
    ]
    
    results = []
    all_match = true
    
    for spec in test_species
        Z, charge, expected = spec.Z, spec.charge, spec.expected
        
        # Korg.jl representation
        korg_species = Korg.Species(Korg.Formula(Z), charge)
        korg_str = string(korg_species)
        
        # Jorg representation via Python
        jorg_output = read(`python -c "
import sys
sys.path.insert(0, 'Jorg/src')
from jorg.statmech.species import Species
species = Species.from_atomic_number($Z, $charge)
print(str(species))
"`, String)
        jorg_str = strip(jorg_output)
        
        strings_match = korg_str == jorg_str == expected
        all_match &= strings_match
        
        test_case = (
            atomic_number=Z, charge=charge, expected=expected,
            korg_str=korg_str, jorg_str=jorg_str, match=strings_match
        )
        push!(results, test_case)
        
        status = strings_match ? "✅" : "❌"
        @printf "  Z=%2d charge=%d: Korg='%s' Jorg='%s' %s\n" Z charge korg_str jorg_str status
    end
    
    println("  Overall species representations: $(all_match ? "PASS" : "FAIL")")
    return results
end

function benchmark_performance()
    """Benchmark performance of key functions."""
    println("\n=== Performance Benchmarks ===")
    
    # Benchmark translational_U
    print("  Benchmarking translational_U...")
    times_trans = Float64[]
    for _ in 1:1000
        t = @elapsed Korg.translational_U(Korg.electron_mass_cgs, 5778.0)
        push!(times_trans, t)
    end
    trans_mean = mean(times_trans)
    trans_std = std(times_trans)
    
    # Benchmark saha_ion_weights  
    print(" saha_ion_weights...")
    times_saha = Float64[]
    for _ in 1:100
        t = @elapsed Korg.saha_ion_weights(5778.0, 1e13, 26, Korg.ionization_energies, Korg.default_partition_funcs)
        push!(times_saha, t)
    end
    saha_mean = mean(times_saha)
    saha_std = std(times_saha)
    
    println(" done.")
    
    results = (
        translational = (mean_ms = trans_mean * 1000, std_ms = trans_std * 1000, 
                        calls_per_sec = 1.0 / trans_mean),
        saha = (mean_ms = saha_mean * 1000, std_ms = saha_std * 1000,
               calls_per_sec = 1.0 / saha_mean)
    )
    
    @printf "  Translational U: %.1f ± %.1f μs (%.0f calls/sec)\n" (trans_mean*1e6) (trans_std*1e6) (1.0/trans_mean)
    @printf "  Saha equation:   %.2f ± %.2f ms (%.0f calls/sec)\n" (saha_mean*1e3) (saha_std*1e3) (1.0/saha_mean)
    
    return results
end

function run_chemical_equilibrium_comparison()
    """Compare chemical equilibrium solver results."""
    println("\n=== Comparing Chemical Equilibrium Solvers ===")
    
    # Solar conditions
    T = 5778.0
    nt = 1e15  # cm^-3
    model_atm_ne = 1e13  # cm^-3
    
    # Get solar abundances
    A_X = Korg.format_A_X()
    
    # Create absolute abundances dictionary 
    absolute_abundances = Dict{Int,Float64}()
    total_abundance = 0.0
    for Z in 1:92
        if haskey(A_X, Z)
            linear_abundance = 10^(A_X[Z] - 12.0)
            absolute_abundances[Z] = linear_abundance 
            total_abundance += linear_abundance
        end
    end
    
    # Normalize
    for Z in keys(absolute_abundances)
        absolute_abundances[Z] /= total_abundance
    end
    
    println("  Running Korg.jl chemical equilibrium...")
    
    # Korg.jl calculation
    start_time = time()
    ne_korg, densities_korg = Korg.chemical_equilibrium(
        T, nt, model_atm_ne, absolute_abundances,
        Korg.ionization_energies, Korg.default_partition_funcs, 
        Korg.default_log_equilibrium_constants
    )
    korg_time = time() - start_time
    
    println("  Running Jorg chemical equilibrium...")
    
    # Create Python abundance array string
    abundance_dict_str = "{" * join(["$k: $v" for (k,v) in absolute_abundances], ", ") * "}"
    
    # Jorg calculation via Python
    jorg_output = read(`python -c "
import sys
import time
sys.path.insert(0, 'Jorg/src')
from jorg.statmech.chemical_equilibrium import chemical_equilibrium
from jorg.statmech.partition_functions import create_default_partition_functions
from jorg.statmech.saha_equation import create_default_ionization_energies  
from jorg.statmech.molecular import create_default_log_equilibrium_constants

# Load data
partition_funcs = create_default_partition_functions()
ionization_energies = create_default_ionization_energies()
log_equilibrium_constants = create_default_log_equilibrium_constants()

# Abundances
absolute_abundances = $abundance_dict_str

# Run calculation
start_time = time.time()
try:
    ne, number_densities = chemical_equilibrium(
        $T, $nt, $model_atm_ne, absolute_abundances,
        ionization_energies, partition_funcs, log_equilibrium_constants
    )
    calc_time = time.time() - start_time
    
    # Get key species
    from jorg.statmech.species import Species
    h_i = Species.from_string('H I')
    h_ii = Species.from_string('H II')
    fe_i = Species.from_string('Fe I') 
    fe_ii = Species.from_string('Fe II')
    
    print(f'SUCCESS {ne} {calc_time}')
    print(f'HI {number_densities.get(h_i, 0.0)}')
    print(f'HII {number_densities.get(h_ii, 0.0)}')
    print(f'FeI {number_densities.get(fe_i, 0.0)}')
    print(f'FeII {number_densities.get(fe_ii, 0.0)}')
except Exception as e:
    print(f'ERROR {str(e)}')
"`, String)
    
    jorg_lines = split(strip(jorg_output), '\n')
    
    if startswith(jorg_lines[1], "SUCCESS")
        success_parts = split(jorg_lines[1])
        ne_jorg = parse(Float64, success_parts[2])
        jorg_time = parse(Float64, success_parts[3])
        
        # Parse species densities
        jorg_densities = Dict{String, Float64}()
        for line in jorg_lines[2:end]
            parts = split(line)
            if length(parts) == 2
                jorg_densities[parts[1]] = parse(Float64, parts[2])
            end
        end
        
        # Compare key results
        h_i_korg = densities_korg[Korg.Species(Korg.Formula(1), 0)]
        h_ii_korg = densities_korg[Korg.Species(Korg.Formula(1), 1)]
        fe_i_korg = densities_korg[Korg.Species(Korg.Formula(26), 0)]
        fe_ii_korg = densities_korg[Korg.Species(Korg.Formula(26), 1)]
        
        h_i_jorg = jorg_densities["HI"]
        h_ii_jorg = jorg_densities["HII"] 
        fe_i_jorg = jorg_densities["FeI"]
        fe_ii_jorg = jorg_densities["FeII"]
        
        # Calculate relative differences
        ne_rel_diff = abs(ne_korg - ne_jorg) / ne_korg
        h_i_rel_diff = abs(h_i_korg - h_i_jorg) / h_i_korg
        h_ii_rel_diff = abs(h_ii_korg - h_ii_jorg) / h_ii_korg
        fe_i_rel_diff = abs(fe_i_korg - fe_i_jorg) / fe_i_korg
        fe_ii_rel_diff = abs(fe_ii_korg - fe_ii_jorg) / fe_ii_korg
        
        results = (
            electron_density = (korg=ne_korg, jorg=ne_jorg, rel_diff=ne_rel_diff),
            h_i = (korg=h_i_korg, jorg=h_i_jorg, rel_diff=h_i_rel_diff),
            h_ii = (korg=h_ii_korg, jorg=h_ii_jorg, rel_diff=h_ii_rel_diff), 
            fe_i = (korg=fe_i_korg, jorg=fe_i_jorg, rel_diff=fe_i_rel_diff),
            fe_ii = (korg=fe_ii_korg, jorg=fe_ii_jorg, rel_diff=fe_ii_rel_diff),
            timing = (korg_sec=korg_time, jorg_sec=jorg_time)
        )
        
        @printf "  Electron density: %.3e vs %.3e (%.2e rel diff)\n" ne_korg ne_jorg ne_rel_diff
        @printf "  H I density:      %.3e vs %.3e (%.2e rel diff)\n" h_i_korg h_i_jorg h_i_rel_diff
        @printf "  H II density:     %.3e vs %.3e (%.2e rel diff)\n" h_ii_korg h_ii_jorg h_ii_rel_diff
        @printf "  Fe I density:     %.3e vs %.3e (%.2e rel diff)\n" fe_i_korg fe_i_jorg fe_i_rel_diff
        @printf "  Fe II density:    %.3e vs %.3e (%.2e rel diff)\n" fe_ii_korg fe_ii_jorg fe_ii_rel_diff
        @printf "  Timing:           %.3f vs %.3f seconds\n" korg_time jorg_time
        
        # Overall assessment
        max_rel_diff = max(ne_rel_diff, h_i_rel_diff, h_ii_rel_diff, fe_i_rel_diff, fe_ii_rel_diff)
        status = max_rel_diff < 1e-2 ? "EXCELLENT" : max_rel_diff < 1e-1 ? "GOOD" : "NEEDS_WORK"
        println("  Overall chemical equilibrium: $status (max rel diff: $(max_rel_diff:.2e))")
        
        return results
    else
        println("  ERROR: Jorg chemical equilibrium failed")
        println("  Output: $jorg_output")
        return nothing
    end
end

function generate_summary_report(all_results)
    """Generate comprehensive summary report."""
    println("\n" * "="^60)
    println("JORG vs KORG.JL STATISTICAL MECHANICS COMPARISON")
    println("="^60)
    
    # Extract status from each test category
    categories = [
        ("Physical Constants", get(all_results, :constants, nothing)),
        ("Translational Partition", get(all_results, :translational, nothing)),
        ("Saha Equation", get(all_results, :saha, nothing)),
        ("Species Representations", get(all_results, :species, nothing)),
        ("Chemical Equilibrium", get(all_results, :chemical_eq, nothing)),
    ]
    
    passed = 0
    total = 0
    
    for (name, result) in categories
        if result !== nothing
            total += 1
            
            # Determine status based on result type
            if name == "Physical Constants"
                status = all(r.status == "PASS" for r in values(result)) ? "PASS" : "FAIL"
            elseif name == "Translational Partition" 
                status = all(r.rel_diff < 1e-12 for r in result) ? "PASS" : "FAIL"
            elseif name == "Saha Equation"
                status = all(r.rel_diff_frac < 1e-2 for r in result) ? "PASS" : "WARNING"
            elseif name == "Species Representations"
                status = all(r.match for r in result) ? "PASS" : "FAIL" 
            elseif name == "Chemical Equilibrium"
                max_diff = max(result.electron_density.rel_diff, result.h_i.rel_diff, 
                             result.h_ii.rel_diff, result.fe_i.rel_diff, result.fe_ii.rel_diff)
                status = max_diff < 1e-2 ? "PASS" : max_diff < 1e-1 ? "WARNING" : "FAIL"
            else
                status = "UNKNOWN"
            end
            
            if status == "PASS"
                status_icon = "✅"
                passed += 1
            elseif status == "WARNING"
                status_icon = "⚠️"
                passed += 0.5
            else
                status_icon = "❌"
            end
            
            @printf "%-25s: %s %s\n" name status_icon status
        end
    end
    
    # Overall assessment
    pass_rate = total > 0 ? passed / total : 0.0
    
    if pass_rate >= 0.9
        overall = "EXCELLENT"
    elseif pass_rate >= 0.7
        overall = "GOOD"
    elseif pass_rate >= 0.5
        overall = "ACCEPTABLE"
    else
        overall = "NEEDS_IMPROVEMENT"
    end
    
    @printf "\nOverall Assessment: %s (%.1f%% pass rate)\n" overall (pass_rate * 100)
    
    # Performance summary
    if haskey(all_results, :performance)
        perf = all_results[:performance]
        println("\nPerformance Summary (Korg.jl):")
        @printf "  Translational U: %.0f calls/sec\n" perf.translational.calls_per_sec
        @printf "  Saha equation: %.0f calls/sec\n" perf.saha.calls_per_sec
    end
    
    return (pass_rate=pass_rate, overall=overall)
end

function main()
    """Run complete comparison suite."""
    println("="^60)
    println("COMPREHENSIVE JORG vs KORG.JL COMPARISON")
    println("="^60)
    
    # Run all comparison tests
    all_results = Dict()
    
    all_results[:constants] = test_physical_constants()
    all_results[:translational] = test_translational_partition_functions()
    all_results[:saha] = test_saha_equation_comprehensive()
    all_results[:species] = test_species_representations()
    all_results[:performance] = benchmark_performance()
    all_results[:chemical_eq] = run_chemical_equilibrium_comparison()
    
    # Generate summary
    summary = generate_summary_report(all_results)
    
    # Save results to JSON
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    filename = "jorg_korg_comparison_$timestamp.json"
    
    # Convert to JSON-serializable format
    json_results = Dict()
    for (key, val) in all_results
        if key == :constants
            json_results[string(key)] = Dict(string(k) => Dict(
                "korg" => v.korg, "jorg" => v.jorg, 
                "rel_diff" => v.rel_diff, "status" => v.status
            ) for (k, v) in val)
        elseif key == :translational || key == :saha || key == :species
            json_results[string(key)] = [Dict(string(k) => v for (k, v) in pairs(case)) for case in val]
        elseif key == :performance
            json_results[string(key)] = Dict(
                "translational" => Dict(string(k) => v for (k, v) in pairs(val.translational)),
                "saha" => Dict(string(k) => v for (k, v) in pairs(val.saha))
            )
        elseif key == :chemical_eq && val !== nothing
            json_results[string(key)] = Dict(
                "electron_density" => Dict(string(k) => v for (k, v) in pairs(val.electron_density)),
                "h_i" => Dict(string(k) => v for (k, v) in pairs(val.h_i)),
                "h_ii" => Dict(string(k) => v for (k, v) in pairs(val.h_ii)),
                "fe_i" => Dict(string(k) => v for (k, v) in pairs(val.fe_i)),
                "fe_ii" => Dict(string(k) => v for (k, v) in pairs(val.fe_ii)),
                "timing" => Dict(string(k) => v for (k, v) in pairs(val.timing))
            )
        end
    end
    
    json_results["summary"] = Dict("pass_rate" => summary.pass_rate, "overall" => summary.overall)
    
    open(filename, "w") do f
        JSON.print(f, json_results, 2)
    end
    
    println("\nDetailed results saved to: $filename")
    
    return filename
end

# Run the comparison if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end