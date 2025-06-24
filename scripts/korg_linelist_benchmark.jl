#!/usr/bin/env julia
"""
Korg.jl linelist reading performance benchmark

This script tests the performance of reading linelists in Korg.jl
to compare with the Jorg implementation.
"""

using Korg
using BenchmarkTools
using Statistics
using Printf
using JSON3

function create_sample_linelist(filename::String, n_lines::Int = 1000)
    """Create a sample linelist file for testing"""
    
    # Sample line data (wavelength, species, log_gf, E_lower, vdW)
    templates = [
        (5889.951, 11.0, 0.108, 0.000, 1.4e-7),
        (5895.924, 11.0, -0.194, 0.000, 1.4e-7), 
        (6562.801, 1.0, 0.640, 10.199, 2.8e-5),
        (4861.323, 1.0, -0.020, 10.199, 2.8e-5),
        (5167.321, 12.0, -0.890, 4.912, 1.2e-7),
        (5857.476, 26.0, -2.158, 4.283, 1.3e-7),
        (5862.353, 26.0, -0.058, 4.549, 1.3e-7),
    ]
    
    open(filename, "w") do f
        println(f, "# Korg test linelist")
        println(f, "# wavelength(Å)  species  log_gf  E_lower(eV)  vdW")
        
        for i in 1:n_lines
            template = templates[((i-1) % length(templates)) + 1]
            
            # Add some variation
            wl = template[1] + 5.0 * (rand() - 0.5)
            species = template[2]
            log_gf = template[3] + 0.5 * (rand() - 0.5)
            E_lower = template[4] + 0.1 * (rand() - 0.5)
            vdw = template[5]
            
            @printf(f, "%8.3f %5.1f %6.3f %6.3f %8.2e\n", 
                    wl, species, log_gf, E_lower, vdw)
        end
    end
    
    println("Created sample linelist: $(filename) with $(n_lines) lines")
end

function benchmark_korg_linelist_reading(filename::String, n_trials::Int = 5)
    """Benchmark Korg linelist reading performance"""
    
    times = Float64[]
    n_lines = 0
    
    # Check if Korg has linelist reading capability
    if !hasmethod(Korg.read_linelist, Tuple{String})
        println("⚠️  Korg.read_linelist not available - testing alternative approach")
        
        # Try alternative approach - create synthetic linelist
        for trial in 1:n_trials
            start_time = time()
            
            try
                # Read file manually and create line data
                lines = readlines(filename)
                data_lines = filter(line -> !startswith(line, "#") && !isempty(strip(line)), lines)
                
                wavelengths = Float64[]
                species_ids = Int[]
                log_gfs = Float64[]
                E_lowers = Float64[]
                
                for line in data_lines
                    parts = split(strip(line))
                    if length(parts) >= 4
                        push!(wavelengths, parse(Float64, parts[1]))
                        push!(species_ids, Int(round(parse(Float64, parts[2]) * 100)))
                        push!(log_gfs, parse(Float64, parts[3]))
                        push!(E_lowers, parse(Float64, parts[4]))
                    end
                end
                
                n_lines = length(wavelengths)
                
                # Force evaluation
                mean_wl = mean(wavelengths)
                
            catch e
                return Dict("error" => string(e))
            end
            
            end_time = time()
            push!(times, end_time - start_time)
        end
        
    else
        # Use native Korg linelist reading
        for trial in 1:n_trials
            start_time = time()
            
            try
                linelist = Korg.read_linelist(filename)
                n_lines = length(linelist.wavelength)
                
                # Force evaluation
                mean_wl = mean(linelist.wavelength)
                
            catch e
                return Dict("error" => string(e))
            end
            
            end_time = time()
            push!(times, end_time - start_time)
        end
    end
    
    return Dict(
        "times" => times,
        "mean_time" => mean(times),
        "std_time" => std(times),
        "min_time" => minimum(times),
        "max_time" => maximum(times),
        "n_lines" => n_lines,
        "lines_per_second" => n_lines / mean(times)
    )
end

function main()
    println("🧪 Korg.jl Linelist Reading Performance Test")
    println("=" * 50)
    
    # Test different sizes
    test_sizes = [1000, 5000, 10000]
    results = Dict()
    
    for n_lines in test_sizes
        println("\n📊 Testing with $(n_lines) lines...")
        
        # Create test file
        test_file = "korg_test_$(n_lines).dat"
        create_sample_linelist(test_file, n_lines)
        
        try
            # Benchmark reading
            result = benchmark_korg_linelist_reading(test_file, 3)
            results[n_lines] = result
            
            if haskey(result, "error")
                println("❌ Error: $(result["error"])")
            else
                @printf("✅ %d lines: %.3fs (%.0f lines/s)\n", 
                       n_lines, result["mean_time"], result["lines_per_second"])
            end
            
        finally
            # Clean up
            if isfile(test_file)
                rm(test_file)
            end
        end
    end
    
    # Print summary
    println("\n" * "=" * 50)
    println("📈 KORG.JL PERFORMANCE SUMMARY")
    println("=" * 50)
    
    println(@sprintf("%-8s %-10s %-15s", "Lines", "Time (s)", "Lines/sec"))
    println("-" * 35)
    
    for n_lines in test_sizes
        if haskey(results, n_lines) && !haskey(results[n_lines], "error")
            result = results[n_lines]
            println(@sprintf("%-8d %-10.3f %-15.0f", 
                           n_lines, result["mean_time"], result["lines_per_second"]))
        else
            println(@sprintf("%-8d %-10s %-15s", n_lines, "ERROR", "N/A"))
        end
    end
    
    # Output JSON for Python integration
    println("\n🔗 JSON Output for Integration:")
    println(JSON3.write(results))
    
    println("\n✨ Korg.jl benchmark completed!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end