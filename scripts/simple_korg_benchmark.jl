#!/usr/bin/env julia
"""
Simple Korg.jl performance test for file I/O and basic operations
"""

using Statistics
using Printf

function create_sample_data(filename::String, n_lines::Int = 1000)
    """Create sample data file"""
    
    open(filename, "w") do f
        println(f, "# Sample data for performance testing")
        println(f, "# wavelength  value1  value2  value3")
        
        for i in 1:n_lines
            wl = 5000.0 + i * 0.1 + 5.0 * (rand() - 0.5)
            val1 = rand()
            val2 = rand() * 10.0
            val3 = rand() * 100.0
            
            @printf(f, "%8.3f %8.5f %8.3f %8.1f\n", wl, val1, val2, val3)
        end
    end
    
    println("Created sample file: $(filename) with $(n_lines) lines")
end

function benchmark_file_reading(filename::String, n_trials::Int = 5)
    """Benchmark file reading performance"""
    
    times = Float64[]
    n_lines = 0
    
    for trial in 1:n_trials
        start_time = time()
        
        # Read and parse file
        data = []
        open(filename, "r") do f
            for line in eachline(f)
                line = strip(line)
                if !isempty(line) && !startswith(line, "#")
                    parts = split(line)
                    if length(parts) >= 4
                        try
                            wl = parse(Float64, parts[1])
                            val1 = parse(Float64, parts[2])
                            val2 = parse(Float64, parts[3])
                            val3 = parse(Float64, parts[4])
                            push!(data, (wl, val1, val2, val3))
                        catch e
                            # Skip malformed lines
                        end
                    end
                end
            end
        end
        
        n_lines = length(data)
        
        # Force evaluation
        if n_lines > 0
            mean_wl = mean([d[1] for d in data])
        end
        
        end_time = time()
        push!(times, end_time - start_time)
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
    println("🧪 Simple Julia File I/O Performance Test")
    println("=" ^ 50)
    
    # Test different sizes
    test_sizes = [1000, 5000, 10000, 25000]
    results = Dict()
    
    for n_lines in test_sizes
        println("\n📊 Testing with $(n_lines) lines...")
        
        # Create test file
        test_file = "julia_test_$(n_lines).dat"
        create_sample_data(test_file, n_lines)
        
        try
            # Benchmark reading
            result = benchmark_file_reading(test_file, 3)
            results[n_lines] = result
            
            @printf("✅ %d lines: %.3fs (%.0f lines/s)\n", 
                   n_lines, result["mean_time"], result["lines_per_second"])
            
        finally
            # Clean up
            if isfile(test_file)
                rm(test_file)
            end
        end
    end
    
    # Print summary  
    println("\n" * ("=" ^ 50))
    println("📈 JULIA I/O PERFORMANCE SUMMARY")
    println("=" ^ 50)
    
    println(@sprintf("%-8s %-10s %-15s", "Lines", "Time (s)", "Lines/sec"))
    println("-" ^ 35)
    
    for n_lines in test_sizes
        if haskey(results, n_lines)
            result = results[n_lines]
            println(@sprintf("%-8d %-10.3f %-15.0f", 
                           n_lines, result["mean_time"], result["lines_per_second"]))
        end
    end
    
    println("\n✨ Julia benchmark completed!")
    println("📝 Note: This tests basic file I/O, not Korg-specific functionality")
    println("🔗 For Korg linelist functionality, see existing linelist files in data/")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end