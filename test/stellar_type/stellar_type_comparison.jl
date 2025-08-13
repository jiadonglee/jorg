#!/usr/bin/env julia
"""
Stellar Type Comparison with Korg.jl
====================================

Compare synthesis for different stellar types:
1. Solar-like star: Teff=5771K, logg=4.44, [M/H]=0.0
2. Arcturus-like star: Teff=4250K, logg=1.4, [Fe/H]=-0.5
3. Metal-poor K giant: Teff=4500K, logg=1.5, [Fe/H]=-2.5
"""

using Korg
using Printf
using Statistics
using Dates

println("🌟 STELLAR TYPE COMPARISON - KORG.JL")
println("=" ^ 55)

# Define stellar parameters
stellar_types = [
    (name="Solar-like", Teff=5771, logg=4.44, m_H=0.0, description="G2V dwarf (Sun-like)"),
    (name="Arcturus-like", Teff=4250, logg=1.4, m_H=-0.5, description="K1.5 III giant"),
    (name="Metal-poor K giant", Teff=4500, logg=1.5, m_H=-2.5, description="K2 III halo giant")
]

# Synthesis parameters
wavelength_range = (5000, 5200)  # Å
vmic = 1.0  # km/s

println("Synthesis parameters:")
println("  Wavelength range: $wavelength_range Å")
println("  Microturbulence: $vmic km/s")
println("  Using synthesize() for complete diagnostics")
println()

# Load line list
println("📋 Loading VALD line list...")
linelist = []
try
    # Try to get VALD solar linelist first
    try
        linelist = get_VALD_solar_linelist()
        println("✅ Loaded VALD stellar solar threshold linelist")
    catch e
        # Fallback: try direct file paths
        vald_paths = [
            joinpath("..", "..", "..", "data", "linelists", "vald_extract_stellar_solar_threshold001.vald"),
            joinpath("..", "..", "..", "misc", "Tutorial notebooks", "basics", "linelist.vald")
        ]
        
        loaded = false
        for vald_path in vald_paths
            if isfile(vald_path)
                linelist = read_linelist(vald_path)
                println("✅ Loaded VALD linelist from: $vald_path")
                loaded = true
                break
            end
        end
        
        if !loaded
            println("⚠️  VALD linelist file not found")
            println("   Using empty linelist for continuum-only synthesis")
        end
    end
    
    if !isempty(linelist)
        println("✅ Linelist loaded: $(length(linelist)) lines")
    else
        println("ℹ️  Running continuum-only synthesis")
    end
    
catch e
    println("⚠️  Linelist loading failed: $e")
    println("   Using empty linelist for continuum-only synthesis")
end

println()

# Store results
results = Dict()

# Synthesize each stellar type
for (i, star) in enumerate(stellar_types)
    println("$i. $(star.name) ($(star.description))")
    println("   Teff=$(star.Teff)K, logg=$(star.logg), [M/H]=$(star.m_H)")
    
    try
        # Create atmosphere and abundance array
        A_X = format_A_X(star.m_H)
        atm = interpolate_marcs(star.Teff, star.logg, A_X)
        
        # Run synthesis
        start_time = time()
        
        result = synthesize(
            atm,                    # Model atmosphere
            linelist,               # Line list
            A_X,                    # Abundance array
            wavelength_range,       # Wavelength range
            vmic=vmic,              # Microturbulent velocity
            return_cntm=true        # Return continuum
        )
        
        elapsed = time() - start_time
        
        # Store results
        results[star.name] = (
            result=result,
            parameters=star,
            synthesis_time=elapsed
        )
        
        # Analyze spectrum - calculate normalized flux
        normalized_flux = result.cntm !== nothing ? result.flux ./ result.cntm : result.flux
        min_flux = minimum(normalized_flux)
        max_absorption = (1.0 - min_flux) * 100
        line_features = sum(normalized_flux .< 0.95)
        
        println("   ✅ Synthesis complete: $(round(elapsed, digits=2))s")
        println("      Flux range: $(round(min_flux, digits=3)) - $(round(maximum(normalized_flux), digits=3))")
        println("      Deepest line: $(round(max_absorption, digits=1))% absorption")
        println("      Line features: $line_features pixels >5% depth")
        
    catch e
        println("   ❌ Synthesis failed: $e")
        results[star.name] = nothing
    end
    
    println()
end

# Save results to files
println("💾 SAVING SYNTHESIS RESULTS")
println("-" ^ 35)

timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")

for (star_name, result_data) in results
    if result_data === nothing
        continue
    end
    
    result = result_data.result
    params = result_data.parameters
    
    # Create safe filename
    safe_name = lowercase(replace(replace(star_name, " " => "_"), "-" => "_"))
    spectrum_file = "korg_$(safe_name)_spectrum_$timestamp.txt"
    
    # Save spectrum
    open(spectrum_file, "w") do file
        println(file, "# Korg.jl Stellar Synthesis - $star_name")
        println(file, "# Generated: $(now())")
        println(file, "# Parameters: Teff=$(params.Teff)K, logg=$(params.logg), [M/H]=$(params.m_H)")
        println(file, "# Synthesis time: $(round(result_data.synthesis_time, digits=2)) seconds")
        println(file, "# Line list: $(length(linelist)) lines")
        println(file, "#")
        
        if result.cntm !== nothing
            println(file, "# Wavelength(Å)    Flux              Continuum         NormalizedFlux")
            normalized_flux = result.flux ./ result.cntm
            for i in 1:length(result.wavelengths)
                @printf(file, "%12.4f  %15.6e  %15.6e  %12.6f\n", 
                       result.wavelengths[i], result.flux[i], result.cntm[i], normalized_flux[i])
            end
        else
            println(file, "# Wavelength(Å)    Flux")
            for i in 1:length(result.wavelengths)
                @printf(file, "%12.4f  %15.6e\n", result.wavelengths[i], result.flux[i])
            end
        end
    end
    
    file_size = round(filesize(spectrum_file) / 1024, digits=1)
    println("✅ $star_name: $spectrum_file ($(file_size) KB)")
end

# Create comparison file
comparison_file = "korg_stellar_comparison_$timestamp.txt"
open(comparison_file, "w") do file
    println(file, "# Korg.jl Stellar Type Comparison Summary")
    println(file, "# Generated: $(now())")
    println(file, "#")
    println(file, "# Star Type           Teff(K)  logg   [M/H]  SynthTime(s)  MinFlux  MaxAbsorption(%)")
    
    for (star_name, result_data) in results
        if result_data === nothing
            continue
        end
        
        result = result_data.result
        params = result_data.parameters
        
        normalized_flux = result.cntm !== nothing ? result.flux ./ result.cntm : result.flux
        min_flux = minimum(normalized_flux)
        max_abs = (1.0 - min_flux) * 100
        
        @printf(file, "%-18s  %5d  %5.2f  %5.1f  %10.2f  %7.3f  %13.1f\n",
               star_name, params.Teff, params.logg, params.m_H, 
               result_data.synthesis_time, min_flux, max_abs)
    end
end

println("✅ Comparison summary: $comparison_file")

println()
println("📊 STELLAR TYPE ANALYSIS")
println("-" ^ 25)

# Compare spectral properties
valid_results = filter(x -> x.second !== nothing, results)

if length(valid_results) >= 2
    println("Relative spectral differences:")
    
    star_names = collect(keys(valid_results))
    
    for i in 1:length(star_names)
        for j in (i+1):length(star_names)
            star1, star2 = star_names[i], star_names[j]
            result1 = valid_results[star1].result
            result2 = valid_results[star2].result
            
            # Calculate normalized fluxes
            norm_flux1 = result1.cntm !== nothing ? result1.flux ./ result1.cntm : result1.flux
            norm_flux2 = result2.cntm !== nothing ? result2.flux ./ result2.cntm : result2.flux
            
            # Find common wavelength range
            wl1, wl2 = result1.wavelengths, result2.wavelengths
            wl_min = max(minimum(wl1), minimum(wl2))
            wl_max = min(maximum(wl1), maximum(wl2))
            
            if wl_max > wl_min
                # Create masks for common wavelength range
                mask1 = (wl1 .>= wl_min) .& (wl1 .<= wl_max)
                mask2 = (wl2 .>= wl_min) .& (wl2 .<= wl_max)
                
                if sum(mask1) > 100 && sum(mask2) > 100
                    # Simple comparison using overlapping points
                    flux1_sub = norm_flux1[mask1]
                    flux2_sub = norm_flux2[mask2]
                    
                    # Use shorter array length for comparison
                    min_len = min(length(flux1_sub), length(flux2_sub))
                    if min_len > 100
                        flux_diff = abs.(flux1_sub[1:min_len] .- flux2_sub[1:min_len])
                        mean_diff = mean(flux_diff)
                        max_diff = maximum(flux_diff)
                        rms_diff = sqrt(mean(flux_diff.^2))
                        
                        println("  $star1 vs $star2:")
                        println("    Mean difference: $(round(mean_diff, digits=4))")
                        println("    Max difference: $(round(max_diff, digits=4))")
                        println("    RMS difference: $(round(rms_diff, digits=4))")
                        println("    Comparison points: $min_len")
                    end
                end
            end
        end
    end
end

println()
println("🎉 KORG.JL STELLAR TYPE COMPARISON COMPLETE!")
println("   Results saved with timestamp: $timestamp")
println("   Ready for comparison with Jorg results")