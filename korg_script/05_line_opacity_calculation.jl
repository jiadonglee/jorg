#!/usr/bin/env julia
"""
Korg.jl Line Opacity Calculation Test
Tests line opacity with VALD linelist for comparison with Jorg.
"""

using Korg
using Printf

println("KORG.JL LINE OPACITY CALCULATION TEST")
println("="^50)

# Load VALD linelist
vald_paths = [
    "/Users/jdli/Project/Korg.jl/data/linelists/vald_extract_stellar_solar_threshold001.vald",
    "/Users/jdli/Project/Korg.jl/misc/Tutorial notebooks/basics/linelist.vald"
]

global linelist = nothing
global linelist_loaded = false

for vald_path in vald_paths
    if isfile(vald_path)
        try
            start_time = time()
            global linelist = read_linelist(vald_path)
            load_time = time() - start_time
            println("✅ VALD linelist loaded: $(length(linelist)) lines ($(round(load_time, digits=2))s)")
            global linelist_loaded = true
            break
        catch e
            println("❌ Failed to load $vald_path: $e")
        end
    end
end

if !linelist_loaded
    println("❌ Could not load VALD linelist")
    global linelist = []
end

# Test parameters
Teff, logg, m_H = 5780.0, 4.44, 0.0
wavelength_start, wavelength_end = 5000.0, 5010.0
println("Test parameters: Teff=$(Teff)K, logg=$(logg), [M/H]=$(m_H), λ=$(wavelength_start)-$(wavelength_end) Å")

# Run synthesis with lines
if linelist_loaded
    try
        println("Running synthesis with line opacity...")
        start_time = time()
        
        # Create atmosphere and abundances
        atm = interpolate_marcs(Teff, logg, m_H)
        A_X = zeros(92)
        A_X[1] = 12.0  # Hydrogen
        solar_elements = [12.00, 10.91, 0.96, 1.38, 2.70, 8.46, 7.83, 8.69, 4.40, 8.06]
        A_X[1:length(solar_elements)] = solar_elements
        A_X[2:end] .+= m_H  # Apply metallicity
        
        # Wavelength grid
        wavelengths = range(wavelength_start, wavelength_end, length=2001)
        
        # Synthesis
        result = synthesize(atm, linelist, A_X, wavelengths)
        wls = wavelengths  # Use input wavelengths
        flux = result.flux
        continuum = result.cntm
        
        synthesis_time = time() - start_time
        println("✅ Synthesis completed ($(round(synthesis_time, digits=2))s)")
        println("   Wavelengths: $(length(wls)) points")
        println("   Flux range: $(round(minimum(flux), sigdigits=3)) - $(round(maximum(flux), sigdigits=3))")
        println("   Continuum range: $(round(minimum(continuum), sigdigits=3)) - $(round(maximum(continuum), sigdigits=3))")
        
        # Analyze line depths
        flux_ratio = flux ./ max.(continuum, 1e-10)
        max_line_depth = 1.0 - minimum(flux_ratio)
        
        println("   Maximum line depth: $(round(max_line_depth*100, digits=1))%")
        println("   Points with >1% absorption: $(count(flux_ratio .< 0.99))")
        
        # Calculate line opacity (difference from continuum)
        line_opacity = continuum .- flux
        
        # Save data for comparison
        using DelimitedFiles
        writedlm("korg_line_opacity_data.txt", [wls line_opacity], header=false)
        println("   Saved opacity data to korg_line_opacity_data.txt")
        
        # Validation checks
        checks = [
            ("Flux positive", all(flux .>= 0)),
            ("Continuum positive", all(continuum .>= 0)),
            ("Realistic flux", 1e14 <= maximum(flux) <= 1e17),
            ("Line absorption active", max_line_depth > 0.01),
            ("Performance good", synthesis_time < 20.0)
        ]
        
        println("\nValidation:")
        all_passed = true
        for (check_name, passed) in checks
            status = passed ? "✅ PASS" : "❌ FAIL"
            println("  $(rpad(check_name, 20)): $status")
            all_passed = all_passed && passed
        end
        
        if all_passed
            println("\n🎉 KORG.JL LINE OPACITY CALCULATION: PRODUCTION READY")
        else
            println("\n⚠️  KORG.JL LINE OPACITY CALCULATION: NEEDS ATTENTION")
        end
        
    catch e
        println("❌ Synthesis failed: $e")
    end
else
    println("⚠️  Cannot test without VALD linelist")
end

println("Test complete!")