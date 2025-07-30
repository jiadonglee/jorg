#!/usr/bin/env julia
"""
Brief Total Opacity Calculator for Korg.jl
===========================================
Calculates total opacity (continuum + line) for stellar atmosphere synthesis
Using the high-level synthesize() function to extract opacity information
"""

using Korg
using Printf
using DelimitedFiles

"""
    calculate_total_opacity(wavelengths_angstrom; Teff=5780, logg=4.44, m_H=0.0, 
                          linelist_path=nothing)

Calculate total opacity for given stellar parameters using Korg.jl's synthesize function

# Parameters
- `wavelengths_angstrom`: Wavelengths in Angstrom
- `Teff`: Effective temperature (K)
- `logg`: Surface gravity
- `m_H`: Metallicity [M/H]
- `linelist_path`: Path to VALD linelist file

# Returns
Dictionary containing opacity components and metadata
"""
function calculate_total_opacity(wavelengths_angstrom; Teff=5780, logg=4.44, m_H=0.0, 
                                linelist_path=nothing)
    
    @printf "Calculating total opacity for Teff=%dK, logg=%.2f, [M/H]=%.1f\n" Teff logg m_H
    @printf "Wavelength range: %.1f - %.1f Å\n" wavelengths_angstrom[1] wavelengths_angstrom[end]
    
    # 1. Setup abundances and atmosphere
    A_X = format_A_X(m_H)  # Pass metallicity as first positional argument
    atm = interpolate_marcs(Teff, logg, A_X)
    
    # 2. Load linelist - use empty list instead of nothing for continuum-only
    if linelist_path !== nothing && isfile(linelist_path)
        try
            @printf "Reading linelist: %s\n" linelist_path
            linelist = read_linelist(linelist_path)
            @printf "Loaded %d lines\n" length(linelist)
        catch e
            @printf "Warning: Could not read linelist: %s\n" e
            linelist = []  # Empty linelist instead of nothing
        end
    else
        println("No linelist provided - continuum only")
        linelist = []  # Empty linelist for continuum-only synthesis
    end
    
    # 3. Run synthesis to get opacity information
    println("Running stellar synthesis...")
    result = synthesize(atm, linelist, A_X, wavelengths_angstrom,
                       vmic=2.0)  # 2 km/s microturbulence
    
    # 4. Extract opacity data
    # result.alpha contains opacity at each layer and wavelength [layers x wavelengths]
    # result.cntm contains continuum flux
    n_layers, n_wavelengths = size(result.alpha)
    
    @printf "Synthesis complete: %d layers, %d wavelengths\n" n_layers n_wavelengths
    
    # Use layer 25 to match working test conditions (T≈4838K)
    layer_index = 25
    layer_opacity = result.alpha[layer_index, :]
    
    # Extract atmospheric conditions for this layer
    T = atm.layers[layer_index].temp
    P = atm.layers[layer_index].number_density * 1.380649e-16 * T
    n_e = atm.layers[layer_index].electron_number_density
    
    @printf "Layer %d: T=%.1fK, P=%.2e dyn/cm², n_e=%.2e cm⁻³\n" layer_index T P n_e
    
    # For demonstration, we'll estimate continuum vs line contributions
    # This is approximate since Korg doesn't separate them in the output
    if length(linelist) == 0
        # No lines - all opacity is continuum
        alpha_continuum = layer_opacity
        alpha_lines = zeros(length(layer_opacity))
    else
        # Rough estimate: assume minimum opacity is continuum level
        continuum_level = minimum(layer_opacity)
        alpha_continuum = fill(continuum_level, length(layer_opacity))
        alpha_lines = layer_opacity .- continuum_level
    end
    
    alpha_total = layer_opacity
    
    # Create results dictionary
    results = Dict(
        "wavelengths" => collect(result.wavelengths),
        "continuum_opacity" => collect(alpha_continuum),
        "line_opacity" => collect(alpha_lines),
        "total_opacity" => collect(alpha_total),
        "flux" => collect(result.flux),
        "continuum_flux" => result.cntm !== nothing ? collect(result.cntm) : nothing,
        "temperature" => T,
        "pressure" => P,
        "electron_density" => n_e,
        "layer_index" => layer_index,
        "stellar_params" => Dict("Teff" => Teff, "logg" => logg, "m_H" => m_H),
        "full_alpha_matrix" => collect(result.alpha)  # Full opacity matrix
    )
    
    # DETAILED ANALYSIS: Save raw data for comparison
    
    # Save full opacity matrix for analysis
    writedlm("korg_full_opacity_matrix.txt", result.alpha)
    writedlm("korg_wavelengths.txt", result.wavelengths)
    
    # Save layer 25 detailed data
    analysis_data = [
        result.wavelengths layer_opacity alpha_continuum alpha_lines
    ]
    writedlm("korg_layer25_analysis.txt", analysis_data)
    
    # Print detailed analysis
    println("\n" * "="^60)
    println("DETAILED KORG.JL OPACITY ANALYSIS")
    println("="^60)
    @printf "Layer %d conditions:\n" layer_index
    @printf "  Temperature: %.3f K\n" T
    @printf "  Pressure: %.6e dyn/cm²\n" P  
    @printf "  Electron density: %.6e cm⁻³\n" n_e
    
    println("\nOpacity Statistics:")
    @printf "  Total opacity range: %.3e - %.3e cm⁻¹\n" minimum(alpha_total) maximum(alpha_total)
    @printf "  Continuum (min baseline): %.3e cm⁻¹\n" minimum(alpha_total)
    @printf "  Line peak: %.3e cm⁻¹\n" maximum(alpha_lines)
    @printf "  Total peak: %.3e cm⁻¹\n" maximum(alpha_total)
    
    # Wavelength-by-wavelength analysis for first 10 points
    println("\nFirst 10 wavelength points:")
    println("λ(Å)      Total       Continuum   Line")
    for i in 1:min(10, length(result.wavelengths))
        @printf "%.3f   %.3e   %.3e   %.3e\n" result.wavelengths[i] alpha_total[i] alpha_continuum[i] alpha_lines[i]
    end
    
    # Peak analysis
    max_idx = argmax(alpha_total)
    @printf "\nPeak opacity at %.3f Å: %.3e cm⁻¹\n" result.wavelengths[max_idx] alpha_total[max_idx]
    @printf "  Continuum contribution: %.3e cm⁻¹\n" alpha_continuum[max_idx]  
    @printf "  Line contribution: %.3e cm⁻¹\n" alpha_lines[max_idx]
    
    if maximum(alpha_continuum) > 0
        enhancement = maximum(alpha_total) / maximum(alpha_continuum)
        @printf "\nEnhancement factor: %.2fx\n" enhancement
    end
    
    println("\n✅ Raw data saved to:")
    println("  - korg_full_opacity_matrix.txt")
    println("  - korg_wavelengths.txt") 
    println("  - korg_layer25_analysis.txt")
    
    return results
end

# Main execution
function main()
    # Example usage
    wavelengths = collect(range(5000, 5005, length=100))  # Small range for fast calculation
    
    # Basic solar parameters
    results = calculate_total_opacity(
        wavelengths,
        Teff=5780,
        logg=4.44,
        m_H=0.0,
        linelist_path="/Users/jdli/Project/Korg.jl/test/data/linelists/5000-5005.vald"
    )
    
    println("\n✅ Total opacity calculation complete!")
    @printf "Available in 'results' dict with keys: %s\n" join(keys(results), ", ")
    
    # Save results to file (simplified - just print summary)
    println("Results dictionary contains opacity data for $(length(results["wavelengths"])) wavelengths")
    
    return results
end

# Run main function if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end