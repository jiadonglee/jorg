#!/usr/bin/env julia
"""
Korg.jl synthesize() Function - Concise Version
==============================================
"""

using Korg, Printf, Statistics, Dates

# CONFIGURATION
# =============
# Stellar type selection (uncomment one)
stellar_type = "arcturus"  # Options: "solar", "arcturus", "metal_poor_k_giant", "custom"

# Predefined stellar types
stellar_params = Dict{String, NamedTuple}(
    "solar" => (Teff=5771, logg=4.44, m_H=0.0),
    "arcturus" => (Teff=4250, logg=1.4, m_H=-0.5),
    "metal_poor_k_giant" => (Teff=4500, logg=1.5, m_H=-2.5),
    "custom" => (Teff=5780, logg=4.44, m_H=0.0),  # Edit these for custom parameters
)

# Get parameters for selected stellar type
params = stellar_params[stellar_type]
Teff = params.Teff      # Effective temperature (K)
logg = params.logg      # Surface gravity
m_H = params.m_H        # Metallicity [M/H]

# Other synthesis parameters
wavelengths = (5000, 5200)  # Wavelength range (Å)
vmic = 1.0       # Microturbulence (km/s)

# Output file names (automatically includes stellar type)
output_prefix = "korg_$(stellar_type)"  # Base name includes stellar type
spectrum_file = "$(output_prefix)_spectrum.txt"
atmosphere_file = "$(output_prefix)_atmosphere.txt"
opacity_file = "$(output_prefix)_opacity.txt"
species_file = "$(output_prefix)_species.txt"
rt_file = "$(output_prefix)_radiative_transfer.txt"

println("🔬 Korg.jl Synthesis: $stellar_type star (Teff=$Teff K, logg=$logg, [M/H]=$m_H)")

try
    # Setup atmosphere and linelist
    A_X = format_A_X(m_H)
    atm = interpolate_marcs(Teff, logg, A_X)
    
    linelist = []
    try
        linelist = get_VALD_solar_linelist()
    catch
        # Use empty linelist for continuum-only synthesis
    end
    
    println("📊 $(length(atm.layers)) layers, $(length(linelist)) lines")
    
    # Run synthesis
    start_time = time()
    result = synthesize(atm, linelist, A_X, wavelengths, vmic=vmic, return_cntm=true)
    elapsed_time = time() - start_time
    
    println("✅ Synthesis complete ($(round(elapsed_time, digits=1))s)")
    
    # Brief result summary
    normalized_flux = result.cntm !== nothing ? result.flux ./ result.cntm : result.flux
    max_absorption = result.cntm !== nothing ? (1.0 - minimum(normalized_flux)) * 100 : 0.0
    
    println("📈 $(length(result.wavelengths)) wavelengths, max absorption: $(round(max_absorption, digits=1))%")
    println("💾 Saving to files with prefix: $output_prefix")
    
    # Save results to files
    # 1. Spectrum data
    open(spectrum_file, "w") do file
        println(file, "# Korg.jl Synthesis - Teff=$Teff K, logg=$logg, [M/H]=$m_H")
        println(file, "# Generated: $(now())")
        if result.cntm !== nothing
            println(file, "# Wavelength(Å)    Flux              Continuum         NormalizedFlux")
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
    
    # 2. Atmosphere structure
    open(atmosphere_file, "w") do file
        println(file, "# Atmospheric Structure - $(now())")
        println(file, "# Layer  ElectronDensity(cm⁻³)")
        for i in 1:length(result.electron_number_density)
            @printf(file, "%5d  %15.6e\n", i, result.electron_number_density[i])
        end
    end
    
    # 3. Opacity summary
    open(opacity_file, "w") do file
        println(file, "# Opacity Matrix Summary - $(now())")
        println(file, "# Shape: $(size(result.alpha)) (layers × wavelengths)")
        println(file, "# Layer    MinOpacity      MaxOpacity      MeanOpacity")
        for layer in 1:size(result.alpha, 1)
            layer_opacity = result.alpha[layer, :]
            @printf(file, "%5d  %12.6e  %12.6e  %12.6e\n", 
                   layer, minimum(layer_opacity), maximum(layer_opacity), mean(layer_opacity))
        end
    end
    
    # 4. Species densities
    open(species_file, "w") do file
        println(file, "# Species Number Densities - $(now())")
        println(file, "# Species                Min Density     Max Density")
        for (species, densities) in result.number_densities
            @printf(file, "%-20s  %12.6e  %12.6e\n", 
                   string(species), minimum(densities), maximum(densities))
        end
    end
    
    # 5. Radiative transfer
    open(rt_file, "w") do file
        println(file, "# Radiative Transfer - $(now())")
        println(file, "# Intensity shape: $(size(result.intensity))")
        println(file, "# Index    μ_value      Weight")
        for (i, (mu_val, weight)) in enumerate(result.mu_grid)
            @printf(file, "%5d  %10.6f  %10.6f\n", i, mu_val, weight)
        end
    end
    
    println("✅ Files saved: $spectrum_file, $atmosphere_file, $opacity_file, $species_file, $rt_file")
    println("🎉 SUCCESS! Synthesis complete.")
    
catch e
    println("❌ Synthesis failed: $e")
    println("Check: Korg.jl installation, project environment, model atmosphere data")
end

println("🏁 Analysis complete!")