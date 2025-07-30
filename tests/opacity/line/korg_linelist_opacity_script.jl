#!/usr/bin/env julia
"""
Korg.jl Linelist Opacity Calculation Script
=========================================

Script to calculate line opacity using Korg.jl with a full linelist
covering the 5000-6000 Å range.
"""

using Korg
using Printf
using Statistics

function calculate_korg_linelist_opacity()
    """Complete Korg.jl linelist opacity calculation"""
    
    println("🌟 KORG.JL LINELIST OPACITY CALCULATION")
    println("="^50)
    
    # === STEP 1: Define atmospheric conditions ===
    println("📊 Setting up atmospheric conditions...")
    
    temperature = 5780.0          # K (solar effective temperature)
    electron_density = 1e14       # cm⁻³
    hydrogen_density = 1e16       # cm⁻³
    microturbulence_kms = 2.0     # km/s
    
    # Convert microturbulence to cm/s (IMPORTANT!)
    microturbulence_cms = microturbulence_kms * 1e5
    
    @printf "  Temperature: %.0f K\n" temperature
    @printf "  Electron density: %.1e cm⁻³\n" electron_density
    @printf "  Hydrogen density: %.1e cm⁻³\n" hydrogen_density
    @printf "  Microturbulence: %.1f km/s = %.1e cm/s\n" microturbulence_kms microturbulence_cms
    
    # === STEP 2: Load linelist ===
    println("\n📖 Loading linelist...")
    
    # Try multiple linelist files in order of preference - USE SMALL LINELIST FIRST FOR DEBUGGING
    linelist_files = [
        "/Users/jdli/Project/Korg.jl/test/data/linelists/5000-5005.vald",
        "/Users/jdli/Project/Korg.jl/data/linelists/vald_extract_stellar_solar_threshold001.vald"
    ]
    
    linelist_file = nothing
    for file in linelist_files
        if isfile(file)
            linelist_file = file
            break
        end
    end
    
    if linelist_file === nothing
        println("❌ No suitable linelist found. Tried:")
        for file in linelist_files
            println("   $file")
        end
        println("   Please ensure a VALD format linelist is available.")
        return nothing
    end
    
    # Read linelist using Korg.jl
    linelist = Korg.read_linelist(linelist_file)
    
    @printf "  Linelist file: %s\n" linelist_file
    @printf "  Number of lines: %d\n" length(linelist)
    
    # Print first few lines for verification
    println("\n  First 5 lines:")
    for i in 1:min(5, length(linelist))
        line = linelist[i]
        @printf "    %.1f Å, %s, log(gf)=%.2f, E_low=%.2f eV\n" (line.wl * 1e8) string(line.species) line.log_gf line.E_lower
    end
    
    # === STEP 3: Create wavelength grid ===
    println("\n📏 Creating wavelength grid...")
    
    λ_start, λ_stop = 5000.0, 5005.0  # Å (FULL RANGE)
    n_points = 100  # High resolution for full range
    wl_range = range(λ_start, λ_stop, length=n_points)
    
    # Create Korg Wavelengths object (converts to cm internally)
    λs = Korg.Wavelengths(wl_range)
    
    @printf "  Range: %.1f - %.1f Å\n" λ_start λ_stop
    @printf "  Points: %d\n" n_points
    @printf "  Resolution: %.3f Å\n" (λ_stop - λ_start) / (n_points - 1)
    
    # === STEP 4: Setup default atmosphere and abundances ===
    println("\n🧮 Setting up default atmosphere and abundances...")
    
    # Use Korg's default solar abundances (Asplund et al. 2009)
    A_X = Korg.format_A_X(0.0)  # Solar abundances ([M/H] = 0.0)
    abs_abundances = @. 10^(A_X - 12)  # Convert from log scale
    abs_abundances ./= sum(abs_abundances)  # Normalize to sum to 1
    
    @printf "  Using default solar abundances (Asplund et al. 2009)\n"
    @printf "  Atmospheric conditions:\n"
    @printf "    Temperature: %.0f K\n" temperature
    @printf "    Hydrogen density: %.1e cm⁻³\n" hydrogen_density
    @printf "    Electron density: %.1e cm⁻³\n" electron_density
    @printf "    Microturbulence: %.1e cm/s\n" microturbulence_cms
    
    # Calculate chemical equilibrium to get proper number densities
    println("\n  Calculating chemical equilibrium...")
    nₑ, number_densities = Korg.chemical_equilibrium(
        temperature,
        hydrogen_density,
        electron_density,
        abs_abundances,
        Korg.ionization_energies,
        Korg.default_partition_funcs,
        Korg.default_log_equilibrium_constants
    )
    
    @printf "  Chemical equilibrium calculated: nₑ = %.2e cm⁻³\n" nₑ
    
    # Show key species
    key_species = [
        ("H I", Korg.species"H_I"),
        ("Fe I", Korg.species"Fe_I"),
        ("Fe II", Korg.species"Fe_II"),
        ("Ti I", Korg.species"Ti_I"),
        ("Ca I", Korg.species"Ca_I")
    ]
    
    println("  Key species number densities:")
    for (name, species) in key_species
        if haskey(number_densities, species)
            @printf "    %-6s: %.2e cm⁻³\n" name number_densities[species]
        end
    end
    
    # === STEP 5: Setup continuum opacity ===
    println("\n🌊 Setting up continuum opacity...")
    
    # Use default continuum opacity (set to zero for line-only comparison)
    α_cntm = [λ -> 0.0]  # Zero continuum for Jorg comparison
    
    @printf "  Continuum opacity: 0.0 cm⁻¹ (ZERO for Jorg comparison)\n"
    
    # === STEP 6: Calculate line absorption ===
    println("\n🔄 Calculating line absorption...")
    
    # Pre-allocate opacity array: [n_layers × n_wavelengths]
    α = zeros(1, length(λs))
    
    # Calculate line absorption using Korg.jl default method
    start_time = time()
    
    Korg.line_absorption!(
        α,                                    # opacity array (modified in-place)
        linelist,                            # list of spectral lines
        λs,                                  # wavelength grid
        [temperature],                       # temperature(s) - vector
        [nₑ],                               # electron density(ies) - vector
        number_densities,                    # species number densities
        Korg.default_partition_funcs,        # partition functions
        microturbulence_cms,                 # microturbulence in cm/s
        α_cntm                              # continuum opacity function(s)
    )
    calc_time = time() - start_time
    
    @printf "✅ Calculation completed in %.3f seconds\n" calc_time
    
    # === STEP 7: Analyze results ===
    println("\n📈 Analyzing results...")
    
    max_opacity = maximum(α)
    max_idx = argmax(α[1, :])
    peak_wavelength = wl_range[max_idx]
    mean_opacity = mean(α[1, :])
    integrated_opacity = sum(α[1, :]) * (λ_stop - λ_start) / n_points * 1e-8  # cm⁻¹⋅cm
    
    @printf "  Maximum opacity: %.3e cm⁻¹\n" max_opacity
    @printf "  Peak wavelength: %.2f Å\n" peak_wavelength
    @printf "  Mean opacity: %.3e cm⁻¹\n" mean_opacity
    @printf "  Integrated opacity: %.3e cm⁻¹⋅cm\n" integrated_opacity
    
    # Find top 10 opacity values and their wavelengths
    sorted_indices = sortperm(α[1, :], rev=true)
    println("\n🔍 Top 10 opacity peaks:")
    for i in 1:min(10, length(sorted_indices))
        idx = sorted_indices[i]
        wl = wl_range[idx]
        opacity = α[1, idx]
        @printf "    %.2f Å: %.3e cm⁻¹\n" wl opacity
    end
    
    # === STEP 8: Save data ===
    println("\n💾 Saving results...")
    
    # Save as simple text file
    output_file = "korg_line_opacity_0716.txt"
    open(output_file, "w") do f
        println(f, "# Korg.jl VALD Linelist Opacity Results - Range 5000-6000 Å")
        println(f, "# Linelist: $(linelist_file)")
        println(f, "# Number of lines: $(length(linelist))")
        println(f, "# Temperature: $(temperature) K")
        println(f, "# Electron density: $(electron_density) cm⁻³")
        println(f, "# Hydrogen density: $(hydrogen_density) cm⁻³")
        println(f, "# Microturbulence: $(microturbulence_kms) km/s")
        println(f, "# Wavelength(Å)  Opacity(cm⁻¹)")
        for (i, wl) in enumerate(wl_range)
            println(f, "$(wl)  $(α[1, i])")
        end
    end
    
    println("💾 Data saved as: $output_file")
    
    return α, wl_range, linelist

end

# === MAIN EXECUTION ===
if abspath(PROGRAM_FILE) == @__FILE__
    results = calculate_korg_linelist_opacity()
    
    println("\n" * "="^50)
    println("🎉 Script completed successfully!")
    println("Key files created:")
    println("  • korg_vald_opacity_results.txt - Numerical results")
    println("  • Ready for Jorg comparison with VALD format data")
end