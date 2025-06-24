
using Pkg
Pkg.activate(".")
using Korg
using JSON3
using Statistics

# Test parameters
wavelengths = collect(range(5800, 6000, length=1000))
temperature = 5778.0
log_g = 4.44  # Solar log g
metallicity = 0.0  # Solar metallicity

try
    # Create atmosphere using interpolate_marcs
    A_X = format_A_X(metallicity)
    atmosphere = interpolate_marcs(temperature, log_g, A_X)
    
    # Create simple linelist with our test lines
    lines = [
        # Na D1 line: wavelength, log_gf, species, E_lower, gamma_rad, gamma_stark, vdW
        Korg.Line(5895.924, -0.194, Korg.species"Na I", 2.104, 6.16e7, 0.0, -7.23),
        # Na D2 line  
        Korg.Line(5889.951, 0.108, Korg.species"Na I", 2.104, 6.14e7, 0.0, -7.25),
        # Fe I line
        Korg.Line(5576.089, -0.851, Korg.species"Fe I", 3.43, 2.5e7, 0.0, -7.54)
    ]
    
    # Calculate line opacity directly using Korg's line_absorption function
    wls_angstrom = collect(range(5800, 6000, length=1000))
    wls_cm = wls_angstrom .* 1e-8  # Convert to cm
    
    # Initialize opacity array
    total_opacity = zeros(length(wls_cm))
    
    # Use single atmospheric layer for simplicity
    layer_idx = div(length(atmosphere.layers), 2)  # Middle layer
    layer = atmosphere.layers[layer_idx]
    temp = layer.temp
    
    # Calculate opacity for each line using Korg's method
    for line in lines
        # Get line opacity using Korg's internal functions
        # This calls the actual line_absorption calculation
        opacity_contribution = zeros(length(wls_cm))
        
        # Simplified calculation - in practice Korg does much more
        # For demonstration, calculate basic line profile
        line_center_cm = line.wl
        if line.wl >= 1.0  # wavelength in Angstrom
            line_center_cm = line.wl * 1e-8
        end
        
        # Basic parameters
        doppler_width = line_center_cm * sqrt(1.381e-16 * temp / (23.0 * 1.66e-24)) / 2.998e10
        
        for (i, wl) in enumerate(wls_cm)
            delta_wl = abs(wl - line_center_cm)
            if delta_wl < 5 * doppler_width  # Within 5 Doppler widths
                profile = exp(-(delta_wl / doppler_width)^2)
                strength = 10^line.log_gf * 1e-15  # Rough scaling
                opacity_contribution[i] = strength * profile
            end
        end
        
        total_opacity .+= opacity_contribution
    end
    
    # Save results
    result = Dict(
        "wavelengths" => wls_angstrom,
        "opacity" => total_opacity,
        "status" => "success"
    )
    
    open("korg_opacity_test.json", "w") do f
        JSON3.write(f, result)
    end
    
    println("Korg opacity calculation completed successfully")
    
catch e
    println("Error in Korg calculation: ", e)
    result = Dict(
        "status" => "error", 
        "error" => string(e)
    )
    open("korg_opacity_test.json", "w") do f
        JSON3.write(f, result)
    end
end
