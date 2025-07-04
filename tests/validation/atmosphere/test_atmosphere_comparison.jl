using Korg

"""
Test Atmosphere Interpolation: Direct Korg Implementation
========================================================

This script tests Korg's atmosphere interpolation directly and outputs
results in a format that can be compared with Jorg's implementation.
"""

function test_korg_atmosphere_interpolation()
    println("KORG ATMOSPHERE INTERPOLATION TEST")
    println(repeat("=", 60))
    
    # Test cases matching Jorg test
    test_cases = [
        ("Solar-type G star", 5777.0, 4.44, 0.0),
        ("Cool K-type star", 4500.0, 4.5, 0.0),
        ("Cool M dwarf", 3500.0, 4.8, 0.0),
        ("Giant K star", 4500.0, 2.5, 0.0),
        ("Metal-poor G star", 5777.0, 4.44, -1.0),
        ("Metal-rich G star", 5777.0, 4.44, +0.3)
    ]
    
    all_results = []
    
    for (description, Teff, logg, M_H) in test_cases
        println("\nTesting: $description")
        println("Stellar Parameters: Teff=$(Teff)K, logg=$(logg), [M/H]=$(M_H)")
        
        try
            # Format abundances
            A_X = Korg.format_A_X(M_H)
            
            # Interpolate atmosphere
            atm = Korg.interpolate_marcs(Teff, logg, A_X)
            
            println("✅ Interpolation successful")
            println("   Layers: $(length(atm.layers))")
            println("   Type: $(typeof(atm) <: Korg.ShellAtmosphere ? "Spherical" : "Planar")")
            if typeof(atm) <: Korg.ShellAtmosphere
                println("   Radius: $(atm.R:.2e) cm")
            end
            
            # Test specific layers
            test_layers = [15, 25, 35]
            layer_results = []
            
            for layer_idx in test_layers
                if layer_idx <= length(atm.layers)
                    layer = atm.layers[layer_idx]
                    
                    layer_result = (
                        layer = layer_idx,
                        T = layer.temp,
                        nt = layer.number_density,
                        ne = layer.electron_number_density,
                        P = layer.number_density * Korg.kboltz_cgs * layer.temp,
                        tau_5000 = layer.tau_5000,
                        z = layer.z
                    )
                    push!(layer_results, layer_result)
                    
                    println("   Layer $layer_idx: T=$(round(layer.temp, digits=1))K, " *
                           "nt=$(round(layer.number_density, sigdigits=3)), " *
                           "ne=$(round(layer.electron_number_density, sigdigits=3))")
                end
            end
            
            # Validate physical consistency
            validate_atmosphere_physics(atm, description)
            
            push!(all_results, (description, Teff, logg, M_H, atm, layer_results))
            
        catch e
            println("❌ Interpolation failed: $e")
        end
    end
    
    return all_results
end

function validate_atmosphere_physics(atm, description)
    """Validate that the atmosphere is physically consistent"""
    
    println("   Physical validation:")
    
    # Check temperature structure
    temps = [layer.temp for layer in atm.layers]
    temp_increasing = all(temps[i] <= temps[i+1] for i in 1:length(temps)-1)
    println("     Temperature increases with depth: $(temp_increasing ? "✅" : "❌")")
    
    # Check density structure  
    densities = [layer.number_density for layer in atm.layers]
    density_increasing = all(densities[i] <= densities[i+1] for i in 1:length(densities)-1)
    println("     Density increases with depth: $(density_increasing ? "✅" : "❌")")
    
    # Check optical depth structure
    taus = [layer.tau_5000 for layer in atm.layers]
    tau_increasing = all(taus[i] <= taus[i+1] for i in 1:length(taus)-1)
    println("     Optical depth increases with depth: $(tau_increasing ? "✅" : "❌")")
    
    # Check electron density fractions
    valid_ne_fractions = true
    for layer in atm.layers
        ne_fraction = layer.electron_number_density / layer.number_density
        if !(1e-15 < ne_fraction < 0.5)
            valid_ne_fractions = false
            break
        end
    end
    println("     Electron density fractions reasonable: $(valid_ne_fractions ? "✅" : "❌")")
end

function generate_comparison_data(results)
    """Generate data for comparison with Jorg"""
    
    println("\n\nGENERATING COMPARISON DATA FOR JORG")
    println(repeat("=", 60))
    
    println("# Korg atmosphere interpolation results for comparison")
    println("korg_atmosphere_results = {")
    
    for (i, (description, Teff, logg, M_H, atm, layer_results)) in enumerate(results)
        println("    \"$description\": {")
        println("        \"Teff\": $Teff,")
        println("        \"logg\": $logg,")
        println("        \"M_H\": $M_H,")
        println("        \"n_layers\": $(length(atm.layers)),")
        println("        \"spherical\": $(typeof(atm) <: Korg.ShellAtmosphere),")
        if typeof(atm) <: Korg.ShellAtmosphere
            println("        \"radius\": $(atm.R),")
        end
        println("        \"test_layers\": {")
        
        for layer_result in layer_results
            println("            $(layer_result.layer): {")
            println("                \"T\": $(layer_result.T),")
            println("                \"nt\": $(layer_result.nt),")
            println("                \"ne\": $(layer_result.ne),")
            println("                \"P\": $(layer_result.P),")
            println("                \"tau_5000\": $(layer_result.tau_5000),")
            println("                \"z\": $(layer_result.z),")
            println("            },")
        end
        
        println("        }")
        print("    }")
        if i < length(results)
            println(",")
        else
            println()
        end
    end
    
    println("}")
end

function test_interpolation_methods()
    """Test different interpolation methods used by Korg"""
    
    println("\n\nTESTING INTERPOLATION METHODS")
    println(repeat("=", 60))
    
    # Test cases that trigger different interpolation schemes
    test_cases = [
        ("Standard interpolation", 5777.0, 4.44, 0.0),     # Standard SDSS grid
        ("Cool dwarf cubic", 3500.0, 4.8, 0.0),           # Cool dwarf cubic spline
        ("Low metallicity", 5777.0, 4.44, -3.0),          # Low-Z grid
        ("Giant spherical", 4500.0, 2.0, 0.0),            # Should be spherical
    ]
    
    for (description, Teff, logg, M_H) in test_cases
        println("\nTesting interpolation method: $description")
        println("Parameters: Teff=$(Teff)K, logg=$(logg), [M/H]=$(M_H)")
        
        try
            A_X = Korg.format_A_X(M_H)
            atm = Korg.interpolate_marcs(Teff, logg, A_X)
            
            println("✅ Success: $(length(atm.layers)) layers, $(typeof(atm) <: Korg.ShellAtmosphere ? "spherical" : "planar")")
            
            # Identify which interpolation method was used
            if Teff <= 4000 && logg >= 3.5 && M_H >= -2.5
                println("   Method: Cool dwarf cubic spline interpolation")
            elseif M_H < -2.5
                println("   Method: Low-metallicity grid interpolation")
            else
                println("   Method: Standard SDSS grid interpolation")
            end
            
        catch e
            println("❌ Failed: $e")
        end
    end
end

function main()
    println("KORG ATMOSPHERE INTERPOLATION VALIDATION")
    println(repeat("=", 80))
    println("Testing Korg atmosphere interpolation for comparison with Jorg")
    println()
    
    # Test main interpolation functionality
    results = test_korg_atmosphere_interpolation()
    
    # Test different interpolation methods
    test_interpolation_methods()
    
    # Generate comparison data
    generate_comparison_data(results)
    
    println("\n" * "=" * 80)
    println("✅ KORG ATMOSPHERE INTERPOLATION TEST COMPLETE")
    println("Tested $(length(results)) stellar types successfully")
    println("Copy the comparison data above for use in Jorg validation")
    println(repeat("=", 80))
    
    return results
end

# Run the test
main()