
using Pkg
Pkg.activate("/Users/jdli/Project/Korg.jl")
using Korg
using JSON

# Test parameters - simpler approach
test_conditions = [
    Dict("name" => "Cool", "Teff" => 3500, "logg" => 4.5, "m_H" => 0.0),
    Dict("name" => "Solar", "Teff" => 5777, "logg" => 4.44, "m_H" => 0.0),
    Dict("name" => "Hot", "Teff" => 8000, "logg" => 4.0, "m_H" => 0.0)
]

results = Dict()

for condition in test_conditions
    try
        name = condition["name"]
        Teff = condition["Teff"]
        logg = condition["logg"]
        m_H = condition["m_H"]
        
        println("Processing $name...")
        
        # Generate atmosphere using simpler approach
        atm = interpolate_marcs(Teff, logg, m_H)
        
        # Extract basic atmosphere data
        n_layers = length(atm.layers)
        layer_data = []
        
        for (i, layer) in enumerate(atm.layers[1:min(5, n_layers)])  # First 5 layers
            push!(layer_data, Dict(
                "temperature" => layer.temp,
                "total_density" => layer.number_density,
                "electron_density" => layer.electron_number_density,
                "tau_5000" => layer.tau_5000
            ))
        end
        
        # Calculate pressures using Korg's approach (atmosphere.jl:139)
        pressure_data = []
        for layer in atm.layers[1:min(5, n_layers)]
            P_gas = layer.number_density * kboltz_cgs * layer.temp
            P_e = layer.electron_number_density * kboltz_cgs * layer.temp
            push!(pressure_data, Dict(
                "gas_pressure" => P_gas,
                "electron_pressure" => P_e,
                "pressure_ratio" => P_e / P_gas
            ))
        end
        
        # Test basic synthesis on tiny wavelength range
        try
            wls = 5500.0:10.0:5520.0  # Very small range
            wavelengths, flux, continuum = synth(atm, wls)
            
            synthesis_success = true
            flux_stats = Dict(
                "mean_flux" => mean(flux),
                "mean_continuum" => mean(continuum),
                "n_points" => length(flux)
            )
        catch e
            println("Synthesis failed: $e")
            synthesis_success = false
            flux_stats = Dict("error" => string(e))
        end
        
        results[name] = Dict(
            "stellar_params" => condition,
            "atmosphere_basics" => Dict(
                "n_layers" => n_layers,
                "layer_sample" => layer_data,
                "pressure_sample" => pressure_data
            ),
            "synthesis_test" => Dict(
                "success" => synthesis_success,
                "stats" => flux_stats
            )
        )
        
    catch e
        println("Error with $name: $e")
        results[condition["name"]] = Dict("error" => string(e))
    end
end

# Test some basic calculations
println("Testing basic calculations...")

# EOS calculations
test_eos = Dict()
T_test = 5777.0
n_test = 1e16
ne_test = 1e14

P_gas_test = n_test * kboltz_cgs * T_test
P_e_test = ne_test * kboltz_cgs * T_test
n_from_P = P_gas_test / (kboltz_cgs * T_test)

test_eos["basic_eos"] = Dict(
    "temperature" => T_test,
    "input_density" => n_test,
    "calculated_pressure" => P_gas_test,
    "recovered_density" => n_from_P,
    "electron_test" => Dict(
        "input_ne" => ne_test,
        "calculated_Pe" => P_e_test
    )
)

results["basic_calculations"] = test_eos

# Save results
open("korg_focused_data.json", "w") do f
    JSON.print(f, results, 2)
end

println("Korg focused data generation complete")
