
using Pkg
Pkg.activate("/Users/jdli/Project/Korg.jl")
using Korg
using JSON

# Import the constant
import Korg: kboltz_cgs

println("Korg kboltz_cgs = ", kboltz_cgs)

# Test basic EOS calculations
test_cases = [
    Dict("T" => 3500.0, "n_total" => 1e15, "n_e" => 1e13),
    Dict("T" => 5777.0, "n_total" => 1e16, "n_e" => 1e14),
    Dict("T" => 8000.0, "n_total" => 1e14, "n_e" => 1e14)
]

results = Dict("eos_tests" => [])

for case in test_cases
    T = case["T"]
    n_total = case["n_total"]
    n_e = case["n_e"]
    
    # Calculate pressures using Korg's method (atmosphere.jl:139)
    P_gas = n_total * kboltz_cgs * T
    P_e = n_e * kboltz_cgs * T
    
    # Calculate density from pressure (inverse)
    n_recovered = P_gas / (kboltz_cgs * T)
    
    push!(results["eos_tests"], Dict(
        "input" => case,
        "korg_P_gas" => P_gas,
        "korg_P_e" => P_e,
        "korg_n_recovered" => n_recovered,
        "pressure_ratio" => P_e / P_gas
    ))
end

# Test atmosphere generation for one simple case
try
    println("Testing atmosphere generation...")
    atm = interpolate_marcs(5777.0, 4.44, 0.0)  # Solar case
    
    # Get first layer data
    first_layer = atm.layers[1]
    
    atm_data = Dict(
        "success" => true,
        "n_layers" => length(atm.layers),
        "first_layer" => Dict(
            "temperature" => first_layer.temp,
            "total_density" => first_layer.number_density,
            "electron_density" => first_layer.electron_number_density,
            "tau_5000" => first_layer.tau_5000
        )
    )
    
    # Calculate pressures for first layer
    layer_P_gas = first_layer.number_density * kboltz_cgs * first_layer.temp
    layer_P_e = first_layer.electron_number_density * kboltz_cgs * first_layer.temp
    
    atm_data["first_layer"]["P_gas"] = layer_P_gas
    atm_data["first_layer"]["P_e"] = layer_P_e
    
    results["atmosphere_test"] = atm_data
    
catch e
    println("Atmosphere generation failed: ", e)
    results["atmosphere_test"] = Dict("success" => false, "error" => string(e))
end

# Test basic synthesis
try
    println("Testing basic synthesis...")
    atm = interpolate_marcs(5777.0, 4.44, 0.0)
    wls = 5500.0:50.0:5550.0  # Very small range
    
    wavelengths, flux, continuum = synth(atm, wls)
    
    synth_data = Dict(
        "success" => true,
        "n_points" => length(flux),
        "wavelength_range" => [minimum(wavelengths), maximum(wavelengths)],
        "flux_range" => [minimum(flux), maximum(flux)],
        "continuum_range" => [minimum(continuum), maximum(continuum)],
        "mean_flux" => mean(flux),
        "mean_continuum" => mean(continuum)
    )
    
    results["synthesis_test"] = synth_data
    
catch e
    println("Synthesis failed: ", e)
    results["synthesis_test"] = Dict("success" => false, "error" => string(e))
end

# Save results
open("simple_korg_results.json", "w") do f
    JSON.print(f, results, 2)
end

println("Simple Korg test complete")
