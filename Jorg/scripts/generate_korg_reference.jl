
using Korg
using JSON

# Test individual Voigt-Hjerting function
println("Testing Voigt-Hjerting function...")

voigt_test_cases = [
    (0.0, 0.0), (0.0, 1.0), (0.0, 2.0),
    (0.1, 0.0), (0.1, 1.0), (0.1, 5.0),
    (0.5, 0.0), (0.5, 2.0),
    (1.0, 1.0), (2.0, 1.0)
]

voigt_results = []
for (alpha, v) in voigt_test_cases
    H = Korg.voigt_hjerting(alpha, v)
    push!(voigt_results, Dict("alpha" => alpha, "v" => v, "H" => H))
    println("H($alpha, $v) = $H")
end

# Test line profile
println("\nTesting line profile...")

lambda_0 = 5000e-8  # 5000 Å in cm
sigma = 0.5e-8      # 0.5 Å Doppler width
gamma = 0.1e-8      # 0.1 Å Lorentz width
amplitude = 1.0

# Test at a few specific wavelengths
test_wavelengths = [lambda_0 - 2e-8, lambda_0 - 1e-8, lambda_0, lambda_0 + 1e-8, lambda_0 + 2e-8]
profile_results = []

for wl in test_wavelengths
    profile_val = Korg.line_profile(lambda_0, sigma, gamma, amplitude, wl)
    push!(profile_results, Dict("wavelength" => wl, "profile_value" => profile_val))
    println("Profile at λ = $(wl*1e8) Å: $profile_val")
end

# Test Harris series
println("\nTesting Harris series...")
harris_test_v = [0.0, 0.5, 1.0, 2.0, 4.0]
harris_results = []

for v in harris_test_v
    if v < 5.0  # Harris series only valid for v < 5
        H0, H1, H2 = Korg.harris_series(v)
        push!(harris_results, Dict("v" => v, "H0" => H0, "H1" => H1, "H2" => H2))
        println("Harris($v): H0=$H0, H1=$H1, H2=$H2")
    end
end

# Save all results
reference_data = Dict(
    "voigt_hjerting" => voigt_results,
    "line_profile" => profile_results, 
    "harris_series" => harris_results,
    "test_parameters" => Dict(
        "lambda_0" => lambda_0,
        "sigma" => sigma,
        "gamma" => gamma,
        "amplitude" => amplitude
    )
)

open("korg_reference_voigt.json", "w") do f
    JSON.print(f, reference_data, 2)
end

println("\nReference data saved to korg_reference_voigt.json")
