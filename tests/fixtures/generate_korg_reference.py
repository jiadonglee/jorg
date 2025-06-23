"""
Generate reference data from Korg.jl for direct comparison
"""

def generate_julia_script():
    """Generate Julia script to create reference Voigt data"""
    julia_script = '''
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
println("\\nTesting line profile...")

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
println("\\nTesting Harris series...")
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

open("../korg_reference_voigt.json", "w") do f
    JSON.print(f, reference_data, 2)
end

println("\\nReference data saved to ../korg_reference_voigt.json")
'''
    
    with open("../generate_korg_reference.jl", "w") as f:
        f.write(julia_script)
    
    print("Julia script written to ../generate_korg_reference.jl")
    print("Run this with: julia ../generate_korg_reference.jl")
    return julia_script

def test_against_reference():
    """Test jorg implementation against Korg reference if available"""
    import json
    import numpy as np
    from jorg.lines import voigt_hjerting, line_profile
    from jorg.lines.profiles import harris_series
    
    try:
        with open("../korg_reference_voigt.json", "r") as f:
            ref_data = json.load(f)
        
        print("=== Comparing against Korg.jl reference data ===")
        
        # Test Voigt-Hjerting function
        print("\\nVoigt-Hjerting comparison:")
        print("Alpha    v       Korg        Jorg       Diff       Rel.Err")
        print("-" * 65)
        
        max_voigt_error = 0.0
        for case in ref_data["voigt_hjerting"]:
            alpha, v = case["alpha"], case["v"]
            korg_H = case["H"]
            jorg_H = float(voigt_hjerting(alpha, v))
            
            diff = abs(jorg_H - korg_H)
            rel_err = diff / abs(korg_H) if korg_H != 0 else 0
            max_voigt_error = max(max_voigt_error, rel_err)
            
            print(f"{alpha:5.1f}  {v:5.1f}  {korg_H:10.6e}  {jorg_H:10.6e}  {diff:8.2e}  {rel_err:8.2%}")
        
        # Test Harris series  
        print("\\nHarris series comparison:")
        print("v       Component   Korg        Jorg       Diff       Rel.Err")
        print("-" * 70)
        
        max_harris_error = 0.0
        for case in ref_data["harris_series"]:
            v = case["v"]
            jorg_H = harris_series(v)
            
            for i, component in enumerate(["H0", "H1", "H2"]):
                korg_val = case[component]
                jorg_val = float(jorg_H[i])
                
                diff = abs(jorg_val - korg_val)
                rel_err = diff / abs(korg_val) if korg_val != 0 else 0
                max_harris_error = max(max_harris_error, rel_err)
                
                print(f"{v:5.1f}   {component:9s}  {korg_val:10.6e}  {jorg_val:10.6e}  {diff:8.2e}  {rel_err:8.2%}")
        
        # Test line profile
        print("\\nLine profile comparison:")
        print("Wavelength(Å)    Korg        Jorg       Diff       Rel.Err")
        print("-" * 60)
        
        params = ref_data["test_parameters"]
        lambda_0 = params["lambda_0"]
        sigma = params["sigma"]
        gamma = params["gamma"]
        amplitude = params["amplitude"]
        
        max_profile_error = 0.0
        for case in ref_data["line_profile"]:
            wl = case["wavelength"]
            korg_prof = case["profile_value"]
            jorg_prof = float(line_profile(lambda_0, sigma, gamma, amplitude, np.array([wl]))[0])
            
            diff = abs(jorg_prof - korg_prof)
            rel_err = diff / abs(korg_prof) if korg_prof != 0 else 0
            max_profile_error = max(max_profile_error, rel_err)
            
            print(f"{wl*1e8:12.1f}  {korg_prof:10.6e}  {jorg_prof:10.6e}  {diff:8.2e}  {rel_err:8.2%}")
        
        print(f"\\n=== Summary ===")
        print(f"Max Voigt-Hjerting error: {max_voigt_error:.2%}")
        print(f"Max Harris series error: {max_harris_error:.2%}")
        print(f"Max line profile error: {max_profile_error:.2%}")
        
        return max_voigt_error, max_harris_error, max_profile_error
        
    except FileNotFoundError:
        print("Reference data not found. Run ../generate_korg_reference.jl first.")
        return None, None, None

if __name__ == "__main__":
    generate_julia_script()
    test_against_reference()