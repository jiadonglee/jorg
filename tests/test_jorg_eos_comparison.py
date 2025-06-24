#!/usr/bin/env python3
"""
Test script comparing Jorg EOS implementation with Korg behavior.

This script validates that Jorg's equation of state functions reproduce
the same results as Korg.jl for pressure calculations and thermodynamic
properties.
"""

import numpy as np
import json
import subprocess
import sys
import os

# Add Jorg to Python path
sys.path.insert(0, '/Users/jdli/Project/Korg.jl/Jorg/src')

from jorg.statmech import (
    gas_pressure, electron_pressure, total_pressure,
    number_density_from_pressure, pressure_scale_height,
    ideal_gas_density, pressure_from_density
)


def generate_korg_reference_data():
    """Generate reference data from Korg.jl for EOS comparison."""
    
    julia_script = '''
using Pkg
Pkg.activate("/Users/jdli/Project/Korg.jl")
using Korg
using Random
using JSON

# Test conditions - realistic stellar atmosphere values
temperatures = [3000.0, 4000.0, 5780.0, 8000.0, 12000.0]  # K
number_densities = [1e12, 1e14, 1e16, 1e18, 1e20]  # cm^-3
electron_densities = [1e10, 1e12, 1e14, 1e16, 1e18]  # cm^-3
gravities = [1e2, 1e3, 1e4, 1e5, 1e6]  # cm s^-2

results = Dict()

# Test gas pressure calculations (matching atmosphere.jl:139)
results["gas_pressures"] = []
for T in temperatures
    for n in number_densities
        P_gas = n * kboltz_cgs * T
        push!(results["gas_pressures"], Dict(
            "temperature" => T,
            "number_density" => n,
            "pressure" => P_gas
        ))
    end
end

# Test electron pressure calculations
results["electron_pressures"] = []
for T in temperatures
    for ne in electron_densities
        P_e = ne * kboltz_cgs * T
        push!(results["electron_pressures"], Dict(
            "temperature" => T,
            "electron_density" => ne,
            "pressure" => P_e
        ))
    end
end

# Test number density from pressure (inverse calculation)
results["density_from_pressure"] = []
for T in temperatures
    for P in [1e3, 1e5, 1e7, 1e9, 1e11]  # dyne cm^-2
        n = P / (kboltz_cgs * T)
        push!(results["density_from_pressure"], Dict(
            "temperature" => T,
            "pressure" => P,
            "number_density" => n
        ))
    end
end

# Write results to JSON
open("korg_eos_reference.json", "w") do f
    JSON.print(f, results, 2)
end

println("Korg reference data generated successfully")
'''
    
    with open('generate_korg_eos_reference.jl', 'w') as f:
        f.write(julia_script)
    
    try:
        result = subprocess.run(['julia', 'generate_korg_eos_reference.jl'], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"Julia script failed: {result.stderr}")
            return None
        
        with open('korg_eos_reference.json', 'r') as f:
            return json.load(f)
    
    except subprocess.TimeoutExpired:
        print("Julia script timed out")
        return None
    except FileNotFoundError:
        print("Julia not found. Please ensure Julia is installed and in PATH.")
        return None
    except Exception as e:
        print(f"Error running Julia script: {e}")
        return None


def test_gas_pressure(korg_data):
    """Test gas pressure calculations against Korg."""
    print("Testing gas pressure calculations...")
    
    max_rel_error = 0.0
    n_tests = 0
    
    for test_case in korg_data["gas_pressures"]:
        T = test_case["temperature"]
        n = test_case["number_density"]
        korg_pressure = test_case["pressure"]
        
        jorg_pressure = float(gas_pressure(n, T))
        
        rel_error = abs(jorg_pressure - korg_pressure) / abs(korg_pressure)
        max_rel_error = max(max_rel_error, rel_error)
        n_tests += 1
        
        if rel_error > 1e-12:
            print(f"  Large error: T={T}, n={n:.2e}, Korg={korg_pressure:.6e}, Jorg={jorg_pressure:.6e}, rel_err={rel_error:.2e}")
    
    print(f"  Gas pressure: {n_tests} tests, max relative error: {max_rel_error:.2e}")
    return max_rel_error < 1e-12


def test_electron_pressure(korg_data):
    """Test electron pressure calculations against Korg."""
    print("Testing electron pressure calculations...")
    
    max_rel_error = 0.0
    n_tests = 0
    
    for test_case in korg_data["electron_pressures"]:
        T = test_case["temperature"]
        ne = test_case["electron_density"]
        korg_pressure = test_case["pressure"]
        
        jorg_pressure = float(electron_pressure(ne, T))
        
        rel_error = abs(jorg_pressure - korg_pressure) / abs(korg_pressure)
        max_rel_error = max(max_rel_error, rel_error)
        n_tests += 1
        
        if rel_error > 1e-12:
            print(f"  Large error: T={T}, ne={ne:.2e}, Korg={korg_pressure:.6e}, Jorg={jorg_pressure:.6e}, rel_err={rel_error:.2e}")
    
    print(f"  Electron pressure: {n_tests} tests, max relative error: {max_rel_error:.2e}")
    return max_rel_error < 1e-12


def test_density_from_pressure(korg_data):
    """Test number density from pressure calculations against Korg."""
    print("Testing number density from pressure calculations...")
    
    max_rel_error = 0.0
    n_tests = 0
    
    for test_case in korg_data["density_from_pressure"]:
        T = test_case["temperature"]
        P = test_case["pressure"]
        korg_density = test_case["number_density"]
        
        jorg_density = float(number_density_from_pressure(P, T))
        
        rel_error = abs(jorg_density - korg_density) / abs(korg_density)
        max_rel_error = max(max_rel_error, rel_error)
        n_tests += 1
        
        if rel_error > 1e-12:
            print(f"  Large error: T={T}, P={P:.2e}, Korg={korg_density:.6e}, Jorg={jorg_density:.6e}, rel_err={rel_error:.2e}")
    
    print(f"  Density from pressure: {n_tests} tests, max relative error: {max_rel_error:.2e}")
    return max_rel_error < 1e-12


def test_consistency():
    """Test internal consistency of EOS functions."""
    print("Testing internal consistency...")
    
    # Test that P = n*k*T and n = P/(k*T) are consistent
    temperatures = np.array([3000.0, 5780.0, 10000.0])
    densities = np.array([1e12, 1e16, 1e20])
    
    max_error = 0.0
    n_tests = 0
    
    for T in temperatures:
        for n in densities:
            # Forward: n -> P
            P = float(gas_pressure(n, T))
            # Backward: P -> n
            n_recovered = float(number_density_from_pressure(P, T))
            
            rel_error = abs(n_recovered - n) / n
            max_error = max(max_error, rel_error)
            n_tests += 1
    
    print(f"  Consistency: {n_tests} tests, max relative error: {max_error:.2e}")
    return max_error < 1e-14


def test_additional_functions():
    """Test additional EOS functions for physical reasonableness."""
    print("Testing additional EOS functions...")
    
    # Test pressure scale height
    T = 5780.0  # Solar temperature
    g = 2.74e4  # Solar surface gravity
    mu = 1.3    # Typical mean molecular weight
    
    H_p = float(pressure_scale_height(T, g, mu))
    
    # For the Sun, pressure scale height should be ~100-200 km
    expected_range = (1e7, 3e7)  # cm
    
    print(f"  Pressure scale height: H_p = {H_p:.2e} cm ({H_p/1e5:.1f} km)")
    h_ok = expected_range[0] < H_p < expected_range[1]
    
    # Test density calculations
    P = 1e6  # dyne cm^-2
    rho = float(ideal_gas_density(P, T, mu))
    P_recovered = float(pressure_from_density(rho, T, mu))
    
    density_consistency = abs(P_recovered - P) / P < 1e-14
    
    print(f"  Density consistency: P={P:.2e} -> rho={rho:.2e} -> P={P_recovered:.2e}")
    print(f"    Relative error: {abs(P_recovered - P) / P:.2e}")
    
    return h_ok and density_consistency


def main():
    """Main test function."""
    print("=" * 60)
    print("Jorg EOS Implementation Test vs Korg.jl")
    print("=" * 60)
    
    # Generate reference data from Korg
    print("Generating Korg reference data...")
    korg_data = generate_korg_reference_data()
    
    if korg_data is None:
        print("Failed to generate Korg reference data. Exiting.")
        return False
    
    print(f"Generated {len(korg_data['gas_pressures'])} gas pressure test cases")
    print(f"Generated {len(korg_data['electron_pressures'])} electron pressure test cases")
    print(f"Generated {len(korg_data['density_from_pressure'])} density test cases")
    print()
    
    # Run tests
    all_passed = True
    
    test_results = {
        "Gas pressure": test_gas_pressure(korg_data),
        "Electron pressure": test_electron_pressure(korg_data),
        "Density from pressure": test_density_from_pressure(korg_data),
        "Internal consistency": test_consistency(),
        "Additional functions": test_additional_functions()
    }
    
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print("=" * 60)
    
    for test_name, passed in test_results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name:.<25} {status}")
        all_passed = all_passed and passed
    
    print(f"\nOverall result: {'PASS' if all_passed else 'FAIL'}")
    
    # Clean up temporary files
    for temp_file in ['generate_korg_eos_reference.jl', 'korg_eos_reference.json']:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)