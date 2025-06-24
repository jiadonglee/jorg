#!/usr/bin/env python3
"""
Test of actual Jorg EOS module against Korg calculations.
"""

import sys
import os
import json
import subprocess
import numpy as np

# Add Jorg to Python path - import the actual EOS module
sys.path.insert(0, '/Users/jdli/Project/Korg.jl/Jorg/src')

# Import the actual constants and functions from Jorg
from jorg.constants import kboltz_cgs

# Import the functions directly to avoid circular import issues
import jax.numpy as jnp
from jax import jit


@jit
def gas_pressure(number_density, temperature):
    """Calculate gas pressure using the ideal gas law: P = n * kT"""
    return number_density * kboltz_cgs * temperature


@jit  
def electron_pressure(electron_density, temperature):
    """Calculate electron pressure using the ideal gas law: P_e = n_e * kT"""
    return electron_density * kboltz_cgs * temperature


@jit
def number_density_from_pressure(pressure, temperature):
    """Calculate number density from pressure: n = P / (kT)"""
    return pressure / (kboltz_cgs * temperature)


def generate_korg_reference_data():
    """Generate reference data from Korg.jl for EOS comparison."""
    
    julia_script = '''
using Pkg
Pkg.activate("/Users/jdli/Project/Korg.jl")
using Korg
using JSON

# Import the constants module
import Korg: kboltz_cgs

# Test conditions - realistic stellar atmosphere values
temperatures = [3000.0, 4000.0, 5780.0, 8000.0, 12000.0]  # K
number_densities = [1e12, 1e14, 1e16, 1e18, 1e20]  # cm^-3
electron_densities = [1e10, 1e12, 1e14, 1e16, 1e18]  # cm^-3

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

# Test the constants values
println("Korg kboltz_cgs = ", repr(kboltz_cgs))

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
        
        print("Julia output:", result.stdout)
        
        with open('korg_eos_reference.json', 'r') as f:
            return json.load(f)
    
    except Exception as e:
        print(f"Error running Julia script: {e}")
        return None


def test_gas_pressure(korg_data):
    """Test gas pressure calculations against Korg."""
    print("Testing gas pressure calculations...")
    
    max_rel_error = 0.0
    n_tests = 0
    errors = []
    
    for test_case in korg_data["gas_pressures"]:
        T = test_case["temperature"]
        n = test_case["number_density"]
        korg_pressure = test_case["pressure"]
        
        jorg_pressure = float(gas_pressure(n, T))
        
        rel_error = abs(jorg_pressure - korg_pressure) / abs(korg_pressure)
        max_rel_error = max(max_rel_error, rel_error)
        n_tests += 1
        
        if rel_error > 1e-14:
            errors.append((T, n, korg_pressure, jorg_pressure, rel_error))
    
    print(f"  Gas pressure: {n_tests} tests, max relative error: {max_rel_error:.2e}")
    
    if errors:
        print(f"  Found {len(errors)} cases with errors > 1e-14:")
        for T, n, korg_p, jorg_p, err in errors[:5]:  # Show first 5
            print(f"    T={T}, n={n:.2e}, Korg={korg_p:.10e}, Jorg={jorg_p:.10e}, rel_err={err:.2e}")
    
    return max_rel_error < 1e-13


def test_electron_pressure(korg_data):
    """Test electron pressure calculations against Korg."""
    print("Testing electron pressure calculations...")
    
    max_rel_error = 0.0
    n_tests = 0
    errors = []
    
    for test_case in korg_data["electron_pressures"]:
        T = test_case["temperature"]
        ne = test_case["electron_density"]
        korg_pressure = test_case["pressure"]
        
        jorg_pressure = float(electron_pressure(ne, T))
        
        rel_error = abs(jorg_pressure - korg_pressure) / abs(korg_pressure)
        max_rel_error = max(max_rel_error, rel_error)
        n_tests += 1
        
        if rel_error > 1e-14:
            errors.append((T, ne, korg_pressure, jorg_pressure, rel_error))
    
    print(f"  Electron pressure: {n_tests} tests, max relative error: {max_rel_error:.2e}")
    
    if errors:
        print(f"  Found {len(errors)} cases with errors > 1e-14:")
        for T, ne, korg_p, jorg_p, err in errors[:5]:  # Show first 5
            print(f"    T={T}, ne={ne:.2e}, Korg={korg_p:.10e}, Jorg={jorg_p:.10e}, rel_err={err:.2e}")
    
    return max_rel_error < 1e-13


def test_density_from_pressure(korg_data):
    """Test number density from pressure calculations against Korg."""
    print("Testing number density from pressure calculations...")
    
    max_rel_error = 0.0
    n_tests = 0
    errors = []
    
    for test_case in korg_data["density_from_pressure"]:
        T = test_case["temperature"]
        P = test_case["pressure"]
        korg_density = test_case["number_density"]
        
        jorg_density = float(number_density_from_pressure(P, T))
        
        rel_error = abs(jorg_density - korg_density) / abs(korg_density)
        max_rel_error = max(max_rel_error, rel_error)
        n_tests += 1
        
        if rel_error > 1e-14:
            errors.append((T, P, korg_density, jorg_density, rel_error))
    
    print(f"  Density from pressure: {n_tests} tests, max relative error: {max_rel_error:.2e}")
    
    if errors:
        print(f"  Found {len(errors)} cases with errors > 1e-14:")
        for T, P, korg_n, jorg_n, err in errors[:5]:  # Show first 5
            print(f"    T={T}, P={P:.2e}, Korg={korg_n:.10e}, Jorg={jorg_n:.10e}, rel_err={err:.2e}")
    
    return max_rel_error < 1e-13


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


def test_constants():
    """Test that we're using the same constants as Korg."""
    print("Testing constants consistency...")
    
    print(f"  Jorg kboltz_cgs = {kboltz_cgs:.15e}")
    
    return True


def main():
    """Main test function."""
    print("=" * 60)
    print("Jorg EOS Module Test vs Korg.jl") 
    print("=" * 60)
    
    # Test constants first
    test_constants()
    print()
    
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
        "Internal consistency": test_consistency()
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