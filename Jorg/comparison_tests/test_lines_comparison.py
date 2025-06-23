"""
Compare jorg.lines results with Korg.jl reference calculations

This script tests line absorption calculations by comparing JAX implementations
with reference results from Korg.jl.
"""

import numpy as np
import jax.numpy as jnp
import json
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jorg.lines import line_absorption, line_profile, voigt_hjerting
from jorg.lines.main import LineData, create_line_data
from jorg.constants import c_cgs, kboltz_eV, pi


def generate_korg_line_reference_data():
    """
    Generate reference line absorption data using Korg.jl
    
    This would be run from Julia to create reference data files.
    """
    julia_script = '''
using Korg
using JSON

# Test parameters
λs = collect(range(5890.0, 5900.0, length=200)) * 1e-8  # Convert Å to cm
temp = 5778.0  # Sun-like temperature
nₑ = 1e15     # Electron density
ξ = 1e5       # Microturbulent velocity (1 km/s)

# Create a simple test linelist with Na D lines
test_lines = [
    # Na I D2 line at 5889.95 Å
    (wavelength=5889.95e-8, log_gf=0.108, E_lower=0.0, species=species"Na_I", 
     gamma_rad=6.14e7, gamma_stark=2.8e-5, vdW=(1.4e-7, 0.3)),
    # Na I D1 line at 5895.92 Å  
    (wavelength=5895.92e-8, log_gf=-0.194, E_lower=0.0, species=species"Na_I",
     gamma_rad=6.14e7, gamma_stark=2.8e-5, vdW=(1.4e-7, 0.3))
]

# Number densities (simplified)
n_densities = Dict(
    species"Na_I" => 1e10,  # Sodium density
    species"H_I" => 1e15    # Hydrogen density for vdW broadening
)

# Partition functions (simplified - constant)
partition_fns = Dict(
    species"Na_I" => (log_T) -> 2.0
)

# Calculate line absorption
α = zeros(length(λs))
line_absorption!(α, test_lines, Korg.Wavelengths(λs), [temp], [nₑ], 
                [n_densities], partition_fns, [ξ])

# Also test individual components
# Voigt profile calculation
λ₀ = 5889.95e-8
σ = Korg.doppler_width(λ₀, temp, 23 * 1.66e-24, ξ)  # Na mass
γ = 1e-8  # Simple Lorentz width

voigt_test = [Korg.line_profile(λ₀, σ, γ, 1.0, λ) for λ in λs]

# Save reference data
reference_data = Dict(
    "wavelengths_cm" => λs,
    "temperature" => temp,
    "electron_density" => nₑ,
    "microturbulent_velocity" => ξ,
    "line_absorption_total" => α,
    "voigt_test_lambda0" => λ₀,
    "voigt_test_sigma" => σ,
    "voigt_test_gamma" => γ,
    "voigt_test_profile" => voigt_test,
    "test_lines" => [
        Dict(
            "wavelength_cm" => line.wavelength,
            "log_gf" => line.log_gf,
            "E_lower_eV" => line.E_lower,
            "species_id" => 11,  # Na
            "gamma_rad" => line.gamma_rad,
            "gamma_stark" => line.gamma_stark,
            "vdw_param1" => line.vdW[1],
            "vdw_param2" => line.vdW[2]
        ) for line in test_lines
    ]
)

# Write to JSON file
open("line_reference_data.json", "w") do f
    JSON.print(f, reference_data, 2)
end

println("Reference data saved to line_reference_data.json")
'''
    
    return julia_script


def create_mock_reference_data():
    """
    Create mock reference data for testing when Julia isn't available
    """
    # Create test wavelength grid
    wavelengths = np.linspace(5890e-8, 5900e-8, 200)  # 200 points from 5890-5900 Å
    
    # Mock line absorption data (realistic-looking absorption profile)
    center1 = 5889.95e-8
    center2 = 5895.92e-8
    
    # Create mock absorption profiles
    def mock_voigt_profile(wl, center, depth, width):
        x = (wl - center) / width
        return depth * np.exp(-x**2) / (1 + x**2)  # Simplified Voigt-like shape
    
    alpha_total = (mock_voigt_profile(wavelengths, center1, 0.5, 0.5e-8) + 
                   mock_voigt_profile(wavelengths, center2, 0.3, 0.5e-8))
    
    # Mock individual Voigt profile test
    voigt_profile_test = mock_voigt_profile(wavelengths, center1, 1.0, 0.3e-8)
    
    reference_data = {
        "wavelengths_cm": wavelengths.tolist(),
        "temperature": 5778.0,
        "electron_density": 1e15,
        "microturbulent_velocity": 1e5,
        "line_absorption_total": alpha_total.tolist(),
        "voigt_test_lambda0": center1,
        "voigt_test_sigma": 0.2e-8,
        "voigt_test_gamma": 0.1e-8,
        "voigt_test_profile": voigt_profile_test.tolist(),
        "test_lines": [
            {
                "wavelength_cm": center1,
                "log_gf": 0.108,
                "E_lower_eV": 0.0,
                "species_id": 11,
                "gamma_rad": 6.14e7,
                "gamma_stark": 2.8e-5,
                "vdw_param1": 1.4e-7,
                "vdw_param2": 0.3
            },
            {
                "wavelength_cm": center2,
                "log_gf": -0.194,
                "E_lower_eV": 0.0,
                "species_id": 11,
                "gamma_rad": 6.14e7,
                "gamma_stark": 2.8e-5,
                "vdw_param1": 1.4e-7,
                "vdw_param2": 0.3
            }
        ]
    }
    
    return reference_data


def test_voigt_profile_comparison():
    """Test individual Voigt profile against reference"""
    print("Testing Voigt profile comparison...")
    
    # Load or create reference data
    try:
        with open("line_reference_data.json", "r") as f:
            ref_data = json.load(f)
    except FileNotFoundError:
        print("Reference data not found, using mock data")
        ref_data = create_mock_reference_data()
    
    # Extract reference parameters
    wavelengths = jnp.array(ref_data["wavelengths_cm"])
    lambda0 = ref_data["voigt_test_lambda0"]
    sigma = ref_data["voigt_test_sigma"]
    gamma = ref_data["voigt_test_gamma"]
    ref_profile = np.array(ref_data["voigt_test_profile"])
    
    # Calculate using jorg.lines
    jorg_profile = line_profile(lambda0, sigma, gamma, 1.0, wavelengths)
    
    # Compare results
    max_diff = np.max(np.abs(jorg_profile - ref_profile))
    rms_diff = np.sqrt(np.mean((jorg_profile - ref_profile)**2))
    
    print(f"Voigt profile comparison:")
    print(f"  Maximum difference: {max_diff:.2e}")
    print(f"  RMS difference: {rms_diff:.2e}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    wl_angstrom = wavelengths * 1e8
    plt.plot(wl_angstrom, ref_profile, 'k-', label='Korg.jl reference', linewidth=2)
    plt.plot(wl_angstrom, jorg_profile, 'r--', label='jorg.lines', linewidth=2)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Profile value')
    plt.title('Voigt Profile Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(wl_angstrom, (jorg_profile - ref_profile), 'b-', linewidth=1)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Difference (jorg - korg)')
    plt.title('Difference Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('voigt_profile_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return max_diff, rms_diff


def test_line_absorption_comparison():
    """Test full line absorption calculation against reference"""
    print("Testing line absorption comparison...")
    
    # Load or create reference data
    try:
        with open("line_reference_data.json", "r") as f:
            ref_data = json.load(f)
    except FileNotFoundError:
        print("Reference data not found, using mock data")
        ref_data = create_mock_reference_data()
    
    # Extract parameters
    wavelengths = jnp.array(ref_data["wavelengths_cm"])
    temperature = ref_data["temperature"]
    electron_density = ref_data["electron_density"]
    xi = ref_data["microturbulent_velocity"]
    ref_absorption = np.array(ref_data["line_absorption_total"])
    
    # Create linelist from reference data
    linelist = []
    for line_data in ref_data["test_lines"]:
        line = create_line_data(
            wavelength_cm=line_data["wavelength_cm"],
            log_gf=line_data["log_gf"],
            E_lower_eV=line_data["E_lower_eV"],
            species_id=line_data["species_id"],
            gamma_rad=line_data["gamma_rad"],
            gamma_stark=line_data["gamma_stark"],
            vdw_param1=line_data["vdw_param1"],
            vdw_param2=line_data["vdw_param2"]
        )
        linelist.append(line)
    
    # Set up number densities and partition functions
    number_densities = {11: 1e10}  # Na I density
    def mock_partition_fn(log_T):
        return 2.0
    partition_functions = {11: mock_partition_fn}
    
    # Calculate using jorg.lines
    jorg_absorption = line_absorption(
        wavelengths=wavelengths,
        linelist=linelist,
        temperature=temperature,
        electron_density=electron_density,
        number_densities=number_densities,
        partition_functions=partition_functions,
        microturbulent_velocity=xi
    )
    
    # Compare results
    max_diff = np.max(np.abs(jorg_absorption - ref_absorption))
    rms_diff = np.sqrt(np.mean((jorg_absorption - ref_absorption)**2))
    
    # Calculate relative differences where reference is significant
    mask = ref_absorption > 0.01 * np.max(ref_absorption)
    if np.any(mask):
        rel_diff = np.abs((jorg_absorption[mask] - ref_absorption[mask]) / ref_absorption[mask])
        mean_rel_diff = np.mean(rel_diff)
        max_rel_diff = np.max(rel_diff)
    else:
        mean_rel_diff = 0.0
        max_rel_diff = 0.0
    
    print(f"Line absorption comparison:")
    print(f"  Maximum absolute difference: {max_diff:.2e}")
    print(f"  RMS difference: {rms_diff:.2e}")
    print(f"  Maximum relative difference: {max_rel_diff:.1%}")
    print(f"  Mean relative difference: {mean_rel_diff:.1%}")
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    wl_angstrom = wavelengths * 1e8
    
    plt.subplot(3, 1, 1)
    plt.plot(wl_angstrom, ref_absorption, 'k-', label='Korg.jl reference', linewidth=2)
    plt.plot(wl_angstrom, jorg_absorption, 'r--', label='jorg.lines', linewidth=2)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Absorption coeff (cm⁻¹)')
    plt.title('Line Absorption Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(wl_angstrom, (jorg_absorption - ref_absorption), 'b-', linewidth=1)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Difference (jorg - korg)')
    plt.title('Absolute Difference')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    if np.any(mask):
        rel_diff_full = np.zeros_like(jorg_absorption)
        rel_diff_full[mask] = (jorg_absorption[mask] - ref_absorption[mask]) / ref_absorption[mask]
        plt.plot(wl_angstrom, rel_diff_full * 100, 'g-', linewidth=1)
        plt.ylabel('Relative difference (%)')
    else:
        plt.plot(wl_angstrom, np.zeros_like(jorg_absorption), 'g-', linewidth=1)
        plt.ylabel('Relative difference (%)')
    plt.xlabel('Wavelength (Å)')
    plt.title('Relative Difference')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('line_absorption_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return max_diff, rms_diff, max_rel_diff, mean_rel_diff


def test_individual_functions():
    """Test individual functions for basic correctness"""
    print("Testing individual functions...")
    
    # Test Voigt-Hjerting function at known points
    # At α=0, v=0: should give exp(0) = 1.0
    H_00 = voigt_hjerting(0.0, 0.0)
    print(f"H(0,0) = {H_00:.6f} (expected ≈ 1.0)")
    
    # At α=0, v=1: should give exp(-1) ≈ 0.368
    H_01 = voigt_hjerting(0.0, 1.0)
    expected_01 = np.exp(-1.0)
    print(f"H(0,1) = {H_01:.6f} (expected ≈ {expected_01:.6f})")
    
    # Test large v limit: H(α, v) → α/(π v²) for large v
    alpha = 0.1
    v_large = 50.0
    H_large = voigt_hjerting(alpha, v_large)
    expected_large = alpha / (pi * v_large**2)
    print(f"H({alpha},{v_large}) = {H_large:.2e} (expected ≈ {expected_large:.2e})")
    
    # Test differences
    diff_00 = abs(H_00 - 1.0)
    diff_01 = abs(H_01 - expected_01)
    diff_large = abs(H_large - expected_large)
    
    print(f"\nFunction test results:")
    print(f"  H(0,0) error: {diff_00:.2e}")
    print(f"  H(0,1) error: {diff_01:.2e}")
    print(f"  H(α,v_large) error: {diff_large:.2e}")
    
    return diff_00, diff_01, diff_large


def main():
    """Run all comparison tests"""
    print("=== Jorg Lines Comparison Tests ===\n")
    
    # Test individual functions
    func_errors = test_individual_functions()
    
    print("\n" + "="*50 + "\n")
    
    # Test Voigt profile comparison
    voigt_errors = test_voigt_profile_comparison()
    
    print("\n" + "="*50 + "\n")
    
    # Test full line absorption comparison
    absorption_errors = test_line_absorption_comparison()
    
    print("\n=== Summary ===")
    print(f"Function tests - Max error: {max(func_errors):.2e}")
    print(f"Voigt profile - Max diff: {voigt_errors[0]:.2e}, RMS: {voigt_errors[1]:.2e}")
    print(f"Line absorption - Max diff: {absorption_errors[0]:.2e}, RMS: {absorption_errors[1]:.2e}")
    print(f"Line absorption - Max rel: {absorption_errors[2]:.1%}, Mean rel: {absorption_errors[3]:.1%}")
    
    # Save summary
    summary = {
        "function_tests": {
            "max_error": float(max(func_errors))
        },
        "voigt_profile": {
            "max_diff": float(voigt_errors[0]),
            "rms_diff": float(voigt_errors[1])
        },
        "line_absorption": {
            "max_diff": float(absorption_errors[0]),
            "rms_diff": float(absorption_errors[1]),
            "max_rel_diff": float(absorption_errors[2]),
            "mean_rel_diff": float(absorption_errors[3])
        }
    }
    
    with open("lines_comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\nResults saved to lines_comparison_summary.json")
    print("Plots saved as voigt_profile_comparison.png and line_absorption_comparison.png")


if __name__ == "__main__":
    main()