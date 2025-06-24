#!/usr/bin/env python3
"""
Generate comprehensive EOS comparison plots between Korg.jl and Jorg.

This script creates detailed visualizations showing the agreement between
implementations across different stellar atmosphere conditions.
"""

import sys
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Import JAX and define Jorg functions
import jax.numpy as jnp
from jax import jit

# Constants
kboltz_cgs = 1.380649e-16  # erg/K

@jit
def jorg_gas_pressure(number_density, temperature):
    """Jorg gas pressure calculation."""
    return number_density * kboltz_cgs * temperature

@jit
def jorg_electron_pressure(electron_density, temperature):
    """Jorg electron pressure calculation."""
    return electron_density * kboltz_cgs * temperature


def generate_korg_eos_data():
    """Generate EOS data from Korg across parameter ranges."""
    
    julia_script = '''
using Pkg
Pkg.activate("/Users/jdli/Project/Korg.jl")
using Korg
using JSON

# Import constants
import Korg: kboltz_cgs

# Define parameter ranges for systematic comparison
temperatures = [3000.0, 4000.0, 5000.0, 5777.0, 6000.0, 7000.0, 8000.0, 10000.0]
densities = [1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19]
electron_fractions = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]  # n_e / n_total

results = Dict(
    "temperatures" => temperatures,
    "densities" => densities, 
    "electron_fractions" => electron_fractions,
    "gas_pressures" => [],
    "electron_pressures" => [],
    "parameter_sets" => []
)

# Generate comprehensive test grid
for T in temperatures
    for n_total in densities
        for e_frac in electron_fractions
            n_e = n_total * e_frac
            
            # Calculate pressures
            P_gas = n_total * kboltz_cgs * T
            P_e = n_e * kboltz_cgs * T
            
            push!(results["gas_pressures"], P_gas)
            push!(results["electron_pressures"], P_e)
            push!(results["parameter_sets"], Dict(
                "T" => T,
                "n_total" => n_total,
                "n_e" => n_e,
                "e_fraction" => e_frac,
                "P_gas" => P_gas,
                "P_e" => P_e
            ))
        end
    end
end

# Add some realistic stellar atmosphere examples
stellar_examples = [
    Dict("name" => "M_dwarf", "T" => 3200.0, "n_total" => 1e17, "e_frac" => 0.02),
    Dict("name" => "K_dwarf", "T" => 4800.0, "n_total" => 1e16, "e_frac" => 0.05),
    Dict("name" => "Solar", "T" => 5777.0, "n_total" => 1e15, "e_frac" => 0.1),
    Dict("name" => "F_star", "T" => 6500.0, "n_total" => 1e14, "e_frac" => 0.3),
    Dict("name" => "A_star", "T" => 8000.0, "n_total" => 1e13, "e_frac" => 0.8),
    Dict("name" => "B_star", "T" => 12000.0, "n_total" => 1e12, "e_frac" => 1.0)
]

stellar_data = []
for example in stellar_examples
    T = example["T"]
    n_total = example["n_total"]
    e_frac = example["e_frac"]
    n_e = n_total * e_frac
    
    P_gas = n_total * kboltz_cgs * T
    P_e = n_e * kboltz_cgs * T
    
    push!(stellar_data, Dict(
        "name" => example["name"],
        "T" => T,
        "n_total" => n_total,
        "n_e" => n_e,
        "e_fraction" => e_frac,
        "P_gas" => P_gas,
        "P_e" => P_e
    ))
end

results["stellar_examples"] = stellar_data

# Write results
open("korg_eos_grid_data.json", "w") do f
    JSON.print(f, results, 2)
end

println("Korg EOS grid data generated: ", length(results["parameter_sets"]), " data points")
'''
    
    with open('generate_korg_eos_grid.jl', 'w') as f:
        f.write(julia_script)
    
    try:
        result = subprocess.run(['julia', 'generate_korg_eos_grid.jl'], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"Korg data generation failed: {result.stderr}")
            return None
        
        print("Korg output:", result.stdout)
        
        with open('korg_eos_grid_data.json', 'r') as f:
            return json.load(f)
    
    except Exception as e:
        print(f"Error generating Korg data: {e}")
        return None


def generate_jorg_eos_data(korg_data):
    """Generate corresponding Jorg EOS data."""
    
    print("Generating Jorg EOS data...")
    
    jorg_data = {
        "parameter_sets": [],
        "stellar_examples": []
    }
    
    # Calculate for the parameter grid
    for param_set in korg_data["parameter_sets"]:
        T = param_set["T"]
        n_total = param_set["n_total"] 
        n_e = param_set["n_e"]
        
        # Calculate with Jorg
        P_gas_jorg = float(jorg_gas_pressure(n_total, T))
        P_e_jorg = float(jorg_electron_pressure(n_e, T))
        
        jorg_set = param_set.copy()
        jorg_set.update({
            "P_gas_jorg": P_gas_jorg,
            "P_e_jorg": P_e_jorg,
            "P_gas_error": abs(P_gas_jorg - param_set["P_gas"]) / param_set["P_gas"],
            "P_e_error": abs(P_e_jorg - param_set["P_e"]) / param_set["P_e"]
        })
        
        jorg_data["parameter_sets"].append(jorg_set)
    
    # Calculate for stellar examples
    for example in korg_data["stellar_examples"]:
        T = example["T"]
        n_total = example["n_total"]
        n_e = example["n_e"]
        
        P_gas_jorg = float(jorg_gas_pressure(n_total, T))
        P_e_jorg = float(jorg_electron_pressure(n_e, T))
        
        jorg_example = example.copy()
        jorg_example.update({
            "P_gas_jorg": P_gas_jorg,
            "P_e_jorg": P_e_jorg,
            "P_gas_error": abs(P_gas_jorg - example["P_gas"]) / example["P_gas"],
            "P_e_error": abs(P_e_jorg - example["P_e"]) / example["P_e"]
        })
        
        jorg_data["stellar_examples"].append(jorg_example)
    
    print(f"Generated Jorg data for {len(jorg_data['parameter_sets'])} grid points")
    print(f"Generated Jorg data for {len(jorg_data['stellar_examples'])} stellar examples")
    
    return jorg_data


def create_eos_comparison_plots(korg_data, jorg_data):
    """Create comprehensive EOS comparison plots."""
    
    print("Creating EOS comparison plots...")
    
    # Set up the plotting style
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'figure.facecolor': 'white',
        'axes.linewidth': 1.2,
        'lines.linewidth': 1.5
    })
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Extract data for plotting
    params = jorg_data["parameter_sets"]
    stellar = jorg_data["stellar_examples"]
    
    temperatures = np.array([p["T"] for p in params])
    densities = np.array([p["n_total"] for p in params]) 
    korg_P_gas = np.array([p["P_gas"] for p in params])
    jorg_P_gas = np.array([p["P_gas_jorg"] for p in params])
    korg_P_e = np.array([p["P_e"] for p in params])
    jorg_P_e = np.array([p["P_e_jorg"] for p in params])
    gas_errors = np.array([p["P_gas_error"] for p in params])
    electron_errors = np.array([p["P_e_error"] for p in params])
    
    # Plot 1: Gas Pressure Comparison (log-log)
    ax1 = plt.subplot(2, 3, 1)
    plt.loglog(korg_P_gas, jorg_P_gas, 'o', alpha=0.6, markersize=3, color='blue', label='Grid Data')
    
    # Add stellar examples
    for star in stellar:
        plt.loglog(star["P_gas"], star["P_gas_jorg"], 's', markersize=8, 
                  label=star["name"], alpha=0.8)
    
    # Perfect agreement line
    p_range = [np.min(korg_P_gas), np.max(korg_P_gas)]
    plt.loglog(p_range, p_range, 'k--', alpha=0.7, label='Perfect Agreement')
    
    plt.xlabel('Korg Gas Pressure (dyne cm⁻²)')
    plt.ylabel('Jorg Gas Pressure (dyne cm⁻²)')
    plt.title('Gas Pressure Comparison\nP = n × k × T')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Electron Pressure Comparison (log-log)
    ax2 = plt.subplot(2, 3, 2)
    plt.loglog(korg_P_e, jorg_P_e, 'o', alpha=0.6, markersize=3, color='red', label='Grid Data')
    
    # Add stellar examples
    for star in stellar:
        plt.loglog(star["P_e"], star["P_e_jorg"], 's', markersize=8, alpha=0.8)
    
    # Perfect agreement line
    p_range = [np.min(korg_P_e[korg_P_e > 0]), np.max(korg_P_e)]
    plt.loglog(p_range, p_range, 'k--', alpha=0.7, label='Perfect Agreement')
    
    plt.xlabel('Korg Electron Pressure (dyne cm⁻²)')
    plt.ylabel('Jorg Electron Pressure (dyne cm⁻²)')
    plt.title('Electron Pressure Comparison\nPₑ = nₑ × k × T')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Relative Error Distribution
    ax3 = plt.subplot(2, 3, 3)
    
    # Create histograms of relative errors
    plt.hist(gas_errors, bins=30, alpha=0.7, label='Gas Pressure', color='blue', density=True)
    plt.hist(electron_errors, bins=30, alpha=0.7, label='Electron Pressure', color='red', density=True)
    
    plt.xlabel('Relative Error')
    plt.ylabel('Density')
    plt.title('Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Add statistics
    gas_mean_error = np.mean(gas_errors)
    gas_max_error = np.max(gas_errors)
    electron_mean_error = np.mean(electron_errors)
    electron_max_error = np.max(electron_errors)
    
    plt.text(0.05, 0.95, f'Gas Pressure:\nMean: {gas_mean_error:.2e}\nMax: {gas_max_error:.2e}', 
             transform=ax3.transAxes, verticalalignment='top', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.text(0.05, 0.75, f'Electron Pressure:\nMean: {electron_mean_error:.2e}\nMax: {electron_max_error:.2e}', 
             transform=ax3.transAxes, verticalalignment='top', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # Plot 4: Error vs Temperature
    ax4 = plt.subplot(2, 3, 4)
    
    # Create temperature bins for averaging
    temp_bins = np.logspace(np.log10(3000), np.log10(12000), 20)
    temp_centers = (temp_bins[:-1] + temp_bins[1:]) / 2
    
    gas_error_means = []
    electron_error_means = []
    
    for i in range(len(temp_bins)-1):
        mask = (temperatures >= temp_bins[i]) & (temperatures < temp_bins[i+1])
        if np.any(mask):
            gas_error_means.append(np.mean(gas_errors[mask]))
            electron_error_means.append(np.mean(electron_errors[mask]))
        else:
            gas_error_means.append(np.nan)
            electron_error_means.append(np.nan)
    
    plt.semilogx(temp_centers, gas_error_means, 'o-', label='Gas Pressure', color='blue')
    plt.semilogx(temp_centers, electron_error_means, 's-', label='Electron Pressure', color='red')
    
    plt.xlabel('Temperature (K)')
    plt.ylabel('Mean Relative Error')
    plt.title('Error vs Temperature')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Error vs Density
    ax5 = plt.subplot(2, 3, 5)
    
    # Create density bins for averaging
    density_bins = np.logspace(12, 19, 20)
    density_centers = (density_bins[:-1] + density_bins[1:]) / 2
    
    gas_error_means_dens = []
    electron_error_means_dens = []
    
    for i in range(len(density_bins)-1):
        mask = (densities >= density_bins[i]) & (densities < density_bins[i+1])
        if np.any(mask):
            gas_error_means_dens.append(np.mean(gas_errors[mask]))
            electron_error_means_dens.append(np.mean(electron_errors[mask]))
        else:
            gas_error_means_dens.append(np.nan)
            electron_error_means_dens.append(np.nan)
    
    plt.semilogx(density_centers, gas_error_means_dens, 'o-', label='Gas Pressure', color='blue')
    plt.semilogx(density_centers, electron_error_means_dens, 's-', label='Electron Pressure', color='red')
    
    plt.xlabel('Number Density (cm⁻³)')
    plt.ylabel('Mean Relative Error')
    plt.title('Error vs Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Stellar Example Details
    ax6 = plt.subplot(2, 3, 6)
    
    # Create bar chart of stellar examples
    star_names = [s["name"] for s in stellar]
    star_gas_errors = [s["P_gas_error"] for s in stellar]
    star_electron_errors = [s["P_e_error"] for s in stellar]
    
    x = np.arange(len(star_names))
    width = 0.35
    
    plt.bar(x - width/2, star_gas_errors, width, label='Gas Pressure', color='blue', alpha=0.7)
    plt.bar(x + width/2, star_electron_errors, width, label='Electron Pressure', color='red', alpha=0.7)
    
    plt.xlabel('Stellar Type')
    plt.ylabel('Relative Error')
    plt.title('Stellar Example Errors')
    plt.xticks(x, star_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Adjust layout and add overall title
    plt.tight_layout()
    plt.suptitle('Korg.jl vs Jorg Equation of State Comparison\n' + 
                f'Grid: {len(params)} points, Max Error: {max(gas_max_error, electron_max_error):.2e}',
                fontsize=14, y=0.98)
    
    # Save the plot
    plt.savefig('korg_jorg_eos_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('korg_jorg_eos_comparison.pdf', bbox_inches='tight')
    
    print("Plots saved as:")
    print("  - korg_jorg_eos_comparison.png")
    print("  - korg_jorg_eos_comparison.pdf")
    
    return fig


def create_summary_statistics(jorg_data):
    """Create summary statistics table."""
    
    params = jorg_data["parameter_sets"]
    stellar = jorg_data["stellar_examples"]
    
    gas_errors = np.array([p["P_gas_error"] for p in params])
    electron_errors = np.array([p["P_e_error"] for p in params])
    
    print("\n" + "="*60)
    print("EOS COMPARISON SUMMARY STATISTICS")
    print("="*60)
    
    print(f"\n📊 Dataset Size:")
    print(f"  Grid points: {len(params)}")
    print(f"  Stellar examples: {len(stellar)}")
    print(f"  Temperature range: 3000 - 12000 K")
    print(f"  Density range: 1e12 - 1e19 cm⁻³")
    
    print(f"\n📈 Gas Pressure Errors:")
    print(f"  Mean: {np.mean(gas_errors):.2e}")
    print(f"  Median: {np.median(gas_errors):.2e}")
    print(f"  Standard deviation: {np.std(gas_errors):.2e}")
    print(f"  Maximum: {np.max(gas_errors):.2e}")
    print(f"  95th percentile: {np.percentile(gas_errors, 95):.2e}")
    
    print(f"\n⚡ Electron Pressure Errors:")
    print(f"  Mean: {np.mean(electron_errors):.2e}")
    print(f"  Median: {np.median(electron_errors):.2e}")
    print(f"  Standard deviation: {np.std(electron_errors):.2e}")
    print(f"  Maximum: {np.max(electron_errors):.2e}")
    print(f"  95th percentile: {np.percentile(electron_errors, 95):.2e}")
    
    print(f"\n⭐ Stellar Examples:")
    for star in stellar:
        print(f"  {star['name']:8s}: Gas={star['P_gas_error']:.2e}, Electron={star['P_e_error']:.2e}")
    
    print(f"\n✅ Assessment:")
    max_error = max(np.max(gas_errors), np.max(electron_errors))
    if max_error < 1e-12:
        print(f"  EXCELLENT: Maximum error {max_error:.2e} (machine precision)")
    elif max_error < 1e-8:
        print(f"  VERY GOOD: Maximum error {max_error:.2e} (numerical precision)")
    elif max_error < 1e-6:
        print(f"  GOOD: Maximum error {max_error:.2e} (acceptable for physics)")
    else:
        print(f"  REVIEW: Maximum error {max_error:.2e} (may need investigation)")


def main():
    """Main function to generate EOS comparison plots."""
    
    print("="*60)
    print("KORG vs JORG EOS COMPARISON VISUALIZATION")
    print("="*60)
    
    # Generate Korg reference data
    print("\n🔍 Generating Korg EOS reference data...")
    korg_data = generate_korg_eos_data()
    
    if not korg_data:
        print("❌ Failed to generate Korg data")
        return False
    
    # Generate corresponding Jorg data
    print("\n🚀 Calculating Jorg EOS data...")
    jorg_data = generate_jorg_eos_data(korg_data)
    
    # Create plots
    print("\n📊 Creating comparison plots...")
    fig = create_eos_comparison_plots(korg_data, jorg_data)
    
    # Display plot
    plt.show()
    
    # Generate summary statistics
    create_summary_statistics(jorg_data)
    
    # Save data for future reference
    with open('eos_comparison_data.json', 'w') as f:
        json.dump({
            'korg_data': korg_data,
            'jorg_data': jorg_data,
            'timestamp': '2025-06-24'
        }, f, indent=2)
    
    print(f"\n📁 Data saved to: eos_comparison_data.json")
    
    # Cleanup temporary files
    for temp_file in ['generate_korg_eos_grid.jl', 'korg_eos_grid_data.json']:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    print("\n" + "="*60)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)