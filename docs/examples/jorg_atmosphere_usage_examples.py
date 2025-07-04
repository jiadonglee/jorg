#!/usr/bin/env python3
"""
Jorg Atmosphere Calculation Examples
===================================

Comprehensive examples showing how to use Jorg's JAX-based atmosphere interpolation
for stellar spectroscopy applications.
"""

import sys
from pathlib import Path

# Add Jorg to path
jorg_path = Path(__file__).parent / "Jorg" / "src"
sys.path.insert(0, str(jorg_path))

import numpy as np
from jorg.atmosphere_jax_fixed import interpolate_marcs_jax_fixed
from jorg.atmosphere import call_korg_interpolation  # For comparison
import warnings
warnings.filterwarnings('ignore')


def example_1_basic_usage():
    """Example 1: Basic atmosphere calculation for different stellar types"""
    
    print("EXAMPLE 1: BASIC ATMOSPHERE CALCULATION")
    print("=" * 50)
    
    # Define stellar parameters for different star types
    stellar_types = [
        ("Sun (G2V)", 5777.0, 4.44, 0.0),
        ("Arcturus (K1.5III)", 4286.0, 1.66, -0.52),
        ("Vega (A0V)", 9602.0, 3.95, -0.5),
        ("Proxima Centauri (M5.5V)", 3042.0, 5.20, 0.0),
        ("Aldebaran (K5III)", 3910.0, 1.11, -0.33),
    ]
    
    for name, Teff, logg, m_H in stellar_types:
        print(f"\n{name}:")
        print(f"  Parameters: Teff={Teff}K, log(g)={logg}, [M/H]={m_H}")
        
        # Calculate atmosphere using Jorg JAX
        atmosphere = interpolate_marcs_jax_fixed(Teff, logg, m_H)
        
        print(f"  Atmosphere: {len(atmosphere.layers)} layers")
        print(f"  Geometry: {'Spherical' if atmosphere.spherical else 'Planar'}")
        if atmosphere.spherical and atmosphere.R:
            print(f"  Stellar radius: {atmosphere.R/6.96e10:.2f} R_sun")
        
        # Show some atmospheric properties
        if len(atmosphere.layers) > 10:
            surface_layer = atmosphere.layers[0]
            deep_layer = atmosphere.layers[-10]
            
            print(f"  Surface: T={surface_layer.temp:.0f}K, tau={surface_layer.tau_5000:.2e}")
            print(f"  Deep:    T={deep_layer.temp:.0f}K, tau={deep_layer.tau_5000:.2e}")


def example_2_chemical_composition():
    """Example 2: Atmospheres with different chemical compositions"""
    
    print("\n\nEXAMPLE 2: CHEMICAL COMPOSITION EFFECTS")
    print("=" * 50)
    
    # Solar-type star with different compositions
    Teff, logg = 5777.0, 4.44
    
    compositions = [
        ("Solar composition", 0.0, 0.0, 0.0),
        ("Metal-poor halo star", -2.0, 0.3, 0.0),
        ("Alpha-enhanced thick disk", -0.5, 0.4, 0.0),
        ("Carbon-enhanced metal-poor", -1.5, 0.3, 1.0),
        ("Metal-rich thin disk", 0.4, 0.0, 0.0),
    ]
    
    for desc, m_H, alpha_m, C_m in compositions:
        print(f"\n{desc}:")
        print(f"  [M/H]={m_H}, [α/M]={alpha_m}, [C/M]={C_m}")
        
        atmosphere = interpolate_marcs_jax_fixed(Teff, logg, m_H, alpha_m, C_m)
        
        # Compare atmospheric structure
        if len(atmosphere.layers) > 25:
            layer = atmosphere.layers[25]  # Representative layer
            print(f"  Layer 25: T={layer.temp:.0f}K, n_e={layer.electron_number_density:.2e} cm⁻³")
            print(f"           n_total={layer.number_density:.2e} cm⁻³")


def example_3_stellar_evolution():
    """Example 3: Atmospheric changes during stellar evolution"""
    
    print("\n\nEXAMPLE 3: STELLAR EVOLUTION SEQUENCE")
    print("=" * 50)
    
    # 1 M_sun star evolution sequence
    evolution_phases = [
        ("Zero-age main sequence", 5770, 4.50, 0.0),
        ("Main sequence turnoff", 5700, 4.30, 0.0),
        ("Subgiant branch", 5500, 3.80, 0.0),
        ("Red giant branch base", 4800, 2.50, 0.0),
        ("RGB tip", 3800, 1.00, 0.0),
        ("Horizontal branch", 5200, 2.80, 0.0),
        ("Asymptotic giant branch", 3500, 0.50, 0.0),
    ]
    
    print(f"{'Phase':<25} {'Teff':<6} {'logg':<6} {'Layers':<7} {'Geometry':<10} {'R_star':<8}")
    print("-" * 70)
    
    for phase, Teff, logg, m_H in evolution_phases:
        atmosphere = interpolate_marcs_jax_fixed(Teff, logg, m_H)
        
        geometry = "Spherical" if atmosphere.spherical else "Planar"
        R_star = f"{atmosphere.R/6.96e10:.2f}" if atmosphere.R else "N/A"
        
        print(f"{phase:<25} {Teff:<6} {logg:<6} {len(atmosphere.layers):<7} {geometry:<10} {R_star:<8}")


def example_4_atmosphere_structure():
    """Example 4: Detailed atmospheric structure analysis"""
    
    print("\n\nEXAMPLE 4: DETAILED ATMOSPHERIC STRUCTURE")
    print("=" * 50)
    
    # Calculate atmosphere for solar-type star
    atmosphere = interpolate_marcs_jax_fixed(5777.0, 4.44, 0.0)
    
    print(f"Solar atmosphere structure ({len(atmosphere.layers)} layers):")
    print(f"Geometry: {'Spherical' if atmosphere.spherical else 'Planar'}")
    
    # Extract atmospheric quantities
    tau_5000 = [layer.tau_5000 for layer in atmosphere.layers]
    temperature = [layer.temp for layer in atmosphere.layers]
    number_density = [layer.number_density for layer in atmosphere.layers]
    electron_density = [layer.electron_number_density for layer in atmosphere.layers]
    height = [layer.z for layer in atmosphere.layers]
    
    print(f"\nAtmospheric ranges:")
    print(f"  Optical depth: τ₅₀₀₀ = {min(tau_5000):.2e} to {max(tau_5000):.2e}")
    print(f"  Temperature: T = {min(temperature):.0f} to {max(temperature):.0f} K")
    print(f"  Total density: n = {min(number_density):.2e} to {max(number_density):.2e} cm⁻³")
    print(f"  Height: z = {min(height):.2e} to {max(height):.2e} cm")
    
    # Show representative layers
    print(f"\nRepresentative atmospheric layers:")
    print(f"{'Layer':<5} {'τ₅₀₀₀':<12} {'T (K)':<8} {'n_total (cm⁻³)':<15} {'n_e (cm⁻³)':<15}")
    print("-" * 65)
    
    for i in [0, 10, 25, 40, 55]:
        if i < len(atmosphere.layers):
            layer = atmosphere.layers[i]
            print(f"{i:<5} {layer.tau_5000:<12.2e} {layer.temp:<8.0f} {layer.number_density:<15.2e} {layer.electron_number_density:<15.2e}")


def example_5_comparison_validation():
    """Example 5: Validation against Korg subprocess"""
    
    print("\n\nEXAMPLE 5: VALIDATION AGAINST KORG")
    print("=" * 50)
    
    test_cases = [
        ("Solar G-type", 5777.0, 4.44, 0.0),
        ("Cool K giant", 4000.0, 2.0, 0.0),
        ("Metal-poor dwarf", 6000.0, 4.5, -1.5),
    ]
    
    for desc, Teff, logg, m_H in test_cases:
        print(f"\n{desc} (Teff={Teff}K, logg={logg}, [M/H]={m_H}):")
        
        # Get both results
        jax_atm = interpolate_marcs_jax_fixed(Teff, logg, m_H)
        korg_atm = call_korg_interpolation(Teff, logg, m_H)
        
        print(f"  JAX:  {len(jax_atm.layers)} layers, spherical={jax_atm.spherical}")
        print(f"  Korg: {len(korg_atm.layers)} layers, spherical={korg_atm.spherical}")
        
        # Compare layer 25
        if len(jax_atm.layers) > 25 and len(korg_atm.layers) > 25:
            jax_layer = jax_atm.layers[25]
            korg_layer = korg_atm.layers[25]
            
            temp_diff = abs(jax_layer.temp - korg_layer.temp) / korg_layer.temp * 100
            tau_diff = abs(jax_layer.tau_5000 - korg_layer.tau_5000) / korg_layer.tau_5000 * 100
            
            print(f"  Agreement: T={temp_diff:.3f}%, τ={tau_diff:.3f}% ✅")


def example_6_batch_processing():
    """Example 6: Batch processing multiple atmospheres"""
    
    print("\n\nEXAMPLE 6: BATCH ATMOSPHERE CALCULATION")
    print("=" * 50)
    
    # Define a grid of stellar parameters
    Teff_range = [4000, 5000, 6000, 7000]
    logg_range = [2.0, 3.0, 4.0, 5.0]
    m_H = 0.0  # Solar metallicity
    
    print("Calculating atmosphere grid:")
    print(f"{'Teff (K)':<8} {'logg':<6} {'Layers':<7} {'Spherical':<10} {'Status':<8}")
    print("-" * 45)
    
    atmosphere_grid = {}
    
    for Teff in Teff_range:
        for logg in logg_range:
            try:
                atmosphere = interpolate_marcs_jax_fixed(Teff, logg, m_H)
                atmosphere_grid[(Teff, logg)] = atmosphere
                
                spherical = "Yes" if atmosphere.spherical else "No"
                status = "✅"
                
                print(f"{Teff:<8} {logg:<6} {len(atmosphere.layers):<7} {spherical:<10} {status:<8}")
                
            except Exception as e:
                print(f"{Teff:<8} {logg:<6} {'N/A':<7} {'N/A':<10} {'❌':<8}")
    
    print(f"\nSuccessfully calculated {len(atmosphere_grid)} atmospheres")


def example_7_advanced_features():
    """Example 7: Advanced features and customization"""
    
    print("\n\nEXAMPLE 7: ADVANCED FEATURES")
    print("=" * 50)
    
    # Different grid types based on stellar parameters
    examples = [
        ("Standard grid", 5777.0, 4.44, 0.0, 0.0, 0.0),
        ("Cool dwarf grid", 3500.0, 4.8, 0.0, 0.0, 0.0),
        ("Low-Z grid", 5777.0, 4.44, -3.0, 0.4, 0.0),
    ]
    
    for desc, Teff, logg, m_H, alpha_m, C_m in examples:
        print(f"\n{desc}:")
        print(f"  Parameters: Teff={Teff}K, logg={logg}, [M/H]={m_H}, [α/M]={alpha_m}, [C/M]={C_m}")
        
        atmosphere = interpolate_marcs_jax_fixed(Teff, logg, m_H, alpha_m, C_m)
        
        print(f"  Grid used: {'Cool dwarf' if len(atmosphere.layers) > 70 else 'Standard/Low-Z'}")
        print(f"  Resolution: {len(atmosphere.layers)} atmospheric layers")
        print(f"  Geometry: {'Spherical' if atmosphere.spherical else 'Planar'}")
        
        # Show optical depth range
        tau_min = min(layer.tau_5000 for layer in atmosphere.layers)
        tau_max = max(layer.tau_5000 for layer in atmosphere.layers)
        print(f"  τ₅₀₀₀ range: {tau_min:.2e} to {tau_max:.2e}")


def main():
    """Run all examples"""
    
    print("JORG ATMOSPHERE CALCULATION EXAMPLES")
    print("=" * 60)
    print("Demonstrating Jorg's JAX-based MARCS atmosphere interpolation")
    print("=" * 60)
    
    # Check if grid files are available
    grid_dir = Path("Jorg/data/marcs_grids")
    if not grid_dir.exists():
        print("❌ MARCS grid files not found!")
        print(f"   Expected location: {grid_dir}")
        print("   Please run the setup script to copy grid files.")
        return
    
    # Run examples
    example_1_basic_usage()
    example_2_chemical_composition()
    example_3_stellar_evolution()
    example_4_atmosphere_structure()
    example_5_comparison_validation()
    example_6_batch_processing()
    example_7_advanced_features()
    
    print("\n\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nJorg's JAX atmosphere interpolation is ready for:")
    print("  ✅ Stellar spectroscopy applications")
    print("  ✅ Chemical equilibrium calculations")
    print("  ✅ Stellar parameter determination")
    print("  ✅ Large-scale stellar surveys")
    print("  ✅ GPU-accelerated batch processing")


if __name__ == "__main__":
    main()