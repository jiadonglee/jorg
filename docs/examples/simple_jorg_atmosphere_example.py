#!/usr/bin/env python3
"""
Simple Jorg Atmosphere Usage Examples
====================================

Clear, focused examples showing how to use Jorg for atmosphere calculations.
"""

import sys
from pathlib import Path

# Add Jorg to path
jorg_path = Path(__file__).parent / "Jorg" / "src"
sys.path.insert(0, str(jorg_path))

from jorg.atmosphere_jax_fixed import interpolate_marcs_jax_fixed
import warnings
warnings.filterwarnings('ignore')


def basic_example():
    """Basic atmosphere calculation"""
    print("=== BASIC JORG ATMOSPHERE CALCULATION ===")
    
    # Calculate atmosphere for the Sun
    Teff = 5777.0  # Effective temperature in K
    logg = 4.44    # Surface gravity log(g) in cgs
    m_H = 0.0      # Metallicity [M/H]
    
    atmosphere = interpolate_marcs_jax_fixed(Teff, logg, m_H)
    
    print(f"Solar atmosphere:")
    print(f"  Temperature: {Teff}K")
    print(f"  Surface gravity: log(g) = {logg}")
    print(f"  Metallicity: [M/H] = {m_H}")
    print(f"  Result: {len(atmosphere.layers)} atmospheric layers")
    print(f"  Geometry: {'Spherical' if atmosphere.spherical else 'Planar'}")
    
    # Show some layer properties
    surface = atmosphere.layers[0]
    photosphere = atmosphere.layers[25]  # Representative photospheric layer
    
    print(f"\n  Surface layer:")
    print(f"    Temperature: {surface.temp:.0f} K")
    print(f"    Optical depth: τ₅₀₀₀ = {surface.tau_5000:.2e}")
    print(f"    Gas density: {surface.number_density:.2e} cm⁻³")
    
    print(f"\n  Photosphere (layer 25):")
    print(f"    Temperature: {photosphere.temp:.0f} K")
    print(f"    Optical depth: τ₅₀₀₀ = {photosphere.tau_5000:.2e}")
    print(f"    Gas density: {photosphere.number_density:.2e} cm⁻³")


def stellar_types_example():
    """Different stellar types"""
    print("\n\n=== DIFFERENT STELLAR TYPES ===")
    
    stars = [
        ("Sun (G2V)", 5777, 4.44, 0.0),
        ("Arcturus (K1III)", 4286, 1.66, -0.52),
        ("Proxima Cen (M5V)", 3042, 5.20, 0.0),
        ("Sirius (A1V)", 9940, 4.33, 0.5),
    ]
    
    for name, Teff, logg, m_H in stars:
        atmosphere = interpolate_marcs_jax_fixed(Teff, logg, m_H)
        geometry = "Spherical" if atmosphere.spherical else "Planar"
        
        print(f"\n{name}:")
        print(f"  Teff={Teff}K, logg={logg}, [M/H]={m_H}")
        print(f"  → {len(atmosphere.layers)} layers, {geometry} geometry")
        
        if atmosphere.spherical and atmosphere.R:
            R_solar = atmosphere.R / 6.96e10  # Convert to solar radii
            print(f"  → Stellar radius: {R_solar:.1f} R☉")


def composition_example():
    """Chemical composition effects"""
    print("\n\n=== CHEMICAL COMPOSITION EFFECTS ===")
    
    # Solar-type star with different compositions
    Teff, logg = 5777, 4.44
    
    compositions = [
        ("Solar", 0.0, 0.0, 0.0),
        ("Metal-poor", -1.0, 0.3, 0.0),
        ("α-enhanced", -0.5, 0.4, 0.0),
        ("Metal-rich", 0.3, 0.0, 0.0),
    ]
    
    print(f"Solar-type star (Teff={Teff}K, logg={logg}) with different compositions:")
    
    for desc, m_H, alpha_m, C_m in compositions:
        atmosphere = interpolate_marcs_jax_fixed(Teff, logg, m_H, alpha_m, C_m)
        
        # Get photospheric properties
        layer = atmosphere.layers[25]
        
        print(f"\n  {desc}: [M/H]={m_H}, [α/M]={alpha_m}, [C/M]={C_m}")
        print(f"    Photosphere: T={layer.temp:.0f}K, n_e={layer.electron_number_density:.2e} cm⁻³")


def access_atmosphere_data():
    """How to access atmospheric data"""
    print("\n\n=== ACCESSING ATMOSPHERIC DATA ===")
    
    atmosphere = interpolate_marcs_jax_fixed(5777, 4.44, 0.0)
    
    print(f"Atmosphere object has {len(atmosphere.layers)} layers")
    print(f"Each layer contains:")
    
    # Show properties of one layer
    layer = atmosphere.layers[25]
    print(f"  - tau_5000: {layer.tau_5000:.3e} (optical depth at 5000Å)")
    print(f"  - temp: {layer.temp:.1f} K (temperature)")
    print(f"  - number_density: {layer.number_density:.2e} cm⁻³ (total particle density)")
    print(f"  - electron_number_density: {layer.electron_number_density:.2e} cm⁻³")
    print(f"  - z: {layer.z:.2e} cm (height)")
    
    print(f"\nExtracting arrays for analysis:")
    
    # Extract arrays for plotting/analysis
    temperatures = [layer.temp for layer in atmosphere.layers]
    optical_depths = [layer.tau_5000 for layer in atmosphere.layers]
    densities = [layer.number_density for layer in atmosphere.layers]
    
    print(f"  Temperature range: {min(temperatures):.0f} - {max(temperatures):.0f} K")
    print(f"  Optical depth range: {min(optical_depths):.2e} - {max(optical_depths):.2e}")
    print(f"  Density range: {min(densities):.2e} - {max(densities):.2e} cm⁻³")


def practical_usage():
    """Practical usage patterns"""
    print("\n\n=== PRACTICAL USAGE PATTERNS ===")
    
    print("1. Single atmosphere calculation:")
    print("   from jorg.atmosphere_jax_fixed import interpolate_marcs_jax_fixed")
    print("   atmosphere = interpolate_marcs_jax_fixed(5777, 4.44, 0.0)")
    
    print("\n2. Batch processing multiple stars:")
    stellar_params = [
        (5777, 4.44, 0.0),
        (4500, 2.5, -0.5),
        (6000, 4.0, 0.2),
    ]
    
    atmospheres = []
    for Teff, logg, m_H in stellar_params:
        atm = interpolate_marcs_jax_fixed(Teff, logg, m_H)
        atmospheres.append(atm)
    
    print(f"   → Calculated {len(atmospheres)} atmospheres")
    
    print("\n3. Error handling:")
    try:
        # This will work fine
        atmosphere = interpolate_marcs_jax_fixed(5777, 4.44, 0.0)
        print("   ✅ Valid parameters - success")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n4. Access different grids automatically:")
    examples = [
        ("Standard SDSS", 5777, 4.44, 0.0),
        ("Cool dwarf", 3500, 4.8, 0.0),  # Uses high-res cool dwarf grid
        ("Giant", 4000, 2.0, 0.0),       # Uses spherical geometry
    ]
    
    for desc, Teff, logg, m_H in examples:
        atm = interpolate_marcs_jax_fixed(Teff, logg, m_H)
        grid_type = "Cool dwarf" if len(atm.layers) > 70 else "Standard"
        print(f"   {desc}: {grid_type} grid, {len(atm.layers)} layers")


def main():
    print("JORG ATMOSPHERE CALCULATION EXAMPLES")
    print("=" * 50)
    print("Using JAX-based MARCS atmosphere interpolation")
    print()
    
    basic_example()
    stellar_types_example()
    composition_example()
    access_atmosphere_data()
    practical_usage()
    
    print("\n\n" + "=" * 50)
    print("✅ ALL EXAMPLES COMPLETED")
    print("Jorg is ready for stellar atmosphere calculations!")
    print("=" * 50)


if __name__ == "__main__":
    main()