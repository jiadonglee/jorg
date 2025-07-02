#!/usr/bin/env python3
"""
Comprehensive example of using Jorg linelist reading capabilities

This example demonstrates how to read different linelist formats and use them
for stellar spectral synthesis, matching and extending Korg.jl functionality.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add jorg to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    
    from jorg.lines import (
        read_linelist, save_linelist, LineList,
        line_absorption, create_line_data,
        parse_species, species_id_to_name,
        air_to_vacuum, angstrom_to_cm
    )
    JAX_AVAILABLE = True
    print("✅ JAX and Jorg.lines successfully imported")
except ImportError as e:
    print(f"❌ JAX import error: {e}")
    JAX_AVAILABLE = False


def create_sample_vald_linelist():
    """Create a realistic VALD-format linelist for demonstration"""
    
    vald_content = """# VALD3 Extract All Request 
# Wavelength range: 5800.0000 - 6000.0000
# Element(s): Na, Ca, Fe, Mg, H
# Extraction timestamp: 2024-01-15
# 
'5889.9510',   0.108,   0.000,'Na 1', 6.14e7, 2.80e-5, 1.40e-7, 0.30
'5895.9242',  -0.194,   0.000,'Na 1', 6.14e7, 2.80e-5, 1.40e-7, 0.30
'5895.9243',  -0.193,   0.000,'Na 1', 6.14e7, 2.80e-5, 1.40e-7, 0.30
'5889.9511',   0.107,   0.000,'Na 1', 6.14e7, 2.80e-5, 1.40e-7, 0.30
'5895.9244',  -0.192,   0.000,'Na 1', 6.14e7, 2.80e-5, 1.40e-7, 0.30
'5889.9512',   0.106,   0.000,'Na 1', 6.14e7, 2.80e-5, 1.40e-7, 0.30
'5801.3310',  -5.234,   2.710,'Ca 1', 3.00e6, 1.20e-6, 8.50e-8, 0.25
'5808.1200',  -6.532,   4.458,'Ca 1', 3.00e6, 1.20e-6, 8.50e-8, 0.25
'5857.4510',  -1.989,   2.933,'Ca 1', 3.00e6, 1.20e-6, 8.50e-8, 0.25
'5867.5620',  -1.570,   2.933,'Ca 1', 3.00e6, 1.20e-6, 8.50e-8, 0.25
'5916.2470',  -2.298,   2.524,'Ca 1', 3.00e6, 1.20e-6, 8.50e-8, 0.25
'5956.6940',  -0.401,   2.933,'Ca 1', 3.00e6, 1.20e-6, 8.50e-8, 0.25
'5983.6880',  -0.155,   0.961,'Ca 1', 3.00e6, 1.20e-6, 8.50e-8, 0.25
'5857.4760',  -2.158,   4.283,'Fe 1', 2.50e6, 8.40e-7, 1.30e-7, 0.22
'5862.3530',  -0.058,   4.549,'Fe 1', 2.50e6, 8.40e-7, 1.30e-7, 0.22
'5883.8180',  -1.313,   3.396,'Fe 1', 2.50e6, 8.40e-7, 1.30e-7, 0.22
'5905.6720',  -0.729,   4.652,'Fe 1', 2.50e6, 8.40e-7, 1.30e-7, 0.22
'5916.2470',  -2.994,   2.453,'Fe 1', 2.50e6, 8.40e-7, 1.30e-7, 0.22
'5927.7890',  -1.090,   4.652,'Fe 1', 2.50e6, 8.40e-7, 1.30e-7, 0.22
'5956.6940',  -4.605,   0.859,'Fe 1', 2.50e6, 8.40e-7, 1.30e-7, 0.22
'5976.7780',  -1.163,   3.943,'Fe 1', 2.50e6, 8.40e-7, 1.30e-7, 0.22
'5987.0670',  -0.553,   4.795,'Fe 1', 2.50e6, 8.40e-7, 1.30e-7, 0.22
"""
    
    return vald_content


def demonstrate_linelist_reading():
    """Demonstrate reading and parsing linelists"""
    
    print("\n📖 Linelist Reading Demonstration")
    print("=" * 50)
    
    # Create sample VALD linelist
    with open("sample_lines.vald", "w") as f:
        f.write(create_sample_vald_linelist())
    
    print("Created sample VALD linelist: sample_lines.vald")
    
    # Read the linelist
    print("\n--- Reading VALD Linelist ---")
    linelist = read_linelist("sample_lines.vald", format="vald")
    
    print(f"Total lines read: {len(linelist)}")
    if len(linelist) > 0:
        print(f"Wavelength range: {linelist.wavelengths_angstrom().min():.2f} - {linelist.wavelengths_angstrom().max():.2f} Å")
        
        # Analyze species content
        species_count = {}
        for line in linelist:
            species_name = species_id_to_name(line.species_id)
            species_count[species_name] = species_count.get(species_name, 0) + 1
        
        print("\nSpecies distribution:")
        for species, count in sorted(species_count.items()):
            print(f"  {species}: {count} lines")
    else:
        print("No lines parsed successfully")
    
    return linelist


def main():
    """Main demonstration function"""
    
    if not JAX_AVAILABLE:
        print("❌ Cannot run demonstration - JAX not available")
        return
    
    print("🎯 Jorg Linelist Reading Demonstration")
    print("=" * 50)
    
    try:
        # Read linelist
        linelist = demonstrate_linelist_reading()
        
        if len(linelist) > 0:
            print(f"\n✅ Successfully demonstrated linelist reading!")
            print(f"   Lines read: {len(linelist)}")
            print(f"   Formats supported: VALD, Kurucz, MOOG, Turbospectrum, Korg HDF5")
            print(f"   Features: Auto-detection, species parsing, wavelength conversion")
        else:
            print(f"\n⚠️  Linelist reading needs refinement for VALD format")
    
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if Path("sample_lines.vald").exists():
            Path("sample_lines.vald").unlink()


if __name__ == "__main__":
    main()