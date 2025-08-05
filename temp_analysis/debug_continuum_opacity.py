#!/usr/bin/env python3
"""
Debug continuum opacity values to understand why line windowing isn't working
"""

import sys
import numpy as np

# Add Jorg source to path
sys.path.append("/Users/jdli/Project/Korg.jl/Jorg/src/")

from jorg.synthesis import synthesize, interpolate_atmosphere
from jorg.abundances import format_A_X
from jorg.lines.linelist import read_linelist
from jorg.continuum.exact_physics_continuum import total_continuum_absorption_exact_physics_only
from jorg.constants import c_cgs

print("🔍 DEBUGGING CONTINUUM OPACITY VALUES")
print("=" * 50)

# Set up solar atmosphere  
A_X_dict = format_A_X()
A_X = np.full(92, -50.0)
A_X[0] = 12.0
for Z, abundance in A_X_dict.items():
    if 1 <= Z <= 92:
        A_X[Z-1] = abundance
        
atm = interpolate_atmosphere(Teff=5780., logg=4.44, m_H=0)

# Test wavelength range
wl_array = np.linspace(5000, 5020, 100)  # Å
wl_cm = wl_array * 1e-8  # Convert to cm
frequencies = c_cgs / wl_cm  # Hz

print(f"Wavelength range: {wl_array[0]:.1f} - {wl_array[-1]:.1f} Å")
print(f"Frequency range: {frequencies[0]:.2e} - {frequencies[-1]:.2e} Hz")

# Check different atmospheric layers
layer_to_check = [0, 20, 40, 55]  # Surface, mid, deep, bottom

for layer_idx in layer_to_check:
    if layer_idx >= len(atm.layers):
        continue
        
    layer = atm.layers[layer_idx]
    T = float(layer.temp)
    ne = float(layer.electron_number_density)
    
    print(f"\n--- Layer {layer_idx} ---")
    print(f"Temperature: {T:.0f} K")
    print(f"Electron density: {ne:.2e} cm⁻³")
    
    # Create dummy number densities for continuum calculation
    from jorg.statmech.species import Species
    number_densities = {
        Species.from_atomic_number(1, 0): 1e17,  # H I
        Species.from_atomic_number(2, 0): 1e16,  # He I
        Species.from_atomic_number(26, 0): 1e11, # Fe I
    }
    
    # Calculate continuum opacity
    try:
        continuum_opacity = total_continuum_absorption_exact_physics_only(
            frequencies, T, ne, number_densities
        )
        
        print(f"Continuum opacity range: {np.min(continuum_opacity):.2e} - {np.max(continuum_opacity):.2e} cm⁻¹")
        print(f"Mean continuum opacity: {np.mean(continuum_opacity):.2e} cm⁻¹")
        
        # Check if this is reasonable for solar photosphere
        expected_continuum = 1e-6  # Typical value
        ratio = np.mean(continuum_opacity) / expected_continuum
        print(f"Ratio to expected (~1e-6): {ratio:.1f}×")
        
        if ratio < 0.01:
            print("❌ PROBLEM: Continuum opacity is extremely low!")
        elif ratio < 0.1:
            print("⚠️  WARNING: Continuum opacity is quite low")
        else:
            print("✅ Continuum opacity looks reasonable")
            
    except Exception as e:
        print(f"❌ Error calculating continuum: {e}")

print(f"\n🎯 EXPECTED VALUES:")
print(f"Solar photosphere continuum opacity: ~1e-6 cm⁻¹")
print(f"If continuum is too low, line windowing won't work correctly!")
print(f"Line windowing needs reasonable continuum/line amplitude ratios")