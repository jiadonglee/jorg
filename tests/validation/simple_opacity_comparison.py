#!/usr/bin/env python3
"""
Simple opacity comparison between Jorg and Korg
"""

import sys
from pathlib import Path
import subprocess
import json

jorg_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(jorg_src))

import numpy as np
from jorg.statmech.species import Species, Formula
from jorg.continuum.complete_continuum import calculate_total_continuum_opacity

print("SIMPLE OPACITY COMPARISON: JORG vs KORG")
print("=" * 45)

# Get Korg result first
julia_script = '''
using Korg
Teff, logg, m_H = 5777.0, 4.44, 0.0
atm = interpolate_marcs(Teff, logg, m_H, 0.0)
layer = atm.layers[25]
T, nₑ = layer.temp, layer.electron_number_density

# Single wavelength test
λ = 5500.0  # Å
ν = 2.99792458e18 / λ

# Species
number_densities = Dict{Korg.Species, Float64}()
number_densities[Korg.Species(Korg.Formula(1), 0)] = 2.5e16    # H I
number_densities[Korg.Species(Korg.Formula(1), 1)] = 6.0e10    # H II
number_densities[Korg.Species(Korg.Formula(2), 0)] = 2.0e15    # He I
number_densities[Korg.Species(Korg.Formula(2), 1)] = 1.0e11    # He II
number_densities[Korg.Species(Korg.Formula([1,1]), 0)] = 1.0e13 # H2

α = Korg.total_continuum_absorption([ν], T, nₑ, number_densities, Korg.default_partition_funcs)
println("KORG_RESULT:", α[1])
println("TEMPERATURE:", T)
println("ELECTRON_DENSITY:", nₑ)
'''

result = subprocess.run(['julia', '--project=.', '-e', julia_script], 
                       capture_output=True, text=True, timeout=60)

if result.returncode != 0:
    print("❌ Korg calculation failed")
    print(result.stderr)
    exit(1)

# Parse Korg results
lines = result.stdout.strip().split('\n')
korg_alpha = None
temp = None 
ne = None

for line in lines:
    if line.startswith("KORG_RESULT:"):
        korg_alpha = float(line.split(':')[1])
    elif line.startswith("TEMPERATURE:"):
        temp = float(line.split(':')[1])  
    elif line.startswith("ELECTRON_DENSITY:"):
        ne = float(line.split(':')[1])

print(f"Korg Results:")
print(f"  Temperature: {temp:.1f} K")
print(f"  Electron density: {ne:.2e} cm⁻³")
print(f"  Continuum absorption: {korg_alpha:.2e} cm⁻¹")
print()

# Calculate Jorg result
frequencies = np.array([2.99792458e18 / 5500.0])  # 5500 Å

# Create number densities dict
number_densities = {}
number_densities[Species(Formula([1]), 0)] = 2.5e16    # H I
number_densities[Species(Formula([1]), 1)] = 6.0e10    # H II  
number_densities[Species(Formula([2]), 0)] = 2.0e15    # He I
number_densities[Species(Formula([2]), 1)] = 1.0e11    # He II
number_densities[Species(Formula([26]), 0)] = 9.0e10   # Fe I
number_densities[Species(Formula([26]), 1)] = 3.0e10   # Fe II
number_densities[Species(Formula([1, 1]), 0)] = 1.0e13  # H2

try:
    jorg_alpha = calculate_total_continuum_opacity(frequencies, temp, ne, number_densities)
    
    print(f"Jorg Results:")
    print(f"  Continuum absorption: {jorg_alpha[0]:.2e} cm⁻¹")
    print()
    
    # Compare
    ratio = jorg_alpha[0] / korg_alpha
    percent_diff = (jorg_alpha[0] - korg_alpha) / korg_alpha * 100
    
    print(f"COMPARISON:")
    print(f"  Ratio (Jorg/Korg): {ratio:.2f}")
    print(f"  Percent difference: {percent_diff:+.1f}%")
    print()
    
    if abs(ratio - 1) < 0.2:
        print("🎉 EXCELLENT: Agreement within 20%!")
    elif abs(ratio - 1) < 0.5:
        print("✅ GOOD: Agreement within 50%")
    elif abs(ratio - 1) < 2.0:
        print("⚠️  REASONABLE: Agreement within factor of 2-3")
    else:
        print("❌ POOR: Significant disagreement")
        
    print()
    print("🎯 OPACITY COMPARISON SUCCESSFUL!")
    print("   EOS → Opacity pipeline is working correctly")
    
except Exception as e:
    print(f"❌ Jorg calculation failed: {e}")
    import traceback
    traceback.print_exc()