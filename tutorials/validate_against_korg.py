#!/usr/bin/env python3
"""
Direct validation of fixed Jorg implementation against Korg.jl
"""

import sys
import os
sys.path.append('/Users/jdli/Project/Korg.jl/Jorg/src')

import numpy as np
import jax.numpy as jnp
from jorg.lines.opacity import calculate_line_opacity_korg_method
from jorg.statmech.partition_functions import create_default_partition_functions
from jorg.statmech.species import Species

def create_korg_reference_script():
    """
    Create a Julia script to generate Korg.jl reference values
    """
    script_content = '''
using Pkg
Pkg.activate("/Users/jdli/Project/Korg.jl")
using Korg

# Exact same parameters as Jorg test
λs = Korg.Wavelengths(5430.0:0.02:5440.0)
temperature = 5780.0
electron_density = 1e14
hydrogen_density = 1e16
abundance = 3.16e-5
microturbulence_cms = 200000.0  # 2 km/s in cm/s

# Set up densities
n_densities = Dict(
    Korg.species"Fe I" => [abundance * hydrogen_density],
    Korg.species"H_I" => [hydrogen_density]
)

# Create the line
line = Korg.Line(5434.5, -2.12, Korg.species"Fe I", 1.01)

# Calculate
α = zeros(1, length(λs))
Korg.line_absorption!(α, [line], λs, [temperature], [electron_density], 
                     n_densities, Korg.default_partition_funcs, 
                     microturbulence_cms, [λ -> 1e-26])

println("=== KORG.JL REFERENCE VALUES ===")
peak_idx = argmax(α[1, :])
peak_opacity = α[1, peak_idx]
peak_wavelength = λs[peak_idx] * 1e8  # Convert to Angstroms

println("Peak opacity: ", peak_opacity, " cm⁻¹")
println("Peak wavelength: ", peak_wavelength, " Å")
println("Line center: 5434.5 Å")

# Sample values around the peak
sample_indices = [peak_idx-10, peak_idx-5, peak_idx, peak_idx+5, peak_idx+10]
println("\\n=== SAMPLE VALUES ===")
for i in sample_indices
    if 1 <= i <= length(λs)
        wl = λs[i] * 1e8
        opacity = α[1, i]
        println("λ = ", round(wl, digits=2), " Å: ", opacity, " cm⁻¹")
    end
end
'''
    
    with open('/Users/jdli/Project/Korg.jl/Jorg/tutorials/korg_reference.jl', 'w') as f:
        f.write(script_content)
    
    return '/Users/jdli/Project/Korg.jl/Jorg/tutorials/korg_reference.jl'

def validate_jorg_implementation():
    """
    Test Jorg implementation and compare with expected Korg values
    """
    print("=== JORG VALIDATION ===\n")
    
    # Same exact parameters
    wavelengths = jnp.linspace(5430, 5440, 501)
    line_wavelength = 5434.5
    excitation_potential = 1.01
    log_gf = -2.12
    temperature = 5780.0
    electron_density = 1e14
    hydrogen_density = 1e16
    abundance = 3.16e-5
    atomic_mass = 55.845
    microturbulence = 2.0  # km/s
    
    # Get exact partition function
    partition_funcs = create_default_partition_functions()
    fe_i_species = Species.from_atomic_number(26, 0)
    U_exact = float(partition_funcs[fe_i_species](jnp.log(temperature)))
    
    # Calculate with fixed method
    opacity_jorg = calculate_line_opacity_korg_method(
        wavelengths=wavelengths,
        line_wavelength=line_wavelength,
        excitation_potential=excitation_potential,
        log_gf=log_gf,
        temperature=temperature,
        electron_density=electron_density,
        hydrogen_density=hydrogen_density,
        abundance=abundance,
        atomic_mass=atomic_mass,
        microturbulence=microturbulence,
        partition_function=U_exact
    )
    
    # Find peak
    peak_idx = jnp.argmax(opacity_jorg)
    peak_opacity = float(opacity_jorg[peak_idx])
    peak_wavelength = float(wavelengths[peak_idx])
    
    print(f"Jorg Results:")
    print(f"  Peak opacity: {peak_opacity:.6e} cm⁻¹")
    print(f"  Peak wavelength: {peak_wavelength:.2f} Å")
    
    # Sample around peak
    print(f"\n  Sample values around peak:")
    for offset in [-10, -5, 0, 5, 10]:
        idx = peak_idx + offset
        if 0 <= idx < len(wavelengths):
            wl = float(wavelengths[idx])
            opacity = float(opacity_jorg[idx])
            print(f"    λ = {wl:.2f} Å: {opacity:.6e} cm⁻¹")
    
    return opacity_jorg, peak_opacity

if __name__ == "__main__":
    # Create Korg reference script
    korg_script = create_korg_reference_script()
    print(f"Created Korg reference script: {korg_script}")
    print("Run with: julia korg_reference.jl")
    print()
    
    # Test Jorg
    opacity_jorg, peak_opacity = validate_jorg_implementation()
    
    print(f"\n" + "="*60)
    print(f"JORG PEAK OPACITY: {peak_opacity:.6e} cm⁻¹")
    print(f"")
    print(f"To compare with Korg.jl, run:")
    print(f"  cd /Users/jdli/Project/Korg.jl/Jorg/tutorials")
    print(f"  julia korg_reference.jl")
    print(f"="*60)