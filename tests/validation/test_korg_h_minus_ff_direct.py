#!/usr/bin/env python3
"""
Direct comparison of our H⁻ ff with Korg's using identical inputs
"""

import sys
from pathlib import Path
jorg_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(jorg_src))

import numpy as np
import jax.numpy as jnp
import subprocess
import json

print("DIRECT KORG H⁻ FREE-FREE COMPARISON")
print("=" * 38)

# Test conditions
T = 4838.3  # K
ne = 2.28e12  # cm⁻³
n_HI = 2.5e16  # cm⁻³
frequency = 5.451e14  # Hz (5500 Å)
wavelength_angstrom = 2.998e18 / frequency

print(f"Temperature: {T} K")
print(f"Electron density: {ne:.2e} cm⁻³")
print(f"H I density: {n_HI:.2e} cm⁻³")
print(f"Wavelength: {wavelength_angstrom:.0f} Å")
print()

# Create Julia script to get Korg's H⁻ ff value
julia_script = f'''
using Pkg
Pkg.activate(".")
using Korg

# Test conditions
T = {T}
ne = {ne}
nH_I = {n_HI}
frequency = {frequency}

# Calculate nH_I_div_partition
# For H I, partition function ≈ 2 at these temperatures
U_H_I = 2.0
nH_I_div_partition = nH_I / U_H_I

println("KORG H⁻ FREE-FREE CALCULATION:")
println("T = ", T, " K")
println("ne = ", ne, " cm⁻³")
println("nH_I = ", nH_I, " cm⁻³") 
println("nH_I_div_partition = ", nH_I_div_partition, " cm⁻³")
println("frequency = ", frequency, " Hz")
println()

# Call Korg's H⁻ ff function directly
try
    alpha_h_minus_ff = Korg.ContinuumAbsorption.Hminus_ff([frequency], T, nH_I_div_partition, ne)[1]
    println("Korg H⁻ ff: ", alpha_h_minus_ff, " cm⁻¹")
    
    # Also get the manual calculation components
    λ = 2.998e10 / frequency * 1e8  # wavelengths in Angstroms
    θ = 5040.0 / T
    println("λ = ", λ, " Å")
    println("θ = ", θ)
    
    # Try to access the interpolation table directly
    try
        K_value = 1e-26 * Korg.ContinuumAbsorption._Hminus_ff_absorption_interp(λ, θ)
        Pe = ne * Korg.kboltz_cgs * T
        nHI_gs = 2 * nH_I_div_partition
        
        println("K value: ", K_value, " cm⁴/dyn")
        println("Pe: ", Pe, " dyn/cm²")
        println("nHI_gs: ", nHI_gs, " cm⁻³")
        println("Manual α = K × Pe × nHI_gs = ", K_value * Pe * nHI_gs, " cm⁻¹")
    catch e
        println("Could not access interpolation table: ", e)
    end
    
catch e
    println("Error calling Korg H⁻ ff: ", e)
end
'''

# Write and run Julia script
with open('/tmp/test_korg_h_minus_ff.jl', 'w') as f:
    f.write(julia_script)

try:
    result = subprocess.run(
        ['julia', '/tmp/test_korg_h_minus_ff.jl'], 
        capture_output=True, text=True, cwd='/Users/jdli/Project/Korg.jl'
    )
    
    print("KORG OUTPUT:")
    print(result.stdout)
    
    if result.stderr:
        print("KORG ERRORS:")
        print(result.stderr)
    
    # Try to extract the numerical value
    lines = result.stdout.split('\n')
    korg_h_minus_ff = None
    korg_K_value = None
    
    for line in lines:
        if 'Korg H⁻ ff:' in line:
            try:
                korg_h_minus_ff = float(line.split(':')[1].split()[0])
            except:
                pass
        elif 'K value:' in line:
            try:
                korg_K_value = float(line.split(':')[1].split()[0])
            except:
                pass
    
    if korg_h_minus_ff is not None:
        print(f"\nEXTRACTED VALUES:")
        print(f"Korg H⁻ ff: {korg_h_minus_ff:.2e} cm⁻¹")
        if korg_K_value is not None:
            print(f"Korg K value: {korg_K_value:.2e} cm⁴/dyn")
    
        # Compare with our implementation
        from jorg.continuum.complete_continuum import h_minus_ff_absorption_coefficient
        
        # Use exactly the same inputs as Korg
        nH_I_div_partition = n_HI / 2.0  # Partition function = 2
        nHI_gs = 2 * nH_I_div_partition  # = n_HI
        
        jorg_h_minus_ff = h_minus_ff_absorption_coefficient(frequency, T, nHI_gs, ne)
        
        print(f"\nCOMPARISON:")
        print(f"Korg:  {korg_h_minus_ff:.2e} cm⁻¹")
        print(f"Jorg:  {float(jorg_h_minus_ff):.2e} cm⁻¹")
        print(f"Ratio (Korg/Jorg): {korg_h_minus_ff / float(jorg_h_minus_ff):.1f}")
        
        if korg_K_value is not None:
            # Calculate what our K value should be
            Pe = ne * 1.381e-16 * T
            our_K = float(jorg_h_minus_ff) / (Pe * nHI_gs)
            print(f"\nK VALUE COMPARISON:")
            print(f"Korg K:  {korg_K_value:.2e} cm⁴/dyn")
            print(f"Jorg K:  {our_K:.2e} cm⁴/dyn")
            print(f"Ratio (Korg/Jorg): {korg_K_value / our_K:.1f}")
        
        # Determine the issue
        ratio = korg_h_minus_ff / float(jorg_h_minus_ff)
        if ratio > 20:
            print(f"\n🔧 MAJOR SCALING ISSUE: Factor {ratio:.0f} too small")
            print("   Check Bell & Berrington table interpolation")
        elif ratio > 5:
            print(f"\n⚠️  MODERATE ISSUE: Factor {ratio:.0f} too small") 
            print("   Likely Bell & Berrington K values or pressure")
        elif ratio > 1.5:
            print(f"\n✅ CLOSE: Factor {ratio:.1f} difference")
            print("   Small adjustments needed")
        else:
            print("\n🎉 EXCELLENT AGREEMENT!")
    else:
        print("\n❌ Could not extract Korg H⁻ ff value from output")
        
except Exception as e:
    print(f"Error running Julia: {e}")
    print("Manual analysis needed")