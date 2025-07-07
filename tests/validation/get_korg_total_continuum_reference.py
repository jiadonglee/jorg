#!/usr/bin/env python3
"""
Get the actual Korg total continuum reference value to ensure we're comparing correctly
"""

import subprocess

print("KORG TOTAL CONTINUUM REFERENCE")
print("=" * 34)

# Create Julia script to get Korg's total continuum
julia_script = '''
using Pkg
Pkg.activate(".")
using Korg

# Test conditions (same as our tests)
T = 4838.3  # K
ne = 2.28e12  # cm⁻³
n_HI = 2.5e16  # cm⁻³
n_HII = 6.0e10  # cm⁻³  
n_HeI = 2.0e15  # cm⁻³
n_HeII = 1.0e11  # cm⁻³
frequency = 5.451e14  # Hz (5500 Å)

println("KORG TOTAL CONTINUUM CALCULATION:")
println("T = ", T, " K")
println("ne = ", ne, " cm⁻³")
println("frequency = ", frequency, " Hz (5500 Å)")
println()

# Create number densities dict
number_densities = Dict(
    Korg.species"H_I" => n_HI,
    Korg.species"H_II" => n_HII,
    Korg.species"He_I" => n_HeI,
    Korg.species"He_II" => n_HeII,
    Korg.species"H2" => 1.0e13  # Approximate H2 density
)

# Add some metals (approximate densities)
number_densities[Korg.species"Fe_I"] = 9.0e10
number_densities[Korg.species"Fe_II"] = 3.0e10
number_densities[Korg.species"Mg_I"] = 3.0e10
number_densities[Korg.species"Si_I"] = 4.0e10
number_densities[Korg.species"Ca_I"] = 2.0e10

# Partition functions (approximate)
partition_funcs = Dict()
for species in keys(number_densities)
    partition_funcs[species] = log(T) -> 2.0  # Simplified
end

println("Number densities:")
for (species, density) in number_densities
    println("  ", species, ": ", density, " cm⁻³")
end
println()

# Calculate total continuum absorption
try
    alpha_total = Korg.ContinuumAbsorption.total_continuum_absorption(
        [frequency], T, ne, number_densities, partition_funcs
    )[1]
    
    println("KORG TOTAL CONTINUUM: ", alpha_total, " cm⁻¹")
    println()
    
    # Calculate individual components for comparison
    nH_I_div_partition = n_HI / 2.0
    
    # H⁻ bound-free
    alpha_h_minus_bf = Korg.ContinuumAbsorption.Hminus_bf([frequency], T, nH_I_div_partition, ne)[1]
    println("H⁻ bound-free: ", alpha_h_minus_bf, " cm⁻¹")
    
    # H⁻ free-free  
    alpha_h_minus_ff = Korg.ContinuumAbsorption.Hminus_ff([frequency], T, nH_I_div_partition, ne)[1]
    println("H⁻ free-free: ", alpha_h_minus_ff, " cm⁻¹")
    
    # H I bound-free
    alpha_h_i_bf = Korg.ContinuumAbsorption.H_I_bf([frequency], T, n_HI, n_HeI, ne, 1/2.0)
    println("H I bound-free: ", alpha_h_i_bf, " cm⁻¹")
    
    # Thomson scattering
    alpha_thomson = Korg.ContinuumAbsorption.electron_scattering(ne)
    println("Thomson scattering: ", alpha_thomson, " cm⁻¹")
    
    # Rayleigh scattering
    alpha_rayleigh = Korg.ContinuumAbsorption.rayleigh([frequency], n_HI, n_HeI, 1.0e13)[1]
    println("Rayleigh scattering: ", alpha_rayleigh, " cm⁻¹")
    
    # Sum of major components
    alpha_major = alpha_h_minus_bf + alpha_h_minus_ff + alpha_h_i_bf + alpha_thomson + alpha_rayleigh
    println("Sum of major components: ", alpha_major, " cm⁻¹")
    
    # Missing/other components
    alpha_missing = alpha_total - alpha_major
    println("Missing/other components: ", alpha_missing, " cm⁻¹")
    println("Missing fraction: ", alpha_missing / alpha_total * 100, "%")
    
catch e
    println("Error calculating continuum: ", e)
end
'''

# Write and run Julia script
with open('/tmp/get_korg_continuum.jl', 'w') as f:
    f.write(julia_script)

try:
    result = subprocess.run(
        ['julia', '/tmp/get_korg_continuum.jl'], 
        capture_output=True, text=True, cwd='/Users/jdli/Project/Korg.jl'
    )
    
    print("KORG OUTPUT:")
    print(result.stdout)
    
    if result.stderr:
        print("KORG ERRORS:")
        print(result.stderr)
    
    # Try to extract the total continuum value
    lines = result.stdout.split('\n')
    korg_total = None
    
    for line in lines:
        if 'KORG TOTAL CONTINUUM:' in line:
            try:
                korg_total = float(line.split(':')[1].split()[0])
                break
            except:
                pass
    
    if korg_total is not None:
        print(f"\n" + "="*50)
        print(f"EXTRACTED KORG TOTAL CONTINUUM: {korg_total:.2e} cm⁻¹")
        print(f"Previous reference value: 3.5e-9 cm⁻¹")
        print(f"Ratio: {korg_total / 3.5e-9:.2f}")
        
        if abs(korg_total / 3.5e-9 - 1.0) > 0.5:
            print("⚠️  SIGNIFICANT DIFFERENCE from previous reference!")
            print("   Our comparison may have been using wrong reference value")
        else:
            print("✅ Consistent with previous reference value")
    else:
        print("\n❌ Could not extract Korg total continuum value")
        
except Exception as e:
    print(f"Error running Julia: {e}")
    print("Could not get Korg reference value")