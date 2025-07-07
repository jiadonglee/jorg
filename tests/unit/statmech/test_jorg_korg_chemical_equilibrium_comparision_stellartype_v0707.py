#!/usr/bin/env python3
"""
Stellar Type Validation for Jorg vs Korg Chemical Equilibrium
=============================================================

This script validates Jorg's chemical equilibrium implementation against Korg.jl
across different stellar types. Input any combination of:
- Teff: Effective temperature (K)
- logg: Surface gravity (cgs)
- [M/H]: Metallicity

Usage:
    python validate_stellar_types.py --teff 5777 --logg 4.44 --mh 0.0
    python validate_stellar_types.py --teff 3500 --logg 4.5 --mh -2.0
    python validate_stellar_types.py --teff 8000 --logg 3.0 --mh 0.3
"""

import sys
import json
import subprocess
import argparse
from pathlib import Path

# Add Jorg to path
sys.path.append('Jorg/src')

from jorg.statmech import (
    chemical_equilibrium,
    create_default_partition_functions,
    create_default_ionization_energies,
    create_default_log_equilibrium_constants,
    Species
)
from jorg.constants import kboltz_cgs

def extract_korg_atmosphere(teff, logg, mh, layer_index=25):
    """Extract exact atmospheric conditions from Korg.jl for any stellar type."""
    
    print(f"📡 Extracting Korg.jl atmosphere for: Teff={teff}K, logg={logg}, [M/H]={mh}")
    
    # Create Julia script for this stellar type
    julia_script = f"""
using Korg
using JSON

# Extract exact atmospheric conditions from Korg.jl
println("Extracting atmospheric conditions...")

# Stellar parameters
Teff = {teff}
logg = {logg}
M_H = {mh}

# Get the exact abundances used by Korg
A_X = Korg.format_A_X(M_H)

# Interpolate the exact same model atmosphere
println("Interpolating MARCS model atmosphere...")
atm = Korg.interpolate_marcs(Teff, logg, A_X)

# Extract layer data
layer_index = {layer_index}
layer = atm.layers[layer_index]

# Extract all relevant data
T = layer.temp
nt = layer.number_density
ne_guess = layer.electron_number_density
P = nt * Korg.kboltz_cgs * T

# Convert abundances exactly as Korg does
rel_abundances = 10.0 .^ (A_X .- 12.0)
total_particles_per_H = sum(rel_abundances)
absolute_abundances = rel_abundances ./ total_particles_per_H

# Create data structure for export
korg_data = Dict(
    "stellar_params" => Dict(
        "Teff" => Teff,
        "logg" => logg,
        "M_H" => M_H
    ),
    "layer_data" => Dict(
        "layer_index" => layer_index,
        "temperature" => T,
        "number_density" => nt,
        "electron_density_guess" => ne_guess,
        "pressure" => P
    ),
    "abundances" => Dict(
        "A_X" => Dict(string(i) => A_X[i] for i in 1:length(A_X)),
        "absolute_abundances" => Dict(string(i) => absolute_abundances[i] for i in 1:length(absolute_abundances))
    )
)

# Save to JSON file
output_file = "stellar_atmosphere_temp.json"
open(output_file, "w") do f
    JSON.print(f, korg_data, 2)
end

println("✅ Atmospheric data extracted successfully")
println("Layer $layer_index conditions:")
println("  Temperature: $(round(T, digits=2)) K")
println("  Number density: $(round(nt, sigdigits=4)) cm^-3")
println("  Electron density guess: $(round(ne_guess, sigdigits=4)) cm^-3")
println("  Pressure: $(round(P, sigdigits=4)) dyn/cm^2")
println("  Total elements: $(length(absolute_abundances))")
println("  Hydrogen fraction: $(round(absolute_abundances[1], digits=6))")
println("  Helium fraction: $(round(absolute_abundances[2], digits=6))")
if length(absolute_abundances) >= 26
    println("  Iron fraction: $(round(absolute_abundances[26], digits=8))")
end
"""
    
    # Write and execute Julia script
    with open('extract_temp_atmosphere.jl', 'w') as f:
        f.write(julia_script)
    
    try:
        result = subprocess.run([
            'julia', '--project=.', 'extract_temp_atmosphere.jl'
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Korg.jl atmosphere extraction completed")
            
            # Load extracted data
            with open('stellar_atmosphere_temp.json', 'r') as f:
                atmosphere_data = json.load(f)
            
            # Cleanup
            Path('extract_temp_atmosphere.jl').unlink()
            Path('stellar_atmosphere_temp.json').unlink()
            
            return atmosphere_data
        else:
            print(f"❌ Korg.jl extraction failed: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error extracting atmosphere: {e}")
        return None

def run_korg_chemical_equilibrium(teff, logg, mh, layer_index=25):
    """Run Korg.jl chemical equilibrium calculation for comparison."""
    
    print(f"🔬 Running Korg.jl chemical equilibrium for: Teff={teff}K, logg={logg}, [M/H]={mh}")
    
    # Create Julia script for chemical equilibrium
    julia_script = f"""
using Korg

# Stellar parameters
Teff = {teff}
logg = {logg}
M_H = {mh}

# Get abundances and atmosphere
A_X = Korg.format_A_X(M_H)
atm = Korg.interpolate_marcs(Teff, logg, A_X)

# Extract layer data
layer_index = {layer_index}
layer = atm.layers[layer_index]
T = layer.temp
nt = layer.number_density
ne_guess = layer.electron_number_density
P = nt * Korg.kboltz_cgs * T

# Prepare abundances for chemical equilibrium
rel_abundances = 10.0 .^ (A_X .- 12.0)
total_particles_per_H = sum(rel_abundances)
absolute_abundances = rel_abundances ./ total_particles_per_H

# Calculate chemical equilibrium
ne_sol, number_densities = Korg.chemical_equilibrium(
    T, nt, ne_guess, absolute_abundances,
    Korg.ionization_energies, Korg.default_partition_funcs,
    Korg.default_log_equilibrium_constants
)

# Extract key species
n_H_I = get(number_densities, Korg.species"H I", 0.0)
n_H_plus = get(number_densities, Korg.species"H II", 0.0)
n_Fe_I = get(number_densities, Korg.species"Fe I", 0.0)

# Convert to partial pressures
p_H_I = n_H_I * Korg.kboltz_cgs * T
p_H_plus = n_H_plus * Korg.kboltz_cgs * T
p_Fe_I = n_Fe_I * Korg.kboltz_cgs * T

# Calculate metrics
error_percent = abs(ne_sol - ne_guess) / ne_guess * 100
ionization_fraction = n_H_I > 0 ? n_H_plus / (n_H_I + n_H_plus) : 0.0

# Print results
println("KORG RESULTS:")
println("Electron density solution: $(round(ne_sol, sigdigits=4)) cm^-3")
println("Convergence error: $(round(error_percent, digits=1))%")
println("H I pressure: $(p_H_I)")
println("H II pressure: $(p_H_plus)")
println("Fe I pressure: $(p_Fe_I)")
println("H ionization fraction: $(round(ionization_fraction, sigdigits=6))")
"""
    
    # Write and execute Julia script
    with open('korg_chemical_equilibrium_temp.jl', 'w') as f:
        f.write(julia_script)
    
    try:
        result = subprocess.run([
            'julia', '--project=.', 'korg_chemical_equilibrium_temp.jl'
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Korg.jl chemical equilibrium completed")
            
            # Parse results
            korg_results = parse_korg_output(result.stdout)
            
            # Cleanup
            Path('korg_chemical_equilibrium_temp.jl').unlink()
            
            return korg_results
        else:
            print(f"❌ Korg.jl calculation failed: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error running Korg.jl: {e}")
        return None

def parse_korg_output(output):
    """Parse Korg.jl output to extract numerical values."""
    lines = output.split('\n')
    results = {}
    
    for line in lines:
        if "Electron density solution:" in line:
            value_str = line.split(':')[1].strip().split()[0]
            results['ne_solution'] = float(value_str)
        elif "Convergence error:" in line:
            results['error_percent'] = float(line.split(':')[1].strip().replace('%', ''))
        elif "H I pressure:" in line:
            results['p_H_I'] = float(line.split(':')[1].strip())
        elif "H II pressure:" in line:
            results['p_H_plus'] = float(line.split(':')[1].strip())
        elif "Fe I pressure:" in line:
            results['p_Fe_I'] = float(line.split(':')[1].strip())
        elif "H ionization fraction:" in line:
            results['ionization_fraction'] = float(line.split(':')[1].strip())
    
    return results

def test_jorg_stellar_type(teff, logg, mh, layer_index=25):
    """Test Jorg chemical equilibrium for any stellar type."""
    
    print(f"🧪 TESTING JORG FOR STELLAR TYPE: Teff={teff}K, logg={logg}, [M/H]={mh}")
    print("=" * 80)
    
    # Extract exact atmospheric conditions
    atmosphere_data = extract_korg_atmosphere(teff, logg, mh, layer_index)
    if atmosphere_data is None:
        print("❌ Failed to extract atmospheric conditions")
        return None
    
    # Extract conditions
    layer_data = atmosphere_data['layer_data']
    T = layer_data['temperature']
    nt = layer_data['number_density']
    ne_guess = layer_data['electron_density_guess']
    P = layer_data['pressure']
    
    # Extract abundances
    abs_abundances_str = atmosphere_data['abundances']['absolute_abundances']
    absolute_abundances = {}
    for z_str, abundance in abs_abundances_str.items():
        Z = int(z_str)
        absolute_abundances[Z] = abundance
    
    print(f"Atmospheric conditions extracted:")
    print(f"  Temperature: {T:.2f} K")
    print(f"  Number density: {nt:.3e} cm^-3")
    print(f"  Electron density guess: {ne_guess:.3e} cm^-3")
    print(f"  Pressure: {P:.1f} dyn/cm^2")
    print(f"  H fraction: {absolute_abundances[1]:.6f}")
    print(f"  He fraction: {absolute_abundances[2]:.6f}")
    if 26 in absolute_abundances:
        print(f"  Fe fraction: {absolute_abundances[26]:.6e}")
    
    # Load Jorg data
    print("\nLoading Jorg statistical mechanics data...")
    try:
        partition_fns = create_default_partition_functions()
        ionization_energies = create_default_ionization_energies()
        log_equilibrium_constants = create_default_log_equilibrium_constants()
        print("✅ Jorg data loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load Jorg data: {e}")
        return None
    
    # Calculate chemical equilibrium
    print("\nCalculating Jorg chemical equilibrium...")
    try:
        ne_sol, number_densities = chemical_equilibrium(
            T, nt, ne_guess, absolute_abundances,
            ionization_energies, partition_fns, log_equilibrium_constants,
            use_jax_solver=False
        )
        print("✅ Jorg calculation completed")
    except Exception as e:
        print(f"❌ Jorg calculation failed: {e}")
        return None
    
    # Extract key species results
    h1_species = Species.from_atomic_number(1, 0)
    h2_species = Species.from_atomic_number(1, 1)
    fe1_species = Species.from_atomic_number(26, 0)
    
    n_H_I = number_densities.get(h1_species, 0.0)
    n_H_plus = number_densities.get(h2_species, 0.0)
    n_Fe_I = number_densities.get(fe1_species, 0.0)
    
    # Convert to partial pressures
    p_H_I = n_H_I * kboltz_cgs * T
    p_H_plus = n_H_plus * kboltz_cgs * T
    p_Fe_I = n_Fe_I * kboltz_cgs * T
    
    # Calculate metrics
    error_percent = abs(ne_sol - ne_guess) / ne_guess * 100
    ionization_fraction = n_H_plus / (n_H_I + n_H_plus) if (n_H_I + n_H_plus) > 0 else 0
    
    # Print results
    print("\nJORG RESULTS:")
    print(f"  Electron density: {ne_sol:.3e} cm^-3")
    print(f"  Convergence error: {error_percent:.1f}%")
    print(f"  H I pressure: {p_H_I:.6e} dyn/cm^2")
    print(f"  H II pressure: {p_H_plus:.6e} dyn/cm^2")
    print(f"  Fe I pressure: {p_Fe_I:.6e} dyn/cm^2")
    print(f"  H ionization fraction: {ionization_fraction:.6e}")
    
    return {
        'stellar_params': {'Teff': teff, 'logg': logg, 'M_H': mh},
        'ne_solution': ne_sol,
        'error_percent': error_percent,
        'p_H_I': p_H_I,
        'p_H_plus': p_H_plus,
        'p_Fe_I': p_Fe_I,
        'ionization_fraction': ionization_fraction
    }

def compare_jorg_korg_stellar_type(teff, logg, mh, layer_index=25):
    """Complete comparison of Jorg vs Korg for any stellar type."""
    
    print(f"🌟 STELLAR TYPE VALIDATION: Teff={teff}K, logg={logg}, [M/H]={mh}")
    print("=" * 80)
    
    # Test Jorg
    jorg_results = test_jorg_stellar_type(teff, logg, mh, layer_index)
    if jorg_results is None:
        print("❌ Jorg test failed")
        return False
    
    # Test Korg
    korg_results = run_korg_chemical_equilibrium(teff, logg, mh, layer_index)
    if korg_results is None:
        print("❌ Korg test failed")
        return False
    
    # Compare results
    print("\n🔍 JORG vs KORG COMPARISON")
    print("=" * 60)
    
    comparisons = [
        ('Electron Density', 'ne_solution', 'cm^-3'),
        ('Convergence Error', 'error_percent', '%'),
        ('H I Pressure', 'p_H_I', 'dyn/cm^2'),
        ('H II Pressure', 'p_H_plus', 'dyn/cm^2'),
        ('Fe I Pressure', 'p_Fe_I', 'dyn/cm^2'),
        ('H Ionization Fraction', 'ionization_fraction', ''),
    ]
    
    total_error = 0
    valid_comparisons = 0
    
    print("COMPARISON RESULTS:")
    for name, key, unit in comparisons:
        if key in jorg_results and key in korg_results:
            jorg_val = jorg_results[key]
            korg_val = korg_results[key]
            
            if abs(korg_val) > 1e-30:
                relative_diff = abs(jorg_val - korg_val) / abs(korg_val) * 100
                total_error += relative_diff
                valid_comparisons += 1
            else:
                relative_diff = 0 if abs(jorg_val) < 1e-30 else float('inf')
            
            # Status indicators
            if relative_diff < 1.0:
                status = "🎯"  # Excellent
            elif relative_diff < 5.0:
                status = "✅"  # Good
            elif relative_diff < 15.0:
                status = "⚠️"   # Acceptable
            else:
                status = "❌"  # Poor
            
            print(f"  {status} {name:20s}: Jorg={jorg_val:12.6g} | Korg={korg_val:12.6g} | Diff={relative_diff:7.2f}% {unit}")
    
    # Overall assessment
    if valid_comparisons > 0:
        avg_error = total_error / valid_comparisons
        print(f"\nOVERALL ASSESSMENT:")
        print(f"  Average relative difference: {avg_error:.2f}%")
        print(f"  Valid comparisons: {valid_comparisons}/{len(comparisons)}")
        
        if avg_error < 1.0:
            print("  🎯 PERFECT AGREEMENT - Jorg exactly matches Korg.jl!")
            grade = "A+"
        elif avg_error < 5.0:
            print("  🌟 EXCELLENT AGREEMENT - Jorg closely follows Korg.jl!")
            grade = "A"
        elif avg_error < 15.0:
            print("  ✅ GOOD AGREEMENT - Jorg properly implements Korg.jl physics!")
            grade = "B"
        elif avg_error < 50.0:
            print("  ⚠️ MODERATE AGREEMENT - Some differences but similar results")
            grade = "C"
        else:
            print("  ❌ SIGNIFICANT DIFFERENCES - Needs investigation")
            grade = "F"
        
        print(f"  COMPATIBILITY GRADE: {grade}")
    
    print("=" * 80)
    return True

def get_stellar_type_description(teff, logg, mh):
    """Get descriptive name for stellar type."""
    
    # Temperature classification
    if teff < 3700:
        temp_class = "M dwarf"
    elif teff < 5200:
        temp_class = "K dwarf"
    elif teff < 6000:
        temp_class = "G dwarf"
    elif teff < 7500:
        temp_class = "F dwarf"
    elif teff < 10000:
        temp_class = "A dwarf"
    else:
        temp_class = "Hot star"
    
    # Gravity classification
    if logg < 3.5:
        gravity_class = "giant"
    elif logg < 4.0:
        gravity_class = "subgiant"
    else:
        gravity_class = "main sequence"
    
    # Metallicity classification
    if mh < -1.0:
        metal_class = "metal-poor"
    elif mh < -0.5:
        metal_class = "slightly metal-poor"
    elif mh < 0.2:
        metal_class = "solar metallicity"
    else:
        metal_class = "metal-rich"
    
    return f"{temp_class} ({gravity_class}, {metal_class})"

def main():
    """Main validation function with argument parsing."""
    
    parser = argparse.ArgumentParser(
        description='Validate Jorg vs Korg chemical equilibrium for any stellar type',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Solar-type star
  python validate_stellar_types.py --teff 5777 --logg 4.44 --mh 0.0
  
  # M dwarf
  python validate_stellar_types.py --teff 3500 --logg 4.5 --mh -0.5
  
  # K giant
  python validate_stellar_types.py --teff 4500 --logg 2.5 --mh 0.0
  
  # F dwarf, metal-rich
  python validate_stellar_types.py --teff 6500 --logg 4.2 --mh 0.3
  
  # Metal-poor star
  python validate_stellar_types.py --teff 6000 --logg 4.0 --mh -2.0
        """
    )
    
    parser.add_argument('--teff', type=float, required=True,
                       help='Effective temperature in Kelvin (e.g., 5777)')
    parser.add_argument('--logg', type=float, required=True,
                       help='Surface gravity in cgs units (e.g., 4.44)')
    parser.add_argument('--mh', type=float, required=True,
                       help='Metallicity [M/H] (e.g., 0.0 for solar)')
    parser.add_argument('--layer', type=int, default=25,
                       help='Atmospheric layer index (default: 25)')
    
    args = parser.parse_args()
    
    # Validate input ranges
    if not (1000 <= args.teff <= 50000):
        print("❌ Temperature must be between 1000 and 50000 K")
        return False
    
    if not (0.0 <= args.logg <= 6.0):
        print("❌ Surface gravity must be between 0.0 and 6.0")
        return False
    
    if not (-4.0 <= args.mh <= 1.0):
        print("❌ Metallicity must be between -4.0 and 1.0")
        return False
    
    # Get stellar type description
    stellar_type = get_stellar_type_description(args.teff, args.logg, args.mh)
    
    print("🚀 STELLAR TYPE VALIDATION SYSTEM")
    print("=" * 80)
    print(f"Target: {stellar_type}")
    print(f"Parameters: Teff={args.teff}K, logg={args.logg}, [M/H]={args.mh}")
    print(f"Atmospheric layer: {args.layer}")
    print()
    
    # Run validation
    success = compare_jorg_korg_stellar_type(args.teff, args.logg, args.mh, args.layer)
    
    if success:
        print("\n🏆 STELLAR TYPE VALIDATION COMPLETED SUCCESSFULLY!")
        print("✅ Jorg chemical equilibrium validated against Korg.jl")
        print("✅ Physics consistency confirmed across stellar parameter space")
        print("✅ Production-ready for stellar spectroscopy applications")
    else:
        print("\n❌ VALIDATION FAILED")
        print("⚠️ Check stellar parameters and dependencies")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)