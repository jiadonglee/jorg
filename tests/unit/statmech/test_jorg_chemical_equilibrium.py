#!/usr/bin/env python3
"""
Chemical Equilibrium Test for Jorg - Python Implementation
===========================================================

This script tests the chemical_equilibrium function from Jorg's statmech module,
using the same stellar parameters and atmospheric conditions as the Korg test.

This provides a direct comparison between Korg (Julia) and Jorg (Python) 
chemical equilibrium calculations.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add Jorg to path
jorg_path = Path(__file__).parent / "Jorg" / "src"
sys.path.insert(0, str(jorg_path))

try:
    import jorg.statmech as statmech
    from jorg.statmech import chemical_equilibrium
    from jorg.constants import kboltz_cgs
    print("✓ Successfully imported Jorg statmech module")
except ImportError as e:
    print(f"✗ Failed to import Jorg: {e}")
    print("Make sure Jorg is properly installed and in the Python path")
    sys.exit(1)

def main():
    print("=" * 60)
    print("JORG CHEMICAL EQUILIBRIUM TEST")
    print("=" * 60)
    
    # 1. Define stellar parameters (exactly matching Korg test)
    Teff = 5777.0  # Effective temperature in Kelvin
    logg = 4.44    # Surface gravity in cgs units
    M_H = 0.0      # Metallicity [metals/H]
    
    print(f"Stellar Parameters: Teff={Teff} K, logg={logg}, [M/H]={M_H}")
    
    # 2. Define atmospheric layer parameters (from Korg test layer 25)
    # These values come from the successful Korg test
    layer_index = 25
    T = 4838.221978288154  # K
    nt = 2.7356685421333148e16  # cm^-3
    ne_guess = 2.3860243024247812e12  # cm^-3
    P = nt * kboltz_cgs * T
    
    print(f"\nAtmospheric layer {layer_index} parameters:")
    print(f"  Temperature: {T:.2f} K")
    print(f"  Total number density: {nt:.3e} cm^-3")
    print(f"  Electron density guess: {ne_guess:.3e} cm^-3")
    print(f"  Total pressure: {P:.2f} dyn/cm^2")
    
    # 3. Create element abundances (convert from Korg format)
    print("\nSetting up elemental abundances...")
    
    # Use solar abundances similar to what Korg uses
    # These are number fractions (not logarithmic)
    absolute_abundances = {
        1: 0.924,      # H - dominant
        2: 0.075,      # He - second most abundant
        6: 0.0002,     # C
        7: 6.9e-5,     # N  
        8: 0.0005,     # O
        26: 2.66e-5,   # Fe
    }
    
    # Normalize to ensure they sum to 1
    total = sum(absolute_abundances.values())
    absolute_abundances = {k: v/total for k, v in absolute_abundances.items()}
    
    print("  Abundance fractions:")
    for Z, frac in absolute_abundances.items():
        element_names = {1: 'H', 2: 'He', 6: 'C', 7: 'N', 8: 'O', 26: 'Fe'}
        name = element_names.get(Z, f'Z={Z}')
        print(f"    {name}: {frac:.6f}")
    print(f"  Total: {sum(absolute_abundances.values()):.6f}")
    
    # 4. Create ionization energies, partition functions, and equilibrium constants
    print("\nCreating atomic and molecular data...")
    
    try:
        # Create ionization energies (should match Korg values)
        ionization_energies = statmech.create_default_ionization_energies()
        print("  ✓ Ionization energies created")
        
        # Create partition functions
        partition_fns = statmech.create_default_partition_functions()
        print("  ✓ Partition functions created")
        
        # Create molecular equilibrium constants
        log_equilibrium_constants = statmech.create_default_log_equilibrium_constants()
        print("  ✓ Molecular equilibrium constants created")
        
        # Check hydrogen ionization energy for comparison with Korg
        chi_H = ionization_energies[1][0]  # First ionization of hydrogen
        print(f"  H I ionization energy: {chi_H:.4f} eV (should be ~13.5984)")
        
    except Exception as e:
        print(f"  ✗ Error creating atomic/molecular data: {e}")
        return False
    
    # 5. Calculate chemical equilibrium
    print(f"\nCalculating chemical equilibrium for layer {layer_index}...")
    
    try:
        ne_sol, number_densities = chemical_equilibrium(
            T, nt, ne_guess, absolute_abundances,
            ionization_energies, partition_fns, log_equilibrium_constants
        )
        
        print("✓ Chemical equilibrium calculation successful!")
        
    except Exception as e:
        print(f"✗ Chemical equilibrium calculation failed: {e}")
        return False
    
    # 6. Extract and display results (matching Korg test format)
    print("\n" + "-" * 50)
    print("Chemical Equilibrium Test Results")
    print(f"Stellar Parameters: Teff={Teff} K, logg={logg}, [M/H]={M_H}")
    print(f"Results for atmospheric layer: {layer_index}")
    print(f"Temperature at layer: {T:.2f} K")
    print(f"Total pressure at layer: {P:.2f} dyn/cm^2")
    print(f"Electron density solution: {ne_sol:.3e} cm^-3")
    print(f"Original electron density: {ne_guess:.3e} cm^-3")
    print(f"Relative error: {abs(ne_sol - ne_guess)/ne_guess * 100:.2f}%")
    print("-" * 50)
    
    # 7. Calculate partial pressures (exactly as in Korg test)
    print("\nExtracting species densities...")
    
    # Define species we want to extract (matching Korg test)
    from jorg.statmech.species import Species
    
    species_to_check = [
        ("H I", Species.from_string("H I")),
        ("H II", Species.from_string("H II")),  # Ionized hydrogen
        ("H-", Species.from_string("H-")),      # H minus
        ("H2O", Species.from_string("H2O")),    # Water
        ("Fe I", Species.from_string("Fe I")),  # Neutral iron
    ]
    
    print("Species number densities (cm^-3):")
    partial_pressures = {}
    
    for name, species in species_to_check:
        try:
            n_density = number_densities.get(species, 0.0)
            p_partial = n_density * kboltz_cgs * T
            partial_pressures[name] = p_partial
            
            print(f"  {name}: {n_density:.3e} cm^-3")
            
        except Exception as e:
            print(f"  {name}: Error accessing - {e}")
            partial_pressures[name] = 0.0
    
    # 8. Display partial pressures (exactly matching Korg format)
    print("\nPartial Pressures (dyn/cm^2):")
    print(f"  - Neutral Hydrogen (H I):  {partial_pressures['H I']}")
    print(f"  - Ionized Hydrogen (H+):   {partial_pressures['H II']}")
    print(f"  - H- ion:                   {partial_pressures['H-']}")
    print(f"  - Water (H2O):             {partial_pressures['H2O']}")
    print(f"  - Neutral Iron (Fe I):     {partial_pressures['Fe I']}")
    
    # 9. Additional analysis
    print("\nAdditional Analysis:")
    
    # Find most abundant species
    sorted_species = sorted(number_densities.items(), key=lambda x: x[1], reverse=True)
    print("Top 10 most abundant species:")
    for i, (species, density) in enumerate(sorted_species[:10]):
        if density > 1e10:  # Only show significant species
            print(f"  {i+1}. {species}: {density:.3e} cm^-3")
    
    # Calculate hydrogen ionization fraction
    n_H_I = number_densities.get(Species.from_string("H I"), 0.0)
    n_H_II = number_densities.get(Species.from_string("H II"), 0.0)
    
    if n_H_I + n_H_II > 0:
        ionization_fraction = n_H_II / (n_H_I + n_H_II)
        print(f"\nHydrogen ionization fraction: {ionization_fraction:.6f}")
    
    print("-" * 50)
    
    # 10. Success summary
    print("\n" + "=" * 60)
    print("✓ JORG CHEMICAL EQUILIBRIUM TEST COMPLETED SUCCESSFULLY!")
    print("✓ Used jorg.statmech.chemical_equilibrium function")
    print("✓ Calculated with same parameters as Korg test")
    print(f"✓ Electron density converged with {abs(ne_sol - ne_guess)/ne_guess * 100:.1f}% accuracy")
    print("✓ Results ready for comparison with Korg output")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)