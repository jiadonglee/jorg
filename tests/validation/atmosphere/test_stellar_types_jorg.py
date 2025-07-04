#!/usr/bin/env python3
"""
Test Jorg Chemical Equilibrium Across Different Stellar Types
============================================================

This script tests Jorg's chemical equilibrium calculation across various stellar types:
- Hot stars (O/B type): High temperature, high ionization
- Solar-type stars (G type): Moderate temperature, balanced ionization
- Cool stars (M dwarf): Low temperature, mostly neutral
- Giant stars: Lower gravity, different pressure structure
- Metal-poor stars: Different abundance patterns
"""

import sys
import numpy as np
from pathlib import Path

# Add Jorg to path
jorg_path = Path(__file__).parent / "Jorg" / "src"
sys.path.insert(0, str(jorg_path))

import jorg.statmech as statmech
from jorg.statmech import chemical_equilibrium
from jorg.statmech.species import Species
from jorg.constants import kboltz_cgs

def get_marcs_atmosphere_params(Teff, logg, M_H, layer_index):
    """
    Get approximate atmospheric parameters for different stellar types.
    This simulates MARCS atmosphere interpolation.
    """
    # Rough scaling relations for different stellar types
    if Teff >= 10000:  # Hot stars
        base_nt = 1e15
        T_ratio = (layer_index / 50.0) ** 0.3  # Shallow temperature gradient
    elif Teff >= 6000:  # Solar-type
        base_nt = 2e16
        T_ratio = (layer_index / 50.0) ** 0.5
    else:  # Cool stars
        base_nt = 5e16
        T_ratio = (layer_index / 50.0) ** 0.7  # Steep temperature gradient
    
    # Gravity effect
    gravity_factor = 10**(logg - 4.4)
    
    # Layer-specific parameters
    T = Teff * (0.7 + 0.3 * T_ratio)  # Temperature decreases with depth
    nt = base_nt * gravity_factor * (1 + layer_index / 100.0)
    
    # Rough electron density estimate (depends on ionization)
    if T > 8000:
        ne_frac = 0.1  # Highly ionized
    elif T > 5000:
        ne_frac = 0.001  # Moderately ionized
    else:
        ne_frac = 0.0001  # Mostly neutral
    
    ne_guess = nt * ne_frac
    
    return T, nt, ne_guess

def get_abundances(M_H):
    """Get solar abundances scaled by metallicity"""
    # Solar abundances (Asplund et al. 2020)
    solar_abundances = {
        1: 12.0,     # H
        2: 10.91,    # He
        3: 0.96,     # Li
        4: 1.38,     # Be
        5: 2.7,      # B
        6: 8.46,     # C
        7: 7.83,     # N
        8: 8.69,     # O
        9: 4.4,      # F
        10: 8.06,    # Ne
        11: 6.24,    # Na
        12: 7.6,     # Mg
        13: 6.45,    # Al
        14: 7.51,    # Si
        15: 5.41,    # P
        16: 7.12,    # S
        17: 5.25,    # Cl
        18: 6.4,     # Ar
        19: 5.04,    # K
        20: 6.34,    # Ca
        21: 3.15,    # Sc
        22: 4.95,    # Ti
        23: 3.93,    # V
        24: 5.64,    # Cr
        25: 5.42,    # Mn
        26: 7.46,    # Fe
        27: 4.94,    # Co
        28: 6.2,     # Ni
        29: 4.18,    # Cu
        30: 4.56,    # Zn
    }
    
    # Apply metallicity scaling (H and He unchanged)
    A_X_log = {}
    for Z in range(1, 93):
        if Z == 1:  # Hydrogen
            A_X_log[Z] = 12.0
        elif Z == 2:  # Helium  
            A_X_log[Z] = 10.91
        elif Z in solar_abundances:
            # Scale metals by [M/H]
            A_X_log[Z] = solar_abundances[Z] + M_H
        else:
            # Default very low abundance
            A_X_log[Z] = 1.0 + M_H
    
    # Convert to absolute abundances exactly as Korg does
    rel_abundances = {}
    for Z in range(1, 93):
        rel_abundances[Z] = 10**(A_X_log[Z] - 12.0)
    
    total_rel = sum(rel_abundances.values())
    absolute_abundances = {Z: rel / total_rel for Z, rel in rel_abundances.items()}
    
    return absolute_abundances

def test_stellar_type(description, Teff, logg, M_H):
    """Test chemical equilibrium for a specific stellar type"""
    print("\n" + "="*70)
    print(f"TESTING: {description}")
    print(f"Stellar Parameters: Teff={Teff} K, logg={logg}, [M/H]={M_H}")
    print("="*70)
    
    try:
        # Get abundances
        absolute_abundances = get_abundances(M_H)
        
        # Create atomic and molecular data
        ionization_energies = statmech.create_default_ionization_energies()
        partition_fns = statmech.create_default_partition_functions()
        log_equilibrium_constants = statmech.create_default_log_equilibrium_constants()
        
        # Test multiple atmospheric layers
        layer_indices = [15, 25, 35]  # Different depths
        results = []
        
        for layer_index in layer_indices:
            try:
                # Get atmospheric parameters
                T, nt, ne_guess = get_marcs_atmosphere_params(Teff, logg, M_H, layer_index)
                P = nt * kboltz_cgs * T
                
                # Calculate chemical equilibrium
                ne_sol, number_densities = chemical_equilibrium(
                    T, nt, ne_guess, absolute_abundances,
                    ionization_energies, partition_fns, log_equilibrium_constants
                )
                
                error_percent = abs(ne_sol - ne_guess) / ne_guess * 100
                
                # Extract key species
                n_H_I = number_densities.get(Species.from_string("H I"), 0.0)
                n_H_II = number_densities.get(Species.from_string("H II"), 0.0)
                n_H2O = number_densities.get(Species.from_string("H2O"), 0.0)
                n_Fe_I = number_densities.get(Species.from_string("Fe I"), 0.0)
                
                # Calculate ionization fraction
                ionization_fraction = n_H_II / (n_H_I + n_H_II) if (n_H_I + n_H_II) > 0 else 0.0
                
                # Store results
                result = {
                    'layer': layer_index,
                    'T': T,
                    'P': P,
                    'ne_error': error_percent,
                    'ne_sol': ne_sol,
                    'ne_guess': ne_guess,
                    'ionization_fraction': ionization_fraction,
                    'p_H_I': n_H_I * kboltz_cgs * T,
                    'p_H_II': n_H_II * kboltz_cgs * T,
                    'p_H2O': n_H2O * kboltz_cgs * T,
                    'p_Fe_I': n_Fe_I * kboltz_cgs * T
                }
                results.append(result)
                
                print(f"Layer {layer_index}: T={T:.1f}K, Error={error_percent:.1f}%, Ion_frac={ionization_fraction:.3g}")
                
            except Exception as e:
                print(f"❌ Layer {layer_index}: Chemical equilibrium failed - {e}")
        
        # Summary for this stellar type
        if results:
            avg_error = sum(r['ne_error'] for r in results) / len(results)
            max_error = max(r['ne_error'] for r in results)
            min_error = min(r['ne_error'] for r in results)
            
            print(f"\nSUMMARY for {description}:")
            print(f"  Layers tested: {len(results)}")
            print(f"  Average error: {avg_error:.1f}%")
            print(f"  Error range: {min_error:.1f}% - {max_error:.1f}%")
            
            # Show ionization range
            ion_fracs = [r['ionization_fraction'] for r in results]
            print(f"  Ionization range: {min(ion_fracs):.3g} - {max(ion_fracs):.3g}")
            
            return results
        else:
            print(f"❌ No successful calculations for {description}")
            return []
            
    except Exception as e:
        print(f"❌ Failed to test {description}: {e}")
        return []

def main():
    print("JORG CHEMICAL EQUILIBRIUM ACROSS STELLAR TYPES")
    print("=" * 80)
    print("Testing chemical equilibrium convergence for various stellar types")
    
    # Define stellar types to test
    stellar_types = [
        ("Hot B-type star", 15000.0, 4.0, 0.0),
        ("Hot A-type star", 9000.0, 4.2, 0.0),
        ("Solar-type G star", 5777.0, 4.44, 0.0),
        ("Cool K-type star", 4500.0, 4.5, 0.0),
        ("Cool M dwarf", 3500.0, 4.8, 0.0),
        ("Giant K star", 4500.0, 2.5, 0.0),
        ("Metal-poor G star", 5777.0, 4.44, -1.0),
        ("Metal-rich G star", 5777.0, 4.44, +0.3)
    ]
    
    all_results = []
    
    for description, Teff, logg, M_H in stellar_types:
        results = test_stellar_type(description, Teff, logg, M_H)
        if results:
            all_results.append((description, results))
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL JORG PERFORMANCE SUMMARY")
    print("="*80)
    
    if all_results:
        print("Stellar Type                | Avg Error | Error Range | Ion Range")
        print("-" * 70)
        
        for description, results in all_results:
            avg_error = sum(r['ne_error'] for r in results) / len(results)
            max_error = max(r['ne_error'] for r in results)
            min_error = min(r['ne_error'] for r in results)
            ion_fracs = [r['ionization_fraction'] for r in results]
            min_ion = min(ion_fracs)
            max_ion = max(ion_fracs)
            
            print(f"{description:<25} | {avg_error:7.1f}%  | {min_error:4.1f}%-{max_error:4.1f}% | {min_ion:4.2g}-{max_ion:4.2g}")
        
        # Calculate overall statistics
        all_errors = []
        for _, results in all_results:
            for r in results:
                all_errors.append(r['ne_error'])
        
        print(f"\nOverall Jorg Statistics:")
        print(f"  Total tests: {len(all_errors)}")
        print(f"  Mean error: {sum(all_errors)/len(all_errors):.2f}%")
        print(f"  Median error: {sorted(all_errors)[len(all_errors)//2]:.2f}%")
        print(f"  Max error: {max(all_errors):.1f}%")
        print(f"  Success rate: {len(all_errors)/(len(stellar_types)*3)*100:.1f}%")
    else:
        print("❌ No successful tests completed")
    
    print("="*80)
    print("✅ JORG STELLAR TYPE TESTING COMPLETE")
    print("="*80)
    
    return all_results

if __name__ == "__main__":
    results = main()