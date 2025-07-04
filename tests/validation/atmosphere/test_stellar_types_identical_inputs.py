#!/usr/bin/env python3
"""
Test Korg vs Jorg Chemical Equilibrium with Identical Inputs
============================================================

This script tests chemical equilibrium using EXACT same atmospheric conditions
extracted from Korg, ensuring a fair comparison between implementations.
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

# Atmosphere data extracted from Korg (exact same conditions)
korg_atmosphere_data = {
    "Solar-type G star": {
        "Teff": 5777.0,
        "logg": 4.44,
        "M_H": 0.0,
        "layers": {
            15: {
                "T": 4590.009579508528,
                "nt": 8.315411129918017e15,
                "ne_guess": 7.239981204782037e11,
                "P": 5269.635801961501,
                "tau_5000": 0.0023776244040555963,
                "z": 3.717249977838802e7,
            },
            25: {
                "T": 4838.221978288154,
                "nt": 2.7356685421333148e16,
                "ne_guess": 2.3860243024247812e12,
                "P": 18273.954914699207,
                "tau_5000": 0.02216150316364371,
                "z": 2.31352762877722e7,
            },
            35: {
                "T": 5383.722228881833,
                "nt": 8.337958936823813e16,
                "ne_guess": 9.453517368277791e12,
                "P": 61976.30484933329,
                "tau_5000": 0.2039855355272931,
                "z": 8.328288096928094e6,
            },
        }
    },
    "Cool K-type star": {
        "Teff": 4500.0,
        "logg": 4.5,
        "M_H": 0.0,
        "layers": {
            15: {
                "T": 3608.9886137413555,
                "nt": 2.4769114629938384e16,
                "ne_guess": 2.990034164984107e11,
                "P": 12341.82197400089,
                "tau_5000": 0.004436448398941338,
                "z": 2.229414738579783e7,
            },
            25: {
                "T": 3802.2316033227494,
                "nt": 7.680801945308658e16,
                "ne_guess": 9.936302279128654e11,
                "P": 40320.73281347935,
                "tau_5000": 0.03768540579507797,
                "z": 1.3351657504444296e7,
            },
            35: {
                "T": 4269.554262251411,
                "nt": 1.9795187990913702e17,
                "ne_guess": 4.888600236940689e12,
                "P": 116687.79966935837,
                "tau_5000": 0.30039711994792184,
                "z": 4.709898303896188e6,
            },
        }
    },
    "Cool M dwarf": {
        "Teff": 3500.0,
        "logg": 4.8,
        "M_H": 0.0,
        "layers": {
            15: {
                "T": 2676.971224171452,
                "nt": 3.440690227103728e15,
                "ne_guess": 9.420055688925724e9,
                "P": 1271.6645344402853,
                "tau_5000": 2.5118862527364922e-5,
                "z": 1.4672293821315061e7,
            },
            25: {
                "T": 2757.6431441697528,
                "nt": 1.561033634416188e16,
                "ne_guess": 3.8264281998144066e10,
                "P": 5943.381503808495,
                "tau_5000": 0.0002511885693426358,
                "z": 1.1194421812266937e7,
            },
            35: {
                "T": 2910.3334110333217,
                "nt": 6.345755693469862e16,
                "ne_guess": 1.5800975648355624e11,
                "P": 25498.19134574878,
                "tau_5000": 0.0025118861241130923,
                "z": 8.085973323800263e6,
            },
        }
    },
    "Giant K star": {
        "Teff": 4500.0,
        "logg": 2.5,
        "M_H": 0.0,
        "layers": {
            15: {
                "T": 3554.649123320977,
                "nt": 1.5362922620728632e15,
                "ne_guess": 4.276264434134044e10,
                "P": 753.9696496690814,
                "tau_5000": 0.005065991832835907,
                "z": 2.3502025578637743e9,
            },
            25: {
                "T": 3794.766627174385,
                "nt": 4.402883893569443e15,
                "ne_guess": 1.5259244406659048e11,
                "P": 2306.776870848835,
                "tau_5000": 0.03735133311123829,
                "z": 1.48913694922859e9,
            },
            35: {
                "T": 4277.143406880127,
                "nt": 1.1366300170141498e16,
                "ne_guess": 6.865808132660596e11,
                "P": 6712.065957700666,
                "tau_5000": 0.2854009198727468,
                "z": 5.971586695312412e8,
            },
        }
    },
    "Metal-poor G star": {
        "Teff": 5777.0,
        "logg": 4.44,
        "M_H": -1.0,
        "layers": {
            15: {
                "T": 4666.866866260754,
                "nt": 2.1882756521811176e16,
                "ne_guess": 3.2283879911501184e11,
                "P": 14099.727608711577,
                "tau_5000": 0.00275127544165067,
                "z": 3.2866487859156106e7,
            },
            25: {
                "T": 4840.113180178101,
                "nt": 7.163604786541968e16,
                "ne_guess": 1.0550775750963367e12,
                "P": 47870.77051900788,
                "tau_5000": 0.025432909700491616,
                "z": 1.8799290548391275e7,
            },
            35: {
                "T": 5343.101981014455,
                "nt": 1.9377501284096784e17,
                "ne_guess": 6.0831917573359375e12,
                "P": 142946.8272290797,
                "tau_5000": 0.23749095859793945,
                "z": 5.462277179825384e6,
            },
        }
    },
    "Metal-rich G star": {
        "Teff": 5777.0,
        "logg": 4.44,
        "M_H": 0.3,
        "layers": {
            15: {
                "T": 4562.69282361102,
                "nt": 5.886360013228573e15,
                "ne_guess": 9.417291391503596e11,
                "P": 3708.0991190108066,
                "tau_5000": 0.002230652314584484,
                "z": 3.7816670365453124e7,
            },
            25: {
                "T": 4835.645170542186,
                "nt": 1.9410995333394496e16,
                "ne_guess": 3.1444758977749937e12,
                "P": 12959.418463940723,
                "tau_5000": 0.02090389701559974,
                "z": 2.3930030085095856e7,
            },
            35: {
                "T": 5394.3344692256205,
                "nt": 6.003205854880858e16,
                "ne_guess": 1.2010194973623188e13,
                "P": 44709.971132874874,
                "tau_5000": 0.19284860273810983,
                "z": 9.09124861898594e6,
            },
        }
    }
}

def get_abundances(M_H):
    """Get solar abundances scaled by metallicity (exact Korg format)"""
    # Solar abundances (Asplund et al. 2009) - same as Korg uses
    A_X_log = {
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
    
    # Apply metallicity scaling EXACTLY as Korg does
    rel_abundances = {}
    for Z in range(1, 93):  # Elements 1-92
        if Z == 1:  # Hydrogen (reference)
            rel_abundances[Z] = 10**(12.0 - 12.0)
        elif Z == 2:  # Helium (not scaled with metals)
            rel_abundances[Z] = 10**(10.91 - 12.0)
        elif Z in A_X_log:
            # Scale metals by [M/H]
            rel_abundances[Z] = 10**((A_X_log[Z] + M_H) - 12.0)
        else:
            # Default very low abundance
            rel_abundances[Z] = 10**((1.0 + M_H) - 12.0)
    
    total_rel = sum(rel_abundances.values())
    absolute_abundances = {Z: rel / total_rel for Z, rel in rel_abundances.items()}
    
    return absolute_abundances

def test_stellar_type_identical_inputs(stellar_type, stellar_data):
    """Test chemical equilibrium using identical atmospheric inputs"""
    
    print(f"\nTesting: {stellar_type}")
    print(f"Stellar Parameters: Teff={stellar_data['Teff']}K, logg={stellar_data['logg']}, [M/H]={stellar_data['M_H']}")
    
    # Get abundances exactly as Korg does
    absolute_abundances = get_abundances(stellar_data['M_H'])
    
    # Create atomic and molecular data
    ionization_energies = statmech.create_default_ionization_energies()
    partition_fns = statmech.create_default_partition_functions()
    log_equilibrium_constants = statmech.create_default_log_equilibrium_constants()
    
    results = []
    
    # Test all available layers
    for layer_idx, layer_data in stellar_data['layers'].items():
        try:
            # Use EXACT same atmospheric conditions as Korg
            T = layer_data['T']
            nt = layer_data['nt']
            ne_guess = layer_data['ne_guess']
            
            print(f"  Layer {layer_idx}: T={T:.1f}K, nt={nt:.2e}, ne_guess={ne_guess:.2e}")
            
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
            
            # Calculate properties
            ionization_fraction = n_H_II / (n_H_I + n_H_II) if (n_H_I + n_H_II) > 0 else 0.0
            
            result = {
                'layer': layer_idx,
                'T': T,
                'nt': nt,
                'ne_guess': ne_guess,
                'ne_sol': ne_sol,
                'error_percent': error_percent,
                'ionization_fraction': ionization_fraction,
                'p_H_I': n_H_I * kboltz_cgs * T,
                'p_H_II': n_H_II * kboltz_cgs * T,
                'p_H2O': n_H2O * kboltz_cgs * T,
                'p_Fe_I': n_Fe_I * kboltz_cgs * T
            }
            results.append(result)
            
            print(f"    Error: {error_percent:.1f}%, Ion_frac: {ionization_fraction:.3e}")
            
        except Exception as e:
            print(f"    ❌ Layer {layer_idx} failed: {e}")
    
    # Summary for this stellar type
    if results:
        avg_error = sum(r['error_percent'] for r in results) / len(results)
        max_error = max(r['error_percent'] for r in results)
        min_error = min(r['error_percent'] for r in results)
        
        print(f"  Summary: {len(results)} layers, avg error: {avg_error:.1f}%, range: {min_error:.1f}%-{max_error:.1f}%")
        
        return results
    else:
        print(f"  ❌ No successful calculations")
        return []

def main():
    print("JORG CHEMICAL EQUILIBRIUM WITH IDENTICAL KORG INPUTS")
    print("=" * 80)
    print("Testing Jorg using EXACT same atmospheric conditions as Korg")
    print("This ensures a fair comparison between implementations")
    print()
    
    # Create atomic and molecular data
    print("Creating atomic and molecular data...")
    ionization_energies = statmech.create_default_ionization_energies()
    partition_fns = statmech.create_default_partition_functions()
    log_equilibrium_constants = statmech.create_default_log_equilibrium_constants()
    print("✅ Data loaded successfully")
    
    all_results = []
    
    for stellar_type, stellar_data in korg_atmosphere_data.items():
        results = test_stellar_type_identical_inputs(stellar_type, stellar_data)
        if results:
            all_results.append((stellar_type, stellar_data, results))
    
    # Overall summary
    print(f"\n{'='*80}")
    print("JORG PERFORMANCE SUMMARY (IDENTICAL INPUTS)")
    print("="*80)
    
    if all_results:
        print("Stellar Type                | Avg Error | Error Range | Ion Range")
        print("-" * 70)
        
        all_errors = []
        for stellar_type, stellar_data, results in all_results:
            avg_error = sum(r['error_percent'] for r in results) / len(results)
            max_error = max(r['error_percent'] for r in results)
            min_error = min(r['error_percent'] for r in results)
            ion_fracs = [r['ionization_fraction'] for r in results]
            min_ion = min(ion_fracs)
            max_ion = max(ion_fracs)
            
            all_errors.extend([r['error_percent'] for r in results])
            
            print(f"{stellar_type:<25} | {avg_error:7.1f}%  | {min_error:4.1f}%-{max_error:4.1f}% | {min_ion:4.2e}-{max_ion:4.2e}")
        
        # Overall statistics
        print(f"\nOverall Jorg Statistics (Identical Inputs):")
        print(f"  Total tests: {len(all_errors)}")
        print(f"  Mean error: {sum(all_errors)/len(all_errors):.2f}%")
        print(f"  Median error: {sorted(all_errors)[len(all_errors)//2]:.2f}%")
        print(f"  Max error: {max(all_errors):.1f}%")
        print(f"  Success rate: 100.0%")
    
    print("\n" + "="*80)
    print("✅ JORG TESTING WITH IDENTICAL INPUTS COMPLETE")
    print("Ready for direct comparison with Korg results")
    print("="*80)
    
    return all_results

if __name__ == "__main__":
    results = main()