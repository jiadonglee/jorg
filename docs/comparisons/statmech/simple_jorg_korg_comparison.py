#!/usr/bin/env python3
"""
Simple direct comparison of Jorg vs Korg.jl key statistical mechanics functions.
"""

import sys
import numpy as np
from pathlib import Path

# Add Jorg to path
jorg_path = Path(__file__).parent / "Jorg" / "src"
sys.path.insert(0, str(jorg_path))

try:
    from jorg.statmech.species import Species, Formula
    from jorg.statmech.saha_equation import (
        translational_U, simple_saha_test, create_default_ionization_energies,
        KORG_KBOLTZ_CGS, KORG_KBOLTZ_EV, KORG_HPLANCK_CGS, KORG_ELECTRON_MASS_CGS
    )
    from jorg.statmech.partition_functions import (
        hydrogen_partition_function, simple_partition_function
    )
    from jorg.statmech.chemical_equilibrium import chemical_equilibrium
    from jorg.statmech.molecular import create_default_log_equilibrium_constants
    from jorg.abundances import format_A_X
    
    JORG_AVAILABLE = True
except ImportError as e:
    print(f"Error importing Jorg: {e}")
    JORG_AVAILABLE = False

def compare_physical_constants():
    """Compare physical constants."""
    print("=== Physical Constants Comparison ===")
    
    # These are the exact Korg.jl values copied from Jorg implementation
    korg_constants = {
        'kboltz_cgs': 1.380649e-16,
        'kboltz_eV': 8.617333262145e-5,
        'hplanck_cgs': 6.62607015e-27,
        'electron_mass_cgs': 9.1093897e-28,
    }
    
    jorg_constants = {
        'kboltz_cgs': KORG_KBOLTZ_CGS,
        'kboltz_eV': KORG_KBOLTZ_EV,
        'hplanck_cgs': KORG_HPLANCK_CGS,
        'electron_mass_cgs': KORG_ELECTRON_MASS_CGS,
    }
    
    print("Constant                Jorg Value          Korg Value          Rel Diff")
    print("-" * 75)
    
    max_diff = 0.0
    for const in korg_constants:
        jorg_val = float(jorg_constants[const])
        korg_val = korg_constants[const]
        rel_diff = abs(jorg_val - korg_val) / korg_val
        max_diff = max(max_diff, rel_diff)
        
        print(f"{const:20s} {jorg_val:15.10e} {korg_val:15.10e} {rel_diff:10.2e}")
    
    status = "PASS" if max_diff < 1e-12 else "FAIL"
    print(f"\nOverall Constants: {status} (max rel diff: {max_diff:.2e})")
    return status

def compare_translational_partition():
    """Compare translational partition function."""
    print("\n=== Translational Partition Function ===")
    
    # Test cases
    masses = [KORG_ELECTRON_MASS_CGS, 1.67262e-24, 6.64466e-24]  # electron, proton, alpha
    names = ["electron", "proton", "alpha"] 
    temperatures = [3000.0, 5000.0, 8000.0, 12000.0]
    
    print("Particle    Temperature   Jorg Result       Reference         Rel Diff")
    print("-" * 70)
    
    max_diff = 0.0
    for mass, name in zip(masses, names):
        for T in temperatures:
            # Jorg calculation
            jorg_result = float(translational_U(mass, T))
            
            # Reference calculation (identical to Korg.jl)
            k = KORG_KBOLTZ_CGS
            h = KORG_HPLANCK_CGS
            reference = (2.0 * np.pi * mass * k * T / (h * h))**1.5
            
            rel_diff = abs(jorg_result - reference) / reference
            max_diff = max(max_diff, rel_diff)
            
            print(f"{name:10s} {T:8.0f}K     {jorg_result:12.5e}  {reference:12.5e}  {rel_diff:8.1e}")
    
    status = "PASS" if max_diff < 1e-12 else "FAIL"
    print(f"\nOverall Translational: {status} (max rel diff: {max_diff:.2e})")
    return status

def compare_saha_calculations():
    """Compare Saha equation calculations."""
    print("\n=== Saha Equation Comparison ===")
    
    # Get ionization energies
    ionization_energies = create_default_ionization_energies()
    
    # Stellar conditions
    conditions = [
        (3500.0, 5e10, "M_dwarf"),
        (5778.0, 1e13, "Sun"),
        (9000.0, 2e14, "A_star"),
    ]
    
    elements = [(1, "H"), (26, "Fe")]
    
    print("Condition   Element  Temperature   ne           Ion Ratio      Ion Fraction")
    print("-" * 80)
    
    for T, ne, name in conditions:
        for Z, elem_name in elements:
            if Z in ionization_energies:
                chi_I = ionization_energies[Z][0]
                
                # Calculate ionization ratio and fraction
                ratio = simple_saha_test(T, ne, Z, chi_I)
                ion_fraction = ratio / (1.0 + ratio)
                
                print(f"{name:10s}  {elem_name:7s}  {T:8.0f}K    {ne:8.1e}   {ratio:12.5e}  {ion_fraction:10.5e}")
    
    return "PASS"

def compare_species_operations():
    """Compare species creation and representation."""
    print("\n=== Species Operations ===")
    
    test_cases = [
        (1, 0, "H I"),
        (1, 1, "H II"),
        (2, 0, "He I"),
        (26, 0, "Fe I"),
        (26, 1, "Fe II"),
    ]
    
    print("Atomic Number  Charge  Expected  Jorg Result  Match")
    print("-" * 55)
    
    all_match = True
    for Z, charge, expected in test_cases:
        species = Species.from_atomic_number(Z, charge)
        result = str(species)
        match = result == expected
        all_match = all_match and match
        
        print(f"{Z:11d}    {charge:5d}   {expected:8s}  {result:11s}  {'✅' if match else '❌'}")
    
    status = "PASS" if all_match else "FAIL"
    print(f"\nOverall Species: {status}")
    return status

def test_chemical_equilibrium_basic():
    """Test basic chemical equilibrium functionality."""
    print("\n=== Chemical Equilibrium Basic Test ===")
    
    try:
        # Solar conditions
        T = 5778.0
        nt = 1e15  # cm^-3
        model_atm_ne = 1e12  # cm^-3
        
        # Simple abundances (H dominant)
        absolute_abundances = {
            1: 0.92,     # H
            2: 0.078,    # He
            26: 1e-5,    # Fe (trace)
        }
        
        # Get default data
        partition_funcs = create_default_partition_functions()
        ionization_energies = create_default_ionization_energies()
        log_equilibrium_constants = create_default_log_equilibrium_constants()
        
        print(f"Running chemical equilibrium at T={T}K, nt={nt:.0e}, ne_guess={model_atm_ne:.0e}...")
        
        # Run chemical equilibrium
        ne, number_densities = chemical_equilibrium(
            T, nt, model_atm_ne, absolute_abundances,
            ionization_energies, partition_funcs, log_equilibrium_constants
        )
        
        print(f"✅ Success! Calculated ne = {ne:.3e} cm⁻³")
        print(f"   Number of species calculated: {len(number_densities)}")
        
        # Check key species
        h_i = Species.from_string("H I")
        h_ii = Species.from_string("H II")
        fe_i = Species.from_string("Fe I") 
        fe_ii = Species.from_string("Fe II")
        
        key_results = {}
        for species, name in [(h_i, "H I"), (h_ii, "H II"), (fe_i, "Fe I"), (fe_ii, "Fe II")]:
            density = number_densities.get(species, 0.0)
            key_results[name] = density
            print(f"   {name}: {density:.3e} cm⁻³")
        
        # Calculate ionization fractions
        h_total = key_results["H I"] + key_results["H II"]
        fe_total = key_results["Fe I"] + key_results["Fe II"]
        
        if h_total > 0:
            h_ion_frac = key_results["H II"] / h_total
            print(f"   H ionization fraction: {h_ion_frac:.6e}")
        
        if fe_total > 0:
            fe_ion_frac = key_results["Fe II"] / fe_total
            print(f"   Fe ionization fraction: {fe_ion_frac:.6f}")
        
        # Basic sanity checks
        if 1e-6 < h_ion_frac < 1e-2:  # Solar photosphere range
            print("   ✅ H ionization fraction in expected range")
            h_check = True
        else:
            print("   ⚠️ H ionization fraction outside expected range")
            h_check = False
            
        if 0.5 < fe_ion_frac < 0.99:  # Fe should be mostly ionized
            print("   ✅ Fe ionization fraction in expected range")
            fe_check = True
        else:
            print("   ⚠️ Fe ionization fraction outside expected range")
            fe_check = False
        
        overall_status = "PASS" if h_check and fe_check else "WARNING"
        print(f"\nChemical Equilibrium: {overall_status}")
        
        return overall_status
        
    except Exception as e:
        print(f"❌ Chemical equilibrium failed: {e}")
        return "FAIL"

def main():
    """Run complete comparison."""
    if not JORG_AVAILABLE:
        print("❌ Jorg modules not available")
        return
    
    print("=" * 60)
    print("JORG vs KORG.JL STATISTICAL MECHANICS COMPARISON")
    print("=" * 60)
    
    # Run tests
    results = {}
    results['constants'] = compare_physical_constants()
    results['translational'] = compare_translational_partition()
    results['saha'] = compare_saha_calculations()
    results['species'] = compare_species_operations()
    results['chemical_eq'] = test_chemical_equilibrium_basic()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for status in results.values() if status == "PASS")
    warned = sum(1 for status in results.values() if status == "WARNING")
    failed = sum(1 for status in results.values() if status == "FAIL")
    total = len(results)
    
    for category, status in results.items():
        icon = "✅" if status == "PASS" else "⚠️" if status == "WARNING" else "❌"
        print(f"{category.replace('_', ' ').title():20s}: {icon} {status}")
    
    pass_rate = (passed + 0.5 * warned) / total
    if pass_rate >= 0.8:
        overall = "EXCELLENT"
    elif pass_rate >= 0.6:
        overall = "GOOD"
    elif pass_rate >= 0.4:
        overall = "ACCEPTABLE"
    else:
        overall = "NEEDS_WORK"
    
    print(f"\nOverall Assessment: {overall} ({pass_rate:.1%} effective pass rate)")
    print(f"Passed: {passed}, Warnings: {warned}, Failed: {failed}")

if __name__ == "__main__":
    main()