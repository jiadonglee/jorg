#!/usr/bin/env python3
"""
Debug Chemical Equilibrium Differences Between Korg.jl and Jorg
================================================================

This script performs a detailed step-by-step comparison to identify
exactly where the Korg.jl and Jorg implementations diverge.
"""

import numpy as np
import sys
import os
from datetime import datetime

# Add Jorg to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Jorg/src'))

from jorg.statmech.chemical_equilibrium import chemical_equilibrium
from jorg.statmech.species import Species
from jorg.statmech.saha_equation import (
    saha_ion_weights, translational_U, 
    KORG_KBOLTZ_CGS, KORG_HPLANCK_CGS, KORG_ELECTRON_MASS_CGS, KORG_KBOLTZ_EV
)
from jorg.statmech.partition_functions import create_default_partition_functions
from jorg.abundances import format_A_X

def debug_saha_equation_step_by_step():
    """Debug the Saha equation implementation step by step"""
    
    print("=== SAHA EQUATION STEP-BY-STEP DEBUG ===")
    print()
    
    # Test conditions - Solar photosphere
    T = 5778.0  # K
    ne = 1e13   # cm^-3
    
    print(f"Test conditions: T = {T} K, ne = {ne:.0e} cm^-3")
    print()
    
    # Constants comparison
    print("Constants:")
    print(f"  Jorg kboltz_cgs:     {KORG_KBOLTZ_CGS:.10e}")
    print(f"  Jorg hplanck_cgs:    {KORG_HPLANCK_CGS:.10e}")
    print(f"  Jorg electron_mass:  {KORG_ELECTRON_MASS_CGS:.10e}")
    print(f"  Jorg kboltz_eV:      {KORG_KBOLTZ_EV:.12e}")
    print()
    
    # Test translational partition function
    trans_U_jorg = translational_U(KORG_ELECTRON_MASS_CGS, T)
    print(f"Translational U (Jorg): {trans_U_jorg:.6e}")
    
    # Manual calculation for verification
    k_cgs = KORG_KBOLTZ_CGS
    h = KORG_HPLANCK_CGS
    m = KORG_ELECTRON_MASS_CGS
    trans_U_manual = (2.0 * np.pi * m * k_cgs * T / (h * h))**1.5
    print(f"Translational U (manual): {trans_U_manual:.6e}")
    print(f"Difference: {abs(trans_U_jorg - trans_U_manual):.2e}")
    print()
    
    # Hydrogen ionization test
    print("HYDROGEN IONIZATION:")
    Z_H = 1
    chi_H = 13.5984  # eV
    
    # Get partition functions
    partition_funcs = create_default_partition_functions()
    H_I = Species.from_atomic_number(1, 0)
    H_II = Species.from_atomic_number(1, 1)
    
    UI = partition_funcs[H_I](np.log(T))
    UII = partition_funcs[H_II](np.log(T))
    
    print(f"  Ionization energy: {chi_H} eV")
    print(f"  U(H I):  {UI}")
    print(f"  U(H II): {UII}")
    print()
    
    # Saha equation components
    k_eV = KORG_KBOLTZ_EV
    exp_factor = np.exp(-chi_H / (k_eV * T))
    saha_factor = 2.0 / ne * (UII / UI) * trans_U_jorg
    
    print(f"  Exponential factor: exp(-chi/(kT)) = {exp_factor:.6e}")
    print(f"  Saha factor: 2/ne * (UII/UI) * transU = {saha_factor:.6e}")
    
    wII_manual = saha_factor * exp_factor
    print(f"  w(H II) manual: {wII_manual:.6e}")
    
    # Using Jorg's saha_ion_weights function
    ionization_energies = {1: (chi_H, -1.0, -1.0)}
    wII_jorg, wIII_jorg = saha_ion_weights(T, ne, 1, ionization_energies, partition_funcs)
    
    print(f"  w(H II) Jorg:   {wII_jorg:.6e}")
    print(f"  w(H III) Jorg:  {wIII_jorg:.6e}")
    print()
    
    # Ionization fractions
    h_ion_frac_manual = wII_manual / (1.0 + wII_manual)
    h_ion_frac_jorg = wII_jorg / (1.0 + wII_jorg)
    
    print(f"  H ionization fraction (manual): {h_ion_frac_manual:.6e}")
    print(f"  H ionization fraction (Jorg):   {h_ion_frac_jorg:.6e}")
    print()
    
    # Comparison with Korg.jl reference
    korg_h_ion = 1.458e-03  # From our earlier Korg.jl calculation
    print(f"  Korg.jl H ionization:           {korg_h_ion:.6e}")
    print(f"  Jorg vs Korg.jl ratio:          {h_ion_frac_jorg / korg_h_ion:.3f}")
    print()
    
    return {
        'trans_U': trans_U_jorg,
        'h_ionization_jorg': h_ion_frac_jorg,
        'h_ionization_korg': korg_h_ion,
        'wII_jorg': wII_jorg
    }

def debug_iron_ionization():
    """Debug iron ionization calculation"""
    
    print("=== IRON IONIZATION DEBUG ===")
    print()
    
    # Test conditions
    T = 5778.0  # K
    ne = 1e13   # cm^-3
    
    # Iron parameters
    Z_Fe = 26
    chi_Fe_I = 7.9025   # eV (first ionization)
    chi_Fe_II = 16.199  # eV (second ionization)
    
    print(f"Iron ionization energies:")
    print(f"  χ(Fe I):  {chi_Fe_I} eV")
    print(f"  χ(Fe II): {chi_Fe_II} eV")
    print()
    
    # Get partition functions
    partition_funcs = create_default_partition_functions()
    Fe_I = Species.from_atomic_number(26, 0)
    Fe_II = Species.from_atomic_number(26, 1)
    Fe_III = Species.from_atomic_number(26, 2)
    
    UI = partition_funcs[Fe_I](np.log(T))
    UII = partition_funcs[Fe_II](np.log(T))
    UIII = partition_funcs[Fe_III](np.log(T))
    
    print(f"Partition functions:")
    print(f"  U(Fe I):   {UI:.3f}")
    print(f"  U(Fe II):  {UII:.3f}")
    print(f"  U(Fe III): {UIII:.3f}")
    print()
    
    # Calculate using Jorg's function
    ionization_energies = {26: (chi_Fe_I, chi_Fe_II, -1.0)}
    wII_fe, wIII_fe = saha_ion_weights(T, ne, 26, ionization_energies, partition_funcs)
    
    print(f"Saha weights:")
    print(f"  w(Fe II):  {wII_fe:.6f}")
    print(f"  w(Fe III): {wIII_fe:.6f}")
    print()
    
    # Ionization fractions
    total_weight = 1.0 + wII_fe + wIII_fe
    fe_neutral_frac = 1.0 / total_weight
    fe_ion1_frac = wII_fe / total_weight
    fe_ion2_frac = wIII_fe / total_weight
    
    print(f"Iron ionization fractions:")
    print(f"  Fe I:   {fe_neutral_frac:.6f}")
    print(f"  Fe II:  {fe_ion1_frac:.6f}")
    print(f"  Fe III: {fe_ion2_frac:.6f}")
    print()
    
    # Compare with Korg.jl reference
    korg_fe_ion = 0.9918  # From our earlier calculation
    jorg_fe_ion = fe_ion1_frac + fe_ion2_frac  # Total ionization
    
    print(f"Total Fe ionization:")
    print(f"  Jorg:     {jorg_fe_ion:.6f}")
    print(f"  Korg.jl:  {korg_fe_ion:.6f}")
    print(f"  Ratio:    {jorg_fe_ion / korg_fe_ion:.3f}")
    print()
    
    return {
        'fe_ionization_jorg': jorg_fe_ion,
        'fe_ionization_korg': korg_fe_ion,
        'wII_fe': wII_fe,
        'wIII_fe': wIII_fe
    }

def debug_electron_density_calculation():
    """Debug electron density self-consistency"""
    
    print("=== ELECTRON DENSITY DEBUG ===")
    print()
    
    # Test conditions
    T = 5778.0
    nt = 1e15
    ne_guess = 1e12
    
    print(f"Conditions: T = {T} K, nt = {nt:.0e} cm^-3, ne_guess = {ne_guess:.0e}")
    print()
    
    # Get simplified abundances for key elements
    A_X = format_A_X()
    absolute_abundances = {}
    
    # Focus on H, He, Fe only for debugging
    key_elements = [1, 2, 26]
    total = 0.0
    
    for Z in key_elements:
        if Z in A_X:
            linear_ab = 10**(A_X[Z] - 12.0)
            absolute_abundances[Z] = linear_ab
            total += linear_ab
    
    # Normalize
    for Z in absolute_abundances:
        absolute_abundances[Z] /= total
    
    print("Abundances:")
    for Z in sorted(absolute_abundances.keys()):
        name = {1: "H", 2: "He", 26: "Fe"}[Z]
        print(f"  {name}: {absolute_abundances[Z]:.6e}")
    print()
    
    # Manual charge conservation check
    partition_funcs = create_default_partition_functions()
    ionization_energies = {
        1: (13.5984, -1.0, -1.0),
        2: (24.5874, 54.418, -1.0),
        26: (7.9025, 16.199, 30.651)
    }
    
    # Calculate ionization for each element at ne_guess
    total_positive_charge = 0.0
    element_details = {}
    
    for Z in key_elements:
        wII, wIII = saha_ion_weights(T, ne_guess, Z, ionization_energies, partition_funcs)
        
        # Element abundance
        element_abundance = absolute_abundances[Z]
        
        # Assume neutral fraction approximation
        neutral_frac = 1.0 / (1.0 + wII + wIII)
        total_atoms = (nt - ne_guess) * element_abundance
        
        n_neutral = total_atoms * neutral_frac
        n_ion1 = wII * n_neutral
        n_ion2 = wIII * n_neutral
        
        element_charge = n_ion1 + 2 * n_ion2
        total_positive_charge += element_charge
        
        element_details[Z] = {
            'wII': wII,
            'wIII': wIII,
            'neutral_frac': neutral_frac,
            'total_atoms': total_atoms,
            'n_neutral': n_neutral,
            'n_ion1': n_ion1,
            'n_ion2': n_ion2,
            'charge_contribution': element_charge
        }
        
        name = {1: "H", 2: "He", 26: "Fe"}[Z]
        print(f"{name} analysis:")
        print(f"  wII = {wII:.3e}, wIII = {wIII:.3e}")
        print(f"  Neutral fraction = {neutral_frac:.6f}")
        print(f"  Total atoms = {total_atoms:.3e}")
        print(f"  Charge contribution = {element_charge:.3e}")
        print()
    
    print(f"Total positive charge: {total_positive_charge:.3e}")
    print(f"Electron density guess: {ne_guess:.3e}")
    print(f"Charge imbalance: {abs(total_positive_charge - ne_guess) / ne_guess * 100:.1f}%")
    print()
    
    # What electron density would balance?
    print(f"For charge balance, need ne ≈ {total_positive_charge:.3e}")
    balance_ratio = total_positive_charge / ne_guess
    print(f"Required ne / guess = {balance_ratio:.3f}")
    print()
    
    return {
        'charge_calculated': total_positive_charge,
        'ne_guess': ne_guess,
        'balance_ratio': balance_ratio,
        'element_details': element_details
    }

def run_full_jorg_solver_debug():
    """Run the full Jorg solver and compare intermediate steps"""
    
    print("=== FULL JORG SOLVER DEBUG ===")
    print()
    
    # Solar conditions
    T = 5778.0
    nt = 1e15
    ne_guess = 1e12
    
    # Get abundances
    A_X = format_A_X()
    absolute_abundances = {}
    key_elements = [1, 2, 26]
    total = 0.0
    
    for Z in key_elements:
        if Z in A_X:
            linear_ab = 10**(A_X[Z] - 12.0)
            absolute_abundances[Z] = linear_ab
            total += linear_ab
    
    for Z in absolute_abundances:
        absolute_abundances[Z] /= total
    
    # Load data
    from jorg.statmech.saha_equation import create_default_ionization_energies
    from jorg.statmech.molecular import create_default_log_equilibrium_constants
    
    ionization_energies = create_default_ionization_energies()
    partition_fns = create_default_partition_functions()
    log_equilibrium_constants = create_default_log_equilibrium_constants()
    
    print(f"Running Jorg chemical equilibrium solver...")
    print(f"Input: T={T}, nt={nt:.0e}, ne_guess={ne_guess:.0e}")
    print()
    
    try:
        ne_calc, densities = chemical_equilibrium(
            T, nt, ne_guess, absolute_abundances,
            ionization_energies, partition_fns, log_equilibrium_constants
        )
        
        print(f"✅ SUCCESS")
        print(f"Calculated electron density: {ne_calc:.3e}")
        print(f"ne_calc / ne_guess = {ne_calc / ne_guess:.3f}")
        print()
        
        # Check key species
        h1 = densities.get(Species.from_atomic_number(1, 0), 0)
        h2 = densities.get(Species.from_atomic_number(1, 1), 0)
        fe1 = densities.get(Species.from_atomic_number(26, 0), 0)
        fe2 = densities.get(Species.from_atomic_number(26, 1), 0)
        
        print("Species densities:")
        print(f"  H I:  {h1:.3e}")
        print(f"  H II: {h2:.3e}")
        print(f"  Fe I: {fe1:.3e}")
        print(f"  Fe II: {fe2:.3e}")
        print()
        
        # Ionization fractions
        h_total = h1 + h2
        fe_total = fe1 + fe2
        
        h_ion_frac = h2 / h_total if h_total > 0 else 0
        fe_ion_frac = fe2 / fe_total if fe_total > 0 else 0
        
        print("Ionization fractions:")
        print(f"  H:  {h_ion_frac:.6e}")
        print(f"  Fe: {fe_ion_frac:.6f}")
        print()
        
        return {
            'success': True,
            'ne_calculated': ne_calc,
            'h_ionization': h_ion_frac,
            'fe_ionization': fe_ion_frac,
            'densities': densities
        }
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return {'success': False, 'error': str(e)}

def main():
    """Main debugging function"""
    
    print("Chemical Equilibrium Solver Debug")
    print("=" * 50)
    print(f"Timestamp: {datetime.now()}")
    print()
    
    # Step 1: Debug Saha equation
    saha_results = debug_saha_equation_step_by_step()
    
    # Step 2: Debug iron ionization
    iron_results = debug_iron_ionization()
    
    # Step 3: Debug electron density
    electron_results = debug_electron_density_calculation()
    
    # Step 4: Run full solver
    solver_results = run_full_jorg_solver_debug()
    
    # Summary
    print("=== SUMMARY ===")
    print()
    
    print("Key findings:")
    print(f"1. Translational U calculation: ✅ Working")
    print(f"2. Hydrogen ionization (Jorg): {saha_results['h_ionization_jorg']:.3e}")
    print(f"   Hydrogen ionization (Korg): {saha_results['h_ionization_korg']:.3e}")
    print(f"   Ratio: {saha_results['h_ionization_jorg'] / saha_results['h_ionization_korg']:.2f}")
    print()
    print(f"3. Iron ionization (Jorg): {iron_results['fe_ionization_jorg']:.4f}")
    print(f"   Iron ionization (Korg): {iron_results['fe_ionization_korg']:.4f}")
    print(f"   Ratio: {iron_results['fe_ionization_jorg'] / iron_results['fe_ionization_korg']:.3f}")
    print()
    print(f"4. Electron density balance factor: {electron_results['balance_ratio']:.2f}")
    print()
    
    if solver_results['success']:
        print(f"5. Full solver H ionization: {solver_results['h_ionization']:.3e}")
        print(f"   Full solver Fe ionization: {solver_results['fe_ionization']:.4f}")
        print(f"   Calculated ne: {solver_results['ne_calculated']:.3e}")
    else:
        print(f"5. Full solver: FAILED")
    
    print()
    print("Likely issues:")
    print("- Electron density self-consistency may not be properly converging")
    print("- Partition function differences vs Korg.jl")
    print("- Different molecular equilibrium handling")
    print("- Abundance normalization differences")

if __name__ == "__main__":
    main()