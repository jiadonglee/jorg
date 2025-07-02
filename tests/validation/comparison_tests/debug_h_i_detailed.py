#!/usr/bin/env python3
"""
Detailed step-by-step debug of H I bound-free calculation
"""
import numpy as np
import sys
import os

# Add Jorg to path
sys.path.insert(0, '/Users/jdli/Project/Korg.jl/Jorg')

from jorg.constants import RydbergH_eV, hplanck_eV, kboltz_eV, bf_sigma_const
import jax.numpy as jnp

def debug_h_i_detailed():
    """Debug H I bound-free step by step"""
    
    print("=== DETAILED H I BOUND-FREE DEBUG ===")
    
    # Test with single frequency and quantum number
    wavelength = 5500.0  # Å
    c_cgs = 2.99792458e10
    frequency = c_cgs * 1e8 / wavelength  # Hz
    
    n = 2  # First excited state (Balmer series)
    T = 5778.0  # K
    
    print(f"Test conditions:")
    print(f"  Wavelength: {wavelength} Å")
    print(f"  Frequency: {frequency:.3e} Hz")
    print(f"  Quantum number n: {n}")
    print(f"  Temperature: {T} K")
    print()
    
    # Step 1: Calculate ionization threshold
    print("Step 1: Ionization threshold")
    chi_h = 13.598434005136  # H I ionization energy from ground state (eV)
    ionization_energy_n = chi_h * (1.0 - 1.0/n**2)  # Energy to ionize from level n
    nu_threshold = ionization_energy_n / hplanck_eV  # Threshold frequency
    
    print(f"  H I ionization energy (ground): {chi_h} eV")
    print(f"  Ionization energy from n={n}: {ionization_energy_n:.6f} eV")
    print(f"  Threshold frequency: {nu_threshold:.3e} Hz")
    print(f"  Test frequency: {frequency:.3e} Hz")
    print(f"  Above threshold: {frequency > nu_threshold}")
    print()
    
    # Step 2: Calculate cross section
    print("Step 2: Cross section calculation")
    inv_n = 1.0 / n
    inv_n2 = inv_n * inv_n
    inv_n5 = inv_n2 * inv_n2 * inv_n
    inv_freq3 = (1.0 / frequency)**3
    
    print(f"  1/n: {inv_n}")
    print(f"  1/n^2: {inv_n2:.6f}")
    print(f"  1/n^5: {inv_n5:.6e}")
    print(f"  1/ν^3: {inv_freq3:.6e}")
    print(f"  bf_sigma_const: {bf_sigma_const:.3e}")
    
    # Cross section in megabarns
    cross_section_mb = bf_sigma_const * inv_n5 * inv_freq3
    cross_section_cm2 = cross_section_mb * 1e-18  # Convert to cm^2
    
    print(f"  Cross section (Mb): {cross_section_mb:.3e}")
    print(f"  Cross section (cm^2): {cross_section_cm2:.3e}")
    print()
    
    # Step 3: Occupation probability
    print("Step 3: Occupation probability")
    degeneracy = 2 * n**2
    excitation_energy = chi_h * (1.0 - 1.0/n**2)
    boltzmann_factor = np.exp(-excitation_energy / (kboltz_eV * T))
    occupation_prob = degeneracy * boltzmann_factor
    
    print(f"  Degeneracy (2n^2): {degeneracy}")
    print(f"  Excitation energy: {excitation_energy:.6f} eV")
    print(f"  kT: {kboltz_eV * T:.6f} eV")
    print(f"  Boltzmann factor: {boltzmann_factor:.6e}")
    print(f"  Occupation probability: {occupation_prob:.6e}")
    print()
    
    # Step 4: Final absorption coefficient
    print("Step 4: Absorption coefficient")
    nH_I = 1e16  # cm^-3
    U_H_I = 2.0
    inv_U_H = 1.0 / U_H_I
    
    # Stimulated emission
    photon_energy = hplanck_eV * frequency
    stim_emission = 1.0 - np.exp(-photon_energy / (kboltz_eV * T))
    
    alpha_n = nH_I * inv_U_H * occupation_prob * cross_section_cm2 * stim_emission
    
    print(f"  n(H I): {nH_I:.3e} cm^-3")
    print(f"  1/U(H I): {inv_U_H}")
    print(f"  Photon energy: {photon_energy:.6f} eV")
    print(f"  Stimulated emission factor: {stim_emission:.6f}")
    print(f"  Alpha for n={n}: {alpha_n:.3e} cm^-1")
    print()
    
    # Test vectorized calculation
    print("Step 5: Vectorized calculation test")
    frequencies_test = np.array([frequency])
    
    # Manual implementation of the Jorg function logic
    n_levels = np.arange(1, 7)  # Test first 6 levels
    total_alpha = 0.0
    
    for n_test in n_levels:
        inv_n_test = 1.0 / n_test
        inv_n2_test = inv_n_test * inv_n_test
        
        # Threshold (corrected to match Korg)
        nu_thresh_test = chi_h * inv_n2_test / hplanck_eV
        above_thresh = frequency > nu_thresh_test
        
        if above_thresh:
            # Cross section
            cross_sec = bf_sigma_const * (inv_n2_test**2 * inv_n_test) * (1.0/frequency)**3 * 1e-18
            
            # Occupation
            deg = 2 * n_test**2
            exc_energy = chi_h * (1.0 - inv_n2_test)
            boltz = np.exp(-exc_energy / (kboltz_eV * T))
            occ = deg * boltz
            
            # Contribution
            alpha_contrib = nH_I * inv_U_H * occ * cross_sec * stim_emission
            total_alpha += alpha_contrib
            
            print(f"  n={n_test}: nu_thresh={nu_thresh_test:.3e}, cross_sec={cross_sec:.3e}, alpha={alpha_contrib:.3e}")
        else:
            print(f"  n={n_test}: below threshold (nu_thresh={nu_thresh_test:.3e})")
    
    print(f"  Total alpha (sum): {total_alpha:.3e} cm^-1")
    
    # Compare with expected magnitude
    expected = 1e-8  # Very rough estimate
    print(f"  Expected magnitude: ~{expected:.0e} cm^-1")
    print(f"  Ratio: {total_alpha/expected:.1f}")

if __name__ == "__main__":
    debug_h_i_detailed()