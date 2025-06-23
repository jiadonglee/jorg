#!/usr/bin/env python3
"""
Test H I bound-free with different n ranges to find contributing levels
"""
import numpy as np
import sys
import os

# Add Jorg to path
sys.path.insert(0, '/Users/jdli/Project/Korg.jl/Jorg')

from jorg.constants import hplanck_eV, kboltz_eV, bf_sigma_const

def test_n_range():
    """Test which n levels contribute to H I bound-free at 5500 Å"""
    
    print("=== H I BOUND-FREE N-LEVEL CONTRIBUTION TEST ===")
    
    # Test conditions
    wavelength = 5500.0  # Å
    c_cgs = 2.99792458e10
    frequency = c_cgs * 1e8 / wavelength  # Hz
    T = 5778.0  # K
    chi_h = 13.598434005136  # eV
    
    print(f"Test frequency: {frequency:.3e} Hz ({wavelength} Å)")
    print(f"Temperature: {T} K")
    print()
    
    # Test n from 1 to 100
    print("Finding contributing n levels:")
    print("n    nu_threshold (Hz)    Above thresh?    Cross section (cm^2)    Alpha contrib (cm^-1)")
    print("-" * 85)
    
    total_alpha = 0.0
    nH_I = 1e16  # cm^-3
    U_H_I = 2.0
    inv_U_H = 1.0 / U_H_I
    
    # Stimulated emission factor
    photon_energy = hplanck_eV * frequency
    stim_emission = 1.0 - np.exp(-photon_energy / (kboltz_eV * T))
    
    for n in range(1, 101):
        inv_n = 1.0 / n
        inv_n2 = inv_n * inv_n
        
        # Threshold frequency
        nu_threshold = chi_h * inv_n2 / hplanck_eV
        above_threshold = frequency > nu_threshold
        
        if above_threshold:
            # Cross section
            cross_section = bf_sigma_const * (inv_n2**2 * inv_n) * (1.0/frequency)**3 * 1e-18
            
            # Occupation probability
            degeneracy = 2 * n**2
            excitation_energy = chi_h * (1.0 - inv_n2)
            boltzmann_factor = np.exp(-excitation_energy / (kboltz_eV * T))
            occupation_prob = degeneracy * boltzmann_factor
            
            # Alpha contribution
            alpha_contrib = nH_I * inv_U_H * occupation_prob * cross_section * stim_emission
            total_alpha += alpha_contrib
            
            print(f"{n:2d}   {nu_threshold:.3e}        Yes           {cross_section:.3e}        {alpha_contrib:.3e}")
            
            if n <= 20:  # Show first 20 contributing levels
                continue
            elif alpha_contrib / total_alpha < 0.01:  # Stop when contribution < 1%
                print(f"... (stopping at n={n}, contribution < 1% of total)")
                break
        else:
            if n <= 10:  # Show first 10 non-contributing levels
                print(f"{n:2d}   {nu_threshold:.3e}        No            0.000e+00               0.000e+00")
    
    print("-" * 85)
    print(f"Total H I bound-free alpha: {total_alpha:.3e} cm^-1")
    
    # Expected from typical stellar models
    expected = 1e-8  # Very rough estimate
    print(f"Expected magnitude: ~{expected:.0e} cm^-1")
    print(f"Ratio actual/expected: {total_alpha/expected:.1f}")
    
    return total_alpha

if __name__ == "__main__":
    test_n_range()