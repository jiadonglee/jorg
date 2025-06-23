#!/usr/bin/env python3
"""
Debug script to create accurate H^- cross section interpolation matching Korg exactly
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import jax.numpy as jnp
import jax

# Physical constants
h_planck_eV = 4.135667696e-15  # eV⋅s
h_planck_cgs = 6.6260755e-27   # erg⋅s
c_cgs = 2.99792458e10          # cm/s
kboltz_eV = 8.617333262e-5     # eV/K
H_minus_ion_energy = 0.754204  # eV

def load_mclaughlin_data():
    """Load McLaughlin 2017 H^- cross section data"""
    with h5py.File('/Users/jdli/Project/Korg.jl/data/McLaughlin2017Hminusbf.h5', 'r') as f:
        nu_data = f['nu'][:]  # Hz
        sigma_data = f['sigma'][:]  # cm^2
    return nu_data, sigma_data

def create_korg_accurate_hminus_cross_section():
    """Create H^- cross section function that exactly matches Korg"""
    nu_data, sigma_data = load_mclaughlin_data()
    
    # Create interpolation function
    interp_func = interp1d(nu_data, sigma_data, kind='linear', 
                          bounds_error=False, fill_value=0.0)
    
    # Constants from Korg
    H_minus_ion_nu = H_minus_ion_energy / h_planck_eV  # Hz
    min_interp_nu = np.min(nu_data)
    
    # Calculate the low-frequency coefficient exactly like Korg
    sigma_at_min = interp_func(min_interp_nu)
    low_nu_coef = sigma_at_min / (min_interp_nu - H_minus_ion_nu)**1.5
    
    print(f"H^- ionization frequency: {H_minus_ion_nu:.6e} Hz")
    print(f"Minimum interpolation frequency: {min_interp_nu:.6e} Hz")
    print(f"Cross section at min freq: {sigma_at_min:.6e} cm^2")
    print(f"Low frequency coefficient: {low_nu_coef:.6e} cm^2/Hz^1.5")
    
    def hminus_cross_section(nu):
        """H^- cross section exactly matching Korg implementation"""
        nu = np.asarray(nu)
        result = np.zeros_like(nu)
        
        # Above ionization threshold
        above_threshold = nu > H_minus_ion_nu
        
        # Low frequency region (below interpolation range)
        low_freq = above_threshold & (nu < min_interp_nu)
        if np.any(low_freq):
            result[low_freq] = low_nu_coef * (nu[low_freq] - H_minus_ion_nu)**1.5
        
        # High frequency region (interpolation range)
        high_freq = nu >= min_interp_nu
        if np.any(high_freq):
            result[high_freq] = interp_func(nu[high_freq])
        
        return result
    
    return hminus_cross_section

def test_cross_section_accuracy():
    """Test our cross section against sample frequencies"""
    hminus_cs = create_korg_accurate_hminus_cross_section()
    
    # Test frequencies in optical range (5000-6000 Å)
    wavelengths_angstrom = np.linspace(5000, 6000, 100)
    frequencies = c_cgs * 1e8 / wavelengths_angstrom  # Hz
    
    cross_sections = hminus_cs(frequencies)
    
    # Plot the cross section
    plt.figure(figsize=(10, 6))
    plt.loglog(wavelengths_angstrom, cross_sections * 1e18, 'b-', linewidth=2)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('H^- Cross Section (Mb)')
    plt.title('H^- Bound-Free Cross Section (McLaughlin 2017)')
    plt.grid(True, alpha=0.3)
    plt.savefig('./hminus_cross_section_debug.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Cross section range: {np.min(cross_sections):.3e} to {np.max(cross_sections):.3e} cm^2")
    print(f"Cross section range: {np.min(cross_sections)*1e18:.3f} to {np.max(cross_sections)*1e18:.3f} Mb")

if __name__ == "__main__":
    test_cross_section_accuracy()
    print("Cross section analysis complete!")