#!/usr/bin/env python3
"""
Create McLaughlin H^- cross section data file for Jorg
"""
import numpy as np
import h5py
import json

def export_mclaughlin_data():
    """Export McLaughlin data to JSON for Jorg"""
    with h5py.File('/Users/jdli/Project/Korg.jl/data/McLaughlin2017Hminusbf.h5', 'r') as f:
        nu_data = f['nu'][:]  # Hz
        sigma_data = f['sigma'][:]  # cm^2
    
    # Constants
    h_planck_eV = 4.135667696e-15  # eV⋅s
    H_minus_ion_energy = 0.754204  # eV
    H_minus_ion_nu = H_minus_ion_energy / h_planck_eV  # Hz
    min_interp_nu = np.min(nu_data)
    
    # Calculate low frequency coefficient exactly like Korg
    sigma_at_min = sigma_data[0]  # First value
    low_nu_coef = sigma_at_min / (min_interp_nu - H_minus_ion_nu)**1.5
    
    data = {
        'frequencies_hz': nu_data.tolist(),
        'cross_sections_cm2': sigma_data.tolist(),
        'h_minus_ion_nu_hz': float(H_minus_ion_nu),
        'min_interp_nu_hz': float(min_interp_nu),
        'low_nu_coefficient': float(low_nu_coef),
        'description': 'McLaughlin 2017 H^- bound-free cross sections'
    }
    
    with open('/Users/jdli/Project/Korg.jl/Jorg/jorg/data/mclaughlin_hminus.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported {len(nu_data)} data points")
    print(f"Frequency range: {np.min(nu_data):.3e} to {np.max(nu_data):.3e} Hz")
    print(f"Cross section range: {np.min(sigma_data):.3e} to {np.max(sigma_data):.3e} cm^2")
    print(f"Low frequency coefficient: {low_nu_coef:.3e} cm^2/Hz^1.5")

if __name__ == "__main__":
    export_mclaughlin_data()