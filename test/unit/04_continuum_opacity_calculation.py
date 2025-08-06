#!/usr/bin/env python3
import sys
import numpy as np
sys.path.insert(0, '../../src')

# Setup
from jorg.continuum.exact_physics_continuum import total_continuum_absorption_exact_physics_only  
from jorg.constants import kboltz_cgs, c_cgs, hplanck_cgs

# Physical conditions
T = 5778.0  # K
n_electron = 1e13  # cm⁻³
n_H_I = 1e17  # cm⁻³
wavelengths = np.linspace(5000, 5010, 100)  # Å

# Core test
try:
    total_continuum_opacity = total_continuum_absorption_exact_physics_only(
        wavelengths=wavelengths,
        temperature=T,
        n_H_I=n_H_I,
        n_electron=n_electron,
        layer_number_densities={}
    )
    print(f"Total opacity: {total_continuum_opacity.min():.2e} - {total_continuum_opacity.max():.2e} cm⁻¹")
    assert np.all(total_continuum_opacity >= 0), "Opacity must be positive"
    print("✅ Continuum opacity test passed")
except Exception as e:
    print(f"❌ Continuum opacity test failed: {e}")