#!/usr/bin/env python3
"""
Debug Jorg opacity calculation
"""

import numpy as np
import jax.numpy as jnp
import sys
from pathlib import Path

# Add Jorg to path
sys.path.append(str(Path(__file__).parent / "Jorg"))

from jorg.lines.opacity import calculate_line_opacity, thermal_doppler_width, boltzmann_factor
from jorg.utils.constants import PLANCK_H, BOLTZMANN_K, SPEED_OF_LIGHT, ELEMENTARY_CHARGE, ELECTRON_MASS, PI


def debug_opacity_calculation():
    """Debug each step of the opacity calculation"""
    print("Debugging Jorg opacity calculation step by step...")
    
    # Test parameters
    wavelengths = jnp.linspace(5890, 5900, 100)  # Narrow range around Na D lines
    line_wavelength = 5895.924  # Na D1
    excitation_potential = 2.104  # eV
    log_gf = -0.194
    temperature = 5778.0  # K
    electron_density = 1e14  # cm^-3
    hydrogen_density = 1e16  # cm^-3
    abundance = 1.738e-06  # Linear abundance
    atomic_mass = 23.0  # Na atomic mass
    
    print(f"Line wavelength: {line_wavelength} Å")
    print(f"Excitation potential: {excitation_potential} eV") 
    print(f"log_gf: {log_gf}")
    print(f"Temperature: {temperature} K")
    print(f"Abundance: {abundance:.2e}")
    
    # Step 1: Convert log_gf to linear
    gf = 10**log_gf
    print(f"Linear gf: {gf:.3e}")
    
    # Step 2: Thermal Doppler width
    doppler_width = thermal_doppler_width(line_wavelength, temperature, atomic_mass)
    print(f"Doppler width: {doppler_width:.3e} Å")
    
    # Step 3: Boltzmann factor
    boltzmann = boltzmann_factor(excitation_potential, temperature)
    print(f"Boltzmann factor: {boltzmann:.3e}")
    
    # Step 4: Population factor
    population_factor = abundance * hydrogen_density * boltzmann
    print(f"Population factor: {population_factor:.3e}")
    
    # Step 5: Line strength
    line_strength = (PI * ELEMENTARY_CHARGE**2) / (ELECTRON_MASS * SPEED_OF_LIGHT) * \
                   gf * population_factor / SPEED_OF_LIGHT
    print(f"Line strength: {line_strength:.3e}")
    
    # Step 6: Profile calculation
    delta_lambda = wavelengths - line_wavelength
    x = delta_lambda / doppler_width
    print(f"Delta lambda range: {jnp.min(delta_lambda):.3f} to {jnp.max(delta_lambda):.3f} Å")
    print(f"x range: {jnp.min(x):.3f} to {jnp.max(x):.3f}")
    
    # Simple Gaussian profile for testing
    profile = jnp.exp(-x**2) / (doppler_width * jnp.sqrt(PI))
    print(f"Profile max: {jnp.max(profile):.3e}")
    
    # Step 7: Final opacity
    opacity = line_strength * profile
    print(f"Opacity max: {jnp.max(opacity):.3e}")
    print(f"Opacity at line center: {opacity[jnp.argmin(jnp.abs(delta_lambda))]:.3e}")
    
    return wavelengths, opacity


def test_constants():
    """Test that constants are reasonable"""
    print("\nTesting physical constants:")
    print(f"SPEED_OF_LIGHT: {SPEED_OF_LIGHT:.3e} cm/s")
    print(f"PLANCK_H: {PLANCK_H:.3e} erg·s")
    print(f"BOLTZMANN_K: {BOLTZMANN_K:.3e} erg/K")
    print(f"ELEMENTARY_CHARGE: {ELEMENTARY_CHARGE:.3e} esu")
    print(f"ELECTRON_MASS: {ELECTRON_MASS:.3e} g")
    print(f"PI: {PI:.6f}")


if __name__ == "__main__":
    test_constants()
    debug_opacity_calculation()