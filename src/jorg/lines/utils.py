"""
Utility functions for line absorption calculations
"""

import jax
import jax.numpy as jnp
import numpy as np
from ..constants import SPEED_OF_LIGHT, hplanck_eV, PI


@jax.jit 
def inverse_gaussian_density(rho: float, sigma: float) -> float:
    """
    Calculate the inverse of a (0-centered) Gaussian PDF with standard deviation σ.
    
    Returns the value of x for which ρ = exp(-0.5 x²/σ²) / √(2π), which is 
    given by σ √(-2 log(√(2π)σρ)). Returns 0 when ρ is larger than any value 
    taken on by the PDF.
    
    Parameters
    ----------
    rho : float
        Density value 
    sigma : float
        Standard deviation
        
    Returns
    -------
    float
        Distance x where PDF equals rho
    """
    max_density = 1.0 / (jnp.sqrt(2.0 * PI) * sigma)
    return jnp.where(
        rho > max_density,
        0.0,
        sigma * jnp.sqrt(-2.0 * jnp.log(jnp.sqrt(2.0 * PI) * sigma * rho))
    )


@jax.jit
def inverse_lorentz_density(rho: float, gamma: float) -> float:
    """
    Calculate the inverse of a (0-centered) Lorentz PDF with width γ.
    
    Returns the value of x for which ρ = 1 / (π γ (1 + x²/γ²)), which is 
    given by √(γ/(πρ) - γ²). Returns 0 when ρ is larger than any value 
    taken on by the PDF.
    
    Parameters
    ----------
    rho : float
        Density value
    gamma : float
        Lorentz width parameter
        
    Returns
    -------
    float
        Distance x where PDF equals rho
    """
    max_density = 1.0 / (PI * gamma)
    return jnp.where(
        rho > max_density,
        0.0,
        jnp.sqrt(gamma / (PI * rho) - gamma**2)
    )


@jax.jit
def sigma_line(wl: float) -> float:
    """
    Compute the cross-section divided by gf at a given wavelength.
    
    This implements the quantum mechanical line cross-section formula:
    σ/gf = (π e² / m_e c) * (λ² / 4π)
    
    Parameters
    ----------
    wl : float
        Wavelength in cm
        
    Returns
    -------
    float
        Cross-section divided by gf in cm²
    """
    # Constants for the line cross-section calculation
    # π e² / (m_e c) ≈ 2.654e-2 in cgs units
    prefactor = 2.6540005269103687e-2
    
    return prefactor * wl**2 / (4.0 * PI)


def get_mass_from_species_id(species_id: int) -> float:
    """
    Get atomic mass from species ID.
    
    This is a simplified implementation. In practice, this would use
    a lookup table for all species masses.
    
    Parameters
    ----------
    species_id : int
        Species identifier
        
    Returns
    -------
    float
        Atomic mass in grams
    """
    # Simplified implementation - would need full lookup table
    # Common species masses in amu, converted to grams
    amu_to_grams = 1.66053906660e-24
    
    # Use if-elif chain for JAX compatibility
    if species_id == 1:
        return 1.008 * amu_to_grams      # H I
    elif species_id == 2:
        return 4.003 * amu_to_grams      # He I  
    elif species_id == 6:
        return 12.011 * amu_to_grams     # C I
    elif species_id == 7:
        return 14.007 * amu_to_grams     # N I
    elif species_id == 8:
        return 15.999 * amu_to_grams     # O I
    elif species_id == 11:
        return 22.990 * amu_to_grams     # Na I
    elif species_id == 12:
        return 24.305 * amu_to_grams     # Mg I
    elif species_id == 13:
        return 26.982 * amu_to_grams     # Al I
    elif species_id == 14:
        return 28.085 * amu_to_grams     # Si I
    elif species_id == 16:
        return 32.066 * amu_to_grams     # S I
    elif species_id == 20:
        return 40.078 * amu_to_grams     # Ca I
    elif species_id == 22:
        return 47.867 * amu_to_grams     # Ti I
    elif species_id == 24:
        return 51.996 * amu_to_grams     # Cr I
    elif species_id == 25:
        return 54.938 * amu_to_grams     # Mn I
    elif species_id == 26:
        return 55.845 * amu_to_grams     # Fe I
    elif species_id == 28:
        return 58.693 * amu_to_grams     # Ni I
    else:
        return 55.845 * amu_to_grams     # Default to Fe mass


@jax.jit
def calculate_window_size(doppler_window: float, lorentz_window: float) -> float:
    """
    Calculate the effective window size for line profile calculation.
    
    Parameters
    ----------
    doppler_window : float
        Doppler broadening window size
    lorentz_window : float  
        Lorentz broadening window size
        
    Returns
    -------
    float
        Effective window size
    """
    return jnp.sqrt(lorentz_window**2 + doppler_window**2)