"""
Scattering opacity implementations in JAX

This module implements Thomson scattering by free electrons
and Rayleigh scattering by neutral atoms and molecules.
"""

import jax
import jax.numpy as jnp

from ..constants import c_cgs, sigma_thomson


@jax.jit
def thomson_scattering(electron_density: float) -> float:
    """
    Calculate Thomson scattering opacity by free electrons
    
    Thomson scattering is frequency-independent for X-ray to optical wavelengths.
    
    Parameters
    ----------
    electron_density : float
        Electron density in cm^-3
        
    Returns
    -------
    float
        Thomson scattering opacity in cm^-1
    """
    return electron_density * sigma_thomson


@jax.jit
def rayleigh_scattering(
    frequencies: jnp.ndarray,
    n_h_i: float,
    n_he_i: float,
    n_h2: float
) -> jnp.ndarray:
    """
    Calculate Rayleigh scattering opacity by neutral atoms and molecules
    
    Rayleigh scattering has a ν^4 frequency dependence and is important
    at short wavelengths (UV).
    
    Parameters
    ----------
    frequencies : jnp.ndarray
        Frequencies in Hz
    n_h_i : float
        H I number density in cm^-3
    n_he_i : float
        He I number density in cm^-3
    n_h2 : float
        H2 number density in cm^-3
        
    Returns
    -------
    jnp.ndarray
        Rayleigh scattering opacity in cm^-1
    """
    # Convert frequency to wavelength in cm
    wavelength = c_cgs / frequencies
    
    # Rayleigh scattering cross sections (approximate values)
    # These are rough approximations - real implementation needs proper formulas
    
    # H I Rayleigh scattering cross section
    # σ_H ≈ 4.5e-26 * (λ₀/λ)^4 cm^2, where λ₀ = 1215 Å (Lyman alpha)
    lambda_0_h = 1215e-8  # cm
    sigma_h_rayleigh = 4.5e-26 * (lambda_0_h / wavelength)**4
    
    # He I Rayleigh scattering cross section (smaller than H I)
    lambda_0_he = 584e-8  # cm (He I resonance line)
    sigma_he_rayleigh = 1.0e-26 * (lambda_0_he / wavelength)**4
    
    # H2 Rayleigh scattering cross section
    # Molecular cross sections are typically larger
    lambda_0_h2 = 1000e-8  # cm (approximate)
    sigma_h2_rayleigh = 8.0e-26 * (lambda_0_h2 / wavelength)**4
    
    # Total Rayleigh scattering opacity
    alpha_rayleigh = (
        n_h_i * sigma_h_rayleigh +
        n_he_i * sigma_he_rayleigh +
        n_h2 * sigma_h2_rayleigh
    )
    
    return alpha_rayleigh