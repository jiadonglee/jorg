"""
Utility functions for continuum absorption calculations
"""

import jax
import jax.numpy as jnp
from ..constants import c_cgs


@jax.jit
def frequency_to_wavelength(frequency: jnp.ndarray) -> jnp.ndarray:
    """Convert frequency (Hz) to wavelength (cm)"""
    return c_cgs / frequency


@jax.jit  
def wavelength_to_frequency(wavelength: jnp.ndarray) -> jnp.ndarray:
    """Convert wavelength (cm) to frequency (Hz)"""
    return c_cgs / wavelength


@jax.jit
def stimulated_emission_factor(frequency: jnp.ndarray, temperature: float) -> jnp.ndarray:
    """
    Calculate stimulated emission correction factor (1 - exp(-hν/kT))
    
    Parameters
    ----------
    frequency : jnp.ndarray
        Frequencies in Hz
    temperature : float
        Temperature in K
        
    Returns
    -------
    jnp.ndarray
        Stimulated emission correction factor
    """
    from ..constants import hplanck_cgs, kboltz_cgs
    
    x = hplanck_cgs * frequency / (kboltz_cgs * temperature)
    # Use expm1 for numerical stability when x is small
    return -jnp.expm1(-x)


@jax.jit
def planck_function(frequency: jnp.ndarray, temperature: float) -> jnp.ndarray:
    """
    Calculate Planck function B_ν(T)
    
    Parameters
    ----------
    frequency : jnp.ndarray
        Frequencies in Hz
    temperature : float
        Temperature in K
        
    Returns
    -------
    jnp.ndarray
        Planck function in erg s^-1 cm^-2 Hz^-1 sr^-1
    """
    from ..constants import hplanck_cgs, kboltz_cgs, c_cgs
    
    h_nu_kt = hplanck_cgs * frequency / (kboltz_cgs * temperature)
    
    # Planck function: B_ν = (2hν³/c²) / (exp(hν/kT) - 1)
    prefactor = 2.0 * hplanck_cgs * frequency**3 / c_cgs**2
    return prefactor / jnp.expm1(h_nu_kt)