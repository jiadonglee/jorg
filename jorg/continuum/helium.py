"""
Helium continuum absorption implementations in JAX
"""

import jax
import jax.numpy as jnp

from ..constants import kboltz_cgs
from .utils import stimulated_emission_factor


@jax.jit
def he_minus_ff_absorption(
    frequencies: jnp.ndarray,
    temperature: float,
    n_he_i_div_u: float,
    electron_density: float
) -> jnp.ndarray:
    """
    Calculate He^- free-free absorption coefficient
    
    This is a simplified implementation that approximates the
    Helium negative ion free-free absorption. The contribution
    is typically small compared to hydrogen sources.
    
    Parameters
    ----------
    frequencies : jnp.ndarray
        Frequencies in Hz
    temperature : float
        Temperature in K
    n_he_i_div_u : float
        He I number density divided by partition function
    electron_density : float
        Electron density in cm^-3
        
    Returns
    -------
    jnp.ndarray
        He^- free-free absorption coefficient in cm^-1
    """
    # Simplified implementation - He^- ff is generally much smaller than H^- ff
    # This would need detailed calculations or fits to be more accurate
    
    # Ground state He I density (degeneracy = 1, Boltzmann factor = 1)
    n_he_i_ground = 1.0 * n_he_i_div_u
    
    # Electron pressure  
    P_e = electron_density * kboltz_cgs * temperature
    
    # Very rough approximation - scale down from H^- ff by ~100x
    # Real implementation would need proper cross sections
    K_he_approx = 1e-28  # Much smaller than H^- 
    
    return K_he_approx * P_e * n_he_i_ground