"""
Main continuum absorption calculation combining all opacity sources
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any

from .hydrogen import (
    h_i_bf_absorption,
    h_minus_bf_absorption,
    h_minus_ff_absorption, 
    h2_plus_bf_ff_absorption
)
from .helium import he_minus_ff_absorption
from .scattering import thomson_scattering, rayleigh_scattering
from ..constants import SPEED_OF_LIGHT


def total_continuum_absorption(
    frequencies: jnp.ndarray,
    temperature: float,
    electron_density: float,
    number_densities: Dict[str, float],
    partition_functions: Dict[str, Any],
    include_stimulated_emission: bool = True
) -> jnp.ndarray:
    """
    Calculate total continuum absorption coefficient at given frequencies
    
    This is the JAX equivalent of Korg's total_continuum_absorption function.
    
    Parameters
    ----------
    frequencies : jnp.ndarray
        Frequencies in Hz (must be sorted)
    temperature : float
        Temperature in K
    electron_density : float  
        Electron number density in cm^-3
    number_densities : Dict[str, float]
        Number densities for each species in cm^-3
        Expected keys: 'H_I', 'H_II', 'He_I', 'H2'
    partition_functions : Dict[str, Any]
        Partition functions for each species
        Expected keys: 'H_I', 'He_I'
    include_stimulated_emission : bool, optional
        Whether to include stimulated emission correction (default: True)
        
    Returns
    -------
    jnp.ndarray
        Linear absorption coefficient in cm^-1
        
    Notes
    -----
    For efficiency, frequencies must be sorted (ascending order).
    This function computes all major continuum opacity sources:
    - H I bound-free absorption
    - H^- bound-free and free-free absorption  
    - H2^+ bound-free and free-free absorption
    - He^- free-free absorption
    - Thomson scattering by free electrons
    - Rayleigh scattering by H I, He I, and H2
    """
    
    # Extract commonly used densities
    n_h_i = number_densities.get('H_I', 0.0)
    n_h_ii = number_densities.get('H_II', 0.0) 
    n_he_i = number_densities.get('He_I', 0.0)
    n_h2 = number_densities.get('H2', 0.0)
    
    # Partition function values  
    u_h_i = partition_functions['H_I'](jnp.log(temperature))
    u_he_i = partition_functions['He_I'](jnp.log(temperature))
    
    # Use JIT-compiled version for the main calculation
    alpha_total = _total_continuum_absorption_jit(
        frequencies, temperature, electron_density,
        n_h_i, n_h_ii, n_he_i, n_h2, u_h_i, u_he_i,
        include_stimulated_emission
    )
    
    return alpha_total


@jax.jit
def _total_continuum_absorption_jit(
    frequencies: jnp.ndarray,
    temperature: float,
    electron_density: float,
    n_h_i: float,
    n_h_ii: float,
    n_he_i: float,
    n_h2: float,
    u_h_i: float,
    u_he_i: float,
    include_stimulated_emission: bool
) -> jnp.ndarray:
    """JIT-compiled version of continuum absorption calculation"""
    
    # Initialize total absorption array
    alpha_total = jnp.zeros_like(frequencies, dtype=jnp.float64)
    
    # Number density divided by partition function (commonly used)
    n_h_i_div_u = n_h_i / u_h_i
    n_he_i_div_u = n_he_i / u_he_i
    
    # Hydrogen continuum absorption
    alpha_h_i_bf = h_i_bf_absorption(
        frequencies, temperature, n_h_i, n_he_i, electron_density, 1.0/u_h_i
    )
    alpha_total += alpha_h_i_bf
    
    # H^- bound-free absorption  
    alpha_h_minus_bf = h_minus_bf_absorption(
        frequencies, temperature, n_h_i_div_u, electron_density,
        include_stimulated_emission
    )
    alpha_total += alpha_h_minus_bf
    
    # H^- free-free absorption
    alpha_h_minus_ff = h_minus_ff_absorption(
        frequencies, temperature, n_h_i_div_u, electron_density
    )
    alpha_total += alpha_h_minus_ff
    
    # H2^+ bound-free and free-free absorption
    alpha_h2_plus = h2_plus_bf_ff_absorption(
        frequencies, temperature, n_h_i, n_h_ii, include_stimulated_emission
    )
    alpha_total += alpha_h2_plus
    
    # He^- free-free absorption
    alpha_he_minus_ff = he_minus_ff_absorption(
        frequencies, temperature, n_he_i_div_u, electron_density
    )
    alpha_total += alpha_he_minus_ff
    
    # Thomson scattering by free electrons
    alpha_thomson = thomson_scattering(electron_density)
    alpha_total += alpha_thomson
    
    # Rayleigh scattering
    alpha_rayleigh = rayleigh_scattering(frequencies, n_h_i, n_he_i, n_h2)
    alpha_total += alpha_rayleigh
    
    return alpha_total