"""
Mathematical utility functions for stellar spectroscopy

This module contains general mathematical functions used throughout Jorg,
including special functions, interpolation routines, and numerical utilities.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Union, Tuple, Optional

# Export common functions for easy access
__all__ = [
    'voigt_hjerting',
    'inverse_gaussian_density', 
    'inverse_lorentz_density',
    'linear_interpolation',
    'safe_log',
    'safe_exp',
    'safe_sqrt'
]


@jax.jit
def safe_log(x: Union[float, jnp.ndarray], 
             min_val: float = 1e-300) -> Union[float, jnp.ndarray]:
    """
    Safe logarithm that avoids -inf for zero inputs
    
    Parameters
    ----------
    x : float or array
        Input value(s)
    min_val : float
        Minimum value to use instead of zero
        
    Returns
    -------
    float or array
        log(max(x, min_val))
    """
    return jnp.log(jnp.maximum(x, min_val))


@jax.jit
def safe_exp(x: Union[float, jnp.ndarray], 
             max_val: float = 700.0) -> Union[float, jnp.ndarray]:
    """
    Safe exponential that avoids overflow
    
    Parameters
    ----------
    x : float or array
        Input value(s)
    max_val : float
        Maximum exponent value
        
    Returns
    -------
    float or array
        exp(min(x, max_val))
    """
    return jnp.exp(jnp.minimum(x, max_val))


@jax.jit
def safe_sqrt(x: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
    """
    Safe square root that handles negative inputs
    
    Parameters
    ----------
    x : float or array
        Input value(s)
        
    Returns
    -------
    float or array
        sqrt(max(x, 0))
    """
    return jnp.sqrt(jnp.maximum(x, 0.0))


@jax.jit
def linear_interpolation(x: jnp.ndarray, y: jnp.ndarray, 
                        x_new: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
    """
    Linear interpolation using JAX
    
    Parameters
    ----------
    x : array
        Input x coordinates (must be sorted)
    y : array
        Input y coordinates
    x_new : float or array
        Points to interpolate at
        
    Returns
    -------
    float or array
        Interpolated values
    """
    return jnp.interp(x_new, x, y)


# These functions will be moved here from lines/utils.py as they are general mathematical utilities
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
    max_density = 1.0 / (jnp.sqrt(2.0 * jnp.pi) * sigma)
    return jnp.where(
        rho > max_density,
        0.0,
        sigma * jnp.sqrt(-2.0 * safe_log(jnp.sqrt(2.0 * jnp.pi) * sigma * rho))
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
    max_density = 1.0 / (jnp.pi * gamma)
    return jnp.where(
        rho > max_density,
        0.0,
        safe_sqrt(gamma / (jnp.pi * rho) - gamma**2)
    )