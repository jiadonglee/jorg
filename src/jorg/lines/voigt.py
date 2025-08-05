"""
Voigt profile calculations for spectral line synthesis

This module provides the main voigt_profile function used by the line opacity
calculation system. It wraps the more complete implementation in profiles.py
with a simplified interface for compatibility with the test suite.
"""

from .profiles import voigt_profile as _voigt_profile_full
from .profiles import voigt_hjerting, line_profile
import jax.numpy as jnp


def voigt_profile(x_values, a_voigt):
    """
    Compute normalized Voigt profile values at given dimensionless frequency offsets.
    
    This function provides the interface expected by the line opacity test script.
    It computes the Voigt profile H(a, v) using the Hjerting function.
    
    Parameters
    ----------
    x_values : array_like
        Dimensionless frequency offsets from line center: x = (λ - λ₀) / Δλ_D
        where Δλ_D is the Doppler width
    a_voigt : float
        Voigt damping parameter: a = Δλ_L / Δλ_D
        where Δλ_L is the Lorentzian width and Δλ_D is the Doppler width
        
    Returns
    -------
    jnp.ndarray
        Normalized Voigt profile values H(a, v) / sqrt(π)
        
    Notes
    -----
    The Voigt profile is the convolution of Gaussian (Doppler) and Lorentzian 
    (pressure + natural) broadening mechanisms. This function uses the Hjerting
    function implementation that exactly matches Korg.jl.
    
    The normalization follows the standard definition where the integral over
    the entire profile equals sqrt(π).
    
    Examples
    --------
    >>> x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> a = 0.1  # Small damping (Doppler-dominated)
    >>> profile = voigt_profile(x, a)
    >>> print(profile.shape)
    (5,)
    """
    x_values = jnp.asarray(x_values)
    
    if jnp.ndim(x_values) == 0:
        # Scalar case
        return voigt_hjerting(a_voigt, jnp.abs(x_values))
    else:
        # Array case - vectorize over x_values
        return jnp.array([voigt_hjerting(a_voigt, jnp.abs(x)) for x in x_values])


def voigt_profile_wavelength(wavelengths, line_center, doppler_width, lorentz_width, amplitude=1.0):
    """
    Compute Voigt profile at specific wavelengths.
    
    This is the full wavelength-based interface that wraps the complete
    line_profile function from profiles.py.
    
    Parameters
    ----------
    wavelengths : array_like
        Wavelengths at which to evaluate the profile (any units)
    line_center : float
        Central wavelength of the line (same units as wavelengths) 
    doppler_width : float
        Doppler broadening width (standard deviation, same units)
    lorentz_width : float
        Lorentzian broadening HWHM (same units)
    amplitude : float, optional
        Total line strength/amplitude (default: 1.0)
        
    Returns
    -------
    jnp.ndarray
        Voigt profile values at the specified wavelengths
    """
    return line_profile(line_center, doppler_width, lorentz_width, amplitude, wavelengths)


# Export the main functions
__all__ = ['voigt_profile', 'voigt_profile_wavelength', 'voigt_hjerting']