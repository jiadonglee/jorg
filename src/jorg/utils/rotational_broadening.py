"""
Rotational broadening for stellar spectra.

This module implements rotational broadening following Gray (2005) and
matching Korg.jl's implementation for consistency.
"""

import numpy as np
import jax.numpy as jnp
from jax import jit
from scipy import special
from typing import Union, Tuple


def rotational_kernel(delta_lambda: float, lambda0: float, vsini: float, 
                      limb_darkening: float = 0.6) -> float:
    """
    Calculate the rotational broadening kernel at a given wavelength offset.
    
    Based on Gray (2005) "The Observation and Analysis of Stellar Photospheres"
    equation 18.14.
    
    Parameters
    ----------
    delta_lambda : float
        Wavelength offset from line center in Angstroms
    lambda0 : float
        Central wavelength in Angstroms
    vsini : float
        Projected rotational velocity in km/s
    limb_darkening : float
        Linear limb darkening coefficient (default 0.6 for solar-type stars)
        
    Returns
    -------
    float
        Rotational kernel value (normalized)
    """
    if vsini <= 0:
        # No rotation - return delta function
        return 1.0 if abs(delta_lambda) < 1e-10 else 0.0
    
    # Convert vsini to wavelength units
    c = 299792.458  # Speed of light in km/s
    delta_lambda_max = lambda0 * vsini / c
    
    # Normalized wavelength shift
    x = delta_lambda / delta_lambda_max
    
    if abs(x) >= 1.0:
        return 0.0
    
    # Rotational profile from Gray (2005)
    # G(x) = (2(1-ε)/π)(1-x²)^0.5 + (ε/2)(1-x²)
    # where ε is the limb darkening coefficient
    
    sqrt_term = np.sqrt(1.0 - x**2)
    
    # Two components: continuum and limb darkened
    continuum_term = (2.0 * (1.0 - limb_darkening) / np.pi) * sqrt_term
    limb_term = (limb_darkening / 2.0) * (1.0 - x**2)
    
    kernel = continuum_term + limb_term
    
    # Normalize
    kernel /= delta_lambda_max
    
    return kernel


def apply_rotational_broadening(wavelengths: np.ndarray, flux: np.ndarray,
                               vsini: float, limb_darkening: float = 0.6) -> np.ndarray:
    """
    Apply rotational broadening to a spectrum.
    
    Parameters
    ----------
    wavelengths : array
        Wavelength array in Angstroms
    flux : array
        Flux array (normalized or absolute)
    vsini : float
        Projected rotational velocity in km/s
    limb_darkening : float
        Linear limb darkening coefficient (default 0.6)
        
    Returns
    -------
    array
        Rotationally broadened flux
    """
    if vsini <= 0:
        return flux
    
    # Speed of light in km/s
    c = 299792.458
    
    # Create output array
    broadened_flux = np.zeros_like(flux)
    
    # Wavelength spacing (assumed uniform)
    dlambda = wavelengths[1] - wavelengths[0]
    
    # For each wavelength point
    for i, lambda0 in enumerate(wavelengths):
        # Maximum wavelength shift due to rotation
        delta_lambda_max = lambda0 * vsini / c
        
        # Range to consider (3x the maximum shift)
        n_points = int(3 * delta_lambda_max / dlambda) + 1
        
        # Accumulate convolution
        total_kernel = 0.0
        
        for j in range(max(0, i - n_points), min(len(wavelengths), i + n_points + 1)):
            # Wavelength difference
            delta_lambda = wavelengths[j] - lambda0
            
            # Calculate kernel value
            kernel_val = rotational_kernel(delta_lambda, lambda0, vsini, limb_darkening)
            
            # Add contribution
            broadened_flux[i] += flux[j] * kernel_val * dlambda
            total_kernel += kernel_val * dlambda
        
        # Normalize
        if total_kernel > 0:
            broadened_flux[i] /= total_kernel
        else:
            broadened_flux[i] = flux[i]
    
    return broadened_flux


@jit
def rotational_kernel_jax(x: float, limb_darkening: float = 0.6) -> float:
    """
    JAX-compatible rotational broadening kernel.
    
    Parameters
    ----------
    x : float
        Normalized wavelength shift (-1 < x < 1)
    limb_darkening : float
        Linear limb darkening coefficient
        
    Returns
    -------
    float
        Kernel value
    """
    # Ensure x is in valid range
    x = jnp.clip(x, -0.9999, 0.9999)
    
    # Rotational profile
    sqrt_term = jnp.sqrt(1.0 - x**2)
    continuum_term = (2.0 * (1.0 - limb_darkening) / jnp.pi) * sqrt_term
    limb_term = (limb_darkening / 2.0) * (1.0 - x**2)
    
    return continuum_term + limb_term


def get_rotation_kernel_array(vsini: float, wavelength: float, 
                             wavelength_grid: np.ndarray,
                             limb_darkening: float = 0.6) -> np.ndarray:
    """
    Get the rotational broadening kernel as an array for convolution.
    
    Parameters
    ----------
    vsini : float
        Projected rotational velocity in km/s
    wavelength : float
        Central wavelength in Angstroms
    wavelength_grid : array
        Wavelength grid for kernel calculation
    limb_darkening : float
        Linear limb darkening coefficient
        
    Returns
    -------
    array
        Normalized rotational kernel
    """
    if vsini <= 0:
        # No rotation - return delta function
        kernel = np.zeros_like(wavelength_grid)
        idx = np.argmin(np.abs(wavelength_grid - wavelength))
        kernel[idx] = 1.0
        return kernel
    
    c = 299792.458  # km/s
    delta_lambda_max = wavelength * vsini / c
    
    # Calculate kernel
    kernel = np.zeros_like(wavelength_grid)
    for i, wl in enumerate(wavelength_grid):
        delta_lambda = wl - wavelength
        kernel[i] = rotational_kernel(delta_lambda, wavelength, vsini, limb_darkening)
    
    # Normalize
    kernel_sum = np.trapz(kernel, wavelength_grid)
    if kernel_sum > 0:
        kernel /= kernel_sum
    
    return kernel


def estimate_vsini_from_line_width(line_fwhm: float, wavelength: float,
                                   thermal_fwhm: float = 0.0) -> float:
    """
    Estimate v sin i from observed line width.
    
    This is a rough estimate assuming the line width is dominated by rotation.
    
    Parameters
    ----------
    line_fwhm : float
        Observed line FWHM in Angstroms
    wavelength : float
        Line wavelength in Angstroms
    thermal_fwhm : float
        Thermal/microturbulent FWHM in Angstroms (to subtract)
        
    Returns
    -------
    float
        Estimated v sin i in km/s
    """
    c = 299792.458  # km/s
    
    # Subtract thermal broadening in quadrature
    if thermal_fwhm > 0 and thermal_fwhm < line_fwhm:
        rotation_fwhm = np.sqrt(line_fwhm**2 - thermal_fwhm**2)
    else:
        rotation_fwhm = line_fwhm
    
    # For rotational broadening, FWHM ≈ 1.7 * Δλ_max
    # where Δλ_max = λ * v sin i / c
    vsini = (rotation_fwhm / 1.7) * c / wavelength
    
    return vsini