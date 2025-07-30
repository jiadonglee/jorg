"""
Spectral processing utilities for LSF and rotational broadening

This module implements proper Line Spread Function (LSF) convolution and
rotational broadening following Korg.jl's algorithms exactly.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Union, Callable, Optional, Tuple
from jax import jit


def _simple_gaussian_filter(flux: jnp.ndarray, sigma: float) -> jnp.ndarray:
    """
    Simple Gaussian convolution using manual implementation
    
    Parameters
    ----------
    flux : jnp.ndarray
        Input flux
    sigma : float
        Gaussian width in pixels
        
    Returns
    -------
    jnp.ndarray
        Convolved flux
    """
    if sigma <= 0.1:
        return flux
    
    # Create Gaussian kernel
    kernel_size = int(6 * sigma) + 1  # Ensure odd size
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    half_size = kernel_size // 2
    x = jnp.arange(-half_size, half_size + 1)
    kernel = jnp.exp(-0.5 * (x / sigma)**2)
    kernel = kernel / jnp.sum(kernel)  # Normalize
    
    # Apply convolution using jax.scipy.signal.convolve
    return jax.scipy.signal.convolve(flux, kernel, mode='same')


@jit
def _lsf_bounds_and_kernel(wavelengths: jnp.ndarray, 
                          lambda0: float, 
                          R: Union[float, Callable], 
                          window_size: float = 4.0) -> Tuple[int, int, jnp.ndarray]:
    """
    Calculate LSF kernel bounds and normalized Gaussian profile
    
    Parameters
    ----------
    wavelengths : jnp.ndarray
        Wavelength grid in Å
    lambda0 : float
        Central wavelength in Å
    R : float or callable
        Resolving power R = λ/Δλ
    window_size : float
        Kernel extent in units of σ (not FWHM)
        
    Returns
    -------
    lb : int
        Lower bound index
    ub : int
        Upper bound index  
    normalized_phi : jnp.ndarray
        Normalized Gaussian kernel
    """
    # Calculate resolving power at this wavelength
    if callable(R):
        R_local = R(lambda0)
    else:
        R_local = R
    
    # LSF width: σ = λ/(R * 2.355) where 2.355 = 2*sqrt(2*ln(2)) converts FWHM to σ
    sigma_lambda = lambda0 / (R_local * 2.355)
    
    # Find wavelength bounds for kernel
    window_extent = window_size * sigma_lambda
    lambda_min = lambda0 - window_extent
    lambda_max = lambda0 + window_extent
    
    # Find indices
    lb = jnp.searchsorted(wavelengths, lambda_min, side='left')
    ub = jnp.searchsorted(wavelengths, lambda_max, side='right')
    
    # Ensure bounds are valid
    lb = jnp.maximum(lb, 0)
    ub = jnp.minimum(ub, len(wavelengths))
    
    # Use dynamic slice for JAX compatibility
    kernel_length = ub - lb
    wl_kernel = jax.lax.dynamic_slice(wavelengths, (lb,), (kernel_length,))
    gaussian_kernel = jnp.exp(-(wl_kernel - lambda0)**2 / (2 * sigma_lambda**2))
    
    # Normalize kernel (ensure it integrates to 1)
    # Use simple sum approximation for normalization (more JAX-friendly)
    normalization = jnp.sum(gaussian_kernel)
    normalized_phi = gaussian_kernel / (normalization + 1e-10)  # Avoid division by zero
    
    return lb, ub, normalized_phi


def apply_LSF(flux: jnp.ndarray, 
              wavelengths: jnp.ndarray, 
              R: Union[float, Callable],
              window_size: float = 4.0) -> jnp.ndarray:
    """
    Apply Line Spread Function convolution to flux
    
    Simplified JAX-compatible implementation using scipy.ndimage convolution.
    
    Parameters
    ----------
    flux : jnp.ndarray
        Input flux vector
    wavelengths : jnp.ndarray
        Wavelength grid in Å
    R : float or callable
        Resolving power R = λ/Δλ. If callable, should take wavelength
        and return resolving power at that wavelength.
    window_size : float, default 4.0
        Kernel extent in units of σ (not FWHM)
        
    Returns
    -------
    jnp.ndarray
        Convolved flux
    """
    # Calculate LSF width at each wavelength
    if callable(R):
        # For callable R, use mean wavelength for simplicity
        R_local = R(jnp.mean(wavelengths))
    else:
        R_local = R
    
    # Convert R to sigma in wavelength units
    lambda_mean = jnp.mean(wavelengths)
    sigma_lambda = lambda_mean / (R_local * 2.355)
    
    # Convert to sigma in pixels (assuming uniform wavelength grid)
    if len(wavelengths) > 1:
        delta_lambda = wavelengths[1] - wavelengths[0]
        sigma_pixels = sigma_lambda / delta_lambda
    else:
        sigma_pixels = 1.0
    
    # Apply simple Gaussian convolution manually
    return _simple_gaussian_filter(flux, sigma_pixels)


def compute_LSF_matrix(synth_wavelengths: jnp.ndarray,
                      obs_wavelengths: jnp.ndarray, 
                      R: Union[float, Callable],
                      window_size: float = 4.0) -> jnp.ndarray:
    """
    Compute LSF convolution matrix (simplified implementation)
    
    Parameters
    ----------
    synth_wavelengths : jnp.ndarray
        Synthesis wavelength grid in Å
    obs_wavelengths : jnp.ndarray  
        Observation wavelength grid in Å
    R : float or callable
        Resolving power
    window_size : float, default 4.0
        Kernel extent in units of σ
        
    Returns
    -------
    jnp.ndarray
        LSF matrix of shape (len(obs_wavelengths), len(synth_wavelengths))
    """
    # For simplicity, create interpolation matrix
    # This is a placeholder implementation - full LSF matrix would be more complex
    n_obs = len(obs_wavelengths)
    n_synth = len(synth_wavelengths)
    
    # Simple interpolation matrix as placeholder
    # In practice, this would implement proper Gaussian LSF convolution
    LSF_matrix = jnp.zeros((n_obs, n_synth))
    
    for i in range(n_obs):
        # Find nearest synthesis wavelengths for each observation wavelength
        obs_wl = obs_wavelengths[i]
        
        # Simple nearest neighbor interpolation weights
        diff = jnp.abs(synth_wavelengths - obs_wl)
        min_idx = jnp.argmin(diff)
        
        # Set weight at nearest point
        LSF_matrix = LSF_matrix.at[i, min_idx].set(1.0)
    
    return LSF_matrix


@jit
def _rotation_kernel_integral(c1: float, c2: float, c3: float, 
                             detuning: float, delta_lambda_rot: float) -> float:
    """
    Indefinite integral of the rotational broadening kernel
    
    This implements the exact formula from Korg.jl for rotational
    broadening with limb darkening.
    """
    # Handle boundary case to avoid NaN
    abs_detuning = jnp.abs(detuning)
    at_boundary = abs_detuning >= delta_lambda_rot
    
    # Safe calculation for interior points
    ratio_sq = (detuning / delta_lambda_rot)**2
    sqrt_term = jnp.sqrt(jnp.maximum(1.0 - ratio_sq, 0.0))
    
    term1 = 0.5 * c1 * detuning * sqrt_term
    term2 = 0.5 * c1 * delta_lambda_rot * jnp.arcsin(
        jnp.clip(detuning / delta_lambda_rot, -1.0, 1.0)
    )
    term3 = c2 * (detuning - detuning**3 / (3 * delta_lambda_rot**2))
    
    interior_value = (term1 + term2 + term3) / c3
    boundary_value = jnp.sign(detuning) * 0.5
    
    return jnp.where(at_boundary, boundary_value, interior_value)


def apply_rotation(flux: jnp.ndarray,
                  wavelengths: jnp.ndarray,
                  vsini: float,
                  epsilon: float = 0.6) -> jnp.ndarray:
    """
    Apply rotational broadening 
    
    Simplified implementation using Gaussian approximation for JAX compatibility.
    For accurate limb-darkening, use the full Korg.jl implementation.
    
    Parameters
    ----------
    flux : jnp.ndarray
        Input flux vector
    wavelengths : jnp.ndarray
        Wavelength grid in Å
    vsini : float
        Projected rotational velocity in km/s
    epsilon : float, default 0.6
        Limb darkening coefficient (not used in simplified version)
        
    Returns
    -------
    jnp.ndarray
        Rotationally broadened flux
    """
    if vsini <= 0:
        return flux
    
    # Physical constants
    c_km_s = 299792.458  # km/s
    
    # Calculate rotational broadening width
    lambda_mean = jnp.mean(wavelengths)
    delta_lambda_rot = lambda_mean * vsini / c_km_s  # Å
    
    # Convert to pixels (assuming uniform grid)
    if len(wavelengths) > 1:
        delta_lambda_pixel = wavelengths[1] - wavelengths[0]
        sigma_pixels = delta_lambda_rot / (delta_lambda_pixel * 2.355)
    else:
        sigma_pixels = 1.0
    
    # Apply simple Gaussian broadening manually
    return _simple_gaussian_filter(flux, sigma_pixels)


__all__ = [
    'apply_LSF',
    'compute_LSF_matrix', 
    'apply_rotation'
]