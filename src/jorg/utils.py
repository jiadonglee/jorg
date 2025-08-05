"""
Utility functions for stellar spectral synthesis post-processing

This module implements proper Line Spread Function (LSF) convolution and
rotational broadening following Korg.jl's algorithms exactly.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Union, Callable, Optional, Tuple
from jax import jit


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
    
    # Calculate Gaussian kernel
    wl_kernel = wavelengths[lb:ub]
    gaussian_kernel = jnp.exp(-(wl_kernel - lambda0)**2 / (2 * sigma_lambda**2))
    
    # Normalize kernel (ensure it integrates to 1)
    normalization = jnp.trapz(gaussian_kernel, wl_kernel)
    normalized_phi = gaussian_kernel / normalization
    
    return lb, ub, normalized_phi


def apply_LSF(flux: jnp.ndarray, 
              wavelengths: jnp.ndarray, 
              R: Union[float, Callable],
              window_size: float = 4.0) -> jnp.ndarray:
    """
    Apply Line Spread Function convolution to flux
    
    This function implements the exact algorithm from Korg.jl's apply_LSF
    function, convolving the input flux with a Gaussian LSF.
    
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
        
    Notes
    -----
    The LSF is modeled as a Gaussian with σ = λ/(R * 2.355).
    This function is intended to be run on a fine wavelength grid,
    then optionally downsampled to observational resolution.
    """
    convolved_flux = jnp.zeros_like(flux)
    
    def convolve_point(i, cf):
        lambda0 = wavelengths[i]
        lb, ub, normalized_phi = _lsf_bounds_and_kernel(
            wavelengths, lambda0, R, window_size
        )
        
        # Handle case where kernel extends beyond array bounds
        flux_segment = flux[lb:ub]
        
        # Ensure shapes match
        min_len = min(len(flux_segment), len(normalized_phi))
        flux_segment = flux_segment[:min_len]
        normalized_phi = normalized_phi[:min_len]
        
        convolved_value = jnp.sum(flux_segment * normalized_phi)
        return cf.at[i].set(convolved_value)
    
    # Use scan for efficient loop
    convolved_flux = jax.lax.fori_loop(
        0, len(wavelengths),
        lambda i, cf: convolve_point(i, cf),
        convolved_flux
    )
    
    return convolved_flux


def compute_LSF_matrix(synth_wavelengths: jnp.ndarray,
                      obs_wavelengths: jnp.ndarray, 
                      R: Union[float, Callable],
                      window_size: float = 4.0) -> jnp.ndarray:
    """
    Compute LSF convolution matrix for efficient repeated convolutions
    
    This creates a sparse matrix that, when multiplied with a flux vector,
    applies Gaussian LSF convolution and resamples to observation wavelengths.
    
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
        
    Notes
    -----
    This function is slow to compute but enables fast convolution via
    matrix multiplication: conv_flux = LSF_matrix @ synth_flux
    """
    n_obs = len(obs_wavelengths)
    n_synth = len(synth_wavelengths)
    
    # Initialize sparse matrix representation
    LSF_matrix = jnp.zeros((n_obs, n_synth))
    
    def fill_row(i, matrix):
        lambda0 = obs_wavelengths[i]
        lb, ub, normalized_phi = _lsf_bounds_and_kernel(
            synth_wavelengths, lambda0, R, window_size
        )
        
        # Ensure bounds are valid
        lb = jnp.maximum(lb, 0)
        ub = jnp.minimum(ub, n_synth)
        
        # Fill matrix row
        kernel_length = ub - lb
        phi_trimmed = normalized_phi[:kernel_length]
        
        return matrix.at[i, lb:ub].set(phi_trimmed)
    
    # Fill matrix row by row
    LSF_matrix = jax.lax.fori_loop(
        0, n_obs, fill_row, LSF_matrix
    )
    
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
    Apply rotational broadening with limb darkening
    
    This implements the exact rotational broadening algorithm from Korg.jl,
    including proper limb darkening treatment.
    
    Parameters
    ----------
    flux : jnp.ndarray
        Input flux vector
    wavelengths : jnp.ndarray
        Wavelength grid in Å
    vsini : float
        Projected rotational velocity in km/s
    epsilon : float, default 0.6
        Limb darkening coefficient
        
    Returns
    -------
    jnp.ndarray
        Rotationally broadened flux
        
    Notes
    -----
    Uses the Gray (2005) rotational broadening kernel with limb darkening.
    The kernel accounts for the projected velocity distribution across
    the stellar disk.
    """
    if vsini <= 0:
        return flux
    
    # Physical constants
    c_cgs = 2.99792458e10  # cm/s
    
    # Limb darkening coefficients  
    c1 = 2 * (1 - epsilon) / (jnp.pi * (1 - epsilon/3))
    c2 = 0.5 * epsilon / (1 - epsilon/3)
    c3 = jnp.pi * (1 - epsilon/3)
    
    # Initialize output
    broadened_flux = jnp.zeros_like(flux)
    
    def broaden_point(i, bf):
        lambda0 = wavelengths[i]
        
        # Rotational wavelength shift
        delta_lambda_rot = lambda0 * vsini * 1e5 / c_cgs  # Convert km/s to cm/s
        
        # Find wavelength window
        lambda_min = lambda0 - delta_lambda_rot
        lambda_max = lambda0 + delta_lambda_rot
        
        lb = jnp.searchsorted(wavelengths, lambda_min, side='left')
        ub = jnp.searchsorted(wavelengths, lambda_max, side='right')
        
        # Ensure valid bounds
        lb = jnp.maximum(lb, 0)
        ub = jnp.minimum(ub, len(wavelengths))
        
        # Get flux window
        flux_window = flux[lb:ub]
        
        # Calculate wavelength step (assume uniform grid)
        if len(wavelengths) > 1:
            wl_step = wavelengths[1] - wavelengths[0]
        else:
            wl_step = 1.0
        
        # Calculate detunings for kernel integration
        indices = jnp.arange(lb, ub)
        detunings_start = (indices - i - 0.5) * wl_step
        detunings_end = (indices - i + 0.5) * wl_step
        
        # Add boundary points
        detunings_all = jnp.concatenate([
            jnp.array([-delta_lambda_rot]),
            detunings_start,
            jnp.array([delta_lambda_rot])
        ])
        
        # Calculate kernel integrals
        kernel_integrals = jax.vmap(
            lambda d: _rotation_kernel_integral(c1, c2, c3, d, delta_lambda_rot)
        )(detunings_all)
        
        # Apply finite difference to get kernel values
        kernel_values = kernel_integrals[1:] - kernel_integrals[:-1]
        
        # Convolve with flux
        broadened_value = jnp.sum(
            flux_window * kernel_values[:len(flux_window)]
        )
        
        return bf.at[i].set(broadened_value)
    
    # Apply broadening to each wavelength point
    broadened_flux = jax.lax.fori_loop(
        0, len(wavelengths), broaden_point, broadened_flux
    )
    
    return broadened_flux


def air_to_vacuum(wavelength: jnp.ndarray, cgs: bool = None) -> jnp.ndarray:
    """
    Convert wavelengths from air to vacuum
    
    Uses the Birch and Downs (1994) formula as implemented in Korg.jl.
    
    Parameters
    ----------
    wavelength : jnp.ndarray
        Wavelengths in Å (if ≥ 1) or cm (if < 1)
    cgs : bool, optional
        If True, wavelengths are in cm. If None, auto-detect based on magnitude.
        
    Returns
    -------
    jnp.ndarray
        Vacuum wavelengths in same units as input
    """
    if cgs is None:
        cgs = jnp.all(wavelength < 1)
    
    wl = wavelength.copy()
    if cgs:
        wl *= 1e8  # Convert cm to Å
    
    # Birch and Downs (1994) formula
    s = 1e4 / wl  # Wavenumber in μm⁻¹
    n = (1 + 0.00008336624212083 + 
         0.02408926869968 / (130.1065924522 - s**2) +
         0.0001599740894897 / (38.92568793293 - s**2))
    
    vacuum_wl = wl * n
    
    if cgs:
        vacuum_wl *= 1e-8  # Convert back to cm
    
    return vacuum_wl


def vacuum_to_air(wavelength: jnp.ndarray, cgs: bool = None) -> jnp.ndarray:
    """
    Convert wavelengths from vacuum to air
    
    Uses iterative solution to invert the air_to_vacuum transformation.
    
    Parameters
    ----------
    wavelength : jnp.ndarray
        Vacuum wavelengths in Å (if ≥ 1) or cm (if < 1)
    cgs : bool, optional
        If True, wavelengths are in cm. If None, auto-detect based on magnitude.
        
    Returns
    -------
    jnp.ndarray
        Air wavelengths in same units as input
    """
    if cgs is None:
        cgs = jnp.all(wavelength < 1)
    
    # Initial guess: vacuum wavelength
    air_wl = wavelength.copy()
    
    # Newton-Raphson iteration to solve: air_to_vacuum(air_wl) = wavelength
    for _ in range(3):  # Usually converges in 2-3 iterations
        vacuum_calc = air_to_vacuum(air_wl, cgs=cgs)
        correction = wavelength - vacuum_calc
        air_wl += correction  # CRITICAL FIX: Remove 0.999 damping factor to match Korg.jl exactly
    
    return air_wl


__all__ = [
    'apply_LSF',
    'compute_LSF_matrix', 
    'apply_rotation',
    'air_to_vacuum',
    'vacuum_to_air'
]