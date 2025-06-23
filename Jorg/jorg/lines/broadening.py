"""
Broadening mechanisms for stellar spectral lines
"""

import jax
import jax.numpy as jnp
from typing import Union, Tuple
from ..constants import c_cgs, kboltz_cgs


@jax.jit
def doppler_width(lambda_0: float, temperature: float, mass: float, xi: float) -> float:
    """
    Calculate the standard deviation of the Doppler-broadening profile.
    
    In standard spectroscopy texts, the Doppler width often refers to σ√2, 
    but this returns σ directly.
    
    Parameters
    ----------
    lambda_0 : float
        Central wavelength in cm
    temperature : float
        Temperature in K
    mass : float  
        Atomic mass in grams
    xi : float
        Microturbulent velocity in cm/s
        
    Returns
    -------
    float
        Doppler width σ in cm
    """
    thermal_velocity_sq = kboltz_cgs * temperature / mass
    microturbulent_velocity_sq = xi**2 / 2.0
    
    return lambda_0 * jnp.sqrt(thermal_velocity_sq + microturbulent_velocity_sq) / c_cgs


@jax.jit  
def scaled_stark(gamma_stark: float, temperature: float, T0: float = 10000.0) -> float:
    """
    Scale Stark broadening gamma according to its temperature dependence.
    
    Parameters
    ----------
    gamma_stark : float
        Stark broadening parameter at reference temperature
    temperature : float
        Current temperature in K
    T0 : float, optional
        Reference temperature in K (default: 10,000 K)
        
    Returns
    -------
    float
        Temperature-scaled Stark broadening parameter
    """
    return gamma_stark * (temperature / T0)**(1.0 / 6.0)


@jax.jit
def scaled_vdw_simple(gamma_vdw: float, temperature: float, T0: float = 10000.0) -> float:
    """
    Simple van der Waals broadening scaling with temperature.
    
    Parameters
    ----------
    gamma_vdw : float
        vdW broadening parameter at reference temperature
    temperature : float
        Current temperature in K  
    T0 : float, optional
        Reference temperature in K (default: 10,000 K)
        
    Returns
    -------
    float
        Temperature-scaled vdW broadening parameter
    """
    return gamma_vdw * (temperature / T0)**0.3


@jax.jit
def scaled_vdw_abo(sigma: float, alpha: float, mass: float, temperature: float) -> float:
    """
    van der Waals broadening using ABO (Anstee, Barklem, O'Mara) theory.
    
    Parameters
    ----------
    sigma : float
        ABO cross-section parameter  
    alpha : float
        ABO velocity exponent parameter
    mass : float
        Atomic mass in grams
    temperature : float
        Temperature in K
        
    Returns
    -------
    float
        ABO-scaled vdW broadening parameter
    """
    # Reference velocity: 10^6 cm/s = 10 km/s
    v0 = 1e6
    
    # Mean relative velocity from kinetic theory
    reduced_mass = mass * 1.67262192e-24 / (mass + 1.67262192e-24)  # H atom mass
    mean_velocity = jnp.sqrt(8.0 * kboltz_cgs * temperature / (jnp.pi * reduced_mass))
    
    # ABO formula
    velocity_ratio = mean_velocity / v0
    return sigma * velocity_ratio**alpha


def scaled_vdw(vdw_params: Union[float, Tuple[float, float]], 
               mass: float, 
               temperature: float, 
               T0: float = 10000.0) -> float:
    """
    van der Waals broadening scaling with temperature dependence.
    
    Supports either simple scaling or ABO (Anstee, Barklem, O'Mara) theory.
    See Anstee & O'Mara (1995) or Paul Barklem's notes for ABO definitions.
    
    Parameters
    ----------
    vdw_params : float or tuple
        Either γ_vdW evaluated at 10,000 K, or tuple containing ABO params (σ, α)
    mass : float
        Atomic mass in grams (ignored for simple scaling)
    temperature : float
        Temperature in K
    T0 : float, optional
        Reference temperature for simple scaling (default: 10,000 K)
        
    Returns
    -------
    float
        Temperature-scaled vdW broadening parameter
    """
    if isinstance(vdw_params, tuple):
        sigma, alpha = vdw_params
        if alpha == -1:
            # Special case: use simple T^0.3 scaling
            return sigma * (temperature / T0)**0.3
        else:
            # Use full ABO theory
            return scaled_vdw_abo(sigma, alpha, mass, temperature)
    else:
        # Simple temperature scaling
        return scaled_vdw_simple(vdw_params, temperature, T0)


@jax.jit
def natural_broadening_gamma(A_ul: float, lambda_0: float) -> float:
    """
    Calculate natural (radiative) broadening parameter.
    
    Parameters
    ----------
    A_ul : float
        Einstein A coefficient (transition rate) in s^-1
    lambda_0 : float
        Central wavelength in cm
        
    Returns
    -------
    float
        Natural broadening parameter in angular frequency units
    """
    # Convert A coefficient to gamma in angular frequency
    # γ = A_ul (for single upper level)
    return A_ul


@jax.jit
def total_broadening_gamma(gamma_natural: float,
                          gamma_stark: float, 
                          gamma_vdw: float) -> float:
    """
    Calculate total broadening parameter from all mechanisms.
    
    Parameters
    ----------
    gamma_natural : float
        Natural (radiative) broadening parameter
    gamma_stark : float
        Stark broadening parameter  
    gamma_vdw : float
        van der Waals broadening parameter
        
    Returns
    -------
    float
        Total broadening parameter (FWHM in angular frequency)
    """
    return gamma_natural + gamma_stark + gamma_vdw


@jax.jit
def convert_broadening_to_wavelength(gamma_angular: float, lambda_0: float) -> float:
    """
    Convert broadening parameter from angular frequency to wavelength units.
    
    This involves the approximation that λ(ν) is linear over the line window.
    The conversion factor is λ²/c for |dλ/dν|, 1/2π for angular vs cyclical 
    frequency, and 1/2 for FWHM vs HWHM.
    
    Parameters
    ----------
    gamma_angular : float
        Broadening parameter in angular frequency (FWHM)
    lambda_0 : float
        Central wavelength in cm
        
    Returns
    -------
    float
        Broadening parameter in wavelength units (HWHM)
    """
    return gamma_angular * lambda_0**2 / (c_cgs * 4.0 * jnp.pi)