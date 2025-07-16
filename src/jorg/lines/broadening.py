"""
Broadening mechanisms for stellar spectral lines

This module provides both JAX-optimized broadening functions and exact
Korg.jl-compatible broadening parameter calculations.
"""

import jax
import jax.numpy as jnp
from typing import Union, Tuple
from ..constants import c_cgs, kboltz_cgs, bohr_radius_cgs

# Import exact Korg.jl broadening functions
from .broadening_korg import (
    approximate_radiative_gamma,
    approximate_gammas,
    approximate_stark_broadening,
    approximate_vdw_broadening,
    approximate_line_strength,
    get_default_broadening_parameters,
    validate_broadening_parameters,
    process_vdw_parameter
)


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


# Additional broadening functions for compatibility with test suite
@jax.jit
def natural_broadening(wavelength: float, oscillator_strength: float) -> float:
    """
    Calculate natural (radiative) broadening
    
    Parameters
    ----------
    wavelength : float
        Wavelength in Angstroms
    oscillator_strength : float
        Oscillator strength (dimensionless)
        
    Returns
    -------
    float
        Natural broadening parameter in Angstroms
    """
    # Convert to cm
    lambda_cm = wavelength * 1e-8
    
    # Natural broadening constant
    # γ_rad = (8π²e²f)/(mₑλ²c) in CGS units
    e_cgs = 4.803e-10  # esu
    me_cgs = 9.109e-28  # g
    
    gamma_rad = (8 * jnp.pi**2 * e_cgs**2 * oscillator_strength) / (me_cgs * lambda_cm**2 * c_cgs)
    
    # Convert back to Angstroms
    return convert_broadening_to_wavelength(gamma_rad, lambda_cm) * 1e8


@jax.jit 
def stark_broadening(wavelength: float, temperature: float, electron_density: float, stark_constant: float) -> float:
    """
    Calculate Stark (pressure) broadening
    
    Parameters
    ----------
    wavelength : float
        Wavelength in Angstroms
    temperature : float
        Temperature in K
    electron_density : float
        Electron density in cm^-3
    stark_constant : float
        Stark broadening constant
        
    Returns
    -------
    float
        Stark broadening parameter in Angstroms
    """
    # Stark broadening scales with electron density and temperature
    gamma_stark = stark_constant * electron_density * (temperature / 10000.0)**(1.0/6.0)
    return gamma_stark


@jax.jit
def vdw_broadening(wavelength: float, temperature: float, neutral_density: float, vdw_constant: float) -> float:
    """
    Calculate van der Waals broadening
    
    Parameters
    ----------
    wavelength : float
        Wavelength in Angstroms
    temperature : float
        Temperature in K
    neutral_density : float
        Neutral atom density in cm^-3
    vdw_constant : float
        van der Waals broadening constant
        
    Returns
    -------
    float
        van der Waals broadening parameter in Angstroms
    """
    # vdW broadening scales with neutral density and temperature^0.3
    gamma_vdw = vdw_constant * neutral_density * (temperature / 10000.0)**0.3
    return gamma_vdw


# Korg.jl-compatible broadening parameter calculation
def get_korg_broadening_parameters(species, wl_cm: float, log_gf: float, E_lower: float,
                                  provided_gamma_rad: float = None,
                                  provided_gamma_stark: float = None,
                                  provided_vdw: Union[float, Tuple[float, float]] = None):
    """
    Calculate broadening parameters using exact Korg.jl methodology.
    
    This function implements the same logic as Korg.jl's Line constructor,
    providing default approximations when parameters are not provided.
    
    Parameters
    ----------
    species : Species
        Chemical species object
    wl_cm : float
        Wavelength in cm
    log_gf : float
        log₁₀(gf) oscillator strength
    E_lower : float
        Lower energy level in eV
    provided_gamma_rad : float, optional
        Provided radiative broadening parameter
    provided_gamma_stark : float, optional
        Provided Stark broadening parameter
    provided_vdw : float or tuple, optional
        Provided van der Waals parameter(s)
        
    Returns
    -------
    Dict[str, float]
        Dictionary with broadening parameters
    """
    # Start with defaults
    broadening_params = get_default_broadening_parameters(species, wl_cm, log_gf, E_lower)
    
    # Override with provided values if available
    if provided_gamma_rad is not None:
        broadening_params['gamma_rad'] = provided_gamma_rad
    
    if provided_gamma_stark is not None:
        broadening_params['gamma_stark'] = provided_gamma_stark
    
    if provided_vdw is not None:
        if isinstance(provided_vdw, tuple):
            broadening_params['vdw_param1'] = provided_vdw[0]
            broadening_params['vdw_param2'] = provided_vdw[1]
        else:
            # Process according to Korg.jl rules
            vdw_param1, vdw_param2 = process_vdw_parameter(provided_vdw, species, E_lower, wl_cm)
            broadening_params['vdw_param1'] = vdw_param1
            broadening_params['vdw_param2'] = vdw_param2
    
    return broadening_params


@jax.jit
def calculate_line_broadening(wl_cm: float, temperature: float, electron_density: float,
                             neutral_density: float, gamma_rad: float, gamma_stark: float,
                             vdw_param1: float, vdw_param2: float, mass: float) -> float:
    """
    Calculate total line broadening parameter for given conditions.
    
    This JAX-optimized function scales the broadening parameters to the
    given atmospheric conditions.
    
    Parameters
    ----------
    wl_cm : float
        Wavelength in cm
    temperature : float
        Temperature in K
    electron_density : float
        Electron density in cm⁻³
    neutral_density : float
        Neutral atom density in cm⁻³
    gamma_rad : float
        Radiative broadening parameter
    gamma_stark : float
        Stark broadening parameter (at 10,000 K)
    vdw_param1 : float
        van der Waals parameter 1
    vdw_param2 : float
        van der Waals parameter 2
    mass : float
        Atomic mass in grams
        
    Returns
    -------
    float
        Total line broadening parameter
    """
    # Radiative broadening (temperature independent)
    gamma_rad_scaled = gamma_rad
    
    # Stark broadening (temperature and electron density dependent)
    gamma_stark_scaled = scaled_stark(gamma_stark, temperature) * electron_density
    
    # van der Waals broadening - use JAX-compatible conditional
    gamma_vdw_simple_scaled = scaled_vdw_simple(vdw_param1, temperature) * neutral_density
    gamma_vdw_abo_scaled = scaled_vdw_abo(vdw_param1, vdw_param2, mass, temperature) * neutral_density
    
    # JAX-compatible conditional
    gamma_vdw_scaled = jnp.where(
        vdw_param2 == -1.0,
        gamma_vdw_simple_scaled,
        gamma_vdw_abo_scaled
    )
    
    # Total broadening
    gamma_total = gamma_rad_scaled + gamma_stark_scaled + gamma_vdw_scaled
    
    return gamma_total


def validate_line_broadening_parameters(broadening_params: dict) -> bool:
    """
    Validate broadening parameters for physical reasonableness.
    
    Parameters
    ----------
    broadening_params : dict
        Dictionary containing broadening parameters
        
    Returns
    -------
    bool
        True if parameters are valid
    """
    return validate_broadening_parameters(
        broadening_params['gamma_rad'],
        broadening_params['gamma_stark'],
        broadening_params['vdw_param1'],
        broadening_params['vdw_param2']
    )