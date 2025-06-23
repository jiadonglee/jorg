"""
Main line absorption calculation combining all line opacity sources
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Any, NamedTuple, Union

from .profiles import line_profile
from .broadening import (
    doppler_width, 
    scaled_stark, 
    scaled_vdw,
    convert_broadening_to_wavelength
)
from .utils import (
    inverse_gaussian_density,
    inverse_lorentz_density, 
    sigma_line,
    get_mass_from_species_id,
    calculate_window_size
)
from ..constants import c_cgs, hplanck_eV, kboltz_eV, pi


class LineData(NamedTuple):
    """Structure to hold line data compatible with JAX"""
    wavelength: float          # Central wavelength in cm
    log_gf: float             # log(gf) value  
    E_lower: float            # Lower level energy in eV
    species_id: int           # Species identifier
    gamma_rad: float          # Natural broadening parameter
    gamma_stark: float        # Stark broadening parameter  
    vdw_param1: float         # vdW parameter (either γ_vdW or σ for ABO)
    vdw_param2: float         # Second vdW parameter (α for ABO, unused for simple)


def line_absorption(
    wavelengths: jnp.ndarray,
    linelist: List[LineData],
    temperature: float,
    electron_density: float,
    number_densities: Dict[int, float],
    partition_functions: Dict[int, Any],
    microturbulent_velocity: float,
    continuum_absorption_fn: Any = None,
    cutoff_threshold: float = 3e-4
) -> jnp.ndarray:
    """
    Calculate line absorption coefficient at given wavelengths.
    
    This is the JAX equivalent of Korg's line_absorption! function.
    
    Parameters
    ----------
    wavelengths : jnp.ndarray
        Wavelengths in cm (must be sorted)
    linelist : List[LineData] 
        List of line data structures
    temperature : float
        Temperature in K
    electron_density : float
        Electron number density in cm^-3
    number_densities : Dict[int, float]
        Number densities for each species in cm^-3
    partition_functions : Dict[int, Any]
        Partition functions for each species
    microturbulent_velocity : float
        Microturbulent velocity in cm/s
    continuum_absorption_fn : callable, optional
        Function returning continuum opacity at given wavelength
    cutoff_threshold : float, optional
        Threshold for line window calculation (default: 3e-4)
        
    Returns
    -------
    jnp.ndarray
        Line absorption coefficient in cm^-1
        
    Notes
    -----
    This function processes all lines in the linelist and computes their
    combined absorption coefficient. Lines are processed with appropriate
    window sizes based on their broadening parameters.
    """
    
    if len(linelist) == 0:
        return jnp.zeros_like(wavelengths)
    
    # Initialize total absorption array
    alpha_total = jnp.zeros_like(wavelengths)
    
    # Precompute temperature-dependent factors
    beta = 1.0 / (kboltz_eV * temperature)
    
    # Precompute number density / partition function for each species
    n_div_U = {}
    for species_id in set(line.species_id for line in linelist):
        if species_id in number_densities and species_id in partition_functions:
            n_dens = number_densities[species_id]
            partition_fn = partition_functions[species_id]
            U_val = partition_fn(jnp.log(temperature))
            n_div_U[species_id] = n_dens / U_val
    
    # Process each line
    for line in linelist:
        if line.species_id not in n_div_U:
            continue
            
        # Get atomic mass for this species
        mass = get_mass_from_species_id(line.species_id)
        
        alpha_line = _process_single_line(
            line, wavelengths, temperature, beta, electron_density,
            n_div_U[line.species_id], mass, microturbulent_velocity,
            continuum_absorption_fn, cutoff_threshold
        )
        
        alpha_total = alpha_total + alpha_line
    
    return alpha_total


def _process_single_line(
    line: LineData,
    wavelengths: jnp.ndarray, 
    temperature: float,
    beta: float,
    electron_density: float,
    n_div_U: float,
    mass: float,
    xi: float,
    continuum_fn: Any,
    cutoff_threshold: float
) -> jnp.ndarray:
    """
    Process a single line and return its absorption contribution.
    
    This function is JIT-compiled for performance.
    """
    
    # Doppler broadening width (σ, NOT √2σ)
    sigma = doppler_width(line.wavelength, temperature, mass, xi)
    
    # Calculate damping parameters (FWHM values in angular frequency)
    gamma_total = line.gamma_rad
    
    # Add Stark broadening (for atoms only)
    if line.species_id < 100:  # Simple check for atomic species vs molecules
        gamma_stark_scaled = scaled_stark(line.gamma_stark, temperature)
        gamma_total += electron_density * gamma_stark_scaled
        
        # Add van der Waals broadening  
        n_h_neutral = 1e15  # Placeholder - would need H I density from number_densities
        if line.vdw_param2 == 0.0:
            # Simple vdW scaling
            gamma_vdw_scaled = scaled_vdw(line.vdw_param1, mass, temperature)
        else:
            # ABO parameters
            gamma_vdw_scaled = scaled_vdw((line.vdw_param1, line.vdw_param2), mass, temperature)
        
        gamma_total += n_h_neutral * gamma_vdw_scaled
    
    # Convert broadening from angular frequency to wavelength units
    gamma_wavelength = convert_broadening_to_wavelength(gamma_total, line.wavelength)
    
    # Calculate level population factors
    E_upper = line.E_lower + c_cgs * hplanck_eV / line.wavelength
    levels_factor = jnp.exp(-beta * line.E_lower) - jnp.exp(-beta * E_upper)
    
    # Total wavelength-integrated absorption coefficient
    amplitude = (10.0**line.log_gf * sigma_line(line.wavelength) * 
                levels_factor * n_div_U)
    
    # Calculate line window size
    if continuum_fn is not None:
        continuum_at_line = continuum_fn(line.wavelength)
        rho_crit = continuum_at_line * cutoff_threshold / amplitude
    else:
        rho_crit = cutoff_threshold / amplitude
    
    # Calculate window sizes for Doppler and Lorentz components
    doppler_window = inverse_gaussian_density(rho_crit, sigma)
    lorentz_window = inverse_lorentz_density(rho_crit, gamma_wavelength)
    window_size = calculate_window_size(doppler_window, lorentz_window)
    
    # Find wavelength indices within the window
    line_center = line.wavelength
    wl_min = line_center - window_size
    wl_max = line_center + window_size
    
    # Create mask for wavelengths within the line window
    in_window = (wavelengths >= wl_min) & (wavelengths <= wl_max)
    
    # Calculate line profile only for wavelengths within the window
    alpha_line = jnp.where(
        in_window,
        line_profile(line_center, sigma, gamma_wavelength, amplitude, wavelengths),
        0.0
    )
    
    return alpha_line


def create_line_data(wavelength_cm: float,
                    log_gf: float, 
                    E_lower_eV: float,
                    species_id: int,
                    gamma_rad: float = 0.0,
                    gamma_stark: float = 0.0,
                    vdw_param1: float = 0.0,
                    vdw_param2: float = 0.0) -> LineData:
    """
    Create a LineData structure from individual parameters.
    
    Parameters
    ----------
    wavelength_cm : float
        Central wavelength in cm
    log_gf : float
        Logarithm of oscillator strength times statistical weight
    E_lower_eV : float
        Lower level energy in eV
    species_id : int
        Species identifier (atomic number for atoms)
    gamma_rad : float, optional
        Natural radiative broadening parameter (default: 0.0)
    gamma_stark : float, optional  
        Stark broadening parameter (default: 0.0)
    vdw_param1 : float, optional
        First vdW parameter (default: 0.0)
    vdw_param2 : float, optional
        Second vdW parameter for ABO theory (default: 0.0)
        
    Returns
    -------
    LineData
        Structured line data
    """
    return LineData(
        wavelength=wavelength_cm,
        log_gf=log_gf,
        E_lower=E_lower_eV,
        species_id=species_id,
        gamma_rad=gamma_rad,
        gamma_stark=gamma_stark,
        vdw_param1=vdw_param1,
        vdw_param2=vdw_param2
    )


def convert_linelist_from_korg_format(korg_linelist: List[Dict]) -> List[LineData]:
    """
    Convert a Korg-format linelist to JAX-compatible LineData structures.
    
    Parameters
    ----------
    korg_linelist : List[Dict]
        List of line dictionaries in Korg format
        
    Returns
    -------
    List[LineData]
        List of LineData structures suitable for JAX processing
    """
    jax_linelist = []
    
    for line_dict in korg_linelist:
        # Convert wavelength from Angstroms to cm
        wavelength_cm = line_dict['wavelength'] * 1e-8
        
        # Extract species information
        species_id = line_dict.get('species_id', 26)  # Default to Fe I
        
        # Extract broadening parameters
        gamma_rad = line_dict.get('gamma_rad', 0.0)
        gamma_stark = line_dict.get('gamma_stark', 0.0)
        
        # Handle vdW parameters (could be single value or tuple)
        vdw = line_dict.get('vdw', 0.0)
        if isinstance(vdw, (list, tuple)):
            vdw_param1, vdw_param2 = vdw[0], vdw[1]
        else:
            vdw_param1, vdw_param2 = vdw, 0.0
        
        line_data = create_line_data(
            wavelength_cm=wavelength_cm,
            log_gf=line_dict['log_gf'],
            E_lower_eV=line_dict['E_lower'],
            species_id=species_id,
            gamma_rad=gamma_rad,
            gamma_stark=gamma_stark,
            vdw_param1=vdw_param1,
            vdw_param2=vdw_param2
        )
        
        jax_linelist.append(line_data)
    
    return jax_linelist