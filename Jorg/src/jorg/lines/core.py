"""
Core line absorption calculations for stellar spectral synthesis

This module contains the main line absorption functions that combine
line profiles, broadening mechanisms, and opacity calculations.
"""

import jax
import jax.numpy as jnp
from typing import List, Dict, Optional, Union

from .linelist import LineList, LineData
from .opacity import calculate_line_opacity_korg_method
from .profiles import voigt_profile
from .broadening import thermal_doppler_width, van_der_waals_broadening
from ..constants import SPEED_OF_LIGHT, BOLTZMANN_K, ATOMIC_MASS_UNIT

__all__ = ['total_line_absorption', 'line_absorption', 'calculate_line_profile']


def total_line_absorption(wavelengths: jnp.ndarray,
                         linelist: Union[List[LineData], LineList],
                         temperature: float,
                         log_g: float,
                         abundances: Optional[Dict[int, float]] = None,
                         electron_density: float = 1e14,
                         hydrogen_density: float = 1e16,
                         microturbulence: float = 0.0,
                         **kwargs) -> jnp.ndarray:
    """
    Calculate total line absorption coefficient from a linelist
    
    This is the main function for calculating line opacity from a list
    of spectral lines, similar to Korg.jl's line_absorption! function.
    
    Parameters
    ----------
    wavelengths : array
        Wavelength grid in Angstroms
    linelist : list or LineList
        List of spectral lines
    temperature : float
        Temperature in K
    log_g : float
        Surface gravity (log10(g) in cm/s²)
    abundances : dict, optional
        Element abundances {element_id: abundance}
    electron_density : float
        Electron density in cm^-3
    hydrogen_density : float
        Hydrogen density in cm^-3
    microturbulence : float
        Microturbulent velocity in km/s
    **kwargs
        Additional parameters
        
    Returns
    -------
    array
        Total line absorption coefficient in cm^-1
    """
    # Handle different input types
    if isinstance(linelist, LineList):
        lines = linelist.lines
    else:
        lines = linelist
    
    if not lines:
        return jnp.zeros_like(wavelengths)
    
    # Default abundance pattern if not provided
    if abundances is None:
        abundances = {
            1: 1.0,      # H
            2: 0.1,      # He
            11: 1e-6,    # Na
            26: 1e-4,    # Fe
        }
    
    total_opacity = jnp.zeros_like(wavelengths)
    
    # Calculate opacity for each line
    for line in lines:
        # Extract line parameters
        if hasattr(line, 'wavelength'):
            line_wl = line.wavelength
            species_id = line.species
            excitation_potential = line.excitation_potential
            log_gf = line.log_gf
        else:
            # Handle dictionary format
            line_wl = line['wavelength']
            species_id = line['species'] 
            excitation_potential = line['excitation_potential']
            log_gf = line['log_gf']
        
        # Get element ID and abundance
        element_id = species_id // 100
        if element_id not in abundances:
            continue
            
        abundance = abundances[element_id]
        
        # Estimate atomic mass (simplified)
        atomic_mass = _get_atomic_mass(element_id)
        
        # Calculate line opacity
        line_opacity = calculate_line_opacity_korg_method(
            wavelengths, line_wl, excitation_potential, log_gf,
            temperature, electron_density, hydrogen_density, abundance,
            atomic_mass=atomic_mass, microturbulence=microturbulence,
            **kwargs
        )
        
        total_opacity += line_opacity
    
    return total_opacity


def line_absorption(wavelength_grid: jnp.ndarray,
                   line_center: float,
                   oscillator_strength: float,
                   excitation_potential: float,
                   temperature: float,
                   number_density: float,
                   atomic_mass: float,
                   microturbulence: float = 0.0,
                   gamma_rad: float = 0.0,
                   gamma_vdw: float = 0.0,
                   gamma_stark: float = 0.0) -> jnp.ndarray:
    """
    Calculate absorption coefficient for a single spectral line
    
    Parameters
    ----------
    wavelength_grid : array
        Wavelength grid in Angstroms
    line_center : float
        Line center wavelength in Angstroms
    oscillator_strength : float
        Oscillator strength (linear, not log)
    excitation_potential : float
        Lower level excitation potential in eV
    temperature : float
        Temperature in K
    number_density : float
        Number density of absorbing species in cm^-3
    atomic_mass : float
        Atomic mass in amu
    microturbulence : float
        Microturbulent velocity in km/s
    gamma_rad : float
        Radiative broadening parameter
    gamma_vdw : float
        Van der Waals broadening parameter
    gamma_stark : float
        Stark broadening parameter
        
    Returns
    -------
    array
        Line absorption coefficient in cm^-1
    """
    # Calculate Doppler width
    doppler_width = thermal_doppler_width(
        line_center, temperature, atomic_mass, microturbulence
    )
    
    # Calculate total Lorentz width
    lorentz_width = gamma_rad + gamma_vdw + gamma_stark
    
    # Calculate line profile
    profile = voigt_profile(wavelength_grid, line_center, doppler_width, lorentz_width)
    
    # Calculate line strength (simplified)
    line_strength = oscillator_strength * number_density
    
    return line_strength * profile


def calculate_line_profile(wavelengths: jnp.ndarray,
                          line_center: float,
                          doppler_width: float,
                          lorentz_width: float) -> jnp.ndarray:
    """
    Calculate normalized line profile
    
    Parameters
    ----------
    wavelengths : array
        Wavelength grid in Angstroms
    line_center : float
        Line center wavelength in Angstroms
    doppler_width : float
        Gaussian (Doppler) width in Angstroms
    lorentz_width : float
        Lorentzian width in Angstroms
        
    Returns
    -------
    array
        Normalized line profile
    """
    return voigt_profile(wavelengths, line_center, doppler_width, lorentz_width)


def _get_atomic_mass(element_id: int) -> float:
    """
    Get atomic mass from element ID
    
    Simplified lookup table for common elements.
    In a full implementation, this would use a comprehensive database.
    """
    masses = {
        1: 1.008,    # H
        2: 4.003,    # He
        3: 6.941,    # Li
        4: 9.012,    # Be
        5: 10.811,   # B
        6: 12.011,   # C
        7: 14.007,   # N
        8: 15.999,   # O
        9: 18.998,   # F
        10: 20.180,  # Ne
        11: 22.990,  # Na
        12: 24.305,  # Mg
        13: 26.982,  # Al
        14: 28.085,  # Si
        15: 30.974,  # P
        16: 32.066,  # S
        17: 35.453,  # Cl
        18: 39.948,  # Ar
        19: 39.098,  # K
        20: 40.078,  # Ca
        21: 44.956,  # Sc
        22: 47.867,  # Ti
        23: 50.942,  # V
        24: 51.996,  # Cr
        25: 54.938,  # Mn
        26: 55.845,  # Fe
        27: 58.933,  # Co
        28: 58.693,  # Ni
        29: 63.546,  # Cu
        30: 65.409,  # Zn
    }
    
    return masses.get(element_id, 55.845)  # Default to Fe mass