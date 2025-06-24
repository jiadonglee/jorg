"""
Stellar atmosphere utilities using equation of state.

This module provides helper functions for working with stellar atmosphere
models and calculating thermodynamic properties.
"""

import jax.numpy as jnp
from jax import jit

from .eos import gas_pressure, electron_pressure, number_density_from_pressure


@jit
def calculate_pressure_from_atmosphere(temperature, total_number_density, electron_number_density):
    """
    Calculate gas and electron pressures from atmosphere layer data.
    
    This matches the approach used in Korg.jl atmosphere.jl:139 and related functions.
    
    Parameters
    ----------
    temperature : float or array
        Temperature in K
    total_number_density : float or array
        Total number density in cm^-3
    electron_number_density : float or array
        Electron number density in cm^-3
        
    Returns
    -------
    tuple of (gas_pressure, electron_pressure)
        Both pressures in dyne cm^-2 (cgs units)
    """
    P_gas = gas_pressure(total_number_density, temperature)
    P_e = electron_pressure(electron_number_density, temperature)
    
    return P_gas, P_e


@jit
def calculate_densities_from_pressure(temperature, gas_pressure_val, electron_pressure_val):
    """
    Calculate number densities from pressures (inverse of calculate_pressure_from_atmosphere).
    
    Parameters
    ----------
    temperature : float or array
        Temperature in K
    gas_pressure_val : float or array
        Gas pressure in dyne cm^-2 (cgs units)
    electron_pressure_val : float or array
        Electron pressure in dyne cm^-2 (cgs units)
        
    Returns
    -------
    tuple of (total_number_density, electron_number_density)
        Both densities in cm^-3
    """
    n_total = number_density_from_pressure(gas_pressure_val, temperature)
    n_e = number_density_from_pressure(electron_pressure_val, temperature)
    
    return n_total, n_e


@jit
def pressure_ratio(electron_pressure_val, gas_pressure_val):
    """
    Calculate the ratio of electron pressure to gas pressure.
    
    This is useful for characterizing the ionization state of the atmosphere.
    
    Parameters
    ----------
    electron_pressure_val : float or array
        Electron pressure in dyne cm^-2
    gas_pressure_val : float or array
        Gas pressure in dyne cm^-2
        
    Returns
    -------
    float or array
        Ratio P_e / P_gas (dimensionless)
    """
    return electron_pressure_val / gas_pressure_val


@jit
def electron_density_fraction(electron_density, total_density):
    """
    Calculate the electron number density as a fraction of total number density.
    
    Parameters
    ----------
    electron_density : float or array
        Electron number density in cm^-3
    total_density : float or array
        Total number density in cm^-3
        
    Returns
    -------
    float or array
        Fraction n_e / n_total (dimensionless)
    """
    return electron_density / total_density


@jit
def mean_molecular_weight_from_densities(total_mass_density, total_number_density):
    """
    Calculate mean molecular weight from mass and number densities.
    
    μ = ρ / (n * m_u)
    
    Parameters
    ----------
    total_mass_density : float or array
        Total mass density in g cm^-3
    total_number_density : float or array
        Total number density in cm^-3
        
    Returns
    -------
    float or array
        Mean molecular weight in atomic mass units
    """
    # Atomic mass unit in grams
    m_u = 1.66053906660e-24  # g
    
    return total_mass_density / (total_number_density * m_u)