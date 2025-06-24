"""
Equation of state calculations for stellar atmospheres.

This module provides functions for calculating pressure and thermodynamic
properties using the ideal gas law and related relations, following the
approach used in Korg.jl.
"""

import jax.numpy as jnp
from jax import jit

from ..constants import kboltz_cgs


@jit
def gas_pressure(number_density, temperature):
    """
    Calculate gas pressure using the ideal gas law.
    
    P = n * kT
    
    Parameters
    ----------
    number_density : float or array
        Total number density in cm^-3
    temperature : float or array
        Temperature in K
        
    Returns
    -------
    float or array
        Gas pressure in dyne cm^-2 (cgs units)
    """
    return number_density * kboltz_cgs * temperature


@jit
def electron_pressure(electron_density, temperature):
    """
    Calculate electron pressure using the ideal gas law.
    
    P_e = n_e * kT
    
    Parameters
    ----------
    electron_density : float or array
        Electron number density in cm^-3
    temperature : float or array
        Temperature in K
        
    Returns
    -------
    float or array
        Electron pressure in dyne cm^-2 (cgs units)
    """
    return electron_density * kboltz_cgs * temperature


@jit
def total_pressure(number_density, electron_density, temperature):
    """
    Calculate total pressure including both gas and electron contributions.
    
    For most stellar atmosphere applications, this is equivalent to gas_pressure
    since electrons are already counted in the total number density.
    
    Parameters
    ----------
    number_density : float or array
        Total number density in cm^-3
    electron_density : float or array
        Electron number density in cm^-3
    temperature : float or array
        Temperature in K
        
    Returns
    -------
    float or array
        Total pressure in dyne cm^-2 (cgs units)
    """
    return gas_pressure(number_density, temperature)


@jit
def number_density_from_pressure(pressure, temperature):
    """
    Calculate number density from pressure using the ideal gas law.
    
    n = P / (kT)
    
    Parameters
    ----------
    pressure : float or array
        Pressure in dyne cm^-2 (cgs units)
    temperature : float or array
        Temperature in K
        
    Returns
    -------
    float or array
        Number density in cm^-3
    """
    return pressure / (kboltz_cgs * temperature)


@jit
def pressure_scale_height(temperature, gravity, mean_molecular_weight=1.3):
    """
    Calculate pressure scale height for hydrostatic equilibrium.
    
    H_P = kT / (μ * m_u * g)
    
    Parameters
    ----------
    temperature : float or array
        Temperature in K
    gravity : float or array
        Surface gravity in cm s^-2
    mean_molecular_weight : float, optional
        Mean molecular weight in atomic mass units (default: 1.3)
        
    Returns
    -------
    float or array
        Pressure scale height in cm
    """
    # Atomic mass unit in grams
    m_u = 1.66053906660e-24  # g
    
    return (kboltz_cgs * temperature) / (mean_molecular_weight * m_u * gravity)


@jit
def ideal_gas_density(pressure, temperature, mean_molecular_weight=1.3):
    """
    Calculate mass density using the ideal gas law.
    
    ρ = P * μ * m_u / (kT)
    
    Parameters
    ----------
    pressure : float or array
        Pressure in dyne cm^-2 (cgs units)
    temperature : float or array
        Temperature in K
    mean_molecular_weight : float, optional
        Mean molecular weight in atomic mass units (default: 1.3)
        
    Returns
    -------
    float or array
        Mass density in g cm^-3
    """
    # Atomic mass unit in grams
    m_u = 1.66053906660e-24  # g
    
    return pressure * mean_molecular_weight * m_u / (kboltz_cgs * temperature)


@jit
def pressure_from_density(density, temperature, mean_molecular_weight=1.3):
    """
    Calculate pressure from mass density using the ideal gas law.
    
    P = ρ * kT / (μ * m_u)
    
    Parameters
    ----------
    density : float or array
        Mass density in g cm^-3
    temperature : float or array
        Temperature in K
    mean_molecular_weight : float, optional
        Mean molecular weight in atomic mass units (default: 1.3)
        
    Returns
    -------
    float or array
        Pressure in dyne cm^-2 (cgs units)
    """
    # Atomic mass unit in grams
    m_u = 1.66053906660e-24  # g
    
    return density * kboltz_cgs * temperature / (mean_molecular_weight * m_u)