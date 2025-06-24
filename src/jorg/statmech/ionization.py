"""
Ionization equilibrium calculations using the Saha equation.

Based on Korg.jl statistical mechanics implementation.
"""

import jax.numpy as jnp
from jax import jit
from typing import Tuple, Dict, Any
from ..constants import (
    kboltz_cgs, ELECTRON_MASS, PLANCK_H, PI, EV_TO_ERG,
    me_cgs, hplanck_cgs, pi
)


# Korg compatibility constants  
kboltz_eV = kboltz_cgs / EV_TO_ERG  # Boltzmann constant in eV/K
electron_mass_cgs = me_cgs


@jit
def translational_u(m: float, T: float) -> float:
    """
    Calculate the translational contribution to partition function.
    Used in the Saha equation for free particle motion.
    
    Parameters:
    -----------
    m : float
        Particle mass in g
    T : float
        Temperature in K
        
    Returns:
    --------
    float
        Translational partition function contribution
    """
    k = kboltz_cgs
    h = hplanck_cgs
    return (2 * pi * m * k * T / h**2)**1.5


@jit
def saha_ion_weights(T: float, ne: float, atom: int, 
                    ionization_energies: Dict[int, Tuple[float, float, float]],
                    partition_funcs: Dict[str, Any]) -> Tuple[float, float]:
    """
    Calculate ionization weights using the Saha equation.
    
    Returns the ratio of singly ionized to neutral atoms (wII) and 
    doubly ionized to neutral atoms (wIII) for a given element.
    
    Parameters:
    -----------
    T : float
        Temperature in K
    ne : float
        Electron number density in cm^-3
    atom : int
        Atomic number of the element
    ionization_energies : dict
        Dictionary mapping atomic numbers to (χI, χII, χIII) in eV
    partition_funcs : dict
        Dictionary mapping species to partition functions
        
    Returns:
    --------
    Tuple[float, float]
        (wII, wIII) - ratios of ionized to neutral number densities
    """
    # Get ionization energies for this element
    if atom in ionization_energies:
        chi_I, chi_II, chi_III = ionization_energies[atom]
    else:
        # Default values - should be replaced with actual data
        chi_I, chi_II, chi_III = 13.6, 54.4, 122.4  # Hydrogen-like scaling
    
    # Get partition functions
    # Simplified species keys - in practice would use proper Species objects
    UI = partition_funcs.get(f"{atom}_0", lambda log_T: 2.0)(jnp.log(T))
    UII = partition_funcs.get(f"{atom}_1", lambda log_T: 1.0)(jnp.log(T))
    
    k = kboltz_eV
    trans_U = translational_u(electron_mass_cgs, T)
    
    # Saha equation for first ionization
    wII = 2.0 / ne * (UII / UI) * trans_U * jnp.exp(-chi_I / (k * T))
    
    # Second ionization (skip for hydrogen)
    if atom == 1:  # Hydrogen
        wIII = 0.0
    else:
        UIII = partition_funcs.get(f"{atom}_2", lambda log_T: 1.0)(jnp.log(T))
        wIII = wII * 2.0 / ne * (UIII / UII) * trans_U * jnp.exp(-chi_II / (k * T))
    
    return wII, wIII


# Default ionization energies for common elements (eV)
DEFAULT_IONIZATION_ENERGIES = {
    1: (13.598, 0.0, 0.0),      # Hydrogen (only first ionization)
    2: (24.587, 54.418, 0.0),   # Helium  
    3: (5.392, 75.640, 122.454), # Lithium
    4: (9.323, 18.211, 153.896), # Beryllium
    5: (8.298, 25.155, 37.931),  # Boron
    6: (11.260, 24.383, 47.888), # Carbon
    7: (14.534, 29.601, 47.449), # Nitrogen
    8: (13.618, 35.121, 54.936), # Oxygen
    # Add more elements as needed
}


def create_default_ionization_energies() -> Dict[int, Tuple[float, float, float]]:
    """
    Create default ionization energies dictionary.
    
    Returns:
    --------
    Dict[int, Tuple[float, float, float]]
        Dictionary mapping atomic numbers to (χI, χII, χIII) in eV
    """
    return DEFAULT_IONIZATION_ENERGIES.copy()