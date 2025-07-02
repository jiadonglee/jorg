"""
Solar abundance patterns and utilities for Jorg

Provides standard abundance references including Asplund et al. 2009
"""

import jax.numpy as jnp
from typing import Dict, Any

# Asplund et al. 2009 solar abundances (log scale, H=12.0)
ASPLUND_2009 = {
    'H': 12.00,   # Hydrogen (reference)
    'He': 10.93,  # Helium
    'Li': 1.05,   # Lithium  
    'Be': 1.38,   # Beryllium
    'B': 2.70,    # Boron
    'C': 8.43,    # Carbon
    'N': 7.83,    # Nitrogen
    'O': 8.69,    # Oxygen
    'F': 4.56,    # Fluorine
    'Ne': 7.93,   # Neon
    'Na': 6.24,   # Sodium
    'Mg': 7.60,   # Magnesium
    'Al': 6.45,   # Aluminum
    'Si': 7.51,   # Silicon
    'P': 5.41,    # Phosphorus
    'S': 7.12,    # Sulfur
    'Cl': 5.50,   # Chlorine
    'Ar': 6.40,   # Argon
    'K': 5.03,    # Potassium
    'Ca': 6.34,   # Calcium
    'Sc': 3.15,   # Scandium
    'Ti': 4.95,   # Titanium
    'V': 3.93,    # Vanadium
    'Cr': 5.64,   # Chromium
    'Mn': 5.43,   # Manganese
    'Fe': 7.50,   # Iron
    'Co': 4.99,   # Cobalt
    'Ni': 6.22,   # Nickel
    'Cu': 4.19,   # Copper
    'Zn': 4.56    # Zinc
}

def get_asplund_abundances(m_H: float = 0.0) -> jnp.ndarray:
    """
    Get Asplund et al. 2009 abundances as array for EOS calculations
    
    Parameters:
    -----------
    m_H : float
        Metallicity [M/H] scaling factor
        
    Returns:
    --------
    jnp.ndarray
        Linear abundances relative to hydrogen, scaled by metallicity
    """
    # Convert log abundances to linear scale
    abundances = jnp.zeros(30)  # H through Zn
    
    # Element indices (0-based)
    element_indices = {
        'H': 0, 'He': 1, 'Li': 2, 'Be': 3, 'B': 4, 'C': 5, 'N': 6, 'O': 7,
        'F': 8, 'Ne': 9, 'Na': 10, 'Mg': 11, 'Al': 12, 'Si': 13, 'P': 14,
        'S': 15, 'Cl': 16, 'Ar': 17, 'K': 18, 'Ca': 19, 'Sc': 20, 'Ti': 21,
        'V': 22, 'Cr': 23, 'Mn': 24, 'Fe': 25, 'Co': 26, 'Ni': 27, 'Cu': 28, 'Zn': 29
    }
    
    # Set abundances with metallicity scaling
    for element, log_abundance in ASPLUND_2009.items():
        if element in element_indices:
            idx = element_indices[element]
            if element in ['H', 'He']:
                # H and He don't scale with metallicity
                linear_abundance = 10**(log_abundance - 12.0)
            else:
                # Metals scale with [M/H]
                linear_abundance = 10**(log_abundance - 12.0 + m_H)
            abundances = abundances.at[idx].set(linear_abundance)
    
    return abundances


def get_asplund_ionization_energies() -> Dict[int, jnp.ndarray]:
    """
    Get ionization energies for Asplund abundance elements
    
    Returns:
    --------
    Dict[int, jnp.ndarray]
        Dictionary mapping atomic number to [χ_I, χ_II, χ_III] in eV
    """
    # Ionization energies in eV (from NIST and other sources)
    ionization_energies = {
        1: jnp.array([13.598, 0.0, 0.0]),        # H
        2: jnp.array([24.587, 54.418, 0.0]),     # He
        3: jnp.array([5.392, 75.640, 122.454]),  # Li
        4: jnp.array([9.323, 18.211, 153.896]),  # Be
        5: jnp.array([8.298, 25.155, 37.930]),   # B
        6: jnp.array([11.260, 24.383, 47.888]),  # C
        7: jnp.array([14.534, 29.601, 47.449]),  # N
        8: jnp.array([13.618, 35.121, 54.936]),  # O
        9: jnp.array([17.423, 34.971, 62.708]),  # F
        10: jnp.array([21.565, 40.963, 63.45]),  # Ne
        11: jnp.array([5.139, 47.287, 71.620]),  # Na
        12: jnp.array([7.646, 15.035, 80.144]),  # Mg
        13: jnp.array([5.986, 18.829, 28.448]),  # Al
        14: jnp.array([8.152, 16.346, 33.493]),  # Si
        15: jnp.array([10.487, 19.769, 30.203]), # P
        16: jnp.array([10.360, 23.338, 34.79]),  # S
        17: jnp.array([12.968, 23.814, 39.61]),  # Cl
        18: jnp.array([15.760, 27.630, 40.74]),  # Ar
        19: jnp.array([4.341, 31.625, 45.806]),  # K
        20: jnp.array([6.113, 11.872, 50.913]),  # Ca
        21: jnp.array([6.561, 12.800, 24.757]),  # Sc
        22: jnp.array([6.828, 13.576, 27.492]),  # Ti
        23: jnp.array([6.746, 14.618, 29.311]),  # V
        24: jnp.array([6.767, 16.486, 30.96]),   # Cr
        25: jnp.array([7.434, 15.640, 33.668]),  # Mn
        26: jnp.array([7.902, 16.199, 30.652]),  # Fe
        27: jnp.array([7.881, 17.084, 33.50]),   # Co
        28: jnp.array([7.640, 18.169, 35.19]),   # Ni
        29: jnp.array([7.726, 20.292, 36.841]),  # Cu
        30: jnp.array([9.394, 17.964, 39.723])   # Zn
    }
    
    return ionization_energies


def format_A_X() -> Dict[int, float]:
    """
    Format abundances as A(X) values mapped by atomic number for tutorial compatibility.
    
    Returns Asplund et al. 2009 solar abundances in the format expected
    by the tutorial: atomic number -> A(X) where A(X) = log10(N_X/N_H) + 12.0
    
    Returns:
    --------
    Dict[int, float]
        Dictionary mapping atomic numbers to A(X) values
    """
    # Map element symbols to atomic numbers
    element_to_z = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
        'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
        'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
        'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30
    }
    
    # Convert to atomic number keys
    abundances = {}
    for element, log_abundance in ASPLUND_2009.items():
        if element in element_to_z:
            Z = element_to_z[element]
            abundances[Z] = log_abundance
    
    return abundances


def calculate_eos_with_asplund(temperature: float, 
                              total_density: float,
                              electron_density_guess: float,
                              m_H: float = 0.0) -> tuple:
    """
    Calculate EOS using Asplund abundances and Jorg's chemical_equilibrium_full
    
    Parameters:
    -----------
    temperature : float
        Temperature in K
    total_density : float
        Total particle density in cm^-3
    electron_density_guess : float
        Initial guess for electron density in cm^-3
    m_H : float
        Metallicity [M/H]
        
    Returns:
    --------
    tuple
        (electron_density, number_densities) from chemical_equilibrium_full
    """
    from .statmech.complete_eos import chemical_equilibrium_full
    
    # Get Asplund abundances and ionization energies
    abundances = get_asplund_abundances(m_H)
    ionization_energies = get_asplund_ionization_energies()
    
    # Calculate equilibrium
    return chemical_equilibrium_full(
        temperature, total_density, electron_density_guess,
        abundances, ionization_energies
    )