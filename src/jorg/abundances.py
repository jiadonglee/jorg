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


# Alpha elements (O, Ne, Mg, Si, S, Ar, Ca, Ti) - elements with even atomic numbers from 8-22
DEFAULT_ALPHA_ELEMENTS = [8, 10, 12, 14, 16, 18, 20, 22]  # O to Ti

# Element symbol to atomic number mapping
ELEMENT_TO_Z = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
    'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
    'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
    'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38,
    'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46,
    'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54,
    'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62,
    'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
    'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
    'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86,
    'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92
}

# Asplund et al. 2020 solar abundances - EXACT MATCH to Korg.jl atomic_data.jl
# Source: /Users/jdli/Project/Korg.jl/src/atomic_data.jl lines 42-52
ASPLUND_2020_SOLAR_ABUNDANCES = jnp.array([
    12.00,  # 1 H
    10.91,  # 2 He  
    0.96,   # 3 Li    ← CORRECTED: was 1.05, now matches Korg.jl
    1.38,   # 4 Be
    2.70,   # 5 B
    8.46,   # 6 C
    7.83,   # 7 N
    8.69,   # 8 O
    4.40,   # 9 F
    8.06,   # 10 Ne
    6.22,   # 11 Na   ← CORRECTED: was 6.24, now matches Korg.jl
    7.55,   # 12 Mg   ← CORRECTED: was 7.60, now matches Korg.jl
    6.43,   # 13 Al   ← CORRECTED: was 6.45, now matches Korg.jl
    7.51,   # 14 Si
    5.41,   # 15 P
    7.12,   # 16 S
    5.31,   # 17 Cl   ← CORRECTED: was 5.50, now matches Korg.jl
    6.38,   # 18 Ar   ← CORRECTED: was 6.40, now matches Korg.jl
    5.07,   # 19 K    ← CORRECTED: was 5.03, now matches Korg.jl
    6.30,   # 20 Ca   ← CORRECTED: was 6.34, now matches Korg.jl
    3.14,   # 21 Sc   ← CORRECTED: was 3.15, now matches Korg.jl
    4.97,   # 22 Ti   ← CORRECTED: was 4.95, now matches Korg.jl
    3.90,   # 23 V    ← CORRECTED: was 3.93, now matches Korg.jl
    5.62,   # 24 Cr   ← CORRECTED: was 5.64, now matches Korg.jl
    5.42,   # 25 Mn   ← CORRECTED: was 5.43, now matches Korg.jl
    7.46,   # 26 Fe   ← CORRECTED: was 7.50, now matches Korg.jl
    4.94,   # 27 Co   ← CORRECTED: was 4.99, now matches Korg.jl
    6.20,   # 28 Ni   ← CORRECTED: was 6.22, now matches Korg.jl
    4.18,   # 29 Cu   ← CORRECTED: was 4.19, now matches Korg.jl
    4.56,   # 30 Zn
    3.02,   # 31 Ga   ← CORRECTED: was 3.04, now matches Korg.jl
    3.62,   # 32 Ge   ← CORRECTED: was 3.65, now matches Korg.jl
    2.30,   # 33 As
    3.34,   # 34 Se
    2.54,   # 35 Br
    3.12,   # 36 Kr   ← CORRECTED: was 3.25, now matches Korg.jl
    2.32,   # 37 Rb   ← CORRECTED: was 2.52, now matches Korg.jl
    2.83,   # 38 Sr   ← CORRECTED: was 2.87, now matches Korg.jl
    2.21,   # 39 Y
    2.59,   # 40 Zr   ← CORRECTED: was 2.58, now matches Korg.jl
    1.47,   # 41 Nb   ← CORRECTED: was 1.46, now matches Korg.jl
    1.88,   # 42 Mo
    -5.00,  # 43 Tc   ← CORRECTED: was 0.00, now matches Korg.jl (-5.00 for no stable isotopes)
    1.75,   # 44 Ru
    0.78,   # 45 Rh   ← CORRECTED: was 0.91, now matches Korg.jl
    1.57,   # 46 Pd
    0.96,   # 47 Ag
    1.71,   # 48 Cd
    0.80,   # 49 In
    2.02,   # 50 Sn   ← CORRECTED: was 2.04, now matches Korg.jl
    1.01,   # 51 Sb
    2.18,   # 52 Te
    1.55,   # 53 I
    2.22,   # 54 Xe   ← CORRECTED: was 2.24, now matches Korg.jl
    1.08,   # 55 Cs
    2.27,   # 56 Ba   ← CORRECTED: was 2.18, now matches Korg.jl
    1.11,   # 57 La   ← CORRECTED: was 1.10, now matches Korg.jl
    1.58,   # 58 Ce
    0.75,   # 59 Pr   ← CORRECTED: was 0.72, now matches Korg.jl
    1.42,   # 60 Nd
    -5.00,  # 61 Pm   ← CORRECTED: was 0.00, now matches Korg.jl (-5.00 for no stable isotopes)
    0.95,   # 62 Sm   ← CORRECTED: was 0.96, now matches Korg.jl
    0.52,   # 63 Eu
    1.08,   # 64 Gd   ← CORRECTED: was 1.07, now matches Korg.jl
    0.31,   # 65 Tb   ← CORRECTED: was 0.30, now matches Korg.jl
    1.10,   # 66 Dy
    0.48,   # 67 Ho
    0.93,   # 68 Er   ← CORRECTED: was 0.92, now matches Korg.jl
    0.11,   # 69 Tm   ← CORRECTED: was 0.10, now matches Korg.jl
    0.85,   # 70 Yb   ← CORRECTED: was 0.84, now matches Korg.jl
    0.10,   # 71 Lu
    0.85,   # 72 Hf
    -0.15,  # 73 Ta   ← CORRECTED: was -0.12, now matches Korg.jl
    0.79,   # 74 W    ← CORRECTED: was 0.85, now matches Korg.jl
    0.26,   # 75 Re
    1.35,   # 76 Os   ← CORRECTED: was 1.40, now matches Korg.jl
    1.32,   # 77 Ir   ← CORRECTED: was 1.38, now matches Korg.jl
    1.61,   # 78 Pt   ← CORRECTED: was 1.62, now matches Korg.jl
    0.91,   # 79 Au   ← CORRECTED: was 0.92, now matches Korg.jl
    1.17,   # 80 Hg
    0.92,   # 81 Tl   ← CORRECTED: was 0.90, now matches Korg.jl
    1.95,   # 82 Pb
    0.65,   # 83 Bi
    -5.00,  # 84 Po   ← CORRECTED: was 0.00, now matches Korg.jl (-5.00 for no stable isotopes)
    -5.00,  # 85 At   ← CORRECTED: was 0.00, now matches Korg.jl (-5.00 for no stable isotopes)
    -5.00,  # 86 Rn   ← CORRECTED: was 0.00, now matches Korg.jl (-5.00 for no stable isotopes)
    -5.00,  # 87 Fr   ← CORRECTED: was 0.00, now matches Korg.jl (-5.00 for no stable isotopes)
    -5.00,  # 88 Ra   ← CORRECTED: was 0.00, now matches Korg.jl (-5.00 for no stable isotopes)
    -5.00,  # 89 Ac   ← CORRECTED: was 0.00, now matches Korg.jl (-5.00 for no stable isotopes)
    0.03,   # 90 Th   ← CORRECTED: was 0.02, now matches Korg.jl
    -5.00,  # 91 Pa   ← CORRECTED: was 0.00, now matches Korg.jl (-5.00 for no stable isotopes)
    -0.54   # 92 U
])


def format_abundances(default_metals_H=0.0, default_alpha_H=None, abundances=None,
                     solar_relative=True, solar_abundances=None, 
                     alpha_elements=None):
    """
    Format abundances as 92-element A(X) array matching Korg.jl format_A_X()
    
    Parameters
    ----------
    default_metals_H : float, default 0.0
        Metallicity [metals/H] - log10 solar-relative abundance of elements heavier than He
    default_alpha_H : float, optional
        Alpha element enhancement [α/H]. If None, defaults to default_metals_H
    abundances : dict, optional
        Individual element abundances. Keys can be atomic numbers (int) or 
        element symbols (str). Values are [X/H] if solar_relative=True, 
        or A(X) values if solar_relative=False
    solar_relative : bool, default True
        If True, interpret abundances as [X/H] format. If False, as A(X) format
    solar_abundances : array_like, optional
        Solar abundance reference. Defaults to Asplund 2020
    alpha_elements : array_like, optional  
        List of atomic numbers of alpha elements. Defaults to O, Ne, Mg, Si, S, Ar, Ca, Ti
        
    Returns
    -------
    jnp.ndarray
        92-element array of A(X) abundances (log10(N_X/N_H) + 12)
        
    Notes
    -----
    This function exactly matches Korg.jl's format_A_X() behavior:
    - Returns 92-element vector with A(H) = 12.0
    - Supports separate alpha enhancement from overall metallicity
    - Individual elements override defaults
    - Handles both atomic numbers and element symbols as keys
    
    Examples
    --------
    >>> # Basic metallicity scaling
    >>> A_X = format_abundances(default_metals_H=-0.5)
    
    >>> # Alpha-enhanced metal-poor star  
    >>> A_X = format_abundances(default_metals_H=-1.0, default_alpha_H=-0.5)
    
    >>> # Individual element abundances
    >>> A_X = format_abundances(default_metals_H=-0.3, abundances={'Fe': -0.5, 'C': 0.2})
    >>> A_X = format_abundances(abundances={26: -0.5, 6: 0.2})  # Same as above
    """
    # Set defaults
    if default_alpha_H is None:
        default_alpha_H = default_metals_H
    if abundances is None:
        abundances = {}
    if solar_abundances is None:
        solar_abundances = ASPLUND_2020_SOLAR_ABUNDANCES  
    if alpha_elements is None:
        alpha_elements = DEFAULT_ALPHA_ELEMENTS
        
    # Initialize with solar abundances
    A_X = jnp.array(solar_abundances[:92])  # Ensure exactly 92 elements
    
    # Apply metallicity scaling to metals (Z > 2)
    for Z in range(3, 93):  # Li to U
        if Z-1 < len(solar_abundances):
            if Z in alpha_elements:
                # Alpha elements get alpha enhancement
                A_X = A_X.at[Z-1].set(solar_abundances[Z-1] + default_alpha_H)
            else:
                # Other metals get metallicity scaling
                A_X = A_X.at[Z-1].set(solar_abundances[Z-1] + default_metals_H)
    
    # Apply individual element abundances (override defaults)
    clean_abundances = {}
    for key, value in abundances.items():
        if isinstance(key, str):
            # Convert element symbol to atomic number
            if key in ELEMENT_TO_Z:
                Z = ELEMENT_TO_Z[key]
                clean_abundances[Z] = float(value)
            else:
                raise ValueError(f"Unknown element symbol: {key}")
        elif isinstance(key, int):
            # Direct atomic number
            if 1 <= key <= 92:
                clean_abundances[key] = float(value)  
            else:
                raise ValueError(f"Atomic number must be 1-92, got {key}")
        else:
            raise ValueError(f"Abundance keys must be int (atomic number) or str (element symbol), got {type(key)}")
    
    # Apply individual abundances
    for Z, abundance_value in clean_abundances.items():
        if solar_relative:
            # [X/H] format: add to solar value
            A_X = A_X.at[Z-1].set(solar_abundances[Z-1] + abundance_value)
        else:
            # A(X) format: use directly
            A_X = A_X.at[Z-1].set(abundance_value)
    
    # Ensure A(H) = 12.0 (required by Korg.jl)
    A_X = A_X.at[0].set(12.0)
    
    return A_X


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
    # Convert to atomic number keys  
    abundances = {}
    for element, log_abundance in ASPLUND_2009.items():
        if element in ELEMENT_TO_Z:
            Z = ELEMENT_TO_Z[element]
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