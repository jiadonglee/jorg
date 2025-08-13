"""
Atomic data constants for stellar spectroscopy.

This module provides comprehensive atomic data matching Korg.jl's atomic_data.jl,
including element symbols, atomic masses, ionization energies, and abundances.
"""

from typing import Dict, List, Optional
import numpy as np

# Element symbols for all elements supported by Korg.jl
ATOMIC_SYMBOLS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U"
]

MAX_ATOMIC_NUMBER = len(ATOMIC_SYMBOLS)

# Atomic number mapping
ATOMIC_NUMBERS: Dict[str, int] = {symbol: i + 1 for i, symbol in enumerate(ATOMIC_SYMBOLS)}

# Atomic masses in atomic mass units (amu), converted to grams in Korg.jl
# From Korg.jl atomic_data.jl
ATOMIC_MASSES_AMU = np.array([
    1.008, 4.003, 6.941, 9.012, 10.81, 12.01, 14.01, 16.00, 19.00, 20.18,
    22.99, 24.31, 26.98, 28.08, 30.97, 32.06, 35.45, 39.95, 39.10, 40.08,
    44.96, 47.90, 50.94, 52.00, 54.94, 55.85, 58.93, 58.71, 63.55, 65.37,
    69.72, 72.59, 74.92, 78.96, 79.90, 83.80, 85.47, 87.62, 88.91, 91.22,
    92.91, 95.94, 98.91, 101.1, 102.9, 106.4, 107.9, 112.4, 114.8, 118.7,
    121.8, 127.6, 126.9, 131.3, 132.9, 137.3, 138.9, 140.1, 140.9, 144.2,
    145.0, 150.4, 152.0, 157.3, 158.9, 162.5, 164.9, 167.3, 168.9, 173.0,
    175.0, 178.5, 181.0, 183.9, 186.2, 190.2, 192.2, 195.1, 197.0, 200.6,
    204.4, 207.2, 209.0, 210.0, 210.0, 222.0, 223.0, 226.0, 227.0, 232.0,
    231.0, 238.0
])

# Convert to grams (Korg.jl multiplies by amu_cgs)
AMU_CGS = 1.6605402e-24  # grams per amu
ATOMIC_MASSES_GRAMS = ATOMIC_MASSES_AMU * AMU_CGS

# Asplund et al. (2009) solar abundances (A_X - 12)
# From Korg.jl atomic_data.jl
ASPLUND_2009_SOLAR_ABUNDANCES = np.array([
    12.00, 10.93, 1.05, 1.38, 2.70, 8.43, 7.83, 8.69, 4.56, 7.93,
    6.24, 7.60, 6.45, 7.51, 5.41, 7.12, 5.50, 6.40, 5.03, 6.34,
    3.15, 4.95, 3.93, 5.64, 5.43, 7.50, 4.99, 6.22, 4.19, 4.56,
    3.04, 3.65, 2.30, 3.34, 2.54, 3.25, 2.52, 2.87, 2.21, 2.58,
    1.46, 1.88, -5.00, 1.75, 0.91, 1.57, 0.94, 1.71, 0.80, 2.04,
    1.01, 2.18, 1.55, 2.24, 1.08, 2.18, 1.10, 1.58, 0.72, 1.42,
    -5.00, 0.96, 0.52, 1.07, 0.30, 1.10, 0.48, 0.92, 0.10, 0.84,
    0.10, 0.85, -0.12, 0.85, 0.26, 1.40, 1.38, 1.62, 0.92, 1.17,
    0.90, 1.75, 0.65, -5.00, -5.00, -5.00, -5.00, -5.00, -5.00, 0.02,
    -5.00, -0.54
])

# Asplund et al. (2020) solar abundances (A_X - 12)
# From Korg.jl atomic_data.jl
ASPLUND_2020_SOLAR_ABUNDANCES = np.array([
    12.00, 10.91, 0.96, 1.38, 2.70, 8.46, 7.83, 8.69, 4.40, 8.06,
    6.22, 7.55, 6.43, 7.51, 5.41, 7.12, 5.31, 6.38, 5.07, 6.30,
    3.14, 4.97, 3.90, 5.62, 5.42, 7.46, 4.94, 6.20, 4.18, 4.56,
    3.02, 3.62, 2.30, 3.34, 2.54, 3.12, 2.32, 2.83, 2.21, 2.59,
    1.47, 1.88, -5.00, 1.75, 0.78, 1.57, 0.96, 1.71, 0.80, 2.02,
    1.01, 2.18, 1.55, 2.22, 1.08, 2.27, 1.11, 1.58, 0.75, 1.42,
    -5.00, 0.95, 0.52, 1.08, 0.31, 1.10, 0.48, 0.93, 0.11, 0.85,
    0.10, 0.85, -0.15, 0.79, 0.26, 1.35, 1.32, 1.61, 0.91, 1.17,
    0.90, 1.75, 0.65, -5.00, -5.00, -5.00, -5.00, -5.00, -5.00, 0.02,
    -5.00, -0.54
])

# Roman numerals for ionization states
ROMAN_NUMERALS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]

# Ionization energies in eV
# This is a simplified version - in practice, Korg.jl loads these from HDF5 files
# These are first ionization energies from NIST
# Fixed IONIZATION_ENERGIES with third ionization energies
# Data from Barklem & Collet 2016 (same as Korg.jl)
# Format: atomic_number: [χ_I, χ_II, χ_III] in eV
IONIZATION_ENERGIES = {
    1: [13.5984, 0.000, 0.000],  # H
    2: [24.5874, 54.418, 0.000],  # He
    3: [5.3917, 75.640, 122.454],  # Li
    4: [9.3227, 18.211, 153.896],  # Be
    5: [8.2980, 25.155, 37.931],  # B
    6: [11.2603, 24.385, 47.888],  # C *** Fixed: added χ_III
    7: [14.5341, 29.601, 47.445],  # N *** Fixed: added χ_III
    8: [13.6181, 35.121, 54.936],  # O *** Fixed: added χ_III
    9: [17.4228, 34.971, 62.708],  # F
    10: [21.5645, 40.963, 63.423],  # Ne
    11: [5.1391, 47.286, 71.620],  # Na
    12: [7.6462, 15.035, 80.144],  # Mg
    13: [5.9858, 18.829, 28.448],  # Al
    14: [8.1517, 16.346, 33.493],  # Si
    15: [10.4867, 19.769, 30.203],  # P
    16: [10.3600, 23.338, 34.856],  # S
    17: [12.9676, 23.814, 39.800],  # Cl
    18: [15.7596, 27.630, 40.735],  # Ar
    19: [4.3407, 31.625, 45.803],  # K
    20: [6.1132, 11.872, 50.913],  # Ca
    21: [6.5615, 12.800, 24.757],  # Sc
    22: [6.8281, 13.575, 27.492],  # Ti
    23: [6.7462, 14.620, 29.311],  # V
    24: [6.7665, 16.486, 30.960],  # Cr
    25: [7.4340, 15.640, 33.668],  # Mn
    26: [7.9025, 16.199, 30.651],  # Fe *** Fixed: added χ_III
    27: [7.8810, 17.084, 33.500],  # Co
    28: [7.6398, 18.169, 35.190],  # Ni
    29: [7.7264, 20.292, 36.841],  # Cu
    30: [9.3942, 17.964, 39.723],  # Zn
    31: [5.9993, 20.514, 30.713],  # Ga
    32: [7.8994, 15.935, 34.225],  # Ge
    33: [9.7886, 18.589, 28.345],  # As
    34: [9.7524, 21.190, 30.820],  # Se
    35: [11.8138, 21.808, 35.985],  # Br
    36: [13.9996, 24.359, 36.950],  # Kr
    37: [4.1771, 27.290, 39.247],  # Rb
    38: [5.6949, 11.030, 42.890],  # Sr
    39: [6.2173, 12.224, 20.524],  # Y
    40: [6.6339, 13.130, 22.987],  # Zr
    41: [6.7589, 14.100, 25.046],  # Nb
    42: [7.0924, 16.160, 27.160],  # Mo
    43: [7.2800, 15.260, 29.540],  # Tc
    44: [7.3605, 16.760, 28.466],  # Ru
    45: [7.4589, 18.078, 31.062],  # Rh
    46: [8.3369, 19.428, 32.927],  # Pd
    47: [7.5762, 21.484, 34.828],  # Ag
    48: [8.9938, 16.908, 37.483],  # Cd
    49: [5.7864, 18.870, 28.032],  # In
    50: [7.3439, 14.632, 30.503],  # Sn
    51: [8.6084, 16.631, 25.321],  # Sb
    52: [9.0096, 18.600, 27.960],  # Te
    53: [10.4513, 19.131, 32.123],  # I
    54: [12.1298, 20.975, 32.123],  # Xe
    55: [3.8939, 23.158, 33.195],  # Cs
    56: [5.2117, 10.004, 19.177],  # Ba
    57: [5.5769, 11.185, 19.177],  # La
    58: [5.5387, 10.850, 20.198],  # Ce
    59: [5.4730, 10.550, 21.624],  # Pr
    60: [5.5250, 10.730, 22.070],  # Nd
    61: [5.5820, 10.900, 22.300],  # Pm
    62: [5.6437, 11.070, 23.424],  # Sm
    63: [5.6704, 11.250, 24.920],  # Eu
    64: [6.1502, 12.090, 20.634],  # Gd
    65: [5.8639, 11.520, 21.910],  # Tb
    66: [5.9389, 11.670, 22.800],  # Dy
    67: [6.0215, 11.800, 22.840],  # Ho
    68: [6.1077, 11.930, 22.740],  # Er
    69: [6.1843, 12.050, 23.681],  # Tm
    70: [6.2542, 12.176, 25.052],  # Yb
    71: [5.4259, 13.900, 20.960],  # Lu
    72: [6.8251, 14.900, 23.320],  # Hf
    73: [7.5496, 16.100, 0.000],  # Ta
    74: [7.8640, 17.600, 0.000],  # W
    75: [7.8335, 17.200, 0.000],  # Re
    76: [8.4382, 17.000, 0.000],  # Os
    77: [8.9670, 18.563, 0.000],  # Ir
    78: [8.9588, 18.561, 0.000],  # Pt
    79: [9.2255, 20.521, 0.000],  # Au
    80: [10.4375, 18.757, 34.490],  # Hg
    81: [6.1082, 20.428, 29.829],  # Tl
    82: [7.4167, 15.028, 31.937],  # Pb
    83: [7.2856, 16.688, 25.558],  # Bi
    84: [8.4170, 19.400, 27.300],  # Po
    85: [9.3175, 20.100, 29.400],  # At
    86: [10.7485, 21.400, 29.400],  # Rn
    87: [4.0727, 22.500, 33.500],  # Fr
    88: [5.2784, 10.147, 31.000],  # Ra
    89: [5.3800, 12.100, 19.000],  # Ac
    90: [6.3067, 11.900, 20.000],  # Th
    91: [5.8900, 0.000, 0.000],  # Pa
    92: [6.1941, 14.500, 0.000],  # U
}

# Isotopic abundances from NIST
# Format: {atomic_number: {mass_number: abundance}}
ISOTOPIC_ABUNDANCES = {
    1: {1: 0.999885, 2: 0.000115},  # H
    2: {3: 0.00000137, 4: 0.99999863},  # He
    3: {6: 0.0759, 7: 0.9241},  # Li
    4: {9: 1.0},  # Be
    5: {10: 0.199, 11: 0.801},  # B
    6: {12: 0.9893, 13: 0.0107},  # C
    7: {14: 0.99636, 15: 0.00364},  # N
    8: {16: 0.99757, 17: 0.00038, 18: 0.00205},  # O
    9: {19: 1.0},  # F
    10: {20: 0.9048, 21: 0.0027, 22: 0.0925},  # Ne
    11: {23: 1.0},  # Na
    12: {24: 0.7899, 25: 0.1000, 26: 0.1101},  # Mg
    13: {27: 1.0},  # Al
    14: {28: 0.92297, 29: 0.04683, 30: 0.0302},  # Si
    15: {31: 1.0},  # P
    16: {32: 0.9499, 33: 0.0075, 34: 0.0425, 36: 0.0001},  # S
    17: {35: 0.7576, 37: 0.2424},  # Cl
    18: {36: 0.00337, 38: 0.00063, 40: 0.996},  # Ar
    19: {39: 0.932581, 40: 0.000117, 41: 0.067302},  # K
    20: {40: 0.96941, 42: 0.00647, 43: 0.00135, 44: 0.02086, 46: 0.00004, 48: 0.00187},  # Ca
    21: {45: 1.0},  # Sc
    22: {46: 0.0825, 47: 0.0744, 48: 0.7372, 49: 0.0541, 50: 0.0518},  # Ti - CRITICAL FOR VALD
    23: {50: 0.00250, 51: 0.99750},  # V
    24: {50: 0.04345, 52: 0.83789, 53: 0.09501, 54: 0.02365},  # Cr
    25: {55: 1.0},  # Mn
    26: {54: 0.05845, 56: 0.91754, 57: 0.02119, 58: 0.00282},  # Fe
    27: {59: 1.0},  # Co
    28: {58: 0.68077, 60: 0.26223, 61: 0.01140, 62: 0.03635, 64: 0.00925},  # Ni
    29: {63: 0.6915, 65: 0.3085},  # Cu
    30: {64: 0.4917, 66: 0.2773, 67: 0.0404, 68: 0.1845, 70: 0.0061},  # Zn
    # Rare earth elements
    57: {138: 0.00090, 139: 0.99910},  # La
    58: {136: 0.00185, 138: 0.00251, 140: 0.88450, 142: 0.11114},  # Ce
    60: {142: 0.27152, 143: 0.12174, 144: 0.23798, 145: 0.08293, 146: 0.17189, 148: 0.05756, 150: 0.05638},  # Nd
    # Add more elements as needed
}


def get_atomic_symbol(atomic_number: int) -> str:
    """Get element symbol from atomic number."""
    if 1 <= atomic_number <= len(ATOMIC_SYMBOLS):
        return ATOMIC_SYMBOLS[atomic_number - 1]
    else:
        return f"El{atomic_number}"


def get_atomic_number(symbol: str) -> int:
    """Get atomic number from element symbol."""
    return ATOMIC_NUMBERS.get(symbol, 0)


def get_atomic_mass(atomic_number: int, unit: str = 'grams') -> float:
    """
    Get atomic mass.
    
    Parameters
    ----------
    atomic_number : int
        Atomic number (1-based)
    unit : str
        Unit for mass ('grams' or 'amu')
        
    Returns
    -------
    float
        Atomic mass in specified units
    """
    if 1 <= atomic_number <= len(ATOMIC_MASSES_AMU):
        if unit == 'amu':
            return ATOMIC_MASSES_AMU[atomic_number - 1]
        else:
            return ATOMIC_MASSES_GRAMS[atomic_number - 1]
    else:
        if unit == 'amu':
            return 1.0
        else:
            return 1.0 * AMU_CGS  # Default to 1 amu if unknown


def get_ionization_energy(atomic_number: int, ionization_stage: int) -> float:
    """
    Get ionization energy in eV.
    
    Parameters
    ----------
    atomic_number : int
        Atomic number (1-based)
    ionization_stage : int
        Ionization stage (0=neutral, 1=singly ionized, etc.)
        
    Returns
    -------
    float
        Ionization energy in eV
    """
    if atomic_number in IONIZATION_ENERGIES:
        energies = IONIZATION_ENERGIES[atomic_number]
        if ionization_stage < len(energies):
            return energies[ionization_stage]
    
    # CRITICAL FIX: Use proper ionization energies instead of hydrogen-like approximation
    from ..statmech.proper_ionization_energies import proper_ionization_energy
    return proper_ionization_energy(atomic_number, ionization_stage + 1)


def get_isotopic_abundance(atomic_number: int, mass_number: int) -> float:
    """Get isotopic abundance fraction."""
    if atomic_number in ISOTOPIC_ABUNDANCES:
        isotopes = ISOTOPIC_ABUNDANCES[atomic_number]
        return isotopes.get(mass_number, 0.0)
    else:
        return 1.0  # Default to 100% if unknown


def get_most_abundant_isotope(atomic_number: int) -> int:
    """Get mass number of most abundant isotope."""
    if atomic_number in ISOTOPIC_ABUNDANCES:
        isotopes = ISOTOPIC_ABUNDANCES[atomic_number]
        return max(isotopes.keys(), key=lambda k: isotopes[k])
    else:
        return 2 * atomic_number  # Rough approximation


def get_solar_abundance(atomic_number: int, year: int = 2009) -> float:
    """
    Get solar abundance A_X - 12 where A_X = log10(N_X/N_H) + 12.
    
    Parameters
    ----------
    atomic_number : int
        Atomic number (1-based)
    year : int
        Year of abundance compilation (2009 or 2020)
        
    Returns
    -------
    float
        Solar abundance A_X - 12
    """
    if year == 2020:
        abundances = ASPLUND_2020_SOLAR_ABUNDANCES
    else:
        abundances = ASPLUND_2009_SOLAR_ABUNDANCES
    
    if 1 <= atomic_number <= len(abundances):
        return abundances[atomic_number - 1]
    else:
        return -5.0  # Default very low abundance


def format_species_name(atomic_number: int, ionization_stage: int) -> str:
    """
    Format species name in spectroscopic notation.
    
    Parameters
    ----------
    atomic_number : int
        Atomic number
    ionization_stage : int
        Ionization stage (0=neutral, 1=singly ionized, etc.)
        
    Returns
    -------
    str
        Species name like "Fe I", "Ca II", etc.
    """
    symbol = get_atomic_symbol(atomic_number)
    if ionization_stage < len(ROMAN_NUMERALS):
        roman = ROMAN_NUMERALS[ionization_stage]
    else:
        roman = str(ionization_stage + 1)
    
    return f"{symbol} {roman}"


def parse_species_name(species_name: str) -> tuple:
    """
    Parse species name to get atomic number and ionization stage.
    
    Parameters
    ----------
    species_name : str
        Species name like "Fe I", "Ca II", etc.
        
    Returns
    -------
    tuple
        (atomic_number, ionization_stage)
    """
    parts = species_name.strip().split()
    if len(parts) != 2:
        raise ValueError(f"Invalid species name: {species_name}")
    
    symbol, roman = parts
    atomic_number = get_atomic_number(symbol)
    if atomic_number == 0:
        raise ValueError(f"Unknown element: {symbol}")
    
    if roman in ROMAN_NUMERALS:
        ionization_stage = ROMAN_NUMERALS.index(roman)
    else:
        # Try to parse as integer
        try:
            ionization_stage = int(roman) - 1
        except ValueError:
            raise ValueError(f"Invalid ionization stage: {roman}")
    
    return atomic_number, ionization_stage


def get_all_atomic_species(max_ionization: int = 2) -> List[tuple]:
    """
    Get all atomic species up to specified ionization stage.
    
    Parameters
    ----------
    max_ionization : int
        Maximum ionization stage to include
        
    Returns
    -------
    List[tuple]
        List of (atomic_number, ionization_stage) tuples
    """
    species = []
    for atomic_number in range(1, MAX_ATOMIC_NUMBER + 1):
        for ionization_stage in range(min(max_ionization + 1, atomic_number + 1)):
            species.append((atomic_number, ionization_stage))
    
    return species


def get_abundances_dict(abundance_type: str = "solar") -> Dict[str, float]:
    """
    Get abundances dictionary for linelist parsing.
    
    Parameters
    ----------
    abundance_type : str
        Type of abundances ("solar", "asplund2009", "asplund2020")
        
    Returns
    -------
    Dict[str, float]
        Dictionary mapping element symbols to number abundances relative to hydrogen
    """
    abundances = {}
    
    for i, symbol in enumerate(ATOMIC_SYMBOLS):
        atomic_number = i + 1
        if abundance_type in ["solar", "asplund2009"]:
            abundance_log = ASPLUND_2009_SOLAR_ABUNDANCES[i]
        else:  # asplund2020
            abundance_log = ASPLUND_2020_SOLAR_ABUNDANCES[i]
        
        # Convert from A_X - 12 to number abundance relative to hydrogen
        # A_X = log10(N_X/N_H) + 12, so N_X/N_H = 10^(A_X - 12)
        if abundance_log > -4.0:  # Skip elements with very low abundances
            abundances[symbol] = 10**(abundance_log - 12.0)
    
    # Add special cases
    abundances['OH'] = abundances.get('O', 4.90e-4)  # OH uses oxygen abundance
    
    # Add molecular species abundances (fixes missing 1,383 lines)
    abundances['CH'] = abundances.get('C', 2.69e-4)   # CH uses carbon abundance
    abundances['CN'] = abundances.get('C', 2.69e-4)   # CN uses carbon abundance  
    abundances['NH'] = abundances.get('N', 6.76e-5)   # NH uses nitrogen abundance
    abundances['C2'] = abundances.get('C', 2.69e-4)   # C2 uses carbon abundance
    abundances['TiO'] = abundances.get('Ti', 8.91e-8) # TiO uses titanium abundance
    abundances['VO'] = abundances.get('V', 8.51e-9)   # VO uses vanadium abundance
    abundances['ZrO'] = abundances.get('Zr', 2.51e-10) # ZrO uses zirconium abundance
    abundances['FeH'] = abundances.get('Fe', 3.16e-5)  # FeH uses iron abundance
    abundances['MgH'] = abundances.get('Mg', 3.98e-5)  # MgH uses magnesium abundance
    
    return abundances


def get_atomic_masses_dict() -> Dict[str, float]:
    """
    Get atomic masses dictionary for linelist parsing.
    
    Returns
    -------
    Dict[str, float]
        Dictionary mapping element symbols to atomic masses in amu
    """
    masses = {}
    
    for i, symbol in enumerate(ATOMIC_SYMBOLS):
        masses[symbol] = ATOMIC_MASSES_AMU[i]
    
    # Add molecular species masses (in amu)
    masses['OH'] = 17.007   # OH molecular mass
    masses['CH'] = 13.018   # CH molecular mass (C + H)
    masses['CN'] = 26.018   # CN molecular mass (C + N)  
    masses['NH'] = 15.015   # NH molecular mass (N + H)
    masses['C2'] = 24.020   # C2 molecular mass (2 * C)
    masses['TiO'] = 63.900  # TiO molecular mass (Ti + O)
    masses['VO'] = 66.940   # VO molecular mass (V + O)
    masses['ZrO'] = 107.220 # ZrO molecular mass (Zr + O)
    masses['FeH'] = 56.858  # FeH molecular mass (Fe + H)
    masses['MgH'] = 25.318  # MgH molecular mass (Mg + H)
    
    return masses


def get_atomic_numbers_dict() -> Dict[str, int]:
    """
    Get atomic numbers dictionary for linelist parsing.
    
    Returns
    -------
    Dict[str, int]
        Dictionary mapping element symbols to atomic numbers
    """
    numbers = ATOMIC_NUMBERS.copy()
    
    # Add molecular species with representative atomic numbers
    # These are used for partition function lookups and species identification
    numbers['OH'] = 8   # OH uses oxygen atomic number for partition functions
    numbers['CH'] = 6   # CH uses carbon atomic number  
    numbers['CN'] = 6   # CN uses carbon atomic number
    numbers['NH'] = 7   # NH uses nitrogen atomic number
    numbers['C2'] = 6   # C2 uses carbon atomic number
    numbers['TiO'] = 22 # TiO uses titanium atomic number
    numbers['VO'] = 23  # VO uses vanadium atomic number
    numbers['ZrO'] = 40 # ZrO uses zirconium atomic number
    numbers['FeH'] = 26 # FeH uses iron atomic number
    numbers['MgH'] = 12 # MgH uses magnesium atomic number
    
    return numbers