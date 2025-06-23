"""
Species identification and parsing for stellar spectroscopy

This module handles atomic and molecular species identification,
matching Korg.jl functionality for species codes and names.
"""

import re
from typing import Dict, Optional, Union


# Element name to atomic number mapping
ELEMENT_NAMES = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
    'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
    'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
    'Pa': 91, 'U': 92
}

# Reverse mapping for atomic number to element name
ELEMENT_SYMBOLS = {v: k for k, v in ELEMENT_NAMES.items()}

# Roman numeral to ionization state mapping
ROMAN_NUMERALS = {
    'I': 0, 'II': 1, 'III': 2, 'IV': 3, 'V': 4, 'VI': 5, 'VII': 6, 'VIII': 7, 'IX': 8, 'X': 9,
    'XI': 10, 'XII': 11, 'XIII': 12, 'XIV': 13, 'XV': 14, 'XVI': 15, 'XVII': 16, 'XVIII': 17,
    'XIX': 18, 'XX': 19
}

# Common molecular species
MOLECULE_CODES = {
    'H2': 100001,    # Hydrogen molecule
    'OH': 108001,    # Hydroxyl
    'CH': 106001,    # Carbon hydride
    'NH': 107001,    # Nitrogen hydride
    'CN': 106007,    # Cyanogen
    'CO': 106008,    # Carbon monoxide
    'SiO': 114008,   # Silicon monoxide
    'TiO': 122008,   # Titanium oxide
    'VO': 123008,    # Vanadium oxide
    'FeH': 126001,   # Iron hydride
    'CaH': 120001,   # Calcium hydride
    'MgH': 112001,   # Magnesium hydride
    'SiH': 114001,   # Silicon hydride
    'C2': 106106,    # Dicarbon
    'N2': 107107,    # Nitrogen molecule
    'O2': 108108,    # Oxygen molecule
}


class Species:
    """Container for species information"""
    
    def __init__(self, element_id: int, ion_state: int = 0, isotope: Optional[int] = None):
        self.element_id = element_id
        self.ion_state = ion_state
        self.isotope = isotope
        self._species_id = None
    
    @property
    def species_id(self) -> int:
        """Get numeric species ID (element_id * 100 + ion_state)"""
        if self._species_id is None:
            self._species_id = self.element_id * 100 + self.ion_state
        return self._species_id
    
    @property
    def symbol(self) -> str:
        """Get element symbol"""
        return ELEMENT_SYMBOLS.get(self.element_id, f"El{self.element_id}")
    
    @property
    def name(self) -> str:
        """Get human-readable species name"""
        symbol = self.symbol
        if self.ion_state == 0:
            ion_name = "I"
        else:
            # Convert to Roman numeral
            ion_name = list(ROMAN_NUMERALS.keys())[list(ROMAN_NUMERALS.values()).index(self.ion_state)]
        
        if self.isotope:
            return f"{self.isotope}{symbol} {ion_name}"
        else:
            return f"{symbol} {ion_name}"
    
    def __repr__(self):
        return f"Species({self.name})"
    
    def __eq__(self, other):
        if isinstance(other, Species):
            return (self.element_id == other.element_id and 
                   self.ion_state == other.ion_state and
                   self.isotope == other.isotope)
        return False


def parse_species(species_str: Union[str, float, int]) -> int:
    """
    Parse various species string formats and return numeric species ID
    
    Supports formats:
    - "Fe I", "Fe II" (element + Roman numeral)
    - "26.00", "26.01" (element.ionization)
    - "FeH", "TiO" (molecules)
    - "606.01213" (molecular codes)
    - 26, 2600, 2601 (numeric codes)
    
    Returns:
    --------
    int : Species ID (element_id * 100 + ion_state)
    """
    
    # Handle numeric inputs
    if isinstance(species_str, (int, float)):
        if species_str < 100:
            # Just element number, assume neutral
            return int(species_str * 100)
        elif species_str < 1000:
            # Already in correct format
            return int(species_str)
        else:
            # Molecular or complex format
            return int(species_str)
    
    species_str = str(species_str).strip()
    
    # Try molecular species first
    if species_str in MOLECULE_CODES:
        return MOLECULE_CODES[species_str]
    
    # Try element + Roman numeral format (e.g., "Fe I", "Ca II")
    match = re.match(r'([A-Z][a-z]?)\s+([IVX]+)', species_str)
    if match:
        element_symbol = match.group(1)
        roman_numeral = match.group(2)
        
        if element_symbol in ELEMENT_NAMES and roman_numeral in ROMAN_NUMERALS:
            element_id = ELEMENT_NAMES[element_symbol]
            ion_state = ROMAN_NUMERALS[roman_numeral]
            return element_id * 100 + ion_state
    
    # Try element + numeric ionization format (e.g., "Na 1", "Ca 2") 
    match = re.match(r'([A-Z][a-z]?)\s+(\d+)', species_str)
    if match:
        element_symbol = match.group(1)
        ion_number = int(match.group(2))
        
        if element_symbol in ELEMENT_NAMES:
            element_id = ELEMENT_NAMES[element_symbol]
            ion_state = ion_number - 1  # Convert to 0-based (neutral = 0)
            return element_id * 100 + ion_state
    
    # Try element.ionization format (e.g., "26.00", "26.01")
    match = re.match(r'(\d+)\.(\d+)', species_str)
    if match:
        element_id = int(match.group(1))
        ion_state = int(match.group(2))
        return element_id * 100 + ion_state
    
    # Try molecular formula (e.g., "FeH", "TiO")
    if len(species_str) >= 2 and species_str.isalpha():
        # Try to parse as molecule
        molecular_id = parse_molecule(species_str)
        if molecular_id:
            return molecular_id
    
    # Try pure numeric format
    try:
        numeric_value = float(species_str)
        if numeric_value < 100:
            return int(numeric_value * 100)  # Assume neutral
        else:
            return int(numeric_value)
    except ValueError:
        pass
    
    # Fallback: try just element symbol
    if species_str in ELEMENT_NAMES:
        return ELEMENT_NAMES[species_str] * 100  # Assume neutral
    
    raise ValueError(f"Could not parse species: {species_str}")


def parse_molecule(formula: str) -> Optional[int]:
    """Parse molecular formula to species ID"""
    
    # Simple two-element molecules
    if len(formula) == 2:
        if formula[0] in ELEMENT_NAMES and formula[1] in ELEMENT_NAMES:
            elem1 = ELEMENT_NAMES[formula[0]]
            elem2 = ELEMENT_NAMES[formula[1]]
            # Create molecular ID: larger element * 1000 + smaller element
            return max(elem1, elem2) * 1000 + min(elem1, elem2)
    
    elif len(formula) == 3:
        # Three character molecules like "FeH", "TiO"
        if formula[:2] in ELEMENT_NAMES and formula[2] in ELEMENT_NAMES:
            elem1 = ELEMENT_NAMES[formula[:2]]
            elem2 = ELEMENT_NAMES[formula[2]]
            return max(elem1, elem2) * 1000 + min(elem1, elem2)
        elif formula[0] in ELEMENT_NAMES and formula[1:] in ELEMENT_NAMES:
            elem1 = ELEMENT_NAMES[formula[0]]
            elem2 = ELEMENT_NAMES[formula[1:]]
            return max(elem1, elem2) * 1000 + min(elem1, elem2)
    
    # Check predefined molecules
    if formula in MOLECULE_CODES:
        return MOLECULE_CODES[formula]
    
    return None


def species_id_to_name(species_id: int) -> str:
    """Convert numeric species ID to human-readable name"""
    
    if species_id > 100000:
        # Molecular species
        return f"Molecule_{species_id}"
    
    element_id = species_id // 100
    ion_state = species_id % 100
    
    if element_id in ELEMENT_SYMBOLS:
        symbol = ELEMENT_SYMBOLS[element_id]
        if ion_state < len(ROMAN_NUMERALS):
            roman = list(ROMAN_NUMERALS.keys())[ion_state]
            return f"{symbol} {roman}"
        else:
            return f"{symbol} +{ion_state}"
    else:
        return f"Element_{element_id}_{ion_state}"


def get_element_mass(element_id: int) -> float:
    """Get atomic mass for element (in atomic mass units)"""
    
    # Standard atomic masses (most abundant isotope or average)
    ATOMIC_MASSES = {
        1: 1.008, 2: 4.003, 3: 6.941, 4: 9.012, 5: 10.811, 6: 12.011, 7: 14.007, 8: 15.999,
        9: 18.998, 10: 20.180, 11: 22.990, 12: 24.305, 13: 26.982, 14: 28.086, 15: 30.974,
        16: 32.065, 17: 35.453, 18: 39.948, 19: 39.098, 20: 40.078, 21: 44.956, 22: 47.867,
        23: 50.942, 24: 51.996, 25: 54.938, 26: 55.845, 27: 58.933, 28: 58.693, 29: 63.546,
        30: 65.38, 31: 69.723, 32: 72.64, 33: 74.922, 34: 78.96, 35: 79.904, 36: 83.798,
        37: 85.468, 38: 87.62, 39: 88.906, 40: 91.224, 41: 92.906, 42: 95.96, 43: 98.0,
        44: 101.07, 45: 102.906, 46: 106.42, 47: 107.868, 48: 112.411, 49: 114.818, 50: 118.710,
        51: 121.760, 52: 127.60, 53: 126.905, 54: 131.293, 55: 132.905, 56: 137.327,
        57: 138.905, 58: 140.116, 59: 140.908, 60: 144.242, 61: 145.0, 62: 150.36, 63: 151.964,
        64: 157.25, 65: 158.925, 66: 162.500, 67: 164.930, 68: 167.259, 69: 168.934, 70: 173.054,
        71: 174.967, 72: 178.49, 73: 180.948, 74: 183.84, 75: 186.207, 76: 190.23, 77: 192.217,
        78: 195.084, 79: 196.967, 80: 200.59, 81: 204.383, 82: 207.2, 83: 208.980, 84: 209.0,
        85: 210.0, 86: 222.0, 87: 223.0, 88: 226.0, 89: 227.0, 90: 232.038, 91: 231.036, 92: 238.029
    }
    
    return ATOMIC_MASSES.get(element_id, 1.0)  # Default to 1 amu if unknown


def is_molecule(species_id: int) -> bool:
    """Check if species ID represents a molecule"""
    return species_id > 100000 or species_id in MOLECULE_CODES.values()


def get_species_info(species_id: int) -> Dict:
    """Get comprehensive information about a species"""
    
    if is_molecule(species_id):
        return {
            'species_id': species_id,
            'type': 'molecule',
            'name': species_id_to_name(species_id),
            'element_id': None,
            'ion_state': None,
            'mass_amu': 1.0  # Would need molecular mass calculation
        }
    else:
        element_id = species_id // 100
        ion_state = species_id % 100
        
        return {
            'species_id': species_id,
            'type': 'atom',
            'name': species_id_to_name(species_id),
            'element_id': element_id,
            'ion_state': ion_state,
            'element_symbol': ELEMENT_SYMBOLS.get(element_id, f"El{element_id}"),
            'mass_amu': get_element_mass(element_id)
        }