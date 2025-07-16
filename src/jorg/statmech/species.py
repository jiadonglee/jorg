"""
Species and Formula types for chemical representation in stellar atmosphere modeling.

Based exactly on Korg.jl's Species and Formula implementation.
"""

import re
from typing import List, Tuple, Union, Dict, Optional
from dataclasses import dataclass, field
from functools import cached_property
import numpy as np

# Atomic data (up to Uranium, Z=92)
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

MAX_ATOMIC_NUMBER = len(ATOMIC_SYMBOLS)  # 92
MAX_ATOMS_PER_MOLECULE = 6

# Create atomic number lookup
ATOMIC_NUMBERS = {symbol: i + 1 for i, symbol in enumerate(ATOMIC_SYMBOLS)}

# Atomic masses in atomic mass units (AMU)
ATOMIC_MASSES = np.array([
    1.008, 4.003, 6.94, 9.012, 10.81, 12.01, 14.01, 16.00, 19.00, 20.18,
    22.99, 24.31, 26.98, 28.09, 30.97, 32.06, 35.45, 39.95, 39.10, 40.08,
    44.96, 47.87, 50.94, 52.00, 54.94, 55.85, 58.93, 58.69, 63.55, 65.38,
    69.72, 72.63, 74.92, 78.97, 79.90, 83.80, 85.47, 87.62, 88.91, 91.22,
    92.91, 95.95, 98.00, 101.1, 102.9, 106.4, 107.9, 112.4, 114.8, 118.7,
    121.8, 127.6, 126.9, 131.3, 132.9, 137.3, 138.9, 140.1, 140.9, 144.2,
    145.0, 150.4, 152.0, 157.3, 158.9, 162.5, 164.9, 167.3, 168.9, 173.1,
    175.0, 178.5, 181.0, 183.8, 186.2, 190.2, 192.2, 195.1, 197.0, 200.6,
    204.4, 207.2, 209.0, 209.0, 210.0, 222.0, 223.0, 226.0, 227.0, 232.0,
    231.0, 238.0
])

# Roman numerals for ionization states
ROMAN_NUMERALS = {
    0: "I", 1: "II", 2: "III", 3: "IV", 4: "V", 5: "VI", 6: "VII", 7: "VIII"
}
ROMAN_TO_CHARGE = {v: k for k, v in ROMAN_NUMERALS.items()}


@dataclass(frozen=True)
class Formula:
    """
    Represents a chemical formula (atom or molecule), irrespective of charge.
    
    Exactly matches Korg.jl's Formula struct behavior.
    
    Attributes:
        atoms: Tuple of atomic numbers (up to 6 atoms), sorted
    """
    atoms: Tuple[int, ...] = field(default_factory=tuple)
    
    def __post_init__(self):
        # Validate and sort atoms
        if len(self.atoms) > MAX_ATOMS_PER_MOLECULE:
            raise ValueError(f"Too many atoms: {len(self.atoms)} > {MAX_ATOMS_PER_MOLECULE}")
        
        for atom in self.atoms:
            if not (1 <= atom <= MAX_ATOMIC_NUMBER):
                raise ValueError(f"Invalid atomic number: {atom}")
        
        # Sort atoms and pad to consistent size for hashing
        sorted_atoms = tuple(sorted(self.atoms))
        object.__setattr__(self, 'atoms', sorted_atoms)
    
    @classmethod
    def from_atomic_number(cls, Z: int) -> 'Formula':
        """Create Formula for single atom."""
        return cls((Z,))
    
    @classmethod
    def from_atomic_numbers(cls, Zs: List[int]) -> 'Formula':
        """Create Formula from list of atomic numbers."""
        return cls(tuple(Zs))
    
    @classmethod
    def from_string(cls, formula_str: str) -> 'Formula':
        """Parse Formula from string representation."""
        # Handle simple atomic symbols first
        if formula_str in ATOMIC_NUMBERS:
            return cls.from_atomic_number(ATOMIC_NUMBERS[formula_str])
        
        # Handle MOOG-style numeric codes (e.g., "0608" for CO)
        if formula_str.isdigit() and len(formula_str) == 4:
            try:
                # MOOG format: XXYY where XX and YY are atomic numbers
                z1 = int(formula_str[:2])
                z2 = int(formula_str[2:])
                if z1 > 0 and z2 > 0 and z1 <= MAX_ATOMIC_NUMBER and z2 <= MAX_ATOMIC_NUMBER:
                    # Order atoms by atomic number (convention)
                    if z1 <= z2:
                        atoms = [z1, z2]
                    else:
                        atoms = [z2, z1]
                    return cls.from_atomic_numbers(atoms)
            except (ValueError, IndexError):
                pass
        
        # Handle molecular formulas (simplified)
        atoms = []
        
        # Simple parser for common molecules
        if formula_str == "H2":
            atoms = [1, 1]
        elif formula_str == "OH":
            atoms = [1, 8]
        elif formula_str == "CO":
            atoms = [6, 8]
        elif formula_str == "H2O":
            atoms = [1, 1, 8]
        elif formula_str == "CH":
            atoms = [1, 6]
        elif formula_str == "NH":
            atoms = [1, 7]
        elif formula_str == "CN":
            atoms = [6, 7]
        elif formula_str == "C2":
            atoms = [6, 6]
        elif formula_str == "N2":
            atoms = [7, 7]
        elif formula_str == "O2":
            atoms = [8, 8]
        elif formula_str == "NO":
            atoms = [7, 8]
        elif formula_str == "SiO":
            atoms = [8, 14]
        elif formula_str == "TiO":
            atoms = [8, 22]
        elif formula_str == "VO":
            atoms = [8, 23]
        elif formula_str == "CaH":
            atoms = [1, 20]
        elif formula_str == "MgH":
            atoms = [1, 12]
        elif formula_str == "FeH":
            atoms = [1, 26]
        else:
            # Try to parse as element symbol
            if formula_str in ATOMIC_NUMBERS:
                atoms = [ATOMIC_NUMBERS[formula_str]]
            else:
                # Try general molecular parser for Barklem & Collet data
                atoms = cls._parse_general_molecular_formula(formula_str)
                if not atoms:
                    raise ValueError(f"Cannot parse formula: {formula_str}. "
                                   f"Supported formats: element symbols (e.g., 'H', 'Fe'), "
                                   f"molecule names (e.g., 'CO', 'H2O'), "
                                   f"or MOOG-style codes (e.g., '0608' for CO)")
        
        return cls.from_atomic_numbers(atoms)
    
    @classmethod
    def _parse_general_molecular_formula(cls, formula_str: str) -> List[int]:
        """
        Parse general molecular formula for Barklem & Collet data.
        
        Handles cases like:
        - "ClCl" -> [Cl, Cl]
        - "ONa" -> [Na, O] (reordered)
        - "OO" -> [O, O]
        - "HC" -> [H, C]
        
        Returns list of atomic numbers, or empty list if parsing fails.
        """
        atoms = []
        
        # Use regex to find all element symbols (with optional numbers)
        pattern = r'([A-Z][a-z]?)(\d*)'
        matches = re.findall(pattern, formula_str)
        
        if not matches:
            return []
        
        for element, count_str in matches:
            if element not in ATOMIC_NUMBERS:
                return []  # Unknown element
            
            count = int(count_str) if count_str else 1
            Z = ATOMIC_NUMBERS[element]
            atoms.extend([Z] * count)
        
        if len(atoms) == 0:
            return []
        
        # Sort atoms by atomic number for consistent representation
        atoms.sort()
        return atoms
    
    @cached_property
    def is_atom(self) -> bool:
        """True if this is a single atom."""
        return len(self.atoms) == 1
    
    @cached_property
    def is_molecule(self) -> bool:
        """True if this is a molecule (multiple atoms)."""
        return len(self.atoms) > 1
    
    @cached_property
    def n_atoms(self) -> int:
        """Total number of atoms."""
        return len(self.atoms)
    
    @cached_property
    def mass(self) -> float:
        """Molecular mass in AMU."""
        return sum(ATOMIC_MASSES[Z - 1] for Z in self.atoms)
    
    def get_atoms(self) -> Tuple[int, ...]:
        """Get atomic numbers of all atoms."""
        return self.atoms
    
    def get_atom(self) -> int:
        """Get atomic number (for single atoms only)."""
        if not self.is_atom:
            raise ValueError("get_atom() only valid for single atoms")
        return self.atoms[0]
    
    def __str__(self) -> str:
        if self.is_atom:
            return ATOMIC_SYMBOLS[self.atoms[0] - 1]
        else:
            # Simple molecular representation
            atom_counts = {}
            for Z in self.atoms:
                symbol = ATOMIC_SYMBOLS[Z - 1]
                atom_counts[symbol] = atom_counts.get(symbol, 0) + 1
            
            parts = []
            for symbol in sorted(atom_counts.keys()):
                count = atom_counts[symbol]
                if count == 1:
                    parts.append(symbol)
                else:
                    parts.append(f"{symbol}{count}")
            return "".join(parts)
    
    def __eq__(self, other) -> bool:
        if isinstance(other, int):
            # Allow comparison with atomic number for single atoms
            return self.is_atom and self.atoms[0] == other
        return isinstance(other, Formula) and self.atoms == other.atoms
    
    def __hash__(self) -> int:
        return hash(self.atoms)


@dataclass(frozen=True)
class Species:
    """
    Represents a chemical species (Formula + charge state).
    
    Exactly matches Korg.jl's Species struct behavior.
    
    Attributes:
        formula: Chemical formula
        charge: Ionization state (0=neutral, 1=singly ionized, etc.)
    """
    formula: Formula
    charge: int = 0
    
    def __post_init__(self):
        if not isinstance(self.formula, Formula):
            # Auto-convert from atomic number
            if isinstance(self.formula, int):
                object.__setattr__(self, 'formula', Formula.from_atomic_number(self.formula))
            else:
                raise TypeError("formula must be Formula or int")
    
    @classmethod
    def from_atomic_number(cls, Z: int, charge: int = 0) -> 'Species':
        """Create Species for single atom with charge."""
        return cls(Formula.from_atomic_number(Z), charge)
    
    @classmethod
    def from_string(cls, species_str: str) -> 'Species':
        """Parse Species from string representation."""
        species_str = species_str.strip()
        
        # Handle Roman numeral notation (e.g., "H I", "Fe II")
        roman_match = re.match(r'^([A-Z][a-z]?)\s+([IVX]+)$', species_str)
        if roman_match:
            element, roman = roman_match.groups()
            if element in ATOMIC_NUMBERS and roman in ROMAN_TO_CHARGE:
                Z = ATOMIC_NUMBERS[element]
                charge = ROMAN_TO_CHARGE[roman]
                return cls.from_atomic_number(Z, charge)
        
        # Handle charge notation (e.g., "H+", "OH-")
        if species_str.endswith('+'):
            formula_str = species_str[:-1]
            charge = 1
        elif species_str.endswith('-'):
            formula_str = species_str[:-1]
            charge = -1
        else:
            formula_str = species_str
            charge = 0
        
        # Handle numeric codes (e.g., "01.00" for H I)
        numeric_match = re.match(r'^(\d{2})\.(\d{2})$', species_str)
        if numeric_match:
            Z = int(numeric_match.group(1))
            charge = int(numeric_match.group(2))
            return cls.from_atomic_number(Z, charge)
        
        formula = Formula.from_string(formula_str)
        return cls(formula, charge)
    
    @cached_property
    def is_atom(self) -> bool:
        """True if this is an atomic species."""
        return self.formula.is_atom
    
    @cached_property
    def is_molecule(self) -> bool:
        """True if this is a molecular species."""
        return self.formula.is_molecule
    
    @cached_property
    def is_neutral(self) -> bool:
        """True if this is a neutral species."""
        return self.charge == 0
    
    @cached_property
    def is_ion(self) -> bool:
        """True if this is an ionized species."""
        return self.charge != 0
    
    @cached_property
    def mass(self) -> float:
        """Molecular mass in AMU."""
        return self.formula.mass
    
    @cached_property
    def n_atoms(self) -> int:
        """Total number of atoms."""
        return self.formula.n_atoms
    
    def get_atoms(self) -> Tuple[int, ...]:
        """Get atomic numbers of all atoms."""
        return self.formula.get_atoms()
    
    def get_atom(self) -> int:
        """Get atomic number (for atomic species only)."""
        return self.formula.get_atom()
    
    def __str__(self) -> str:
        if self.is_atom:
            # Use Roman numerals for atoms
            element = ATOMIC_SYMBOLS[self.formula.atoms[0] - 1]
            if 0 <= self.charge < len(ROMAN_NUMERALS):
                roman = ROMAN_NUMERALS[self.charge]
                return f"{element} {roman}"
            else:
                return f"{element}({self.charge:+d})"
        else:
            # Use +/- notation for molecules
            mol_str = str(self.formula)
            if self.charge == 0:
                return mol_str
            elif self.charge == 1:
                return f"{mol_str}+"
            elif self.charge == -1:
                return f"{mol_str}-"
            else:
                return f"{mol_str}({self.charge:+d})"
    
    def __eq__(self, other) -> bool:
        return (isinstance(other, Species) and 
                self.formula == other.formula and 
                self.charge == other.charge)
    
    def __lt__(self, other) -> bool:
        """Define ordering for Species objects (needed for JAX sorting)."""
        if not isinstance(other, Species):
            return NotImplemented
        
        # First compare by formula (atomic numbers)
        if self.formula.atoms != other.formula.atoms:
            return self.formula.atoms < other.formula.atoms
        
        # Then compare by charge
        return self.charge < other.charge
    
    def __le__(self, other) -> bool:
        return self == other or self < other
    
    def __gt__(self, other) -> bool:
        return not (self <= other)
    
    def __ge__(self, other) -> bool:
        return not (self < other)
    
    def __hash__(self) -> int:
        return hash((self.formula, self.charge))


def all_atomic_species() -> List[Species]:
    """Generate all atomic species (Z=1-92, charge=0-2)."""
    species = []
    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        max_charge = min(2, Z)  # Don't exceed number of electrons
        for charge in range(max_charge + 1):
            species.append(Species.from_atomic_number(Z, charge))
    return species


def parse_species(s: Union[str, int, Species]) -> Species:
    """
    Parse species from various input formats.
    
    Parameters:
        s: String, atomic number, or Species object
        
    Returns:
        Species object
    """
    if isinstance(s, Species):
        return s
    elif isinstance(s, int):
        return Species.from_atomic_number(s, 0)
    elif isinstance(s, str):
        return Species.from_string(s)
    else:
        raise TypeError(f"Cannot parse species from {type(s)}")


# Convenience functions matching Korg.jl naming
def create_formula(arg):
    """Create Formula (matching Korg.jl constructor)."""
    if isinstance(arg, int):
        return Formula.from_atomic_number(arg)
    elif isinstance(arg, (list, tuple)):
        return Formula.from_atomic_numbers(list(arg))
    elif isinstance(arg, str):
        return Formula.from_string(arg)
    else:
        raise TypeError(f"Cannot create Formula from {type(arg)}")


def create_species(formula, charge=0):
    """Create Species (matching Korg.jl constructor)."""
    if isinstance(formula, int):
        return Species.from_atomic_number(formula, charge)
    elif isinstance(formula, Formula):
        return Species(formula, charge)
    elif isinstance(formula, str):
        return Species.from_string(formula)
    else:
        return Species(create_formula(formula), charge)


# Additional utilities
def get_mass(species) -> float:
    """Get mass of species in AMU."""
    if isinstance(species, str):
        # Convert string to Species object
        species = Species.from_string(species)
    return species.mass


def ismolecule(species: Species) -> bool:
    """Check if species is a molecule."""
    return species.is_molecule


def n_atoms(species: Species) -> int:
    """Get number of atoms in species."""
    return species.formula.n_atoms


def get_atoms(species: Species) -> Tuple[int, ...]:
    """Get atomic numbers of all atoms in species."""
    return species.get_atoms()


def get_atom(species: Species) -> int:
    """Get atomic number (for atomic species only)."""
    return species.get_atom()