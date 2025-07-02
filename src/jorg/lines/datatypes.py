"""
Line data structures for stellar spectral synthesis.

This module provides line data structures that strictly follow Korg.jl conventions
for maximum compatibility and consistency.
"""

from typing import NamedTuple, Union, Tuple, List
from dataclasses import dataclass
import numpy as np


MAX_ATOMS_PER_MOLECULE = 6


@dataclass(frozen=True)
class Formula:
    """
    Represents molecular formula following Korg.jl structure.
    
    Stores atomic numbers in a fixed-size array, matching Korg.jl's
    SVector{6,UInt8} implementation.
    """
    atoms: Tuple[int, ...]  # Up to MAX_ATOMS_PER_MOLECULE atoms, sorted by atomic number
    
    def __post_init__(self):
        # Ensure atoms tuple has exactly MAX_ATOMS_PER_MOLECULE elements (padded with zeros)
        if len(self.atoms) > MAX_ATOMS_PER_MOLECULE:
            raise ValueError(f"Formula supports at most {MAX_ATOMS_PER_MOLECULE} atoms")
        
        # Pad with zeros to match Korg.jl structure
        padded_atoms = tuple(list(self.atoms) + [0] * (MAX_ATOMS_PER_MOLECULE - len(self.atoms)))
        object.__setattr__(self, 'atoms', padded_atoms)
    
    @classmethod
    def from_atomic_number(cls, atomic_number: int) -> 'Formula':
        """Create formula for single atom."""
        return cls((atomic_number,))
    
    @classmethod
    def from_atomic_numbers(cls, atomic_numbers: List[int]) -> 'Formula':
        """Create formula from list of atomic numbers (will be sorted)."""
        if not atomic_numbers:
            raise ValueError("Cannot create empty formula")
        sorted_atoms = tuple(sorted(atomic_numbers))
        return cls(sorted_atoms)
    
    def is_molecule(self) -> bool:
        """Check if this formula represents a molecule (more than one atom)."""
        # Count non-zero atoms - if more than one atom, it's a molecule
        non_zero_atoms = sum(1 for atom in self.atoms if atom != 0)
        return non_zero_atoms > 1
    
    def get_atoms(self) -> List[int]:
        """Return list of atomic numbers, excluding padding zeros."""
        return [atom for atom in self.atoms if atom != 0]


@dataclass(frozen=True)
class Species:
    """
    Chemical species representation matching Korg.jl exactly.
    
    Combines Formula (atomic/molecular composition) with charge state.
    """
    formula: Formula
    charge: int  # 0=neutral, 1=singly ionized, -1=negative ion, etc.
    
    def __post_init__(self):
        if self.charge < -1:
            raise ValueError(f"Cannot construct species with charge < -1: {self.formula} with charge {self.charge}")
    
    @classmethod
    def from_element(cls, atomic_number: int, charge: int = 0) -> 'Species':
        """Create species for single element."""
        return cls(Formula.from_atomic_number(atomic_number), charge)
    
    @classmethod
    def from_molecule(cls, atomic_numbers: List[int], charge: int = 0) -> 'Species':
        """Create species for molecule."""
        return cls(Formula.from_atomic_numbers(atomic_numbers), charge)
    
    def is_molecule(self) -> bool:
        """Check if this species is molecular."""
        return self.formula.is_molecule()
    
    def __str__(self) -> str:
        atoms = self.formula.get_atoms()
        if len(atoms) == 1:
            # Atomic species
            element_symbols = {1: 'H', 2: 'He', 6: 'C', 7: 'N', 8: 'O', 11: 'Na', 12: 'Mg', 
                             13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 20: 'Ca', 22: 'Ti', 26: 'Fe'}
            symbol = element_symbols.get(atoms[0], f'El{atoms[0]}')
            if self.charge == 0:
                return f'{symbol} I'
            elif self.charge == 1:
                return f'{symbol} II'
            else:
                return f'{symbol} {self.charge+1}'
        else:
            # Molecular species
            # Simplified molecular representation
            return f'Molecule({atoms})'


@dataclass
class Line:
    """
    Spectral line representation strictly following Korg.jl structure.
    
    This exactly matches Korg.jl's Line struct with proper units and types.
    """
    wl: float                           # Wavelength in cm (NOT Angstroms!)
    log_gf: float                       # log₁₀(gf) oscillator strength (unitless)
    species: Species                    # Chemical species (Formula + charge)
    E_lower: float                      # Lower energy level in eV (excitation potential)
    gamma_rad: float                    # Radiative damping parameter in s⁻¹
    gamma_stark: float                  # Stark broadening parameter in s⁻¹
    vdW: Tuple[float, float]           # van der Waals parameters: (σ, α) or (γ_vdW, -1.0)
    
    def __post_init__(self):
        """Validate line data after construction."""
        if self.wl <= 0:
            raise ValueError("Wavelength must be positive")
        if not isinstance(self.species, Species):
            raise TypeError("Species must be a Species object")
        if len(self.vdW) != 2:
            raise ValueError("vdW must be a tuple of exactly 2 elements")


# Legacy format for backward compatibility
class LineData(NamedTuple):
    """
    Legacy line data format (will be deprecated).
    
    This is kept for backward compatibility but new code should use Line class.
    """
    wavelength: float    # Wavelength in cm (converted from input)
    species: int        # Integer species code (will be converted to Species)
    log_gf: float       # log₁₀(gf) oscillator strength
    E_lower: float      # Lower energy level in eV
    gamma_rad: float = 0.0      # Radiative damping in s⁻¹
    gamma_stark: float = 0.0    # Stark damping in s⁻¹
    vdw_param1: float = 0.0     # vdW parameter 1
    vdw_param2: float = 0.0     # vdW parameter 2


def create_line_data(wavelength: float,
                    species: int,
                    log_gf: float,
                    E_lower: float,
                    gamma_rad: float = 0.0,
                    gamma_stark: float = 0.0,
                    vdw_param1: float = 0.0,
                    vdw_param2: float = 0.0,
                    wavelength_unit: str = 'auto') -> LineData:
    """
    Create a LineData object with proper unit conversions following Korg.jl conventions.
    
    Parameters
    ----------
    wavelength : float
        Line wavelength (will be converted to cm following Korg.jl logic)
    species : int
        Species ID (integer encoding)
    log_gf : float
        Logarithm base 10 of oscillator strength (unitless)
    E_lower : float
        Lower energy level in eV
    gamma_rad : float, optional
        Radiative broadening parameter in s⁻¹
    gamma_stark : float, optional
        Stark broadening parameter in s⁻¹
    vdw_param1 : float, optional
        Van der Waals parameter 1
    vdw_param2 : float, optional
        Van der Waals parameter 2
    wavelength_unit : str, optional
        Unit of input wavelength ('auto', 'angstrom', 'cm')
        
    Returns
    -------
    LineData
        Line data object with wavelength in cm (Korg.jl convention)
    """
    # Convert wavelength to cm following Korg.jl convention
    if wavelength_unit == 'auto':
        # Korg.jl logic: wavelengths >= 1 assumed to be in Å, < 1 assumed to be in cm
        if wavelength >= 1.0:
            wavelength_cm = wavelength * 1e-8  # Å to cm
        else:
            wavelength_cm = wavelength  # Already in cm
    elif wavelength_unit == 'angstrom':
        wavelength_cm = wavelength * 1e-8  # Å to cm
    elif wavelength_unit == 'cm':
        wavelength_cm = wavelength  # Already in cm
    else:
        raise ValueError(f"Unknown wavelength unit: {wavelength_unit}")
    
    return LineData(
        wavelength=wavelength_cm,
        species=species,
        log_gf=log_gf,
        E_lower=E_lower,
        gamma_rad=gamma_rad,
        gamma_stark=gamma_stark,
        vdw_param1=vdw_param1,
        vdw_param2=vdw_param2
    )


def create_line(wl: float,
                log_gf: float, 
                species: Species,
                E_lower: float,
                gamma_rad: float = 0.0,
                gamma_stark: float = 0.0,
                vdW: Tuple[float, float] = (0.0, -1.0),
                wavelength_unit: str = 'auto') -> Line:
    """
    Create a Line object following Korg.jl structure exactly.
    
    Parameters
    ----------
    wl : float
        Wavelength (will be converted to cm)
    log_gf : float
        log₁₀(gf) oscillator strength
    species : Species
        Chemical species object
    E_lower : float
        Lower energy level in eV
    gamma_rad : float, optional
        Radiative damping in s⁻¹
    gamma_stark : float, optional
        Stark damping in s⁻¹  
    vdW : Tuple[float, float], optional
        van der Waals parameters (σ, α) or (γ_vdW, -1.0)
    wavelength_unit : str, optional
        Unit of input wavelength
        
    Returns
    -------
    Line
        Line object matching Korg.jl structure
    """
    # Convert wavelength to cm following Korg.jl convention
    if wavelength_unit == 'auto':
        if wl >= 1.0:
            wl_cm = wl * 1e-8  # Å to cm
        else:
            wl_cm = wl  # Already in cm
    elif wavelength_unit == 'angstrom':
        wl_cm = wl * 1e-8  # Å to cm
    elif wavelength_unit == 'cm':
        wl_cm = wl  # Already in cm
    else:
        raise ValueError(f"Unknown wavelength unit: {wavelength_unit}")
    
    return Line(
        wl=wl_cm,
        log_gf=log_gf,
        species=species,
        E_lower=E_lower,
        gamma_rad=gamma_rad,
        gamma_stark=gamma_stark,
        vdW=vdW
    )


def species_from_integer(species_id: int) -> Species:
    """
    Convert integer species ID to Species object following Korg.jl conventions.
    
    This handles the mapping from integer codes (like 2600 for Fe I, 801 for H2O)
    to proper Species objects with Formula and charge.
    
    Parameters
    ----------
    species_id : int
        Integer species identifier
        
    Returns
    -------
    Species
        Species object with proper Formula and charge
    """
    if species_id < 100:
        # Simple atomic species: species_id = atomic_number
        return Species.from_element(species_id, charge=0)
    else:
        # Decode element and ionization from Korg.jl convention
        element_id = species_id // 100
        ionization = (species_id % 100) - 1  # 1-indexed to 0-indexed
        
        # Handle molecular species (crude detection for now)
        if species_id >= 601:  # Molecular species range
            # This is a simplified mapping - in practice would need full molecular database
            molecular_mapping = {
                801: [1, 1, 8],     # H2O
                2208: [22, 8],      # TiO
                608: [6, 8],        # CO
                108: [1, 8],        # OH
                601: [6, 1],        # CH
                607: [6, 7],        # CN
                # Add more as needed
            }
            
            if species_id in molecular_mapping:
                atoms = molecular_mapping[species_id]
                return Species.from_molecule(atoms, charge=0)
            else:
                # Fallback for unknown molecular species
                return Species.from_element(element_id, charge=0)
        else:
            # Atomic species
            return Species.from_element(element_id, charge=ionization)
