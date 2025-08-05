"""
Molecular species support for stellar spectral synthesis.

This module provides molecular species identification, partition functions,
and equilibrium calculations matching Korg.jl's molecular capabilities.
"""

import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import h5py
from pathlib import Path

from .species import Species
from ..constants import BOLTZMANN_K, PLANCK_H, c_cgs


@dataclass
class MolecularSpecies:
    """
    Molecular species information.
    
    Attributes
    ----------
    name : str
        Molecular name (e.g., 'H2O', 'TiO')
    formula : str
        Chemical formula
    species_id : int
        Numerical identifier
    atomic_composition : Dict[int, int]
        {element_id: count} for constituent atoms
    mass_amu : float
        Molecular mass in atomic mass units
    dissociation_energy : float
        Dissociation energy in eV
    """
    name: str
    formula: str
    species_id: int
    atomic_composition: Dict[int, int]
    mass_amu: float
    dissociation_energy: float = 0.0


# Common stellar molecular species
STELLAR_MOLECULES = {
    'H2O': MolecularSpecies(
        name='H2O', formula='H2O', species_id=801,
        atomic_composition={1: 2, 8: 1}, mass_amu=18.015,
        dissociation_energy=5.1  # eV
    ),
    'TiO': MolecularSpecies(
        name='TiO', formula='TiO', species_id=2208,
        atomic_composition={22: 1, 8: 1}, mass_amu=63.866,
        dissociation_energy=6.87  # eV
    ),
    'VO': MolecularSpecies(
        name='VO', formula='VO', species_id=2308,
        atomic_composition={23: 1, 8: 1}, mass_amu=66.941,
        dissociation_energy=6.31  # eV
    ),
    'OH': MolecularSpecies(
        name='OH', formula='OH', species_id=108,
        atomic_composition={1: 1, 8: 1}, mass_amu=17.007,
        dissociation_energy=4.39  # eV
    ),
    'CH': MolecularSpecies(
        name='CH', formula='CH', species_id=601,
        atomic_composition={6: 1, 1: 1}, mass_amu=13.019,
        dissociation_energy=3.47  # eV
    ),
    'CN': MolecularSpecies(
        name='CN', formula='CN', species_id=607,
        atomic_composition={6: 1, 7: 1}, mass_amu=26.018,
        dissociation_energy=7.76  # eV
    ),
    'CO': MolecularSpecies(
        name='CO', formula='CO', species_id=608,
        atomic_composition={6: 1, 8: 1}, mass_amu=28.014,
        dissociation_energy=11.09  # eV
    ),
    'NH': MolecularSpecies(
        name='NH', formula='NH', species_id=701,
        atomic_composition={7: 1, 1: 1}, mass_amu=15.015,
        dissociation_energy=3.47  # eV
    ),
    'SiO': MolecularSpecies(
        name='SiO', formula='SiO', species_id=1408,
        atomic_composition={14: 1, 8: 1}, mass_amu=44.085,
        dissociation_energy=8.26  # eV
    ),
    'CaH': MolecularSpecies(
        name='CaH', formula='CaH', species_id=2001,
        atomic_composition={20: 1, 1: 1}, mass_amu=41.086,
        dissociation_energy=1.7   # eV
    ),
    'FeH': MolecularSpecies(
        name='FeH', formula='FeH', species_id=2601,
        atomic_composition={26: 1, 1: 1}, mass_amu=56.853,
        dissociation_energy=1.6   # eV
    ),
    'MgH': MolecularSpecies(
        name='MgH', formula='MgH', species_id=1201,
        atomic_composition={12: 1, 1: 1}, mass_amu=25.313,
        dissociation_energy=1.3   # eV
    ),
    'AlH': MolecularSpecies(
        name='AlH', formula='AlH', species_id=1301,
        atomic_composition={13: 1, 1: 1}, mass_amu=27.990,
        dissociation_energy=3.1   # eV
    ),
    'SiH': MolecularSpecies(
        name='SiH', formula='SiH', species_id=1401,
        atomic_composition={14: 1, 1: 1}, mass_amu=29.093,
        dissociation_energy=3.06  # eV
    ),
    'H2': MolecularSpecies(
        name='H2', formula='H2', species_id=101,
        atomic_composition={1: 2}, mass_amu=2.016,
        dissociation_energy=4.48  # eV
    ),
    'C2': MolecularSpecies(
        name='C2', formula='C2', species_id=606,
        atomic_composition={6: 2}, mass_amu=24.022,
        dissociation_energy=6.21  # eV
    ),
    'N2': MolecularSpecies(
        name='N2', formula='N2', species_id=707,
        atomic_composition={7: 2}, mass_amu=28.014,
        dissociation_energy=9.76  # eV
    ),
    'O2': MolecularSpecies(
        name='O2', formula='O2', species_id=808,
        atomic_composition={8: 2}, mass_amu=31.998,
        dissociation_energy=5.12  # eV
    ),
}


def get_molecular_species(name: str) -> Optional[MolecularSpecies]:
    """
    Get molecular species by name.
    
    Parameters
    ----------
    name : str
        Molecular species name
        
    Returns
    -------
    MolecularSpecies or None
        Molecular species object
    """
    # Try exact name first, then uppercase
    return STELLAR_MOLECULES.get(name) or STELLAR_MOLECULES.get(name.upper())


def get_molecular_species_by_id(species_id: int) -> Optional[MolecularSpecies]:
    """
    Get molecular species by ID.
    
    Parameters
    ----------
    species_id : int
        Species identifier
        
    Returns
    -------
    MolecularSpecies or None
        Molecular species object
    """
    for molecule in STELLAR_MOLECULES.values():
        if molecule.species_id == species_id:
            return molecule
    return None


def is_molecular_species(species_id: int) -> bool:
    """
    Check if species ID corresponds to a molecule.
    
    Parameters
    ----------
    species_id : int
        Species identifier
        
    Returns
    -------
    bool
        True if molecular species
    """
    return get_molecular_species_by_id(species_id) is not None


class MolecularPartitionFunction:
    """
    Molecular partition function calculator.
    
    This class provides partition function calculations for molecular species,
    including rotational, vibrational, and electronic contributions.
    """
    
    def __init__(self, molecule: MolecularSpecies):
        self.molecule = molecule
        
    def calculate(self, temperature: float) -> float:
        """
        Calculate molecular partition function at given temperature.
        
        This is a simplified implementation. For production use,
        would need full ExoMol or HITRAN data.
        
        Parameters
        ----------
        temperature : float
            Temperature in K
            
        Returns
        -------
        float
            Partition function value
        """
        # Simplified partition function calculation
        # In practice, would use tabulated data from ExoMol/HITRAN
        
        # Electronic contribution (ground state only)
        Q_elec = 1.0
        
        # Rotational contribution (rigid rotor approximation)
        if len(self.molecule.atomic_composition) == 2:
            # Diatomic molecule
            B_cm = self._estimate_rotational_constant()  # cm^-1
            B_K = B_cm * 1.44  # Convert to K
            Q_rot = temperature / B_K
        else:
            # Polyatomic (very rough approximation)
            Q_rot = (temperature / 100.0)**(3/2)
        
        # Vibrational contribution (harmonic oscillator)
        omega_cm = self._estimate_vibrational_frequency()  # cm^-1
        omega_K = omega_cm * 1.44  # Convert to K
        if omega_K > 0:
            Q_vib = 1.0 / (1.0 - jnp.exp(-omega_K / temperature))
        else:
            Q_vib = 1.0
        
        return Q_elec * Q_rot * Q_vib
    
    def _estimate_rotational_constant(self) -> float:
        """Estimate rotational constant for diatomic molecule."""
        # Very rough scaling with molecular mass
        mass_factor = (2.0 / self.molecule.mass_amu)**0.5
        return 60.0 * mass_factor  # cm^-1
    
    def _estimate_vibrational_frequency(self) -> float:
        """Estimate vibrational frequency."""
        # Rough scaling with dissociation energy
        return 2000.0 * (self.molecule.dissociation_energy / 5.0)**0.5  # cm^-1


class MolecularEquilibrium:
    """
    Molecular chemical equilibrium calculator.
    
    Calculates molecular number densities in chemical equilibrium
    using mass action law and partition functions.
    """
    
    def __init__(self, molecules: List[MolecularSpecies]):
        self.molecules = molecules
        self.partition_functions = {
            mol.species_id: MolecularPartitionFunction(mol) 
            for mol in molecules
        }
    
    def calculate_number_densities(
        self,
        temperature: float,
        pressure: float,
        element_abundances: Dict[int, float],
        total_hydrogen_density: float
    ) -> Dict[int, float]:
        """
        Calculate molecular number densities in chemical equilibrium.
        
        This is a simplified implementation of molecular equilibrium.
        For production use, would need full thermodynamic data.
        
        Parameters
        ----------
        temperature : float
            Temperature in K
        pressure : float
            Gas pressure in dyne/cm²
        element_abundances : Dict[int, float]
            Element abundances relative to hydrogen
        total_hydrogen_density : float
            Total hydrogen density in cm^-3
            
        Returns
        -------
        Dict[int, float]
            Molecular number densities by species ID in cm^-3
        """
        number_densities = {}
        
        for molecule in self.molecules:
            # Calculate equilibrium constant (simplified)
            K_eq = self._calculate_equilibrium_constant(molecule, temperature)
            
            # Apply mass action law (very simplified)
            # For diatomic AB: K = [AB] / ([A] * [B])
            if len(molecule.atomic_composition) == 2:
                elements = list(molecule.atomic_composition.keys())
                if len(elements) == 2:
                    elem1, elem2 = elements
                    n1 = element_abundances.get(elem1, 1e-10) * total_hydrogen_density
                    n2 = element_abundances.get(elem2, 1e-10) * total_hydrogen_density
                    
                    # Simplified equilibrium (ignoring competing reactions)
                    n_mol = K_eq * n1 * n2 / (1.0 + K_eq * n1)
                    number_densities[molecule.species_id] = min(n_mol, min(n1, n2))
                else:
                    # Homonuclear diatomic
                    elem = elements[0]
                    n_elem = element_abundances.get(elem, 1e-10) * total_hydrogen_density
                    n_mol = K_eq * n_elem**2 / (1.0 + K_eq * n_elem)
                    number_densities[molecule.species_id] = min(n_mol, n_elem / 2)
            else:
                # Polyatomic (very crude approximation)
                min_abundance = min(
                    element_abundances.get(elem, 1e-10) 
                    for elem in molecule.atomic_composition.keys()
                )
                n_mol = K_eq * min_abundance * total_hydrogen_density / 1000.0
                number_densities[molecule.species_id] = n_mol
        
        return number_densities
    
    def _calculate_equilibrium_constant(
        self, 
        molecule: MolecularSpecies, 
        temperature: float
    ) -> float:
        """Calculate equilibrium constant for molecular formation."""
        # K = (Q_mol / Q_atoms) * exp(-D_e / kT)
        
        # Partition function contribution
        Q_mol = self.partition_functions[molecule.species_id].calculate(temperature)
        
        # Rough atomic partition functions (simplified)
        Q_atoms = 1.0
        for element_id, count in molecule.atomic_composition.items():
            Q_atom = 2.0 if element_id == 1 else 1.0  # H has J=1/2
            Q_atoms *= Q_atom**count
        
        # Boltzmann factor
        kT_eV = BOLTZMANN_K * temperature / 1.602e-19  # Convert to eV
        boltzmann_factor = jnp.exp(-molecule.dissociation_energy / kT_eV)
        
        return (Q_mol / Q_atoms) * boltzmann_factor


def load_exomol_partition_functions(data_dir: str) -> Dict[str, callable]:
    """
    Load ExoMol partition functions from data files.
    
    This would interface with ExoMol partition function data.
    Currently returns placeholder functions.
    
    Parameters
    ----------
    data_dir : str
        Path to ExoMol data directory
        
    Returns
    -------
    Dict[str, callable]
        Dictionary of partition function interpolators by species
    """
    # Placeholder - in practice would load from ExoMol .pf files
    partition_functions = {}
    
    for name, molecule in STELLAR_MOLECULES.items():
        pf_calc = MolecularPartitionFunction(molecule)
        partition_functions[name] = pf_calc.calculate
    
    return partition_functions


def get_apogee_molecular_species() -> List[str]:
    """
    Get molecular species commonly found in APOGEE spectra.
    
    Returns
    -------
    List[str]
        List of molecular species names
    """
    return ['H2O', 'OH', 'CO', 'CN', 'C2']


def get_cool_star_molecules() -> List[str]:
    """
    Get molecular species important for cool star spectroscopy.
    
    Returns
    -------
    List[str]
        List of molecular species names
    """
    return ['TiO', 'VO', 'H2O', 'OH', 'FeH', 'CaH', 'MgH']


def molecular_species_summary() -> None:
    """Print summary of available molecular species."""
    print("🧬 Available Molecular Species:")
    print("=" * 50)
    
    for name, molecule in STELLAR_MOLECULES.items():
        composition = " + ".join([
            f"{count}{elem}" if count > 1 else f"{elem}"
            for elem, count in molecule.atomic_composition.items()
        ])
        
        print(f"{name:>6} | ID: {molecule.species_id:>4} | "
              f"Mass: {molecule.mass_amu:>6.3f} amu | "
              f"D_e: {molecule.dissociation_energy:>5.2f} eV | "
              f"Composition: {composition}")
    
    print("=" * 50)
    print(f"Total: {len(STELLAR_MOLECULES)} molecular species")


# Molecular opacity helper functions

def molecular_opacity_temperature_scaling(temperature: float) -> float:
    """
    Temperature scaling factor for molecular opacity.
    
    Molecular opacity typically decreases with temperature due to
    thermal dissociation.
    
    Parameters
    ----------
    temperature : float
        Temperature in K
        
    Returns
    -------
    float
        Scaling factor
    """
    # Empirical scaling based on typical stellar atmospheres
    T_ref = 3000.0  # Reference temperature
    return (temperature / T_ref)**(-1.5)


def estimate_molecular_fraction(
    molecule_name: str,
    temperature: float,
    pressure: float,
    metallicity: float = 0.0
) -> float:
    """
    Estimate molecular fraction in stellar atmosphere.
    
    Very rough empirical estimates for molecular abundances.
    
    Parameters
    ----------
    molecule_name : str
        Molecular species name
    temperature : float
        Temperature in K
    pressure : float
        Gas pressure in dyne/cm²
    metallicity : float
        Metallicity [M/H]
        
    Returns
    -------
    float
        Estimated molecular fraction relative to total
    """
    molecule = get_molecular_species(molecule_name)
    if molecule is None:
        return 0.0
    
    # Temperature-dependent formation
    kT_eV = BOLTZMANN_K * temperature / 1.602e-19
    formation_factor = jnp.exp(-molecule.dissociation_energy / (2 * kT_eV))
    
    # Pressure enhancement
    pressure_factor = (pressure / 1e6)**0.2  # Rough scaling
    
    # Metallicity dependence
    if any(elem > 2 for elem in molecule.atomic_composition.keys()):
        # Metal-bearing molecules
        metal_factor = 10**metallicity
    else:
        # H, He molecules
        metal_factor = 1.0
    
    # CRITICAL FIX: Remove hardcoded molecular abundances to match Korg.jl
    # Korg.jl uses proper chemical equilibrium with equilibrium constants (log_nK)
    # See Korg.jl/src/statmech.jl:317-337 for proper molecular equilibrium calculation
    #
    # For production use, molecular abundances should come from:
    # 1. Chemical equilibrium solver (statmech.chemical_equilibrium)  
    # 2. Equilibrium constants from thermodynamic databases
    # 3. Mass action law: n_mol = 10^(sum(n_atoms) - log_nK)
    #
    # These hardcoded values are rough approximations only for development/testing
    base_fractions = {
        'H2O': 0.0, 'TiO': 0.0, 'VO': 0.0, 'OH': 0.0,     # Use chemical equilibrium instead
        'CO': 0.0, 'CH': 0.0, 'CN': 0.0, 'NH': 0.0,       # Use chemical equilibrium instead  
        'FeH': 0.0, 'CaH': 0.0, 'MgH': 0.0,               # Use chemical equilibrium instead
        'H2': 0.0, 'SiO': 0.0                              # Use chemical equilibrium instead
    }
    
    base_fraction = base_fractions.get(molecule_name.upper(), 1e-12)
    
    return base_fraction * formation_factor * pressure_factor * metal_factor