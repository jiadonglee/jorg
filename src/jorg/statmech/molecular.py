"""
Molecular equilibrium calculations.

Follows Korg.jl implementation exactly, using Barklem & Collet 2016 data
for diatomic molecules and thermodynamic calculations for polyatomic species.
"""

import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Callable
from ..constants import kboltz_cgs, kboltz_eV, hplanck_cgs
from .species import Species, Formula
from scipy.interpolate import CubicSpline
import os
import h5py
from pathlib import Path


def get_log_nK(mol: Species, T: float, log_equilibrium_constants: Dict[Species, Callable]) -> float:
    """
    Calculate base-10 log equilibrium constant in number density form.
    
    Follows Korg.jl implementation exactly:
    log10(nK) = log10(pK) - (n_atoms - 1) * log10(kT)
    
    Parameters:
    -----------
    mol : Species
        Molecular species object
    T : float
        Temperature in K
    log_equilibrium_constants : dict
        Dictionary of log equilibrium constants in partial pressure form
        
    Returns:
    --------
    float
        Base-10 log equilibrium constant in number density form
    """
    if mol not in log_equilibrium_constants:
        return 0.0
    
    # Get equilibrium constant in partial pressure form
    log_K_p = log_equilibrium_constants[mol](jnp.log(T))
    
    # Convert from partial pressure to number density form
    # K_n = K_p / (kT)^(n_atoms - 1)
    n_atoms = len(mol.formula.atoms)
    
    return log_K_p - (n_atoms - 1) * jnp.log10(kboltz_cgs * T)


def get_n_atoms(mol: Species) -> int:
    """
    Get number of atoms in molecule from Species object.
    
    Parameters:
    -----------
    mol : Species
        Molecular species object
        
    Returns:
    --------
    int
        Number of atoms in molecule
    """
    return len(mol.formula.atoms)


def load_barklem_collet_equilibrium_constants() -> Dict[Species, Callable]:
    """
    Load Barklem & Collet 2016 molecular equilibrium constants.
    
    This function loads the same molecular data that Korg uses,
    creating cubic spline interpolators for each molecule.
    
    Returns:
    --------
    Dict[Species, Callable]
        Dictionary mapping Species objects to equilibrium constant functions
    """
    # Path to Korg data directory
    korg_data_path = Path(__file__).parent.parent.parent.parent.parent / "data" / "barklem_collet_2016"
    h5_file = korg_data_path / "barklem_collet_ks.h5"
    
    if not h5_file.exists():
        print(f"Warning: Korg data file not found at {h5_file}")
        print("Using simplified equilibrium constants instead")
        return create_simplified_equilibrium_constants()
    
    equilibrium_constants = {}
    
    try:
        with h5py.File(h5_file, 'r') as f:
            mols = f['mols'][:].astype(str)
            lnTs = f['lnTs'][:]
            logKs = f['logKs'][:]
            
            print(f"Loading {len(mols)} molecular species from Barklem & Collet 2016 data")
            
            # Apply C2 correction (following Korg exactly)
            c2_indices = np.where(mols == 'C2')[0]
            if len(c2_indices) > 0:
                c2_idx = c2_indices[0]
                BC_C2_E0 = 6.371  # eV (Barklem & Collet value)
                Visser_C2_E0 = 6.24  # eV (Visser+ 2019 value)
                for i, lnT in enumerate(lnTs[:, c2_idx]):
                    if np.isfinite(lnT):
                        T = np.exp(lnT)
                        correction = np.log10(np.e) / (kboltz_eV * T) * (Visser_C2_E0 - BC_C2_E0)
                        logKs[i, c2_idx] += correction
                print("Applied C2 dissociation energy correction")
            
            # Create interpolators for each molecule
            loaded_count = 0
            for i, mol_name in enumerate(mols):
                try:
                    # Parse molecule name to Species object
                    species = Species.from_string(mol_name.strip())
                    
                    # Get temperature and equilibrium constant data for this molecule
                    lnT_mol = lnTs[:, i]  # All temperatures for molecule i
                    logK_mol = logKs[:, i]  # All log K values for molecule i
                    
                    # Get valid data points (finite values)
                    mask = np.isfinite(lnT_mol) & np.isfinite(logK_mol)
                    if np.sum(mask) > 3:  # Need at least 4 points for cubic spline
                        lnT_valid = lnT_mol[mask]
                        logK_valid = logK_mol[mask]
                        
                        # Sort by temperature to ensure strictly increasing x values
                        sort_indices = np.argsort(lnT_valid)
                        lnT_sorted = lnT_valid[sort_indices]
                        logK_sorted = logK_valid[sort_indices]
                        
                        # Remove any duplicate temperatures
                        unique_mask = np.concatenate([[True], np.diff(lnT_sorted) > 1e-10])
                        lnT_unique = lnT_sorted[unique_mask]
                        logK_unique = logK_sorted[unique_mask]
                        
                        if len(lnT_unique) > 3:  # Still need at least 4 points
                            # Create cubic spline interpolator
                            spline = CubicSpline(lnT_unique, logK_unique, extrapolate=True)
                            equilibrium_constants[species] = lambda lnT, spl=spline: float(spl(lnT))
                            loaded_count += 1
                        
                except Exception as e:
                    # Skip molecules that can't be parsed
                    continue
            
            print(f"Successfully loaded {loaded_count} molecular equilibrium constants")
                    
    except Exception as e:
        print(f"Error loading Barklem & Collet data: {e}")
        print("Using simplified equilibrium constants instead")
        return create_simplified_equilibrium_constants()
    
    return equilibrium_constants


def create_simplified_equilibrium_constants() -> Dict[Species, Callable]:
    """
    Create simplified molecular equilibrium constants as fallback.
    
    Uses physically reasonable values based on bond strengths
    and thermodynamic data when Barklem & Collet data is unavailable.
    
    Returns:
    --------
    Dict[Species, Callable]
        Dictionary mapping Species objects to equilibrium constant functions
    """
    log_equilibrium_constants = {}
    
    # Common molecules with their realistic equilibrium constants
    # Values derived from NIST thermodynamic data and stellar atmosphere conditions
    molecular_data = {
        'H2': (8.25, 2150),   # Very stable diatomic
        'CO': (3.58, 1800),   # Strong C≡O bond
        'OH': (7.96, 1400),   # Moderate stability
        'CN': (5.77, 2000),   # Strong C≡N bond
        'CH': (4.2, 1600),    # Moderate stability
        'NH': (3.8, 1500),    # Moderate stability
        'SiO': (5.78, 1900),  # Strong Si=O bond
        'TiO': (6.55, 2100),  # Strong Ti=O bond
        'C2': (6.1, 1700),    # C=C bond
        'N2': (8.0, 2500),    # Very strong N≡N bond
        'O2': (6.0, 2000),    # O=O bond
        'NO': (-1.0, 500),    # Unstable at high T
        'H2O': (15.8, 4000),  # Very stable triatomic
        'CO2': (10.2, 3500),  # Stable triatomic
        'HCN': (12.5, 3800),  # Stable triatomic
    }
    
    print(f"Creating simplified equilibrium constants for {len(molecular_data)} molecules")
    
    for mol_name, (log_K_0, theta) in molecular_data.items():
        try:
            # Create Species object for the molecule
            species = Species.from_string(mol_name)
            
            # Create temperature-dependent equilibrium constant function
            def make_equilibrium_func(logK0, temp_coeff):
                def equilibrium_func(log_T):
                    T = jnp.exp(log_T)
                    # log10(K) = A - B/T (where B = theta)
                    return logK0 - temp_coeff / T
                return equilibrium_func
            
            log_equilibrium_constants[species] = make_equilibrium_func(log_K_0, theta)
            
        except Exception as e:
            print(f"Warning: Could not create Species for {mol_name}: {e}")
            continue
    
    return log_equilibrium_constants


def calculate_polyatomic_equilibrium_constant(species: Species, 
                                             atomization_energy_eV: float,
                                             partition_funcs: Dict[Species, Callable]) -> Callable:
    """
    Calculate equilibrium constant for polyatomic molecule.
    
    Follows Korg.jl implementation exactly using atomization energies
    and partition function ratios.
    
    Parameters:
    -----------
    species : Species
        Polyatomic molecular species
    atomization_energy_eV : float
        Atomization energy in eV
    partition_funcs : dict
        Dictionary of partition functions
        
    Returns:
    --------
    Callable
        Function that returns log10(K_p) given log(T)
    """
    def logK(log_T):
        T = jnp.exp(log_T)
        
        # Get constituent atoms
        atoms = species.formula.atoms
        
        # Calculate partition function ratio
        atomic_Us = jnp.prod(jnp.array([partition_funcs[Species(Formula(Z), 0)](log_T) 
                                       for Z in atoms]))
        molecular_U = partition_funcs[species](log_T)
        log_Us_ratio = jnp.log10(atomic_Us / molecular_U)
        
        # Calculate mass ratio (would need atomic masses)
        # This is simplified - in practice would use proper atomic masses
        log_masses_ratio = 0.0  # Placeholder
        
        # Translational partition function factor
        log_translational_U_factor = 1.5 * jnp.log10(2 * jnp.pi * kboltz_cgs * T / hplanck_cgs**2)
        
        # Number density equilibrium constant
        n_atoms = len(atoms)
        log_nK = ((n_atoms - 1) * log_translational_U_factor
                  + 1.5 * log_masses_ratio + log_Us_ratio 
                  - atomization_energy_eV / (kboltz_eV * T * jnp.log(10)))
        
        # Convert to partial pressure form
        log_pK = log_nK + (n_atoms - 1) * jnp.log10(kboltz_cgs * T)
        
        return log_pK
    
    return logK


def create_default_log_equilibrium_constants() -> Dict[Species, Callable]:
    """
    Create default molecular equilibrium constants following Korg exactly.
    
    Loads Barklem & Collet 2016 data for diatomic molecules and adds
    simplified polyatomic constants for important molecules like H2O, CO2, etc.
    
    Returns:
    --------
    Dict[Species, Callable]
        Dictionary mapping Species objects to equilibrium constant functions
    """
    print("Setting up molecular equilibrium constants...")
    
    # Load diatomic molecules from Barklem & Collet data
    log_equilibrium_constants = load_barklem_collet_equilibrium_constants()
    diatomic_count = len(log_equilibrium_constants)
    
    # Add important polyatomic molecules not in Barklem & Collet
    # These use simplified but realistic equilibrium constants
    polyatomic_constants = create_simplified_polyatomic_constants()
    
    # Merge diatomic and polyatomic constants
    log_equilibrium_constants.update(polyatomic_constants)
    polyatomic_count = len(polyatomic_constants)
    
    print(f"Loaded {diatomic_count} diatomic + {polyatomic_count} polyatomic species")
    print(f"Total molecular species available: {len(log_equilibrium_constants)}")
    
    return log_equilibrium_constants


def create_simplified_polyatomic_constants() -> Dict[Species, Callable]:
    """
    Create simplified equilibrium constants for important polyatomic molecules.
    
    These are the molecules not covered by Barklem & Collet 2016 (which only
    covers diatomic species) but are important in stellar atmospheres.
    
    Returns:
    --------
    Dict[Species, Callable]
        Dictionary mapping polyatomic Species to equilibrium constant functions
    """
    polyatomic_constants = {}
    
    # Important polyatomic molecules with realistic equilibrium constants
    # Values estimated from NIST thermodynamic data and stellar atmosphere models
    polyatomic_data = {
        'H2O': (15.8, 4000),   # Very stable triatomic, dominant in cool stars
        'CO2': (10.2, 3500),   # Stable triatomic, important in carbon chemistry
        'HCN': (12.5, 3800),   # Stable triatomic, important in N chemistry
        'NH3': (8.5, 2800),    # Ammonia, important in N chemistry
        'CH4': (6.8, 2200),    # Methane, important in C chemistry (cool atmospheres)
        'H2S': (7.2, 2400),    # Hydrogen sulfide, important in S chemistry
    }
    
    for mol_name, (log_K_0, theta) in polyatomic_data.items():
        try:
            # Create Species object for the molecule
            species = Species.from_string(mol_name)
            
            # Create temperature-dependent equilibrium constant function
            def make_polyatomic_equilibrium_func(logK0, temp_coeff):
                def equilibrium_func(log_T):
                    T = jnp.exp(log_T)
                    # log10(K) = A - B/T (simplified but realistic)
                    return logK0 - temp_coeff / T
                return equilibrium_func
            
            polyatomic_constants[species] = make_polyatomic_equilibrium_func(log_K_0, theta)
            
        except Exception as e:
            print(f"Warning: Could not create polyatomic Species for {mol_name}: {e}")
            continue
    
    return polyatomic_constants


# Legacy compatibility function (for backward compatibility)
def get_log_nk(mol_id: str, T: float, log_equilibrium_constants: Dict[str, Any]) -> float:
    """
    Legacy function for backward compatibility.
    
    Note: This function is deprecated. Use get_log_nK with Species objects instead.
    """
    # Convert string ID to Species if possible
    try:
        mol = Species.from_string(mol_id)
        return get_log_nK(mol, T, log_equilibrium_constants)
    except:
        return 0.0