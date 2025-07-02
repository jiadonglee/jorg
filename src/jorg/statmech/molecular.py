"""
Molecular equilibrium calculations.

Based on Korg.jl statistical mechanics implementation.
"""

import jax.numpy as jnp
from jax import jit
from typing import Dict, Any
from ..constants import kboltz_cgs


@jit
def get_log_nk(mol_id: str, T: float, log_equilibrium_constants: Dict[str, Any]) -> float:
    """
    Calculate base-10 log equilibrium constant in number density form.
    
    Converts from partial pressure form to number density form:
    log10(nK) where nK = n(A)n(B)/n(AB)
    
    Parameters:
    -----------
    mol_id : str
        Molecular species identifier
    T : float
        Temperature in K
    log_equilibrium_constants : dict
        Dictionary of log equilibrium constants in partial pressure form
        
    Returns:
    --------
    float
        Base-10 log equilibrium constant in number density form
    """
    if mol_id not in log_equilibrium_constants:
        return 0.0
    
    # Get equilibrium constant function and evaluate at this temperature
    log_K_p = log_equilibrium_constants[mol_id](jnp.log(T))
    
    # Convert from partial pressure to number density form
    # For diatomic molecule: K_n = K_p / (kT)
    # For polyatomic: K_n = K_p / (kT)^(n_atoms - 1)
    n_atoms = get_n_atoms(mol_id)  # Would need proper molecular formula parsing
    
    return log_K_p - (n_atoms - 1) * jnp.log10(kboltz_cgs * T)


def get_n_atoms(mol_id: str) -> int:
    """
    Get number of atoms in molecule from molecular identifier.
    
    Parameters:
    -----------
    mol_id : str
        Molecular species identifier
        
    Returns:
    --------
    int
        Number of atoms in molecule
    """
    # Simplified implementation - in practice would parse molecular formula
    common_molecules = {
        'CO': 2,
        'OH': 2, 
        'CH': 2,
        'CN': 2,
        'H2': 2,
        'C2': 2,
        'NH': 2,
        'H2O': 3,
        'CO2': 3,
    }
    return common_molecules.get(mol_id, 2)  # Default to diatomic


def create_default_equilibrium_constants() -> Dict[str, Any]:
    """
    Create default molecular equilibrium constants.
    
    In practice, these would be interpolated from tabulated data
    over temperature ranges.
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary mapping molecule IDs to equilibrium constant functions
    """
    # Simplified implementation for testing
    equilibrium_constants = {}
    
    # Example: CO molecule (very simplified)
    def co_equilibrium(log_T):
        # Simplified temperature dependence
        return -5.0 + 2000.0 / jnp.exp(log_T)
    
    equilibrium_constants['CO'] = co_equilibrium
    
    return equilibrium_constants


def create_default_log_equilibrium_constants() -> Dict[Any, Any]:
    """
    Create default log molecular equilibrium constants for tutorial compatibility.
    
    This function creates equilibrium constants that work with Species objects
    as expected by the tutorial examples.
    
    Returns:
    --------
    Dict
        Dictionary mapping Species objects to log equilibrium constant functions
    """
    from .species import Species, Formula
    
    log_equilibrium_constants = {}
    
    # Common molecules and their simplified equilibrium constants
    molecules = [
        ('CO', [6, 8]),     # Carbon monoxide
        ('OH', [1, 8]),     # Hydroxyl  
        ('CH', [1, 6]),     # Methylidyne
        ('CN', [6, 7]),     # Cyanogen
        ('H2', [1, 1]),     # Hydrogen molecule
        ('O2', [8, 8]),     # Oxygen molecule
        ('N2', [7, 7]),     # Nitrogen molecule
        ('NO', [7, 8]),     # Nitric oxide
        ('H2O', [1, 1, 8]), # Water
    ]
    
    for mol_name, atoms in molecules:
        try:
            # Create Species object for the molecule
            formula = Formula.from_atomic_numbers(atoms)
            species = Species(formula, 0)  # Neutral molecule
            
            # Create simplified equilibrium constant function
            def make_equilibrium_func(name):
                def equilibrium_func(log_T):
                    T = jnp.exp(log_T)
                    # Very simplified temperature dependence
                    # In practice, these would be from detailed molecular data
                    if name == 'CO':
                        return -5.0 + 2000.0 / T
                    elif name == 'OH':
                        return -3.0 + 1500.0 / T
                    elif name == 'H2':
                        return -2.0 + 1000.0 / T
                    elif name == 'H2O':
                        return -8.0 + 3000.0 / T
                    else:
                        return -4.0 + 1800.0 / T
                return equilibrium_func
            
            log_equilibrium_constants[species] = make_equilibrium_func(mol_name)
            
        except Exception:
            # Skip if Species creation fails
            continue
    
    return log_equilibrium_constants