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