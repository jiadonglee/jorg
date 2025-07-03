"""
Partition function calculations for atoms and molecules.

Based on Korg.jl statistical mechanics implementation.
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Dict, Any
from ..constants import kboltz_cgs, RYDBERG, ELECTRON_MASS, EV_TO_ERG


@jit
def hydrogen_partition_function(log_T: float) -> float:
    """
    Calculate partition function for neutral hydrogen.
    
    Parameters:
    -----------
    log_T : float
        Natural logarithm of temperature in K
        
    Returns:
    --------
    float
        Partition function for H I
    """
    # For hydrogen, the partition function is simply the statistical weight
    # of the ground state (2 for spin degeneracy) at most temperatures
    # relevant for stellar atmospheres
    return 2.0


@jit
def simple_partition_function(species_id: int, log_T: float) -> float:
    """
    Simple partition function calculation for atoms.
    
    Parameters:
    -----------
    species_id : int
        Species identifier (atomic number for neutral atoms)
    log_T : float
        Natural logarithm of temperature in K
        
    Returns:
    --------
    float
        Partition function value
    """
    T = jnp.exp(log_T)
    
    # Use jnp.where for JAX compatibility
    hydrogen_case = 2.0
    helium_case = 1.0
    iron_case = 25.0 * (T / 5000.0)**0.3
    default_case = 2.0 * (T / 5000.0)**0.1
    
    # Nested where statements to handle multiple conditions
    result = jnp.where(
        species_id == 1,
        hydrogen_case,
        jnp.where(
            species_id == 2,
            helium_case,
            jnp.where(
                species_id == 26,
                iron_case,
                default_case
            )
        )
    )
    
    return result


# Additional functions for test compatibility
@jit
def atomic_partition_function(element: int, ionization: int, temperatures: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate atomic partition function for given element and ionization stage
    
    Parameters
    ----------
    element : int
        Atomic number
    ionization : int
        Ionization stage (0=neutral, 1=singly ionized, etc.)
    temperatures : jnp.ndarray
        Temperatures in K
        
    Returns
    -------
    jnp.ndarray
        Partition function values
    """
    log_T = jnp.log(temperatures)
    
    # Apply simple_partition_function to each temperature
    def calc_U(log_t):
        return simple_partition_function(element, log_t)
    
    U_values = jax.vmap(calc_U)(log_T)
    
    # Adjust for ionization stage using jnp.where for JAX compatibility
    neutral_case = U_values
    singly_ionized_case = U_values * 0.5
    higher_ionized_case = jnp.ones_like(temperatures)
    
    result = jnp.where(
        ionization == 0,
        neutral_case,
        jnp.where(
            ionization == 1,
            singly_ionized_case,
            higher_ionized_case
        )
    )
    
    return result


def partition_function(element_symbol: str, ionization: int, temperature: float) -> float:
    """
    Wrapper function for partition function calculation
    
    Parameters
    ----------
    element_symbol : str
        Element symbol ('H', 'He', 'Fe', etc.)
    ionization : int
        Ionization stage
    temperature : float
        Temperature in K
        
    Returns
    -------
    float
        Partition function value
    """
    # Element symbol to atomic number mapping
    element_map = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
        'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
        'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
        'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30
    }
    
    element_number = element_map.get(element_symbol, 1)
    temperatures = jnp.array([temperature])
    
    return float(atomic_partition_function(element_number, ionization, temperatures)[0])


def create_partition_function_dict() -> Dict[Any, Any]:
    """
    Create a dictionary of partition functions compatible with Korg.jl format.
    
    Returns:
    --------
    Dict
        Dictionary mapping species to partition function callables
    """
    # This would be populated with actual partition function data
    # For now, return a simple implementation
    partition_funcs = {}
    
    # Add simple functions for testing
    for Z in range(1, 93):  # Elements 1-92
        for charge in range(3):  # I, II, III
            species_key = f"{Z}_{charge}"  # Simplified key format
            partition_funcs[species_key] = lambda log_T, z=Z, c=charge: simple_partition_function(z, log_T)
    
    return partition_funcs


def create_default_partition_functions() -> Dict[Any, Any]:
    """
    Create partition functions exactly matching Korg.jl.
    
    This function now uses the exact partition function values from Korg.jl's
    NIST atomic level data, eliminating the 20-40% discrepancies from simplified
    approximations identified in the root cause analysis.
    
    Returns:
    --------
    Dict
        Dictionary mapping Species objects to partition function callables
        that exactly match Korg.jl values
    """
    try:
        # Use exact Korg.jl partition functions
        from .korg_partition_functions import create_korg_partition_functions
        return create_korg_partition_functions()
    except ImportError as e:
        print(f"⚠️ Warning: Could not load Korg.jl partition functions ({e})")
        print("    Falling back to simplified partition functions.")
        return create_simplified_partition_functions()


def create_simplified_partition_functions() -> Dict[Any, Any]:
    """
    Create simplified partition functions (fallback).
    
    This function creates the old simplified partition functions as a fallback
    when the exact Korg.jl data is not available.
    
    Returns:
    --------
    Dict
        Dictionary mapping Species objects to simplified partition function callables
    """
    from .species import Species, Formula
    
    partition_funcs = {}
    
    # Create partition functions for common species using Species objects as keys
    for Z in range(1, 93):  # Elements 1-92
        for charge in range(3):  # I, II, III (neutral, singly, doubly ionized)
            try:
                # Create Species object
                formula = Formula.from_atomic_number(Z)
                species = Species(formula, charge)
                
                # Create partition function that properly handles ionization stages
                def make_partition_func(atomic_num, chrg):
                    def partition_func(log_T):
                        T = jnp.exp(log_T)
                        
                        # Hydrogen - exact Korg.jl behavior
                        if atomic_num == 1:
                            if chrg == 0:
                                return 2.0  # H I: ground state degeneracy
                            elif chrg == 1:
                                return 1.0  # H II: bare proton
                            else:
                                return 1.0  # H III: impossible, but avoid errors
                        
                        # Helium - realistic behavior
                        elif atomic_num == 2:
                            if chrg == 0:
                                return 1.0  # He I: ground state
                            elif chrg == 1:
                                return 2.0  # He II: hydrogen-like
                            else:
                                return 1.0  # He III: bare nucleus
                        
                        # Other light elements (Z <= 10)
                        elif atomic_num <= 10:
                            if chrg == 0:
                                # Neutral atoms: modest temperature dependence
                                base = 2.0 + (atomic_num - 1) * 0.5
                                return base * (T / 5778.0)**0.1
                            elif chrg == 1:
                                # Singly ionized: simpler structure
                                base = 1.0 + (atomic_num - 1) * 0.2
                                return base * (T / 5778.0)**0.05
                            else:
                                # Doubly ionized: very simple
                                return 1.0
                        
                        # Heavy elements (Z > 10) - including iron
                        else:
                            if chrg == 0:
                                # Neutral heavy atoms: complex structure
                                if atomic_num == 26:  # Iron
                                    return 25.0 * (T / 5778.0)**0.3
                                else:
                                    base = 5.0 + (atomic_num - 11) * 1.0
                                    return base * (T / 5778.0)**0.2
                            elif chrg == 1:
                                # Singly ionized: moderate complexity
                                if atomic_num == 26:  # Iron II
                                    return 30.0 * (T / 5778.0)**0.25
                                else:
                                    base = 2.0 + (atomic_num - 11) * 0.5
                                    return base * (T / 5778.0)**0.15
                            else:
                                # Doubly ionized: simpler
                                if atomic_num == 26:  # Iron III
                                    return 25.0 * (T / 5778.0)**0.2
                                else:
                                    base = 1.0 + (atomic_num - 11) * 0.2
                                    return base * (T / 5778.0)**0.1
                    
                    return partition_func
                
                partition_funcs[species] = make_partition_func(Z, charge)
                
            except Exception:
                # Skip if Species creation fails
                continue
    
    return partition_funcs