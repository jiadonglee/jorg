"""
Core chemical equilibrium solver for stellar atmospheres.

Based on Korg.jl statistical mechanics implementation using JAX.
"""

import jax.numpy as jnp
from jax import jit
from jax.scipy.optimize import minimize
import numpy as np
from typing import Dict, Tuple, Any, Optional
from ..constants import kboltz_cgs
from .ionization import saha_ion_weights, create_default_ionization_energies
from .partition_functions import create_partition_function_dict
from .molecular import get_log_nk, create_default_equilibrium_constants

# Maximum atomic number to consider
MAX_ATOMIC_NUMBER = 92


class ChemicalEquilibriumError(Exception):
    """Exception raised when chemical equilibrium calculation fails."""
    def __init__(self, msg: str):
        self.msg = msg
        super().__init__(f"Chemical equilibrium failed: {msg}")


@jit
def compute_neutral_fraction_guess(T: float, ne_model: float, 
                                 ionization_energies: Dict[int, Tuple[float, float, float]],
                                 partition_funcs: Dict[str, Any]) -> jnp.ndarray:
    """
    Compute initial guess for neutral fractions by neglecting molecules.
    
    Parameters:
    -----------
    T : float
        Temperature in K
    ne_model : float
        Model atmosphere electron density in cm^-3
    ionization_energies : dict
        Ionization energies for each element
    partition_funcs : dict
        Partition functions for each species
        
    Returns:
    --------
    jnp.ndarray
        Initial neutral fraction guess for each element
    """
    neutral_fractions = []
    
    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        wII, wIII = saha_ion_weights(T, ne_model, Z, ionization_energies, partition_funcs)
        neutral_fraction = 1.0 / (1.0 + wII + wIII)
        neutral_fractions.append(neutral_fraction)
    
    return jnp.array(neutral_fractions)


@jit 
def chemical_equilibrium_residuals(x: jnp.ndarray, T: float, n_total: float,
                                 absolute_abundances: jnp.ndarray,
                                 ionization_energies: Dict[int, Tuple[float, float, float]],
                                 partition_funcs: Dict[str, Any],
                                 log_equilibrium_constants: Dict[str, Any]) -> jnp.ndarray:
    """
    Calculate residuals for chemical equilibrium system.
    
    Parameters:
    -----------
    x : jnp.ndarray
        Solution vector: [neutral_fractions, ne_scaled]
    T : float
        Temperature in K
    n_total : float
        Total number density in cm^-3
    absolute_abundances : jnp.ndarray
        Absolute abundances (N_X/N_total)
    ionization_energies : dict
        Ionization energies
    partition_funcs : dict
        Partition functions
    log_equilibrium_constants : dict
        Molecular equilibrium constants
        
    Returns:
    --------
    jnp.ndarray
        Residual vector
    """
    # Extract variables
    neutral_fractions = jnp.abs(x[:MAX_ATOMIC_NUMBER])
    ne = jnp.abs(x[MAX_ATOMIC_NUMBER]) * n_total * 1e-5
    
    # Atom number densities
    atom_densities = absolute_abundances * (n_total - ne)
    neutral_densities = atom_densities * neutral_fractions
    
    residuals = jnp.zeros(MAX_ATOMIC_NUMBER + 1)
    
    # Atomic conservation equations
    for Z in range(MAX_ATOMIC_NUMBER):
        atom_idx = Z
        wII, wIII = saha_ion_weights(T, ne, Z + 1, ionization_energies, partition_funcs)
        
        # Conservation: total atoms = neutral + ionized
        total_in_all_stages = (1.0 + wII + wIII) * neutral_densities[Z]
        residuals = residuals.at[atom_idx].set(atom_densities[Z] - total_in_all_stages)
        
        # Electron conservation contribution
        electrons_from_this_element = (wII + 2.0 * wIII) * neutral_densities[Z]
        residuals = residuals.at[MAX_ATOMIC_NUMBER].add(electrons_from_this_element)
    
    # Complete electron conservation equation
    residuals = residuals.at[MAX_ATOMIC_NUMBER].add(-ne)
    
    # Normalize residuals
    residuals = residuals.at[:MAX_ATOMIC_NUMBER].set(
        residuals[:MAX_ATOMIC_NUMBER] / atom_densities
    )
    residuals = residuals.at[MAX_ATOMIC_NUMBER].set(
        residuals[MAX_ATOMIC_NUMBER] / (ne * 1e-5)
    )
    
    return residuals


def chemical_equilibrium(T: float, n_total: float, ne_model: float,
                        absolute_abundances: np.ndarray,
                        ionization_energies: Optional[Dict] = None,
                        partition_funcs: Optional[Dict] = None,
                        log_equilibrium_constants: Optional[Dict] = None,
                        electron_density_warn_threshold: float = 0.1,
                        electron_density_warn_min_value: float = 1e-4) -> Tuple[float, Dict[str, float]]:
    """
    Solve for chemical equilibrium in stellar atmosphere.
    
    Parameters:
    -----------
    T : float
        Temperature in K
    n_total : float
        Total number density in cm^-3
    ne_model : float
        Model atmosphere electron density in cm^-3
    absolute_abundances : np.ndarray
        Absolute abundances (N_X/N_total) for elements 1-92
    ionization_energies : dict, optional
        Ionization energies for each element
    partition_funcs : dict, optional
        Partition functions
    log_equilibrium_constants : dict, optional
        Molecular equilibrium constants
    electron_density_warn_threshold : float
        Threshold for electron density warning
    electron_density_warn_min_value : float
        Minimum value for electron density warning
        
    Returns:
    --------
    Tuple[float, Dict[str, float]]
        (electron_density, species_number_densities)
    """
    # Use defaults if not provided
    if ionization_energies is None:
        ionization_energies = create_default_ionization_energies()
    if partition_funcs is None:
        partition_funcs = create_partition_function_dict()
    if log_equilibrium_constants is None:
        log_equilibrium_constants = create_default_equilibrium_constants()
    
    # Convert to JAX arrays
    abs_abundances_jax = jnp.array(absolute_abundances[:MAX_ATOMIC_NUMBER])
    
    # Compute initial guess
    neutral_guess = compute_neutral_fraction_guess(T, ne_model, ionization_energies, partition_funcs)
    ne_guess_scaled = ne_model / n_total * 1e5
    
    x0 = jnp.concatenate([neutral_guess, jnp.array([ne_guess_scaled])])
    
    # Define objective function for optimization
    def objective(x):
        residuals = chemical_equilibrium_residuals(x, T, n_total, abs_abundances_jax,
                                                 ionization_energies, partition_funcs,
                                                 log_equilibrium_constants)
        return jnp.sum(residuals**2)
    
    # Solve using JAX optimization
    try:
        result = minimize(objective, x0, method='BFGS')
        if not result.success:
            raise ChemicalEquilibriumError("Optimization failed to converge")
        
        solution = result.x
    except Exception as e:
        raise ChemicalEquilibriumError(f"Solver failed: {e}")
    
    # Extract results
    neutral_fractions = jnp.abs(solution[:MAX_ATOMIC_NUMBER])
    ne = jnp.abs(solution[MAX_ATOMIC_NUMBER]) * n_total * 1e-5
    
    # Check electron density convergence
    if (ne / n_total > electron_density_warn_min_value and 
        abs((ne - ne_model) / ne_model) > electron_density_warn_threshold):
        print(f"Warning: Electron density differs from model by factor > {electron_density_warn_threshold}")
        print(f"Calculated ne = {ne:.3e}, model ne = {ne_model:.3e}")
    
    # Build number densities dictionary
    number_densities = {}
    
    # Neutral atomic species
    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        neutral_density = (n_total - ne) * absolute_abundances[Z-1] * neutral_fractions[Z-1]
        number_densities[f"H_{Z}_0"] = float(neutral_density)  # Neutral
        
        # Ionized species
        wII, wIII = saha_ion_weights(T, ne, Z, ionization_energies, partition_funcs)
        number_densities[f"H_{Z}_1"] = float(wII * neutral_density)  # Singly ionized
        number_densities[f"H_{Z}_2"] = float(wIII * neutral_density)  # Doubly ionized
    
    # Add molecular species (simplified)
    for mol_id in log_equilibrium_constants:
        # Simplified molecular calculation
        log_nK = get_log_nk(mol_id, T, log_equilibrium_constants)
        # Would need proper molecular equilibrium calculation here
        number_densities[mol_id] = 1e-10  # Placeholder
    
    return float(ne), number_densities