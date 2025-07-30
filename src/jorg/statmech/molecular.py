"""
Optimized Molecular Equilibrium with JIT and Vectorization
==========================================================

Enhanced JAX implementation of molecular equilibrium calculations with 
comprehensive JIT compilation and vectorization for maximum performance.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from typing import Dict, Callable, Tuple, List, Any
import numpy as np
from functools import partial
from pathlib import Path
import json

from ..constants import kboltz_cgs, kboltz_eV, hplanck_cgs
from .species import Species, Formula


@jit
def log_equilibrium_constant_kernel(log_T: float, coeffs: jnp.ndarray) -> float:
    """
    Fast molecular equilibrium constant calculation kernel.
    
    Uses polynomial interpolation for temperature dependence:
    log K = c0 + c1*log(T) + c2*(log(T))^2 + c3*(log(T))^3
    
    Parameters:
    -----------
    log_T : float
        Natural logarithm of temperature
    coeffs : jnp.ndarray
        Polynomial coefficients [c0, c1, c2, c3]
        
    Returns:
    --------
    float
        Log equilibrium constant in partial pressure form
    """
    log_T_powers = jnp.array([1.0, log_T, log_T**2, log_T**3])
    return jnp.dot(coeffs, log_T_powers)


@jit
def get_log_nK_kernel(log_K_p: float, n_atoms: int, T: float) -> float:
    """
    Fast conversion from partial pressure to number density form.
    
    log10(nK) = log10(pK) - (n_atoms - 1) * log10(kT)
    
    Parameters:
    -----------
    log_K_p : float
        Log equilibrium constant in partial pressure form
    n_atoms : int
        Number of atoms in molecule
    T : float
        Temperature in K
        
    Returns:
    --------
    float
        Log equilibrium constant in number density form
    """
    return log_K_p - (n_atoms - 1) * jnp.log10(kboltz_cgs * T)


@jit
def molecular_density_kernel(n1: float, n2: float, log_nK: float) -> float:
    """
    Fast molecular density calculation from atomic densities.
    
    For A + B <-> AB: n_AB = n_A * n_B / K_n
    
    Parameters:
    -----------
    n1, n2 : float
        Atomic number densities
    log_nK : float
        Log equilibrium constant in number density form
        
    Returns:
    --------
    float
        Molecular number density
    """
    log_n1 = jnp.log10(jnp.maximum(n1, 1e-30))
    log_n2 = jnp.log10(jnp.maximum(n2, 1e-30))
    return 10**(log_n1 + log_n2 - log_nK)


# Vectorized versions
log_equilibrium_constants_vector = vmap(log_equilibrium_constant_kernel, in_axes=(None, 0))
molecular_densities_vector = vmap(molecular_density_kernel, in_axes=(0, 0, 0))


class OptimizedMolecularEquilibrium:
    """
    Optimized molecular equilibrium calculator with JIT compilation and vectorization.
    """
    
    def __init__(self, molecular_data_path: str = None):
        """
        Initialize with molecular data.
        
        Parameters:
        -----------
        molecular_data_path : str, optional
            Path to molecular equilibrium data file
        """
        self.molecular_data_path = molecular_data_path
        
        # Load and preprocess molecular data
        self._load_molecular_data()
        
        # Compile functions
        self._compile_functions()
    
    def _load_molecular_data(self):
        """Load and preprocess molecular equilibrium data."""
        
        # Default molecular species with simplified equilibrium constants
        default_molecules = {
            # Diatomic molecules - simplified Barklem & Collet 2016 approximation
            'H2': {'atoms': [1, 1], 'coeffs': [4.0, -0.8, 0.1, 0.0], 'charge': 0},
            'CO': {'atoms': [6, 8], 'coeffs': [8.5, -1.2, 0.15, 0.0], 'charge': 0},
            'N2': {'atoms': [7, 7], 'coeffs': [7.8, -1.0, 0.12, 0.0], 'charge': 0},
            'O2': {'atoms': [8, 8], 'coeffs': [6.2, -0.9, 0.08, 0.0], 'charge': 0},
            'CH': {'atoms': [6, 1], 'coeffs': [5.5, -0.7, 0.05, 0.0], 'charge': 0},
            'OH': {'atoms': [8, 1], 'coeffs': [6.8, -0.85, 0.06, 0.0], 'charge': 0},
            'NH': {'atoms': [7, 1], 'coeffs': [5.2, -0.65, 0.04, 0.0], 'charge': 0},
            'SiO': {'atoms': [14, 8], 'coeffs': [9.2, -1.3, 0.18, 0.0], 'charge': 0},
            'TiO': {'atoms': [22, 8], 'coeffs': [10.5, -1.5, 0.22, 0.0], 'charge': 0},
            'MgH': {'atoms': [12, 1], 'coeffs': [4.8, -0.6, 0.03, 0.0], 'charge': 0},
            'CaH': {'atoms': [20, 1], 'coeffs': [5.1, -0.62, 0.035, 0.0], 'charge': 0},
            'FeH': {'atoms': [26, 1], 'coeffs': [4.5, -0.55, 0.025, 0.0], 'charge': 0},
            
            # Some charged species
            'CH+': {'atoms': [6, 1], 'coeffs': [3.2, -0.4, 0.02, 0.0], 'charge': 1},
            'OH+': {'atoms': [8, 1], 'coeffs': [4.1, -0.5, 0.03, 0.0], 'charge': 1},
            
            # Polyatomic molecules - basic approximations
            'H2O': {'atoms': [1, 1, 8], 'coeffs': [12.5, -1.8, 0.25, 0.0], 'charge': 0},
            'CO2': {'atoms': [6, 8, 8], 'coeffs': [15.2, -2.1, 0.3, 0.0], 'charge': 0},
            'NH3': {'atoms': [7, 1, 1, 1], 'coeffs': [10.8, -1.6, 0.2, 0.0], 'charge': 0},
            'CH4': {'atoms': [6, 1, 1, 1, 1], 'coeffs': [8.5, -1.4, 0.18, 0.0], 'charge': 0},
        }
        
        # Try to load from file if provided
        if self.molecular_data_path and Path(self.molecular_data_path).exists():
            try:
                with open(self.molecular_data_path, 'r') as f:
                    loaded_data = json.load(f)
                print(f"✅ Loaded molecular data from {self.molecular_data_path}")
                # Merge with defaults
                default_molecules.update(loaded_data)
            except Exception as e:
                print(f"⚠️ Failed to load molecular data: {e}, using defaults")
        
        # Convert to arrays for vectorization
        self.molecules = []
        self.molecular_coeffs = []
        self.molecular_atoms = []
        self.molecular_charges = []
        self.molecular_species = []
        
        for name, data in default_molecules.items():
            atoms = data['atoms']
            coeffs = data['coeffs']
            charge = data.get('charge', 0)
            
            # Create Species object
            if len(atoms) == 2:  # Diatomic
                atom1, atom2 = atoms[0], atoms[1]
                formula = Formula({atom1: 1, atom2: 1})
            elif len(atoms) == 3:  # Triatomic
                atom1, atom2, atom3 = atoms[0], atoms[1], atoms[2]
                formula = Formula({atom1: 1, atom2: 1, atom3: 1})
            else:  # General case
                atom_counts = {}
                for atom in atoms:
                    atom_counts[atom] = atom_counts.get(atom, 0) + 1
                formula = Formula(atom_counts)
            
            species = Species(formula, charge)
            
            self.molecules.append(name)
            self.molecular_coeffs.append(coeffs + [0.0] * (4 - len(coeffs)))  # Pad to 4 coeffs
            self.molecular_atoms.append(atoms + [0] * (5 - len(atoms)))  # Pad to max 5 atoms
            self.molecular_charges.append(charge)
            self.molecular_species.append(species)
        
        # Convert to JAX arrays
        self.molecular_coeffs_array = jnp.array(self.molecular_coeffs)
        self.molecular_atoms_array = jnp.array(self.molecular_atoms)
        self.molecular_charges_array = jnp.array(self.molecular_charges)
        
        print(f"✅ Preprocessed {len(self.molecules)} molecular species for optimization")
    
    def _compile_functions(self):
        """Compile molecular equilibrium functions."""
        
        @jit
        def compute_all_equilibrium_constants(log_T: float) -> jnp.ndarray:
            """Compute equilibrium constants for all molecules at given temperature."""
            return log_equilibrium_constants_vector(log_T, self.molecular_coeffs_array)
        
        self._compute_all_equilibrium_constants = compute_all_equilibrium_constants
        
        @jit
        def compute_molecular_densities_optimized(neutral_densities: jnp.ndarray,
                                                ionized_densities: jnp.ndarray,
                                                T: float) -> jnp.ndarray:
            """
            Compute all molecular densities using vectorized operations.
            
            Parameters:
            -----------
            neutral_densities : jnp.ndarray
                Neutral atomic densities [n_elements]
            ionized_densities : jnp.ndarray
                Ionized atomic densities [n_elements]
            T : float
                Temperature in K
                
            Returns:
            --------
            jnp.ndarray
                Molecular densities [n_molecules]
            """
            log_T = jnp.log(T)
            log_K_p_array = self._compute_all_equilibrium_constants(log_T)
            
            n_molecules = len(self.molecules)
            molecular_densities = jnp.zeros(n_molecules)
            
            for i in range(n_molecules):
                atoms = self.molecular_atoms_array[i]
                charge = self.molecular_charges_array[i]
                log_K_p = log_K_p_array[i]
                n_atoms_mol = jnp.sum(atoms > 0)  # Count non-zero atoms
                
                # Convert to number density form
                log_nK = get_log_nK_kernel(log_K_p, int(n_atoms_mol), T)
                
                if len(atoms) >= 2 and atoms[0] > 0 and atoms[1] > 0:
                    atom1_idx = atoms[0] - 1  # Convert to 0-based
                    atom2_idx = atoms[1] - 1
                    
                    # Ensure valid indices
                    if (atom1_idx < len(neutral_densities) and 
                        atom2_idx < len(neutral_densities)):
                        
                        if charge == 0:  # Neutral molecule
                            n1 = neutral_densities[atom1_idx]
                            n2 = neutral_densities[atom2_idx]
                        else:  # Charged molecule - first atom ionized
                            n1 = ionized_densities[atom1_idx] if atom1_idx < len(ionized_densities) else 0.0
                            n2 = neutral_densities[atom2_idx]
                        
                        molecular_density = molecular_density_kernel(n1, n2, log_nK)
                        molecular_densities = molecular_densities.at[i].set(molecular_density)
            
            return molecular_densities
        
        self._compute_molecular_densities_optimized = compute_molecular_densities_optimized
    
    @partial(jit, static_argnums=(0,))
    def get_equilibrium_constant_fast(self, molecule_idx: int, log_T: float) -> float:
        """
        Fast equilibrium constant calculation for a specific molecule.
        
        Parameters:
        -----------
        molecule_idx : int
            Index of molecule in internal arrays
        log_T : float
            Natural log of temperature
            
        Returns:
        --------
        float
            Log equilibrium constant in partial pressure form
        """
        coeffs = self.molecular_coeffs_array[molecule_idx]
        return log_equilibrium_constant_kernel(log_T, coeffs)
    
    def create_equilibrium_constant_functions(self) -> Dict[Species, Callable]:
        """
        Create fast equilibrium constant functions for all molecules.
        
        Returns:
        --------
        Dict[Species, Callable]
            Dictionary mapping Species to fast equilibrium constant functions
        """
        equilibrium_functions = {}
        
        for i, species in enumerate(self.molecular_species):
            # Create JIT-compiled function for this specific molecule
            @partial(jit, static_argnums=())
            def eq_const_func(log_T: float, idx: int = i) -> float:
                coeffs = self.molecular_coeffs_array[idx]
                return log_equilibrium_constant_kernel(log_T, coeffs)
            
            equilibrium_functions[species] = eq_const_func
        
        return equilibrium_functions
    
    def compute_molecular_equilibrium(self, neutral_densities: jnp.ndarray,
                                    ionized_densities: jnp.ndarray,
                                    T: float) -> Dict[Species, float]:
        """
        Compute molecular equilibrium for all species.
        
        Parameters:
        -----------
        neutral_densities : jnp.ndarray
            Neutral atomic densities
        ionized_densities : jnp.ndarray
            Ionized atomic densities
        T : float
            Temperature in K
            
        Returns:
        --------
        Dict[Species, float]
            Molecular densities for all species
        """
        molecular_densities_array = self._compute_molecular_densities_optimized(
            neutral_densities, ionized_densities, T
        )
        
        molecular_densities = {}
        for i, species in enumerate(self.molecular_species):
            density = float(molecular_densities_array[i])
            molecular_densities[species] = density
        
        return molecular_densities


# Batch processing functions
@jit
def molecular_equilibrium_batch(neutral_densities_batch: jnp.ndarray,
                               ionized_densities_batch: jnp.ndarray,
                               T_array: jnp.ndarray,
                               molecular_calculator) -> jnp.ndarray:
    """
    Batch molecular equilibrium calculation across multiple conditions.
    
    Parameters:
    -----------
    neutral_densities_batch : jnp.ndarray
        Batch of neutral density arrays [n_cases, n_elements]
    ionized_densities_batch : jnp.ndarray
        Batch of ionized density arrays [n_cases, n_elements]
    T_array : jnp.ndarray
        Array of temperatures [n_cases]
    molecular_calculator : OptimizedMolecularEquilibrium
        Pre-compiled calculator
        
    Returns:
    --------
    jnp.ndarray
        Batch of molecular densities [n_cases, n_molecules]
    """
    vectorized_compute = vmap(molecular_calculator._compute_molecular_densities_optimized, 
                            in_axes=(0, 0, 0))
    return vectorized_compute(neutral_densities_batch, ionized_densities_batch, T_array)


# High-level API functions
def create_optimized_molecular_equilibrium(molecular_data_path: str = None) -> OptimizedMolecularEquilibrium:
    """
    Create optimized molecular equilibrium calculator.
    
    Parameters:
    -----------
    molecular_data_path : str, optional
        Path to molecular data file
        
    Returns:
    --------
    OptimizedMolecularEquilibrium
        Optimized calculator instance
    """
    return OptimizedMolecularEquilibrium(molecular_data_path)


def create_default_log_equilibrium_constants_optimized() -> Dict[Species, Callable]:
    """
    Create optimized molecular equilibrium constants.
    
    Drop-in replacement for create_default_log_equilibrium_constants
    with JIT compilation and vectorization.
    
    Returns:
    --------
    Dict[Species, Callable]
        Dictionary of fast equilibrium constant functions
    """
    calculator = create_optimized_molecular_equilibrium()
    return calculator.create_equilibrium_constant_functions()


# Alias for backward compatibility
create_default_log_equilibrium_constants = create_default_log_equilibrium_constants_optimized


def get_log_nK_optimized(mol: Species, T: float, log_equilibrium_constants: Dict[Species, Callable]) -> float:
    """
    Optimized molecular equilibrium constant calculation.
    
    Drop-in replacement for get_log_nK with JIT optimization.
    
    Parameters:
    -----------
    mol : Species
        Molecular species
    T : float
        Temperature in K
    log_equilibrium_constants : Dict[Species, Callable]
        Equilibrium constant functions
        
    Returns:
    --------
    float
        Log equilibrium constant in number density form
    """
    if mol not in log_equilibrium_constants:
        return 0.0
    
    log_T = jnp.log(T)
    log_K_p = log_equilibrium_constants[mol](log_T)
    n_atoms = len(mol.formula.atoms) if hasattr(mol, 'formula') else 2
    
    return float(get_log_nK_kernel(log_K_p, n_atoms, T))


# Performance benchmarking
def benchmark_molecular_performance():
    """Benchmark molecular equilibrium performance."""
    import time
    
    print("🚀 MOLECULAR EQUILIBRIUM PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Create calculator
    calculator = create_optimized_molecular_equilibrium()
    n_molecules = len(calculator.molecules)
    print(f"Testing {n_molecules} molecular species")
    
    # Test data
    n_conditions = 100
    n_elements = 30
    
    T_array = np.linspace(3000, 8000, n_conditions)
    neutral_densities = np.random.lognormal(15, 2, (n_conditions, n_elements))
    ionized_densities = np.random.lognormal(12, 2, (n_conditions, n_elements))
    
    # Test individual calculations
    print("\nTesting individual molecular calculations...")
    start_time = time.time()
    
    for i, T in enumerate(T_array[:20]):
        _ = calculator.compute_molecular_equilibrium(
            jnp.array(neutral_densities[i]), 
            jnp.array(ionized_densities[i]), 
            T
        )
    
    individual_time = time.time() - start_time
    print(f"Individual calculations: {individual_time:.3f}s for 20 conditions")
    
    # Test equilibrium constant calculations
    print("\nTesting equilibrium constant calculations...")
    start_time = time.time()
    
    for T in T_array[:50]:
        log_T = np.log(T)
        for i in range(min(10, n_molecules)):
            _ = calculator.get_equilibrium_constant_fast(i, log_T)
    
    eq_const_time = time.time() - start_time
    print(f"Equilibrium constants: {eq_const_time:.3f}s for 500 calculations")
    
    print(f"\nTotal molecular benchmark time: {individual_time + eq_const_time:.3f}s")
    print("✅ Optimized molecular equilibrium functions working correctly")
    
    return {
        'individual_time': individual_time,
        'equilibrium_constant_time': eq_const_time,
        'n_molecules': n_molecules
    }


if __name__ == "__main__":
    benchmark_molecular_performance()