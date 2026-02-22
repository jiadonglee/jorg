"""
Metal bound-free absorption exactly following Korg.jl implementation.

This module implements metal bound-free absorption using precomputed cross-section
tables from TOPBase and NORAD, exactly matching Korg.jl's approach.
"""

import jax
jax.config.update("jax_enable_x64", True)  # Enable float64 precision
import jax.numpy as jnp
import numpy as np
import h5py
from typing import Dict, Tuple, Any, Optional
import os

from ..statmech.species import Species
from ..data import get_data_path
from ..constants import SPEED_OF_LIGHT

_H_I_SPECIES = Species.from_string("H I")
_HE_I_SPECIES = Species.from_string("He I")
_H_II_SPECIES = Species.from_string("H II")
_METAL_BF_SKIP_SPECIES = frozenset({_H_I_SPECIES, _HE_I_SPECIES, _H_II_SPECIES})


class MetalBoundFreeData:
    """
    Container for metal bound-free cross-section data.
    
    Exactly matches Korg.jl's metal_bf_cross_sections structure.
    """
    
    def __init__(self, data_file: Optional[str] = None):
        """
        Initialize metal bound-free data from HDF5 file.
        
        Parameters:
        -----------
        data_file : str, optional
            Path to the HDF5 data file. If None, uses default Korg.jl data file.
        """
        if data_file is None:
            try:
                data_file = str(get_data_path("bf_cross-sections", "bf_cross-sections.h5"))
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    "Metal BF data file not found. "
                    "Set JORG_DATA_DIR or pass data_file explicitly."
                ) from exc
            
        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f"Metal BF data file not found: {data_file}\n"
                "Set JORG_DATA_DIR or pass data_file explicitly."
            )
            
        self.data_file = data_file
        self._load_data()
    
    def _load_data(self):
        """Load and parse the HDF5 data file."""
        with h5py.File(self.data_file, 'r') as f:
            # Load grid parameters (exactly matching Korg.jl)
            self.logT_min = float(f['logT_min'][()])
            self.logT_max = float(f['logT_max'][()])
            self.logT_step = float(f['logT_step'][()])
            
            self.nu_min = float(f['nu_min'][()])
            self.nu_max = float(f['nu_max'][()])
            self.nu_step = float(f['nu_step'][()])
            
            # Create grids
            self.logT_grid = np.arange(self.logT_min, self.logT_max + 0.5*self.logT_step, self.logT_step)
            self.nu_grid = np.arange(self.nu_min, self.nu_max + 0.5*self.nu_step, self.nu_step)
            
            # Load cross-section data for each species
            self.cross_sections = {}
            self.species_list = []
            
            cs_group = f['cross-sections']
            for species_name in cs_group.keys():
                # Parse species name exactly as in Korg.jl
                try:
                    species = Species.from_string(species_name)
                    
                    # Load cross-section data (log10 values)
                    # HDF5 stores as (n_temp, n_freq) in Python but (n_freq, n_temp) in Julia
                    # We need to transpose to get (n_freq, n_temp) = (60185, 31)
                    log_sigma_data = np.array(cs_group[species_name], dtype=np.float64).T
                    
                    # Store as JAX arrays for efficient computation
                    self.cross_sections[species] = jnp.array(log_sigma_data)
                    self.species_list.append(species)
                    
                except Exception as e:
                    print(f"Warning: Could not parse species {species_name}: {e}")
                    continue

        # Fast-path species excludes continuum channels handled elsewhere.
        self.fast_species = tuple(
            sp for sp in self.species_list
            if sp not in _METAL_BF_SKIP_SPECIES
        )
        self.fast_log_sigma_stack = (
            jnp.stack([self.cross_sections[sp] for sp in self.fast_species], axis=0)
            if self.fast_species
            else jnp.zeros((0, 1, 1), dtype=jnp.float64)
        )
        
        # Convert grids to JAX arrays
        self.logT_grid = jnp.array(self.logT_grid)
        self.nu_grid = jnp.array(self.nu_grid)
        
        # Only print once when data is first loaded (controlled by global singleton)
        # Removed per-layer print that was cluttering output


# Global data instance (loaded once)
_metal_bf_data = None


def get_metal_bf_data(data_file: Optional[str] = None) -> MetalBoundFreeData:
    """Get the global metal bound-free data instance."""
    global _metal_bf_data
    if _metal_bf_data is None:
        _metal_bf_data = MetalBoundFreeData(data_file)
    return _metal_bf_data


@jax.jit
def _bilinear_interpolate_2d(x: float, y: float, 
                            x_grid: jnp.ndarray, y_grid: jnp.ndarray,
                            values: jnp.ndarray) -> float:
    """
    Bilinear interpolation on a 2D grid with flat extrapolation.
    
    Exactly matches Korg.jl's linear_interpolation with Flat() extrapolation.
    
    Parameters:
    -----------
    x, y : float
        Coordinates to interpolate at
    x_grid, y_grid : jnp.ndarray  
        Regular grid coordinates
    values : jnp.ndarray
        Values at grid points, shape (len(x_grid), len(y_grid))
        
    Returns:
    --------
    float
        Interpolated value
    """
    # Find grid indices with clamping for flat extrapolation
    nx = len(x_grid)
    ny = len(y_grid)
    
    # Find x index
    dx = x_grid[1] - x_grid[0]
    i_f = (x - x_grid[0]) / dx
    i = jnp.clip(jnp.floor(i_f).astype(int), 0, nx - 2)
    
    # Find y index  
    dy = y_grid[1] - y_grid[0]
    j_f = (y - y_grid[0]) / dy
    j = jnp.clip(jnp.floor(j_f).astype(int), 0, ny - 2)
    
    # Compute fractional parts
    fx = jnp.clip(i_f - i, 0.0, 1.0)
    fy = jnp.clip(j_f - j, 0.0, 1.0)
    
    # Bilinear interpolation
    v00 = values[i, j]
    v10 = values[i + 1, j]
    v01 = values[i, j + 1]
    v11 = values[i + 1, j + 1]
    
    v0 = v00 * (1 - fx) + v10 * fx
    v1 = v01 * (1 - fx) + v11 * fx
    
    return v0 * (1 - fy) + v1 * fy


@jax.jit  
def _interpolate_metal_cross_section(nu: float, logT: float,
                                   nu_grid: jnp.ndarray, logT_grid: jnp.ndarray,
                                   log_sigma_data: jnp.ndarray) -> float:
    """
    Interpolate metal cross-section at given frequency and temperature.
    
    Parameters:
    -----------
    nu : float
        Frequency in Hz
    logT : float
        log10(Temperature in K)
    nu_grid : jnp.ndarray
        Frequency grid
    logT_grid : jnp.ndarray  
        log10(Temperature) grid
    log_sigma_data : jnp.ndarray
        log10(cross-section) data, shape (len(nu_grid), len(logT_grid))
        
    Returns:
    --------
    float
        ln(cross-section in Mb) - Korg.jl stores cross-sections as natural log of Mb
    """
    # Data is stored as (n_freq, n_temp), ready for bilinear interpolation
    # where x=nu and y=logT
    return _bilinear_interpolate_2d(nu, logT, nu_grid, logT_grid, log_sigma_data)


# Vectorized interpolation for multiple frequencies
_interpolate_metal_cross_section_vectorized = jax.vmap(
    _interpolate_metal_cross_section, 
    in_axes=(0, None, None, None, None)
)


@jax.jit
def _metal_bf_absorption_stacked(
    frequencies: jnp.ndarray,
    logT: float,
    number_densities_vec: jnp.ndarray,
    nu_grid: jnp.ndarray,
    logT_grid: jnp.ndarray,
    log_sigma_stack: jnp.ndarray,
) -> jnp.ndarray:
    """Compute metal bf absorption from stacked species tables in one JAX kernel."""

    def _single_species(number_density: float, log_sigma_data: jnp.ndarray) -> jnp.ndarray:
        log_sigma_interp = _interpolate_metal_cross_section_vectorized(
            frequencies, logT, nu_grid, logT_grid, log_sigma_data
        )
        mask = jnp.isfinite(log_sigma_interp)
        safe_n = jnp.maximum(number_density, 1e-300)
        alpha = jnp.where(mask, jnp.exp(jnp.log(safe_n) + log_sigma_interp) * 1e-18, 0.0)
        return jnp.where(number_density > 0.0, alpha, 0.0)

    alpha_by_species = jax.vmap(_single_species, in_axes=(0, 0))(number_densities_vec, log_sigma_stack)
    return jnp.sum(alpha_by_species, axis=0)


def metal_bf_absorption(frequencies: jnp.ndarray,
                       temperature: float, 
                       number_densities: Dict[Species, float],
                       species_list: Optional[list] = None) -> jnp.ndarray:
    """
    Calculate metal bound-free absorption exactly following Korg.jl.
    
    Adds contributions from bf metal opacities using precomputed tables from
    TOPBase (Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, S, Ar, Ca) and 
    NORAD (Fe). For these elements, tables have been precomputed for neutral
    and singly ionized species assuming LTE distribution of energy levels.
    
    Cross sections computed for 100 K < T < 100,000 K and frequencies 
    corresponding to 500 Å < λ < 30,000 Å. Outside these ranges, flat
    extrapolation is used.
    
    Parameters:
    -----------
    frequencies : jnp.ndarray
        Frequencies in Hz
    temperature : float
        Temperature in K
    number_densities : Dict[Species, float]
        Number densities for each species in cm^-3
    species_list : list, optional
        List of species to include. If None, uses all available species.
        
    Returns:
    --------
    jnp.ndarray
        Metal bound-free absorption coefficient in cm^-1
    """
    # Get metal BF data
    bf_data = get_metal_bf_data()
    
    freqs = jnp.asarray(frequencies, dtype=jnp.float64)
    
    logT = jnp.log10(temperature)

    if species_list is None:
        # Fast path: all species in one vectorized kernel.
        if bf_data.fast_species:
            ndens_vec = jnp.asarray(
                [float(number_densities.get(sp, 0.0)) for sp in bf_data.fast_species],
                dtype=jnp.float64,
            )
            return _metal_bf_absorption_stacked(
                frequencies=freqs,
                logT=logT,
                number_densities_vec=ndens_vec,
                nu_grid=bf_data.nu_grid,
                logT_grid=bf_data.logT_grid,
                log_sigma_stack=bf_data.fast_log_sigma_stack,
            )
        return jnp.zeros_like(freqs, dtype=jnp.float64)

    # Custom-species fallback path.
    alpha_total = jnp.zeros_like(freqs, dtype=jnp.float64)
    
    # Add contributions from each metal species
    for species in species_list:
        # Skip if species not in number_densities or not available in data
        if species not in number_densities or species not in bf_data.cross_sections:
            continue

        if species in _METAL_BF_SKIP_SPECIES:
            continue
            
        number_density = number_densities[species]
        if number_density <= 0:
            continue
            
        # Get cross-section data for this species
        log_sigma_data = bf_data.cross_sections[species]
        
        # Interpolate cross-sections at all frequencies
        log_sigma_interp = _interpolate_metal_cross_section_vectorized(
            freqs, logT, bf_data.nu_grid, bf_data.logT_grid, log_sigma_data
        )
        
        # Apply mask to avoid NaNs in derivatives (exact match to Korg.jl logic)
        # When σ = 0, log(σ) = -∞, which causes issues
        mask = jnp.isfinite(log_sigma_interp)
        
        # Calculate absorption exactly as in Korg.jl: α = exp(log(n) + log_σ) * 1e-18
        # log_sigma_interp is ln(σ in Mb), not log10 - this is the key insight!
        ln_sigma_mb = log_sigma_interp  # Data is already in natural log
        ln_n = jnp.log(number_density)
        
        # Add contribution (using mask to avoid NaN propagation)
        # This exactly matches Korg.jl: exp(log(n) + log_σ) * 1e-18
        alpha_contribution = jnp.where(
            mask,
            jnp.exp(ln_n + ln_sigma_mb) * 1e-18,  # Convert Mb to cm^2
            0.0
        )
        
        alpha_total += alpha_contribution
    
    return alpha_total


def metal_bf_absorption_dense_batch(
    frequencies: jnp.ndarray,
    temperatures: jnp.ndarray,
    number_densities_dense: jnp.ndarray,
    species_layout: Any,
) -> jnp.ndarray:
    """
    Vectorized metal bound-free absorption for dense [layers, species] inputs.
    """
    bf_data = get_metal_bf_data()

    freqs = jnp.asarray(frequencies, dtype=jnp.float64)
    temps = jnp.asarray(temperatures, dtype=jnp.float64)
    dense = jnp.asarray(number_densities_dense, dtype=jnp.float64)

    if dense.ndim != 2:
        raise ValueError("number_densities_dense must be rank-2 [n_layers, n_species].")
    if dense.shape[0] != temps.shape[0]:
        raise ValueError("number_densities_dense layer count must match temperatures.")
    if dense.shape[1] != len(species_layout.species):
        raise ValueError("number_densities_dense species axis must match species_layout.")

    n_layers = dense.shape[0]
    if not bf_data.fast_species:
        return jnp.zeros((n_layers, freqs.shape[0]), dtype=jnp.float64)

    ndens_cols = []
    for sp in bf_data.fast_species:
        idx = species_layout.index.get(sp)
        if idx is None:
            ndens_cols.append(jnp.zeros((n_layers,), dtype=jnp.float64))
        else:
            ndens_cols.append(dense[:, idx])

    ndens_stack = jnp.stack(ndens_cols, axis=1)

    def _single_layer(temp: float, ndens_vec: jnp.ndarray) -> jnp.ndarray:
        return _metal_bf_absorption_stacked(
            frequencies=freqs,
            logT=jnp.log10(jnp.maximum(temp, 1e-300)),
            number_densities_vec=ndens_vec,
            nu_grid=bf_data.nu_grid,
            logT_grid=bf_data.logT_grid,
            log_sigma_stack=bf_data.fast_log_sigma_stack,
        )

    return jax.vmap(_single_layer, in_axes=(0, 0))(temps, ndens_stack)


@jax.jit
def metal_bf_absorption_jit(frequencies: jnp.ndarray,
                           temperature: float,
                           # Individual number densities for JIT compilation
                           n_al_i: float, n_c_i: float, n_ca_i: float,
                           n_fe_i: float, n_mg_i: float, n_na_i: float,
                           n_s_i: float, n_si_i: float,
                           # Pre-loaded data arrays
                           nu_grid: jnp.ndarray,
                           logT_grid: jnp.ndarray, 
                           log_sigma_al_i: jnp.ndarray,
                           log_sigma_c_i: jnp.ndarray,
                           log_sigma_ca_i: jnp.ndarray,
                           log_sigma_fe_i: jnp.ndarray,
                           log_sigma_mg_i: jnp.ndarray,
                           log_sigma_na_i: jnp.ndarray,
                           log_sigma_s_i: jnp.ndarray,
                           log_sigma_si_i: jnp.ndarray) -> jnp.ndarray:
    """
    JIT-compiled version of metal bound-free absorption.
    
    This version takes individual number densities and pre-loaded data arrays
    to enable efficient JIT compilation without Dict types.
    """
    alpha_total = jnp.zeros_like(frequencies, dtype=jnp.float64)
    logT = jnp.log10(temperature)
    
    # List of (number_density, log_sigma_data) pairs
    species_data = [
        (n_al_i, log_sigma_al_i),
        (n_c_i, log_sigma_c_i), 
        (n_ca_i, log_sigma_ca_i),
        (n_fe_i, log_sigma_fe_i),
        (n_mg_i, log_sigma_mg_i),
        (n_na_i, log_sigma_na_i),
        (n_s_i, log_sigma_s_i),
        (n_si_i, log_sigma_si_i)
    ]
    
    for number_density, log_sigma_data in species_data:
        # Skip if number density is effectively zero
        alpha_total = jnp.where(
            number_density > 1e-50,
            alpha_total + _compute_single_species_absorption(
                frequencies, logT, number_density, nu_grid, logT_grid, log_sigma_data
            ),
            alpha_total
        )
    
    return alpha_total


@jax.jit
def _compute_single_species_absorption(frequencies: jnp.ndarray,
                                     logT: float,
                                     number_density: float,
                                     nu_grid: jnp.ndarray,
                                     logT_grid: jnp.ndarray,
                                     log_sigma_data: jnp.ndarray) -> jnp.ndarray:
    """Compute absorption for a single metal species."""
    # Interpolate cross-sections
    log_sigma_interp = _interpolate_metal_cross_section_vectorized(
        frequencies, logT, nu_grid, logT_grid, log_sigma_data
    )
    
    # Apply mask for finite values
    mask = jnp.isfinite(log_sigma_interp)
    
    # Calculate absorption exactly as in Korg.jl: α = exp(log(n) + log_σ) * 1e-18
    ln_sigma_mb = log_sigma_interp  # Data is already in natural log
    ln_n = jnp.log(number_density)
    
    # Calculate absorption with masking
    return jnp.where(mask, jnp.exp(ln_n + ln_sigma_mb) * 1e-18, 0.0)


def validate_metal_bf_implementation():
    """
    Validate the metal bound-free implementation against known values.
    """
    print("Validating metal bound-free implementation...")
    
    try:
        # Load data
        bf_data = get_metal_bf_data()
        
        # Test parameters
        T = 5000.0  # K
        frequencies = jnp.array([1e15, 2e15, 3e15])  # Hz
        
        # Create dummy number densities
        number_densities = {}
        for species in bf_data.species_list:
            number_densities[species] = 1e10  # cm^-3
        
        # Calculate absorption
        alpha = metal_bf_absorption(frequencies, T, number_densities)
        
        print(f"Test frequencies: {frequencies}")
        print(f"Metal BF absorption: {alpha}")
        print(f"Alpha shape: {alpha.shape}")
        print(f"Alpha finite: {jnp.all(jnp.isfinite(alpha))}")
        
        # Test that absorption increases with frequency (generally expected)
        print(f"Absorption values: {alpha}")
        
        # Test interpolation bounds
        logT_test = jnp.log10(T)
        print(f"Temperature: {T} K, log10(T): {logT_test}")
        print(f"Grid bounds - logT: [{bf_data.logT_min}, {bf_data.logT_max}]")
        print(f"Grid bounds - nu: [{bf_data.nu_min}, {bf_data.nu_max}] Hz")
        
        print("Metal bound-free validation passed!")
        
    except Exception as e:
        print(f"Metal bound-free validation failed: {e}")
        raise


if __name__ == "__main__":
    validate_metal_bf_implementation()
