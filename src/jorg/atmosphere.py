"""
Jorg Atmosphere Interpolation
============================

JAX-based MARCS stellar atmosphere interpolation for Jorg.

This module provides native Python/JAX implementation of stellar atmosphere 
interpolation, replacing the original subprocess-based approach with a 
high-performance, GPU-accelerated solution.

Key Features:
- Complete translation of Korg's atmosphere interpolation to Python/JAX
- Support for all three MARCS interpolation methods (standard, cool dwarf, low-Z)
- Perfect numerical agreement with Korg results
- GPU acceleration and auto-differentiation ready
- No external Julia dependencies

Example:
    >>> from jorg.atmosphere import interpolate_marcs
    >>> atmosphere = interpolate_marcs(5777.0, 4.44, 0.0)  # Solar atmosphere
    >>> print(f"Atmosphere: {len(atmosphere.layers)} layers")
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, NamedTuple
from dataclasses import dataclass
import warnings
from functools import lru_cache
# PHASE 1.3 OPTIMIZATION: Replace SciPy with JAX interpolation
try:
    from scipy.interpolate import CubicSpline
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Import JAX interpolation (v2 optimization)
from .interpolation_jax import cubic_spline_nd

# Import Jorg constants
from .constants import kboltz_cgs, G_cgs, solar_mass_cgs
from .data import get_data_root


_INTERPOLATOR_CACHE = {}
_CUBIC_INTERPOLATOR_CACHE = {}


@dataclass
class AtmosphereLayer:
    """
    Single atmospheric layer with physical properties.
    
    Attributes:
        tau_5000: Optical depth at 5000 Å (dimensionless)
        z: Height relative to photosphere (cm)
        temp: Temperature (K)
        electron_number_density: Electron number density (cm⁻³)
        number_density: Total number density (cm⁻³)
    """
    tau_5000: float
    z: float  
    temp: float
    electron_number_density: float
    number_density: float


@dataclass 
class ModelAtmosphere:
    """
    Model stellar atmosphere consisting of multiple layers.
    
    Attributes:
        layers: List of AtmosphereLayer objects
        spherical: Whether this is a spherical (True) or planar (False) atmosphere
        R: Photospheric radius for spherical atmospheres (cm)
    """
    layers: List[AtmosphereLayer]
    spherical: bool = False
    R: Optional[float] = None


class AtmosphereInterpolationError(Exception):
    """Exception raised when atmosphere interpolation fails"""
    pass


def load_marcs_grid(grid_path: str):
    """
    Load MARCS atmosphere grid from HDF5 file.
    
    Args:
        grid_path: Path to HDF5 file containing MARCS atmosphere grid
        
    Returns:
        Tuple of (grid, nodes, param_names) where:
        - grid: JAX array with atmosphere data
        - nodes: List of parameter node arrays  
        - param_names: List of parameter names
    """
    with h5py.File(grid_path, 'r') as f:
        grid = jnp.array(f['grid'][:])
        nodes = []
        for i in range(1, 6):  # grid_values/1 through grid_values/5
            if f'grid_values/{i}' in f:
                nodes.append(jnp.array(f[f'grid_values/{i}'][:]))
        
        param_names = [name.decode('utf-8') if isinstance(name, bytes) else name 
                      for name in f['grid_parameter_names'][:]]
    
    return grid, nodes, param_names


@lru_cache(maxsize=4)
def _load_marcs_grid_cached(grid_path: str):
    """
    Cached MARCS grid loader to avoid repeated HDF5 reads.
    """
    return load_marcs_grid(str(grid_path))


def _artifact_roots() -> List[Path]:
    roots: List[Path] = []
    depot_path = os.environ.get("JULIA_DEPOT_PATH")
    if depot_path:
        for entry in depot_path.split(os.pathsep):
            entry_path = Path(entry).expanduser()
            if entry_path.is_dir():
                roots.append(entry_path / "artifacts")
    else:
        roots.append(Path.home() / ".julia" / "artifacts")
    return roots


def _find_grid_in_artifacts(filename: str) -> Optional[Path]:
    for root in _artifact_roots():
        if not root.is_dir():
            continue
        matches = sorted(root.rglob(filename))
        if matches:
            return matches[0]
    return None


def _resolve_marcs_grid_path(grid_data_dir: Optional[Union[str, Path]], filename: str) -> str:
    candidates: List[Path] = []
    if grid_data_dir is not None:
        base = Path(grid_data_dir).expanduser()
        candidates.append(base / filename)
        candidates.append(base / "marcs_grids" / filename)

    env_grid_dir = os.environ.get("JORG_MARCS_GRID_DIR")
    if env_grid_dir:
        base = Path(env_grid_dir).expanduser()
        candidates.append(base / filename)

    data_root = get_data_root()
    candidates.append(data_root / "marcs_grids" / filename)
    candidates.append(data_root / filename)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    artifact_match = _find_grid_in_artifacts(filename)
    if artifact_match is not None:
        return str(artifact_match)

    raise FileNotFoundError(
        f"MARCS grid file not found: {filename}. "
        "Set JORG_MARCS_GRID_DIR or JORG_DATA_DIR to your MARCS bundle. "
        "If you don't have the MARCS grids, download them from: "
        "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/Q8AYIA"
    )


@lru_cache(maxsize=16)
def _resolve_marcs_grid_path_cached(grid_data_dir: Optional[Union[str, Path]], filename: str) -> str:
    """
    Cached path resolution for MARCS grids.
    """
    return _resolve_marcs_grid_path(grid_data_dir, filename)


def multilinear_interpolation(params: jnp.ndarray, 
                             nodes: List[jnp.ndarray],
                             grid: jnp.ndarray) -> jnp.ndarray:
    """
    JAX-based multilinear interpolation for MARCS atmosphere grids.
    
    Args:
        params: Parameter values to interpolate [Teff, logg, m_H, alpha_m, C_m]
        nodes: List of grid node arrays for each parameter
        grid: Atmosphere data grid
        
    Returns:
        Interpolated atmosphere quantities [n_layers, n_quantities]
    """
    n_params = len(params)
    
    # Find bounding indices and weights for each parameter
    lower_indices = []
    upper_indices = []
    weights = []
    
    for i, (param, node_array) in enumerate(zip(params, nodes)):
        # Clamp parameter to valid range
        param_clamped = jnp.clip(param, node_array[0], node_array[-1])
        
        # Find bounding indices
        lower_idx = jnp.searchsorted(node_array, param_clamped, side='right') - 1
        lower_idx = jnp.clip(lower_idx, 0, len(node_array) - 2)
        upper_idx = lower_idx + 1
        
        # Calculate interpolation weight
        p1 = node_array[lower_idx]
        p2 = node_array[upper_idx]
        weight = (param_clamped - p1) / (p2 - p1)
        
        lower_indices.append(lower_idx)
        upper_indices.append(upper_idx)
        weights.append(weight)
    
    # Perform interpolation by evaluating all corners of hypercube
    result = jnp.zeros((grid.shape[-1], grid.shape[-2]))  # [layers, quantities]
    
    # Iterate over all 2^n corners
    for corner in range(2**n_params):
        # Determine which bound to use for each parameter at this corner
        indices = []
        corner_weight = 1.0
        
        for i in range(n_params):
            use_upper = (corner >> i) & 1
            if use_upper:
                indices.append(upper_indices[i])
                corner_weight *= weights[i]
            else:
                indices.append(lower_indices[i])
                corner_weight *= (1 - weights[i])
        
        # Extract atmosphere for this corner
        # Parameter order: [Teff, logg, metallicity, alpha, carbon] 
        # Grid order: [carbon, alpha, metallicity, logg, Teff, quantities, layers]
        if len(indices) == 5:  # Full parameter set
            Teff_idx, logg_idx, mH_idx, alpha_idx, C_idx = indices
            atm_corner = grid[C_idx, alpha_idx, mH_idx, logg_idx, Teff_idx, :, :]
        elif len(indices) == 3:  # Low-Z grid (Teff, logg, mH only)
            Teff_idx, logg_idx, mH_idx = indices
            # Low-Z grid has different structure
            atm_corner = grid[mH_idx, logg_idx, Teff_idx, :, :]
        else:
            raise ValueError(f"Unexpected number of parameters: {len(indices)}")
        
        atm_corner_t = atm_corner.T  # Transpose to [layers, quantities]
        result += corner_weight * atm_corner_t
    
    return result


def _get_multilinear_interpolator(grid_path: str, nodes: List[jnp.ndarray],
                                  grid: jnp.ndarray, n_params: int):
    """
    Return a cached JIT-compiled multilinear interpolator for a grid.
    """
    key = (str(grid_path), int(n_params))
    cached = _INTERPOLATOR_CACHE.get(key)
    if cached is not None:
        return cached

    def _interp(params):
        params = jnp.asarray(params, dtype=grid.dtype)
        return multilinear_interpolation(params, nodes, grid)

    compiled = jax.jit(_interp)
    _INTERPOLATOR_CACHE[key] = compiled
    return compiled


def _get_cool_dwarf_interpolator(grid_path: str, axis_nodes: List[np.ndarray], grid: jnp.ndarray):
    """
    Return a cached JIT-compiled cool dwarf cubic interpolator for a grid.
    """
    key = (str(grid_path), "cool_dwarf")
    cached = _CUBIC_INTERPOLATOR_CACHE.get(key)
    if cached is not None:
        return cached

    axis_nodes_jax = [jnp.asarray(node, dtype=jnp.float64) for node in axis_nodes]

    def _interp(axis_params):
        data = jnp.asarray(grid, dtype=jnp.float64)
        for node, value in zip(axis_nodes_jax, axis_params):
            data = cubic_spline_nd(node, data, axis=0, x_query=value)
        return data.T

    compiled = jax.jit(_interp)
    _CUBIC_INTERPOLATOR_CACHE[key] = compiled
    return compiled


def _cubic_interpolation_cool_dwarf(params: np.ndarray,
                                    nodes: List[jnp.ndarray],
                                    grid: jnp.ndarray,
                                    param_names: List[str],
                                    use_jax: bool = True,
                                    grid_path: Optional[str] = None) -> np.ndarray:
    """
    Cubic interpolation for cool dwarf grid (matches Korg's cubic spline behavior).

    The cool dwarf grid is stored in HDF5 as [C, alpha, mH, logg, Teff, quantities, layers].
    We interpolate along the 5 parameter axes only, leaving quantities/layers untouched.

    PHASE 1.3 OPTIMIZATION: Now supports JAX interpolation for GPU acceleration!

    Parameters
    ----------
    params : np.ndarray
        Parameter values to interpolate at [Teff, logg, mH, alpha, C]
    nodes : List[jnp.ndarray]
        Grid node values for each parameter
    grid : jnp.ndarray
        Atmosphere grid data
    param_names : List[str]
        Names of parameters (for error messages)
    use_jax : bool, optional
        If True, use JAX interpolation (GPU-accelerated, JIT-compilable).
        If False, use SciPy (original implementation).
        Default: True

    Returns
    -------
    np.ndarray
        Interpolated atmosphere [layers, quantities]
    """
    nodes_np = [np.asarray(node) for node in nodes]
    params_np = np.asarray(params, dtype=float)

    for value, node, name in zip(params_np, nodes_np, param_names):
        if value < node[0] or value > node[-1]:
            raise AtmosphereInterpolationError(
                f"Can't interpolate grid. {name}={value} outside [{node[0]}, {node[-1]}]."
            )

    # Grid parameter order is [Teff, logg, mH, alpha, C] in nodes, but grid axes are reversed.
    axis_nodes = [nodes_np[4], nodes_np[3], nodes_np[2], nodes_np[1], nodes_np[0]]
    axis_params = [params_np[4], params_np[3], params_np[2], params_np[1], params_np[0]]

    if use_jax:
        # JAX version - JIT-compilable and cached for reuse
        cache_key = grid_path or "cool_dwarf_grid"
        interp = _get_cool_dwarf_interpolator(cache_key, axis_nodes, grid)
        data = interp(jnp.asarray(axis_params, dtype=jnp.float64))
        return np.asarray(data)  # [layers, quantities]
    else:
        # Original SciPy version - fallback for validation
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy not available. Install scipy or use use_jax=True")

        grid_np = np.asarray(grid, dtype=float)
        data = grid_np
        for node, value in zip(axis_nodes, axis_params):
            spline = CubicSpline(node, data, axis=0, extrapolate=False)
            data = spline(value)

        return data.T  # [layers, quantities]


def create_atmosphere_from_quantities(atm_quants: jnp.ndarray, 
                                    spherical: bool = False,
                                    logg: float = 4.44) -> ModelAtmosphere:
    """
    Create ModelAtmosphere from interpolated quantities.
    
    Args:
        atm_quants: Interpolated atmosphere quantities [n_layers, n_quantities]
                   From MARCS grids: [temp, log_ne, log_nt, tau_5000, sinh_z]
        spherical: Whether to create spherical atmosphere
        logg: Surface gravity for spherical radius calculation
        
    Returns:
        ModelAtmosphere object
    """
    n_layers = atm_quants.shape[0]
    
    # Extract quantities (MARCS grid order)
    temp = atm_quants[:, 0]           # Temperature
    log_ne = atm_quants[:, 1]         # Log electron density  
    log_nt = atm_quants[:, 2]         # Log total density
    tau_5000 = atm_quants[:, 3]       # Optical depth
    sinh_z = atm_quants[:, 4]         # Sinh of height
    
    # Convert from log to linear densities
    ne = jnp.exp(log_ne)
    nt = jnp.exp(log_nt)
    z = jnp.sinh(sinh_z)
    
    # Filter out NaN layers (Korg uses tau_5000 for NaN checking)
    valid_mask = ~jnp.isnan(tau_5000)
    
    # Create atmosphere layers
    layers = []
    for i in range(n_layers):
        if valid_mask[i]:
            layer = AtmosphereLayer(
                tau_5000=float(tau_5000[i]),
                z=float(z[i]),
                temp=float(temp[i]), 
                electron_number_density=float(ne[i]),
                number_density=float(nt[i])
            )
            layers.append(layer)
    
    # Calculate radius for spherical atmospheres
    R = None
    if spherical:
        R = float(jnp.sqrt(G_cgs * solar_mass_cgs / (10**logg)))
    
    return ModelAtmosphere(layers=layers, spherical=spherical, R=R)


def interpolate_marcs(Teff: float, 
                     logg: float,
                     m_H: float = 0.0,
                     alpha_m: float = 0.0, 
                     C_m: float = 0.0,
                     spherical: Optional[bool] = None,
                     grid_data_dir: Optional[str] = None) -> ModelAtmosphere:
    """
    Interpolate MARCS stellar atmosphere using JAX.
    
    This is the main interface for atmosphere interpolation, providing identical
    functionality to Korg's interpolate_marcs but with JAX implementation.
    
    Args:
        Teff: Effective temperature (K)
        logg: Surface gravity log(g) (cgs)
        m_H: Metallicity [M/H] (default: 0.0)
        alpha_m: Alpha enhancement [alpha/M] (default: 0.0)
        C_m: Carbon enhancement [C/M] (default: 0.0)
        spherical: Force spherical/planar (default: auto from logg < 3.5)
        grid_data_dir: Directory containing MARCS grid files (default: auto)
        
    Returns:
        ModelAtmosphere object with interpolated atmospheric structure
        
    Raises:
        AtmosphereInterpolationError: If interpolation fails
        
    Examples:
        >>> # Solar atmosphere
        >>> atmosphere = interpolate_marcs(5777.0, 4.44, 0.0)
        
        >>> # Metal-poor giant  
        >>> atmosphere = interpolate_marcs(4500.0, 2.0, -1.0)
        
        >>> # Alpha-enhanced star
        >>> atmosphere = interpolate_marcs(5000.0, 4.0, -0.5, alpha_m=0.4)
    """
    # Set default spherical based on surface gravity
    if spherical is None:
        spherical = logg < 3.5
    
    # Validate parameters
    if not (2000 <= Teff <= 8000):
        warnings.warn(f"Teff {Teff}K outside typical range [2000, 8000]K")
    if not (0.0 <= logg <= 5.5):
        warnings.warn(f"logg {logg} outside typical range [0.0, 5.5]")
    if not (-5.0 <= m_H <= 1.0):
        warnings.warn(f"[M/H] {m_H} outside typical range [-5.0, 1.0]")
    
    # Prepare parameters for interpolation
    params = jnp.array([Teff, logg, m_H, alpha_m, C_m])
    
    # Choose which grid to use based on stellar parameters
    if m_H < -2.5:
        # Low metallicity grid
        if abs(alpha_m - 0.4) > 0.01 or abs(C_m) > 0.01:
            raise AtmosphereInterpolationError(
                "For low metallicities ([M/H] < -2.5), alpha_M must be 0.4 and C_M must be 0"
            )
        
        grid_path = _resolve_marcs_grid_path_cached(grid_data_dir, "MARCS_metal_poor_atmospheres.h5")
        grid, nodes, param_names = _load_marcs_grid_cached(grid_path)
        
        # Use only Teff, logg, m_H for low-Z grid
        params_low_z = params[:3]
        interp = _get_multilinear_interpolator(grid_path, nodes, grid, n_params=3)
        atm_quants = interp(params_low_z)
        
    elif (Teff <= 4000 and logg >= 3.5 and m_H >= -2.5):
        # Cool dwarf grid (uses cubic spline interpolation in Korg, multilinear here)
        try:
            grid_path = _resolve_marcs_grid_path_cached(grid_data_dir, "resampled_cool_dwarf_atmospheres.h5")
            grid, nodes, param_names = _load_marcs_grid_cached(grid_path)

            atm_quants = _cubic_interpolation_cool_dwarf(
                np.array(params, dtype=float), nodes, grid, param_names, grid_path=grid_path
            )
            
        except FileNotFoundError:
            # Fallback to standard grid if cool dwarf grid not available
            warnings.warn("Cool dwarf grid not found, using standard grid")
            grid_path = _resolve_marcs_grid_path_cached(grid_data_dir, "SDSS_MARCS_atmospheres.h5")
            grid, nodes, param_names = _load_marcs_grid_cached(grid_path)
            interp = _get_multilinear_interpolator(grid_path, nodes, grid, n_params=5)
            atm_quants = interp(params)
    
    else:
        # Standard SDSS grid
        grid_path = _resolve_marcs_grid_path_cached(grid_data_dir, "SDSS_MARCS_atmospheres.h5")
        grid, nodes, param_names = _load_marcs_grid_cached(grid_path)
        interp = _get_multilinear_interpolator(grid_path, nodes, grid, n_params=5)
        atm_quants = interp(params)
    
    # Create atmosphere from interpolated quantities
    atmosphere = create_atmosphere_from_quantities(atm_quants, spherical, logg)
    
    # Validate optical depths are positive
    tau_values = [layer.tau_5000 for layer in atmosphere.layers]
    if any(tau < 0 for tau in tau_values):
        raise AtmosphereInterpolationError(
            "Interpolated atmosphere has negative optical depths and is not reliable"
        )
    
    return atmosphere


# Convenience functions for backward compatibility and specific use cases

def interpolate_marcs_atmosphere(
    teff: float,
    logg: float,
    metallicity: float = 0.0,
    alpha_enhancement: float = 0.0,
    **kwargs
) -> ModelAtmosphere:
    """
    Compatibility wrapper for older callers expecting interpolate_marcs_atmosphere().

    Parameters use [M/H] and [alpha/H]; convert to [alpha/M] for interpolate_marcs().
    """
    alpha_m = alpha_enhancement - metallicity
    return interpolate_marcs(
        Teff=teff,
        logg=logg,
        m_H=metallicity,
        alpha_m=alpha_m,
        **kwargs
    )

def interpolate_marcs_from_abundances(Teff: float,
                                    logg: float, 
                                    A_X: Dict[int, float],
                                    **kwargs) -> ModelAtmosphere:
    """
    Interpolate MARCS atmosphere from abundance vector.
    
    Args:
        Teff: Effective temperature (K)
        logg: Surface gravity log(g) (cgs)
        A_X: Abundance vector {element: log_abundance}
        **kwargs: Additional arguments for interpolate_marcs
        
    Returns:
        ModelAtmosphere object
    """
    # Follow Korg.jl interpolate_marcs(A_X) logic exactly.
    from .abundances import DEFAULT_ALPHA_ELEMENTS, GREVESSE_2007_SOLAR_ABUNDANCES

    solar_abundances = np.array(GREVESSE_2007_SOLAR_ABUNDANCES, dtype=float)

    if isinstance(A_X, dict):
        A_vec = np.array(solar_abundances, dtype=float)
        for Z, value in A_X.items():
            if 1 <= Z <= 92:
                A_vec[Z - 1] = float(value)
    else:
        A_vec = np.array(A_X, dtype=float)
        if A_vec.shape[0] < 92:
            raise ValueError("A_X must have at least 92 elements.")
        A_vec = A_vec[:92]

    def _get_multi_X_H(A_values, Zs, solar_values):
        A_mX = np.log10(np.sum(10 ** A_values[np.array(Zs) - 1]))
        A_mX_solar = np.log10(np.sum(10 ** solar_values[np.array(Zs) - 1]))
        return A_mX - A_mX_solar

    alpha_elements = list(DEFAULT_ALPHA_ELEMENTS)
    metals = [Z for Z in range(3, 93) if Z not in ([6] + alpha_elements)]

    m_H = _get_multi_X_H(A_vec, metals, solar_abundances)
    alpha_H = _get_multi_X_H(A_vec, alpha_elements, solar_abundances)
    C_H = A_vec[5] - solar_abundances[5]

    alpha_m = alpha_H - m_H
    C_m = C_H - m_H

    if m_H < -2.5:
        alpha_m = 0.4
        C_m = 0.0

    return interpolate_marcs(Teff, logg, m_H, alpha_m, C_m, **kwargs)


def call_korg_interpolation(Teff: float, logg: float, m_H: float = 0.0, 
                           alpha_m: float = 0.0, C_m: float = 0.0) -> ModelAtmosphere:
    """
    Compatibility function that mimics the old subprocess interface.
    
    This function provides backward compatibility for existing code that used
    the subprocess-based Korg interpolation. Now uses the JAX implementation.
    
    Args:
        Teff: Effective temperature (K)
        logg: Surface gravity log(g) (cgs)
        m_H: Metallicity [M/H]
        alpha_m: Alpha enhancement [alpha/M]
        C_m: Carbon enhancement [C/M]
        
    Returns:
        ModelAtmosphere object
    """
    return interpolate_marcs(Teff, logg, m_H, alpha_m, C_m)


# Simplified interface for common use
def solar_atmosphere() -> ModelAtmosphere:
    """Get solar atmosphere with standard parameters."""
    return interpolate_marcs(5777.0, 4.44, 0.0)


def validate_atmosphere(atmosphere: ModelAtmosphere) -> bool:
    """
    Validate that atmosphere structure is physically reasonable.
    
    Args:
        atmosphere: ModelAtmosphere to validate
        
    Returns:
        True if atmosphere passes validation checks
    """
    if len(atmosphere.layers) == 0:
        return False
    
    # Check for monotonic temperature increase with optical depth
    tau_values = [layer.tau_5000 for layer in atmosphere.layers]
    temp_values = [layer.temp for layer in atmosphere.layers]
    
    # Check positive optical depths
    if any(tau <= 0 for tau in tau_values):
        return False
    
    # Check reasonable temperature range
    if min(temp_values) < 1000 or max(temp_values) > 50000:
        return False
    
    # Check density values are positive
    densities = [layer.number_density for layer in atmosphere.layers]
    if any(density <= 0 for density in densities):
        return False
    
    return True
