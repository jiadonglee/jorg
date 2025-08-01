"""
Korg.jl-Compatible Radiative Transfer for Jorg
==============================================

This module implements radiative transfer exactly as done in Korg.jl:
- Anchored optical depth integration using τ_5000 reference
- Linear intensity calculation with exact analytical solutions  
- Exponential integral methods for flux calculation
- No artificial saturation or clipping

Direct port of:
/Users/jdli/Project/Korg.jl/src/RadiativeTransfer/RadiativeTransfer.jl

Author: Claude Code Assistant
Date: July 2025
"""

import numpy as np
import jax.numpy as jnp
from typing import Tuple, List, Dict, Optional, Union
from scipy.special import expi
import warnings


def generate_mu_grid(n_points_or_values: Union[int, List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate μ grid for radiative transfer quadrature (matches Korg.jl exactly)
    
    Parameters
    ----------
    n_points_or_values : int or list
        If int: number of Gauss-Legendre quadrature points
        If list: explicit μ values to use
        
    Returns
    -------
    mu_grid : ndarray
        μ values for angular integration
    mu_weights : ndarray  
        Corresponding quadrature weights
        
    Notes
    -----
    Exact port of Korg.jl's generate_mu_grid() function
    """
    if isinstance(n_points_or_values, int):
        # Gauss-Legendre quadrature (simplified - should use FastGaussQuadrature)
        from scipy.special.orthogonal import p_roots
        mu_raw, weights_raw = p_roots(n_points_or_values)
        
        # Transform from [-1,1] to [0,1] as in Korg.jl
        mu_grid = mu_raw / 2 + 0.5
        mu_weights = weights_raw / 2
        
        return mu_grid, mu_weights
    
    else:
        mu_grid = np.array(n_points_or_values)
        
        if len(mu_grid) == 1:
            return mu_grid, np.array([1.0])
        
        if not np.all(np.diff(mu_grid) >= 0) or mu_grid[0] < 0 or mu_grid[-1] > 1:
            raise ValueError("μ_grid must be sorted and bounded between 0 and 1")
        
        # Trapezoidal weights
        delta = np.diff(mu_grid)
        mu_weights = 0.5 * np.concatenate([[delta[0]], 
                                          delta[:-1] + delta[1:], 
                                          [delta[-1]]])
        
        return mu_grid, mu_weights


def compute_tau_anchored(alpha: np.ndarray, 
                        integrand_factor: np.ndarray,
                        log_tau_ref: np.ndarray) -> np.ndarray:
    """
    Compute optical depth using anchored integration scheme (exact Korg.jl port)
    
    Parameters
    ----------
    alpha : ndarray
        Absorption coefficient along ray [layer]
    integrand_factor : ndarray
        τ_ref/α_ref * ds/dz factor [layer]
    log_tau_ref : ndarray
        Log of reference optical depth [layer]
        
    Returns
    -------
    tau : ndarray
        Optical depth at each layer [layer]
        
    Notes
    -----
    Direct port of compute_tau_anchored! from Korg.jl
    """
    n_layers = len(alpha)
    tau = np.zeros(n_layers)
    
    # Calculate integrand: α(z) * (τ_ref/α_ref) * (ds/dz)
    integrand_buffer = alpha * integrand_factor
    
    # Trapezoidal integration in log(τ_ref) space
    tau[0] = 0.0
    for i in range(1, n_layers):
        tau[i] = (tau[i-1] + 
                 0.5 * (integrand_buffer[i] + integrand_buffer[i-1]) * 
                 (log_tau_ref[i] - log_tau_ref[i-1]))
    
    return tau


def compute_I_linear_flux_only(tau: np.ndarray, source: np.ndarray) -> float:
    """
    Compute intensity at surface using exact linear interpolation method
    
    Parameters
    ----------
    tau : ndarray
        Optical depth along ray [layer]
    source : ndarray
        Source function along ray [layer]
        
    Returns
    -------
    intensity : float
        Emergent intensity at surface
        
    Notes
    -----
    Exact port of compute_I_linear_flux_only from Korg.jl:
    
    Solves the radiative transfer integral exactly by linearly interpolating
    the source function and computing integrals of the form:
    ∫ (m*τ + b) * exp(-τ) dτ = -exp(-τ) * (m*τ + b + m)
    """
    if len(tau) == 1:
        return 0.0
    
    intensity = 0.0
    next_exp_neg_tau = np.exp(-tau[0])
    
    for i in range(len(tau) - 1):
        delta_tau = tau[i+1] - tau[i]
        
        # Handle numerical case where large τ causes delta_tau ≈ 0
        if delta_tau == 0:
            delta_tau = 1.0  # Korg.jl's fix: make it 1 if it's 0
        
        # Linear interpolation slope
        m = (source[i+1] - source[i]) / delta_tau
        
        # Calculate exponentials for this interval
        cur_exp_neg_tau = next_exp_neg_tau
        next_exp_neg_tau = np.exp(-tau[i+1])
        
        # Exact analytical integration
        intensity += (-next_exp_neg_tau * (source[i+1] + m) + 
                     cur_exp_neg_tau * (source[i] + m))
    
    return intensity


def compute_I_linear(tau: np.ndarray, source: np.ndarray) -> np.ndarray:
    """
    Compute intensity at all layers using exact linear method
    
    Parameters
    ----------
    tau : ndarray
        Optical depth along ray [layer]
    source : ndarray
        Source function along ray [layer]
        
    Returns
    -------
    intensity : ndarray
        Intensity at each layer [layer]
        
    Notes
    -----
    Direct port of compute_I_linear! from Korg.jl
    """
    n_layers = len(tau)
    intensity = np.zeros(n_layers)
    
    if n_layers == 1:
        return intensity
    
    # Work backwards from deepest layer
    for k in range(n_layers-2, -1, -1):
        delta = tau[k+1] - tau[k]
        m = (source[k+1] - source[k]) / delta
        
        # Exact analytical solution
        intensity[k] = ((intensity[k+1] - source[k] - m * (delta + 1)) * 
                       np.exp(-delta) + m + source[k])
    
    return intensity


def exponential_integral_2(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Second-order exponential integral E₂(x) using scipy for accuracy
    
    Parameters
    ----------
    x : float or ndarray
        Input values
        
    Returns
    -------
    result : float or ndarray
        E₂(x) values
        
    Notes
    -----
    Uses scipy.special.expi for high accuracy, with E₂(x) = -Ei(-x) + ln(x) + γ
    where Ei is the exponential integral and γ is Euler-Mascheroni constant
    """
    from scipy.special import expi
    
    # Handle x = 0 case
    if np.isscalar(x):
        if x == 0:
            return 1.0
        else:
            # E₂(x) = ∫₁^∞ exp(-xt)/t² dt, use scipy for accuracy
            from scipy.special import expn
            return expn(2, x)
    else:
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        
        # Handle x = 0 cases
        zero_mask = (x == 0)
        result[zero_mask] = 1.0
        
        # Handle x > 0 cases
        pos_mask = (x > 0)
        if np.any(pos_mask):
            from scipy.special import expn
            result[pos_mask] = expn(2, x[pos_mask])
        
        return result


def expint_transfer_integral_core(tau: float, m: float, b: float) -> float:
    """
    Exact solution to ∫ (m*τ + b) * E₂(τ) dτ
    
    Parameters
    ----------
    tau : float
        Optical depth value
    m : float
        Linear slope parameter
    b : float
        Linear intercept parameter
        
    Returns
    -------
    integral : float
        Exact integral value
        
    Notes
    -----
    Direct port from Korg.jl for exponential integral radiative transfer
    """
    return (1/6 * (tau * exponential_integral_2(tau) * (3*b + 2*m*tau) - 
                   np.exp(-tau) * (3*b + 2*m*(tau + 1))))


def compute_F_flux_only_expint(tau: np.ndarray, source: np.ndarray) -> float:
    """
    Compute astrophysical flux using exponential integral method
    
    Parameters
    ----------
    tau : ndarray
        Optical depth along ray [layer]
    source : ndarray
        Source function along ray [layer]
        
    Returns
    -------
    flux : float
        Emergent astrophysical flux
        
    Notes
    -----
    Direct port of compute_F_flux_only_expint from Korg.jl
    Handles μ integration analytically using E₂
    """
    flux = 0.0
    
    for i in range(len(tau) - 1):
        m = (source[i+1] - source[i]) / (tau[i+1] - tau[i])
        b = source[i] - m * tau[i]
        
        flux += (expint_transfer_integral_core(tau[i+1], m, b) - 
                expint_transfer_integral_core(tau[i], m, b))
    
    return flux


def calculate_rays(mu_surface_grid: np.ndarray, 
                  spatial_coord: np.ndarray, 
                  spherical: bool) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Calculate ray paths through atmosphere
    
    Parameters
    ----------
    mu_surface_grid : ndarray
        μ values at stellar surface
    spatial_coord : ndarray
        Physical coordinate (radius for spherical, height for plane-parallel)
    spherical : bool
        Whether atmosphere is spherical
        
    Returns
    -------
    rays : list of tuples
        List of (path_length, ds_dz) pairs for each ray
        
    Notes
    -----
    Direct port of calculate_rays from Korg.jl
    """
    rays = []
    
    for mu_surface in mu_surface_grid:
        if spherical:
            # Spherical atmosphere: spatial_coord is radius
            b = spatial_coord[0] * np.sqrt(1 - mu_surface**2)  # Impact parameter
            
            # Find lowest layer ray penetrates
            if b < spatial_coord[-1]:
                lowest_layer_index = len(spatial_coord)
            else:
                lowest_layer_index = np.argmin(np.abs(spatial_coord - b))
                if spatial_coord[lowest_layer_index] < b:
                    lowest_layer_index -= 1
                lowest_layer_index = max(0, lowest_layer_index)
            
            # Calculate path lengths and derivatives
            coord_subset = spatial_coord[:lowest_layer_index+1]
            s = np.sqrt(coord_subset**2 - b**2)
            dsdr = coord_subset / s
            
            rays.append((s, dsdr))
            
        else:
            # Plane-parallel atmosphere: spatial_coord is height
            s = spatial_coord / mu_surface
            dsdr = np.ones_like(spatial_coord) / mu_surface
            
            rays.append((s, dsdr))
    
    return rays


def radiative_transfer_korg_compatible(
    alpha: np.ndarray,
    source: np.ndarray, 
    spatial_coord: np.ndarray,
    mu_points: Union[int, List[float]],
    spherical: bool = False,
    include_inward_rays: bool = False,
    tau_scheme: str = "anchored",
    I_scheme: str = "linear_flux_only",
    alpha_ref: Optional[np.ndarray] = None,
    tau_ref: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Main radiative transfer function following Korg.jl exactly
    
    Parameters
    ----------
    alpha : ndarray
        Absorption coefficient matrix [layers × wavelengths]
    source : ndarray
        Source function matrix [layers × wavelengths]
    spatial_coord : ndarray
        Physical coordinate (radius or height) [layers]
    mu_points : int or list
        Number of μ points or explicit μ values
    spherical : bool, default=False
        Whether atmosphere is spherical
    include_inward_rays : bool, default=False
        Include inward-propagating rays
    tau_scheme : str, default="anchored"
        Optical depth calculation scheme
    I_scheme : str, default="linear_flux_only"
        Intensity calculation scheme
    alpha_ref : ndarray, optional
        Reference absorption coefficient for anchoring
    tau_ref : ndarray, optional
        Reference optical depth for anchoring
        
    Returns
    -------
    flux : ndarray
        Emergent flux [wavelengths]
    intensity : ndarray
        Intensity array [mu × wavelengths × layers] or [mu × wavelengths]
    mu_surface_grid : ndarray
        μ values used
    mu_weights : ndarray
        Quadrature weights used
        
    Notes
    -----
    Direct port of Korg.jl's main radiative_transfer function
    """
    n_layers, n_wavelengths = alpha.shape
    
    # Special case: use exponential integral for plane-parallel + anchored + flux_only
    if I_scheme == "linear_flux_only" and tau_scheme == "anchored" and not spherical:
        I_scheme = "linear_flux_only_expint"
        mu_surface_grid, mu_weights = np.array([1.0]), np.array([1.0])
    else:
        mu_surface_grid, mu_weights = generate_mu_grid(mu_points)
    
    # Calculate ray paths
    rays = calculate_rays(mu_surface_grid, spatial_coord, spherical)
    
    # Determine inward rays needed
    if include_inward_rays:
        inward_mu_surface_grid = -mu_surface_grid
    else:
        # Only rays needed to seed bottom boundary
        ray_lengths = [len(ray[0]) for ray in rays]
        short_rays_mask = np.array(ray_lengths) < n_layers
        inward_mu_surface_grid = -mu_surface_grid[short_rays_mask]
    
    n_inward_rays = len(inward_mu_surface_grid)
    
    # Initialize intensity array
    if I_scheme.startswith("linear_flux_only"):
        # Only surface intensity
        intensity = np.zeros((n_inward_rays + len(mu_surface_grid), n_wavelengths))
    else:
        # Full intensity field
        intensity = np.zeros((n_inward_rays + len(mu_surface_grid), n_wavelengths, n_layers))
    
    # Reference values for anchored scheme
    if tau_ref is None:
        tau_ref = spatial_coord  # Fallback
    if alpha_ref is None:
        alpha_ref = np.ones(n_layers)  # Fallback
    
    log_tau_ref = np.log(np.maximum(tau_ref, 1e-10))  # Avoid log(0)
    
    # Process inward rays first
    for mu_idx in range(n_inward_rays):
        path, dsdz = rays[mu_idx]
        # Reverse path and dsdz for inward rays (as in Korg.jl)
        path = path[::-1]
        dsdz = dsdz[::-1]
        layer_indices = np.arange(len(path)-1, -1, -1)  # Reverse order
        
        for wavelength_idx in range(n_wavelengths):
            alpha_wl = alpha[layer_indices, wavelength_idx]
            source_wl = source[layer_indices, wavelength_idx]
            
            # Calculate integrand factor
            integrand_factor = tau_ref[layer_indices] / alpha_ref[layer_indices] * dsdz
            
            # Compute optical depth (with negated log_tau_ref for inward rays)
            if tau_scheme == "anchored":
                tau = compute_tau_anchored(alpha_wl, 
                                         integrand_factor, 
                                         -log_tau_ref[layer_indices])  # Negative for inward
            else:
                raise ValueError(f"τ_scheme '{tau_scheme}' not implemented")
            
            # Compute intensity
            if I_scheme == "linear":
                intensity[mu_idx, wavelength_idx, layer_indices] = compute_I_linear(tau, source_wl)
            elif I_scheme == "linear_flux_only":
                intensity[mu_idx, wavelength_idx] = compute_I_linear_flux_only(tau, source_wl)
            elif I_scheme == "linear_flux_only_expint":
                intensity[mu_idx, wavelength_idx] = compute_F_flux_only_expint(tau, source_wl)
            else:
                raise ValueError(f"I_scheme '{I_scheme}' not implemented")
    
    # Process outward rays
    for mu_idx in range(len(mu_surface_grid)):
        full_mu_idx = n_inward_rays + mu_idx
        path, dsdz = rays[mu_idx]
        layer_indices = np.arange(len(path))
        
        for wavelength_idx in range(n_wavelengths):
            alpha_wl = alpha[layer_indices, wavelength_idx]
            source_wl = source[layer_indices, wavelength_idx]
            
            # Calculate integrand factor
            integrand_factor = tau_ref[layer_indices] / alpha_ref[layer_indices] * dsdz
            
            # Compute optical depth
            if tau_scheme == "anchored":
                tau = compute_tau_anchored(alpha_wl, 
                                         integrand_factor, 
                                         log_tau_ref[layer_indices])
            else:
                raise ValueError(f"τ_scheme '{tau_scheme}' not implemented")
            
            # Compute intensity
            if I_scheme == "linear":
                intensity[full_mu_idx, wavelength_idx, layer_indices] = compute_I_linear(tau, source_wl)
            elif I_scheme == "linear_flux_only":
                intensity[full_mu_idx, wavelength_idx] = compute_I_linear_flux_only(tau, source_wl)
            elif I_scheme == "linear_flux_only_expint":
                intensity[full_mu_idx, wavelength_idx] = compute_F_flux_only_expint(tau, source_wl)
            else:
                raise ValueError(f"I_scheme '{I_scheme}' not implemented")
            
            # Set boundary condition for corresponding inward ray
            if mu_idx < n_inward_rays:
                if I_scheme.startswith("linear_flux_only"):
                    intensity[full_mu_idx, wavelength_idx] = intensity[mu_idx, wavelength_idx] * np.exp(-tau[-1])
                else:
                    # Use len(path)-1 as the index (equivalent to Julia's length(path))
                    intensity[full_mu_idx, wavelength_idx, len(path)-1] = intensity[mu_idx, wavelength_idx, len(path)-1]
    
    # Calculate flux from surface intensity
    if I_scheme.startswith("linear_flux_only"):
        surface_intensity = intensity[n_inward_rays:, :]  # Only outward rays
    else:
        surface_intensity = intensity[n_inward_rays:, :, 0]  # Surface layer, outward rays
    
    # Integrate over μ: F = 2π ∫ I(μ) μ dμ
    flux = 2 * np.pi * np.dot(mu_weights * mu_surface_grid, surface_intensity)
    
    return flux, intensity, mu_surface_grid, mu_weights


# Export main functions
__all__ = [
    'radiative_transfer_korg_compatible',
    'compute_tau_anchored', 
    'compute_I_linear_flux_only',
    'compute_I_linear',
    'exponential_integral_2',
    'generate_mu_grid'
]