"""
Radiative Transfer Module for Jorg

This module provides a complete implementation of radiative transfer that is
strictly consistent with Korg.jl's RadiativeTransfer module. It includes:

- Formal solution of the radiative transfer equation
- Optical depth calculation with anchored and bezier schemes
- Angular quadrature (μ-grids) with Gauss-Legendre integration
- Source function treatment
- Intensity calculation with multiple schemes
- Support for plane-parallel and spherical atmospheres

All algorithms and numerical methods exactly match Korg.jl implementation.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import expit
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from scipy.special import roots_legendre
from dataclasses import dataclass

# Import exponential integral functions
from jax.scipy.special import exp1


@dataclass
class RadiativeTransferResult:
    """Results from radiative transfer calculation"""
    flux: jnp.ndarray
    intensity: jnp.ndarray  
    mu_grid: jnp.ndarray
    mu_weights: jnp.ndarray


def generate_mu_grid(n_points_or_values: Union[int, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate μ quadrature grid for angular integration
    
    Exactly matches Korg.jl's generate_mu_grid function.
    
    Parameters
    ----------
    n_points_or_values : int or array
        If int: generate Gauss-Legendre grid with n points
        If array: use provided μ values with trapezoidal weights
        
    Returns
    -------
    mu_grid : jnp.ndarray
        μ values (cosine of angle from normal)  
    mu_weights : jnp.ndarray
        Quadrature weights
    """
    if isinstance(n_points_or_values, int):
        # Gauss-Legendre quadrature: μ ∈ [0,1] from standard [-1,1]
        mu_raw, weights_raw = roots_legendre(n_points_or_values)
        mu_grid = jnp.array(mu_raw / 2 + 0.5)  # Transform [-1,1] -> [0,1]
        mu_weights = jnp.array(weights_raw / 2)  # Scale weights for [0,1] interval
        return mu_grid, mu_weights
    else:
        # User-provided μ values
        mu_grid = jnp.array(n_points_or_values)
        
        if len(mu_grid) == 1:
            return mu_grid, jnp.array([1.0])
            
        # Check bounds and sorting
        if not jnp.all(jnp.diff(mu_grid) >= 0) or mu_grid[0] < 0 or mu_grid[-1] > 1:
            raise ValueError("μ_grid must be sorted and bounded between 0 and 1")
            
        # Trapezoidal rule weights
        delta = jnp.diff(mu_grid)
        mu_weights = jnp.concatenate([
            jnp.array([delta[0]]) * 0.5,
            (delta[:-1] + delta[1:]) * 0.5, 
            jnp.array([delta[-1]]) * 0.5
        ])
        
        return mu_grid, mu_weights


def calculate_rays(mu_surface_grid: jnp.ndarray, 
                  spatial_coord: jnp.ndarray, 
                  spherical: bool) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Calculate ray paths through atmosphere
    
    Exactly matches Korg.jl's calculate_rays function.
    
    Parameters
    ----------
    mu_surface_grid : array
        μ values at stellar surface  
    spatial_coord : array
        Physical coordinate (radius for spherical, height for plane-parallel)
    spherical : bool
        Whether atmosphere is spherical
        
    Returns
    -------
    rays : list
        List of (path_distance, ds_dz) tuples for each ray
    """
    rays = []
    
    if spherical:
        # Spherical atmosphere: spatial_coord is radius
        for mu_surface in mu_surface_grid:
            # Impact parameter
            b = spatial_coord[0] * jnp.sqrt(1 - mu_surface**2)
            
            # Find lowest layer ray passes through
            if b < spatial_coord[-1]:
                # Ray goes below atmosphere
                lowest_layer_index = len(spatial_coord)
            else:
                # Find closest approach layer
                lowest_layer_index = jnp.argmin(jnp.abs(spatial_coord - b))
                if spatial_coord[lowest_layer_index] < b:
                    lowest_layer_index -= 1
                    
            # Ray path distances 
            layers_in_ray = spatial_coord[:lowest_layer_index]
            s = jnp.sqrt(layers_in_ray**2 - b**2)
            dsdr = layers_in_ray / s
            
            rays.append((s, dsdr))
    else:
        # Plane-parallel atmosphere: spatial_coord is height
        for mu_surface in mu_surface_grid:
            path = spatial_coord / mu_surface
            dsdz = jnp.ones_like(spatial_coord) / mu_surface
            rays.append((path, dsdz))
            
    return rays


def compute_tau_anchored(alpha: jnp.ndarray,
                        integrand_factor: jnp.ndarray, 
                        log_tau_ref: jnp.ndarray) -> jnp.ndarray:
    """
    Compute optical depth using anchored scheme
    
    Exactly matches Korg.jl's compute_tau_anchored! function.
    
    Parameters
    ----------
    alpha : array
        Absorption coefficient along ray
    integrand_factor : array  
        Integration factors (τ_ref/α_ref * ds/dz)
    log_tau_ref : array
        Log of reference optical depth
        
    Returns
    -------
    tau : array
        Optical depth along ray
    """
    # Integrand for trapezoidal rule
    integrand = alpha * integrand_factor
    
    # Initialize optical depth
    tau = jnp.zeros_like(alpha)
    
    # Trapezoidal integration outward
    for i in range(1, len(log_tau_ref)):
        delta_log_tau = log_tau_ref[i] - log_tau_ref[i-1]
        tau_increment = 0.5 * (integrand[i] + integrand[i-1]) * delta_log_tau
        tau = tau.at[i].set(tau[i-1] + tau_increment)
        
    return tau


def compute_tau_bezier(s: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
    """
    Compute optical depth using Bezier scheme
    
    Matches Korg.jl's compute_tau_bezier! function.
    Based on de la Cruz Rodríguez and Piskunov 2013.
    
    Parameters
    ----------
    s : array
        Path distance along ray
    alpha : array
        Absorption coefficient
        
    Returns
    -------  
    tau : array
        Optical depth along ray
    """
    tau = jnp.zeros_like(s)
    tau = tau.at[0].set(1e-5)  # Small non-zero value at first layer
    
    # Fritsch-Butland control points
    C = fritsch_butland_C(s, alpha)
    
    # Clamp for numerical stability
    C = jnp.clip(C, 0.5 * jnp.min(alpha), 2 * jnp.max(alpha))
    
    # Bezier integration
    for i in range(1, len(alpha)):
        ds = s[i-1] - s[i]
        tau_increment = ds / 3 * (alpha[i] + alpha[i-1] + C[i-1])
        tau = tau.at[i].set(tau[i-1] + tau_increment)
        
    return tau


def fritsch_butland_C(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Bezier control points using Fritsch & Butland method
    
    Exactly matches Korg.jl's fritsch_butland_C function.
    
    Parameters
    ----------
    x : array
        Independent variable (path distance)
    y : array  
        Dependent variable (absorption coefficient)
        
    Returns
    -------
    C : array
        Control points for Bezier interpolation
    """
    h = jnp.diff(x)  # h[k] = x[k+1] - x[k]
    alpha = (1/3) * (1 + h[1:] / (h[1:] + h[:-1]))  # α[k] w.r.t h[k] and h[k-1]
    d = jnp.diff(y) / h  # d[k] is slope at midpoint
    
    # Derivative at interior points
    yprime = (d[:-1] * d[1:]) / (alpha * d[1:] + (1 - alpha) * d[:-1])
    
    # Control points
    C0 = y[1:-1] + h[:-1] * yprime / 2
    C1 = y[1:-1] - h[1:] * yprime / 2
    
    # Average control points at boundaries
    C_full = (jnp.concatenate([C0, jnp.array([C1[-1]])]) + 
              jnp.concatenate([jnp.array([C0[0]]), C1])) / 2
              
    return C_full


def compute_I_linear(tau: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
    """
    Compute intensity using linear interpolation scheme
    
    Exactly matches Korg.jl's compute_I_linear! function.
    
    Solves: I = ∫ S(τ) exp(-τ) dτ with linear interpolation of S.
    
    Parameters
    ----------
    tau : array
        Optical depth along ray
    S : array
        Source function along ray
        
    Returns
    -------
    I : array
        Intensity at each layer
    """
    I = jnp.zeros_like(tau)
    
    if len(tau) <= 1:
        return I
    
    # Backward integration from deep layers
    for k in range(len(tau)-2, -1, -1):
        delta_tau = tau[k+1] - tau[k]
        
        if delta_tau > 0:
            # Linear slope of source function
            m = (S[k+1] - S[k]) / delta_tau
            
            # Exact integration: ∫(mτ + b)exp(-τ)dτ = -exp(-τ)(mτ + b + m)
            exp_neg_delta = jnp.exp(-delta_tau)
            I_new = (I[k+1] - S[k] - m * (delta_tau + 1)) * exp_neg_delta + m + S[k]
            I = I.at[k].set(I_new)
        else:
            I = I.at[k].set(I[k+1])
            
    return I


def compute_I_linear_flux_only(tau: jnp.ndarray, S: jnp.ndarray) -> float:
    """
    Compute surface intensity using linear scheme (flux-only optimization)
    
    Exactly matches Korg.jl's compute_I_linear_flux_only function.
    
    Parameters
    ----------
    tau : array
        Optical depth along ray
    S : array  
        Source function along ray
        
    Returns
    -------
    I_surface : float
        Intensity at stellar surface
    """
    if len(tau) <= 1:
        return 0.0
        
    I = 0.0
    next_exp_neg_tau = jnp.exp(-tau[0])
    
    for i in range(len(tau)-1):
        delta_tau = tau[i+1] - tau[i]
        
        # Handle numerical issues with very small delta_tau
        delta_tau = jnp.where(delta_tau == 0, 1.0, delta_tau)
        
        # Linear slope
        m = (S[i+1] - S[i]) / delta_tau
        
        # Current and next exponential terms
        cur_exp_neg_tau = next_exp_neg_tau
        next_exp_neg_tau = jnp.exp(-tau[i+1])
        
        # Add contribution to integral
        contribution = (-next_exp_neg_tau * (S[i+1] + m) + 
                       cur_exp_neg_tau * (S[i] + m))
        I += contribution
        
    return I


def compute_I_bezier(tau: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
    """
    Compute intensity using Bezier interpolation scheme
    
    Matches Korg.jl's compute_I_bezier! function.
    Based on de la Cruz Rodríguez and Piskunov 2013.
    
    Parameters
    ----------
    tau : array
        Optical depth along ray
    S : array
        Source function along ray
        
    Returns
    -------
    I : array
        Intensity at each layer
    """
    I = jnp.zeros_like(tau)
    I = I.at[-1].set(0.0)  # Boundary condition
    
    if len(tau) <= 1:
        return I
        
    # Bezier control points for source function
    C = fritsch_butland_C(tau, S)
    
    # Backward integration
    for k in range(len(tau)-2, -1, -1):
        delta_tau = tau[k+1] - tau[k]
        
        if delta_tau > 0:
            # Bezier coefficients from de la Cruz Rodríguez & Piskunov 2013
            exp_neg_delta = jnp.exp(-delta_tau)
            delta_sq = delta_tau**2
            
            alpha = (2 + delta_sq - 2*delta_tau - 2*exp_neg_delta) / delta_sq
            beta = (2 - (2 + 2*delta_tau + delta_sq)*exp_neg_delta) / delta_sq  
            gamma = (2*delta_tau - 4 + (2*delta_tau + 4)*exp_neg_delta) / delta_sq
            
            I_new = (I[k+1] * exp_neg_delta + 
                    alpha * S[k] + beta * S[k+1] + gamma * C[k])
            I = I.at[k].set(I_new)
        else:
            I = I.at[k].set(I[k+1])
    
    # Apply surface correction if τ[0] ≠ 0
    I = I.at[0].multiply(jnp.exp(-tau[0]))
    
    return I


def exponential_integral_2(x: jnp.ndarray) -> jnp.ndarray:
    """
    Second-order exponential integral E₂(x)
    
    Exactly matches Korg.jl's exponential_integral_2 function.
    Uses piecewise polynomial approximations for accuracy.
    
    Parameters
    ----------
    x : array
        Input values
        
    Returns
    -------
    E2 : array
        E₂(x) values
    """
    def _expint_small(x):
        # Euler-Mascheroni constant
        gamma = 0.57721566490153286060651209008240243104215933593992
        return (1 + 
                ((jnp.log(x) + gamma - 1) +
                 (-0.5 + (0.08333333333333333 + 
                         (-0.013888888888888888 + 0.0020833333333333333 * x) * x) * x) * x) * x)
    
    def _expint_large(x):
        invx = 1 / x
        return (jnp.exp(-x) * 
                (1 + (-2 + (6 + (-24 + 120 * invx) * invx) * invx) * invx) * invx)
    
    def _expint_2(x):  # x ∈ [1.1, 2.5]
        x = x - 2
        return (0.037534261820486914 +
                (-0.04890051070806112 +
                 (0.033833820809153176 +
                  (-0.016916910404576574 +
                   (0.007048712668573576 - 0.0026785108140579598 * x) * x) * x) * x) * x)
    
    # Apply appropriate approximation based on x value
    result = jnp.where(x == 0, 1.0,
             jnp.where(x < 1.1, _expint_small(x),
             jnp.where(x < 2.5, _expint_2(x), _expint_large(x))))
    
    return result


def compute_F_flux_only_expint(tau: jnp.ndarray, S: jnp.ndarray) -> float:
    """
    Compute astrophysical flux using exponential integral method
    
    Exactly matches Korg.jl's compute_F_flux_only_expint function.
    
    Parameters
    ----------
    tau : array
        Optical depth along ray
    S : array
        Source function along ray
        
    Returns
    -------
    F : float
        Astrophysical flux
    """
    I = 0.0
    
    for i in range(len(tau)-1):
        delta_tau = tau[i+1] - tau[i]
        if delta_tau > 0:
            m = (S[i+1] - S[i]) / delta_tau
            b = S[i] - m * tau[i]
            
            # Exact solution to ∫(mτ + b)E₂(τ)dτ
            contribution = (expint_transfer_integral_core(tau[i+1], m, b) -
                          expint_transfer_integral_core(tau[i], m, b))
            I += contribution
            
    return I


def expint_transfer_integral_core(tau: float, m: float, b: float) -> float:
    """
    Core exponential integral for transfer equation
    
    Exactly matches Korg.jl's expint_transfer_integral_core function.
    
    Evaluates: ∫(mτ + b)E₂(τ)dτ
    
    Parameters
    ----------
    tau : float
        Optical depth
    m : float
        Linear slope coefficient  
    b : float
        Linear intercept coefficient
        
    Returns
    -------
    integral : float
        Integral value
    """
    E2_tau = exponential_integral_2(tau)
    exp_neg_tau = jnp.exp(-tau)
    
    return (1/6 * (tau * E2_tau * (3*b + 2*m*tau) - 
                   exp_neg_tau * (3*b + 2*m*(tau + 1))))


def radiative_transfer_core(mu_ind: int,
                           layer_inds: List[int],
                           n_inward_rays: int, 
                           path: jnp.ndarray,
                           dsdz: jnp.ndarray,
                           tau_buffer: jnp.ndarray,
                           integrand_buffer: jnp.ndarray, 
                           log_tau_ref: jnp.ndarray,
                           alpha: jnp.ndarray,
                           S: jnp.ndarray,
                           I: jnp.ndarray,
                           tau_ref: jnp.ndarray,
                           alpha_ref: jnp.ndarray,
                           tau_scheme: str,
                           I_scheme: str) -> jnp.ndarray:
    """
    Core radiative transfer calculation for single ray
    
    Exactly matches Korg.jl's radiative_transfer_core function.
    
    Parameters
    ----------
    mu_ind : int
        Ray index
    layer_inds : array
        Layer indices along ray
    n_inward_rays : int  
        Number of inward rays
    path : array
        Path distances
    dsdz : array
        Path derivatives
    tau_buffer : array
        Optical depth workspace
    integrand_buffer : array
        Integration workspace
    log_tau_ref : array
        Log reference optical depth
    alpha : array
        Absorption matrix
    S : array
        Source function matrix
    I : array
        Intensity array (modified in place)
    tau_ref : array
        Reference optical depth
    alpha_ref : array
        Reference absorption
    tau_scheme : str
        Optical depth scheme
    I_scheme : str
        Intensity scheme
        
    Returns
    -------
    I : array
        Updated intensity array
    """
    if len(path) == 1 and (I_scheme == "bezier" or tau_scheme == "bezier"):
        # Bezier schemes require minimum 2 layers
        if I_scheme.startswith("linear_flux_only"):
            I = I.at[mu_ind, :].set(0.0)
        else:
            I = I.at[mu_ind, :, 0].set(0.0)
        return I
    
    # Integration factor for this ray
    layer_inds_array = jnp.array(layer_inds)
    integrand_factor = tau_ref[layer_inds_array] / alpha_ref[layer_inds_array] * dsdz
    
    # Process each wavelength
    for lam_ind in range(alpha.shape[1]):
        alpha_ray = alpha[layer_inds_array, lam_ind]
        S_ray = S[layer_inds_array, lam_ind]
        
        # Compute optical depth
        if tau_scheme == "anchored":
            tau = compute_tau_anchored(alpha_ray, integrand_factor, log_tau_ref[layer_inds_array])
        elif tau_scheme == "bezier":
            tau = compute_tau_bezier(path, alpha_ray)
        else:
            raise ValueError("tau_scheme must be 'anchored' or 'bezier'")
        
        # Compute intensity
        if I_scheme == "linear":
            I_ray = compute_I_linear(tau, S_ray)
            I = I.at[mu_ind, lam_ind, layer_inds_array].set(I_ray)
        elif I_scheme == "linear_flux_only":
            I_surface = compute_I_linear_flux_only(tau, S_ray)
            I = I.at[mu_ind, lam_ind].add(I_surface)
        elif I_scheme == "linear_flux_only_expint":
            F_surface = compute_F_flux_only_expint(tau, S_ray)
            I = I.at[mu_ind, lam_ind].add(F_surface)
        elif I_scheme == "bezier":
            I_ray = compute_I_bezier(tau, S_ray)
            I = I.at[mu_ind, lam_ind, layer_inds_array].set(I_ray)
        else:
            raise ValueError("I_scheme must be 'linear', 'bezier', or 'linear_flux_only'")
        
        # Set boundary condition for outward ray if this is inward ray
        if mu_ind < n_inward_rays:
            if I_scheme.startswith("linear_flux_only"):
                # Intensity at bottom attenuated by optical depth
                I = I.at[mu_ind + n_inward_rays, lam_ind].set(
                    I[mu_ind, lam_ind] * jnp.exp(-tau[-1]))
            else:
                I = I.at[mu_ind + n_inward_rays, lam_ind, len(path)-1].set(
                    I[mu_ind, lam_ind, len(path)-1])
    
    return I


def radiative_transfer(alpha: jnp.ndarray,
                      S: jnp.ndarray, 
                      spatial_coord: jnp.ndarray,
                      mu_points: Union[int, jnp.ndarray],
                      spherical: bool = False,
                      include_inward_rays: bool = False,
                      alpha_ref: Optional[jnp.ndarray] = None,
                      tau_ref: Optional[jnp.ndarray] = None,
                      tau_scheme: str = "anchored",
                      I_scheme: str = "linear_flux_only") -> RadiativeTransferResult:
    """
    Solve radiative transfer equation
    
    Exactly matches Korg.jl's radiative_transfer function interface and algorithms.
    
    Parameters
    ----------
    alpha : array, shape (n_layers, n_wavelengths)
        Absorption coefficient matrix [cm⁻¹]
    S : array, shape (n_layers, n_wavelengths)  
        Source function matrix [erg cm⁻² s⁻¹ sr⁻¹ Hz⁻¹]
    spatial_coord : array
        Physical coordinate (radius or height) [cm]
    mu_points : int or array
        Number of μ points or specific μ values
    spherical : bool, default False
        Whether atmosphere is spherical
    include_inward_rays : bool, default False
        Include all inward rays (not just boundary conditions)
    alpha_ref : array, optional
        Reference absorption for anchored scheme
    tau_ref : array, optional  
        Reference optical depth for anchored scheme
    tau_scheme : str, default "anchored"
        Optical depth calculation scheme
    I_scheme : str, default "linear_flux_only"
        Intensity calculation scheme
        
    Returns
    -------
    result : RadiativeTransferResult
        Complete radiative transfer solution
    """
    n_layers, n_wavelengths = alpha.shape
    
    # Set up μ quadrature
    if I_scheme == "linear_flux_only_expint" and tau_scheme == "anchored" and not spherical:
        # Special case: use exponential integral optimization
        mu_surface_grid, mu_weights = jnp.array([1.0]), jnp.array([1.0])
    else:
        mu_surface_grid, mu_weights = generate_mu_grid(mu_points)
    
    # Calculate ray paths
    rays = calculate_rays(mu_surface_grid, spatial_coord, spherical)
    
    # Determine inward rays needed
    if include_inward_rays:
        inward_mu_surface_grid = -mu_surface_grid
    else:
        # Only rays needed for boundary conditions
        ray_lengths = [len(ray[0]) for ray in rays]
        incomplete_rays = [i for i, length in enumerate(ray_lengths) 
                          if length < len(spatial_coord)]
        inward_mu_surface_grid = -mu_surface_grid[jnp.array(incomplete_rays)] if incomplete_rays else jnp.array([])
    
    n_inward_rays = len(inward_mu_surface_grid)
    n_total_rays = n_inward_rays + len(mu_surface_grid)
    
    # Initialize intensity array
    if I_scheme.startswith("linear_flux_only"):
        I = jnp.zeros((n_total_rays, n_wavelengths))
    else:
        I = jnp.zeros((n_total_rays, n_wavelengths, n_layers))
    
    # Set up reference arrays for anchored scheme
    if tau_scheme == "anchored":
        if tau_ref is None or alpha_ref is None:
            raise ValueError("tau_ref and alpha_ref required for anchored scheme")
        log_tau_ref = jnp.log(tau_ref)
    else:
        log_tau_ref = jnp.zeros(n_layers)
        tau_ref = jnp.ones(n_layers)
        alpha_ref = jnp.ones(n_layers)
    
    # Workspace arrays
    tau_buffer = jnp.zeros(n_layers)
    integrand_buffer = jnp.zeros(n_layers)
    
    # Process inward rays
    for mu_ind in range(n_inward_rays):
        ray_ind = jnp.where(mu_surface_grid == -inward_mu_surface_grid[mu_ind])[0][0]
        path, dsdz = rays[ray_ind]
        
        # Reverse for inward integration
        path = path[::-1]
        dsdz = dsdz[::-1] 
        layer_inds = list(range(len(path)-1, -1, -1))
        
        I = radiative_transfer_core(
            mu_ind, layer_inds, n_inward_rays, -path, dsdz,
            tau_buffer, integrand_buffer, -log_tau_ref, alpha, S, I,
            tau_ref, alpha_ref, tau_scheme, I_scheme)
    
    # Process outward rays  
    for mu_ind in range(n_inward_rays, n_total_rays):
        ray_ind = mu_ind - n_inward_rays
        path, dsdz = rays[ray_ind]
        layer_inds = list(range(len(path)))
        
        I = radiative_transfer_core(
            mu_ind, layer_inds, n_inward_rays, path, dsdz,
            tau_buffer, integrand_buffer, log_tau_ref, alpha, S, I, 
            tau_ref, alpha_ref, tau_scheme, I_scheme)
    
    # Calculate emergent flux from surface intensities
    if I_scheme.startswith("linear_flux_only"):
        surface_I = I[n_inward_rays:, :]  # Only outward rays
    else:
        surface_I = I[n_inward_rays:, :, 0]  # Surface layer, outward rays
    
    # Integrate over μ: F = 2π ∫ I(μ) μ dμ
    flux = 2 * jnp.pi * jnp.sum(
        surface_I * (mu_weights[:, None] * mu_surface_grid[:, None]), axis=0)
    
    return RadiativeTransferResult(
        flux=flux,
        intensity=I, 
        mu_grid=mu_surface_grid,
        mu_weights=mu_weights
    )