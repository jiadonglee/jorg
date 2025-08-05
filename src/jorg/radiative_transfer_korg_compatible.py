"""
Korg-compatible radiative transfer implementation
"""

import numpy as np
from typing import Tuple, Union

def radiative_transfer_korg_compatible(
    alpha: np.ndarray,
    source: np.ndarray, 
    spatial_coord: np.ndarray,
    mu_points: int = 20,
    spherical: bool = False,
    include_inward_rays: bool = False,
    tau_scheme: str = "anchored",
    I_scheme: str = "linear_flux_only",
    alpha_ref: np.ndarray = None,
    tau_ref: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Radiative transfer calculation matching Korg.jl's methodology.
    
    Parameters
    ----------
    alpha : np.ndarray
        Opacity matrix [layers × wavelengths]
    source : np.ndarray
        Source function matrix [layers × wavelengths]
    spatial_coord : np.ndarray
        Height coordinate for each layer
    mu_points : int
        Number of μ angle points
    spherical : bool
        Use spherical geometry (not implemented)
    include_inward_rays : bool
        Include inward rays (not implemented)
    tau_scheme : str
        Optical depth scheme ("anchored" or "integrated")
    I_scheme : str
        Intensity scheme ("linear_flux_only")
    alpha_ref : np.ndarray
        Reference opacity at 5000Å
    tau_ref : np.ndarray
        Reference optical depth scale
        
    Returns
    -------
    flux : np.ndarray
        Emergent flux spectrum
    intensity : np.ndarray
        Intensity at each μ angle
    mu_grid : np.ndarray
        μ values
    mu_weights : np.ndarray
        Integration weights
    """
    # Use the exact radiative transfer implementation
    from .radiative_transfer_exact import radiative_transfer
    
    # Generate μ grid
    mu_grid = np.linspace(0.05, 1.0, mu_points)
    mu_weights = np.ones(mu_points) / mu_points
    
    # Call the exact radiative transfer
    flux, intensity, actual_mu_grid, actual_mu_weights = radiative_transfer(
        alpha=alpha,
        source=source,
        spatial_coord=spatial_coord,
        mu_points=mu_points,
        spherical=spherical,
        include_inward_rays=include_inward_rays,
        alpha_ref=alpha_ref,
        tau_ref=tau_ref,
        I_scheme=I_scheme,
        tau_scheme=tau_scheme
    )
    
    # Use the actual grids returned by the RT function
    mu_grid = actual_mu_grid
    mu_weights = actual_mu_weights
    
    return flux, intensity, mu_grid, mu_weights


def generate_mu_grid(mu_values):
    """
    Generate μ grid for radiative transfer.
    
    Parameters
    ----------
    mu_values : int or list
        Number of μ points or explicit μ values
        
    Returns
    -------
    mu_grid : np.ndarray
        μ values
    mu_weights : np.ndarray
        Integration weights
    """
    if isinstance(mu_values, int):
        # Generate Gaussian quadrature points
        mu_grid = np.linspace(0.05, 1.0, mu_values)
        mu_weights = np.ones(mu_values) / mu_values
    else:
        # Use provided values
        mu_grid = np.array(mu_values)
        mu_weights = np.ones(len(mu_values)) / len(mu_values)
    
    return mu_grid, mu_weights