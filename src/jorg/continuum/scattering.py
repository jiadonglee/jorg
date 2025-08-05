"""
Scattering opacity implementations in JAX

This module implements Thomson scattering by free electrons
and Rayleigh scattering by neutral atoms and molecules.
"""

import jax
import jax.numpy as jnp

from ..constants import c_cgs, sigma_thomson


@jax.jit
def thomson_scattering(electron_density: float) -> float:
    """
    Calculate Thomson scattering opacity by free electrons
    
    Thomson scattering is frequency-independent for X-ray to optical wavelengths.
    
    Parameters
    ----------
    electron_density : float
        Electron density in cm^-3
        
    Returns
    -------
    float
        Thomson scattering opacity in cm^-1
    """
    return electron_density * sigma_thomson


@jax.jit
def rayleigh_scattering(
    frequencies: jnp.ndarray,
    n_h_i: float,
    n_he_i: float,
    n_h2: float
) -> jnp.ndarray:
    """
    Calculate Rayleigh scattering opacity by neutral atoms and molecules
    
    Implements proper Rayleigh scattering formulations from Korg.jl:
    - H I and He I: Colgan+ 2016 formulation
    - H2: Dalgarno & Williams 1962 formulation
    
    Direct port from Korg.jl/src/ContinuumAbsorption/scattering.jl
    
    Parameters
    ----------
    frequencies : jnp.ndarray
        Frequencies in Hz
    n_h_i : float
        H I number density in cm^-3
    n_he_i : float
        He I number density in cm^-3
    n_h2 : float
        H2 number density in cm^-3
        
    Returns
    -------
    jnp.ndarray
        Rayleigh scattering opacity in cm^-1
    """
    from ..constants import hplanck_eV, Rydberg_eV
    
    # Thompson scattering cross section (from Korg.jl)
    sigma_thomson = 6.65246e-25  # cm^2
    
    # Colgan+ 2016 formulation for H I and He I
    # (ħω/2E_H)^2 = (photon energy / 2 Rydberg)^2
    E_2Ryd_2 = (hplanck_eV * frequencies / (2 * Rydberg_eV))**2
    E_2Ryd_4 = E_2Ryd_2**2
    E_2Ryd_6 = E_2Ryd_2 * E_2Ryd_4
    E_2Ryd_8 = E_2Ryd_4**2
    
    # Colgan+ 2016 equation 6 for H I
    sigma_H_over_sigma_th = 20.24 * E_2Ryd_4 + 239.2 * E_2Ryd_6 + 2256 * E_2Ryd_8
    
    # Colgan+ 2016 equation 7 for He I  
    sigma_He_over_sigma_th = 1.913 * E_2Ryd_4 + 4.52 * E_2Ryd_6 + 7.90 * E_2Ryd_8
    
    # H I and He I contribution
    alpha_H_He = (n_h_i * sigma_H_over_sigma_th + n_he_i * sigma_He_over_sigma_th) * sigma_thomson
    
    # Dalgarno & Williams 1962 equation 3 for H2 (assumes λ in Å)
    # Convert frequency to wavelength in Angstroms for this formula
    inv_lambda_A = frequencies / (1e8 * c_cgs)  # 1/λ where λ is in Å
    inv_lambda2 = inv_lambda_A**2
    inv_lambda4 = inv_lambda2**2
    inv_lambda6 = inv_lambda2 * inv_lambda4
    inv_lambda8 = inv_lambda4**2
    
    # H2 Rayleigh scattering (Dalgarno & Williams 1962)
    alpha_H2 = (8.14e-13 * inv_lambda4 + 1.28e-6 * inv_lambda6 + 1.61 * inv_lambda8) * n_h2
    
    return alpha_H_He + alpha_H2