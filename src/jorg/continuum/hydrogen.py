"""
Hydrogen continuum absorption implementations in JAX

This module implements H I bound-free, H^- bound-free/free-free, 
and H2^+ bound-free/free-free absorption.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional
import json
import os

from ..constants import (
    c_cgs, hplanck_cgs, kboltz_cgs, hplanck_eV, kboltz_eV, 
    RydbergH_eV, H_minus_ion_energy, bf_sigma_const
)
from .utils import stimulated_emission_factor

# Load McLaughlin 2017 H^- cross section data
def _load_mclaughlin_data():
    """Load McLaughlin H^- cross section data"""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'mclaughlin_hminus.json')
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

# Global McLaughlin data (loaded once)
_MCLAUGHLIN_DATA = _load_mclaughlin_data()
_MCLAUGHLIN_NU = jnp.array(_MCLAUGHLIN_DATA['frequencies_hz'])
_MCLAUGHLIN_SIGMA = jnp.array(_MCLAUGHLIN_DATA['cross_sections_cm2'])
_H_MINUS_ION_NU = _MCLAUGHLIN_DATA['h_minus_ion_nu_hz']
_MIN_INTERP_NU = _MCLAUGHLIN_DATA['min_interp_nu_hz']
_LOW_NU_COEF = _MCLAUGHLIN_DATA['low_nu_coefficient']


@jax.jit
def simple_hydrogen_bf_cross_section(n: jnp.ndarray, frequency: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate H I bound-free cross section using simple hydrogenic approximation
    
    This implements equation 5.5 from Kurucz (1970) with the Z^4 correction.
    Used for n >= 7 levels where detailed cross sections aren't available.
    
    Parameters
    ----------
    n : jnp.ndarray
        Principal quantum number (scalar or array)
    frequency : jnp.ndarray
        Frequencies in Hz
        
    Returns
    -------
    jnp.ndarray
        Cross section in cm^2 (megabarns converted to cm^2)
    """
    inv_n = 1.0 / n
    inv_n2 = inv_n * inv_n
    
    # Ionization threshold frequency for level n
    # Use the actual H I ionization energy (chi_h), not the Rydberg constant
    chi_h = 13.598434005136  # eV, H I ionization energy (same as in main function)
    nu_threshold = chi_h * inv_n2 / hplanck_eV
    
    # Cross section is zero below threshold
    above_threshold = frequency > nu_threshold
    
    # Hydrogenic cross section formula
    # 64*π^4*e^10*m_e / (c*h^6*3*sqrt(3)) * n^-5 * ν^-3
    cross_section = jnp.where(
        above_threshold,
        bf_sigma_const * (inv_n2 * inv_n2 * inv_n) * (1.0 / frequency)**3 * 1e-18,  # convert Mb to cm^2
        0.0
    )
    
    return cross_section


@jax.jit
def h_i_bf_absorption(
    frequencies: jnp.ndarray,
    temperature: float,
    n_h_i: float,
    n_he_i: float,
    electron_density: float,
    inv_u_h: float,
    n_max_detailed: int = 6,
    n_max_total: int = 40
) -> jnp.ndarray:
    """
    Calculate H I bound-free absorption coefficient
    
    This is a simplified version of Korg's H_I_bf function that uses
    analytic cross sections for all levels. The full implementation
    would require loading detailed cross sections from data files.
    
    Parameters
    ----------
    frequencies : jnp.ndarray
        Frequencies in Hz (sorted)
    temperature : float
        Temperature in K
    n_h_i : float
        H I number density in cm^-3
    n_he_i : float
        He I number density in cm^-3 (used for level dissolution)
    electron_density : float
        Electron density in cm^-3
    inv_u_h : float
        Inverse H I partition function
    n_max_detailed : int, optional
        Maximum n for detailed cross sections (default: 6)
    n_max_total : int, optional
        Maximum n to include (default: 40)
        
    Returns
    -------
    jnp.ndarray
        H I bound-free absorption coefficient in cm^-1
    """
    chi_h = 13.598434005136  # H I ionization energy in eV
    
    total_cross_section = jnp.zeros_like(frequencies, dtype=jnp.float64)
    
    # Vectorized calculation over energy levels
    n_levels = jnp.arange(1, n_max_total + 1)
    
    # Occupation probabilities (simplified - no MHD dissolution)
    degeneracies = 2 * n_levels**2
    excitation_energies = chi_h * (1.0 - 1.0/n_levels**2)
    boltzmann_factors = jnp.exp(-excitation_energies / (kboltz_eV * temperature))
    occupation_probs = degeneracies * boltzmann_factors
    
    # Calculate cross sections for all levels
    def level_cross_section(n):
        return simple_hydrogen_bf_cross_section(n, frequencies)
    
    # Vectorize over levels
    cross_sections = jax.vmap(level_cross_section)(n_levels)
    
    # Weight by occupation probabilities and sum
    # Shape: (n_levels, n_frequencies) -> (n_frequencies)
    total_cross_section = jnp.sum(occupation_probs[:, None] * cross_sections, axis=0)
    
    # Include stimulated emission correction
    stim_emission = stimulated_emission_factor(frequencies, temperature)
    
    return n_h_i * inv_u_h * total_cross_section * stim_emission


@jax.jit
def h_minus_number_density(
    n_h_i_div_u: float,
    electron_density: float,
    temperature: float
) -> float:
    """
    Calculate H^- number density using Saha equation
    
    This implements equation 5.10 from Kurucz (1970).
    
    Parameters
    ----------
    n_h_i_div_u : float
        H I number density divided by partition function
    electron_density : float
        Electron density in cm^-3
    temperature : float
        Temperature in K
        
    Returns
    -------
    float
        H^- number density in cm^-3
    """
    # Ground state H I density (degeneracy = 2, Boltzmann factor = 1)
    n_h_i_ground = 2.0 * n_h_i_div_u
    
    # Saha equation coefficient: (h^2/(2*π*m_e))^1.5
    coef = 3.31283018e-22  # cm^3 * eV^1.5
    beta = 1.0 / (kboltz_eV * temperature)
    
    # H^- density from Saha equation
    n_h_minus = 0.25 * n_h_i_ground * electron_density * coef * beta**1.5 * jnp.exp(H_minus_ion_energy * beta)
    
    return n_h_minus


@jax.jit
def h_minus_bf_cross_section(frequency: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate H^- bound-free cross section using exact McLaughlin 2017 data
    
    This exactly matches Korg's implementation:
    - For ν < min_interp_nu: use coefficient * (ν - ν_ion)^1.5
    - For ν >= min_interp_nu: use linear interpolation of McLaughlin data
    
    Parameters
    ----------
    frequency : jnp.ndarray
        Frequencies in Hz
        
    Returns
    -------
    jnp.ndarray
        H^- bound-free cross section in cm^2
    """
    # Above ionization threshold
    above_threshold = frequency > _H_MINUS_ION_NU
    
    # Low frequency region (below interpolation range)
    low_freq = above_threshold & (frequency < _MIN_INTERP_NU)
    
    # High frequency region (interpolation range)
    high_freq = frequency >= _MIN_INTERP_NU
    
    # Initialize result
    result = jnp.zeros_like(frequency)
    
    # Low frequency: coefficient * (ν - ν_ion)^1.5
    result = jnp.where(
        low_freq,
        _LOW_NU_COEF * jnp.power(frequency - _H_MINUS_ION_NU, 1.5),
        result
    )
    
    # High frequency: interpolate McLaughlin data
    result = jnp.where(
        high_freq,
        jnp.interp(frequency, _MCLAUGHLIN_NU, _MCLAUGHLIN_SIGMA),
        result
    )
    
    return result


@jax.jit
def h_minus_bf_absorption(
    frequencies: jnp.ndarray,
    temperature: float,
    n_h_i_div_u: float,
    electron_density: float,
    include_stimulated_emission: bool = True
) -> jnp.ndarray:
    """
    Calculate H^- bound-free absorption coefficient
    
    Parameters
    ----------
    frequencies : jnp.ndarray
        Frequencies in Hz
    temperature : float
        Temperature in K
    n_h_i_div_u : float
        H I number density divided by partition function
    electron_density : float
        Electron density in cm^-3
    include_stimulated_emission : bool, optional
        Whether to include stimulated emission correction
        
    Returns
    -------
    jnp.ndarray
        H^- bound-free absorption coefficient in cm^-1
    """
    # H^- number density
    n_h_minus = h_minus_number_density(n_h_i_div_u, electron_density, temperature)
    
    # Cross section
    cross_section = h_minus_bf_cross_section(frequencies)
    
    # Absorption coefficient
    alpha = n_h_minus * cross_section
    
    # Include stimulated emission if requested
    stim_factor = jnp.where(
        include_stimulated_emission,
        stimulated_emission_factor(frequencies, temperature),
        1.0
    )
    alpha *= stim_factor
    
    return alpha


# Bell & Berrington (1987) H^- free-free absorption data
# Table from https://doi.org/10.1088/0022-3700/20/4/019
_BELL_BERRINGTON_THETA = jnp.array([0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.8, 3.6])

_BELL_BERRINGTON_LAMBDA = jnp.array([
    1823, 2278, 2604, 3038, 3645, 4557, 5063, 5696, 6510, 7595, 9113,
    10126, 11392, 13019, 15189, 18227, 22784, 30378, 45567, 91134,
    113918, 151890
])

# K values in units of 10^-26 cm^4/dyn 
# Table organized as [wavelength_index, theta_index] exactly matching Korg.jl
_BELL_BERRINGTON_K = jnp.array([
    [0.0178, 0.0222, 0.0308, 0.0402, 0.0498, 0.0596, 0.0695, 0.0795, 0.0896, 0.131, 0.172],  # 1823 Å
    [0.0228, 0.0280, 0.0388, 0.0499, 0.0614, 0.0732, 0.0851, 0.0972, 0.110, 0.160, 0.211],   # 2278 Å
    [0.0277, 0.0342, 0.0476, 0.0615, 0.0760, 0.0908, 0.105, 0.121, 0.136, 0.199, 0.262],    # 2604 Å
    [0.0364, 0.0447, 0.0616, 0.0789, 0.0966, 0.114, 0.132, 0.150, 0.169, 0.243, 0.318],     # 3038 Å
    [0.0520, 0.0633, 0.0859, 0.108, 0.131, 0.154, 0.178, 0.201, 0.225, 0.321, 0.418],       # 3645 Å
    [0.0791, 0.0959, 0.129, 0.161, 0.194, 0.227, 0.260, 0.293, 0.327, 0.463, 0.602],        # 4557 Å
    [0.0965, 0.117, 0.157, 0.195, 0.234, 0.272, 0.311, 0.351, 0.390, 0.549, 0.711],         # 5063 Å
    [0.121, 0.146, 0.195, 0.241, 0.288, 0.334, 0.381, 0.428, 0.475, 0.667, 0.861],          # 5696 Å
    [0.154, 0.188, 0.249, 0.309, 0.367, 0.424, 0.482, 0.539, 0.597, 0.830, 1.07],           # 6510 Å
    [0.208, 0.250, 0.332, 0.409, 0.484, 0.557, 0.630, 0.702, 0.774, 1.06, 1.36],            # 7595 Å
    [0.293, 0.354, 0.468, 0.576, 0.677, 0.777, 0.874, 0.969, 1.06, 1.45, 1.83],             # 9113 Å
    [0.358, 0.432, 0.572, 0.702, 0.825, 0.943, 1.06, 1.17, 1.28, 1.73, 2.17],               # 10126 Å
    [0.448, 0.539, 0.711, 0.871, 1.02, 1.16, 1.29, 1.43, 1.57, 2.09, 2.60],                 # 11392 Å
    [0.579, 0.699, 0.924, 1.13, 1.33, 1.51, 1.69, 1.86, 2.02, 2.67, 3.31],                  # 13019 Å
    [0.781, 0.940, 1.24, 1.52, 1.78, 2.02, 2.26, 2.48, 2.69, 3.52, 4.31],                   # 15189 Å
    [1.11, 1.34, 1.77, 2.17, 2.53, 2.87, 3.20, 3.51, 3.80, 4.92, 5.97],                     # 18227 Å
    [1.73, 2.08, 2.74, 3.37, 3.90, 4.50, 5.01, 5.50, 5.95, 7.59, 9.06],                     # 22784 Å
    [3.04, 3.65, 4.80, 5.86, 6.86, 7.79, 8.67, 9.50, 10.3, 13.2, 15.6],                     # 30378 Å
    [6.79, 8.16, 10.7, 13.1, 15.3, 17.4, 19.4, 21.2, 23.0, 29.5, 35.0],                     # 45567 Å
    [27.0, 32.4, 42.6, 51.9, 60.7, 68.9, 76.8, 84.2, 91.4, 117.0, 140.0],                   # 91134 Å
    [42.3, 50.6, 66.4, 80.8, 94.5, 107.0, 120.0, 131.0, 142.0, 183.0, 219.0],               # 113918 Å
    [75.1, 90.0, 118.0, 144.0, 168.0, 191.0, 212.0, 234.0, 253.0, 325.0, 388.0]            # 151890 Å
])


@jax.jit  
def _interpolate_bell_berrington(wavelength_angstrom: float, theta: float) -> float:
    """
    Interpolate Bell & Berrington (1987) H^- free-free absorption table
    
    This implements the exact same 2D linear interpolation as Korg.jl
    
    Parameters
    ----------
    wavelength_angstrom : float
        Wavelength in Angstroms
    theta : float
        Temperature parameter θ = 5040/T
        
    Returns
    -------
    float
        Interpolated K value in units of 10^-26 cm^4/dyn
    """
    # Clamp to table bounds
    lambda_min, lambda_max = _BELL_BERRINGTON_LAMBDA[0], _BELL_BERRINGTON_LAMBDA[-1]
    theta_min, theta_max = _BELL_BERRINGTON_THETA[0], _BELL_BERRINGTON_THETA[-1]
    
    wl_clamped = jnp.clip(wavelength_angstrom, lambda_min, lambda_max)
    theta_clamped = jnp.clip(theta, theta_min, theta_max)
    
    # Find interpolation indices
    lambda_idx = jnp.searchsorted(_BELL_BERRINGTON_LAMBDA, wl_clamped) - 1
    lambda_idx = jnp.clip(lambda_idx, 0, len(_BELL_BERRINGTON_LAMBDA) - 2)
    
    theta_idx = jnp.searchsorted(_BELL_BERRINGTON_THETA, theta_clamped) - 1
    theta_idx = jnp.clip(theta_idx, 0, len(_BELL_BERRINGTON_THETA) - 2)
    
    # Get interpolation fractions
    lambda_frac = ((wl_clamped - _BELL_BERRINGTON_LAMBDA[lambda_idx]) /
                   (_BELL_BERRINGTON_LAMBDA[lambda_idx + 1] - _BELL_BERRINGTON_LAMBDA[lambda_idx]))
    theta_frac = ((theta_clamped - _BELL_BERRINGTON_THETA[theta_idx]) /
                  (_BELL_BERRINGTON_THETA[theta_idx + 1] - _BELL_BERRINGTON_THETA[theta_idx]))
    
    # Bilinear interpolation
    k00 = _BELL_BERRINGTON_K[lambda_idx, theta_idx]
    k01 = _BELL_BERRINGTON_K[lambda_idx, theta_idx + 1]
    k10 = _BELL_BERRINGTON_K[lambda_idx + 1, theta_idx]
    k11 = _BELL_BERRINGTON_K[lambda_idx + 1, theta_idx + 1]
    
    k_interp = (k00 * (1 - lambda_frac) * (1 - theta_frac) +
                k01 * (1 - lambda_frac) * theta_frac +
                k10 * lambda_frac * (1 - theta_frac) +
                k11 * lambda_frac * theta_frac)
    
    return k_interp


@jax.jit
def h_minus_ff_absorption(
    frequencies: jnp.ndarray,
    temperature: float,
    n_h_i_div_u: float,
    electron_density: float
) -> jnp.ndarray:
    """
    Calculate H^- free-free absorption coefficient exactly following Korg.jl implementation
    
    This implements the exact Bell & Berrington (1987) table interpolation used in Korg.jl.
    The reaction is: photon + e^- + H I -> e^- + H I
    
    Parameters
    ----------
    frequencies : jnp.ndarray
        Frequencies in Hz
    temperature : float
        Temperature in K
    n_h_i_div_u : float
        H I number density divided by partition function
    electron_density : float
        Electron density in cm^-3
        
    Returns
    -------
    jnp.ndarray
        H^- free-free absorption coefficient in cm^-1
        
    Notes
    -----
    Based on Table 1 from Bell & Berrington (1987) which tabulates K values
    in units of cm^4/dyn. Must be multiplied by electron pressure and 
    ground-state H I density to get absorption coefficient.
    
    Valid ranges:
    - Wavelength: 1823-151890 Å  
    - Temperature: 1400-10080 K (θ = 0.5-3.6)
    """
    # Convert frequency to wavelength in Angstroms
    wavelength_angstrom = c_cgs * 1e8 / frequencies
    
    # Temperature parameter θ = 5040/T (exactly as in Korg.jl)
    theta = 5040.0 / temperature
    
    # Vectorized interpolation over the Bell & Berrington table
    def interpolate_single(wl):
        return _interpolate_bell_berrington(wl, theta)
    
    # Apply interpolation to each wavelength
    K_values = jax.vmap(interpolate_single)(wavelength_angstrom)
    
    # K is in units of 10^-26 cm^4/dyn, so multiply by 1e-26
    K = K_values * 1e-26
    
    # Electron pressure in dyn/cm^2 (cgs units)
    P_e = electron_density * kboltz_cgs * temperature
    
    # Ground state H I density: n(H I, n=1) = 2 * n(H I)/U(T) 
    # (degeneracy=2, Boltzmann factor=1 for ground state)
    n_h_i_ground_state = 2.0 * n_h_i_div_u
    
    # Final absorption coefficient: K * P_e * n(H I, ground state)
    return K * P_e * n_h_i_ground_state


@jax.jit
def h2_plus_bf_ff_absorption(
    frequencies: jnp.ndarray,
    temperature: float,
    n_h_i: float,
    n_h_ii: float,
    include_stimulated_emission: bool = True
) -> jnp.ndarray:
    """
    Calculate H2^+ bound-free and free-free absorption coefficient
    
    This is a simplified implementation that would need detailed
    cross sections from Stancil (1994) for full accuracy.
    
    Parameters
    ----------
    frequencies : jnp.ndarray
        Frequencies in Hz
    temperature : float
        Temperature in K
    n_h_i : float
        H I number density in cm^-3
    n_h_ii : float
        H II number density in cm^-3
    include_stimulated_emission : bool, optional
        Whether to include stimulated emission correction
        
    Returns
    -------
    jnp.ndarray
        H2^+ bound-free and free-free absorption coefficient in cm^-1
    """
    # This is a placeholder implementation
    # The full version would need Stancil (1994) cross sections and equilibrium constants
    
    # Simple approximation: assume small contribution
    # Real implementation needs tabulated data
    alpha = jnp.zeros_like(frequencies, dtype=jnp.float64)
    
    # Include stimulated emission if requested
    stim_factor = jnp.where(
        include_stimulated_emission,
        stimulated_emission_factor(frequencies, temperature),
        1.0
    )
    alpha *= stim_factor
    
    return alpha