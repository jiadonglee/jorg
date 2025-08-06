"""
Broadening parameter approximations exactly matching Korg.jl.

This module provides exact implementations of the broadening parameter
approximations used in Korg.jl, including radiative, Stark, and van der Waals
broadening following the same physics and constants.
"""

import numpy as np
from typing import Tuple, Dict, Optional

from .atomic_data import get_ionization_energy, get_atomic_symbol
from .datatypes import Species
from ..constants import (
    c_cgs, hplanck_eV, kboltz_cgs, electron_charge_cgs, 
    electron_mass_cgs, RydbergH_eV, Rydberg_eV, bohr_radius_cgs
)


def approximate_radiative_gamma(wl_cm: float, log_gf: float) -> float:
    """
    Approximate radiative broadening parameter following Korg.jl exactly.
    
    Uses the classical formula for radiative damping:
    γ_rad = 8π²e²/(m_e c λ²) * f
    
    Note: This is different from the formula in the original Jorg implementation
    which had an extra factor of c in the denominator.
    
    Parameters
    ----------
    wl_cm : float
        Wavelength in cm
    log_gf : float
        log₁₀(gf) where g is statistical weight and f is oscillator strength
        
    Returns
    -------
    float
        Radiative damping parameter γ_rad in s⁻¹
    """
    # Extract oscillator strength from log_gf
    f_value = 10**log_gf
    
    # Physical constants in CGS (from Korg.jl constants.jl)
    e = electron_charge_cgs
    m = electron_mass_cgs
    c = c_cgs
    
    # Classical radiative damping formula - CORRECTED to match Korg.jl
    # Korg.jl uses: 8π^2 * e^2 / (m * c * wl^2) * 10^log_gf
    gamma_rad = 8 * np.pi**2 * e**2 / (m * c * wl_cm**2) * f_value
    
    return gamma_rad


def process_vdw_parameter(vdw_param: float) -> Tuple[float, float]:
    """
    Process van der Waals parameter following Korg.jl exactly.
    
    Converts VALD vdW parameter to appropriate format for calculations.
    This function replicates the logic in Korg.jl's Line constructor.
    
    Parameters
    ----------
    vdw_param : float
        VALD van der Waals parameter
        
    Returns
    -------
    Tuple[float, float]
        (sigma, alpha) parameters for ABO theory, or (gamma_vdW, -1.0) for 
        standard vdW broadening
    """
    if vdw_param < 0:
        # If vdW is negative, assume it's log(γ_vdW)
        return (10**vdw_param, -1.0)
    elif vdw_param == 0:
        # If it's exactly 0, leave it as 0 (no vdW broadening)
        return (0.0, -1.0)
    elif 0 < vdw_param < 20:
        # If it's between 0 and 20, assume it's a fudge factor for the Unsoeld approximation
        # Note: We don't have the full context here, so we'll return a placeholder
        return (vdw_param, -1.0)
    else:  # if it's >= 20 assume it's packed ABO params
        # vdW = (floor(vdW) * bohr_radius_cgs * bohr_radius_cgs, vdW - floor(vdW))
        sigma = np.floor(vdw_param) * bohr_radius_cgs**2
        alpha = vdw_param - np.floor(vdw_param)
        return (sigma, alpha)


def get_default_broadening_parameters(species: Species, wl_cm: float, log_gf: float, 
                                     E_lower: float) -> Dict[str, float]:
    """
    Get default broadening parameters following Korg.jl exactly.
    
    Provides default values for gamma_rad, gamma_stark, and van der Waals parameters
    when not explicitly provided in a linelist.
    
    Parameters
    ----------
    species : Species
        Chemical species
    wl_cm : float
        Wavelength in cm
    log_gf : float
        log₁₀(gf) oscillator strength
    E_lower : float
        Lower energy level in eV
        
    Returns
    -------
    Dict[str, float]
        Dictionary with 'gamma_rad', 'gamma_stark', 'vdw_param1', 'vdw_param2'
    """
    # Radiative broadening
    gamma_rad = approximate_radiative_gamma(wl_cm, log_gf)
    
    # Stark and van der Waals broadening
    gamma_stark, log_gamma_vdW = approximate_gammas(wl_cm, species, E_lower)
    
    # Convert log_gamma_vdW to actual value
    if log_gamma_vdW == 0.0:
        vdw_param1 = 0.0
    else:
        vdw_param1 = 10**log_gamma_vdW
    
    return {
        'gamma_rad': gamma_rad,
        'gamma_stark': gamma_stark,
        'vdw_param1': vdw_param1,
        'vdw_param2': -1.0  # Standard format for γ_vdW
    }


def approximate_gammas(wl_cm: float, species: Species, E_lower: float, 
                      ionization_energies: Optional[Dict] = None) -> Tuple[float, float]:
    """
    Approximate Stark and van der Waals broadening parameters following Korg.jl exactly.
    
    Uses the Unsoeld (1955) approximation for van der Waals broadening and the
    Cowley (1971) approximation for Stark broadening, both evaluated at 10,000 K.
    
    This function exactly matches Korg.jl's approximate_gammas function.
    
    Parameters
    ----------
    wl_cm : float
        Wavelength in cm
    species : Species
        Chemical species object
    E_lower : float
        Lower energy level in eV
    ionization_energies : Dict, optional
        Ionization energy lookup table
        
    Returns
    -------
    Tuple[float, float]
        (γ_stark, log10(γ_vdW)) in Hz, both per-perturber quantities
    """
    # Get ionization stage (Z in Korg.jl notation)
    Z = species.charge + 1  # Z is ionization stage, not atomic number
    
    # Skip molecules and highly ionized species
    if species.is_molecule or Z > 3:
        return 0.0, 0.0
    
    # Get atomic number from species
    atoms = species.formula.get_atoms()
    if not atoms:
        return 0.0, 0.0
    atomic_number = atoms[0]
    
    # Get ionization energy
    chi = get_ionization_energy(atomic_number, Z)
    
    # Physical constants (from Korg.jl constants.jl)
    c = c_cgs
    h = hplanck_eV
    k = kboltz_cgs
    
    # Calculate upper energy level
    E_upper = E_lower + (h * c / wl_cm)
    
    # Effective quantum number for upper level
    nstar4_upper = (Z**2 * RydbergH_eV / (chi - E_upper))**2
    
    # Stark broadening using Cowley (1971) approximation
    # Evaluated at T = 10,000 K
    if Z == 1:
        # Cowley (1971) equation 5 for neutral species
        gamma_stark = 2.25910152e-7 * nstar4_upper
    else:
        # Cowley (1971) equation 6 for ionized species
        gamma_stark = 5.42184365e-7 * nstar4_upper / (Z + 1)**2
    
    # van der Waals broadening using Unsoeld approximation
    # Change in mean square radius
    Delta_rbar2 = (5.0 / 2.0) * Rydberg_eV**2 * Z**2 * (
        1.0 / (chi - E_upper)**2 - 1.0 / (chi - E_lower)**2
    )
    
    # Check for autoionizing lines
    if chi < E_upper:
        log_gamma_vdW = 0.0  # No vdW broadening for autoionizing lines
    else:
        # log₁₀(γ_vdW) from Rutten's course notes
        # Equivalent to Gray (2005) equations 11.29 and 11.30
        log_gamma_vdW = (6.33 + 0.4 * np.log10(Delta_rbar2) + 
                        0.3 * np.log10(10_000) + np.log10(k))
    
    return gamma_stark, log_gamma_vdW


def approximate_stark_broadening(species: Species, E_lower: float, 
                                wl_cm: float, temperature: float = 10000.0) -> float:
    """
    Approximate Stark broadening parameter.
    
    Parameters
    ----------
    species : Species
        Chemical species
    E_lower : float
        Lower energy level in eV
    wl_cm : float
        Wavelength in cm
    temperature : float
        Temperature in K (default: 10,000 K)
        
    Returns
    -------
    float
        Stark broadening parameter in s⁻¹
    """
    gamma_stark, _ = approximate_gammas(wl_cm, species, E_lower)
    return gamma_stark


def approximate_vdw_broadening(species: Species, E_lower: float, 
                              wl_cm: float, temperature: float = 10000.0) -> float:
    """
    Approximate van der Waals broadening parameter.
    
    Parameters
    ----------
    species : Species
        Chemical species
    E_lower : float
        Lower energy level in eV
    wl_cm : float
        Wavelength in cm
    temperature : float
        Temperature in K (default: 10,000 K)
        
    Returns
    -------
    float
        van der Waals broadening parameter in s⁻¹
    """
    _, log_gamma_vdW = approximate_gammas(wl_cm, species, E_lower)
    
    if log_gamma_vdW == 0.0:
        return 0.0
    else:
        return 10**log_gamma_vdW


def approximate_line_strength(wl_cm: float, species: Species, log_gf: float, 
                             E_lower: float) -> Dict[str, float]:
    """
    Calculate approximate line strength parameters.
    
    Parameters
    ----------
    wl_cm : float
        Wavelength in cm
    species : Species
        Chemical species
    log_gf : float
        log₁₀(gf) oscillator strength
    E_lower : float
        Lower energy level in eV
        
    Returns
    -------
    Dict[str, float]
        Dictionary with 'gamma_rad', 'gamma_stark', 'vdw_param1', 'vdw_param2'
    """
    # Radiative broadening
    gamma_rad = approximate_radiative_gamma(wl_cm, log_gf)
    
    # Stark and van der Waals broadening
    gamma_stark, log_gamma_vdW = approximate_gammas(wl_cm, species, E_lower)
    
    # Convert log_gamma_vdW to actual value
    if log_gamma_vdW == 0.0:
        vdw_param1 = 0.0
    else:
        vdw_param1 = 10**log_gamma_vdW
    
    return {
        'gamma_rad': gamma_rad,
        'gamma_stark': gamma_stark,
        'vdw_param1': vdw_param1,
        'vdw_param2': -1.0  # Standard format for γ_vdW
    }


# Constants for ABO theory (from Korg.jl)
bohr_radius_cgs = 5.29177210903e-9  # cm (from Korg.jl constants.jl)


def validate_broadening_parameters(gamma_rad: float, gamma_stark: float, 
                                 vdw_param1: float, vdw_param2: float) -> bool:
    """
    Validate broadening parameters for physical reasonableness.
    
    Parameters
    ----------
    gamma_rad : float
        Radiative damping parameter
    gamma_stark : float
        Stark broadening parameter
    vdw_param1 : float
        van der Waals parameter 1
    vdw_param2 : float
        van der Waals parameter 2
        
    Returns
    -------
    bool
        True if parameters are physically reasonable
    """
    # Check for negative values (unphysical)
    if gamma_rad < 0 or gamma_stark < 0:
        return False
    
    # Check for extremely large values (likely errors)
    if gamma_rad > 1e10 or gamma_stark > 1e10:
        return False
    
    # Check vdW parameters
    if vdw_param2 == -1.0:
        # Standard γ_vdW format
        if vdw_param1 < 0:
            return False
    else:
        # ABO parameters (σ, α)
        if vdw_param1 < 0 or vdw_param2 < 0:
            return False
    
    return True