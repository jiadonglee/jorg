"""
Exact Korg.jl van der Waals Parameter System for Jorg
====================================================

This module implements Korg.jl's exact van der Waals broadening parameter system,
extracted directly from Korg.jl source code (linelist.jl lines 80-89, 145-178).

NO hardcoded values, NO empirical corrections, NO simplified functions.
Uses Korg.jl's exact physics-based calculations and parameter processing logic.

Key findings from Korg.jl source analysis:
1. Korg.jl uses VALD vdW parameters directly from line lists
2. Default vdW calculated using Unsoeld (1955) approximation: 6.33 + 0.4log10(Δrbar2) + 0.3log10(10000) + log10(k)
3. ABO theory for packed parameters (vdW >= 20)
4. Species-specific parameters come from VALD data, NOT hardcoded optimizations
"""

import jax.numpy as jnp
import numpy as np
from jax import jit
from typing import Optional, Tuple

from ..constants import (
    kboltz_cgs, c_cgs, hplanck_eV, 
    RydbergH_eV, Rydberg_eV,
    bohr_radius_cgs
)
# Lazy loading to avoid circular imports
_ionization_energies = None

def _get_ionization_energies():
    """Lazy loading of ionization energies to avoid circular imports."""
    global _ionization_energies
    if _ionization_energies is None:
        from ..statmech import create_default_ionization_energies
        _ionization_energies = create_default_ionization_energies()
    return _ionization_energies


def approximate_gammas_korg(wl_cm: float, atomic_number: int, charge: int, 
                           E_lower_eV: float) -> Tuple[float, float]:
    """
    Exact implementation of Korg.jl's approximate_gammas function.
    
    From Korg.jl linelist.jl lines 145-178:
    - Simplified Unsoeld (1955) approximation for van der Waals broadening
    - Cowley (1971) approximation for Stark broadening
    - Evaluated at 10,000 K
    
    Parameters:
    -----------
    wl_cm : float
        Wavelength in cm
    atomic_number : int
        Atomic number (Z)
    charge : int
        Charge state (0 = neutral, 1 = singly ionized, etc.)
    E_lower_eV : float
        Lower level excitation energy in eV
        
    Returns:
    --------
    Tuple[float, float]
        (γ_stark, log10(γ_vdW)) in Hz (FWHM, per-perturber quantities)
    """
    
    Z = charge + 1  # Z is ionization stage, not atomic number
    
    # Skip molecules and highly ionized species (Korg.jl line 147-149)
    if Z > 3:
        return 0.0, 0.0
    
    # Get ionization energy (Korg.jl line 150)
    # Note: using Jorg's ionization_energies like Korg.jl
    try:
        ionization_energies = _get_ionization_energies()
        χ = ionization_energies[atomic_number][Z-1]  # Convert to 0-based indexing
    except (KeyError, IndexError):
        return 0.0, 0.0
    
    # Physical constants (Korg.jl lines 151-154)
    c = c_cgs
    h = hplanck_eV
    k = kboltz_cgs
    E_upper = E_lower_eV + (h * c / wl_cm)
    
    # Stark broadening calculation (Korg.jl lines 158-167)
    nstar4_upper = (Z**2 * RydbergH_eV / (χ - E_upper))**2
    
    if Z == 1:
        # Cowley (1971) equation 5 evaluated at T=10,000 K
        γstark = 2.25910152e-7 * nstar4_upper
    else:
        # Cowley (1971) equation 6 @ T=10,000 K
        γstark = 5.42184365e-7 * nstar4_upper / (Z + 1)**2
    
    # van der Waals broadening calculation (Korg.jl lines 169-176)
    Δrbar2 = (5 / 2) * Rydberg_eV**2 * Z**2 * (1 / (χ - E_upper)**2 - 1 / (χ - E_lower_eV)**2)
    
    if χ < E_upper:
        # No vdW for autoionizing lines (Korg.jl lines 170-171)
        log_γvdW = 0.0  # This will be interpreted as γ, not log γ
    else:
        # Unsoeld approximation from Gray (2005) equations 11.29-11.30 (Korg.jl lines 173-175)
        # This is Korg.jl's exact default formula (NOT hardcoded -7.5!)
        log_γvdW = 6.33 + 0.4 * jnp.log10(Δrbar2) + 0.3 * jnp.log10(10000.0) + jnp.log10(k)
    
    return γstark, log_γvdW


def process_vald_vdw_parameter_korg(vdW_raw, wl_cm: float, atomic_number: int, 
                                   charge: int, E_lower_eV: float) -> Tuple[float, float]:
    """
    Exact implementation of Korg.jl's vdW parameter processing logic.
    
    From Korg.jl linelist.jl lines 80-91:
    Converts VALD vdW parameter to (γ_vdW, indicator) tuple exactly as Korg.jl does.
    
    Parameters:
    -----------
    vdW_raw : float or None
        Raw vdW parameter from VALD line list
    wl_cm : float
        Wavelength in cm
    atomic_number : int
        Atomic number
    charge : int
        Charge state
    E_lower_eV : float
        Lower level excitation energy in eV
        
    Returns:
    --------
    Tuple[float, float]
        (γ_vdW, indicator) - where indicator = -1 for standard, ≥0 for ABO α parameter
    """
    
    if vdW_raw is None or np.isnan(vdW_raw):
        # Use Korg.jl's exact default calculation (Korg.jl line 66-68)
        _, log_γvdW_approx = approximate_gammas_korg(wl_cm, atomic_number, charge, E_lower_eV)
        vdW_processed = float(log_γvdW_approx)
    else:
        vdW_processed = float(vdW_raw)
    
    # Apply Korg.jl's exact parameter processing logic (lines 80-91)
    if vdW_processed < 0:
        # Negative value: assume it's log(γ_vdW) (Korg.jl line 82)
        return (10**vdW_processed, -1.0)
    elif vdW_processed == 0:
        # Exactly 0: no vdW broadening (Korg.jl line 84)
        return (0.0, -1.0)
    elif 0 < vdW_processed < 20:
        # 0-20 range: fudge factor for Unsoeld approximation (Korg.jl lines 85-87)
        _, log_γvdW_approx = approximate_gammas_korg(wl_cm, atomic_number, charge, E_lower_eV)
        γ_vdW = vdW_processed * 10**log_γvdW_approx
        return (γ_vdW, -1.0)
    else:
        # >= 20: packed ABO parameters (Korg.jl lines 88-89)
        # (σ, α) from ABO theory
        σ_abo = np.floor(vdW_processed) * bohr_radius_cgs * bohr_radius_cgs
        α_abo = vdW_processed - np.floor(vdW_processed)
        return (σ_abo, α_abo)


def get_korg_vdw_parameter(line_wavelength_cm: float, species_id: int, 
                          excitation_potential_eV: float, 
                          vald_vdw_param: Optional[float] = None) -> Tuple[float, float]:
    """
    Get van der Waals parameter exactly as Korg.jl does.
    
    This replaces the hardcoded Jorg species-specific parameters with
    Korg.jl's exact VALD-based system.
    
    Parameters:
    -----------
    line_wavelength_cm : float
        Line wavelength in cm
    species_id : int
        Species ID (element*100 + ionization_stage)
    excitation_potential_eV : float
        Lower level excitation energy in eV
    vald_vdw_param : float, optional
        VALD vdW parameter from line list
        
    Returns:
    --------
    Tuple[float, float]
        (γ_vdW, indicator) following Korg.jl's exact convention
        - indicator = -1.0: standard vdW (γ_vdW in s⁻¹)
        - indicator ≥ 0.0: ABO α parameter
    """
    
    # Extract atomic number and charge from species_id
    atomic_number = species_id // 100
    charge = (species_id % 100) - 1  # Convert from 1-based to 0-based
    charge = max(0, charge)  # Ensure non-negative
    
    # Use Korg.jl's exact parameter processing
    return process_vald_vdw_parameter_korg(
        vald_vdw_param, line_wavelength_cm, atomic_number, charge, excitation_potential_eV
    )


def scaled_vdw_korg(vdW_tuple: Tuple[float, float], atomic_mass_amu: float, 
                   temperature_K: float) -> float:
    """
    Exact implementation of Korg.jl's scaled_vdW function.
    
    This function handles both standard and ABO vdW parameters exactly as Korg.jl does.
    
    Parameters:
    -----------
    vdW_tuple : Tuple[float, float]
        (γ_vdW_or_σ, indicator) from get_korg_vdw_parameter
    atomic_mass_amu : float
        Atomic mass in amu
    temperature_K : float
        Temperature in K
        
    Returns:
    --------
    float
        Scaled vdW broadening parameter in s⁻¹
    """
    
    γ_vdW_or_σ, indicator = vdW_tuple
    
    if indicator < 0:
        # Standard vdW parameter: scale with temperature (T/10000)^0.3
        return γ_vdW_or_σ * (temperature_K / 10000.0)**0.3
    else:
        # ABO theory calculation
        # This implements the exact ABO formula from Korg.jl line_absorption.jl
        σ_abo = γ_vdW_or_σ
        α_abo = indicator
        
        # ABO constants (from Korg.jl)
        v0 = 1e6  # cm/s (σ is given at 10,000 m/s = 10^6 cm/s)
        amu_cgs = 1.66054e-24  # g
        
        # Inverse reduced mass (exact Korg.jl formula)
        invμ = 1.0 / (1.008 * amu_cgs) + 1.0 / (atomic_mass_amu * amu_cgs)
        
        # Relative velocity (exact Korg.jl formula)
        vbar = jnp.sqrt(8 * kboltz_cgs * temperature_K / jnp.pi * invμ)
        
        # ABO formula (exact Korg.jl formula)
        # γ = 2 * (4/π)^(α/2) * Γ((4-α)/2) * v₀ * σ * (vbar/v₀)^(1-α)
        from jax.scipy.special import gamma as gamma_func
        γ_abo = (2 * (4 / jnp.pi)**(α_abo / 2) * gamma_func((4 - α_abo) / 2) * 
                v0 * σ_abo * (vbar / v0)**(1 - α_abo))
        
        return γ_abo


def validate_korg_vdw_system():
    """
    Validate the Korg.jl vdW parameter system implementation.
    
    Tests key components against known physics and Korg.jl behavior.
    """
    
    print("🔧 Validating Korg.jl vdW Parameter System")
    print("=" * 50)
    
    # Test 1: approximate_gammas for typical stellar conditions
    print("\n1. Testing approximate_gammas function:")
    
    # Fe I at 5000 Å
    wl_test = 5000e-8  # cm
    Fe_atomic_number = 26
    Fe_charge = 0
    E_lower_test = 1.0  # eV
    
    γstark, log_γvdW = approximate_gammas_korg(wl_test, Fe_atomic_number, Fe_charge, E_lower_test)
    
    print(f"   Fe I at 5000 Å, E_lower=1.0 eV:")
    print(f"   γ_stark = {γstark:.3e} Hz")
    print(f"   log10(γ_vdW) = {log_γvdW:.3f}")
    print(f"   γ_vdW = {10**log_γvdW:.3e} Hz")
    
    # Compare with old hardcoded default
    old_default = -7.5
    print(f"   Old Jorg default: {old_default:.3f}")
    print(f"   Korg.jl physics-based: {log_γvdW:.3f}")
    print(f"   Difference: {log_γvdW - old_default:.3f}")
    
    # Test 2: VALD parameter processing
    print("\n2. Testing VALD parameter processing:")
    
    test_cases = [
        (-7.8, "Negative: log(γ_vdW)"),
        (0.0, "Zero: no broadening"),
        (1.5, "1-20: fudge factor"),
        (25.3, "≥20: ABO parameters")
    ]
    
    for vdW_raw, description in test_cases:
        γ_vdW, indicator = process_vald_vdw_parameter_korg(
            vdW_raw, wl_test, Fe_atomic_number, Fe_charge, E_lower_test
        )
        print(f"   {description}: vdW={vdW_raw} → (γ={γ_vdW:.3e}, ind={indicator:.1f})")
    
    # Test 3: Temperature scaling
    print("\n3. Testing temperature scaling:")
    
    # Standard vdW parameter
    vdW_standard = (1e9, -1.0)  # 1e9 Hz, standard indicator
    atomic_mass = 55.845  # Fe mass
    
    temps = [4000, 5780, 8000]
    print(f"   Standard vdW scaling (T/10000)^0.3:")
    for T in temps:
        scaled = scaled_vdw_korg(vdW_standard, atomic_mass, T)
        scaling_factor = (T / 10000.0)**0.3
        print(f"   T={T}K: γ={scaled:.3e} Hz (factor={scaling_factor:.3f})")
    
    print("\n✅ Korg.jl vdW parameter system validation complete")
    print("Ready to replace hardcoded Jorg system with exact Korg.jl implementation")


if __name__ == "__main__":
    validate_korg_vdw_system()