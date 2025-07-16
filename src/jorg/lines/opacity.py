"""
Line opacity calculations for stellar spectral synthesis
"""

import jax.numpy as jnp
import jax
from jax import jit
from typing import Optional

from ..constants import (
    PLANCK_H, BOLTZMANN_K, SPEED_OF_LIGHT, ELECTRON_MASS, 
    ELEMENTARY_CHARGE, VACUUM_PERMEABILITY, PI, AVOGADRO
)


@jit
def harris_series_korg(v):
    """Harris series from Korg.jl - exact implementation"""
    v2 = v * v
    H0 = jnp.exp(-v2)
    
    # Coefficients from Korg.jl - using JAX conditional operations
    def h1_case1():
        return -1.12470432 + (-0.15516677 + (3.288675912 + (-2.34357915 + 0.42139162 * v) * v) * v) * v
    
    def h1_case2():
        return -0.50637523 + (-0.0946391 + (0.48058512 + (-0.12194999 + 0.009740301 * v) * v) * v) * v
    
    def h1_case3():
        return 0.21233787 + (-0.08020186 + (0.013842842 + (-0.0007160979 + 0.000005118 * v) * v) * v) * v
    
    def h1_case4():
        return 0.08506242 + (-0.008477697 + (0.0002728695 + (-0.00000174 + 0.000000004 * v) * v) * v) * v
    
    H1 = jnp.where(v < 1.3, h1_case1(),
                   jnp.where(v < 2.3, h1_case2(),
                             jnp.where(v < 3.3, h1_case3(), h1_case4())))
    
    # H2 calculation
    H2 = 2 * v2 * H0 - 1.12837916709551 + H1 * v
    
    return H0, H1, H2

@jit
def voigt_hjerting(a, v):
    """
    Exact Korg.jl Voigt-Hjerting function implementation
    
    Translated directly from Korg.jl line_absorption.jl:270-294
    
    Parameters:
    - a: damping parameter (γ/(σ√2))  
    - v: frequency parameter (|λ-λ₀|/(σ√2))
    
    Returns:
    - H(a,v) Voigt-Hjerting function value
    """
    v2 = v * v
    
    # Case 1: α ≤ 0.2 && v ≥ 5
    def case1():
        invv2 = 1 / v2
        return (a / jnp.sqrt(jnp.pi) * invv2) * (1 + 1.5 * invv2 + 3.75 * invv2**2)
    
    # Case 2: α ≤ 0.2 && v < 5
    def case2():
        H0, H1, H2 = harris_series_korg(v)
        return H0 + (H1 + H2 * a) * a
    
    # Case 3: α ≤ 1.4 && α + v < 3.2 (modified harris series)
    def case3():
        H0, H1, H2 = harris_series_korg(v)
        M0 = H0
        M1 = H1 + 2 / jnp.sqrt(jnp.pi) * M0
        M2 = H2 - M0 + 2 / jnp.sqrt(jnp.pi) * M1
        M3 = 2 / (3 * jnp.sqrt(jnp.pi)) * (1 - H2) - (2 / 3) * v2 * M1 + (2 / jnp.sqrt(jnp.pi)) * M2
        M4 = 2 / 3 * v2 * v2 * M0 - 2 / (3 * jnp.sqrt(jnp.pi)) * M1 + 2 / jnp.sqrt(jnp.pi) * M3
        psi = 0.979895023 + (-0.962846325 + (0.532770573 - 0.122727278 * a) * a) * a
        return psi * (M0 + (M1 + (M2 + (M3 + M4 * a) * a) * a) * a)
    
    # Case 4: α > 1.4 or (α > 0.2 and α + v > 3.2)
    def case4():
        r2 = v2 / (a * a)
        alpha_invu = 1 / jnp.sqrt(2) / ((r2 + 1) * a)
        alpha2_invu2 = alpha_invu * alpha_invu
        return jnp.sqrt(2 / jnp.pi) * alpha_invu * (1 + (3 * r2 - 1 + ((r2 - 2) * 15 * r2 + 2) * alpha2_invu2) * alpha2_invu2)
    
    # Apply conditions exactly as in Korg.jl
    condition1 = (a <= 0.2) & (v >= 5)
    condition2 = (a <= 0.2) & (v < 5)
    condition3 = (a <= 1.4) & (a + v < 3.2) & ~condition1 & ~condition2
    
    return jnp.where(condition1, case1(),
                     jnp.where(condition2, case2(),
                               jnp.where(condition3, case3(), case4())))


@jit
def voigt_profile(x, sigma, gamma):
    """
    Calculate Voigt profile using Voigt-Hjerting function
    
    Parameters:
    - x: Distance from line center in units of Doppler width
    - sigma: Gaussian component (thermal broadening)
    - gamma: Lorentzian component (natural + collisional broadening)
    
    Returns:
    - Voigt profile value
    """
    # Voigt parameters
    a = gamma / (sigma * jnp.sqrt(2))
    v = jnp.abs(x) / (sigma * jnp.sqrt(2))
    
    # Call Voigt-Hjerting function
    return voigt_hjerting(a, v)


@jit
def thermal_doppler_width(wavelength, temperature, atomic_mass):
    """
    Calculate thermal Doppler broadening width
    
    Parameters:
    - wavelength: Line wavelength in Angstroms
    - temperature: Temperature in K
    - atomic_mass: Atomic mass in amu
    
    Returns:
    - Doppler width in Angstroms
    """
    # Convert wavelength to cm
    wavelength_cm = wavelength * 1e-8
    
    # Thermal velocity
    v_thermal = jnp.sqrt(2 * BOLTZMANN_K * temperature / (atomic_mass * 1.66054e-24))  # amu to g
    
    # Doppler width
    delta_lambda = wavelength_cm * v_thermal / SPEED_OF_LIGHT
    
    # Convert back to Angstroms
    return delta_lambda * 1e8


@jit
def natural_broadening_gamma(gf_value, wavelength):
    """
    Calculate natural broadening parameter
    
    Parameters:
    - gf_value: Oscillator strength (linear, not log)
    - wavelength: Wavelength in Angstroms
    
    Returns:
    - Natural broadening gamma in s^-1
    """
    # Convert to CGS units
    wavelength_cm = wavelength * 1e-8
    
    # Natural broadening (Einstein A coefficient)
    gamma_nat = (8 * PI**2 * ELEMENTARY_CHARGE**2) / (ELECTRON_MASS * SPEED_OF_LIGHT**3) * \
                (1 / wavelength_cm**2) * gf_value
    
    return gamma_nat


@jit
def van_der_waals_broadening(log_gamma_vdw, temperature, hydrogen_density):
    """
    Calculate van der Waals broadening
    
    Parameters:
    - log_gamma_vdw: Log of van der Waals broadening parameter
    - temperature: Temperature in K
    - hydrogen_density: Hydrogen density in cm^-3
    
    Returns:
    - Van der Waals broadening gamma in s^-1
    """
    # Reference values
    T_ref = 10000.0  # K
    
    # Temperature dependence
    temp_factor = (temperature / T_ref)**0.3
    
    # Van der Waals broadening
    gamma_vdw = 10**log_gamma_vdw * hydrogen_density * temp_factor
    
    return gamma_vdw


@jit
def saha_equation(temperature, electron_density, ionization_energy):
    """
    Saha equation for ionization equilibrium
    
    Parameters:
    - temperature: Temperature in K
    - electron_density: Electron density in cm^-3
    - ionization_energy: Ionization energy in eV
    
    Returns:
    - Ratio of ionized to neutral population
    """
    # Saha constant
    saha_const = 2.41e15  # cm^-3 K^-3/2
    
    # Partition function ratio (simplified)
    partition_ratio = 2.0  # Assume ratio of partition functions
    
    # Exponential factor
    exp_factor = jnp.exp(-ionization_energy * ELEMENTARY_CHARGE / (BOLTZMANN_K * temperature))
    
    # Saha equation
    ratio = (saha_const * partition_ratio * temperature**1.5 / electron_density) * exp_factor
    
    return ratio


@jit
def boltzmann_factor(excitation_energy, temperature):
    """
    Boltzmann factor for level population
    
    Parameters:
    - excitation_energy: Excitation energy in eV
    - temperature: Temperature in K
    
    Returns:
    - Boltzmann factor
    """
    # Convert eV to erg: 1 eV = 1.602176634e-12 erg
    excitation_erg = excitation_energy * 1.602176634e-12
    return jnp.exp(-excitation_erg / (BOLTZMANN_K * temperature))


def calculate_line_opacity_korg_method(wavelengths, line_wavelength, excitation_potential, log_gf, 
                                      temperature, electron_density, hydrogen_density, abundance,
                                      atomic_mass=None, gamma_rad=6.16e7, gamma_stark=0.0, 
                                      log_gamma_vdw=-7.5, vald_vdw_param=None, microturbulence=0.0, partition_function=None):
    """
    Calculate line opacity using Korg.jl's exact formulation
    
    This function automatically chooses between ABO and standard vdW calculations
    based on the vald_vdw_param value.
    """
    # Pre-process vdW parameter to choose calculation method
    if vald_vdw_param is not None and vald_vdw_param >= 20.0:
        # Use ABO calculation
        return _calculate_line_opacity_abo(
            wavelengths, line_wavelength, excitation_potential, log_gf,
            temperature, electron_density, hydrogen_density, abundance,
            atomic_mass, gamma_rad, gamma_stark, vald_vdw_param, microturbulence, partition_function
        )
    else:
        # Use standard calculation
        return _calculate_line_opacity_standard(
            wavelengths, line_wavelength, excitation_potential, log_gf,
            temperature, electron_density, hydrogen_density, abundance,
            atomic_mass, gamma_rad, gamma_stark, log_gamma_vdw, microturbulence, partition_function
        )


@jit
def _calculate_line_opacity_standard(wavelengths, line_wavelength, excitation_potential, log_gf, 
                                    temperature, electron_density, hydrogen_density, abundance,
                                    atomic_mass, gamma_rad, gamma_stark, log_gamma_vdw, microturbulence, partition_function):
    """
    Calculate line opacity using standard van der Waals broadening
    
    Parameters:
    - wavelengths: Array of wavelengths in Angstroms
    - line_wavelength: Line center wavelength in Angstroms
    - excitation_potential: Lower level excitation energy in eV
    - log_gf: Log of oscillator strength
    - temperature: Temperature in K
    - electron_density: Electron density in cm^-3
    - hydrogen_density: Hydrogen density in cm^-3
    - abundance: Element abundance relative to hydrogen
    - atomic_mass: Atomic mass in amu
    - gamma_rad: Radiative broadening parameter in s^-1
    - gamma_stark: Stark broadening parameter in s^-1
    - log_gamma_vdw: Log van der Waals broadening parameter
    - microturbulence: Microturbulent velocity in km/s
    - partition_function: Exact partition function value (if None, uses approximation)
    
    Returns:
    - Opacity array at each wavelength in cm^-1
    """
    # Convert wavelengths to cm (Korg internal units)
    wl_cm = wavelengths * 1e-8
    line_wl_cm = line_wavelength * 1e-8
    
    # Convert log_gf to linear gf
    gf = 10**log_gf
    
    # Estimate atomic mass if not provided
    if atomic_mass is None:
        atomic_mass = 23.0  # Default to sodium
    
    # Convert atomic mass to grams
    atomic_mass_g = atomic_mass * 1.66054e-24
    
    # Convert microturbulence to cm/s
    xi_cms = microturbulence * 1e5
    
    # === STEP 1: Cross-section factor (sigma_line in Korg) ===
    # Korg's sigma_line = (π e² / mₑ c) × (λ² / c) exactly as implemented
    cross_section_factor = (PI * ELEMENTARY_CHARGE**2) / (ELECTRON_MASS * SPEED_OF_LIGHT) * \
                          (line_wl_cm**2 / SPEED_OF_LIGHT)
    
    # === STEP 2: Level population factor ===
    # E_upper = E_lower + hc/λ
    E_upper_eV = excitation_potential + (PLANCK_H * SPEED_OF_LIGHT) / (line_wl_cm * 1.602176634e-12)
    
    # β = 1 / (k T) in eV^-1
    beta_eV = 1.0 / (8.617333262145e-5 * temperature)  # kboltz_eV from Korg
    
    # Level factor: exp(-β*E_lower) - exp(-β*E_upper)
    levels_factor = jnp.exp(-beta_eV * excitation_potential) - jnp.exp(-beta_eV * E_upper_eV)
    
    # === STEP 3: Number density factor ===
    # Use provided partition function or fall back to simplified approximation
    U_value = jnp.where(
        partition_function is None,
        25.0 * (temperature / 5778.0)**0.3,  # Simplified Fe I approximation
        partition_function  # Exact value
    )
    
    n_div_U = abundance * hydrogen_density / U_value
    
    # === STEP 4: Line amplitude ===
    amplitude = gf * cross_section_factor * levels_factor * n_div_U
    
    # === STEP 5: Broadening parameters ===
    # Doppler width: σ = λ₀ * sqrt(kT/m + ξ²/2) / c
    thermal_velocity_sq = BOLTZMANN_K * temperature / atomic_mass_g
    total_velocity_sq = thermal_velocity_sq + (xi_cms**2) / 2
    # CRITICAL FIX: Calculate Doppler width in cm (like Korg.jl), not Angstroms
    doppler_width_cm = line_wl_cm * jnp.sqrt(total_velocity_sq) / SPEED_OF_LIGHT
    
    # Van der Waals broadening - standard log(γ_vdW) format
    gamma_vdw = 10**log_gamma_vdw * hydrogen_density * (temperature / 10000.0)**0.3
    
    # Total Lorentz width in frequency units
    gamma_total_freq = gamma_rad + electron_density * gamma_stark + gamma_vdw
    
    # Convert to wavelength HWHM: γ = Γ * λ²/(4πc)
    gamma_wavelength = gamma_total_freq * line_wl_cm**2 / (4 * PI * SPEED_OF_LIGHT)
    
    # === STEP 6: Voigt profile parameters ===
    # Normalized parameters for Voigt function (using cm units like Korg.jl)
    inv_sigma_sqrt2 = 1.0 / (doppler_width_cm * jnp.sqrt(2))
    
    # Calculate profile for each wavelength
    def calculate_single_opacity(wl):
        # Calculate Voigt parameters in cm (matching Korg.jl exactly)
        v = jnp.abs(wl - line_wl_cm) * inv_sigma_sqrt2  # frequency parameter
        a = gamma_wavelength * inv_sigma_sqrt2           # damping parameter
        
        # Use proper Voigt-Hjerting function (matches Korg exactly)
        voigt_value = voigt_hjerting(a, v)
        
        # Apply scaling factor exactly like Korg.jl
        # inv_sigma_sqrt2 is already in cm⁻¹, so use directly
        scaling = inv_sigma_sqrt2 / jnp.sqrt(PI) * amplitude
        
        return voigt_value * scaling
    
    # Vectorized calculation
    opacity_cm = jax.vmap(calculate_single_opacity)(wl_cm)
    
    return opacity_cm


@jit
def _calculate_line_opacity_abo(wavelengths, line_wavelength, excitation_potential, log_gf, 
                               temperature, electron_density, hydrogen_density, abundance,
                               atomic_mass, gamma_rad, gamma_stark, vald_vdw_param, microturbulence, partition_function):
    """
    Calculate line opacity using ABO theory for van der Waals broadening
    
    Parameters:
    - vald_vdw_param: VALD van der Waals parameter (>= 20, treated as packed ABO format)
    - Other parameters same as standard calculation
    
    Returns:
    - Opacity array at each wavelength in cm^-1
    """
    # Convert wavelengths to cm (Korg internal units)
    wl_cm = wavelengths * 1e-8
    line_wl_cm = line_wavelength * 1e-8
    
    # Convert log_gf to linear gf
    gf = 10**log_gf
    
    # Estimate atomic mass if not provided
    if atomic_mass is None:
        atomic_mass = 23.0  # Default to sodium
    
    # Convert atomic mass to grams
    atomic_mass_g = atomic_mass * 1.66054e-24
    
    # Convert microturbulence to cm/s
    xi_cms = microturbulence * 1e5
    
    # === STEP 1: Cross-section factor (sigma_line in Korg) ===
    # Korg's sigma_line = (π e² / mₑ c) × (λ² / c) exactly as implemented
    cross_section_factor = (PI * ELEMENTARY_CHARGE**2) / (ELECTRON_MASS * SPEED_OF_LIGHT) * \
                          (line_wl_cm**2 / SPEED_OF_LIGHT)
    
    # === STEP 2: Level population factor ===
    # E_upper = E_lower + hc/λ
    E_upper_eV = excitation_potential + (PLANCK_H * SPEED_OF_LIGHT) / (line_wl_cm * 1.602176634e-12)
    
    # β = 1 / (k T) in eV^-1
    beta_eV = 1.0 / (8.617333262145e-5 * temperature)  # kboltz_eV from Korg
    
    # Level factor: exp(-β*E_lower) - exp(-β*E_upper)
    levels_factor = jnp.exp(-beta_eV * excitation_potential) - jnp.exp(-beta_eV * E_upper_eV)
    
    # === STEP 3: Number density factor ===
    # Use provided partition function or fall back to simplified approximation
    U_value = jnp.where(
        partition_function is None,
        25.0 * (temperature / 5778.0)**0.3,  # Simplified Fe I approximation
        partition_function  # Exact value
    )
    
    n_div_U = abundance * hydrogen_density / U_value
    
    # === STEP 4: Line amplitude ===
    amplitude = gf * cross_section_factor * levels_factor * n_div_U
    
    # === STEP 5: Broadening parameters ===
    # Doppler width: σ = λ₀ * sqrt(kT/m + ξ²/2) / c
    thermal_velocity_sq = BOLTZMANN_K * temperature / atomic_mass_g
    total_velocity_sq = thermal_velocity_sq + (xi_cms**2) / 2
    # CRITICAL FIX: Calculate Doppler width in cm (like Korg.jl), not Angstroms
    doppler_width_cm = line_wl_cm * jnp.sqrt(total_velocity_sq) / SPEED_OF_LIGHT
    
    # Van der Waals broadening - ABO format
    # CRITICAL FIX: Use exact Korg.jl ABO parameter unpacking
    # From Korg.jl linelist.jl line 89:
    # vdW = (floor(vdW) * bohr_radius_cgs * bohr_radius_cgs, vdW - floor(vdW))
    bohr_radius_cgs = 5.29177210903e-9  # cm (exact Korg constant)
    sigma_abo = jnp.floor(vald_vdw_param) * bohr_radius_cgs**2  # cm² (exact Korg unpacking)
    alpha_abo = vald_vdw_param - jnp.floor(vald_vdw_param)    # fractional part (exact Korg unpacking)
    
    # ABO theory calculation (exact Korg.jl implementation from line_absorption.jl lines 196-203)
    v0 = 1e6  # cm/s (σ is given at 10,000 m/s = 10^6 cm/s)
    amu_cgs = 1.66054e-24  # g (exact Korg constant)
    
    # Inverse reduced mass (exact Korg.jl formula)
    invμ = 1.0 / (1.008 * amu_cgs) + 1.0 / atomic_mass_g  # inverse reduced mass
    
    # Relative velocity (exact Korg.jl formula)
    vbar = jnp.sqrt(8 * BOLTZMANN_K * temperature / PI * invμ)  # relative velocity
    
    # ABO formula (exact Korg.jl formula from line_absorption.jl line 203)
    # γ = 2 * (4/π)^(α/2) * Γ((4-α)/2) * v₀ * σ * (vbar/v₀)^(1-α)
    from jax.scipy.special import gamma as gamma_func
    gamma_abo_raw = (2 * (4 / PI)**(alpha_abo / 2) * gamma_func((4 - alpha_abo) / 2) * 
                    v0 * sigma_abo * (vbar / v0)**(1 - alpha_abo))
    
    # Scale by hydrogen density (per-unit-density → actual density)
    gamma_vdw = gamma_abo_raw * hydrogen_density
    
    # Total Lorentz width in frequency units
    gamma_total_freq = gamma_rad + electron_density * gamma_stark + gamma_vdw
    
    # Convert to wavelength HWHM: γ = Γ * λ²/(4πc)
    gamma_wavelength = gamma_total_freq * line_wl_cm**2 / (4 * PI * SPEED_OF_LIGHT)
    
    # === STEP 6: Voigt profile parameters ===
    # Normalized parameters for Voigt function (using cm units like Korg.jl)
    inv_sigma_sqrt2 = 1.0 / (doppler_width_cm * jnp.sqrt(2))
    
    # Calculate profile for each wavelength
    def calculate_single_opacity(wl):
        # Calculate Voigt parameters in cm (matching Korg.jl exactly)
        v = jnp.abs(wl - line_wl_cm) * inv_sigma_sqrt2  # frequency parameter
        a = gamma_wavelength * inv_sigma_sqrt2           # damping parameter
        
        # Use proper Voigt-Hjerting function (matches Korg exactly)
        voigt_value = voigt_hjerting(a, v)
        
        # Apply scaling factor exactly like Korg.jl
        # inv_sigma_sqrt2 is already in cm⁻¹, so use directly
        scaling = inv_sigma_sqrt2 / jnp.sqrt(PI) * amplitude
        
        return voigt_value * scaling
    
    # Vectorized calculation
    opacity_cm = jax.vmap(calculate_single_opacity)(wl_cm)
    
    return opacity_cm


@jit
def calculate_linelist_opacity(wavelengths, lines_data, temperature, electron_density, 
                              hydrogen_density, abundances):
    """
    Calculate total opacity from a list of lines
    
    Parameters:
    - wavelengths: Array of wavelengths in Angstroms
    - lines_data: List of line dictionaries with keys:
        - wavelength, species, excitation_potential, log_gf
    - temperature: Temperature in K
    - electron_density: Electron density in cm^-3
    - hydrogen_density: Hydrogen density in cm^-3
    - abundances: Dictionary of element abundances {element_id: abundance}
    
    Returns:
    - Total opacity array
    """
    total_opacity = jnp.zeros_like(wavelengths)
    
    for line in lines_data:
        # Extract line parameters
        line_wavelength = line['wavelength']
        species_id = line['species']
        element_id = species_id // 100
        
        excitation_potential = line['excitation_potential']
        log_gf = line['log_gf']
        
        # Get abundance
        if element_id in abundances:
            abundance = abundances[element_id]
            
            # Calculate line opacity
            line_opacity = calculate_line_opacity(
                wavelengths, line_wavelength, excitation_potential, log_gf,
                temperature, electron_density, hydrogen_density, abundance
            )
            
            total_opacity += line_opacity
    
    return total_opacity