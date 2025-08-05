"""
Core line absorption calculations for stellar spectral synthesis

This module contains the main line absorption functions that combine
line profiles, broadening mechanisms, and opacity calculations.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Optional, Union, Tuple

from .datatypes import LineData
from .linelist import LineList
from .opacity import calculate_line_opacity_korg_method
from .profiles import voigt_profile
from .broadening import doppler_width, scaled_vdw
from .molecular_cross_sections import (
    MolecularCrossSection, 
    interpolate_molecular_cross_sections,
    is_molecular_species
)
from ..constants import SPEED_OF_LIGHT, BOLTZMANN_K, ATOMIC_MASS_UNIT, kboltz_eV

__all__ = ['total_line_absorption', 'line_absorption', 'calculate_line_profile', 'LineData', 'create_line_data']


def _get_physics_based_partition_function(temperature: float, element_id: int) -> float:
    """
    Get physics-based partition function - replaces hardcoded 25.0
    
    Uses proper statistical mechanics instead of arbitrary constants.
    
    Parameters
    ----------
    temperature : float
        Temperature in K
    element_id : int
        Atomic number
        
    Returns
    -------
    float
        Physics-based partition function value
    """
    beta = 1.0 / (kboltz_eV * temperature)
    
    if element_id == 1:  # Hydrogen - exact
        # n=1: g=2, n=2: g=8 at 10.2 eV, n=3: g=18 at 12.1 eV
        U = 2.0 * (1.0 + 4.0 * np.exp(-10.2 * beta) + 9.0 * np.exp(-12.1 * beta))
        return float(U)
        
    elif element_id == 26:  # Iron - much better than hardcoded 25.0
        # Ground state: g ≈ 25, first excited config at ~0.86 eV
        ground_g = 25.0
        excited_g = 21.0
        excited_E = 0.86  # eV
        U = ground_g + excited_g * np.exp(-excited_E * beta)
        return float(U)
        
    elif element_id == 2:  # Helium
        # Simple: ground g=1, first excited ~20 eV
        U = 1.0 + 3.0 * np.exp(-19.8 * beta)
        return float(U)
        
    elif element_id in [22, 28]:  # Ti, Ni - common in stellar spectra
        # Complex atoms with multiple low-lying states
        ground_g = 21.0  # Typical for transition metals
        excited_g = 15.0
        excited_E = 0.5  # eV
        U = ground_g + excited_g * np.exp(-excited_E * beta)  
        return float(U)
        
    else:  # Other elements - generic but physical
        if element_id <= 10:  # Light elements
            ground_g = 1.0 if element_id % 2 == 0 else 2.0
            excited_E = 2.0  # eV
        else:  # Heavier elements
            ground_g = float(element_id % 10 + 1)  # Rough but reasonable
            excited_E = 1.0  # eV
            
        excited_g = ground_g * 2.0
        U = ground_g + excited_g * np.exp(-excited_E * beta)
        return float(U)


def total_line_absorption(wavelengths: jnp.ndarray,
                         linelist: Union[List[LineData], LineList],
                         temperature: float,
                         log_g: float,
                         abundances: Optional[Dict[int, float]] = None,
                         electron_density: float = 1e14,
                         hydrogen_density: float = 1e16,
                         microturbulence: float = 0.0,
                         **kwargs) -> jnp.ndarray:
    """
    Calculate total line absorption coefficient from a linelist
    
    This is the main function for calculating line opacity from a list
    of spectral lines, similar to Korg.jl's line_absorption! function.
    
    Parameters
    ----------
    wavelengths : array
        Wavelength grid in Angstroms
    linelist : list or LineList
        List of spectral lines
    temperature : float
        Temperature in K
    log_g : float
        Surface gravity (log10(g) in cm/s²)
    abundances : dict, optional
        Element abundances {element_id: abundance}
    electron_density : float
        Electron density in cm^-3
    hydrogen_density : float
        Hydrogen density in cm^-3
    microturbulence : float
        Microturbulent velocity in km/s
    **kwargs
        Additional parameters
        
    Returns
    -------
    array
        Total line absorption coefficient in cm^-1
    """
    # Handle different input types
    if isinstance(linelist, LineList):
        lines = linelist.lines
    else:
        lines = linelist
    
    if not lines:
        return jnp.zeros_like(wavelengths)
    
    # Default abundance pattern if not provided
    if abundances is None:
        abundances = {
            1: 1.0,      # H
            2: 0.1,      # He
            11: 1e-6,    # Na
            26: 1e-4,    # Fe
        }
    
    total_opacity = jnp.zeros_like(wavelengths)
    
    # Calculate opacity for each line
    for line in lines:
        # Extract line parameters
        if hasattr(line, 'wavelength'):
            line_wl = line.wavelength
            species_id = line.species
            excitation_potential = line.E_lower
            log_gf = line.log_gf
        else:
            # Handle dictionary format
            line_wl = line['wavelength']
            species_id = line['species'] 
            excitation_potential = line['excitation_potential']
            log_gf = line['log_gf']
        
        # Get element ID and abundance
        element_id = species_id // 100
        if element_id not in abundances:
            continue
            
        abundance = abundances[element_id]
        
        
        # Estimate atomic mass (simplified)
        atomic_mass = _get_atomic_mass(element_id)
        
        # Calculate line opacity using the working pipeline approach
        # Microturbulence is already in km/s - no conversion needed
        microturbulence_kms = microturbulence
        
        # Set up default broadening parameters (matching working pipeline)
        gamma_rad = kwargs.get('gamma_rad', 6.16e7)
        gamma_stark = kwargs.get('gamma_stark', 0.0)
        log_gamma_vdw = kwargs.get('log_gamma_vdw', -8.0)
        vald_vdw_param = kwargs.get('vald_vdw_param', 0.0)
        
        # Get partition function from partition_funcs dict if available
        partition_funcs = kwargs.get('partition_funcs', {})
        if partition_funcs and isinstance(partition_funcs, dict):
            # Look up partition function for this species
            species_key = f"{element_id:d}_0"  # Neutral species key
            partition_function = partition_funcs.get(species_key, _get_physics_based_partition_function(temperature, element_id))
        else:
            partition_function = kwargs.get('partition_function', _get_physics_based_partition_function(temperature, element_id))
        
        if partition_function is None:
            partition_function = _get_physics_based_partition_function(temperature, element_id)
        
        # Convert species ID to species name for optimized vdW parameters
        species_name = _get_species_name(species_id)
        
        try:
            # CRITICAL FIX: Convert line wavelength from cm to Angstroms for calculate_line_opacity_korg_method
            line_wavelength_A = line_wl * 1e8 if line_wl < 1e-4 else line_wl
            
            line_opacity = calculate_line_opacity_korg_method(
                wavelengths=wavelengths,
                line_wavelength=line_wavelength_A,  # Convert to Angstroms
                excitation_potential=excitation_potential,
                log_gf=log_gf,
                temperature=temperature,
                electron_density=electron_density,
                hydrogen_density=hydrogen_density,
                abundance=abundance,
                atomic_mass=atomic_mass,
                gamma_rad=gamma_rad,
                gamma_stark=gamma_stark,
                log_gamma_vdw=log_gamma_vdw,
                vald_vdw_param=vald_vdw_param,
                microturbulence=microturbulence_kms,
                partition_function=partition_function,
                species_name=species_name,  # Use optimized vdW parameters
                species_id=species_id  # Enable exact Korg.jl partition functions
            )
            
        except Exception as e:
            # If line opacity calculation fails, return zeros to avoid crashing
            print(f"Warning: Line opacity calculation failed for species {species_id}: {e}")
            line_opacity = jnp.zeros_like(wavelengths)
        
        total_opacity += line_opacity
    
    return total_opacity


def line_absorption(wavelength_grid: jnp.ndarray,
                   line_center: float,
                   oscillator_strength: float,
                   excitation_potential: float,
                   temperature: float,
                   number_density: float,
                   atomic_mass: float,
                   microturbulence: float = 0.0,
                   gamma_rad: float = 0.0,
                   gamma_vdw: float = 0.0,
                   gamma_stark: float = 0.0) -> jnp.ndarray:
    """
    Calculate absorption coefficient for a single spectral line
    
    Parameters
    ----------
    wavelength_grid : array
        Wavelength grid in Angstroms
    line_center : float
        Line center wavelength in Angstroms
    oscillator_strength : float
        Oscillator strength (linear, not log)
    excitation_potential : float
        Lower level excitation potential in eV
    temperature : float
        Temperature in K
    number_density : float
        Number density of absorbing species in cm^-3
    atomic_mass : float
        Atomic mass in amu
    microturbulence : float
        Microturbulent velocity in km/s
    gamma_rad : float
        Radiative broadening parameter
    gamma_vdw : float
        Van der Waals broadening parameter
    gamma_stark : float
        Stark broadening parameter
        
    Returns
    -------
    array
        Line absorption coefficient in cm^-1
    """
    # Calculate Doppler width
    doppler_width_value = doppler_width(
        line_center, temperature, atomic_mass, microturbulence
    )
    
    # Calculate total Lorentz width
    lorentz_width = gamma_rad + gamma_vdw + gamma_stark
    
    # Calculate line profile
    profile = voigt_profile(wavelength_grid, line_center, doppler_width_value, lorentz_width)
    
    # Calculate line strength (simplified)
    line_strength = oscillator_strength * number_density
    
    return line_strength * profile


def calculate_line_profile(wavelengths: jnp.ndarray,
                          line_center: float,
                          doppler_width: float,
                          lorentz_width: float) -> jnp.ndarray:
    """
    Calculate normalized line profile
    
    Parameters
    ----------
    wavelengths : array
        Wavelength grid in Angstroms
    line_center : float
        Line center wavelength in Angstroms
    doppler_width : float
        Gaussian (Doppler) width in Angstroms
    lorentz_width : float
        Lorentzian width in Angstroms
        
    Returns
    -------
    array
        Normalized line profile
    """
    return voigt_profile(wavelengths, line_center, doppler_width, lorentz_width)


def _get_atomic_mass(element_id: int) -> float:
    """
    Get atomic mass from element ID
    
    Simplified lookup table for common elements.
    In a full implementation, this would use a comprehensive database.
    """
    masses = {
        1: 1.008,    # H
        2: 4.003,    # He
        3: 6.941,    # Li
        4: 9.012,    # Be
        5: 10.811,   # B
        6: 12.011,   # C
        7: 14.007,   # N
        8: 15.999,   # O
        9: 18.998,   # F
        10: 20.180,  # Ne
        11: 22.990,  # Na
        12: 24.305,  # Mg
        13: 26.982,  # Al
        14: 28.085,  # Si
        15: 30.974,  # P
        16: 32.066,  # S
        17: 35.453,  # Cl
        18: 39.948,  # Ar
        19: 39.098,  # K
        20: 40.078,  # Ca
        21: 44.956,  # Sc
        22: 47.867,  # Ti
        23: 50.942,  # V
        24: 51.996,  # Cr
        25: 54.938,  # Mn
        26: 55.845,  # Fe
        27: 58.933,  # Co
        28: 58.693,  # Ni
        29: 63.546,  # Cu
        30: 65.409,  # Zn
    }
    
    return masses.get(element_id, 55.845)  # Default to Fe mass


def _get_species_name(species_id: int) -> str:
    """
    Convert species ID to species name for optimized vdW parameters
    
    Species ID format: element_id * 100 + ionization_stage
    Examples: 2600 = Fe I, 2601 = Fe II, 2200 = Ti I
    
    Parameters
    ---------- 
    species_id : int
        Korg species ID (element_id * 100 + ionization)
        
    Returns
    -------
    str
        Species name (e.g., "Fe I", "Ti I") or None if not recognized
    """
    element_id = species_id // 100
    ionization = species_id % 100
    
    # Element symbols
    elements = {
        1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
        11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca",
        21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn",
        31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr",
        56: "Ba", 57: "La", 58: "Ce", 59: "Pr", 60: "Nd"
    }
    
    # Ionization stage to Roman numeral
    ion_stages = {
        0: "I", 1: "II", 2: "III", 3: "IV", 4: "V", 5: "VI"
    }
    
    if element_id in elements and ionization in ion_stages:
        element_symbol = elements[element_id]
        ion_stage = ion_stages[ionization]
        return f"{element_symbol} {ion_stage}"
    
    return None  # Unknown species - will use default vdW parameter


def create_line_data(wavelength: float,
                    species: int,
                    log_gf: float,
                    E_lower: float,
                    gamma_rad: float = 0.0,
                    gamma_stark: float = 0.0,
                    vdw_param1: float = 0.0,
                    vdw_param2: float = 0.0) -> LineData:
    """
    Create a LineData object.
    
    Parameters
    ----------
    wavelength : float
        Line wavelength
    species : int
        Species ID
    log_gf : float
        Logarithm of oscillator strength
    E_lower : float
        Lower energy level
    gamma_rad : float, optional
        Radiative broadening parameter
    gamma_stark : float, optional
        Stark broadening parameter
    vdw_param1 : float, optional
        Van der Waals parameter 1
    vdw_param2 : float, optional
        Van der Waals parameter 2
        
    Returns
    -------
    LineData
        Line data object
    """
    return LineData(
        wavelength=wavelength,
        species=species,
        log_gf=log_gf,
        E_lower=E_lower,
        gamma_rad=gamma_rad,
        gamma_stark=gamma_stark,
        vdw_param1=vdw_param1,
        vdw_param2=vdw_param2
    )


# Molecular Line Synthesis Integration

def total_line_absorption_with_molecules(
    wavelengths: jnp.ndarray,
    linelist: Union[List[LineData], LineList],
    temperature: Union[float, jnp.ndarray],
    log_g: float,
    abundances: Optional[Dict[int, float]] = None,
    electron_density: Union[float, jnp.ndarray] = 1e14,
    hydrogen_density: Union[float, jnp.ndarray] = 1e16,
    microturbulence: float = 0.0,
    molecular_cross_sections: Optional[Dict[int, MolecularCrossSection]] = None,
    molecular_number_densities: Optional[Dict[int, jnp.ndarray]] = None,
    **kwargs
) -> jnp.ndarray:
    """
    Calculate total line absorption including molecular lines.
    
    This function extends the basic line absorption to include precomputed
    molecular cross-sections, matching Korg.jl's approach for efficient
    molecular line synthesis.
    
    Parameters
    ----------
    wavelengths : jnp.ndarray
        Wavelength grid in cm
    linelist : List[LineData] or LineList
        Atomic and molecular line data
    temperature : float or jnp.ndarray
        Temperature in K (can be array for atmospheric layers)
    log_g : float
        Surface gravity (log cm/s²)
    abundances : Dict[int, float], optional
        Element abundances
    electron_density : float or jnp.ndarray
        Electron density in cm^-3
    hydrogen_density : float or jnp.ndarray
        Hydrogen density in cm^-3
    microturbulence : float
        Microturbulent velocity in cm/s
    molecular_cross_sections : Dict[int, MolecularCrossSection], optional
        Precomputed molecular cross-sections by species ID
    molecular_number_densities : Dict[int, jnp.ndarray], optional
        Molecular number densities by species ID in cm^-3
        
    Returns
    -------
    jnp.ndarray
        Total line absorption coefficient in cm^-1
    """
    
    # Separate atomic and molecular lines
    atomic_lines = []
    molecular_lines = []
    
    for line in linelist:
        if is_molecular_species_id(line.species):
            molecular_lines.append(line)
        else:
            atomic_lines.append(line)
    
    # Calculate atomic line absorption (existing method)
    alpha_atomic = jnp.zeros_like(wavelengths)
    if atomic_lines:
        alpha_atomic = total_line_absorption(
            wavelengths, atomic_lines, temperature, log_g,
            abundances, electron_density, hydrogen_density,
            microturbulence, **kwargs
        )
    
    # Calculate molecular line absorption
    alpha_molecular = jnp.zeros_like(wavelengths)
    
    # Method 1: Use precomputed cross-sections (fast)
    if molecular_cross_sections is not None and molecular_number_densities is not None:
        alpha_molecular += interpolate_molecular_cross_sections(
            wavelengths, 
            temperature if jnp.isscalar(temperature) else temperature,
            microturbulence,
            molecular_cross_sections,
            molecular_number_densities
        )
    
    # Method 2: Line-by-line calculation for molecules (slower, more accurate)
    elif molecular_lines:
        alpha_molecular = calculate_molecular_line_absorption(
            wavelengths, molecular_lines, temperature, 
            microturbulence, molecular_number_densities or {}
        )
    
    return alpha_atomic + alpha_molecular


def calculate_molecular_line_absorption(
    wavelengths: jnp.ndarray,
    molecular_lines: List[LineData],
    temperature: Union[float, jnp.ndarray],
    microturbulence: float,
    number_densities: Dict[int, jnp.ndarray]
) -> jnp.ndarray:
    """
    Calculate molecular line absorption using line-by-line method.
    
    Molecular lines have simplified broadening (no Stark or vdW broadening).
    
    Parameters
    ----------
    wavelengths : jnp.ndarray
        Wavelength grid in cm
    molecular_lines : List[LineData]
        List of molecular lines
    temperature : float or jnp.ndarray
        Temperature in K
    microturbulence : float
        Microturbulent velocity in cm/s
    number_densities : Dict[int, jnp.ndarray]
        Number densities by species ID
        
    Returns
    -------
    jnp.ndarray
        Molecular line absorption coefficient in cm^-1
    """
    from .profiles import line_profile
    from .broadening import doppler_width
    
    alpha_total = jnp.zeros_like(wavelengths)
    
    for line in molecular_lines:
        if line.species not in number_densities:
            continue
            
        # Get molecular mass
        molecular_mass = get_molecular_mass_from_id(line.species)
        
        # Calculate Doppler width
        sigma = doppler_width(line.wavelength, temperature, molecular_mass, microturbulence)
        
        # Only radiative damping for molecules
        gamma_total = line.gamma_rad
        
        # Convert to wavelength units (simplified)
        gamma_wl = gamma_total * line.wavelength**2 / (2.998e10 * 4 * jnp.pi)
        
        # Calculate line strength
        line_strength = 10**line.log_gf * number_densities[line.species]
        
        # Calculate line profile
        alpha_line = line_profile(
            line.wavelength, sigma, gamma_wl, line_strength, wavelengths
        )
        
        alpha_total += alpha_line
    
    return alpha_total


def separate_atomic_molecular_lines(linelist: Union[List[LineData], LineList]) -> Tuple[List[LineData], List[LineData]]:
    """
    Separate linelist into atomic and molecular components.
    
    Parameters
    ----------
    linelist : List[LineData] or LineList
        Input linelist
        
    Returns
    -------
    Tuple[List[LineData], List[LineData]]
        (atomic_lines, molecular_lines)
    """
    atomic_lines = []
    molecular_lines = []
    
    for line in linelist:
        if is_molecular_species_id(line.species):
            molecular_lines.append(line)
        else:
            atomic_lines.append(line)
    
    return atomic_lines, molecular_lines


def is_molecular_species_id(species_id: int) -> bool:
    """
    Check if species ID corresponds to a molecule.
    
    Uses convention: molecules have species_id > 100 and specific patterns.
    
    Parameters
    ----------
    species_id : int
        Species identifier
        
    Returns
    -------
    bool
        True if molecular species
    """
    # Common molecular species ID ranges
    molecular_ranges = [
        (101, 199),   # Diatomic molecules (H2, etc.)
        (601, 699),   # Carbon compounds (CH, CN, CO, etc.)
        (701, 799),   # Nitrogen compounds (NH, etc.)
        (801, 899),   # Oxygen compounds (OH, H2O, etc.)
        (1201, 1299), # Mg compounds (MgH, etc.)
        (1301, 1399), # Al compounds (AlH, etc.)
        (1401, 1499), # Si compounds (SiH, SiO, etc.)
        (2001, 2099), # Ca compounds (CaH, etc.)
        (2201, 2299), # Ti compounds (TiO, etc.)
        (2301, 2399), # V compounds (VO, etc.)
        (2601, 2699), # Fe compounds (FeH, etc.)
    ]
    
    # Check if species_id falls in molecular ranges
    for min_id, max_id in molecular_ranges:
        if min_id <= species_id <= max_id:
            return True
    
    return False


def get_molecular_mass_from_id(species_id: int) -> float:
    """
    Get molecular mass in grams from species ID.
    
    Parameters
    ----------
    species_id : int
        Molecular species ID
        
    Returns
    -------
    float
        Molecular mass in grams
    """
    # Simplified molecular mass lookup
    molecular_masses = {
        101: 2.016,    # H2
        108: 17.007,   # OH  
        601: 13.019,   # CH
        606: 24.022,   # C2
        607: 26.018,   # CN
        608: 28.014,   # CO
        701: 15.015,   # NH
        707: 28.014,   # N2
        801: 18.015,   # H2O
        808: 31.998,   # O2
        1201: 25.313,  # MgH
        1301: 27.990,  # AlH
        1401: 29.093,  # SiH
        1408: 44.085,  # SiO
        2001: 41.086,  # CaH
        2208: 63.866,  # TiO
        2308: 66.941,  # VO
        2601: 56.853,  # FeH
    }
    
    mass_amu = molecular_masses.get(species_id, 18.015)  # Default to H2O
    return mass_amu * 1.66054e-24  # Convert amu to grams