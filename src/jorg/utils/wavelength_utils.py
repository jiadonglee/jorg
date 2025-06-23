"""
Wavelength conversion utilities for stellar spectroscopy

This module provides wavelength conversion functions between air and vacuum,
unit conversions, and wavelength detection utilities.
"""

import numpy as np
from typing import Union


def air_to_vacuum(wavelength: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert air wavelengths to vacuum wavelengths
    
    Uses the formula from Birch and Downs (1994) as implemented in VALD
    
    Parameters:
    -----------
    wavelength : float or array
        Wavelength in cm (air)
        
    Returns:
    --------
    float or array
        Wavelength in cm (vacuum)
    """
    
    # Convert to Angstroms for calculation
    if isinstance(wavelength, np.ndarray):
        wl_angstrom = wavelength * 1e8
    else:
        wl_angstrom = wavelength * 1e8
    
    # Birch and Downs (1994) formula
    # Valid for wavelengths > 2000 Å
    s = 1e4 / wl_angstrom  # wavenumber in μm^-1
    
    n = 1 + 0.0000834254 + 0.02406147 / (130 - s**2) + 0.00015998 / (38.9 - s**2)
    
    # Convert back to cm
    vacuum_wavelength_angstrom = wl_angstrom * n
    
    if isinstance(wavelength, np.ndarray):
        return vacuum_wavelength_angstrom * 1e-8
    else:
        return vacuum_wavelength_angstrom * 1e-8


def vacuum_to_air(wavelength: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert vacuum wavelengths to air wavelengths
    
    Uses iterative solution of the Birch and Downs formula
    
    Parameters:
    -----------
    wavelength : float or array
        Wavelength in cm (vacuum)
        
    Returns:
    --------
    float or array
        Wavelength in cm (air)
    """
    
    # Convert to Angstroms
    if isinstance(wavelength, np.ndarray):
        vacuum_angstrom = wavelength * 1e8
        air_angstrom = np.zeros_like(vacuum_angstrom)
        
        for i, wl_vac in enumerate(vacuum_angstrom):
            air_angstrom[i] = _vacuum_to_air_single(wl_vac)
            
        return air_angstrom * 1e-8
    else:
        air_angstrom = _vacuum_to_air_single(wavelength * 1e8)
        return air_angstrom * 1e-8


def _vacuum_to_air_single(vacuum_angstrom: float) -> float:
    """Convert single vacuum wavelength to air (in Angstroms)"""
    
    # Initial guess
    air_angstrom = vacuum_angstrom
    
    # Iterate to find air wavelength
    for _ in range(5):  # Usually converges in 2-3 iterations
        s = 1e4 / air_angstrom
        n = 1 + 0.0000834254 + 0.02406147 / (130 - s**2) + 0.00015998 / (38.9 - s**2)
        air_angstrom = vacuum_angstrom / n
    
    return air_angstrom


def detect_wavelength_unit(wavelengths: np.ndarray) -> str:
    """
    Detect whether wavelengths are in Angstroms or cm
    
    Parameters:
    -----------
    wavelengths : array
        Array of wavelengths
        
    Returns:
    --------
    str
        "angstrom" or "cm"
    """
    
    median_wl = np.median(wavelengths)
    
    if median_wl > 100:
        return "angstrom"
    else:
        return "cm"


def detect_air_or_vacuum(wavelengths: np.ndarray) -> str:
    """
    Attempt to detect if wavelengths are air or vacuum
    
    This is challenging and not always reliable. Uses heuristics
    based on typical differences between air and vacuum wavelengths.
    
    Parameters:
    -----------
    wavelengths : array
        Array of wavelengths
        
    Returns:
    --------
    str
        "air" or "vacuum" (best guess)
    """
    
    # Convert to Angstroms if needed
    if detect_wavelength_unit(wavelengths) == "cm":
        wl_angstrom = wavelengths * 1e8
    else:
        wl_angstrom = wavelengths
    
    # Check against known air/vacuum line positions
    # This is a very rough heuristic
    
    # Look for common lines where air/vacuum difference is significant
    known_air_lines = [
        5889.95,  # Na D2 (air)
        5895.92,  # Na D1 (air)
        6562.80,  # H-alpha (air)
        4861.33,  # H-beta (air)
    ]
    
    known_vacuum_lines = [
        5891.58,  # Na D2 (vacuum)
        5897.56,  # Na D1 (vacuum)
        6564.61,  # H-alpha (vacuum)
        4862.68,  # H-beta (vacuum)
    ]
    
    air_matches = 0
    vacuum_matches = 0
    
    for wl in wl_angstrom:
        # Check for close matches (within 0.5 Å)
        for air_line in known_air_lines:
            if abs(wl - air_line) < 0.5:
                air_matches += 1
                
        for vacuum_line in known_vacuum_lines:
            if abs(wl - vacuum_line) < 0.5:
                vacuum_matches += 1
    
    if air_matches > vacuum_matches:
        return "air"
    elif vacuum_matches > air_matches:
        return "vacuum"
    else:
        # Default assumption for most linelists
        return "air"


def angstrom_to_cm(wavelength: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert Angstroms to cm"""
    return wavelength * 1e-8


def cm_to_angstrom(wavelength: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert cm to Angstroms"""
    return wavelength * 1e8


def frequency_to_wavelength(frequency: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert frequency (Hz) to wavelength (cm)
    
    Parameters:
    -----------
    frequency : float or array
        Frequency in Hz
        
    Returns:
    --------
    float or array
        Wavelength in cm
    """
    c_cgs = 2.99792458e10  # Speed of light in cm/s
    return c_cgs / frequency


def wavelength_to_frequency(wavelength: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert wavelength (cm) to frequency (Hz)
    
    Parameters:
    -----------
    wavelength : float or array
        Wavelength in cm
        
    Returns:
    --------
    float or array
        Frequency in Hz
    """
    c_cgs = 2.99792458e10  # Speed of light in cm/s
    return c_cgs / wavelength


def wavelength_to_wavenumber(wavelength: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert wavelength (cm) to wavenumber (cm^-1)
    
    Parameters:
    -----------
    wavelength : float or array
        Wavelength in cm
        
    Returns:
    --------
    float or array
        Wavenumber in cm^-1
    """
    return 1.0 / wavelength


def wavenumber_to_wavelength(wavenumber: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert wavenumber (cm^-1) to wavelength (cm)
    
    Parameters:
    -----------
    wavenumber : float or array
        Wavenumber in cm^-1
        
    Returns:
    --------
    float or array
        Wavelength in cm
    """
    return 1.0 / wavenumber


def energy_to_wavelength(energy_eV: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert photon energy (eV) to wavelength (cm)
    
    Parameters:
    -----------
    energy_eV : float or array
        Photon energy in eV
        
    Returns:
    --------
    float or array
        Wavelength in cm
    """
    # hc = 1.24e-4 eV⋅cm
    return 1.24e-4 / energy_eV


def wavelength_to_energy(wavelength: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert wavelength (cm) to photon energy (eV)
    
    Parameters:
    -----------
    wavelength : float or array
        Wavelength in cm
        
    Returns:
    --------
    float or array
        Photon energy in eV
    """
    # hc = 1.24e-4 eV⋅cm
    return 1.24e-4 / wavelength


def doppler_shift(wavelength: Union[float, np.ndarray], 
                 velocity: float) -> Union[float, np.ndarray]:
    """
    Apply Doppler shift to wavelength
    
    Parameters:
    -----------
    wavelength : float or array
        Rest wavelength in cm
    velocity : float
        Velocity in cm/s (positive = redshift)
        
    Returns:
    --------
    float or array
        Shifted wavelength in cm
    """
    c_cgs = 2.99792458e10  # Speed of light in cm/s
    
    # Non-relativistic approximation for stellar velocities
    if abs(velocity) < 0.1 * c_cgs:
        return wavelength * (1 + velocity / c_cgs)
    else:
        # Relativistic formula
        beta = velocity / c_cgs
        gamma = 1 / np.sqrt(1 - beta**2)
        return wavelength * gamma * (1 + beta)


def redshift_to_velocity(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert redshift to velocity
    
    Parameters:
    -----------
    z : float or array
        Redshift (dimensionless)
        
    Returns:
    --------
    float or array
        Velocity in cm/s
    """
    c_cgs = 2.99792458e10  # Speed of light in cm/s
    
    # Non-relativistic approximation for small z
    if isinstance(z, np.ndarray):
        small_z = np.abs(z) < 0.1
        velocity = np.zeros_like(z)
        
        # Non-relativistic
        velocity[small_z] = z[small_z] * c_cgs
        
        # Relativistic
        large_z = ~small_z
        velocity[large_z] = c_cgs * ((1 + z[large_z])**2 - 1) / ((1 + z[large_z])**2 + 1)
        
        return velocity
    else:
        if abs(z) < 0.1:
            return z * c_cgs
        else:
            return c_cgs * ((1 + z)**2 - 1) / ((1 + z)**2 + 1)


def velocity_to_redshift(velocity: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert velocity to redshift
    
    Parameters:
    -----------
    velocity : float or array
        Velocity in cm/s
        
    Returns:
    --------
    float or array
        Redshift (dimensionless)
    """
    c_cgs = 2.99792458e10  # Speed of light in cm/s
    beta = velocity / c_cgs
    
    # Non-relativistic approximation for small velocities
    if isinstance(beta, np.ndarray):
        small_v = np.abs(beta) < 0.1
        redshift = np.zeros_like(beta)
        
        # Non-relativistic
        redshift[small_v] = beta[small_v]
        
        # Relativistic
        large_v = ~small_v
        redshift[large_v] = np.sqrt((1 + beta[large_v]) / (1 - beta[large_v])) - 1
        
        return redshift
    else:
        if abs(beta) < 0.1:
            return beta
        else:
            return np.sqrt((1 + beta) / (1 - beta)) - 1