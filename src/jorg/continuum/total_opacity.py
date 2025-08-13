"""
Total opacity calculation: continuum + lines

This module combines continuum and line opacity calculations to provide
the total opacity needed for radiative transfer, following Korg's approach.
"""

import jax.numpy as jnp
import numpy as np
from typing import Dict

from .complete_continuum import calculate_total_continuum_opacity
from ..lines.hydrogen_lines import hydrogen_line_opacity_simple
from ..lines.opacity import atomic_line_opacity_simple
from ..statmech.species import Species, Formula


def calculate_total_opacity(
    frequencies: np.ndarray,
    temperature: float, 
    electron_density: float,
    number_densities: Dict,
    include_lines: bool = True,
    include_hydrogen_lines: bool = True,
    line_cutoff_threshold: float = 1e-4
) -> np.ndarray:
    """
    Calculate total opacity = continuum + lines
    
    Parameters:
    -----------
    frequencies : np.ndarray
        Frequencies in Hz
    temperature : float
        Temperature in K
    electron_density : float
        Electron density in cm⁻³
    number_densities : Dict
        Dictionary mapping Species to number densities
    include_lines : bool
        Whether to include atomic/molecular line opacity
    include_hydrogen_lines : bool
        Whether to include hydrogen line opacity
    line_cutoff_threshold : float
        Cutoff threshold for weak lines
        
    Returns:
    --------
    np.ndarray
        Total opacity coefficient in cm⁻¹
    """
    
    # 1. Calculate continuum opacity
    alpha_continuum = calculate_total_continuum_opacity(
        frequencies, temperature, electron_density, number_densities
    )
    
    alpha_total = alpha_continuum.copy()
    
    if not include_lines:
        return alpha_total
    
    # 2. Add hydrogen line opacity (if requested)
    if include_hydrogen_lines:
        try:
            # Extract hydrogen densities
            h_i_density = 0.0
            he_i_density = 0.0
            
            for species, density in number_densities.items():
                if hasattr(species, 'formula') and hasattr(species, 'charge'):
                    if len(species.formula.atoms) == 1:
                        Z = species.formula.atoms[0]
                        charge = species.charge
                        
                        if Z == 1 and charge == 0:  # H I
                            h_i_density = density
                        elif Z == 2 and charge == 0:  # He I  
                            he_i_density = density
            
            if h_i_density > 0:
                # Convert frequencies to wavelengths for hydrogen line calculation
                wavelengths = 2.998e18 / frequencies  # Å
                
                # Simple hydrogen line opacity (Balmer series dominates in optical)
                alpha_hydrogen = calculate_hydrogen_line_opacity(
                    wavelengths, temperature, electron_density, h_i_density, he_i_density
                )
                
                alpha_total += alpha_hydrogen
                
        except Exception as e:
            print(f"Warning: Hydrogen line opacity failed: {e}")
    
    # 3. Add atomic/molecular line opacity (simplified)
    if include_lines:
        try:
            # Simple estimate for atomic line opacity
            # In a full implementation, this would read linelists and calculate Voigt profiles
            alpha_lines = estimate_atomic_line_opacity(
                frequencies, temperature, electron_density, number_densities,
                line_cutoff_threshold
            )
            
            alpha_total += alpha_lines
            
        except Exception as e:
            print(f"Warning: Atomic line opacity failed: {e}")
    
    return alpha_total


def calculate_hydrogen_line_opacity(
    wavelengths: np.ndarray,
    temperature: float,
    electron_density: float, 
    h_i_density: float,
    he_i_density: float
) -> np.ndarray:
    """
    Calculate hydrogen line opacity for Balmer and higher series
    
    This is a simplified implementation focusing on the strongest lines
    in the optical range (Balmer series: Hα, Hβ, Hγ, Hδ)
    """
    
    alpha_hydrogen = np.zeros_like(wavelengths)
    
    # Balmer series wavelengths (Å, air)
    balmer_wavelengths = {
        'H_alpha': 6562.8,
        'H_beta': 4861.3, 
        'H_gamma': 4340.5,
        'H_delta': 4101.7,
        'H_epsilon': 3970.1,
        'H_zeta': 3889.1
    }
    
    # Simple line opacity calculation for each Balmer line
    for line_name, line_wl in balmer_wavelengths.items():
        
        # Check if this wavelength is in our range
        if np.min(wavelengths) <= line_wl <= np.max(wavelengths):
            
            # Simple Voigt profile parameters (very approximate)
            gamma_natural = 1e8  # Natural damping (s⁻¹)
            gamma_stark = electron_density * 1e-16  # Stark damping
            gamma_total = gamma_natural + gamma_stark
            
            # Thermal Doppler width
            atomic_mass = 1.008  # Hydrogen atomic mass
            doppler_width = line_wl * np.sqrt(2 * 1.381e-16 * temperature / (atomic_mass * 1.661e-24)) / 2.998e10
            
            # Line strength (very simplified)
            line_strength = h_i_density * 1e-15  # Rough estimate
            
            # Simple Voigt profile (Gaussian approximation)
            for i, wl in enumerate(wavelengths):
                if abs(wl - line_wl) < 10.0:  # Within 10 Å of line center
                    delta_lambda = wl - line_wl
                    profile = np.exp(-(delta_lambda / doppler_width)**2) / (doppler_width * np.sqrt(np.pi))
                    alpha_hydrogen[i] += line_strength * profile
    
    return alpha_hydrogen


def estimate_atomic_line_opacity(
    frequencies: np.ndarray,
    temperature: float,
    electron_density: float,
    number_densities: Dict,
    cutoff_threshold: float
) -> np.ndarray:
    """
    Estimate atomic line opacity from metal species
    
    This is a very simplified implementation that estimates the contribution
    from strong metal lines. A full implementation would require atomic data.
    """
    
    alpha_lines = np.zeros_like(frequencies)
    
    # Estimate based on metal densities and temperature
    metal_density_total = 0.0
    
    for species, density in number_densities.items():
        if hasattr(species, 'formula') and hasattr(species, 'charge'):
            if len(species.formula.atoms) == 1:
                Z = species.formula.atoms[0]
                if Z > 2:  # Metals (Z > 2)
                    metal_density_total += density
    
    if metal_density_total > 0:
        # Very rough estimate: line opacity ~ 10^-3 × metal_density × temperature dependence
        # This is just a placeholder - real calculation requires linelists
        
        # Convert frequencies to wavelengths
        wavelengths = 2.998e18 / frequencies  # Å
        
        # Assume most lines are in optical range (4000-7000 Å)
        optical_mask = (wavelengths >= 4000) & (wavelengths <= 7000)
        
        # Simple scaling
        line_opacity_estimate = metal_density_total * 1e-18 * np.exp(-5000 / temperature)
        
        alpha_lines[optical_mask] = line_opacity_estimate
    
    return alpha_lines


if __name__ == "__main__":
    # Test the total opacity calculation
    print("Testing total opacity calculation...")
    
    # Test conditions
    frequencies = np.array([5.451e14])  # 5500 Å
    T = 4838.3  # K
    ne = 2.28e12  # cm⁻³
    
    # Create number densities
    number_densities = {}
    number_densities[Species(Formula([1]), 0)] = 2.5e16    # H I
    number_densities[Species(Formula([1]), 1)] = 6.0e10    # H II
    number_densities[Species(Formula([2]), 0)] = 2.0e15    # He I
    number_densities[Species(Formula([2]), 1)] = 1.0e11    # He II
    number_densities[Species(Formula([26]), 0)] = 9.0e10   # Fe I
    number_densities[Species(Formula([26]), 1)] = 3.0e10   # Fe II
    number_densities[Species(Formula([1, 1]), 0)] = 1.0e13  # H2
    
    # Calculate opacities
    alpha_continuum = calculate_total_continuum_opacity(frequencies, T, ne, number_densities)
    alpha_total = calculate_total_opacity(frequencies, T, ne, number_densities, 
                                        include_lines=True, include_hydrogen_lines=True)
    
    print(f"Continuum opacity: {alpha_continuum[0]:.2e} cm⁻¹")
    print(f"Total opacity: {alpha_total[0]:.2e} cm⁻¹")
    print(f"Line contribution: {(alpha_total[0] - alpha_continuum[0]):.2e} cm⁻¹")
    print(f"Line/continuum ratio: {(alpha_total[0] - alpha_continuum[0]) / alpha_continuum[0]:.2f}")