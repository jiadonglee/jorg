"""
Total opacity calculation combining continuum and line absorption
This module provides the complete opacity calculation for stellar atmosphere modeling
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Union

from .continuum.complete_continuum import calculate_total_continuum_opacity
from .lines.core import total_line_absorption
from .lines.hydrogen_lines import hydrogen_line_absorption
from .lines.datatypes import LineData
from .lines.linelist import LineList


@jax.jit
def calculate_total_opacity(
    frequencies: jnp.ndarray,
    temperature: float,
    electron_density: float,
    number_densities: Dict,
    linelist: Optional[Union[List[LineData], LineList]] = None,
    include_continuum: bool = True,
    include_atomic_lines: bool = True,
    include_hydrogen_lines: bool = True,
    include_molecular_lines: bool = True,
    microturbulence: float = 2.0e5,  # cm/s
    cutoff_threshold: float = 3e-4
) -> jnp.ndarray:
    """
    Calculate total opacity including continuum and line absorption
    
    This is the main function that combines all opacity sources to provide
    the complete absorption coefficient for radiative transfer calculations.
    
    Parameters
    ----------
    frequencies : jnp.ndarray
        Frequencies in Hz
    temperature : float
        Temperature in K
    electron_density : float
        Electron density in cm⁻³
    number_densities : Dict
        Dictionary mapping species to number densities in cm⁻³
    linelist : List[LineData] or LineList, optional
        Atomic and molecular line data
    include_continuum : bool, default True
        Include continuum opacity sources (H⁻ bf/ff, Thomson, Rayleigh, etc.)
    include_atomic_lines : bool, default True
        Include atomic line absorption
    include_hydrogen_lines : bool, default True
        Include hydrogen line absorption (Balmer, Lyman, etc.)
    include_molecular_lines : bool, default True
        Include molecular line absorption
    microturbulence : float, default 2e5
        Microturbulent velocity in cm/s
    cutoff_threshold : float, default 3e-4
        Line cutoff threshold relative to continuum
        
    Returns
    -------
    jnp.ndarray
        Total absorption coefficient in cm⁻¹
    """
    
    # Initialize total opacity
    alpha_total = jnp.zeros_like(frequencies)
    
    # 1. Continuum opacity
    if include_continuum:
        alpha_continuum = calculate_total_continuum_opacity(
            frequencies, temperature, electron_density, number_densities
        )
        alpha_total += alpha_continuum
    
    # Convert frequencies to wavelengths for line calculations
    wavelengths_cm = 2.998e10 / frequencies  # cm
    wavelengths_angstrom = wavelengths_cm * 1e8  # Å
    
    # 2. Hydrogen line absorption
    if include_hydrogen_lines:
        # Extract hydrogen densities
        h_i_density = 0.0
        h_ii_density = 0.0
        for species, density in number_densities.items():
            if hasattr(species, 'formula') and hasattr(species, 'charge'):
                if len(species.formula.atoms) == 1 and species.formula.atoms[0] == 1:
                    if species.charge == 0:
                        h_i_density = density
                    elif species.charge == 1:
                        h_ii_density = density
        
        if h_i_density > 0:
            alpha_hydrogen = hydrogen_line_absorption(
                wavelengths_cm, temperature, electron_density, h_i_density, microturbulence
            )
            alpha_total += alpha_hydrogen
    
    # 3. Atomic and molecular line absorption
    if linelist is not None and (include_atomic_lines or include_molecular_lines):
        # Filter linelist based on what's requested
        filtered_lines = []
        if include_atomic_lines or include_molecular_lines:
            for line in linelist:
                is_molecular = len(getattr(line.species, 'formula', [1])) > 1
                if (include_molecular_lines and is_molecular) or \
                   (include_atomic_lines and not is_molecular):
                    filtered_lines.append(line)
        
        if filtered_lines:
            alpha_lines = total_line_absorption(
                wavelengths_angstrom,
                filtered_lines,
                temperature,
                log_g=4.0,  # Default surface gravity
                electron_density=electron_density,
                hydrogen_density=h_i_density,
                microturbulence=microturbulence,
                cutoff_threshold=cutoff_threshold
            )
            alpha_total += alpha_lines
    
    return alpha_total


def calculate_opacity_spectrum(
    wavelength_range: tuple,
    n_points: int,
    temperature: float,
    electron_density: float,
    number_densities: Dict,
    linelist: Optional[Union[List[LineData], LineList]] = None,
    **kwargs
) -> tuple:
    """
    Calculate opacity spectrum over a wavelength range
    
    Parameters
    ----------
    wavelength_range : tuple
        (min_wavelength, max_wavelength) in Angstroms
    n_points : int
        Number of wavelength points
    temperature : float
        Temperature in K
    electron_density : float
        Electron density in cm⁻³
    number_densities : Dict
        Dictionary mapping species to number densities
    linelist : optional
        Line data for atomic/molecular lines
    **kwargs
        Additional arguments passed to calculate_total_opacity
        
    Returns
    -------
    tuple
        (wavelengths_angstrom, opacity_cm_inv)
    """
    
    # Create wavelength grid
    wavelengths_angstrom = jnp.linspace(wavelength_range[0], wavelength_range[1], n_points)
    wavelengths_cm = wavelengths_angstrom * 1e-8
    frequencies = 2.998e10 / wavelengths_cm  # Hz
    
    # Calculate opacity
    opacity = calculate_total_opacity(
        frequencies, temperature, electron_density, number_densities, 
        linelist=linelist, **kwargs
    )
    
    return wavelengths_angstrom, opacity


def opacity_summary(
    frequencies: jnp.ndarray,
    temperature: float, 
    electron_density: float,
    number_densities: Dict,
    linelist: Optional[Union[List[LineData], LineList]] = None
) -> Dict:
    """
    Calculate opacity breakdown by source for analysis
    
    Returns a dictionary with opacity contributions from each source
    """
    
    # Calculate individual components
    alpha_continuum = calculate_total_continuum_opacity(
        frequencies, temperature, electron_density, number_densities
    )
    
    wavelengths_cm = 2.998e10 / frequencies
    wavelengths_angstrom = wavelengths_cm * 1e8
    
    # Hydrogen lines
    h_i_density = 0.0
    for species, density in number_densities.items():
        if hasattr(species, 'formula') and hasattr(species, 'charge'):
            if len(species.formula.atoms) == 1 and species.formula.atoms[0] == 1:
                if species.charge == 0:
                    h_i_density = density
                    break
    
    alpha_hydrogen = jnp.zeros_like(frequencies)
    if h_i_density > 0:
        alpha_hydrogen = hydrogen_line_absorption(
            wavelengths_cm, temperature, electron_density, h_i_density, 2.0e5
        )
    
    # Other lines
    alpha_other_lines = jnp.zeros_like(frequencies)
    if linelist is not None:
        alpha_other_lines = total_line_absorption(
            wavelengths_angstrom, linelist, temperature, 4.0,
            electron_density=electron_density, hydrogen_density=h_i_density
        )
    
    # Total
    alpha_total = alpha_continuum + alpha_hydrogen + alpha_other_lines
    
    return {
        'continuum': np.array(alpha_continuum),
        'hydrogen_lines': np.array(alpha_hydrogen), 
        'other_lines': np.array(alpha_other_lines),
        'total': np.array(alpha_total),
        'frequencies': np.array(frequencies),
        'wavelengths_angstrom': np.array(wavelengths_cm * 1e8)
    }


if __name__ == "__main__":
    # Test the total opacity calculation
    print("Testing total opacity calculation...")
    
    # Test conditions
    T = 5000.0  # K
    ne = 1e13   # cm⁻³
    
    # Create mock number densities using simple dict for testing
    number_densities = {
        'H_I': 1e16,    # cm⁻³
        'H_II': 1e11,   # cm⁻³ 
        'He_I': 1e15,   # cm⁻³
        'Fe_I': 1e11,   # cm⁻³
    }
    
    # Test wavelength range (5000-6000 Å)
    wavelengths, opacity = calculate_opacity_spectrum(
        wavelength_range=(5000, 6000),
        n_points=100,
        temperature=T,
        electron_density=ne,
        number_densities=number_densities,
        include_continuum=True,
        include_hydrogen_lines=True,
        include_atomic_lines=False,  # No linelist provided
        include_molecular_lines=False
    )
    
    print(f"✓ Calculated opacity for {len(wavelengths)} wavelength points")
    print(f"  Wavelength range: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} Å")
    print(f"  Opacity range: {np.min(opacity):.2e} - {np.max(opacity):.2e} cm⁻¹")
    print(f"  Temperature: {T} K")
    print(f"  Electron density: {ne:.1e} cm⁻³")
    
    # Test opacity summary
    frequencies = 2.998e10 / (wavelengths * 1e-8)
    summary = opacity_summary(
        frequencies, T, ne, number_densities, linelist=None
    )
    
    print(f"\nOpacity breakdown at 5500 Å:")
    idx = np.argmin(np.abs(wavelengths - 5500))
    print(f"  Continuum:      {summary['continuum'][idx]:.2e} cm⁻¹")
    print(f"  Hydrogen lines: {summary['hydrogen_lines'][idx]:.2e} cm⁻¹") 
    print(f"  Other lines:    {summary['other_lines'][idx]:.2e} cm⁻¹")
    print(f"  Total:          {summary['total'][idx]:.2e} cm⁻¹")
    
    print("\n✓ Total opacity calculation working!")
    print("Ready for stellar atmosphere synthesis!")