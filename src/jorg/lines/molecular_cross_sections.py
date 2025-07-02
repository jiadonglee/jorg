"""
Molecular cross-section precomputation and interpolation for stellar spectral synthesis.

This module implements the molecular line handling system matching Korg.jl's approach,
including precomputed cross-sections for efficient synthesis of spectra with large
molecular linelists (e.g., H2O from APOGEE, TiO/VO bands, ExoMol data).
"""

import jax
import jax.numpy as jnp
import numpy as np
import h5py
from typing import Optional, Union, Dict, Tuple, List
from dataclasses import dataclass
from scipy.interpolate import RegularGridInterpolator
import os

# Removed circular import - line_absorption_coefficient not needed
from .species import Species
from .datatypes import LineData
from ..constants import c_cgs, hplanck_eV
from ..utils.wavelength_utils import air_to_vacuum, vacuum_to_air


@dataclass
class MolecularCrossSection:
    """
    Precomputed molecular absorption cross-sections on a 3D grid.
    
    This class matches Korg.jl's MolecularCrossSection struct and provides
    efficient interpolation for molecular line synthesis.
    
    Attributes
    ----------
    wavelengths : jnp.ndarray
        Wavelength grid in cm
    vmic_grid : jnp.ndarray
        Microturbulent velocity grid in cm/s
    log_temp_grid : jnp.ndarray
        Log temperature grid (log K)
    cross_sections : jnp.ndarray
        3D cross-section grid [n_vmic, n_temp, n_wavelength] in cm^2
    species : Species
        Molecular species identifier
    interpolator : RegularGridInterpolator
        Scipy interpolator for efficient evaluation
    """
    wavelengths: jnp.ndarray
    vmic_grid: jnp.ndarray
    log_temp_grid: jnp.ndarray
    cross_sections: jnp.ndarray
    species: Species
    interpolator: Optional[RegularGridInterpolator] = None
    
    def __post_init__(self):
        """Initialize the interpolator after construction."""
        if self.interpolator is None:
            # Create interpolator using scipy for compatibility
            self.interpolator = RegularGridInterpolator(
                (self.vmic_grid, self.log_temp_grid, self.wavelengths),
                self.cross_sections,
                method='linear',
                bounds_error=False,
                fill_value=0.0
            )
    
    def interpolate(self, 
                   wavelengths: jnp.ndarray, 
                   temperatures: jnp.ndarray,
                   vmic: Union[float, jnp.ndarray],
                   number_density: Union[float, jnp.ndarray]) -> jnp.ndarray:
        """
        Interpolate molecular cross-sections for given conditions.
        
        Parameters
        ----------
        wavelengths : jnp.ndarray
            Wavelengths to evaluate at in cm
        temperatures : jnp.ndarray
            Temperatures for each atmospheric layer in K
        vmic : float or jnp.ndarray
            Microturbulent velocity in cm/s
        number_density : float or jnp.ndarray  
            Molecular number density in cm^-3
            
        Returns
        -------
        jnp.ndarray
            Absorption coefficients in cm^-1
        """
        n_layers = len(temperatures)
        n_wavelengths = len(wavelengths)
        alpha = jnp.zeros((n_layers, n_wavelengths))
        
        # Ensure vmic is an array
        if jnp.isscalar(vmic):
            vmic = jnp.full(n_layers, vmic)
        if jnp.isscalar(number_density):
            number_density = jnp.full(n_layers, number_density)
            
        log_temps = jnp.log10(temperatures)
        
        # Interpolate for each atmospheric layer
        for i in range(n_layers):
            # Create interpolation points
            points = jnp.array([
                [vmic[i], log_temps[i], wl] 
                for wl in wavelengths
            ])
            
            # Interpolate cross-sections
            cross_section_values = self.interpolator(points)
            
            # Scale by number density to get absorption coefficient
            alpha = alpha.at[i, :].set(cross_section_values * number_density[i])
            
        return alpha


def create_molecular_cross_section(
    linelist: List[LineData],
    wavelength_range: Tuple[float, float],
    wavelength_step: float = 1e-8,  # 0.01 Å in cm
    vmic_vals: Optional[jnp.ndarray] = None,
    log_temp_vals: Optional[jnp.ndarray] = None,
    cutoff_alpha: float = 1e-32,
    air_wavelengths: bool = False
) -> MolecularCrossSection:
    """
    Create precomputed molecular cross-sections from a linelist.
    
    This function matches Korg.jl's MolecularCrossSection constructor approach,
    computing absorption cross-sections on a 3D grid for efficient interpolation.
    
    Parameters
    ----------
    linelist : List[Line]
        List of molecular lines
    wavelength_range : Tuple[float, float]
        Wavelength range in cm (min, max)
    wavelength_step : float
        Wavelength step size in cm (default: 0.01 Å)
    vmic_vals : jnp.ndarray, optional
        Microturbulent velocity grid in cm/s
        Default: [0, 2e5, 4e5] cm/s (0, 2, 4 km/s)
    log_temp_vals : jnp.ndarray, optional
        Log temperature grid (log K)
        Default: 3.0 to 5.0 in steps of 0.04 (1000K to 100000K)
    cutoff_alpha : float
        Minimum opacity threshold (default: 1e-32)
    air_wavelengths : bool
        Whether input wavelengths are in air (default: False, vacuum)
        
    Returns
    -------
    MolecularCrossSection
        Precomputed cross-section object
    """
    
    # Default grids matching Korg.jl
    if vmic_vals is None:
        vmic_vals = jnp.array([0.0, 2e5, 4e5])  # 0, 2, 4 km/s in cm/s
    if log_temp_vals is None:
        log_temp_vals = jnp.arange(3.0, 5.01, 0.04)  # log K from 3.0 to 5.0
        
    # Create wavelength grid
    wl_min, wl_max = wavelength_range
    wavelengths = jnp.arange(wl_min, wl_max + wavelength_step, wavelength_step)
    
    # Convert to vacuum if needed
    if air_wavelengths:
        wavelengths = air_to_vacuum(wavelengths * 1e8) / 1e8  # Convert Å to cm and back
    
    # Get molecular species (assume all lines are same species)
    if len(linelist) == 0:
        raise ValueError("Empty linelist provided")
    species_id = linelist[0].species
    # Convert species ID to Species object
    species = Species(element_id=species_id, ion_state=0)
    
    # Initialize cross-section grid
    n_vmic = len(vmic_vals)
    n_temp = len(log_temp_vals) 
    n_wavelength = len(wavelengths)
    cross_sections = jnp.zeros((n_vmic, n_temp, n_wavelength))
    
    print(f"Computing molecular cross-sections for {species}")
    print(f"Grid: {n_vmic} vmic × {n_temp} temp × {n_wavelength} wavelength")
    
    # Compute cross-sections for each grid point
    for i, vmic in enumerate(vmic_vals):
        for j, log_temp in enumerate(log_temp_vals):
            temp = 10**log_temp
            
            # Compute line absorption for this temperature/vmic combination
            # Use simplified approach - in practice would call full line_absorption
            alpha = jnp.zeros(n_wavelength)
            
            for line in linelist:
                # Calculate line contribution
                line_alpha = compute_molecular_line_opacity(
                    line, wavelengths, temp, vmic, cutoff_alpha
                )
                alpha += line_alpha
            
            cross_sections = cross_sections.at[i, j, :].set(alpha)
            
    return MolecularCrossSection(
        wavelengths=wavelengths,
        vmic_grid=vmic_vals,
        log_temp_grid=log_temp_vals,
        cross_sections=cross_sections,
        species=species
    )


def compute_molecular_line_opacity(
    line: LineData,
    wavelengths: jnp.ndarray,
    temperature: float,
    vmic: float,
    cutoff_alpha: float
) -> jnp.ndarray:
    """
    Compute opacity contribution from a single molecular line.
    
    This is a simplified version for molecular cross-section precomputation.
    Molecular lines skip Stark and van der Waals broadening.
    
    Parameters
    ----------
    line : LineData
        Molecular line data
    wavelengths : jnp.ndarray
        Wavelength grid in cm
    temperature : float
        Temperature in K
    vmic : float
        Microturbulent velocity in cm/s
    cutoff_alpha : float
        Minimum opacity threshold
        
    Returns
    -------
    jnp.ndarray
        Line opacity in cm^2 (cross-section per molecule)
    """
    from .broadening import doppler_width
    from .profiles import line_profile
    
    # Get molecular mass
    mass = get_molecular_mass(line.species)
    
    # Calculate Doppler width
    sigma = doppler_width(line.wavelength, temperature, mass, vmic)
    
    # Only radiative damping for molecules (no Stark or vdW)
    gamma_total = line.gamma_rad
    
    # Convert to wavelength units
    gamma_wl = gamma_total * line.wavelength**2 / (c_cgs * 4 * jnp.pi)
    
    # Calculate line strength (simplified - would need partition functions)
    # For now, use the log_gf directly
    line_strength = 10**line.log_gf
    
    # Calculate cross-section at line center (approximate)
    sigma_line_center = (jnp.pi * 2.818e-13**2 * line.wavelength**2) / (2 * c_cgs)
    amplitude = line_strength * sigma_line_center
    
    # Calculate line profile
    alpha = line_profile(
        line.wavelength, sigma, gamma_wl, amplitude, wavelengths
    )
    
    # Apply cutoff threshold
    alpha = jnp.where(alpha < cutoff_alpha, 0.0, alpha)
    
    return alpha


def get_molecular_mass(species_id: int) -> float:
    """
    Get molecular mass in grams.
    
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


def save_molecular_cross_section(
    cross_section: MolecularCrossSection,
    filename: str
) -> None:
    """
    Save molecular cross-section to HDF5 file.
    
    Parameters
    ----------
    cross_section : MolecularCrossSection
        Cross-section object to save
    filename : str
        Output filename
    """
    with h5py.File(filename, 'w') as f:
        f.create_dataset('wavelengths', data=cross_section.wavelengths)
        f.create_dataset('vmic_grid', data=cross_section.vmic_grid)  
        f.create_dataset('log_temp_grid', data=cross_section.log_temp_grid)
        f.create_dataset('cross_sections', data=cross_section.cross_sections)
        
        # Save species information
        f.attrs['species_element_id'] = cross_section.species.element_id
        f.attrs['species_ion_state'] = cross_section.species.ion_state
        if cross_section.species.isotope is not None:
            f.attrs['species_isotope'] = cross_section.species.isotope


def load_molecular_cross_section(filename: str) -> MolecularCrossSection:
    """
    Load molecular cross-section from HDF5 file.
    
    Parameters
    ----------
    filename : str
        Input filename
        
    Returns
    -------
    MolecularCrossSection
        Loaded cross-section object
    """
    with h5py.File(filename, 'r') as f:
        wavelengths = jnp.array(f['wavelengths'][:])
        vmic_grid = jnp.array(f['vmic_grid'][:])
        log_temp_grid = jnp.array(f['log_temp_grid'][:])
        cross_sections = jnp.array(f['cross_sections'][:])
        
        # Restore species
        species = Species(
            element_id=f.attrs['species_element_id'],
            ion_state=f.attrs['species_ion_state'],
            isotope=f.attrs.get('species_isotope', None)
        )
        
    return MolecularCrossSection(
        wavelengths=wavelengths,
        vmic_grid=vmic_grid, 
        log_temp_grid=log_temp_grid,
        cross_sections=cross_sections,
        species=species
    )


def interpolate_molecular_cross_sections(
    wavelengths: jnp.ndarray,
    temperatures: jnp.ndarray,
    vmic: Union[float, jnp.ndarray],
    molecular_cross_sections: Dict[int, MolecularCrossSection],
    number_densities: Dict[int, jnp.ndarray]
) -> jnp.ndarray:
    """
    Interpolate molecular cross-sections for synthesis.
    
    This function matches Korg.jl's interpolate_molecular_cross_sections! function.
    
    Parameters
    ----------
    wavelengths : jnp.ndarray
        Wavelength grid in cm
    temperatures : jnp.ndarray
        Temperature profile in K
    vmic : float or jnp.ndarray
        Microturbulent velocity in cm/s
    molecular_cross_sections : Dict[int, MolecularCrossSection]
        Dictionary of precomputed cross-sections by species ID
    number_densities : Dict[int, jnp.ndarray]
        Number densities by species ID in cm^-3
        
    Returns
    -------
    jnp.ndarray
        Total molecular absorption coefficient in cm^-1
    """
    n_layers = len(temperatures)
    n_wavelengths = len(wavelengths)
    alpha_total = jnp.zeros((n_layers, n_wavelengths))
    
    # Sum contributions from all molecular species
    for species_id, cross_section in molecular_cross_sections.items():
        if species_id in number_densities:
            alpha_species = cross_section.interpolate(
                wavelengths, temperatures, vmic, number_densities[species_id]
            )
            alpha_total += alpha_species
            
    return alpha_total


# Utility functions for molecular species identification
def is_molecular_species(species: Species) -> bool:
    """Check if a species is molecular (more than one atom)."""
    atom_count = sum(count for _, count in species.formula.atoms if count > 0)
    return atom_count > 1


def get_common_molecular_species() -> Dict[str, Species]:
    """
    Get common stellar molecular species.
    
    Returns
    -------
    Dict[str, Species]
        Dictionary mapping molecular names to Species objects
    """
    molecules = {
        'H2O': Species(element_id=801, ion_state=0),  # Water
        'TiO': Species(element_id=2208, ion_state=0), # Titanium oxide
        'VO': Species(element_id=2308, ion_state=0),  # Vanadium oxide  
        'OH': Species(element_id=108, ion_state=0),   # Hydroxyl
        'CH': Species(element_id=601, ion_state=0),   # Methylidyne
        'CN': Species(element_id=607, ion_state=0),   # Cyanogen
        'CO': Species(element_id=608, ion_state=0),   # Carbon monoxide
        'NH': Species(element_id=701, ion_state=0),   # Imidogen
        'SiO': Species(element_id=1408, ion_state=0), # Silicon monoxide
        'CaH': Species(element_id=2001, ion_state=0), # Calcium hydride
    }
    return molecules