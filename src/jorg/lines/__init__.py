"""
Line absorption calculations for stellar spectral synthesis

This module implements atomic and molecular line opacity calculations
in JAX for high-performance stellar spectral synthesis.

Key Features:
- Sophisticated hydrogen line treatment with MHD formalism
- ABO van der Waals broadening for Balmer lines
- Griem Stark broadening for Brackett lines
- Complete pressure ionization physics
"""

from .core import line_absorption, LineData, create_line_data
from .profiles import (
    line_profile,
    voigt_hjerting,
    harris_series
)
from .voigt import (
    voigt_profile,
    voigt_profile_wavelength
)
from .broadening import (
    doppler_width,
    scaled_stark,
    scaled_vdw
)
from .utils import (
    inverse_gaussian_density,
    inverse_lorentz_density,
    sigma_line
)
from .linelist import (
    read_linelist,
    save_linelist,
    LineList
)
from .linelist_data import (
    get_VALD_solar_linelist,
    get_APOGEE_DR17_linelist,
    get_GALAH_DR3_linelist,
    get_GES_linelist
)
from .species import (
    parse_species,
    species_id_to_name,
    get_species_info,
    Species
)
from .atomic_data import (
    get_atomic_symbol,
    get_atomic_number,
    get_atomic_mass,
    get_ionization_energy,
    format_species_name,
    get_solar_abundance,
    ATOMIC_SYMBOLS,
    ATOMIC_NUMBERS
)
from .broadening_korg import (
    approximate_radiative_gamma,
    approximate_gammas,
    approximate_stark_broadening,
    approximate_vdw_broadening,
    approximate_line_strength,
    get_default_broadening_parameters
)
from .broadening import (
    get_korg_broadening_parameters,
    calculate_line_broadening,
    validate_line_broadening_parameters
)
from .linelist_utils import (
    merge_linelists,
    remove_duplicate_lines,
    filter_lines_by_strength,
    filter_lines_by_species,
    prune_weak_lines,
    get_linelist_statistics,
    print_linelist_summary,
    create_line_window,
    validate_linelist_physics,
    convert_linelist_to_korg_format,
    split_linelist_by_species
)
from .datatypes import (
    LineData,
    Line,
    Formula,
    Species as KorgSpecies,
    create_line_data,
    create_line,
    species_from_integer
)
from ..utils.wavelength_utils import (
    air_to_vacuum,
    vacuum_to_air,
    angstrom_to_cm,
    cm_to_angstrom
)

# Hydrogen lines - sophisticated treatment  
try:
    from .hydrogen_lines import (
        hydrogen_line_absorption,
        hummer_mihalas_w
    )
    _hydrogen_available = True
except ImportError:
    _hydrogen_available = False

__all__ = [
    # Core line absorption
    "line_absorption",
    "LineData", 
    "create_line_data",
    
    # Line profiles and physics
    "line_profile",
    "voigt_hjerting", 
    "harris_series",
    "voigt_profile",
    "voigt_profile_wavelength",
    "doppler_width",
    "scaled_stark",
    "scaled_vdw",
    
    # Utilities
    "inverse_gaussian_density",
    "inverse_lorentz_density",
    "sigma_line",
    
    # Linelist reading and management
    "read_linelist",
    "save_linelist", 
    "LineList",
    
    # Built-in linelist functions (Korg.jl compatibility)
    "get_VALD_solar_linelist",
    "get_APOGEE_DR17_linelist", 
    "get_GALAH_DR3_linelist",
    "get_GES_linelist",
    
    # Species handling
    "parse_species",
    "species_id_to_name",
    "get_species_info",
    "Species",
    
    # Wavelength utilities
    "air_to_vacuum",
    "vacuum_to_air",
    "angstrom_to_cm",
    "cm_to_angstrom",
    
    # Atomic data
    "get_atomic_symbol",
    "get_atomic_number",
    "get_atomic_mass",
    "get_ionization_energy",
    "format_species_name",
    "get_solar_abundance",
    "ATOMIC_SYMBOLS",
    "ATOMIC_NUMBERS",
    
    # Korg.jl-compatible broadening
    "approximate_radiative_gamma",
    "approximate_gammas",
    "approximate_stark_broadening",
    "approximate_vdw_broadening",
    "approximate_line_strength",
    "get_default_broadening_parameters",
    "get_korg_broadening_parameters",
    "calculate_line_broadening",
    "validate_line_broadening_parameters",
    
    # Linelist utilities
    "merge_linelists",
    "remove_duplicate_lines",
    "filter_lines_by_strength",
    "filter_lines_by_species",
    "prune_weak_lines",
    "get_linelist_statistics",
    "print_linelist_summary",
    "create_line_window",
    "validate_linelist_physics",
    "convert_linelist_to_korg_format",
    "split_linelist_by_species",
    
    # Korg.jl-compatible datatypes
    "Line",
    "Formula",
    "KorgSpecies",
    "create_line",
    "species_from_integer"
]

# Add hydrogen lines to exports if available
if _hydrogen_available:
    __all__.extend([
        "hydrogen_line_absorption",
        "hummer_mihalas_w"
    ])