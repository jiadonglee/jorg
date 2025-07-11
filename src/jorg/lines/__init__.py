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
from .species import (
    parse_species,
    species_id_to_name,
    get_species_info,
    Species
)
from ..utils.wavelength_utils import (
    air_to_vacuum,
    vacuum_to_air,
    angstrom_to_cm,
    cm_to_angstrom
)

# Hydrogen lines - sophisticated treatment
try:
    from .hydrogen_lines_simple import (
        hydrogen_line_absorption_balmer,
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
    
    # Species handling
    "parse_species",
    "species_id_to_name",
    "get_species_info",
    "Species",
    
    # Wavelength utilities
    "air_to_vacuum",
    "vacuum_to_air",
    "angstrom_to_cm",
    "cm_to_angstrom"
]

# Add hydrogen lines to exports if available
if _hydrogen_available:
    __all__.extend([
        "hydrogen_line_absorption_balmer",
        "hummer_mihalas_w"
    ])