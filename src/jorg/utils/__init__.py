"""
Utility functions for Jorg
"""

from .math import (
    inverse_gaussian_density, inverse_lorentz_density,
    linear_interpolation, safe_log, safe_exp, safe_sqrt
)
from .wavelength_utils import (
    air_to_vacuum, vacuum_to_air, detect_wavelength_unit
)

__all__ = [
    # Math utilities
    "inverse_gaussian_density", "inverse_lorentz_density", 
    "linear_interpolation", "safe_log", "safe_exp", "safe_sqrt",
    # Wavelength utilities  
    "air_to_vacuum", "vacuum_to_air", "detect_wavelength_unit"
]