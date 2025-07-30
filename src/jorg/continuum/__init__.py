"""
Continuum absorption calculations for stellar spectral synthesis

This module implements bound-free, free-free, and scattering opacity sources
in JAX for high-performance stellar spectral synthesis.
"""

from .core import total_continuum_absorption
from .hydrogen import (
    h_i_bf_absorption,
    h_minus_bf_absorption, 
    h_minus_ff_absorption,
    h2_plus_bf_ff_absorption
)
from .helium import he_minus_ff_absorption
from .scattering import thomson_scattering, rayleigh_scattering
from .metals_bf import metal_bf_absorption
from .utils import frequency_to_wavelength, wavelength_to_frequency
from .h_i_bf_api import H_I_bf, H_I_bf_fast, H_I_bf_stellar

__all__ = [
    "total_continuum_absorption",
    "h_i_bf_absorption",
    "h_minus_bf_absorption", 
    "h_minus_ff_absorption",
    "h2_plus_bf_ff_absorption",
    "he_minus_ff_absorption",
    "metal_bf_absorption",
    "thomson_scattering",
    "rayleigh_scattering",
    "frequency_to_wavelength",
    "wavelength_to_frequency",
    "H_I_bf",
    "H_I_bf_fast", 
    "H_I_bf_stellar"
]