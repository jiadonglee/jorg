"""
Statistical mechanics module for stellar spectroscopy calculations.

This module provides functions for chemical equilibrium, ionization balance,
partition functions, and equation of state following the Korg.jl implementation.
"""

from .core import chemical_equilibrium
from .partition_functions import hydrogen_partition_function, simple_partition_function
from .ionization import saha_ion_weights, translational_u
from .molecular import get_log_nk
from .eos import (
    gas_pressure, electron_pressure, total_pressure,
    number_density_from_pressure, pressure_scale_height,
    ideal_gas_density, pressure_from_density
)
from .atmosphere import (
    calculate_pressure_from_atmosphere, calculate_densities_from_pressure,
    pressure_ratio, electron_density_fraction, mean_molecular_weight_from_densities
)

__all__ = [
    'chemical_equilibrium',
    'hydrogen_partition_function', 
    'simple_partition_function',
    'saha_ion_weights',
    'translational_u',
    'get_log_nk',
    'gas_pressure',
    'electron_pressure', 
    'total_pressure',
    'number_density_from_pressure',
    'pressure_scale_height',
    'ideal_gas_density',
    'pressure_from_density',
    'calculate_pressure_from_atmosphere',
    'calculate_densities_from_pressure',
    'pressure_ratio',
    'electron_density_fraction',
    'mean_molecular_weight_from_densities'
]