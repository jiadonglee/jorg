"""
Statmech utilities required for synthesis.

This module exposes the minimal surface area used by the synthesis pipeline
and line/opacity helpers.
"""

from .korg_equilibrium_constants import (
    create_default_partition_functions_korg as create_default_partition_functions,
    create_default_log_equilibrium_constants_korg as create_default_log_equilibrium_constants,
)
from .proper_ionization_energies import (
    create_proper_ionization_energy_dict as create_default_ionization_energies,
    get_proper_ionization_energies,
)
from .korg_chemical_equilibrium import (
    chemical_equilibrium,
    saha_ion_weights,
    translational_U,
)
from .species import Species, Formula, MAX_ATOMIC_NUMBER
from .hummer_mihalas import hummer_mihalas_w, hummer_mihalas_U_H

__all__ = [
    "chemical_equilibrium",
    "saha_ion_weights",
    "translational_U",
    "create_default_partition_functions",
    "create_default_log_equilibrium_constants",
    "create_default_ionization_energies",
    "get_proper_ionization_energies",
    "Species",
    "Formula",
    "MAX_ATOMIC_NUMBER",
    "hummer_mihalas_w",
    "hummer_mihalas_U_H",
]

__description__ = "Statmech utilities for synthesis"
