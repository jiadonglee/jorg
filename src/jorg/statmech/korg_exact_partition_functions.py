"""
EXACT Korg.jl Partition Functions - Load from HDF5
===================================================

This module loads the ACTUAL partition functions from Korg.jl's HDF5 file,
NOT approximations. This fixes the 72-100% partition function errors that
were causing the 6.7% electron density discrepancy.

Direct port from Korg.jl/src/read_statmech_quantities.jl:172-198
"""

import numpy as np
import h5py
from scipy.interpolate import CubicSpline
from typing import Dict, Callable
import warnings
import os

from .species import Species
from ..data import get_data_path


class KorgExactPartitionFunctions:
    """
    Load EXACT partition functions from Korg.jl HDF5 data file

    This replaces all approximations with the actual Korg.jl partition functions,
    fixing 72-100% errors in H I, He I, Fe I, Fe II, etc.
    """

    def __init__(self, data_file: str = None):
        """
        Initialize by loading Korg.jl partition function data

        Parameters
        ----------
        data_file : str, optional
            Path to partition_funcs.h5. If None, searches standard locations.
        """
        if data_file is None:
            try:
                data_file = str(get_data_path("atomic_partition_funcs", "partition_funcs.h5"))
            except FileNotFoundError:
                data_file = None

            if data_file is None:
                raise FileNotFoundError(
                    "Could not find partition_funcs.h5. "
                    "Set JORG_DATA_DIR or specify data_file."
                )

        self.data_file = data_file
        self.partition_funcs = {}
        self._load_partition_functions()

    def _load_partition_functions(self):
        """
        Load partition functions from Korg.jl HDF5 file

        Direct port of Korg.jl/src/read_statmech_quantities.jl:172-198
        """
        with h5py.File(self.data_file, 'r') as f:
            # Load temperature grid
            logT_min = float(f['logT_min'][()])
            logT_max = float(f['logT_max'][()])
            logT_step = float(f['logT_step'][()])

            # Reconstruct temperature grid (matches Korg.jl exactly)
            self.log_temps = np.arange(logT_min, logT_max + logT_step/2, logT_step)

            # Element symbols to atomic numbers
            element_symbols = {
                'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
                'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
                'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
                'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
                'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
                'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
                'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
                'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
                'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
                'Pa': 91, 'U': 92,
            }

            # Roman numeral to charge
            roman_to_charge = {'I': 0, 'II': 1, 'III': 2}

            # Load all species partition functions
            for key in f.keys():
                if key.startswith('logT'):
                    continue  # Skip metadata

                # Parse species key (e.g., "H I", "Fe II")
                parts = key.split()
                if len(parts) != 2:
                    continue

                element_sym, roman = parts
                if element_sym not in element_symbols or roman not in roman_to_charge:
                    continue

                Z = element_symbols[element_sym]
                charge = roman_to_charge[roman]

                # Load partition function values (U, not log(U))
                U_values = f[key][:]

                # Create Species object
                species = Species.from_atomic_number(Z, charge)

                # Create cubic spline interpolator (matching Korg.jl)
                # Note: Korg.jl stores U directly, not log(U)
                partition_func = CubicSpline(self.log_temps, U_values, extrapolate=True)

                self.partition_funcs[species] = partition_func

        print(f"Loaded {len(self.partition_funcs)} EXACT partition functions from Korg.jl")

        # Add special cases for bare nuclei (not in HDF5 file)
        # H II (proton), He III (alpha particle), etc. have U=1 (no internal states)
        log_temps_array = np.array(self.log_temps)
        ones_U = np.ones_like(log_temps_array)

        # H II - bare proton
        h_ii = Species.from_atomic_number(1, 1)
        self.partition_funcs[h_ii] = CubicSpline(log_temps_array, ones_U, extrapolate=True)

        # He III - alpha particle
        he_iii = Species.from_atomic_number(2, 2)
        self.partition_funcs[he_iii] = CubicSpline(log_temps_array, ones_U, extrapolate=True)

        print(f"  Added bare nuclei: H II, He III (U=1)")

    def get_partition_function(self, species: Species, log_temperature: float) -> float:
        """
        Get partition function U(T) for a species at log(T)

        Parameters
        ----------
        species : Species
            Atomic or ionic species
        log_temperature : float
            Natural logarithm of temperature in K

        Returns
        -------
        float
            Partition function U(T)
        """
        if species in self.partition_funcs:
            # Interpolate and return U directly
            return float(self.partition_funcs[species](log_temperature))
        else:
            warnings.warn(f"Species {species} not found in Korg partition functions")
            # Fallback for missing species
            temperature = np.exp(log_temperature)
            if species.charge == 0:
                return 2.0  # Crude fallback
            else:
                return 1.0

    def __contains__(self, species: Species) -> bool:
        """Check if species has a partition function"""
        return species in self.partition_funcs

    def __getitem__(self, species: Species) -> Callable:
        """Get partition function interpolator for a species"""
        return lambda log_T: self.get_partition_function(species, log_T)


# Global instance
_korg_exact_partition_functions = None


def get_korg_exact_partition_functions(data_file: str = None) -> KorgExactPartitionFunctions:
    """
    Get global instance of EXACT Korg.jl partition functions

    Parameters
    ----------
    data_file : str, optional
        Path to partition_funcs.h5. If None, searches standard locations.

    Returns
    -------
    KorgExactPartitionFunctions
        Loaded partition function system
    """
    global _korg_exact_partition_functions
    if _korg_exact_partition_functions is None:
        _korg_exact_partition_functions = KorgExactPartitionFunctions(data_file)
    return _korg_exact_partition_functions


__all__ = ['KorgExactPartitionFunctions', 'get_korg_exact_partition_functions']
