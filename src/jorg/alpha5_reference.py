"""
Alpha5 Reference Implementation
Provides alpha5 reference opacity calculation for radiative transfer.
"""

from functools import lru_cache

import numpy as np

from .constants import c_cgs
from .continuum.exact_physics_continuum import total_continuum_absorption_exact_physics_only
from .opacity.korg_line_processor import KorgLineProcessor
from .statmech.korg_chemical_equilibrium import chemical_equilibrium
from .statmech import create_default_ionization_energies, create_default_log_equilibrium_constants, create_default_partition_functions
from .lines.linelist import read_linelist
from .data import get_data_path

ALPHA_5000_WL_ANG = 5000.0
ALPHA_5000_WL_CM = ALPHA_5000_WL_ANG * 1e-8
ALPHA_5000_BUFFER_ANG = 21.0

_alpha_5000_default_linelist = None


@lru_cache(maxsize=1)
def _load_alpha_5000_default_linelist():
    """Load and cache the default alpha 5000 linelist."""
    global _alpha_5000_default_linelist
    if _alpha_5000_default_linelist is not None:
        return _alpha_5000_default_linelist

    default_path = get_data_path("linelists", "alpha_5000", "alpha_5000_lines.csv")
    _alpha_5000_default_linelist = read_linelist(default_path, format="alpha_5000")
    return _alpha_5000_default_linelist


def _line_wavelength_cm(line):
    if hasattr(line, "wavelength"):
        return float(line.wavelength)
    if hasattr(line, "wl"):
        return float(line.wl)
    return None


def _get_alpha_5000_linelist(linelist):
    """
    Get lines near 5000 Å for alpha5 reference calculation.

    Uses the provided linelist if it covers the 5000 Å region.
    Only falls back to default alpha_5000 linelist if needed.
    """
    # First, try to extract 5000 Å lines from the provided linelist
    if linelist is not None and len(linelist) > 0:
        lines = linelist.lines if hasattr(linelist, "lines") else linelist
        wl_min_cm = (ALPHA_5000_WL_ANG - ALPHA_5000_BUFFER_ANG) * 1e-8
        wl_max_cm = (ALPHA_5000_WL_ANG + ALPHA_5000_BUFFER_ANG) * 1e-8
        linelist5 = [line for line in lines
                     if (wl := _line_wavelength_cm(line)) is not None and wl_min_cm <= wl <= wl_max_cm]

        if len(linelist5) > 0:
            linelist5 = sorted(linelist5, key=_line_wavelength_cm)
            min_wl = _line_wavelength_cm(linelist5[0])
            max_wl = _line_wavelength_cm(linelist5[-1])

            # Check if the linelist fully covers 5000 Å
            if min_wl is not None and max_wl is not None:
                if min_wl <= ALPHA_5000_WL_CM <= max_wl:
                    # Provided linelist covers 5000 Å - use it directly
                    return linelist5

                # Partial coverage - try to supplement with default if available
                try:
                    default_linelist = _load_alpha_5000_default_linelist()
                    if min_wl > ALPHA_5000_WL_CM:
                        fallback = [line for line in default_linelist if _line_wavelength_cm(line) < min_wl]
                        return sorted(fallback + linelist5, key=_line_wavelength_cm)
                    if max_wl < ALPHA_5000_WL_CM:
                        fallback = [line for line in default_linelist if _line_wavelength_cm(line) > max_wl]
                        return sorted(linelist5 + fallback, key=_line_wavelength_cm)
                except FileNotFoundError:
                    # No default available, use what we have
                    return linelist5

    # No provided linelist or it doesn't cover the region - try default
    try:
        return _load_alpha_5000_default_linelist()
    except FileNotFoundError:
        # No default linelist available - return empty (continuum-only)
        return []


def calculate_alpha5_reference(atm, A_X, linelist=None, number_densities=None,
                               electron_densities=None, partition_funcs=None,
                               ionization_energies=None, log_equilibrium_constants=None,
                               microturbulence_kms=1.0, line_cutoff_threshold=3e-4,
                               use_chemical_equilibrium_from=None, verbose=False):
    """
    Calculate alpha5 reference opacity for radiative transfer anchoring.

    Parameters
    ----------
    use_chemical_equilibrium_from : dict, optional
        Pre-computed chemical equilibrium results with keys:
        - 'electron_densities': array of electron densities per layer
        - 'number_densities': dict of species number densities per layer
        When provided, reuses these results instead of recalculating (Korg.jl optimization).
    """
    if verbose:
        print("Calculating alpha5 reference opacity...")

    # Handle both dictionary and ModelAtmosphere formats
    if hasattr(atm, "layers"):
        layers = atm.layers
        temperatures = np.array([layer.temp for layer in layers])
        number_density_layers = np.array([layer.number_density for layer in layers])
        electron_density_guess = np.array([layer.electron_number_density for layer in layers])
    else:
        temperatures = np.array(atm["temperature"])
        number_density_layers = None
        electron_density_guess = None
        if "number_density" in atm:
            number_density_layers = np.array(atm["number_density"])
        if "electron_density" in atm:
            electron_density_guess = np.array(atm["electron_density"])

    n_layers = len(temperatures)

    if ionization_energies is None:
        ionization_energies = create_default_ionization_energies()
    if log_equilibrium_constants is None:
        log_equilibrium_constants = create_default_log_equilibrium_constants()
    if partition_funcs is None:
        partition_funcs = create_default_partition_functions()

    # OPTIMIZATION: Reuse chemical equilibrium results if provided (Korg.jl approach)
    if use_chemical_equilibrium_from is not None:
        if verbose:
            print("   🔄 Reusing chemical equilibrium results (no CE recalculation)")
        electron_densities = np.asarray(use_chemical_equilibrium_from['electron_densities'])
        number_densities = use_chemical_equilibrium_from['number_densities']
    elif number_densities is None or electron_densities is None:
        if number_density_layers is None or electron_density_guess is None:
            raise ValueError("atm must include number_density and electron_density when computing chemical equilibrium.")
        # Only calculate CE if not provided and not reused
        abs_abundances = 10 ** (A_X - 12)
        abs_abundances = abs_abundances / np.sum(abs_abundances)
        abs_abundances = {Z: float(abs_abundances[Z - 1]) for Z in range(1, 93)}

        number_densities = {}
        electron_densities = np.zeros(n_layers)

        if verbose:
            print("   ⚠️  Calculating chemical equilibrium (consider passing use_chemical_equilibrium_from)")

        for i in range(n_layers):
            ne, n_dict = chemical_equilibrium(
                temp=float(temperatures[i]),
                nt=float(number_density_layers[i]),
                model_atm_ne=float(electron_density_guess[i]),
                absolute_abundances=abs_abundances,
                ionization_energies=ionization_energies,
                partition_funcs=partition_funcs,
                log_equilibrium_constants=log_equilibrium_constants
            )
            electron_densities[i] = ne
            for spec, dens in n_dict.items():
                if spec not in number_densities:
                    number_densities[spec] = np.zeros(n_layers)
                number_densities[spec][i] = dens
    else:
        electron_densities = np.asarray(electron_densities)

    alpha5_continuum = np.zeros(n_layers)
    frequency_5000 = c_cgs / ALPHA_5000_WL_CM

    for i in range(n_layers):
        layer_number_densities = {spec: dens[i] for spec, dens in number_densities.items()}
        alpha5_continuum[i] = float(total_continuum_absorption_exact_physics_only(
            np.array([frequency_5000]),
            float(temperatures[i]),
            float(electron_densities[i]),
            layer_number_densities,
            partition_funcs=partition_funcs
        )[0])

    # IMPORTANT: τ_5000 in the atmosphere is defined relative to the *continuum* opacity
    # at 5000 Å. For anchored optical-depth integration, alpha_ref must therefore be the
    # continuum opacity at 5000 Å as well. Including line opacity here breaks the
    # consistency between (tau_ref, alpha_ref) and distorts line depths.
    alpha5_ref = alpha5_continuum

    if verbose:
        print(f"Alpha5 reference range: {alpha5_ref.min():.2e} - {alpha5_ref.max():.2e} cm^-1")

    return alpha5_ref
