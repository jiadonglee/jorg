"""
Positive ion free-free absorption (Korg.jl compatibility).

This includes all free-free interactions involving positively charged ions.
"""

from typing import Dict, Optional
import numpy as np

from ..constants import c_cgs, hplanck_eV, Rydberg_eV
from ..statmech.species import Species
from .hydrogenic_bf_ff import hydrogenic_ff_absorption, get_gaunt_factor_bounds
from .peach1970 import get_all_departure_coefficients

_DEPARTURE_COEFFICIENTS: Optional[Dict[Species, object]] = None


def _load_departure_coefficients() -> Dict[Species, object]:
    global _DEPARTURE_COEFFICIENTS
    if _DEPARTURE_COEFFICIENTS is None:
        _DEPARTURE_COEFFICIENTS = {
            Species.from_string(name): interp
            for name, interp in get_all_departure_coefficients().items()
        }
    return _DEPARTURE_COEFFICIENTS


def positive_ion_ff_absorption(
    frequencies: np.ndarray,
    temperature: float,
    number_densities: Dict[Species, float],
    electron_density: float,
    departure_coefficients: Optional[Dict[Species, object]] = None
) -> np.ndarray:
    """
    Compute free-free absorption for positive ions using hydrogenic Gaunt factors.
    """
    t_bounds, lambda_bounds = get_gaunt_factor_bounds()
    if not (t_bounds[0] <= temperature <= t_bounds[1]):
        return np.zeros_like(frequencies, dtype=float)

    freq_arr = np.asarray(frequencies, dtype=float)
    scalar_input = np.isscalar(frequencies)
    if scalar_input:
        freq_arr = freq_arr.reshape(1)

    freq_min = c_cgs / lambda_bounds[1]
    freq_max = c_cgs / lambda_bounds[0]
    idx = (freq_arr > freq_min) & (freq_arr < freq_max)

    alpha = np.zeros_like(freq_arr, dtype=float)
    if not np.any(idx):
        return float(alpha[0]) if scalar_input else alpha

    if departure_coefficients is None:
        departure_coefficients = _load_departure_coefficients()

    ndens_Z1 = 0.0
    ndens_Z2 = 0.0

    for spec, ndens in number_densities.items():
        if spec.charge <= 0:
            continue
        if spec in departure_coefficients:
            D = departure_coefficients[spec]
            sigma = freq_arr[idx] / (spec.charge ** 2) * (hplanck_eV / Rydberg_eV)
            correction = 1.0 + D(temperature, sigma)
            alpha[idx] += (
                hydrogenic_ff_absorption(freq_arr[idx], temperature, spec.charge, ndens, electron_density)
                * correction
            )
        else:
            if spec.charge == 1:
                ndens_Z1 += ndens
            elif spec.charge == 2:
                ndens_Z2 += ndens
            else:
                raise ValueError("Triply ionized species not supported")

    if ndens_Z1 > 0.0:
        alpha[idx] += hydrogenic_ff_absorption(freq_arr[idx], temperature, 1, ndens_Z1, electron_density)
    if ndens_Z2 > 0.0:
        alpha[idx] += hydrogenic_ff_absorption(freq_arr[idx], temperature, 2, ndens_Z2, electron_density)

    return float(alpha[0]) if scalar_input else alpha
