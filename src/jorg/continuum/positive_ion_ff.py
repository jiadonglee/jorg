"""
Positive ion free-free absorption (Korg.jl compatibility).

This includes all free-free interactions involving positively charged ions.
"""

from typing import Dict, Optional, Any
import numpy as np
import jax.numpy as jnp

from ..constants import c_cgs, hplanck_eV, Rydberg_eV
from ..statmech.species import Species
from .hydrogenic_bf_ff import (
    hydrogenic_ff_absorption,
    hydrogenic_ff_absorption_jax,
    get_gaunt_factor_bounds,
)
from .peach1970 import get_all_departure_coefficients
from .interp_jax import interp2_linear_clamped

_DEPARTURE_COEFFICIENTS: Optional[Dict[Species, object]] = None
_GAUNT_FACTOR_BOUNDS = get_gaunt_factor_bounds()
_SIGMA_SCALE = hplanck_eV / Rydberg_eV


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
    t_bounds, lambda_bounds = _GAUNT_FACTOR_BOUNDS
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
    freq_sel = freq_arr[idx]

    if departure_coefficients is None:
        departure_coefficients = _load_departure_coefficients()

    ndens_Z1 = 0.0
    ndens_Z2 = 0.0

    for spec, ndens in number_densities.items():
        if spec.charge <= 0 or ndens <= 0.0:
            continue
        if spec in departure_coefficients:
            D = departure_coefficients[spec]
            sigma = freq_sel / (spec.charge ** 2) * _SIGMA_SCALE
            correction = 1.0 + D(temperature, sigma)
            alpha[idx] += (
                hydrogenic_ff_absorption(freq_sel, temperature, spec.charge, ndens, electron_density)
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
        alpha[idx] += hydrogenic_ff_absorption(freq_sel, temperature, 1, ndens_Z1, electron_density)
    if ndens_Z2 > 0.0:
        alpha[idx] += hydrogenic_ff_absorption(freq_sel, temperature, 2, ndens_Z2, electron_density)

    return float(alpha[0]) if scalar_input else alpha


def positive_ion_ff_absorption_dense_batch(
    frequencies: jnp.ndarray,
    temperatures: jnp.ndarray,
    number_densities_dense: jnp.ndarray,
    electron_densities: jnp.ndarray,
    species_layout: Any,
    departure_coefficients: Optional[Dict[Species, object]] = None,
) -> jnp.ndarray:
    """
    Vectorized positive-ion free-free absorption on dense [layers, species] inputs.
    """
    freq = jnp.asarray(frequencies, dtype=jnp.float64)
    temps = jnp.asarray(temperatures, dtype=jnp.float64)
    ne = jnp.asarray(electron_densities, dtype=jnp.float64)
    dense = jnp.asarray(number_densities_dense, dtype=jnp.float64)

    if dense.ndim != 2:
        raise ValueError("number_densities_dense must be rank-2 [n_layers, n_species].")
    if dense.shape[0] != temps.shape[0]:
        raise ValueError("number_densities_dense layer count must match temperatures.")
    if dense.shape[1] != len(species_layout.species):
        raise ValueError("number_densities_dense species axis must match species_layout.")

    n_layers = dense.shape[0]
    alpha = jnp.zeros((n_layers, freq.shape[0]), dtype=jnp.float64)

    t_bounds, lambda_bounds = _GAUNT_FACTOR_BOUNDS
    temp_mask = (temps >= t_bounds[0]) & (temps <= t_bounds[1])
    freq_min = c_cgs / lambda_bounds[1]
    freq_max = c_cgs / lambda_bounds[0]
    freq_mask = (freq > freq_min) & (freq < freq_max)

    if departure_coefficients is None:
        departure_coefficients = _load_departure_coefficients()

    ndens_z1 = jnp.zeros((n_layers,), dtype=jnp.float64)
    ndens_z2 = jnp.zeros((n_layers,), dtype=jnp.float64)
    temp_grid = temps[:, None]
    ne_grid = ne[:, None]
    freq_grid = freq[None, :]

    for idx, spec in enumerate(species_layout.species):
        charge = int(getattr(spec, "charge", 0))
        if charge <= 0:
            continue

        ndens = dense[:, idx]

        if spec in departure_coefficients:
            dep = departure_coefficients[spec]
            sigma = freq_grid / float(charge * charge) * _SIGMA_SCALE
            if not (
                hasattr(dep, "_T_vals_jnp")
                and hasattr(dep, "_sigma_vals_jnp")
                and hasattr(dep, "_table_vals_jnp")
            ):
                raise ValueError("Departure coefficient interpolator must expose JAX table attributes.")
            correction = interp2_linear_clamped(
                temp_grid,
                sigma,
                dep._T_vals_jnp,
                dep._sigma_vals_jnp,
                dep._table_vals_jnp,
                x_mode="zero",
                y_mode="zero",
            )

            base = hydrogenic_ff_absorption_jax(
                freq_grid,
                temp_grid,
                charge,
                ndens[:, None],
                ne_grid,
            )
            alpha = alpha + base * (1.0 + correction)
            continue

        if charge == 1:
            ndens_z1 = ndens_z1 + ndens
        elif charge == 2:
            ndens_z2 = ndens_z2 + ndens
        else:
            raise ValueError("Triply ionized species not supported")

    alpha = alpha + hydrogenic_ff_absorption_jax(
        freq_grid,
        temp_grid,
        1,
        ndens_z1[:, None],
        ne_grid,
    )
    alpha = alpha + hydrogenic_ff_absorption_jax(
        freq_grid,
        temp_grid,
        2,
        ndens_z2[:, None],
        ne_grid,
    )

    mask = temp_mask[:, None] & freq_mask[None, :]
    return jnp.where(mask, alpha, 0.0)
