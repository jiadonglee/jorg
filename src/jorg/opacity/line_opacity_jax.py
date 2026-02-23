"""
Experimental JAX-native line opacity backend.

This module keeps a pure-JAX compute core for line opacity accumulation while
using a static Python/NumPy preprocessing pass to pack linelist objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.scipy.special import gamma as jax_gamma
import numpy as np

from ..constants import (
    PI,
    amu_cgs,
    c_cgs,
    electron_charge_cgs,
    electron_mass_cgs,
    hplanck_eV,
    kboltz_cgs,
    kboltz_eV,
)
from ..core.state_jax import DenseSpeciesLayout
from ..lines.profiles import line_profile
from ..statmech.species import Species


_PARTITION_LOGU_CACHE: Dict[Tuple[int, Tuple[Tuple[int, int], ...]], "SpeciesPartitionLogUTable"] = {}


@dataclass(frozen=True)
class LineTensorPack:
    """Dense line tensors for the JAX line-opacity core."""

    wavelength: jnp.ndarray
    gf: jnp.ndarray
    log_gf_base: jnp.ndarray
    E_lower: jnp.ndarray
    delta_E: jnp.ndarray
    gamma_rad: jnp.ndarray
    gamma_stark: jnp.ndarray
    vdw_sigma: jnp.ndarray
    vdw_alpha: jnp.ndarray
    gamma_factor: jnp.ndarray
    vdw_base_gamma: jnp.ndarray
    species_idx: jnp.ndarray
    atomic_mass: jnp.ndarray
    is_molecule: jnp.ndarray
    cross_section: jnp.ndarray
    source_line_index: jnp.ndarray

    @property
    def n_lines(self) -> int:
        return int(self.wavelength.shape[0])


@dataclass(frozen=True)
class SpeciesPartitionLogUTable:
    """Partition-function lookup tables on a shared log(T) grid."""

    logT_grid: jnp.ndarray
    logU_by_species: jnp.ndarray  # [n_species, n_logT]


def _line_wavelength_cm(line: Any) -> Optional[float]:
    """Extract line wavelength in cm from mixed line object formats."""
    wl = getattr(line, "wavelength", None)
    if wl is None:
        wl = getattr(line, "wl", None)
    if wl is None:
        return None
    wl_val = float(wl)
    if not np.isfinite(wl_val) or wl_val <= 0.0:
        return None
    # Defensive fallback for Angstrom-valued objects.
    if wl_val > 1.0:
        wl_val = wl_val * 1e-8
    return wl_val


def _line_vdw_params(line: Any) -> tuple[float, float]:
    """Return (sigma, alpha) vdW parameters from line object."""
    if hasattr(line, "vdW"):
        sigma, alpha = line.vdW
    elif hasattr(line, "vdw_param1") and hasattr(line, "vdw_param2"):
        sigma = getattr(line, "vdw_param1")
        alpha = getattr(line, "vdw_param2")
    else:
        sigma, alpha = 0.0, -1.0
    sigma = float(sigma)
    alpha = float(alpha)
    if not np.isfinite(sigma) or sigma < 0.0:
        sigma = 0.0
    if not np.isfinite(alpha):
        alpha = -1.0
    return sigma, alpha


def _infer_species_identity(species: Any) -> Optional[Tuple[int, int]]:
    """
    Return `(atomic_number, charge)` for atomic species, else None.
    """
    try:
        if not bool(getattr(species, "is_atom", False)):
            return None
        z = int(species.get_atom())
        charge = int(getattr(species, "charge", 0))
        if z < 1:
            return None
        return z, charge
    except Exception:
        return None


def _interp_table_jax(x: jnp.ndarray, grid: jnp.ndarray, table: jnp.ndarray) -> jnp.ndarray:
    """
    Piecewise-linear interpolation on the last axis of `table`.
    """
    x = jnp.clip(x, grid[0], grid[-1])
    idx = jnp.searchsorted(grid, x, side="right") - 1
    idx = jnp.clip(idx, 0, grid.shape[0] - 2)
    x0 = grid[idx]
    x1 = grid[idx + 1]
    denom = jnp.where((x1 - x0) != 0.0, (x1 - x0), 1.0)
    t = (x - x0) / denom
    y0 = jnp.take(table, idx, axis=-1)
    y1 = jnp.take(table, idx + 1, axis=-1)
    return y0 + t * (y1 - y0)


def pack_line_tensors(
    linelist: Optional[Iterable[Any]],
    species_layout: DenseSpeciesLayout,
) -> Optional[LineTensorPack]:
    """
    Pack a mixed-format linelist into static tensors.

    Unknown species (not present in `species_layout`) are skipped.
    """
    if linelist is None:
        return None

    sigma_line_const = PI * electron_charge_cgs**2 / (electron_mass_cgs * c_cgs**2)

    wavelength = []
    gf = []
    log_gf_base = []
    E_lower = []
    delta_E = []
    gamma_rad = []
    gamma_stark = []
    vdw_sigma = []
    vdw_alpha = []
    gamma_factor = []
    vdw_base_gamma = []
    species_idx = []
    atomic_mass = []
    is_molecule = []
    cross_section = []
    source_line_index = []

    for line_i, line in enumerate(linelist):
        wl_val = _line_wavelength_cm(line)
        if wl_val is None:
            continue
        species = getattr(line, "species", None)
        if species is None:
            continue
        idx = species_layout.index.get(species)
        if idx is None:
            continue

        log_gf_val = float(getattr(line, "log_gf", np.nan))
        E_lower_val = float(getattr(line, "E_lower", np.nan))
        if not np.isfinite(log_gf_val) or not np.isfinite(E_lower_val):
            continue

        gamma_rad_val = float(getattr(line, "gamma_rad", 6.16e7))
        if not np.isfinite(gamma_rad_val) or gamma_rad_val < 0.0:
            gamma_rad_val = 6.16e7
        gamma_stark_val = float(getattr(line, "gamma_stark", 0.0))
        if not np.isfinite(gamma_stark_val) or gamma_stark_val < 0.0:
            gamma_stark_val = 0.0

        vdws, vdwa = _line_vdw_params(line)
        if vdwa == -2.0:
            # Unsold-style mode needs a pre-factor from detailed broadening;
            # keep a stable fallback in this experimental backend.
            vdw_base = 1.0
        else:
            vdw_base = 1.0

        mass_cgs = float(getattr(species, "mass", 1.0)) * amu_cgs
        if not np.isfinite(mass_cgs) or mass_cgs <= 0.0:
            mass_cgs = amu_cgs

        wavelength.append(wl_val)
        gf.append(float(10.0 ** log_gf_val))
        log_gf_base.append(log_gf_val)
        E_lower.append(E_lower_val)
        delta_E.append(float(hplanck_eV * c_cgs / wl_val))
        gamma_rad.append(gamma_rad_val)
        gamma_stark.append(gamma_stark_val)
        vdw_sigma.append(vdws)
        vdw_alpha.append(vdwa)
        gamma_factor.append(float(jax_gamma((4.0 - vdwa) / 2.0)))
        vdw_base_gamma.append(vdw_base)
        species_idx.append(int(idx))
        atomic_mass.append(mass_cgs)
        is_molecule.append(float(not bool(getattr(species, "is_atom", True))))
        cross_section.append(float(sigma_line_const * wl_val * wl_val))
        source_line_index.append(int(line_i))

    if not wavelength:
        return None

    return LineTensorPack(
        wavelength=jnp.asarray(np.asarray(wavelength, dtype=np.float64)),
        gf=jnp.asarray(np.asarray(gf, dtype=np.float64)),
        log_gf_base=jnp.asarray(np.asarray(log_gf_base, dtype=np.float64)),
        E_lower=jnp.asarray(np.asarray(E_lower, dtype=np.float64)),
        delta_E=jnp.asarray(np.asarray(delta_E, dtype=np.float64)),
        gamma_rad=jnp.asarray(np.asarray(gamma_rad, dtype=np.float64)),
        gamma_stark=jnp.asarray(np.asarray(gamma_stark, dtype=np.float64)),
        vdw_sigma=jnp.asarray(np.asarray(vdw_sigma, dtype=np.float64)),
        vdw_alpha=jnp.asarray(np.asarray(vdw_alpha, dtype=np.float64)),
        gamma_factor=jnp.asarray(np.asarray(gamma_factor, dtype=np.float64)),
        vdw_base_gamma=jnp.asarray(np.asarray(vdw_base_gamma, dtype=np.float64)),
        species_idx=jnp.asarray(np.asarray(species_idx, dtype=np.int32)),
        atomic_mass=jnp.asarray(np.asarray(atomic_mass, dtype=np.float64)),
        is_molecule=jnp.asarray(np.asarray(is_molecule, dtype=np.float64)),
        cross_section=jnp.asarray(np.asarray(cross_section, dtype=np.float64)),
        source_line_index=jnp.asarray(np.asarray(source_line_index, dtype=np.int32)),
    )


def _continuum_at_line_centers(
    wl_array: jnp.ndarray,
    line_wavelengths: jnp.ndarray,
    continuum_opacity: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolate continuum opacity to each line center."""
    n_wavelengths = wl_array.shape[0]
    if n_wavelengths == 1:
        return jnp.tile(continuum_opacity[:, 0], (line_wavelengths.shape[0], 1))

    idx = jnp.searchsorted(wl_array, line_wavelengths)
    idx = jnp.clip(idx, 1, n_wavelengths - 1)
    x0 = wl_array[idx - 1]
    x1 = wl_array[idx]
    dx = x1 - x0
    frac = jnp.where(dx != 0.0, (line_wavelengths - x0) / dx, 0.0)
    cont0 = jnp.take(continuum_opacity, idx - 1, axis=1)
    cont1 = jnp.take(continuum_opacity, idx, axis=1)
    cont = cont0 + (cont1 - cont0) * frac[None, :]

    below = line_wavelengths <= wl_array[0]
    above = line_wavelengths >= wl_array[-1]
    cont = jnp.where(below[None, :], continuum_opacity[:, 0][:, None], cont)
    cont = jnp.where(above[None, :], continuum_opacity[:, -1][:, None], cont)
    return jnp.transpose(cont, (1, 0))


def _build_species_partition_logu_table(
    *,
    partition_funcs: Dict[Any, Any],
    species_layout: DenseSpeciesLayout,
) -> SpeciesPartitionLogUTable:
    """
    Build species-aligned log(U) lookup table from statmech partition data.

    Table build is static (Python/NumPy). Runtime interpolation remains JAX-native.
    """
    from ..statmech.chem_eq_jax import prepare_partition_function_tables

    pf_tables = prepare_partition_function_tables(partition_funcs, oversample=10)
    logT_grid = np.asarray(pf_tables.logT_grid, dtype=np.float64)
    n_species = len(species_layout.species)
    n_logT = int(logT_grid.size)
    logU = np.zeros((n_species, n_logT), dtype=np.float64)

    for species, idx in species_layout.index.items():
        ident = _infer_species_identity(species)
        if ident is None:
            # Molecules/unknown species default to logU=0 => U=1.
            continue
        z, charge = ident
        z_idx = z - 1
        if charge == 0:
            if bool(pf_tables.mask_I[z_idx]):
                logU[idx, :] = np.asarray(pf_tables.logU_I_table[z_idx], dtype=np.float64)
        elif charge == 1:
            if bool(pf_tables.mask_II[z_idx]):
                logU[idx, :] = np.asarray(pf_tables.logU_II_table[z_idx], dtype=np.float64)
        elif charge == 2:
            if bool(pf_tables.mask_III[z_idx]):
                logU[idx, :] = np.asarray(pf_tables.logU_III_table[z_idx], dtype=np.float64)

    return SpeciesPartitionLogUTable(
        logT_grid=jnp.asarray(logT_grid, dtype=jnp.float64),
        logU_by_species=jnp.asarray(logU, dtype=jnp.float64),
    )


def _get_species_partition_logu_table(
    *,
    partition_funcs: Dict[Any, Any],
    species_layout: DenseSpeciesLayout,
) -> SpeciesPartitionLogUTable:
    """
    Return cached species-aligned partition-function log(U) table.
    """
    species_key = tuple(
        _infer_species_identity(species) or (-1, 0)
        for species in species_layout.species
    )
    cache_key = (id(partition_funcs), species_key)
    cached = _PARTITION_LOGU_CACHE.get(cache_key)
    if cached is not None:
        return cached
    table = _build_species_partition_logu_table(
        partition_funcs=partition_funcs,
        species_layout=species_layout,
    )
    _PARTITION_LOGU_CACHE[cache_key] = table
    return table


def compute_line_opacity_jax(
    *,
    wl_array: jnp.ndarray,
    temps: jnp.ndarray,
    electron_densities: jnp.ndarray,
    number_density_dense: jnp.ndarray,
    species_layout: DenseSpeciesLayout,
    partition_funcs: Dict[Any, Any],
    linelist: Optional[Iterable[Any]],
    microturbulence_kms: float,
    continuum_opacity: jnp.ndarray,
    line_loggf_deltas: Optional[jnp.ndarray] = None,
    cutoff_threshold: float = 3e-4,
    soft_window_temperature: float = 3.0,
) -> jnp.ndarray:
    """
    Compute line opacity matrix [layers, wavelengths] with a JAX-native core.

    The static line packing and partition-function lookup are done in Python.
    All profile accumulation is done with JAX tensors and control flow.
    """
    wl_array = jnp.asarray(wl_array, dtype=jnp.float64)
    # NOTE: line-profile kernels use cgs wavelengths (cm) internally.
    wl_array_cm = wl_array * 1e-8
    temps = jnp.asarray(temps, dtype=jnp.float64)
    electron_densities = jnp.asarray(electron_densities, dtype=jnp.float64)
    number_density_dense = jnp.asarray(number_density_dense, dtype=jnp.float64)
    continuum_opacity = jnp.asarray(continuum_opacity, dtype=jnp.float64)

    if linelist is None:
        line_sequence = None
    elif isinstance(linelist, (list, tuple)):
        line_sequence = linelist
    else:
        line_sequence = list(linelist)

    line_count = 0 if line_sequence is None else len(line_sequence)
    if line_loggf_deltas is not None:
        line_loggf_deltas = jnp.asarray(line_loggf_deltas, dtype=jnp.float64)
        if line_loggf_deltas.ndim != 1:
            raise ValueError("line_loggf_deltas must be a 1-D array when provided.")
        if line_loggf_deltas.shape[0] != line_count:
            raise ValueError(
                "line_loggf_deltas length must match input linelist length "
                f"(got {line_loggf_deltas.shape[0]}, expected {line_count})."
            )

    pack = pack_line_tensors(line_sequence, species_layout)
    if pack is None or pack.n_lines == 0:
        return jnp.zeros_like(continuum_opacity)

    if line_loggf_deltas is None:
        packed_loggf_deltas = jnp.zeros((pack.n_lines,), dtype=jnp.float64)
    else:
        packed_loggf_deltas = jnp.take(line_loggf_deltas, pack.source_line_index, axis=0)

    partition_table = _get_species_partition_logu_table(
        partition_funcs=partition_funcs,
        species_layout=species_layout,
    )
    logT = jnp.log(jnp.clip(temps, 1e-300, None))
    logU_dense = jax.vmap(
        lambda logTi: _interp_table_jax(logTi, partition_table.logT_grid, partition_table.logU_by_species),
        in_axes=0,
        out_axes=1,
    )(logT)

    # n/U by line and layer: [n_lines, n_layers]
    line_density = jnp.take(number_density_dense, pack.species_idx, axis=1).T
    line_logU = jnp.take(logU_dense, pack.species_idx, axis=0)
    line_pf = jnp.exp(jnp.clip(line_logU, -700.0, 700.0))
    n_div_u = line_density / jnp.maximum(line_pf, 1e-30)

    beta = 1.0 / (kboltz_eV * temps)
    vmic_cm_s = jnp.asarray(microturbulence_kms, dtype=jnp.float64) * 1e5

    sigma = pack.wavelength[:, None] * jnp.sqrt(
        kboltz_cgs * temps[None, :] / jnp.maximum(pack.atomic_mass[:, None], 1e-30)
        + (vmic_cm_s**2) / 2.0
    ) / c_cgs
    sigma = jnp.maximum(sigma, 1e-30)

    temp_stark = (temps / 10000.0) ** (1.0 / 6.0)
    temp_vdw = (temps / 10000.0) ** 0.3
    temp_vbar = jnp.sqrt(8.0 * kboltz_cgs * temps / PI)

    is_atom = 1.0 - pack.is_molecule
    gamma_total = pack.gamma_rad[:, None] + (
        electron_densities[None, :] * (pack.gamma_stark[:, None] * temp_stark[None, :]) * is_atom[:, None]
    )

    h_neutral_idx = species_layout.index.get(Species.from_atomic_number(1, 0))
    if h_neutral_idx is None:
        n_h_neutral = jnp.zeros_like(temps)
    else:
        n_h_neutral = number_density_dense[:, int(h_neutral_idx)]

    inv_mu_const = 1.0 / (1.008 * amu_cgs)
    inv_mu = inv_mu_const + 1.0 / jnp.maximum(pack.atomic_mass, 1e-30)
    vbar = temp_vbar[None, :] * jnp.sqrt(inv_mu[:, None])
    v0 = 1e6

    vdw_abo = (
        2.0
        * (4.0 / PI) ** (pack.vdw_alpha[:, None] / 2.0)
        * pack.gamma_factor[:, None]
        * v0
        * pack.vdw_sigma[:, None]
        * (vbar / v0) ** (1.0 - pack.vdw_alpha[:, None])
    )
    vdw_simple = pack.vdw_sigma[:, None] * temp_vdw[None, :]
    vdw_unsold = pack.vdw_sigma[:, None] * pack.vdw_base_gamma[:, None] * temp_vdw[None, :]
    vdw_gamma = jnp.where(
        pack.vdw_alpha[:, None] == -1.0,
        vdw_simple,
        jnp.where(pack.vdw_alpha[:, None] == -2.0, vdw_unsold, vdw_abo),
    )
    gamma_total = gamma_total + (n_h_neutral[None, :] * vdw_gamma) * is_atom[:, None]

    gamma = gamma_total * pack.wavelength[:, None] ** 2 / (4.0 * PI * c_cgs)
    gamma = jnp.maximum(gamma, 1e-30)

    E_upper = pack.E_lower + pack.delta_E
    levels_factor = jnp.exp(-beta[None, :] * pack.E_lower[:, None]) - jnp.exp(
        -beta[None, :] * E_upper[:, None]
    )
    loggf_eff = jnp.clip(pack.log_gf_base + packed_loggf_deltas, -300.0, 300.0)
    gf_eff = jnp.exp(jnp.log(10.0) * loggf_eff)
    amplitude = gf_eff[:, None] * pack.cross_section[:, None] * levels_factor * n_div_u

    continuum_line = _continuum_at_line_centers(wl_array_cm, pack.wavelength, continuum_opacity)
    rho_crit = (continuum_line * cutoff_threshold) / jnp.maximum(jnp.abs(amplitude), 1e-50)

    sqrt_2pi = jnp.sqrt(2.0 * PI)
    threshold_g = 1.0 / (sqrt_2pi * sigma)
    safe_g = jnp.clip(sqrt_2pi * sigma * rho_crit, 1e-300, 1.0 - 1e-12)
    doppler_arg = jnp.maximum(-2.0 * jnp.log(safe_g), 1e-12)
    doppler_val = sigma * jnp.sqrt(doppler_arg)
    doppler_windows = jnp.where(rho_crit <= threshold_g, doppler_val, 0.0)

    threshold_l = 1.0 / (PI * gamma)
    safe_rho = jnp.maximum(rho_crit, 1e-300)
    lorentz_arg = gamma / (PI * safe_rho) - gamma * gamma
    lorentz_val = jnp.sqrt(jnp.maximum(lorentz_arg, 1e-12))
    lorentz_windows = jnp.where(rho_crit <= threshold_l, lorentz_val, 0.0)

    doppler_window = jnp.max(doppler_windows, axis=1)
    lorentz_window = jnp.max(lorentz_windows, axis=1)
    window_size = jnp.sqrt(doppler_window**2 + lorentz_window**2)

    sigma_ref = jnp.mean(sigma, axis=1)
    gate_scale = jnp.maximum(soft_window_temperature * sigma_ref, 1e-30)
    distance = jnp.abs(wl_array_cm[None, :] - pack.wavelength[:, None])
    soft_gate = jax.nn.sigmoid((window_size[:, None] - distance) / gate_scale[:, None])

    def _add_one_line(alpha_acc: jnp.ndarray, line_idx: jnp.ndarray):
        amp_i = amplitude[line_idx]
        sigma_i = sigma[line_idx]
        gamma_i = gamma[line_idx]
        wl0 = pack.wavelength[line_idx]
        gate = soft_gate[line_idx]

        line_alpha = jax.vmap(
            lambda amp_layer, sigma_layer, gamma_layer: line_profile(
                wl0,
                jnp.maximum(sigma_layer, 1e-30),
                jnp.maximum(gamma_layer, 1e-30),
                amp_layer,
                wl_array_cm,
            ),
            in_axes=(0, 0, 0),
            out_axes=0,
        )(amp_i, sigma_i, gamma_i)
        line_alpha = jnp.nan_to_num(line_alpha, nan=0.0, posinf=0.0, neginf=0.0)
        alpha_next = alpha_acc + line_alpha * gate[None, :]
        return alpha_next, None

    alpha0 = jnp.zeros_like(continuum_opacity)
    alpha_matrix, _ = jax.lax.scan(_add_one_line, alpha0, jnp.arange(pack.n_lines))
    return alpha_matrix


__all__ = ["LineTensorPack", "pack_line_tensors", "compute_line_opacity_jax"]
