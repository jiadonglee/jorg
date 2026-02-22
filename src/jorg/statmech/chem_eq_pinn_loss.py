"""Physics-constrained losses for chemical-equilibrium PINN training."""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from ..constants import hplanck_cgs, kboltz_cgs, kboltz_eV, me_cgs
from .chem_eq_jax import (
    ChemEqData,
    _LOG_MAX,
    _LOG_MIN,
    _eval_cubic_jax,
    _interp_table_jax,
)
from .species import MAX_ATOMIC_NUMBER


def _prepare_atomic_inputs(
    neutral_fractions: jnp.ndarray,
    ne: jnp.ndarray,
    T: jnp.ndarray,
    n_total: jnp.ndarray,
    abundances: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, bool]:
    """Broadcast scalar/batched inputs into consistent batch-shaped arrays."""

    neutral_fractions = jnp.asarray(neutral_fractions, dtype=jnp.float64)
    ne = jnp.asarray(ne, dtype=jnp.float64)
    T = jnp.asarray(T, dtype=jnp.float64)
    n_total = jnp.asarray(n_total, dtype=jnp.float64)
    abundances = jnp.asarray(abundances, dtype=jnp.float64)

    input_is_scalar = (
        neutral_fractions.ndim == 1
        and abundances.ndim == 1
        and ne.ndim == 0
        and T.ndim == 0
        and n_total.ndim == 0
    )

    if neutral_fractions.ndim == 1:
        neutral_fractions = neutral_fractions[None, :]
    if abundances.ndim == 1:
        abundances = abundances[None, :]

    if neutral_fractions.ndim != 2 or abundances.ndim != 2:
        raise ValueError("neutral_fractions and abundances must be 1D or 2D arrays.")

    if neutral_fractions.shape[-1] != MAX_ATOMIC_NUMBER:
        raise ValueError(
            f"neutral_fractions must have trailing size {MAX_ATOMIC_NUMBER}, "
            f"got {neutral_fractions.shape[-1]}."
        )

    if abundances.shape[-1] != MAX_ATOMIC_NUMBER:
        raise ValueError(
            f"abundances must have trailing size {MAX_ATOMIC_NUMBER}, "
            f"got {abundances.shape[-1]}."
        )

    batch_size = neutral_fractions.shape[0]
    if abundances.shape[0] not in (1, batch_size):
        raise ValueError("abundances batch size must be 1 or equal to neutral_fractions batch size.")

    if abundances.shape[0] == 1 and batch_size > 1:
        abundances = jnp.broadcast_to(abundances, (batch_size, MAX_ATOMIC_NUMBER))

    def _to_batch_1d(x: jnp.ndarray, name: str) -> jnp.ndarray:
        if x.ndim == 0:
            return jnp.full((batch_size,), x)
        if x.ndim == 1 and x.shape[0] == batch_size:
            return x
        raise ValueError(f"{name} must be scalar or shape (batch,).")

    ne = _to_batch_1d(ne, "ne")
    T = _to_batch_1d(T, "T")
    n_total = _to_batch_1d(n_total, "n_total")

    return neutral_fractions, ne, T, n_total, abundances, input_is_scalar


def _compute_saha_weights_single(T: jnp.ndarray, ne: jnp.ndarray, chem_data: ChemEqData):
    logT = jnp.log(T)

    logT_grid = jnp.asarray(chem_data.pf.logT_grid)
    logU_I_table = jnp.asarray(chem_data.pf.logU_I_table)
    logU_II_table = jnp.asarray(chem_data.pf.logU_II_table)
    logU_III_table = jnp.asarray(chem_data.pf.logU_III_table)

    cubic_logT_grid = None
    U_I_coeffs = None
    U_II_coeffs = None
    U_III_coeffs = None
    cubic_mask_I = None
    cubic_mask_II = None
    cubic_mask_III = None

    if chem_data.pf.cubic_logT_grid is not None and chem_data.pf.U_I_coeffs is not None:
        cubic_logT_grid = jnp.asarray(chem_data.pf.cubic_logT_grid)
        U_I_coeffs = jnp.asarray(chem_data.pf.U_I_coeffs)
        U_II_coeffs = jnp.asarray(chem_data.pf.U_II_coeffs)
        U_III_coeffs = jnp.asarray(chem_data.pf.U_III_coeffs)
        if chem_data.pf.cubic_mask_I is not None:
            cubic_mask_I = jnp.asarray(chem_data.pf.cubic_mask_I)
            cubic_mask_II = jnp.asarray(chem_data.pf.cubic_mask_II)
            cubic_mask_III = jnp.asarray(chem_data.pf.cubic_mask_III)

    chi1 = jnp.asarray(chem_data.ion.chi1)
    chi2 = jnp.asarray(chem_data.ion.chi2)

    mask1 = (
        jnp.asarray(chem_data.ion.mask1)
        & jnp.asarray(chem_data.pf.mask_I)
        & jnp.asarray(chem_data.pf.mask_II)
    )
    mask2 = (
        jnp.asarray(chem_data.ion.mask2)
        & jnp.asarray(chem_data.pf.mask_II)
        & jnp.asarray(chem_data.pf.mask_III)
    )

    logU_I = _interp_table_jax(logT, logT_grid, logU_I_table)
    logU_II = _interp_table_jax(logT, logT_grid, logU_II_table)
    logU_III = _interp_table_jax(logT, logT_grid, logU_III_table)

    if cubic_logT_grid is not None and U_I_coeffs is not None:
        U_I = _eval_cubic_jax(logT, cubic_logT_grid, U_I_coeffs)
        U_II = _eval_cubic_jax(logT, cubic_logT_grid, U_II_coeffs)
        U_III = _eval_cubic_jax(logT, cubic_logT_grid, U_III_coeffs)

        logU_I_c = jnp.log(jnp.clip(U_I, 1e-300, None))
        logU_II_c = jnp.log(jnp.clip(U_II, 1e-300, None))
        logU_III_c = jnp.log(jnp.clip(U_III, 1e-300, None))

        if cubic_mask_I is not None:
            logU_I = jnp.where(cubic_mask_I, logU_I_c, logU_I)
            logU_II = jnp.where(cubic_mask_II, logU_II_c, logU_II)
            logU_III = jnp.where(cubic_mask_III, logU_III_c, logU_III)
        else:
            logU_I = logU_I_c
            logU_II = logU_II_c
            logU_III = logU_III_c

    log_trans_U = 1.5 * jnp.log(2.0 * jnp.pi * me_cgs * kboltz_cgs * T / (hplanck_cgs ** 2))
    inv_kT = 1.0 / (kboltz_eV * T)

    log2 = jnp.log(2.0)
    logC1 = log2 + logU_II - logU_I + log_trans_U - chi1 * inv_kT
    logC2 = logC1 + log2 + logU_III - logU_II + log_trans_U - chi2 * inv_kT

    logC1 = jnp.where(mask1, logC1, -jnp.inf)
    logC2 = jnp.where(mask2, logC2, -jnp.inf)

    log_ne = jnp.log(jnp.clip(ne, 1e-300, None))
    log_wII = jnp.clip(logC1 - log_ne, _LOG_MIN, _LOG_MAX)
    log_wIII = jnp.clip(logC2 - 2.0 * log_ne, _LOG_MIN, _LOG_MAX)

    wII = jnp.where(mask1, jnp.exp(log_wII), 0.0)
    wIII = jnp.where(mask2, jnp.exp(log_wIII), 0.0)
    return wII, wIII


def compute_saha_weights_jax(
    T: jnp.ndarray,
    ne: jnp.ndarray,
    chem_data: ChemEqData,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Saha ionization weight arrays for all 92 elements."""

    T = jnp.asarray(T, dtype=jnp.float64)
    ne = jnp.asarray(ne, dtype=jnp.float64)

    if T.ndim == 0 and ne.ndim == 0:
        return _compute_saha_weights_single(T, ne, chem_data)

    if T.ndim == 0:
        T = jnp.broadcast_to(T, ne.shape)
    if ne.ndim == 0:
        ne = jnp.broadcast_to(ne, T.shape)

    if T.shape != ne.shape:
        raise ValueError("T and ne must have matching shapes after broadcasting.")

    flat_T = T.reshape((-1,))
    flat_ne = ne.reshape((-1,))

    wII, wIII = jax.vmap(
        lambda t_i, ne_i: _compute_saha_weights_single(t_i, ne_i, chem_data)
    )(flat_T, flat_ne)

    out_shape = T.shape + (MAX_ATOMIC_NUMBER,)
    return wII.reshape(out_shape), wIII.reshape(out_shape)


def compute_atomic_species_densities(
    neutral_fractions: jnp.ndarray,
    ne: jnp.ndarray,
    T: jnp.ndarray,
    n_total: jnp.ndarray,
    abundances: jnp.ndarray,
    chem_data: ChemEqData,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute atomic nucleus, neutral, singly and doubly-ionized number densities."""

    neutral_fractions, ne, T, n_total, abundances, _ = _prepare_atomic_inputs(
        neutral_fractions, ne, T, n_total, abundances
    )

    wII, wIII = compute_saha_weights_jax(T, ne, chem_data)

    n_nuc = abundances * (n_total[:, None] - ne[:, None])
    n0 = n_nuc * neutral_fractions
    n1 = wII * n0
    n2 = wIII * n0
    return n_nuc, n0, n1, n2


def element_conservation_residual(
    neutral_fractions: jnp.ndarray,
    ne: jnp.ndarray,
    T: jnp.ndarray,
    n_total: jnp.ndarray,
    abundances: jnp.ndarray,
    chem_data: ChemEqData,
) -> jnp.ndarray:
    """Residual of elemental conservation equation normalized by nucleus density."""

    neutral_fractions, ne, T, n_total, abundances, input_is_scalar = _prepare_atomic_inputs(
        neutral_fractions, ne, T, n_total, abundances
    )

    _, n0, n1, n2 = compute_atomic_species_densities(
        neutral_fractions, ne, T, n_total, abundances, chem_data
    )

    n_nuc = abundances * (n_total[:, None] - ne[:, None])
    total_pred = n0 + n1 + n2
    denom = jnp.maximum(n_nuc, 1e-100)
    residual = (total_pred - n_nuc) / denom

    if input_is_scalar:
        return residual[0]
    return residual


def charge_neutrality_residual(
    neutral_fractions: jnp.ndarray,
    ne: jnp.ndarray,
    T: jnp.ndarray,
    n_total: jnp.ndarray,
    abundances: jnp.ndarray,
    chem_data: ChemEqData,
) -> jnp.ndarray:
    """Residual of charge neutrality equation normalized by electron density."""

    neutral_fractions, ne, T, n_total, abundances, input_is_scalar = _prepare_atomic_inputs(
        neutral_fractions, ne, T, n_total, abundances
    )

    _, _, n1, n2 = compute_atomic_species_densities(
        neutral_fractions, ne, T, n_total, abundances, chem_data
    )

    positive_charge = jnp.sum(n1 + 2.0 * n2, axis=-1)
    residual = (positive_charge - ne) / jnp.maximum(ne, 1e-100)

    if input_is_scalar:
        return residual[0]
    return residual


def element_conservation_loss(
    neutral_fractions: jnp.ndarray,
    ne: jnp.ndarray,
    T: jnp.ndarray,
    n_total: jnp.ndarray,
    abundances: jnp.ndarray,
    chem_data: ChemEqData,
) -> jnp.ndarray:
    """Weighted MSE of elemental conservation residuals."""

    abundances = jnp.asarray(abundances, dtype=jnp.float64)
    residual = element_conservation_residual(
        neutral_fractions, ne, T, n_total, abundances, chem_data
    )

    if residual.ndim == 1:
        residual = residual[None, :]
        abundances = abundances[None, :] if abundances.ndim == 1 else abundances

    if abundances.ndim == 1:
        abundances = jnp.broadcast_to(abundances[None, :], residual.shape)

    weights = jnp.clip(abundances, 1e-12, None)
    weights = weights / jnp.mean(weights, axis=-1, keepdims=True)
    sample_losses = jnp.mean(weights * residual ** 2, axis=-1)
    return jnp.mean(sample_losses)


def charge_neutrality_loss(
    neutral_fractions: jnp.ndarray,
    ne: jnp.ndarray,
    T: jnp.ndarray,
    n_total: jnp.ndarray,
    abundances: jnp.ndarray,
    chem_data: ChemEqData,
) -> jnp.ndarray:
    """MSE of charge neutrality residuals."""

    residual = charge_neutrality_residual(
        neutral_fractions, ne, T, n_total, abundances, chem_data
    )
    return jnp.mean(residual ** 2)


def total_physics_loss(
    neutral_fractions: jnp.ndarray,
    ne: jnp.ndarray,
    T: jnp.ndarray,
    n_total: jnp.ndarray,
    abundances: jnp.ndarray,
    chem_data: ChemEqData,
    *,
    w_element: float = 1.0,
    w_charge: float = 10.0,
) -> Tuple[jnp.ndarray, dict]:
    """Combined physics-informed loss and components."""

    elem = element_conservation_loss(neutral_fractions, ne, T, n_total, abundances, chem_data)
    charge = charge_neutrality_loss(neutral_fractions, ne, T, n_total, abundances, chem_data)
    total = w_element * elem + w_charge * charge
    return total, {"element": elem, "charge": charge}


__all__ = [
    "compute_saha_weights_jax",
    "compute_atomic_species_densities",
    "element_conservation_residual",
    "charge_neutrality_residual",
    "element_conservation_loss",
    "charge_neutrality_loss",
    "total_physics_loss",
]
