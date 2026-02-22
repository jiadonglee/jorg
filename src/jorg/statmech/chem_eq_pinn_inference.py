"""Inference helpers to inject PINN CE outputs into synthesis.

This module bridges a trained chemical-equilibrium PINN checkpoint into the
``use_chemical_equilibrium_from`` interface expected by synthesis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .chem_eq_jax import ChemEqData
from .chem_eq_pinn import ChemicalEquilibriumPINNSolver
from .chem_eq_pinn_loss import compute_atomic_species_densities
from .species import MAX_ATOMIC_NUMBER, Species


def _normalize_inputs(
    temperatures: np.ndarray,
    n_totals: np.ndarray,
    abundances: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    temps = np.asarray(temperatures, dtype=np.float64)
    totals = np.asarray(n_totals, dtype=np.float64)
    abund = np.asarray(abundances, dtype=np.float64)

    if temps.ndim != 1:
        raise ValueError("temperatures must be a 1-D array.")
    if totals.ndim != 1:
        raise ValueError("n_totals must be a 1-D array.")
    if temps.shape[0] != totals.shape[0]:
        raise ValueError("temperatures and n_totals must have the same length.")
    if np.any(~np.isfinite(temps)) or np.any(temps <= 0.0):
        raise ValueError("temperatures must be finite and positive.")
    if np.any(~np.isfinite(totals)) or np.any(totals <= 0.0):
        raise ValueError("n_totals must be finite and positive.")

    n_layers = temps.shape[0]
    if abund.ndim == 1:
        if abund.shape[0] != MAX_ATOMIC_NUMBER:
            raise ValueError(
                f"abundances 1-D input must have length {MAX_ATOMIC_NUMBER}, "
                f"got {abund.shape[0]}."
            )
        abund = np.broadcast_to(abund[None, :], (n_layers, MAX_ATOMIC_NUMBER)).copy()
    elif abund.ndim == 2:
        if abund.shape != (n_layers, MAX_ATOMIC_NUMBER):
            raise ValueError(
                "abundances 2-D input must have shape "
                f"({n_layers}, {MAX_ATOMIC_NUMBER}), got {abund.shape}."
            )
    else:
        raise ValueError("abundances must be a 1-D or 2-D array.")

    row_sums = np.sum(abund, axis=1)
    if np.any(~np.isfinite(abund)) or np.any(abund < 0.0):
        raise ValueError("abundances must be finite and non-negative.")
    if np.any(row_sums <= 0.0):
        raise ValueError("abundances rows must have positive sums.")

    abund /= row_sums[:, None]
    return temps, totals, abund


def _validate_prediction_shapes(
    ne_pred: np.ndarray,
    neutral_fractions: np.ndarray,
    n_layers: int,
):
    if ne_pred.shape != (n_layers,):
        raise RuntimeError(
            f"Unexpected ne prediction shape {ne_pred.shape}; expected {(n_layers,)}."
        )
    if neutral_fractions.shape != (n_layers, MAX_ATOMIC_NUMBER):
        raise RuntimeError(
            "Unexpected neutral-fraction prediction shape "
            f"{neutral_fractions.shape}; expected {(n_layers, MAX_ATOMIC_NUMBER)}."
        )
    if np.any(~np.isfinite(ne_pred)) or np.any(ne_pred <= 0.0):
        raise RuntimeError("PINN predicted non-finite or non-positive electron densities.")
    if np.any(~np.isfinite(neutral_fractions)):
        raise RuntimeError("PINN predicted non-finite neutral fractions.")


def load_pinn_solver_from_checkpoint(
    model_path: str | Path,
    chem_data: ChemEqData,
) -> ChemicalEquilibriumPINNSolver:
    """Load and return a PINN solver instance from checkpoint."""

    ckpt = Path(model_path).expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"PINN checkpoint not found: {ckpt}")
    return ChemicalEquilibriumPINNSolver.load(str(ckpt), chem_data=chem_data)


def predict_atomic_ce_from_solver(
    solver: ChemicalEquilibriumPINNSolver,
    temperatures: np.ndarray,
    n_totals: np.ndarray,
    abundances: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict (ne, neutral_fractions) for all layers using a loaded solver."""

    temps, totals, abund = _normalize_inputs(temperatures, n_totals, abundances)
    ne_pred, neutral_fractions = solver.solve_batch(temps, totals, abund)
    ne_pred = np.asarray(ne_pred, dtype=np.float64)
    neutral_fractions = np.asarray(neutral_fractions, dtype=np.float64)
    _validate_prediction_shapes(ne_pred, neutral_fractions, temps.shape[0])
    return ne_pred, neutral_fractions


def predict_atomic_ce_from_checkpoint(
    model_path: str | Path,
    temperatures: np.ndarray,
    n_totals: np.ndarray,
    abundances: np.ndarray,
    chem_data: ChemEqData,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict (ne, neutral_fractions) for all layers from a PINN checkpoint."""

    solver = load_pinn_solver_from_checkpoint(model_path=model_path, chem_data=chem_data)
    return predict_atomic_ce_from_solver(
        solver=solver,
        temperatures=temperatures,
        n_totals=n_totals,
        abundances=abundances,
    )


def build_atomic_ce_source_from_solver(
    solver: ChemicalEquilibriumPINNSolver,
    temperatures: np.ndarray,
    n_totals: np.ndarray,
    abundances: np.ndarray,
    chem_data: ChemEqData,
) -> Dict[str, object]:
    """Build synthesis CE-source dict from a loaded PINN solver."""

    temps, totals, abund = _normalize_inputs(temperatures, n_totals, abundances)
    ne_pred, neutral_fractions = predict_atomic_ce_from_solver(
        solver=solver,
        temperatures=temps,
        n_totals=totals,
        abundances=abund,
    )

    _, n0, n1, n2 = compute_atomic_species_densities(
        neutral_fractions,
        ne_pred,
        temps,
        totals,
        abund,
        chem_data,
    )

    n0 = np.asarray(n0, dtype=np.float64)
    n1 = np.asarray(n1, dtype=np.float64)
    n2 = np.asarray(n2, dtype=np.float64)

    number_densities: Dict[Species, np.ndarray] = {}
    for z in range(1, MAX_ATOMIC_NUMBER + 1):
        idx = z - 1
        number_densities[Species.from_atomic_number(z, 0)] = n0[:, idx]
        number_densities[Species.from_atomic_number(z, 1)] = n1[:, idx]
        number_densities[Species.from_atomic_number(z, 2)] = n2[:, idx]

    return {
        "electron_densities": np.asarray(ne_pred, dtype=np.float64),
        "number_densities": number_densities,
    }


def build_atomic_ce_source_from_checkpoint(
    model_path: str | Path,
    temperatures: np.ndarray,
    n_totals: np.ndarray,
    abundances: np.ndarray,
    chem_data: ChemEqData,
) -> Dict[str, object]:
    """Build synthesis CE-source dict from a PINN checkpoint.

    Returns a dictionary compatible with ``use_chemical_equilibrium_from``:
    ``{"electron_densities": ne, "number_densities": species_map}``.
    """

    solver = load_pinn_solver_from_checkpoint(model_path=model_path, chem_data=chem_data)
    return build_atomic_ce_source_from_solver(
        solver=solver,
        temperatures=temperatures,
        n_totals=n_totals,
        abundances=abundances,
        chem_data=chem_data,
    )


__all__ = [
    "load_pinn_solver_from_checkpoint",
    "predict_atomic_ce_from_solver",
    "build_atomic_ce_source_from_solver",
    "predict_atomic_ce_from_checkpoint",
    "build_atomic_ce_source_from_checkpoint",
]
