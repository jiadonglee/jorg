"""
Joint log(gf) + stellar-parameter alternating optimization utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .stellar_parameters import StellarParameters


@dataclass
class JointFitConfig:
    """Configuration for alternating joint fitting."""

    outer_iterations: int = 3
    stellar_damping: float = 1.0
    loggf_damping: float = 1.0
    convergence_dteff: float = 5.0
    convergence_dlogg: float = 0.02
    convergence_dmh: float = 0.02
    convergence_dloggf: float = 0.01
    objective_tolerance: float = 1e-6
    consecutive_convergence_required: int = 2


@dataclass
class JointStarState:
    """Per-star state used by the joint fit loop."""

    source_id: int
    star_label: str
    initial_parameters: StellarParameters
    current_parameters: StellarParameters
    atmosphere: Any
    abundances: np.ndarray
    quality_flag: str = "ok"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JointLineState:
    """Per-line state used by the joint fit loop."""

    line_index: int
    species: str
    wavelength_A: float
    loggf_ges: float
    delta: float = 0.0
    n_used: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JointFitResult:
    """Output container for joint fitting."""

    config: JointFitConfig
    star_states: List[JointStarState]
    line_states: List[JointLineState]
    line_deltas_dense: np.ndarray
    update_mask: np.ndarray
    history: List[Dict[str, Any]]
    star_failures: List[Dict[str, Any]]
    line_failures: List[Dict[str, Any]]
    converged: bool
    n_outer_iterations: int


StellarBlockUpdate = Callable[
    [List[JointStarState], np.ndarray, int, JointFitConfig],
    Tuple[List[JointStarState], List[Dict[str, Any]]],
]
LogGFBlockUpdate = Callable[
    [List[JointStarState], List[JointLineState], np.ndarray, np.ndarray, int, JointFitConfig],
    Tuple[np.ndarray, List[JointLineState], List[Dict[str, Any]]],
]
ObjectiveFunction = Callable[
    [List[JointStarState], List[JointLineState], np.ndarray],
    float,
]


def _copy_stellar_params(p: StellarParameters) -> StellarParameters:
    return StellarParameters(
        Teff=float(p.Teff),
        logg=float(p.logg),
        m_h=float(p.m_h),
        alpha_fe=float(p.alpha_fe),
    )


def _clone_star_state(state: JointStarState) -> JointStarState:
    return JointStarState(
        source_id=int(state.source_id),
        star_label=str(state.star_label),
        initial_parameters=_copy_stellar_params(state.initial_parameters),
        current_parameters=_copy_stellar_params(state.current_parameters),
        atmosphere=state.atmosphere,
        abundances=np.asarray(state.abundances, dtype=float).copy(),
        quality_flag=str(state.quality_flag),
        metadata=dict(state.metadata),
    )


def _clone_line_state(state: JointLineState) -> JointLineState:
    return JointLineState(
        line_index=int(state.line_index),
        species=str(state.species),
        wavelength_A=float(state.wavelength_A),
        loggf_ges=float(state.loggf_ges),
        delta=float(state.delta),
        n_used=int(state.n_used),
        metrics=dict(state.metrics),
    )


def _max_star_deltas(
    previous_states: Sequence[JointStarState],
    current_states: Sequence[JointStarState],
) -> Tuple[float, float, float]:
    prev = {int(s.source_id): s for s in previous_states}
    max_dteff = 0.0
    max_dlogg = 0.0
    max_dmh = 0.0
    for state in current_states:
        old = prev.get(int(state.source_id))
        if old is None:
            continue
        max_dteff = max(
            max_dteff, abs(float(state.current_parameters.Teff) - float(old.current_parameters.Teff))
        )
        max_dlogg = max(
            max_dlogg, abs(float(state.current_parameters.logg) - float(old.current_parameters.logg))
        )
        max_dmh = max(
            max_dmh, abs(float(state.current_parameters.m_h) - float(old.current_parameters.m_h))
        )
    return float(max_dteff), float(max_dlogg), float(max_dmh)


def fit_joint_loggf_stellar_parameters(
    *,
    config: JointFitConfig,
    star_states: Sequence[JointStarState],
    line_states: Sequence[JointLineState],
    line_deltas_dense: np.ndarray,
    update_mask: np.ndarray,
    stellar_block_update: StellarBlockUpdate,
    loggf_block_update: LogGFBlockUpdate,
    objective_function: Optional[ObjectiveFunction] = None,
) -> JointFitResult:
    """
    Run alternating stellar/log(gf) updates until convergence or max iterations.
    """
    n_lines = int(np.asarray(line_deltas_dense).size)
    line_deltas = np.asarray(line_deltas_dense, dtype=float).copy()
    if line_deltas.ndim != 1:
        raise ValueError("line_deltas_dense must be a 1-D array.")
    mask = np.asarray(update_mask, dtype=bool).reshape(-1)
    if mask.shape != line_deltas.shape:
        raise ValueError(
            "update_mask shape must match line_deltas_dense shape "
            f"(got {mask.shape}, expected {line_deltas.shape})."
        )

    star_states_local = [_clone_star_state(s) for s in star_states]
    line_states_local = [_clone_line_state(s) for s in line_states]

    history: List[Dict[str, Any]] = []
    star_failures: List[Dict[str, Any]] = []
    line_failures: List[Dict[str, Any]] = []
    prev_objective: Optional[float] = None
    consecutive_converged = 0
    converged = False
    n_completed = 0

    for outer_iter in range(1, max(1, int(config.outer_iterations)) + 1):
        n_completed = outer_iter
        prev_star_states = [_clone_star_state(s) for s in star_states_local]
        prev_line_deltas = line_deltas.copy()

        iter_star_failures: List[Dict[str, Any]] = []
        iter_line_failures: List[Dict[str, Any]] = []

        try:
            updated_star_states, reported_failures = stellar_block_update(
                star_states_local,
                line_deltas.copy(),
                outer_iter,
                config,
            )
            star_states_local = [_clone_star_state(s) for s in updated_star_states]
            for item in reported_failures:
                row = dict(item)
                row.setdefault("outer_iter", outer_iter)
                iter_star_failures.append(row)
        except Exception as exc:  # pragma: no cover - defensive path
            iter_star_failures.append(
                {
                    "outer_iter": outer_iter,
                    "scope": "stellar_block",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            star_states_local = prev_star_states

        try:
            proposed_deltas, updated_line_states, reported_failures = loggf_block_update(
                star_states_local,
                line_states_local,
                line_deltas.copy(),
                mask.copy(),
                outer_iter,
                config,
            )
            proposed_deltas = np.asarray(proposed_deltas, dtype=float).reshape(-1)
            if proposed_deltas.shape != (n_lines,):
                raise ValueError(
                    "loggf_block_update returned invalid delta vector shape "
                    f"{proposed_deltas.shape}, expected {(n_lines,)}."
                )

            damping = float(np.clip(config.loggf_damping, 0.0, 1.0))
            line_deltas[mask] = line_deltas[mask] + damping * (
                proposed_deltas[mask] - line_deltas[mask]
            )
            line_states_local = [_clone_line_state(s) for s in updated_line_states]
            for item in reported_failures:
                row = dict(item)
                row.setdefault("outer_iter", outer_iter)
                iter_line_failures.append(row)
        except Exception as exc:  # pragma: no cover - defensive path
            iter_line_failures.append(
                {
                    "outer_iter": outer_iter,
                    "scope": "loggf_block",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            line_deltas = prev_line_deltas

        for line_state in line_states_local:
            idx = int(line_state.line_index)
            if 0 <= idx < n_lines:
                line_state.delta = float(line_deltas[idx])

        max_dteff, max_dlogg, max_dmh = _max_star_deltas(prev_star_states, star_states_local)
        max_dloggf = (
            float(np.max(np.abs(line_deltas[mask] - prev_line_deltas[mask])))
            if np.any(mask)
            else 0.0
        )

        objective = np.nan
        if objective_function is not None:
            try:
                objective = float(
                    objective_function(star_states_local, line_states_local, line_deltas.copy())
                )
            except Exception:  # pragma: no cover - defensive path
                objective = np.nan

        objective_delta = np.nan
        objective_small = False
        if prev_objective is not None and np.isfinite(prev_objective) and np.isfinite(objective):
            objective_delta = float(prev_objective - objective)
            objective_small = bool(
                abs(objective_delta) <= float(max(0.0, config.objective_tolerance))
            )

        param_small = (
            max_dteff <= float(config.convergence_dteff)
            and max_dlogg <= float(config.convergence_dlogg)
            and max_dmh <= float(config.convergence_dmh)
            and max_dloggf <= float(config.convergence_dloggf)
        )
        iter_converged = bool(param_small and objective_small)
        if iter_converged:
            consecutive_converged += 1
        else:
            consecutive_converged = 0

        history.append(
            {
                "outer_iter": outer_iter,
                "objective": objective,
                "objective_delta": objective_delta,
                "max_abs_dteff": max_dteff,
                "max_abs_dlogg": max_dlogg,
                "max_abs_dmh": max_dmh,
                "max_abs_dloggf": max_dloggf,
                "n_star_failures": len(iter_star_failures),
                "n_line_failures": len(iter_line_failures),
                "iter_converged": iter_converged,
                "consecutive_converged": consecutive_converged,
            }
        )

        if np.isfinite(objective):
            prev_objective = objective
        star_failures.extend(iter_star_failures)
        line_failures.extend(iter_line_failures)

        if consecutive_converged >= max(1, int(config.consecutive_convergence_required)):
            converged = True
            break

    return JointFitResult(
        config=config,
        star_states=star_states_local,
        line_states=line_states_local,
        line_deltas_dense=line_deltas,
        update_mask=mask,
        history=history,
        star_failures=star_failures,
        line_failures=line_failures,
        converged=bool(converged),
        n_outer_iterations=int(n_completed),
    )


__all__ = [
    "JointFitConfig",
    "JointStarState",
    "JointLineState",
    "JointFitResult",
    "fit_joint_loggf_stellar_parameters",
]
