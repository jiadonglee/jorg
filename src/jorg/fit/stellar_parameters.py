"""
Stellar parameter fitting pipeline for Teff, logg, [M/H], and [alpha/Fe].

This is a Korg-inspired direct spectral fitting workflow:
1. Parameter scaling to an unbounded space.
2. Optional fitting windows and per-window linear continuum adjustment.
3. Two-stage optimization with optional JAX-accelerated local surrogate.
4. Approximate covariance from finite-difference Hessian.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..synthesis import synth

try:
    import jax.numpy as jnp
    from jaxopt import LBFGSB

    _JAX_ACCEL_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency path
    jnp = None
    LBFGSB = None
    _JAX_ACCEL_AVAILABLE = False


PARAMETER_ORDER = ("Teff", "logg", "m_h", "alpha_fe")

DEFAULT_STELLAR_BOUNDS: Dict[str, Tuple[float, float]] = {
    "Teff": (2800.0, 8000.0),
    "logg": (-0.5, 5.5),
    "m_h": (-5.0, 1.0),
    "alpha_fe": (-1.5, 1.5),
}

DEFAULT_HESSIAN_STEPS: Dict[str, float] = {
    "Teff": 25.0,
    "logg": 0.03,
    "m_h": 0.02,
    "alpha_fe": 0.02,
}

LARGE_CHI2 = 1e30


ModelFunction = Callable[["StellarParameters", np.ndarray, Mapping[str, Any]], np.ndarray]


@dataclass(frozen=True)
class StellarParameters:
    """
    Stellar parameters in Teff/logg/[M/H]/[alpha/Fe] convention.
    """

    Teff: float
    logg: float
    m_h: float
    alpha_fe: float = 0.0

    @property
    def alpha_h(self) -> float:
        """[alpha/H] implied by [M/H] + [alpha/Fe]."""
        return self.m_h + self.alpha_fe

    def to_dict(self) -> Dict[str, float]:
        return {
            "Teff": float(self.Teff),
            "logg": float(self.logg),
            "m_h": float(self.m_h),
            "alpha_fe": float(self.alpha_fe),
            "alpha_h": float(self.alpha_h),
        }


@dataclass
class StellarFitResult:
    """Output container for stellar parameter fitting."""

    best_parameters: StellarParameters
    initial_parameters: StellarParameters
    best_fit_flux: np.ndarray
    fit_wavelengths: np.ndarray
    chi2: float
    reduced_chi2: float
    converged: bool
    n_iterations: int
    n_evaluations: int
    parameter_uncertainties: Dict[str, float]
    covariance: np.ndarray
    bounds: Dict[str, Tuple[float, float]]
    history: List[Dict[str, float]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "Stellar Parameter Fit Summary",
            "============================",
            (
                f"Teff = {self.best_parameters.Teff:.1f} +/- "
                f"{self.parameter_uncertainties.get('Teff', np.nan):.1f} K"
            ),
            (
                f"logg = {self.best_parameters.logg:.3f} +/- "
                f"{self.parameter_uncertainties.get('logg', np.nan):.3f}"
            ),
            (
                f"[M/H] = {self.best_parameters.m_h:.3f} +/- "
                f"{self.parameter_uncertainties.get('m_h', np.nan):.3f}"
            ),
            (
                f"[alpha/Fe] = {self.best_parameters.alpha_fe:.3f} +/- "
                f"{self.parameter_uncertainties.get('alpha_fe', np.nan):.3f}"
            ),
            f"chi2 = {self.chi2:.3f}",
            f"reduced chi2 = {self.reduced_chi2:.3f}",
            f"converged = {self.converged}",
            f"iterations = {self.n_iterations}",
            f"evaluations = {self.n_evaluations}",
        ]
        return "\n".join(lines)


def _normalize_param_name(name: str) -> str:
    key = (
        name.strip()
        .lower()
        .replace("[", "")
        .replace("]", "")
        .replace("/", "_")
        .replace("-", "_")
    )

    aliases = {
        "teff": "Teff",
        "logg": "logg",
        "m_h": "m_h",
        "mh": "m_h",
        "metallicity": "m_h",
        "alpha_fe": "alpha_fe",
        "alphafe": "alpha_fe",
        "alpha_h": "alpha_h",
        "alphah": "alpha_h",
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown parameter name '{name}'. Use Teff, logg, m_h (or M/H), "
            "alpha_fe (or alpha/Fe), or alpha_h."
        )
    return aliases[key]


def _merge_windows(windows: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    cleaned: List[Tuple[float, float]] = []
    for lower, upper in windows:
        lo = float(lower)
        hi = float(upper)
        if lo > hi:
            lo, hi = hi, lo
        cleaned.append((lo, hi))

    if not cleaned:
        return []

    cleaned.sort(key=lambda p: p[0])
    merged = [cleaned[0]]
    for lo, hi in cleaned[1:]:
        prev_lo, prev_hi = merged[-1]
        if lo <= prev_hi:
            merged[-1] = (prev_lo, max(prev_hi, hi))
        else:
            merged.append((lo, hi))
    return merged


def _build_wavelength_mask(
    wavelengths: np.ndarray, windows: Sequence[Tuple[float, float]]
) -> np.ndarray:
    mask = np.zeros_like(wavelengths, dtype=bool)
    for lo, hi in windows:
        mask |= (wavelengths >= lo) & (wavelengths <= hi)
    return mask


def _tan_scale(value: float, lower: float, upper: float) -> float:
    if lower >= upper:
        raise ValueError(f"Invalid bounds ({lower}, {upper})")
    width = upper - lower
    eps = max(1e-12 * width, 1e-12)
    clipped = float(np.clip(value, lower + eps, upper - eps))
    return float(np.tan(np.pi * (((clipped - lower) / width) - 0.5)))


def _tan_unscale(value: float, lower: float, upper: float) -> float:
    return float((np.arctan(value) / np.pi + 0.5) * (upper - lower) + lower)


def _cache_key_from_components(
    components: Mapping[str, float], precision: int = 8
) -> Tuple[float, float, float, float]:
    return tuple(round(float(components[name]), precision) for name in PARAMETER_ORDER)


def _estimate_error_from_flux(flux: np.ndarray) -> np.ndarray:
    if flux.size < 3:
        return np.full_like(flux, 0.01, dtype=float)
    diffs = np.diff(flux)
    med = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - med)))
    sigma = 1.4826 * mad / np.sqrt(2.0)
    sigma = float(np.clip(sigma, 1e-4, 0.2))
    return np.full_like(flux, sigma, dtype=float)


def _apply_linear_continuum_adjustment(
    wavelengths: np.ndarray,
    windows: Sequence[Tuple[float, float]],
    model_flux: np.ndarray,
    obs_flux: np.ndarray,
    obs_err: np.ndarray,
) -> np.ndarray:
    adjusted = model_flux.copy()
    for lo, hi in windows:
        window_mask = (wavelengths >= lo) & (wavelengths <= hi)
        if int(np.count_nonzero(window_mask)) < 3:
            continue

        wl = wavelengths[window_mask]
        mdl = adjusted[window_mask]
        dat = obs_flux[window_mask]
        ivar = 1.0 / np.maximum(obs_err[window_mask], 1e-12) ** 2

        x0 = mdl
        x1 = mdl * wl

        s00 = np.sum(ivar * x0 * x0)
        s01 = np.sum(ivar * x0 * x1)
        s11 = np.sum(ivar * x1 * x1)
        t0 = np.sum(ivar * x0 * dat)
        t1 = np.sum(ivar * x1 * dat)

        det = s00 * s11 - s01 * s01
        if (not np.isfinite(det)) or abs(det) < 1e-30:
            continue

        beta0 = (t0 * s11 - t1 * s01) / det
        beta1 = (s00 * t1 - s01 * t0) / det
        adjusted[window_mask] = mdl * (beta0 + beta1 * wl)
    return adjusted


def _default_model_function(
    params: StellarParameters,
    wavelengths: np.ndarray,
    context: Mapping[str, Any],
) -> np.ndarray:
    wl_model, flux_model, _ = synth(
        params.Teff,
        params.logg,
        params.m_h,
        alpha_H=params.alpha_h,
        wavelengths=wavelengths,
        linelist=context.get("linelist"),
        rectify=True,
        R=context.get("R", float("inf")),
        vmic=context.get("vmic", 1.0),
        vsini=context.get("vsini", 0.0),
        verbose=False,
        **context.get("synth_kwargs", {}),
    )

    wl_model = np.asarray(wl_model, dtype=float)
    flux_model = np.asarray(flux_model, dtype=float)
    if wl_model.shape == wavelengths.shape and np.allclose(wl_model, wavelengths):
        return flux_model
    return np.interp(
        wavelengths, wl_model, flux_model, left=flux_model[0], right=flux_model[-1]
    )


def _normalize_initial_guess(
    initial_guess: Optional[Mapping[str, float]]
) -> StellarParameters:
    values: Dict[str, float] = {"Teff": 5777.0, "logg": 4.44, "m_h": 0.0, "alpha_fe": 0.0}
    alpha_h_value: Optional[float] = None

    if initial_guess:
        for raw_key, raw_value in initial_guess.items():
            canonical = _normalize_param_name(raw_key)
            if canonical == "alpha_h":
                alpha_h_value = float(raw_value)
            else:
                values[canonical] = float(raw_value)

    explicit_alpha_fe = bool(
        initial_guess and any(_normalize_param_name(k) == "alpha_fe" for k in initial_guess)
    )
    if alpha_h_value is not None and not explicit_alpha_fe:
        values["alpha_fe"] = float(alpha_h_value - values["m_h"])

    return StellarParameters(**values)


def _normalize_fixed_params(
    fixed_params: Optional[Mapping[str, float]],
    reference_m_h: float,
) -> Dict[str, float]:
    fixed: Dict[str, float] = {}
    alpha_h_value: Optional[float] = None

    if fixed_params:
        for raw_key, raw_value in fixed_params.items():
            canonical = _normalize_param_name(raw_key)
            if canonical == "alpha_h":
                alpha_h_value = float(raw_value)
            else:
                fixed[canonical] = float(raw_value)

    if alpha_h_value is not None and "alpha_fe" not in fixed:
        fixed["alpha_fe"] = float(alpha_h_value - fixed.get("m_h", reference_m_h))
    return fixed


def _normalize_bounds(
    bounds: Optional[Mapping[str, Tuple[float, float]]]
) -> Dict[str, Tuple[float, float]]:
    result = dict(DEFAULT_STELLAR_BOUNDS)
    if not bounds:
        return result

    for raw_key, pair in bounds.items():
        canonical = _normalize_param_name(raw_key)
        if canonical == "alpha_h":
            raise ValueError(
                "Bounds for alpha_h are not directly supported. Use alpha_fe bounds."
            )
        if len(pair) != 2:
            raise ValueError(f"Bounds for {raw_key} must be (lower, upper).")
        lower = float(pair[0])
        upper = float(pair[1])
        if lower >= upper:
            raise ValueError(f"Invalid bounds for {raw_key}: ({lower}, {upper})")
        result[canonical] = (lower, upper)
    return result


def _params_from_components(components: Mapping[str, float]) -> StellarParameters:
    return StellarParameters(
        Teff=float(components["Teff"]),
        logg=float(components["logg"]),
        m_h=float(components["m_h"]),
        alpha_fe=float(components["alpha_fe"]),
    )


def _compute_hessian_and_covariance(
    objective_no_prior: Callable[[np.ndarray], float],
    best_values: np.ndarray,
    fit_names: Sequence[str],
    bounds: Mapping[str, Tuple[float, float]],
) -> Tuple[np.ndarray, Dict[str, float]]:
    n = len(fit_names)
    if n == 0:
        return np.zeros((0, 0), dtype=float), {}

    steps = np.zeros(n, dtype=float)
    for i, name in enumerate(fit_names):
        lower, upper = bounds[name]
        base_step = DEFAULT_HESSIAN_STEPS[name]
        margin = min(best_values[i] - lower, upper - best_values[i])
        max_centered_step = 0.45 * max(margin, 0.0)
        if max_centered_step <= 0.0:
            return np.full((n, n), np.nan, dtype=float), {
                key: np.nan for key in fit_names
            }
        steps[i] = min(base_step, max_centered_step)

    f0 = objective_no_prior(best_values)
    if not np.isfinite(f0):
        return np.full((n, n), np.nan, dtype=float), {
            key: np.nan for key in fit_names
        }

    hessian = np.zeros((n, n), dtype=float)
    for i in range(n):
        ei = np.zeros(n, dtype=float)
        ei[i] = steps[i]
        fp = objective_no_prior(best_values + ei)
        fm = objective_no_prior(best_values - ei)
        hessian[i, i] = (fp - 2.0 * f0 + fm) / (steps[i] ** 2)

    for i in range(n):
        for j in range(i + 1, n):
            ei = np.zeros(n, dtype=float)
            ej = np.zeros(n, dtype=float)
            ei[i] = steps[i]
            ej[j] = steps[j]
            fpp = objective_no_prior(best_values + ei + ej)
            fpm = objective_no_prior(best_values + ei - ej)
            fmp = objective_no_prior(best_values - ei + ej)
            fmm = objective_no_prior(best_values - ei - ej)
            hij = (fpp - fpm - fmp + fmm) / (4.0 * steps[i] * steps[j])
            hessian[i, j] = hij
            hessian[j, i] = hij

    try:
        covariance = 2.0 * np.linalg.inv(hessian)
        diagonal = np.diag(covariance)
        uncertainties = {
            name: float(np.sqrt(val)) if val >= 0.0 else np.nan
            for name, val in zip(fit_names, diagonal)
        }
    except np.linalg.LinAlgError:
        covariance = np.full((n, n), np.nan, dtype=float)
        uncertainties = {name: np.nan for name in fit_names}

    return covariance, uncertainties


def _jax_surrogate_step(
    p_center: np.ndarray,
    flux_center: np.ndarray,
    jacobian: np.ndarray,
    obs_flux: np.ndarray,
    obs_err: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    *,
    maxiter: int,
    tol: float,
) -> Tuple[np.ndarray, int, bool]:
    """
    Minimize a local linearized objective with JAX L-BFGS-B.
    """
    if not _JAX_ACCEL_AVAILABLE:
        return np.asarray(p_center, dtype=float), 0, False

    p0 = jnp.asarray(p_center, dtype=jnp.float64)
    f0 = jnp.asarray(flux_center, dtype=jnp.float64)
    J = jnp.asarray(jacobian, dtype=jnp.float64)
    y = jnp.asarray(obs_flux, dtype=jnp.float64)
    sigma = jnp.asarray(obs_err, dtype=jnp.float64)
    lb = jnp.asarray(lower_bounds, dtype=jnp.float64)
    ub = jnp.asarray(upper_bounds, dtype=jnp.float64)

    # Tiny margin avoids tan() singularity at exact bounds in the weak prior.
    eps = jnp.maximum(1e-10, 1e-10 * (ub - lb))
    lb_safe = lb + eps
    ub_safe = ub - eps

    def objective(p):
        p_clip = jnp.clip(p, lb_safe, ub_safe)
        pred_flux = f0 + J @ (p_clip - p0)
        resid = (pred_flux - y) / sigma
        chi2 = jnp.sum(resid * resid)
        scaled = jnp.tan(jnp.pi * (((p_clip - lb) / (ub - lb)) - 0.5))
        prior = jnp.sum((scaled / 100.0) ** 2)
        return chi2 + prior

    solver = LBFGSB(fun=objective, maxiter=int(maxiter), tol=float(tol), jit=True)
    result = solver.run(p0, bounds=(lb, ub))

    params = np.asarray(result.params, dtype=float)
    state = result.state
    iter_num = int(getattr(state, "iter_num", 0))
    error = float(getattr(state, "error", np.inf))
    success = np.isfinite(error)
    return params, iter_num, bool(success)


def fit_stellar_parameters(
    obs_wavelengths: np.ndarray,
    obs_flux: np.ndarray,
    linelist: Optional[Sequence[Any]],
    *,
    obs_error: Optional[np.ndarray] = None,
    initial_guess: Optional[Mapping[str, float]] = None,
    fixed_params: Optional[Mapping[str, float]] = None,
    bounds: Optional[Mapping[str, Tuple[float, float]]] = None,
    windows: Optional[Sequence[Tuple[float, float]]] = None,
    R: float = 20_000.0,
    vmic: float = 1.0,
    vsini: float = 0.0,
    adjust_continuum: bool = True,
    optimizer: str = "auto",
    global_samples: int = 32,
    max_iterations: int = 80,
    initial_scaled_step: float = 1.0,
    scaled_tolerance: float = 2e-3,
    shrink: float = 0.5,
    surrogate_outer_iterations: int = 2,
    surrogate_step_fraction: float = 0.03,
    surrogate_maxiter: int = 60,
    surrogate_tolerance: float = 1e-3,
    compute_uncertainties: bool = False,
    random_seed: Optional[int] = 1234,
    model_function: Optional[ModelFunction] = None,
    model_context: Optional[Mapping[str, Any]] = None,
    return_history: bool = True,
) -> StellarFitResult:
    """
    Fit Teff/logg/[M/H]/[alpha/Fe] to a rectified observed spectrum.

    Parameters
    ----------
    optimizer : {"auto", "jax_surrogate", "coordinate"}
        - "coordinate": original NumPy coordinate search.
        - "jax_surrogate": local linearization + JAX L-BFGS-B acceleration.
        - "auto": use "jax_surrogate" when JAX is available, otherwise fallback
          to "coordinate".
    compute_uncertainties : bool, default False
        If True, estimate covariance with finite-difference Hessian. This can
        be expensive because it triggers many extra full syntheses.
    """
    obs_wavelengths = np.asarray(obs_wavelengths, dtype=float)
    obs_flux = np.asarray(obs_flux, dtype=float)
    if obs_wavelengths.ndim != 1 or obs_flux.ndim != 1:
        raise ValueError("obs_wavelengths and obs_flux must be 1D arrays")
    if obs_wavelengths.shape != obs_flux.shape:
        raise ValueError("obs_wavelengths and obs_flux must have the same shape")
    if obs_wavelengths.size < 10:
        raise ValueError("Need at least 10 pixels for fitting")

    order = np.argsort(obs_wavelengths)
    obs_wavelengths = obs_wavelengths[order]
    obs_flux = obs_flux[order]

    if obs_error is None:
        obs_error = _estimate_error_from_flux(obs_flux)
    else:
        obs_error = np.asarray(obs_error, dtype=float)[order]
    if obs_error.shape != obs_flux.shape:
        raise ValueError("obs_error must have the same shape as obs_flux")
    if np.any(~np.isfinite(obs_wavelengths)) or np.any(~np.isfinite(obs_flux)):
        raise ValueError("obs_wavelengths and obs_flux must be finite")
    if np.any(~np.isfinite(obs_error)) or np.any(obs_error <= 0.0):
        raise ValueError("obs_error must be finite and strictly positive")

    if windows is None:
        windows = [(float(obs_wavelengths[0]), float(obs_wavelengths[-1]))]
    merged_windows = _merge_windows(windows)
    if not merged_windows:
        raise ValueError("windows must contain at least one valid interval")

    fit_mask = _build_wavelength_mask(obs_wavelengths, merged_windows)
    if not np.any(fit_mask):
        raise ValueError("No observed pixels fall inside requested windows")

    fit_wavelengths = obs_wavelengths[fit_mask]
    fit_flux = obs_flux[fit_mask]
    fit_error = obs_error[fit_mask]

    bounds_map = _normalize_bounds(bounds)
    initial_params = _normalize_initial_guess(initial_guess)
    fixed_map = _normalize_fixed_params(fixed_params, initial_params.m_h)

    for name in PARAMETER_ORDER:
        value = float(fixed_map[name]) if name in fixed_map else float(
            getattr(initial_params, name)
        )
        lower, upper = bounds_map[name]
        if not (lower <= value <= upper):
            raise ValueError(
                f"Initial/fixed {name}={value} outside bounds ({lower}, {upper})"
            )

    fit_names = [name for name in PARAMETER_ORDER if name not in fixed_map]

    if model_function is None:
        if linelist is None:
            raise ValueError("linelist is required when model_function is not supplied")
        model_function = _default_model_function

    context: Dict[str, Any] = dict(model_context or {})
    context.setdefault("linelist", linelist)
    context.setdefault("R", R)
    context.setdefault("vmic", vmic)
    context.setdefault("vsini", vsini)
    context.setdefault("synth_kwargs", {})

    optimizer_key = str(optimizer).strip().lower()
    if optimizer_key not in {"auto", "jax_surrogate", "coordinate"}:
        raise ValueError(
            f"optimizer must be one of ('auto', 'jax_surrogate', 'coordinate'), got {optimizer!r}"
        )
    if optimizer_key == "auto":
        optimizer_key = "jax_surrogate" if _JAX_ACCEL_AVAILABLE else "coordinate"
    if optimizer_key == "jax_surrogate" and not _JAX_ACCEL_AVAILABLE:
        optimizer_key = "coordinate"

    def build_components_from_scaled(scaled_vector: np.ndarray) -> Dict[str, float]:
        components: Dict[str, float] = dict(fixed_map)
        for idx, name in enumerate(fit_names):
            lower, upper = bounds_map[name]
            components[name] = _tan_unscale(float(scaled_vector[idx]), lower, upper)
        for name in PARAMETER_ORDER:
            if name not in components:
                components[name] = float(getattr(initial_params, name))
        return components

    def build_components_from_values(values: np.ndarray) -> Dict[str, float]:
        components: Dict[str, float] = dict(fixed_map)
        for idx, name in enumerate(fit_names):
            lower, upper = bounds_map[name]
            components[name] = float(np.clip(values[idx], lower, upper))
        for name in PARAMETER_ORDER:
            if name not in components:
                components[name] = float(getattr(initial_params, name))
        return components

    def values_from_components(components: Mapping[str, float]) -> np.ndarray:
        return np.array([float(components[name]) for name in fit_names], dtype=float)

    def scaled_from_components(components: Mapping[str, float]) -> np.ndarray:
        return np.array(
            [_tan_scale(components[name], *bounds_map[name]) for name in fit_names],
            dtype=float,
        )

    def prior_from_components(components: Mapping[str, float]) -> float:
        if not fit_names:
            return 0.0
        scaled = scaled_from_components(components)
        return float(np.sum((scaled / 100.0) ** 2))

    evaluation_count = 0
    store_flux_in_cache = optimizer_key == "jax_surrogate"
    objective_cache: Dict[Tuple[float, float, float, float], Tuple[float, Optional[np.ndarray]]] = {}

    def evaluate_components(
        components: Mapping[str, float],
        *,
        return_flux: bool = False,
    ) -> Tuple[float, Optional[np.ndarray]]:
        nonlocal evaluation_count
        key = _cache_key_from_components(components)
        cached = objective_cache.get(key)
        if cached is not None:
            cached_chi2, cached_flux = cached
            if return_flux and cached_flux is not None:
                return float(cached_chi2), np.asarray(cached_flux, dtype=float)
            if not return_flux:
                return float(cached_chi2), None

        evaluation_count += 1

        params = _params_from_components(components)
        try:
            model_flux = np.asarray(
                model_function(params, fit_wavelengths, context), dtype=float
            )
        except Exception:
            return LARGE_CHI2, None

        if model_flux.shape != fit_flux.shape:
            if model_flux.ndim != 1 or model_flux.size < 2:
                return LARGE_CHI2, None
            synth_wl = np.linspace(fit_wavelengths[0], fit_wavelengths[-1], model_flux.size)
            model_flux = np.interp(
                fit_wavelengths, synth_wl, model_flux, left=model_flux[0], right=model_flux[-1]
            )

        if np.any(~np.isfinite(model_flux)):
            return LARGE_CHI2, None

        adjusted_flux = (
            _apply_linear_continuum_adjustment(
                fit_wavelengths, merged_windows, model_flux, fit_flux, fit_error
            )
            if adjust_continuum
            else model_flux
        )

        residual = (adjusted_flux - fit_flux) / fit_error
        chi2 = float(np.sum(residual * residual))

        flux_to_cache = (
            np.asarray(adjusted_flux, dtype=float).copy() if store_flux_in_cache else None
        )
        objective_cache[key] = (chi2, flux_to_cache)

        if return_flux:
            if flux_to_cache is not None:
                return chi2, flux_to_cache
            return chi2, np.asarray(adjusted_flux, dtype=float)
        return chi2, None

    def evaluate_scaled(
        scaled_vector: np.ndarray,
        *,
        include_prior: bool = True,
        return_flux: bool = False,
    ) -> Tuple[float, float, Optional[np.ndarray]]:
        components = build_components_from_scaled(scaled_vector)
        chi2, flux = evaluate_components(components, return_flux=return_flux)
        prior = prior_from_components(components) if include_prior else 0.0
        total = chi2 + prior
        return total, chi2, flux if return_flux else None

    if fit_names:
        initial_components = {
            "Teff": float(initial_params.Teff),
            "logg": float(initial_params.logg),
            "m_h": float(initial_params.m_h),
            "alpha_fe": float(initial_params.alpha_fe),
        }
        x0 = scaled_from_components(initial_components)
        best_x = x0.copy()
        best_total, _, _ = evaluate_scaled(best_x, include_prior=True, return_flux=False)
        history: List[Dict[str, float]] = []
        converged = False
        n_iterations = 0

        rng = np.random.default_rng(random_seed)
        if global_samples > 0:
            n_local = max(0, global_samples // 2)
            n_global = max(0, global_samples - n_local)

            for _ in range(n_local):
                trial = x0 + rng.normal(0.0, 1.5, size=len(fit_names))
                total, _, _ = evaluate_scaled(trial, include_prior=True, return_flux=False)
                if total < best_total:
                    best_total = total
                    best_x = trial

            for _ in range(n_global):
                sampled = {name: rng.uniform(*bounds_map[name]) for name in fit_names}
                trial = np.array(
                    [_tan_scale(sampled[name], *bounds_map[name]) for name in fit_names],
                    dtype=float,
                )
                total, _, _ = evaluate_scaled(trial, include_prior=True, return_flux=False)
                if total < best_total:
                    best_total = total
                    best_x = trial

        if optimizer_key == "coordinate":
            step = np.full(len(fit_names), float(initial_scaled_step), dtype=float)
            for iteration in range(1, max_iterations + 1):
                n_iterations = iteration
                improved = False
                for i in range(len(fit_names)):
                    local_best_total = best_total
                    local_best_x = best_x
                    for direction in (1.0, -1.0):
                        trial = best_x.copy()
                        trial[i] += direction * step[i]
                        total, _, _ = evaluate_scaled(
                            trial, include_prior=True, return_flux=False
                        )
                        if total < local_best_total:
                            local_best_total = total
                            local_best_x = trial
                    if local_best_total < best_total:
                        best_total = local_best_total
                        best_x = local_best_x
                        improved = True

                _, pure_chi2, _ = evaluate_scaled(
                    best_x, include_prior=False, return_flux=False
                )
                components = build_components_from_scaled(best_x)
                if return_history:
                    history.append(
                        {
                            "iteration": float(iteration),
                            "chi2": float(pure_chi2),
                            "Teff": float(components["Teff"]),
                            "logg": float(components["logg"]),
                            "m_h": float(components["m_h"]),
                            "alpha_fe": float(components["alpha_fe"]),
                            "optimizer": "coordinate",
                        }
                    )

                if not improved:
                    step *= float(shrink)
                    if float(np.max(step)) < float(scaled_tolerance):
                        converged = True
                        break
        else:
            # JAX-accelerated local linear surrogate + L-BFGS-B.
            lower_bounds = np.array([bounds_map[name][0] for name in fit_names], dtype=float)
            upper_bounds = np.array([bounds_map[name][1] for name in fit_names], dtype=float)

            best_components = build_components_from_scaled(best_x)
            center_values = values_from_components(best_components)
            center_total = best_total

            for outer in range(1, max(1, surrogate_outer_iterations) + 1):
                n_iterations = outer
                center_components = build_components_from_values(center_values)
                chi2_center, flux_center = evaluate_components(
                    center_components, return_flux=True
                )
                if flux_center is None or not np.isfinite(chi2_center):
                    break
                center_total = chi2_center + prior_from_components(center_components)

                jacobian = np.zeros((fit_flux.size, len(fit_names)), dtype=float)
                jacobian_ok = True
                for j, name in enumerate(fit_names):
                    lo, hi = bounds_map[name]
                    width = hi - lo
                    base_step = max(DEFAULT_HESSIAN_STEPS[name], surrogate_step_fraction * width)
                    max_step = 0.45 * min(center_values[j] - lo, hi - center_values[j])
                    if max_step <= 0.0:
                        jacobian_ok = False
                        break
                    step = min(base_step, max_step)

                    p_plus = center_values.copy()
                    p_minus = center_values.copy()
                    p_plus[j] += step
                    p_minus[j] -= step

                    chi2_plus, flux_plus = evaluate_components(
                        build_components_from_values(p_plus), return_flux=True
                    )
                    chi2_minus, flux_minus = evaluate_components(
                        build_components_from_values(p_minus), return_flux=True
                    )
                    if (
                        flux_plus is None
                        or flux_minus is None
                        or not np.isfinite(chi2_plus)
                        or not np.isfinite(chi2_minus)
                    ):
                        jacobian_ok = False
                        break

                    denom = p_plus[j] - p_minus[j]
                    if denom <= 0.0 or not np.isfinite(denom):
                        jacobian_ok = False
                        break
                    jacobian[:, j] = (flux_plus - flux_minus) / denom

                if not jacobian_ok:
                    break

                candidate_values, inner_iters, _ = _jax_surrogate_step(
                    p_center=center_values,
                    flux_center=flux_center,
                    jacobian=jacobian,
                    obs_flux=fit_flux,
                    obs_err=fit_error,
                    lower_bounds=lower_bounds,
                    upper_bounds=upper_bounds,
                    maxiter=max(5, int(surrogate_maxiter)),
                    tol=float(surrogate_tolerance),
                )

                # Conservative line search on the true forward model.
                improved = False
                best_local_values = center_values.copy()
                best_local_total = center_total
                for frac in (1.0, 0.5, 0.25):
                    trial_values = center_values + frac * (candidate_values - center_values)
                    trial_values = np.clip(trial_values, lower_bounds, upper_bounds)
                    trial_components = build_components_from_values(trial_values)
                    trial_chi2, _ = evaluate_components(trial_components, return_flux=False)
                    trial_total = trial_chi2 + prior_from_components(trial_components)
                    if trial_total < best_local_total:
                        best_local_total = trial_total
                        best_local_values = trial_values
                        improved = True

                prev_center_values = center_values.copy()
                center_values = best_local_values
                center_total = best_local_total
                best_x = scaled_from_components(build_components_from_values(center_values))
                best_total = center_total

                _, pure_chi2, _ = evaluate_scaled(
                    best_x, include_prior=False, return_flux=False
                )
                if return_history:
                    comp = build_components_from_scaled(best_x)
                    history.append(
                        {
                            "iteration": float(outer),
                            "chi2": float(pure_chi2),
                            "Teff": float(comp["Teff"]),
                            "logg": float(comp["logg"]),
                            "m_h": float(comp["m_h"]),
                            "alpha_fe": float(comp["alpha_fe"]),
                            "optimizer": "jax_surrogate",
                            "inner_lbfgsb_iters": float(inner_iters),
                        }
                    )

                if not improved:
                    converged = True
                    break

                delta = np.max(
                    np.abs(center_values - prev_center_values)
                    / np.maximum(upper_bounds - lower_bounds, 1e-12)
                )
                if float(delta) < float(scaled_tolerance):
                    converged = True
                    break
    else:
        best_x = np.zeros(0, dtype=float)
        history = []
        converged = True
        n_iterations = 0

    if fit_names:
        best_components = build_components_from_scaled(best_x)
    else:
        best_components = {
            "Teff": float(fixed_map["Teff"]),
            "logg": float(fixed_map["logg"]),
            "m_h": float(fixed_map["m_h"]),
            "alpha_fe": float(fixed_map["alpha_fe"]),
        }

    best_parameters = _params_from_components(best_components)
    _, best_chi2, best_fit_flux = evaluate_scaled(
        best_x, include_prior=False, return_flux=True
    )
    if best_fit_flux is None:
        raise RuntimeError("Forward model failed at best-fit parameters")

    n_data = fit_flux.size
    n_fit = max(len(fit_names), 1)
    dof = max(n_data - n_fit, 1)
    reduced_chi2 = float(best_chi2 / dof)

    if fit_names and compute_uncertainties:
        best_fit_values = np.array([best_components[name] for name in fit_names], dtype=float)

        def objective_no_prior(values: np.ndarray) -> float:
            trial_components = dict(best_components)
            for i, name in enumerate(fit_names):
                lower, upper = bounds_map[name]
                trial_components[name] = float(np.clip(values[i], lower, upper))
            trial_scaled = np.array(
                [_tan_scale(trial_components[name], *bounds_map[name]) for name in fit_names],
                dtype=float,
            )
            _, chi2_value, _ = evaluate_scaled(
                trial_scaled, include_prior=False, return_flux=False
            )
            return float(chi2_value)

        cov_fit, unc_fit = _compute_hessian_and_covariance(
            objective_no_prior, best_fit_values, fit_names, bounds_map
        )
    else:
        cov_fit = np.full((len(fit_names), len(fit_names)), np.nan, dtype=float)
        unc_fit = {name: np.nan for name in fit_names}

    covariance = np.full((len(PARAMETER_ORDER), len(PARAMETER_ORDER)), np.nan, dtype=float)
    for i, name_i in enumerate(PARAMETER_ORDER):
        if name_i in fixed_map:
            covariance[i, i] = 0.0
            continue
        ii = fit_names.index(name_i)
        for j, name_j in enumerate(PARAMETER_ORDER):
            if name_j in fixed_map:
                covariance[i, j] = 0.0
                continue
            jj = fit_names.index(name_j)
            covariance[i, j] = cov_fit[ii, jj]

    uncertainties = {name: 0.0 for name in PARAMETER_ORDER}
    for name, value in unc_fit.items():
        uncertainties[name] = float(value)

    return StellarFitResult(
        best_parameters=best_parameters,
        initial_parameters=initial_params,
        best_fit_flux=np.asarray(best_fit_flux, dtype=float),
        fit_wavelengths=np.asarray(fit_wavelengths, dtype=float),
        chi2=float(best_chi2),
        reduced_chi2=float(reduced_chi2),
        converged=bool(converged),
        n_iterations=int(n_iterations),
        n_evaluations=int(evaluation_count),
        parameter_uncertainties=uncertainties,
        covariance=covariance,
        bounds=bounds_map,
        history=history,
    )


__all__ = [
    "DEFAULT_STELLAR_BOUNDS",
    "PARAMETER_ORDER",
    "StellarParameters",
    "StellarFitResult",
    "fit_stellar_parameters",
]
