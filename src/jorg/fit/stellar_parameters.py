"""
Stellar parameter fitting pipeline for Teff, logg, [M/H], and [alpha/Fe].

This is a Korg-inspired direct spectral fitting workflow:
1. Parameter scaling to an unbounded space.
2. Optional fitting windows and per-window linear continuum adjustment.
3. Two-stage optimization with optional JAX-accelerated local surrogate.
4. Approximate covariance from finite-difference Hessian.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
from ..synthesis import _get_default_statmech_data, synth, synthesize_jax
from ..atmosphere import clear_interpolator_caches
from ..statmech.chem_eq_jax import clear_solver_caches as clear_chem_eq_solver_caches
from ..continuum.mclaughlin_hminus import clear_mclaughlin_cache
from ..continuum.stancil1994 import clear_stancil_cache
from ..continuum.helium import clear_helium_cache
from ..continuum.metals_bf import clear_metal_bf_cache
from ..continuum.nahar_h_i_bf import clear_nahar_h_i_cache

try:
    import jax
    import jax.numpy as jnp
    from jaxopt import LBFGSB

    _JAX_ACCEL_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency path
    jax = None
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
_FIT_LINE_WINDOW_CACHE = OrderedDict()
_FIT_LINE_WINDOW_CACHE_MAX = 64
_AUTODIFF_OBJECTIVE_CACHE = OrderedDict()
_AUTODIFF_OBJECTIVE_CACHE_MAX = 8
_AUTODIFF_EXECUTION_PROFILES = {
    "debug": {
        "value_and_grad_atmosphere_interp_jit": False,
        "value_and_grad_objective_jit": False,
        "value_and_grad_solver_jit": False,
        "value_and_grad_solver_maxls": 2,
        "value_and_grad_solver_step_fraction": None,
        "value_and_grad_solver_backtracking": (1.0, 0.5),
        "value_and_grad_ce_warm_start": True,
        "value_and_grad_solver_implicit_diff": False,
    },
    "interactive": {
        "value_and_grad_atmosphere_interp_jit": True,
        "value_and_grad_objective_jit": True,
        "value_and_grad_solver_jit": False,
        "value_and_grad_solver_maxls": 2,
        "value_and_grad_solver_step_fraction": None,
        "value_and_grad_solver_backtracking": (1.0, 0.5),
        "value_and_grad_ce_warm_start": True,
        "value_and_grad_solver_implicit_diff": False,
    },
    "batch": {
        "value_and_grad_atmosphere_interp_jit": True,
        "value_and_grad_objective_jit": True,
        "value_and_grad_solver_jit": True,
        "value_and_grad_solver_maxls": 4,
        "value_and_grad_solver_step_fraction": 0.12,
        "value_and_grad_solver_backtracking": (1.0, 0.5, 0.25, 0.1),
        "value_and_grad_ce_warm_start": True,
        "value_and_grad_solver_implicit_diff": False,
    },
}


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


def _stable_array_cache_key(values: np.ndarray) -> Tuple[Tuple[int, ...], int]:
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return tuple(int(dim) for dim in arr.shape), hash(arr.tobytes())


def _line_wavelength_cm(line: Any) -> Optional[float]:
    wl = getattr(line, "wavelength", None)
    if wl is None:
        wl = getattr(line, "wl", None)
    if wl is None:
        return None
    wl_val = float(wl)
    if not np.isfinite(wl_val) or wl_val <= 0.0:
        return None
    if wl_val > 1.0:
        wl_val *= 1e-8
    return wl_val


def _get_fit_relevant_lines_cached(
    linelist: Optional[Sequence[Any]],
    wl_min_cm: float,
    wl_max_cm: float,
) -> Tuple[Tuple[Any, ...], np.ndarray]:
    if linelist is None:
        return (), np.zeros(0, dtype=np.int32)

    key = (id(linelist), round(float(wl_min_cm), 16), round(float(wl_max_cm), 16))
    cached = _FIT_LINE_WINDOW_CACHE.get(key)
    if cached is not None:
        _FIT_LINE_WINDOW_CACHE.move_to_end(key)
        lines, indices = cached
        return lines, np.asarray(indices, dtype=np.int32)

    selected_lines: List[Any] = []
    selected_indices: List[int] = []
    for idx, line in enumerate(linelist):
        wl_val = _line_wavelength_cm(line)
        if wl_val is None:
            continue
        if wl_min_cm <= wl_val <= wl_max_cm:
            selected_lines.append(line)
            selected_indices.append(int(idx))

    cached_value = (
        tuple(selected_lines),
        np.asarray(selected_indices, dtype=np.int32),
    )
    _FIT_LINE_WINDOW_CACHE[key] = cached_value
    if len(_FIT_LINE_WINDOW_CACHE) > _FIT_LINE_WINDOW_CACHE_MAX:
        _FIT_LINE_WINDOW_CACHE.popitem(last=False)

    return cached_value


def _normalize_execution_profile(
    execution_profile: str,
) -> str:
    profile = str(execution_profile).strip().lower()
    if profile not in _AUTODIFF_EXECUTION_PROFILES:
        allowed = ", ".join(sorted(_AUTODIFF_EXECUTION_PROFILES))
        raise ValueError(
            f"execution_profile must be one of {allowed}; got {execution_profile!r}."
        )
    return profile


def _get_cached_autodiff_objective(
    cache_key: Tuple[Any, ...],
    builder: Callable[[], Any],
) -> Any:
    cached = _AUTODIFF_OBJECTIVE_CACHE.get(cache_key)
    if cached is not None:
        _AUTODIFF_OBJECTIVE_CACHE.move_to_end(cache_key)
        return cached

    value = builder()
    _AUTODIFF_OBJECTIVE_CACHE[cache_key] = value
    if len(_AUTODIFF_OBJECTIVE_CACHE) > _AUTODIFF_OBJECTIVE_CACHE_MAX:
        _AUTODIFF_OBJECTIVE_CACHE.popitem(last=False)
    return value


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


def _apply_linear_continuum_adjustment_jax(
    wavelengths: "jnp.ndarray",
    windows: Sequence[Tuple[float, float]],
    model_flux: "jnp.ndarray",
    obs_flux: "jnp.ndarray",
    obs_err: "jnp.ndarray",
) -> "jnp.ndarray":
    """
    JAX-compatible continuum adjustment used by the value_and_grad optimizer.
    """
    adjusted = model_flux
    inv_var_all = 1.0 / jnp.maximum(obs_err, 1e-12) ** 2
    for lo, hi in windows:
        window_mask = (wavelengths >= lo) & (wavelengths <= hi)
        mask_f = window_mask.astype(adjusted.dtype)
        ivar = inv_var_all * mask_f

        # Keep array shapes static; if the window is too small, det/solve is benign and
        # the update is masked out below.
        x0 = adjusted
        x1 = adjusted * wavelengths

        s00 = jnp.sum(ivar * x0 * x0)
        s01 = jnp.sum(ivar * x0 * x1)
        s11 = jnp.sum(ivar * x1 * x1)
        t0 = jnp.sum(ivar * x0 * obs_flux)
        t1 = jnp.sum(ivar * x1 * obs_flux)

        det = s00 * s11 - s01 * s01
        safe_det = jnp.where(jnp.abs(det) > 1e-30, det, 1.0)
        beta0 = (t0 * s11 - t1 * s01) / safe_det
        beta1 = (s00 * t1 - s01 * t0) / safe_det
        update_all = adjusted * (beta0 + beta1 * wavelengths)

        valid = (jnp.sum(window_mask) >= 3) & jnp.isfinite(det) & (jnp.abs(det) > 1e-30)
        adjusted = jnp.where(window_mask & valid, update_all, adjusted)
    return adjusted


def _validate_jax_value_and_grad_synthesize_kwargs(
    context: Mapping[str, Any],
) -> Tuple[Dict[str, Any], str, bool, bool]:
    """
    Validate strict autodiff requirements for optimizer='jax_value_and_grad'.
    """
    synth_kwargs = dict(context.get("synth_kwargs", {}))
    synthesize_kwargs = dict(synth_kwargs.get("synthesize_kwargs", {}))

    line_backend = str(synthesize_kwargs.get("line_backend", "jax")).strip().lower()
    autodiff_strict = bool(synthesize_kwargs.get("autodiff_strict", True))
    ce_jit = bool(synthesize_kwargs.get("ce_jit", True))

    if line_backend != "jax":
        raise ValueError(
            "optimizer='jax_value_and_grad' requires line_backend='jax' "
            "inside model_context['synth_kwargs']['synthesize_kwargs']."
        )
    if not autodiff_strict:
        raise ValueError(
            "optimizer='jax_value_and_grad' requires autodiff_strict=True "
            "inside model_context['synth_kwargs']['synthesize_kwargs']."
        )
    if not ce_jit:
        raise ValueError(
            "optimizer='jax_value_and_grad' requires ce_jit=True "
            "inside model_context['synth_kwargs']['synthesize_kwargs']."
        )

    return synthesize_kwargs, line_backend, autodiff_strict, ce_jit


def _merge_autodiff_fit_model_context(
    model_context: Optional[Mapping[str, Any]],
    *,
    execution_profile: Literal["interactive", "batch", "debug"] = "interactive",
) -> Dict[str, Any]:
    """
    Build a model_context with the strict autodiff defaults for
    optimizer='jax_value_and_grad'.

    User-provided keys are preserved unless this helper is responsible for
    supplying the default.
    """
    profile = _normalize_execution_profile(execution_profile)
    context: Dict[str, Any] = dict(model_context or {})

    synth_kwargs = dict(context.get("synth_kwargs", {}))
    synthesize_kwargs = dict(synth_kwargs.get("synthesize_kwargs", {}))

    synth_kwargs.setdefault("engine", "jax")
    synth_kwargs.setdefault("ce_solver", "jax")

    synthesize_kwargs.setdefault("line_backend", "jax")
    synthesize_kwargs.setdefault("autodiff_strict", True)
    synthesize_kwargs.setdefault("ce_jit", True)
    synthesize_kwargs.setdefault("cntm_step", 1.0)
    synthesize_kwargs.setdefault("line_buffer", 10.0)
    synthesize_kwargs.setdefault("line_cutoff_threshold", 3e-4)
    synthesize_kwargs.setdefault("rectify_mode", "continuum")

    synth_kwargs["synthesize_kwargs"] = synthesize_kwargs
    context["synth_kwargs"] = synth_kwargs

    for key, value in _AUTODIFF_EXECUTION_PROFILES[profile].items():
        context.setdefault(key, value)

    return context


def _default_model_function(
    params: StellarParameters,
    wavelengths: np.ndarray,
    context: Mapping[str, Any],
) -> np.ndarray:
    synth_kwargs = dict(context.get("synth_kwargs", {}))
    if "synthesize_kwargs" in synth_kwargs and synth_kwargs["synthesize_kwargs"] is not None:
        # synth() pops engine/ce_solver from this nested dictionary; copy it per call.
        synth_kwargs["synthesize_kwargs"] = dict(synth_kwargs["synthesize_kwargs"])

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
        **synth_kwargs,
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


def _normalize_parameter_priors(
    parameter_priors: Optional[Mapping[str, Tuple[float, float]]],
    *,
    fit_names: Sequence[str],
) -> Dict[str, Tuple[float, float]]:
    if not parameter_priors:
        return {}

    fit_name_set = set(fit_names)
    out: Dict[str, Tuple[float, float]] = {}
    for raw_key, raw_value in parameter_priors.items():
        canonical = _normalize_param_name(raw_key)
        if canonical == "alpha_h":
            raise ValueError(
                "Priors for alpha_h are not directly supported. Use alpha_fe priors."
            )
        if canonical not in fit_name_set:
            continue

        if not isinstance(raw_value, (tuple, list)) or len(raw_value) != 2:
            raise ValueError(
                f"Prior for {raw_key!r} must be a (mu, sigma) pair; got {raw_value!r}."
            )
        mu = float(raw_value[0])
        sigma = float(raw_value[1])
        if (not np.isfinite(mu)) or (not np.isfinite(sigma)) or sigma <= 0.0:
            raise ValueError(
                f"Prior for {raw_key!r} must have finite mu and positive sigma; "
                f"got ({raw_value[0]!r}, {raw_value[1]!r})."
            )
        out[canonical] = (mu, sigma)
    return out


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
    solver_jit: bool = True,
    solver_implicit_diff: bool = True,
    prior_mu: Optional[np.ndarray] = None,
    prior_sigma: Optional[np.ndarray] = None,
    prior_strength: float = 1.0,
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
    prior_mu_j = (
        None
        if prior_mu is None
        else jnp.asarray(np.asarray(prior_mu, dtype=np.float64), dtype=jnp.float64)
    )
    prior_sigma_j = (
        None
        if prior_sigma is None
        else jnp.asarray(np.asarray(prior_sigma, dtype=np.float64), dtype=jnp.float64)
    )
    prior_strength_j = jnp.asarray(float(prior_strength), dtype=jnp.float64)

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
        if prior_mu_j is not None and prior_sigma_j is not None:
            valid_prior = jnp.isfinite(prior_mu_j) & jnp.isfinite(prior_sigma_j) & (prior_sigma_j > 0.0)
            prior_resid = jnp.where(
                valid_prior,
                ((p_clip - prior_mu_j) / prior_sigma_j) ** 2,
                0.0,
            )
            prior = prior + prior_strength_j * jnp.sum(prior_resid)
        return chi2 + prior

    solver = LBFGSB(
        fun=objective,
        maxiter=int(maxiter),
        tol=float(tol),
        jit=bool(solver_jit),
        implicit_diff=bool(solver_implicit_diff),
    )
    result = solver.run(p0, bounds=(lb, ub))

    params = np.asarray(result.params, dtype=float)
    state = result.state
    iter_num = int(getattr(state, "iter_num", 0))
    error = float(getattr(state, "error", np.inf))
    success = np.isfinite(error)
    return params, iter_num, bool(success)


def fit_stellar_parameters_autodiff(
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
    global_samples: int = 0,
    max_iterations: int = 80,
    surrogate_maxiter: int = 60,
    surrogate_tolerance: float = 1e-3,
    parameter_priors: Optional[Mapping[str, Tuple[float, float]]] = None,
    prior_strength: float = 1.0,
    compute_uncertainties: bool = False,
    random_seed: Optional[int] = 1234,
    model_context: Optional[Mapping[str, Any]] = None,
    execution_profile: Literal["interactive", "batch", "debug"] = "interactive",
    return_history: bool = True,
    **fit_kwargs: Any,
) -> StellarFitResult:
    """
    Convenience wrapper for the strict JAX autodiff fitting path.

    This pins optimizer='jax_value_and_grad' and injects the recommended
    strict synthesis defaults, so callers do not need to build the nested
    model_context['synth_kwargs']['synthesize_kwargs'] structure manually.

    Extra keyword arguments are forwarded to fit_stellar_parameters(), except
    for 'optimizer' and 'model_function', which are intentionally fixed by
    this wrapper.
    """
    if "optimizer" in fit_kwargs:
        raise ValueError(
            "fit_stellar_parameters_autodiff() fixes optimizer='jax_value_and_grad'; "
            "do not pass optimizer explicitly."
        )
    if "model_function" in fit_kwargs:
        raise ValueError(
            "fit_stellar_parameters_autodiff() requires the built-in synthesis model; "
            "do not pass model_function explicitly."
        )

    context = _merge_autodiff_fit_model_context(
        model_context,
        execution_profile=execution_profile,
    )

    return fit_stellar_parameters(
        obs_wavelengths=obs_wavelengths,
        obs_flux=obs_flux,
        obs_error=obs_error,
        linelist=linelist,
        initial_guess=initial_guess,
        fixed_params=fixed_params,
        bounds=bounds,
        windows=windows,
        R=R,
        vmic=vmic,
        vsini=vsini,
        adjust_continuum=adjust_continuum,
        optimizer="jax_value_and_grad",
        global_samples=global_samples,
        max_iterations=max_iterations,
        surrogate_maxiter=surrogate_maxiter,
        surrogate_tolerance=surrogate_tolerance,
        parameter_priors=parameter_priors,
        prior_strength=prior_strength,
        compute_uncertainties=compute_uncertainties,
        random_seed=random_seed,
        model_context=context,
        return_history=return_history,
        **fit_kwargs,
    )


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
    parameter_priors: Optional[Mapping[str, Tuple[float, float]]] = None,
    prior_strength: float = 1.0,
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
    optimizer : {"auto", "jax_value_and_grad", "jax_surrogate", "coordinate"}
        - "coordinate": original NumPy coordinate search.
        - "jax_value_and_grad": full objective optimized with JAX value_and_grad.
          Requires JAX and the built-in synthesis model path.
        - "jax_surrogate": local linearization + JAX L-BFGS-B acceleration.
        - "auto": use "jax_value_and_grad" when JAX is available, otherwise
          fallback to "coordinate".
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
    if not np.isfinite(float(prior_strength)) or float(prior_strength) < 0.0:
        raise ValueError("prior_strength must be a finite non-negative number.")
    prior_strength_value = float(prior_strength)
    normalized_priors = _normalize_parameter_priors(
        parameter_priors,
        fit_names=fit_names,
    )

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
    if optimizer_key not in {"auto", "jax_value_and_grad", "jax_surrogate", "coordinate"}:
        raise ValueError(
            "optimizer must be one of "
            "('auto', 'jax_value_and_grad', 'jax_surrogate', 'coordinate'), "
            f"got {optimizer!r}"
        )
    if optimizer_key == "auto":
        optimizer_key = "jax_value_and_grad" if _JAX_ACCEL_AVAILABLE else "coordinate"
    if optimizer_key in {"jax_value_and_grad", "jax_surrogate"} and not _JAX_ACCEL_AVAILABLE:
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

    def parameter_prior_from_components(components: Mapping[str, float]) -> float:
        if (not normalized_priors) or prior_strength_value <= 0.0:
            return 0.0
        total = 0.0
        for name in fit_names:
            prior = normalized_priors.get(name)
            if prior is None:
                continue
            mu, sigma = prior
            total += ((float(components[name]) - mu) / sigma) ** 2
        return float(prior_strength_value * total)

    def prior_from_components(components: Mapping[str, float]) -> float:
        if not fit_names:
            return 0.0
        scaled = scaled_from_components(components)
        boundary_prior = float(np.sum((scaled / 100.0) ** 2))
        return boundary_prior + parameter_prior_from_components(components)

    prior_mu_vector = (
        np.array(
            [
                normalized_priors[name][0] if name in normalized_priors else np.nan
                for name in fit_names
            ],
            dtype=float,
        )
        if fit_names and normalized_priors
        else None
    )
    prior_sigma_vector = (
        np.array(
            [
                normalized_priors[name][1] if name in normalized_priors else np.nan
                for name in fit_names
            ],
            dtype=float,
        )
        if fit_names and normalized_priors
        else None
    )

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

    best_chi2_jax: Optional[float] = None
    best_fit_flux_jax: Optional[np.ndarray] = None

    if fit_names:
        initial_components = {
            "Teff": float(initial_params.Teff),
            "logg": float(initial_params.logg),
            "m_h": float(initial_params.m_h),
            "alpha_fe": float(initial_params.alpha_fe),
        }
        x0 = scaled_from_components(initial_components)
        best_x = x0.copy()
        best_total = np.inf
        if optimizer_key != "jax_value_and_grad":
            best_total, _, _ = evaluate_scaled(best_x, include_prior=True, return_flux=False)
        history: List[Dict[str, float]] = []
        converged = False
        n_iterations = 0

        if optimizer_key == "jax_value_and_grad":
            if model_function is not _default_model_function:
                raise ValueError(
                    "optimizer='jax_value_and_grad' currently requires the built-in "
                    "model_function (leave model_function=None)."
                )

            try:
                from ..abundances import format_abundances
                from ..atmosphere import interpolate_marcs
                from ..utils.spectral_processing import apply_LSF, apply_rotation
            except Exception as exc:  # pragma: no cover - optional path
                raise RuntimeError(
                    "Failed to initialize dependencies for optimizer='jax_value_and_grad'."
                ) from exc

            synthesize_kwargs, line_backend, autodiff_strict, ce_jit = (
                _validate_jax_value_and_grad_synthesize_kwargs(context)
            )

            teff0 = float(initial_params.Teff)
            logg0 = float(initial_params.logg)
            mh0 = float(initial_params.m_h)
            grid_mode = context.get("grid_mode")
            if grid_mode is None:
                if mh0 < -2.5:
                    grid_mode = "low_z"
                elif teff0 <= 4000.0 and logg0 >= 3.5 and mh0 >= -2.5:
                    # Cool-dwarf cubic path currently contains non-JAX interpolation
                    # and is not tracer-safe. Use standard multilinear grid in grad mode.
                    grid_mode = "standard"
                else:
                    grid_mode = "standard"

            spherical = bool(context.get("spherical", logg0 < 3.5))
            grid_data_dir = context.get("grid_data_dir")
            ionization_energies = context.get("ionization_energies")
            partition_funcs = context.get("partition_funcs")
            log_equilibrium_constants = context.get("log_equilibrium_constants")
            if (
                ionization_energies is None
                or partition_funcs is None
                or log_equilibrium_constants is None
            ):
                (
                    default_ionization_energies,
                    default_partition_funcs,
                    default_log_equilibrium_constants,
                ) = _get_default_statmech_data()
                if ionization_energies is None:
                    ionization_energies = default_ionization_energies
                if partition_funcs is None:
                    partition_funcs = default_partition_funcs
                if log_equilibrium_constants is None:
                    log_equilibrium_constants = default_log_equilibrium_constants
            atmosphere_interp_jit = bool(context.get("value_and_grad_atmosphere_interp_jit", True))
            ce_warm_start = bool(context.get("value_and_grad_ce_warm_start", True))
            requested_objective_jit = bool(context.get("value_and_grad_objective_jit", False))
            requested_solver_jit = bool(context.get("value_and_grad_solver_jit", False))
            allow_experimental_objective_jit = bool(
                context.get("value_and_grad_enable_experimental_objective_jit", False)
            )
            solver_maxls = max(1, int(context.get("value_and_grad_solver_maxls", 5)))
            solver_implicit_diff = bool(
                context.get("value_and_grad_solver_implicit_diff", True)
            )

            lower_bounds = jnp.asarray(
                [bounds_map[name][0] for name in fit_names], dtype=jnp.float64
            )
            upper_bounds = jnp.asarray(
                [bounds_map[name][1] for name in fit_names], dtype=jnp.float64
            )
            fit_wavelengths_j = jnp.asarray(fit_wavelengths, dtype=jnp.float64)
            fit_flux_j = jnp.asarray(fit_flux, dtype=jnp.float64)
            fit_error_j = jnp.asarray(fit_error, dtype=jnp.float64)
            vmic_j = jnp.asarray(context.get("vmic", vmic), dtype=jnp.float64)
            R_local = float(context.get("R", R))
            vsini_local = float(context.get("vsini", vsini))
            line_buffer = float(synthesize_kwargs.get("line_buffer", 10.0))
            filtered_linelist = context.get("linelist")
            filtered_line_loggf_deltas = synthesize_kwargs.get("line_loggf_deltas")
            if filtered_linelist:
                wl_lo = max(0.0, float(np.min(fit_wavelengths)) - line_buffer)
                wl_hi = float(np.max(fit_wavelengths)) + line_buffer
                filtered_linelist, filtered_line_indices = _get_fit_relevant_lines_cached(
                    filtered_linelist,
                    wl_lo * 1e-8,
                    wl_hi * 1e-8,
                )
                if filtered_line_loggf_deltas is not None:
                    filtered_line_loggf_deltas = np.asarray(
                        filtered_line_loggf_deltas,
                        dtype=np.float64,
                    )[filtered_line_indices]
            else:
                filtered_linelist = () if filtered_linelist is None else tuple(filtered_linelist)

            x0_values = jnp.asarray(
                [initial_components[name] for name in fit_names], dtype=jnp.float64
            )
            prior_mu_values_j = (
                None
                if prior_mu_vector is None
                else jnp.asarray(prior_mu_vector, dtype=jnp.float64)
            )
            prior_sigma_values_j = (
                None
                if prior_sigma_vector is None
                else jnp.asarray(prior_sigma_vector, dtype=jnp.float64)
            )
            prior_strength_j = jnp.asarray(prior_strength_value, dtype=jnp.float64)

            def _components_from_values_jax(values: "jnp.ndarray") -> Dict[str, "jnp.ndarray"]:
                comps: Dict[str, "jnp.ndarray"] = {}
                idx = 0
                for name in PARAMETER_ORDER:
                    if name in fixed_map:
                        comps[name] = jnp.asarray(float(fixed_map[name]), dtype=jnp.float64)
                    elif name in fit_names:
                        lo, hi = bounds_map[name]
                        comps[name] = jnp.clip(
                            values[idx],
                            jnp.asarray(lo, dtype=jnp.float64),
                            jnp.asarray(hi, dtype=jnp.float64),
                        )
                        idx += 1
                    else:
                        comps[name] = jnp.asarray(
                            float(getattr(initial_params, name)),
                            dtype=jnp.float64,
                        )
                return comps

            def _synthesize_flux_jax(values: "jnp.ndarray") -> "jnp.ndarray":
                comps = _components_from_values_jax(values)
                teff_j = comps["Teff"]
                logg_j = comps["logg"]
                mh_j = comps["m_h"]
                alpha_fe_j = comps["alpha_fe"]
                alpha_h_j = mh_j + alpha_fe_j

                atm_dense = interpolate_marcs(
                    Teff=teff_j,
                    logg=logg_j,
                    m_H=mh_j,
                    alpha_m=alpha_fe_j,
                    C_m=0.0,
                    spherical=spherical,
                    grid_data_dir=grid_data_dir,
                    grid_mode=grid_mode,
                    return_dense=True,
                    interpolator_jit=atmosphere_interp_jit,
                )
                A_X = format_abundances(
                    default_metals_H=mh_j,
                    default_alpha_H=alpha_h_j,
                )

                state = synthesize_jax(
                    atm=atm_dense,
                    linelist=filtered_linelist,
                    A_X=A_X,
                    wavelengths=fit_wavelengths_j,
                    vmic=vmic_j,
                    mu_values=int(synthesize_kwargs.get("mu_values", 20)),
                    line_buffer=line_buffer,
                    cntm_step=float(synthesize_kwargs.get("cntm_step", 1.0)),
                    rectify=True,
                    rectify_mode=str(synthesize_kwargs.get("rectify_mode", "continuum")),
                    rectify_percentile=float(synthesize_kwargs.get("rectify_percentile", 99.5)),
                    autodiff_strict=autodiff_strict,
                    line_backend=line_backend,
                    line_loggf_deltas=filtered_line_loggf_deltas,
                    ce_jit=ce_jit,
                    ce_warm_start=ce_warm_start,
                    ionization_energies=ionization_energies,
                    partition_funcs=partition_funcs,
                    log_equilibrium_constants=log_equilibrium_constants,
                    line_cutoff_threshold=float(synthesize_kwargs.get("line_cutoff_threshold", 3e-4)),
                    verbose=False,
                )

                flux_j = state.flux
                if np.isfinite(R_local) and R_local > 0.0:
                    flux_j = apply_LSF(flux_j, state.wavelengths, R_local)
                if vsini_local > 0.0:
                    flux_j = apply_rotation(flux_j, state.wavelengths, vsini_local)
                return jnp.asarray(flux_j, dtype=jnp.float64)

            objective_cache_key_base = (
                _stable_array_cache_key(fit_wavelengths),
                tuple(fit_names),
                tuple(
                    (float(bounds_map[name][0]), float(bounds_map[name][1])) for name in fit_names
                ),
                tuple((float(lo), float(hi)) for lo, hi in merged_windows),
                id(filtered_linelist),
                (
                    None
                    if filtered_line_loggf_deltas is None
                    else _stable_array_cache_key(np.asarray(filtered_line_loggf_deltas))
                ),
                bool(adjust_continuum),
                float(context.get("vmic", vmic)),
                float(R_local),
                float(vsini_local),
                bool(spherical),
                str(grid_mode),
                str(grid_data_dir),
                bool(atmosphere_interp_jit),
                int(synthesize_kwargs.get("mu_values", 20)),
                float(line_buffer),
                float(synthesize_kwargs.get("cntm_step", 1.0)),
                str(synthesize_kwargs.get("rectify_mode", "continuum")),
                float(synthesize_kwargs.get("rectify_percentile", 99.5)),
                bool(autodiff_strict),
                str(line_backend),
                bool(ce_jit),
                float(synthesize_kwargs.get("line_cutoff_threshold", 3e-4)),
                (
                    None
                    if prior_mu_vector is None
                    else tuple(float(v) for v in prior_mu_vector)
                ),
                (
                    None
                    if prior_sigma_vector is None
                    else tuple(float(v) for v in prior_sigma_vector)
                ),
                float(prior_strength_value),
                tuple(
                    (name, float(fixed_map[name]))
                    for name in PARAMETER_ORDER
                    if name in fixed_map
                ),
                tuple(
                    (name, float(getattr(initial_params, name)))
                    for name in PARAMETER_ORDER
                    if name not in fit_names
                ),
                id(ionization_energies),
                id(partition_funcs),
                id(log_equilibrium_constants),
            )

            def _build_objective_bundle(use_objective_jit: bool) -> Dict[str, Any]:
                def _chi2_and_flux_jax(
                    values: "jnp.ndarray",
                    obs_flux_j: "jnp.ndarray",
                    obs_error_j: "jnp.ndarray",
                ) -> Tuple["jnp.ndarray", "jnp.ndarray"]:
                    model_flux_j = _synthesize_flux_jax(values)
                    adjusted_flux_j = (
                        _apply_linear_continuum_adjustment_jax(
                            fit_wavelengths_j,
                            merged_windows,
                            model_flux_j,
                            obs_flux_j,
                            obs_error_j,
                        )
                        if adjust_continuum
                        else model_flux_j
                    )
                    resid = (adjusted_flux_j - obs_flux_j) / obs_error_j
                    chi2 = jnp.sum(resid * resid)
                    return chi2, adjusted_flux_j

                def _chi2_only_jax(
                    values: "jnp.ndarray",
                    obs_flux_j: "jnp.ndarray",
                    obs_error_j: "jnp.ndarray",
                ) -> "jnp.ndarray":
                    chi2, _ = _chi2_and_flux_jax(values, obs_flux_j, obs_error_j)
                    return chi2

                def _objective_total_with_aux_jax(
                    values: "jnp.ndarray",
                    obs_flux_j: "jnp.ndarray",
                    obs_error_j: "jnp.ndarray",
                ) -> Tuple["jnp.ndarray", Tuple["jnp.ndarray", "jnp.ndarray"]]:
                    values_clip = jnp.clip(values, lower_bounds, upper_bounds)
                    chi2, adjusted_flux_j = _chi2_and_flux_jax(
                        values_clip, obs_flux_j, obs_error_j
                    )
                    scaled = jnp.tan(
                        jnp.pi
                        * (((values_clip - lower_bounds) / (upper_bounds - lower_bounds)) - 0.5)
                    )
                    prior = jnp.sum((scaled / 100.0) ** 2)
                    if prior_mu_values_j is not None and prior_sigma_values_j is not None:
                        valid_prior = (
                            jnp.isfinite(prior_mu_values_j)
                            & jnp.isfinite(prior_sigma_values_j)
                            & (prior_sigma_values_j > 0.0)
                        )
                        prior_resid = jnp.where(
                            valid_prior,
                            ((values_clip - prior_mu_values_j) / prior_sigma_values_j) ** 2,
                            0.0,
                        )
                        prior = prior + prior_strength_j * jnp.sum(prior_resid)
                    total = chi2 + prior
                    return total, (chi2, adjusted_flux_j)

                def _objective_total_jax(
                    values: "jnp.ndarray",
                    obs_flux_j: "jnp.ndarray",
                    obs_error_j: "jnp.ndarray",
                ) -> "jnp.ndarray":
                    values_clip = jnp.clip(values, lower_bounds, upper_bounds)
                    chi2 = _chi2_only_jax(values_clip, obs_flux_j, obs_error_j)
                    scaled = jnp.tan(
                        jnp.pi
                        * (((values_clip - lower_bounds) / (upper_bounds - lower_bounds)) - 0.5)
                    )
                    prior = jnp.sum((scaled / 100.0) ** 2)
                    if prior_mu_values_j is not None and prior_sigma_values_j is not None:
                        valid_prior = (
                            jnp.isfinite(prior_mu_values_j)
                            & jnp.isfinite(prior_sigma_values_j)
                            & (prior_sigma_values_j > 0.0)
                        )
                        prior_resid = jnp.where(
                            valid_prior,
                            ((values_clip - prior_mu_values_j) / prior_sigma_values_j) ** 2,
                            0.0,
                        )
                        prior = prior + prior_strength_j * jnp.sum(prior_resid)
                    return chi2 + prior

                objective_with_grad = jax.value_and_grad(_objective_total_jax)
                objective_with_grad_with_aux = jax.value_and_grad(
                    _objective_total_with_aux_jax,
                    has_aux=True,
                )
                if use_objective_jit:
                    objective_with_grad = jax.jit(objective_with_grad)
                    objective_with_grad_with_aux = jax.jit(objective_with_grad_with_aux)

                return {
                    "value_and_grad": objective_with_grad,
                    "value_and_grad_with_aux": objective_with_grad_with_aux,
                    "chi2_only": _chi2_only_jax,
                }

            def _load_objective_bundle(use_objective_jit: bool) -> Dict[str, Any]:
                return _get_cached_autodiff_objective(
                    objective_cache_key_base + (bool(use_objective_jit),),
                    lambda: _build_objective_bundle(use_objective_jit),
                )

            objective_jit = requested_objective_jit and allow_experimental_objective_jit
            if requested_objective_jit and not objective_jit:
                atmosphere_interp_jit = False
            objective_bundle = _load_objective_bundle(objective_jit)
            objective_with_grad_cached = objective_bundle["value_and_grad"]
            objective_with_grad_with_aux_cached = objective_bundle["value_and_grad_with_aux"]
            chi2_only_jax = objective_bundle["chi2_only"]

            def _objective_for_solver(values: "jnp.ndarray") -> Tuple["jnp.ndarray", "jnp.ndarray"]:
                return objective_with_grad_cached(values, fit_flux_j, fit_error_j)

            initial_objective_value: Optional[float] = None
            final_objective_value: Optional[float] = None
            final_best_flux_value: Optional[np.ndarray] = None
            final_best_chi2_value: Optional[float] = None
            solver_jit = requested_solver_jit and objective_jit
            max_solver_iters = max(1, int(surrogate_maxiter))
            lower_bounds_np = np.asarray(jax.device_get(lower_bounds), dtype=float)
            upper_bounds_np = np.asarray(jax.device_get(upper_bounds), dtype=float)
            x0_clipped_np = np.clip(
                np.asarray(jax.device_get(x0_values), dtype=float),
                lower_bounds_np,
                upper_bounds_np,
            )
            evaluation_count = 0

            def _eval_objective(values_np: np.ndarray) -> Tuple[float, np.ndarray, float, np.ndarray]:
                nonlocal evaluation_count, objective_jit, objective_with_grad_cached
                nonlocal objective_with_grad_with_aux_cached, solver_jit
                nonlocal atmosphere_interp_jit
                values_j = jnp.asarray(values_np, dtype=jnp.float64)
                try:
                    (total_j, (chi2_j, flux_j)), grad_j = objective_with_grad_with_aux_cached(
                        values_j,
                        fit_flux_j,
                        fit_error_j,
                    )
                except Exception:
                    if not objective_jit:
                        raise
                    objective_jit = False
                    atmosphere_interp_jit = False
                    clear_interpolator_caches()
                    clear_chem_eq_solver_caches()
                    clear_mclaughlin_cache()
                    clear_stancil_cache()
                    clear_helium_cache()
                    clear_metal_bf_cache()
                    clear_nahar_h_i_cache()
                    objective_bundle_fallback = _load_objective_bundle(False)
                    objective_with_grad_cached = objective_bundle_fallback["value_and_grad"]
                    objective_with_grad_with_aux_cached = objective_bundle_fallback["value_and_grad_with_aux"]
                    solver_jit = False
                    (total_j, (chi2_j, flux_j)), grad_j = objective_with_grad_with_aux_cached(
                        values_j,
                        fit_flux_j,
                        fit_error_j,
                    )
                total = float(np.asarray(jax.device_get(total_j)))
                grad = np.asarray(jax.device_get(grad_j), dtype=float)
                chi2_val = float(np.asarray(jax.device_get(chi2_j)))
                flux_val = np.asarray(jax.device_get(flux_j), dtype=float)
                evaluation_count += 1
                return total, grad, chi2_val, flux_val

            probe_total, probe_grad, probe_chi2, probe_flux = _eval_objective(x0_clipped_np)
            initial_objective_value = probe_total

            if solver_jit:
                solver = LBFGSB(
                    fun=_objective_for_solver,
                    value_and_grad=True,
                    maxiter=max_solver_iters,
                    tol=float(surrogate_tolerance),
                    maxls=solver_maxls,
                    jit=solver_jit,
                    implicit_diff=solver_implicit_diff,
                )

                result = solver.run(x0_values, bounds=(lower_bounds, upper_bounds))
                best_values = np.asarray(result.params, dtype=float)
                state = result.state
                n_iterations = int(getattr(state, "iter_num", 0))
                # Fall back gracefully when solver state does not expose num_fun_eval.
                evaluation_count = max(
                    evaluation_count,
                    int(getattr(state, "num_fun_eval", n_iterations + 1)),
                )
                error = float(getattr(state, "error", np.inf))
                converged = bool(np.isfinite(error))
                final_objective_value = None
            else:
                widths = np.maximum(
                    np.asarray(jax.device_get(upper_bounds - lower_bounds), dtype=float),
                    1e-12,
                )
                step_fraction_override = context.get("value_and_grad_solver_step_fraction", None)
                if step_fraction_override is None:
                    grad_step_fraction = 0.12 if len(merged_windows) > 1 else 0.15
                else:
                    grad_step_fraction = float(step_fraction_override)
                backtracking_factors = tuple(
                    float(v)
                    for v in context.get(
                        "value_and_grad_solver_backtracking",
                        (1.0, 0.5, 0.25, 0.1, 0.05),
                    )
                )
                x_curr = np.asarray(x0_clipped_np, dtype=float)
                curr_total, curr_grad = probe_total, probe_grad
                curr_chi2 = probe_chi2
                curr_flux = np.asarray(probe_flux, dtype=float)
                grad_scale = np.max(np.abs(curr_grad * widths)) if curr_grad.size else 0.0
                error = float(grad_scale)
                converged = bool(np.isfinite(curr_total))
                n_iterations = 0

                for iter_idx in range(max_solver_iters):
                    n_iterations = iter_idx + 1
                    grad_scale = np.max(np.abs(curr_grad * widths)) if curr_grad.size else 0.0
                    error = float(grad_scale)
                    if (not np.isfinite(curr_total)) or (not np.isfinite(grad_scale)):
                        converged = False
                        break
                    if grad_scale <= float(surrogate_tolerance):
                        converged = True
                        break

                    direction = -curr_grad
                    direction_norm = np.max(np.abs(direction * widths)) if direction.size else 0.0
                    if (not np.isfinite(direction_norm)) or direction_norm <= 0.0:
                        converged = True
                        break

                    improved = False
                    for factor in backtracking_factors[: max(1, solver_maxls)]:
                        step = grad_step_fraction * float(factor)
                        trial = np.clip(
                            x_curr + step * (direction * widths / direction_norm),
                            lower_bounds_np,
                            upper_bounds_np,
                        )
                        if np.allclose(trial, x_curr, rtol=0.0, atol=1e-12):
                            continue
                        trial_total, trial_grad, trial_chi2, trial_flux = _eval_objective(trial)
                        if np.isfinite(trial_total) and trial_total < curr_total:
                            x_curr = trial
                            curr_total = trial_total
                            curr_grad = trial_grad
                            curr_chi2 = trial_chi2
                            curr_flux = np.asarray(trial_flux, dtype=float)
                            improved = True
                            break

                    if not improved:
                        converged = grad_scale <= float(surrogate_tolerance)
                        break

                best_values = np.asarray(x_curr, dtype=float)
                final_objective_value = float(curr_total)
                final_best_chi2_value = float(curr_chi2)
                final_best_flux_value = np.asarray(curr_flux, dtype=float)

            best_components = build_components_from_values(best_values)
            best_x = scaled_from_components(best_components)
            best_values_j = jnp.asarray(best_values, dtype=jnp.float64)
            if final_objective_value is not None:
                best_total = float(final_objective_value)
            elif initial_objective_value is not None and np.array_equal(
                np.asarray(x0_values, dtype=float),
                best_values,
            ):
                best_total = initial_objective_value
            else:
                best_total_j, _ = _objective_for_solver(best_values_j)
                best_total = float(np.asarray(jax.device_get(best_total_j)))

            if final_best_flux_value is not None and final_best_chi2_value is not None:
                best_fit_flux_jax = np.asarray(final_best_flux_value, dtype=float)
                best_chi2_jax = float(final_best_chi2_value)
            else:
                best_flux_arr = _synthesize_flux_jax(best_values_j)
                if adjust_continuum:
                    best_flux_arr = _apply_linear_continuum_adjustment_jax(
                        fit_wavelengths_j,
                        merged_windows,
                        best_flux_arr,
                        fit_flux_j,
                        fit_error_j,
                    )
                best_resid_arr = (best_flux_arr - fit_flux_j) / fit_error_j
                best_chi2_jax = float(
                    np.asarray(jax.device_get(jnp.sum(best_resid_arr * best_resid_arr)))
                )
                best_fit_flux_jax = np.asarray(jax.device_get(best_flux_arr), dtype=float)

            if return_history:
                history.append(
                    {
                        "iteration": float(n_iterations),
                        "chi2": float(best_chi2_jax),
                        "Teff": float(best_components["Teff"]),
                        "logg": float(best_components["logg"]),
                        "m_h": float(best_components["m_h"]),
                        "alpha_fe": float(best_components["alpha_fe"]),
                        "optimizer": "jax_value_and_grad",
                    }
                )
        else:
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
                surrogate_solver_jit = bool(context.get("surrogate_solver_jit", True))
                surrogate_solver_implicit_diff = bool(
                    context.get("surrogate_solver_implicit_diff", True)
                )
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
                        solver_jit=surrogate_solver_jit,
                        solver_implicit_diff=surrogate_solver_implicit_diff,
                        prior_mu=prior_mu_vector,
                        prior_sigma=prior_sigma_vector,
                        prior_strength=prior_strength_value,
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
    if best_chi2_jax is not None and best_fit_flux_jax is not None:
        best_chi2 = float(best_chi2_jax)
        best_fit_flux = np.asarray(best_fit_flux_jax, dtype=float)
    else:
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
    "fit_stellar_parameters_autodiff",
    "fit_stellar_parameters",
]
