"""Gaia DR3 RVS-like synthetic spectrum utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike1D = Union[Sequence[float], np.ndarray]
ModelCallable = Callable[[np.ndarray], ArrayLike1D]

C_LIGHT_KMS = 299_792.458
GAIA_RVS_WAVE_MIN_NM = 846.0
GAIA_RVS_WAVE_MAX_NM = 870.0
GAIA_RVS_STEP_NM = 0.01

_DEFAULT_CAT_CENTERS_NM = (849.8, 854.2, 866.2)


@dataclass(frozen=True)
class RVSSpectrum:
    """Container for RVS-like spectra on the Gaia DR3 mean-spectrum grid."""

    wave_nm: np.ndarray
    flux: np.ndarray
    flux_norm: np.ndarray
    continuum: np.ndarray
    metadata: Dict[str, Any]
    observed_wave_nm: Optional[np.ndarray] = None
    observed_flux: Optional[np.ndarray] = None
    observed_flux_norm: Optional[np.ndarray] = None
    observed_continuum: Optional[np.ndarray] = None


def gaia_rvs_wavelength_grid() -> np.ndarray:
    """Return Gaia DR3 RVS mean-spectrum wavelength grid in vacuum nm."""
    return np.arange(
        GAIA_RVS_WAVE_MIN_NM,
        GAIA_RVS_WAVE_MAX_NM + GAIA_RVS_STEP_NM / 2.0,
        GAIA_RVS_STEP_NM,
        dtype=float,
    )


def make_rvs_like(
    wave_nm_hr: Union[ArrayLike1D, ModelCallable],
    flux_hr: Optional[ArrayLike1D] = None,
    R: float = 11500.0,
    rv_kms: Optional[float] = None,
    normalise: bool = True,
    mode: str = "lnlambda",
    poly_order: int = 3,
    n_iter: int = 6,
    clip_sigma: float = 2.5,
    cat_mask_half_width_nm: float = 0.2,
    cat_centers_nm: Sequence[float] = _DEFAULT_CAT_CENTERS_NM,
    extra_line_masks_nm: Optional[Sequence[Tuple[float, float]]] = None,
    snr: Optional[float] = None,
    noise_model: Optional[Callable[[np.ndarray, np.random.Generator], np.ndarray]] = None,
    rng: Optional[Union[int, np.random.Generator]] = None,
    model_wave_min_nm: float = 845.0,
    model_wave_max_nm: float = 871.0,
    model_step_nm: float = 0.001,
) -> RVSSpectrum:
    """
    Generate a Gaia DR3 RVS-like synthetic spectrum.

    Parameters
    ----------
    wave_nm_hr
        Either high-resolution wavelength array (vacuum nm), or a callable model
        ``model(wave_nm) -> flux``.
    flux_hr
        High-resolution flux array when ``wave_nm_hr`` is an array.
        If ``wave_nm_hr`` is callable, this may optionally provide the model
        evaluation grid.
    R
        Resolving power for Gaussian LSF. Default is 11500.
    rv_kms
        Optional radial velocity in km/s. If set, observed-frame arrays are
        included in the returned dataclass.
    normalise
        If True, estimate and apply pseudo-continuum normalization.
    mode
        Convolution mode: ``"lnlambda"`` (default) or ``"quick"``.
    poly_order, n_iter, clip_sigma, cat_mask_half_width_nm, cat_centers_nm,
    extra_line_masks_nm
        Controls for pseudo-continuum estimation.
    snr
        Optional per-sample SNR for additive Gaussian-like noise.
    noise_model
        Optional custom noise model ``noise_model(flux, rng) -> noisy_flux``.
    rng
        Random seed or NumPy Generator used for noise simulation.
    model_wave_min_nm, model_wave_max_nm, model_step_nm
        Default model-evaluation grid settings when ``wave_nm_hr`` is callable
        and no explicit grid is passed via ``flux_hr``.
    """
    if R <= 0.0:
        raise ValueError("R must be positive.")
    if mode not in ("lnlambda", "quick"):
        raise ValueError("mode must be 'lnlambda' or 'quick'.")
    if poly_order < 0:
        raise ValueError("poly_order must be >= 0.")
    if n_iter < 1:
        raise ValueError("n_iter must be >= 1.")
    if clip_sigma <= 0.0:
        raise ValueError("clip_sigma must be > 0.")
    if cat_mask_half_width_nm < 0.0:
        raise ValueError("cat_mask_half_width_nm must be >= 0.")

    wave_hr, flux_in, input_kind = _prepare_high_res_input(
        wave_nm_hr=wave_nm_hr,
        flux_hr=flux_hr,
        model_wave_min_nm=model_wave_min_nm,
        model_wave_max_nm=model_wave_max_nm,
        model_step_nm=model_step_nm,
    )

    if mode == "lnlambda":
        wave_conv, flux_conv = _convolve_constant_R_lnlambda(wave_hr, flux_in, R=R)
        method = "constant-R Gaussian LSF on uniform ln(lambda) grid"
    else:
        wave_conv, flux_conv = _convolve_quick_constant_fwhm(wave_hr, flux_in, R=R)
        method = (
            "quick approximation: constant FWHM Gaussian at lambda0=860 nm on "
            "uniform lambda grid"
        )

    wave_rvs = gaia_rvs_wavelength_grid()
    flux_rvs = flux_conserving_resample(wave_conv, flux_conv, wave_rvs)

    rng_obj = _coerce_rng(rng)
    noise_applied = False
    if snr is not None:
        if snr <= 0.0:
            raise ValueError("snr must be > 0 when provided.")
        if noise_model is None:
            scale = np.nanmedian(np.abs(flux_rvs)) / snr

            def _default_noise_model(flux: np.ndarray, gen: np.random.Generator) -> np.ndarray:
                return flux + gen.normal(0.0, scale, size=flux.shape)

            noise_model_use = _default_noise_model
        else:
            noise_model_use = noise_model
        flux_rvs = np.asarray(noise_model_use(flux_rvs, rng_obj), dtype=float)
        noise_applied = True
    elif noise_model is not None:
        flux_rvs = np.asarray(noise_model(flux_rvs, rng_obj), dtype=float)
        noise_applied = True

    if normalise:
        continuum = estimate_pseudo_continuum(
            wave_nm=wave_rvs,
            flux=flux_rvs,
            poly_order=poly_order,
            n_iter=n_iter,
            clip_sigma=clip_sigma,
            cat_mask_half_width_nm=cat_mask_half_width_nm,
            cat_centers_nm=cat_centers_nm,
            extra_line_masks_nm=extra_line_masks_nm,
        )
        flux_norm = flux_rvs / np.maximum(continuum, 1e-12)
    else:
        continuum = np.ones_like(flux_rvs)
        flux_norm = flux_rvs.copy()

    observed_wave_nm: Optional[np.ndarray] = None
    observed_flux: Optional[np.ndarray] = None
    observed_flux_norm: Optional[np.ndarray] = None
    observed_continuum: Optional[np.ndarray] = None
    doppler_factor: Optional[float] = None
    if rv_kms is not None:
        doppler_factor = _relativistic_doppler_factor(rv_kms)
        observed_wave_nm = wave_rvs * doppler_factor
        observed_flux = flux_rvs.copy()
        observed_flux_norm = flux_norm.copy()
        observed_continuum = continuum.copy()

    metadata: Dict[str, Any] = {
        "R": float(R),
        "mode": mode,
        "method": method,
        "input_kind": input_kind,
        "rv_kms": None if rv_kms is None else float(rv_kms),
        "doppler_factor": doppler_factor,
        "normalised": bool(normalise),
        "poly_order": int(poly_order),
        "n_iter": int(n_iter),
        "clip_sigma": float(clip_sigma),
        "cat_centers_nm": tuple(float(x) for x in cat_centers_nm),
        "cat_mask_half_width_nm": float(cat_mask_half_width_nm),
        "extra_line_masks_nm": None
        if extra_line_masks_nm is None
        else tuple((float(a), float(b)) for a, b in extra_line_masks_nm),
        "noise_applied": noise_applied,
        "snr": None if snr is None else float(snr),
        "resampling": "flux-conserving bin-average via cumulative trapezoid",
        "gaia_grid": {
            "wave_min_nm": GAIA_RVS_WAVE_MIN_NM,
            "wave_max_nm": GAIA_RVS_WAVE_MAX_NM,
            "step_nm": GAIA_RVS_STEP_NM,
            "n_samples": int(wave_rvs.size),
        },
    }
    if mode == "quick":
        metadata["quick_lambda0_nm"] = 860.0
        metadata["quick_fwhm_nm"] = 860.0 / R

    return RVSSpectrum(
        wave_nm=wave_rvs,
        flux=flux_rvs,
        flux_norm=flux_norm,
        continuum=continuum,
        metadata=metadata,
        observed_wave_nm=observed_wave_nm,
        observed_flux=observed_flux,
        observed_flux_norm=observed_flux_norm,
        observed_continuum=observed_continuum,
    )


def estimate_pseudo_continuum(
    wave_nm: ArrayLike1D,
    flux: ArrayLike1D,
    poly_order: int = 3,
    n_iter: int = 6,
    clip_sigma: float = 2.5,
    cat_mask_half_width_nm: float = 0.2,
    cat_centers_nm: Sequence[float] = _DEFAULT_CAT_CENTERS_NM,
    extra_line_masks_nm: Optional[Sequence[Tuple[float, float]]] = None,
) -> np.ndarray:
    """
    Estimate pseudo-continuum in the Ca II triplet region.

    The fit uses iterative asymmetric clipping: lower residual outliers are
    removed while points above the fit are favored in weighting.
    """
    wave = np.asarray(wave_nm, dtype=float)
    y = np.asarray(flux, dtype=float)
    _validate_wave_flux(wave, y)

    base_mask = np.isfinite(wave) & np.isfinite(y)
    for center in cat_centers_nm:
        base_mask &= np.abs(wave - float(center)) > cat_mask_half_width_nm
    if extra_line_masks_nm is not None:
        for a, b in extra_line_masks_nm:
            lo = min(float(a), float(b))
            hi = max(float(a), float(b))
            base_mask &= (wave < lo) | (wave > hi)

    if np.count_nonzero(base_mask) < poly_order + 2:
        raise ValueError(
            "Not enough unmasked points to estimate continuum. "
            "Reduce poly_order or relax masks."
        )

    wave_span = wave[-1] - wave[0]
    if wave_span <= 0.0:
        raise ValueError("Wavelength array must span a non-zero range.")
    x = 2.0 * (wave - 0.5 * (wave[0] + wave[-1])) / wave_span

    keep = base_mask.copy()
    weights = np.ones_like(y)
    continuum = np.ones_like(y)

    for _ in range(n_iter):
        if np.count_nonzero(keep) < poly_order + 2:
            break

        deg = min(poly_order, np.count_nonzero(keep) - 1)
        coeff = np.polyfit(x[keep], y[keep], deg=deg, w=weights[keep])
        continuum = np.polyval(coeff, x)
        continuum = np.maximum(continuum, 1e-12)

        residual = y - continuum
        sigma = _robust_sigma(residual[keep])
        if not np.isfinite(sigma) or sigma <= 0.0:
            break

        lower_bound = -clip_sigma * sigma
        upper_bound = 5.0 * clip_sigma * sigma
        keep = base_mask & (residual > lower_bound) & (residual < upper_bound)
        weights = np.where(residual >= 0.0, 3.0, 1.0)

    return continuum


def flux_conserving_resample(
    wave_in_nm: ArrayLike1D,
    flux_in: ArrayLike1D,
    wave_out_nm: ArrayLike1D,
) -> np.ndarray:
    """Resample by integrating the input spectrum over output sample bins."""
    wave_in = np.asarray(wave_in_nm, dtype=float)
    flux = np.asarray(flux_in, dtype=float)
    wave_out = np.asarray(wave_out_nm, dtype=float)

    _validate_wave_flux(wave_in, flux)
    _validate_wave(wave_out)

    edges_out = _sample_centers_to_edges(wave_out)
    cumulative = _cumulative_trapezoid(wave_in, flux)

    cumulative_at_edges = _interp_cumulative(
        xq=edges_out,
        x=wave_in,
        cumulative=cumulative,
        y=flux,
    )

    widths = np.diff(edges_out)
    if np.any(widths <= 0.0):
        raise ValueError("Output wavelength bins must have positive widths.")

    return np.diff(cumulative_at_edges) / widths


def _prepare_high_res_input(
    wave_nm_hr: Union[ArrayLike1D, ModelCallable],
    flux_hr: Optional[ArrayLike1D],
    model_wave_min_nm: float,
    model_wave_max_nm: float,
    model_step_nm: float,
) -> Tuple[np.ndarray, np.ndarray, str]:
    if callable(wave_nm_hr):
        model = wave_nm_hr
        if flux_hr is None:
            if model_step_nm <= 0.0:
                raise ValueError("model_step_nm must be positive.")
            if model_wave_max_nm <= model_wave_min_nm:
                raise ValueError("model_wave_max_nm must be > model_wave_min_nm.")
            wave = np.arange(
                model_wave_min_nm,
                model_wave_max_nm + model_step_nm / 2.0,
                model_step_nm,
                dtype=float,
            )
        else:
            wave = np.asarray(flux_hr, dtype=float)
        flux = np.asarray(model(wave), dtype=float)
        input_kind = "callable"
    else:
        if flux_hr is None:
            raise ValueError("flux_hr is required when wave_nm_hr is an array.")
        wave = np.asarray(wave_nm_hr, dtype=float)
        flux = np.asarray(flux_hr, dtype=float)
        input_kind = "arrays"

    _validate_wave_flux(wave, flux)
    return wave, flux, input_kind


def _convolve_constant_R_lnlambda(
    wave_nm: np.ndarray,
    flux: np.ndarray,
    R: float,
    n_sigma_kernel: float = 6.0,
) -> Tuple[np.ndarray, np.ndarray]:
    _validate_wave_flux(wave_nm, flux)
    ln_wave = np.log(wave_nm)
    dln_native = np.median(np.diff(ln_wave))

    sigma_ln = 1.0 / (2.355 * R)
    dln_target = max(dln_native, sigma_ln / 6.0)

    ln_uniform = np.arange(ln_wave[0], ln_wave[-1] + dln_target / 2.0, dln_target)
    flux_uniform = np.interp(ln_uniform, ln_wave, flux)

    sigma_pix = sigma_ln / dln_target
    flux_conv_uniform = _gaussian_convolve_uniform(
        flux_uniform,
        sigma_pix=sigma_pix,
        n_sigma_kernel=n_sigma_kernel,
    )

    wave_uniform = np.exp(ln_uniform)
    return wave_uniform, flux_conv_uniform


def _convolve_quick_constant_fwhm(
    wave_nm: np.ndarray,
    flux: np.ndarray,
    R: float,
    lambda0_nm: float = 860.0,
    n_sigma_kernel: float = 6.0,
) -> Tuple[np.ndarray, np.ndarray]:
    _validate_wave_flux(wave_nm, flux)

    fwhm_nm = lambda0_nm / R
    sigma_nm = fwhm_nm / 2.355

    dlam_native = np.median(np.diff(wave_nm))
    dlam_target = max(dlam_native, sigma_nm / 6.0)

    wave_uniform = np.arange(wave_nm[0], wave_nm[-1] + dlam_target / 2.0, dlam_target)
    flux_uniform = np.interp(wave_uniform, wave_nm, flux)

    sigma_pix = sigma_nm / dlam_target
    flux_conv_uniform = _gaussian_convolve_uniform(
        flux_uniform,
        sigma_pix=sigma_pix,
        n_sigma_kernel=n_sigma_kernel,
    )
    return wave_uniform, flux_conv_uniform


def _gaussian_convolve_uniform(
    flux: np.ndarray,
    sigma_pix: float,
    n_sigma_kernel: float = 6.0,
) -> np.ndarray:
    if sigma_pix <= 0.0:
        raise ValueError("sigma_pix must be positive.")
    if sigma_pix < 1e-10:
        return flux.copy()

    half_width = int(np.ceil(n_sigma_kernel * sigma_pix))
    grid = np.arange(-half_width, half_width + 1, dtype=float)
    kernel = np.exp(-0.5 * (grid / sigma_pix) ** 2)
    kernel /= np.sum(kernel)

    flux_pad = np.pad(flux, (half_width, half_width), mode="edge")
    conv_pad = np.convolve(flux_pad, kernel, mode="same")
    return conv_pad[half_width:-half_width]


def _sample_centers_to_edges(centers: np.ndarray) -> np.ndarray:
    _validate_wave(centers)
    if centers.size == 1:
        raise ValueError("At least two output wavelength points are required.")

    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def _cumulative_trapezoid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = np.diff(x)
    increments = 0.5 * (y[:-1] + y[1:]) * dx
    cumulative = np.empty_like(x)
    cumulative[0] = 0.0
    cumulative[1:] = np.cumsum(increments)
    return cumulative


def _interp_cumulative(
    xq: np.ndarray,
    x: np.ndarray,
    cumulative: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    out = np.interp(xq, x, cumulative)

    left = xq < x[0]
    if np.any(left):
        out[left] = cumulative[0] + y[0] * (xq[left] - x[0])

    right = xq > x[-1]
    if np.any(right):
        out[right] = cumulative[-1] + y[-1] * (xq[right] - x[-1])

    return out


def _robust_sigma(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    median = np.median(vals)
    mad = np.median(np.abs(vals - median))
    sigma = 1.4826 * mad
    if sigma > 0.0 and np.isfinite(sigma):
        return float(sigma)
    return float(np.std(vals))


def _coerce_rng(rng: Optional[Union[int, np.random.Generator]]) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _validate_wave(wave_nm: np.ndarray) -> None:
    if wave_nm.ndim != 1:
        raise ValueError("Wavelength input must be 1D.")
    if wave_nm.size < 2:
        raise ValueError("Wavelength input must contain at least 2 samples.")
    if not np.all(np.isfinite(wave_nm)):
        raise ValueError("Wavelength input contains non-finite values.")
    if not np.all(np.diff(wave_nm) > 0.0):
        raise ValueError("Wavelength input must be strictly increasing.")


def _validate_wave_flux(wave_nm: np.ndarray, flux: np.ndarray) -> None:
    _validate_wave(wave_nm)
    if flux.ndim != 1:
        raise ValueError("Flux input must be 1D.")
    if flux.shape != wave_nm.shape:
        raise ValueError("wave_nm and flux arrays must have the same shape.")
    if not np.all(np.isfinite(flux)):
        raise ValueError("Flux input contains non-finite values.")


def _relativistic_doppler_factor(rv_kms: float) -> float:
    beta = float(rv_kms) / C_LIGHT_KMS
    if abs(beta) >= 1.0:
        raise ValueError("|rv_kms| must be < speed of light.")
    return float(np.sqrt((1.0 + beta) / (1.0 - beta)))
