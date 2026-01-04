"""
Equivalent Width Calculation for Stellar Spectral Lines

This module provides native Python implementation of equivalent width (EW)
calculations for spectral lines, compatible with Jorg synthesis outputs.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import integrate

from ..synthesis import SynthesisResult


@dataclass
class EWResult:
    """
    Result of equivalent width calculation for a single line.

    Attributes
    ----------
    wavelength : float
        Line center wavelength in Angstroms
    ew_mA : float
        Equivalent width in milli-Angstroms (mA)
    ew_angstrom : float
        Equivalent width in Angstroms
    continuum_level : float
        Continuum flux level at line position
    line_depth : float
        Maximum line depth (1 - min_flux/continuum)
    bounds : tuple
        Integration bounds (wl_min, wl_max) in Angstroms
    blended : bool
        True if line appears to be blended with neighbors
    """
    wavelength: float
    ew_mA: float
    ew_angstrom: float
    continuum_level: float
    line_depth: float
    bounds: Tuple[float, float]
    blended: bool = False

    def __repr__(self) -> str:
        return (f"EWResult(wl={self.wavelength:.2f} Å, "
                f"EW={self.ew_mA:.1f} mA, depth={self.line_depth:.3f})")


def calculate_equivalent_width(
    wavelengths: np.ndarray,
    flux: np.ndarray,
    continuum: Optional[np.ndarray] = None,
    line_center: float = 0.0,
    window_size: float = 2.0,
    method: str = "trapz",
    auto_bounds: bool = True,
    continuum_threshold: float = 0.99
) -> EWResult:
    """
    Calculate equivalent width for a single spectral line.

    The equivalent width is defined as:
        EW = ∫ (1 - F(λ)/F_cont(λ)) dλ
    where the integral is over the line profile.

    Parameters
    ----------
    wavelengths : np.ndarray
        Wavelength array in Angstroms
    flux : np.ndarray
        Normalized flux array (0-1 scale) or raw flux
    continuum : np.ndarray, optional
        Continuum flux array. If None, assumes flux is already normalized.
    line_center : float, optional
        Line center wavelength in Angstroms. If 0.0, uses array center.
    window_size : float, optional
        Window size in Angstroms for line integration (default: 2.0 Å)
    method : str, optional
        Integration method: 'trapz' (trapezoidal) or 'simps' (Simpson's rule)
    auto_bounds : bool, optional
        If True (default), auto-detect line boundaries
    continuum_threshold : float, optional
        Flux threshold (relative to continuum) for boundary detection

    Returns
    -------
    EWResult
        Equivalent width result with details

    Examples
    --------
    >>> from jorg.synthesis import synth
    >>> from jorg.fit.equivalent_width import calculate_equivalent_width
    >>>
    >>> wl, flux, cntm = synth(5780, 4.44, 0.0, wavelengths=(5000, 5002))
    >>> result = calculate_equivalent_width(wl, flux, cntm, line_center=5001.2, window_size=1.0)
    >>> print(f"EW = {result.ew_mA:.1f} mA")
    """
    # Normalize flux if continuum provided
    if continuum is not None:
        normalized_flux = flux / continuum
        continuum_level = continuum[np.argmin(np.abs(wavelengths - line_center))]
    else:
        normalized_flux = flux
        continuum_level = 1.0

    # Determine line center if not provided
    if line_center == 0.0:
        line_center = (wavelengths[0] + wavelengths[-1]) / 2.0

    # Find window indices
    wl_min = line_center - window_size / 2
    wl_max = line_center + window_size / 2

    mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
    wl_window = wavelengths[mask]
    flux_window = normalized_flux[mask]

    if len(wl_window) < 3:
        raise ValueError(f"Window too small: only {len(wl_window)} points in range")

    # Auto-detect boundaries if requested
    if auto_bounds:
        # Find where flux returns to continuum level
        # Start from center and work outward
        center_idx = np.argmin(np.abs(wl_window - line_center))

        # Look left from center
        left_idx = 0
        for i in range(center_idx, 0, -1):
            if flux_window[i] >= continuum_threshold:
                left_idx = i
                break

        # Look right from center
        right_idx = len(flux_window) - 1
        for i in range(center_idx, len(flux_window)):
            if flux_window[i] >= continuum_threshold:
                right_idx = i
                break

        wl_min_bound = wl_window[left_idx]
        wl_max_bound = wl_window[right_idx]
        bounds = (wl_min_bound, wl_max_bound)

        # Trim to these bounds
        bound_mask = (wl_window >= wl_min_bound) & (wl_window <= wl_max_bound)
        wl_window = wl_window[bound_mask]
        flux_window = flux_window[bound_mask]
    else:
        bounds = (wl_min, wl_max)

    # Calculate equivalent width
    # EW = ∫ (1 - F/F_cont) dλ
    absorption = 1.0 - flux_window

    if method == "trapz":
        ew_angstrom = np.trapz(absorption, wl_window)
    elif method == "simps":
        ew_angstrom = integrate.simpson(absorption, wl_window)
    else:
        raise ValueError(f"Unknown integration method: {method}")

    # Convert to milli-Angstroms
    ew_mA = ew_angstrom * 1000.0

    # Calculate line depth
    min_flux = np.min(flux_window)
    line_depth = 1.0 - min_flux / continuum_level

    # Check for blending (simple heuristic: asymmetric profile)
    center_idx = np.argmin(np.abs(wl_window - line_center))
    left_half = flux_window[:center_idx]
    right_half = flux_window[center_idx:]

    if len(left_half) > 2 and len(right_half) > 2:
        # Compare left and right halves
        left_ew = np.trapz(1.0 - left_half, wl_window[:center_idx])
        right_ew = np.trapz(1.0 - right_half, wl_window[center_idx:])
        blended = abs(left_ew - right_ew) / (left_ew + right_ew) > 0.3
    else:
        blended = False

    return EWResult(
        wavelength=line_center,
        ew_mA=ew_mA,
        ew_angstrom=ew_angstrom,
        continuum_level=continuum_level,
        line_depth=line_depth,
        bounds=bounds,
        blended=blended
    )


def calculate_equivalent_widths(
    result: Union[SynthesisResult, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    line_centers: Optional[np.ndarray] = None,
    window_size: float = 2.0,
    method: str = "trapz",
    auto_bounds: bool = True
) -> Dict[float, EWResult]:
    """
    Calculate equivalent widths for multiple spectral lines.

    Parameters
    ----------
    result : SynthesisResult or tuple
        Synthesis result or (wavelengths, flux, continuum) tuple
    line_centers : np.ndarray, optional
        Array of line center wavelengths in Angstroms.
        If None, attempts to detect lines automatically.
    window_size : float, optional
        Window size in Angstroms for each line (default: 2.0)
    method : str, optional
        Integration method (default: 'trapz')
    auto_bounds : bool, optional
        Auto-detect line boundaries (default: True)

    Returns
    -------
    dict
        Dictionary mapping {wavelength: EWResult}

    Examples
    --------
    >>> from jorg.synthesis import synth
    >>> from jorg.fit.equivalent_width import calculate_equivalent_widths
    >>>
    >>> wl, flux, cntm = synth(5780, 4.44, 0.0, wavelengths=(5000, 5020))
    >>> lines = [5001.2, 5005.8, 5010.3]
    >>> ews = calculate_equivalent_widths((wl, flux, cntm), line_centers=lines)
    >>> for wl, result in ews.items():
    ...     print(f"{wl}: EW = {result.ew_mA:.1f} mA")
    """
    # Unpack result
    if isinstance(result, SynthesisResult):
        wavelengths = result.wavelengths
        flux = result.flux
        continuum = result.cntm
    else:
        wavelengths, flux, continuum = result

    # Auto-detect lines if not provided
    if line_centers is None:
        line_centers = _detect_lines(wavelengths, flux, continuum)

    results = {}

    for wl in line_centers:
        try:
            ew_result = calculate_equivalent_width(
                wavelengths, flux, continuum,
                line_center=wl,
                window_size=window_size,
                method=method,
                auto_bounds=auto_bounds
            )
            results[wl] = ew_result
        except ValueError as e:
            import warnings
            warnings.warn(f"Could not calculate EW at {wl:.2f} Å: {e}")

    return results


def _detect_lines(
    wavelengths: np.ndarray,
    flux: np.ndarray,
    continuum: Optional[np.ndarray] = None,
    min_depth: float = 0.01
) -> np.ndarray:
    """
    Automatically detect spectral lines.

    Parameters
    ----------
    wavelengths : np.ndarray
        Wavelength array
    flux : np.ndarray
        Flux array
    continuum : np.ndarray, optional
        Continuum array
    min_depth : float, optional
        Minimum line depth to detect (default: 1%)

    Returns
    -------
    np.ndarray
        Array of detected line centers
    """
    if continuum is not None:
        normalized = flux / continuum
    else:
        normalized = flux

    # Find local minima
    from scipy.signal import find_peaks
    # Invert to find minima as peaks
    inverted = 1.0 - normalized
    peaks, _ = find_peaks(inverted, prominence=min_depth)

    return wavelengths[peaks]


def equivalent_width_from_linelist(
    linelist: List,
    atm: Dict,
    A_X: np.ndarray,
    wavelengths: Union[Tuple[float, float], np.ndarray],
    line_centers: Optional[np.ndarray] = None,
    window_size: float = 2.0,
    **synth_kwargs
) -> Dict[float, EWResult]:
    """
    Convenience function: synthesize spectrum and calculate EWs in one call.

    Parameters
    ----------
    linelist : list
        Spectral line list
    atm : dict
        Atmospheric model
    A_X : np.ndarray
        Abundance array
    wavelengths : tuple or np.ndarray
        Wavelength range or grid
    line_centers : np.ndarray, optional
        Line centers to measure. If None, uses linelist wavelengths.
    window_size : float, optional
        Window size for EW calculation (default: 2.0)
    **synth_kwargs
        Additional arguments passed to synthesize()

    Returns
    -------
    dict
        Dictionary mapping {wavelength: EWResult}

    Examples
    --------
    >>> from jorg.synthesis import synthesize_korg_compatible
    >>> from jorg.fit.equivalent_width import equivalent_width_from_linelist
    >>>
    >>> ews = equivalent_width_from_linelist(
    ...     linelist, atm, A_X, wavelengths=(5000, 5010),
    ...     vmic=1.0, logg=4.44
    ... )
    """
    from ..synthesis import synthesize_korg_compatible

    result = synthesize_korg_compatible(
        atm=atm,
        linelist=linelist,
        A_X=A_X,
        wavelengths=wavelengths,
        **synth_kwargs
    )

    # Use linelist wavelengths if not provided
    if line_centers is None:
        from ..lines.datatypes import Line, LineData
        line_centers = []
        for line in linelist:
            if isinstance(line, Line):
                wl_angstrom = line.wl * 1e8
            else:
                wl_angstrom = line.wavelength * 1e8
            # Check if within wavelength range
            if hasattr(result, 'wavelengths'):
                wl_min, wl_max = result.wavelengths.min(), result.wavelengths.max()
            else:
                # Assume tuple was passed
                if isinstance(wavelengths, tuple):
                    wl_min, wl_max = wavelengths
                else:
                    wl_min, wl_max = wavelengths.min(), wavelengths.max()

            if wl_min <= wl_angstrom <= wl_max:
                line_centers.append(wl_angstrom)

        line_centers = np.array(line_centers) if line_centers else None

    return calculate_equivalent_widths(
        result,
        line_centers=line_centers,
        window_size=window_size
    )


class EWCalculator:
    """
    Calculator for equivalent widths with stateful configuration.

    This class is useful for repeated EW calculations with the same
    atmospheric model and wavelength grid.

    Parameters
    ----------
    atm : dict
        Atmospheric model
    A_X : np.ndarray
        Abundance array
    wavelength_range : tuple
        (wl_min, wl_max) in Angstroms
    window_size : float, optional
        Default window size for EW calculation (default: 2.0)
    **synth_kwargs
        Additional arguments for synthesis

    Examples
    --------
    >>> from jorg.fit.equivalent_width import EWCalculator
    >>>
    >>> calculator = EWCalculator(atm, A_X, (5000, 5010), vmic=1.0, logg=4.44)
    >>>
    >>> # Calculate EWs for different linelists
    >>> ews_original = calculator.calculate(linelist_original)
    >>> ews_modified = calculator.calculate(linelist_modified)
    >>>
    >>> # Compare
    >>> for wl in ews_original:
    ...     delta = ews_modified[wl].ew_mA - ews_original[wl].ew_mA
    ...     print(f"{wl}: ΔEW = {delta:.1f} mA")
    """

    def __init__(
        self,
        atm: Dict,
        A_X: np.ndarray,
        wavelength_range: Tuple[float, float],
        window_size: float = 2.0,
        **synth_kwargs
    ):
        self.atm = atm
        self.A_X = A_X
        self.wavelength_range = wavelength_range
        self.window_size = window_size
        self.synth_kwargs = synth_kwargs
        self._cached_result = None
        self._cached_linelist = None

    def calculate(
        self,
        linelist: List,
        line_centers: Optional[np.ndarray] = None,
        force_resynthesize: bool = False
    ) -> Dict[float, EWResult]:
        """
        Calculate EWs for a linelist.

        Parameters
        ----------
        linelist : list
            Spectral line list
        line_centers : np.ndarray, optional
            Specific line centers to measure
        force_resynthesize : bool, optional
            Force resynthesis even if linelist matches cached version

        Returns
        -------
        dict
            Dictionary mapping {wavelength: EWResult}
        """
        from ..synthesis import synthesize_korg_compatible

        # Check cache
        if not force_resynthesize and self._cached_linelist is linelist:
            result = self._cached_result
        else:
            result = synthesize_korg_compatible(
                atm=self.atm,
                linelist=linelist,
                A_X=self.A_X,
                wavelengths=self.wavelength_range,
                **self.synth_kwargs
            )
            self._cached_result = result
            self._cached_linelist = linelist

        return calculate_equivalent_widths(
            result,
            line_centers=line_centers,
            window_size=self.window_size
        )

    def compare_ews(
        self,
        linelist1: List,
        linelist2: List,
        line_centers: Optional[np.ndarray] = None
    ) -> Dict[float, Dict[str, float]]:
        """
        Compare EWs between two linelists.

        Parameters
        ----------
        linelist1, linelist2 : list
            Linelists to compare
        line_centers : np.ndarray, optional
            Specific lines to compare

        Returns
        -------
        dict
            Dictionary mapping {wavelength: {'ew1': ..., 'ew2': ..., 'delta': ...}}
        """
        ews1 = self.calculate(linelist1, line_centers)
        ews2 = self.calculate(linelist2, line_centers)

        comparison = {}

        all_wls = set(ews1.keys()) | set(ews2.keys())

        for wl in all_wls:
            ew1 = ews1.get(wl, EWResult(wl, 0, 0, 1, 0, (wl, wl)))
            ew2 = ews2.get(wl, EWResult(wl, 0, 0, 1, 0, (wl, wl)))

            comparison[wl] = {
                'ew1_mA': ew1.ew_mA,
                'ew2_mA': ew2.ew_mA,
                'delta_mA': ew2.ew_mA - ew1.ew_mA,
                'delta_percent': ((ew2.ew_mA - ew1.ew_mA) / ew1.ew_mA * 100) if ew1.ew_mA > 0 else np.inf
            }

        return comparison


# Alias for compatibility with Jorg naming conventions
calculate_EWs = calculate_equivalent_widths
calculate_EW = calculate_equivalent_width


@dataclass
class EWFitResult:
    """
    Result of fitting log(gf) to match observed equivalent widths.

    Attributes
    ----------
    wavelength : float
        Line wavelength in Angstroms
    observed_ew : float
        Observed equivalent width in mA
    initial_ew : float
        Initial synthetic EW (before fitting) in mA
    final_ew : float
        Final synthetic EW (after fitting) in mA
    initial_loggf : float
        Initial log(gf) value
    final_loggf : float
        Fitted log(gf) value
    delta_loggf : float
        Change in log(gf)
    residual : float
        Difference between observed and final EW (mA)
    converged : bool
        Whether fit converged
    iterations : int
        Number of iterations performed
    """
    wavelength: float
    observed_ew: float
    initial_ew: float
    final_ew: float
    initial_loggf: float
    final_loggf: float
    delta_loggf: float
    residual: float
    converged: bool
    iterations: int

    def __repr__(self) -> str:
        return (f"EWFitResult(wl={self.wavelength:.2f} Å, "
                f"log(gf): {self.initial_loggf:.3f}→{self.final_loggf:.3f}, "
                f"EW: {self.final_ew:.1f} mA, residual: {self.residual:.1f} mA)")


__all__ = [
    'calculate_equivalent_width',
    'calculate_equivalent_widths',
    'equivalent_width_from_linelist',
    'EWCalculator',
    'EWResult',
    'EWFitResult',
    'calculate_EW',
    'calculate_EWs',
]
