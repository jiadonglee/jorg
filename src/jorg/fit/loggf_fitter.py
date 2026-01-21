"""
LogGF Fitter - Optimized fitting for oscillator strengths

This module provides high-level APIs for fitting loggf values to observed spectra,
with automatic caching of continuum calculations for 2-5x speedup.

The LogGFFitter class is designed for:
- Fitting individual line loggf values
- Batch fitting multiple lines
- Interactive exploration of loggf adjustments

Usage:
    >>> from jorg.fit import LogGFFitter
    >>> fitter = LogGFFitter(atm, A_X, (5000, 5010), linelist)
    >>> result = fitter.fit_line(5001.2, observed_flux)
    >>> print(f"Best loggf: {result.best_loggf:.3f}")
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass
from scipy.optimize import minimize

# Import Jorg modules
from ..synthesis import synthesize_korg_compatible, resynthesize_from_continuum, SynthesisResult
from ..lines.linelist_modifier import LogGFModifier
from ..lines.linelist import read_linelist
from ..atmosphere import interpolate_marcs
from ..synthesis import create_korg_compatible_abundance_array


@dataclass
class FitResult:
    """
    Result of a loggf fitting operation.

    Attributes
    ----------
    best_delta_loggf : float
        Optimal delta_loggf adjustment to fit the observed spectrum
    best_loggf : float
        Final loggf value (original + delta)
    wavelengths : np.ndarray
        Wavelength grid for the fit
    best_fit_flux : np.ndarray
        Normalized flux from best-fit synthesis
    observed_flux : np.ndarray
        Observed flux (for comparison)
    success : bool
        Whether optimization converged successfully
    message : str
        Optimizer message
    n_evaluations : int
        Number of function evaluations
    """
    best_delta_loggf: float
    best_loggf: float
    wavelengths: np.ndarray
    best_fit_flux: np.ndarray
    observed_flux: np.ndarray
    success: bool
    message: str
    n_evaluations: int


class LogGFFitter:
    """
    Optimized fitter for loggf values using cached continuum calculations.

    This class is designed for fitting oscillator strengths to observed spectra,
    avoiding redundant calculations of continuum opacity and chemical equilibrium.

    Parameters
    ----------
    atm : dict
        Model atmosphere (from interpolate_marcs or similar)
    A_X : np.ndarray
        Abundance array (92-element, A(X) = log(X/H) + 12)
    wavelengths : tuple or np.ndarray
        Wavelength range (wl_min, wl_max) in Å or explicit wavelength array
    linelist : list
        Initial linelist (will be modified during fitting)
    vmic : float, default=1.0
        Microturbulent velocity in km/s
    cntm_step : float, default=1.0
        Continuum calculation step in Å
    hydrogen_lines : bool, default=False
        Include hydrogen lines (usually False for metal line fitting)
    verbose : bool, default=False
        Print progress information

    Examples
    --------
    >>> from jorg.fit import LogGFFitter
    >>> from jorg.atmosphere import interpolate_marcs
    >>> from jorg.synthesis import create_korg_compatible_abundance_array
    >>>
    >>> # Setup
    >>> atm = interpolate_marcs(5777, 4.44, 0.0)
    >>> A_X = create_korg_compatible_abundance_array(0.0)
    >>> linelist = read_linelist('solar_lines.vald')
    >>>
    >>> # Initialize fitter (performs one-time continuum calculation)
    >>> fitter = LogGFFitter(atm, A_X, (5000, 5010), linelist)
    >>>
    >>> # Fit a single line
    >>> result = fitter.fit_line(5001.2, observed_flux)
    >>> print(f"Best loggf adjustment: {result.best_delta_loggf:.3f}")
    >>>
    >>> # Quick synthesis with custom loggf
    >>> flux = fitter.synthesize_with_loggf(0.1, 5001.2)
    """

    def __init__(
        self,
        atm: Dict,
        A_X: np.ndarray,
        wavelengths: Union[Tuple[float, float], np.ndarray],
        linelist: List,
        vmic: float = 1.0,
        cntm_step: float = 1.0,
        hydrogen_lines: bool = False,
        verbose: bool = False
    ):
        """
        Initialize fitter with one-time continuum synthesis.

        This performs a full synthesis with continuum caching, which is reused
        for all subsequent fitting operations.
        """
        self.atm = atm
        self.A_X = A_X
        self.wavelengths = wavelengths
        self.base_linelist = linelist
        self.vmic = vmic
        self.cntm_step = cntm_step
        self.hydrogen_lines = hydrogen_lines
        self.verbose = verbose

        # Cached quantities
        self._cached_result: Optional[SynthesisResult] = None
        self._original_loggfs: Dict[float, float] = {}

        # Perform initial synthesis with caching
        self._initialize_cache()

    def _initialize_cache(self):
        """Perform initial synthesis and cache continuum-dependent quantities."""
        if self.verbose:
            print("🔄 LogGFFitter: Performing initial synthesis with continuum caching...")

        # Store original loggf values for all lines
        for line in self.base_linelist:
            wl_angstrom = float(line.wavelength) * 1e8  # Convert cm to Å
            self._original_loggfs[wl_angstrom] = float(line.log_gf)

        # Perform initial synthesis with continuum caching
        self._cached_result = synthesize_korg_compatible(
            atm=self.atm,
            linelist=self.base_linelist,
            A_X=self.A_X,
            wavelengths=self.wavelengths,
            vmic=self.vmic,
            cntm_step=self.cntm_step,
            hydrogen_lines=self.hydrogen_lines,
            export_intermediate_results=True,  # Need temperature for resynthesis
            verbose=self.verbose
        )

        if self.verbose:
            print(f"✅ LogGFFitter: Initial synthesis complete, continuum cached")
            print(f"   Wavelengths: {len(self._cached_result.wavelengths)} points")
            print(f"   Alpha continuum shape: {self._cached_result.alpha_continuum.shape}")

    def synthesize_with_loggf(
        self,
        delta_loggf: float,
        line_center: float,
        wavelength_tolerance: float = 0.01
    ) -> np.ndarray:
        """
        Quick synthesis with adjusted loggf using cached continuum.

        Parameters
        ----------
        delta_loggf : float
            Adjustment to loggf (in dex)
        line_center : float
            Line center wavelength in Angstroms
        wavelength_tolerance : float, default=0.01
            Tolerance for matching line wavelengths in Angstroms

        Returns
        -------
        np.ndarray
            Normalized flux (flux / continuum)

        Examples
        --------
        >>> fitter = LogGFFitter(atm, A_X, (5000, 5010), linelist)
        >>> flux = fitter.synthesize_with_loggf(0.1, 5001.2)  # Increase loggf by 0.1 dex
        """
        # Create modified linelist
        modifier = LogGFModifier(self.base_linelist, wavelength_tolerance=wavelength_tolerance)
        modifier.adjust_line(line_center, delta_loggf=delta_loggf)
        modified_linelist = modifier.apply_modifications()

        # Fast resynthesis using cached continuum
        result = resynthesize_from_continuum(
            previous_result=self._cached_result,
            linelist=modified_linelist,
            wavelengths=self.wavelengths,
            vmic=self.vmic,
            hydrogen_lines=self.hydrogen_lines,
            verbose=False
        )

        # Return normalized flux
        if result.cntm is not None:
            return np.asarray(result.flux) / np.asarray(result.cntm)
        else:
            return np.asarray(result.flux)

    def fit_line(
        self,
        line_center: float,
        observed_flux: np.ndarray,
        observed_wavelengths: Optional[np.ndarray] = None,
        observed_err: Optional[np.ndarray] = None,
        postprocess_flux: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        wavelength_tolerance: float = 0.01,
        method: str = 'BFGS',
        bounds: Optional[Tuple[float, float]] = None,
        **opt_kwargs
    ) -> FitResult:
        """
        Fit loggf for a single line to match observed flux.

        Parameters
        ----------
        line_center : float
            Line center wavelength in Angstroms
        observed_flux : np.ndarray
            Observed normalized flux (flux/continuum)
        observed_wavelengths : np.ndarray, optional
            Wavelength array for observed flux. If None, uses fitter's wavelength grid.
        observed_err : np.ndarray, optional
            Observational errors (for chi-squared weighting)
        postprocess_flux : callable, optional
            Function applied to model flux before comparison, e.g., instrumental convolution.
            Must accept (flux, wavelengths) and return processed flux on the same grid.
        wavelength_tolerance : float, default=0.01
            Tolerance for matching line wavelengths in Angstroms
        method : str, default='BFGS'
            Optimization method (see scipy.optimize.minimize)
        bounds : tuple, optional
            Bounds for delta_loggf, e.g., (-1.0, 1.0)
        **opt_kwargs : dict
            Additional arguments passed to scipy.optimize.minimize

        Returns
        -------
        FitResult
            Fitting result with best loggf value and diagnostic information

        Examples
        --------
        >>> fitter = LogGFFitter(atm, A_X, (5000, 5010), linelist)
        >>> result = fitter.fit_line(5001.2, observed_flux, method='BFGS')
        >>> print(f"Best delta_loggf: {result.best_delta_loggf:.3f}")
        >>> print(f"Final loggf: {result.best_loggf:.3f}")
        >>> print(f"Success: {result.success}")
        """
        # Get original loggf for this line
        original_loggf = self._original_loggfs.get(
            round(line_center, 4),  # Round to avoid floating point issues
            self._find_closest_loggf(line_center)
        )

        # Setup wavelength grid for fitting
        if observed_wavelengths is None:
            fit_wavelengths = self._cached_result.wavelengths
        else:
            fit_wavelengths = observed_wavelengths

        # Define objective function
        def objective(x):
            delta = float(x[0])
            model_flux = self.synthesize_with_loggf(delta, line_center, wavelength_tolerance)

            if postprocess_flux is not None:
                model_flux = postprocess_flux(model_flux, fit_wavelengths)

            # Interpolate to observed wavelength grid if needed
            if observed_wavelengths is not None:
                model_flux = np.interp(observed_wavelengths, fit_wavelengths, model_flux)

            # Chi-squared or MSE depending on whether errors are provided
            if observed_err is not None:
                return float(np.sum(((model_flux - observed_flux) / observed_err) ** 2))
            else:
                return float(np.mean((model_flux - observed_flux) ** 2))

        # Initial guess (no adjustment)
        x0 = np.array([0.0])

        # Set default options (method-specific)
        if method == 'BFGS':
            # BFGS doesn't accept 'disp', uses 'gtol' and 'maxiter'
            default_options = {'gtol': 1e-4, 'maxiter': 20}
        else:
            # Other methods may accept different options
            default_options = {'maxiter': 20}
        default_options.update(opt_kwargs)

        # Run optimization
        if bounds is not None:
            from scipy.optimize import minimize_scalar
            result = minimize_scalar(
                objective,
                bounds=bounds,
                method='bounded',
                options=default_options
            )
            best_delta = result.x
            success = result.success
            message = result.message
            n_eval = result.nfev
        else:
            result = minimize(objective, x0, method=method, options=default_options)
            best_delta = float(result.x[0])
            success = result.success
            message = result.message
            n_eval = result.nfev

        # Get best-fit spectrum
        best_fit_flux = self.synthesize_with_loggf(best_delta, line_center, wavelength_tolerance)

        if postprocess_flux is not None:
            best_fit_flux = postprocess_flux(best_fit_flux, fit_wavelengths)

        # Interpolate to observed wavelength grid if needed
        if observed_wavelengths is not None:
            best_fit_flux = np.interp(observed_wavelengths, fit_wavelengths, best_fit_flux)

        # Get original loggf value
        closest_loggf = self._find_closest_loggf(line_center)

        return FitResult(
            best_delta_loggf=best_delta,
            best_loggf=closest_loggf + best_delta,
            wavelengths=observed_wavelengths if observed_wavelengths is not None else fit_wavelengths,
            best_fit_flux=best_fit_flux,
            observed_flux=observed_flux,
            success=success,
            message=message,
            n_evaluations=n_eval
        )

    def _find_closest_loggf(self, line_center: float) -> float:
        """Find the loggf value for the line closest to the given wavelength."""
        wavelengths = np.array(list(self._original_loggfs.keys()))
        idx = np.argmin(np.abs(wavelengths - line_center))
        closest_wl = wavelengths[idx]
        return self._original_loggfs[closest_wl]

    def get_speedup(self) -> float:
        """
        Return estimated speedup factor from using cached continuum.

        This is a rough estimate based on the relative computational cost
        of continuum vs. line calculations.

        Returns
        -------
        float
            Estimated speedup factor (typically 2-5x)
        """
        # Based on typical computational costs:
        # - Chemical equilibrium: ~50-60%
        # - Continuum opacity: ~15-20%
        # - Line opacity: ~20-30%
        # - Radiative transfer: ~10-15%
        #
        # Reusing continuum saves CE + continuum calculation: ~70-75%
        # So speedup is roughly 1/(1-0.7) ≈ 3-4x
        return 3.5  # Conservative estimate


# Convenience function for quick fitting
def fit_loggf_quick(
    Teff: float,
    logg: float,
    m_H: float,
    wavelengths: Union[Tuple[float, float], np.ndarray],
    linelist: List,
    line_center: float,
    observed_flux: np.ndarray,
    alpha_H: Optional[float] = None,
    vmic: float = 1.0,
    **fit_kwargs
) -> FitResult:
    """
    Quick loggf fitting with automatic atmosphere and abundance setup.

    This is a convenience function that handles atmosphere interpolation and
    abundance array creation automatically.

    Parameters
    ----------
    Teff : float
        Effective temperature in K
    logg : float
        Surface gravity (log g)
    m_H : float
        Metallicity [metals/H]
    wavelengths : tuple or array
        Wavelength range for fitting
    linelist : list
        Line list
    line_center : float
        Line center wavelength in Angstroms
    observed_flux : np.ndarray
        Observed normalized flux
    alpha_H : float, optional
        Alpha enhancement (defaults to m_H)
    vmic : float, default=1.0
        Microturbulent velocity in km/s
    **fit_kwargs : dict
        Additional arguments passed to LogGFFitter.fit_line()

    Returns
    -------
    FitResult
        Fitting result

    Examples
    --------
    >>> from jorg.fit import fit_loggf_quick
    >>> from jorg.lines import read_linelist
    >>>
    >>> linelist = read_linelist('solar.vald')
    >>> result = fit_loggf_quick(
    ...     5777, 4.44, 0.0, (5000, 5010), linelist,
    ...     5001.2, observed_flux
    ... )
    >>> print(f"Best loggf: {result.best_loggf:.3f}")
    """
    # Setup atmosphere and abundances
    atm = interpolate_marcs(Teff, logg, m_H)
    A_X = create_korg_compatible_abundance_array(m_H, alpha_H if alpha_H is not None else m_H)

    # Create fitter
    fitter = LogGFFitter(atm, A_X, wavelengths, linelist, vmic=vmic, verbose=False)

    # Fit line
    return fitter.fit_line(line_center, observed_flux, **fit_kwargs)
