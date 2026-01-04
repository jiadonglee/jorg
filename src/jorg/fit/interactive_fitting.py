"""
Interactive Log(gf) Fitting Tools for Stellar Spectral Lines

This module provides interactive tools for fitting oscillator strengths
to observed equivalent widths, designed for Jupyter notebook usage.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from scipy.optimize import minimize_scalar

from ..synthesis import SynthesisResult, synthesize_korg_compatible
from .equivalent_width import calculate_equivalent_width, EWResult, EWCalculator
from ..lines.linelist_modifier import LogGFModifier
from ..lines.datatypes import Line, LineData


@dataclass
class LineFitState:
    """
    State tracking for a single line during interactive fitting.

    Attributes
    ----------
    wavelength : float
        Line wavelength in Angstroms
    original_loggf : float
        Original log(gf) value from linelist
    current_loggf : float
        Current log(gf) value (after modifications)
    observed_ew : Optional[float]
        Observed equivalent width in mA (if provided)
    synthetic_ew : Optional[float]
        Current synthetic equivalent width in mA
    residual : float
        Difference (observed - synthetic) in mA
    iteration : int
        Number of iterations performed
    """
    wavelength: float
    original_loggf: float
    current_loggf: float
    observed_ew: Optional[float] = None
    synthetic_ew: Optional[float] = None
    residual: float = 0.0
    iteration: int = 0

    @property
    def delta_loggf(self) -> float:
        """Change in log(gf) from original."""
        return self.current_loggf - self.original_loggf

    def copy(self) -> 'LineFitState':
        """Create a copy of this state."""
        return LineFitState(
            wavelength=self.wavelength,
            original_loggf=self.original_loggf,
            current_loggf=self.current_loggf,
            observed_ew=self.observed_ew,
            synthetic_ew=self.synthetic_ew,
            residual=self.residual,
            iteration=self.iteration
        )


@dataclass
class FittingSessionResult:
    """
    Result of a fitting session for multiple lines.

    Attributes
    ----------
    lines : Dict[float, LineFitState]
        Dictionary mapping wavelength to fit state for each line
    converged : int
        Number of lines that successfully converged
    chi_squared : float
        Total chi-squared of the fit
    """
    lines: Dict[float, LineFitState] = field(default_factory=dict)
    converged: int = 0
    chi_squared: float = 0.0

    def as_dataframe(self):
        """Convert results to pandas DataFrame for easy viewing."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for DataFrame export")

        data = []
        for wl, state in self.lines.items():
            data.append({
                'wavelength': wl,
                'original_loggf': state.original_loggf,
                'final_loggf': state.current_loggf,
                'delta_loggf': state.delta_loggf,
                'observed_ew_mA': state.observed_ew,
                'synthetic_ew_mA': state.synthetic_ew,
                'residual_mA': state.residual,
                'iterations': state.iteration
            })

        return pd.DataFrame(data)

    def summary(self) -> str:
        """Get a text summary of fitting results."""
        lines = []
        lines.append(f"=== Fitting Session Summary ===")
        lines.append(f"Total lines: {len(self.lines)}")
        lines.append(f"Converged: {self.converged}")
        lines.append(f"Chi-squared: {self.chi_squared:.3f}")
        lines.append("")

        for wl, state in sorted(self.lines.items()):
            status = "✓" if state.observed_ew and abs(state.residual) < 5 else " "
            lines.append(
                f"{status} {wl:7.2f} Å: log(gf) {state.original_loggf:7.3f} → "
                f"{state.current_loggf:7.3f} (Δ{state.delta_loggf:+6.3f})"
            )
            if state.observed_ew:
                lines.append(
                    f"   EW: {state.observed_ew:.1f} → {state.synthetic_ew:.1f} mA "
                    f"(residual: {state.residual:+.1f})"
                )

        return "\n".join(lines)


class LineFittingSession:
    """
    Interactive session for fitting log(gf) to observed equivalent widths.

    This class manages the fitting process for multiple spectral lines,
    allowing interactive adjustment and optimization of oscillator strengths.

    Parameters
    ----------
    atm : dict
        Atmospheric model
    A_X : np.ndarray
        Abundance array
    linelist : list
        Spectral line list
    observed_ews : dict
        Dictionary of {wavelength: observed_EW_mA} for lines to fit
    wavelength_range : tuple, optional
        Wavelength range for synthesis (default: auto-detected)
    window_size : float, optional
        Window size for EW calculation in Angstroms (default: 2.0)
    vmic : float, optional
        Microturbulent velocity in km/s (default: 1.0)
    wavelength_tolerance : float, optional
        Tolerance for matching wavelengths in Angstroms (default: 0.01)
    **synth_kwargs
        Additional arguments passed to synthesize_korg_compatible()

    Examples
    --------
    >>> from jorg.synthesis import synthesize_korg_compatible
    >>> from jorg.atmosphere import interpolate_marcs
    >>> from jorg.lines.linelist import read_linelist
    >>> from jorg.fit.interactive_fitting import LineFittingSession
    >>> from jorg.synthesis import create_korg_compatible_abundance_array
    >>>
    >>> # Setup
    >>> atm = interpolate_marcs(5780, 4.44, 0.0)
    >>> A_X = create_korg_compatible_abundance_array(0.0)
    >>> linelist = read_linelist('lines.vald', format='vald')
    >>>
    >>> # Observed EWs (in mA)
    >>> observed = {5001.2: 85.3, 5005.8: 42.1, 5010.3: 67.8}
    >>>
    >>> # Create session
    >>> session = LineFittingSession(atm, A_X, linelist, observed)
    >>>
    >>> # Fit a single line
    >>> result = session.fit_single_line(5001.2)
    >>> print(f"Best log(gf): {result['best_loggf']:.3f}")
    >>>
    >>> # Fit all lines
    >>> session.fit_all()
    >>> print(session.summary())
    >>>
    >>> # Plot comparison
    >>> session.plot_comparison(5001.2)
    >>>
    >>> # Get results as table
    >>> df = session.get_results().as_dataframe()
    """

    def __init__(
        self,
        atm: Dict,
        A_X: np.ndarray,
        linelist: List,
        observed_ews: Dict[float, float],
        wavelength_range: Optional[Tuple[float, float]] = None,
        window_size: float = 2.0,
        vmic: float = 1.0,
        wavelength_tolerance: float = 0.01,
        **synth_kwargs
    ):
        self.atm = atm
        self.A_X = A_X
        self.original_linelist = linelist
        self.observed_ews = observed_ews
        self.window_size = window_size
        self.vmic = vmic
        self.wavelength_tolerance = wavelength_tolerance
        self.synth_kwargs = synth_kwargs

        # Auto-detect wavelength range if not provided
        if wavelength_range is None:
            wl_min = min(observed_ews.keys()) - 5.0
            wl_max = max(observed_ews.keys()) + 5.0
            wavelength_range = (wl_min, wl_max)
        self.wavelength_range = wavelength_range

        # Create modifier for linelist adjustments
        self.modifier = LogGFModifier(linelist, wavelength_tolerance=wavelength_tolerance)

        # Track line states
        self._line_states: Dict[float, LineFitState] = {}
        self._initialize_states()

        # EW calculator for efficient repeated calculations
        self._calculator = None

        # History for undo/analysis
        self._history: List[Dict[float, float]] = []

    def _initialize_states(self):
        """Initialize fit states for all observed lines."""
        for wl in self.observed_ews.keys():
            try:
                line = self.modifier.get_original_line(wl)
                if line is None:
                    continue

                original_loggf = line.log_gf if isinstance(line, Line) else line.log_gf

                self._line_states[wl] = LineFitState(
                    wavelength=wl,
                    original_loggf=original_loggf,
                    current_loggf=original_loggf,
                    observed_ew=self.observed_ews[wl]
                )
            except ValueError:
                import warnings
                warnings.warn(f"Could not find line at {wl:.2f} Å in linelist")

    def _get_synthetic_ew(self, wavelength: float) -> float:
        """Get synthetic EW for a line at current state."""
        # Recalculate EWs
        modified_linelist = self.modifier.apply_modifications()

        result = synthesize_korg_compatible(
            atm=self.atm,
            linelist=modified_linelist,
            A_X=self.A_X,
            wavelengths=self.wavelength_range,
            vmic=self.vmic,
            **self.synth_kwargs
        )

        ew_result = calculate_equivalent_width(
            result.wavelengths,
            result.flux,
            result.cntm,
            line_center=wavelength,
            window_size=self.window_size
        )

        return ew_result.ew_mA

    def _objective_function(self, delta_loggf: float, wavelength: float) -> float:
        """
        Objective function for optimization.

        Parameters
        ----------
        delta_loggf : float
            Change in log(gf) to test
        wavelength : float
            Line wavelength

        Returns
        -------
        float
            Squared residual between observed and synthetic EW
        """
        # Save current state
        current_state = self._line_states[wavelength].copy()
        original_loggf = current_state.original_loggf

        # Apply test modification
        test_loggf = original_loggf + delta_loggf
        self.modifier.set_line(wavelength, test_loggf)

        # Calculate EW
        try:
            synthetic_ew = self._get_synthetic_ew(wavelength)
            residual = self.observed_ews[wavelength] - synthetic_ew
            return residual ** 2
        except Exception:
            return np.inf
        finally:
            # Restore original
            self.modifier.set_line(wavelength, current_state.current_loggf)

    def fit_single_line(
        self,
        wavelength: float,
        method: str = 'brent',
        bounds: Tuple[float, float] = (-2.0, 2.0),
        tolerance: float = 0.1,
        verbose: bool = True
    ) -> Dict:
        """
        Fit log(gf) for a single line to match observed EW.

        Parameters
        ----------
        wavelength : float
            Line wavelength in Angstroms
        method : str, optional
            Optimization method ('brent', 'golden', or 'bounded')
        bounds : tuple, optional
            Search bounds for delta_loggf (default: ±2.0 dex)
        tolerance : float, optional
            EW tolerance for convergence in mA (default: 0.1)
        verbose : bool, optional
            Print progress messages

        Returns
        -------
        dict
            Fitting result with keys:
            - 'wavelength': line wavelength
            - 'best_loggf': best fit log(gf)
            - 'synthetic_ew': synthetic EW at best fit
            - 'residual': residual (observed - synthetic)
            - 'converged': whether fit converged
            - 'iterations': optimization iterations

        Examples
        --------
        >>> result = session.fit_single_line(5001.2)
        >>> if result['converged']:
        ...     print(f"Best log(gf): {result['best_loggf']:.3f}")
        ... else:
        ...     print("Fit did not converge")
        """
        if wavelength not in self._line_states:
            raise ValueError(f"No observed EW for line at {wavelength:.2f} Å")

        state = self._line_states[wavelength]
        target_ew = self.observed_ews[wavelength]

        if verbose:
            print(f"Fitting line at {wavelength:.2f} Å")
            print(f"  Target EW: {target_ew:.1f} mA")
            print(f"  Initial log(gf): {state.original_loggf:.3f}")

        # Define objective for scipy
        def objective(delta):
            return self._objective_function(delta, wavelength)

        # Optimize (bounds require method='bounded' in scipy)
        method_to_use = method
        minimize_kwargs = {'method': method_to_use, 'options': {'xatol': 0.01}}
        if bounds is not None:
            if method_to_use in ('brent', 'golden'):
                method_to_use = 'bounded'
            minimize_kwargs['method'] = method_to_use
            minimize_kwargs['bounds'] = bounds

        result = minimize_scalar(objective, **minimize_kwargs)

        # Apply best fit
        best_delta = result.x
        best_loggf = state.original_loggf + best_delta

        self.modifier.set_line(wavelength, best_loggf)

        # Calculate final EW
        synthetic_ew = self._get_synthetic_ew(wavelength)
        residual = target_ew - synthetic_ew

        # Update state
        state.current_loggf = best_loggf
        state.synthetic_ew = synthetic_ew
        state.residual = residual
        state.iteration += 1

        converged = abs(residual) < tolerance

        if verbose:
            print(f"  Final log(gf): {best_loggf:.3f} (Δ{best_delta:+.3f})")
            print(f"  Synthetic EW: {synthetic_ew:.1f} mA")
            print(f"  Residual: {residual:+.1f} mA")
            print(f"  Converged: {converged}")

        return {
            'wavelength': wavelength,
            'best_loggf': best_loggf,
            'synthetic_ew': synthetic_ew,
            'residual': residual,
            'converged': converged,
            'iterations': result.nfev
        }

    def fit_all(
        self,
        tolerance: float = 0.1,
        verbose: bool = True
    ) -> FittingSessionResult:
        """
        Fit all lines in the session.

        Parameters
        ----------
        tolerance : float, optional
            EW tolerance for convergence in mA
        verbose : bool, optional
            Print progress for each line

        Returns
        -------
        FittingSessionResult
            Summary of fitting results for all lines

        Examples
        --------
        >>> results = session.fit_all()
        >>> print(results.summary())
        """
        converged = 0
        chi_squared = 0.0

        for wl in self._line_states.keys():
            result = self.fit_single_line(wl, tolerance=tolerance, verbose=verbose)

            if result['converged']:
                converged += 1

            chi_squared += (result['residual'] / tolerance) ** 2

        session_result = FittingSessionResult(
            lines={wl: state.copy() for wl, state in self._line_states.items()},
            converged=converged,
            chi_squared=chi_squared
        )

        # Save to history
        self._history.append({
            wl: state.current_loggf for wl, state in self._line_states.items()
        })

        return session_result

    def iterate_line(
        self,
        wavelength: float,
        delta_loggf: float,
        verbose: bool = True
    ) -> EWResult:
        """
        Manually adjust log(gf) by a step and see resulting EW.

        Parameters
        ----------
        wavelength : float
            Line wavelength in Angstroms
        delta_loggf : float
            Amount to adjust log(gf) (can be positive or negative)
        verbose : bool, optional
            Print progress

        Returns
        -------
        EWResult
            Resulting equivalent width measurement

        Examples
        --------
        >>> # Try increasing log(gf) by 0.1 dex
        >>> ew_result = session.iterate_line(5001.2, 0.1)
        >>> print(f"New EW: {ew_result.ew_mA:.1f} mA")
        """
        if wavelength not in self._line_states:
            raise ValueError(f"No state for line at {wavelength:.2f} Å")

        state = self._line_states[wavelength]

        # Apply adjustment
        self.modifier.adjust_line(wavelength, delta_loggf)

        # Get new EW
        synthetic_ew = self._get_synthetic_ew(wavelength)

        # Update state
        new_loggf = state.current_loggf + delta_loggf
        state.current_loggf = new_loggf
        state.synthetic_ew = synthetic_ew
        state.residual = (state.observed_ew or 0) - synthetic_ew
        state.iteration += 1

        if verbose:
            print(f"{wavelength:.2f} Å: log(gf) {state.original_loggf:.3f} → "
                  f"{new_loggf:.3f} (Δ{delta_loggf:+.3f})")
            print(f"  EW: {synthetic_ew:.1f} mA")

        # Return full EW result
        modified_linelist = self.modifier.apply_modifications()
        result = synthesize_korg_compatible(
            atm=self.atm,
            linelist=modified_linelist,
            A_X=self.A_X,
            wavelengths=self.wavelength_range,
            vmic=self.vmic,
            **self.synth_kwargs
        )

        return calculate_equivalent_width(
            result.wavelengths,
            result.flux,
            result.cntm,
            line_center=wavelength,
            window_size=self.window_size
        )

    def plot_comparison(
        self,
        wavelength: float,
        window_size: Optional[float] = None,
        show_observed: bool = True,
        figsize: Tuple[float, float] = (10, 4)
    ):
        """
        Plot observed vs synthetic spectrum around a line.

        Parameters
        ----------
        wavelength : float
            Line wavelength in Angstroms
        window_size : float, optional
            Plot window size in Angstroms (default: uses session default)
        show_observed : bool, optional
            Show observed EW as horizontal bar
        figsize : tuple, optional
            Figure size

        Examples
        --------
        >>> session.plot_comparison(5001.2)
        """
        if window_size is None:
            window_size = self.window_size * 2

        # Get current spectrum
        modified_linelist = self.modifier.apply_modifications()
        result = synthesize_korg_compatible(
            atm=self.atm,
            linelist=modified_linelist,
            A_X=self.A_X,
            wavelengths=self.wavelength_range,
            vmic=self.vmic,
            **self.synth_kwargs
        )

        # Get spectrum around line
        wl = result.wavelengths
        flux = result.flux
        cntm = result.cntm

        mask = (wl >= wavelength - window_size/2) & (wl <= wavelength + window_size/2)
        wl_plot = wl[mask]
        flux_plot = flux[mask]
        cntm_plot = cntm[mask]

        # Normalize
        if cntm_plot is not None:
            flux_plot = flux_plot / cntm_plot

        # Plot
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(wl_plot, flux_plot, 'k-', linewidth=1.5, label='Synthetic')

        # Mark line center
        ax.axvline(wavelength, color='r', linestyle='--', alpha=0.5, label='Line center')

        # Show observed EW if available
        if show_observed and wavelength in self.observed_ews:
            observed_ew = self.observed_ews[wavelength]
            # Convert EW to approximate depth for visualization
            # This is a rough approximation for display only
            ax.axhline(1 - observed_ew/1000, color='b', linestyle=':',
                      label=f'Observed EW: {observed_ew:.1f} mA', alpha=0.5)

        ax.set_xlabel('Wavelength (Å)')
        ax.set_ylabel('Normalized Flux')
        ax.set_title(f'Line at {wavelength:.2f} Å')
        ax.legend(loc='best')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_ew_comparison(self, figsize: Tuple[float, float] = (8, 6)):
        """
        Plot observed vs synthetic EWs for all lines.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size

        Examples
        --------
        >>> session.fit_all(verbose=False)
        >>> session.plot_ew_comparison()
        """
        modified_linelist = self.modifier.apply_modifications()
        result = synthesize_korg_compatible(
            atm=self.atm,
            linelist=modified_linelist,
            A_X=self.A_X,
            wavelengths=self.wavelength_range,
            vmic=self.vmic,
            **self.synth_kwargs
        )

        # Calculate EWs for all lines
        observed = []
        synthetic = []
        wavelengths = []

        for wl in self._line_states.keys():
            ew_result = calculate_equivalent_width(
                result.wavelengths,
                result.flux,
                result.cntm,
                line_center=wl,
                window_size=self.window_size
            )

            wavelengths.append(wl)
            observed.append(self.observed_ews[wl])
            synthetic.append(ew_result.ew_mA)

        # Plot
        fig, ax = plt.subplots(figsize=figsize)

        ax.scatter(observed, synthetic, s=50, alpha=0.7)

        # Add 1:1 line
        min_ew = min(min(observed), min(synthetic))
        max_ew = max(max(observed), max(synthetic))
        ax.plot([min_ew, max_ew], [min_ew, max_ew], 'k--', alpha=0.5, label='1:1')

        # Label points
        for wl, obs, syn in zip(wavelengths, observed, synthetic):
            ax.annotate(f'{wl:.0f}', (obs, syn), fontsize=8, alpha=0.7)

        ax.set_xlabel('Observed EW (mA)')
        ax.set_ylabel('Synthetic EW (mA)')
        ax.set_title('Observed vs Synthetic Equivalent Widths')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def get_results(self) -> FittingSessionResult:
        """
        Get current fitting results.

        Returns
        -------
        FittingSessionResult
            Current state of all line fits

        Examples
        --------
        >>> results = session.get_results()
        >>> df = results.as_dataframe()
        >>> print(df)
        """
        # Calculate chi-squared
        chi_squared = 0.0
        converged = 0

        for state in self._line_states.values():
            if state.observed_ew and state.synthetic_ew:
                residual = state.residual
                chi_squared += residual ** 2
                if abs(residual) < 5.0:  # 5 mA tolerance
                    converged += 1

        return FittingSessionResult(
            lines={wl: state.copy() for wl, state in self._line_states.items()},
            converged=converged,
            chi_squared=chi_squared
        )

    def export_results(self, filename: str, format: str = 'csv'):
        """
        Export fitting results to file.

        Parameters
        ----------
        filename : str
            Output filename
        format : str, optional
            Output format ('csv' or 'json')

        Examples
        --------
        >>> session.export_results('fitted_loggf.csv')
        """
        results = self.get_results()

        if format.lower() == 'csv':
            try:
                import pandas as pd
            except ImportError:
                raise ImportError("pandas is required for CSV export")

            df = results.as_dataframe()
            df.to_csv(filename, index=False)

        elif format.lower() == 'json':
            import json
            data = {
                wl: {
                    'original_loggf': state.original_loggf,
                    'final_loggf': state.current_loggf,
                    'delta_loggf': state.delta_loggf,
                    'observed_ew_mA': state.observed_ew,
                    'synthetic_ew_mA': state.synthetic_ew,
                    'residual_mA': state.residual
                }
                for wl, state in results.lines.items()
            }
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)

        else:
            raise ValueError(f"Unknown format: {format}")

    def get_modified_linelist(self):
        """
        Get the modified linelist with current log(gf) values.

        Returns
        -------
        list
            Modified linelist

        Examples
        --------
        >>> modified_ll = session.get_modified_linelist()
        >>> # Save for later use
        >>> from jorg.lines.linelist_modifier import LogGFModifier
        >>> modifier = LogGFModifier(modified_ll)
        >>> modifier.to_file('fitted_lines.vald')
        """
        return self.modifier.apply_modifications()

    def reset_line(self, wavelength: float):
        """
        Reset a line to its original log(gf) value.

        Parameters
        ----------
        wavelength : float
            Line wavelength in Angstroms

        Examples
        --------
        >>> session.reset_line(5001.2)
        """
        if wavelength in self._line_states:
            state = self._line_states[wavelength]
            self.modifier.reset_line(wavelength)
            state.current_loggf = state.original_loggf
            state.synthetic_ew = None
            state.residual = 0.0

    def reset_all(self):
        """Reset all lines to original log(gf) values.

        Examples
        --------
        >>> session.reset_all()
        """
        self.modifier.reset_all()

        for state in self._line_states.values():
            state.current_loggf = state.original_loggf
            state.synthetic_ew = None
            state.residual = 0.0
            state.iteration = 0

    def summary(self) -> str:
        """Get a text summary of current state."""
        results = self.get_results()
        return results.summary()

    def __repr__(self) -> str:
        return (f"LineFittingSession(n_lines={len(self._line_states)}, "
                f"wavelength_range={self.wavelength_range})")


__all__ = [
    'LineFittingSession',
    'LineFitState',
    'FittingSessionResult',
]
