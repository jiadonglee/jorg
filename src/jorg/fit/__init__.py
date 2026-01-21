"""
Parameter Fitting Module for Jorg - JAX-Optimized Stellar Spectral Analysis
=========================================================================

This module provides JAX-optimized parameter fitting capabilities for stellar spectral
analysis, including both spectral fitting and equivalent width analysis.

Key Features:
- GPU-accelerated optimization with automatic differentiation
- Spectral fitting with chi-squared minimization
- Optimized log(gf) fitting with cached continuum calculations (2-5x speedup)
- Interactive log(gf) fitting tools for oscillator strength determination
- Equivalent width calculation from synthetic spectra
- Classical stellar parameter determination via equivalent widths

Main Functions:
- fit_spectrum: Full spectral fitting with BFGS optimization
- calculate_equivalent_width: Equivalent width calculation for single line
- calculate_equivalent_widths: Batch equivalent width computation
- LineFittingSession: Interactive log(gf) fitting to observed EWs
- LogGFFitter: Optimized loggf fitting with continuum caching
- fit_loggf_quick: Convenience function for quick loggf fitting

Performance:
- JAX JIT compilation for optimized machine code
- GPU acceleration for 10-100x speedup
- Vectorized operations for batch processing
- Memory-efficient gradient computation
- Continuum caching for 2-5x speedup in loggf fitting

Author: JAX Implementation Team
Created: July 2025
Status: Development Phase
"""

# Core fitting functions
_core_import_error = None
try:
    from .core import (
        fit_spectrum,
        FitResult,
        FitParameters,
        validate_fit_parameters
    )
except Exception as exc:  # pragma: no cover - optional core dependency
    _core_import_error = exc

    def _core_unavailable(*_args, **_kwargs):
        raise ImportError(
            "jorg.fit.core could not be imported; fit_spectrum and related APIs "
            "are unavailable in this environment."
        ) from _core_import_error

    fit_spectrum = _core_unavailable
    validate_fit_parameters = _core_unavailable
    FitResult = None
    FitParameters = None

# Equivalent width calculation (NEW: native Python implementation)
from .equivalent_width import (
    calculate_equivalent_width,
    calculate_equivalent_widths,
    equivalent_width_from_linelist,
    EWCalculator,
    EWResult,
    EWFitResult,
    calculate_EW,
    calculate_EWs,
)

# Interactive log(gf) fitting (NEW)
from .interactive_fitting import (
    LineFittingSession,
    LineFitState,
    FittingSessionResult,
)

# Optimized loggf fitting with continuum caching (NEW - January 2026)
from .loggf_fitter import (
    LogGFFitter,
    fit_loggf_quick,
    FitResult as LogGFFitResult,
)

__all__ = [
    # Core fitting
    'fit_spectrum',
    'FitResult',
    'FitParameters',
    'validate_fit_parameters',

    # Equivalent width calculation (NEW)
    'calculate_equivalent_width',
    'calculate_equivalent_widths',
    'equivalent_width_from_linelist',
    'EWCalculator',
    'EWResult',
    'EWFitResult',
    'calculate_EW',
    'calculate_EWs',

    # Interactive log(gf) fitting (NEW)
    'LineFittingSession',
    'LineFitState',
    'FittingSessionResult',

    # Optimized loggf fitting with continuum caching (NEW)
    'LogGFFitter',
    'fit_loggf_quick',
    'LogGFFitResult',
]

# Module metadata
__version__ = '0.2.0'
__author__ = 'Jorg Development Team'
__email__ = 'jorg@stellar.synthesis'
__description__ = 'JAX-optimized parameter fitting for stellar spectral analysis'
