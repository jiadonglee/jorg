"""RVS-like synthetic spectrum utilities."""

from .synthetic import (
    RVSSpectrum,
    estimate_pseudo_continuum,
    flux_conserving_resample,
    gaia_rvs_wavelength_grid,
    make_rvs_like,
)

__all__ = [
    "RVSSpectrum",
    "estimate_pseudo_continuum",
    "flux_conserving_resample",
    "gaia_rvs_wavelength_grid",
    "make_rvs_like",
]
