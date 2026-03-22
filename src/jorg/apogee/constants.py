"""Constants and shared metadata for APOGEE/TransformerPayne integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Tuple

import numpy as np

APOGEE_RESOLUTION: Final[int] = 22_500
APOGEE_WAVE_LOG10_START: Final[float] = float(np.log10(15100.802))
APOGEE_WAVE_LOG10_STEP: Final[float] = 6e-6
APOGEE_PIXELS: Final[int] = 8575
APOGEE_WAVE_RANGE_ANGSTROM: Final[Tuple[float, float]] = (15100.802, 16999.847)

DEFAULT_SYNTHESIS_START_ANGSTROM: Final[float] = 15_000.0
DEFAULT_SYNTHESIS_STOP_ANGSTROM: Final[float] = 17_000.0
DEFAULT_SYNTHESIS_STEP_ANGSTROM: Final[float] = 0.01

TP_COMPACT_LABEL_NAMES: Final[Tuple[str, ...]] = (
    "Teff",
    "logg",
    "m_h",
    "alpha_fe",
    "c_fe",
    "vmic",
    "mu",
)

TP_ALPHA_ELEMENTS: Final[Tuple[str, ...]] = ("O", "Mg", "Si", "S", "Ca", "Ti")


def apogee_wavelength_grid() -> np.ndarray:
    """Return the standard APOGEE DR17 vacuum wavelength grid in Angstrom."""
    indices = np.arange(APOGEE_PIXELS, dtype=np.float64)
    return np.power(10.0, APOGEE_WAVE_LOG10_START + APOGEE_WAVE_LOG10_STEP * indices)


def repo_root() -> Path:
    """Return the current Jorg repository root."""
    return Path(__file__).resolve().parents[3]


def workspace_root() -> Path:
    """Return the outer workspace root containing Jorg and Korg."""
    return repo_root().parent


def apogee_root() -> Path:
    """Return the top-level workspace folder reserved for the APOGEE workflow."""
    return repo_root() / "apogee"


def default_korg_root() -> Path:
    """Return the local Korg checkout used by the APOGEE bridge."""
    return workspace_root() / "Korg.jl-1.0.1"


@dataclass(frozen=True)
class SynthesisGrid:
    """Wavelength grids used by the Korg APOGEE synthesis pipeline."""

    synthesis_start: float = DEFAULT_SYNTHESIS_START_ANGSTROM
    synthesis_stop: float = DEFAULT_SYNTHESIS_STOP_ANGSTROM
    synthesis_step: float = DEFAULT_SYNTHESIS_STEP_ANGSTROM
    resolution: int = APOGEE_RESOLUTION

    @property
    def synthesis_wavelengths(self) -> np.ndarray:
        step = float(self.synthesis_step)
        return np.arange(
            float(self.synthesis_start),
            float(self.synthesis_stop) + 0.5 * step,
            step,
            dtype=np.float64,
        )

    @property
    def apogee_wavelengths(self) -> np.ndarray:
        return apogee_wavelength_grid()
