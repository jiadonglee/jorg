"""
Korg-compatible ionization energies.

This module mirrors Korg.jl's `setup_ionization_energies` behavior:
- parse `BarklemCollet2016-ionization_energies.dat`
- keep values exactly as provided (including `-1` sentinels)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from ..data import get_data_path

MAX_ATOMIC_NUMBER = 92


class ProperIonizationEnergies:
    """
    Korg-compatible ionization energy loader.

    Values are stored as `{Z: (chi1, chi2, chi3)}` in eV.
    """

    def __init__(self, data_file: str | Path | None = None):
        self._data_file = Path(data_file) if data_file is not None else get_data_path(
            "barklem_collet_2016", "BarklemCollet2016-ionization_energies.dat"
        )
        self.ionization_energies: Dict[int, Tuple[float, float, float]] = self._load()

    def _load(self) -> Dict[int, Tuple[float, float, float]]:
        out: Dict[int, Tuple[float, float, float]] = {}
        with open(self._data_file, "r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                toks = line.split()
                if len(toks) < 5:
                    continue
                Z = int(toks[0])
                chi1, chi2, chi3 = map(float, toks[2:5])
                out[Z] = (chi1, chi2, chi3)

        missing = [Z for Z in range(1, MAX_ATOMIC_NUMBER + 1) if Z not in out]
        if missing:
            raise ValueError(f"Missing ionization energies for Z={missing}")
        return out

    def get_ionization_energy(self, element: int, ionization_stage: int = 1) -> float:
        if not (1 <= element <= MAX_ATOMIC_NUMBER):
            raise ValueError(f"Invalid atomic number: {element}")
        if ionization_stage < 1:
            raise ValueError(f"Invalid ionization stage: {ionization_stage}")
        vals = self.ionization_energies[element]
        if ionization_stage <= 3:
            return vals[ionization_stage - 1]
        # Korg data table only provides first 3 stages.
        return -1.0

    def get_all_ionization_energies(self) -> Dict[int, Tuple[float, float, float]]:
        return dict(self.ionization_energies)

    def validate_against_hardcoded(self) -> Dict[int, Dict[str, float]]:
        """Utility retained for compatibility with existing callers."""
        out: Dict[int, Dict[str, float]] = {}
        for Z in (1, 2, 6, 8, 22, 26, 28):
            chi1 = self.get_ionization_energy(Z, 1)
            out[Z] = {"korg_chi1_eV": chi1, "legacy_13p6Z2": 13.6 * Z * Z}
        return out


_proper_ionization_energies: ProperIonizationEnergies | None = None


def get_proper_ionization_energies() -> ProperIonizationEnergies:
    global _proper_ionization_energies
    if _proper_ionization_energies is None:
        _proper_ionization_energies = ProperIonizationEnergies()
    return _proper_ionization_energies


def proper_ionization_energy(element: int, ionization_stage: int = 1) -> float:
    return get_proper_ionization_energies().get_ionization_energy(element, ionization_stage)


def create_proper_ionization_energy_dict() -> Dict[int, Tuple[float, float, float]]:
    return get_proper_ionization_energies().get_all_ionization_energies()


def validate_ionization_energy_improvements():
    return get_proper_ionization_energies().validate_against_hardcoded()


__all__ = [
    "ProperIonizationEnergies",
    "get_proper_ionization_energies",
    "proper_ionization_energy",
    "create_proper_ionization_energy_dict",
    "validate_ionization_energy_improvements",
]
