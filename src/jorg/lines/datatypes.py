from typing import NamedTuple

class LineData(NamedTuple):
    """A single spectral line"""
    wavelength: float
    species: int
    log_gf: float
    E_lower: float
    gamma_rad: float = 0.0
    gamma_stark: float = 0.0
    vdw_param1: float = 0.0
    vdw_param2: float = 0.0
