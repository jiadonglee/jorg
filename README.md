# Jorg

Jorg (`jorg`) is a JAX-based stellar spectrum synthesis library for Python.

## Highlights
- Korg.jl-compatible `synth` and `synthesize` APIs
- JAX JIT acceleration with optional GPU support
- MARCS atmosphere interpolation and exact radiative transfer
- Line and continuum opacity with built-in VALD solar linelist helpers
- Diagnostic outputs via `SynthesisResult`

## Install
```bash
pip install -e .
pip install -e ".[gpu]"  # optional CUDA support
```

## Quick start
```python
from jorg.synthesis import synth
from jorg.lines import get_VALD_solar_linelist

wl, flux, cntm = synth(
    5780, 4.44, 0.0,
    wavelengths=(5000, 6000),
    linelist=get_VALD_solar_linelist(),
)
```

## Data files
Jorg relies on external data (linelists, partition functions, opacity tables, MARCS grids).
Small tables live under `data/` in the repo; large MARCS grids are not checked in.

Set these environment variables to point at your local data bundle:
```bash
export JORG_DATA_DIR=/path/to/jorg/data
export JORG_MARCS_GRID_DIR=/path/to/marcs_grids  # optional if separate
```

When running from a cloned repo, Jorg will automatically use the local `data/` directory
if `JORG_DATA_DIR` is not set.

## API at a glance
- `synth(Teff, logg, m_H, ...)` returns `(wavelengths, flux, continuum)`
- `synthesize(atm, linelist, A_X, ...)` returns `SynthesisResult` with diagnostics
- `interpolate_atmosphere(...)` and `format_abundances(...)` are available in `jorg.synthesis`

## Requirements
- Python >= 3.8
- JAX >= 0.4
- NumPy >= 1.20

## Status
- v0.3.0 (alpha)

## Chemical Equilibrium (JAX)
- `jorg.statmech.chem_eq_jax.chemical_equilibrium_jax` provides a fully JAX-native solver with implicit differentiation.
- Partition functions and logK tables are linearly interpolated on a precomputed logT grid (no SciPy in the JIT path).
- Molecule support is limited to neutral species and +1 diatomics; polyatomic logK is approximated via grid interpolation.
