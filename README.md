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
# From GitHub (recommended for clean/tagged snapshots)
pip install "git+https://github.com/jorg-project/jorg.git@v0.3.1-clean"

# Local development
pip install -e .
pip install -e ".[gpu]"  # optional CUDA support
```

By default, legacy synthesis will try a PINN chemical-equilibrium checkpoint first.
Checkpoint lookup priority:
1. `ce_pinn_checkpoint=...` argument
2. `JORG_PINN_CKPT` environment variable
3. default model paths under `data/models/`
If none is found, synthesis automatically falls back to the JAX engine.

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

## Stellar Parameter Fitting (Experimental)
Jorg includes a direct-fitting pipeline for:
- `Teff`
- `logg`
- `[M/H]`
- `[alpha/Fe]`

```python
from jorg.fit import fit_stellar_parameters
from jorg.lines import get_VALD_solar_linelist

result = fit_stellar_parameters(
    obs_wavelengths=obs_wave,
    obs_flux=obs_flux,
    obs_error=obs_err,              # optional; auto-estimated if omitted
    linelist=get_VALD_solar_linelist(),
    initial_guess={"Teff": 5600, "logg": 4.3, "m_h": -0.2, "alpha_fe": 0.1},
    windows=[(5166.0, 5190.0), (5205.0, 5240.0)],
    R=50_000,
    optimizer="jax_surrogate",      # JAX-accelerated local surrogate optimizer
    compute_uncertainties=False,    # set True only when covariance is needed
)

print(result.summary())
```

Notes:
- Internally, `[alpha/Fe]` is converted to `alpha_H = [M/H] + [alpha/Fe]`.
- The fitter uses Korg-style parameter scaling, weak regularization, and window-level continuum adjustment.

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

## Autodiff modes
- `synthesize(..., engine="legacy")` remains the default stable path.
- `synthesize(..., engine="jax")` uses the differentiable JAX pipeline.
- `synthesize_jax(..., autodiff_strict=True)` enforces strict autodiff behavior:
  - with non-empty `linelist`, `line_backend` must be `"jax"`.
  - `line_backend="numpy"` is legacy compatibility mode and not fully autodiff-safe.
- `line_backend` default is `"jax"` for full line-opacity autodiff in JAX engine.
- `line_loggf_deltas` (JAX backend only) enables direct differentiation wrt line `log_gf`:
  - shape must match input `linelist` length and index order.
  - filtered/invalid lines are ignored in opacity accumulation, so their effective gradient is zero.

## Requirements
- Python >= 3.8
- JAX >= 0.4
- NumPy >= 1.20

## Status
- v0.3.1 (alpha)

## Chemical Equilibrium (JAX)
- `jorg.statmech.chem_eq_jax.chemical_equilibrium_jax` provides a fully JAX-native solver with implicit differentiation.
- Partition functions and logK tables are linearly interpolated on a precomputed logT grid (no SciPy in the JIT path).
- Molecule support is limited to neutral species and +1 diatomics; polyatomic logK is approximated via grid interpolation.
