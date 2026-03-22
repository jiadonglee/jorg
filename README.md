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

By default, synthesis now uses the JAX engine with strict autodiff settings.
Legacy PINN-based CE remains available only when you explicitly select:
- `engine="legacy"`
- `ce_solver="pinn"`
- optional `ce_pinn_checkpoint=...` / `JORG_PINN_CKPT`

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
from jorg.fit import fit_stellar_parameters_autodiff
from jorg.lines import get_VALD_solar_linelist

result = fit_stellar_parameters_autodiff(
    obs_wavelengths=obs_wave,
    obs_flux=obs_flux,
    obs_error=obs_err,
    linelist=get_VALD_solar_linelist(),
    initial_guess={"Teff": 5600.0, "logg": 4.3, "m_h": -0.2, "alpha_fe": 0.1},
    bounds={
        "Teff": (4800.0, 6500.0),
        "logg": (3.5, 5.0),
        "m_h": (-1.5, 0.5),
        "alpha_fe": (-0.2, 0.6),
    },
    windows=[(5166.0, 5190.0), (5205.0, 5240.0)],
    R=50_000,
    execution_profile="interactive",
    compute_uncertainties=False,
)

print(result.summary())
```

Notes:
- Internally, `[alpha/Fe]` is converted to `alpha_H = [M/H] + [alpha/Fe]`.
- The fitter uses Korg-style parameter scaling, weak regularization, and window-level continuum adjustment.
- `fit_stellar_parameters_autodiff(...)` fixes `optimizer="jax_value_and_grad"` and
  injects the strict JAX defaults required by the full autodiff path.
- `execution_profile="interactive"` is now the default:
  - `"debug"` keeps atmosphere/objective/solver JIT off for tracing and diagnostics.
  - `"interactive"` requests atmosphere/objective JIT with solver JIT off.
  - `"batch"` requests atmosphere/objective/solver JIT for repeated same-shape fits.
  - In the current physical `jax_value_and_grad` path, objective/solver JIT are still
    disabled at runtime because nested tracing through atmosphere/CE caches is not yet
    tracer-safe; atmosphere interpolation JIT still applies.
- In the autodiff fit path, JAX line synthesis now prefilters the linelist to the active
  fitting wavelength span plus `line_buffer`, so small windows no longer scan the full line table.
- If you need the lower-level API, call `fit_stellar_parameters(..., optimizer="jax_value_and_grad")`
  and ensure `model_context["synth_kwargs"]["synthesize_kwargs"]` includes:
  `line_backend="jax"`, `autodiff_strict=True`, and `ce_jit=True`.

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
- Default path: `synthesize(...)`, `synthesize_spectrum(...)`, and `synth(...)` run with
  `engine="jax"` and `ce_solver="jax"`.
- `synthesize(..., engine="legacy")` is still supported, but must be explicit.
- `synthesize_jax(..., autodiff_strict=True)` is now the default strict mode:
  - with non-empty `linelist`, `line_backend` must be `"jax"`;
  - `ce_jit` must be `True` (strict mode rejects `ce_jit=False`);
  - `ce_warm_start` defaults to `False` to avoid layer-to-layer solver drift and
    keep Jorg/Korg agreement stable (you can set `ce_warm_start=True` for speed experiments);
  - `line_backend="numpy"` is only allowed when `autodiff_strict=False`.
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
