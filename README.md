# Jorg v0.1

Jorg is a JAX-based stellar spectrum synthesis library for Python. This is the
v0.1.0 release, focused on Korg.jl API compatibility and fast CPU/GPU execution.

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

## API at a glance
- `synth(Teff, logg, m_H, ...)` returns `(wavelengths, flux, continuum)`
- `synthesize(atm, linelist, A_X, ...)` returns `SynthesisResult` with diagnostics
- `interpolate_atmosphere(...)` and `format_abundances(...)` are available in `jorg.synthesis`

## Requirements
- Python >= 3.8
- JAX >= 0.4
- NumPy >= 1.20

## Status
- v0.1.0 (alpha)
