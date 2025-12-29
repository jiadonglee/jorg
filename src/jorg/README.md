# Jorg Package Documentation

JAX-based stellar spectral synthesis, API-compatible with Korg.jl.

## Overview

Jorg synthesises stellar spectra using radiative transfer through model atmospheres. It achieves 90–96.5% agreement with Korg.jl across the stellar parameter space whilst providing a pure Python/JAX implementation.

**Version**: 0.1.0  
**Status**: Production-ready for research applications  
**Compatibility**: Korg.jl v0.20.0 API-compatible

## Quick Start

```python
from jorg.synthesis import synth

# Solar spectrum synthesis
wavelengths, flux, continuum = synth(
    Teff=5780,                    # Effective temperature [K]
    logg=4.44,                    # Surface gravity [log₁₀(cm/s²)]
    m_H=0.0,                      # Metallicity [M/H]
    wavelengths=(5000, 6000),     # Wavelength range [Å]
    vmic=1.0,                     # Microturbulence [km/s]
    rectify=True                  # Continuum-normalised output
)
```

## Package Structure

```
jorg/
├── synthesis.py          # Main synthesis API (synth, synthesize)
├── atmosphere.py         # MARCS atmosphere interpolation
├── abundances.py         # Elemental abundance formatting
├── constants.py          # Physical constants (CGS units)
├── continuum/            # Continuum opacity sources
│   ├── hydrogen.py       # H⁻ bound-free/free-free
│   ├── metals_bf.py      # Metal bound-free (TOPBase)
│   └── scattering.py     # Thomson and Rayleigh scattering
├── lines/                # Spectral line formation
│   ├── linelist.py       # VALD linelist parsing
│   ├── profiles.py       # Voigt profile calculation
│   ├── hydrogen_lines.py # Hydrogen line absorption
│   └── broadening.py     # Pressure broadening
├── statmech/             # Statistical mechanics
│   ├── chemical_equilibrium.py
│   ├── partition_functions.py
│   └── species.py
├── opacity/              # Opacity calculation
│   └── korg_line_processor.py  # Line windowing algorithm
├── radiative_transfer/   # RT solvers
└── utils/                # Utilities
```

## Core API

### `synth()` — High-Level Synthesis

```python
synth(Teff, logg, m_H, alpha_H=None, wavelengths=(5000, 6000),
      linelist=None, rectify=True, R=inf, vsini=0, vmic=1.0, **abundances)
```

Returns `(wavelengths, flux, continuum)` tuple.

### `synthesize()` — Full Diagnostic Output

```python
synthesize(atm, linelist, A_X, wavelengths, vmic=1.0, ...)
```

Returns `SynthesisResult` dataclass with complete diagnostics.

## Output Data Structures

### SynthesisResult

| Field | Type | Shape | Units | Description |
|-------|------|-------|-------|-------------|
| `flux` | ndarray | (N_λ,) | erg/s/cm²/cm or dimensionless | Emergent spectrum |
| `cntm` | ndarray | (N_λ,) | erg/s/cm²/cm | Continuum flux |
| `alpha` | ndarray | (N_layers, N_λ) | cm⁻¹ | Opacity matrix |
| `intensity` | ndarray | (N_μ, N_λ) | erg/s/cm²/sr/cm | Specific intensity |
| `number_densities` | dict | Species → (N_layers,) | cm⁻³ | Species densities |
| `electron_number_density` | ndarray | (N_layers,) | cm⁻³ | Electron density |
| `wavelengths` | ndarray | (N_λ,) | Å | Vacuum wavelengths |
| `mu_grid` | list | (μ, weight) pairs | — | Angular quadrature |

### Atmosphere Dictionary

| Key | Shape | Units | Description |
|-----|-------|-------|-------------|
| `temperature` | (N_layers,) | K | Temperature profile |
| `pressure` | (N_layers,) | dyn/cm² | Gas pressure |
| `electron_density` | (N_layers,) | cm⁻³ | Electron number density |
| `tau_5000` | (N_layers,) | dimensionless | Optical depth at 5000 Å |
| `height` | (N_layers,) | cm | Height coordinate |

## Installation

```bash
pip install -e .              # Basic installation
pip install -e ".[dev]"       # With development dependencies
pip install -e ".[gpu]"       # With GPU support (JAX CUDA)
```

**Requirements**: Python ≥3.8, JAX ≥0.4.0, NumPy ≥1.20.0

## Performance

| Wavelength Range | Approximate Time | Memory |
|------------------|------------------|--------|
| 100 Å | 5–15 s | ~20 MB |
| 1000 Å | 30–120 s | ~100 MB |

First call includes JAX JIT compilation overhead (~1–3 s).

## Documentation

- [SYNTHESIS_DOCUMENTATION.md](../../docs/SYNTHESIS_DOCUMENTATION.md) — Main synthesis module
- [implementation/](../../docs/implementation/) — Implementation details

## Limitations

- Plane-parallel atmospheres only (spherical geometry not yet implemented)
- LTE assumption (no NLTE corrections)
- Air wavelength conversion incomplete
