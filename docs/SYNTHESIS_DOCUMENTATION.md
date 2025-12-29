# Jorg Synthesis Module

API documentation for the core synthesis functions.

## Overview

The synthesis module (`jorg/synthesis.py`) provides two main interfaces:

1. **`synth()`** — High-level interface with stellar parameters as input
2. **`synthesize()`** — Low-level interface with pre-computed atmosphere

Both functions implement Korg.jl's synthesis pipeline:
1. Abundance vector construction
2. Atmosphere interpolation (MARCS grid)
3. Chemical equilibrium calculation (layer-by-layer)
4. Opacity computation (continuum + lines)
5. Radiative transfer solution
6. Post-processing (rectification, broadening)

---

## Functions

### `synth()`

High-level synthesis interface. Accepts stellar parameters directly and handles atmosphere interpolation internally.

```python
def synth(
    Teff: float,
    logg: float,
    m_H: float,
    alpha_H: float = None,
    wavelengths: Tuple[float, float] = (5000.0, 6000.0),
    linelist: Optional[List] = None,
    rectify: bool = True,
    R: float = float('inf'),
    vsini: float = 0,
    vmic: float = 1.0,
    hydrogen_lines: bool = True,
    mu_points: int = 20,
    rt_method: str = "korg_default",
    use_cubic_interpolation: bool = False,
    use_exact_partition_functions: bool = True,
    use_full_molecular_equilibrium: bool = True,
    format_A_X_kwargs: Optional[Dict] = None,
    synthesize_kwargs: Optional[Dict] = None,
    verbose: bool = False,
    **abundances
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Teff` | float | — | Effective temperature [K]. Valid range: 3000–50000 |
| `logg` | float | — | Surface gravity [log₁₀(cm/s²)]. Valid range: 0–6 |
| `m_H` | float | — | Metallicity [M/H] [dex]. Valid range: −4 to +1 |
| `alpha_H` | float | `m_H` | Alpha enhancement [α/H] [dex] |
| `wavelengths` | tuple | (5000, 6000) | Wavelength range (start, end) [Å] |
| `linelist` | list | None | Spectral lines. If None, continuum-only synthesis |
| `rectify` | bool | True | Normalise flux by continuum |
| `R` | float | inf | Spectral resolving power λ/Δλ for LSF convolution |
| `vsini` | float | 0 | Projected rotational velocity [km/s] |
| `vmic` | float | 1.0 | Microturbulent velocity [km/s] |
| `hydrogen_lines` | bool | True | Include hydrogen line absorption |
| `mu_points` | int | 20 | Number of angular quadrature points |
| `rt_method` | str | "korg_default" | Radiative transfer method |
| `verbose` | bool | False | Print progress information |
| `**abundances` | — | — | Individual element abundances [X/H] by symbol (e.g., `Fe=-0.5`) |

#### Returns

| Index | Type | Shape | Units | Description |
|-------|------|-------|-------|-------------|
| 0 | ndarray | (N_λ,) | Å | Wavelength array |
| 1 | ndarray | (N_λ,) | dimensionless or erg/s/cm²/cm | Flux (rectified if `rectify=True`) |
| 2 | ndarray | (N_λ,) | erg/s/cm²/cm | Continuum flux |

#### Examples

```python
from jorg.synthesis import synth
from jorg.lines.linelist_data import get_VALD_solar_linelist

# Solar spectrum with built-in linelist
linelist = get_VALD_solar_linelist()
wl, flux, cont = synth(5780, 4.44, 0.0, linelist=linelist)

# Metal-poor alpha-enhanced star
wl, flux, cont = synth(
    Teff=6000, logg=4.0, m_H=-1.5, alpha_H=-1.0,
    wavelengths=(4000, 7000), linelist=linelist
)

# Individual element abundances
wl, flux, cont = synth(
    Teff=5500, logg=4.2, m_H=-0.5,
    Fe=-0.8, C=0.2, O=-0.3,  # [X/H] format
    linelist=linelist
)

# Physical units (not rectified)
wl, flux, cont = synth(5780, 4.44, 0.0, rectify=False, linelist=None)
# flux ~ 10¹⁵ erg/s/cm²/cm
```

---

### `synthesize()`

Low-level synthesis interface. Requires pre-computed atmosphere and abundance array.

```python
def synthesize(
    atm: Dict,
    linelist: Optional[List] = None,
    A_X: Optional[np.ndarray] = None,
    wavelengths: Union[Tuple[float, float], np.ndarray] = (4000.0, 7000.0),
    vmic: float = 1.0,
    line_buffer: float = 10.0,
    cntm_step: float = 1.0,
    air_wavelengths: bool = False,
    hydrogen_lines: bool = True,
    use_MHD_for_hydrogen_lines: bool = True,
    hydrogen_line_window_size: float = 150.0,
    mu_values: Union[int, List[float]] = 20,
    line_cutoff_threshold: float = 3e-4,
    return_cntm: bool = True,
    I_scheme: str = "linear_flux_only",
    tau_scheme: str = "anchored",
    rt_method: str = "korg_default",
    rectify: bool = False,
    verbose: bool = False,
    debug_mode: bool = False,
    export_intermediate_results: bool = False,
    **kwargs
) -> SynthesisResult
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `atm` | dict | — | Atmosphere structure (see below) |
| `linelist` | list | None | Spectral lines |
| `A_X` | ndarray | None | 92-element abundance array. A_X[0] must equal 12 |
| `wavelengths` | tuple/ndarray | (4000, 7000) | Wavelength range or explicit grid [Å] |
| `vmic` | float | 1.0 | Microturbulence [km/s] |
| `line_buffer` | float | 10.0 | Buffer for line inclusion [Å] |
| `cntm_step` | float | 1.0 | Continuum grid spacing [Å] |
| `air_wavelengths` | bool | False | Input wavelengths in air (not vacuum) |
| `hydrogen_lines` | bool | True | Include hydrogen lines |
| `use_MHD_for_hydrogen_lines` | bool | True | Use MHD occupation probability |
| `hydrogen_line_window_size` | float | 150.0 | Window size for H lines [Å] |
| `mu_values` | int/list | 20 | Angular quadrature points or explicit values |
| `line_cutoff_threshold` | float | 3e-4 | Line profile truncation threshold |
| `return_cntm` | bool | True | Include continuum in output |
| `I_scheme` | str | "linear_flux_only" | Intensity calculation scheme |
| `tau_scheme` | str | "anchored" | Optical depth calculation scheme |
| `rectify` | bool | False | Normalise flux by continuum |
| `verbose` | bool | False | Print progress information |
| `debug_mode` | bool | False | Enable diagnostic output |

#### Returns

`SynthesisResult` dataclass with fields:

| Field | Type | Shape | Units | Description |
|-------|------|-------|-------|-------------|
| `flux` | ndarray | (N_λ,) | erg/s/cm²/cm | Emergent flux |
| `cntm` | ndarray | (N_λ,) | erg/s/cm²/cm | Continuum flux (if `return_cntm=True`) |
| `intensity` | ndarray | (N_μ, N_λ) | erg/s/cm²/sr/cm | Specific intensity |
| `alpha` | ndarray | (N_layers, N_λ) | cm⁻¹ | Absorption coefficient matrix |
| `mu_grid` | list | — | — | List of (μ, weight) tuples |
| `number_densities` | dict | Species → (N_layers,) | cm⁻³ | Species number densities |
| `electron_number_density` | ndarray | (N_layers,) | cm⁻³ | Electron density profile |
| `wavelengths` | ndarray | (N_λ,) | Å | Vacuum wavelengths |
| `subspectra` | list | — | — | Wavelength window slices |
| `debug_data` | dict | — | — | Diagnostic data (if `debug_mode=True`) |
| `intermediate_results` | dict | — | — | Intermediate values (if `export_intermediate_results=True`) |

#### Expected Atmosphere Structure

The `atm` dictionary must contain:

| Key | Type | Shape | Units | Required |
|-----|------|-------|-------|----------|
| `temperature` | ndarray | (N_layers,) | K | Yes |
| `pressure` | ndarray | (N_layers,) | dyn/cm² | Yes |
| `electron_density` | ndarray | (N_layers,) | cm⁻³ | Yes |
| `tau_5000` | ndarray | (N_layers,) | dimensionless | Yes |
| `height` | ndarray | (N_layers,) | cm | No (estimated if absent) |

Alternatively, pass a `ModelAtmosphere` object with `.layers` attribute.

#### Examples

```python
from jorg.synthesis import synthesize
from jorg.atmosphere import interpolate_marcs
from jorg.abundances import format_abundances
import numpy as np

# Create atmosphere and abundances
A_X = format_abundances(default_metals_H=-0.5, default_alpha_H=-0.3)
atm = interpolate_marcs(5500, 4.0, A_X)

# Define wavelength grid
wavelengths = np.linspace(5000, 5100, 1000)

# Full synthesis with diagnostics
result = synthesize(
    atm=atm,
    linelist=my_linelist,
    A_X=A_X,
    wavelengths=wavelengths,
    vmic=1.5,
    verbose=True
)

# Access results
print(f"Flux range: {result.flux.min():.3e} - {result.flux.max():.3e}")
print(f"Opacity matrix shape: {result.alpha.shape}")
print(f"Species tracked: {len(result.number_densities)}")
```

---

## Supporting Functions

### `format_abundances()`

Construct 92-element abundance array from stellar parameters.

```python
def format_abundances(
    default_metals_H: float = 0.0,
    default_alpha_H: float = None,
    abundances: Dict = None,
    solar_relative: bool = True,
    solar_abundances: np.ndarray = None,
    alpha_elements: List[int] = None
) -> np.ndarray
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `default_metals_H` | float | 0.0 | Metallicity [M/H] |
| `default_alpha_H` | float | `default_metals_H` | Alpha enhancement [α/H] |
| `abundances` | dict | None | Individual element overrides. Keys: atomic number or symbol |
| `solar_relative` | bool | True | Interpret values as [X/H] (True) or A(X) (False) |
| `solar_abundances` | ndarray | Asplund 2020 | Reference solar abundances |
| `alpha_elements` | list | [8,10,12,14,16,18,20,22] | Atomic numbers of alpha elements |

#### Returns

92-element `ndarray` with A(X) = log₁₀(N_X/N_H) + 12 for each element. A_X[0] = 12.0 (hydrogen).

#### Alpha Elements

Default alpha elements (atomic numbers): O (8), Ne (10), Mg (12), Si (14), S (16), Ar (18), Ca (20), Ti (22).

---

### `interpolate_marcs()` / `interpolate_atmosphere()`

Interpolate MARCS model atmosphere grid.

```python
from jorg.atmosphere import interpolate_marcs

atm = interpolate_marcs(Teff, logg, A_X)
```

Returns `ModelAtmosphere` object or dictionary with atmosphere structure.

---

## Physics Implementation

### Synthesis Pipeline

1. **Wavelength grid**: Default spacing 0.01 Å (10 mÅ)
2. **Chemical equilibrium**: Newton solver with Saha–Boltzmann ionisation, molecular equilibrium
3. **Continuum opacity**: H⁻ bf/ff (McLaughlin 2017), H I bf, metal bf (TOPBase), Thomson/Rayleigh scattering
4. **Line opacity**: KorgLineProcessor with windowing algorithm (cutoff threshold 3×10⁻⁴)
5. **Source function**: LTE Planck function B_λ(T)
6. **Radiative transfer**: Anchored optical depth, linear intensity scheme, Gauss–Legendre μ quadrature

### Radiative Transfer Methods

| Method | Description |
|--------|-------------|
| `korg_default` | Standard Korg.jl scheme (anchored τ, linear S) |
| `feautrier` | Feautrier method (2nd-order accurate) |
| `short_char` | Short characteristics |

### Units

All internal calculations use CGS units:
- Wavelength: cm (converted from Å at boundaries)
- Flux: erg s⁻¹ cm⁻² cm⁻¹
- Opacity: cm⁻¹
- Number density: cm⁻³
- Temperature: K
- Pressure: dyn cm⁻²

---

## Comparison with Korg.jl

| Feature | Korg.jl | Jorg | Notes |
|---------|---------|------|-------|
| API | `synth()`, `synthesize()` | Identical | Same signatures and defaults |
| Atmosphere | MARCS interpolation | MARCS interpolation | Grid-based |
| Continuum | Full physics | Full physics | H⁻, H I, metals, scattering |
| Lines | VALD parsing | VALD parsing | 99.9% line count agreement |
| RT | Multiple schemes | Multiple schemes | Default: anchored + linear |
| Performance | Julia native | JAX JIT | Similar after compilation |

Agreement with Korg.jl: 90–96.5% across stellar parameter space (Teff: 4000–7000 K, logg: 2–5, [M/H]: −2 to +0.5).
