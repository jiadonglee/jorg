# Synthesis API Reference

Complete API reference for the Jorg synthesis module.

## Module: `jorg.synthesis`

### Functions

---

#### `synth()`

```python
synth(
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
    format_A_X_kwargs: Optional[Dict] = None,
    synthesize_kwargs: Optional[Dict] = None,
    verbose: bool = False,
    **abundances
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

High-level stellar spectrum synthesis. Matches Korg.jl's `synth()` API.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `Teff` | float | *required* | Effective temperature [K] |
| `logg` | float | *required* | Surface gravity [log₁₀(cm/s²)] |
| `m_H` | float | *required* | Metallicity [M/H] [dex] |
| `alpha_H` | float | `m_H` | Alpha enhancement [α/H] [dex] |
| `wavelengths` | tuple | (5000, 6000) | Wavelength range (start, end) [Å] |
| `linelist` | list | None | Spectral line list |
| `rectify` | bool | True | Normalise flux by continuum |
| `R` | float | inf | Resolving power for LSF convolution |
| `vsini` | float | 0 | Rotational velocity [km/s] |
| `vmic` | float | 1.0 | Microturbulence [km/s] |
| `hydrogen_lines` | bool | True | Include H line absorption |
| `mu_points` | int | 20 | Angular quadrature points |
| `rt_method` | str | "korg_default" | RT method: "korg_default", "feautrier", "short_char" |
| `use_cubic_interpolation` | bool | False | Use cubic atmosphere interpolation |
| `format_A_X_kwargs` | dict | None | Options for `format_abundances()` |
| `synthesize_kwargs` | dict | None | Options for `synthesize()` |
| `verbose` | bool | False | Print progress |
| `**abundances` | — | — | Element abundances by symbol (e.g., `Fe=-0.5`) |

**Returns**

| Index | Type | Shape | Units |
|-------|------|-------|-------|
| 0 | ndarray | (N_λ,) | Å |
| 1 | ndarray | (N_λ,) | dimensionless (rectified) or erg/s/cm²/cm |
| 2 | ndarray | (N_λ,) | erg/s/cm²/cm |

---

#### `synthesize()`

```python
synthesize(
    atm: Union[Dict, ModelAtmosphere],
    linelist: Optional[List] = None,
    A_X: Optional[np.ndarray] = None,
    wavelengths: Union[Tuple[float, float], np.ndarray] = (4000.0, 7000.0),
    verbose: bool = True,
    **kwargs
) -> SynthesisResult
```

Full synthesis with diagnostic output. Wrapper for `synthesize_korg_compatible()`.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `atm` | dict/ModelAtmosphere | *required* | Atmosphere structure |
| `linelist` | list | None | Spectral lines |
| `A_X` | ndarray | None | 92-element abundance array |
| `wavelengths` | tuple/ndarray | (4000, 7000) | Wavelength range or grid [Å] |
| `verbose` | bool | True | Print progress |
| `**kwargs` | — | — | Additional options (see below) |

**Additional kwargs**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `vmic` | float | 1.0 | Microturbulence [km/s] |
| `line_buffer` | float | 10.0 | Line inclusion buffer [Å] |
| `cntm_step` | float | 1.0 | Continuum grid spacing [Å] |
| `air_wavelengths` | bool | False | Input in air wavelengths |
| `hydrogen_lines` | bool | True | Include H lines |
| `use_MHD_for_hydrogen_lines` | bool | True | Use MHD occupation probability |
| `hydrogen_line_window_size` | float | 150.0 | H line window [Å] |
| `mu_values` | int/list | 20 | Angular quadrature |
| `line_cutoff_threshold` | float | 3e-4 | Profile truncation threshold |
| `return_cntm` | bool | True | Return continuum |
| `I_scheme` | str | "linear_flux_only" | Intensity scheme |
| `tau_scheme` | str | "anchored" | Optical depth scheme |
| `rt_method` | str | "korg_default" | RT method |
| `rectify` | bool | False | Normalise by continuum |
| `debug_mode` | bool | False | Enable diagnostics |
| `export_intermediate_results` | bool | False | Export intermediates |
| `logg` | float | 4.44 | Surface gravity (for RT) |

**Returns**

`SynthesisResult` dataclass.

---

### Classes

#### `SynthesisResult`

```python
@dataclass
class SynthesisResult:
    flux: np.ndarray
    cntm: Optional[np.ndarray]
    intensity: np.ndarray
    alpha: np.ndarray
    mu_grid: List[Tuple[float, float]]
    number_densities: Dict[Species, np.ndarray]
    electron_number_density: np.ndarray
    wavelengths: np.ndarray
    subspectra: List[slice]
    debug_data: Optional[Dict] = None
    intermediate_results: Optional[Dict] = None
```

**Attributes**

| Name | Type | Shape | Units | Description |
|------|------|-------|-------|-------------|
| `flux` | ndarray | (N_λ,) | erg/s/cm²/cm | Emergent spectrum |
| `cntm` | ndarray | (N_λ,) | erg/s/cm²/cm | Continuum flux |
| `intensity` | ndarray | (N_μ, N_λ) | erg/s/cm²/sr/cm | Specific intensity |
| `alpha` | ndarray | (N_layers, N_λ) | cm⁻¹ | Opacity matrix |
| `mu_grid` | list | — | — | (μ, weight) tuples |
| `number_densities` | dict | Species → (N_layers,) | cm⁻³ | Species densities |
| `electron_number_density` | ndarray | (N_layers,) | cm⁻³ | Electron density |
| `wavelengths` | ndarray | (N_λ,) | Å | Vacuum wavelengths |
| `subspectra` | list | — | — | Window slices |
| `debug_data` | dict | — | — | Diagnostic data |
| `intermediate_results` | dict | — | — | Intermediate values |

---

## Module: `jorg.abundances`

#### `format_abundances()`

```python
format_abundances(
    default_metals_H: float = 0.0,
    default_alpha_H: float = None,
    abundances: Dict = None,
    solar_relative: bool = True,
    solar_abundances: np.ndarray = None,
    alpha_elements: List[int] = None
) -> np.ndarray
```

Format 92-element abundance array.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `default_metals_H` | float | 0.0 | Metallicity [M/H] |
| `default_alpha_H` | float | `default_metals_H` | Alpha enhancement |
| `abundances` | dict | None | Individual overrides (atomic number or symbol) |
| `solar_relative` | bool | True | Values are [X/H] (True) or A(X) (False) |
| `solar_abundances` | ndarray | Asplund 2020 | Reference abundances |
| `alpha_elements` | list | [8,10,12,14,16,18,20,22] | Alpha element Z values |

**Returns**

92-element ndarray with A(X) = log₁₀(N_X/N_H) + 12.

---

## Module: `jorg.atmosphere`

#### `interpolate_marcs()`

```python
interpolate_marcs(
    Teff: float,
    logg: float,
    A_X: np.ndarray
) -> ModelAtmosphere
```

Interpolate MARCS atmosphere grid.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `Teff` | float | Effective temperature [K] |
| `logg` | float | Surface gravity [log₁₀(cm/s²)] |
| `A_X` | ndarray | 92-element abundance array |

**Returns**

`ModelAtmosphere` object with `.layers` attribute.

---

#### `ModelAtmosphere`

```python
@dataclass
class ModelAtmosphere:
    layers: List[AtmosphereLayer]
    spherical: bool = False
    R: Optional[float] = None
```

#### `AtmosphereLayer`

```python
@dataclass
class AtmosphereLayer:
    tau_5000: float          # Optical depth at 5000 Å
    z: float                 # Height [cm]
    temp: float              # Temperature [K]
    electron_number_density: float  # [cm⁻³]
    number_density: float    # Total number density [cm⁻³]
```

---

## Module: `jorg.constants`

Physical constants in CGS units.

| Name | Value | Units | Description |
|------|-------|-------|-------------|
| `kboltz_cgs` | 1.380649e-16 | erg/K | Boltzmann constant |
| `hplanck_cgs` | 6.62607015e-27 | erg·s | Planck constant |
| `c_cgs` | 2.99792458e10 | cm/s | Speed of light |
| `electron_mass_cgs` | 9.1093897e-28 | g | Electron mass |
| `electron_charge_cgs` | 4.80320425e-10 | statcoulomb | Electron charge |
| `amu_cgs` | 1.6605402e-24 | g | Atomic mass unit |
| `bohr_radius_cgs` | 5.29177210903e-9 | cm | Bohr radius |
| `solar_mass_cgs` | 1.9884e33 | g | Solar mass |
| `G_cgs` | 6.67430e-8 | cm³/g/s² | Gravitational constant |
| `eV_to_cgs` | 1.602e-12 | erg/eV | Energy conversion |
| `kboltz_eV` | 8.617333262145e-5 | eV/K | Boltzmann in eV |
| `RydbergH_eV` | 13.598287264 | eV | Hydrogen Rydberg |

---

## Module: `jorg.lines.linelist_data`

#### `get_VALD_solar_linelist()`

```python
get_VALD_solar_linelist() -> List
```

Load built-in solar linelist (VALD format).

**Returns**

List of spectral lines (~19,000 lines).

---

## Module: `jorg.lines.linelist`

#### `read_linelist()`

```python
read_linelist(
    filename: str,
    format: str = "auto"
) -> List
```

Read linelist from file.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `filename` | str | *required* | Path to linelist file |
| `format` | str | "auto" | Format: "auto", "vald", "kurucz" |

**Returns**

List of spectral lines.
