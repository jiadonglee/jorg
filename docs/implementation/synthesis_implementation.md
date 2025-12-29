# Synthesis Implementation Guide

Technical documentation of Jorg's synthesis module internals.

## Architecture

```
synthesis.py
├── synth()                         # High-level API
├── synthesize()                    # Wrapper for synthesize_korg_compatible
├── synthesize_korg_compatible()    # Core implementation
│   ├── Wavelength grid setup
│   ├── Abundance validation
│   ├── LayerProcessor initialisation
│   ├── KorgLineProcessor initialisation
│   ├── Layer-by-layer processing
│   │   ├── Chemical equilibrium
│   │   ├── Continuum opacity
│   │   └── Line opacity (via KorgLineProcessor)
│   ├── Radiative transfer
│   └── Post-processing
└── Helper functions
    ├── _setup_mu_grid()
    ├── _calculate_line_opacity_multilayer()
    └── _calculate_radiative_transfer()
```

## Core Pipeline

### 1. Wavelength Grid Setup

```python
# Default: 10 mÅ spacing (matches Korg.jl)
spacing = 0.01  # Å
n_points = int((λ_stop - λ_start) / spacing) + 1
wl_array = np.linspace(λ_start, λ_stop, n_points)
```

The wavelength grid is internally stored in Ångströms but converted to cm for opacity calculations.

### 2. Abundance Array Validation

The 92-element abundance array must satisfy:
- `len(A_X) == 92`
- `A_X[0] == 12` (hydrogen reference)

Abundances are converted to absolute fractions:
```python
abs_abundances = 10**(A_X - 12)
abs_abundances = abs_abundances / np.sum(abs_abundances)
```

### 3. Chemical Equilibrium

Chemical equilibrium is calculated layer-by-layer using a Newton solver (`korg_chemical_equilibrium.py`). For each atmospheric layer:

1. Input: T [K], P [dyn/cm²], absolute abundances
2. Solve Saha equation for ionisation fractions
3. Solve molecular equilibrium for molecule formation
4. Output: electron density [cm⁻³], species number densities [cm⁻³]

Key species tracked: H I, H II, H⁻, He I, He II, H₂, and all metals with their ionisation states.

### 4. Opacity Calculation

Opacity is computed in two stages:

#### Stage 1: Continuum Opacity

```python
alpha_continuum = layer_processor.process_all_layers(
    atm, abs_abundances, wl_array,
    linelist=None,      # No lines
    hydrogen_lines=False
)
```

Continuum sources:
- **H⁻ bound-free**: McLaughlin (2017) cross-sections
- **H⁻ free-free**: Bell & Berrington (1987)
- **H I bound-free**: Hydrogenic approximation
- **Metal bound-free**: TOPBase photoionisation cross-sections
- **Thomson scattering**: σ_T = 6.65×10⁻²⁵ cm²
- **Rayleigh scattering**: H I and He I with λ⁻⁴ dependence

#### Stage 2: Line Opacity

```python
line_opacity = _calculate_line_opacity_multilayer(
    wl_array, temps, electron_densities, number_densities,
    partition_funcs, linelist, line_buffer, vmic,
    continuum_opacity=alpha_continuum,
    cutoff_threshold=3e-4
)
alpha_matrix = alpha_continuum + line_opacity
```

The `KorgLineProcessor` implements Korg.jl's windowing algorithm:
1. For each line, calculate the window where line/continuum > cutoff threshold
2. Only compute Voigt profile within this window
3. This reduces computational cost from O(N_lines × N_wavelengths) to O(N_lines × window_width)

### 5. Source Function

LTE source function equals the Planck function:

```python
# B_ν = (2hν³/c²) / (exp(hν/kT) - 1)
frequencies = c_cgs / (wavelengths * 1e-8)
h_nu_over_kt = hplanck_cgs * frequencies / (kboltz_cgs * temperatures)
B_nu = 2 * hplanck_cgs * frequencies**3 / c_cgs**2 / (np.exp(h_nu_over_kt) - 1)

# Convert to per-wavelength: B_λ = B_ν × c/λ²
source_function = B_nu * c_cgs / (wavelengths * 1e-8)**2
```

Units: erg s⁻¹ cm⁻² sr⁻¹ cm⁻¹

### 6. Radiative Transfer

The RT equation is solved using the anchored optical depth scheme:

```python
flux, intensity, mu_grid, weights = radiative_transfer(
    alpha=alpha_matrix,
    source=source_function,
    spatial_coord=atm['height'],
    mu_points=20,
    spherical=False,
    tau_scheme="anchored",
    I_scheme="linear_flux_only",
    alpha_ref=alpha5_reference,
    tau_ref=tau_5000
)
```

#### Anchored Optical Depth

```
τ_λ = τ_ref × (α_λ / α_ref)
```

This preserves the atmosphere's optical depth scale whilst allowing wavelength-dependent opacity.

#### Angular Quadrature

Gauss–Legendre quadrature with 20 points (default). The emergent flux is:

```
F_λ = 2π ∫₀¹ I_λ(μ) μ dμ ≈ 2π Σᵢ wᵢ μᵢ I_λ(μᵢ)
```

### 7. Continuum Calculation

Continuum flux is calculated separately using the continuum-only opacity:

```python
continuum_flux, _, _, _ = radiative_transfer(
    alpha=alpha_continuum,  # Stage 1 opacity only
    source=source_function,
    ...
)
```

### 8. Rectification

If `rectify=True`:
```python
flux = flux / np.maximum(continuum, 1e-10)
flux = np.clip(flux, 0.0, 2.0)  # Allow mild emission, prevent negative
continuum = np.ones_like(continuum)  # Normalised
```

## Data Flow

```
Input: Teff, logg, m_H, wavelengths, linelist
         │
         ▼
┌─────────────────────┐
│ format_abundances() │ → A_X[92]
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ interpolate_marcs() │ → atm{T, P, ne, τ, z}
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ LayerProcessor      │
│ ├── chemical_eq()   │ → number_densities, ne
│ └── continuum_abs() │ → alpha_continuum[layers, λ]
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ KorgLineProcessor   │ → line_opacity[layers, λ]
└─────────────────────┘
         │
         ▼
alpha_matrix = alpha_continuum + line_opacity
         │
         ▼
┌─────────────────────┐
│ radiative_transfer()│ → flux, continuum, intensity
└─────────────────────┘
         │
         ▼
Output: SynthesisResult
```

## Key Classes

### SynthesisResult

```python
@dataclass
class SynthesisResult:
    flux: np.ndarray                    # (N_λ,)
    cntm: Optional[np.ndarray]          # (N_λ,)
    intensity: np.ndarray               # (N_μ, N_λ)
    alpha: np.ndarray                   # (N_layers, N_λ)
    mu_grid: List[Tuple[float, float]]  # [(μ, weight), ...]
    number_densities: Dict[Species, np.ndarray]
    electron_number_density: np.ndarray # (N_layers,)
    wavelengths: np.ndarray             # (N_λ,)
    subspectra: List[slice]
    debug_data: Optional[Dict]
    intermediate_results: Optional[Dict]
```

### LayerProcessor

Handles layer-by-layer opacity computation:
- Chemical equilibrium solver integration
- Continuum opacity sources
- Species density tracking

### KorgLineProcessor

Implements Korg.jl's line windowing algorithm:
- Line profile calculation with Voigt functions
- Cutoff threshold for profile truncation
- Multi-layer batch processing

## Constants

Physical constants are defined in `constants.py` (CGS units):

| Constant | Value | Units |
|----------|-------|-------|
| `kboltz_cgs` | 1.380649×10⁻¹⁶ | erg/K |
| `hplanck_cgs` | 6.62607015×10⁻²⁷ | erg·s |
| `c_cgs` | 2.99792458×10¹⁰ | cm/s |
| `electron_mass_cgs` | 9.1093897×10⁻²⁸ | g |
| `electron_charge_cgs` | 4.80320425×10⁻¹⁰ | statcoulomb |

## Performance Notes

- First call includes JAX JIT compilation (~1–3 s overhead)
- Memory scales as O(N_layers × N_wavelengths)
- Typical values: 56–72 layers, 10³–10⁵ wavelength points
- Line windowing reduces line computation by ~100× for dense linelists

## Error Handling

The synthesis validates:
- Abundance array length (must be 92)
- Hydrogen abundance reference (A_X[0] must equal 12)
- Atmosphere structure completeness
- Numerical stability (clips extreme values, prevents division by zero)
