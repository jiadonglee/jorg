# Jorg Lines Module Documentation

## Overview

The `lines` module (`/Jorg/src/jorg/lines/`) provides comprehensive atomic and molecular line absorption calculations for stellar spectral synthesis. It implements sophisticated line profile calculations, broadening mechanisms, and opacity calculations following Korg.jl's proven methodologies.

## Architecture

### Core Design Philosophy

1. **Comprehensive Line Physics**: Full implementation of Voigt profiles, broadening mechanisms, and opacity calculations
2. **Hydrogen Line Specialization**: Sophisticated treatment using MHD formalism and Stark broadening
3. **Molecular Line Support**: Efficient cross-section interpolation and line-by-line calculation
4. **JAX Optimization**: High-performance computation with JIT compilation
5. **Korg.jl Compatibility**: API and physics exactly matching Korg.jl implementation

### Module Structure

```
lines/
├── __init__.py           # Module exports and API
├── core.py              # Main line absorption functions
├── profiles.py          # Line profile calculations (Voigt, Harris)
├── broadening.py        # Broadening mechanisms (Doppler, vdW, Stark)
├── hydrogen_lines.py    # Advanced hydrogen line treatment
├── hydrogen_lines_simple.py  # Simplified hydrogen implementation
├── linelist.py          # Line list management and I/O
├── opacity.py           # Line opacity calculations
├── molecular_*.py       # Molecular line support
├── datatypes.py         # Data structures (LineData, etc.)
├── species.py           # Species identification and handling
└── utils.py            # Utility functions
```

---

## Main API Functions

### 1. `total_line_absorption()` - Master Line Opacity Function

**Purpose**: Calculate total line absorption coefficient from a complete linelist.

#### Signature
```python
def total_line_absorption(wavelengths: jnp.ndarray,
                         linelist: Union[List[LineData], LineList],
                         temperature: float,
                         log_g: float,
                         abundances: Optional[Dict[int, float]] = None,
                         electron_density: float = 1e14,
                         hydrogen_density: float = 1e16,
                         microturbulence: float = 0.0,
                         **kwargs) -> jnp.ndarray
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `wavelengths` | array | - | Wavelength grid [Å] |
| `linelist` | List/LineList | - | Spectral line data |
| `temperature` | float | - | Temperature [K] |
| `log_g` | float | - | Surface gravity [log cgs] |
| `abundances` | Dict | `{1:1.0, 2:0.1, 11:1e-6, 26:1e-4}` | Element abundances |
| `electron_density` | float | 1e14 | Electron density [cm⁻³] |
| `hydrogen_density` | float | 1e16 | Hydrogen density [cm⁻³] |
| `microturbulence` | float | 0.0 | Microturbulent velocity [km/s] |

#### Returns
```python
jnp.ndarray  # Line absorption coefficient [cm⁻¹]
```

#### Logic Flow

1. **Input Processing**:
   ```python
   # Handle different linelist formats
   if isinstance(linelist, LineList):
       lines = linelist.lines
   else:
       lines = linelist
   ```

2. **Line-by-Line Calculation**:
   ```python
   for line in lines:
       # Extract line parameters
       line_wl = line.wavelength          # [Å]
       species_id = line.species           # Species identifier
       excitation_potential = line.E_lower # [eV]
       log_gf = line.log_gf               # log(oscillator strength)
       
       # Get element abundance
       element_id = species_id // 100
       abundance = abundances[element_id]
       atomic_mass = _get_atomic_mass(element_id)
       
       # Calculate line opacity
       line_opacity = calculate_line_opacity_korg_method(
           wavelengths, line_wl, excitation_potential, log_gf,
           temperature, electron_density, hydrogen_density, abundance,
           atomic_mass=atomic_mass, microturbulence=microturbulence
       )
       
       total_opacity += line_opacity
   ```

#### API Called
- `calculate_line_opacity_korg_method()`: Core opacity calculation following Korg.jl methodology

---

### 2. `line_absorption()` - Single Line Opacity

**Purpose**: Calculate absorption coefficient for a single spectral line.

#### Signature
```python
def line_absorption(wavelength_grid: jnp.ndarray,
                   line_center: float,
                   oscillator_strength: float,
                   excitation_potential: float,
                   temperature: float,
                   number_density: float,
                   atomic_mass: float,
                   microturbulence: float = 0.0,
                   gamma_rad: float = 0.0,
                   gamma_vdw: float = 0.0,
                   gamma_stark: float = 0.0) -> jnp.ndarray
```

#### Logic Flow

1. **Doppler Width Calculation**:
   ```python
   doppler_width_value = doppler_width(
       line_center, temperature, atomic_mass, microturbulence
   )
   ```

2. **Total Broadening**:
   ```python
   lorentz_width = gamma_rad + gamma_vdw + gamma_stark
   ```

3. **Profile Calculation**:
   ```python
   profile = voigt_profile(wavelength_grid, line_center, 
                          doppler_width_value, lorentz_width)
   ```

4. **Line Strength and Opacity**:
   ```python
   line_strength = oscillator_strength * number_density
   return line_strength * profile
   ```

#### APIs Called
- `doppler_width()`: Thermal and turbulent Doppler broadening
- `voigt_profile()`: Voigt line profile calculation

---

### 3. `total_line_absorption_with_molecules()` - Enhanced Molecular Support

**Purpose**: Advanced line absorption including efficient molecular cross-section interpolation.

#### Key Features

1. **Atomic/Molecular Separation**:
   ```python
   # Separate lines by type
   atomic_lines = []
   molecular_lines = []
   
   for line in linelist:
       if is_molecular_species_id(line.species):
           molecular_lines.append(line)
       else:
           atomic_lines.append(line)
   ```

2. **Dual Molecular Methods**:
   ```python
   # Method 1: Fast cross-section interpolation
   if molecular_cross_sections is not None:
       alpha_molecular += interpolate_molecular_cross_sections(
           wavelengths, temperature, microturbulence,
           molecular_cross_sections, molecular_number_densities
       )
   
   # Method 2: Line-by-line calculation (more accurate)
   elif molecular_lines:
       alpha_molecular = calculate_molecular_line_absorption(
           wavelengths, molecular_lines, temperature, 
           microturbulence, molecular_number_densities
       )
   ```

#### APIs Called
- `interpolate_molecular_cross_sections()`: Fast molecular opacity from precomputed cross-sections
- `calculate_molecular_line_absorption()`: Line-by-line molecular calculation

---

## Line Profile Physics

### 4. Voigt Profile Implementation

#### Core Function: `voigt_hjerting()`

**Purpose**: Exact implementation of Hjerting function following Hunger (1965).

```python
@jax.jit
def voigt_hjerting(alpha: float, v: float) -> float:
    """
    Compute H(α, v) for Voigt profile calculation
    
    Parameters:
    -----------
    alpha : float
        Damping parameter (Lorentz/Doppler width ratio)
    v : float  
        Dimensionless frequency offset from line center
        
    Returns:
    --------
    float
        Hjerting function value H(α, v)
    """
```

#### Implementation Logic

1. **Case Classification**:
   ```python
   # Small alpha, large v: asymptotic expansion
   if alpha <= 0.2 and v >= 5:
       invv2 = 1.0 / v2
       return (alpha / sqrt_pi * invv2) * (1.0 + 1.5 * invv2 + 3.75 * invv2**2)
   
   # Small alpha, small v: Harris series
   elif alpha <= 0.2 and v < 5:
       H = harris_series(v)
       H0, H1, H2 = H[0], H[1], H[2]
       return H0 + (H1 + H2 * alpha) * alpha
   
   # Intermediate regime: modified Harris series
   elif alpha <= 1.4 and alpha + v < 3.2:
       # Complex polynomial calculations...
   ```

2. **Harris Series Calculation**:
   ```python
   @jax.jit
   def harris_series(v: float) -> jnp.ndarray:
       # Piecewise polynomial approximations
       # H1 calculation with three regions
       # H0 = exp(-v²), H2 = (1 - 2v²) * H0
       return jnp.array([H0, H1, H2])
   ```

#### APIs Called
- `harris_series()`: Polynomial approximations for Voigt function

---

## Broadening Mechanisms

### 5. `doppler_width()` - Thermal and Turbulent Broadening

**Purpose**: Calculate Gaussian (Doppler) line width from thermal and turbulent motions.

#### Formula
```
Δλ_D = (λ₀/c) * √(2kT/m + ξ²)
```

Where:
- `λ₀`: Line center wavelength
- `T`: Temperature
- `m`: Atomic mass
- `ξ`: Microturbulent velocity

### 6. `scaled_vdw()` - van der Waals Broadening

**Purpose**: Calculate pressure broadening from neutral hydrogen collisions.

#### Implementation
Following Barklem, Piskunov & O'Mara (2000) for enhanced precision.

### 7. `scaled_stark()` - Stark Broadening

**Purpose**: Calculate broadening from charged particle collisions.

#### Methods
- **Impact approximation**: For weak fields
- **Quasistatic approximation**: For strong fields
- **Stehlé & Hutcheon (1999)**: Advanced profiles for hydrogen

---

## Hydrogen Line Treatment

### 8. `hydrogen_line_absorption()` - Advanced Hydrogen Lines

**Purpose**: Sophisticated hydrogen line treatment with MHD formalism.

#### Key Features

1. **MHD Occupation Probability**:
   ```python
   def hummer_mihalas_w(T: float, n_eff: float, nH: float, 
                       nHe: float, ne: float) -> float:
       """
       Calculate occupation probability correction following
       Hummer & Mihalas (1988) formalism
       """
       # Neutral term contribution
       r_level = sqrt(5/2 * n_eff⁴ + 1/2 * n_eff²) * a₀
       neutral_term = (nH * (r_level + √3 * a₀)³ + 
                      nHe * (r_level + 1.02 * a₀)³)
       
       # Charged term with quantum correction K
       K = 16/3 * (n_eff/(n_eff+1))² * ((n_eff+7/6)/(n_eff²+n_eff+0.5))
       charged_term = 16 * ((e²)/(χ * √K))³ * ne
       
       return exp(-4π/3 * (neutral_term + charged_term))
   ```

2. **Series-Specific Treatment**:
   - **Lyman series**: Full quantum mechanical treatment
   - **Balmer series**: ABO van der Waals broadening
   - **Paschen series**: Simplified approach
   - **Brackett series**: Griem Stark broadening

3. **Oscillator Strength Calculation**:
   ```python
   def brackett_oscillator_strength(n: int, m: int) -> float:
       """Accurate to 10⁻⁴ for Brackett series (Goldwire 1968)"""
       # Complex polynomial fitting from Peterson & Kurucz
   ```

#### APIs Called
- `hummer_mihalas_w()`: MHD occupation probability
- `sigma_line()`: Quantum mechanical line cross-section
- `brackett_oscillator_strength()`: Oscillator strengths for hydrogen

---

## Molecular Line Support

### 9. Molecular Cross-Section Interpolation

#### `MolecularCrossSection` Class
```python
@dataclass
class MolecularCrossSection:
    wavelengths: jnp.ndarray      # Wavelength grid [cm]
    temperatures: jnp.ndarray     # Temperature grid [K]
    cross_sections: jnp.ndarray   # Cross-sections [cm²] (2D: T×λ)
    species_id: int               # Molecular species identifier
```

#### `interpolate_molecular_cross_sections()`
**Purpose**: Fast molecular opacity using precomputed cross-section tables.

```python
def interpolate_molecular_cross_sections(
    wavelengths: jnp.ndarray,
    temperature: float,
    microturbulence: float,
    molecular_cross_sections: Dict[int, MolecularCrossSection],
    molecular_number_densities: Dict[int, jnp.ndarray]
) -> jnp.ndarray:
    """
    Efficient molecular line opacity via interpolation
    
    Logic:
    1. For each molecular species
    2. Interpolate cross-section at (temperature, wavelength)
    3. Apply Doppler convolution if microturbulence > 0
    4. Multiply by number density
    5. Sum contributions
    """
```

### 10. Molecular Species Identification

#### `is_molecular_species_id()`
**Purpose**: Identify molecular species from species ID.

```python
def is_molecular_species_id(species_id: int) -> bool:
    """
    Uses ID ranges to identify molecules:
    - 101-199:   Diatomic molecules (H₂, etc.)
    - 601-699:   Carbon compounds (CH, CN, CO)
    - 701-799:   Nitrogen compounds (NH, N₂)
    - 801-899:   Oxygen compounds (OH, H₂O, O₂)
    - 1201-1299: Mg compounds (MgH)
    - 1401-1499: Si compounds (SiH, SiO)
    - 2208:      TiO
    - 2601-2699: Fe compounds (FeH)
    """
```

---

## Data Structures

### 11. `LineData` Dataclass

```python
@dataclass
class LineData:
    wavelength: float      # Line center wavelength [Å]
    species: int          # Species ID (atomic or molecular)
    log_gf: float         # log₁₀(oscillator strength)
    E_lower: float        # Lower excitation potential [eV]
    gamma_rad: float      # Radiative broadening parameter
    gamma_stark: float    # Stark broadening parameter
    vdw_param1: float     # van der Waals parameter 1
    vdw_param2: float     # van der Waals parameter 2
```

### 12. `LineList` Class

```python
class LineList:
    """
    Container for spectral line data with I/O capabilities
    
    Features:
    - VALD format reading
    - Kurucz format support
    - Line filtering and selection
    - Wavelength range extraction
    """
    
    def __init__(self, lines: List[LineData]):
        self.lines = lines
    
    def filter_by_wavelength(self, wl_min: float, wl_max: float) -> 'LineList'
    def filter_by_species(self, species_ids: List[int]) -> 'LineList'  
    def filter_by_strength(self, min_log_gf: float) -> 'LineList'
```

---

## Utility Functions

### 13. Line Profile Utilities

#### `calculate_line_profile()`
**Purpose**: Calculate normalized line profile.

```python
def calculate_line_profile(wavelengths: jnp.ndarray,
                          line_center: float,
                          doppler_width: float,
                          lorentz_width: float) -> jnp.ndarray:
    """Returns normalized Voigt profile"""
    return voigt_profile(wavelengths, line_center, doppler_width, lorentz_width)
```

#### `sigma_line()`
**Purpose**: Quantum mechanical line cross-section.

```python
@jax.jit  
def sigma_line(wavelength: float) -> float:
    """
    σ = (π * e² * λ²) / (mₑ * c²)
    
    Classical electron radius times λ²/π
    """
    return pi * electron_charge_cgs**2 * wavelength**2 / (electron_mass_cgs * c_cgs**2)
```

---

## Performance Optimizations

### 14. JAX Compilation Strategy

1. **JIT-Compiled Core Functions**:
   ```python
   @jax.jit
   def voigt_hjerting(alpha: float, v: float) -> float
   
   @jax.jit
   def harris_series(v: float) -> jnp.ndarray
   
   @jax.jit
   def hummer_mihalas_w(T, n_eff, nH, nHe, ne) -> float
   ```

2. **Vectorized Calculations**:
   ```python
   # Vectorize over wavelength grid
   profiles = jax.vmap(voigt_hjerting, in_axes=(None, 0))(alpha, v_grid)
   
   # Vectorize over line list
   line_opacities = jax.vmap(single_line_opacity)(line_parameters)
   ```

3. **Memory-Efficient Line Processing**:
   - Chunked line list processing for large line lists
   - Selective line filtering by wavelength range
   - Efficient molecular cross-section interpolation

### 15. Line Cutoff and Filtering

#### Wavelength-Based Filtering
```python
def filter_lines_by_wavelength(lines: List[LineData], 
                              wl_min: float, wl_max: float,
                              buffer: float = 10.0) -> List[LineData]:
    """
    Filter lines within wavelength range + buffer
    
    Buffer accounts for line wing contributions
    """
    effective_min = wl_min - buffer
    effective_max = wl_max + buffer
    
    return [line for line in lines 
            if effective_min <= line.wavelength <= effective_max]
```

#### Strength-Based Filtering
```python
def filter_weak_lines(lines: List[LineData], 
                     min_strength: float = -8.0) -> List[LineData]:
    """Remove lines weaker than threshold"""
    return [line for line in lines if line.log_gf >= min_strength]
```

---

## Integration with Synthesis Pipeline

### 16. Synthesis Integration Points

The lines module integrates with the main synthesis pipeline at several key points:

1. **Chemical Equilibrium Input**:
   ```python
   # Uses chemical equilibrium results for species number densities
   number_densities = equilibrium_result.number_densities
   electron_density = equilibrium_result.ne
   ```

2. **Atmosphere Structure Input**:
   ```python
   # Uses atmosphere structure for layer-by-layer calculation
   for i in range(atm['n_layers']):
       T = atm['temperature'][i]
       ne = atm['electron_density'][i]
       # Calculate line opacity for this layer
   ```

3. **Wavelength Grid Input**:
   ```python
   # Uses synthesis wavelength grid
   wavelengths = synthesis_wavelengths  # [Å]
   frequencies = c / (wavelengths * 1e-8)  # [Hz]
   ```

4. **Output to Radiative Transfer**:
   ```python
   # Line opacity adds to total absorption coefficient
   alpha_total = alpha_continuum + alpha_lines + alpha_hydrogen_lines
   ```

---

## Comparison with Korg.jl

### API Compatibility ✅

| Feature | Korg.jl | Jorg | Status |
|---------|---------|------|--------|
| **Core Functions** | | | |
| `line_absorption!()` | ✓ | `total_line_absorption()` | **Equivalent API** |
| Voigt profiles | ✓ | ✓ | **Exact implementation** |
| Harris series | ✓ | ✓ | **Exact coefficients** |
| **Hydrogen Lines** | | | |
| MHD formalism | ✓ | ✓ | **Full implementation** |
| Stark broadening | ✓ | ✓ | **Stehlé & Hutcheon** |
| ABO vdW broadening | ✓ | ✓ | **Barklem+ 2000** |
| **Molecular Lines** | | | |
| Cross-section interpolation | ✓ | ✓ | **Enhanced support** |
| Line-by-line calculation | ✓ | ✓ | **Full physics** |

### Physics Implementation ✅

| Component | Korg.jl Method | Jorg Implementation | Accuracy |
|-----------|----------------|-------------------|----------|
| **Voigt Profile** | Hjerting function (Hunger 1965) | `voigt_hjerting()` | **Exact** |
| **Harris Series** | Polynomial coefficients | `harris_series()` | **Exact** |
| **MHD Occupation** | Hummer & Mihalas (1988) | `hummer_mihalas_w()` | **Exact** |
| **Stark Broadening** | Stehlé & Hutcheon (1999) | Profile interpolation | **Exact** |
| **vdW Broadening** | Barklem+ 2000 | `scaled_vdw()` | **Exact** |
| **Molecular Physics** | Cross-section tables | Interpolation + LBL | **Enhanced** |

### Performance Characteristics

| Aspect | Korg.jl | Jorg | Notes |
|--------|---------|------|-------|
| **Line Profile Calculation** | Native Julia speed | JAX JIT-compiled | **Similar performance** |
| **Voigt Function** | Optimized algorithms | `@jax.jit` compiled | **High performance** |
| **Large Line Lists** | Efficient filtering | Vectorized processing | **Scalable** |
| **Molecular Lines** | Table interpolation | JAX interpolation | **Optimized** |
| **Memory Usage** | Julia GC | JAX memory management | **Efficient** |

---

## Usage Examples

### Basic Line Opacity Calculation
```python
from jorg.lines import total_line_absorption, LineData

# Create sample line list
lines = [
    LineData(wavelength=5000.0, species=2600, log_gf=-1.5, E_lower=2.0),
    LineData(wavelength=5001.0, species=1100, log_gf=-2.0, E_lower=1.5),
]

# Calculate line absorption
wavelengths = jnp.linspace(4999, 5002, 100)
alpha_lines = total_line_absorption(
    wavelengths, lines, 
    temperature=5000, log_g=4.5,
    abundances={26: 1e-4, 11: 1e-6},
    electron_density=1e13,
    microturbulence=2.0
)
```

### Advanced Hydrogen Line Treatment
```python
from jorg.lines.hydrogen_lines import hydrogen_line_absorption

# Calculate hydrogen line opacity with MHD formalism
alpha_h = hydrogen_line_absorption(
    wavelengths * 1e-8,  # Convert to cm
    temperature=6000,
    electron_density=1e13,
    nH_I=1e16, nHe_I=1e15,
    UH_I=2.0,
    microturbulence=2e5,  # cm/s
    use_MHD=True,
    adaptive_window=True
)
```

### Molecular Line Processing
```python
from jorg.lines import total_line_absorption_with_molecules

# Include molecular cross-sections
alpha_total = total_line_absorption_with_molecules(
    wavelengths, complete_linelist,
    temperature, log_g, abundances,
    molecular_cross_sections=co_cross_sections,
    molecular_number_densities={'CO': 1e12}
)
```

---

## Conclusion

The Jorg lines module provides:

1. **Complete Line Physics**: Full implementation of atomic and molecular line absorption
2. **Advanced Hydrogen Treatment**: MHD formalism with sophisticated broadening
3. **High Performance**: JAX-optimized computation with vectorization
4. **Korg.jl Compatibility**: Exact API and physics matching
5. **Molecular Support**: Enhanced molecular line capabilities
6. **Extensible Design**: Clean architecture for adding new line treatments

The implementation successfully replicates Korg.jl's proven line absorption methodology while providing performance optimizations and enhanced molecular line support in Python.