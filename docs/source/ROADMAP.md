# Jorg Development Roadmap: Complete JAX-based Stellar Spectral Synthesis

## Executive Summary

**Jorg** is the JAX-based translation of Korg.jl, designed to provide identical scientific accuracy while achieving 10-50x performance improvements through GPU acceleration and automatic differentiation. This roadmap defines the complete API structure and implementation plan based on comprehensive analysis of Korg.jl's architecture.

## Current Implementation Status

### ✅ Completed Components
- **Project Structure**: Modular architecture with `continuum`, `lines`, `statmech`, `utils`
- **Basic APIs**: `synth()`, `synthesize()`, `SynthesisResult` interfaces
- **Data Structures**: `LineData`, `Species`, basic JAX array handling
- **Foundation**: Partial continuum absorption, line absorption framework
- **Statistical Mechanics**: Chemical equilibrium solver foundation

### ✅ **NEW: Hydrogen Lines Complete**
- **Hydrogen Lines**: ✅ **COMPLETED** - Sophisticated treatment with MHD formalism
  - MHD (Hummer & Mihalas 1988) occupation probability corrections
  - ABO (Anstee-Barklem-O'Mara) van der Waals broadening for Balmer lines  
  - Griem 1960/1967 Stark broadening theory for Brackett lines
  - Pressure ionization and level dissolution effects
  - **Validation**: Exact agreement with Korg.jl to 6 decimal places

### ❌ Missing Critical Components
- **Radiative Transfer**: Complete formal solution implementation
- **Atmosphere Models**: MARCS interpolation and atmosphere handling
- **Complete Continuum**: All opacity sources (H⁻, metals, scattering)
- **Line Physics**: Complete Voigt profiles and broadening mechanisms
- **Integration**: End-to-end synthesis pipeline
- **Performance**: JIT optimization and GPU vectorization

## Complete API Reference

Following Korg.jl's architecture exactly, here are all APIs to implement:

### Phase 1: Core Foundation APIs (6-8 weeks)

#### 1.1 Atmosphere Handling (`jorg/atmosphere.py`)

```python
@jax.jit
def interpolate_marcs(
    Teff: float,                    # Effective temperature [K]
    logg: float,                    # Surface gravity [cgs]
    metallicity: float,             # [M/H] metallicity [dex]
    atmosphere_grid: AtmosphereGrid = None  # Pre-loaded MARCS grid
) -> AtmosphereModel:
    """
    Interpolate MARCS stellar atmosphere models
    
    Equivalent to Korg.jl's interpolate_marcs() function.
    Performs trilinear interpolation in (Teff, logg, [M/H]) space.
    
    Returns
    -------
    AtmosphereModel with T, P, ρ, nₑ profiles for each layer
    """

def read_model_atmosphere(filename: str) -> AtmosphereModel:
    """Parse MARCS atmosphere files (.mod format)"""

def load_marcs_grid(grid_path: str) -> AtmosphereGrid:
    """Load preprocessed MARCS atmosphere grid for fast interpolation"""

@dataclass
class AtmosphereModel:
    """Stellar atmosphere structure"""
    layers: jnp.ndarray            # [n_layers, 4] -> T, P, ρ, nₑ
    tau_5000: jnp.ndarray          # [n_layers] reference optical depths  
    z: jnp.ndarray                 # [n_layers] height coordinate [cm]
    n_layers: int                  # Number of atmospheric layers

@dataclass  
class AtmosphereGrid:
    """Pre-loaded MARCS model grid for interpolation"""
    teff_grid: jnp.ndarray         # Temperature grid points
    logg_grid: jnp.ndarray         # Surface gravity grid points
    metallicity_grid: jnp.ndarray  # Metallicity grid points
    atmosphere_data: jnp.ndarray   # [n_teff, n_logg, n_mh, n_layers, n_props]
```

#### 1.2 Chemical Equilibrium (`jorg/statmech/complete_eos.py`)

```python
@jax.jit
def chemical_equilibrium(
    temp: float,                   # Temperature [K]
    n_total: float,                # Total number density [cm⁻³]
    ne_model: float,               # Model atmosphere electron density [cm⁻³]
    absolute_abundances: jnp.ndarray,  # [92] normalized abundances
    ionization_energies: Dict[int, Tuple[float, float, float]], # χ_I, χ_II, χ_III
    partition_funcs: Dict[str, Callable],  # U(T) functions
    log_equilibrium_constants: Dict[str, Callable],  # Molecular K_eq(T) 
    electron_density_warn_threshold: float = 0.1,
    electron_density_warn_min_value: float = 1e-4
) -> Tuple[float, Dict[str, float]]:
    """
    Solve chemical equilibrium using Newton-Raphson method
    
    Equivalent to Korg.jl's chemical_equilibrium() function.
    Determines ionization balance and molecular equilibrium.
    
    Returns
    -------
    Tuple of (corrected_electron_density, species_number_densities)
    """

@jax.jit
def saha_ion_weights(
    T: float,                      # Temperature [K]
    ne: float,                     # Electron density [cm⁻³]
    atom: int,                     # Atomic number
    ionization_energies: Tuple[float, float, float],
    partition_funcs: Tuple[float, float, float]  # UI, UII, UIII
) -> Tuple[float, float]:
    """Calculate ionization equilibrium weights (wII, wIII)"""

@jax.jit  
def translational_partition_function(mass: float, T: float) -> float:
    """Translational partition function for free particles"""

def setup_chemical_equilibrium_residuals(
    T: float, n_total: float, absolute_abundances: jnp.ndarray,
    ionization_energies: Dict, partition_funcs: Dict, 
    log_equilibrium_constants: Dict
) -> Callable:
    """Setup residual function for Newton-Raphson solver"""
```

#### 1.3 Complete Continuum Absorption (`jorg/continuum/complete_continuum.py`)

```python
@jax.jit
def total_continuum_absorption(
    frequencies: jnp.ndarray,      # [n_freq] frequencies [Hz] (sorted)
    temperature: float,            # Temperature [K]
    electron_density: float,       # Electron density [cm⁻³]
    number_densities: Dict[str, float],  # Species densities [cm⁻³]
    partition_funcs: Dict[str, Callable],  # Partition functions
    error_oobounds: bool = False   # Error on out-of-bounds frequencies
) -> jnp.ndarray:
    """
    Calculate total continuum absorption coefficient
    
    Equivalent to Korg.jl's total_continuum_absorption() function.
    Includes all continuum sources: H, He, metals, scattering.
    
    Returns
    -------
    jnp.ndarray of shape [n_freq] with absorption coefficients [cm⁻¹]
    """

# Individual continuum sources (following Korg.jl structure):

@jax.jit
def h_i_bf_absorption(frequencies: jnp.ndarray, T: float, nH_I: float, 
                      nHe_I: float, ne: float, U_inv: float) -> jnp.ndarray:
    """H I bound-free photo-ionization"""

@jax.jit  
def h_minus_bf_absorption(frequencies: jnp.ndarray, T: float, 
                          nH_I_div_U: float, ne: float) -> jnp.ndarray:
    """H⁻ bound-free photo-detachment"""

@jax.jit
def h_minus_ff_absorption(frequencies: jnp.ndarray, T: float,
                          nH_I_div_U: float, ne: float) -> jnp.ndarray:
    """H⁻ free-free absorption"""

@jax.jit
def h2_plus_bf_ff_absorption(frequencies: jnp.ndarray, T: float,
                             nH_I: float, nH_II: float) -> jnp.ndarray:
    """H₂⁺ bound-free and free-free processes"""

@jax.jit
def he_minus_ff_absorption(frequencies: jnp.ndarray, T: float,
                           nHe_I_div_U: float, ne: float) -> jnp.ndarray:
    """He⁻ free-free absorption"""

@jax.jit
def positive_ion_ff_absorption(frequencies: jnp.ndarray, T: float,
                               number_densities: Dict[str, float], ne: float) -> jnp.ndarray:
    """Free-free absorption by positive ions"""

@jax.jit
def metal_bf_absorption(frequencies: jnp.ndarray, T: float,
                        number_densities: Dict[str, float]) -> jnp.ndarray:
    """Metal bound-free cross-sections from TOPBase/NORAD"""

@jax.jit
def electron_scattering(ne: float) -> float:
    """Thomson scattering by free electrons"""

@jax.jit
def rayleigh_scattering(frequencies: jnp.ndarray, nH_I: float, 
                        nHe_I: float, nH2: float) -> jnp.ndarray:
    """Rayleigh scattering by neutral atoms and molecules"""
```

#### 1.4 Line Absorption (`jorg/lines/core.py`)

```python
@jax.jit
def line_absorption(
    alpha: jnp.ndarray,            # [n_layers, n_wavelengths] absorption matrix
    linelist: jnp.ndarray,         # Line data array
    wavelengths: jnp.ndarray,      # [n_wavelengths] wavelength grid [cm]
    temperatures: jnp.ndarray,     # [n_layers] layer temperatures [K]
    electron_densities: jnp.ndarray,  # [n_layers] electron densities [cm⁻³]
    number_densities: Dict[str, jnp.ndarray],  # Species densities [cm⁻³]
    partition_funcs: Dict[str, Callable],  # Partition functions
    microturbulence: Union[float, jnp.ndarray],  # Microturbulent velocity [cm/s]
    alpha_continuum: jnp.ndarray,  # [n_layers] continuum opacity interpolators
    cutoff_threshold: float = 3e-4,  # Line profile cutoff threshold
    line_buffer: float = 10e-8,    # Line calculation buffer [cm]
    verbose: bool = False          # Progress output
) -> jnp.ndarray:
    """
    Calculate line absorption and add to total opacity matrix (in-place)
    
    Equivalent to Korg.jl's line_absorption!() function.
    Uses vectorized operations over lines and wavelengths.
    
    Returns
    -------
    Modified absorption matrix alpha (in-place modification)
    """

@jax.jit
def hydrogen_line_absorption(
    alpha_layer: jnp.ndarray,      # [n_wavelengths] single layer absorption
    wavelengths: jnp.ndarray,      # [n_wavelengths] wavelength grid [cm]
    temperature: float,            # Temperature [K]
    electron_density: float,       # Electron density [cm⁻³]
    nH_I: float,                   # Neutral hydrogen density [cm⁻³]
    nHe_I: float,                  # Neutral helium density [cm⁻³]  
    U_H_I: float,                  # H I partition function
    microturbulence: float,        # Microturbulent velocity [cm/s]
    window_size: float = 150e-8,   # Line window size [cm]
    use_MHD: bool = True           # Use MHD occupation probability formalism
) -> jnp.ndarray:
    """
    ✅ **IMPLEMENTED** - Special hydrogen line treatment with advanced physics
    
    **Status: COMPLETE** - Sophisticated implementation matching Korg.jl exactly:
    - MHD (Hummer & Mihalas 1988) occupation probability formalism
    - ABO van der Waals broadening for Balmer lines (Hα, Hβ, Hγ)
    - Griem 1960/1967 Stark broadening theory for Brackett lines
    - Pressure ionization effects validated across stellar conditions
    
    **Files**: `src/jorg/lines/hydrogen_lines.py`, `hydrogen_lines_simple.py`
    **Validation**: Exact agreement with Korg.jl (6 decimal places)
    
    Returns
    -------
    Modified layer absorption (in-place modification)
    """
```

#### 1.5 Line Profile Physics (`jorg/lines/profiles.py`)

```python
@jax.jit
def voigt_profile(
    nu: jnp.ndarray,               # [n_freq] frequencies [Hz]
    nu0: float,                    # Line center frequency [Hz]
    gamma_L: float,                # Lorentzian HWHM [Hz]
    gamma_G: float                 # Gaussian HWHM [Hz]
) -> jnp.ndarray:
    """
    JAX-optimized Voigt profile calculation
    
    Uses complex error function (jax.scipy.special.wofz) for accuracy.
    Equivalent to Korg.jl's line profile calculations.
    
    Returns
    -------
    Normalized line profile [unitless]
    """

@jax.jit
def doppler_width(
    wavelength: float,             # Line wavelength [cm]
    temperature: float,            # Temperature [K]
    mass: float,                   # Atomic mass [amu]
    microturbulence: float         # Microturbulent velocity [cm/s]
) -> float:
    """Thermal + turbulent Doppler broadening width"""

@jax.jit
def scaled_stark(
    gamma_stark: float,            # Reference Stark broadening [s⁻¹]
    temperature: float             # Temperature [K]
) -> float:
    """Temperature-scaled Stark broadening parameter"""

@jax.jit
def scaled_vdw(
    vdw_params: Tuple[float, float],  # van der Waals parameters
    mass: float,                   # Atomic mass [amu]
    temperature: float             # Temperature [K]
) -> float:
    """van der Waals broadening calculation"""

@jax.jit
def sigma_line(wavelength: float) -> float:
    """Quantum mechanical line cross-section"""

@jax.jit
def inverse_gaussian_density(rho_crit: float, sigma: float) -> float:
    """Calculate window size for Gaussian line core"""

@jax.jit
def inverse_lorentz_density(rho_crit: float, gamma: float) -> float:
    """Calculate window size for Lorentzian line wings"""
```

#### 1.6 Radiative Transfer (`jorg/radiative_transfer.py`)

```python
@jax.jit
def radiative_transfer(
    alpha: jnp.ndarray,            # [n_layers, n_wavelengths] absorption matrix
    source_function: jnp.ndarray,  # [n_layers, n_wavelengths] source function
    spatial_coord: jnp.ndarray,    # [n_layers] height/radius coordinate [cm]
    mu_points: Union[int, jnp.ndarray],  # μ quadrature points or values
    spherical: bool = False,       # Spherical vs plane-parallel geometry
    include_inward_rays: bool = False,  # Include inward propagation
    alpha_ref: Optional[jnp.ndarray] = None,  # Reference absorption for τ scheme
    tau_ref: Optional[jnp.ndarray] = None,    # Reference optical depths
    tau_scheme: str = "anchored",  # Optical depth calculation method
    I_scheme: str = "linear_flux_only"  # Intensity calculation method
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Solve radiative transfer equation for emergent flux
    
    Equivalent to Korg.jl's radiative_transfer() function.
    Implements anchored τ scheme and linear source function method.
    
    Returns
    -------
    Tuple of (flux, intensity, mu_grid, mu_weights)
    - flux: [n_wavelengths] emergent flux [erg/s/cm²/Å]
    - intensity: [n_mu, n_wavelengths] or [n_mu, n_layers, n_wavelengths]
    - mu_grid: [n_mu] μ values used
    - mu_weights: [n_mu] integration weights
    """

@jax.jit
def calculate_rays(
    mu_surface_grid: jnp.ndarray,  # μ values at stellar surface
    spatial_coord: jnp.ndarray,    # Height/radius coordinate [cm]
    spherical: bool                # Geometry flag
) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
    """Calculate ray paths through atmosphere"""

@jax.jit
def compute_tau_anchored(
    tau: jnp.ndarray,              # [n_layers] optical depth array
    alpha: jnp.ndarray,            # [n_layers] absorption coefficients
    integrand_factor: jnp.ndarray, # Integration weights
    log_tau_ref: jnp.ndarray       # [n_layers] reference log optical depths
) -> jnp.ndarray:
    """Compute optical depth using anchored scheme"""

@jax.jit
def compute_intensity_linear(
    tau: jnp.ndarray,              # [n_layers] optical depth along ray
    source_function: jnp.ndarray   # [n_layers] source function along ray
) -> float:
    """Compute intensity with linear source function interpolation"""

@jax.jit
def blackbody_source_function(temperature: float, wavelength: float) -> float:
    """Planck blackbody function for LTE source function"""
```

### Phase 2: High-Level Integration (4-6 weeks)

#### 2.1 Complete Synthesis Pipeline (`jorg/synthesis.py`)

```python
@jax.jit
def synthesize(
    atmosphere: AtmosphereModel,   # Stellar atmosphere structure
    linelist: jnp.ndarray,         # Atomic/molecular lines
    abundances: jnp.ndarray,       # [92] abundance vector
    wavelengths: Union[Tuple[float, float], jnp.ndarray],  # Wavelength specification
    vmic: float = 1.0,             # Microturbulent velocity [km/s]
    line_buffer: float = 10.0,     # Line calculation buffer [Å]
    cntm_step: float = 1.0,        # Continuum grid spacing [Å]
    air_wavelengths: bool = False, # Input wavelengths in air
    hydrogen_lines: bool = True,   # Include H lines
    use_MHD_for_hydrogen_lines: bool = True,  # Use MHD formalism
    hydrogen_line_window_size: float = 150.0,  # H line window [Å]
    mu_values: int = 20,           # μ quadrature points
    line_cutoff_threshold: float = 3e-4,  # Line profile cutoff
    return_continuum: bool = True, # Return continuum flux
    I_scheme: str = "linear_flux_only",  # Intensity calculation scheme
    tau_scheme: str = "anchored",  # Optical depth scheme
    verbose: bool = False          # Progress output
) -> SynthesisResult:
    """
    Core synthesis function following Korg.jl exactly
    
    This is the main computational engine that orchestrates:
    1. Wavelength processing and validation
    2. Linelist filtering by wavelength range
    3. Per-layer chemical equilibrium calculation
    4. Continuum opacity calculation with interpolation
    5. Line opacity calculation (including hydrogen lines)
    6. Radiative transfer computation
    
    Returns
    -------
    SynthesisResult with detailed diagnostic information
    """

def synth(
    Teff: float = 5000,            # Effective temperature [K]
    logg: float = 4.5,             # Surface gravity [cgs]
    m_H: float = 0.0,              # Metallicity [M/H] [dex]
    alpha_H: Optional[float] = None,  # Alpha enhancement [α/H] [dex]
    wavelengths: Union[Tuple[float, float], jnp.ndarray] = (5000, 6000),  # [Å]
    linelist: Optional[jnp.ndarray] = None,  # Line data
    rectify: bool = True,          # Continuum normalize
    R: Union[float, Callable] = float('inf'),  # Resolving power
    vsini: float = 0.0,            # Projected rotation velocity [km/s]
    vmic: float = 1.0,             # Microturbulent velocity [km/s]
    **kwargs                       # Additional synthesis options
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    High-level user interface matching Korg.jl API exactly
    
    This function provides the same interface as Korg.jl's synth() function,
    with automatic atmosphere interpolation and post-processing.
    
    Returns
    -------
    Tuple of (wavelengths, flux, continuum)
    - wavelengths: [n_wavelengths] wavelengths [Å]
    - flux: [n_wavelengths] rectified flux (0-1) or absolute flux
    - continuum: [n_wavelengths] continuum flux [erg/s/cm²/Å]
    """

@dataclass
class SynthesisResult:
    """
    Container for detailed synthesis results
    
    Equivalent to Korg.jl's SynthesisResult structure.
    """
    flux: jnp.ndarray              # [n_wavelengths] emergent flux
    continuum: Optional[jnp.ndarray]  # [n_wavelengths] continuum flux
    intensity: jnp.ndarray         # Intensity at all μ values and layers
    alpha: jnp.ndarray             # [n_layers, n_wavelengths] absorption matrix
    mu_grid: jnp.ndarray           # μ values and weights
    number_densities: Dict[str, jnp.ndarray]  # Species densities
    electron_number_density: jnp.ndarray      # [n_layers] electron density
    wavelengths: jnp.ndarray       # [n_wavelengths] vacuum wavelengths [Å]
    subspectra: List[slice]        # Wavelength window indices
```

#### 2.2 Supporting Functions (`jorg/abundances.py`)

```python
def format_abundances(
    m_H: float,                    # Metallicity [M/H] [dex]
    alpha_H: float,                # Alpha enhancement [α/H] [dex]
    abundances: Dict[str, float]   # Element overrides [X/H] [dex]
) -> jnp.ndarray:
    """
    Create abundance vector [92 elements]
    
    Equivalent to Korg.jl's format_A_X() function.
    Returns A(X) = log(N_X/N_H) + 12 for all elements.
    """

def get_solar_abundances() -> jnp.ndarray:
    """Standard solar abundance pattern"""

def apply_abundance_pattern(
    base_abundances: jnp.ndarray,
    metallicity: float,
    alpha_enhancement: float,
    element_overrides: Dict[str, float]
) -> jnp.ndarray:
    """Apply abundance pattern modifications"""
```

#### 2.3 Data Management (`jorg/data/`)

```python
def read_linelist(
    filename: str,                 # Path to linelist file
    format: str = "vald",          # Format: "vald", "kurucz", "moog", etc.
    isotopic_abundances: Optional[Dict] = None  # Isotope data
) -> jnp.ndarray:
    """
    Parse linelist files into JAX arrays
    
    Equivalent to Korg.jl's read_linelist() function.
    Supports VALD, Kurucz, MOOG, Turbospectrum formats.
    """

def save_linelist(
    path: str,                     # Output file path
    linelist: jnp.ndarray          # Linelist to save
) -> None:
    """Save linelist in optimized HDF5 format"""

@dataclass
class LineData:
    """Single spectral line representation"""
    wavelength: float              # Wavelength [cm]
    log_gf: float                 # log(gf) oscillator strength
    species: int                  # Species identifier
    E_lower: float                # Lower energy level [eV]
    gamma_rad: float              # Radiative damping [s⁻¹]
    gamma_stark: float            # Stark broadening [s⁻¹]
    vdw_params: Tuple[float, float]  # van der Waals parameters

class WavelengthGrid:
    """JAX-optimized wavelength handling"""
    def __init__(self, ranges: List[Tuple[float, float]], air_wavelengths: bool = False):
        self.ranges = ranges
        self.air_wavelengths = air_wavelengths
        
    def to_vacuum(self) -> 'WavelengthGrid':
        """Convert air wavelengths to vacuum"""
        
    def to_frequency(self) -> jnp.ndarray:
        """Convert wavelengths to frequencies"""
```

### Phase 3: Advanced JAX Features (3-4 weeks)

#### 3.1 Performance Optimization

```python
# JIT-compiled core functions
@jax.jit
def fast_synth(params: jnp.ndarray, wavelengths: jnp.ndarray, 
               linelist: jnp.ndarray) -> jnp.ndarray:
    """Ultra-fast synthesis with minimal overhead"""

# Vectorized synthesis for parameter grids
batch_synth = jax.vmap(synth, in_axes=(0, 0, 0, None, None))

# Multi-GPU synthesis
parallel_synth = jax.pmap(synth, axis_name='device')

# Memory-optimized chunked processing
@jax.jit
def chunked_line_absorption(
    alpha: jnp.ndarray,
    linelist_chunks: List[jnp.ndarray],
    chunk_size: int = 1000
) -> jnp.ndarray:
    """Process large linelists in memory-efficient chunks"""
```

#### 3.2 Automatic Differentiation

```python
# Gradient-enabled synthesis
grad_synth = jax.grad(synth, argnums=(0, 1, 2))  # w.r.t. Teff, logg, m_H

# Hessian for uncertainty estimation
hess_synth = jax.hessian(synth, argnums=(0, 1, 2))

# Custom derivatives for physical functions
@jax.custom_vjp
def physical_function_with_analytic_grad(x):
    """Function with custom analytic gradient for numerical stability"""
```

#### 3.3 Advanced Features

```python
def fit_spectrum(
    observed_wavelengths: jnp.ndarray,
    observed_flux: jnp.ndarray,
    linelist: jnp.ndarray,
    initial_params: Dict[str, float],
    parameters_to_fit: List[str] = ["Teff", "logg", "m_H"]
) -> Dict[str, float]:
    """
    Gradient-based parameter fitting using JAX optimization
    
    Leverages automatic differentiation for robust convergence.
    """

def parameter_sensitivity_analysis(
    params: Dict[str, float],
    wavelengths: jnp.ndarray,
    linelist: jnp.ndarray,
    param_variations: Dict[str, float]
) -> Dict[str, jnp.ndarray]:
    """Compute parameter sensitivities using automatic differentiation"""

def uncertainty_propagation(
    params: Dict[str, float],
    param_uncertainties: Dict[str, float],
    wavelengths: jnp.ndarray,
    linelist: jnp.ndarray
) -> jnp.ndarray:
    """Propagate parameter uncertainties to spectrum using gradients"""
```

## Implementation Timeline

### Phase 1: Core Foundation (Weeks 1-8)
**Priority: Critical Path**

1. **Weeks 1-2**: Atmosphere handling and interpolation
   - `interpolate_marcs()`, `read_model_atmosphere()`
   - MARCS grid preprocessing and loading
   - Atmosphere data structures

2. **Weeks 3-4**: Complete chemical equilibrium
   - `chemical_equilibrium()` with full Newton-Raphson solver
   - Saha equation and molecular equilibrium
   - Species number density calculations

3. **Weeks 5-6**: Complete continuum absorption
   - All continuum sources: H, He, metals, scattering
   - `total_continuum_absorption()` function
   - Performance optimization with vectorization

4. **Weeks 7-8**: Line absorption physics
   - Voigt profiles and broadening mechanisms
   - `line_absorption()` function with proper windowing
   - Hydrogen line special treatment

### Phase 2: Integration (Weeks 9-14)
**Priority: System Integration**

1. **Weeks 9-10**: Radiative transfer implementation
   - `radiative_transfer()` with anchored τ scheme
   - Ray calculation and intensity integration
   - Multiple integration schemes

2. **Weeks 11-12**: Complete synthesis pipeline
   - `synthesize()` function integration
   - End-to-end testing with reference comparisons
   - Data flow optimization

3. **Weeks 13-14**: High-level API and post-processing
   - `synth()` function with user-friendly interface
   - LSF application and rotational broadening
   - Error handling and validation

### Phase 3: Performance & Features (Weeks 15-19)
**Priority: Optimization**

1. **Weeks 15-16**: JAX optimization
   - JIT compilation for all kernels
   - Vectorization and memory optimization
   - GPU acceleration benchmarking

2. **Weeks 17-18**: Advanced features
   - Automatic differentiation integration
   - Batched synthesis and parameter fitting
   - Multi-GPU parallelization

3. **Week 19**: Performance tuning
   - Memory profiling and optimization
   - Computational bottleneck elimination
   - Scaling tests

### Phase 4: Validation & Documentation (Weeks 20-23)
**Priority: Production Readiness**

1. **Week 20**: Accuracy validation
   - Reference comparisons with Korg.jl (<0.1% difference)
   - Physical sanity checks
   - Edge case testing

2. **Week 21**: Performance benchmarking
   - Speed comparisons (target: 10-50x speedup)
   - Memory usage optimization
   - Multi-GPU scaling tests

3. **Weeks 22-23**: Documentation and examples
   - Complete API documentation
   - Jupyter notebook tutorials
   - Performance guides and best practices

## Expected Performance Gains

### Computational Improvements
- **10-50x speedup** from GPU acceleration (line absorption bottleneck)
- **2-5x speedup** from JIT compilation optimization
- **Linear scaling** with additional GPUs using `pmap`
- **Memory efficiency** through vectorized operations

### Scientific Benefits
- **Identical accuracy** to Korg.jl (validated to <0.1% difference)
- **Automatic differentiation** enables gradient-based parameter fitting
- **Batched operations** for efficient parameter space exploration
- **Reproducibility** through deterministic JAX computations

### Development Advantages
- **GPU acceleration** for large-scale stellar surveys
- **Parameter fitting** with robust gradient-based optimization
- **Uncertainty quantification** through automatic differentiation
- **Scalability** to supercomputing environments

## Success Criteria

### Technical Metrics
- **Accuracy**: <0.1% RMS difference vs Korg.jl for solar spectrum
- **Performance**: >10x speedup on GPU vs Julia multithreaded CPU
- **Memory**: <2GB for typical synthesis (5000-6000 Å, R=50,000)
- **Scaling**: Linear performance improvement with additional GPUs

### API Completeness
- **100% compatibility** with Korg.jl's main synthesis APIs
- **Complete physics** implementation (all continuum/line sources)
- **Advanced features** (gradients, batching, multi-GPU)
- **Production stability** with comprehensive error handling

This roadmap ensures Jorg delivers on the promise of high-performance stellar spectral synthesis while maintaining the scientific rigor and accuracy of the original Korg.jl implementation.