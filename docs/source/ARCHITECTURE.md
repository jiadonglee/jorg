# Jorg Architecture Analysis

**Detailed technical analysis of Korg.jl architecture for JAX translation**

## Current Korg.jl Architecture Analysis

### Main Modules and Their Purposes

#### Core Entry Points
- **`synth()`** (`src/synth.jl`): High-level user-friendly interface returning `(wavelengths, flux, continuum)`
- **`synthesize()`** (`src/synthesize.jl`): Lower-level interface returning detailed `SynthesisResult` with full diagnostics

#### Key Computational Modules

1. **Line Absorption** (`src/line_absorption.jl`)
   - **Purpose**: Compute opacity coefficients from atomic/molecular lines
   - **Key functions**: `line_absorption!()`, Voigt profile calculations, line broadening
   - **Computational bottleneck**: Dominates synthesis runtime, heavily multithreaded

2. **Continuum Absorption** (`src/ContinuumAbsorption/`)
   - **Purpose**: Bound-free, free-free, and scattering opacity calculations
   - **Sub-modules**: H/He absorption, metal bound-free, positive ion free-free, scattering
   - **Key function**: `total_continuum_absorption()`

3. **Radiative Transfer** (`src/RadiativeTransfer/`)
   - **Purpose**: Formal solution of radiative transfer equation
   - **Methods**: Linear interpolation schemes, Bezier (deprecated), exponential integral optimizations
   - **Key function**: `radiative_transfer()`

4. **Statistical Mechanics** (`src/statmech.jl`)
   - **Purpose**: Chemical equilibrium, ionization balance, molecular equilibrium
   - **Dependencies**: NLsolve.jl for nonlinear equation solving
   - **Key function**: `chemical_equilibrium()`

5. **Hydrogen Line Absorption** (`src/hydrogen_line_absorption.jl`)
   - **Purpose**: Special treatment for hydrogen lines with MHD occupation probability formalism
   - **Separate from general line absorption due to complexity**

## Key Data Structures

### Atmospheric Models
```julia
struct PlanarAtmosphereLayer{F1,F2,F3,F4,F5}
    tau_5000::F1                # optical depth at 5000 Å
    z::F2                       # height (cm)
    temp::F3                    # temperature (K)
    electron_number_density::F4 # cm^-3
    number_density::F5          # cm^-3
end
```

**JAX Translation Strategy**:
```python
@dataclass
class AtmosphereLayer:
    """JAX-compatible atmospheric layer"""
    tau_5000: jnp.ndarray       # (n_layers,) optical depth at 5000 Å
    z: jnp.ndarray             # (n_layers,) height (cm)
    temp: jnp.ndarray          # (n_layers,) temperature (K)
    electron_density: jnp.ndarray  # (n_layers,) cm^-3
    number_density: jnp.ndarray    # (n_layers,) cm^-3
    
    def __post_init__(self):
        # Validate array shapes match
        shapes = [arr.shape for arr in [self.tau_5000, self.z, self.temp, 
                                       self.electron_density, self.number_density]]
        assert all(s == shapes[0] for s in shapes), "All arrays must have same shape"
```

### Line Data
```julia
struct Line{F1,F2,F3,F4,F5,F6}
    wl::F1                  # wavelength (cm)
    log_gf::F2             # log oscillator strength
    species::Species       # chemical species
    E_lower::F3            # lower energy level (eV)
    gamma_rad::F4          # radiative damping (s^-1)
    gamma_stark::F5        # Stark broadening (s^-1)
    vdW::Tuple{F6,F6}     # van der Waals broadening parameters
end
```

**JAX Translation Strategy**:
```python
@dataclass
class LineList:
    """JAX-optimized line list representation"""
    wl: jnp.ndarray           # (n_lines,) wavelength (cm)
    log_gf: jnp.ndarray       # (n_lines,) log oscillator strength
    species_id: jnp.ndarray   # (n_lines,) integer species ID
    E_lower: jnp.ndarray      # (n_lines,) lower energy level (eV)
    gamma_rad: jnp.ndarray    # (n_lines,) radiative damping (s^-1)
    gamma_stark: jnp.ndarray  # (n_lines,) Stark broadening (s^-1)
    vdW_a: jnp.ndarray        # (n_lines,) vdW parameter A
    vdW_b: jnp.ndarray        # (n_lines,) vdW parameter B
    
    @property
    def n_lines(self) -> int:
        return len(self.wl)
```

### Species Representation
```julia
struct Formula
    atoms::SVector{6,UInt8}  # Up to 6 atoms per molecule
end

struct Species
    formula::Formula
    charge::Int8             # ionization state
end
```

**JAX Translation Strategy**:
```python
class SpeciesRegistry:
    """Efficient integer-based species encoding for JAX"""
    
    def __init__(self):
        self.species_to_id = {}  # Species -> int
        self.id_to_species = {}  # int -> Species
        self.next_id = 0
    
    def register_species(self, formula: str, charge: int) -> int:
        """Register a species and return its integer ID"""
        species_key = (formula, charge)
        if species_key not in self.species_to_id:
            self.species_to_id[species_key] = self.next_id
            self.id_to_species[self.next_id] = species_key
            self.next_id += 1
        return self.species_to_id[species_key]
    
    def get_species_array(self, species_list) -> jnp.ndarray:
        """Convert list of species to JAX array of IDs"""
        return jnp.array([self.register_species(s.formula, s.charge) 
                         for s in species_list])
```

### Wavelength Handling
```julia
struct Wavelengths{F,R} <: AbstractArray{F,1}
    wl_ranges::Vector{R}     # wavelength ranges (cm)
    all_wls::Vector{F}       # precomputed wavelengths
    all_freqs::Vector{F}     # precomputed frequencies
end
```

**JAX Translation Strategy**:
```python
@dataclass
class Wavelengths:
    """JAX-optimized wavelength/frequency handling"""
    wl: jnp.ndarray          # (n_wl,) wavelengths (cm)
    freq: jnp.ndarray        # (n_wl,) frequencies (Hz)
    
    @classmethod
    def from_range(cls, wl_min: float, wl_max: float, 
                   n_points: int = None, delta_wl: float = None):
        """Create wavelength grid from range"""
        if n_points is not None:
            wl = jnp.linspace(wl_min, wl_max, n_points)
        elif delta_wl is not None:
            wl = jnp.arange(wl_min, wl_max + delta_wl, delta_wl)
        else:
            raise ValueError("Must specify either n_points or delta_wl")
        
        freq = constants.c_cgs / wl  # Hz
        return cls(wl=wl, freq=freq)
    
    @property
    def n_wavelengths(self) -> int:
        return len(self.wl)
```

## Current Parallelization Strategy

### Multithreading Implementation
**Location**: `src/line_absorption.jl` lines 46-111

**Current Strategy**: Chunk-based task parallelism over spectral lines
```julia
n_chunks = tasks_per_thread * Threads.nthreads()
chunk_size = max(1, length(linelist) ÷ n_chunks + (length(linelist) % n_chunks > 0))
linelist_chunks = partition(linelist, chunk_size)
tasks = map(linelist_chunks) do linelist_chunk
    Threads.@spawn begin
        # Local computation for each chunk
    end
end
```

**JAX Translation Strategy**:
```python
@jax.jit
def line_absorption_vectorized(frequencies, linelist, atmosphere, abundances):
    """Fully vectorized line absorption calculation"""
    
    # Compute all line profiles simultaneously using vmap
    line_profiles = jax.vmap(
        compute_line_profile, 
        in_axes=(None, 0, 0, 0, 0, None)  # broadcast over lines
    )(frequencies, linelist.wl, linelist.log_gf, linelist.E_lower, 
      linelist.species_id, atmosphere)
    
    # Sum contributions from all lines
    return jnp.sum(line_profiles, axis=0)

@jax.jit  
def compute_line_profile(frequencies, wl_line, log_gf, E_lower, species_id, atmosphere):
    """Compute absorption profile for a single line"""
    # Vectorized Voigt profile calculation
    return voigt_profile(frequencies, wl_line, gamma_total, doppler_width)
```

### Parallelization Characteristics
- **Current**: Embarrassingly parallel, static chunking
- **JAX Approach**: Fully vectorized, automatic GPU parallelization
- **Memory pattern**: Eliminated need for manual chunking through vectorization
- **Load balancing**: Automatic through JAX's compilation

## Computational Bottlenecks

### Primary Bottlenecks (Performance Impact)

1. **Line Absorption Calculations** (~80-90% of runtime)
   - **Current**: Nested loops over lines and wavelengths
   - **JAX Solution**: Vectorized tensor operations
   ```python
   # Instead of nested loops:
   # for line in linelist:
   #     for wl in wavelengths:
   #         profile[wl] += voigt(...)
   
   # JAX vectorized approach:
   profiles = jax.vmap(jax.vmap(voigt, in_axes=(0, None)), in_axes=(None, 0))
   total_absorption = jnp.sum(profiles, axis=0)
   ```

2. **Chemical Equilibrium Solving** (~5-10% of runtime)
   - **Current**: NLsolve.jl nonlinear solver
   - **JAX Solution**: `jax.scipy.optimize` with autodiff
   ```python
   @jax.jit
   def chemical_equilibrium_jax(T, P, abundances):
       def residual(x):
           return saha_equations(x, T, P, abundances)
       
       # Automatic differentiation provides Jacobian
       solution = jax.scipy.optimize.fsolve(
           residual, initial_guess, jac=jax.jacfwd(residual)
       )
       return solution
   ```

3. **Continuum Absorption** (~3-5% of runtime)
   - **Current**: Sequential calculation of opacity sources
   - **JAX Solution**: Parallel computation of all sources
   ```python
   @jax.jit
   def total_continuum_absorption(frequencies, T, ne, densities):
       # Compute all opacity sources in parallel
       h_minus_bf = jax.vmap(h_minus_bound_free)(frequencies, T, ne)
       h_minus_ff = jax.vmap(h_minus_free_free)(frequencies, T, ne)
       thomson = thomson_scattering(ne)
       rayleigh = jax.vmap(rayleigh_scattering)(frequencies, densities)
       
       return h_minus_bf + h_minus_ff + thomson + rayleigh
   ```

4. **Radiative Transfer** (~2-5% of runtime)
   - **Current**: Sequential layer-by-layer integration
   - **JAX Solution**: Batched operations over atmospheric layers

### Performance-Critical Functions

#### Voigt Function Implementation
```python
@jax.jit
def voigt_profile(nu, nu0, gamma_L, gamma_G):
    """JAX-optimized Voigt profile using Faddeeva function"""
    # Normalized frequency offset
    x = (nu - nu0) / gamma_G
    y = gamma_L / gamma_G
    
    # Use JAX's complex error function
    z = x + 1j * y
    w = jax.scipy.special.wofz(z)
    
    # Voigt profile normalization
    return w.real / (gamma_G * jnp.sqrt(jnp.pi))
```

#### Broadening Calculations
```python
@jax.jit
def compute_broadening(T, ne, nH, species_id, line_data):
    """Vectorized broadening parameter calculation"""
    
    # Doppler width (thermal + microturbulent)
    doppler_width = jnp.sqrt(
        2 * constants.kboltz * T / species_mass[species_id] +
        vmic**2
    ) * line_data.wl / constants.c_cgs
    
    # Stark broadening (vectorized lookup)
    stark_gamma = stark_broadening_coeffs[species_id] * ne**(2/3)
    
    # van der Waals broadening
    vdW_gamma = (line_data.vdW_a * (T/10000)**(-line_data.vdW_b) * 
                 (nH/1e15)**(2/5))
    
    return doppler_width, stark_gamma, vdW_gamma
```

## External Dependencies → JAX Equivalents

### Scientific Computing Libraries

| Julia Package | JAX Equivalent | Purpose | Translation Notes |
|---------------|----------------|---------|------------------|
| `ForwardDiff.jl` | `jax.grad`, `jax.jacfwd` | Automatic differentiation | JAX provides superior AD performance |
| `NLsolve.jl` | `jax.scipy.optimize.fsolve` | Nonlinear equation solving | JAX autodiff improves convergence |
| `Optim.jl` | `optax`, `jax.scipy.optimize` | Optimization routines | JAX enables GPU-accelerated optimization |
| `Interpolations.jl` | `jax.scipy.interpolate` | Interpolation functions | May need custom implementations |
| `SpecialFunctions.jl` | `jax.scipy.special` | Special mathematical functions | Most functions available |

### Numerical Methods

| Julia Package | JAX Equivalent | Purpose | Translation Notes |
|---------------|----------------|---------|------------------|
| `FastGaussQuadrature.jl` | Custom implementation | Gaussian quadrature | Need to implement using JAX arrays |
| `StaticArrays.jl` | `jnp.array` | Fixed-size arrays | JAX arrays are naturally static |

### Key Algorithms Requiring JAX Implementation

1. **Voigt Function**: ✅ Available via `jax.scipy.special.wofz`
2. **Exponential Integrals**: ⚠️ Need custom implementation
3. **Chemical Equilibrium Solver**: ✅ Can use `jax.scipy.optimize.fsolve`
4. **Cubic Spline Interpolation**: ⚠️ May need custom implementation
5. **Gaussian Quadrature**: ⚠️ Need custom implementation

### Custom Implementations Needed

#### Exponential Integrals
```python
@jax.jit
def exponential_integral_2(x):
    """Second-order exponential integral E2(x)"""
    # Implement using series expansions as in Korg.jl
    return jnp.where(
        x == 0, 1.0,
        jnp.where(x < 1.1, _expint_small(x),
        jnp.where(x < 2.5, _expint_2(x),
        # ... additional ranges
        _expint_large(x)))
    )
```

#### Gaussian Quadrature
```python
@jax.jit
def gaussian_quadrature(n_points):
    """Generate Gauss-Legendre quadrature points and weights"""
    # Pre-computed coefficients for common n_points values
    # Or use JAX-compatible implementation of Golub-Welsch algorithm
    return points, weights
```

## JAX-Specific Optimizations

### Memory Layout Optimization
```python
# Structure of Arrays (SoA) for better GPU memory access
@dataclass
class LineListSoA:
    """Structure of Arrays for optimal GPU memory access"""
    wl: jnp.ndarray          # (n_lines,) contiguous wavelengths
    log_gf: jnp.ndarray      # (n_lines,) contiguous oscillator strengths
    species_id: jnp.ndarray  # (n_lines,) contiguous species IDs
    # ... other parameters
    
    @classmethod
    def from_aos(cls, lines_aos):
        """Convert Array of Structures to Structure of Arrays"""
        return cls(
            wl=jnp.array([line.wl for line in lines_aos]),
            log_gf=jnp.array([line.log_gf for line in lines_aos]),
            # ... other fields
        )
```

### Batched Operations
```python
@jax.jit
def batch_synthesis(stellar_params, wavelengths, linelist):
    """Synthesize multiple stellar spectra simultaneously"""
    
    # Vectorize over stellar parameters
    batch_synth = jax.vmap(
        single_synthesis, 
        in_axes=(0, None, None)  # batch over stellar params
    )
    
    return batch_synth(stellar_params, wavelengths, linelist)

# Usage:
stellar_params = {
    'Teff': jnp.array([5778, 6000, 5500]),
    'logg': jnp.array([4.44, 4.0, 4.5]),
    'm_H': jnp.array([0.0, -0.5, 0.2])
}
batch_flux = batch_synthesis(stellar_params, wavelengths, linelist)
```

### Custom Gradient Implementations
```python
@jax.custom_jvp
def voigt_profile_with_custom_grad(nu, nu0, gamma_L, gamma_G):
    """Voigt profile with custom gradient for numerical stability"""
    return voigt_profile_forward(nu, nu0, gamma_L, gamma_G)

@voigt_profile_with_custom_grad.defjvp
def voigt_profile_jvp(primals, tangents):
    """Custom Jacobian-vector product for Voigt profile"""
    # Implement analytic derivatives for better numerical stability
    return forward_pass, jvp_result
```

## Performance Projections

### Expected Speedups by Component

| Component | Current Runtime % | Expected JAX Speedup | Projected Runtime % |
|-----------|------------------|---------------------|-------------------|
| Line Absorption | 80-90% | 20-100x | 5-10% |
| Chemical Equilibrium | 5-10% | 2-5x | 3-5% |
| Continuum Absorption | 3-5% | 5-10x | 1-2% |
| Radiative Transfer | 2-5% | 2-5x | 1-3% |

### Overall Performance Targets
- **Conservative**: 10-20x overall speedup
- **Optimistic**: 50-100x overall speedup (with GPU acceleration)
- **Multi-GPU**: Linear scaling with additional devices

### Memory Efficiency
- **Reduced allocations**: JAX's functional programming reduces intermediate allocations
- **GPU memory**: Efficient utilization of GPU memory through proper batching
- **Streaming**: Large parameter sweeps through memory-efficient streaming

This architecture analysis provides the technical foundation for implementing a high-performance JAX-based stellar spectral synthesis package that maintains the scientific accuracy of Korg.jl while achieving significant computational improvements.