# Korg-JAX Translation Roadmap

**Jorg**: JAX-based stellar spectral synthesis (Korg.jl → Python JAX translation)

## Overview

This document outlines the comprehensive roadmap for translating Korg.jl (Julia) to Jorg (Python JAX), maintaining scientific accuracy while achieving significant performance improvements through GPU acceleration and automatic differentiation.

## Phase 1: Foundation & Core Infrastructure (4-6 weeks)

### 1.1 Data Structures & Type System

**Objective**: Establish JAX-compatible data foundations

#### Tasks:
- **JAX PyTrees for complex data**: Replace Julia structs with JAX-compatible nested dictionaries/dataclasses
- **Atmospheric models**: Convert `PlanarAtmosphereLayer` to flat arrays with consistent indexing
- **Species handling**: Create efficient integer-based species encoding system
- **Wavelength grids**: Implement `jnp.linspace`-based wavelength handling with automatic frequency conversion

#### Deliverables:
```python
# Target data structures
@dataclass
class AtmosphereLayer:
    tau_5000: jnp.ndarray       # optical depth at 5000 Å
    z: jnp.ndarray             # height (cm)  
    temp: jnp.ndarray          # temperature (K)
    electron_density: jnp.ndarray  # cm^-3
    number_density: jnp.ndarray    # cm^-3

class Species:
    """Integer-encoded species for efficient JAX operations"""
    species_id: int
    charge: int
    
class Wavelengths:
    """JAX-optimized wavelength/frequency handling"""
    wl: jnp.ndarray            # wavelengths (Å)
    freq: jnp.ndarray          # frequencies (Hz)
```

### 1.2 Mathematical Foundations

**Objective**: Port core mathematical functions to JAX

#### Tasks:
- **Voigt function**: Implement using JAX-compatible complex error function (`jax.scipy.special.wofz`)
- **Special functions**: Port exponential integrals (E1, E2) using series expansions
- **Interpolation**: Create JAX-native cubic spline interpolation for continuum opacity
- **Chemical equilibrium solver**: Implement Newton-Raphson using `jax.scipy.optimize`

#### Deliverables:
```python
# Core mathematical functions
@jax.jit
def voigt_profile(nu, nu0, gamma_L, gamma_G):
    """JAX-optimized Voigt profile calculation"""
    
@jax.jit  
def exponential_integral_2(x):
    """Second-order exponential integral for radiative transfer"""
    
@jax.jit
def chemical_equilibrium_solver(T, P, abundances):
    """Newton-Raphson solver for chemical equilibrium"""
```

### 1.3 Data Loading & Preprocessing

**Objective**: Efficient data pipeline for JAX arrays

#### Tasks:
- **Atmosphere model reader**: Convert MARCS files to preprocessed JAX arrays
- **Linelist parser**: Create efficient numpy/JAX-compatible line data format
- **Atomic data**: Port partition functions, ionization energies to lookup tables

#### Deliverables:
- `jorg/data/loaders.py`: Data loading utilities
- `jorg/data/preprocessors.py`: Format conversion tools
- Preprocessed data files in HDF5/NPZ format

## Phase 2: Core Computational Kernels (6-8 weeks)

### 2.1 Continuum Absorption (`jorg.continuum`)

**Priority: High (foundational)**

#### Target API:
```python
alpha_continuum = total_continuum_absorption(
    frequencies,  # jnp.array shape (n_freq,)
    temperature,  # float
    electron_density,  # float  
    number_densities,  # dict -> jnp.array
)
```

#### Tasks:
- Port H⁻ bound-free/free-free calculations
- Implement metal bound-free opacity (vectorized lookups)
- Add Thomson/Rayleigh scattering
- **JAX optimization**: Use `vmap` for wavelength-parallel computation

#### Performance Target:
- 5-10x speedup over Julia version through vectorization

### 2.2 Line Absorption (`jorg.lines`)

**Priority: Critical (performance bottleneck - 80-90% of runtime)**

#### Target API:
```python
alpha_lines = line_absorption(
    frequencies,     # (n_freq,) 
    linelist,       # structured array
    atmosphere,     # dict of arrays
    abundances,     # (n_species,)
)
```

#### Tasks:
- **Vectorized Voigt profiles**: Compute all lines simultaneously using `vmap`
- **Broadening calculations**: Doppler, Stark, van der Waals in parallel
- **Memory optimization**: Chunked processing for large linelists
- **Key optimization**: Replace nested loops with tensor operations

#### Performance Target:
- 20-100x speedup through GPU acceleration and vectorization

### 2.3 Statistical Mechanics (`jorg.statmech`)

#### Target API:
```python
electron_density, number_densities = chemical_equilibrium(
    temperature, pressure, abundances, ionization_energies
)
```

#### Tasks:
- Newton-Raphson solver using `jax.scipy.optimize.fsolve`
- Saha equation vectorization  
- Molecular equilibrium constants
- **JAX benefit**: Automatic differentiation for robust convergence

## Phase 3: Radiative Transfer & Integration (4-5 weeks)

### 3.1 Radiative Transfer (`jorg.rt`)

#### Target API:
```python
flux, intensity = radiative_transfer(
    absorption_coeff,  # (n_layers, n_freq)
    source_function,   # (n_layers, n_freq)  
    atmosphere,        # atmospheric structure
    mu_points=5        # angular quadrature
)
```

#### Tasks:
- Linear interpolation scheme (primary method)
- Exponential integral optimization for plane-parallel case
- **JAX optimization**: Batched operations over wavelengths

### 3.2 High-Level Interface (`jorg.synthesis`)

#### Target API (matching original Korg):
```python
wavelengths, flux, continuum = synth(
    Teff=5778, logg=4.44, m_H=0.0,
    wavelengths=(5000, 6000),
    linelist=default_linelist
)
```

#### Tasks:
- User-friendly interface matching Korg.jl API
- Parameter validation and error handling
- Documentation and examples

## Phase 4: Performance Optimization (3-4 weeks)

### 4.1 JAX-Specific Optimizations

#### Tasks:
- **JIT compilation**: Add `@jax.jit` to all computational kernels
- **Vectorization**: Use `vmap` for parameter sweeps (temperature, metallicity grids)
- **Parallelization**: Implement `pmap` for multi-GPU synthesis
- **Memory management**: Optimize array chunking and intermediate allocations

#### Performance Targets:
- 10-50x overall speedup vs Julia multithreaded version
- Linear scaling with additional GPUs

### 4.2 Advanced Features

#### Tasks:
- **Automatic differentiation**: Enable gradient-based fitting
- **Batched synthesis**: Compute multiple stellar spectra simultaneously
- **Custom derivatives**: Analytic gradients for key physical functions

#### Target APIs:
```python
# Gradient-enabled synthesis
grad_synth = jax.grad(synth, argnums=(0, 1, 2))  # w.r.t. Teff, logg, m_H

# Batched parameter sweeps
batch_synth = jax.vmap(synth, in_axes=(0, 0, 0, None, None))
```

## Phase 5: Testing & Validation (3-4 weeks)

### 5.1 Accuracy Validation

#### Tasks:
- **Reference comparisons**: Test against original Korg.jl outputs
- **Physical checks**: Verify continuum levels, line depths, equivalent widths
- **Edge cases**: Extreme stellar parameters, sparse linelists

#### Success Criteria:
- <0.1% RMS difference in synthetic spectra vs Korg.jl
- All physical sanity checks pass
- Robust performance across parameter space

### 5.2 Performance Benchmarking

#### Tasks:
- **Speed comparisons**: JAX vs Julia multithreading
- **Memory profiling**: Optimize large wavelength ranges
- **Scaling tests**: Multi-GPU performance

## JAX Ecosystem Dependencies

### Core Scientific Stack
```python
jax>=0.4.0              # Core JAX functionality
jax[cuda]               # GPU support  
numpy>=1.24.0           # Array operations
scipy>=1.10.0           # Special functions, optimization
```

### Specialized Libraries
```python
chex                    # Type checking, testing utilities
optax                   # Advanced optimization algorithms  
flax                    # Neural network components (if ML features)
jaxlib                  # Compiled JAX operations
```

### Data & I/O
```python
h5py                    # HDF5 file reading
pandas                  # Tabular data processing
astropy                 # Astronomical utilities
```

## Implementation Strategy

### 1. Modular Translation Approach
1. Start with **continuum absorption** (well-defined, medium complexity)
2. Move to **line absorption** (highest impact on performance)
3. Integrate with **radiative transfer** (puts everything together)
4. Add high-level interfaces last

### 2. Performance-First Design
- **Memory layout**: Structure arrays for optimal GPU memory access
- **Batch operations**: Design for parameter sweeps from the start
- **Compilation boundaries**: Minimize JAX recompilation overhead

### 3. Validation Strategy
- **Unit tests**: Each module against reference calculations
- **Integration tests**: Full synthesis comparisons  
- **Physical tests**: Known stellar spectra reproduction

## Expected Performance Gains

### Computational Improvements
- **10-50x speedup** from GPU acceleration (line absorption)
- **2-5x speedup** from JIT compilation optimization
- **Linear scaling** with additional GPUs using `pmap`

### Development Benefits
- **Automatic differentiation**: Enable gradient-based parameter fitting
- **Batched operations**: Efficient parameter space exploration
- **Reproducibility**: Deterministic computation across platforms

## Project Structure

```
Jorg/
├── ROADMAP.md                 # This document
├── ARCHITECTURE.md            # Detailed architecture analysis
├── setup.py                  # Package configuration
├── requirements.txt           # Dependencies
├── jorg/
│   ├── __init__.py
│   ├── synthesis.py           # High-level interfaces (synth, synthesize)
│   ├── continuum/            # Continuum absorption
│   │   ├── __init__.py
│   │   ├── hydrogen.py       # H, H⁻ absorption
│   │   ├── helium.py         # He absorption  
│   │   ├── metals.py         # Metal bound-free
│   │   └── scattering.py     # Thomson, Rayleigh
│   ├── lines/                # Line absorption
│   │   ├── __init__.py
│   │   ├── profiles.py       # Voigt, Lorentzian profiles
│   │   ├── broadening.py     # Doppler, Stark, vdW
│   │   └── hydrogen_lines.py # Special H line treatment
│   ├── rt/                   # Radiative transfer
│   │   ├── __init__.py
│   │   ├── transfer.py       # Core RT solver
│   │   └── schemes.py        # Integration schemes
│   ├── statmech/             # Statistical mechanics
│   │   ├── __init__.py
│   │   ├── equilibrium.py    # Chemical equilibrium
│   │   └── partition.py      # Partition functions
│   ├── data/                 # Data handling
│   │   ├── __init__.py
│   │   ├── loaders.py        # File I/O
│   │   ├── species.py        # Species definitions
│   │   └── constants.py      # Physical constants
│   └── utils/                # Utilities
│       ├── __init__.py
│       ├── math.py           # Mathematical functions
│       └── interpolation.py  # Interpolation routines
├── tests/                    # Test suite
├── benchmarks/               # Performance benchmarks
├── examples/                 # Usage examples
└── docs/                     # Documentation
```

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| 1: Foundation | 4-6 weeks | Data structures, math functions, data pipeline |
| 2: Core Kernels | 6-8 weeks | Continuum/line absorption, statistical mechanics |
| 3: Integration | 4-5 weeks | Radiative transfer, high-level interfaces |
| 4: Optimization | 3-4 weeks | Performance tuning, advanced features |
| 5: Validation | 3-4 weeks | Testing, benchmarking, documentation |
| **Total** | **20-27 weeks** | **Production-ready Jorg package** |

This roadmap provides a systematic path from Korg.jl to Jorg while preserving scientific accuracy and achieving significant performance improvements through JAX's GPU acceleration and automatic differentiation capabilities.