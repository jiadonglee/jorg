# Jorg Project Structure

This document outlines the planned directory structure and organization for the Jorg (JAX-based Korg) project.

## Directory Structure

```
Jorg/
├── README.md                     # Project overview and quick start
├── ROADMAP.md                    # Development roadmap and timeline
├── ARCHITECTURE.md               # Technical architecture analysis
├── project_structure.md          # This file
├── LICENSE                       # MIT License
├── setup.py                      # Package configuration
├── pyproject.toml               # Modern Python packaging
├── requirements.txt             # Core dependencies
├── requirements-dev.txt         # Development dependencies
├── requirements-gpu.txt         # GPU-specific dependencies
├── .gitignore                   # Git ignore patterns
├── .github/                     # GitHub workflows and templates
│   ├── workflows/
│   │   ├── ci.yml              # Continuous integration tests
│   │   ├── gpu-tests.yml       # GPU-specific testing
│   │   └── benchmarks.yml      # Performance benchmarking
│   └── ISSUE_TEMPLATE.md       # Issue templates
│
├── jorg/                        # Main package directory
│   ├── __init__.py             # Package initialization and main API
│   ├── version.py              # Version information
│   ├── constants.py            # Physical constants and unit conversions
│   │
│   ├── synthesis.py            # High-level synthesis interfaces
│   │   # synth(), synthesize(), batch_synth()
│   │
│   ├── continuum/              # Continuum absorption calculations
│   │   ├── __init__.py
│   │   ├── core.py            # Main total_continuum_absorption()
│   │   ├── hydrogen.py        # H I, H⁻ absorption
│   │   ├── helium.py          # He I, He II absorption
│   │   ├── metals.py          # Metal bound-free absorption
│   │   ├── scattering.py      # Thomson, Rayleigh scattering
│   │   └── positive_ions.py   # Free-free from positive ions
│   │
│   ├── lines/                  # Line absorption calculations
│   │   ├── __init__.py
│   │   ├── core.py            # Main line_absorption() function
│   │   ├── profiles.py        # Voigt, Lorentzian, Gaussian profiles
│   │   ├── broadening.py      # Doppler, Stark, van der Waals broadening
│   │   ├── hydrogen_lines.py  # Special hydrogen line treatment
│   │   └── molecular.py       # Molecular line absorption
│   │
│   ├── rt/                     # Radiative transfer
│   │   ├── __init__.py
│   │   ├── core.py            # Main radiative_transfer() function
│   │   ├── schemes.py         # Linear, Bezier integration schemes
│   │   ├── geometry.py        # Ray calculation for different geometries
│   │   └── intensity.py       # Intensity computation methods
│   │
│   ├── statmech/               # Statistical mechanics and equilibrium
│   │   ├── __init__.py
│   │   ├── equilibrium.py     # Chemical equilibrium solver
│   │   ├── partition.py       # Partition function calculations
│   │   ├── ionization.py      # Saha equation and ionization balance
│   │   └── molecular.py       # Molecular equilibrium
│   │
│   ├── atmosphere/             # Atmospheric model handling
│   │   ├── __init__.py
│   │   ├── models.py          # Atmosphere data structures
│   │   ├── interpolation.py   # MARCS interpolation
│   │   ├── io.py              # Atmosphere file I/O
│   │   └── structure.py       # Atmospheric layer calculations
│   │
│   ├── data/                   # Data handling and species management
│   │   ├── __init__.py
│   │   ├── loaders.py         # File I/O utilities
│   │   ├── species.py         # Species definitions and registry
│   │   ├── linelists.py       # Line list handling
│   │   ├── abundances.py      # Abundance patterns
│   │   └── atomic_data.py     # Atomic/molecular data management
│   │
│   ├── utils/                  # Utility functions
│   │   ├── __init__.py
│   │   ├── math.py            # Mathematical functions (Voigt, expint, etc.)
│   │   ├── interpolation.py   # Interpolation routines
│   │   ├── units.py           # Unit conversions
│   │   ├── wavelengths.py     # Wavelength/frequency handling
│   │   └── validation.py      # Input validation and error checking
│   │
│   └── optimization/           # JAX-specific optimizations
│       ├── __init__.py
│       ├── batching.py        # Batched operations and vmapping
│       ├── gradients.py       # Custom gradient implementations
│       ├── memory.py          # Memory optimization utilities
│       └── compilation.py     # JIT compilation helpers
│
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── conftest.py            # Pytest configuration and fixtures
│   ├── test_synthesis.py      # High-level synthesis tests
│   ├── test_continuum/        # Continuum absorption tests
│   │   ├── test_hydrogen.py
│   │   ├── test_helium.py
│   │   ├── test_metals.py
│   │   └── test_scattering.py
│   ├── test_lines/            # Line absorption tests
│   │   ├── test_profiles.py
│   │   ├── test_broadening.py
│   │   └── test_hydrogen_lines.py
│   ├── test_rt/               # Radiative transfer tests
│   │   ├── test_transfer.py
│   │   └── test_schemes.py
│   ├── test_statmech/         # Statistical mechanics tests
│   │   ├── test_equilibrium.py
│   │   └── test_partition.py
│   ├── test_utils/            # Utility function tests
│   │   ├── test_math.py
│   │   └── test_interpolation.py
│   ├── reference/             # Reference data for validation
│   │   ├── korg_outputs/      # Reference outputs from Korg.jl
│   │   └── test_spectra/      # Test stellar spectra
│   └── integration/           # Full integration tests
│       ├── test_accuracy.py   # Accuracy vs Korg.jl
│       ├── test_performance.py # Performance benchmarks
│       └── test_physics.py    # Physical sanity checks
│
├── benchmarks/                 # Performance benchmarking
│   ├── __init__.py
│   ├── run_benchmarks.py      # Main benchmark runner
│   ├── synthesis_bench.py     # Synthesis performance tests
│   ├── memory_bench.py        # Memory usage benchmarks
│   ├── gpu_scaling.py         # Multi-GPU scaling tests
│   └── comparison_plots.py    # Visualization of benchmark results
│
├── examples/                   # Usage examples and tutorials
│   ├── __init__.py
│   ├── basic_synthesis.py     # Simple synthesis example
│   ├── parameter_fitting.py   # Gradient-based fitting example
│   ├── batch_processing.py    # Batch synthesis example
│   ├── gpu_acceleration.py    # GPU usage examples
│   ├── custom_atmospheres.py  # Custom atmosphere models
│   └── notebooks/             # Jupyter notebooks
│       ├── tutorial.ipynb     # Getting started tutorial
│       ├── performance.ipynb  # Performance comparison
│       └── fitting.ipynb      # Parameter fitting examples
│
├── docs/                       # Documentation
│   ├── conf.py                # Sphinx configuration
│   ├── index.rst             # Documentation main page
│   ├── installation.rst      # Installation instructions
│   ├── quickstart.rst        # Quick start guide
│   ├── api/                  # API documentation
│   │   ├── synthesis.rst
│   │   ├── continuum.rst
│   │   ├── lines.rst
│   │   ├── rt.rst
│   │   └── utils.rst
│   ├── tutorials/            # Detailed tutorials
│   │   ├── basic_usage.rst
│   │   ├── advanced_features.rst
│   │   └── performance_tips.rst
│   ├── theory/               # Theoretical background
│   │   ├── stellar_atmospheres.rst
│   │   ├── radiative_transfer.rst
│   │   └── line_formation.rst
│   └── development/          # Development documentation
│       ├── contributing.rst
│       ├── architecture.rst
│       └── testing.rst
│
├── data/                       # Data files (not in git, downloaded separately)
│   ├── atmospheres/           # MARCS atmosphere models
│   ├── linelists/            # Atomic and molecular line lists
│   ├── atomic_data/          # Partition functions, ionization energies
│   └── molecular_data/       # Molecular cross-sections
│
├── scripts/                   # Utility scripts
│   ├── setup_data.py         # Download and setup data files
│   ├── convert_korg_data.py  # Convert Korg.jl data to JAX format
│   ├── generate_tests.py     # Generate reference test data
│   └── performance_profiling.py # Performance profiling utilities
│
└── tools/                     # Development tools
    ├── code_generators/       # Code generation utilities
    ├── data_converters/       # Data format conversion tools
    └── validation/            # Validation against Korg.jl
        ├── compare_outputs.py
        └── generate_references.py
```

## Module Organization Principles

### 1. Physical Separation
Each major physical process gets its own module:
- `continuum/`: All continuum opacity sources
- `lines/`: All line absorption calculations  
- `rt/`: Radiative transfer solution
- `statmech/`: Chemical and ionization equilibrium

### 2. Computational Separation
JAX-specific optimizations are separated:
- `optimization/`: JAX compilation, batching, custom gradients
- Core physics modules focus on algorithms, not optimization

### 3. Data Flow Separation
- `data/`: Input data handling and species management
- `atmosphere/`: Atmospheric model structures and interpolation
- `utils/`: General utility functions

### 4. Testing Strategy
- Unit tests for each module
- Integration tests for complete synthesis
- Reference tests against Korg.jl
- Performance benchmarks

## Key Design Decisions

### 1. Flat Module Structure
Avoid deep nesting to keep imports simple:
```python
from jorg import synth
from jorg.continuum import total_continuum_absorption
from jorg.lines import line_absorption
```

### 2. Functional Programming
JAX encourages functional programming:
- Pure functions without side effects
- Immutable data structures
- Clear separation of computation and state

### 3. Vectorization-First
Design all functions for vectorized operation:
```python
# Single spectrum
flux = synth(Teff=5778, logg=4.44, m_H=0.0, ...)

# Multiple spectra (same function, vectorized inputs)
flux = synth(Teff=[5778, 6000, 5500], logg=[4.44, 4.0, 4.5], ...)
```

### 4. Gradients as First-Class Feature
Enable automatic differentiation throughout:
```python
# Any synthesis function can be differentiated
grad_synth = jax.grad(synth, argnums=(0, 1, 2))  # w.r.t. Teff, logg, m_H
```

## Development Workflow

### 1. Module Development Order
1. `utils/math.py` - Core mathematical functions
2. `data/` - Data structures and loading
3. `continuum/` - Continuum absorption (well-defined physics)
4. `lines/` - Line absorption (most complex, highest impact)
5. `rt/` - Radiative transfer (integration of above)
6. `synthesis.py` - High-level interfaces

### 2. Testing Strategy
- Write tests alongside each module
- Use reference data from Korg.jl for validation
- Continuous integration with CPU and GPU testing
- Performance regression testing

### 3. Documentation
- Docstrings for all public functions
- Sphinx-generated API documentation
- Jupyter notebook tutorials
- Performance comparison documentation

This structure provides a solid foundation for the Jorg project, emphasizing modularity, testability, and JAX optimization while maintaining scientific accuracy.