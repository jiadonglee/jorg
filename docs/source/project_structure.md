# Jorg Project Structure Documentation

This document provides a comprehensive overview of the Jorg project organization, including the rationale behind the structure, dependency relationships, and development workflow.

## 📁 Directory Structure

```
Jorg/
├── 📄 README.md                          # Project overview and quick start
├── 📄 LICENSE                            # MIT license
├── 📄 pyproject.toml                     # Modern Python packaging configuration
├── 📄 setup.py                           # Setuptools compatibility
├── 📄 .gitignore                         # Git ignore patterns
├── 📄 PROJECT_STRUCTURE.md               # This file
│
├── 📁 src/jorg/                          # 🎯 Main package source code
│   ├── 📄 __init__.py                    # Package initialization and main API
│   ├── 📄 synthesis.py                   # High-level synthesis interface
│   ├── 📄 constants.py                   # Physical constants and unit conversions
│   │
│   ├── 📁 continuum/                     # Continuum absorption calculations
│   │   ├── 📄 __init__.py                # Module exports
│   │   ├── 📄 core.py                    # Main total_continuum_absorption()
│   │   ├── 📄 hydrogen.py                # H I, H⁻ bound-free and free-free
│   │   ├── 📄 helium.py                  # He I, He II absorption
│   │   ├── 📄 scattering.py              # Thomson, Rayleigh scattering
│   │   └── 📄 utils.py                   # Continuum-specific utilities
│   │
│   ├── 📁 lines/                         # Line absorption calculations
│   │   ├── 📄 __init__.py                # Module exports
│   │   ├── 📄 core.py                    # Main line_absorption() functions
│   │   ├── 📄 hydrogen_lines.py          # 🌟 Sophisticated hydrogen line treatment
│   │   ├── 📄 hydrogen_lines_simple.py   # Simplified hydrogen lines (Balmer focus)
│   │   ├── 📄 linelist.py                # Linelist reading and parsing
│   │   ├── 📄 opacity.py                 # Line opacity calculations
│   │   ├── 📄 profiles.py                # Voigt, Gaussian, Lorentzian profiles
│   │   ├── 📄 broadening.py              # Doppler, van der Waals, Stark broadening
│   │   ├── 📄 species.py                 # Species identification and handling
│   │   └── 📄 utils.py                   # Line-specific utilities
│   │
│   ├── 📁 utils/                         # General utility functions
│   │   ├── 📄 __init__.py                # Utility exports
│   │   ├── 📄 math.py                    # Mathematical functions and safe operations
│   │   └── 📄 wavelength_utils.py        # Wavelength/frequency conversions
│   │
│   └── 📁 data/                          # Package data files
│       └── 📄 mclaughlin_hminus.json     # H⁻ absorption data
│
├── 📁 tests/                             # 🧪 Test suite
│   ├── 📄 __init__.py                    # Test package initialization
│   ├── 📄 conftest.py                    # Pytest configuration and fixtures
│   │
│   ├── 📁 unit/                          # Unit tests for individual modules
│   │   ├── 📄 __init__.py
│   │   ├── 📁 continuum/                 # Continuum module tests
│   │   │   ├── 📄 __init__.py
│   │   │   └── 📄 test_continuum.py
│   │   ├── 📁 lines/                     # Line module tests
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 test_lines.py
│   │   │   ├── 📄 test_linelist_reading.py
│   │   │   └── 📄 test_voigt_accuracy.py
│   │   └── 📁 utils/                     # Utility tests
│   │       └── 📄 __init__.py
│   │
│   ├── 📁 integration/                   # Integration and end-to-end tests
│   │   ├── 📄 __init__.py
│   │   ├── 📄 test_accuracy.py           # Accuracy vs Korg.jl comparison
│   │   ├── 📄 test_korg_jorg_comparison.py # Detailed comparison tests
│   │   └── 📄 test_plotting.py           # Visualization tests
│   │
│   └── 📁 fixtures/                      # Test data and reference outputs
│       ├── 📄 __init__.py
│       ├── 📄 generate_korg_reference.py # Reference data generation
│       └── 📁 reference_data/            # Korg.jl reference outputs
│           ├── 📄 korg_detailed_reference.json
│           ├── 📄 korg_reference_data.json
│           ├── 📄 korg_reference_voigt.json
│           └── 📄 lines_comparison_summary.json
│
├── 📁 examples/                          # 📚 Usage examples and tutorials
│   └── 📄 basic_linelist_usage.py        # Linelist reading example
│
├── 📁 benchmarks/                        # ⚡ Performance benchmarking
│   ├── 📄 speed_comparison.py            # Jorg vs Korg speed comparison
│   └── 📄 linelist_reading_benchmark.py  # Linelist I/O performance
│
├── 📁 scripts/                           # 🔧 Development and utility scripts
│   ├── 📄 debug_vald_parser.py           # VALD parser debugging
│   ├── 📄 generate_korg_reference.jl     # Reference data generation (Julia)
│   ├── 📄 korg_linelist_benchmark.jl     # Korg benchmarking (Julia)
│   └── 📄 simple_korg_benchmark.jl       # Simple Korg timing (Julia)
│
├── 📁 docs/                              # 📖 Documentation
│   ├── 📁 source/                        # Sphinx source files
│   │   ├── 📄 ARCHITECTURE.md
│   │   ├── 📄 LINES_IMPLEMENTATION_SUMMARY.md
│   │   ├── 📄 ORGANIZATION_SUMMARY.md
│   │   ├── 📄 project_structure.md
│   │   ├── 📄 ROADMAP.md
│   │   └── 📄 SPEED_TEST_RESULTS.md
│   ├── 📁 api/                           # API documentation
│   └── 📁 tutorials/                     # User tutorials
│
└── 📁 comparison_tests/                  # 🔬 Legacy comparison tests (to be migrated)
    ├── 📄 README.md
    ├── 📄 component_wise_comparison.py
    ├── 📄 create_final_comparison_plots.py
    └── 📄 ... (other comparison scripts)
```

## 🏗️ Architectural Principles

### 1. **Modular Design**
- **Physical separation**: Each major physical process (continuum, lines, radiative transfer) has its own module
- **Computational separation**: JAX-specific optimizations are isolated from physics algorithms
- **Clear interfaces**: Well-defined APIs between modules

### 2. **Dependency Hierarchy**
```
🌳 Dependency Tree:
   utils/ ← Base utilities (no internal dependencies)
      ↑
   constants.py ← Physical constants
      ↑
   continuum/, lines/ ← Physics modules (depend on utils, constants)
      ↑
   synthesis.py ← High-level interface (depends on continuum, lines)
      ↑
   tests/ ← Test everything (depends on all modules)
```

### 3. **Source Layout (src/)**
Following Python best practices with the `src/` layout:
- **Prevents accidental imports** from development directory
- **Clear separation** between source and tests
- **Better packaging** and distribution

### 4. **Test Organization**
- **Unit tests**: Test individual functions/classes in isolation
- **Integration tests**: Test module interactions and end-to-end workflows
- **Fixtures**: Shared test data and reference outputs
- **Separate by scope**: Unit vs integration vs performance tests

## 🔄 Data Flow and Module Interactions

### High-Level Synthesis Flow
```
📊 synthesis.py
    ↓ calls
🌡️ continuum/core.py ← continuum/hydrogen.py, helium.py, scattering.py
    ↓ combines with
📊 lines/core.py ← lines/linelist.py, opacity.py, profiles.py
    ↓ uses
🔧 utils/math.py, wavelength_utils.py
    ↓ references
📐 constants.py
```

### Import Strategy
```python
# High-level user imports
from jorg import synth, synthesize

# Module-level imports
from jorg.continuum import total_continuum_absorption
from jorg.lines import total_line_absorption, LineList

# Utility imports
from jorg.utils import air_to_vacuum, voigt_hjerting
from jorg.constants import SPEED_OF_LIGHT, BOLTZMANN_K
```

## 🧪 Testing Strategy

### Test Categories
1. **Unit Tests** (`tests/unit/`)
   - Individual function testing
   - Mock external dependencies
   - Fast execution (<1s per test)

2. **Integration Tests** (`tests/integration/`)
   - Multi-module interactions
   - Accuracy vs reference implementations
   - End-to-end synthesis workflows

3. **Performance Tests** (`benchmarks/`)
   - Speed comparisons vs Korg.jl
   - Memory usage profiling
   - Scaling analysis

### Test Data Management
- **Small test data**: Included in repository
- **Large reference data**: Downloaded separately or generated on-demand
- **Reproducible**: Version-controlled reference outputs

## 📦 Package Configuration

### Modern Python Packaging
- **pyproject.toml**: Primary configuration (PEP 518/621)
- **setup.py**: Minimal compatibility shim
- **Optional dependencies**: GPU, development, documentation extras

### Key Features
- **Automatic version discovery**: From git tags or explicit version
- **Entry points**: Command-line interfaces (future)
- **Data files**: Included package data (constants, reference spectra)
- **Comprehensive metadata**: For PyPI publication

## 🔧 Development Workflow

### Setup
```bash
# Clone and install in development mode
git clone https://github.com/jorg-project/jorg.git
cd jorg
pip install -e ".[dev,docs,gpu]"
```

### Common Tasks
```bash
# Run tests
pytest                          # All tests
pytest tests/unit/             # Unit tests only
pytest -m "not slow"           # Skip slow tests
pytest --cov=jorg             # With coverage

# Code quality
black src/ tests/              # Format code
flake8 src/ tests/             # Lint
mypy src/                      # Type checking

# Documentation
cd docs/ && make html          # Build docs
```

### Release Process
1. **Version bump**: Update version in `src/jorg/__init__.py`
2. **Testing**: Full test suite including slow tests
3. **Documentation**: Update docs and changelog
4. **Build**: `python -m build`
5. **Publish**: `twine upload dist/*`

## 🎯 Future Enhancements

### Planned Additions
- **Radiative Transfer Module** (`src/jorg/rt/`)
- **Atmosphere Interpolation** (`src/jorg/atmosphere/`)
- **Statistical Mechanics** (`src/jorg/statmech/`)
- **Optimization Utilities** (`src/jorg/optimization/`)

### Infrastructure Improvements
- **Continuous Integration**: GitHub Actions with CPU/GPU tests
- **Documentation**: Sphinx with auto-generated API docs
- **Performance Monitoring**: Automated benchmark tracking
- **Example Gallery**: Jupyter notebook tutorials

## 📚 Documentation Structure

### User Documentation
- **README.md**: Quick start and overview
- **Installation Guide**: Detailed setup instructions
- **Tutorials**: Step-by-step examples
- **API Reference**: Auto-generated from docstrings

### Developer Documentation
- **This File**: Project structure and architecture
- **Contributing Guide**: Code standards and workflow
- **Architecture Documents**: Design decisions and rationale
- **Performance Analysis**: Benchmark results and optimization notes

---

This structure provides a solid foundation for the Jorg project, emphasizing:
- ✅ **Modularity** for maintainable code
- ✅ **Testability** with comprehensive test coverage  
- ✅ **Performance** through JAX optimization
- ✅ **Usability** with clear APIs and documentation
- ✅ **Extensibility** for future enhancements

The organization follows Python best practices and enables efficient development, testing, and deployment of the stellar spectral synthesis package.