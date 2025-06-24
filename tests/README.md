# Jorg Tests Directory

This directory contains all test scripts, comparison tools, and debugging utilities for the Jorg project.

## Directory Structure

```
tests/
├── README.md                           # This file
├── __init__.py                         # Python package init
├── test_continuum.py                   # Unit tests for continuum absorption  
├── test_lines.py                       # Unit tests for line absorption
├── test_voigt_accuracy.py              # Voigt-Hjerting accuracy tests
│
├── comparison_script.py                # Basic Korg vs Jorg comparison
├── simple_comparison.py                # Simple continuum demonstration
├── plot_line_comparison.py             # Comprehensive line data plots
├── korg_jorg_comparison.py             # Detailed continuum comparison
├── corrected_comparison.py             # Corrected comparison analysis
├── final_korg_jorg_comparison.py       # Final comparison with performance
│
├── debug_comparison.py                 # Debug comparison differences
├── debug_components.py                 # Debug individual components
├── debug_cross_sections.py             # Debug cross-section calculations
├── debug_ordering.py                   # Debug wavelength ordering
├── final_debug.py                      # Final debugging utilities
│
└── generate_korg_reference.py          # Generate reference data from Korg.jl
```

## Test Categories

### 1. Unit Tests
- **test_continuum.py**: Tests for continuum absorption components
- **test_lines.py**: Tests for line absorption components  
- **test_voigt_accuracy.py**: Voigt-Hjerting function accuracy validation

### 2. Comparison Tests
- **comparison_script.py**: Basic comparison framework
- **simple_comparison.py**: Solar-like conditions demonstration
- **plot_line_comparison.py**: Publication-quality line comparison plots
- **korg_jorg_comparison.py**: Comprehensive continuum analysis
- **final_korg_jorg_comparison.py**: Final validation with performance metrics

### 3. Debug Tools
- **debug_comparison.py**: Investigate accuracy differences
- **debug_components.py**: Component-wise analysis
- **debug_cross_sections.py**: Cross-section debugging
- **debug_ordering.py**: Wavelength ordering validation
- **final_debug.py**: Complete debugging pipeline

### 4. Reference Data Generation
- **generate_korg_reference.py**: Creates Julia scripts to generate reference data from Korg.jl

## Usage

### Running Unit Tests

```bash
# Run all tests
cd tests/
python test_continuum.py
python test_lines.py
python test_voigt_accuracy.py

# Or use pytest if available
pytest test_*.py
```

### Running Comparison Tests

```bash
# Basic comparison
python simple_comparison.py

# Comprehensive line comparison with plots
python plot_line_comparison.py

# Detailed continuum analysis
python final_korg_jorg_comparison.py
```

### Generating Reference Data

```bash
# Generate Julia script for reference data
python generate_korg_reference.py

# Run the generated Julia script (from parent directory)
cd ..
julia generate_korg_reference.jl
```

## Test Outputs

All test figures are saved to `../test_fig/` directory:

- **accuracy_summary.png**: Overall accuracy assessment
- **voigt_hjerting_comparison.png**: Voigt function validation
- **harris_series_comparison.png**: Harris series accuracy
- **line_profile_comparison.png**: Line profile validation
- **jorg_continuum_example.png**: Basic continuum example
- **korg_jorg_comparison.png**: Continuum comparison plots

## Dependencies

Test scripts require:
- JAX and JAX-numpy
- NumPy  
- Matplotlib
- JSON (built-in)
- pathlib (built-in)

Reference data generation requires:
- Julia with Korg.jl package installed
- JSON.jl package for Julia

## Import Path Handling

All test scripts automatically add the parent directory to the Python path:
```python
sys.path.insert(0, str(Path(__file__).parent.parent))
```

This allows importing the `jorg` package from the parent directory.

## Test Results Summary

### Accuracy Achieved
- **Voigt-Hjerting Function**: 1.9e-16 max error (machine precision)
- **Harris Series**: 1.9e-16 max error (machine precision)  
- **Line Profiles**: 2.7e-16 max error (machine precision)
- **Continuum Absorption**: ~1% typical error vs Korg.jl

### Status
✅ All core algorithms implemented and validated  
✅ Machine precision accuracy for line absorption  
✅ JAX compatibility and optimization ready  
✅ Comprehensive test coverage  

## Next Steps

1. **Integration Tests**: End-to-end stellar synthesis validation
2. **Performance Tests**: GPU acceleration benchmarking  
3. **Parameter Sweeps**: Testing across stellar parameter grids
4. **Regression Tests**: Continuous validation framework