# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Korg.jl is a Julia package for 1D LTE (Local Thermodynamic Equilibrium) stellar spectral synthesis. It generates synthetic stellar spectra by modeling radiative transfer through stellar atmospheres, including atomic and molecular line absorption, continuum absorption, and various physical processes.

## Development Commands

### Testing
- `julia --project=. -e "using Pkg; Pkg.test()"` - Run the full test suite
- `julia --project=. test/runtests.jl` - Alternative way to run tests
- Tests are automatically run with 4 threads via GitHub Actions
- Individual test files can be run: `julia --project=. test/[filename].jl`

### Documentation
- `julia --project=docs docs/make.jl` - Build documentation locally
- Documentation is automatically built and deployed via GitHub Actions

### Package Management
- `julia --project=. -e "using Pkg; Pkg.instantiate()"` - Install dependencies
- Standard Julia package development workflow applies

## Code Architecture

### Core Synthesis Functions
- **`synth()`** (`src/synth.jl`): High-level, user-friendly interface that returns `(wavelengths, flux, continuum)`
- **`synthesize()`** (`src/synthesize.jl`): Lower-level interface returning detailed `SynthesisResult` with full diagnostic information

### Key Components
- **Radiative Transfer** (`src/RadiativeTransfer/`): Formal solution of radiative transfer equation
- **Continuum Absorption** (`src/ContinuumAbsorption/`): Bound-free, free-free, and scattering opacity calculations
- **Line Absorption** (`src/line_absorption.jl`, `src/hydrogen_line_absorption.jl`): Atomic and molecular line profiles
- **Statistical Mechanics** (`src/statmech.jl`): Chemical equilibrium, partition functions, ionization balance
- **Model Atmospheres** (`src/atmosphere.jl`): Reading and interpolating stellar atmosphere models
- **Linelists** (`src/linelist.jl`): Parsing and managing atomic/molecular line data

### Data Handling
- **Wavelengths** (`src/wavelengths.jl`): Custom type for handling wavelength ranges and air/vacuum conversions
- **Species** (`src/species.jl`): Chemical species representation and isotopic data
- **Abundances** (`src/abundances.jl`): Stellar abundance patterns and solar scaling

### Fitting and Analysis
- **Fit Module** (`src/Fit/`): Parameter fitting via equivalent widths or full spectral synthesis
- **Utils** (`src/utils.jl`): Line spread functions, rotational broadening, wavelength conversions

## Physical Considerations

### Multithreading
- Korg leverages Julia's multithreading for line opacity calculations
- Launch Julia with `-t N` or set `JULIA_NUM_THREADS=N` for optimal performance
- Line absorption calculations dominate synthesis runtime

### Wavelength Limitations
- Minimum wavelength: 1300 Å (due to Rayleigh scattering implementation)
- Default wavelength range: 5000-6000 Å
- Long wavelengths (>15000 Å): Consider `use_MHD_for_hydrogen_lines=false` for APOGEE-like applications

### Performance Parameters
- `line_cutoff_threshold`: Controls line profile truncation (default: 3e-4)
- `line_buffer`: Distance from synthesis window to include lines (default: 10 Å)
- `cntm_step`: Continuum opacity calculation spacing (default: 1 Å)

## Testing Strategy

The test suite is comprehensive and includes:
- Individual component tests for each major module
- Comparison tests against reference calculations
- Autodifferentiation compatibility tests
- Aqua.jl quality assurance checks
- Performance-sensitive tests marked as "slow"

Key test files:
- `test/synth.jl` and `test/synthesize.jl`: Core synthesis functionality
- `test/continuum_absorption.jl`: Opacity calculations
- `test/fit.jl`: Parameter fitting routines
- `test/autodiff.jl`: Automatic differentiation compatibility

## External Dependencies

Major dependencies include:
- **Scientific Computing**: ForwardDiff.jl, Optim.jl, NLsolve.jl, Interpolations.jl
- **Data Handling**: HDF5.jl, CSV.jl, DataFrames.jl
- **Numerical Methods**: FastGaussQuadrature.jl, SpecialFunctions.jl

## Development Notes

- Code follows Julia conventions with comprehensive docstrings
- Physical constants defined in `src/constants.jl`
- Atomic data and linelists stored in `data/` directory
- Tutorial notebooks available in `misc/Tutorial notebooks/`
- Supports both Julia and Python interfaces via PythonCall.jl

## Model Atmosphere Integration

- Primary support for MARCS stellar atmosphere models
- Interpolation capabilities for different stellar parameters (Teff, log g, [M/H])
- Atmosphere files typically have `.mod` extension
- Custom atmosphere reading via `read_model_atmosphere()`