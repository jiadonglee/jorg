# Jorg

A JAX-based Python package for stellar spectral synthesis calculations, translated from [Korg.jl](https://github.com/ajwheeler/Korg.jl).

## Current Implementation

Jorg provides JAX-optimized implementations of key stellar physics components:

### Statistical Mechanics
- Equation of state calculations
- Gas and electron pressure computations
- Basic atmospheric structure utilities

### Continuum Absorption
- H I bound-free absorption
- H⁻ bound-free and free-free absorption (McLaughlin 2017 data)
- Thomson scattering
- Rayleigh scattering
- He⁻ free-free absorption

### Line Absorption  
- Voigt line profiles with exact Hjerting function
- Doppler, Stark, and van der Waals broadening
- Basic linelist management
- Species identification and parsing

### Utilities
- Physical constants (CGS units)
- Wavelength conversions (air/vacuum)
- Mathematical utility functions

## Installation

```bash
git clone https://github.com/jiadonglee/jorg.git
cd jorg
pip install -e .
```

## Basic Usage

```python
import jax.numpy as jnp
from jorg.statmech.eos import gas_pressure, electron_pressure
from jorg.lines.profiles import line_profile
from jorg.continuum.hydrogen import h_minus_bf_absorption

# Example: Calculate gas pressure
T = 5777.0  # K
n_total = 1e16  # cm^-3
P_gas = gas_pressure(n_total, T)

# Example: Calculate Voigt profile
wavelength = jnp.linspace(5889.0, 5897.0, 100)
profile = line_profile(wavelength, 5889.95, 0.1, 0.05)
```

## Current Limitations

- Main synthesis functions (`synth`, `synthesize`) are placeholder implementations
- No radiative transfer solver
- No MARCS atmosphere interpolation
- Limited linelist format support
- No fitting or analysis tools

## Development Status

**Implemented and Validated:**
- Statistical mechanics (EOS calculations validated against Korg.jl with <1e-7 relative error)
- Line profiles (exact mathematical agreement with Korg.jl)
- Basic continuum absorption components

**In Development:**
- Complete synthesis pipeline
- Radiative transfer equation solver
- Atmosphere model interpolation

**Not Implemented:**
- Full spectral synthesis
- Parameter fitting
- Multi-format linelist readers
- Analysis tools

## Testing

```bash
pytest tests/
```

## Package Structure

```
src/jorg/
├── constants.py           # Physical constants
├── statmech/             # Statistical mechanics and EOS
├── continuum/            # Continuum absorption
├── lines/                # Line absorption and profiles
└── utils/                # Utility functions
```

## Dependencies

- JAX (for JIT compilation and automatic differentiation)
- NumPy
- SciPy (for special functions)

## Acknowledgments

Jorg is translated from the excellent [Korg.jl](https://github.com/ajwheeler/Korg.jl) package. We acknowledge the Korg.jl development team for their foundational work in stellar spectral synthesis.

## License

MIT License
