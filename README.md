# Jorg: JAX-based Optimized Radiative Transfer for Generating-spectra

**Jorg** is a Python/JAX implementation of stellar spectral synthesis, translated from Korg.jl. It provides stellar atmosphere modeling and spectral line formation using JAX for high-performance computing.

## Status: Not Production-Ready

Jorg is currently under development and **not ready for production use**. While some physics modules are working, significant issues remain with opacity calculations and overall system validation.

## Features

- **MARCS atmosphere interpolation** for stellar structure
- **Chemical equilibrium** calculations with 288 molecular species  
- **Continuum opacity** physics (H⁻, Thomson scattering, metals)
- **Line opacity** calculations with VALD format support
- **JAX-based** implementation for GPU acceleration
- **Radiative transfer** solvers

## Installation

```bash
cd Jorg
pip install -e .
```

## Usage

```python
from jorg.synthesis import synth

# Basic continuum synthesis
wavelengths, flux, continuum = synth(
    Teff=5780, logg=4.44, m_H=0.0,
    wavelengths=(5000, 5010)
)
```

---

*Translated from Korg.jl - Research-grade stellar spectral synthesis*