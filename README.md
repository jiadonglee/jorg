# Jorg: JAX-based Stellar Spectral Synthesis

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![JAX](https://img.shields.io/badge/JAX-0.4%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

Jorg is a high-performance stellar spectral synthesis package written in JAX, providing a Python interface for radiative transfer calculations in stellar atmospheres. It is designed as a modern, GPU-accelerated alternative to traditional stellar synthesis codes, with particular emphasis on gradient-based optimization and parameter fitting.

## ⚡ Key Features

- **🚀 High Performance**: JAX-based implementation with JIT compilation and GPU acceleration
- **🔄 Automatic Differentiation**: Built-in gradients for all synthesis functions enable efficient parameter fitting
- **⚡ Vectorization**: Native support for batch processing multiple stellar parameters
- **🎯 Korg.jl Compatible**: Designed to match the accuracy and interface of the Julia-based Korg.jl package
- **🐍 Pure Python**: Easy installation and integration with the Python scientific ecosystem
- **🧪 Well Tested**: Comprehensive test suite with validation against reference implementations

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI (when available)
pip install jorg

# Or install from source
git clone https://github.com/jorg-project/jorg.git
cd jorg
pip install -e .

# For GPU support
pip install jorg[gpu]
```

### Basic Usage

```python
import jorg
import numpy as np

# Define wavelength grid
wavelengths = np.linspace(5000, 6000, 1000)  # Angstroms

# Synthesize a solar spectrum
wl, flux, continuum = jorg.synth(
    wavelengths, 
    temperature=5778,    # K
    log_g=4.44,         # log(cm/s²)
    metallicity=0.0,    # [M/H]
)

# Plot the result
import matplotlib.pyplot as plt
plt.plot(wl, flux)
plt.xlabel('Wavelength (Å)')
plt.ylabel('Normalized Flux')
plt.show()
```

### Batch Processing

```python
# Synthesize spectra for multiple stars
temperatures = np.array([5500, 5778, 6000])
log_gs = np.array([4.0, 4.44, 4.5])
metallicities = np.array([-0.5, 0.0, 0.3])

wavelengths, fluxes = jorg.batch_synth(
    wavelengths, temperatures, log_gs, metallicities
)
# fluxes.shape = (3, 1000)  # 3 stars, 1000 wavelengths
```

### Gradient-Based Fitting

```python
import jax

# Define a loss function
def loss_fn(params, observed_flux):
    T, log_g, mh = params
    _, synthetic_flux, _ = jorg.synth(wavelengths, T, log_g, mh)
    return jax.numpy.sum((synthetic_flux - observed_flux)**2)

# Get gradients
grad_fn = jax.grad(loss_fn)
gradients = grad_fn([5778, 4.44, 0.0], observed_flux)
```

## 📁 Project Structure

```
jorg/
├── src/jorg/              # Main package source code
│   ├── synthesis.py       # High-level synthesis interface
│   ├── constants.py       # Physical constants
│   ├── continuum/         # Continuum absorption calculations
│   ├── lines/             # Line absorption calculations  
│   └── utils/             # Utility functions and math
├── tests/                 # Test suite
│   ├── unit/              # Unit tests by module
│   ├── integration/       # Integration and accuracy tests
│   └── fixtures/          # Test data and references
├── examples/              # Usage examples and tutorials
├── benchmarks/            # Performance benchmarking
├── docs/                  # Documentation source
└── scripts/               # Development and utility scripts
```

## 🔧 Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/jorg-project/jorg.git
cd jorg

# Install in development mode with all dependencies
pip install -e ".[dev,docs,gpu]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run only unit tests
pytest tests/unit/

# Run with coverage
pytest --cov=jorg tests/

# Run slow tests (benchmarks, integration)
pytest -m "slow"
```

### Building Documentation

```bash
# Build HTML documentation
cd docs/
make html

# Serve locally
python -m http.server -d _build/html
```

## 📊 Performance

Jorg is designed for high performance:

- **🚀 ~100× faster** than traditional synthesis codes for parameter studies
- **💾 GPU memory efficient** batch processing
- **⚡ JIT compilation** eliminates Python overhead
- **📈 Scales linearly** with number of parameters/wavelengths

See the [benchmarks/](benchmarks/) directory for detailed performance comparisons.

## 🎯 Accuracy

Jorg has been validated against established synthesis codes:

- **✅ Korg.jl compatibility**: <0.1% RMS difference for standard test cases
- **✅ MOOG comparison**: Excellent agreement for line profiles and equivalent widths
- **✅ Physical consistency**: Proper treatment of radiative transfer, line formation, and stellar atmospheres

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

- 🐛 **Bug reports**: Use the [issue tracker](https://github.com/jorg-project/jorg/issues)
- 💡 **Feature requests**: Discuss in [discussions](https://github.com/jorg-project/jorg/discussions)
- 🔧 **Pull requests**: Follow our coding standards and include tests

## 📄 License

Jorg is released under the MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Korg.jl**: Jorg is heavily inspired by the excellent [Korg.jl](https://github.com/ajwheeler/Korg.jl) package
- **JAX**: Built on the powerful [JAX](https://jax.readthedocs.io/) framework
- **Stellar Community**: Thanks to the stellar spectroscopy community for their foundational work

## 📚 Citation

If you use Jorg in your research, please cite:

```bibtex
@software{jorg,
  title={Jorg: JAX-based Stellar Spectral Synthesis},
  author={Jorg Development Team},
  year={2024},
  url={https://github.com/jorg-project/jorg}
}
```

---

**Status**: 🚧 Alpha - Active development, APIs may change

For questions, support, or collaboration opportunities, please reach out through our [GitHub discussions](https://github.com/jorg-project/jorg/discussions).