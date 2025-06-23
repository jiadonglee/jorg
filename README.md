# Jorg: JAX-based Stellar Spectral Synthesis

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![JAX](https://img.shields.io/badge/JAX-0.4%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

Jorg is a high-performance stellar spectral synthesis package written in JAX based on [Korg.jl](https://github.com/ajwheeler/Korg.jl), providing a Python interface for radiative transfer calculations in stellar atmospheres. It is designed as a modern, GPU-accelerated alternative to traditional stellar synthesis codes, with particular emphasis on gradient-based optimization and parameter fitting.

## ⚡ Key Features

- **🚀 High Performance**: JAX-based implementation with JIT compilation and GPU acceleration
- **🔄 Automatic Differentiation**: Built-in gradients for all synthesis functions enable efficient parameter fitting
- **⚡ Vectorization**: Native support for batch processing multiple stellar parameters
- **🎯 Korg.jl Compatible**: Designed to match the accuracy and interface of the Julia-based Korg.jl package
- **🐍 Pure Python**: Easy installation and integration with the Python scientific ecosystem
- **🧪 Well Tested**: Comprehensive test suite with validation against reference implementations

---

**Status**: 🚧 Alpha - Active development, APIs may change

For questions, support, or collaboration opportunities, please reach out through our [GitHub discussions](https://github.com/jorg-project/jorg/discussions).
