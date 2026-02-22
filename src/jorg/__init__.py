"""
Jorg - JAX Stellar Synthesis
============================

JAX-accelerated stellar spectral synthesis compatible with Korg.jl inputs.
"""

from .jax_runtime import configure_jax_runtime

configure_jax_runtime()

__version__ = "0.3.1"
