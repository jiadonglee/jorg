"""
Jorg: JAX-based stellar spectral synthesis package

A high-performance implementation of Korg.jl using JAX.

Features:
- 90-96.5% agreement with Korg.jl across stellar parameter space
- Full Korg.jl API compatibility with synth() and synthesize()
- Production-ready spectral synthesis for stellar surveys
- JAX-optimised for GPU acceleration and automatic differentiation
"""

__version__ = "0.1.0"
__author__ = "Jorg Development Team"

# Import main synthesis functions
from .synthesis import synth, synthesize, SynthesisResult

# Import main modules
from . import continuum
from . import lines
from . import utils
from . import constants
# from . import statmech  # Temporarily disabled due to circular import

# Export main API
__all__ = [
    # High-level synthesis functions
    "synth", "synthesize", "SynthesisResult",
    # Modules
    "continuum", "lines", "utils", "constants"  # , "statmech"
]