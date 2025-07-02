"""
Jorg: JAX-based stellar spectral synthesis package
A high-performance translation of Korg.jl using JAX

Key Features:
- Sophisticated hydrogen line treatment with MHD formalism
- JAX-optimized for GPU acceleration and automatic differentiation
- Complete physics implementation matching Korg.jl accuracy
"""

__version__ = "0.2.0"  # Updated for hydrogen lines implementation
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