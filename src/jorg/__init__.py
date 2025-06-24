"""
Jorg: JAX-based stellar spectral synthesis package
A high-performance translation of Korg.jl using JAX
"""

__version__ = "0.1.0"
__author__ = "Jorg Development Team"

# Import main synthesis functions
from .synthesis import synth, synthesize, SynthesisResult, batch_synth

# Import main modules
from . import continuum
from . import lines
from . import utils
from . import constants
# from . import statmech  # Temporarily disabled due to circular import

# Export main API
__all__ = [
    # High-level synthesis functions
    "synth", "synthesize", "SynthesisResult", "batch_synth",
    # Modules
    "continuum", "lines", "utils", "constants"  # , "statmech"
]