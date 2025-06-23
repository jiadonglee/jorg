"""
Jorg: JAX-based stellar spectral synthesis package
A high-performance translation of Korg.jl using JAX
"""

__version__ = "0.1.0"
__author__ = "Jorg Development Team"

# Import main modules
from . import continuum
from . import constants

__all__ = ["continuum", "constants"]