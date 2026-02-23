"""
Opacity module for Korg-compatible Jorg synthesis
=================================================

This module provides systematic opacity calculation following Korg.jl's
exact architecture while using Jorg's validated physics implementations.
"""

from .layer_processor import LayerProcessor
from .korg_line_processor import KorgLineProcessor
from .line_opacity_jax import compute_line_opacity_jax

__all__ = ['LayerProcessor', 'KorgLineProcessor', 'compute_line_opacity_jax']
