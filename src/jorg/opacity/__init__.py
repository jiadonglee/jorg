"""
Opacity module for Korg-compatible Jorg synthesis
=================================================

This module provides systematic opacity calculation following Korg.jl's
exact architecture while using Jorg's validated physics implementations.
"""

from .layer_processor import LayerProcessor

__all__ = ['LayerProcessor']