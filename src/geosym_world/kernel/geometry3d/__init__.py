"""
GeoSymSolid pilot engine.

This package separates symbolic solid-geometry state from rendering backends.
CadQuery/OpenCascade is used when available for CAD validation/export, while
the renderer layer can use Blender Freestyle or a local textbook-style fallback.
"""

from .kernel import GeoSymSolidKernel
from .pipeline import GeoSymSolidPipeline

__all__ = ["GeoSymSolidKernel", "GeoSymSolidPipeline"]
