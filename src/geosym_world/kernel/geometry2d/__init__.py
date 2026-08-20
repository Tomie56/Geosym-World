"""Analytic 2D geometry and intersections."""

from . import analytic, intersections, regions
from .analytic import is_rectangle_2d
from .regions import RegionGeometry2D, boolean_polygon_boundaries_2d, intersection_polygon_2d, region_geometry_area_2d, resolve_region_geometry_2d

__all__ = [
    "analytic",
    "intersections",
    "regions",
    "RegionGeometry2D",
    "boolean_polygon_boundaries_2d",
    "intersection_polygon_2d",
    "is_rectangle_2d",
    "region_geometry_area_2d",
    "resolve_region_geometry_2d",
]
