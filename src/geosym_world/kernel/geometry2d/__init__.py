"""Analytic 2D geometry and intersections."""

from . import analytic, intersections, regions
from .analytic import is_rectangle_2d
from .intersections import INTERSECTION_STATUSES, IntersectionResult, PredicateDiagnostic, diagnose_collinearity_2d, diagnose_line_circle_tangency_2d
from .regions import RegionGeometry2D, boolean_polygon_boundaries_2d, boolean_region_geometries_2d, intersection_polygon_2d, normalize_polygon_ring_2d, region_geometry_area_2d, resolve_region_geometry_2d

__all__ = [
    "analytic",
    "intersections",
    "regions",
    "RegionGeometry2D",
    "INTERSECTION_STATUSES",
    "IntersectionResult",
    "PredicateDiagnostic",
    "boolean_polygon_boundaries_2d",
    "boolean_region_geometries_2d",
    "intersection_polygon_2d",
    "normalize_polygon_ring_2d",
    "diagnose_collinearity_2d",
    "diagnose_line_circle_tangency_2d",
    "is_rectangle_2d",
    "region_geometry_area_2d",
    "resolve_region_geometry_2d",
]
