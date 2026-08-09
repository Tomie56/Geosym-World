"""Auditable capability registry for the original synthesis toolchain."""

from __future__ import annotations

import inspect
from typing import Any, Dict

from geosym_world.legacy_tools.builder import RandomGeometryBuilder
from geosym_world.legacy_tools.drawer import GeometryDrawer
from geosym_world.legacy_tools.gt import GeometryCalculator
from geosym_world.legacy_tools.qa_template import QAGenerator
from geosym_world.legacy_tools.region import RegionExtractor
from geosym_world.legacy_tools.shader import EnhancedDrawer, GeometryAnnotator
from geosym_world.legacy_tools.shaders import SHADERS
from geosym_world.legacy_tools.template import TemplateGenerator


LEGACY_CAPABILITIES: Dict[str, tuple[str, ...]] = {
    "base_shapes": ("polygon", "circle", "special_triangle", "special_rectangle", "parallelogram", "trapezoid"),
    "derivation_rules": (
        "concentric",
        "translation",
        "circum_inscribe",
        "vertex_on_center",
        "flip",
        "special_triangle_flip",
        "special_triangle_translation",
        "special_rectangle_flip",
        "special_rectangle_translation",
        "parallelogram_flip",
        "parallelogram_translation",
        "trapezoid_flip",
        "trapezoid_translation",
        "side_polygon",
    ),
    "enhancement_operations": ("connect_points", "connect_midpoints", "draw_perpendicular", "draw_diameter"),
    "intersection_types": ("segment_segment", "segment_circle", "line_arc", "circle_circle"),
    "shaders": tuple(sorted(SHADERS)),
    "annotations": ("point", "line", "perpendicular"),
    "ground_truth": ("line_length", "arc_length", "perimeter", "entity_area", "shadow_area"),
    "question_types": (
        "length",
        "angle",
        "perimeter",
        "entity_area",
        "shadow_area",
        "shadow_ratio",
        "shadow_entity_ratio",
        "all_shadow_area",
    ),
    "answer_grading": ("mathd", "sympy", "verl", "boxed_answer"),
}


_CLASSES = {
    "TemplateGenerator": TemplateGenerator,
    "RandomGeometryBuilder": RandomGeometryBuilder,
    "GeometryDrawer": GeometryDrawer,
    "EnhancedDrawer": EnhancedDrawer,
    "GeometryAnnotator": GeometryAnnotator,
    "RegionExtractor": RegionExtractor,
    "GeometryCalculator": GeometryCalculator,
    "QAGenerator": QAGenerator,
}


def compatibility_report() -> Dict[str, Any]:
    """Return runtime evidence that the complete legacy classes are reachable."""

    return {
        "available": True,
        "capabilities": {name: list(values) for name, values in LEGACY_CAPABILITIES.items()},
        "classes": {
            name: {
                "module": cls.__module__,
                "methods": sorted(method_name for method_name, value in inspect.getmembers(cls, inspect.isfunction)),
            }
            for name, cls in _CLASSES.items()
        },
    }
