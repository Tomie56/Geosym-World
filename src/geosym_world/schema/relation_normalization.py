"""Relation and target type definitions."""

from __future__ import annotations

from typing import Dict, Iterable, Set


RELATION_TYPES: Set[str] = {
    "collinear",
    "parallel",
    "perpendicular",
    "midpoint",
    "equal_length",
    "equal_angle",
    "point_on_line",
    "point_on_circle",
    "intersection",
    "tangent",
    "length",
    "angle",
    "area",
    "perimeter",
    "ratio",
    # Theorem-level semantic relations. The current coordinate verifier does
    # not prove these, but they are valid GeoSym-Debug graph facts and should
    # not be treated as unknown schema.
    "arc_measure",
    "chord",
    "circle",
    "congruent",
    "incidence",
    "intersecting_chords",
    "right_triangle",
    "similar",
    "triangle",
}

TARGET_TYPES: Set[str] = {
    "length",
    "angle",
    "perimeter",
    "entity_area",
    "shadow_area",
    "shadow_ratio",
    "shadow_entity_ratio",
    "all_shadow_area",
    "area",
    "ratio",
}

FAILURE_COMPONENTS: Set[str] = {
    "perception_error",
    "grounding_error",
    "topology_error",
    "theorem_or_planning_error",
    "computation_error",
    "pseudo_reasoning",
}


def is_relation_type(value: str) -> bool:
    return value in RELATION_TYPES


def is_target_type(value: str) -> bool:
    return value in TARGET_TYPES


def normalize_relation_type(value: str) -> str:
    value = (value or "").strip().lower().replace(" ", "_").replace("-", "_")
    aliases: Dict[str, str] = {
        "right_angle": "perpendicular",
        "orthogonal": "perpendicular",
        "on_line": "point_on_line",
        "on_circle": "point_on_circle",
        "same_length": "equal_length",
        "equal_segments": "equal_length",
    }
    return aliases.get(value, value)


def normalize_target_type(value: str) -> str:
    value = (value or "").strip().lower().replace(" ", "_").replace("-", "_")
    aliases: Dict[str, str] = {
        "line_length": "length",
        "segment_length": "length",
        "entity_area": "entity_area",
        "shaded_area": "shadow_area",
        "shadow": "shadow_area",
    }
    return aliases.get(value, value)


def unknown_relation_types(relations: Iterable[dict]) -> Set[str]:
    out: Set[str] = set()
    for relation in relations:
        rtype = normalize_relation_type(str(relation.get("type", "")))
        if rtype not in RELATION_TYPES:
            out.add(rtype)
    return out
