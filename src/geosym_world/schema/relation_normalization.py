"""Relation and target type definitions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Iterable, Set

from geosym_world.errors import InvalidGeometryInput
from geosym_world.schema.canonical import stable_checksum
from geosym_world.schema.ids import validate_geometry_id


RELATION_TYPES: Set[str] = {
    "collinear",
    "parallel",
    "perpendicular",
    "midpoint",
    "equal_length",
    "equal_angle",
    "point_on_line",
    "point_on_circle",
    "point_on_arc",
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
    "arc",
    "sector_region",
    "angle_from_line",
    "angle_bisector",
    "foot_of_perpendicular",
    "point_on_segment_ratio",
    "point_on_ray_distance",
    "point_projection_plane",
    "line_projection_plane",
    "dihedral_angle",
    "polyhedron_plane_section",
    "linear_overlap",
    "point_on_ray",
}

SYMMETRIC_RELATIONS = {
    "parallel",
    "perpendicular",
    "equal_length",
    "equal_angle",
    "congruent",
    "similar",
}

TARGET_TYPES: Set[str] = {
    "length",
    "arc_length",
    "angle",
    "perimeter",
    "entity_area",
    "shadow_area",
    "shadow_ratio",
    "shadow_entity_ratio",
    "all_shadow_area",
    "area",
    "ratio",
    "parallel",
    "perpendicular",
    "collinear",
    "tangent_to_circle",
    "equal_length",
    "equal_angle",
    "similar",
    "congruent",
    "midpoint",
    "cyclic_quadrilateral",
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
        "same_angle": "equal_angle",
        "same_direction": "parallel",
        "line_intersection": "intersection",
    }
    return aliases.get(value, value)


def normalize_relation(relation: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize one state relation and assign a deterministic relation ID."""

    if not isinstance(relation, Mapping):
        raise InvalidGeometryInput("Geometry relation must be a mapping")
    normalized = dict(relation)
    relation_type = normalize_relation_type(str(normalized.get("type", "")))
    if not relation_type:
        raise InvalidGeometryInput("Geometry relation type must be non-empty")
    if relation_type not in RELATION_TYPES:
        raise InvalidGeometryInput(f"Unknown geometry relation type: {relation_type}")
    normalized["type"] = relation_type
    if relation_type in SYMMETRIC_RELATIONS and isinstance(normalized.get("objects"), (list, tuple)):
        normalized["objects"] = sorted(str(item) for item in normalized["objects"])
    relation_id = normalized.get("id")
    if relation_id is None:
        identity_payload = {key: value for key, value in normalized.items() if key != "id"}
        relation_id = f"rel_{stable_checksum(identity_payload)[:16]}"
    validate_geometry_id(str(relation_id))
    normalized["id"] = str(relation_id)
    return normalized


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
