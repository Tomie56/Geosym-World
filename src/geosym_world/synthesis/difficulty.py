"""Auditable difficulty scoring for synthetic geometry tasks."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, Mapping

from .task_graph import build_task_dependency_graph


DIFFICULTY_SCHEMA_VERSION = "2.0"
DIFFICULTY_LEVELS = ("easy", "medium", "hard", "expert")
_DIFFICULTY_RANK = {label: index for index, label in enumerate(DIFFICULTY_LEVELS)}
_THRESHOLDS = {"easy": 14, "medium": 25, "hard": 42}
_OPERATION_WEIGHTS = {
    "add_arc": 2,
    "add_circle": 1,
    "add_composite_region": 4,
    "add_polygon": 1,
    "add_region": 2,
    "angle_bisector": 3,
    "circle_circle_intersection": 5,
    "foot_of_perpendicular": 3,
    "intersect": 5,
    "line_circle_intersection": 5,
    "line_line_intersection": 4,
    "mark_sector_region": 3,
    "midpoint": 2,
    "parallel_through": 2,
    "perpendicular_through": 2,
    "point_on_segment_ratio": 2,
    "section_polyhedron_with_plane": 7,
    "tangent_points_from_point": 5,
}
_AXIS_WEIGHTS = {
    "3d_interface": 7,
    "area_ratio": 3,
    "circle_difference": 4,
    "circular_segment": 4,
    "composite_area": 5,
    "cyclic_quadrilateral": 6,
    "multiple_solution_paths": 5,
    "multi_step_area": 5,
    "proof_target": 6,
    "region_boolean": 5,
    "similar_triangle": 6,
    "tangential_quadrilateral": 6,
}
_TARGET_WEIGHTS = {
    "area": 1,
    "arc_length": 2,
    "perimeter": 2,
    "ratio": 4,
    "volume": 3,
}


def score_trace(trace: Iterable[Dict[str, Any]], diagnostic_axes: Iterable[str] = ()) -> tuple[str, Dict[str, Any]]:
    return score_task(trace, diagnostic_axes)


def score_task(
    trace: Iterable[Dict[str, Any]],
    diagnostic_axes: Iterable[str] = (),
    *,
    derivation: Mapping[str, Any] | None = None,
    solution_paths: Mapping[str, Any] | None = None,
) -> tuple[str, Dict[str, Any]]:
    operations = list(trace)
    axes = sorted(set(str(axis) for axis in diagnostic_axes))
    operation_counts = Counter(str(operation.get("op", "")) for operation in operations)
    constructive_ops = sum(count for name, count in operation_counts.items() if name not in {"add_point", "set_target"})
    operation_score = sum(_OPERATION_WEIGHTS.get(name, 1 if name not in {"add_point", "set_target"} else 0) * count for name, count in operation_counts.items())
    task_graph = build_task_dependency_graph(operations)
    construction_score = min(12, operation_score) + min(12, int(task_graph["effective_depth"])) + min(6, 2 * int(task_graph["branch_points"]))
    axis_score = sum(_AXIS_WEIGHTS.get(axis, 0) for axis in axes)
    target = next((operation for operation in reversed(operations) if operation.get("op") == "set_target"), {})
    target_mode = str(target.get("mode", "value"))
    target_type = str(target.get("type", ""))
    target_score = _TARGET_WEIGHTS.get(target_type, 2)
    if target_mode == "proof":
        target_score += 7
    elif target_mode == "constraint":
        target_score += 5

    derivation_steps = list((derivation or {}).get("steps") or ())
    derivation_depth = int(((derivation or {}).get("integrity") or {}).get("max_depth", 0))
    derived_rules = {str(step.get("rule", "")) for step in derivation_steps if step.get("source") == "derived"}
    derivation_score = min(8, len(derived_rules) // 2) + min(8, derivation_depth)

    paths = list((solution_paths or {}).get("paths") or ())
    alternative_count = int((solution_paths or {}).get("alternative_count", 0))
    solution_depth = max((int((path.get("integrity") or {}).get("max_depth", 0)) for path in paths), default=0)
    solution_score = 4 * alternative_count + min(5, solution_depth)

    score = construction_score + axis_score + target_score + derivation_score + solution_score
    label = _label_for_score(score)
    policy = {
        "schema_version": DIFFICULTY_SCHEMA_VERSION,
        "label": label,
        "score": score,
        "thresholds": dict(_THRESHOLDS),
        "features": {
            "construction": {
                "num_operations": len(operations),
                "num_constructive_operations": constructive_ops,
                "operation_counts": dict(sorted(operation_counts.items())),
                "score": construction_score,
                "effective_depth": task_graph["effective_depth"],
                "branch_points": task_graph["branch_points"],
                "live_steps": len(task_graph["live_steps"]),
                "dead_enhancement_steps": list(task_graph["dead_enhancement_steps"]),
                "dependency_graph_checksum": task_graph["checksum"],
            },
            "target": {"mode": target_mode, "type": target_type, "score": target_score},
            "semantic_axes": {"values": axes, "score": axis_score},
            "derivation": {
                "available": derivation is not None,
                "num_steps": len(derivation_steps),
                "num_derived_rules": len(derived_rules),
                "max_depth": derivation_depth,
                "score": derivation_score,
            },
            "solutions": {
                "available": solution_paths is not None,
                "num_paths": len(paths),
                "alternative_count": alternative_count,
                "max_depth": solution_depth,
                "score": solution_score,
            },
        },
        "num_ops": len(operations),
        "axes": axes,
    }
    return label, policy


def difficulty_in_range(label: str, *, minimum: str | None = None, maximum: str | None = None) -> bool:
    if label not in _DIFFICULTY_RANK:
        raise ValueError(f"Unknown difficulty label: {label}")
    if minimum is not None and minimum not in _DIFFICULTY_RANK:
        raise ValueError(f"Unknown minimum difficulty: {minimum}")
    if maximum is not None and maximum not in _DIFFICULTY_RANK:
        raise ValueError(f"Unknown maximum difficulty: {maximum}")
    rank = _DIFFICULTY_RANK[label]
    return (minimum is None or rank >= _DIFFICULTY_RANK[minimum]) and (maximum is None or rank <= _DIFFICULTY_RANK[maximum])


def _label_for_score(score: int) -> str:
    if score < _THRESHOLDS["easy"]:
        return "easy"
    if score < _THRESHOLDS["medium"]:
        return "medium"
    if score < _THRESHOLDS["hard"]:
        return "hard"
    return "expert"
