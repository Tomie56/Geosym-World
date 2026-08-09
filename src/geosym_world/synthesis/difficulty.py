"""Difficulty scoring policy for synthetic records."""

from __future__ import annotations

from typing import Any, Dict, Iterable


DIFFICULTY_LEVELS = ("easy", "medium", "hard", "expert")
_DIFFICULTY_RANK = {label: index for index, label in enumerate(DIFFICULTY_LEVELS)}


def score_trace(trace: Iterable[Dict[str, Any]], diagnostic_axes: Iterable[str] = ()) -> tuple[str, Dict[str, Any]]:
    ops = list(trace)
    axes = set(diagnostic_axes)
    score = len(ops)
    score += 5 * sum(1 for op in ops if "intersection" in str(op.get("op")))
    score += 3 * sum(1 for op in ops if op.get("op") in {"angle_bisector", "tangent_points_from_point", "section_polyhedron_with_plane", "foot_of_perpendicular"})
    score += 2 * sum(1 for op in ops if op.get("op") in {"midpoint", "point_on_segment_ratio", "parallel_through", "perpendicular_through"})
    score += 7 if "3d_interface" in axes else 0
    score += 6 if "similar_triangle" in axes else 0
    score += 5 if "composite_area" in axes else 0
    score += 5 if "multi_step_area" in axes else 0
    score += 4 if "circle_difference" in axes else 0
    score += 4 if "rhombus" in axes else 0
    score += 6 if "cyclic_quadrilateral" in axes else 0
    score += 6 if "tangential_quadrilateral" in axes else 0
    if score < 15:
        label = "easy"
    elif score < 22:
        label = "medium"
    elif score < 40:
        label = "hard"
    else:
        label = "expert"
    return label, {"score": score, "num_ops": len(ops), "axes": sorted(axes)}


def difficulty_in_range(label: str, *, minimum: str | None = None, maximum: str | None = None) -> bool:
    if label not in _DIFFICULTY_RANK:
        raise ValueError(f"Unknown difficulty label: {label}")
    if minimum is not None and minimum not in _DIFFICULTY_RANK:
        raise ValueError(f"Unknown minimum difficulty: {minimum}")
    if maximum is not None and maximum not in _DIFFICULTY_RANK:
        raise ValueError(f"Unknown maximum difficulty: {maximum}")
    rank = _DIFFICULTY_RANK[label]
    return (minimum is None or rank >= _DIFFICULTY_RANK[minimum]) and (maximum is None or rank <= _DIFFICULTY_RANK[maximum])
