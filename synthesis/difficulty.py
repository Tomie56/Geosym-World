"""Difficulty scoring policy for synthetic records."""

from __future__ import annotations

from typing import Any, Dict, Iterable


def score_trace(trace: Iterable[Dict[str, Any]], diagnostic_axes: Iterable[str] = ()) -> tuple[str, Dict[str, Any]]:
    ops = list(trace)
    axes = set(diagnostic_axes)
    score = len(ops)
    score += 4 * sum(1 for op in ops if "intersection" in str(op.get("op")))
    score += 3 * sum(1 for op in ops if op.get("op") in {"angle_bisector", "tangent_points_from_point", "section_polyhedron_with_plane"})
    score += 4 if "3d_interface" in axes else 0
    score += 3 if "similar_triangle" in axes else 0
    if score < 13:
        label = "easy"
    elif score < 22:
        label = "medium"
    else:
        label = "hard"
    return label, {"score": score, "num_ops": len(ops), "axes": sorted(axes)}
