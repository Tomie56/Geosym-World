"""Exact geometric preflight for synthetic enhancement operations."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping

import sympy as sp

from geosym_world.construction.builder import ConstructionBuilder
from geosym_world.errors import GeometryWorldError
from geosym_world.kernel.geometry2d.analytic import cross2, sub
from geosym_world.schema.state import FormalGeometryState


PREFLIGHT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class PreflightResult:
    valid: bool
    reason: str | None = None
    state: FormalGeometryState | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class PreflightDiagnostics:
    checked: int = 0
    accepted: int = 0
    skip_counts: Counter[str] = field(default_factory=Counter)
    examples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    max_examples_per_reason: int = 3

    def record(self, operation: Mapping[str, Any], result: PreflightResult) -> None:
        self.checked += 1
        if result.valid:
            self.accepted += 1
            return
        reason = result.reason or "preflight_failed"
        self.skip_counts[reason] += 1
        examples = self.examples.setdefault(reason, [])
        if len(examples) < self.max_examples_per_reason:
            examples.append({"operation": operation.get("op"), "id": operation.get("id"), **dict(result.metadata)})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "checked": self.checked,
            "accepted": self.accepted,
            "skipped": self.checked - self.accepted,
            "skip_counts": dict(sorted(self.skip_counts.items())),
            "examples": {key: list(value) for key, value in sorted(self.examples.items())},
        }


class EnhancementPreflight:
    """Validate one operation without mutating the persistent candidate state."""

    def check(self, state: FormalGeometryState, operation: Mapping[str, Any]) -> PreflightResult:
        before_ids = state.object_ids()
        candidate = FormalGeometryState.from_dict(state.to_dict())
        try:
            ConstructionBuilder(candidate).execute([dict(operation)])
        except (GeometryWorldError, KeyError, ValueError) as exc:
            return PreflightResult(False, "construction_error", metadata={"error_type": type(exc).__name__, "error": str(exc)})

        new_ids = candidate.object_ids() - before_ids
        for point_id in sorted(new_ids & set(candidate.points)):
            point = candidate.points[point_id]
            for existing_id in sorted(before_ids & set(state.points)):
                if _same_coords(point.coords, state.points[existing_id].coords):
                    return PreflightResult(False, "coincident_point", metadata={"point": point_id, "coincident_with": existing_id})

        for segment_id in sorted(new_ids & set(candidate.segments)):
            segment = candidate.segments[segment_id]
            for existing_id, existing in state.segments.items():
                if _segments_overlap(state, segment.start, segment.end, existing.start, existing.end):
                    return PreflightResult(False, "overlapping_segment", metadata={"segment": segment_id, "overlaps": existing_id})

        new_line_like = (new_ids & set(candidate.lines)) | (new_ids & set(candidate.rays))
        existing_line_like = set(state.lines) | set(state.rays) | set(state.segments)
        for object_id in sorted(new_line_like):
            line = candidate.get_line_like(object_id)
            for existing_id in sorted(existing_line_like):
                existing = state.get_line_like(existing_id)
                if line.dim == existing.dim == 2 and _coincident_lines(line.point, line.direction, existing.point, existing.direction):
                    return PreflightResult(False, "coincident_line", metadata={"line": object_id, "coincident_with": existing_id})
        return PreflightResult(True, state=candidate)


def _same_coords(left, right) -> bool:
    return len(left) == len(right) and all(sp.simplify(a - b) == 0 for a, b in zip(left, right))


def _coincident_lines(point1, direction1, point2, direction2) -> bool:
    return sp.simplify(cross2(direction1, direction2)) == 0 and sp.simplify(cross2(sub(point2, point1), direction1)) == 0


def _segments_overlap(state: FormalGeometryState, a_id: str, b_id: str, c_id: str, d_id: str) -> bool:
    a = state.points[a_id].coords
    b = state.points[b_id].coords
    c = state.points[c_id].coords
    d = state.points[d_id].coords
    direction = sub(b, a)
    if sp.simplify(cross2(direction, sub(c, a))) != 0 or sp.simplify(cross2(direction, sub(d, a))) != 0:
        return False
    axis = 0 if sp.simplify(direction[0]) != 0 else 1
    first = sorted((float(sp.N(a[axis])), float(sp.N(b[axis]))))
    second = sorted((float(sp.N(c[axis])), float(sp.N(d[axis]))))
    return min(first[1], second[1]) - max(first[0], second[0]) > 1e-9
