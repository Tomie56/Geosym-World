"""Resolve model-facing object references against a formal geometry state."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from geosym_world_geometry.state import FormalGeometryState


@dataclass(frozen=True)
class Resolution:
    status: str
    object_id: str | None = None
    object_type: str | None = None
    candidates: tuple[str, ...] = ()


class ObjectReferenceResolver:
    """Conservative resolver that never guesses between ambiguous objects."""

    def resolve(self, reference: str, state: FormalGeometryState, expected_types: Iterable[str] = ()) -> Resolution:
        if not isinstance(reference, str) or not reference.strip():
            return Resolution("invalid")
        expected = set(expected_types)
        objects = self._objects(state)
        if reference in objects and (not expected or objects[reference] in expected):
            return Resolution("resolved", reference, objects[reference])
        normalized = _normalize(reference)
        candidates = [
            object_id for object_id, object_type in objects.items()
            if _normalize(object_id) == normalized and (not expected or object_type in expected)
        ]
        if not candidates:
            compact = normalized.removeprefix("point").removeprefix("line").removeprefix("segment").removeprefix("circle")
            candidates = [
                object_id for object_id, object_type in objects.items()
                if _normalize(object_id) == compact and (not expected or object_type in expected)
            ]
        if len(candidates) == 1:
            object_id = candidates[0]
            return Resolution("resolved", object_id, objects[object_id])
        if candidates:
            return Resolution("ambiguous", candidates=tuple(sorted(candidates)))
        structural = self._structural_candidates(reference, state, expected)
        if len(structural) == 1:
            object_id = structural[0]
            return Resolution("resolved", object_id, objects[object_id])
        if structural:
            return Resolution("ambiguous", candidates=tuple(sorted(structural)))
        return Resolution("unknown")

    def visible_object_ids(self, state: FormalGeometryState) -> tuple[str, ...]:
        return tuple(sorted(self._objects(state)))

    @staticmethod
    def _objects(state: FormalGeometryState) -> dict[str, str]:
        groups = {
            "point": state.points,
            "line": state.lines,
            "segment": state.segments,
            "ray": state.rays,
            "circle": state.circles,
            "arc": state.arcs,
            "polygon": state.polygons,
            "region": state.regions,
            "plane": state.planes,
            "polyhedron": state.polyhedra,
        }
        return {object_id: object_type for object_type, registry in groups.items() for object_id in registry}

    @staticmethod
    def _structural_candidates(reference: str, state: FormalGeometryState, expected: set[str]) -> list[str]:
        compact = _normalize(reference)
        if compact.startswith("line"):
            compact = compact[4:]
        if compact.startswith("segment"):
            compact = compact[7:]
        if len(compact) == 2:
            endpoints = {compact[0].upper(), compact[1].upper()}
            candidates = []
            if not expected or "segment" in expected:
                candidates.extend(object_id for object_id, segment in state.segments.items() if {segment.start.upper(), segment.end.upper()} == endpoints)
            if not expected or "line" in expected:
                candidates.extend(object_id for object_id, line in state.lines.items() if len(line.source_points) == 2 and {item.upper() for item in line.source_points} == endpoints)
            return candidates
        return []


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())
