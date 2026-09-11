"""Conservative spatial grounding for recognized diagram labels."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from geosym_world.errors import InvalidGeometryInput
from geosym_world.schema import (
    ParseConflictGroup,
    ParseEvidence,
    ParseObject,
    ParseObjectKind,
    ParseOrigin,
    ParseReference,
    ParseRelation,
)


_POINT_NAME_PATTERN = re.compile(r"^[A-Z][0-9]?$", re.ASCII)
_LINE_NAME_PATTERN = re.compile(r"^[a-z](?:[A-Za-z_]*[A-Za-z])?$", re.ASCII)
_NON_LINE_NAME_TOKENS = frozenset({"frac", "pi", "sin", "cos", "tan", "sqrt"})


def ground_point_labels(
    objects: Sequence[ParseObject],
    evidence: Mapping[str, ParseEvidence],
    *,
    max_distance: float = 28.0,
    ambiguity_margin: float = 8.0,
) -> tuple[tuple[ParseRelation, ...], tuple[ParseConflictGroup, ...]]:
    """Propose point-name bindings and preserve spatial ambiguity explicitly."""

    if max_distance <= 0 or ambiguity_margin < 0:
        raise InvalidGeometryInput("Point-label grounding distances must be non-negative with a positive maximum")
    points = tuple(item for item in objects if item.kind is ParseObjectKind.POINT and _point_position(item) is not None)
    labels = tuple(
        item
        for item in objects
        if item.kind is ParseObjectKind.TEXT
        and item.visible_name is not None
        and _POINT_NAME_PATTERN.fullmatch(item.visible_name.strip())
        and _label_bbox(item) is not None
    )
    bindings = []
    conflicts = []
    for label in sorted(labels, key=lambda item: item.id):
        ranked = sorted(
            (
                (_point_bbox_distance(_point_position(point), _label_bbox(label)), point)
                for point in points
            ),
            key=lambda item: (item[0], item[1].id),
        )
        if not ranked or ranked[0][0] > max_distance:
            continue
        nearest_distance = ranked[0][0]
        ambiguous = tuple(item for item in ranked if item[0] <= max_distance and item[0] <= nearest_distance + ambiguity_margin)
        selected = ambiguous if len(ambiguous) > 1 else ranked[:1]
        member_ids = []
        for distance, point in selected:
            relation_id = f"binding:{label.id}:{point.id}"
            member_ids.append(relation_id)
            bindings.append(
                ParseRelation(
                    id=relation_id,
                    predicate="binds",
                    arguments=(ParseReference("label", label.id), ParseReference("object", point.id)),
                    origin=ParseOrigin.IMAGE_TEXT,
                    confidence=_binding_confidence(label, point, distance, max_distance),
                    score_kind="point_label_spatial_margin",
                    evidence_refs=_binding_evidence(label, point, evidence),
                    metadata={
                        "binding_kind": "point_name",
                        "label": label.visible_name.strip(),
                        "distance_px": round(distance, 3),
                        "max_distance_px": max_distance,
                        "ambiguity_margin_px": ambiguity_margin,
                    },
                )
            )
        if len(member_ids) > 1:
            conflicts.append(
                ParseConflictGroup(
                    id=f"conflict:{label.id}:point_binding",
                    members=tuple(member_ids),
                    reason="multiple_points_within_spatial_margin",
                )
            )
    conflicts.extend(_binding_uniqueness_conflicts(bindings, conflicts))
    return tuple(bindings), tuple(conflicts)


def ground_line_labels(
    objects: Sequence[ParseObject],
    evidence: Mapping[str, ParseEvidence],
    *,
    max_distance: float = 28.0,
    ambiguity_margin: float = 8.0,
    minimum_projection: float = 0.1,
    maximum_projection: float = 0.9,
) -> tuple[tuple[ParseRelation, ...], tuple[ParseConflictGroup, ...]]:
    """Propose line-name bindings from labels near the interior of supports."""

    if max_distance <= 0.0 or ambiguity_margin < 0.0:
        raise InvalidGeometryInput("Line-label grounding distances require a positive maximum and non-negative margin")
    if not 0.0 <= minimum_projection < maximum_projection <= 1.0:
        raise InvalidGeometryInput("Line-label projection bounds must satisfy 0 <= minimum < maximum <= 1")

    object_index = {candidate.id: candidate for candidate in objects}
    supports = tuple(
        candidate
        for candidate in objects
        if candidate.kind in {ParseObjectKind.LINE, ParseObjectKind.SEGMENT}
        and candidate.visible is not False
        and _support_points(candidate, object_index) is not None
    )
    labels = tuple(
        candidate
        for candidate in objects
        if candidate.kind is ParseObjectKind.TEXT
        and candidate.visible_name is not None
        and _is_line_name(candidate.visible_name)
        and _label_bbox(candidate) is not None
    )

    bindings: list[ParseRelation] = []
    conflicts: list[ParseConflictGroup] = []
    for label in sorted(labels, key=lambda candidate: candidate.id):
        center = _label_center(label)
        ranked = []
        if center is not None:
            for support in supports:
                points = _support_points(support, object_index)
                if points is None:
                    continue
                distance, projection = _point_segment_distance(center, *points)
                if distance <= max_distance and minimum_projection <= projection <= maximum_projection:
                    ranked.append((distance, projection, support))
        ranked.sort(key=lambda item: (item[0], item[2].id))
        if not ranked:
            continue
        nearest_distance = ranked[0][0]
        selected = tuple(item for item in ranked if item[0] <= nearest_distance + ambiguity_margin)
        member_ids = []
        for distance, projection, support in selected:
            relation_id = f"binding:{label.id}:{support.id}:line_name"
            member_ids.append(relation_id)
            bindings.append(
                ParseRelation(
                    id=relation_id,
                    predicate="binds",
                    arguments=(ParseReference("label", label.id), ParseReference("object", support.id)),
                    origin=ParseOrigin.IMAGE_TEXT,
                    confidence=_binding_confidence(label, support, distance, max_distance),
                    score_kind="line_label_spatial_margin",
                    evidence_refs=_binding_evidence(label, support, evidence),
                    metadata={
                        "binding_kind": "line_name",
                        "label": label.visible_name.strip(),
                        "distance_px": round(distance, 3),
                        "projection": round(projection, 6),
                        "max_distance_px": max_distance,
                        "ambiguity_margin_px": ambiguity_margin,
                    },
                )
            )
        if len(member_ids) > 1:
            conflicts.append(
                ParseConflictGroup(
                    id=f"conflict:{label.id}:line_binding",
                    members=tuple(member_ids),
                    reason="multiple_supports_within_spatial_margin",
                )
            )
    conflicts.extend(
        _binding_uniqueness_conflicts(
            bindings,
            conflicts,
            conflict_id_prefix="line_name_uniqueness",
            conflict_reason="line_name_bindings_must_be_one_to_one",
        )
    )
    return tuple(bindings), tuple(conflicts)


def _is_line_name(value: str) -> bool:
    normalized = value.strip()
    return bool(_LINE_NAME_PATTERN.fullmatch(normalized)) and normalized.lower() not in _NON_LINE_NAME_TOKENS


def _support_points(
    candidate: ParseObject,
    object_index: Mapping[str, ParseObject],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    first = _finite_pair(candidate.geometry.get("p1"))
    second = _finite_pair(candidate.geometry.get("p2"))
    if first is not None and second is not None and first != second:
        return first, second
    references = {
        reference.role: object_index.get(reference.ref)
        for reference in candidate.references
        if reference.role in {"start", "end", "point_0", "point_1"}
    }
    start = _point_position(references.get("start") or references.get("point_0"))
    end = _point_position(references.get("end") or references.get("point_1"))
    return (start, end) if start is not None and end is not None and start != end else None


def _finite_pair(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return None
    try:
        pair = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    return pair if all(math.isfinite(item) for item in pair) else None


def _binding_uniqueness_conflicts(
    bindings: Sequence[ParseRelation],
    existing: Sequence[ParseConflictGroup],
    *,
    conflict_id_prefix: str = "point_name_uniqueness",
    conflict_reason: str = "point_name_bindings_must_be_one_to_one",
) -> tuple[ParseConflictGroup, ...]:
    conflicted = {member for conflict in existing for member in conflict.members}
    candidates = [binding for binding in bindings if binding.id not in conflicted]
    parent = {binding.id: binding.id for binding in candidates}

    def find(candidate_id: str) -> str:
        while parent[candidate_id] != candidate_id:
            parent[candidate_id] = parent[parent[candidate_id]]
            candidate_id = parent[candidate_id]
        return candidate_id

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    by_key: dict[tuple[str, str], list[str]] = {}
    for binding in candidates:
        point_id = next((item.ref for item in binding.arguments if item.role == "object"), None)
        label = binding.metadata.get("label")
        if point_id is not None:
            by_key.setdefault(("point", point_id), []).append(binding.id)
        if isinstance(label, str):
            by_key.setdefault(("name", label.strip().upper()), []).append(binding.id)
    for member_ids in by_key.values():
        for member_id in member_ids[1:]:
            union(member_ids[0], member_id)

    components: dict[str, list[str]] = {}
    for candidate_id in parent:
        components.setdefault(find(candidate_id), []).append(candidate_id)
    return tuple(
        ParseConflictGroup(
            id=f"conflict:{conflict_id_prefix}:{index}",
            members=tuple(sorted(member_ids)),
            reason=conflict_reason,
        )
        for index, member_ids in enumerate(sorted(components.values()))
        if len(member_ids) > 1
    )


def _point_position(candidate: ParseObject) -> tuple[float, float] | None:
    value = candidate.geometry.get("pixel")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return None
    return float(value[0]), float(value[1])


def _label_bbox(candidate: ParseObject) -> tuple[float, float, float, float] | None:
    value = candidate.geometry.get("bbox")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        return None
    return tuple(float(item) for item in value)


def _label_center(candidate: ParseObject) -> tuple[float, float] | None:
    bbox = _label_bbox(candidate)
    if bbox is None:
        return None
    left, top, right, bottom = bbox
    return (left + right) / 2.0, (top + bottom) / 2.0


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[float, float]:
    dx, dy = end[0] - start[0], end[1] - start[1]
    squared_length = dx * dx + dy * dy
    if squared_length == 0.0:
        return math.inf, 0.0
    projection = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / squared_length
    nearest = start[0] + projection * dx, start[1] + projection * dy
    return math.dist(point, nearest), projection


def _point_bbox_distance(point, bbox) -> float:
    x, y = point
    left, top, right, bottom = bbox
    dx = max(left - x, 0.0, x - right)
    dy = max(top - y, 0.0, y - bottom)
    return math.hypot(dx, dy)


def _binding_confidence(label: ParseObject, point: ParseObject, distance: float, max_distance: float) -> float:
    spatial_score = max(0.0, 1.0 - distance / max_distance)
    return round(min(label.confidence, point.confidence, 0.5 + 0.5 * spatial_score), 3)


def _binding_evidence(
    label: ParseObject,
    point: ParseObject,
    evidence: Mapping[str, ParseEvidence],
) -> tuple[str, ...]:
    refs = tuple(dict.fromkeys((*label.evidence_refs, *point.evidence_refs)))
    missing = sorted(set(refs) - set(evidence))
    if missing:
        raise InvalidGeometryInput(f"Point-label binding references unknown evidence: {missing}")
    return refs
