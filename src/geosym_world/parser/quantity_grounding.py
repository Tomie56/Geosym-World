"""Conservative grounding of recognized image measurements to geometry objects."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

from geosym_world.errors import InvalidGeometryInput
from geosym_world.schema import (
    ParseConflictGroup,
    ParseEvidence,
    ParseObject,
    ParseObjectKind,
    ParseOrigin,
    ParseQuantity,
    ParseReference,
    math_expression_from_text,
)


_MEASUREMENT_TOKEN_PATTERN = re.compile(r"(?:\d|\\frac|\\sqrt|sqrt|pi|[π√½⅓⅔¼¾])", re.IGNORECASE)
_POINT_NAME_PATTERN = re.compile(r"^[A-Z][0-9]?$", re.ASCII)


def ground_segment_length_labels(
    objects: Sequence[ParseObject],
    evidence: Mapping[str, ParseEvidence],
    *,
    max_distance: float = 32.0,
    ambiguity_margin: float = 8.0,
    minimum_projection: float = 0.1,
    maximum_projection: float = 0.9,
) -> tuple[tuple[ParseQuantity, ...], tuple[ParseConflictGroup, ...], tuple[dict[str, object], ...]]:
    """Propose image length quantities without resolving spatial ambiguity."""

    if max_distance <= 0.0 or ambiguity_margin < 0.0:
        raise InvalidGeometryInput("Image-quantity distances require a positive maximum and non-negative margin")
    if not 0.0 <= minimum_projection < maximum_projection <= 1.0:
        raise InvalidGeometryInput("Image-quantity projection bounds must satisfy 0 <= minimum < maximum <= 1")

    object_index = {candidate.id: candidate for candidate in objects}
    segments = tuple(
        candidate
        for candidate in objects
        if candidate.kind is ParseObjectKind.SEGMENT
        and candidate.visible is not False
        and _segment_points(candidate, object_index) is not None
    )
    labels = tuple(
        candidate
        for candidate in objects
        if candidate.kind is ParseObjectKind.TEXT
        and candidate.visible_name is not None
        and not _POINT_NAME_PATTERN.fullmatch(candidate.visible_name.strip())
        and _MEASUREMENT_TOKEN_PATTERN.search(candidate.visible_name.strip())
        and _label_center(candidate) is not None
    )

    quantities: list[ParseQuantity] = []
    conflicts: list[ParseConflictGroup] = []
    issues: list[dict[str, object]] = []
    for label in sorted(labels, key=lambda candidate: candidate.id):
        raw_text = label.visible_name.strip()
        try:
            expression = math_expression_from_text(raw_text)
        except InvalidGeometryInput as exc:
            issues.append(
                {
                    "label_id": label.id,
                    "raw_text": raw_text,
                    "status": "invalid_expression",
                    "message": str(exc),
                }
            )
            continue

        ranked = _rank_segments(
            label,
            segments,
            object_index,
            maximum_distance=max_distance,
            minimum_projection=minimum_projection,
            maximum_projection=maximum_projection,
        )
        if not ranked:
            issues.append({"label_id": label.id, "raw_text": raw_text, "status": "object_unresolved"})
            continue

        nearest_distance = ranked[0][0]
        selected = tuple(candidate for candidate in ranked if candidate[0] <= nearest_distance + ambiguity_margin)
        member_ids = []
        for distance, projection, segment in selected:
            quantity_id = f"quantity:image:{label.id}:{segment.id}"
            member_ids.append(quantity_id)
            quantities.append(
                ParseQuantity(
                    id=quantity_id,
                    kind="length",
                    arguments=(ParseReference("object", segment.id),),
                    expression=expression,
                    origin=ParseOrigin.IMAGE,
                    confidence=_quantity_confidence(label, segment, distance, max_distance),
                    score_kind="ocr_segment_spatial_margin",
                    raw_text=raw_text,
                    evidence_refs=_candidate_evidence(label, segment, evidence),
                    dependencies=(label.id,),
                    metadata={
                        "binding_kind": "segment_length",
                        "label_object_id": label.id,
                        "binding_status": "ambiguous" if len(selected) > 1 else "proposed",
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
                    id=f"conflict:{label.id}:quantity_binding",
                    members=tuple(member_ids),
                    reason="multiple_segments_within_spatial_margin",
                )
            )
            issues.append(
                {
                    "label_id": label.id,
                    "raw_text": raw_text,
                    "status": "object_ambiguous",
                    "candidate_ids": list(member_ids),
                }
            )
    return tuple(quantities), tuple(conflicts), tuple(issues)


def _rank_segments(
    label: ParseObject,
    segments: Sequence[ParseObject],
    object_index: Mapping[str, ParseObject],
    *,
    maximum_distance: float,
    minimum_projection: float,
    maximum_projection: float,
) -> tuple[tuple[float, float, ParseObject], ...]:
    center = _label_center(label)
    if center is None:
        return ()
    ranked = []
    for segment in segments:
        points = _segment_points(segment, object_index)
        if points is None:
            continue
        distance, projection = _point_segment_distance(center, *points)
        if distance <= maximum_distance and minimum_projection <= projection <= maximum_projection:
            ranked.append((distance, projection, segment))
    return tuple(sorted(ranked, key=lambda item: (item[0], item[2].id)))


def _segment_points(
    candidate: ParseObject,
    object_index: Mapping[str, ParseObject],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    p1 = candidate.geometry.get("p1")
    p2 = candidate.geometry.get("p2")
    if _is_point(p1) and _is_point(p2):
        return (float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))
    refs = {
        reference.role: object_index.get(reference.ref)
        for reference in candidate.references
        if reference.role in {"start", "end"}
    }
    start = _point_position(refs.get("start"))
    end = _point_position(refs.get("end"))
    return (start, end) if start is not None and end is not None else None


def _point_position(candidate: ParseObject | None) -> tuple[float, float] | None:
    value = candidate.geometry.get("pixel") if candidate is not None else None
    if not _is_point(value):
        return None
    return float(value[0]), float(value[1])


def _label_center(candidate: ParseObject) -> tuple[float, float] | None:
    bbox = candidate.geometry.get("bbox")
    if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) != 4:
        return None
    left, top, right, bottom = (float(value) for value in bbox)
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


def _quantity_confidence(
    label: ParseObject,
    segment: ParseObject,
    distance: float,
    max_distance: float,
) -> float:
    spatial_score = max(0.0, 1.0 - distance / max_distance)
    return round(min(label.confidence, segment.confidence, 0.5 + 0.5 * spatial_score), 3)


def _candidate_evidence(
    label: ParseObject,
    segment: ParseObject,
    evidence: Mapping[str, ParseEvidence],
) -> tuple[str, ...]:
    refs = tuple(dict.fromkeys((*label.evidence_refs, *segment.evidence_refs)))
    missing = sorted(set(refs) - set(evidence))
    if missing:
        raise InvalidGeometryInput(f"Image quantity references unknown evidence: {missing}")
    return refs


def _is_point(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
        and all(isinstance(component, (int, float)) and math.isfinite(float(component)) for component in value)
    )
