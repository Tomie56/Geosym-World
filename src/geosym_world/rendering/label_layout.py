"""Deterministic pixel-space label placement shared by SVG and PNG renderers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from .layout import ViewTransform
from .marker_geometry import MARKER_KINDS, marker_pixel_bbox
from .scene import Point2D, RenderLabel, RenderPrimitive, RenderScene

Box = tuple[float, float, float, float]
Segment = tuple[Point2D, Point2D]


@dataclass(frozen=True)
class PositionedLabel:
    label: RenderLabel
    x: float
    y: float
    bbox: Box


def position_labels(scene: RenderScene, transform: ViewTransform) -> tuple[PositionedLabel, ...]:
    point_boxes, marker_boxes, hole_boxes, segments = _obstacles(scene.primitives, transform)
    placed: dict[str, PositionedLabel] = {}
    occupied: list[Box] = []
    ordered = sorted(scene.labels, key=lambda label: (label.id.startswith("measure:"), label.id))
    for label in ordered:
        anchor = transform.point(label.position)
        width = max(11.0, 10.5 * len(label.text))
        candidates = _candidate_offsets(label, width)
        best = min(
            candidates,
            key=lambda offset: _candidate_score(
                anchor,
                offset,
                width,
                transform,
                occupied,
                point_boxes,
                marker_boxes,
                hole_boxes,
                segments,
                label.offset,
            ),
        )
        x, y = anchor[0] + best[0], anchor[1] + best[1]
        positioned = PositionedLabel(label=label, x=x, y=y, bbox=_text_box(x, y, width))
        placed[label.id] = positioned
        occupied.append(positioned.bbox)
    return tuple(placed[label.id] for label in scene.labels)


def _candidate_offsets(label: RenderLabel, width: float) -> tuple[Point2D, ...]:
    preferred = (float(label.offset[0]), float(label.offset[1]))
    if label.id.startswith("label:"):
        generic = [
            (12.0, -14.0),
            (-width - 12.0, -14.0),
            (12.0, 21.0),
            (-width - 12.0, 21.0),
            (-width / 2.0, -22.0),
            (-width / 2.0, 28.0),
            (16.0, 5.0),
            (-width - 16.0, 5.0),
            (-width / 2.0, -34.0),
            (-width / 2.0, 40.0),
            (42.0, -14.0),
            (-width - 42.0, -14.0),
            (42.0, 21.0),
            (-width - 42.0, 21.0),
        ]
    else:
        dx, dy = preferred
        generic = [
            (-dx, -dy),
            (dx + 14.0, dy),
            (dx - 14.0, dy),
            (-dx + 14.0, -dy),
            (-dx - 14.0, -dy),
            (-width / 2.0, -20.0),
            (-width / 2.0, 25.0),
            (-width / 2.0, -34.0),
            (-width / 2.0, 40.0),
            (-width / 2.0, -50.0),
            (-width / 2.0, 56.0),
            (48.0, -14.0),
            (-width - 48.0, -14.0),
            (48.0, 21.0),
            (-width - 48.0, 21.0),
        ]
    generic.sort(key=lambda item: (math.hypot(item[0] - preferred[0], item[1] - preferred[1]), item[0], item[1]))
    unique: list[Point2D] = []
    for candidate in (preferred, *generic):
        if candidate not in unique:
            unique.append(candidate)
    return tuple(unique)


def _candidate_score(
    anchor: Point2D,
    offset: Point2D,
    width: float,
    transform: ViewTransform,
    occupied: Iterable[Box],
    point_boxes: Iterable[Box],
    marker_boxes: Iterable[Box],
    hole_boxes: Iterable[Box],
    segments: Iterable[Segment],
    preferred: Point2D,
) -> float:
    x, y = anchor[0] + offset[0], anchor[1] + offset[1]
    box = _text_box(x, y, width)
    score = 0.12 * math.hypot(offset[0] - preferred[0], offset[1] - preferred[1])
    viewport = (5.0, 5.0, transform.width - 5.0, transform.height - 5.0)
    if not _contains(viewport, box):
        score += 100000.0 + 100.0 * _outside_area(box, viewport)
    for other in occupied:
        score += 10000.0 * _overlap_area(box, other)
        score += 400.0 * _overlap_area(_expand(box, 4.0), _expand(other, 4.0))
    for obstacle in point_boxes:
        score += 1800.0 * _overlap_area(box, obstacle)
    for obstacle in marker_boxes:
        score += 1200.0 * _overlap_area(box, obstacle)
    for obstacle in hole_boxes:
        score += 2500.0 * _overlap_area(box, obstacle)
    for segment in segments:
        if _segment_intersects_box(segment, _expand(box, 2.0)):
            score += 4000.0
    return score


def _obstacles(primitives: Iterable[RenderPrimitive], transform: ViewTransform) -> tuple[list[Box], list[Box], list[Box], list[Segment]]:
    point_boxes: list[Box] = []
    marker_boxes: list[Box] = []
    hole_boxes: list[Box] = []
    segments: list[Segment] = []
    for primitive in primitives:
        geometry = primitive.geometry
        if primitive.kind == "point":
            x, y = transform.point(tuple(geometry["point"]))
            radius = float(geometry.get("radius", 4.0)) + 4.0
            point_boxes.append((x - radius, y - radius, x + radius, y + radius))
        elif primitive.kind == "segment":
            a, b = [transform.point(tuple(point)) for point in geometry["points"]]
            segments.append((a, b))
        elif primitive.kind == "polygon":
            points = [transform.point(tuple(point)) for point in geometry["points"]]
            segments.extend((points[index], points[(index + 1) % len(points)]) for index in range(len(points)))
            for hole in geometry.get("holes", ()):
                hole_points = [transform.point(tuple(point)) for point in hole]
                hole_boxes.append((
                    min(point[0] for point in hole_points),
                    min(point[1] for point in hole_points),
                    max(point[0] for point in hole_points),
                    max(point[1] for point in hole_points),
                ))
        elif primitive.kind in {"line", "ray"}:
            point = tuple(geometry["point"])
            direction = _unit(tuple(geometry["direction"]))
            span = max(transform.bounds.span * 4.0, 2.0)
            start = point if primitive.kind == "ray" else (point[0] - direction[0] * span, point[1] - direction[1] * span)
            end = (point[0] + direction[0] * span, point[1] + direction[1] * span)
            segments.append((transform.point(start), transform.point(end)))
        elif primitive.kind in MARKER_KINDS:
            marker_boxes.append(marker_pixel_bbox(primitive, transform, padding=5.0))
    return point_boxes, marker_boxes, hole_boxes, segments


def _text_box(x: float, baseline_y: float, width: float) -> Box:
    return x - 2.0, baseline_y - 16.0, x + width + 2.0, baseline_y + 5.0


def _expand(box: Box, amount: float) -> Box:
    return box[0] - amount, box[1] - amount, box[2] + amount, box[3] + amount


def _contains(outer: Box, inner: Box) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def _overlap_area(left: Box, right: Box) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    return width * height


def _outside_area(box: Box, viewport: Box) -> float:
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    return max(0.0, area - _overlap_area(box, viewport))


def _segment_intersects_box(segment: Segment, box: Box) -> bool:
    (x1, y1), (x2, y2) = segment
    dx, dy = x2 - x1, y2 - y1
    p = (-dx, dx, -dy, dy)
    q = (x1 - box[0], box[2] - x1, y1 - box[1], box[3] - y1)
    lower, upper = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0:
                return False
            continue
        ratio = qi / pi
        if pi < 0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower > upper:
            return False
    return True


def _unit(direction: Point2D) -> Point2D:
    norm = math.hypot(direction[0], direction[1])
    if norm < 1e-12:
        return 1.0, 0.0
    return direction[0] / norm, direction[1] / norm
