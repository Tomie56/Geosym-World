"""Renderer-neutral geometry for visible relation markers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .layout import ViewTransform
from .scene import Point2D, RenderPrimitive


MARKER_KINDS = frozenset({"angle_marker", "right_angle_marker", "tick_marker", "parallel_marker"})


def marker_world_points(primitive: RenderPrimitive, *, angle_steps: int = 24) -> tuple[Point2D, ...]:
    """Return the world-space polyline actually drawn for a marker."""

    if angle_steps <= 0:
        raise ValueError("angle_steps must be positive")
    geometry = primitive.geometry
    if primitive.kind == "angle_marker":
        return _angle_points(geometry, angle_steps)
    if primitive.kind == "right_angle_marker":
        return _right_angle_points(geometry)
    if primitive.kind == "tick_marker":
        return _tick_points(geometry)
    if primitive.kind == "parallel_marker":
        return tuple(_point(point) for point in geometry["points"])
    raise ValueError(f"Unsupported marker primitive kind: {primitive.kind!r}")


def marker_pixel_bbox(
    primitive: RenderPrimitive,
    transform: ViewTransform,
    *,
    padding: float = 0.0,
) -> tuple[float, float, float, float]:
    """Return the marker's pixel-space path bounds."""

    if padding < 0.0 or not math.isfinite(padding):
        raise ValueError("Marker bbox padding must be finite and non-negative")
    points = tuple(transform.point(point) for point in marker_world_points(primitive))
    return (
        min(point[0] for point in points) - padding,
        min(point[1] for point in points) - padding,
        max(point[0] for point in points) + padding,
        max(point[1] for point in points) + padding,
    )


def _angle_points(geometry: Mapping[str, Any], steps: int) -> tuple[Point2D, ...]:
    vertex = _point(geometry["vertex"])
    arm1 = _point(geometry["arm1"])
    arm2 = _point(geometry["arm2"])
    radius = float(geometry.get("radius", 0.22))
    start_angle = math.atan2(arm1[1] - vertex[1], arm1[0] - vertex[0])
    end_angle = math.atan2(arm2[1] - vertex[1], arm2[0] - vertex[0])
    orientation = geometry.get("orientation")
    if orientation is None:
        delta = (end_angle - start_angle + math.pi) % (2 * math.pi) - math.pi
    elif orientation == "ccw":
        delta = (end_angle - start_angle) % (2 * math.pi)
    elif orientation == "cw":
        delta = -((start_angle - end_angle) % (2 * math.pi))
    else:
        raise ValueError(f"Unsupported marker orientation: {orientation!r}")
    return tuple(
        (
            vertex[0] + radius * math.cos(start_angle + delta * index / steps),
            vertex[1] + radius * math.sin(start_angle + delta * index / steps),
        )
        for index in range(steps + 1)
    )


def _right_angle_points(geometry: Mapping[str, Any]) -> tuple[Point2D, ...]:
    vertex = _point(geometry["vertex"])
    first = _unit(_direction(geometry, "arm1", "direction1", vertex))
    second = _unit(_direction(geometry, "arm2", "direction2", vertex))
    size = float(geometry.get("size", 0.18))
    first_corner = (vertex[0] + first[0] * size, vertex[1] + first[1] * size)
    outer_corner = (first_corner[0] + second[0] * size, first_corner[1] + second[1] * size)
    second_corner = (vertex[0] + second[0] * size, vertex[1] + second[1] * size)
    return first_corner, outer_corner, second_corner


def _tick_points(geometry: Mapping[str, Any]) -> tuple[Point2D, Point2D]:
    start = _point(geometry["start"])
    end = _point(geometry["end"])
    size = float(geometry.get("size", 0.08))
    midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
    direction = _unit((end[0] - start[0], end[1] - start[1]))
    normal = (-direction[1], direction[0])
    return (
        midpoint[0] - normal[0] * size,
        midpoint[1] - normal[1] * size,
    ), (
        midpoint[0] + normal[0] * size,
        midpoint[1] + normal[1] * size,
    )


def _direction(
    geometry: Mapping[str, Any],
    arm_key: str,
    direction_key: str,
    vertex: Point2D,
) -> Point2D:
    if direction_key in geometry:
        return _point(geometry[direction_key])
    arm = _point(geometry[arm_key])
    return arm[0] - vertex[0], arm[1] - vertex[1]


def _point(value: Any) -> Point2D:
    return float(value[0]), float(value[1])


def _unit(direction: Point2D) -> Point2D:
    norm = math.hypot(direction[0], direction[1])
    if norm < 1e-12:
        return 1.0, 0.0
    return direction[0] / norm, direction[1] / norm
