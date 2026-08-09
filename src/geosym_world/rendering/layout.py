"""Viewport and coordinate layout helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

from .scene import Point2D, RenderPrimitive, RenderScene


@dataclass(frozen=True)
class WorldBounds:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    @property
    def span(self) -> float:
        return max(self.width, self.height)


@dataclass(frozen=True)
class ViewTransform:
    width: int
    height: int
    padding: int
    bounds: WorldBounds
    scale: float

    def point(self, xy: Point2D) -> Point2D:
        x, y = xy
        px = self.padding + (x - self.bounds.min_x) * self.scale
        py = self.height - self.padding - (y - self.bounds.min_y) * self.scale
        return px, py

    def length(self, value: float) -> float:
        return value * self.scale

    def world_rect(self) -> tuple[float, float, float, float]:
        return self.bounds.min_x, self.bounds.min_y, self.bounds.max_x, self.bounds.max_y


def build_view_transform(scene: RenderScene, *, width: int = 768, height: int = 768, padding: int = 48) -> ViewTransform:
    bounds = compute_world_bounds(scene)
    usable_width = max(1, width - 2 * padding)
    usable_height = max(1, height - 2 * padding)
    scale = min(usable_width / bounds.width, usable_height / bounds.height)
    return ViewTransform(width=width, height=height, padding=padding, bounds=bounds, scale=scale)


def compute_world_bounds(scene: RenderScene) -> WorldBounds:
    xs: list[float] = []
    ys: list[float] = []
    for primitive in scene.primitives:
        for x, y in _primitive_points(primitive):
            xs.append(float(x))
            ys.append(float(y))
    for label in scene.labels:
        xs.append(float(label.position[0]))
        ys.append(float(label.position[1]))
    if not xs or not ys:
        return WorldBounds(-1.0, -1.0, 1.0, 1.0)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if max_x - min_x < 1e-9:
        min_x -= 1.0
        max_x += 1.0
    if max_y - min_y < 1e-9:
        min_y -= 1.0
        max_y += 1.0
    margin = 0.08 * max(max_x - min_x, max_y - min_y, 1.0)
    return WorldBounds(min_x - margin, min_y - margin, max_x + margin, max_y + margin)


def _primitive_points(primitive: RenderPrimitive) -> Iterable[Point2D]:
    geometry = primitive.geometry
    kind = primitive.kind
    if kind == "point":
        yield tuple(geometry["point"])  # type: ignore[misc]
    elif kind in {"segment", "line", "ray", "polyline", "polygon"}:
        points = geometry.get("points")
        if points:
            for point in points:
                yield tuple(point)  # type: ignore[misc]
        else:
            if "point" in geometry:
                yield tuple(geometry["point"])  # type: ignore[misc]
    elif kind in {"circle", "arc", "sector", "annulus"}:
        center = tuple(geometry["center"])
        radius = float(geometry.get("radius", geometry.get("outer_radius")))
        yield center[0] - radius, center[1] - radius
        yield center[0] + radius, center[1] + radius
    elif kind in {"angle_marker", "right_angle_marker"}:
        yield tuple(geometry["vertex"])  # type: ignore[misc]
        radius = float(geometry.get("radius", 0.25))
        vertex = tuple(geometry["vertex"])
        yield vertex[0] - radius, vertex[1] - radius
        yield vertex[0] + radius, vertex[1] + radius
    elif kind == "tick_marker":
        yield tuple(geometry["start"])  # type: ignore[misc]
        yield tuple(geometry["end"])  # type: ignore[misc]
