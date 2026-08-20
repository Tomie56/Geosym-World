"""SVG renderer for RenderScene."""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Iterable

from .label_layout import position_labels
from .layout import ViewTransform, build_view_transform
from .scene import Point2D, RenderPrimitive, RenderScene, RenderStyle


def write_svg(scene: RenderScene, path: str | Path, *, width: int = 768, height: int = 768, padding: int = 48) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_svg(scene, width=width, height=height, padding=padding), encoding="utf-8")


def render_svg(scene: RenderScene, *, width: int = 768, height: int = 768, padding: int = 48) -> str:
    transform = build_view_transform(scene, width=width, height=height, padding=padding)
    fingerprint = str(scene.metadata.get("scene_fingerprint", ""))
    body: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" data-scene-fingerprint="{html.escape(fingerprint)}">',
        f"<title>{html.escape(scene.id)}</title>",
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
    ]
    for primitive in scene.primitives:
        body.extend(_render_primitive(primitive, transform))
    for positioned in position_labels(scene, transform):
        label = positioned.label
        x, y = positioned.x, positioned.y
        body.append(
            f'<text data-id="{html.escape(label.id)}:halo" x="{_fmt(x)}" y="{_fmt(y)}" font-size="15" font-family="Inter, Arial, sans-serif" '
            f'fill="#ffffff" stroke="#ffffff" stroke-width="4" stroke-linejoin="round">{html.escape(label.text)}</text>'
        )
        body.append(
            f'<text data-id="{html.escape(label.id)}" x="{_fmt(x)}" y="{_fmt(y)}" font-size="15" font-family="Inter, Arial, sans-serif" '
            f'fill="{html.escape(label.style.fill)}">{html.escape(label.text)}</text>'
        )
    body.append("</svg>")
    return "\n".join(body) + "\n"


def _render_primitive(primitive: RenderPrimitive, transform: ViewTransform) -> Iterable[str]:
    kind = primitive.kind
    geometry = primitive.geometry
    style = primitive.style
    if kind == "point":
        x, y = transform.point(tuple(geometry["point"]))
        radius = float(geometry.get("radius", 4.0))
        yield f'<circle data-id="{html.escape(primitive.id)}" cx="{_fmt(x)}" cy="{_fmt(y)}" r="{_fmt(radius)}" {_style(style)}/>'
    elif kind == "segment":
        a, b = [transform.point(tuple(point)) for point in geometry["points"]]
        yield _line_element(primitive.id, a, b, style)
    elif kind == "line":
        a, b = _line_endpoints(tuple(geometry["point"]), tuple(geometry["direction"]), transform)
        yield _line_element(primitive.id, transform.point(a), transform.point(b), style)
    elif kind == "ray":
        a, b = _ray_endpoints(tuple(geometry["point"]), tuple(geometry["direction"]), transform)
        yield _line_element(primitive.id, transform.point(a), transform.point(b), style)
    elif kind == "polygon":
        holes = geometry.get("holes", ())
        components = geometry.get("components", ())
        if not holes and not components:
            points = " ".join(_point_attr(transform.point(tuple(point))) for point in geometry["points"])
            yield f'<polygon data-id="{html.escape(primitive.id)}" points="{points}" {_style(style)}/>'
        else:
            rings = [geometry["points"], *components, *holes]
            path = " ".join(
                "M " + " L ".join(_point_attr(transform.point(tuple(point))).replace(",", " ") for point in ring) + " Z"
                for ring in rings
            )
            yield f'<path data-id="{html.escape(primitive.id)}" d="{path}" fill-rule="evenodd" {_style(style)}/>'
    elif kind == "circle":
        x, y = transform.point(tuple(geometry["center"]))
        radius = transform.length(float(geometry["radius"]))
        yield f'<circle data-id="{html.escape(primitive.id)}" cx="{_fmt(x)}" cy="{_fmt(y)}" r="{_fmt(radius)}" {_style(style)}/>'
    elif kind == "annulus":
        x, y = transform.point(tuple(geometry["center"]))
        outer = transform.length(float(geometry["outer_radius"]))
        inner = transform.length(float(geometry["inner_radius"]))
        path = " ".join(
            (
                _circle_path(x, y, outer),
                _circle_path(x, y, inner),
            )
        )
        yield f'<path data-id="{html.escape(primitive.id)}" d="{path}" fill-rule="evenodd" {_style(style)}/>'
    elif kind == "arc":
        yield _arc_path(primitive, transform, closed=False)
    elif kind == "sector":
        yield _arc_path(primitive, transform, closed=True)
    elif kind == "circular_segment":
        yield _circular_segment_path(primitive, transform)
    elif kind == "lens":
        yield _lens_path(primitive, transform)
    elif kind == "angle_marker":
        yield _angle_marker_path(primitive, transform)
    elif kind == "right_angle_marker":
        yield _right_angle_marker_path(primitive, transform)
    elif kind == "tick_marker":
        yield _tick_marker_path(primitive, transform)


def _line_element(object_id: str, a: Point2D, b: Point2D, style: RenderStyle) -> str:
    return (
        f'<line data-id="{html.escape(object_id)}" x1="{_fmt(a[0])}" y1="{_fmt(a[1])}" '
        f'x2="{_fmt(b[0])}" y2="{_fmt(b[1])}" {_style(style)}/>'
    )


def _circle_path(x: float, y: float, radius: float) -> str:
    return (
        f"M {_fmt(x - radius)} {_fmt(y)} "
        f"A {_fmt(radius)} {_fmt(radius)} 0 1 0 {_fmt(x + radius)} {_fmt(y)} "
        f"A {_fmt(radius)} {_fmt(radius)} 0 1 0 {_fmt(x - radius)} {_fmt(y)} Z"
    )


def _arc_path(primitive: RenderPrimitive, transform: ViewTransform, *, closed: bool) -> str:
    geometry = primitive.geometry
    center = tuple(geometry["center"])
    start = tuple(geometry["start"])
    end = tuple(geometry["end"])
    radius = transform.length(float(geometry["radius"]))
    sx, sy = transform.point(start)
    ex, ey = transform.point(end)
    cx, cy = transform.point(center)
    large_arc = 1 if _arc_delta(center, start, end, geometry.get("orientation", "ccw")) > math.pi else 0
    sweep = 0 if geometry.get("orientation", "ccw") == "ccw" else 1
    if not closed:
        path = f"M {_fmt(sx)} {_fmt(sy)} A {_fmt(radius)} {_fmt(radius)} 0 {large_arc} {sweep} {_fmt(ex)} {_fmt(ey)}"
    else:
        path = (
            f"M {_fmt(cx)} {_fmt(cy)} L {_fmt(sx)} {_fmt(sy)} "
            f"A {_fmt(radius)} {_fmt(radius)} 0 {large_arc} {sweep} {_fmt(ex)} {_fmt(ey)} Z"
        )
    return f'<path data-id="{html.escape(primitive.id)}" d="{path}" {_style(primitive.style)}/>'


def _circular_segment_path(primitive: RenderPrimitive, transform: ViewTransform) -> str:
    geometry = primitive.geometry
    center = tuple(geometry["center"])
    start = tuple(geometry["start"])
    end = tuple(geometry["end"])
    radius = transform.length(float(geometry["radius"]))
    sx, sy = transform.point(start)
    ex, ey = transform.point(end)
    large_arc = 1 if _arc_delta(center, start, end, geometry.get("orientation", "ccw")) > math.pi else 0
    sweep = 0 if geometry.get("orientation", "ccw") == "ccw" else 1
    path = f"M {_fmt(sx)} {_fmt(sy)} A {_fmt(radius)} {_fmt(radius)} 0 {large_arc} {sweep} {_fmt(ex)} {_fmt(ey)} Z"
    return f'<path data-id="{html.escape(primitive.id)}" d="{path}" {_style(primitive.style)}/>'


def _lens_path(primitive: RenderPrimitive, transform: ViewTransform) -> str:
    arcs = primitive.geometry.get("arcs", ())
    if len(arcs) != 2:
        return ""
    first, second = arcs
    start = transform.point(tuple(first["start"]))
    path_parts = [f"M {_fmt(start[0])} {_fmt(start[1])}"]
    for arc in (first, second):
        center = tuple(arc["center"])
        arc_start = tuple(arc["start"])
        arc_end = tuple(arc["end"])
        end = transform.point(arc_end)
        radius = transform.length(float(arc["radius"]))
        large_arc = 1 if _arc_delta(center, arc_start, arc_end, arc.get("orientation", "ccw")) > math.pi else 0
        sweep = 0 if arc.get("orientation", "ccw") == "ccw" else 1
        path_parts.append(f"A {_fmt(radius)} {_fmt(radius)} 0 {large_arc} {sweep} {_fmt(end[0])} {_fmt(end[1])}")
    path_parts.append("Z")
    return f'<path data-id="{html.escape(primitive.id)}" d="{" ".join(path_parts)}" {_style(primitive.style)}/>'


def _line_endpoints(point: Point2D, direction: Point2D, transform: ViewTransform) -> tuple[Point2D, Point2D]:
    dx, dy = _unit(direction)
    span = max(transform.bounds.span * 4.0, 2.0)
    a = (point[0] - dx * span, point[1] - dy * span)
    b = (point[0] + dx * span, point[1] + dy * span)
    return _clip_segment(a, b, transform) or (a, b)


def _ray_endpoints(point: Point2D, direction: Point2D, transform: ViewTransform) -> tuple[Point2D, Point2D]:
    dx, dy = _unit(direction)
    span = max(transform.bounds.span * 4.0, 2.0)
    end = (point[0] + dx * span, point[1] + dy * span)
    return _clip_segment(point, end, transform) or (point, end)


def _angle_marker_path(primitive: RenderPrimitive, transform: ViewTransform) -> str:
    geometry = primitive.geometry
    vertex = tuple(geometry["vertex"])
    arm1 = tuple(geometry["arm1"])
    arm2 = tuple(geometry["arm2"])
    radius = float(geometry.get("radius", 0.22))
    points = [_arc_point_world(vertex, arm1, arm2, radius, idx / 24, orientation=geometry.get("orientation")) for idx in range(25)]
    path = "M " + " L ".join(_point_attr(transform.point(point)).replace(",", " ") for point in points)
    return f'<path data-id="{html.escape(primitive.id)}" d="{path}" {_style(primitive.style)}/>'


def _right_angle_marker_path(primitive: RenderPrimitive, transform: ViewTransform) -> str:
    geometry = primitive.geometry
    vertex = tuple(geometry["vertex"])
    u = _unit(_direction_from_geometry(geometry, "arm1", "direction1", vertex))
    v = _unit(_direction_from_geometry(geometry, "arm2", "direction2", vertex))
    size = float(geometry.get("size", 0.18))
    p1 = (vertex[0] + u[0] * size, vertex[1] + u[1] * size)
    p2 = (p1[0] + v[0] * size, p1[1] + v[1] * size)
    p3 = (vertex[0] + v[0] * size, vertex[1] + v[1] * size)
    d = "M " + " L ".join(_point_attr(transform.point(point)).replace(",", " ") for point in (p1, p2, p3))
    return f'<path data-id="{html.escape(primitive.id)}" d="{d}" {_style(primitive.style)}/>'


def _tick_marker_path(primitive: RenderPrimitive, transform: ViewTransform) -> str:
    geometry = primitive.geometry
    start = tuple(geometry["start"])
    end = tuple(geometry["end"])
    size = float(geometry.get("size", 0.08))
    mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    dx, dy = _unit((end[0] - start[0], end[1] - start[1]))
    normal = (-dy, dx)
    a = (mid[0] - normal[0] * size, mid[1] - normal[1] * size)
    b = (mid[0] + normal[0] * size, mid[1] + normal[1] * size)
    return _line_element(primitive.id, transform.point(a), transform.point(b), primitive.style)


def _arc_point_world(vertex: Point2D, arm1: Point2D, arm2: Point2D, radius: float, t: float, *, orientation: str | None = None) -> Point2D:
    a0 = math.atan2(arm1[1] - vertex[1], arm1[0] - vertex[0])
    a1 = math.atan2(arm2[1] - vertex[1], arm2[0] - vertex[0])
    if orientation is None:
        delta = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
    elif orientation == "ccw":
        delta = (a1 - a0) % (2 * math.pi)
    else:
        delta = -((a0 - a1) % (2 * math.pi))
    angle = a0 + delta * t
    return vertex[0] + radius * math.cos(angle), vertex[1] + radius * math.sin(angle)


def _direction_from_geometry(geometry: dict, arm_key: str, direction_key: str, vertex: Point2D) -> Point2D:
    if direction_key in geometry:
        return tuple(geometry[direction_key])
    arm = tuple(geometry[arm_key])
    return arm[0] - vertex[0], arm[1] - vertex[1]


def _clip_segment(a: Point2D, b: Point2D, transform: ViewTransform) -> tuple[Point2D, Point2D] | None:
    min_x, min_y, max_x, max_y = transform.world_rect()
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    p = [-dx, dx, -dy, dy]
    q = [a[0] - min_x, max_x - a[0], a[1] - min_y, max_y - a[1]]
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0:
                return None
            continue
        r = qi / pi
        if pi < 0:
            u1 = max(u1, r)
        else:
            u2 = min(u2, r)
        if u1 > u2:
            return None
    return (a[0] + u1 * dx, a[1] + u1 * dy), (a[0] + u2 * dx, a[1] + u2 * dy)


def _arc_delta(center: Point2D, start: Point2D, end: Point2D, orientation: str) -> float:
    a0 = math.atan2(start[1] - center[1], start[0] - center[0])
    a1 = math.atan2(end[1] - center[1], end[0] - center[0])
    delta = (a1 - a0) % (2 * math.pi)
    if orientation != "ccw":
        delta = (a0 - a1) % (2 * math.pi)
    return delta


def _unit(direction: Point2D) -> Point2D:
    norm = math.hypot(direction[0], direction[1])
    if norm < 1e-12:
        return 1.0, 0.0
    return direction[0] / norm, direction[1] / norm


def _style(style: RenderStyle) -> str:
    dash = f' stroke-dasharray="{",".join(_fmt(v) for v in style.dasharray)}"' if style.dasharray else ""
    return (
        f'stroke="{html.escape(style.stroke)}" fill="{html.escape(style.fill)}" '
        f'stroke-width="{_fmt(style.stroke_width)}" opacity="{_fmt(style.opacity)}" '
        f'stroke-linecap="round" stroke-linejoin="round"{dash}'
    )


def _point_attr(point: Point2D) -> str:
    return f"{_fmt(point[0])},{_fmt(point[1])}"


def _fmt(value: float) -> str:
    return f"{value:.6g}"
