"""OpenCV PNG renderer for RenderScene."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

from .label_layout import position_labels
from .layout import ViewTransform, build_view_transform
from .scene import Point2D, RenderPrimitive, RenderScene, RenderStyle


def write_png(scene: RenderScene, path: str | Path, *, width: int = 768, height: int = 768, padding: int = 48) -> None:
    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("PNG rendering requires opencv-python, numpy, and Pillow") from exc

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transform = build_view_transform(scene, width=width, height=height, padding=padding)
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    for primitive in scene.primitives:
        _draw_primitive(cv2, np, image, primitive, transform)
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font = _label_font(ImageFont)
    for positioned in position_labels(scene, transform):
        label = positioned.label
        draw.text(
            (positioned.x, positioned.y),
            label.text,
            font=font,
            fill=_rgb(label.style.fill),
            stroke_width=3,
            stroke_fill=(255, 255, 255),
            anchor="ls",
        )
    image = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Failed to write PNG: {output_path}")


def _label_font(ImageFont):
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, 16)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_primitive(cv2, np, image, primitive: RenderPrimitive, transform: ViewTransform) -> None:
    kind = primitive.kind
    geometry = primitive.geometry
    style = primitive.style
    color = _bgr(style.stroke)
    fill = _bgr(style.fill)
    thickness = max(1, int(round(style.stroke_width)))
    if kind == "point":
        center = _pixel(transform.point(tuple(geometry["point"])))
        radius = max(2, int(round(float(geometry.get("radius", 4.0)))))
        cv2.circle(image, center, radius, fill if style.fill != "none" else (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, center, radius, color, thickness, cv2.LINE_AA)
    elif kind == "segment":
        a, b = [_pixel(transform.point(tuple(point))) for point in geometry["points"]]
        _draw_styled_line(cv2, image, a, b, color, thickness, style.dasharray)
    elif kind == "line":
        a, b = _line_endpoints(tuple(geometry["point"]), tuple(geometry["direction"]), transform)
        _draw_styled_line(cv2, image, _pixel(transform.point(a)), _pixel(transform.point(b)), color, thickness, style.dasharray)
    elif kind == "ray":
        a, b = _ray_endpoints(tuple(geometry["point"]), tuple(geometry["direction"]), transform)
        _draw_styled_line(cv2, image, _pixel(transform.point(a)), _pixel(transform.point(b)), color, thickness, style.dasharray)
    elif kind == "polygon":
        points = np.array([_pixel(transform.point(tuple(point))) for point in geometry["points"]], dtype=np.int32)
        if style.fill != "none":
            overlay = image.copy()
            cv2.fillPoly(overlay, [points], fill, cv2.LINE_AA)
            alpha = max(0.0, min(1.0, style.opacity))
            cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, image)
        cv2.polylines(image, [points], True, color, thickness, cv2.LINE_AA)
    elif kind == "circle":
        center = _pixel(transform.point(tuple(geometry["center"])))
        radius = max(1, int(round(transform.length(float(geometry["radius"])))))
        cv2.circle(image, center, radius, color, thickness, cv2.LINE_AA)
    elif kind == "annulus":
        center = _pixel(transform.point(tuple(geometry["center"])))
        outer = max(1, int(round(transform.length(float(geometry["outer_radius"])))))
        inner = max(1, int(round(transform.length(float(geometry["inner_radius"])))))
        overlay = image.copy()
        cv2.circle(overlay, center, outer, fill, -1, cv2.LINE_AA)
        alpha = max(0.0, min(1.0, style.opacity))
        cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, image)
        cv2.circle(image, center, inner, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, center, outer, color, thickness, cv2.LINE_AA)
        cv2.circle(image, center, inner, color, thickness, cv2.LINE_AA)
    elif kind == "arc":
        points = np.array([_pixel(transform.point(point)) for point in _arc_points(geometry, closed=False)], dtype=np.int32)
        cv2.polylines(image, [points], False, color, thickness, cv2.LINE_AA)
    elif kind == "sector":
        points = np.array([_pixel(transform.point(point)) for point in _arc_points(geometry, closed=True)], dtype=np.int32)
        if style.fill != "none":
            overlay = image.copy()
            cv2.fillPoly(overlay, [points], fill, cv2.LINE_AA)
            alpha = max(0.0, min(1.0, style.opacity))
            cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, image)
        cv2.polylines(image, [points], True, color, thickness, cv2.LINE_AA)
    elif kind == "angle_marker":
        points = np.array([_pixel(transform.point(point)) for point in _angle_marker_points(geometry)], dtype=np.int32)
        cv2.polylines(image, [points], False, color, thickness, cv2.LINE_AA)
    elif kind == "right_angle_marker":
        points = np.array([_pixel(transform.point(point)) for point in _right_angle_points(geometry)], dtype=np.int32)
        cv2.polylines(image, [points], False, color, thickness, cv2.LINE_AA)
    elif kind == "tick_marker":
        a, b = _tick_points(geometry)
        _draw_styled_line(cv2, image, _pixel(transform.point(a)), _pixel(transform.point(b)), color, thickness, ())


def _arc_points(geometry: dict, *, closed: bool, steps: int = 64) -> Iterable[Point2D]:
    center = tuple(geometry["center"])
    start = tuple(geometry["start"])
    end = tuple(geometry["end"])
    radius = float(geometry["radius"])
    a0 = math.atan2(start[1] - center[1], start[0] - center[0])
    a1 = math.atan2(end[1] - center[1], end[0] - center[0])
    if geometry.get("orientation", "ccw") == "ccw":
        delta = (a1 - a0) % (2 * math.pi)
    else:
        delta = -((a0 - a1) % (2 * math.pi))
    points: list[Point2D] = []
    if closed:
        points.append(center)
    for idx in range(steps + 1):
        angle = a0 + delta * idx / steps
        points.append((center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle)))
    if closed:
        points.append(center)
    return points


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


def _draw_styled_line(cv2, image, a: tuple[int, int], b: tuple[int, int], color: tuple[int, int, int], thickness: int, dasharray: tuple[float, ...]) -> None:
    if not dasharray:
        cv2.line(image, a, b, color, thickness, cv2.LINE_AA)
        return
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < 1:
        return
    pattern = [max(1.0, float(value)) for value in dasharray]
    distance = 0.0
    pattern_idx = 0
    draw = True
    while distance < length:
        step = pattern[pattern_idx % len(pattern)]
        next_distance = min(length, distance + step)
        if draw:
            start = (int(round(a[0] + dx * distance / length)), int(round(a[1] + dy * distance / length)))
            end = (int(round(a[0] + dx * next_distance / length)), int(round(a[1] + dy * next_distance / length)))
            cv2.line(image, start, end, color, thickness, cv2.LINE_AA)
        distance = next_distance
        pattern_idx += 1
        draw = not draw


def _angle_marker_points(geometry: dict, steps: int = 24) -> list[Point2D]:
    vertex = tuple(geometry["vertex"])
    arm1 = tuple(geometry["arm1"])
    arm2 = tuple(geometry["arm2"])
    radius = float(geometry.get("radius", 0.22))
    a0 = math.atan2(arm1[1] - vertex[1], arm1[0] - vertex[0])
    a1 = math.atan2(arm2[1] - vertex[1], arm2[0] - vertex[0])
    delta = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
    return [(vertex[0] + radius * math.cos(a0 + delta * idx / steps), vertex[1] + radius * math.sin(a0 + delta * idx / steps)) for idx in range(steps + 1)]


def _right_angle_points(geometry: dict) -> list[Point2D]:
    vertex = tuple(geometry["vertex"])
    u = _unit(_direction_from_geometry(geometry, "arm1", "direction1", vertex))
    v = _unit(_direction_from_geometry(geometry, "arm2", "direction2", vertex))
    size = float(geometry.get("size", 0.18))
    p1 = (vertex[0] + u[0] * size, vertex[1] + u[1] * size)
    p2 = (p1[0] + v[0] * size, p1[1] + v[1] * size)
    p3 = (vertex[0] + v[0] * size, vertex[1] + v[1] * size)
    return [p1, p2, p3]


def _tick_points(geometry: dict) -> tuple[Point2D, Point2D]:
    start = tuple(geometry["start"])
    end = tuple(geometry["end"])
    size = float(geometry.get("size", 0.08))
    mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    dx, dy = _unit((end[0] - start[0], end[1] - start[1]))
    normal = (-dy, dx)
    return (mid[0] - normal[0] * size, mid[1] - normal[1] * size), (mid[0] + normal[0] * size, mid[1] + normal[1] * size)


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


def _unit(direction: Point2D) -> Point2D:
    norm = math.hypot(direction[0], direction[1])
    if norm < 1e-12:
        return 1.0, 0.0
    return direction[0] / norm, direction[1] / norm


def _pixel(point: Point2D) -> tuple[int, int]:
    return int(round(point[0])), int(round(point[1]))


def _bgr(color: str) -> tuple[int, int, int]:
    if color == "none":
        return 255, 255, 255
    value = color.lstrip("#")
    if len(value) != 6:
        return 31, 41, 55
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return b, g, r


def _rgb(color: str) -> tuple[int, int, int]:
    blue, green, red = _bgr(color)
    return red, green, blue
