"""Convert formal geometry states into renderer-neutral scenes."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import sympy as sp

from geosym_world_geometry.state import FormalGeometryState

from .scene import Point2D, RenderLabel, RenderPrimitive, RenderScene, RenderStyle
from .style_policy import DeterministicStylePolicy


BASE = RenderStyle(stroke="#1f2937", fill="none", stroke_width=2.0)
AUX = RenderStyle(stroke="#64748b", fill="none", stroke_width=1.4, opacity=0.7, dasharray=(6.0, 5.0))
HIDDEN_EDGE = RenderStyle(stroke="#64748b", fill="none", stroke_width=1.3, opacity=0.65, dasharray=(5.0, 5.0))
POINT = RenderStyle(stroke="#111827", fill="#ffffff", stroke_width=1.7)
TARGET = RenderStyle(stroke="#dc2626", fill="none", stroke_width=3.2)
TARGET_HIDDEN = RenderStyle(stroke="#dc2626", fill="none", stroke_width=2.2, opacity=0.75, dasharray=(5.0, 5.0))
TARGET_FILL = RenderStyle(stroke="#dc2626", fill="#fecaca", stroke_width=3.0, opacity=0.7)
POLYGON_FILL = RenderStyle(stroke="#334155", fill="#dbeafe", stroke_width=1.7, opacity=0.45)
REGION_FILL = RenderStyle(stroke="#0f766e", fill="#ccfbf1", stroke_width=1.7, opacity=0.5)
SECTION_FILL = RenderStyle(stroke="#ea580c", fill="#fed7aa", stroke_width=2.4, opacity=0.75)
PLANE_FILL = RenderStyle(stroke="#0284c7", fill="#e0f2fe", stroke_width=1.4, opacity=0.35, dasharray=(5.0, 5.0))
POLY_EDGE = RenderStyle(stroke="#475569", fill="none", stroke_width=1.5, opacity=0.85)
CIRCLE = RenderStyle(stroke="#7c3aed", fill="none", stroke_width=2.0)
MARKER = RenderStyle(stroke="#0f172a", fill="none", stroke_width=1.8)


def state_to_render_scene(
    state: FormalGeometryState,
    *,
    scene_id: str = "formal_state",
    include_labels: bool = True,
    include_3d: bool | str = "auto",
    style_policy: DeterministicStylePolicy | None = None,
) -> RenderScene:
    """Build a renderer-neutral scene from a formal state.

    The projection is intentionally simple: 2D coordinates pass through unchanged,
    and 3D coordinates use a fixed orthographic projection suitable for debugging.
    """

    target_id = state.target.object if state.target else None
    render_dim = _resolve_render_dim(state, include_3d)
    primitives: list[RenderPrimitive] = []

    primitives.extend(_region_primitives(state, target_id, render_dim))
    primitives.extend(_plane_primitives(state, target_id, render_dim))
    primitives.extend(_polygon_primitives(state, target_id, render_dim))
    primitives.extend(_polyhedron_primitives(state, target_id, render_dim))
    primitives.extend(_line_primitives(state, target_id, render_dim))
    primitives.extend(_circle_primitives(state, target_id, render_dim))
    primitives.extend(_arc_primitives(state, target_id, render_dim))
    primitives.extend(_marker_primitives(state, render_dim))
    primitives.extend(_point_primitives(state, target_id, render_dim))

    labels = tuple(_point_labels(state, render_dim)) if include_labels else ()
    scene = RenderScene(
        id=scene_id,
        primitives=tuple(primitives),
        labels=labels,
        metadata={
            "target": state.target.to_dict() if state.target else None,
            "projection": "identity_2d_fixed_orthographic_3d",
            "render_dim": render_dim,
            "num_points": len(state.points),
            "num_primitives": len(primitives),
        },
    )
    return style_policy.apply(scene) if style_policy else scene


def _region_primitives(state: FormalGeometryState, target_id: str | None, render_dim: int | None) -> list[RenderPrimitive]:
    primitives: list[RenderPrimitive] = []
    for region in state.regions.values():
        if region.kind == "sector_region":
            circle_id = region.metadata.get("circle")
            if not circle_id or circle_id not in state.circles:
                continue
            circle = state.circles[circle_id]
            if not _point_allowed(state, circle.center, render_dim):
                continue
            center = _project(state.points[circle.center].coords)
            start = _project(state.points[region.metadata["start"]].coords)
            end = _project(state.points[region.metadata["end"]].coords)
            style = TARGET_FILL if region.id == target_id else REGION_FILL
            primitives.append(
                RenderPrimitive(
                    id=region.id,
                    kind="sector",
                    role="target" if region.id == target_id else "region",
                    geometry={"center": center, "start": start, "end": end, "radius": _float(circle.radius)},
                    style=style,
                    metadata={"kind": region.kind},
                )
            )
        elif all(point_id in state.points and _point_allowed(state, point_id, render_dim) for point_id in region.boundary):
            style = TARGET_FILL if region.id == target_id else REGION_FILL
            primitives.append(
                RenderPrimitive(
                    id=region.id,
                    kind="polygon",
                    role="target" if region.id == target_id else "region",
                    geometry={"points": [_project(state.points[point_id].coords) for point_id in region.boundary]},
                    style=style,
                    metadata={"kind": region.kind},
                )
            )
    return primitives


def _plane_primitives(state: FormalGeometryState, target_id: str | None, render_dim: int | None) -> list[RenderPrimitive]:
    if render_dim == 2:
        return []
    primitives: list[RenderPrimitive] = []
    for plane in state.planes.values():
        if len(plane.source_points) >= 3 and all(point_id in state.points for point_id in plane.source_points[:3]):
            primitives.append(
                RenderPrimitive(
                    id=plane.id,
                    kind="polygon",
                    role="target" if plane.id == target_id else "plane",
                    geometry={"points": [_project(state.points[point_id].coords) for point_id in plane.source_points[:3]]},
                    style=TARGET_FILL if plane.id == target_id else PLANE_FILL,
                    metadata={"source_points": list(plane.source_points[:3])},
                )
            )
    return primitives


def _polygon_primitives(state: FormalGeometryState, target_id: str | None, render_dim: int | None) -> list[RenderPrimitive]:
    primitives: list[RenderPrimitive] = []
    for polygon in state.polygons.values():
        if not all(_point_allowed(state, point_id, render_dim) for point_id in polygon.vertices):
            continue
        style = SECTION_FILL if polygon.metadata.get("section") else POLYGON_FILL
        if polygon.id == target_id:
            style = TARGET_FILL
        primitives.append(
            RenderPrimitive(
                id=polygon.id,
                kind="polygon",
                role="target" if polygon.id == target_id else "polygon",
                geometry={"points": [_project(state.points[point_id].coords) for point_id in polygon.vertices]},
                style=style,
                metadata=polygon.metadata,
            )
        )
    return primitives


def _polyhedron_primitives(state: FormalGeometryState, target_id: str | None, render_dim: int | None) -> list[RenderPrimitive]:
    if render_dim == 2:
        return []
    primitives: list[RenderPrimitive] = []
    for polyhedron in state.polyhedra.values():
        hidden_edges = _hidden_polyhedron_edges(state, polyhedron.vertices, polyhedron.faces)
        for left, right in _polyhedron_edges(polyhedron.faces):
            edge_id = f"{polyhedron.id}:{left}-{right}"
            hidden = tuple(sorted((left, right))) in hidden_edges
            if hidden:
                style = TARGET_HIDDEN if polyhedron.id == target_id else HIDDEN_EDGE
            else:
                style = TARGET if polyhedron.id == target_id else POLY_EDGE
            primitives.append(
                RenderPrimitive(
                    id=edge_id,
                    kind="segment",
                    role="polyhedron_edge",
                    geometry={"points": [_project(state.points[left].coords), _project(state.points[right].coords)]},
                    style=style,
                    metadata={"polyhedron": polyhedron.id, "edge": [left, right], "hidden": hidden},
                )
            )
    return primitives


def _line_primitives(state: FormalGeometryState, target_id: str | None, render_dim: int | None) -> list[RenderPrimitive]:
    primitives: list[RenderPrimitive] = []
    for segment in state.segments.values():
        if not (_point_allowed(state, segment.start, render_dim) and _point_allowed(state, segment.end, render_dim)):
            continue
        style = TARGET if segment.id == target_id else BASE
        primitives.append(
            RenderPrimitive(
                id=segment.id,
                kind="segment",
                role="target" if segment.id == target_id else "segment",
                geometry={"points": [_project(state.points[segment.start].coords), _project(state.points[segment.end].coords)]},
                style=style,
            )
        )
    for line in state.lines.values():
        if render_dim is not None and line.dim != render_dim:
            continue
        style = TARGET if line.id == target_id else AUX if line.metadata.get("constructed_by") else BASE
        primitives.append(
            RenderPrimitive(
                id=line.id,
                kind="line",
                role="target" if line.id == target_id else "line",
                geometry={"point": _project(line.point), "direction": _project_direction(line.direction)},
                style=style,
                metadata=line.metadata,
            )
        )
    for ray in state.rays.values():
        if not _point_allowed(state, ray.start, render_dim):
            continue
        style = TARGET if ray.id == target_id else AUX
        primitives.append(
            RenderPrimitive(
                id=ray.id,
                kind="ray",
                role="target" if ray.id == target_id else "ray",
                geometry={"point": _project(state.points[ray.start].coords), "direction": _project_direction(ray.direction)},
                style=style,
                metadata=ray.metadata,
            )
        )
    return primitives


def _circle_primitives(state: FormalGeometryState, target_id: str | None, render_dim: int | None) -> list[RenderPrimitive]:
    primitives: list[RenderPrimitive] = []
    for circle in state.circles.values():
        if not _point_allowed(state, circle.center, render_dim):
            continue
        primitives.append(
            RenderPrimitive(
                id=circle.id,
                kind="circle",
                role="target" if circle.id == target_id else "circle",
                geometry={"center": _project(state.points[circle.center].coords), "radius": _float(circle.radius)},
                style=TARGET if circle.id == target_id else CIRCLE,
                metadata=circle.metadata,
            )
        )
    return primitives


def _arc_primitives(state: FormalGeometryState, target_id: str | None, render_dim: int | None) -> list[RenderPrimitive]:
    primitives: list[RenderPrimitive] = []
    for arc in state.arcs.values():
        circle = state.circles[arc.circle]
        if not _point_allowed(state, circle.center, render_dim):
            continue
        primitives.append(
            RenderPrimitive(
                id=arc.id,
                kind="arc",
                role="target" if arc.id == target_id else "arc",
                geometry={
                    "center": _project(state.points[circle.center].coords),
                    "radius": _float(circle.radius),
                    "start": _project(state.points[arc.start].coords),
                    "end": _project(state.points[arc.end].coords),
                    "orientation": arc.orientation,
                },
                style=TARGET if arc.id == target_id else RenderStyle(stroke="#0891b2", fill="none", stroke_width=2.0),
                metadata=arc.metadata,
            )
        )
    return primitives


def _point_primitives(state: FormalGeometryState, target_id: str | None, render_dim: int | None) -> list[RenderPrimitive]:
    primitives: list[RenderPrimitive] = []
    for point in state.points.values():
        if render_dim is not None and point.dim != render_dim:
            continue
        primitives.append(
            RenderPrimitive(
                id=point.id,
                kind="point",
                role="target" if point.id == target_id else "point",
                geometry={"point": _project(point.coords), "radius": 4.0},
                style=TARGET_FILL if point.id == target_id else POINT,
                metadata={"dim": point.dim, **point.metadata},
            )
        )
    return primitives


def _point_labels(state: FormalGeometryState, render_dim: int | None) -> Iterable[RenderLabel]:
    visible = [point for point in state.points.values() if render_dim is None or point.dim == render_dim]
    if not visible:
        return
    projected = [_project(point.coords) for point in visible]
    cx = sum(point[0] for point in projected) / len(projected)
    cy = sum(point[1] for point in projected) / len(projected)
    for point in state.points.values():
        if render_dim is not None and point.dim != render_dim:
            continue
        x, y = _project(point.coords)
        dx, dy = x - cx, y - cy
        norm = math.hypot(dx, dy)
        if norm < 1e-9:
            offset = (14.0, -18.0)
        else:
            offset_x = 22.0 * dx / norm
            offset_y = -22.0 * dy / norm
            if abs(offset_y) < 14.0:
                offset_y = -14.0 if y >= cy else 18.0
            offset = (offset_x, offset_y)
        yield RenderLabel(id=f"label:{point.id}", text=_display_label(point.id), position=(x, y), offset=offset)


def _marker_primitives(state: FormalGeometryState, render_dim: int | None) -> list[RenderPrimitive]:
    if render_dim == 3:
        return []
    primitives: list[RenderPrimitive] = []
    for relation in state.relations:
        rel_type = relation.get("type")
        if rel_type == "midpoint":
            point_id = relation.get("point")
            endpoints = relation.get("of") or ()
            if point_id in state.points and len(endpoints) == 2 and all(_point_allowed(state, item, render_dim) for item in endpoints):
                mid = _project(state.points[point_id].coords)
                left = _project(state.points[endpoints[0]].coords)
                right = _project(state.points[endpoints[1]].coords)
                primitives.append(_tick_marker(f"tick:{endpoints[0]}-{point_id}", left, mid))
                primitives.append(_tick_marker(f"tick:{point_id}-{endpoints[1]}", mid, right))
        elif rel_type == "angle_bisector":
            angle = relation.get("angle") or ()
            if len(angle) == 3 and all(_point_allowed(state, item, render_dim) for item in angle):
                a, vertex, c = angle
                primitives.append(
                    RenderPrimitive(
                        id=f"angle:{a}{vertex}{c}",
                        kind="angle_marker",
                        role="marker",
                        geometry={
                            "vertex": _project(state.points[vertex].coords),
                            "arm1": _project(state.points[a].coords),
                            "arm2": _project(state.points[c].coords),
                            "radius": 0.22,
                        },
                        style=MARKER,
                        metadata={"relation": "angle_bisector", "mode": relation.get("mode", "internal")},
                    )
                )
        elif rel_type == "foot_of_perpendicular":
            foot_id = relation.get("point")
            source_id = relation.get("from")
            line_id = relation.get("line")
            if foot_id in state.points and source_id in state.points and line_id:
                line = state.get_line_like(line_id)
                if line.dim == 2 and _point_allowed(state, foot_id, render_dim) and _point_allowed(state, source_id, render_dim):
                    foot = _project(state.points[foot_id].coords)
                    source = _project(state.points[source_id].coords)
                    line_dir = _project_direction(line.direction)
                    primitives.append(
                        RenderPrimitive(
                            id=f"altitude:{source_id}-{foot_id}",
                            kind="segment",
                            role="auxiliary",
                            geometry={"points": [source, foot]},
                            style=AUX,
                            metadata={"relation": "foot_of_perpendicular", "from": source_id, "line": line_id},
                        )
                    )
                    primitives.append(
                        RenderPrimitive(
                            id=f"right_angle:{foot_id}",
                            kind="right_angle_marker",
                            role="marker",
                            geometry={"vertex": foot, "arm1": source, "direction2": line_dir, "size": 0.18},
                            style=MARKER,
                            metadata={"relation": "foot_of_perpendicular", "from": source_id, "line": line_id},
                        )
                    )
        elif rel_type == "perpendicular":
            object_ids = relation.get("objects") or ()
            if len(object_ids) == 2 and all(object_id in state.lines for object_id in object_ids):
                left = state.lines[object_ids[0]]
                right = state.lines[object_ids[1]]
                if left.dim == 2 and right.dim == 2:
                    vertex = _line_intersection(left.point, left.direction, right.point, right.direction)
                    if vertex is not None:
                        primitives.append(
                            RenderPrimitive(
                                id=f"right_angle:{object_ids[0]}|{object_ids[1]}",
                                kind="right_angle_marker",
                                role="marker",
                                geometry={
                                    "vertex": vertex,
                                    "direction1": _project_direction(left.direction),
                                    "direction2": _project_direction(right.direction),
                                    "size": 0.18,
                                },
                                style=MARKER,
                                metadata={"relation": "perpendicular", "objects": list(object_ids)},
                            )
                        )
    return primitives


def _tick_marker(object_id: str, start: Point2D, end: Point2D) -> RenderPrimitive:
    return RenderPrimitive(
        id=object_id,
        kind="tick_marker",
        role="marker",
        geometry={"start": start, "end": end, "size": 0.08},
        style=MARKER,
        metadata={"relation": "equal_length"},
    )


def _line_intersection(point1: Sequence[sp.Expr], direction1: Sequence[sp.Expr], point2: Sequence[sp.Expr], direction2: Sequence[sp.Expr]) -> Point2D | None:
    x1, y1 = _float(point1[0]), _float(point1[1])
    dx1, dy1 = _float(direction1[0]), _float(direction1[1])
    x2, y2 = _float(point2[0]), _float(point2[1])
    dx2, dy2 = _float(direction2[0]), _float(direction2[1])
    denom = dx1 * dy2 - dy1 * dx2
    if abs(denom) < 1e-12:
        return None
    t = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / denom
    return x1 + t * dx1, y1 + t * dy1


def _hidden_polyhedron_edges(state: FormalGeometryState, vertices: Sequence[str], faces: Sequence[Sequence[str]]) -> set[tuple[str, str]]:
    if not vertices or not faces:
        return set()
    centroid = _centroid([state.points[vertex].coords for vertex in vertices])
    camera = (-0.55, -1.0, 0.35)
    edge_front_counts: dict[tuple[str, str], int] = {edge: 0 for edge in _polyhedron_edges(faces)}
    edge_face_counts: dict[tuple[str, str], int] = {edge: 0 for edge in _polyhedron_edges(faces)}
    for face in faces:
        if len(face) < 3:
            continue
        coords = [state.points[point_id].coords for point_id in face]
        normal = _cross3f(_sub3f(coords[1], coords[0]), _sub3f(coords[2], coords[0]))
        face_centroid = _centroid(coords)
        outward_hint = _sub3f(face_centroid, centroid)
        if _dot3f(normal, outward_hint) < 0:
            normal = (-normal[0], -normal[1], -normal[2])
        front = _dot3f(normal, camera) < 0
        for idx, left in enumerate(face):
            right = face[(idx + 1) % len(face)]
            edge = tuple(sorted((left, right)))
            edge_face_counts[edge] = edge_face_counts.get(edge, 0) + 1
            if front:
                edge_front_counts[edge] = edge_front_counts.get(edge, 0) + 1
    hidden = set()
    for edge, face_count in edge_face_counts.items():
        if face_count and edge_front_counts.get(edge, 0) == 0:
            hidden.add(edge)
    return hidden


def _centroid(points: Sequence[Sequence[sp.Expr]]) -> tuple[float, float, float]:
    return tuple(sum(_float(point[idx]) for point in points) / len(points) for idx in range(3))  # type: ignore[return-value]


def _sub3f(left: Sequence[sp.Expr] | tuple[float, float, float], right: Sequence[sp.Expr] | tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(_float(left[idx]) - _float(right[idx]) for idx in range(3))  # type: ignore[return-value]


def _cross3f(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _dot3f(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _display_label(object_id: str) -> str:
    if len(object_id) <= 8:
        return object_id
    tail = object_id.rsplit("_", 1)[-1]
    if 1 < len(tail) <= 6:
        return tail
    return object_id[:6] + "..."


def _resolve_render_dim(state: FormalGeometryState, include_3d: bool | str) -> int | None:
    if include_3d is True:
        return None
    if include_3d is False:
        return 2
    if include_3d != "auto":
        raise ValueError("include_3d must be True, False, or 'auto'")
    target_dim = _target_dim(state)
    if target_dim is not None:
        return target_dim
    has_2d = any(point.dim == 2 for point in state.points.values())
    has_3d = any(point.dim == 3 for point in state.points.values())
    if has_2d:
        return 2
    if has_3d:
        return 3
    return None


def _target_dim(state: FormalGeometryState) -> int | None:
    if state.target is None:
        return None
    object_id = state.target.object
    target_type = state.target.type
    if target_type.startswith("section") or target_type in {"dihedral_angle", "line_plane_angle", "plane_angle", "space_angle"}:
        return 3
    if object_id in state.points:
        return state.points[object_id].dim
    if object_id in state.lines:
        return state.lines[object_id].dim
    if object_id in state.segments:
        segment = state.segments[object_id]
        return state.points[segment.start].dim
    if object_id in state.rays:
        return state.points[state.rays[object_id].start].dim
    if object_id in state.circles:
        return state.points[state.circles[object_id].center].dim
    if object_id in state.arcs:
        circle = state.circles[state.arcs[object_id].circle]
        return state.points[circle.center].dim
    if object_id in state.polygons:
        polygon = state.polygons[object_id]
        return state.points[polygon.vertices[0]].dim
    if object_id in state.regions and state.regions[object_id].boundary:
        return state.points[state.regions[object_id].boundary[0]].dim
    if "|" in object_id:
        parts = object_id.split("|")
        if any(part in state.planes or part in state.polyhedra for part in parts):
            return 3
        dims = []
        for part in parts:
            if part in state.lines:
                dims.append(state.lines[part].dim)
            elif part in state.segments:
                dims.append(state.points[state.segments[part].start].dim)
        if dims:
            return 3 if 3 in dims else 2
    if state.circles:
        return 2
    return None


def _point_allowed(state: FormalGeometryState, point_id: str, render_dim: int | None) -> bool:
    if point_id not in state.points:
        return False
    return render_dim is None or state.points[point_id].dim == render_dim


def _project(coords: Sequence[sp.Expr]) -> Point2D:
    values = [_float(value) for value in coords]
    if len(values) == 2:
        return values[0], values[1]
    if len(values) == 3:
        x, y, z = values
        return x - 0.55 * y, z + 0.35 * y
    raise ValueError(f"Cannot project coordinate dimension {len(values)}")


def _project_direction(direction: Sequence[sp.Expr]) -> Point2D:
    projected = _project(direction)
    if abs(projected[0]) < 1e-12 and abs(projected[1]) < 1e-12:
        return 1.0, 0.0
    return projected


def _float(value: sp.Expr) -> float:
    return float(sp.N(value))


def _polyhedron_edges(faces: Iterable[Sequence[str]]) -> list[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for face in faces:
        for idx, left in enumerate(face):
            right = face[(idx + 1) % len(face)]
            edges.add(tuple(sorted((left, right))))
    return sorted(edges)
