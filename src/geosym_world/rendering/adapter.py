"""Convert formal geometry states into renderer-neutral scenes."""

from __future__ import annotations

import math
import re
from dataclasses import replace
from typing import Iterable, Sequence

import sympy as sp

from geosym_world.schema.state import FormalGeometryState
from geosym_world.kernel.geometry2d.regions import resolve_region_geometry_2d

from .scene import Point2D, RenderLabel, RenderPrimitive, RenderScene, RenderStyle, scene_fingerprint
from .annotation_policy import AnnotationPolicy
from .math_text import format_math_expression
from .relation_text import format_relation_value
from .quality import inspect_render_scene
from .style_policy import DeterministicStylePolicy


BASE = RenderStyle(stroke="#1f2937", fill="none", stroke_width=2.0)
AUX = RenderStyle(stroke="#64748b", fill="none", stroke_width=1.4, opacity=0.7, dasharray=(6.0, 5.0))
HIDDEN_EDGE = RenderStyle(stroke="#64748b", fill="none", stroke_width=1.3, opacity=0.65, dasharray=(5.0, 5.0))
POINT = RenderStyle(stroke="#111827", fill="#ffffff", stroke_width=1.7)
INTERSECTION = RenderStyle(stroke="#9a3412", fill="#fb923c", stroke_width=2.0)
TARGET = RenderStyle(stroke="#dc2626", fill="none", stroke_width=3.2)
TARGET_HIDDEN = RenderStyle(stroke="#dc2626", fill="none", stroke_width=2.2, opacity=0.75, dasharray=(5.0, 5.0))
TARGET_FILL = RenderStyle(stroke="#dc2626", fill="#fecaca", stroke_width=3.0, opacity=0.7)
POLYGON_FILL = RenderStyle(stroke="#334155", fill="#dbeafe", stroke_width=1.7, opacity=0.45)
REGION_FILL = RenderStyle(stroke="#0f766e", fill="#ccfbf1", stroke_width=1.7, opacity=0.5)
REGION_ALT_FILL = RenderStyle(stroke="#1d4ed8", fill="#dbeafe", stroke_width=1.7, opacity=0.5)
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
    annotation_policy: AnnotationPolicy | None = None,
) -> RenderScene:
    """Build a renderer-neutral scene from a formal state.

    The projection is intentionally simple: 2D coordinates pass through unchanged,
    and 3D coordinates use a fixed orthographic projection suitable for debugging.
    """

    annotation_policy = annotation_policy or AnnotationPolicy()
    target_id = state.target.object if state.target else None
    render_dim = _resolve_render_dim(state, include_3d)
    point_labels = _point_display_labels(state, render_dim)
    relation_intersections = _perpendicular_relation_intersection_primitives(state, render_dim)
    implicit_intersections = _implicit_intersection_primitives(state, render_dim)
    implicit_intersections = [
        primitive
        for primitive in implicit_intersections
        if not any(
            _near(tuple(primitive.geometry["point"]), tuple(relation.geometry["point"]))
            for relation in relation_intersections
        )
    ]
    constructed_intersections = [*relation_intersections, *implicit_intersections]
    implicit_labels = _implicit_intersection_labels(constructed_intersections, point_labels)
    primitives: list[RenderPrimitive] = []

    primitives.extend(_region_primitives(state, target_id, render_dim))
    primitives.extend(_plane_primitives(state, target_id, render_dim))
    primitives.extend(_polygon_primitives(state, target_id, render_dim))
    primitives.extend(_polyhedron_primitives(state, target_id, render_dim))
    primitives.extend(_line_primitives(state, target_id, render_dim))
    primitives.extend(_circle_primitives(state, target_id, render_dim))
    primitives.extend(_arc_primitives(state, target_id, render_dim))
    primitives.extend(
        _marker_primitives(
            state,
            render_dim,
            show_symbols=annotation_policy.relation_markers,
            allow_inferred=annotation_policy.inferred_markers,
        )
    )
    primitives.extend(constructed_intersections)
    primitives.extend(_point_primitives(state, target_id, render_dim, point_labels))

    value_labels = (
        (*_measurement_labels(state, render_dim), *_relation_labels(state, render_dim))
        if annotation_policy.show_public_values
        else ()
    )
    labels = (
        (*_point_labels(state, render_dim, point_labels, show_all_points=annotation_policy.level == "points"), *implicit_labels, *value_labels)
        if include_labels
        else ()
    )
    for index, primitive in enumerate(primitives):
        metadata = dict(primitive.metadata)
        region = state.regions.get(primitive.id) or state.composite_regions.get(primitive.id)
        if region is not None:
            metadata.update({key: region.metadata[key] for key in ("render_fill", "line_style") if key in region.metadata})
        primitives[index] = replace(primitive, metadata=metadata, style=replace(
            primitive.style,
            fill="none" if metadata.get("render_fill") is False else primitive.style.fill,
            dasharray=(6.0, 5.0) if metadata.get("line_style") == "dashed" else primitive.style.dasharray,
        ))
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
            "point_labels": point_labels,
            "annotation_policy": annotation_policy.to_dict(),
            "region_geometry": {
                region_id: resolve_region_geometry_2d(state, region_id).to_dict()
                for region_id in (*state.regions.keys(), *state.composite_regions.keys())
            },
        },
    )
    if style_policy:
        scene = style_policy.apply(scene)
    quality = inspect_render_scene(scene)
    fingerprint = scene_fingerprint(scene)
    return replace(scene, metadata={**scene.metadata, "render_quality": quality.to_dict(), "scene_fingerprint": fingerprint})


def _region_primitives(state: FormalGeometryState, target_id: str | None, render_dim: int | None) -> list[RenderPrimitive]:
    primitives: list[RenderPrimitive] = []
    hole_ids = {hole_id for region in state.regions.values() for hole_id in region.holes}
    materialized_component_ids = {
        component_id
        for composite in state.composite_regions.values()
        if composite.metadata.get("render") is not False
        if resolve_region_geometry_2d(state, composite.id).resolved
        for component_id in composite.components
    }
    for region in state.regions.values():
        if region.metadata.get("render") is False:
            continue
        if region.id in hole_ids or (region.id in materialized_component_ids and region.id != target_id):
            continue
        if region.kind == "annulus_region":
            inner = state.circles.get(region.metadata.get("inner_circle"))
            outer = state.circles.get(region.metadata.get("outer_circle"))
            if inner is None or outer is None or inner.center != outer.center or not _point_allowed(state, outer.center, render_dim):
                continue
            primitives.append(
                RenderPrimitive(
                    id=region.id,
                    kind="annulus",
                    role="target" if region.id == target_id else "region",
                    geometry={
                        "center": _project(state.points[outer.center].coords),
                        "inner_radius": _float(inner.radius),
                        "outer_radius": _float(outer.radius),
                    },
                    style=TARGET_FILL if region.id == target_id else REGION_FILL,
                    metadata={"kind": region.kind, **region.metadata},
                )
            )
        elif region.kind == "lens_region":
            arc_ids = tuple(region.metadata.get("arcs") or ())
            if len(arc_ids) != 2 or any(arc_id not in state.arcs for arc_id in arc_ids):
                continue
            arcs = tuple(state.arcs[arc_id] for arc_id in arc_ids)
            circles = tuple(state.circles[arc.circle] for arc in arcs)
            if any(not _point_allowed(state, circle.center, render_dim) for circle in circles):
                continue
            primitives.append(
                RenderPrimitive(
                    id=region.id,
                    kind="lens",
                    role="target" if region.id == target_id else "region",
                    geometry={
                        "arcs": [
                            {
                                "center": _project(state.points[circle.center].coords),
                                "start": _project(state.points[arc.start].coords),
                                "end": _project(state.points[arc.end].coords),
                                "radius": _float(circle.radius),
                                "orientation": arc.orientation,
                            }
                            for arc, circle in zip(arcs, circles)
                        ]
                    },
                    style=TARGET_FILL if region.id == target_id else REGION_FILL,
                    metadata={"kind": region.kind, **region.metadata},
                )
            )
        elif region.kind == "sector_region":
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
                    geometry={
                        "center": center,
                        "start": start,
                        "end": end,
                        "radius": _float(circle.radius),
                        "orientation": region.metadata.get("orientation", "ccw"),
                    },
                    style=style,
                    metadata={"kind": region.kind},
                )
            )
        elif region.kind == "circular_segment_region":
            sector = state.regions.get(region.metadata.get("sector_region"))
            if sector is None:
                continue
            circle = state.circles.get(sector.metadata.get("circle"))
            if circle is None or not _point_allowed(state, circle.center, render_dim):
                continue
            primitives.append(
                RenderPrimitive(
                    id=region.id,
                    kind="circular_segment",
                    role="target" if region.id == target_id else "region",
                    geometry={
                        "center": _project(state.points[circle.center].coords),
                        "start": _project(state.points[sector.metadata["start"]].coords),
                        "end": _project(state.points[sector.metadata["end"]].coords),
                        "radius": _float(circle.radius),
                        "orientation": sector.metadata.get("orientation", "ccw"),
                    },
                    style=TARGET_FILL if region.id == target_id else REGION_FILL,
                    metadata={"kind": region.kind, **region.metadata},
                )
            )
        elif region.kind == "polygon_region":
            resolved = resolve_region_geometry_2d(state, region.id)
            if resolved.resolved and all(len(point) == 2 for point in resolved.boundary):
                style = TARGET_FILL if region.id == target_id else (REGION_ALT_FILL if region.metadata.get("fill_variant") == "alternate" else REGION_FILL)
                primitives.append(
                    RenderPrimitive(
                        id=region.id,
                        kind="polygon",
                        role="target" if region.id == target_id else "region",
                        geometry={
                            "points": [_project(point) for point in resolved.boundary],
                            "holes": [[_project(point) for point in hole] for hole in resolved.holes],
                        },
                        style=style,
                        metadata={"kind": region.kind, "boundary_status": resolved.status, "source_ids": list(resolved.source_ids)},
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
                        metadata={"kind": region.kind, "boundary_status": resolved.status},
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
    for region in state.composite_regions.values():
        if region.metadata.get("render") is False:
            continue
        resolved = resolve_region_geometry_2d(state, region.id)
        if not resolved.resolved or any(len(point) != 2 for point in resolved.boundary):
            continue
        style = TARGET_FILL if region.id == target_id else REGION_FILL
        primitives.append(
            RenderPrimitive(
                id=region.id,
                kind="polygon",
                role="target" if region.id == target_id else "region",
                geometry={
                    "points": [_project(point) for point in resolved.boundary],
                    "components": [[_project(point) for point in component] for component in resolved.components],
                    "holes": [[_project(point) for point in hole] for hole in resolved.holes],
                },
                style=style,
                metadata={"operation": region.operation, "boundary_status": resolved.status, "source_ids": list(resolved.source_ids)},
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
        if polygon.metadata.get("render") is False:
            continue
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
    primitives = _relation_extension_primitives(state, render_dim)
    # Infinite construction lines and rays form the background. Drawing finite
    # geometry afterward prevents dashed extensions from overwriting solid edges.
    for line in state.lines.values():
        if line.metadata.get("render") is False:
            continue
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
                metadata=segment.metadata,
            )
        )
    return primitives


def _circle_primitives(state: FormalGeometryState, target_id: str | None, render_dim: int | None) -> list[RenderPrimitive]:
    primitives: list[RenderPrimitive] = []
    for circle in state.circles.values():
        if circle.metadata.get("render") is False:
            continue
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
        if arc.metadata.get("render") is False:
            continue
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


def _point_primitives(
    state: FormalGeometryState,
    target_id: str | None,
    render_dim: int | None,
    point_labels: dict[str, str],
) -> list[RenderPrimitive]:
    primitives: list[RenderPrimitive] = []
    for point in state.points.values():
        if point.metadata.get("render") is False or point.metadata.get("show_marker") is False or (render_dim is not None and point.dim != render_dim):
            continue
        constructed_by = str(point.metadata.get("constructed_by", ""))
        is_intersection = constructed_by.startswith("intersection_")
        primitives.append(
            RenderPrimitive(
                id=point.id,
                kind="point",
                role="target" if point.id == target_id else "point",
                geometry={"point": _project(point.coords), "radius": 5.0 if is_intersection else 4.0},
                style=TARGET_FILL if point.id == target_id else INTERSECTION if is_intersection else POINT,
                metadata={"dim": point.dim, **point.metadata, "display_label": point_labels[point.id]},
            )
        )
    return primitives


def _point_labels(
    state: FormalGeometryState,
    render_dim: int | None,
    point_labels: dict[str, str],
    *,
    show_all_points: bool = False,
) -> Iterable[RenderLabel]:
    visible = [point for point in state.points.values() if point.metadata.get("render") is not False and (render_dim is None or point.dim == render_dim)]
    if not visible:
        return
    projected = [_project(point.coords) for point in visible]
    cx = sum(point[0] for point in projected) / len(projected)
    cy = sum(point[1] for point in projected) / len(projected)
    for point in state.points.values():
        if point.metadata.get("render") is False or (
            point.metadata.get("show_label") is False and not show_all_points
        ) or (render_dim is not None and point.dim != render_dim):
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
        yield RenderLabel(id=f"label:{point.id}", text=point_labels[point.id], position=(x, y), offset=offset)


def _measurement_labels(state: FormalGeometryState, render_dim: int | None) -> Iterable[RenderLabel]:
    if render_dim == 3:
        return
    for segment in state.segments.values():
        if not segment.metadata.get("show_length"):
            continue
        if not (_point_allowed(state, segment.start, render_dim) and _point_allowed(state, segment.end, render_dim)):
            continue
        start = state.points[segment.start].coords
        end = state.points[segment.end].coords
        yield _segment_measure_label(
            f"measure:{segment.id}",
            start,
            end,
            display_value=segment.metadata.get("display_value"),
            position=float(segment.metadata.get("label_position", 0.5)),
            side=float(segment.metadata.get("label_side", 1.0)),
        )
    for region in state.regions.values():
        if region.kind != "sector_region" or not region.metadata.get("show_central_angle"):
            continue
        yield _sector_angle_label(state, region)
    for relation in state.relations:
        if relation.get("type") != "foot_of_perpendicular":
            continue
        foot_id = relation.get("point")
        source_id = relation.get("from")
        if foot_id not in state.points or source_id not in state.points:
            continue
        if not state.points[foot_id].metadata.get("show_length"):
            continue
        yield _segment_measure_label(
            f"measure:altitude:{source_id}-{foot_id}",
            state.points[source_id].coords,
            state.points[foot_id].coords,
        )


def _relation_labels(state: FormalGeometryState, render_dim: int | None) -> Iterable[RenderLabel]:
    if render_dim == 3:
        return
    for relation in state.relations:
        metadata = relation.get("metadata") or {}
        channel = metadata.get("annotation_channel")
        if channel != "relation_value" and not metadata.get("show_label"):
            continue
        text = format_relation_value(relation)
        if text is None:
            continue
        anchor_id = str(metadata.get("label_anchor", ""))
        if anchor_id in state.points and _point_allowed(state, anchor_id, render_dim):
            anchor = _project(state.points[anchor_id].coords)
        else:
            point_ids = _relation_reference_points(state, relation)
            if not point_ids or not all(_point_allowed(state, point_id, render_dim) for point_id in point_ids):
                continue
            points = [_project(state.points[point_id].coords) for point_id in point_ids]
            anchor = (
                sum(point[0] for point in points) / len(points),
                sum(point[1] for point in points) / len(points),
            )
        raw_offset = metadata.get("label_offset", (36.0, -32.0))
        offset = (float(raw_offset[0]), float(raw_offset[1]))
        yield RenderLabel(
            id=f"relation-value:{relation['id']}",
            text=text,
            position=anchor,
            offset=offset,
        )


def _relation_reference_points(state: FormalGeometryState, relation: dict) -> tuple[str, ...]:
    point_ids: list[str] = []

    def add_object(object_id: str) -> None:
        if object_id in state.points:
            point_ids.append(object_id)
        elif object_id in state.segments:
            segment = state.segments[object_id]
            point_ids.extend((segment.start, segment.end))

    for key in ("objects", "segments", "ratios", "angles"):
        value = relation.get(key) or ()
        stack = list(value) if isinstance(value, (list, tuple)) else [value]
        while stack:
            item = stack.pop(0)
            if isinstance(item, (list, tuple)):
                stack[0:0] = item
            elif item is not None:
                add_object(str(item))
    add_object(str(relation.get("point", "")))
    return tuple(dict.fromkeys(point_ids))


def _segment_measure_label(
    label_id: str,
    start: Sequence[sp.Expr],
    end: Sequence[sp.Expr],
    *,
    display_value: object | None = None,
    position: float = 0.5,
    side: float = 1.0,
) -> RenderLabel:
    a = _project(start)
    b = _project(end)
    position = max(0.0, min(1.0, position))
    midpoint = (a[0] + position * (b[0] - a[0]), a[1] + position * (b[1] - a[1]))
    dx, dy = b[0] - a[0], b[1] - a[1]
    norm = math.hypot(dx, dy) or 1.0
    offset = (-14.0 * side * dy / norm, 14.0 * side * dx / norm)
    squared = sum(sp.simplify((end[index] - start[index]) ** 2) for index in range(len(start)))
    text = format_math_expression(display_value if display_value is not None else sp.sqrt(squared))
    return RenderLabel(id=label_id, text=text, position=midpoint, offset=offset)


def _sector_angle_label(state: FormalGeometryState, region) -> RenderLabel:
    center = _project(state.points[region.metadata["center"]].coords)
    start = _project(state.points[region.metadata["start"]].coords)
    end = _project(state.points[region.metadata["end"]].coords)
    first = _unit_direction((start[0] - center[0], start[1] - center[1]))
    second = _unit_direction((end[0] - center[0], end[1] - center[1]))
    direction = _unit_direction((first[0] + second[0], first[1] + second[1]))
    if abs(first[0] + second[0]) + abs(first[1] + second[1]) < 1e-9:
        direction = (-first[1], first[0])
    radius = _annotation_size(state, 2) * 2.1
    return RenderLabel(
        id=f"measure:central_angle:{region.id}",
        text=format_math_expression(region.metadata["central_angle"]),
        position=(center[0] + radius * direction[0], center[1] + radius * direction[1]),
        offset=(6.0, -5.0),
    )


def _implicit_intersection_primitives(state: FormalGeometryState, render_dim: int | None) -> list[RenderPrimitive]:
    if render_dim == 3:
        return []
    line_like: list[tuple[str, Point2D, Point2D, str]] = []
    for object_id, segment in state.segments.items():
        a = _project(state.points[segment.start].coords)
        b = _project(state.points[segment.end].coords)
        line_like.append((object_id, a, (b[0] - a[0], b[1] - a[1]), "segment"))
    for object_id, line in state.lines.items():
        if line.dim == 2 and line.metadata.get("render") is not False:
            line_like.append((object_id, _project(line.point), _project_direction(line.direction), "line"))
    for object_id, ray in state.rays.items():
        if state.points[ray.start].dim == 2:
            line_like.append((object_id, _project(state.points[ray.start].coords), _project_direction(ray.direction), "ray"))
    explicit = _visible_projected_points(state, 2)
    seen: list[Point2D] = []
    primitives: list[RenderPrimitive] = []
    for left_index, left in enumerate(line_like):
        for right in line_like[left_index + 1:]:
            if left[3] != "segment" or right[3] != "segment":
                continue
            point = _bounded_line_intersection(left, right)
            if point is None or not _within_core_bounds(point, explicit) or any(_near(point, known) for known in (*explicit, *seen)):
                continue
            seen.append(point)
            primitives.append(
                RenderPrimitive(
                    id=f"implicit_intersection:{left[0]}|{right[0]}",
                    kind="point",
                    role="implicit_intersection",
                    geometry={"point": point, "radius": 4.5},
                    style=INTERSECTION,
                    metadata={"implicit": True, "objects": [left[0], right[0]]},
                )
            )
    return primitives


def _perpendicular_relation_intersection_primitives(
    state: FormalGeometryState,
    render_dim: int | None,
) -> list[RenderPrimitive]:
    if render_dim == 3:
        return []
    explicit = _visible_projected_points(state, 2)
    seen: list[Point2D] = []
    primitives: list[RenderPrimitive] = []
    for relation in state.relations:
        if relation.get("type") != "perpendicular":
            continue
        object_ids = relation.get("objects") or ()
        if len(object_ids) != 2:
            continue
        object_ids = _ordered_perpendicular_objects(state, object_ids)
        try:
            left = state.get_line_like(object_ids[0])
            right = state.get_line_like(object_ids[1])
        except KeyError:
            continue
        if left.dim != 2 or right.dim != 2:
            continue
        point = _line_intersection(left.point, left.direction, right.point, right.direction)
        if point is None or not _within_core_bounds(point, explicit) or any(_near(point, known) for known in (*explicit, *seen)):
            continue
        seen.append(point)
        primitives.append(
            RenderPrimitive(
                id=f"relation_intersection:{object_ids[0]}|{object_ids[1]}",
                kind="point",
                role="relation_intersection",
                geometry={"point": point, "radius": 4.0},
                style=POINT,
                metadata={"implicit": True, "relation": "perpendicular", "objects": list(object_ids)},
            )
        )
    return primitives


def _within_core_bounds(point: Point2D, core_points: Sequence[Point2D]) -> bool:
    if not core_points:
        return True
    min_x = min(item[0] for item in core_points)
    max_x = max(item[0] for item in core_points)
    min_y = min(item[1] for item in core_points)
    max_y = max(item[1] for item in core_points)
    margin = max(max_x - min_x, max_y - min_y, 1.0)
    return min_x - margin <= point[0] <= max_x + margin and min_y - margin <= point[1] <= max_y + margin


def _implicit_intersection_labels(
    primitives: Sequence[RenderPrimitive],
    point_labels: dict[str, str],
) -> tuple[RenderLabel, ...]:
    if not primitives:
        return ()
    used = set(point_labels.values())
    names = iter(_available_point_names(used, len(primitives)))
    points = [tuple(primitive.geometry["point"]) for primitive in primitives]
    cx = sum(point[0] for point in points) / len(points)
    cy = sum(point[1] for point in points) / len(points)
    labels: list[RenderLabel] = []
    for primitive, point in zip(primitives, points):
        name = next(names)
        point_labels[primitive.id] = name
        primitive.metadata["display_label"] = name
        dx, dy = point[0] - cx, point[1] - cy
        norm = math.hypot(dx, dy)
        offset = (12.0, -14.0) if norm < 1e-9 else (16.0 * dx / norm, -16.0 * dy / norm)
        labels.append(RenderLabel(id=f"label:{primitive.id}", text=name, position=point, offset=offset))
    return tuple(labels)


def _relation_extension_primitives(state: FormalGeometryState, render_dim: int | None) -> list[RenderPrimitive]:
    if render_dim == 3:
        return []
    primitives: list[RenderPrimitive] = []
    extension = _annotation_size(state, render_dim) * 2.0
    for relation in state.relations:
        relation_type = relation.get("type")
        if relation_type == "foot_of_perpendicular":
            foot_id = relation.get("point")
            segment_id = relation.get("line")
            if foot_id in state.points and segment_id in state.segments:
                primitive = _segment_extension_primitive(
                    state,
                    segment_id,
                    _project(state.points[foot_id].coords),
                    extension,
                    relation_tag=f"foot:{foot_id}",
                )
                if primitive is not None:
                    primitives.append(primitive)
            continue
        if relation_type != "perpendicular":
            continue
        object_ids = relation.get("objects") or ()
        if len(object_ids) != 2:
            continue
        object_ids = _ordered_perpendicular_objects(state, object_ids)
        try:
            left = state.get_line_like(object_ids[0])
            right = state.get_line_like(object_ids[1])
        except KeyError:
            continue
        vertex = _line_intersection(left.point, left.direction, right.point, right.direction)
        if vertex is None:
            continue
        for object_id in object_ids:
            primitive = _segment_extension_primitive(
                state,
                object_id,
                vertex,
                extension,
                relation_tag=f"{object_ids[0]}|{object_ids[1]}",
            )
            if primitive is not None:
                primitives.append(primitive)
    return primitives


def _segment_extension_primitive(
    state: FormalGeometryState,
    segment_id: str,
    vertex: Point2D,
    extension: float,
    *,
    relation_tag: str,
) -> RenderPrimitive | None:
    segment = state.segments.get(segment_id)
    if segment is None:
        return None
    start = _project(state.points[segment.start].coords)
    end = _project(state.points[segment.end].coords)
    if _point_on_segment(vertex, start, end):
        return None
    endpoint = min((start, end), key=lambda point: math.hypot(vertex[0] - point[0], vertex[1] - point[1]))
    direction = (vertex[0] - endpoint[0], vertex[1] - endpoint[1])
    norm = math.hypot(*direction)
    if norm < 1e-9:
        return None
    beyond = (vertex[0] + extension * direction[0] / norm, vertex[1] + extension * direction[1] / norm)
    return RenderPrimitive(
        id=f"extension:{segment_id}:{relation_tag}",
        kind="segment",
        role="auxiliary",
        geometry={"points": [endpoint, beyond]},
        style=AUX,
        metadata={"relation": "segment_extension", "segment": segment_id},
    )


def _point_on_segment(point: Point2D, start: Point2D, end: Point2D, tolerance: float = 1e-8) -> bool:
    cross = (point[0] - start[0]) * (end[1] - start[1]) - (point[1] - start[1]) * (end[0] - start[0])
    if abs(cross) > tolerance * max(1.0, math.hypot(end[0] - start[0], end[1] - start[1])):
        return False
    dot = (point[0] - start[0]) * (point[0] - end[0]) + (point[1] - start[1]) * (point[1] - end[1])
    return dot <= tolerance


def _bounded_line_intersection(
    left: tuple[str, Point2D, Point2D, str],
    right: tuple[str, Point2D, Point2D, str],
) -> Point2D | None:
    _, p, r, left_kind = left
    _, q, s, right_kind = right
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-10:
        return None
    qmp = (q[0] - p[0], q[1] - p[1])
    t = (qmp[0] * s[1] - qmp[1] * s[0]) / denom
    u = (qmp[0] * r[1] - qmp[1] * r[0]) / denom
    if not _parameter_allowed(t, left_kind) or not _parameter_allowed(u, right_kind):
        return None
    return p[0] + t * r[0], p[1] + t * r[1]


def _parameter_allowed(value: float, kind: str) -> bool:
    if kind == "segment":
        return -1e-9 <= value <= 1.0 + 1e-9
    if kind == "ray":
        return value >= -1e-9
    return True


def _near(left: Point2D, right: Point2D, tolerance: float = 1e-7) -> bool:
    return math.hypot(left[0] - right[0], left[1] - right[1]) <= tolerance


def _marker_primitives(
    state: FormalGeometryState,
    render_dim: int | None,
    *,
    show_symbols: bool,
    allow_inferred: bool,
) -> list[RenderPrimitive]:
    if render_dim == 3:
        return []
    primitives: list[RenderPrimitive] = []
    marker_size = _annotation_size(state, render_dim)
    if show_symbols:
        primitives.extend(_equal_length_markers(state, marker_size))
    for region in state.regions.values():
        if not show_symbols or region.kind != "sector_region" or not region.metadata.get("show_central_angle"):
            continue
        primitives.append(
            RenderPrimitive(
                id=f"central_angle:{region.id}",
                kind="angle_marker",
                role="marker",
                geometry={
                    "vertex": _project(state.points[region.metadata["center"]].coords),
                    "arm1": _project(state.points[region.metadata["start"]].coords),
                    "arm2": _project(state.points[region.metadata["end"]].coords),
                    "radius": marker_size * 1.45,
                    "orientation": region.metadata.get("orientation", "ccw"),
                },
                style=MARKER,
                metadata={
                    "relation": "sector_central_angle",
                    "region": region.id,
                    "points": [
                        region.metadata["start"],
                        region.metadata["center"],
                        region.metadata["end"],
                    ],
                },
            )
        )
    for relation in state.relations:
        rel_type = relation.get("type")
        relation_metadata = relation.get("metadata") or {}
        if relation_metadata.get("annotation_role") == "internal" or relation_metadata.get("render_marker") is False:
            continue
        if rel_type == "midpoint":
            if not show_symbols:
                continue
            point_id = relation.get("point")
            endpoints = relation.get("of") or ()
            if point_id in state.points and len(endpoints) == 2 and all(_point_allowed(state, item, render_dim) for item in endpoints):
                mid = _project(state.points[point_id].coords)
                left = _project(state.points[endpoints[0]].coords)
                right = _project(state.points[endpoints[1]].coords)
                primitives.append(
                    _tick_marker(
                        f"tick:{endpoints[0]}-{point_id}",
                        left,
                        mid,
                        point_ids=(endpoints[0], point_id),
                    )
                )
                primitives.append(
                    _tick_marker(
                        f"tick:{point_id}-{endpoints[1]}",
                        mid,
                        right,
                        point_ids=(point_id, endpoints[1]),
                    )
                )
        elif rel_type == "angle_bisector":
            if not show_symbols:
                continue
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
                            "radius": marker_size * 1.25,
                        },
                        style=MARKER,
                        metadata={
                            "relation": "angle_bisector",
                            "mode": relation.get("mode", "internal"),
                            "points": [a, vertex, c],
                        },
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
                    line_dir = _foot_marker_direction(state, line_id, foot, line.direction)
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
                    if show_symbols:
                        primitives.append(
                            RenderPrimitive(
                                id=f"right_angle:{foot_id}",
                                kind="right_angle_marker",
                                role="marker",
                                geometry={"vertex": foot, "arm1": source, "direction2": line_dir, "size": marker_size},
                                style=MARKER,
                                metadata={
                                    "relation": "foot_of_perpendicular",
                                    "vertex": foot_id,
                                    "from": source_id,
                                    "line": line_id,
                                },
                            )
                        )
        elif rel_type == "parallel":
            if not show_symbols or relation.get("metadata", {}).get("show_marker") is False:
                continue
            object_ids = tuple(str(item) for item in relation.get("objects") or ())
            if len(object_ids) == 2:
                primitives.extend(_parallel_markers(state, object_ids, render_dim, marker_size))
        elif rel_type == "perpendicular":
            if not show_symbols or relation.get("metadata", {}).get("show_marker") is False:
                continue
            object_ids = relation.get("objects") or ()
            if len(object_ids) == 2:
                object_ids = _ordered_perpendicular_objects(state, object_ids)
                try:
                    left = state.get_line_like(object_ids[0])
                    right = state.get_line_like(object_ids[1])
                except KeyError:
                    continue
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
                                    "direction1": _foot_marker_direction(state, object_ids[0], vertex, left.direction),
                                    "direction2": _foot_marker_direction(state, object_ids[1], vertex, right.direction),
                                    "size": marker_size,
                                },
                                style=MARKER,
                                metadata={"relation": "perpendicular", "objects": list(object_ids)},
                            )
                        )
    show_inferred = allow_inferred and show_symbols and (
        not state.target or bool(state.target.metadata.get("allow_inferred_markers", True))
    )
    if show_inferred:
        primitives.extend(_polygon_right_angle_markers(state, render_dim, primitives, marker_size))
    return _deduplicate_right_angle_markers(primitives)


def _equal_length_markers(state: FormalGeometryState, marker_size: float) -> list[RenderPrimitive]:
    groups: list[set[str]] = []
    for relation in state.relations:
        metadata = relation.get("metadata", {})
        if relation.get("type") != "equal_length" or metadata.get("show_marker") is False or metadata.get("render_marker") is False or metadata.get("annotation_role") == "internal":
            continue
        members = set(relation.get("segments", relation.get("objects", ())))
        if len(members) < 2 or not members <= set(state.segments):
            continue
        intersecting = [group for group in groups if group & members]
        groups = [group for group in groups if not group & members]
        groups.append(members.union(*intersecting))
    primitives = []
    for group_index, group in enumerate(sorted(groups, key=lambda members: sorted(members))):
        count = group_index + 1
        for segment_id in sorted(group):
            segment = state.segments[segment_id]
            start, end = (_project(state.points[point].coords) for point in (segment.start, segment.end))
            length = math.dist(start, end)
            if length <= 1e-9:
                continue
            for tick_index in range(count):
                fraction = 0.5 + (tick_index - (count - 1) / 2) * marker_size * 0.65 / length
                primitives.append(RenderPrimitive(
                    id=f"equal_tick:{segment_id}:{tick_index}", kind="tick_marker", role="marker",
                    geometry={"start": start, "end": end, "size": marker_size * 0.45, "position": fraction},
                    style=MARKER,
                    metadata={"relation": "equal_length", "segment": segment_id, "group": sorted(group), "count": count},
                ))
    return primitives


def _ordered_perpendicular_objects(state: FormalGeometryState, object_ids: Sequence[str]) -> tuple[str, str]:
    """Keep the constructed perpendicular first for stable annotation IDs."""

    normalized = tuple(str(item) for item in object_ids)
    constructed = [
        object_id
        for object_id in normalized
        if object_id in state.lines and state.lines[object_id].metadata.get("constructed_by") == "perpendicular_through"
    ]
    if len(constructed) == 1:
        return constructed[0], next(item for item in normalized if item != constructed[0])
    return normalized[0], normalized[1]


def _annotation_size(state: FormalGeometryState, render_dim: int | None) -> float:
    points = _visible_projected_points(state, render_dim)
    if not points:
        return 0.18
    span = max(max(point[0] for point in points) - min(point[0] for point in points), max(point[1] for point in points) - min(point[1] for point in points), 1.0)
    return max(0.14, min(0.34, 0.04 * span))


def _unit_direction(direction: Point2D) -> Point2D:
    length = math.hypot(direction[0], direction[1])
    if length < 1e-12:
        return 1.0, 0.0
    return direction[0] / length, direction[1] / length


def _foot_marker_direction(state: FormalGeometryState, line_id: str, foot: Point2D, fallback: Sequence[sp.Expr]) -> Point2D:
    segment = state.segments.get(line_id)
    endpoint_ids = (segment.start, segment.end) if segment is not None else state.get_line_like(line_id).source_points
    if endpoint_ids:
        candidates = []
        for point_id in endpoint_ids:
            point = _project(state.points[point_id].coords)
            direction = (point[0] - foot[0], point[1] - foot[1])
            candidates.append((math.hypot(*direction), direction))
        if max(length for length, _ in candidates) > 1e-9:
            return max(candidates, key=lambda item: item[0])[1]
    return _project_direction(fallback)


def _polygon_right_angle_markers(
    state: FormalGeometryState,
    render_dim: int | None,
    existing: Sequence[RenderPrimitive],
    marker_size: float,
) -> list[RenderPrimitive]:
    occupied = [tuple(primitive.geometry["vertex"]) for primitive in existing if primitive.kind == "right_angle_marker"]
    markers: list[RenderPrimitive] = []
    for polygon in state.polygons.values():
        if len(polygon.vertices) < 3 or not all(_point_allowed(state, point_id, render_dim) for point_id in polygon.vertices):
            continue
        points = [_project(state.points[point_id].coords) for point_id in polygon.vertices]
        candidates: list[tuple[int, str, Point2D, Point2D, Point2D]] = []
        orientation = _polygon_orientation(points)
        for index, vertex_id in enumerate(polygon.vertices):
            previous = points[index - 1]
            vertex = points[index]
            following = points[(index + 1) % len(points)]
            left = (previous[0] - vertex[0], previous[1] - vertex[1])
            right = (following[0] - vertex[0], following[1] - vertex[1])
            denominator = math.hypot(*left) * math.hypot(*right)
            turn = (vertex[0] - previous[0]) * (following[1] - vertex[1]) - (vertex[1] - previous[1]) * (following[0] - vertex[0])
            if denominator < 1e-12 or abs(left[0] * right[0] + left[1] * right[1]) > 1e-8 * denominator:
                continue
            if orientation and turn * orientation <= 0:
                continue
            priority = 0 if str(state.points[vertex_id].metadata.get("constructed_by", "")).startswith("intersection_") else 1
            candidates.append((priority, vertex_id, vertex, previous, following))
        show_all = bool(polygon.metadata.get("show_all_right_angles"))
        for _, vertex_id, vertex, previous, following in sorted(candidates, key=lambda item: (item[0], item[1]))[: None if show_all else 1]:
            if any(_near(vertex, known) for known in (*occupied, *(tuple(marker.geometry["vertex"]) for marker in markers))):
                continue
            markers.append(
                RenderPrimitive(
                    id=f"right_angle:{polygon.id}:{vertex_id}",
                    kind="right_angle_marker",
                    role="marker",
                    geometry={"vertex": vertex, "arm1": previous, "arm2": following, "size": marker_size},
                    style=MARKER,
                    metadata={"relation": "inferred_polygon_right_angle", "polygon": polygon.id, "vertex": vertex_id},
                )
            )
    return markers


def _polygon_orientation(points: Sequence[Point2D]) -> float:
    area2 = sum(points[index][0] * points[(index + 1) % len(points)][1] - points[(index + 1) % len(points)][0] * points[index][1] for index in range(len(points)))
    if abs(area2) < 1e-12:
        return 0.0
    return 1.0 if area2 > 0 else -1.0


def _deduplicate_right_angle_markers(primitives: Sequence[RenderPrimitive]) -> list[RenderPrimitive]:
    seen: list[Point2D] = []
    result: list[RenderPrimitive] = []
    for primitive in primitives:
        if primitive.kind == "right_angle_marker":
            vertex = tuple(primitive.geometry["vertex"])
            if any(_near(vertex, known) for known in seen):
                continue
            seen.append(vertex)
        result.append(primitive)
    return result


def _tick_marker(
    object_id: str,
    start: Point2D,
    end: Point2D,
    *,
    point_ids: tuple[str, str],
) -> RenderPrimitive:
    return RenderPrimitive(
        id=object_id,
        kind="tick_marker",
        role="marker",
        geometry={"start": start, "end": end, "size": 0.08},
        style=MARKER,
        metadata={"relation": "equal_length", "points": list(point_ids)},
    )


def _parallel_markers(
    state: FormalGeometryState,
    object_ids: tuple[str, str],
    render_dim: int | None,
    marker_size: float,
) -> list[RenderPrimitive]:
    markers: list[RenderPrimitive] = []
    for object_id in object_ids:
        geometry = _parallel_marker_geometry(state, object_id, render_dim, marker_size)
        if geometry is None:
            continue
        center, direction = geometry
        if direction[0] < 0 or (abs(direction[0]) < 1e-12 and direction[1] < 0):
            direction = (-direction[0], -direction[1])
        normal = (-direction[1], direction[0])
        size = marker_size * 0.72
        tail = (center[0] - direction[0] * size * 0.55, center[1] - direction[1] * size * 0.55)
        tip = (center[0] + direction[0] * size * 0.55, center[1] + direction[1] * size * 0.55)
        points = [
            (tail[0] + normal[0] * size * 0.48, tail[1] + normal[1] * size * 0.48),
            tip,
            (tail[0] - normal[0] * size * 0.48, tail[1] - normal[1] * size * 0.48),
        ]
        markers.append(
            RenderPrimitive(
                id=f"parallel:{object_id}",
                kind="parallel_marker",
                role="marker",
                geometry={"points": points, "size": size},
                style=MARKER,
                metadata={"relation": "parallel", "objects": list(object_ids), "support": object_id},
            )
        )
    return markers


def _parallel_marker_geometry(
    state: FormalGeometryState,
    object_id: str,
    render_dim: int | None,
    marker_size: float,
) -> tuple[Point2D, Point2D] | None:
    if object_id in state.segments:
        segment = state.segments[object_id]
        if not _point_allowed(state, segment.start, render_dim) or not _point_allowed(state, segment.end, render_dim):
            return None
        start = _project(state.points[segment.start].coords)
        end = _project(state.points[segment.end].coords)
        return ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2), _unit_direction((end[0] - start[0], end[1] - start[1]))
    if object_id in state.lines:
        line = state.lines[object_id]
        if render_dim is not None and line.dim != render_dim:
            return None
        direction = _unit_direction(_project(line.direction))
        anchor = _project(line.point)
        return (anchor[0] + direction[0] * marker_size * 1.5, anchor[1] + direction[1] * marker_size * 1.5), direction
    if object_id in state.rays:
        ray = state.rays[object_id]
        if not _point_allowed(state, ray.start, render_dim):
            return None
        direction = _unit_direction(_project(ray.direction))
        anchor = _project(state.points[ray.start].coords)
        return (anchor[0] + direction[0] * marker_size * 1.5, anchor[1] + direction[1] * marker_size * 1.5), direction
    return None


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


def _point_display_labels(state: FormalGeometryState, render_dim: int | None) -> dict[str, str]:
    visible = [point for point in state.points.values() if point.metadata.get("render") is not False and (render_dim is None or point.dim == render_dim)]
    labels: dict[str, str] = {}
    used: set[str] = set()
    for point in visible:
        display_name = point.metadata.get("display_name")
        if isinstance(display_name, str) and display_name and display_name not in used:
            labels[point.id] = display_name
            used.add(display_name)
    for point in visible:
        if point.id not in labels and re.fullmatch(r"[A-Z](?:[0-9]+)?", point.id) and point.id not in used:
            labels[point.id] = point.id
            used.add(point.id)

    available = iter(_available_point_names(used, len(visible)))
    for point in visible:
        if point.id not in labels:
            labels[point.id] = next(available)
    return labels


def _visible_projected_points(state: FormalGeometryState, render_dim: int | None) -> list[Point2D]:
    return [
        _project(point.coords)
        for point in state.points.values()
        if point.metadata.get("render") is not False and (render_dim is None or point.dim == render_dim)
    ]


def _available_point_names(used: set[str], count: int) -> list[str]:
    sequence = [chr(code) for code in range(ord("A"), ord("Z") + 1)]
    sequence.extend(f"P{index}" for index in range(1, count + len(used) + 1))
    return [label for label in sequence if label not in used]


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
    if object_id in state.regions:
        region = state.regions[object_id]
        if region.kind in {"sector_region", "annulus_region"}:
            circle_id = region.metadata.get("circle") or region.metadata.get("outer_circle")
            if circle_id in state.circles:
                return state.points[state.circles[circle_id].center].dim
        for reference in region.boundary:
            if reference in state.points:
                return state.points[reference].dim
            if reference in state.segments:
                return state.points[state.segments[reference].start].dim
            if reference in state.arcs:
                return state.points[state.circles[state.arcs[reference].circle].center].dim
            if reference in state.polygons:
                return state.points[state.polygons[reference].vertices[0]].dim
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
