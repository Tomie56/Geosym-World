"""Observed-layout projection from an admitted ParseGraph to RenderScene."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from geosym_world.schema import ParseGraph, ParseObject, ParseObjectKind, ParseRegion

from .math_text import format_math_expression
from .quality import inspect_render_scene
from .scene import RenderLabel, RenderPrimitive, RenderScene, RenderStyle, scene_fingerprint


SOURCE_LINE = RenderStyle(stroke="#1f2937", fill="none", stroke_width=2.0)
SOURCE_POINT = RenderStyle(stroke="#111827", fill="#ffffff", stroke_width=1.7)
SOURCE_CIRCLE = RenderStyle(stroke="#475569", fill="none", stroke_width=2.0)
SOURCE_REGION = RenderStyle(stroke="#0f766e", fill="#ccfbf1", stroke_width=1.7, opacity=0.5)
SOURCE_MARKER = RenderStyle(stroke="#0f172a", fill="none", stroke_width=1.8)


def source_graph_to_render_scene(graph: ParseGraph, *, scene_id: str | None = None) -> RenderScene:
    """Project only explicit observed geometry, names, quantities, and markers."""

    if not isinstance(graph, ParseGraph):
        raise TypeError("source_graph_to_render_scene requires a ParseGraph")
    positions = _point_positions(graph)
    objects = {item.id: item for item in graph.objects}
    point_names = graph.point_names()
    primitives: list[RenderPrimitive] = []
    labels: list[RenderLabel] = []
    unsupported: list[str] = []

    materialized_components = {
        reference
        for region in graph.regions
        for reference in (
            *region.holes,
            *region.components,
            *(region.expression.references() if region.expression is not None else ()),
        )
    }
    for region in graph.regions:
        if region.id in materialized_components:
            continue
        primitive = _region_primitive(region, graph, objects, positions)
        if primitive is None:
            unsupported.append(region.id)
        else:
            primitives.append(primitive)

    for candidate in graph.objects:
        primitive = _object_primitive(candidate, objects, positions)
        if primitive is not None:
            primitives.append(primitive)
        elif candidate.kind not in {ParseObjectKind.TEXT, ParseObjectKind.MARKER}:
            unsupported.append(candidate.id)
        point_name = candidate.visible_name or point_names.get(candidate.id)
        if candidate.kind is ParseObjectKind.POINT and point_name and candidate.id in positions:
            labels.append(
                RenderLabel(
                    id=f"label:{candidate.id}",
                    text=point_name,
                    position=positions[candidate.id],
                    offset=(10.0, -10.0),
                )
            )

    for relation in graph.relations:
        marker = _relation_marker(relation.predicate, relation.id, relation.arguments, objects, positions)
        if marker is not None:
            primitives.extend(marker)

    for quantity in graph.quantities:
        anchor = _quantity_anchor(quantity.arguments[0].ref, objects, positions)
        if anchor is None:
            unsupported.append(quantity.id)
            continue
        labels.append(
            RenderLabel(
                id=f"quantity:{quantity.id}",
                text=format_math_expression(quantity.expression.to_sympy()),
                position=anchor,
                offset=(8.0, -10.0),
            )
        )

    scene = RenderScene(
        id=scene_id or f"source_redraw:{graph.id}",
        primitives=tuple(primitives),
        labels=tuple(labels),
        metadata={
            "output_mode": "source_redraw",
            "parse_graph_checksum": graph.checksum(),
            "layout_scope": "observed_pixel_witness",
            "unsupported_candidates": sorted(set(unsupported)),
            "point_labels": {
                candidate.id: candidate.visible_name or point_names.get(candidate.id)
                for candidate in graph.objects
                if candidate.kind is ParseObjectKind.POINT and (candidate.visible_name is not None or candidate.id in point_names)
            },
            "annotation_projection": "explicit_source_only",
        },
    )
    quality = inspect_render_scene(scene)
    return replace(
        scene,
        metadata={
            **scene.metadata,
            "render_quality": quality.to_dict(),
            "scene_fingerprint": scene_fingerprint(scene),
        },
    )


def _point_positions(graph: ParseGraph) -> dict[str, tuple[float, float]]:
    image_size = graph.metadata.get("image_size")
    height = float(image_size[1]) if isinstance(image_size, (list, tuple)) and len(image_size) == 2 else 0.0
    positions = {}
    for candidate in graph.objects:
        if candidate.kind is not ParseObjectKind.POINT:
            continue
        pixel = candidate.geometry.get("pixel")
        if isinstance(pixel, (list, tuple)) and len(pixel) == 2:
            positions[candidate.id] = float(pixel[0]), height - float(pixel[1])
    return positions


def _object_primitive(
    candidate: ParseObject,
    objects: dict[str, ParseObject],
    positions: dict[str, tuple[float, float]],
) -> RenderPrimitive | None:
    if candidate.kind is ParseObjectKind.POINT and candidate.id in positions:
        if candidate.visible is False:
            return None
        return RenderPrimitive(
            id=candidate.id,
            kind="point",
            role="point",
            geometry={"point": positions[candidate.id], "radius": 4.0},
            style=SOURCE_POINT,
            metadata={"parse_candidate_id": candidate.id},
        )
    if candidate.kind is ParseObjectKind.SEGMENT:
        endpoints = _referenced_points(candidate, positions, ("start", "end"))
        if endpoints is not None:
            return RenderPrimitive(candidate.id, "segment", {"points": endpoints}, SOURCE_LINE, "segment")
    if candidate.kind is ParseObjectKind.LINE:
        points = _ordered_reference_positions(candidate, positions, "point")
        if len(points) >= 2:
            direction = points[1][0] - points[0][0], points[1][1] - points[0][1]
            return RenderPrimitive(candidate.id, "line", {"point": points[0], "direction": direction}, SOURCE_LINE, "line")
    if candidate.kind is ParseObjectKind.RAY:
        endpoints = _referenced_points(candidate, positions, ("start", "through"))
        if endpoints is not None:
            direction = endpoints[1][0] - endpoints[0][0], endpoints[1][1] - endpoints[0][1]
            return RenderPrimitive(candidate.id, "ray", {"point": endpoints[0], "direction": direction}, SOURCE_LINE, "ray")
    if candidate.kind is ParseObjectKind.CIRCLE:
        center_ref = next((reference.ref for reference in candidate.references if reference.role == "center"), None)
        radius = candidate.geometry.get("radius_px")
        if center_ref in positions and isinstance(radius, (int, float)) and math.isfinite(float(radius)):
            return RenderPrimitive(
                candidate.id,
                "circle",
                {"center": positions[center_ref], "radius": float(radius)},
                SOURCE_CIRCLE,
                "circle",
            )
    if candidate.kind is ParseObjectKind.ARC:
        geometry = _arc_geometry(candidate, objects, positions)
        if geometry is not None:
            return RenderPrimitive(
                candidate.id,
                "arc",
                geometry,
                SOURCE_CIRCLE,
                "arc",
            )
    if candidate.kind is ParseObjectKind.POLYGON:
        points = _ordered_reference_positions(candidate, positions, "vertex")
        if len(points) >= 3:
            return RenderPrimitive(candidate.id, "polygon", {"points": points}, SOURCE_LINE, "polygon")
    return None


def _region_primitive(
    region: ParseRegion,
    graph: ParseGraph,
    objects: dict[str, ParseObject],
    positions: dict[str, tuple[float, float]],
) -> RenderPrimitive | None:
    regions = {item.id: item for item in graph.regions}
    ring = _region_ring(region, objects, positions)
    hole_ids = list(dict.fromkeys(region.holes))
    holes = [_region_ring(regions[hole], objects, positions) for hole in hole_ids if hole in regions]
    if ring is None and region.expression is not None and region.expression.operator == "difference":
        operands = region.expression.operands
        if len(operands) == 2 and all(operand.ref in regions for operand in operands):
            ring = _region_ring(regions[operands[0].ref], objects, positions)
            hole = _region_ring(regions[operands[1].ref], objects, positions)
            if hole is not None and operands[1].ref not in hole_ids:
                holes.append(hole)
    if ring is None:
        return None
    return RenderPrimitive(
        region.id,
        "polygon",
        {"points": ring, "holes": [hole for hole in holes if hole is not None]},
        SOURCE_REGION,
        "region",
        {"parse_candidate_id": region.id, "holes": list(region.holes)},
    )


def _region_ring(
    region: ParseRegion,
    objects: dict[str, ParseObject],
    positions: dict[str, tuple[float, float]],
) -> list[tuple[float, float]] | None:
    points: list[tuple[float, float]] = []
    for boundary in region.boundary:
        if boundary.ref in positions:
            points.append(positions[boundary.ref])
            continue
        candidate = objects.get(boundary.ref)
        if candidate is None:
            return None
        if candidate.kind is ParseObjectKind.POLYGON:
            ring = _ordered_reference_positions(candidate, positions, "vertex")
            if boundary.orientation == "reverse":
                ring.reverse()
            points.extend(ring)
        elif candidate.kind is ParseObjectKind.SEGMENT:
            endpoints = _referenced_points(candidate, positions, ("start", "end"))
            if endpoints is None:
                return None
            points.extend(reversed(endpoints) if boundary.orientation == "reverse" else endpoints)
        elif candidate.kind is ParseObjectKind.ARC:
            arc_points = _arc_points(candidate, objects, positions)
            if arc_points is None:
                return None
            points.extend(reversed(arc_points) if boundary.orientation == "reverse" else arc_points)
        else:
            return None
    deduplicated = []
    for point in points:
        if not deduplicated or point != deduplicated[-1]:
            deduplicated.append(point)
    if len(deduplicated) > 1 and deduplicated[0] == deduplicated[-1]:
        deduplicated.pop()
    return deduplicated if len(deduplicated) >= 3 else None


def _arc_geometry(
    candidate: ParseObject,
    objects: dict[str, ParseObject],
    positions: dict[str, tuple[float, float]],
) -> dict[str, Any] | None:
    circle_ref = next((reference.ref for reference in candidate.references if reference.role == "circle"), None)
    start_ref = next((reference.ref for reference in candidate.references if reference.role == "start"), None)
    end_ref = next((reference.ref for reference in candidate.references if reference.role == "end"), None)
    circle = objects.get(circle_ref) if circle_ref is not None else None
    if circle is None or circle.kind is not ParseObjectKind.CIRCLE:
        return None
    center_ref = next((reference.ref for reference in circle.references if reference.role == "center"), None)
    if center_ref not in positions or start_ref not in positions or end_ref not in positions:
        return None
    radius = circle.geometry.get("radius_px")
    if radius is None:
        radius = math.dist(positions[center_ref], positions[start_ref])
    if not isinstance(radius, (int, float)) or not math.isfinite(float(radius)) or float(radius) <= 0.0:
        return None
    orientation = candidate.geometry.get("orientation", "ccw")
    if orientation not in {"ccw", "cw"}:
        return None
    center = positions[center_ref]
    start = positions[start_ref]
    end = positions[end_ref]
    endpoint_radius = math.dist(center, end)
    if abs(endpoint_radius - float(radius)) > max(3.0, 0.05 * float(radius)):
        return None
    return {
        "center": center,
        "start": start,
        "end": end,
        "radius": float(radius),
        "orientation": orientation,
    }


def _arc_points(
    candidate: ParseObject,
    objects: dict[str, ParseObject],
    positions: dict[str, tuple[float, float]],
) -> list[tuple[float, float]] | None:
    geometry = _arc_geometry(candidate, objects, positions)
    if geometry is None:
        return None
    center = geometry["center"]
    start = geometry["start"]
    end = geometry["end"]
    start_angle = math.atan2(start[1] - center[1], start[0] - center[0])
    end_angle = math.atan2(end[1] - center[1], end[0] - center[0])
    if geometry["orientation"] == "ccw":
        delta = (end_angle - start_angle) % (2.0 * math.pi)
    else:
        delta = -((start_angle - end_angle) % (2.0 * math.pi))
    if abs(delta) <= 1e-9:
        return None
    steps = min(128, max(12, int(math.ceil(abs(delta) * geometry["radius"] / 2.0))))
    points = [
        (
            center[0] + geometry["radius"] * math.cos(start_angle + delta * index / steps),
            center[1] + geometry["radius"] * math.sin(start_angle + delta * index / steps),
        )
        for index in range(steps + 1)
    ]
    points[0] = start
    points[-1] = end
    return points


def _relation_marker(predicate, relation_id, arguments, objects, positions) -> list[RenderPrimitive] | None:
    object_ids = [argument.ref for argument in arguments]
    if predicate == "perpendicular" and len(object_ids) == 2:
        left = _line_points(objects.get(object_ids[0]), positions)
        right = _line_points(objects.get(object_ids[1]), positions)
        vertex = _line_intersection(left, right) if left is not None and right is not None else None
        if vertex is not None:
            return [
                RenderPrimitive(
                    f"marker:{relation_id}",
                    "right_angle_marker",
                    {"vertex": vertex, "arm1": left[0], "arm2": right[0], "size": _marker_size(positions)},
                    SOURCE_MARKER,
                    "marker",
                    {"parse_relation_id": relation_id},
                )
            ]
    if predicate == "parallel" and len(object_ids) == 2:
        markers = []
        for index, object_id in enumerate(object_ids):
            points = _line_points(objects.get(object_id), positions)
            if points is None:
                return None
            markers.append(_parallel_marker(f"marker:{relation_id}:{index}", points, index + 1))
        return markers
    return None


def _parallel_marker(marker_id: str, points, count: int) -> RenderPrimitive:
    start, end = points
    midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    dx, dy = end[0] - start[0], end[1] - start[1]
    norm = math.hypot(dx, dy) or 1.0
    tangent = dx / norm, dy / norm
    normal = -tangent[1], tangent[0]
    size = min(max(norm * 0.04, 3.0), 12.0)
    path = []
    for index in range(count):
        shift = (index - (count - 1) / 2) * size * 1.4
        center = midpoint[0] + tangent[0] * shift, midpoint[1] + tangent[1] * shift
        path.extend(
            (
                (center[0] - tangent[0] * size + normal[0] * size, center[1] - tangent[1] * size + normal[1] * size),
                center,
                (center[0] + tangent[0] * size + normal[0] * size, center[1] + tangent[1] * size + normal[1] * size),
            )
        )
    return RenderPrimitive(marker_id, "parallel_marker", {"points": path}, SOURCE_MARKER, "marker")


def _quantity_anchor(object_id, objects, positions):
    candidate = objects.get(object_id)
    if candidate is None:
        return positions.get(object_id)
    line = _line_points(candidate, positions)
    if line is not None:
        return (line[0][0] + line[1][0]) / 2, (line[0][1] + line[1][1]) / 2
    if candidate.kind is ParseObjectKind.CIRCLE:
        center_ref = next((reference.ref for reference in candidate.references if reference.role == "center"), None)
        center = positions.get(center_ref)
        radius = candidate.geometry.get("radius_px")
        if center is not None and isinstance(radius, (int, float)):
            return center[0] + float(radius), center[1]
    return positions.get(object_id)


def _line_points(candidate, positions):
    if candidate is None:
        return None
    if candidate.kind is ParseObjectKind.SEGMENT:
        return _referenced_points(candidate, positions, ("start", "end"))
    if candidate.kind in {ParseObjectKind.LINE, ParseObjectKind.RAY}:
        points = _ordered_reference_positions(candidate, positions, "point")
        return points[:2] if len(points) >= 2 else None
    return None


def _referenced_points(candidate, positions, roles):
    refs = []
    for role in roles:
        ref = next((reference.ref for reference in candidate.references if reference.role == role), None)
        if ref not in positions:
            return None
        refs.append(positions[ref])
    return refs


def _ordered_reference_positions(candidate, positions, prefix):
    indexed = []
    for reference in candidate.references:
        if reference.ref not in positions:
            continue
        if reference.role == prefix:
            index = 0
        elif reference.role.startswith(f"{prefix}_") and reference.role[len(prefix) + 1 :].isdigit():
            index = int(reference.role[len(prefix) + 1 :])
        else:
            continue
        indexed.append((index, reference.ref, positions[reference.ref]))
    return [point for _, _, point in sorted(indexed)]


def _line_intersection(left, right):
    (ax, ay), (bx, by) = left
    (cx, cy), (dx, dy) = right
    denominator = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx)
    if abs(denominator) < 1e-9:
        return None
    t = ((cx - ax) * (dy - cy) - (cy - ay) * (dx - cx)) / denominator
    return ax + t * (bx - ax), ay + t * (by - ay)


def _marker_size(positions):
    if not positions:
        return 4.0
    xs = [point[0] for point in positions.values()]
    ys = [point[1] for point in positions.values()]
    return max(3.0, min(12.0, 0.035 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0)))
