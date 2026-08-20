"""Exact boolean operations for supported simple 2D polygons."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cmp_to_key
from typing import Any, Sequence

import sympy as sp

from geosym_world.errors import DegenerateGeometryError, UnsupportedConstructionError
from .analytic import Coord, cross2, dot, polygon_area_2d, sub


@dataclass(frozen=True)
class RegionGeometry2D:
    """Renderer/solver-neutral boundary resolution result."""

    status: str
    boundary: tuple[Coord, ...] = ()
    holes: tuple[tuple[Coord, ...], ...] = ()
    components: tuple[tuple[Coord, ...], ...] = ()
    source_ids: tuple[str, ...] = ()
    reason: str | None = None

    @property
    def resolved(self) -> bool:
        return self.status in {"polygon", "polygon_with_holes", "multi_polygon"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "boundary": [[str(value) for value in point] for point in self.boundary],
            "holes": [[[str(value) for value in point] for point in hole] for hole in self.holes],
            "components": [[[str(value) for value in point] for point in component] for component in self.components],
            "source_ids": list(self.source_ids),
            "reason": self.reason,
        }


def resolve_region_geometry_2d(state: Any, region_id: str, *, _stack: tuple[str, ...] = ()) -> RegionGeometry2D:
    """Resolve only boundaries that can be represented without guessing.

    Polygon regions and supported simple polygon booleans are materialized.
    Other composite operations remain explicit area-only or unsupported.
    """

    if region_id in _stack:
        return RegionGeometry2D("unsupported", source_ids=(region_id,), reason="cyclic_region_reference")
    if region_id in state.regions:
        region = state.regions[region_id]
        if region.kind != "polygon_region":
            return RegionGeometry2D("parametric", source_ids=(region_id,), reason=f"region_kind:{region.kind}")
        boundary_ids = region.boundary
        polygon_id = region.metadata.get("polygon")
        if polygon_id and len(boundary_ids) == 0:
            boundary_ids = (polygon_id,)
        boundary = _resolve_boundary(state, boundary_ids)
        if boundary is None:
            return RegionGeometry2D("unsupported", source_ids=(region_id,), reason="boundary_not_materializable")
        boundary_issue = _polygon_boundary_issue(boundary)
        if boundary_issue is not None:
            return RegionGeometry2D("unsupported", source_ids=(region_id, *boundary_ids), reason=boundary_issue)
        holes: list[tuple[Coord, ...]] = []
        source_ids = [region_id, *boundary_ids]
        for hole_id in region.holes:
            hole = resolve_region_geometry_2d(state, hole_id, _stack=(*_stack, region_id))
            if hole.status != "polygon" or not hole.boundary:
                return RegionGeometry2D("unsupported", source_ids=tuple((*source_ids, hole_id)), reason="hole_not_materializable")
            if not _polygon_strictly_inside(hole.boundary, boundary):
                return RegionGeometry2D("unsupported", source_ids=tuple((*source_ids, *hole.source_ids)), reason="hole_outside_boundary")
            if any(_polygons_touch_or_overlap(hole.boundary, existing) for existing in holes):
                return RegionGeometry2D(
                    "unsupported",
                    source_ids=tuple((*source_ids, *hole.source_ids)),
                    reason="holes_touch_or_overlap",
                )
            holes.append(hole.boundary)
            source_ids.extend(hole.source_ids)
        return RegionGeometry2D(
            "polygon_with_holes" if holes else "polygon",
            boundary=boundary,
            holes=tuple(holes),
            source_ids=tuple(dict.fromkeys(source_ids)),
        )
    if region_id not in state.composite_regions:
        return RegionGeometry2D("unsupported", source_ids=(region_id,), reason="unknown_region")
    composite = state.composite_regions[region_id]
    if composite.operation == "partition":
        return _resolve_partition_geometry(state, region_id, composite, _stack)
    formula = composite.metadata.get("area_formula")
    if (
        composite.operation == "difference"
        and isinstance(formula, dict)
        and formula.get("rule") == "region_difference"
        and formula.get("mode") == "boundary_notch"
    ):
        return _resolve_rectangular_boundary_notch(state, region_id, composite, _stack)
    if len(composite.components) != 2 or composite.operation not in {"union", "intersection", "difference"}:
        return RegionGeometry2D("area_only", source_ids=(region_id, *composite.components), reason="operation_boundary_not_materialized")
    components = [resolve_region_geometry_2d(state, component, _stack=(*_stack, region_id)) for component in composite.components]
    if not all(item.status == "polygon" and not item.holes and not item.components for item in components):
        return RegionGeometry2D("unsupported", source_ids=(region_id, *composite.components), reason="boolean_requires_simple_polygons")
    try:
        boundaries = boolean_polygon_boundaries_2d(components[0].boundary, components[1].boundary, composite.operation)
    except Exception as exc:
        return RegionGeometry2D("unsupported", source_ids=(region_id, *composite.components), reason=str(exc))
    if not boundaries:
        return RegionGeometry2D("empty", source_ids=(region_id, *composite.components), reason="empty_or_degenerate_boolean_result")
    if any(len(boundary) < 3 or polygon_area_2d(boundary) == 0 for boundary in boundaries):
        return RegionGeometry2D("unsupported", source_ids=(region_id, *composite.components), reason="degenerate_boolean_boundary")
    outer_loops = tuple(boundary for boundary in boundaries if _compare(_signed_area(boundary), 0) > 0)
    holes = tuple(boundary for boundary in boundaries if _compare(_signed_area(boundary), 0) < 0)
    if not outer_loops:
        return RegionGeometry2D("unsupported", source_ids=(region_id, *composite.components), reason="boolean_boundary_has_no_outer_loop")
    outer = outer_loops[0]
    components_out = outer_loops[1:]
    return RegionGeometry2D(
        "multi_polygon" if components_out else ("polygon_with_holes" if holes else "polygon"),
        boundary=outer,
        holes=holes,
        components=components_out,
        source_ids=(region_id, *components[0].source_ids, *components[1].source_ids),
    )


def _resolve_partition_geometry(state: Any, region_id: str, composite: Any, stack: tuple[str, ...]) -> RegionGeometry2D:
    parent_id = composite.metadata.get("parent_region")
    if parent_id not in state.regions:
        return RegionGeometry2D("unsupported", source_ids=(region_id,), reason="partition_parent_missing")
    parent = resolve_region_geometry_2d(state, parent_id, _stack=(*stack, region_id))
    components = [resolve_region_geometry_2d(state, component, _stack=(*stack, region_id)) for component in composite.components]
    if parent.status != "polygon" or not all(item.status == "polygon" for item in components):
        return RegionGeometry2D(
            "unsupported",
            source_ids=(region_id, parent_id, *composite.components),
            reason="partition_requires_simple_polygon_regions",
        )
    if any(any(not _point_in_polygon(point, parent.boundary) for point in item.boundary) for item in components):
        return RegionGeometry2D(
            "unsupported",
            source_ids=(region_id, parent_id, *composite.components),
            reason="partition_component_outside_parent",
        )
    for left_index, left in enumerate(components):
        for right in components[left_index + 1 :]:
            try:
                overlap = intersection_polygon_2d(left.boundary, right.boundary)
            except Exception:
                return RegionGeometry2D(
                    "unsupported",
                    source_ids=(region_id, parent_id, *composite.components),
                    reason="partition_components_must_be_convex",
                )
            if overlap and polygon_area_2d(overlap) != 0:
                return RegionGeometry2D(
                    "unsupported",
                    source_ids=(region_id, parent_id, *composite.components),
                    reason="partition_components_overlap",
                )
    component_area = sum((polygon_area_2d(item.boundary) for item in components), sp.Integer(0))
    if sp.simplify(component_area - polygon_area_2d(parent.boundary)) != 0:
        return RegionGeometry2D(
            "unsupported",
            source_ids=(region_id, parent_id, *composite.components),
            reason="partition_area_does_not_cover_parent",
        )
    return RegionGeometry2D(
        "polygon",
        boundary=parent.boundary,
        source_ids=tuple(dict.fromkeys((region_id, *parent.source_ids, *(source for item in components for source in item.source_ids)))),
    )


def _resolve_rectangular_boundary_notch(state: Any, region_id: str, composite: Any, stack: tuple[str, ...]) -> RegionGeometry2D:
    if len(composite.components) != 2:
        return RegionGeometry2D("unsupported", source_ids=(region_id,), reason="boundary_notch_requires_two_components")
    outer = resolve_region_geometry_2d(state, composite.components[0], _stack=(*stack, region_id))
    notch = resolve_region_geometry_2d(state, composite.components[1], _stack=(*stack, region_id))
    if outer.status != "polygon" or notch.status != "polygon":
        return RegionGeometry2D("unsupported", source_ids=(region_id, *composite.components), reason="boundary_notch_requires_simple_polygons")
    outer_bounds = _axis_aligned_rectangle_bounds(outer.boundary)
    notch_bounds = _axis_aligned_rectangle_bounds(notch.boundary)
    if outer_bounds is None or notch_bounds is None:
        return RegionGeometry2D("unsupported", source_ids=(region_id, *composite.components), reason="boundary_notch_requires_axis_aligned_rectangles")
    ox0, oy0, ox1, oy1 = outer_bounds
    nx0, ny0, nx1, ny1 = notch_bounds
    within_outer = (
        _compare(ox0, nx0) <= 0
        and _compare(nx0, nx1) < 0
        and _compare(nx1, ox1) <= 0
        and _compare(oy0, ny0) <= 0
        and _compare(ny0, ny1) < 0
        and _compare(ny1, oy1) <= 0
    )
    if not within_outer:
        return RegionGeometry2D("unsupported", source_ids=(region_id, *composite.components), reason="boundary_notch_outside_outer_rectangle")
    corners = {
        "bottom_left": _compare(nx0, ox0) == 0 and _compare(ny0, oy0) == 0,
        "bottom_right": _compare(nx1, ox1) == 0 and _compare(ny0, oy0) == 0,
        "top_right": _compare(nx1, ox1) == 0 and _compare(ny1, oy1) == 0,
        "top_left": _compare(nx0, ox0) == 0 and _compare(ny1, oy1) == 0,
    }
    touching = [corner for corner, matched in corners.items() if matched]
    if len(touching) != 1:
        return RegionGeometry2D("unsupported", source_ids=(region_id, *composite.components), reason="boundary_notch_must_remove_one_outer_corner")
    boundary_by_corner = {
        "bottom_left": ((nx1, oy0), (ox1, oy0), (ox1, oy1), (ox0, oy1), (ox0, ny1), (nx1, ny1)),
        "bottom_right": ((ox0, oy0), (nx0, oy0), (nx0, ny1), (ox1, ny1), (ox1, oy1), (ox0, oy1)),
        "top_right": ((ox0, oy0), (ox1, oy0), (ox1, ny0), (nx0, ny0), (nx0, oy1), (ox0, oy1)),
        "top_left": ((ox0, oy0), (ox1, oy0), (ox1, oy1), (nx1, oy1), (nx1, ny0), (ox0, ny0)),
    }
    boundary = tuple(tuple(sp.simplify(value) for value in point) for point in boundary_by_corner[touching[0]])
    issue = _polygon_boundary_issue(boundary)
    if issue is not None:
        return RegionGeometry2D("unsupported", source_ids=(region_id, *composite.components), reason=issue)
    return RegionGeometry2D(
        "polygon",
        boundary=boundary,
        source_ids=tuple(dict.fromkeys((region_id, *outer.source_ids, *notch.source_ids))),
    )


def _axis_aligned_rectangle_bounds(points: Sequence[Coord]) -> tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr] | None:
    if len(points) != 4:
        return None
    xs = tuple(dict.fromkeys(point[0] for point in points))
    ys = tuple(dict.fromkeys(point[1] for point in points))
    if len(xs) != 2 or len(ys) != 2:
        return None
    corners = {(x, y) for x in xs for y in ys}
    if set(points) != corners:
        return None
    x0, x1 = sorted(xs, key=cmp_to_key(_compare))
    y0, y1 = sorted(ys, key=cmp_to_key(_compare))
    if _compare(x0, x1) >= 0 or _compare(y0, y1) >= 0:
        return None
    return x0, y0, x1, y1


def intersection_polygon_2d(left: Sequence[Coord], right: Sequence[Coord]) -> tuple[Coord, ...]:
    first = _as_convex_ccw(left, "left")
    second = _as_convex_ccw(right, "right")
    return _intersection_polygon(first, second)


def boolean_polygon_boundaries_2d(
    left: Sequence[Coord],
    right: Sequence[Coord],
    operation: str,
) -> tuple[tuple[Coord, ...], ...]:
    """Materialize an exact boolean of two simple polygons.

    Collinear overlapping edges remain unsupported because their ownership is
    ambiguous without a dedicated coincident-edge normalization pass.
    """

    if operation not in {"union", "intersection", "difference"}:
        raise ValueError(f"Unsupported polygon boolean operation: {operation}")
    first = _as_simple_ccw(left, "left")
    second = _as_simple_ccw(right, "right")
    if _same_polygon_boundary(first, second):
        return () if operation == "difference" else (first,)
    pieces: list[tuple[Coord, Coord]] = []
    for polygon, other, reverse in ((first, second, False), (second, first, operation == "difference")):
        for start, end in _split_polygon_edges(polygon, other):
            midpoint = tuple(sp.simplify((start[index] + end[index]) / 2) for index in range(2))
            inside_other = _point_in_polygon(midpoint, other)
            if operation == "intersection":
                keep = inside_other
            elif operation == "difference":
                keep = (not inside_other) if polygon is first else _point_in_polygon(midpoint, first)
            else:
                keep = not inside_other
            if keep:
                pieces.append((end, start) if reverse else (start, end))
    return _stitch_boundary_pieces(pieces)


def region_geometry_area_2d(geometry: RegionGeometry2D) -> sp.Expr:
    """Compute exact area from a materialized outer, component, and hole set."""

    if not geometry.resolved:
        raise UnsupportedConstructionError(f"Region geometry is not resolved: {geometry.status}")
    return sp.simplify(
        polygon_area_2d(geometry.boundary)
        + sum((polygon_area_2d(component) for component in geometry.components), sp.Integer(0))
        - sum((polygon_area_2d(hole) for hole in geometry.holes), sp.Integer(0))
    )


def _polygon_boundary_issue(points: Sequence[Coord]) -> str | None:
    if len(points) < 3:
        return "boundary_needs_three_points"
    if any(len(point) != 2 for point in points):
        return "boundary_not_2d"
    if len({tuple(point) for point in points}) != len(points):
        return "boundary_repeats_vertex"
    if polygon_area_2d(points) == 0:
        return "boundary_is_degenerate"
    for left_index in range(len(points)):
        left = (points[left_index], points[(left_index + 1) % len(points)])
        for right_index in range(left_index + 1, len(points)):
            if right_index in {left_index, (left_index + 1) % len(points), (left_index - 1) % len(points)}:
                continue
            right = (points[right_index], points[(right_index + 1) % len(points)])
            if _segments_intersect(left, right):
                return "boundary_self_intersects"
    return None


def _segments_intersect(left: tuple[Coord, Coord], right: tuple[Coord, Coord]) -> bool:
    a, b = left
    c, d = right
    orientations = (
        _orientation(a, b, c),
        _orientation(a, b, d),
        _orientation(c, d, a),
        _orientation(c, d, b),
    )
    if orientations[0] == 0 and _on_segment(c, a, b):
        return True
    if orientations[1] == 0 and _on_segment(d, a, b):
        return True
    if orientations[2] == 0 and _on_segment(a, c, d):
        return True
    if orientations[3] == 0 and _on_segment(b, c, d):
        return True
    return orientations[0] != orientations[1] and orientations[2] != orientations[3]


def _polygon_strictly_inside(inner: Sequence[Coord], outer: Sequence[Coord]) -> bool:
    if any(_point_on_polygon_boundary(point, outer) or not _point_in_polygon(point, outer) for point in inner):
        return False
    return not _polygon_boundaries_intersect(inner, outer)


def _polygons_touch_or_overlap(left: Sequence[Coord], right: Sequence[Coord]) -> bool:
    if _polygon_boundaries_intersect(left, right):
        return True
    return _point_in_polygon(left[0], right) or _point_in_polygon(right[0], left)


def _polygon_boundaries_intersect(left: Sequence[Coord], right: Sequence[Coord]) -> bool:
    return any(
        _segments_intersect(
            (left[left_index], left[(left_index + 1) % len(left)]),
            (right[right_index], right[(right_index + 1) % len(right)]),
        )
        for left_index in range(len(left))
        for right_index in range(len(right))
    )


def _point_on_polygon_boundary(point: Coord, polygon: Sequence[Coord]) -> bool:
    return any(_on_segment(point, polygon[index], polygon[(index + 1) % len(polygon)]) for index in range(len(polygon)))


def _orientation(a: Coord, b: Coord, c: Coord) -> int:
    return _compare(cross2(sub(b, a), sub(c, a)), 0)


def _on_segment(point: Coord, start: Coord, end: Coord) -> bool:
    return (
        _orientation(start, end, point) == 0
        and _compare(point[0], min(start[0], end[0])) >= 0
        and _compare(point[0], max(start[0], end[0])) <= 0
        and _compare(point[1], min(start[1], end[1])) >= 0
        and _compare(point[1], max(start[1], end[1])) <= 0
    )


def _point_in_polygon(point: Coord, polygon: Sequence[Coord]) -> bool:
    inside = False
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        if _on_segment(point, start, end):
            return True
        start_above = _compare(start[1], point[1]) > 0
        end_above = _compare(end[1], point[1]) > 0
        if start_above == end_above:
            continue
        crossing_x = sp.simplify(start[0] + (point[1] - start[1]) * (end[0] - start[0]) / (end[1] - start[1]))
        if _compare(crossing_x, point[0]) > 0:
            inside = not inside
    return inside


def _split_polygon_edges(polygon: Sequence[Coord], other: Sequence[Coord]) -> tuple[tuple[Coord, Coord], ...]:
    pieces: list[tuple[Coord, Coord]] = []
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        parameters: list[sp.Expr] = [sp.Integer(0), sp.Integer(1)]
        edge = sub(end, start)
        for other_index, other_start in enumerate(other):
            other_end = other[(other_index + 1) % len(other)]
            other_edge = sub(other_end, other_start)
            denominator = sp.simplify(cross2(edge, other_edge))
            if denominator == 0:
                if _compare(cross2(sub(other_start, start), edge), 0) == 0:
                    other_parameters = [
                        sp.simplify(dot(sub(point, start), edge) / dot(edge, edge))
                        for point in (other_start, other_end)
                    ]
                    overlap_start = _max_expr(sp.Integer(0), _min_expr(*other_parameters))
                    overlap_end = _min_expr(sp.Integer(1), _max_expr(*other_parameters))
                    if _compare(overlap_end - overlap_start, 0) > 0:
                        raise UnsupportedConstructionError("Boolean operation has collinear overlapping edges")
                continue
            parameter = sp.simplify(cross2(sub(other_start, start), other_edge) / denominator)
            other_parameter = sp.simplify(cross2(sub(other_start, start), edge) / denominator)
            if _between(parameter, 0, 1) and _between(other_parameter, 0, 1):
                parameters.append(parameter)
        parameters = _unique_sorted(parameters)
        for left, right in zip(parameters, parameters[1:]):
            if _compare(left, right) == 0:
                continue
            pieces.append((
                tuple(sp.simplify(start[dim] + left * edge[dim]) for dim in range(2)),
                tuple(sp.simplify(start[dim] + right * edge[dim]) for dim in range(2)),
            ))
    return tuple(pieces)


def _stitch_boundary_pieces(pieces: Sequence[tuple[Coord, Coord]]) -> tuple[tuple[Coord, ...], ...]:
    remaining = list(pieces)
    loops: list[tuple[Coord, ...]] = []
    while remaining:
        start, end = remaining.pop(0)
        loop = [start, end]
        while loop[-1] != loop[0]:
            match = next((index for index, piece in enumerate(remaining) if loop[-1] in piece), None)
            if match is None:
                raise UnsupportedConstructionError("Boolean operation produced an open boundary")
            next_start, next_end = remaining.pop(match)
            if next_end == loop[-1]:
                next_start, next_end = next_end, next_start
            loop.append(next_end)
            if len(loop) > len(pieces) + 1:
                raise UnsupportedConstructionError("Boolean operation produced a cyclic boundary")
        loops.append(tuple(loop[:-1]))
    return tuple(sorted(loops, key=lambda loop: (float(sp.N(min(point[0] for point in loop))), float(sp.N(min(point[1] for point in loop))), len(loop))))


def _point_in_convex(point: Coord, polygon: Sequence[Coord]) -> bool:
    return all(_compare(cross2(sub(polygon[(index + 1) % len(polygon)], polygon[index]), sub(point, polygon[index])), 0) >= 0 for index in range(len(polygon)))


def _contains_polygon(outer: Sequence[Coord], inner: Sequence[Coord]) -> bool:
    return all(_point_in_polygon(point, outer) for point in inner)


def _same_polygon_boundary(left: Sequence[Coord], right: Sequence[Coord]) -> bool:
    if len(left) != len(right):
        return False
    doubled = tuple(right) * 2
    return any(tuple(left) == doubled[offset : offset + len(left)] for offset in range(len(right)))


def _between(value: sp.Expr, lower: int, upper: int) -> bool:
    return _compare(value, lower) >= 0 and _compare(value, upper) <= 0


def _unique_sorted(values: Sequence[sp.Expr]) -> list[sp.Expr]:
    ordered = sorted(values, key=cmp_to_key(_compare))
    unique: list[sp.Expr] = []
    for value in ordered:
        if not unique or _compare(value, unique[-1]) != 0:
            unique.append(value)
    return unique


def _min_expr(*values: sp.Expr) -> sp.Expr:
    result = values[0]
    for value in values[1:]:
        if _compare(value, result) < 0:
            result = value
    return result


def _max_expr(*values: sp.Expr) -> sp.Expr:
    result = values[0]
    for value in values[1:]:
        if _compare(value, result) > 0:
            result = value
    return result


def _resolve_boundary(state: Any, boundary: Sequence[str]) -> tuple[Coord, ...] | None:
    if len(boundary) == 1 and boundary[0] in state.polygons:
        return tuple(state.require_point(point_id).coords for point_id in state.polygons[boundary[0]].vertices)
    if boundary and all(point_id in state.points for point_id in boundary):
        points = tuple(state.require_point(point_id).coords for point_id in boundary)
        return points if all(len(point) == 2 for point in points) else None
    if boundary and all(segment_id in state.segments for segment_id in boundary):
        unused = list(boundary)
        first = state.segments[unused.pop(0)]
        ordered: list[str] = [first.start, first.end]
        while unused:
            current = ordered[-1]
            match_index = next(
                (index for index, segment_id in enumerate(unused)
                 if current in {state.segments[segment_id].start, state.segments[segment_id].end}),
                None,
            )
            if match_index is None:
                return None
            segment_id = unused.pop(match_index)
            segment = state.segments[segment_id]
            ordered.append(segment.end if segment.start == current else segment.start)
        if ordered[-1] != ordered[0]:
            return None
        points = tuple(state.require_point(point_id).coords for point_id in ordered[:-1])
        return points if all(len(point) == 2 for point in points) else None
    return None


def boolean_polygon_area_2d(left: Sequence[Coord], right: Sequence[Coord], operation: str) -> sp.Expr:
    """Return exact area for a supported simple-polygon boolean operation."""

    if operation not in {"union", "intersection", "difference"}:
        raise ValueError(f"Unsupported polygon boolean operation: {operation}")
    try:
        first = _as_convex_ccw(left, "left")
        second = _as_convex_ccw(right, "right")
    except UnsupportedConstructionError:
        boundaries = boolean_polygon_boundaries_2d(left, right, operation)
        return sp.simplify(sum((_signed_area(boundary) for boundary in boundaries), sp.Integer(0)))
    overlap = _intersection_polygon(first, second)
    overlap_area = polygon_area_2d(overlap) if len(overlap) >= 3 else sp.Integer(0)
    first_area = polygon_area_2d(first)
    second_area = polygon_area_2d(second)
    if operation == "intersection":
        return sp.simplify(overlap_area)
    if operation == "difference":
        return sp.simplify(first_area - overlap_area)
    return sp.simplify(first_area + second_area - overlap_area)


def _as_simple_ccw(points: Sequence[Coord], label: str) -> tuple[Coord, ...]:
    if len(points) < 3 or any(len(point) != 2 for point in points):
        raise DegenerateGeometryError(f"{label} polygon must contain at least three 2D points")
    issue = _polygon_boundary_issue(points)
    if issue is not None:
        if issue in {"boundary_repeats_vertex", "boundary_is_degenerate"}:
            raise DegenerateGeometryError(f"{label} polygon is invalid: {issue}")
        raise UnsupportedConstructionError(f"{label} polygon is not simple: {issue}")
    return tuple(reversed(points)) if _compare(_signed_area(points), 0) < 0 else tuple(points)


def _as_convex_ccw(points: Sequence[Coord], label: str) -> tuple[Coord, ...]:
    if len(points) < 3 or any(len(point) != 2 for point in points):
        raise DegenerateGeometryError(f"{label} polygon must contain at least three 2D points")
    if len({tuple(point) for point in points}) != len(points):
        raise DegenerateGeometryError(f"{label} polygon repeats vertices")
    signed_area = _signed_area(points)
    if signed_area == 0:
        raise DegenerateGeometryError(f"{label} polygon is degenerate")
    ordered = tuple(reversed(points)) if _compare(signed_area, 0) < 0 else tuple(points)
    signs = []
    for index in range(len(ordered)):
        a = ordered[index]
        b = ordered[(index + 1) % len(ordered)]
        c = ordered[(index + 2) % len(ordered)]
        turn = sp.simplify(cross2(sub(b, a), sub(c, b)))
        if turn != 0:
            signs.append(_compare(turn, 0))
    if not signs or len(set(signs)) != 1 or signs[0] < 0:
        raise UnsupportedConstructionError(f"{label} polygon boolean requires a simple convex polygon")
    return ordered


def _intersection_polygon(subject: Sequence[Coord], clip: Sequence[Coord]) -> tuple[Coord, ...]:
    output = list(subject)
    for index, clip_start in enumerate(clip):
        clip_end = clip[(index + 1) % len(clip)]
        input_points = output
        output = []
        if not input_points:
            break
        previous = input_points[-1]
        previous_inside = _inside(previous, clip_start, clip_end)
        for current in input_points:
            current_inside = _inside(current, clip_start, clip_end)
            if current_inside != previous_inside:
                output.append(_line_boundary_intersection(previous, current, clip_start, clip_end))
            if current_inside:
                output.append(current)
            previous = current
            previous_inside = current_inside
    return tuple(output)


def _inside(point: Coord, edge_start: Coord, edge_end: Coord) -> bool:
    return _compare(cross2(sub(edge_end, edge_start), sub(point, edge_start)), 0) >= 0


def _line_boundary_intersection(first: Coord, second: Coord, edge_start: Coord, edge_end: Coord) -> Coord:
    direction = sub(second, first)
    edge_direction = sub(edge_end, edge_start)
    denominator = sp.simplify(cross2(direction, edge_direction))
    if denominator == 0:
        return first
    parameter = sp.simplify(cross2(sub(edge_start, first), edge_direction) / denominator)
    return tuple(sp.simplify(first[index] + parameter * direction[index]) for index in range(2))


def _signed_area(points: Sequence[Coord]) -> sp.Expr:
    return sp.simplify(sum(points[index][0] * points[(index + 1) % len(points)][1] - points[(index + 1) % len(points)][0] * points[index][1] for index in range(len(points))) / 2)


def _compare(left: sp.Expr, right: sp.Expr) -> int:
    difference = sp.simplify(left - right)
    if difference == 0 or difference.is_zero is True:
        return 0
    if difference.is_negative is True:
        return -1
    if difference.is_positive is True:
        return 1
    numeric = float(sp.N(difference))
    if abs(numeric) <= 1e-12:
        return 0
    return -1 if numeric < 0 else 1
