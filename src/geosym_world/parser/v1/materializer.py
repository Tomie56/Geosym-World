"""Conservative materialization of Parser v1 witnesses into GeometryState."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp

from geosym_world.errors import InvalidGeometryInput
from geosym_world.schema import (
    Arc, Circle, CompositeRegion, FormalGeometryState, Line, MathExpression,
    Point, Polygon, Ray, Region, Segment, Target,
)
from geosym_world.kernel.geometry2d.analytic import sub

from .contracts import ClauseGraph


@dataclass(frozen=True)
class MaterializationResult:
    state: FormalGeometryState
    unsupported_clause_ids: tuple[str, ...] = ()


def materialize_clause_graph(clause_graph: ClauseGraph) -> MaterializationResult:
    """Materialize admitted clauses into GeometryState and report unsupported clauses."""

    if not isinstance(clause_graph, ClauseGraph):
        raise InvalidGeometryInput("GeometryState materialization requires a ClauseGraph")
    state = FormalGeometryState()
    unsupported: list[str] = []
    for clause in _materialization_order(clause_graph.clauses):
        if clause.status not in {"admitted", "derived"}:
            continue
        if clause.predicate == "object.segment":
            geometry = clause.value if isinstance(clause.value, dict) else {}
            start = geometry.get("p1")
            end = geometry.get("p2")
            if isinstance(start, str) and isinstance(end, str) and start in state.points and end in state.points:
                state.add_segment(Segment(clause.id, start, end, _entity_metadata(clause)))
            else:
                unsupported.append(clause.id)
            continue
        if clause.predicate == "object.line":
            geometry = clause.value if isinstance(clause.value, dict) else {}
            start = geometry.get("p1")
            end = geometry.get("p2")
            if isinstance(start, str) and isinstance(end, str) and start in state.points and end in state.points:
                start_point = state.points[start]
                end_point = state.points[end]
                direction = sub(end_point.coords, start_point.coords)
                state.add_line(Line(clause.id, start_point.coords, direction, (start, end),
                                    _entity_metadata(clause)))
            else:
                unsupported.append(clause.id)
            continue
        if clause.predicate == "object.ray":
            geometry = clause.value if isinstance(clause.value, dict) else {}
            start = geometry.get("start")
            through = geometry.get("through")
            direction = geometry.get("direction")
            if isinstance(start, str) and start in state.points and isinstance(through, str):
                if through in state.points and through != start:
                    state.add_ray(Ray(
                        clause.id,
                        start,
                        sub(state.points[through].coords, state.points[start].coords),
                        through,
                        _entity_metadata(clause),
                    ))
                else:
                    unsupported.append(clause.id)
            elif isinstance(start, str) and start in state.points and isinstance(direction, (list, tuple)):
                state.add_ray(Ray(
                    clause.id,
                    start,
                    tuple(_expression(value) for value in direction),
                    None,
                    _entity_metadata(clause),
                ))
            else:
                unsupported.append(clause.id)
            continue
        if clause.predicate == "object.circle":
            geometry = clause.value if isinstance(clause.value, dict) else {}
            center = geometry.get("center")
            radius = geometry.get("exact_radius")
            if isinstance(center, str) and center in state.points and radius is not None:
                state.add_circle(Circle(
                    clause.id,
                    center,
                    _expression(radius),
                    _entity_metadata(clause),
                ))
            else:
                unsupported.append(clause.id)
            continue
        if clause.predicate == "object.arc":
            geometry = clause.value if isinstance(clause.value, dict) else {}
            circle = geometry.get("circle")
            start = geometry.get("start")
            end = geometry.get("end")
            if (
                isinstance(circle, str) and circle in state.circles
                and isinstance(start, str) and start in state.points
                and isinstance(end, str) and end in state.points and start != end
            ):
                central_angle = geometry.get("central_angle")
                state.add_arc(Arc(
                    clause.id,
                    circle,
                    start,
                    end,
                    _expression(central_angle) if central_angle is not None else None,
                    str(geometry.get("orientation", "ccw")),
                    _entity_metadata(clause),
                ))
            else:
                unsupported.append(clause.id)
            continue
        if clause.predicate == "object.polygon":
            geometry = clause.value if isinstance(clause.value, dict) else {}
            vertices = geometry.get("vertices")
            if isinstance(vertices, (list, tuple)) and len(vertices) >= 3 and all(
                isinstance(vertex, str) and vertex in state.points for vertex in vertices
            ):
                state.add_polygon(Polygon(
                    clause.id,
                    tuple(vertices),
                    _entity_metadata(clause),
                ))
            else:
                unsupported.append(clause.id)
            continue
        if clause.predicate == "region":
            metadata = clause.metadata
            boundary = metadata.get("boundary")
            holes = metadata.get("holes", [])
            components = metadata.get("components", [])
            operation = metadata.get("operation")
            if (
                isinstance(operation, str)
                and operation in {"union", "intersection", "difference", "sum", "partition"}
                and isinstance(components, list)
                and components
                and all(component in state.regions or component in state.composite_regions for component in components)
            ):
                state.add_composite_region(CompositeRegion(
                    clause.id,
                    operation,
                    tuple(components),
                    _entity_metadata(clause, metadata),
                ))
            elif (
                isinstance(boundary, list)
                and len(boundary) >= 3
                and all(isinstance(point_id, str) and point_id in state.points for point_id in boundary)
                and not components
                and isinstance(holes, list)
                and all(
                    isinstance(hole, str)
                    and (
                        hole in state.points
                        or hole in state.segments
                        or hole in state.polygons
                        or hole in state.regions
                    )
                    for hole in holes
                )
            ):
                state.add_region(Region(
                    clause.id,
                    tuple(boundary),
                    metadata=_entity_metadata(clause, metadata),
                    holes=tuple(holes),
                ))
            else:
                unsupported.append(clause.id)
            continue
        if clause.predicate in {"parallel", "perpendicular", "equal_length"}:
            if len(clause.arguments) == 2 and all(_has_line_like(state, item) for item in clause.arguments):
                _add_relation(state, clause, {
                    "id": clause.id, "type": clause.predicate,
                    "objects": list(clause.arguments),
                })
            else:
                unsupported.append(clause.id)
            continue
        if clause.predicate == "equal_angle":
            angles = _equal_angle_arguments(clause.arguments, clause.metadata)
            if angles is not None and all(point_id in state.points for angle in angles for point_id in angle):
                _add_relation(state, clause, {
                    "id": clause.id,
                    "type": clause.predicate,
                    "angles": [list(angle) for angle in angles],
                })
            else:
                unsupported.append(clause.id)
            continue
        if clause.predicate == "collinear":
            if len(clause.arguments) >= 3 and all(item in state.points for item in clause.arguments):
                _add_relation(state, clause, {
                    "id": clause.id, "type": clause.predicate,
                    "points": list(clause.arguments),
                })
            else:
                unsupported.append(clause.id)
            continue
        if clause.predicate in {"point_on_line", "point_on_ray"}:
            if (
                len(clause.arguments) == 2
                and clause.arguments[0] in state.points
                and _has_line_like(state, clause.arguments[1])
            ):
                _add_relation(state, clause, {"id": clause.id, "type": clause.predicate,
                                    "point": clause.arguments[0], "line": clause.arguments[1]})
            else:
                unsupported.append(clause.id)
            continue
        if clause.predicate == "point_on_arc":
            if len(clause.arguments) == 2 and clause.arguments[0] in state.points and clause.arguments[1] in state.arcs:
                _add_relation(state, clause, {"id": clause.id, "type": clause.predicate,
                                    "point": clause.arguments[0], "arc": clause.arguments[1]})
            else:
                unsupported.append(clause.id)
            continue
        if clause.predicate == "point_on_circle":
            if (
                len(clause.arguments) == 2
                and clause.arguments[0] in state.points
                and clause.arguments[1] in state.circles
            ):
                _add_relation(state, clause, {"id": clause.id, "type": clause.predicate,
                                    "point": clause.arguments[0], "circle": clause.arguments[1]})
            else:
                unsupported.append(clause.id)
            continue
        if clause.predicate == "midpoint":
            if (
                len(clause.arguments) == 2
                and clause.arguments[0] in state.points
                and clause.arguments[1] in state.segments
            ):
                _add_relation(state, clause, {"id": clause.id, "type": clause.predicate,
                                    "point": clause.arguments[0], "segment": clause.arguments[1]})
            else:
                unsupported.append(clause.id)
            continue
        if clause.predicate == "foot_of_perpendicular":
            if (
                len(clause.arguments) == 3
                and clause.arguments[0] in state.points
                and clause.arguments[1] in state.points
                and _has_line_like(state, clause.arguments[2])
            ):
                _add_relation(state, clause, {
                    "id": clause.id,
                    "type": clause.predicate,
                    "point": clause.arguments[0],
                    "from": clause.arguments[1],
                    "line": clause.arguments[2],
                })
            else:
                unsupported.append(clause.id)
            continue
        if clause.predicate == "angle_bisector":
            payload = _angle_bisector_payload(clause.arguments, clause.metadata)
            if payload is not None:
                line_id, angle, mode = payload
                if line_id in state.segments or line_id in state.lines or line_id in state.rays:
                    _add_relation(state, clause, {
                        "id": clause.id,
                        "type": clause.predicate,
                        "line": line_id,
                        "angle": list(angle),
                        "mode": mode,
                    })
                else:
                    unsupported.append(clause.id)
            else:
                unsupported.append(clause.id)
            continue
        if clause.predicate == "point_on_segment_ratio":
            ratio = clause.metadata.get("ratio")
            if (
                len(clause.arguments) == 2
                and clause.arguments[0] in state.points
                and clause.arguments[1] in state.segments
                and ratio is not None
            ):
                segment = state.segments[clause.arguments[1]]
                _add_relation(state, clause, {
                    "id": clause.id,
                    "type": clause.predicate,
                    "point": clause.arguments[0],
                    "segment": [segment.start, segment.end],
                    "ratio": str(_expression(ratio)),
                })
            else:
                unsupported.append(clause.id)
            continue
        if clause.predicate == "point_on_ray_distance":
            distance = clause.metadata.get("distance")
            if (
                len(clause.arguments) == 2
                and clause.arguments[0] in state.points
                and clause.arguments[1] in state.rays
                and distance is not None
            ):
                _add_relation(state, clause, {
                    "id": clause.id,
                    "type": clause.predicate,
                    "point": clause.arguments[0],
                    "ray": clause.arguments[1],
                    "distance": str(_expression(distance)),
                })
            else:
                unsupported.append(clause.id)
            continue
        if clause.predicate == "proportional_lengths":
            ratios = clause.metadata.get("ratios")
            value = clause.metadata.get("value")
            if (
                isinstance(ratios, list)
                and len(ratios) >= 2
                and all(isinstance(pair, list) and len(pair) == 2 for pair in ratios)
                and all(
                    all(isinstance(segment_id, str) and segment_id in state.segments for segment_id in pair)
                    for pair in ratios
                )
                and value is not None
            ):
                _add_relation(state, clause, {
                    "id": clause.id,
                    "type": clause.predicate,
                    "ratios": ratios,
                    "value": str(_expression(value)),
                })
            else:
                unsupported.append(clause.id)
            continue
        if clause.predicate == "angle_from_line":
            angle = clause.metadata.get("angle")
            if len(clause.arguments) == 3 and angle is not None:
                line_id, _point_id, base_line_id = clause.arguments
                if _has_line_like(state, line_id) and _has_line_like(state, base_line_id):
                    _add_relation(state, clause, {
                        "id": clause.id,
                        "type": clause.predicate,
                        "line": line_id,
                        "base_line": base_line_id,
                        "angle": str(_expression(angle)),
                    })
                else:
                    unsupported.append(clause.id)
            else:
                unsupported.append(clause.id)
            continue
        if clause.predicate == "intersection":
            objects, points = _intersection_arguments(clause.arguments, clause.metadata)
            if (
                len(objects) == 2
                and len(points) >= 1
                and all(
                    _has_line_like(state, object_id)
                    or object_id in state.circles
                    or object_id in state.arcs
                    for object_id in objects
                )
                and all(point_id in state.points for point_id in points)
            ):
                _add_relation(state, clause, {
                    "id": clause.id,
                    "type": clause.predicate,
                    "objects": list(objects),
                    "points": list(points),
                })
            else:
                unsupported.append(clause.id)
            continue
        if clause.predicate == "chord":
            roles = clause.metadata.get("argument_roles") if isinstance(clause.metadata, dict) else None
            if isinstance(roles, list) and len(roles) == len(clause.arguments):
                values = dict(zip(roles, clause.arguments))
                chord_id = values.get("object") or values.get("segment")
                circle_id = values.get("circle")
                point_ids = tuple(
                    argument for argument, role in zip(clause.arguments, roles)
                    if str(role).startswith("point")
                )
            else:
                chord_id = clause.arguments[0] if clause.arguments else None
                circle_id = clause.arguments[1] if len(clause.arguments) > 1 else None
                point_ids = tuple(clause.arguments[2:])
            if (
                isinstance(chord_id, str) and chord_id in state.segments
                and isinstance(circle_id, str) and circle_id in state.circles
                and (not point_ids or all(point_id in state.points for point_id in point_ids))
            ):
                relation = {"id": clause.id, "type": clause.predicate,
                            "object": chord_id, "circle": circle_id}
                if point_ids:
                    relation["points"] = list(point_ids)
                _add_relation(state, clause, relation)
            else:
                unsupported.append(clause.id)
            continue
        if clause.predicate in {"similar", "congruent"}:
            if len(clause.arguments) == 2 and all(argument in state.polygons for argument in clause.arguments):
                relation = {
                    "id": clause.id,
                    "type": clause.predicate,
                    "polygons": list(clause.arguments),
                }
                mapping = clause.metadata.get("mapping")
                if isinstance(mapping, dict):
                    relation["mapping"] = dict(mapping)
                _add_relation(state, clause, relation)
            else:
                unsupported.append(clause.id)
            continue
        if clause.predicate == "tangent":
            if len(clause.arguments) == 3:
                point_id, line_id, circle_id = clause.arguments
                if point_id in state.points and _has_line_like(state, line_id) and circle_id in state.circles:
                    _add_relation(state, clause, {"id": clause.id, "type": clause.predicate,
                                        "point": point_id, "line": line_id, "circle": circle_id})
                else:
                    unsupported.append(clause.id)
            else:
                unsupported.append(clause.id)
            continue
        if clause.predicate == "target":
            value = clause.value if isinstance(clause.value, dict) else {}
            target_type = value.get("type")
            target_object = value.get("object")
            if isinstance(target_type, str) and isinstance(target_object, str) and _has_object(state, target_object):
                target_metadata = dict(clause.metadata)
                if "expression" in value:
                    target_metadata["expression"] = value["expression"]
                state.set_target(Target(
                    target_type,
                    target_object,
                    target_metadata,
                    str(value.get("mode", "value")),
                    dict(value["expectation"]) if isinstance(value.get("expectation"), dict) else None,
                ))
            else:
                unsupported.append(clause.id)
            continue
        if clause.predicate != "object.point":
            unsupported.append(clause.id)
            continue
        coordinates = clause.value.get("exact_coords") if isinstance(clause.value, dict) else None
        if not isinstance(coordinates, (list, tuple)) or not coordinates:
            unsupported.append(clause.id)
            continue
        point_id = clause.arguments[0]
        point = Point(id=point_id, coords=tuple(_expression(value) for value in coordinates),
                      metadata=_entity_metadata(clause))
        state.add_point(point)
    state.validate()
    return MaterializationResult(state=state, unsupported_clause_ids=tuple(unsupported))


def materialize_exact_point_state(clause_graph: ClauseGraph) -> MaterializationResult:
    """Compatibility alias for the clause graph materializer."""

    return materialize_clause_graph(clause_graph)


def _expression(value: Any) -> sp.Expr:
    if isinstance(value, dict):
        return MathExpression.from_dict(value).to_sympy()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return sp.sympify(value)
    raise InvalidGeometryInput("Exact point coordinates must be numeric or MathExpression payloads")


def _add_relation(state: FormalGeometryState, clause: Any, payload: dict[str, Any]) -> None:
    metadata = {
        **dict(clause.metadata),
        "parser_clause_id": clause.id,
        "provenance": list(clause.provenance),
        "dependencies": list(clause.dependencies),
    }
    state.add_relation({**payload, "metadata": metadata})


def _entity_metadata(clause: Any, metadata: Any = None) -> dict[str, Any]:
    return {**dict(metadata if metadata is not None else clause.metadata), "parser_clause_id": clause.id}


def _has_line_like(state: FormalGeometryState, object_id: str) -> bool:
    return object_id in state.lines or object_id in state.segments or object_id in state.rays


def _has_object(state: FormalGeometryState, object_id: str) -> bool:
    return any(
        object_id in collection
        for collection in (
            state.points,
            state.lines,
            state.segments,
            state.rays,
            state.circles,
            state.arcs,
            state.polygons,
            state.regions,
            state.composite_regions,
        )
    )


def _equal_angle_arguments(arguments: tuple[str, ...], metadata: Any) -> tuple[tuple[str, str, str], ...] | None:
    roles = metadata.get("argument_roles") if isinstance(metadata, dict) else None
    if isinstance(roles, list) and len(roles) == len(arguments):
        grouped: dict[str, dict[str, str]] = {}
        for argument, role in zip(arguments, roles):
            parts = str(role).split("_")
            if len(parts) < 3 or parts[0] != "angle":
                return None
            if parts[2] == "arm" and len(parts) == 4 and parts[3] in {"0", "1"}:
                key = f"arm_{parts[3]}"
            elif parts[2] == "vertex" and len(parts) == 3:
                key = "vertex"
            else:
                return None
            grouped.setdefault(parts[1], {})[key] = argument
        result = []
        for angle_index in sorted(grouped):
            angle = grouped[angle_index]
            if not {"arm_0", "vertex", "arm_1"}.issubset(angle):
                return None
            result.append((angle["arm_0"], angle["vertex"], angle["arm_1"]))
        return tuple(result) if len(result) == 2 else None
    if len(arguments) == 6:
        return (tuple(arguments[:3]), tuple(arguments[3:]))
    return None


def _angle_bisector_payload(
    arguments: tuple[str, ...], metadata: Any
) -> tuple[str, tuple[str, str, str], str] | None:
    roles = metadata.get("argument_roles") if isinstance(metadata, dict) else None
    if not isinstance(roles, list) or len(roles) != len(arguments):
        return None
    role_values = dict(zip(roles, arguments))
    line_id = role_values.get("line")
    vertex = role_values.get("vertex")
    arms = [role_values.get("arm_0"), role_values.get("arm_1")]
    mode = str(metadata.get("mode", "internal"))
    if (
        not isinstance(line_id, str)
        or not isinstance(vertex, str)
        or any(not isinstance(arm, str) for arm in arms)
        or mode not in {"internal", "external"}
    ):
        return None
    return line_id, (arms[0], vertex, arms[1]), mode


def _intersection_arguments(arguments: tuple[str, ...], metadata: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    roles = metadata.get("argument_roles") if isinstance(metadata, dict) else None
    if isinstance(roles, list) and len(roles) == len(arguments):
        objects = tuple(argument for argument, role in zip(arguments, roles) if role in {"support", "object"})
        points = tuple(argument for argument, role in zip(arguments, roles) if str(role).startswith("point"))
        return objects, points
    return arguments[:2], arguments[2:]


def _materialization_order(clauses: tuple[Any, ...]) -> tuple[Any, ...]:
    clause_by_id = {clause.id: clause for clause in clauses}
    original_index = {clause.id: index for index, clause in enumerate(clauses)}
    reference_owner: dict[str, str] = {}
    for clause in clauses:
        if not clause.predicate.startswith("object."):
            continue
        reference_owner[clause.id] = clause.id
    dependencies = {
        clause.id: {
            reference_owner.get(reference, reference)
            for reference in _string_values((clause.arguments, clause.value, clause.metadata, clause.dependencies))
            if reference_owner.get(reference, reference) in clause_by_id
            and reference_owner.get(reference, reference) != clause.id
        }
        for clause in clauses
    }
    ordered: list[Any] = []
    remaining = set(clause_by_id)
    while remaining:
        ready = [
            clause_by_id[clause_id]
            for clause_id in remaining
            if not (dependencies[clause_id] & remaining)
        ]
        if not ready:
            raise InvalidGeometryInput(
                f"Parser clause dependencies contain a cycle: {sorted(remaining)}"
            )
        ready.sort(key=lambda clause: (_materialization_rank(clause.predicate), original_index[clause.id]))
        ordered.extend(ready)
        remaining.difference_update(clause.id for clause in ready)
    return tuple(ordered)


def _string_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, dict):
        return tuple(item for child in value.values() for item in _string_values(child))
    if isinstance(value, (list, tuple)):
        return tuple(item for child in value for item in _string_values(child))
    return ()


def _materialization_rank(predicate: str) -> int:
    if predicate == "object.point":
        return 0
    if predicate.startswith("object."):
        return 1
    if predicate == "region":
        return 2
    if predicate == "target":
        return 4
    return 3
