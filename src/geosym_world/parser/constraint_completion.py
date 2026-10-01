"""Conservative completion of typed parser constraints into replayable traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Any

import sympy as sp
from sympy.core.sympify import SympifyError

from geosym_world.errors import InvalidGeometryInput
from geosym_world.construction.builder import ConstructionBuilder
from geosym_world.schema import MathExpression, ParseGraph, ParseObject, ParseObjectKind
from geosym_world.schema.canonical import stable_checksum
from geosym_world.schema.parse_graph import math_expression_from_text
from geosym_world.schema.problem_normalize import line_id, normalize_id


COMPLETION_PLAN_SCHEMA_VERSION = "geosym.parser.constraint_completion.v1"


class CompletionStatus(str, Enum):
    EXACT = "exact"
    DETERMINISTIC_COMPLETION = "deterministic_completion"
    BRANCHING_COMPLETION = "branching_completion"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class ConstraintCompletionPlan:
    """A replayable completion proposal, never an admitted parser fact."""

    source_id: str
    parse_graph_checksum: str
    status: CompletionStatus
    trace: tuple[dict[str, Any], ...] = ()
    branches: tuple[tuple[dict[str, Any], ...], ...] = ()
    source_candidate_ids: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = COMPLETION_PLAN_SCHEMA_VERSION

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "parse_graph_checksum": self.parse_graph_checksum,
            "status": self.status.value,
            "trace": list(self.trace),
            "branches": [list(branch) for branch in self.branches],
            "branch_count": len(self.branches),
            "source_candidate_ids": list(self.source_candidate_ids),
            "diagnostics": list(self.diagnostics),
            "metadata": self.metadata,
        }

    def checksum(self) -> str:
        return stable_checksum(self._payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "checksum": self.checksum()}


def build_constraint_completion_plan(
    graph: ParseGraph,
) -> ConstraintCompletionPlan:
    """Build only from exact coordinates or explicit numeric 2D constraints.

    Pixel geometry is intentionally ignored. The canonical builder is used only
    after the graph has supplied enough numeric evidence and a complete segment
    topology. Unsupported theorem-only graphs return a diagnostic plan instead
    of receiving fabricated coordinates.
    """

    if not isinstance(graph, ParseGraph):
        raise InvalidGeometryInput("Constraint completion requires a ParseGraph")

    objects = _semantic_objects(graph.objects)
    points = {item.id: item for item in objects if item.kind is ParseObjectKind.POINT}
    parse_checksum = graph.checksum()
    if not points:
        return _unsupported(graph, "no_semantic_points")

    names = _unique_names(objects)
    segments = _segment_topology(objects, names)
    polygons = _polygon_topology(objects, names)
    if not segments and not polygons:
        return _unsupported(graph, "no_replayable_segment_or_polygon_topology")

    exact_coordinates = _exact_coordinates(points)
    has_exact_coordinates = any("exact_coords" in item.geometry for item in points.values())
    if has_exact_coordinates and not exact_coordinates:
        return _unsupported(graph, "exact_coordinates_not_numeric_2d")
    relation_items = tuple(
        (relation, tuple(_relation_line_name(argument.ref, segments, names) for argument in relation.arguments))
        for relation in graph.relations
        if relation.predicate in {"parallel", "perpendicular"}
        and len(relation.arguments) == 2
        and all(_relation_line_name(argument.ref, segments, names) for argument in relation.arguments)
    )
    if exact_coordinates:
        if len(exact_coordinates) != len(points):
            derived_plan = _build_derived_point_plan(
                graph,
                objects=objects,
                points=points,
                exact_coordinates=exact_coordinates,
                segments=segments,
                polygons=polygons,
                names=names,
                relations=relation_items,
            )
            if derived_plan is not None:
                return derived_plan
            return _unsupported(graph, "mixed_exact_and_missing_coordinates")
        trace = _trace_from_geometry(
            coordinates={names[item_id]: coordinate for item_id, coordinate in exact_coordinates.items()},
            segments=segments,
            polygons=polygons,
            relations=relation_items,
            targets=graph.targets,
            names=names,
            source_id=graph.id,
        )
        diagnostics = _validate_trace_coordinates(exact_coordinates)
        if diagnostics:
            return _unsupported(graph, diagnostics[0])
        return ConstraintCompletionPlan(
            source_id=graph.source_id,
            parse_graph_checksum=parse_checksum,
            status=CompletionStatus.EXACT,
            trace=tuple(trace),
            source_candidate_ids=tuple(
                item.id for item in (*graph.objects, *graph.relations, *graph.quantities, *graph.regions, *graph.targets)
            ),
            metadata={"method": "explicit_exact_coords", "coordinate_source": "parse_graph"},
        )

    numeric_lengths = _numeric_lengths(graph, segments, names)
    constraints = _canonical_constraints(graph, segments, names, numeric_lengths)
    if not numeric_lengths:
        return _unsupported(graph, "no_explicit_numeric_length_constraints")
    if not constraints:
        return _unsupported(graph, "no_supported_numeric_constraints")
    if any(item.kind not in {ParseObjectKind.POINT, ParseObjectKind.SEGMENT, ParseObjectKind.LINE,
                             ParseObjectKind.POLYGON} for item in objects):
        return _unsupported(graph, "unsupported_semantic_object_kind")
    if any(len(edge) != 2 for edge in segments.values()):
        return _unsupported(graph, "segment_endpoints_not_resolved")

    rectangle_plan = _build_rectangle_plan(
        graph,
        points=points,
        segments=segments,
        polygons=polygons,
        names=names,
        numeric_lengths=numeric_lengths,
        relations=relation_items,
    )
    if rectangle_plan is not None:
        return rectangle_plan

    perpendicular_chain_plan = _build_perpendicular_chain_plan(
        graph,
        points=points,
        segments=segments,
        polygons=polygons,
        names=names,
        numeric_lengths=numeric_lengths,
        relations=relation_items,
    )
    if perpendicular_chain_plan is not None:
        return perpendicular_chain_plan

    if any(line_id(*edge) not in numeric_lengths for edge in segments.values()):
        return _unsupported(graph, "every_segment_requires_explicit_length")

    sss_plan = _build_sss_plan(
        graph,
        points=points,
        segments=segments,
        polygons=polygons,
        names=names,
        numeric_lengths=numeric_lengths,
    )
    if sss_plan is not None:
        return sss_plan

    return _unsupported(graph, "no_registered_exact_completion_strategy")


def _build_derived_point_plan(
    graph: ParseGraph,
    *,
    objects: tuple[ParseObject, ...],
    points: dict[str, ParseObject],
    exact_coordinates: dict[str, tuple[sp.Expr, sp.Expr]],
    segments: dict[str, tuple[str, str]],
    polygons: dict[str, tuple[str, ...]],
    names: dict[str, str],
    relations: tuple[tuple[Any, tuple[str | None, ...]], ...],
) -> ConstraintCompletionPlan | None:
    """Complete named derived points from an exact geometric scaffold.

    This path deliberately accepts only deterministic construction relations.
    It does not infer a missing point from pixels, a theorem, or an arbitrary
    distance constraint.
    """

    if any(item.kind not in {
        ParseObjectKind.POINT,
        ParseObjectKind.LINE,
        ParseObjectKind.SEGMENT,
        ParseObjectKind.POLYGON,
    } for item in objects):
        return None

    point_plans = _derived_point_relations(graph, points=points, names=names)
    missing_points = set(points) - set(exact_coordinates)
    if not missing_points or set(point_plans) != missing_points:
        return None
    if any(len(point_plans[point_id]) != 1 for point_id in missing_points):
        return None

    object_by_id = {item.id: item for item in objects}
    trace: list[dict[str, Any]] = []
    materialized = {names[point_id] for point_id in exact_coordinates}
    for point_id, coordinate in sorted(exact_coordinates.items(), key=lambda item: names[item[0]]):
        trace.append({
            "op": "add_point",
            "id": names[point_id],
            "coord": [str(sp.simplify(coordinate[0])), str(sp.simplify(coordinate[1]))],
            "metadata": {
                "completion_source": graph.id,
                "source_parse_object_id": point_id,
                "completion_role": "explicit_coordinate",
            },
        })

    emitted_linear: set[str] = set()
    support_ids: dict[str, str] = {}
    pending_points = set(missing_points)
    while pending_points:
        progress = False
        for object_id, edge in segments.items():
            start, end = edge
            output_id = _linear_output_id(object_by_id[object_id], edge, names)
            if output_id in emitted_linear or start not in materialized or end not in materialized:
                continue
            item = object_by_id[object_id]
            metadata = {
                "completion_source": graph.id,
                "source_parse_object_id": object_id,
                "completion_role": "support",
            }
            if item.kind is ParseObjectKind.LINE:
                trace.append({"op": "add_line", "id": output_id, "points": [start, end], "metadata": metadata})
            else:
                trace.append({"op": "add_segment", "id": output_id, "points": [start, end], "metadata": metadata})
            emitted_linear.add(output_id)
            support_ids[object_id] = output_id
            progress = True

        for point_id in sorted(tuple(pending_points), key=lambda item: names[item]):
            relation, operation, dependencies, params = point_plans[point_id][0]
            dependency_names = tuple(names.get(item) for item in dependencies)
            if any(item is None for item in dependency_names):
                return None
            if operation == "midpoint":
                ready = all(item in materialized for item in dependency_names)
                operation_params = {"of": list(dependency_names)}
            elif operation == "foot_of_perpendicular":
                source_name, _ = dependency_names
                support_id = dependencies[1]
                ready = source_name in materialized and support_id in support_ids
                operation_params = {"point": source_name, "line": support_ids.get(support_id, "")}
            elif operation == "point_on_segment_ratio":
                support_id = dependencies[0]
                ready = support_id in support_ids
                operation_params = {"segment": support_ids.get(support_id, ""), "ratio": params["ratio"]}
            else:
                return None
            if not ready or not operation_params.get("line", operation_params.get("segment", True)):
                continue
            metadata = {
                "completion_source": graph.id,
                "source_parse_relation_id": relation.id,
                "completion_role": "derived_point",
            }
            operation_params["metadata"] = metadata
            trace.append({"op": operation, "id": names[point_id], **operation_params})
            materialized.add(names[point_id])
            pending_points.remove(point_id)
            progress = True
        if not progress:
            return None

    for object_id, edge in segments.items():
        if object_id in support_ids:
            continue
        start, end = edge
        if start not in materialized or end not in materialized:
            return None
        item = object_by_id[object_id]
        output_id = _linear_output_id(item, edge, names)
        if item.kind is ParseObjectKind.LINE:
            trace.append({"op": "add_line", "id": output_id, "points": [start, end], "metadata": {
                "completion_source": graph.id, "source_parse_object_id": object_id,
            }})
        else:
            trace.append({"op": "add_segment", "id": output_id, "points": [start, end], "metadata": {
                "completion_source": graph.id, "source_parse_object_id": object_id,
            }})
        support_ids[object_id] = output_id

    for object_id, vertices in polygons.items():
        if not all(vertex in materialized for vertex in vertices):
            return None
        trace.append({"op": "add_polygon", "id": names.get(object_id, normalize_id(object_id)), "vertices": list(vertices),
                      "metadata": {"completion_source": graph.id, "source_parse_object_id": object_id}})
    for relation, _ in relations:
        if relation.predicate not in {"parallel", "perpendicular"} or len(relation.arguments) != 2:
            continue
        line_names = tuple(_linear_reference_output(argument.ref, object_by_id, segments, names, support_ids)
                           for argument in relation.arguments)
        if all(line_names):
            trace.append({"op": "add_relation", "id": f"relation_{normalize_id(relation.id)}", "type": relation.predicate,
                          "objects": list(line_names), "metadata": {"completion_source": graph.id,
                          "source_parse_relation_id": relation.id}})
    _append_value_targets(trace, graph.targets, names, materialized, set(support_ids.values()), set(polygons))

    try:
        state = ConstructionBuilder().execute(trace)
    except Exception:
        return None
    if any(point_id not in state.points for point_id in materialized):
        return None
    source_ids = tuple(item.id for item in (*graph.objects, *graph.relations, *graph.quantities, *graph.regions, *graph.targets))
    return ConstraintCompletionPlan(
        source_id=graph.source_id,
        parse_graph_checksum=graph.checksum(),
        status=CompletionStatus.DETERMINISTIC_COMPLETION,
        trace=tuple(trace),
        source_candidate_ids=source_ids,
        metadata={"method": "derived_construction_v1", "coordinate_source": "explicit_coordinates_and_constraints",
                  "derived_operations": [trace_item["op"] for trace_item in trace if trace_item["op"] in {
                      "midpoint", "foot_of_perpendicular", "point_on_segment_ratio"}]},
    )


def _derived_point_relations(
    graph: ParseGraph,
    *,
    points: dict[str, ParseObject],
    names: dict[str, str],
) -> dict[str, list[tuple[Any, str, tuple[str, ...], dict[str, Any]]]]:
    result: dict[str, list[tuple[Any, str, tuple[str, ...], dict[str, Any]]]] = {}
    for relation in graph.relations:
        by_role: dict[str, list[str]] = {}
        for argument in relation.arguments:
            by_role.setdefault(argument.role, []).append(argument.ref)
        point_refs = by_role.get("point", [])
        if len(point_refs) != 1 or point_refs[0] not in points:
            continue
        output_id = point_refs[0]
        if relation.predicate == "midpoint":
            dependencies = tuple(by_role.get("of", []))
            if len(dependencies) != 2:
                dependencies = tuple(by_role.get(f"of_{index}", [""])[0] for index in range(2))
            if len(dependencies) == 2 and all(item in points for item in dependencies):
                result.setdefault(output_id, []).append((relation, "midpoint", dependencies, {}))
        elif relation.predicate == "foot_of_perpendicular":
            source = _first_role(by_role, "from", "source")
            support = _first_role(by_role, "line", "support")
            if source in points and support:
                result.setdefault(output_id, []).append((relation, "foot_of_perpendicular", (source, support), {}))
        elif relation.predicate == "point_on_segment_ratio":
            support = _first_role(by_role, "segment", "support")
            ratio = _numeric_ratio(relation.metadata.get("ratio"))
            if support and ratio is not None:
                result.setdefault(output_id, []).append((relation, "point_on_segment_ratio", (support,), {"ratio": str(ratio)}))
    return result


def _first_role(by_role: dict[str, list[str]], *roles: str) -> str | None:
    for role in roles:
        values = by_role.get(role, [])
        if len(values) == 1:
            return values[0]
    return None


def _numeric_ratio(value: Any) -> sp.Expr | None:
    try:
        expression = (
            MathExpression.from_dict(value).to_sympy()
            if isinstance(value, dict)
            else math_expression_from_text(str(value)).to_sympy()
        )
    except (InvalidGeometryInput, TypeError, ValueError, SympifyError):
        return None
    if expression.is_number is not True or expression.is_real is not True:
        return None
    if expression.is_positive is not True or (sp.Integer(1) - expression).is_positive is not True:
        return None
    return sp.simplify(expression)


def _linear_output_id(item: ParseObject, edge: tuple[str, str], names: dict[str, str]) -> str:
    if item.kind is ParseObjectKind.LINE:
        return names[item.id]
    return line_id(*edge)


def _linear_reference_output(
    object_id: str,
    object_by_id: dict[str, ParseObject],
    segments: dict[str, tuple[str, str]],
    names: dict[str, str],
    support_ids: dict[str, str],
) -> str | None:
    if object_id in support_ids:
        return support_ids[object_id]
    item = object_by_id.get(object_id)
    edge = segments.get(object_id)
    if item is None or edge is None:
        return None
    return _linear_output_id(item, edge, names)


def _append_value_targets(
    trace: list[dict[str, Any]],
    targets: tuple[Any, ...],
    names: dict[str, str],
    point_ids: set[str],
    linear_ids: set[str],
    polygon_ids: set[str],
) -> None:
    known_objects = point_ids | linear_ids | {names.get(item, normalize_id(item)) for item in polygon_ids}
    for target in targets:
        if target.mode != "value" or not target.arguments:
            continue
        target_name = names.get(target.arguments[0].ref, normalize_id(target.arguments[0].ref))
        if target_name in known_objects:
            trace.append({"op": "set_target", "type": target.kind, "object": target_name,
                          "metadata": {"source_parse_target_id": target.id}})


def _semantic_objects(objects: tuple[ParseObject, ...]) -> tuple[ParseObject, ...]:
    return tuple(
        item for item in objects
        if item.origin.value != "image" and not item.id.startswith("visual:")
    )


def _unique_names(objects: tuple[ParseObject, ...]) -> dict[str, str]:
    names: dict[str, str] = {}
    used: set[str] = set()
    for index, item in enumerate(objects):
        candidate = normalize_id(item.visible_name or item.id.split(":")[-1])
        if not candidate:
            candidate = f"object_{index}"
        if candidate in used:
            candidate = f"{candidate}_{index}"
        used.add(candidate)
        names[item.id] = candidate
    return names


def _canonical_point_aliases(point_names: tuple[str, ...]) -> dict[str, str] | None:
    unique_names = tuple(dict.fromkeys(point_names))
    if len(unique_names) > 26:
        return None
    if all(len(name) == 1 for name in unique_names):
        return {name: name for name in unique_names}
    return {name: chr(ord("A") + index) for index, name in enumerate(sorted(unique_names))}


def _build_sss_plan(
    graph: ParseGraph,
    *,
    points: dict[str, ParseObject],
    segments: dict[str, tuple[str, str]],
    polygons: dict[str, tuple[str, ...]],
    names: dict[str, str],
    numeric_lengths: dict[str, sp.Expr],
) -> ConstraintCompletionPlan | None:
    if len(points) != 3 or len(segments) != 3:
        return None
    point_names = tuple(names[item_id] for item_id in points)
    edge_by_pair = {frozenset(edge): edge for edge in segments.values()}
    if len(edge_by_pair) != 3 or set(edge_by_pair) != {frozenset(pair) for pair in combinations(point_names, 2)}:
        return None
    if any(item.kind not in {ParseObjectKind.POINT, ParseObjectKind.SEGMENT, ParseObjectKind.POLYGON} for item in _semantic_objects(graph.objects)):
        return None

    edges = sorted(edge_by_pair.values(), key=lambda edge: line_id(*edge))
    base_start, base_end = edges[0]
    apex = next(name for name in point_names if name not in {base_start, base_end})
    base_length = _length_for_edge((base_start, base_end), segments, numeric_lengths)
    start_length = _length_for_edge((apex, base_start), segments, numeric_lengths)
    end_length = _length_for_edge((apex, base_end), segments, numeric_lengths)
    projection = sp.simplify((start_length**2 - end_length**2 + base_length**2) / (2 * base_length))
    height_squared = sp.simplify(start_length**2 - projection**2)
    if height_squared.is_positive is not True:
        reason = "sss_triangle_degenerate_or_impossible" if height_squared.is_nonpositive is True else "sss_triangle_numeric_unstable"
        return _unsupported(graph, reason)
    height = sp.sqrt(height_squared)
    coordinates_by_branch = (
        {base_start: (sp.Integer(0), sp.Integer(0)), base_end: (base_length, sp.Integer(0)), apex: (projection, height)},
        {base_start: (sp.Integer(0), sp.Integer(0)), base_end: (base_length, sp.Integer(0)), apex: (projection, -height)},
    )
    branches = tuple(
        tuple(_trace_from_geometry(
            coordinates=coordinates,
            segments=segments,
            polygons=polygons,
            relations=(),
            targets=graph.targets,
            names=names,
            source_id=graph.id,
            length_constraints=numeric_lengths,
            branch_label=branch_label,
        ))
        for branch_label, coordinates in zip(("first", "second"), coordinates_by_branch)
    )
    source_ids = tuple(item.id for item in (*graph.objects, *graph.relations, *graph.quantities, *graph.regions, *graph.targets))
    return ConstraintCompletionPlan(
        source_id=graph.source_id,
        parse_graph_checksum=graph.checksum(),
        status=CompletionStatus.BRANCHING_COMPLETION,
        trace=branches[0],
        branches=branches,
        source_candidate_ids=source_ids,
        metadata={
            "method": "sss_triangle_v1",
            "coordinate_source": "explicit_numeric_side_lengths",
            "realization_branches": ["first", "second"],
            "gauge": {"base": [base_start, base_end], "apex": apex},
        },
    )


def _build_perpendicular_chain_plan(
    graph: ParseGraph,
    *,
    points: dict[str, ParseObject],
    segments: dict[str, tuple[str, str]],
    polygons: dict[str, tuple[str, ...]],
    names: dict[str, str],
    numeric_lengths: dict[str, sp.Expr],
    relations: tuple[tuple[Any, tuple[str | None, ...]], ...],
) -> ConstraintCompletionPlan | None:
    if len(points) != 3 or len(segments) != 2:
        return None
    perpendicular = next(
        (
            relation
            for relation in graph.relations
            if relation.predicate == "perpendicular"
            and len(relation.arguments) == 2
            and all(argument.ref in segments for argument in relation.arguments)
        ),
        None,
    )
    if perpendicular is None:
        return None
    first = segments[perpendicular.arguments[0].ref]
    second = segments[perpendicular.arguments[1].ref]
    shared = set(first) & set(second)
    if len(shared) != 1:
        return None
    first_length = _optional_length_for_edge(first, segments, numeric_lengths)
    second_length = _optional_length_for_edge(second, segments, numeric_lengths)
    if first_length is None or second_length is None:
        return None
    common = next(iter(shared))
    first_other = next(point for point in first if point != common)
    second_other = next(point for point in second if point != common)
    coordinates = {
        first_other: (sp.Integer(0), sp.Integer(0)),
        common: (first_length, sp.Integer(0)),
        second_other: (first_length, second_length),
    }
    if set(coordinates) != {names[point_id] for point_id in points}:
        return None
    trace = _trace_from_geometry(
        coordinates=coordinates,
        segments=segments,
        polygons=polygons,
        relations=relations,
        targets=graph.targets,
        names=names,
        source_id=graph.id,
        length_constraints=numeric_lengths,
    )
    source_ids = tuple(
        item.id for item in (*graph.objects, *graph.relations, *graph.quantities, *graph.regions, *graph.targets)
    )
    return ConstraintCompletionPlan(
        source_id=graph.source_id,
        parse_graph_checksum=graph.checksum(),
        status=CompletionStatus.DETERMINISTIC_COMPLETION,
        trace=tuple(trace),
        source_candidate_ids=source_ids,
        metadata={
            "method": "perpendicular_two_length_chain_v1",
            "coordinate_source": "explicit_numeric_and_structural_constraints",
            "source_coordinate_recovery": False,
            "gauge": {"base": [first_other, common], "positive_side": second_other},
        },
    )


def _build_rectangle_plan(
    graph: ParseGraph,
    *,
    points: dict[str, ParseObject],
    segments: dict[str, tuple[str, str]],
    polygons: dict[str, tuple[str, ...]],
    names: dict[str, str],
    numeric_lengths: dict[str, sp.Expr],
    relations: tuple[tuple[Any, tuple[str | None, ...]], ...],
) -> ConstraintCompletionPlan | None:
    rectangle = next(
        (
            relation for relation in graph.relations
            if relation.predicate == "rectangle"
            and len(relation.arguments) == 1
            and relation.arguments[0].ref in polygons
            and len(polygons[relation.arguments[0].ref]) == 4
        ),
        None,
    )
    if rectangle is None:
        return None

    polygon_id = rectangle.arguments[0].ref
    vertices = polygons[polygon_id]
    start_index = next(
        (
            index for index in range(4)
            if _optional_length_for_edge((vertices[index], vertices[(index + 1) % 4]), segments, numeric_lengths) is not None
            and _optional_length_for_edge((vertices[(index + 1) % 4], vertices[(index + 2) % 4]), segments, numeric_lengths) is not None
        ),
        None,
    )
    if start_index is None:
        return None
    ordered = tuple(vertices[(start_index + offset) % 4] for offset in range(4))
    first_length = _optional_length_for_edge((ordered[0], ordered[1]), segments, numeric_lengths)
    second_length = _optional_length_for_edge((ordered[1], ordered[2]), segments, numeric_lengths)
    if first_length is None or second_length is None:
        return None
    coordinates: dict[str, tuple[sp.Expr, sp.Expr]] = {
        ordered[0]: (sp.Integer(0), sp.Integer(0)),
        ordered[1]: (sp.Integer(0), first_length),
        ordered[2]: (second_length, first_length),
        ordered[3]: (second_length, sp.Integer(0)),
    }

    coordinates = _complete_perpendicular_length_points(
        graph,
        coordinates=coordinates,
        segments=segments,
        names=names,
        numeric_lengths=numeric_lengths,
    )
    coordinates = _complete_collinear_intersections(
        graph,
        coordinates=coordinates,
        names=names,
    )
    point_names = {names[point_id] for point_id in points}
    if set(coordinates) != point_names:
        return None

    trace = _trace_from_geometry(
        coordinates=coordinates,
        segments=segments,
        polygons=polygons,
        relations=relations,
        targets=graph.targets,
        names=names,
        source_id=graph.id,
        length_constraints=numeric_lengths,
    )
    trace.extend(_semantic_relation_trace(graph, names, predicates={"rectangle", "collinear"}))
    try:
        ConstructionBuilder().execute(trace)
    except Exception:
        return None
    source_ids = tuple(
        item.id for item in (*graph.objects, *graph.relations, *graph.quantities, *graph.regions, *graph.targets)
    )
    return ConstraintCompletionPlan(
        source_id=graph.source_id,
        parse_graph_checksum=graph.checksum(),
        status=CompletionStatus.DETERMINISTIC_COMPLETION,
        trace=tuple(trace),
        source_candidate_ids=source_ids,
        metadata={
            "method": "exact_rectangle_constraints_v1",
            "coordinate_source": "explicit_numeric_and_structural_constraints",
            "source_coordinate_recovery": False,
            "gauge": {"rectangle": list(ordered)},
        },
    )


def _complete_perpendicular_length_points(
    graph: ParseGraph,
    *,
    coordinates: dict[str, tuple[sp.Expr, sp.Expr]],
    segments: dict[str, tuple[str, str]],
    names: dict[str, str],
    numeric_lengths: dict[str, sp.Expr],
) -> dict[str, tuple[sp.Expr, sp.Expr]]:
    output = dict(coordinates)
    scaffold = {
        names[item.id]: (float(value[0]), 1.0 - float(value[1]))
        for item in graph.objects
        if item.id in names
        and isinstance((value := item.metadata.get("seed_normalized")), (list, tuple))
        and len(value) == 2
    }
    for relation in graph.relations:
        if relation.predicate != "perpendicular" or len(relation.arguments) != 2:
            continue
        edge_ids = [argument.ref for argument in relation.arguments]
        for support_id, unknown_id in (edge_ids, edge_ids[::-1]):
            support = segments.get(support_id)
            unknown = segments.get(unknown_id)
            if support is None or unknown is None or not all(point in output for point in support):
                continue
            missing = [point for point in unknown if point not in output]
            placed = [point for point in unknown if point in output]
            length = numeric_lengths.get(line_id(*unknown))
            if len(missing) != 1 or len(placed) != 1 or length is None:
                continue
            support_vector = (
                sp.simplify(output[support[1]][0] - output[support[0]][0]),
                sp.simplify(output[support[1]][1] - output[support[0]][1]),
            )
            norm = sp.sqrt(sp.simplify(support_vector[0] ** 2 + support_vector[1] ** 2))
            if norm.is_zero is True:
                continue
            positive = (-support_vector[1] * length / norm, support_vector[0] * length / norm)
            sign = _scaffold_perpendicular_sign(support, (placed[0], missing[0]), scaffold)
            vector = positive if sign >= 0 else (-positive[0], -positive[1])
            origin = output[placed[0]]
            output[missing[0]] = (
                sp.simplify(origin[0] + vector[0]),
                sp.simplify(origin[1] + vector[1]),
            )
    return output


def _scaffold_perpendicular_sign(
    support: tuple[str, str],
    extension: tuple[str, str],
    scaffold: dict[str, tuple[float, float]],
) -> int:
    if not all(point in scaffold for point in (*support, *extension)):
        return -1
    support_vector = (
        scaffold[support[1]][0] - scaffold[support[0]][0],
        scaffold[support[1]][1] - scaffold[support[0]][1],
    )
    extension_vector = (
        scaffold[extension[1]][0] - scaffold[extension[0]][0],
        scaffold[extension[1]][1] - scaffold[extension[0]][1],
    )
    cross = support_vector[0] * extension_vector[1] - support_vector[1] * extension_vector[0]
    return 1 if cross >= 0 else -1


def _complete_collinear_intersections(
    graph: ParseGraph,
    *,
    coordinates: dict[str, tuple[sp.Expr, sp.Expr]],
    names: dict[str, str],
) -> dict[str, tuple[sp.Expr, sp.Expr]]:
    output = dict(coordinates)
    line_supports = {
        item.id: [names.get(reference.ref) for reference in item.references if reference.role in {"start", "end", "through"}]
        for item in graph.objects if item.kind in {ParseObjectKind.SEGMENT, ParseObjectKind.LINE, ParseObjectKind.RAY}
    }
    for point_name in sorted(set(names.values()) - set(output)):
        supports: list[tuple[tuple[sp.Expr, sp.Expr], tuple[sp.Expr, sp.Expr]]] = []
        for relation in graph.relations:
            if relation.predicate == "collinear":
                relation_names = [names.get(argument.ref) for argument in relation.arguments]
            elif relation.predicate in {"point_on_line", "point_on_segment", "point_on_ray", "incidence"}:
                roles = {argument.role: argument.ref for argument in relation.arguments}
                relation_names = [names.get(roles.get("point")), *line_supports.get(roles.get("support"), [])]
            elif relation.predicate == "ordered_points":
                relation_names = [names.get(argument.ref) for argument in relation.arguments if argument.role != "support"]
            else:
                continue
            if point_name not in relation_names:
                continue
            known = [name for name in relation_names if name != point_name and name in output]
            if len(known) >= 2:
                supports.append((output[known[0]], output[known[1]]))
        if len(supports) < 2:
            continue
        intersection = _exact_line_intersection(supports[0], supports[1])
        if intersection is not None:
            output[point_name] = intersection
    return output


def _exact_line_intersection(
    left: tuple[tuple[sp.Expr, sp.Expr], tuple[sp.Expr, sp.Expr]],
    right: tuple[tuple[sp.Expr, sp.Expr], tuple[sp.Expr, sp.Expr]],
) -> tuple[sp.Expr, sp.Expr] | None:
    start, end = left
    other_start, other_end = right
    left_vector = (end[0] - start[0], end[1] - start[1])
    right_vector = (other_end[0] - other_start[0], other_end[1] - other_start[1])
    denominator = sp.simplify(left_vector[0] * right_vector[1] - left_vector[1] * right_vector[0])
    if denominator.is_zero is True:
        return None
    offset = (other_start[0] - start[0], other_start[1] - start[1])
    parameter = sp.simplify((offset[0] * right_vector[1] - offset[1] * right_vector[0]) / denominator)
    return (
        sp.simplify(start[0] + parameter * left_vector[0]),
        sp.simplify(start[1] + parameter * left_vector[1]),
    )


def _semantic_relation_trace(
    graph: ParseGraph,
    names: dict[str, str],
    *,
    predicates: set[str],
) -> list[dict[str, Any]]:
    return [
        {
            "op": "add_relation",
            "id": f"relation_{normalize_id(relation.id)}",
            "type": relation.predicate,
            "objects": [names.get(argument.ref, normalize_id(argument.ref)) for argument in relation.arguments],
            "metadata": {"completion_source": graph.id, "source_parse_relation_id": relation.id},
        }
        for relation in graph.relations
        if relation.predicate in predicates
    ]


def _length_for_edge(
    wanted: tuple[str, str],
    segments: dict[str, tuple[str, str]],
    numeric_lengths: dict[str, sp.Expr],
) -> sp.Expr:
    wanted_pair = frozenset(wanted)
    for edge in segments.values():
        if frozenset(edge) == wanted_pair:
            value = numeric_lengths.get(line_id(*edge))
            if value is not None:
                return value
    raise InvalidGeometryInput(f"Missing numeric length for edge {wanted[0]}-{wanted[1]}")


def _optional_length_for_edge(
    wanted: tuple[str, str],
    segments: dict[str, tuple[str, str]],
    numeric_lengths: dict[str, sp.Expr],
) -> sp.Expr | None:
    wanted_pair = frozenset(wanted)
    for edge in segments.values():
        if frozenset(edge) == wanted_pair:
            value = numeric_lengths.get(line_id(*edge))
            if value is not None:
                return value
    return None


def _point_ref(item: ParseObject, names: dict[str, str], role: str) -> str | None:
    for reference in item.references:
        if reference.role == role and reference.ref in names:
            return names[reference.ref]
    return None


def _segment_topology(objects: tuple[ParseObject, ...], names: dict[str, str]) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for item in objects:
        if item.kind not in {ParseObjectKind.SEGMENT, ParseObjectKind.LINE}:
            continue
        start = _point_ref(item, names, "start")
        end = _point_ref(item, names, "end")
        if start is None or end is None:
            points = [names[reference.ref] for reference in item.references if reference.ref in names]
            if len(points) == 2:
                start, end = points
        if start and end and start != end:
            result[item.id] = (start, end)
    return result


def _polygon_topology(objects: tuple[ParseObject, ...], names: dict[str, str]) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for item in objects:
        if item.kind is not ParseObjectKind.POLYGON:
            continue
        vertices = tuple(names[reference.ref] for reference in item.references if reference.ref in names)
        if len(vertices) >= 3 and len(set(vertices)) == len(vertices):
            result[item.id] = vertices
    return result


def _exact_coordinates(points: dict[str, ParseObject]) -> dict[str, tuple[sp.Expr, sp.Expr]]:
    coordinates: dict[str, tuple[sp.Expr, sp.Expr]] = {}
    for object_id, item in points.items():
        payload = item.geometry.get("exact_coords")
        if payload is None:
            continue
        if not isinstance(payload, (list, tuple)) or len(payload) != 2:
            return {}
        values = tuple(_expression_value(value) for value in payload)
        if any(value is None or value.free_symbols for value in values):
            return {}
        coordinates[object_id] = values  # type: ignore[assignment]
    return coordinates


def _expression_value(value: Any) -> sp.Expr | None:
    if not isinstance(value, dict):
        return None
    try:
        expression = MathExpression.from_dict(value).to_sympy()
    except (InvalidGeometryInput, TypeError, ValueError, SympifyError):
        return None
    if expression.free_symbols or expression.is_real is False or expression.is_finite is False:
        return None
    return sp.simplify(expression)


def _numeric_lengths(
    graph: ParseGraph,
    segments: dict[str, tuple[str, str]],
    names: dict[str, str],
) -> dict[str, sp.Expr]:
    by_object = {
        item.id: line_id(*segments[item.id])
        for item in graph.objects
        if item.id in segments
    }
    output: dict[str, sp.Expr] = {}
    for quantity in graph.quantities:
        if quantity.kind != "length" or not quantity.arguments:
            continue
        segment_name = by_object.get(quantity.arguments[0].ref)
        value = quantity.expression.to_sympy()
        if segment_name and value.is_number and value.is_real is not False and value.is_positive is True:
            output[segment_name] = sp.simplify(value)
    return output


def graph_object(graph: ParseGraph, object_id: str) -> ParseObject | None:
    return next((item for item in graph.objects if item.id == object_id), None)


def _canonical_constraints(
    graph: ParseGraph,
    segments: dict[str, tuple[str, str]],
    names: dict[str, str],
    numeric_lengths: dict[str, sp.Expr],
    *,
    canonical_aliases: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    aliases = canonical_aliases or {name: name for name in names.values()}
    constraints = [
        {
            "type": "length",
            "object": line_id(aliases[edge[0]], aliases[edge[1]]),
            "value": str(value),
        }
        for edge in segments.values()
        if (segment_name := line_id(*edge)) in numeric_lengths
        for value in [numeric_lengths[segment_name]]
    ]
    for relation in graph.relations:
        if relation.predicate not in {"parallel", "perpendicular"} or len(relation.arguments) != 2:
            continue
        line_names = [
            _relation_line_name(argument.ref, segments, names, canonical_aliases=aliases)
            for argument in relation.arguments
        ]
        if all(line_names):
            constraints.append({"type": relation.predicate, "lines": line_names})
    return constraints


def _relation_line_name(
    reference: str,
    segments: dict[str, tuple[str, str]],
    names: dict[str, str],
    *,
    canonical_aliases: dict[str, str] | None = None,
) -> str | None:
    edge = segments.get(reference)
    if edge is None:
        return None
    aliases = canonical_aliases or {name: name for name in names.values()}
    return line_id(aliases[edge[0]], aliases[edge[1]])


def _trace_from_geometry(
    *,
    coordinates: dict[str, tuple[sp.Expr, sp.Expr]],
    segments: dict[str, tuple[str, str]],
    polygons: dict[str, tuple[str, ...]],
    relations: tuple[tuple[Any, tuple[str | None, ...]], ...],
    targets: tuple[Any, ...],
    names: dict[str, str],
    source_id: str,
    length_constraints: dict[str, sp.Expr] | None = None,
    branch_label: str | None = None,
) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    reverse_names = {value: key for key, value in names.items()}
    for point_id, coordinate in sorted(coordinates.items()):
        source_point_id = reverse_names.get(point_id, point_id)
        metadata = {"completion_source": source_id, "source_parse_object_id": source_point_id}
        if branch_label is not None:
            metadata["branch_selection"] = branch_label
        trace.append({
            "op": "add_point",
            "id": point_id,
            "coord": [str(sp.simplify(coordinate[0])), str(sp.simplify(coordinate[1]))],
            "metadata": metadata,
        })
    emitted_segments: set[str] = set()
    for object_id, edge in segments.items():
        segment_id = line_id(*edge)
        if segment_id in emitted_segments:
            continue
        emitted_segments.add(segment_id)
        metadata: dict[str, Any] = {
            "completion_source": source_id,
            "source_parse_object_id": object_id,
        }
        if length_constraints and segment_id in length_constraints:
            metadata["source_length"] = str(length_constraints[segment_id])
            metadata["show_length"] = True
            metadata["display_value"] = str(length_constraints[segment_id])
        trace.append({"op": "add_segment", "id": segment_id, "points": list(edge), "metadata": metadata})
    for object_id, vertices in polygons.items():
        polygon_id = names.get(object_id, normalize_id(object_id))
        trace.append({
            "op": "add_polygon",
            "id": polygon_id,
            "vertices": list(vertices),
            "metadata": {"completion_source": source_id, "source_parse_object_id": object_id},
        })
    for relation, line_names in relations:
        if all(line_names):
            supporting_text = str(relation.metadata.get("supporting_text") or "").lower()
            show_marker = relation.predicate == "perpendicular" or (
                relation.predicate == "parallel"
                and ("arrow" in supporting_text or "mark" in supporting_text)
            )
            trace.append({
                "op": "add_relation",
                "id": f"relation_{normalize_id(relation.id)}",
                "type": relation.predicate,
                "objects": list(line_names),
                "metadata": {
                    "completion_source": source_id,
                    "source_parse_relation_id": relation.id,
                    "show_marker": show_marker,
                },
            })
    for target in targets:
        if target.mode != "value" or not target.arguments:
            continue
        object_id = target.arguments[0].ref
        target_name = names.get(object_id, normalize_id(object_id))
        known_objects = set(coordinates) | {line_id(*edge) for edge in segments.values()} | {
            names.get(item, normalize_id(item)) for item in polygons
        }
        if target_name in known_objects:
            trace.append({
                "op": "set_target",
                "type": target.kind,
                "object": target_name,
                "metadata": {"completion_source": source_id, "source_parse_target_id": target.id},
            })
    return trace


def _validate_trace_coordinates(coordinates: dict[str, tuple[sp.Expr, sp.Expr]]) -> list[str]:
    if any(value.free_symbols or value.is_real is False or value.is_finite is False for point in coordinates.values() for value in point):
        return ["completion_coordinates_not_finite_numeric"]
    values = list(coordinates.values())
    if len(values) >= 3:
        anchor = values[0]
        if all(sp.simplify((point[0] - anchor[0]) * (other[1] - anchor[1]) - (point[1] - anchor[1]) * (other[0] - anchor[0])) == 0
               for index, point in enumerate(values[1:], start=1) for other in values[index + 1:]):
            return ["completion_coordinates_degenerate_collinear"]
    return []


def _unsupported(graph: ParseGraph, reason: str) -> ConstraintCompletionPlan:
    return ConstraintCompletionPlan(
        source_id=graph.source_id,
        parse_graph_checksum=graph.checksum(),
        status=CompletionStatus.UNSUPPORTED,
        diagnostics=(reason,),
        metadata={"method": "none", "coordinate_source": "none"},
    )


__all__ = [
    "COMPLETION_PLAN_SCHEMA_VERSION",
    "CompletionStatus",
    "ConstraintCompletionPlan",
    "build_constraint_completion_plan",
]
