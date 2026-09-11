"""Structural analysis of Parser v1 clauses before geometric solving."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from geosym_world.errors import InvalidGeometryInput
from geosym_world.schema import ParseGraph
from geosym_world.schema.ids import validate_geometry_id

from .contracts import ClauseGraph, ConstraintState, ObservationGraph, Realization, RealizationStatus
from .compiler import compile_parse_graph


def analyze_constraints(
    clause_graph: ClauseGraph,
    *,
    free_parameters: Sequence[str] = (),
    realizations: Sequence[Realization] = (),
) -> ConstraintState:
    """Classify only facts decidable from the current clauses and witnesses.

    This deliberately does not infer geometric entailment. A later solver may
    refine the returned state, but it must start from the recorded graph.
    """

    if not isinstance(clause_graph, ClauseGraph):
        raise InvalidGeometryInput("Constraint analysis requires a ClauseGraph")
    parameters = tuple(free_parameters)
    if any(not isinstance(item, str) or not item.strip() for item in parameters):
        raise InvalidGeometryInput("Free parameter names must be non-empty strings")
    for parameter in parameters:
        validate_geometry_id(parameter)
    branches = tuple(realizations)
    if any(not isinstance(item, Realization) for item in branches):
        raise InvalidGeometryInput("Constraint realizations must be typed Realization values")

    status = _classify(clause_graph.clauses, parameters, branches)
    derived_ids = tuple(item.id for item in clause_graph.clauses if item.status == "derived")
    metadata = {"analyzer": "structural_v1"}
    if status is RealizationStatus.AMBIGUOUS and not branches:
        metadata["ambiguity_reason"] = "unresolved_clause_conflict"
    return ConstraintState(
        id=f"constraints_{clause_graph.id}",
        clause_graph_checksum=clause_graph.checksum(),
        status=status,
        free_parameters=parameters,
        realizations=branches,
        derived_clause_ids=derived_ids,
        metadata=metadata,
    )


def realize_exact_points(clause_graph: ClauseGraph) -> ConstraintState:
    """Create a witness only when every point object has exact coordinates."""

    point_clauses = [item for item in clause_graph.clauses if item.predicate == "object.point"]
    coordinates: dict[str, tuple[Any, ...]] = {}
    for clause in point_clauses:
        if not isinstance(clause.value, Mapping):
            return analyze_constraints(clause_graph, free_parameters=("point_coordinates",))
        raw_coordinates = clause.value.get("exact_coords")
        if not isinstance(raw_coordinates, (list, tuple)) or not raw_coordinates:
            return analyze_constraints(clause_graph, free_parameters=("point_coordinates",))
        object_id = clause.arguments[0]
        coordinates[object_id] = tuple(raw_coordinates)
    if not point_clauses:
        return analyze_constraints(clause_graph, free_parameters=("point_coordinates",))
    realization = Realization(
        branch_id=f"witness_{clause_graph.id}",
        coordinates=coordinates,
        satisfied_clause_ids=tuple(item.id for item in point_clauses),
        metadata={"source": "exact_point_coordinates"},
    )
    return analyze_constraints(clause_graph, realizations=(realization,))


def compile_parse_graph_state(graph: ParseGraph) -> tuple[ObservationGraph, ClauseGraph, ConstraintState]:
    """Compile a ParseGraph and classify its constraints in one explicit step."""

    observation_graph, clause_graph = compile_parse_graph(graph)
    state = realize_exact_points(clause_graph)
    return observation_graph, clause_graph, state


def _classify(
    clauses: Sequence[Any], parameters: tuple[str, ...], branches: tuple[Realization, ...]
) -> RealizationStatus:
    if _has_conflicting_fixed_coordinates(clauses):
        return RealizationStatus.INCONSISTENT
    if any(clause.predicate == "conflict" and clause.metadata.get("status") == "unresolved" for clause in clauses):
        return RealizationStatus.AMBIGUOUS
    if len(branches) > 1:
        return RealizationStatus.AMBIGUOUS
    if len(branches) == 1 and not parameters:
        return RealizationStatus.WELL_CONSTRAINED
    return RealizationStatus.UNDERCONSTRAINED


def _has_conflicting_fixed_coordinates(clauses: Sequence[Any]) -> bool:
    coordinates: dict[str, tuple[Any, ...]] = {}
    for clause in clauses:
        if not clause.predicate.startswith("object.point") or not isinstance(clause.value, Mapping):
            continue
        value = clause.value.get("exact_coords")
        if not isinstance(value, (list, tuple)):
            continue
        coordinate = tuple(value)
        object_id = clause.arguments[0]
        previous = coordinates.get(object_id)
        if previous is not None and previous != coordinate:
            return True
        coordinates[object_id] = coordinate
    return False
