import pytest

from geosym_world.errors import InvalidGeometryInput
from geosym_world.parser.v1 import (
    Clause,
    ClauseGraph,
    RealizationStatus,
    analyze_constraints,
    compile_parse_graph_state,
    realize_exact_points,
)
from geosym_world.parser import (
    Clause as PublicClause,
    Observation as PublicObservation,
    compile_parse_graph_state as public_compile_parse_graph_state,
)


def test_constraint_analysis_reports_underconstrained_without_inventing_coordinates():
    graph = ClauseGraph("clauses_1", "problem_1", (Clause("point_a", "object.point", ("A",), 0.8),))

    state = analyze_constraints(graph, free_parameters=("translation_x", "translation_y"))

    assert state.status is RealizationStatus.UNDERCONSTRAINED
    assert state.realizations == ()
    assert state.clause_graph_checksum == graph.checksum()


def test_constraint_analysis_detects_conflicting_fixed_point_coordinates():
    graph = ClauseGraph(
        "clauses_1",
        "problem_1",
        (
            Clause("point_a_1", "object.point", ("A",), 1.0, value={"exact_coords": [0, 0]}),
            Clause("point_a_2", "object.point", ("A",), 1.0, value={"exact_coords": [1, 0]}),
        ),
    )

    state = analyze_constraints(graph)

    assert state.status is RealizationStatus.INCONSISTENT


def test_constraint_analysis_rejects_untyped_realizations():
    graph = ClauseGraph("clauses_1", "problem_1")
    with pytest.raises(InvalidGeometryInput, match="typed Realization"):
        analyze_constraints(graph, realizations=(object(),))


def test_exact_point_realization_is_created_only_from_explicit_coordinates():
    graph = ClauseGraph(
        "clauses_1",
        "problem_1",
        (Clause("point_a", "object.point", ("A",), 1.0, value={"exact_coords": [0, 1]}),),
    )

    state = realize_exact_points(graph)

    assert state.status is RealizationStatus.WELL_CONSTRAINED
    assert state.realizations[0].coordinates["A"] == (0, 1)


def test_missing_point_coordinates_remain_underconstrained():
    graph = ClauseGraph("clauses_1", "problem_1", (Clause("point_a", "object.point", ("A",), 1.0),))

    state = realize_exact_points(graph)

    assert state.status is RealizationStatus.UNDERCONSTRAINED
    assert state.realizations == ()


def test_unresolved_conflict_is_preserved_as_ambiguity():
    graph = ClauseGraph(
        "clauses_1", "problem_1",
        (Clause("conflict_1", "conflict", ("candidate_a", "candidate_b"), 0.0,
                metadata={"status": "unresolved"}),),
    )

    state = analyze_constraints(graph)

    assert state.status is RealizationStatus.AMBIGUOUS
    assert state.metadata["ambiguity_reason"] == "unresolved_clause_conflict"


def test_compile_parse_graph_state_exposes_all_parser_stages():
    graph = ClauseGraph("clauses_1", "problem_1")
    with pytest.raises(InvalidGeometryInput, match="ParseGraph"):
        compile_parse_graph_state(graph)


def test_parser_public_api_exposes_v1_entrypoint():
    assert public_compile_parse_graph_state is compile_parse_graph_state
    assert PublicClause is Clause
    assert PublicObservation.__name__ == "Observation"
