import pytest

from geosym_world.errors import InvalidGeometryInput
from geosym_world.parser.v1 import (
    Clause,
    ClauseGraph,
    ConstraintState,
    Observation,
    ObservationGraph,
    RealizationStatus,
)


def test_constraint_contract_round_trip_preserves_checksum():
    observation_graph = ObservationGraph(
        id="obs_1",
        source_id="problem_1",
        observations=(Observation("pixel_1", "segment", 0.9, {"points": ["A", "B"]}),),
    )
    clause_graph = ClauseGraph(
        id="clauses_1",
        source_id="problem_1",
        clauses=(Clause("clause_1", "collinear", ("A", "B", "C"), 0.8, ("pixel_1",)),),
    )
    state = ConstraintState(
        id="constraints_1",
        clause_graph_checksum=clause_graph.checksum(),
        status=RealizationStatus.UNDERCONSTRAINED,
        free_parameters=("rotation", "scale"),
    )

    assert ObservationGraph.from_dict(observation_graph.to_dict()).checksum() == observation_graph.checksum()
    assert ClauseGraph.from_dict(clause_graph.to_dict()).checksum() == clause_graph.checksum()
    assert ConstraintState.from_dict(state.to_dict()).checksum() == state.checksum()


def test_underconstrained_state_requires_explicit_free_parameters():
    with pytest.raises(InvalidGeometryInput, match="free parameters"):
        ConstraintState("constraints_1", "a" * 64, RealizationStatus.UNDERCONSTRAINED)


def test_serialized_checksum_rejects_tampering():
    graph = ObservationGraph("obs_1", "problem_1")
    payload = graph.to_dict()
    payload["metadata"] = {"changed": True}

    with pytest.raises(InvalidGeometryInput, match="checksum"):
        ObservationGraph.from_dict(payload)


def test_constraint_state_requires_witness_for_well_constrained_status():
    with pytest.raises(InvalidGeometryInput, match="exactly one realization"):
        ConstraintState("constraints_1", "a" * 64, RealizationStatus.WELL_CONSTRAINED)


def test_clause_graph_rejects_unknown_dependencies():
    with pytest.raises(InvalidGeometryInput, match="unknown dependencies"):
        ClauseGraph(
            "clauses_1",
            "problem_1",
            (Clause("clause_1", "parallel", ("AB", "CD"), 1.0, dependencies=("missing",)),),
        )


def test_clause_rejects_invalid_provenance_reference():
    with pytest.raises(InvalidGeometryInput, match="provenance or dependency"):
        Clause("clause_1", "parallel", ("AB", "CD"), 1.0, provenance=("",))


def test_clause_rejects_natural_language_argument_as_reference():
    with pytest.raises(InvalidGeometryInput):
        Clause("clause_1", "parallel", ("line AB", "CD"), 1.0)


def test_clause_graph_rejects_unknown_targets():
    with pytest.raises(InvalidGeometryInput, match="unknown target ids"):
        ClauseGraph("clauses_1", "problem_1", target_ids=("missing_target",))
