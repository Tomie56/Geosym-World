import pytest

from geosym_world.errors import InvalidGeometryInput
from geosym_world.parser import CLAUSE_GRAPH_SCHEMA_VERSION, clause_graph_from_model_payload


def test_model_payload_is_ingested_as_proposed_clauses():
    graph = clause_graph_from_model_payload(
        {
            "schema_version": CLAUSE_GRAPH_SCHEMA_VERSION,
            "clauses": [{
                "id": "clause_1",
                "predicate": "parallel",
                "arguments": ["AB", "CD"],
                "confidence": 0.6,
                "provenance": ["model_1"],
                "dependencies": [],
                "value": None,
                "status": "proposed",
                "metadata": {"model": "test"},
            }],
            "target_ids": [],
            "metadata": {"backend": "fixture"},
        },
        graph_id="clauses_1",
        source_id="problem_1",
    )

    assert graph.clauses[0].status == "proposed"
    assert graph.clauses[0].metadata["model"] == "test"


def test_model_payload_rejects_missing_fields():
    with pytest.raises(InvalidGeometryInput, match="fields mismatch"):
        clause_graph_from_model_payload({}, graph_id="clauses_1", source_id="problem_1")


def test_model_payload_cannot_self_admit_a_clause():
    with pytest.raises(InvalidGeometryInput, match="proposed hypotheses"):
        clause_graph_from_model_payload(
                {
                "schema_version": CLAUSE_GRAPH_SCHEMA_VERSION,
                "clauses": [{
                    "id": "clause_1", "predicate": "parallel", "arguments": ["AB", "CD"],
                    "confidence": 1.0, "provenance": ["model_1"], "dependencies": [],
                    "value": None, "status": "admitted", "metadata": {},
                }],
                "target_ids": [], "metadata": {},
            },
            graph_id="clauses_1", source_id="problem_1",
        )


def test_model_payload_rejects_invalid_target_and_metadata_types():
    payload = {
        "schema_version": CLAUSE_GRAPH_SCHEMA_VERSION,
        "clauses": [],
        "target_ids": "target_1",
        "metadata": {},
    }
    with pytest.raises(InvalidGeometryInput, match="target_ids must be a list"):
        clause_graph_from_model_payload(payload, graph_id="clauses_1", source_id="problem_1")

    payload["target_ids"] = []
    payload["metadata"] = []
    with pytest.raises(InvalidGeometryInput, match="metadata must be a mapping"):
        clause_graph_from_model_payload(payload, graph_id="clauses_1", source_id="problem_1")


def test_model_payload_rejects_unknown_schema_version():
    payload = {
        "schema_version": "geosym.parser.clause_graph.v0",
        "clauses": [],
        "target_ids": [],
        "metadata": {},
    }
    with pytest.raises(InvalidGeometryInput, match="schema version"):
        clause_graph_from_model_payload(payload, graph_id="clauses_1", source_id="problem_1")
