import pytest

from geosym_world.errors import InvalidGeometryInput, VerificationError
from geosym_world.parser.v1 import Clause, ClauseGraph, admit_clauses


def _graph():
    return ClauseGraph("clauses_1", "problem_1", (Clause("clause_1", "parallel", ("AB", "CD"), 0.7),))


def test_admission_requires_explicit_verifier_acceptance():
    with pytest.raises(VerificationError, match="explicitly verified"):
        admit_clauses(_graph(), ("clause_1",))

    admitted = admit_clauses(_graph(), ("clause_1",), verification={"clause_1": {"accepted": True}})
    assert admitted.clauses[0].status == "admitted"
    assert admitted.metadata["admission"]["admitted_ids"] == ["clause_1"]


def test_admission_rejects_unknown_clause_ids():
    with pytest.raises(InvalidGeometryInput, match="unknown clauses"):
        admit_clauses(_graph(), ("missing",), verification={"missing": {"accepted": True}})


def test_admission_requires_dependency_closure():
    graph = ClauseGraph(
        "clauses_1",
        "problem_1",
        (
            Clause("base", "object.point", ("A",), 1.0),
            Clause("derived", "point_on_line", ("A", "BC"), 1.0, dependencies=("base",)),
        ),
    )
    with pytest.raises(VerificationError, match="unresolved clause dependencies"):
        admit_clauses(graph, ("derived",), verification={"derived": {"accepted": True}})


def test_admission_rejects_malformed_verification_reports():
    with pytest.raises(InvalidGeometryInput, match="verification report"):
        admit_clauses(_graph(), ("clause_1",), verification={"clause_1": "accepted"})


@pytest.mark.parametrize("status", ["rejected", "derived"])
def test_admission_rejects_terminal_status_transition(status):
    graph = ClauseGraph(
        "clauses_1", "problem_1", (Clause("clause_1", "parallel", ("AB", "CD"), 1.0, status=status),)
    )
    with pytest.raises(VerificationError, match="terminal status"):
        admit_clauses(graph, ("clause_1",), verification={"clause_1": {"accepted": True}})


def test_admission_rejects_non_boolean_acceptance():
    with pytest.raises(InvalidGeometryInput, match="accepted flag must be boolean"):
        admit_clauses(_graph(), ("clause_1",), verification={"clause_1": {"accepted": "false"}})
