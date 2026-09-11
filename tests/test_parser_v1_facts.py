import pytest

from geosym_world.errors import InvalidGeometryInput
from geosym_world.parser import Clause, ClauseGraph, admitted_clauses_to_facts
from geosym_world.schema import FactScope


def test_fact_projection_excludes_proposed_and_preserves_provenance():
    graph = ClauseGraph(
        "clauses_1", "problem_1",
        (
            Clause("proposed", "parallel", ("AB", "CD"), 0.8),
            Clause("admitted", "parallel", ("AB", "CD"), 1.0, status="admitted"),
            Clause("derived", "perpendicular", ("AB", "EF"), 1.0, status="derived",
                   dependencies=("admitted",)),
        ),
    )

    facts = admitted_clauses_to_facts(graph, source_id="problem_1")

    assert len(facts.find()) == 2
    assert facts.find("parallel", "AB")[0].scope is FactScope.SOURCE_ASSERTION
    assert facts.find("perpendicular", "AB")[0].scope is FactScope.ENTAILED_FACT
    assert facts.find("perpendicular", "AB")[0].provenance[0].event_id == "derived"
    assert facts.find("parallel", "AB")[0].metadata["parser_clause_id"] == "admitted"


def test_fact_projection_requires_source_id():
    with pytest.raises(InvalidGeometryInput, match="source_id"):
        admitted_clauses_to_facts(ClauseGraph("clauses_1", "problem_1"), source_id="")


def test_quantity_clause_projects_exact_value_to_standard_fact():
    graph = ClauseGraph(
        "clauses_2", "problem_2",
        (Clause(
            "quantity_ab", "quantity.length", ("AB",), 1.0,
            value={"operator": "div", "operands": [
                {"operator": "literal", "value": "3"},
                {"operator": "literal", "value": "2"},
            ]},
            status="admitted",
        ),),
    )

    facts = admitted_clauses_to_facts(graph, source_id="problem_2")

    assert facts.find("length", "AB")[0].value == "3/2"
