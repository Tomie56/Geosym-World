"""Projection from admitted Parser clauses to the existing FactStore."""

from __future__ import annotations

from geosym_world.errors import InvalidGeometryInput
from geosym_world.schema import FactScope, FactStore, MathExpression, Provenance
from geosym_world.schema.provenance import EvidenceKind

from .contracts import ClauseGraph


def admitted_clauses_to_facts(clause_graph: ClauseGraph, *, source_id: str) -> FactStore:
    """Project only admitted or derived clauses into source-reasoning facts."""

    if not isinstance(source_id, str) or not source_id.strip():
        raise InvalidGeometryInput("Fact projection source_id must be non-empty")
    facts = FactStore()
    for clause in clause_graph.clauses:
        if clause.status not in {"admitted", "derived"}:
            continue
        scope = FactScope.SOURCE_ASSERTION if clause.status == "admitted" else FactScope.ENTAILED_FACT
        fact_type = clause.predicate
        value = None
        if clause.predicate.startswith("quantity."):
            fact_type = clause.predicate.removeprefix("quantity.")
            if isinstance(clause.value, dict):
                value = MathExpression.from_dict(clause.value).to_sympy()
        facts.add(
            fact_type,
            clause.arguments[0],
            value,
            source=source_id,
            scope=scope,
            rule="parser_v1",
            dependencies=clause.dependencies,
            metadata={
                **dict(clause.metadata),
                "arguments": list(clause.arguments),
                "value": clause.value,
                "parser_clause_id": clause.id,
            },
            provenance=(Provenance(
                source_id=source_id,
                kind=EvidenceKind.PARSER,
                event_id=clause.id,
                dependencies=clause.dependencies,
                metadata={"clause_id": clause.id},
            ),),
        )
    return facts
