"""Compile the existing ParseGraph candidate layer into Parser v1 clauses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from geosym_world.errors import InvalidGeometryInput
from geosym_world.schema import ParseGraph

from .contracts import Clause, ClauseGraph, Observation, ObservationGraph


def compile_parse_graph(graph: ParseGraph) -> tuple[ObservationGraph, ClauseGraph]:
    """Preserve ParseGraph candidates as explicit, provenance-bearing clauses."""

    if not isinstance(graph, ParseGraph):
        raise InvalidGeometryInput("Parser v1 compiler requires a ParseGraph")

    observations = tuple(_compile_evidence(item) for item in graph.evidence)
    clauses = []
    for item in graph.objects:
        clauses.append(_candidate_clause(item.id, f"object.{item.kind.value}",
                                          tuple(reference.ref for reference in item.references),
                                          item.confidence, item.evidence_refs, value=item.geometry,
                                          metadata={"origin": item.origin.value, "visible_name": item.visible_name,
                                                    "aliases": list(item.aliases)}))
    for item in graph.relations:
        relation_metadata = dict(item.metadata)
        relation_metadata["argument_roles"] = [argument.role for argument in item.arguments]
        clauses.append(_candidate_clause(item.id, item.predicate,
                                          tuple(reference.ref for reference in item.arguments), item.confidence,
                                          item.evidence_refs, item.dependencies, metadata=relation_metadata))
    for item in graph.quantities:
        clauses.append(_candidate_clause(item.id, f"quantity.{item.kind}",
                                          tuple(reference.ref for reference in item.arguments), item.confidence,
                                          item.evidence_refs, item.dependencies, value=item.expression.to_dict(),
                                          metadata={
                                              "unit": item.unit,
                                              "raw_text": item.raw_text,
                                              **dict(item.metadata),
                                          }))
    for item in graph.regions:
        arguments = tuple(reference.ref for reference in item.boundary) + item.holes + item.components
        clauses.append(_candidate_clause(item.id, "region", arguments, item.confidence, item.evidence_refs,
                                          value=item.expression.to_dict() if item.expression else None,
                                          metadata={"boundary": [reference.ref for reference in item.boundary],
                                                    "holes": list(item.holes), "components": list(item.components),
                                                    **dict(item.metadata)}))
    for item in graph.targets:
        arguments = tuple(reference.ref for reference in item.arguments)
        value = {"type": item.kind, "mode": item.mode}
        if arguments:
            value["object"] = arguments[0]
        if item.expression is not None:
            value["expression"] = item.expression.to_dict()
        if item.mode == "constraint":
            expectation = dict(item.metadata.get("expectation", {}))
            if "operator" in item.metadata and "operator" not in expectation:
                expectation["operator"] = item.metadata["operator"]
            if "value" in item.metadata and "value" not in expectation:
                expectation["value"] = item.metadata["value"]
            if expectation:
                value["expectation"] = expectation
        clauses.append(_candidate_clause(
            item.id,
            "target",
            arguments,
            item.confidence,
            item.evidence_refs,
            item.dependencies,
            value=value,
            metadata=dict(item.metadata),
        ))
    for item in graph.conflicts:
        clauses.append(_candidate_clause(
            item.id,
            "conflict",
            item.members,
            0.0 if item.status.value == "unresolved" else 1.0,
            (),
            metadata={"status": item.status.value, "selected": item.selected, "reason": item.reason},
        ))

    clause_graph = ClauseGraph(
        id=f"clauses_{graph.id}",
        source_id=graph.source_id,
        clauses=tuple(clauses),
        target_ids=tuple(item.id for item in graph.targets),
        metadata={"parse_graph_checksum": graph.checksum(), "compiler": "parse_graph_v1"},
    )
    observation_graph = ObservationGraph(
        id=f"observations_{graph.id}",
        source_id=graph.source_id,
        observations=observations,
        metadata={"parse_graph_checksum": graph.checksum()},
    )
    return observation_graph, clause_graph


def _compile_evidence(item: Any) -> Observation:
    geometry: dict[str, Any] = {}
    for name in ("bbox", "point", "span"):
        value = getattr(item, name)
        if value is not None:
            geometry[name] = list(value)
    return Observation(item.id, item.kind.value, 1.0, geometry, item.raw_content, dict(item.metadata))


def _candidate_clause(
    clause_id: str,
    predicate: str,
    arguments: tuple[str, ...],
    confidence: float,
    provenance: tuple[str, ...],
    dependencies: tuple[str, ...] = (),
    value: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> Clause:
    return Clause(
        id=clause_id,
        predicate=predicate,
        arguments=arguments or (clause_id,),
        confidence=confidence,
        provenance=provenance,
        dependencies=dependencies,
        value=value,
        status="proposed",
        metadata=dict(metadata or {}),
    )
