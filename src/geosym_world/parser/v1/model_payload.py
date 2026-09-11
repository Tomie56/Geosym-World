"""Strict ingestion of model-produced Parser v1 clause payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from geosym_world.errors import InvalidGeometryInput

from .contracts import CLAUSE_GRAPH_SCHEMA_VERSION, Clause, ClauseGraph


def clause_graph_from_model_payload(
    payload: Mapping[str, Any],
    *,
    graph_id: str,
    source_id: str,
) -> ClauseGraph:
    """Parse a model response without admitting it into formal geometry."""

    if not isinstance(payload, Mapping):
        raise InvalidGeometryInput("Model Parser payload must be a mapping")
    expected = {"schema_version", "clauses", "target_ids", "metadata"}
    unknown = set(payload) - expected
    missing = expected - set(payload)
    if unknown or missing:
        raise InvalidGeometryInput(
            f"Model Parser payload fields mismatch; unknown={sorted(unknown)}, missing={sorted(missing)}"
        )
    if payload["schema_version"] != CLAUSE_GRAPH_SCHEMA_VERSION:
        raise InvalidGeometryInput(
            f"Unsupported model Parser payload schema version: {payload['schema_version']}"
        )
    if not isinstance(payload["target_ids"], list):
        raise InvalidGeometryInput("Model Parser target_ids must be a list")
    if not isinstance(payload["metadata"], Mapping):
        raise InvalidGeometryInput("Model Parser metadata must be a mapping")
    clauses = payload["clauses"]
    if isinstance(clauses, (str, bytes)) or not isinstance(clauses, list):
        raise InvalidGeometryInput("Model Parser clauses must be a list")
    parsed_clauses = tuple(Clause.from_dict(item) for item in clauses)
    non_proposed = [item.id for item in parsed_clauses if item.status != "proposed"]
    if non_proposed:
        raise InvalidGeometryInput(
            f"Model Parser clauses must be proposed hypotheses: {sorted(non_proposed)}"
        )
    return ClauseGraph(
        id=graph_id,
        source_id=source_id,
        clauses=parsed_clauses,
        target_ids=tuple(payload["target_ids"]),
        metadata=dict(payload["metadata"]),
        schema_version=payload["schema_version"],
    )
