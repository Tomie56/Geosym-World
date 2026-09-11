"""Explicit admission of parser hypotheses after external verification."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from geosym_world.errors import InvalidGeometryInput, VerificationError

from .contracts import Clause, ClauseGraph


def admit_clauses(
    clause_graph: ClauseGraph,
    admitted_ids: Iterable[str],
    *,
    verification: Mapping[str, Mapping[str, Any]] | None = None,
) -> ClauseGraph:
    """Mark only verifier-approved hypotheses as admitted facts."""

    selected = tuple(admitted_ids)
    if len(selected) != len(set(selected)):
        raise InvalidGeometryInput("Admission contains duplicate clause ids")
    clause_by_id = {item.id: item for item in clause_graph.clauses}
    missing = set(selected) - set(clause_by_id)
    if missing:
        raise InvalidGeometryInput(f"Admission references unknown clauses: {sorted(missing)}")
    invalid_statuses = {
        clause_id: clause_by_id[clause_id].status
        for clause_id in selected
        if clause_by_id[clause_id].status in {"rejected", "derived"}
    }
    if invalid_statuses:
        raise VerificationError(
            f"Admission cannot transition clauses from terminal status: {invalid_statuses}"
        )
    if not isinstance(verification, Mapping) and verification is not None:
        raise InvalidGeometryInput("Admission verification must be a mapping")
    unresolved_dependencies = {
        dependency
        for clause_id in selected
        for dependency in clause_by_id[clause_id].dependencies
        if clause_by_id[dependency].status != "admitted" and dependency not in selected
    }
    if unresolved_dependencies:
        raise VerificationError(
            f"Admission has unresolved clause dependencies: {sorted(unresolved_dependencies)}"
        )
    reports = verification or {}
    if any(not isinstance(report, Mapping) for report in reports.values()):
        raise InvalidGeometryInput("Each admission verification report must be a mapping")
    rejected = []
    for clause_id in selected:
        accepted = reports.get(clause_id, {}).get("accepted", False)
        if not isinstance(accepted, bool):
            raise InvalidGeometryInput(
                f"Admission verification accepted flag must be boolean: {clause_id}"
            )
        if not accepted:
            rejected.append(clause_id)
    if rejected:
        raise VerificationError(f"Clauses were not explicitly verified: {sorted(rejected)}")

    clauses = tuple(
        replace(item, status="admitted") if item.id in selected else item
        for item in clause_graph.clauses
    )
    return replace(
        clause_graph,
        clauses=clauses,
        metadata={**dict(clause_graph.metadata), "admission": {"admitted_ids": list(selected),
                                                                  "verification": dict(reports)}},
    )
