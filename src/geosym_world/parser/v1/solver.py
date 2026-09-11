"""Run the existing geometry fact closure on a materialized Parser graph."""

from __future__ import annotations

from dataclasses import dataclass

from geosym_world.schema import FactStore
from geosym_world.solver import FactClosureSolver

from .contracts import ClauseGraph
from .materializer import MaterializationResult, materialize_clause_graph


@dataclass(frozen=True)
class ParserSolveResult:
    materialization: MaterializationResult
    facts: FactStore


def solve_admitted_clause_graph(
    clause_graph: ClauseGraph,
    *,
    solver_profile: str = "auto",
) -> ParserSolveResult:
    """Materialize admitted clauses and run the canonical FactClosureSolver."""

    materialization = materialize_clause_graph(clause_graph)
    facts = FactClosureSolver(profile=solver_profile).solve(materialization.state)
    return ParserSolveResult(materialization=materialization, facts=facts)
