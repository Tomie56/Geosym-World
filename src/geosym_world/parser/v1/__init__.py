"""Versioned contracts for image-to-geometry parsing."""

from .contracts import (
    CLAUSE_GRAPH_SCHEMA_VERSION,
    CONSTRAINT_STATE_SCHEMA_VERSION,
    OBSERVATION_GRAPH_SCHEMA_VERSION,
    REALIZATION_SCHEMA_VERSION,
    Clause,
    ClauseGraph,
    ConstraintState,
    Realization,
    RealizationStatus,
    Observation,
    ObservationGraph,
)
from .compiler import compile_parse_graph
from .analyzer import analyze_constraints, compile_parse_graph_state, realize_exact_points
from .admission import admit_clauses
from .model_payload import clause_graph_from_model_payload
from .facts import admitted_clauses_to_facts
from .materializer import MaterializationResult, materialize_clause_graph, materialize_exact_point_state
from .solver import ParserSolveResult, solve_admitted_clause_graph

__all__ = [
    "CLAUSE_GRAPH_SCHEMA_VERSION",
    "CONSTRAINT_STATE_SCHEMA_VERSION",
    "OBSERVATION_GRAPH_SCHEMA_VERSION",
    "REALIZATION_SCHEMA_VERSION",
    "Clause",
    "ClauseGraph",
    "ConstraintState",
    "Realization",
    "RealizationStatus",
    "Observation",
    "ObservationGraph",
    "compile_parse_graph",
    "analyze_constraints",
    "realize_exact_points",
    "compile_parse_graph_state",
    "admit_clauses",
    "clause_graph_from_model_payload",
    "admitted_clauses_to_facts",
    "MaterializationResult",
    "materialize_clause_graph",
    "materialize_exact_point_state",
    "ParserSolveResult",
    "solve_admitted_clause_graph",
]
