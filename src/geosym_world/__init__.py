"""Geometry state, synthesis, parsing, rendering, and benchmark foundation."""

from geosym_world.construction.builder import ConstructionBuilder
from geosym_world.construction.operations import ConstructionOp
from geosym_world.schema.expected import compare_expected_facts, legacy_expected_facts
from geosym_world.schema.facts import Fact, FactStore
from geosym_world.datasets.adapters.legacy import LegacyGeometryAdapter
from geosym_world.datasets.pilot_exporter import PilotBenchmarkExporter
from geosym_world.solver.closure import FactClosureSolver
from geosym_world.schema.state import FormalGeometryState
from geosym_world.synthesis import SyntheticRecordExporter, SyntheticSampler
from geosym_world.verifier.trace import TraceVerifier, VerificationReport

__all__ = [
    "ConstructionBuilder",
    "ConstructionOp",
    "Fact",
    "FactClosureSolver",
    "FactStore",
    "FormalGeometryState",
    "LegacyGeometryAdapter",
    "PilotBenchmarkExporter",
    "SyntheticRecordExporter",
    "SyntheticSampler",
    "TraceVerifier",
    "VerificationReport",
    "compare_expected_facts",
    "legacy_expected_facts",
]
