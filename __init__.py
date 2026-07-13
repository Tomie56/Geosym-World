"""Executable geometry world for GeoSym-Hard and GeoSym-Debug prototypes."""

from .builder import ConstructionBuilder
from .construction import ConstructionOp
from .expected import compare_expected_facts, legacy_expected_facts
from .fact import Fact, FactStore
from .legacy_adapter import LegacyGeometryAdapter
from .pilot_exporter import PilotBenchmarkExporter
from .solver import FactClosureSolver
from .state import FormalGeometryState
from .synthesis import SyntheticRecordExporter, SyntheticSampler
from .verifier import TraceVerifier, VerificationReport

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
