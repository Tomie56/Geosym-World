"""Geometry state, synthesis, parsing, rendering, and benchmark foundation."""

from geosym_world.construction.builder import ConstructionBuilder
from geosym_world.construction.operations import ConstructionOp
from geosym_world.schema.expected import compare_expected_facts, legacy_expected_facts
from geosym_world.schema.facts import Fact, FactStore
from geosym_world.datasets.adapters.legacy import LegacyGeometryAdapter
from geosym_world.datasets.pilot_exporter import PilotBenchmarkExporter
from geosym_world.solver import FactClosureSolver, IncrementalFactClosure, IncrementalSolveReport
from geosym_world.schema.state import FormalGeometryState
from geosym_world.schema.entities import CompositeRegion
from geosym_world.rendering.quality import RenderQualityReport, inspect_render_scene
from geosym_world.parser.synthetic_debug import SyntheticParserDebugger
from geosym_world.parser.synthetic_v0 import SyntheticParserV0
from geosym_world.parser.contracts import parser_record_audit, validate_proposed_parser_record
from geosym_world.benchmark.parser_slices import ParserBenchmarkSlicer
from geosym_world.synthesis import SyntheticRecordExporter, SyntheticSampler
from geosym_world.verifier.trace import TraceVerifier, VerificationReport
from geosym_world.verifier.admission import AdmissionReport, GroundingReport, HypothesisAdmission

__all__ = [
    "ConstructionBuilder",
    "AdmissionReport",
    "ConstructionOp",
    "CompositeRegion",
    "Fact",
    "FactClosureSolver",
    "FactStore",
    "IncrementalFactClosure",
    "IncrementalSolveReport",
    "FormalGeometryState",
    "GroundingReport",
    "HypothesisAdmission",
    "LegacyGeometryAdapter",
    "PilotBenchmarkExporter",
    "RenderQualityReport",
    "SyntheticRecordExporter",
    "SyntheticParserV0",
    "ParserBenchmarkSlicer",
    "parser_record_audit",
    "SyntheticParserDebugger",
    "SyntheticSampler",
    "TraceVerifier",
    "VerificationReport",
    "compare_expected_facts",
    "inspect_render_scene",
    "legacy_expected_facts",
    "validate_proposed_parser_record",
]
