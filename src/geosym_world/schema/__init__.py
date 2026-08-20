"""Canonical geometry state contract."""

from .entities import Arc, Circle, CompositeRegion, Line, Plane, Point, Polygon, Polyhedron, Ray, Region, Segment, Target
from .canonical import canonical_json, canonicalize, record_checksum, stable_checksum, state_checksum
from .derivations import DERIVATION_SCHEMA_VERSION, DerivationStep, DerivationTracer, DerivationValidation, validate_derivation_payload
from .facts import Fact, FactStore
from .hypotheses import AdmissionStatus, GeometryHypothesis
from .ids import EventId, GeometryId, RecordId, validate_geometry_id
from .provenance import EvidenceKind, Provenance
from .problem import GeoSymDiagnostic, GeoSymGeometry, GeoSymInput, GeoSymProblemState, GeoSymProof, GeoSymSource, GeoSymTarget
from .problem_verifier import GeoSymVerifier
from .quantities import QuantityValue
from .records import GeometryRecord, SCHEMA_VERSION
from .relation_normalization import normalize_relation, normalize_relation_type
from .state import FormalGeometryState, PROOF_TARGET_TYPES, STATE_SCHEMA_VERSION
from .solutions import SOLUTION_PATH_SCHEMA_VERSION, SolutionPathValidation, build_solution_paths, validate_solution_paths
from .transitions import StateTransition

__all__ = [
    "AdmissionStatus",
    "Arc",
    "canonical_json",
    "canonicalize",
    "Circle",
    "CompositeRegion",
    "DERIVATION_SCHEMA_VERSION",
    "DerivationStep",
    "DerivationTracer",
    "DerivationValidation",
    "EventId",
    "EvidenceKind",
    "Fact",
    "FactStore",
    "FormalGeometryState",
    "GeometryHypothesis",
    "GeometryId",
    "GeometryRecord",
    "GeoSymDiagnostic",
    "GeoSymGeometry",
    "GeoSymInput",
    "GeoSymProblemState",
    "GeoSymProof",
    "GeoSymSource",
    "GeoSymTarget",
    "GeoSymVerifier",
    "Line",
    "Plane",
    "Point",
    "Polygon",
    "Polyhedron",
    "Provenance",
    "QuantityValue",
    "normalize_relation",
    "normalize_relation_type",
    "Ray",
    "RecordId",
    "record_checksum",
    "Region",
    "SCHEMA_VERSION",
    "STATE_SCHEMA_VERSION",
    "PROOF_TARGET_TYPES",
    "SOLUTION_PATH_SCHEMA_VERSION",
    "SolutionPathValidation",
    "build_solution_paths",
    "validate_solution_paths",
    "Segment",
    "stable_checksum",
    "state_checksum",
    "StateTransition",
    "Target",
    "validate_geometry_id",
    "validate_derivation_payload",
]
