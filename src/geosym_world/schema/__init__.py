"""Canonical geometry state contract."""

from .entities import Arc, Circle, Line, Plane, Point, Polygon, Polyhedron, Ray, Region, Segment, Target
from .facts import Fact, FactStore
from .hypotheses import AdmissionStatus, GeometryHypothesis
from .ids import EventId, GeometryId, RecordId, validate_geometry_id
from .provenance import EvidenceKind, Provenance
from .problem import GeoSymDiagnostic, GeoSymGeometry, GeoSymInput, GeoSymProblemState, GeoSymProof, GeoSymSource, GeoSymTarget
from .problem_verifier import GeoSymVerifier
from .quantities import QuantityValue
from .records import GeometryRecord, SCHEMA_VERSION
from .state import FormalGeometryState

__all__ = [
    "AdmissionStatus",
    "Arc",
    "Circle",
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
    "Ray",
    "RecordId",
    "Region",
    "SCHEMA_VERSION",
    "Segment",
    "Target",
    "validate_geometry_id",
]
