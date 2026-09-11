"""Typed intermediate contracts for constraint-first diagram parsing."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from geosym_world.errors import InvalidGeometryInput
from geosym_world.schema.canonical import stable_checksum
from geosym_world.schema.ids import validate_geometry_id


OBSERVATION_GRAPH_SCHEMA_VERSION = "geosym.parser.observation_graph.v1"
CLAUSE_GRAPH_SCHEMA_VERSION = "geosym.parser.clause_graph.v1"
CONSTRAINT_STATE_SCHEMA_VERSION = "geosym.parser.constraint_state.v1"
REALIZATION_SCHEMA_VERSION = "geosym.parser.realization.v1"


class RealizationStatus(str, Enum):
    UNDERCONSTRAINED = "underconstrained"
    WELL_CONSTRAINED = "well_constrained"
    OVERCONSTRAINED = "overconstrained"
    INCONSISTENT = "inconsistent"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class Observation:
    id: str
    kind: str
    confidence: float
    geometry: Mapping[str, Any] = field(default_factory=dict)
    text: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_geometry_id(self.id)
        _require_text(self.kind, "Observation kind")
        _require_probability(self.confidence, "Observation confidence")
        _require_mapping(self.geometry, "Observation geometry")
        _require_mapping(self.metadata, "Observation metadata")

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "confidence": self.confidence,
                "geometry": dict(self.geometry), "text": self.text, "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Observation":
        _exact_fields(data, {"id", "kind", "confidence", "geometry", "text", "metadata"})
        return cls(**dict(data))


@dataclass(frozen=True)
class ObservationGraph:
    id: str
    source_id: str
    observations: tuple[Observation, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = OBSERVATION_GRAPH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_id_pair(self.id, self.source_id)
        _validate_version(self.schema_version, OBSERVATION_GRAPH_SCHEMA_VERSION)
        _unique(self.observations, "observation")
        _require_mapping(self.metadata, "Observation graph metadata")

    def _payload(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "id": self.id, "source_id": self.source_id,
                "observations": [item.to_dict() for item in sorted(self.observations, key=lambda item: item.id)],
                "metadata": dict(self.metadata)}

    def to_dict(self) -> dict[str, Any]:
        return _with_checksum(self._payload())

    def checksum(self) -> str:
        return stable_checksum(self._payload())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObservationGraph":
        _verify_checksum(data, {"schema_version", "id", "source_id", "observations", "metadata"})
        return cls(observations=tuple(Observation.from_dict(item) for item in data["observations"]),
                   **{key: data[key] for key in ("id", "source_id", "metadata", "schema_version")})


@dataclass(frozen=True)
class Clause:
    id: str
    predicate: str
    arguments: tuple[str, ...]
    confidence: float
    provenance: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    value: Any = None
    status: str = "proposed"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_geometry_id(self.id)
        _require_text(self.predicate, "Clause predicate")
        if not self.arguments or any(not isinstance(item, str) or not item.strip() for item in self.arguments):
            raise InvalidGeometryInput("Clause arguments must contain non-empty references")
        for argument in self.arguments:
            validate_geometry_id(argument)
        _require_probability(self.confidence, "Clause confidence")
        for reference in (*self.provenance, *self.dependencies):
            if not isinstance(reference, str) or not reference.strip():
                raise InvalidGeometryInput(f"Clause {self.id} contains an empty provenance or dependency reference")
            validate_geometry_id(reference)
        if self.status not in {"proposed", "admitted", "derived", "rejected"}:
            raise InvalidGeometryInput(f"Unknown clause status: {self.status}")
        if self.id in self.dependencies:
            raise InvalidGeometryInput(f"Clause {self.id} cannot depend on itself")
        _require_mapping(self.metadata, f"Clause {self.id} metadata")

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "predicate": self.predicate, "arguments": list(self.arguments),
                "confidence": self.confidence, "provenance": list(self.provenance),
                "dependencies": list(self.dependencies), "value": self.value, "status": self.status,
                "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Clause":
        _exact_fields(data, {
            "id", "predicate", "arguments", "confidence", "provenance",
            "dependencies", "value", "status", "metadata",
        })
        return cls(arguments=tuple(data["arguments"]), provenance=tuple(data["provenance"]),
                   dependencies=tuple(data["dependencies"]), metadata=dict(data["metadata"]),
                   **{key: data[key] for key in ("id", "predicate", "confidence", "value", "status")})


@dataclass(frozen=True)
class ClauseGraph:
    id: str
    source_id: str
    clauses: tuple[Clause, ...] = ()
    target_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = CLAUSE_GRAPH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_id_pair(self.id, self.source_id)
        _validate_version(self.schema_version, CLAUSE_GRAPH_SCHEMA_VERSION)
        _unique(self.clauses, "clause")
        clause_ids = {item.id for item in self.clauses}
        if len(self.target_ids) != len(set(self.target_ids)):
            raise InvalidGeometryInput("Clause graph contains duplicate target ids")
        for target_id in self.target_ids:
            validate_geometry_id(target_id)
        missing_targets = set(self.target_ids) - clause_ids
        if missing_targets:
            raise InvalidGeometryInput(f"Clause graph has unknown target ids: {sorted(missing_targets)}")
        for clause in self.clauses:
            missing = set(clause.dependencies) - clause_ids
            if missing:
                raise InvalidGeometryInput(f"Clause {clause.id} has unknown dependencies: {sorted(missing)}")
        _require_mapping(self.metadata, "Clause graph metadata")

    def _payload(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "id": self.id, "source_id": self.source_id,
                "clauses": [item.to_dict() for item in sorted(self.clauses, key=lambda item: item.id)],
                "target_ids": list(self.target_ids), "metadata": dict(self.metadata)}

    def to_dict(self) -> dict[str, Any]:
        return _with_checksum(self._payload())

    def checksum(self) -> str:
        return stable_checksum(self._payload())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClauseGraph":
        _verify_checksum(data, {"schema_version", "id", "source_id", "clauses", "target_ids", "metadata"})
        return cls(clauses=tuple(Clause.from_dict(item) for item in data["clauses"]),
                   target_ids=tuple(data["target_ids"]),
                   **{key: data[key] for key in ("id", "source_id", "metadata", "schema_version")})


@dataclass(frozen=True)
class Realization:
    branch_id: str
    coordinates: Mapping[str, tuple[Any, ...]]
    satisfied_clause_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = REALIZATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_geometry_id(self.branch_id)
        _validate_version(self.schema_version, REALIZATION_SCHEMA_VERSION)
        if not self.coordinates:
            raise InvalidGeometryInput("Realization requires at least one coordinate")
        for object_id, coordinate in self.coordinates.items():
            validate_geometry_id(object_id)
            if not isinstance(coordinate, tuple) or not coordinate:
                raise InvalidGeometryInput("Realization coordinates must be non-empty tuples")
        _require_mapping(self.metadata, "Realization metadata")

    def _payload(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "branch_id": self.branch_id,
                "coordinates": {key: list(value) for key, value in sorted(self.coordinates.items())},
                "satisfied_clause_ids": list(self.satisfied_clause_ids), "metadata": dict(self.metadata)}

    def to_dict(self) -> dict[str, Any]:
        return _with_checksum(self._payload())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Realization":
        _verify_checksum(data, {"schema_version", "branch_id", "coordinates", "satisfied_clause_ids", "metadata"})
        return cls(coordinates={key: tuple(value) for key, value in data["coordinates"].items()},
                   satisfied_clause_ids=tuple(data["satisfied_clause_ids"]),
                   **{key: data[key] for key in ("branch_id", "metadata", "schema_version")})


@dataclass(frozen=True)
class ConstraintState:
    id: str
    clause_graph_checksum: str
    status: RealizationStatus
    free_parameters: tuple[str, ...] = ()
    realizations: tuple[Realization, ...] = ()
    derived_clause_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = CONSTRAINT_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_geometry_id(self.id)
        _require_sha256(self.clause_graph_checksum, "Clause graph checksum")
        _validate_version(self.schema_version, CONSTRAINT_STATE_SCHEMA_VERSION)
        _unique(self.realizations, "realization", key=lambda item: item.branch_id)
        if self.status == RealizationStatus.UNDERCONSTRAINED and not self.free_parameters:
            raise InvalidGeometryInput("Underconstrained state must expose free parameters")
        if self.status == RealizationStatus.WELL_CONSTRAINED and len(self.realizations) != 1:
            raise InvalidGeometryInput("Well-constrained state requires exactly one realization")
        _require_mapping(self.metadata, "Constraint state metadata")
        if (
            self.status == RealizationStatus.AMBIGUOUS
            and len(self.realizations) < 2
            and not self.metadata.get("ambiguity_reason")
        ):
            raise InvalidGeometryInput("Ambiguous state requires two realizations or an ambiguity reason")

    def _payload(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "id": self.id,
                "clause_graph_checksum": self.clause_graph_checksum, "status": self.status.value,
                "free_parameters": list(self.free_parameters),
                "realizations": [item.to_dict() for item in self.realizations],
                "derived_clause_ids": list(self.derived_clause_ids), "metadata": dict(self.metadata)}

    def to_dict(self) -> dict[str, Any]:
        return _with_checksum(self._payload())

    def checksum(self) -> str:
        return stable_checksum(self._payload())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConstraintState":
        _verify_checksum(data, {
            "schema_version", "id", "clause_graph_checksum", "status",
            "free_parameters", "realizations", "derived_clause_ids", "metadata",
        })
        return cls(realizations=tuple(Realization.from_dict(item) for item in data["realizations"]),
                   status=RealizationStatus(data["status"]), free_parameters=tuple(data["free_parameters"]),
                   derived_clause_ids=tuple(data["derived_clause_ids"]),
                   **{key: data[key] for key in ("id", "clause_graph_checksum", "metadata", "schema_version")})


def _validate_id_pair(first: str, second: str) -> None:
    validate_geometry_id(first)
    validate_geometry_id(second)


def _exact_fields(data: Mapping[str, Any], expected: set[str]) -> None:
    if not isinstance(data, Mapping):
        raise InvalidGeometryInput("Parser contract payload must be a mapping")
    unknown = set(data) - expected
    missing = expected - set(data)
    if unknown or missing:
        raise InvalidGeometryInput(
            f"Parser contract fields mismatch; unknown={sorted(unknown)}, missing={sorted(missing)}"
        )


def _with_checksum(payload: dict[str, Any]) -> dict[str, Any]:
    return {**payload, "checksum": stable_checksum(payload)}


def _verify_checksum(data: Mapping[str, Any], payload_fields: set[str]) -> None:
    _exact_fields(data, {*payload_fields, "checksum"})
    payload = {key: data[key] for key in payload_fields}
    if data["checksum"] != stable_checksum(payload):
        raise InvalidGeometryInput("Parser contract checksum does not match payload")


def _validate_version(value: str, expected: str) -> None:
    if value != expected:
        raise InvalidGeometryInput(f"Unsupported parser contract schema version: {value}")


def _unique(values: tuple[Any, ...], label: str, key=lambda item: item.id) -> None:
    ids = [key(item) for item in values]
    if len(ids) != len(set(ids)):
        raise InvalidGeometryInput(f"Duplicate {label} ids")


def _require_text(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise InvalidGeometryInput(f"{label} must be non-empty")


def _require_probability(value: Any, label: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 <= value <= 1
    ):
        raise InvalidGeometryInput(f"{label} must be finite and in [0, 1]")


def _require_mapping(value: Any, label: str) -> None:
    if not isinstance(value, Mapping):
        raise InvalidGeometryInput(f"{label} must be a mapping")


def _require_sha256(value: Any, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise InvalidGeometryInput(f"{label} must be a lowercase SHA-256 checksum")
