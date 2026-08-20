"""Versioned records shared by synthesis, parsers, benchmarks, and agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .hypotheses import GeometryHypothesis
from .ids import validate_geometry_id
from .state import FormalGeometryState

SCHEMA_VERSION = "1.0"


@dataclass
class GeometryRecord:
    id: str
    source: Mapping[str, Any]
    state: FormalGeometryState | None = None
    hypotheses: list[GeometryHypothesis] = field(default_factory=list)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_geometry_id(self.id)
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported geometry record schema version: {self.schema_version}")

    def to_dict(self, *, include_state: bool = True) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "source": dict(self.source),
            "state": self.state.to_dict() if include_state and self.state else None,
            "hypotheses": [item.to_dict() for item in self.hypotheses],
            "metadata": dict(self.metadata),
        }

    def checksum(self) -> str:
        from .canonical import record_checksum

        return record_checksum(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GeometryRecord":
        version = str(data.get("schema_version", ""))
        if version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported geometry record schema version: {version or '<missing>'}")
        state_payload = data.get("state")
        return cls(
            id=str(data["id"]),
            source=dict(data.get("source", {})),
            state=FormalGeometryState.from_dict(state_payload) if state_payload is not None else None,
            hypotheses=[GeometryHypothesis.from_dict(item) for item in data.get("hypotheses", ())],
            metadata=dict(data.get("metadata", {})),
            schema_version=version,
        )
