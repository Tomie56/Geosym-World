"""Uncertain parser output kept separate from admitted geometry facts."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping

from geosym_world.errors import InvalidGeometryInput
from .ids import validate_geometry_id
from .provenance import Provenance


class AdmissionStatus(str, Enum):
    PROPOSED = "proposed"
    ADMITTED = "admitted"
    REJECTED = "rejected"
    ABSTAINED = "abstained"


@dataclass(frozen=True)
class GeometryHypothesis:
    id: str
    kind: str
    payload: Mapping[str, Any]
    confidence: float
    provenance: tuple[Provenance, ...] = ()
    status: AdmissionStatus = AdmissionStatus.PROPOSED
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_geometry_id(self.id)
        if not self.kind:
            raise InvalidGeometryInput("Hypothesis kind must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise InvalidGeometryInput("Hypothesis confidence must be in [0, 1]")

    def with_status(self, status: AdmissionStatus) -> "GeometryHypothesis":
        return replace(self, status=status)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GeometryHypothesis":
        return cls(
            id=str(data["id"]),
            kind=str(data["kind"]),
            payload=dict(data.get("payload", {})),
            confidence=float(data["confidence"]),
            provenance=tuple(Provenance.from_dict(item) for item in data.get("provenance", ())),
            status=AdmissionStatus(data.get("status", AdmissionStatus.PROPOSED.value)),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "payload": dict(self.payload),
            "confidence": self.confidence,
            "provenance": [item.to_dict() for item in self.provenance],
            "status": self.status.value,
            "metadata": dict(self.metadata),
        }
