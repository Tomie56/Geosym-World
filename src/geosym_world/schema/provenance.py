"""Provenance carried by observations, hypotheses, and facts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class EvidenceKind(str, Enum):
    SOURCE = "source"
    PARSER = "parser"
    CONSTRUCTION = "construction"
    SOLVER = "solver"
    VERIFIER = "verifier"
    AGENT = "agent"


@dataclass(frozen=True)
class Provenance:
    source_id: str
    kind: EvidenceKind
    event_id: str | None = None
    dependencies: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Provenance":
        return cls(
            source_id=str(data["source_id"]),
            kind=EvidenceKind(data["kind"]),
            event_id=str(data["event_id"]) if data.get("event_id") is not None else None,
            dependencies=tuple(str(item) for item in data.get("dependencies", ())),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "kind": self.kind.value,
            "event_id": self.event_id,
            "dependencies": list(self.dependencies),
            "metadata": dict(self.metadata),
        }
