"""Serializable state transitions used by construction and admission."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping

from .canonical import state_checksum


@dataclass(frozen=True)
class StateTransition:
    event_id: str
    status: str
    before_checksum: str
    after_checksum: str
    changed: bool
    added_objects: tuple[str, ...] = ()
    removed_objects: tuple[str, ...] = ()
    added_relation_ids: tuple[str, ...] = ()
    removed_relation_ids: tuple[str, ...] = ()
    target_changed: bool = False
    trace_length_before: int = 0
    trace_length_after: int = 0
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id.strip():
            raise ValueError("State transition event_id must be non-empty")
        if self.status not in {"committed", "rejected", "abstained", "proposed"}:
            raise ValueError(f"Unknown state transition status: {self.status}")
        for name, checksum in (("before_checksum", self.before_checksum), ("after_checksum", self.after_checksum)):
            if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
                raise ValueError(f"State transition {name} must be a SHA-256 checksum")
        if self.trace_length_before < 0 or self.trace_length_after < 0:
            raise ValueError("State transition trace lengths cannot be negative")

    @classmethod
    def from_states(
        cls,
        before: Any,
        after: Any,
        *,
        event_id: str,
        status: str = "committed",
        provenance: Mapping[str, Any] | None = None,
    ) -> "StateTransition":
        before_ids = set(before.object_ids())
        after_ids = set(after.object_ids())
        before_relations = {str(item["id"]): item for item in before.relations}
        after_relations = {str(item["id"]): item for item in after.relations}
        before_digest = state_checksum(before)
        after_digest = state_checksum(after)
        return cls(
            event_id=event_id,
            status=status,
            before_checksum=before_digest,
            after_checksum=after_digest,
            changed=before_digest != after_digest,
            added_objects=tuple(sorted(after_ids - before_ids)),
            removed_objects=tuple(sorted(before_ids - after_ids)),
            added_relation_ids=tuple(sorted(set(after_relations) - set(before_relations))),
            removed_relation_ids=tuple(sorted(set(before_relations) - set(after_relations))),
            target_changed=before.target != after.target,
            trace_length_before=len(before.construction_trace),
            trace_length_after=len(after.construction_trace),
            provenance=dict(provenance or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "status": self.status,
            "before_checksum": self.before_checksum,
            "after_checksum": self.after_checksum,
            "changed": self.changed,
            "added_objects": list(self.added_objects),
            "removed_objects": list(self.removed_objects),
            "added_relation_ids": list(self.added_relation_ids),
            "removed_relation_ids": list(self.removed_relation_ids),
            "target_changed": self.target_changed,
            "trace_length_before": self.trace_length_before,
            "trace_length_after": self.trace_length_after,
            "provenance": dict(self.provenance),
        }
