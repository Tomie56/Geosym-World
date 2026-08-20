"""Bounded candidate diagnostics for synthetic generation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping


def accepted_validation_metadata(report: Any) -> dict[str, Any]:
    return {
        "status": "accepted",
        "reason": getattr(report, "reason", None) or "validated",
        **dict(getattr(report, "metadata", {}) or {}),
    }


@dataclass
class CandidateDiagnostics:
    max_examples_per_reason: int = 3
    checked: int = 0
    accepted: int = 0
    rejection_counts: Counter[str] = field(default_factory=Counter)
    rejection_examples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    skip_counts: Counter[str] = field(default_factory=Counter)
    skip_examples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    @property
    def rejected(self) -> int:
        return sum(self.rejection_counts.values())

    @property
    def skipped(self) -> int:
        return sum(self.skip_counts.values())

    @property
    def considered(self) -> int:
        return self.checked + self.skipped

    def record_accept(self) -> None:
        self.checked += 1
        self.accepted += 1

    def record_rejection(
        self,
        reason: str,
        *,
        candidate_id: str,
        family: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.checked += 1
        self.rejection_counts[reason] += 1
        examples = self.rejection_examples.setdefault(reason, [])
        if len(examples) < self.max_examples_per_reason:
            examples.append({
                "candidate_id": candidate_id,
                "family": family,
                "metadata": dict(metadata or {}),
            })

    def record_skip(
        self,
        reason: str,
        *,
        candidate_id: str,
        family: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.skip_counts[reason] += 1
        examples = self.skip_examples.setdefault(reason, [])
        if len(examples) < self.max_examples_per_reason:
            examples.append({
                "candidate_id": candidate_id,
                "family": family,
                "metadata": dict(metadata or {}),
            })

    def to_dict(self) -> dict[str, Any]:
        return {
            "considered": self.considered,
            "checked": self.checked,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "skipped": self.skipped,
            "rejection_counts": dict(sorted(self.rejection_counts.items())),
            "rejection_examples": {key: list(value) for key, value in sorted(self.rejection_examples.items())},
            "skip_counts": dict(sorted(self.skip_counts.items())),
            "skip_examples": {key: list(value) for key, value in sorted(self.skip_examples.items())},
        }
