"""Bounded candidate diagnostics for synthetic generation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping


DIAGNOSTIC_SCHEMA_VERSION = "1.0"
_REJECTION_CODES = {
    "candidate_validation_failed": ("validation.candidate.failed", "validation", "candidate"),
    "coincident_points": ("geometry.point.coincident", "geometry", "point"),
    "difficulty_out_of_range": ("policy.difficulty.out_of_range", "policy", "difficulty"),
    "dead_enhancement_operations": ("synthesis.task_graph.dead_operation", "synthesis", "task_graph"),
    "duplicate_content": ("policy.duplicate.content", "policy", "duplicate"),
    "duplicate_id": ("policy.duplicate.id", "policy", "duplicate"),
    "execution_failed": ("execution.internal.unexpected", "execution", "internal"),
    "invalid_derivation": ("derivation.graph.invalid", "derivation", "graph"),
    "invalid_solution_paths": ("solution.graph.invalid", "solution", "graph"),
    "missing_target": ("solver.target.missing_contract", "solver", "target"),
    "missing_target_fact": ("solver.target.unresolved", "solver", "target"),
    "nonzero_verification_residual": ("verification.residual.nonzero", "verification", "residual"),
    "state_verification_failed": ("verification.state.invalid", "verification", "state"),
    "target_has_no_value": ("solver.target.no_value", "solver", "target"),
    "trace_verification_failed": ("verification.trace.invalid", "verification", "trace"),
}


def rejection_diagnostic(reason: str, metadata: Mapping[str, Any] | None = None) -> dict[str, str]:
    supplied = dict((metadata or {}).get("diagnostic") or {})
    if supplied:
        return {
            "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
            "code": str(supplied["code"]),
            "stage": str(supplied["stage"]),
            "component": str(supplied["component"]),
        }
    code, stage, component = _REJECTION_CODES.get(reason, (f"candidate.{reason}", "candidate", "unknown"))
    return {"schema_version": DIAGNOSTIC_SCHEMA_VERSION, "code": code, "stage": stage, "component": component}


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
    rejection_code_counts: Counter[str] = field(default_factory=Counter)
    rejection_stage_counts: Counter[str] = field(default_factory=Counter)
    rejection_component_counts: Counter[str] = field(default_factory=Counter)
    rejection_family_counts: Counter[str] = field(default_factory=Counter)
    rejection_examples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    skip_counts: Counter[str] = field(default_factory=Counter)
    skip_examples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    preflight_checked: int = 0
    preflight_accepted: int = 0
    preflight_skip_counts: Counter[str] = field(default_factory=Counter)

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
        diagnostic = rejection_diagnostic(reason, metadata)
        self.rejection_code_counts[diagnostic["code"]] += 1
        self.rejection_stage_counts[diagnostic["stage"]] += 1
        self.rejection_component_counts[diagnostic["component"]] += 1
        self.rejection_family_counts[family] += 1
        examples = self.rejection_examples.setdefault(reason, [])
        if len(examples) < self.max_examples_per_reason:
            examples.append({
                "candidate_id": candidate_id,
                "family": family,
                "diagnostic": diagnostic,
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

    def record_preflight(self, payload: Mapping[str, Any] | None) -> None:
        data = dict(payload or {})
        self.preflight_checked += int(data.get("checked", 0))
        self.preflight_accepted += int(data.get("accepted", 0))
        self.preflight_skip_counts.update({str(key): int(value) for key, value in (data.get("skip_counts") or {}).items()})

    def to_dict(self) -> dict[str, Any]:
        return {
            "considered": self.considered,
            "checked": self.checked,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "skipped": self.skipped,
            "rejection_counts": dict(sorted(self.rejection_counts.items())),
            "rejection_code_counts": dict(sorted(self.rejection_code_counts.items())),
            "rejection_stage_counts": dict(sorted(self.rejection_stage_counts.items())),
            "rejection_component_counts": dict(sorted(self.rejection_component_counts.items())),
            "rejection_family_counts": dict(sorted(self.rejection_family_counts.items())),
            "rejection_examples": {key: list(value) for key, value in sorted(self.rejection_examples.items())},
            "skip_counts": dict(sorted(self.skip_counts.items())),
            "skip_examples": {key: list(value) for key, value in sorted(self.skip_examples.items())},
            "preflight": {
                "checked": self.preflight_checked,
                "accepted": self.preflight_accepted,
                "skipped": self.preflight_checked - self.preflight_accepted,
                "skip_counts": dict(sorted(self.preflight_skip_counts.items())),
            },
        }
