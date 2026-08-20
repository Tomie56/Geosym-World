"""Ground and admit executable geometry hypotheses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from geosym_world.construction.builder import ConstructionBuilder
from geosym_world.construction.operations import normalize_trace
from geosym_world.schema import AdmissionStatus, FormalGeometryState, GeometryHypothesis, StateTransition
from geosym_world.verifier.trace import TraceVerifier
from geosym_world.construction.operations import operation_references


@dataclass(frozen=True)
class GroundingReport:
    status: str
    referenced_objects: tuple[str, ...] = ()
    unresolved_objects: tuple[str, ...] = ()
    issues: tuple[dict[str, Any], ...] = ()

    @property
    def valid(self) -> bool:
        return self.status == "grounded"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "valid": self.valid,
            "referenced_objects": list(self.referenced_objects),
            "unresolved_objects": list(self.unresolved_objects),
            "issues": list(self.issues),
        }


@dataclass(frozen=True)
class AdmissionReport:
    decision: str
    hypothesis_id: str
    reason: str
    grounding: GroundingReport
    verification: Mapping[str, Any] = field(default_factory=dict)
    transition: StateTransition | None = None
    state: FormalGeometryState | None = field(default=None, repr=False, compare=False)
    hypothesis: GeometryHypothesis | None = field(default=None, repr=False, compare=False)

    @property
    def admitted(self) -> bool:
        return self.decision == AdmissionStatus.ADMITTED.value

    def to_dict(self) -> dict[str, Any]:
        verification = self.verification.to_dict() if hasattr(self.verification, "to_dict") else dict(self.verification)
        return {
            "decision": self.decision,
            "hypothesis_id": self.hypothesis_id,
            "reason": self.reason,
            "grounding": self.grounding.to_dict(),
            "verification": verification,
            "transition": self.transition.to_dict() if self.transition is not None else None,
        }


class HypothesisAdmission:
    """Validate a proposed trace before allowing it into persistent state."""

    def __init__(self, *, verifier: TraceVerifier | None = None):
        self.verifier = verifier or TraceVerifier()

    def admit(
        self,
        hypothesis: GeometryHypothesis,
        *,
        trace: Iterable[Mapping[str, Any]] | None = None,
        initial_state: FormalGeometryState | None = None,
        confidence_threshold: float = 0.0,
        event_id: str | None = None,
    ) -> AdmissionReport:
        before = initial_state if initial_state is not None else FormalGeometryState()
        event_id = event_id or f"admission:{hypothesis.id}"
        if hypothesis.status is not AdmissionStatus.PROPOSED:
            return self._report(hypothesis, AdmissionStatus.REJECTED, "hypothesis_not_proposed", before, event_id=event_id)
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if hypothesis.confidence < confidence_threshold:
            return self._report(hypothesis, AdmissionStatus.ABSTAINED, "below_confidence_threshold", before, event_id=event_id)
        operations = list(trace) if trace is not None else _trace_from_payload(hypothesis.payload)
        if operations is None or not operations:
            return self._report(hypothesis, AdmissionStatus.ABSTAINED, "missing_executable_trace", before, event_id=event_id)

        grounding = _ground_trace(self.verifier, operations, before)
        if not grounding.valid:
            return self._report(hypothesis, AdmissionStatus.REJECTED, "grounding_failed", before, grounding=grounding, event_id=event_id)
        verification = self.verifier.verify_trace(operations, initial_state=before)
        if not verification.valid:
            return self._report(hypothesis, AdmissionStatus.REJECTED, "verification_failed", before, grounding=grounding, verification=verification, event_id=event_id)

        candidate = ConstructionBuilder(FormalGeometryState.from_dict(before.to_dict())).execute(operations)
        transition = StateTransition.from_states(
            before,
            candidate,
            event_id=event_id,
            provenance={"hypothesis_id": hypothesis.id, "kind": hypothesis.kind},
        )
        return AdmissionReport(
            decision=AdmissionStatus.ADMITTED.value,
            hypothesis_id=hypothesis.id,
            reason="verified",
            grounding=grounding,
            verification=verification,
            transition=transition,
            state=candidate,
            hypothesis=hypothesis.with_status(AdmissionStatus.ADMITTED),
        )

    def _report(self, hypothesis, decision, reason, before, *, event_id, grounding=None, verification=None):
        grounding = grounding or GroundingReport("not_run")
        transition = StateTransition.from_states(before, before, event_id=event_id, status=decision.value, provenance={"hypothesis_id": hypothesis.id})
        return AdmissionReport(
            decision=decision.value,
            hypothesis_id=hypothesis.id,
            reason=reason,
            grounding=grounding,
            verification=verification or {},
            transition=transition,
            hypothesis=hypothesis.with_status(decision),
        )


def _trace_from_payload(payload: Mapping[str, Any]) -> list[Mapping[str, Any]] | None:
    for key in ("construction_trace", "trace"):
        candidate = payload.get(key)
        if isinstance(candidate, list):
            return candidate
    return None


def _ground_trace(verifier: TraceVerifier, trace, initial_state) -> GroundingReport:
    normalized = normalize_trace(trace)
    report = verifier.verify_trace(normalized, execute=False, initial_state=initial_state)
    issues = tuple(issue.to_dict() for issue in report.issues if issue.code == "trace.unknown_reference")
    unresolved = tuple(sorted({str(issue["metadata"].get("ref_id")) for issue in issues}))
    referenced = tuple(sorted({str(ref_id) for op in normalized for _, ref_id in operation_references(op)}))
    return GroundingReport(
        status="failed" if issues else "grounded",
        referenced_objects=referenced,
        unresolved_objects=unresolved,
        issues=issues,
    )
