"""Degeneracy rejection and duplicate filtering for synthesis."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict

from geosym_world_geometry.builder import ConstructionBuilder
from geosym_world_geometry.solver import FactClosureSolver
from geosym_world_geometry.verifier import TraceVerifier

from .templates import SyntheticSpec


@dataclass
class RejectionReport:
    valid: bool
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class DegeneracyRejector:
    def __init__(self):
        self.verifier = TraceVerifier()
        self.solver = FactClosureSolver()

    def check(self, spec: SyntheticSpec) -> RejectionReport:
        verification = self.verifier.verify_trace(spec.trace)
        if not verification.valid:
            return RejectionReport(False, "trace_verification_failed", verification.to_dict())
        try:
            state = ConstructionBuilder().execute(spec.trace)
            facts = self.solver.solve(state)
        except Exception as exc:  # pragma: no cover - defensive boundary for generator rejection.
            return RejectionReport(False, "execution_failed", {"error": str(exc)})
        if state.target is None:
            return RejectionReport(False, "missing_target")
        target_facts = [fact for fact in facts if fact.type == "target" and fact.object == state.target.object]
        if not target_facts:
            return RejectionReport(False, "missing_target_fact", {"target": state.target.to_dict()})
        target = target_facts[-1]
        if target.value is None:
            return RejectionReport(False, "target_has_no_value", {"target": state.target.to_dict()})
        return RejectionReport(True, metadata={"target_value": str(target.value), "num_facts": len(facts)})


class DuplicateFilter:
    def __init__(self):
        self._seen: set[str] = set()

    def accept(self, spec: SyntheticSpec) -> bool:
        key = stable_spec_hash(spec)
        if key in self._seen:
            return False
        self._seen.add(key)
        return True


def stable_spec_hash(spec: SyntheticSpec) -> str:
    payload = {"family": spec.family, "trace": spec.trace, "question": spec.question}
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]
