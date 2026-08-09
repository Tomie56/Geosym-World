"""Degeneracy rejection and duplicate filtering for synthesis."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict

import sympy as sp

from geosym_world.construction.builder import ConstructionBuilder
from geosym_world.solver.closure import FactClosureSolver
from geosym_world.verifier.trace import TraceVerifier

from .templates import SyntheticSpec


@dataclass
class RejectionReport:
    valid: bool
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class DegeneracyRejector:
    def __init__(self, *, solver_profile: str = "auto"):
        self.verifier = TraceVerifier()
        self.solver = FactClosureSolver(profile=solver_profile)

    def check(self, spec: SyntheticSpec) -> RejectionReport:
        verification = self.verifier.verify_trace(spec.trace, execute=False)
        if not verification.valid:
            return RejectionReport(False, "trace_verification_failed", verification.to_dict())
        try:
            state = ConstructionBuilder().execute(spec.trace)
            self.verifier.verify_state(state, verification)
            if not verification.valid:
                return RejectionReport(False, "state_verification_failed", verification.to_dict())
            coincident = _coincident_points(state)
            if coincident is not None:
                return RejectionReport(False, "coincident_points", coincident)
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


def _coincident_points(state) -> Dict[str, str] | None:
    point_ids = sorted(state.points)
    for index, point_id in enumerate(point_ids):
        point = state.points[point_id]
        for other_id in point_ids[index + 1 :]:
            other = state.points[other_id]
            if other.dim != point.dim:
                continue
            if all(sp.simplify(left - right) == 0 for left, right in zip(point.coords, other.coords)):
                return {"point": point_id, "coincident_with": other_id}
    return None
