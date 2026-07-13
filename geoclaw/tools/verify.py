"""Typed claim verification against the exact fact closure."""

from __future__ import annotations

from typing import Any, Dict

import sympy as sp

from ..feedback import INVALID_ARGUMENT, NOT_VERIFIABLE, UNKNOWN_OBJECT, VALID
from .base import ToolResult


RELATION_TYPES = {
    "exists", "point_on_line", "point_on_segment", "point_on_circle", "collinear",
    "parallel", "perpendicular", "midpoint", "right_triangle", "similar_triangle",
    "angle_bisector", "tangent_to_circle",
}
VALUE_TYPES = {"length", "distance", "angle", "area", "perimeter", "volume", "surface_area", "target"}


class VerifyTool:
    name = "verify"

    def execute(self, environment: Any, arguments: Dict[str, Any]) -> ToolResult:
        claim = arguments.get("claim")
        if not isinstance(claim, dict) or not isinstance(claim.get("type"), str):
            return ToolResult(INVALID_ARGUMENT, errors=[{"code": "verify.invalid_claim"}])
        claim_type = claim["type"]
        object_id = claim.get("object")
        objects = claim.get("objects")
        if object_id is None and isinstance(objects, list):
            object_id = "|".join(str(item) for item in objects)
        if claim_type == "equal_length":
            return self._compare_values(environment, claim, "length")
        if claim_type == "equal_angle":
            return self._compare_values(environment, claim, "angle")
        if claim_type == "target_value":
            object_id = environment.state.target.object if environment.state.target else None
            claim_type = "target"
        if claim_type not in RELATION_TYPES | VALUE_TYPES:
            return ToolResult(NOT_VERIFIABLE, result={"truth": "unknown"}, errors=[{"code": "verify.unsupported_claim", "type": claim_type}])
        if not object_id:
            return ToolResult(INVALID_ARGUMENT, errors=[{"code": "verify.missing_object"}])
        if isinstance(object_id, str) and "|" not in object_id:
            resolution = environment.resolver.resolve(object_id, environment.state)
            if resolution.status == "resolved":
                object_id = resolution.object_id
                claim = {**claim, "object": object_id}
        matches = environment.facts.find(claim_type, str(object_id))
        if not matches and claim_type in RELATION_TYPES:
            matches = self._unordered_matches(environment, claim_type, objects)
        if not matches:
            if self._references_unknown(environment, claim):
                return ToolResult(UNKNOWN_OBJECT, errors=[{"code": "verify.unknown_object"}])
            return ToolResult(VALID, result={"truth": "unknown", "claim": claim, "evidence_strength": "unknown"}, evidence=[])
        expected = claim.get("value")
        if expected is None:
            return ToolResult(VALID, result={"truth": "true", "claim": claim, "evidence_strength": _evidence_strength(matches)}, evidence=[item.to_dict() for item in matches])
        for fact in matches:
            if fact.value is not None and _equal(fact.value, expected, environment.task.numeric_tolerance):
                return ToolResult(VALID, result={"truth": "true", "claim": claim, "evidence_strength": _evidence_strength([fact])}, evidence=[fact.to_dict()])
        valued = [fact.to_dict() for fact in matches if fact.value is not None]
        truth = "false" if valued else "unknown"
        return ToolResult(VALID, result={"truth": truth, "claim": claim, "evidence_strength": "counterexample_found" if truth == "false" else "unknown"}, evidence=valued)

    def _compare_values(self, environment: Any, claim: Dict[str, Any], fact_type: str) -> ToolResult:
        objects = claim.get("objects")
        if not isinstance(objects, list) or len(objects) != 2:
            return ToolResult(INVALID_ARGUMENT, errors=[{"code": "verify.requires_two_objects"}])
        left = environment.facts.find(fact_type, str(objects[0]))
        right = environment.facts.find(fact_type, str(objects[1]))
        if not left or not right:
            return ToolResult(VALID, result={"truth": "unknown", "claim": claim, "evidence_strength": "unknown"})
        truth = "true" if any(_equal(a.value, b.value, environment.task.numeric_tolerance) for a in left for b in right if a.value is not None and b.value is not None) else "false"
        return ToolResult(VALID, result={"truth": truth, "claim": claim, "evidence_strength": "exact_coordinate" if truth == "true" else "counterexample_found"}, evidence=[left[0].to_dict(), right[0].to_dict()])

    @staticmethod
    def _unordered_matches(environment: Any, claim_type: str, objects: Any):
        if not isinstance(objects, list):
            return []
        wanted = {str(item) for item in objects}
        return [fact for fact in environment.facts.find(claim_type) if set(fact.object.split("|")) == wanted]

    @staticmethod
    def _references_unknown(environment: Any, claim: Dict[str, Any]) -> bool:
        refs = claim.get("objects", [claim.get("object")])
        return any(ref is not None and environment.resolver.resolve(str(ref), environment.state).status == "unknown" for ref in refs)


def _equal(left: Any, right: Any, tolerance: float) -> bool:
    try:
        residual = sp.simplify(sp.sympify(left) - sp.sympify(right))
        return residual == 0 or abs(float(sp.N(residual))) <= tolerance
    except Exception:
        return str(left).strip() == str(right).strip()


def _evidence_strength(facts) -> str:
    rules = {fact.rule for fact in facts}
    if any(rule in {"target_lookup", "construction_trace", "state_registry"} for rule in rules):
        return "solver_derived"
    return "exact_coordinate"
