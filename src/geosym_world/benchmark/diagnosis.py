"""Failure attribution for GeoSym-Debug v0."""

from __future__ import annotations

from typing import Any, Dict, List

from geosym_world.schema.problem import GeoSymProblemState
from geosym_world.benchmark.answer_grader import AnswerGrader
from geosym_world.benchmark.output_parser import ModelOutputParser


class DiagnosisEngine:
    def __init__(self):
        self.parser = ModelOutputParser()
        self.grader = AnswerGrader()

    def diagnose(self, state: GeoSymProblemState, model_output: Any) -> Dict[str, Any]:
        parsed = self.parser.parse(model_output)
        gt = state.target.answer if state.target else None
        answer_grade = self.grader.grade(parsed.get("answer"), gt)
        object_errors = self._object_errors(state, parsed.get("referenced_objects") or [])
        claim_errors = self._claim_errors(state, parsed.get("claims") or [])
        components: List[str] = []
        if object_errors:
            components.append("perception_error")
        if claim_errors:
            components.append("topology_error")
        if not answer_grade["correct"] and not object_errors and not claim_errors:
            components.append("computation_error")
        if answer_grade["correct"] and (object_errors or claim_errors):
            components.append("pseudo_reasoning")
        return {
            "correct": answer_grade["correct"],
            "answer_grade": answer_grade,
            "object_errors": object_errors,
            "claim_errors": claim_errors,
            "failure_components": components,
            "parsed_output": parsed,
        }

    def _object_errors(self, state: GeoSymProblemState, refs: list[str]) -> List[Dict[str, str]]:
        point_ids = {p.get("id") for p in state.geometry.points}
        line_ids = {l.get("id") for l in state.geometry.lines}
        entity_ids = {e.get("id") for e in state.geometry.entities}
        known = {item for item in point_ids | line_ids | entity_ids if item}
        errors = []
        for ref in refs:
            if ref not in known:
                # Single-letter refs are often intermediate text, so keep warning
                # severity but still expose the signal.
                errors.append({"object": ref, "error": "unknown_reference"})
        return errors

    def _claim_errors(self, state: GeoSymProblemState, claims: list[dict]) -> List[Dict[str, str]]:
        known_relations = {(r.get("type"), self._relation_args(r)) for r in state.geometry.relations}
        errors = []
        for claim in claims:
            key = (claim.get("type"), self._relation_args(claim))
            if key[0] and key not in known_relations:
                errors.append({"claim": str(claim), "error": "unverified_claim"})
        return errors

    def _relation_args(self, relation: dict) -> tuple[str, ...]:
        args = relation.get("lines") or relation.get("points") or []
        return tuple(sorted(str(item) for item in args))
