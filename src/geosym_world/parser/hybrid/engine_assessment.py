"""Validate VLM self-assessment of GeoSym engine buildability."""

from __future__ import annotations

from typing import Any, Dict


class EngineAssessmentValidator:
    VALID_MODES = {"coordinate_buildable", "theorem_only", "unsupported", "failed"}

    def validate(self, semantic: Dict[str, Any], formal: Dict[str, Any], instruction_plan: Dict[str, Any]) -> Dict[str, Any]:
        assessment = dict(semantic.get("engine_assessment") or {})
        expected = str(assessment.get("expected_mode") or "").strip()
        actual = self._actual_mode(formal, instruction_plan)
        if not expected:
            return {
                "provided": False,
                "expected_mode": None,
                "actual_mode": actual,
                "consistent": None,
                "reason": "VLM did not provide engine_assessment.expected_mode",
            }
        if expected not in self.VALID_MODES:
            return {
                "provided": True,
                "expected_mode": expected,
                "actual_mode": actual,
                "consistent": False,
                "reason": f"Unknown expected_mode: {expected}",
                "vlm_reason": assessment.get("reason", ""),
            }
        return {
            "provided": True,
            "expected_mode": expected,
            "actual_mode": actual,
            "consistent": expected == actual,
            "reason": "expected_mode matches actual_mode" if expected == actual else "expected_mode differs from actual_mode",
            "vlm_reason": assessment.get("reason", ""),
        }

    def _actual_mode(self, formal: Dict[str, Any], instruction_plan: Dict[str, Any]) -> str:
        if instruction_plan.get("buildable"):
            return "coordinate_buildable"
        mode = formal.get("mode")
        if mode in self.VALID_MODES:
            return str(mode)
        return "unsupported"
