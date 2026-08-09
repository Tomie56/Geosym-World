"""Classify where a hybrid real-data parse is currently blocked."""

from __future__ import annotations

from typing import Any, Dict


class HybridStageClassifier:
    def classify(
        self,
        *,
        visual_state: Dict[str, Any],
        semantic: Dict[str, Any],
        grounded: Dict[str, Any],
        formal: Dict[str, Any],
        instruction_plan: Dict[str, Any],
        verification: Dict[str, Any],
        image_alignment: Dict[str, Any] | None = None,
        consistency: Dict[str, Any] | None = None,
        contract_audit: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        metrics = verification.get("metrics") or {}
        issues = verification.get("issues") or []
        flags = {
            "visual_extraction_failed": bool(visual_state.get("error")) or (
                metrics.get("visual_lines", 0) == 0 and metrics.get("visual_circles", 0) == 0
            ),
            "semantic_schema_warning": metrics.get("semantic_schema_warnings", 0) > 0,
            "semantic_empty": not (semantic.get("relations") or semantic.get("build_instructions")),
            "relation_grounding_failed": bool(semantic.get("relations")) and metrics.get("grounded_accepted", 0) == 0,
            "instruction_rejected": metrics.get("build_instructions_rejected", 0) > 0,
            "instruction_synthesized": metrics.get("build_instructions_synthesized", 0) > 0,
            "contract_failed": isinstance(contract_audit, dict) and contract_audit.get("valid") is False,
            "contract_warning": isinstance(contract_audit, dict) and contract_audit.get("num_warnings", 0) > 0,
            "instruction_not_buildable": not instruction_plan.get("buildable"),
            "theorem_only": formal.get("mode") == "theorem_only",
            "unsupported_formal": formal.get("mode") in {"unsupported", "failed"},
            "canonical_build_failed": any(issue.get("code") == "geometry.build_failed" for issue in issues),
            "verification_failed": not bool(verification.get("valid")),
            "image_alignment_failed": isinstance(image_alignment, dict) and not image_alignment.get("valid", False),
            "answer_consistency_failed": isinstance(consistency, dict) and not consistency.get("valid", False),
        }
        primary = self._primary_stage(flags)
        return {
            "primary_stage": primary,
            "stage_flags": flags,
        }

    def _primary_stage(self, flags: Dict[str, bool]) -> str:
        ordered = [
            ("visual_extraction_failed", "visual_extraction"),
            ("semantic_empty", "semantic_extraction"),
            ("theorem_only", "theorem_solver_missing"),
            ("relation_grounding_failed", "relation_grounding"),
            ("unsupported_formal", "formalization"),
            ("contract_failed", "extraction_contract"),
            ("instruction_not_buildable", "instruction_planning"),
            ("canonical_build_failed", "canonical_build"),
            ("answer_consistency_failed", "answer_consistency"),
            ("image_alignment_failed", "image_alignment"),
            ("verification_failed", "verification"),
            ("instruction_rejected", "instruction_grounding_warning"),
        ]
        for flag, stage in ordered:
            if flags.get(flag):
                return stage
        return "accepted"
