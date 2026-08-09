"""Compare semantic build instructions with a task-packet response template."""

from __future__ import annotations

from typing import Any, Dict, List
import json


class ResponseTemplateAdherenceAnalyzer:
    """Measure whether stage-2 instructions follow response_template."""

    def analyze(self, *, packet: Dict[str, Any], semantic: Dict[str, Any]) -> Dict[str, Any]:
        template = packet.get("response_template") or {}
        expected = [item for item in template.get("build_instructions") or [] if isinstance(item, dict)]
        actual = [item for item in semantic.get("build_instructions") or [] if isinstance(item, dict)]
        expected_task_decisions = [
            item for item in template.get("instruction_task_decisions") or packet.get("instruction_task_decision_template") or []
            if isinstance(item, dict) and item.get("candidate_id")
        ]
        actual_task_decisions = [
            item for item in semantic.get("instruction_task_decisions") or []
            if isinstance(item, dict) and item.get("candidate_id")
        ]
        expected_reports = []
        matched_actual_indices: set[int] = set()

        for idx, expected_inst in enumerate(expected):
            matches = [
                actual_idx for actual_idx, actual_inst in enumerate(actual)
                if self._instruction_matches(actual_inst, expected_inst)
            ]
            for actual_idx in matches:
                matched_actual_indices.add(actual_idx)
            expected_reports.append({
                "template_index": idx,
                "action": expected_inst.get("action"),
                "expected_instruction": expected_inst,
                "matched": bool(matches),
                "matched_instruction_indices": matches[:10],
            })

        missing = [item for item in expected_reports if not item.get("matched")]
        extra = [
            {
                "instruction_index": idx,
                "instruction": instruction,
            }
            for idx, instruction in enumerate(actual)
            if idx not in matched_actual_indices
        ]
        actual_task_decision_ids = {str(item.get("candidate_id")) for item in actual_task_decisions}
        expected_task_decision_ids = {str(item.get("candidate_id")) for item in expected_task_decisions}
        task_decision_reports = [
            {
                "template_index": idx,
                "candidate_id": str(item.get("candidate_id")),
                "expected_decision": item,
                "matched": str(item.get("candidate_id")) in actual_task_decision_ids,
                "actual_decisions": [
                    actual for actual in actual_task_decisions
                    if str(actual.get("candidate_id")) == str(item.get("candidate_id"))
                ][:5],
            }
            for idx, item in enumerate(expected_task_decisions)
        ]
        missing_task_decisions = [item for item in task_decision_reports if not item.get("matched")]
        extra_task_decisions = [
            {
                "candidate_id": str(item.get("candidate_id")),
                "decision": item,
            }
            for item in actual_task_decisions
            if str(item.get("candidate_id")) not in expected_task_decision_ids
        ]
        num_expected = len(expected)
        num_matched = len(expected_reports) - len(missing)
        num_expected_task_decisions = len(expected_task_decisions)
        num_matched_task_decisions = len(task_decision_reports) - len(missing_task_decisions)
        return {
            "num_template_instructions": num_expected,
            "num_actual_instructions": len(actual),
            "num_matched_template_instructions": num_matched,
            "num_missing_template_instructions": len(missing),
            "num_extra_instructions": len(extra),
            "num_template_instruction_task_decisions": num_expected_task_decisions,
            "num_actual_instruction_task_decisions": len(actual_task_decisions),
            "num_matched_template_instruction_task_decisions": num_matched_task_decisions,
            "num_missing_template_instruction_task_decisions": len(missing_task_decisions),
            "num_extra_instruction_task_decisions": len(extra_task_decisions),
            "template_instruction_task_decision_adherence": (num_matched_task_decisions / num_expected_task_decisions) if num_expected_task_decisions else None,
            "template_adherence": (num_matched / num_expected) if num_expected else None,
            "missing_template_instructions": missing[:100],
            "extra_instructions": extra[:100],
            "missing_template_instruction_task_decisions": missing_task_decisions[:100],
            "extra_instruction_task_decisions": extra_task_decisions[:100],
        }

    def _instruction_matches(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        if str(actual.get("action") or "") != str(expected.get("action") or ""):
            return False
        if str(expected.get("action") or "") == "set_target":
            return self._target_matches(actual.get("target") or {}, expected.get("target") or {}) and self._candidate_ref_matches(actual, expected)
        for key, value in expected.items():
            if key == "supported_by_current_builder":
                continue
            if key not in actual:
                return False
            if not self._same_value(actual.get(key), value):
                return False
        return True

    def _candidate_ref_matches(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        expected_ref = str(expected.get("from_candidate") or expected.get("candidate_id") or "")
        if not expected_ref:
            return True
        refs = {
            str(actual.get("from_candidate") or ""),
            str(actual.get("candidate_id") or ""),
        }
        target = actual.get("target")
        if isinstance(target, dict):
            refs.add(str(target.get("from_candidate") or ""))
            refs.add(str(target.get("candidate_id") or ""))
        return expected_ref in refs

    def _target_matches(self, actual: Any, expected: Any) -> bool:
        if not isinstance(actual, dict) or not isinstance(expected, dict):
            return False
        for key in ["type", "object", "value_symbol"]:
            if key in expected and str(actual.get(key) or "") != str(expected.get(key) or ""):
                return False
        return True

    def _same_value(self, actual: Any, expected: Any) -> bool:
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                return False
            return all(
                key in actual and self._same_value(actual.get(key), value)
                for key, value in expected.items()
            )
        if isinstance(expected, list):
            if not isinstance(actual, list) or len(actual) != len(expected):
                return False
            return all(self._same_value(a, e) for a, e in zip(actual, expected))
        return json.dumps(actual, sort_keys=True, ensure_ascii=False) == json.dumps(expected, sort_keys=True, ensure_ascii=False)
