"""Coverage of build_plan_seed expected instructions by VLM output."""

from __future__ import annotations

from typing import Any, Dict, List
import json


class BuildPlanSeedCoverageAnalyzer:
    """Check whether stage-2 emitted instructions from build_plan_seed."""

    def analyze(self, *, build_plan_seed: Dict[str, Any], semantic: Dict[str, Any]) -> Dict[str, Any]:
        expected = self._expected(build_plan_seed)
        actual = [item for item in (semantic.get("build_instructions") or []) if isinstance(item, dict)]
        covered: List[Dict[str, Any]] = []
        missing: List[Dict[str, Any]] = []
        for item in expected:
            if any(self._compatible(actual_item, item["instruction"]) for actual_item in actual):
                covered.append(item)
            else:
                missing.append(item)
        return {
            "version": "build-plan-seed-coverage-v0",
            "num_expected_instructions": len(expected),
            "num_covered_instructions": len(covered),
            "num_missing_instructions": len(missing),
            "coverage": (len(covered) / len(expected)) if expected else 1.0,
            "covered": covered[:200],
            "missing": missing[:200],
        }

    def _expected(self, build_plan_seed: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for task in build_plan_seed.get("tasks") or []:
            if not isinstance(task, dict):
                continue
            task_id = str(task.get("id") or "")
            relation_id = str(task.get("from_relation") or "")
            for idx, instruction in enumerate(task.get("expected_instructions") or []):
                if not isinstance(instruction, dict) or not instruction.get("action"):
                    continue
                out.append({
                    "task_id": task_id,
                    "from_relation": relation_id,
                    "index": idx,
                    "action": str(instruction.get("action") or ""),
                    "instruction": dict(instruction),
                })
        return out

    def _compatible(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        if str(actual.get("action") or "") != str(expected.get("action") or ""):
            return False
        for key, expected_value in expected.items():
            if key in {"supported_by_current_builder"}:
                continue
            if key not in actual:
                continue
            if not self._same(actual.get(key), expected_value):
                return False
        return True

    def _same(self, actual: Any, expected: Any) -> bool:
        return json.dumps(actual, sort_keys=True, ensure_ascii=False) == json.dumps(expected, sort_keys=True, ensure_ascii=False)
