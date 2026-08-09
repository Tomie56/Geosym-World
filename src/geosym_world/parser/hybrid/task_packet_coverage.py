"""Coverage analysis for instruction task packets."""

from __future__ import annotations

from typing import Any, Dict, List, Set
import json


class InstructionTaskPacketCoverageAnalyzer:
    """Check whether build instructions cover packet required actions."""

    def analyze(self, *, packet: Dict[str, Any], semantic: Dict[str, Any]) -> Dict[str, Any]:
        instructions = semantic.get("build_instructions") or []
        required_tasks = packet.get("required_tasks") or []
        task_reports = []
        covered_tasks = []
        missing_tasks = []
        covered_requirement_count = 0
        missing_requirement_count = 0

        for task in required_tasks:
            cid = str(task.get("candidate_id") or "")
            required_actions = [str(item) for item in task.get("required_actions") or [] if item]
            requirements = self._requirements_for_task(task)
            if not cid or (not required_actions and not requirements):
                continue
            covered_actions = self._covered_actions_for_candidate(cid, instructions)
            requirement_reports = [
                self._requirement_report(requirement, instructions)
                for requirement in requirements
            ]
            covered_requirements = [item for item in requirement_reports if item.get("covered")]
            missing_requirements = [item for item in requirement_reports if not item.get("covered")]
            missing_actions = self._missing_actions(required_actions, covered_actions, missing_requirements)
            covered_requirement_count += len(covered_requirements)
            missing_requirement_count += len(missing_requirements)
            item = {
                "candidate_id": cid,
                "candidate_type": task.get("candidate_type"),
                "decision": task.get("decision"),
                "required_actions": required_actions,
                "covered_actions": sorted(covered_actions),
                "missing_actions": missing_actions,
                "num_instruction_requirements": len(requirement_reports),
                "num_covered_instruction_requirements": len(covered_requirements),
                "num_missing_instruction_requirements": len(missing_requirements),
                "instruction_requirements": requirement_reports[:50],
            }
            task_reports.append(item)
            if missing_actions or missing_requirements:
                missing_tasks.append(item)
            else:
                covered_tasks.append(item)

        return {
            "num_required_tasks": len(task_reports),
            "num_covered_tasks": len(covered_tasks),
            "num_missing_tasks": len(missing_tasks),
            "num_required_actions": sum(len(item.get("required_actions") or []) for item in task_reports),
            "num_covered_actions": sum(len(item.get("covered_actions") or []) for item in task_reports),
            "num_missing_actions": sum(len(item.get("missing_actions") or []) for item in task_reports),
            "num_instruction_requirements": covered_requirement_count + missing_requirement_count,
            "num_covered_instruction_requirements": covered_requirement_count,
            "num_missing_instruction_requirements": missing_requirement_count,
            "coverage": (len(covered_tasks) / len(task_reports)) if task_reports else None,
            "task_reports": task_reports[:100],
            "missing_tasks": missing_tasks[:100],
        }

    def _covered_actions_for_candidate(self, candidate_id: str, instructions: List[Dict[str, Any]]) -> Set[str]:
        covered: Set[str] = set()
        for instruction in instructions:
            if self._refs_candidate(instruction, candidate_id):
                action = str(instruction.get("action") or "")
                if action:
                    covered.add(action)
        return covered

    def _refs_candidate(self, item: Dict[str, Any], candidate_id: str) -> bool:
        if not isinstance(item, dict):
            return False
        for key in ["candidate_id", "from_candidate"]:
            if str(item.get(key) or "") == candidate_id:
                return True
        for key in ["candidate_ids", "from_candidates"]:
            value = item.get(key)
            if isinstance(value, list) and candidate_id in {str(v) for v in value}:
                return True
            if value and str(value) == candidate_id:
                return True
        target = item.get("target")
        if isinstance(target, dict):
            return self._refs_candidate(target, candidate_id)
        return False

    def _requirements_for_task(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        requirements = [item for item in task.get("instruction_requirements") or [] if isinstance(item, dict)]
        if requirements:
            return requirements
        out = []
        for idx, suggestion in enumerate(task.get("suggested_instructions") or []):
            if not isinstance(suggestion, dict):
                continue
            action = str(suggestion.get("action") or "")
            if action in {str(item) for item in task.get("required_actions") or []}:
                out.append({
                    "requirement_id": f"{task.get('candidate_id')}:{idx}:{action}",
                    "action": action,
                    "expected_instruction": dict(suggestion),
                    "match_rule": {
                        "action": action,
                        "from_candidate": task.get("candidate_id"),
                        "field_values": {
                            key: suggestion.get(key)
                            for key in ["id", "points", "lines", "value", "target"]
                            if key in suggestion
                        },
                    },
                })
        return out

    def _requirement_report(self, requirement: Dict[str, Any], instructions: List[Dict[str, Any]]) -> Dict[str, Any]:
        match_rule = requirement.get("match_rule") or {}
        matches = [
            idx for idx, instruction in enumerate(instructions)
            if self._matches_requirement(instruction, match_rule)
        ]
        return {
            "requirement_id": requirement.get("requirement_id"),
            "action": requirement.get("action") or match_rule.get("action"),
            "expected_instruction": requirement.get("expected_instruction") or {},
            "covered": bool(matches),
            "matched_instruction_indices": matches[:10],
        }

    def _matches_requirement(self, instruction: Dict[str, Any], match_rule: Dict[str, Any]) -> bool:
        action = str(match_rule.get("action") or "")
        if action and str(instruction.get("action") or "") != action:
            return False
        cid = str(match_rule.get("from_candidate") or "")
        if cid and not self._refs_candidate(instruction, cid):
            return False
        for key, expected in (match_rule.get("field_values") or {}).items():
            if key not in instruction:
                return False
            if not self._same_value(instruction.get(key), expected):
                return False
        return True

    def _missing_actions(
        self,
        required_actions: List[str],
        covered_actions: Set[str],
        missing_requirements: List[Dict[str, Any]],
    ) -> List[str]:
        missing = set(required_actions) - covered_actions
        missing.update(
            str(item.get("action") or "")
            for item in missing_requirements
            if item.get("action")
        )
        return sorted(missing)

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
        return self._canonical(actual) == self._canonical(expected)

    def _canonical(self, value: Any) -> str:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
