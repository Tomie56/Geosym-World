"""Adapt stage-2 VLM instructions to the task-packet builder contract."""

from __future__ import annotations

from typing import Any, Dict, List
import json


class InstructionPassAdapter:
    """Attach task-packet candidate refs and exact fields to VLM instructions.

    The adapter only patches instructions that uniquely match an expected
    instruction from `instruction_task_packet.required_tasks`. It preserves raw
    instruction output for diagnostics and does not create missing instructions;
    missing tasks are still handled by downstream repair/coverage analysis.
    """

    def adapt(
        self,
        *,
        instruction_pass: Dict[str, Any],
        instruction_task_packet: Dict[str, Any],
        build_plan_seed: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        adapted = dict(instruction_pass or {})
        expected = self._expected_instructions(instruction_task_packet)
        expected.extend(self._expected_seed_instructions(build_plan_seed or {}))
        expected = self._dedupe_expected(expected)
        patches: List[Dict[str, Any]] = []
        adapted_instructions = []

        for idx, instruction in enumerate(adapted.get("build_instructions") or []):
            if not isinstance(instruction, dict):
                continue
            patched = dict(instruction)
            match = self._unique_match(patched, expected)
            fields: List[str] = []
            if match:
                expected_instruction = match["instruction"]
                cid = str(match.get("candidate_id") or "")
                if cid and not self._refs_candidate(patched, cid):
                    patched["from_candidate"] = cid
                    fields.append("from_candidate")
                for key, value in expected_instruction.items():
                    if key == "supported_by_current_builder":
                        continue
                    if key not in patched or self._is_missing(patched.get(key)):
                        patched[key] = self._copy_value(value)
                        fields.append(key)
                    elif isinstance(patched.get(key), dict) and isinstance(value, dict):
                        merged, changed = self._merge_missing_dict(patched[key], value)
                        if changed:
                            patched[key] = merged
                            fields.append(key)
                if fields:
                    patched["adapter_source"] = "instruction_pass_adapter"
                    patches.append({
                        "index": idx,
                        "candidate_id": cid,
                        "action": patched.get("action"),
                        "fields": fields,
                        "requirement_id": match.get("requirement_id"),
                    })
            adapted_instructions.append(patched)
        adapted["build_instructions"] = adapted_instructions

        metadata = dict(adapted.get("metadata") or {})
        metadata["instruction_pass_adapter"] = {
            "used": bool(patches),
            "num_instruction_patches": len(patches),
            "num_from_candidate_patches": self._count_field(patches, "from_candidate"),
            "num_field_patches": sum(len(item.get("fields") or []) for item in patches),
            "num_expected_items": len(expected),
            "patches": patches[:100],
        }
        adapted["metadata"] = metadata
        return adapted

    def _expected_instructions(self, packet: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for task in packet.get("required_tasks") or []:
            if not isinstance(task, dict):
                continue
            cid = str(task.get("candidate_id") or "")
            for requirement in task.get("instruction_requirements") or []:
                if not isinstance(requirement, dict):
                    continue
                expected = requirement.get("expected_instruction")
                if not isinstance(expected, dict) or not expected.get("action"):
                    continue
                out.append({
                    "candidate_id": cid,
                    "requirement_id": requirement.get("requirement_id"),
                    "instruction": dict(expected),
                })
        return out

    def _expected_seed_instructions(self, build_plan_seed: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for task in build_plan_seed.get("tasks") or []:
            if not isinstance(task, dict):
                continue
            cid = str(task.get("from_relation") or task.get("id") or "")
            task_id = str(task.get("id") or cid)
            for idx, expected in enumerate(task.get("expected_instructions") or []):
                if not isinstance(expected, dict) or not expected.get("action"):
                    continue
                out.append({
                    "candidate_id": cid,
                    "requirement_id": f"build_plan_seed:{task_id}:{idx}:{expected.get('action')}",
                    "instruction": dict(expected),
                })
        return out

    def _dedupe_expected(self, expected_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen = set()
        for item in expected_items:
            instruction = item.get("instruction")
            if not isinstance(instruction, dict):
                continue
            key = self._canonical(instruction)
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    def _unique_match(self, instruction: Dict[str, Any], expected_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        matches = [
            item for item in expected_items
            if self._compatible_instruction(instruction, item.get("instruction") or {})
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return self._merge_compatible_matches(matches)
        return {}

    def _merge_compatible_matches(self, matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_instruction: Dict[str, Any] = {}
        candidate_ids: List[str] = []
        requirement_ids: List[str] = []
        for item in matches:
            cid = str(item.get("candidate_id") or "")
            if cid and cid not in candidate_ids:
                candidate_ids.append(cid)
            rid = str(item.get("requirement_id") or "")
            if rid:
                requirement_ids.append(rid)
            instruction = item.get("instruction") or {}
            if not isinstance(instruction, dict):
                continue
            for key, value in instruction.items():
                if key not in merged_instruction or self._is_missing(merged_instruction.get(key)):
                    merged_instruction[key] = self._copy_value(value)
                elif not self._compatible_value(merged_instruction.get(key), value):
                    return {}
        return {
            "candidate_id": candidate_ids[0] if len(candidate_ids) == 1 else "",
            "candidate_ids": candidate_ids,
            "requirement_id": "|".join(requirement_ids[:5]),
            "instruction": merged_instruction,
        }

    def _compatible_instruction(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        if str(actual.get("action") or "") != str(expected.get("action") or ""):
            return False
        for key, expected_value in expected.items():
            if key in {"action", "supported_by_current_builder"}:
                continue
            if key not in actual or self._is_missing(actual.get(key)):
                continue
            if not self._compatible_value(actual.get(key), expected_value):
                return False
        return True

    def _is_missing(self, value: Any) -> bool:
        return value is None or value == ""

    def _compatible_value(self, actual: Any, expected: Any) -> bool:
        if isinstance(actual, dict) and isinstance(expected, dict):
            return all(
                key in expected and self._compatible_value(value, expected.get(key))
                for key, value in actual.items()
            )
        if isinstance(actual, list) or isinstance(expected, list):
            return self._same_value(actual, expected)
        return self._same_value(actual, expected)

    def _merge_missing_dict(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
        out = dict(actual)
        changed = False
        for key, value in expected.items():
            if key not in out or self._is_missing(out.get(key)):
                out[key] = self._copy_value(value)
                changed = True
            elif isinstance(out.get(key), dict) and isinstance(value, dict):
                merged, nested_changed = self._merge_missing_dict(out[key], value)
                if nested_changed:
                    out[key] = merged
                    changed = True
        return out, changed

    def _refs_candidate(self, instruction: Dict[str, Any], candidate_id: str) -> bool:
        if not candidate_id:
            return False
        for key in ["candidate_id", "from_candidate"]:
            if str(instruction.get(key) or "") == candidate_id:
                return True
        for key in ["candidate_ids", "from_candidates"]:
            value = instruction.get(key)
            if isinstance(value, list) and candidate_id in {str(item) for item in value}:
                return True
            if value and str(value) == candidate_id:
                return True
        target = instruction.get("target")
        if isinstance(target, dict):
            return self._refs_candidate(target, candidate_id)
        return False

    def _same_value(self, actual: Any, expected: Any) -> bool:
        return self._canonical(actual) == self._canonical(expected)

    def _canonical(self, value: Any) -> str:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)

    def _copy_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._copy_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._copy_value(item) for item in value]
        return value

    def _count_field(self, patches: List[Dict[str, Any]], field: str) -> int:
        return sum(1 for patch in patches if field in (patch.get("fields") or []))
