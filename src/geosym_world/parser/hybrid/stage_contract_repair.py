"""Repair missing stage-level bookkeeping without changing raw VLM evidence."""

from __future__ import annotations

from typing import Any, Dict, List, Set
import json


class StageContractRepairer:
    """Fill missing candidate_decisions for final semantic consistency.

    This repair does not invent geometry relations or builder instructions. It
    only makes the final semantic object explicitly account for every script
    candidate in the stage-1 checklist, so downstream task-packet analysis can
    distinguish unreviewed candidates from confirmed/uncertain ones.
    """

    def repair(
        self,
        *,
        semantic: Dict[str, Any],
        evidence_graph: Dict[str, Any],
        instruction_task_packet: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        original_decisions = [
            dict(item) for item in semantic.get("candidate_decisions") or []
            if isinstance(item, dict)
        ]
        original_task_decisions = [
            dict(item) for item in semantic.get("instruction_task_decisions") or []
            if isinstance(item, dict)
        ]
        existing_ids = {
            str(item.get("candidate_id"))
            for item in original_decisions
            if item.get("candidate_id")
        }
        existing_task_ids = {
            str(item.get("candidate_id"))
            for item in original_task_decisions
            if item.get("candidate_id")
        }
        checklist = [
            item for item in evidence_graph.get("candidate_review_checklist") or []
            if isinstance(item, dict) and item.get("candidate_id")
        ]
        relation_refs = self._refs_from_items(semantic.get("relations") or [])
        instruction_refs = self._refs_from_items(semantic.get("build_instructions") or [])
        target_refs = self._target_refs(semantic)
        used_refs = relation_refs | instruction_refs | target_refs
        additions: List[Dict[str, Any]] = []
        task_additions = self._instruction_task_decision_additions(
            semantic=semantic,
            instruction_task_packet=instruction_task_packet or {},
            existing_ids=existing_task_ids,
        )

        for item in checklist:
            cid = str(item.get("candidate_id") or "")
            if not cid or cid in existing_ids:
                continue
            if cid in used_refs:
                decision = "confirm"
                reason = "stage-contract repair: candidate is referenced by relation/target/build instruction"
            else:
                decision = "uncertain"
                reason = "stage-contract repair: VLM did not review this script candidate"
            additions.append({
                "candidate_id": cid,
                "decision": decision,
                "corrected_relation": {},
                "reason": reason,
                "confidence": 0.0,
                "repair_source": "stage_contract_candidate_decision",
            })
            existing_ids.add(cid)

        if not additions and not task_additions:
            return {
                "used": False,
                "num_original_decisions": len(original_decisions),
                "num_added_decisions": 0,
                "num_final_decisions": len(original_decisions),
                "num_original_instruction_task_decisions": len(original_task_decisions),
                "num_added_instruction_task_decisions": 0,
                "num_final_instruction_task_decisions": len(original_task_decisions),
                "additions": [],
                "instruction_task_decision_additions": [],
                "semantic": semantic,
            }

        repaired = dict(semantic)
        if additions:
            repaired["candidate_decisions"] = self._dedupe(original_decisions + additions)
        if task_additions:
            repaired["instruction_task_decisions"] = self._dedupe(original_task_decisions + task_additions)
        metadata = dict(repaired.get("metadata") or {})
        metadata["stage_contract_repair"] = {
            "used": True,
            "num_added": len(additions),
            "num_added_instruction_task_decisions": len(task_additions),
            "source": "candidate_review_checklist",
        }
        repaired["metadata"] = metadata
        return {
            "used": True,
            "num_original_decisions": len(original_decisions),
            "num_added_decisions": len(additions),
            "num_final_decisions": len(repaired.get("candidate_decisions") or original_decisions),
            "num_original_instruction_task_decisions": len(original_task_decisions),
            "num_added_instruction_task_decisions": len(task_additions),
            "num_final_instruction_task_decisions": len(repaired.get("instruction_task_decisions") or original_task_decisions),
            "additions": additions,
            "instruction_task_decision_additions": task_additions,
            "semantic": repaired,
        }

    def _instruction_task_decision_additions(
        self,
        *,
        semantic: Dict[str, Any],
        instruction_task_packet: Dict[str, Any],
        existing_ids: Set[str],
    ) -> List[Dict[str, Any]]:
        required_tasks = [
            item for item in instruction_task_packet.get("required_tasks") or []
            if isinstance(item, dict) and item.get("candidate_id")
        ]
        if not required_tasks:
            return []
        instructions = [
            item for item in semantic.get("build_instructions") or []
            if isinstance(item, dict)
        ]
        additions: List[Dict[str, Any]] = []
        for task in required_tasks:
            cid = str(task.get("candidate_id") or "")
            if not cid or cid in existing_ids:
                continue
            required_actions = [str(item) for item in task.get("required_actions") or [] if item]
            covered_actions = sorted(self._covered_actions_for_candidate(cid, instructions))
            missing_actions = sorted(set(required_actions) - set(covered_actions))
            decision = "emit" if covered_actions and not missing_actions else "uncertain"
            additions.append({
                "candidate_id": cid,
                "decision": decision,
                "emitted_actions": covered_actions,
                "missing_actions": missing_actions,
                "reason": (
                    "stage-contract repair: inferred emit from existing build_instructions"
                    if decision == "emit"
                    else "stage-contract repair: required task was not fully covered by existing build_instructions"
                ),
                "confidence": 0.0,
                "repair_source": "stage_contract_instruction_task_decision",
            })
            existing_ids.add(cid)
        return additions

    def _covered_actions_for_candidate(self, candidate_id: str, instructions: List[Dict[str, Any]]) -> Set[str]:
        actions: Set[str] = set()
        for instruction in instructions:
            if candidate_id in self._refs_from_item(instruction):
                action = str(instruction.get("action") or "")
                if action:
                    actions.add(action)
        return actions

    def _refs_from_items(self, items: List[Dict[str, Any]]) -> Set[str]:
        refs: Set[str] = set()
        for item in items:
            refs.update(self._refs_from_item(item))
        return refs

    def _target_refs(self, semantic: Dict[str, Any]) -> Set[str]:
        refs = self._refs_from_item(semantic.get("target") or {})
        refs.update(self._refs_from_items(semantic.get("targets") or []))
        return refs

    def _refs_from_item(self, item: Dict[str, Any]) -> Set[str]:
        if not isinstance(item, dict):
            return set()
        refs: Set[str] = set()
        for key in ["candidate_id", "from_candidate"]:
            value = item.get(key)
            if value:
                refs.add(str(value))
        for key in ["candidate_ids", "from_candidates"]:
            value = item.get(key)
            if isinstance(value, list):
                refs.update(str(v) for v in value if v)
            elif value:
                refs.add(str(value))
        target = item.get("target")
        if isinstance(target, dict):
            refs.update(self._refs_from_item(target))
        return refs

    def _dedupe(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        seen = set()
        for item in items:
            key = json.dumps(item, sort_keys=True, ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out
