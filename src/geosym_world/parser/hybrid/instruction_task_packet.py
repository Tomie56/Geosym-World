"""Build the explicit task packet for staged instruction generation."""

from __future__ import annotations

from typing import Any, Dict, List
import json

from .relation_object_normalizer import RelationObjectNormalizer


class InstructionTaskPacketBuilder:
    """Create a compact instruction-pass contract from relation-pass output.

    The packet is shown to the VLM in stage 2 and written to disk for audit. It
    tells the model exactly which confirmed/corrected candidates must become
    GeoSym builder instructions, instead of making it rediscover that from the
    full evidence graph.
    """

    def __init__(self) -> None:
        self.object_normalizer = RelationObjectNormalizer()

    def build(self, *, relation_pass: Dict[str, Any], evidence_graph: Dict[str, Any]) -> Dict[str, Any]:
        decisions = self._decision_index(relation_pass)
        relation_refs = self._refs_from_items(relation_pass.get("relations") or [])
        target_refs = self._target_refs(relation_pass)
        relation_visual_ids = self._relation_visual_ids_by_candidate(relation_pass)
        relation_builder_decisions = self._relation_builder_decision_index(relation_pass)
        used_refs = relation_refs | target_refs
        primary_target_ref = self._primary_target_ref(relation_pass)
        canonical_targets_by_ref = self._targets_by_ref(relation_pass)
        required_tasks = []
        deferred_tasks = []
        for task in evidence_graph.get("build_tasks") or []:
            cid = str(task.get("candidate_id") or "")
            if not cid:
                continue
            decision = decisions.get(cid, {})
            decision_value = str(decision.get("decision") or "")
            builder_decision = relation_builder_decisions.get(cid, {})
            used = cid in used_refs
            should_emit = decision_value in set(task.get("must_emit_when_decision") or []) or used
            if task.get("candidate_type") == "target" and primary_target_ref and cid != primary_target_ref:
                should_emit = False
            corrected_requirements = self._corrected_instruction_requirements(cid, decision, task)
            instruction_requirements = corrected_requirements or (task.get("instruction_requirements") or [])
            raw_corrected_relation = dict(decision.get("corrected_relation") or {}) if isinstance(decision.get("corrected_relation"), dict) else {}
            corrected_relation = raw_corrected_relation if decision_value == "correct" and raw_corrected_relation else {}
            corrected_buildability = self._buildability_from_corrected(corrected_relation) if corrected_relation else {}
            effective_builder_mapping = self._builder_mapping_from_corrected(corrected_relation) if corrected_relation else dict(task.get("builder_mapping") or {})
            suggested_instructions = [
                item for item in (task.get("suggested_instructions") or [])
                if item.get("supported_by_current_builder", True)
            ]
            if corrected_requirements:
                suggested_instructions = [
                    dict(item.get("expected_instruction") or {})
                    for item in corrected_requirements
                    if item.get("expected_instruction")
                ]
            elif corrected_relation and not corrected_buildability.get("can_generate_build_instruction"):
                instruction_requirements = []
                suggested_instructions = []
                should_emit = False
            if builder_decision and not self._relation_builder_decision_allows_emit(builder_decision):
                instruction_requirements = []
                suggested_instructions = []
                should_emit = False
            blocked_by_builder_decision = bool(builder_decision and not self._relation_builder_decision_allows_emit(builder_decision))
            canonical_target = canonical_targets_by_ref.get(cid) if task.get("candidate_type") == "target" else {}
            if canonical_target and self._target_rank(canonical_target) > self._target_rank(self._target_from_requirements(instruction_requirements)):
                instruction_requirements = self._target_instruction_requirements(cid, canonical_target, source="relation_pass.target")
                suggested_instructions = [dict(item.get("expected_instruction") or {}) for item in instruction_requirements]
            if task.get("candidate_type") == "target" and not self._target_is_emit_ready(self._target_from_requirements(instruction_requirements) or canonical_target):
                instruction_requirements = []
                suggested_instructions = []
                should_emit = False
            visual_ids = relation_visual_ids.get(cid) or []
            instruction_requirements = self._attach_visual_ids_to_requirements(instruction_requirements, visual_ids)
            suggested_instructions = self._attach_visual_ids_to_instructions(suggested_instructions, visual_ids)
            item = {
                "candidate_id": cid,
                "candidate_type": task.get("candidate_type"),
                "candidate_object": task.get("candidate_object"),
                "effective_candidate_object": (
                    canonical_target.get("object")
                    if canonical_target
                    else self._candidate_object_from_corrected(corrected_relation) if corrected_relation else task.get("candidate_object")
                ),
                "decision": decision_value or ("used_without_decision" if used else "unreviewed"),
                "decision_reason": decision.get("reason") or "",
                "relation_builder_decision": dict(builder_decision),
                "blocked_by_relation_builder_decision": blocked_by_builder_decision,
                "corrected_relation_used": bool(corrected_relation),
                "corrected_relation": corrected_relation,
                "instruction_source": "candidate_decision.corrected_relation" if corrected_requirements else "script_build_suggestion",
                "relation_buildability": corrected_buildability if corrected_relation else dict(task.get("relation_buildability") or {}),
                "builder_mapping": effective_builder_mapping,
                "buildability_mode": (corrected_buildability.get("mode") if corrected_relation else task.get("buildability_mode")),
                "required_actions": (self._required_actions_from_instructions(instruction_requirements) if corrected_relation else (task.get("required_actions") or [])),
                "instruction_requirements": instruction_requirements,
                "optional_actions": task.get("optional_actions") or [],
                "suggested_instructions": suggested_instructions,
                "notes": task.get("notes") or "",
            }
            if should_emit and item["required_actions"]:
                required_tasks.append(item)
            else:
                deferred_tasks.append(item)

        vlm_relation_tasks = self._vlm_relation_tasks(
            relation_pass,
            evidence_graph=evidence_graph,
            relation_builder_decisions=relation_builder_decisions,
        )
        required_tasks.extend(vlm_relation_tasks["required"])
        deferred_tasks.extend(vlm_relation_tasks["deferred"])

        capability_guide = self._capability_guide(required_tasks, deferred_tasks, evidence_graph)
        engine_translation_contract = self._engine_translation_contract(required_tasks, deferred_tasks)
        engine_translation_summary = self._engine_translation_summary(engine_translation_contract)
        response_template = self._response_template(required_tasks, relation_pass, evidence_graph)
        decision_template = self._task_decision_template(required_tasks)
        builder_contract = evidence_graph.get("builder_contract") or {}
        return {
            "version": "instruction-task-packet-v0",
            "purpose": "stage-2 VLM must convert required_tasks into GeoSym build_instructions",
            "builder_contract": {
                "supported_actions": builder_contract.get("supported_actions") or [],
                "action_schema": builder_contract.get("action_schema") or [],
                "supported_constraint_types": builder_contract.get("supported_constraint_types") or [],
                "theorem_relation_types": builder_contract.get("theorem_relation_types") or [],
                "relation_mapping_contract": builder_contract.get("relation_mapping_contract") or [],
                "rule": builder_contract.get("rule") or "",
            },
            "capability_guide": capability_guide,
            "engine_translation_contract": engine_translation_contract,
            "engine_translation_summary": engine_translation_summary,
            "vlm_derived_evidence": self._vlm_derived_evidence(required_tasks, deferred_tasks),
            "required_tasks": required_tasks,
            "deferred_tasks": deferred_tasks,
            "instruction_task_decision_template": decision_template,
            "response_template": response_template,
            "metrics": {
                "num_decisions": len(decisions),
                "num_relation_refs": len(relation_refs),
                "num_target_refs": len(target_refs),
                "num_required_tasks": len(required_tasks),
                "num_instruction_task_decision_template": len(decision_template),
                "num_deferred_tasks": len(deferred_tasks),
                "num_required_actions": sum(len(item.get("required_actions") or []) for item in required_tasks),
                "num_instruction_requirements": sum(len(item.get("instruction_requirements") or []) for item in required_tasks),
                "num_template_build_instructions": len(response_template.get("build_instructions") or []),
                "num_corrected_relation_tasks": sum(1 for item in required_tasks if item.get("corrected_relation_used")),
                "num_builder_mapped_tasks": sum(1 for item in required_tasks if item.get("builder_mapping")),
                "num_relation_builder_decision_blocked_tasks": sum(1 for item in deferred_tasks if item.get("blocked_by_relation_builder_decision")),
                "num_relation_builder_decision_theorem_only_blocked_tasks": sum(
                    1
                    for item in deferred_tasks
                    if item.get("blocked_by_relation_builder_decision")
                    and self._is_theorem_only_blocked_task(item)
                ),
                "num_relation_builder_decision_unexpected_blocked_tasks": sum(
                    1
                    for item in deferred_tasks
                    if item.get("blocked_by_relation_builder_decision")
                    and self._is_unexpected_blocked_task(item)
                ),
                "num_vlm_derived_required_tasks": len(vlm_relation_tasks["required"]),
                "num_vlm_derived_deferred_tasks": len(vlm_relation_tasks["deferred"]),
                "num_vlm_derived_instruction_requirements": sum(len(item.get("instruction_requirements") or []) for item in vlm_relation_tasks["required"]),
                "num_vlm_derived_evidence_items": len(vlm_relation_tasks["required"]) + len(vlm_relation_tasks["deferred"]),
                "num_vlm_derived_evidence_with_visual_ids": sum(
                    1
                    for item in vlm_relation_tasks["required"] + vlm_relation_tasks["deferred"]
                    if ((item.get("provenance") or {}).get("evidence") or {}).get("visual_ids")
                ),
                "num_vlm_derived_evidence_with_named_object_binding_visual_ids": sum(
                    1
                    for item in vlm_relation_tasks["required"] + vlm_relation_tasks["deferred"]
                    if ((item.get("provenance") or {}).get("evidence") or {}).get("named_object_binding_visual_ids")
                ),
                "num_vlm_derived_evidence_named_object_binding_visual_ids": sum(
                    len(((item.get("provenance") or {}).get("evidence") or {}).get("named_object_binding_visual_ids") or [])
                    for item in vlm_relation_tasks["required"] + vlm_relation_tasks["deferred"]
                ),
                "num_vlm_derived_evidence_invalid_visual_ids": sum(
                    len(((item.get("provenance") or {}).get("evidence") or {}).get("invalid_visual_ids") or [])
                    for item in vlm_relation_tasks["required"] + vlm_relation_tasks["deferred"]
                ),
                "num_vlm_derived_evidence_invalid_visual_id_tasks": sum(
                    1
                    for item in vlm_relation_tasks["required"] + vlm_relation_tasks["deferred"]
                    if ((item.get("provenance") or {}).get("evidence") or {}).get("invalid_visual_ids")
                ),
                "num_vlm_derived_evidence_invalid_text_refs": sum(
                    len(((item.get("provenance") or {}).get("evidence") or {}).get("invalid_text_refs") or [])
                    for item in vlm_relation_tasks["required"] + vlm_relation_tasks["deferred"]
                ),
                "num_vlm_derived_evidence_invalid_text_ref_tasks": sum(
                    1
                    for item in vlm_relation_tasks["required"] + vlm_relation_tasks["deferred"]
                    if ((item.get("provenance") or {}).get("evidence") or {}).get("invalid_text_refs")
                ),
                "num_vlm_derived_evidence_invalid_text_ref_blocking_tasks": sum(
                    1
                    for item in vlm_relation_tasks["required"] + vlm_relation_tasks["deferred"]
                    if ((item.get("provenance") or {}).get("evidence") or {}).get("invalid_text_refs")
                    and not ((item.get("evidence_gate") or {}).get("has_explicit_evidence"))
                ),
                "num_vlm_derived_evidence_can_generate_build_instruction": sum(
                    1
                    for item in vlm_relation_tasks["required"] + vlm_relation_tasks["deferred"]
                    if (item.get("relation_buildability") or {}).get("can_generate_build_instruction")
                    and item.get("instruction_requirements")
                ),
                "num_vlm_derived_evidence_engine_can_encode_relation": sum(
                    1
                    for item in vlm_relation_tasks["required"] + vlm_relation_tasks["deferred"]
                    if (item.get("relation_buildability") or {}).get("can_generate_build_instruction")
                ),
                "num_vlm_derived_object_normalization_failed": sum(
                    1
                    for item in vlm_relation_tasks["required"] + vlm_relation_tasks["deferred"]
                    if (
                        (item.get("relation_buildability") or {}).get("can_generate_build_instruction")
                        and not item.get("instruction_requirements")
                    )
                ),
                "num_vlm_derived_missing_relation_builder_decisions": sum(
                    1
                    for item in vlm_relation_tasks["required"] + vlm_relation_tasks["deferred"]
                    if item.get("missing_relation_builder_decision")
                ),
                "num_vlm_derived_blocked_by_relation_builder_decision": sum(
                    1
                    for item in vlm_relation_tasks["required"] + vlm_relation_tasks["deferred"]
                    if item.get("blocked_by_relation_builder_decision")
                ),
                "num_vlm_derived_theorem_only_deferred": sum(
                    1
                    for item in vlm_relation_tasks["required"] + vlm_relation_tasks["deferred"]
                    if item.get("blocked_by_relation_builder_decision")
                    and self._is_theorem_only_blocked_task(item)
                ),
                "num_vlm_derived_unexpected_deferred": sum(
                    1
                    for item in vlm_relation_tasks["required"] + vlm_relation_tasks["deferred"]
                    if item.get("blocked_by_relation_builder_decision")
                    and self._is_unexpected_blocked_task(item)
                ),
                "num_vlm_derived_evidence_gate_passed": sum(
                    1
                    for item in vlm_relation_tasks["required"] + vlm_relation_tasks["deferred"]
                    if (item.get("evidence_gate") or {}).get("has_explicit_evidence")
                ),
                "num_vlm_derived_evidence_required_after_gate": len(vlm_relation_tasks["required"]),
                "num_capability_guide_relation_rules": len(capability_guide.get("relation_to_instruction_rules") or []),
                "num_capability_guide_task_decisions": len(capability_guide.get("task_decision_guide") or []),
                "num_capability_guide_deferred_tasks": len(capability_guide.get("deferred_task_guide") or []),
                "num_engine_translation_contract_items": len(engine_translation_contract.get("items") or []),
                "num_engine_translation_contract_emit_items": sum(1 for item in engine_translation_contract.get("items") or [] if item.get("translation_decision") == "emit"),
                "num_engine_translation_contract_blocked_items": sum(1 for item in engine_translation_contract.get("items") or [] if item.get("translation_decision") != "emit"),
                "num_engine_translation_summary_missing_decision_items": int(engine_translation_summary.get("missing_relation_builder_decision_items") or 0),
                "num_engine_translation_summary_theorem_only_items": int(engine_translation_summary.get("theorem_only_items") or 0),
                "num_engine_translation_summary_unsupported_items": int(engine_translation_summary.get("unsupported_items") or 0),
                "num_engine_translation_summary_evidence_gate_failed_items": int(engine_translation_summary.get("evidence_gate_failed_items") or 0),
                "num_engine_translation_summary_object_normalization_failed_items": int(engine_translation_summary.get("object_normalization_failed_items") or 0),
            },
            "rules": [
                "response_template is the preferred GeoSym stage-2 output draft",
                "copy response_template.build_instructions when it is consistent with stage-1 relations",
                "engine_translation_contract is the authoritative stage-1-to-stage-2 conversion table",
                "emit only engine_translation_contract items whose translation_decision is emit",
                "emit all required_actions for each required_task when possible",
                "emit every instruction_requirements[*].expected_instruction exactly when possible",
                "copy suggested_instructions only when supported by builder_contract.supported_actions",
                "include from_candidate for every emitted instruction",
                "do not output pixel coordinates",
                "do not convert theorem-only tasks into fake coordinate constraints",
                "when corrected_relation_used is true, follow corrected_relation and effective_candidate_object instead of the original candidate_object",
                "use builder_mapping to decide which semantic relations can become GeoSym builder instructions",
                "use builder_contract.relation_mapping_contract as the global source of truth for relation-to-action conversion",
                "use capability_guide.task_decision_guide to choose emit/skip/defer/uncertain for every required_task",
                "use capability_guide.deferred_task_guide to avoid fake coordinate constraints for theorem-only candidates",
                "vlm_derived tasks come from VLM-discovered relations; emit them only when their visual/text evidence is explicit and builder_mapping allows it",
                "review vlm_derived_evidence before trusting VLM-discovered constraints; these constraints require downstream verification and review",
                "emit one instruction_task_decisions item for every required_task",
            ],
        }

    def _engine_translation_summary(self, contract: Dict[str, Any]) -> Dict[str, Any]:
        items = [item for item in contract.get("items") or [] if isinstance(item, dict)]
        by_decision: Dict[str, int] = {}
        by_builder_status: Dict[str, int] = {}
        by_candidate_type: Dict[str, int] = {}
        block_reason_counts: Dict[str, int] = {}
        for item in items:
            decision = str(item.get("translation_decision") or "unknown")
            by_decision[decision] = by_decision.get(decision, 0) + 1
            status = str(item.get("builder_mapping_status") or "unknown")
            by_builder_status[status] = by_builder_status.get(status, 0) + 1
            candidate_type = str(item.get("candidate_type") or "unknown")
            by_candidate_type[candidate_type] = by_candidate_type.get(candidate_type, 0) + 1
            for reason in item.get("block_reasons") or []:
                key = str(reason or "")
                if key:
                    block_reason_counts[key] = block_reason_counts.get(key, 0) + 1
        return {
            "version": "engine-translation-summary-v0",
            "items_total": len(items),
            "emit_items": by_decision.get("emit", 0),
            "defer_items": sum(count for decision, count in by_decision.items() if decision != "emit"),
            "by_translation_decision": by_decision,
            "by_builder_mapping_status": by_builder_status,
            "by_candidate_type": by_candidate_type,
            "block_reason_counts": block_reason_counts,
            "missing_relation_builder_decision_items": sum(
                1
                for item in items
                if any("missing relation_builder_decision" == str(reason) for reason in item.get("block_reasons") or [])
            ),
            "theorem_only_items": sum(
                1
                for item in items
                if str(item.get("builder_mapping_status") or "") in {"theorem_only", "not_buildable"}
                or str(item.get("buildability_mode") or "") == "theorem_only"
                or str(item.get("relation_builder_decision") or "") == "theorem_only"
            ),
            "unsupported_items": sum(
                1
                for item in items
                if str(item.get("builder_mapping_status") or "") == "unsupported"
                or str(item.get("buildability_mode") or "") == "unsupported"
                or str(item.get("relation_builder_decision") or "") == "unsupported"
            ),
            "evidence_gate_failed_items": sum(
                1
                for item in items
                if item.get("review_required") and item.get("evidence_gate_passed") is False
            ),
            "object_normalization_failed_items": sum(
                1
                for item in items
                if any("object normalization failed" == str(reason) for reason in item.get("block_reasons") or [])
            ),
        }

    def _is_theorem_only_blocked_task(self, task: Dict[str, Any]) -> bool:
        mapping = task.get("builder_mapping") or {}
        buildability = task.get("relation_buildability") or {}
        status = str(mapping.get("status") or "").strip().lower()
        mode = str(buildability.get("mode") or mapping.get("buildability_mode") or "").strip().lower()
        return status in {"theorem_only", "not_buildable"} or mode == "theorem_only"

    def _is_unexpected_blocked_task(self, task: Dict[str, Any]) -> bool:
        mapping = task.get("builder_mapping") or {}
        buildability = task.get("relation_buildability") or {}
        status = str(mapping.get("status") or "").strip().lower()
        can_generate = bool(buildability.get("can_generate_build_instruction"))
        has_requirements = bool(task.get("instruction_requirements"))
        return status in {"buildable", "convertible"} or can_generate or has_requirements

    def _decision_index(self, semantic: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        out = {}
        for item in semantic.get("candidate_decisions") or []:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("candidate_id") or "")
            if cid:
                out[cid] = item
        return out

    def _relation_builder_decision_index(self, semantic: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        out = {}
        for item in semantic.get("relation_builder_decisions") or []:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("candidate_id") or item.get("from_candidate") or "")
            if cid:
                out[cid] = item
            if item.get("relation_index") is not None:
                try:
                    out[f"relation[{int(item.get('relation_index'))}]"] = item
                    out[f"vlm_relation[{int(item.get('relation_index'))}]"] = item
                except (TypeError, ValueError):
                    pass
            signature = self._decision_signature(item)
            if signature != "||":
                out[f"signature::{signature}"] = item
        return out

    def _relation_builder_decision_allows_emit(self, decision: Dict[str, Any]) -> bool:
        value = str(decision.get("decision") or "").lower()
        return value in {"buildable", "convertible"}

    def _target_refs(self, semantic: Dict[str, Any]) -> set[str]:
        out = self._refs_from_item(semantic.get("target") or {})
        for item in semantic.get("targets") or []:
            if isinstance(item, dict):
                out.update(self._refs_from_item(item))
        return out

    def _targets_by_ref(self, semantic: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        targets: List[Dict[str, Any]] = []
        if isinstance(semantic.get("target"), dict):
            targets.append(dict(semantic["target"]))
        targets.extend(dict(item) for item in semantic.get("targets") or [] if isinstance(item, dict))
        for target in targets:
            refs = self._refs_from_item(target)
            for ref in refs:
                current = out.get(ref) or {}
                if self._target_rank(target) > self._target_rank(current):
                    out[ref] = dict(target)
        return out

    def _primary_target_ref(self, semantic: Dict[str, Any]) -> str:
        refs = sorted(self._refs_from_item(semantic.get("target") or {}))
        if refs:
            return refs[0]
        for item in semantic.get("targets") or []:
            if not isinstance(item, dict):
                continue
            refs = sorted(self._refs_from_item(item))
            if refs:
                return refs[0]
        return ""

    def _required_actions_from_instructions(self, instructions: List[Dict[str, Any]]) -> List[str]:
        out: List[str] = []
        seen = set()
        for instruction in instructions:
            action = str((instruction.get("expected_instruction") or {}).get("action") or instruction.get("action") or "")
            if action and action not in seen:
                seen.add(action)
                out.append(action)
        return out

    def _refs_from_items(self, items: List[Dict[str, Any]]) -> set[str]:
        out = set()
        for item in items:
            out.update(self._refs_from_item(item))
        return out

    def _refs_from_item(self, item: Dict[str, Any]) -> set[str]:
        if not isinstance(item, dict):
            return set()
        out = set()
        for key in ["candidate_id", "from_candidate"]:
            value = item.get(key)
            if value:
                out.add(str(value))
        for key in ["candidate_ids", "from_candidates"]:
            value = item.get(key)
            if isinstance(value, list):
                out.update(str(v) for v in value if v)
            elif value:
                out.add(str(value))
        target = item.get("target")
        if isinstance(target, dict):
            out.update(self._refs_from_item(target))
        return out

    def _relation_visual_ids_by_candidate(self, relation_pass: Dict[str, Any]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for rel in relation_pass.get("relations") or []:
            if not isinstance(rel, dict):
                continue
            cid = str(rel.get("candidate_id") or rel.get("from_candidate") or "")
            if not cid:
                continue
            visual_ids = [str(item) for item in rel.get("visual_ids") or [] if item]
            if not visual_ids:
                continue
            current = out.setdefault(cid, [])
            for vid in visual_ids:
                if vid not in current:
                    current.append(vid)
        return out

    def _attach_visual_ids_to_requirements(
        self,
        requirements: List[Dict[str, Any]],
        visual_ids: List[str],
    ) -> List[Dict[str, Any]]:
        if not visual_ids:
            return requirements
        out = []
        for requirement in requirements:
            item = dict(requirement)
            expected = item.get("expected_instruction")
            if isinstance(expected, dict):
                item["expected_instruction"] = self._attach_visual_ids_to_instruction(expected, visual_ids)
                match_rule = dict(item.get("match_rule") or {})
                field_values = dict(match_rule.get("field_values") or {})
                if "visual_ids" in item["expected_instruction"]:
                    field_values["visual_ids"] = item["expected_instruction"]["visual_ids"]
                    match_rule["field_values"] = field_values
                    item["match_rule"] = match_rule
            out.append(item)
        return out

    def _attach_visual_ids_to_instructions(
        self,
        instructions: List[Dict[str, Any]],
        visual_ids: List[str],
    ) -> List[Dict[str, Any]]:
        if not visual_ids:
            return instructions
        return [self._attach_visual_ids_to_instruction(item, visual_ids) for item in instructions]

    def _attach_visual_ids_to_instruction(self, instruction: Dict[str, Any], visual_ids: List[str]) -> Dict[str, Any]:
        item = dict(instruction)
        action = str(item.get("action") or "")
        if action not in {"add_segment", "set_length", "set_parallel", "set_perpendicular"}:
            return item
        merged = []
        for vid in [str(value) for value in (item.get("visual_ids") or []) if value] + [str(value) for value in visual_ids if value]:
            if vid not in merged:
                merged.append(vid)
        if merged:
            item["visual_ids"] = merged
        return item

    def _response_template(
        self,
        required_tasks: List[Dict[str, Any]],
        relation_pass: Dict[str, Any],
        evidence_graph: Dict[str, Any],
    ) -> Dict[str, Any]:
        build_instructions = []
        seen = set()
        for task in required_tasks:
            for requirement in task.get("instruction_requirements") or []:
                if not isinstance(requirement, dict):
                    continue
                expected = dict(requirement.get("expected_instruction") or {})
                if not expected:
                    continue
                key = self._stable_key(expected)
                if key in seen:
                    continue
                seen.add(key)
                build_instructions.append(expected)
        target = self._template_target(required_tasks, relation_pass)
        assessment = self._template_engine_assessment(required_tasks, evidence_graph)
        return {
            "build_instructions": build_instructions,
            "instruction_task_decisions": self._task_decision_template(required_tasks),
            "target": target,
            "targets": self._template_targets(relation_pass, target),
            "engine_assessment": assessment,
            "notes": (
                "Draft generated from confirmed/corrected script candidates and current GeoSym builder capability; "
                "stage-2 VLM should copy it unless it conflicts with stage-1 relations or the image/text."
            ),
        }

    def _task_decision_template(self, required_tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for task in required_tasks:
            cid = str(task.get("candidate_id") or "")
            if not cid:
                continue
            out.append({
                "candidate_id": cid,
                "decision": "emit",
                "emitted_actions": list(task.get("required_actions") or []),
                "missing_actions": [],
                "reason": "required task is covered by current GeoSym builder capability",
                "confidence": 0.0,
            })
        return out

    def _vlm_relation_tasks(
        self,
        relation_pass: Dict[str, Any],
        *,
        evidence_graph: Dict[str, Any],
        relation_builder_decisions: Dict[str, Dict[str, Any]] | None = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        required: List[Dict[str, Any]] = []
        deferred: List[Dict[str, Any]] = []
        relation_builder_decisions = relation_builder_decisions or {}
        valid_visual_ids = self._valid_visual_ids_from_evidence(evidence_graph)
        text_corpus = self._text_corpus_from_evidence(evidence_graph)
        binding_index = self._named_object_binding_visual_index(relation_pass)
        seen_candidate_ids = set(self._refs_from_items(relation_pass.get("relations") or []))
        used_ids = set()
        for idx, relation in enumerate(relation_pass.get("relations") or []):
            if not isinstance(relation, dict):
                continue
            if relation.get("candidate_id") or relation.get("from_candidate"):
                continue
            rtype = str(relation.get("type") or "")
            if not rtype:
                continue
            cid = self._vlm_relation_candidate_id(idx, relation, used_ids | seen_candidate_ids)
            used_ids.add(cid)
            builder_decision = self._relation_builder_decision_for_vlm_relation(
                relation_builder_decisions,
                relation=relation,
                relation_index=idx,
                generated_candidate_id=cid,
            )
            builder_decision_allows_emit = bool(builder_decision and self._relation_builder_decision_allows_emit(builder_decision))
            buildability = dict(relation.get("buildability") or {})
            mapping = dict(relation.get("builder_mapping") or {})
            evidence_summary = self._relation_evidence_summary(relation)
            valid_relation_visual_ids, invalid_relation_visual_ids = self._split_valid_visual_ids(
                evidence_summary.get("visual_ids") or [],
                valid_visual_ids,
            )
            binding_evidence = self._named_object_binding_evidence(
                relation,
                binding_index=binding_index,
                valid_visual_ids=valid_visual_ids,
            )
            if binding_evidence.get("visual_ids") or binding_evidence.get("invalid_visual_ids"):
                valid_relation_visual_ids = self._dedupe_list(valid_relation_visual_ids + (binding_evidence.get("visual_ids") or []))
                invalid_relation_visual_ids = self._dedupe_list(invalid_relation_visual_ids + (binding_evidence.get("invalid_visual_ids") or []))
                evidence_summary = dict(evidence_summary)
                evidence_summary["visual_ids"] = valid_relation_visual_ids
                evidence_summary["named_object_binding_visual_ids"] = binding_evidence.get("visual_ids") or []
                evidence_summary["named_object_binding_invalid_visual_ids"] = binding_evidence.get("invalid_visual_ids") or []
                evidence_summary["named_object_binding_objects"] = binding_evidence.get("objects") or []
                evidence_summary["named_object_binding_records"] = binding_evidence.get("bindings") or []
                evidence_summary["has_named_object_binding_evidence"] = bool(binding_evidence.get("visual_ids"))
                evidence_summary["has_visual_evidence"] = bool(valid_relation_visual_ids)
                sources = [str(item) for item in evidence_summary.get("evidence_sources") or [] if item]
                if binding_evidence.get("visual_ids") and "named_object_bindings" not in sources:
                    sources.append("named_object_bindings")
                if sources:
                    evidence_summary["evidence_sources"] = sources
            if invalid_relation_visual_ids:
                evidence_summary = dict(evidence_summary)
                evidence_summary["visual_ids"] = valid_relation_visual_ids
                evidence_summary["invalid_visual_ids"] = invalid_relation_visual_ids
                evidence_summary["has_visual_evidence"] = bool(valid_relation_visual_ids)
            valid_text_refs, invalid_text_refs = self._split_valid_text_refs(
                evidence_summary.get("text_refs") or [],
                text_corpus,
            )
            if valid_text_refs != (evidence_summary.get("text_refs") or []) or invalid_text_refs:
                evidence_summary = dict(evidence_summary)
                evidence_summary["text_refs"] = valid_text_refs
                evidence_summary["invalid_text_refs"] = invalid_text_refs
                evidence_summary["has_text_evidence"] = bool(valid_text_refs)
            relation_for_build = dict(relation)
            relation_for_build["visual_ids"] = valid_relation_visual_ids
            relation_for_build["text_refs"] = valid_text_refs
            if invalid_relation_visual_ids:
                relation_for_build["invalid_visual_ids"] = invalid_relation_visual_ids
            if invalid_text_refs:
                relation_for_build["invalid_text_refs"] = invalid_text_refs
            instructions = self._instructions_from_corrected(cid, relation_for_build, {"candidate_type": rtype})
            requirements = self._requirements_from_instructions(cid, instructions, source="vlm_relation")
            visual_ids = valid_relation_visual_ids
            requirements = self._attach_visual_ids_to_requirements(requirements, visual_ids)
            suggested_instructions = self._attach_visual_ids_to_instructions(
                [dict(item.get("expected_instruction") or {}) for item in requirements if item.get("expected_instruction")],
                visual_ids,
            )
            task = {
                "candidate_id": cid,
                "candidate_type": rtype,
                "candidate_object": self._candidate_object_from_corrected(relation),
                "effective_candidate_object": self._candidate_object_from_corrected(relation),
                "decision": "vlm_derived",
                "decision_reason": "relation was proposed by VLM stage-1 and has no script candidate id",
                "relation_builder_decision": dict(builder_decision),
                "blocked_by_relation_builder_decision": bool(builder_decision and not builder_decision_allows_emit),
                "missing_relation_builder_decision": not bool(builder_decision),
                "corrected_relation_used": False,
                "corrected_relation": {},
                "instruction_source": "vlm_relation",
                "relation_buildability": buildability,
                "builder_mapping": mapping,
                "buildability_mode": buildability.get("mode") or mapping.get("buildability_mode"),
                "required_actions": self._required_actions_from_instructions(requirements),
                "instruction_requirements": requirements,
                "optional_actions": [],
                "suggested_instructions": suggested_instructions,
                "notes": "VLM-derived relation: use for stage-2 conversion only when evidence is explicit; verifier remains authoritative",
                "provenance": {
                    "source": "vlm_relation",
                    "relation_index": idx,
                    "requires_verification": True,
                    "evidence": evidence_summary,
                    "original_relation": relation_for_build,
                },
            }
            has_explicit_evidence = bool(evidence_summary.get("has_visual_evidence") or evidence_summary.get("has_text_evidence"))
            task["evidence_gate"] = {
                "has_explicit_evidence": has_explicit_evidence,
                "valid_visual_ids": valid_relation_visual_ids,
                "invalid_visual_ids": invalid_relation_visual_ids,
                "valid_text_refs": valid_text_refs,
                "invalid_text_refs": invalid_text_refs,
                "required_for_required_task": True,
                "reason": "VLM-derived relations need valid visual_ids or text_refs before they can become required build tasks",
            }
            engine_can_encode_relation = bool(buildability.get("can_generate_build_instruction"))
            object_normalization_failed = engine_can_encode_relation and not bool(requirements)
            if not builder_decision:
                task["relation_builder_decision_issue"] = {
                    "missing": True,
                    "reason": "VLM-derived relation lacks relation_builder_decisions entry; defer until stage-1 maps it to current GeoSym builder capability",
                }
            if object_normalization_failed:
                task["object_normalization"] = {
                    "failed": True,
                    "reason": "relation type is buildable by GeoSym but named objects could not be normalized into builder instructions",
                    "relation_type": rtype,
                    "candidate_object": task.get("effective_candidate_object") or task.get("candidate_object"),
                }
            can_build = (
                engine_can_encode_relation
                and bool(requirements)
                and has_explicit_evidence
                and builder_decision_allows_emit
            )
            if can_build:
                required.append(task)
            else:
                deferred.append(task)
        return {"required": required, "deferred": deferred}

    def _vlm_derived_evidence(self, required_tasks: List[Dict[str, Any]], deferred_tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for status, tasks in [("required", required_tasks), ("deferred", deferred_tasks)]:
            for task in tasks:
                cid = str(task.get("candidate_id") or "")
                if not cid.startswith("vlm_rel_"):
                    continue
                provenance = task.get("provenance") or {}
                relation = provenance.get("original_relation") or {}
                evidence = provenance.get("evidence") or self._relation_evidence_summary(relation)
                mapping = task.get("builder_mapping") or {}
                buildability = task.get("relation_buildability") or {}
                engine_can_encode_relation = bool(buildability.get("can_generate_build_instruction"))
                instruction_requirements = len(task.get("instruction_requirements") or [])
                object_normalization_failed = engine_can_encode_relation and instruction_requirements == 0
                out.append({
                    "candidate_id": cid,
                    "task_status": status,
                    "relation_type": task.get("candidate_type"),
                    "candidate_object": task.get("effective_candidate_object") or task.get("candidate_object"),
                    "evidence": evidence,
                    "invalid_visual_ids": evidence.get("invalid_visual_ids") or [],
                    "invalid_text_refs": evidence.get("invalid_text_refs") or [],
                    "builder_mapping_status": mapping.get("status"),
                    "buildability_mode": task.get("buildability_mode") or buildability.get("mode") or mapping.get("buildability_mode"),
                    "engine_can_encode_relation": engine_can_encode_relation,
                    "can_generate_build_instruction": engine_can_encode_relation and instruction_requirements > 0,
                    "relation_builder_decision": (task.get("relation_builder_decision") or {}).get("decision"),
                    "missing_relation_builder_decision": bool(task.get("missing_relation_builder_decision")),
                    "blocked_by_relation_builder_decision": bool(task.get("blocked_by_relation_builder_decision")),
                    "object_normalization_failed": object_normalization_failed,
                    "object_normalization": task.get("object_normalization") or {},
                    "passes_evidence_gate": bool((task.get("evidence_gate") or {}).get("has_explicit_evidence")),
                    "required_actions": list(task.get("required_actions") or []),
                    "instruction_requirements": instruction_requirements,
                    "review_required": True,
                    "review_reason": (
                        "VLM-discovered relation has no script candidate id; require verifier, image alignment, answer consistency, "
                        "and human/LLM judge evidence before auto-accept"
                    ),
                })
        return out

    def _engine_translation_contract(
        self,
        required_tasks: List[Dict[str, Any]],
        deferred_tasks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        items = []
        for task in required_tasks:
            items.append(self._engine_translation_item(task, task_status="required"))
        for task in deferred_tasks:
            items.append(self._engine_translation_item(task, task_status="deferred"))
        return {
            "version": "engine-translation-contract-v0",
            "purpose": "authoritative conversion table from stage-1 VLM relations/targets to current GeoSym builder instructions",
            "rules": [
                "stage 2 may emit build_instructions only for items with translation_decision=emit",
                "copy expected_instructions exactly for emit items unless they contradict stage-1 semantic relations",
                "for non-emit items, do not fabricate points, lines, lengths, or coordinates; record defer/skip/uncertain in instruction_task_decisions",
                "pixel or overlay ids are evidence only and never mathematical coordinates",
                "the verifier, image alignment, and answer consistency remain authoritative after emission",
            ],
            "items": items,
        }

    def _engine_translation_item(self, task: Dict[str, Any], *, task_status: str) -> Dict[str, Any]:
        mapping = task.get("builder_mapping") or {}
        buildability = task.get("relation_buildability") or {}
        builder_decision = task.get("relation_builder_decision") or {}
        requirements = [
            item for item in task.get("instruction_requirements") or []
            if isinstance(item, dict)
        ]
        expected_instructions = [
            dict(item.get("expected_instruction") or {})
            for item in requirements
            if item.get("expected_instruction")
        ]
        evidence_gate = task.get("evidence_gate") or {}
        evidence = ((task.get("provenance") or {}).get("evidence") or {})
        status = str(mapping.get("status") or "")
        decision = str(builder_decision.get("decision") or "")
        required_actions = [str(item) for item in task.get("required_actions") or [] if item]
        block_reasons: List[str] = []
        if task_status != "required":
            block_reasons.append("task is deferred by relation/task packet selection")
        if task.get("blocked_by_relation_builder_decision"):
            block_reasons.append(f"relation_builder_decision={decision or 'missing'} does not allow emission")
        if task.get("missing_relation_builder_decision"):
            block_reasons.append("missing relation_builder_decision")
        if status in {"theorem_only", "not_buildable"}:
            block_reasons.append(f"builder_mapping.status={status}")
        if buildability and not buildability.get("can_generate_build_instruction"):
            block_reasons.append("relation_buildability cannot generate current GeoSym instruction")
        if task.get("object_normalization"):
            block_reasons.append("object normalization failed")
        candidate_id = str(task.get("candidate_id") or "")
        is_vlm_derived = candidate_id.startswith("vlm_rel_")
        if is_vlm_derived and not evidence_gate.get("has_explicit_evidence"):
            block_reasons.append("VLM-derived relation did not pass explicit evidence gate")
        if not expected_instructions:
            block_reasons.append("no expected GeoSym instruction instances")
        can_emit = task_status == "required" and bool(expected_instructions) and not block_reasons
        return {
            "candidate_id": candidate_id,
            "candidate_type": task.get("candidate_type"),
            "candidate_object": task.get("effective_candidate_object") or task.get("candidate_object"),
            "task_status": task_status,
            "translation_decision": "emit" if can_emit else "defer",
            "block_reasons": block_reasons,
            "builder_mapping_status": status,
            "buildability_mode": task.get("buildability_mode") or buildability.get("mode") or mapping.get("buildability_mode"),
            "relation_builder_decision": decision,
            "required_actions": required_actions,
            "expected_instructions": expected_instructions,
            "evidence_gate_passed": bool(evidence_gate.get("has_explicit_evidence")) if is_vlm_derived else None,
            "valid_visual_ids": evidence_gate.get("valid_visual_ids") or evidence.get("visual_ids") or [],
            "invalid_visual_ids": evidence_gate.get("invalid_visual_ids") or evidence.get("invalid_visual_ids") or [],
            "valid_text_refs": evidence_gate.get("valid_text_refs") or evidence.get("text_refs") or [],
            "invalid_text_refs": evidence_gate.get("invalid_text_refs") or evidence.get("invalid_text_refs") or [],
            "review_required": is_vlm_derived,
        }

    def _relation_builder_decision_for_vlm_relation(
        self,
        decisions: Dict[str, Dict[str, Any]],
        *,
        relation: Dict[str, Any],
        relation_index: int,
        generated_candidate_id: str,
    ) -> Dict[str, Any]:
        for key in [
            generated_candidate_id,
            f"vlm_relation[{relation_index}]",
            f"relation[{relation_index}]",
            f"signature::{self._relation_signature(relation)}",
        ]:
            if decisions.get(key):
                return dict(decisions[key])
        relation_signature = self._relation_signature(relation)
        for decision in decisions.values():
            if not isinstance(decision, dict):
                continue
            if decision.get("relation_index") is not None:
                try:
                    if int(decision.get("relation_index")) == relation_index:
                        return dict(decision)
                except (TypeError, ValueError):
                    pass
            if self._decision_signature(decision) == relation_signature:
                return dict(decision)
        return {}

    def _relation_signature(self, relation: Dict[str, Any]) -> str:
        rtype = str(relation.get("type") or "")
        obj = self._candidate_object_from_corrected(relation)
        value = str(relation.get("value") or "")
        return f"{rtype}|{json.dumps(obj, sort_keys=True, ensure_ascii=False)}|{value}"

    def _decision_signature(self, decision: Dict[str, Any]) -> str:
        rtype = str(decision.get("relation_type") or "")
        obj = self._candidate_object_from_corrected(decision)
        value = str(decision.get("value") or "")
        return f"{rtype}|{json.dumps(obj, sort_keys=True, ensure_ascii=False)}|{value}"

    def _relation_evidence_summary(self, relation: Dict[str, Any]) -> Dict[str, Any]:
        text_refs = relation.get("text_refs")
        if isinstance(text_refs, str):
            text_ref_values = [text_refs]
        elif isinstance(text_refs, list):
            text_ref_values = [str(item) for item in text_refs if item]
        else:
            text_evidence = relation.get("text_evidence")
            if isinstance(text_evidence, dict):
                text_ref_values = [str(value) for value in text_evidence.values() if value]
            elif isinstance(text_evidence, list):
                text_ref_values = [str(value) for value in text_evidence if value]
            elif text_evidence:
                text_ref_values = [str(text_evidence)]
            else:
                text_ref_values = []
        return {
            "source": relation.get("source") or "vlm",
            "visual_ids": [str(item) for item in relation.get("visual_ids") or [] if item],
            "text_refs": text_ref_values,
            "grounded": bool(relation.get("grounded")),
            "confidence": relation.get("confidence"),
            "has_visual_evidence": bool(relation.get("visual_ids")),
            "has_text_evidence": bool(text_ref_values),
        }

    def _named_object_binding_visual_index(self, relation_pass: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {}
        for item in relation_pass.get("named_object_bindings") or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("decision") or "").lower() != "bind":
                continue
            obj = self._normalize_named_object_key(item.get("object"))
            if not obj:
                continue
            visual_ids = self._dedupe_list([str(value) for value in item.get("visual_ids") or [] if value])
            if not visual_ids:
                continue
            record = {
                "proposal_id": item.get("proposal_id"),
                "object": obj,
                "kind": item.get("kind"),
                "visual_ids": visual_ids,
                "confidence": item.get("confidence"),
            }
            out.setdefault(obj, []).append(record)
        return out

    def _named_object_binding_evidence(
        self,
        relation: Dict[str, Any],
        *,
        binding_index: Dict[str, List[Dict[str, Any]]],
        valid_visual_ids: set[str],
    ) -> Dict[str, Any]:
        object_keys = self._relation_named_object_keys(relation)
        valid: List[str] = []
        invalid: List[str] = []
        bindings: List[Dict[str, Any]] = []
        for key in object_keys:
            for record in binding_index.get(key) or []:
                record_valid, record_invalid = self._split_valid_visual_ids(record.get("visual_ids") or [], valid_visual_ids)
                valid.extend(record_valid)
                invalid.extend(record_invalid)
                bindings.append({
                    "proposal_id": record.get("proposal_id"),
                    "object": record.get("object"),
                    "kind": record.get("kind"),
                    "visual_ids": record_valid,
                    "invalid_visual_ids": record_invalid,
                    "confidence": record.get("confidence"),
                })
        return {
            "objects": object_keys,
            "visual_ids": self._dedupe_list(valid),
            "invalid_visual_ids": self._dedupe_list(invalid),
            "bindings": [item for item in bindings if item.get("visual_ids") or item.get("invalid_visual_ids")],
        }

    def _relation_named_object_keys(self, relation: Dict[str, Any]) -> List[str]:
        keys: List[str] = []
        for line in self._lines_from_relation(relation):
            if line:
                keys.append(line)
        line = self._line_from_relation(relation)
        if len(line) >= 2:
            keys.append("".join(line[:2]))
            keys.extend(line[:2])
        for key in ["object", "segment", "line", "target"]:
            keys.extend(self._named_object_keys_from_value(relation.get(key)))
        for key in ["objects", "points", "endpoints", "lines", "segments"]:
            keys.extend(self._named_object_keys_from_value(relation.get(key)))
        return self._dedupe_list([item for item in keys if item])

    def _named_object_keys_from_value(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, dict):
            out: List[str] = []
            for key in ["id", "name", "label", "object", "segment", "line", "point", "points", "endpoints"]:
                if key in value:
                    out.extend(self._named_object_keys_from_value(value.get(key)))
            return out
        if isinstance(value, (list, tuple)):
            out: List[str] = []
            if len(value) >= 2 and all(self.object_normalizer.looks_like_point(item) for item in value[:2]):
                line_id = self.object_normalizer.line_id(value[:2])
                if line_id:
                    out.append(line_id)
                out.extend(self._normalize_named_object_key(item) for item in value[:2])
            for item in value:
                out.extend(self._named_object_keys_from_value(item))
            return [item for item in out if item]
        out: List[str] = []
        normalized = self._normalize_named_object_key(value)
        if normalized:
            out.append(normalized)
        for line in self.object_normalizer.line_ids_from_value(value):
            if line:
                out.append(line)
        return [item for item in out if item]

    def _normalize_named_object_key(self, value: Any) -> str:
        return "".join(ch for ch in str(value or "").upper() if ch.isalnum())

    def _dedupe_list(self, values: List[Any]) -> List[Any]:
        out: List[Any] = []
        seen = set()
        for value in values:
            key = str(value)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(value)
        return out

    def _valid_visual_ids_from_evidence(self, evidence_graph: Dict[str, Any]) -> set[str]:
        out: set[str] = set()
        visual = evidence_graph.get("visual_candidates") or {}
        for key in ["lines", "circles", "text_regions", "intersections", "circle_line_hits", "endpoints"]:
            for item in visual.get(key) or []:
                if isinstance(item, dict) and item.get("id"):
                    out.add(str(item["id"]))
        structure = visual.get("visual_structure_candidates") or {}
        for key in ["point_candidates", "segment_candidates", "angle_candidates", "label_region_candidates"]:
            for item in structure.get(key) or []:
                if not isinstance(item, dict):
                    continue
                if item.get("id"):
                    out.add(str(item["id"]))
                if item.get("visual_id"):
                    out.add(str(item["visual_id"]))
                if item.get("source_visual_id"):
                    out.add(str(item["source_visual_id"]))
                if item.get("intersection_visual_id"):
                    out.add(str(item["intersection_visual_id"]))
        for task in evidence_graph.get("grounding_tasks") or []:
            if not isinstance(task, dict):
                continue
            for choice in task.get("visual_choices") or []:
                if not isinstance(choice, dict):
                    continue
                if choice.get("visual_id"):
                    out.add(str(choice["visual_id"]))
                if choice.get("source_visual_id"):
                    out.add(str(choice["source_visual_id"]))
                if choice.get("intersection_visual_id"):
                    out.add(str(choice["intersection_visual_id"]))
        return out

    def _split_valid_visual_ids(self, visual_ids: List[str], valid_visual_ids: set[str]) -> tuple[List[str], List[str]]:
        valid: List[str] = []
        invalid: List[str] = []
        for visual_id in [str(item) for item in visual_ids if item]:
            if visual_id in valid_visual_ids:
                if visual_id not in valid:
                    valid.append(visual_id)
            elif visual_id not in invalid:
                invalid.append(visual_id)
        return valid, invalid

    def _text_corpus_from_evidence(self, evidence_graph: Dict[str, Any]) -> Dict[str, Any]:
        text_evidence = evidence_graph.get("text_evidence") or {}
        texts: List[str] = []
        source_names: set[str] = set()
        for key in ["problem_text", "known_answer"]:
            value = text_evidence.get(key)
            if value is not None and value != "":
                texts.append(str(value))
        for source in text_evidence.get("text_sources") or []:
            if not isinstance(source, dict):
                continue
            if source.get("name"):
                source_names.add(str(source.get("name")))
            if source.get("text"):
                texts.append(str(source.get("text")))
        for key in [
            "labels_mentioned",
            "segments_mentioned",
            "numbers_mentioned",
            "length_claims",
            "angle_claims",
            "area_claims",
            "perimeter_claims",
            "ratio_claims",
            "triangle_claims",
        ]:
            value = text_evidence.get(key)
            if value:
                texts.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
        for key in ["target_hint", "target_angle_hint", "target_area_hint", "target_perimeter_hint", "target_ratio_hint"]:
            value = text_evidence.get(key)
            if value:
                texts.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
        normalized_texts = [self._normalize_text_ref(text) for text in texts if text]
        return {
            "texts": texts,
            "normalized_texts": normalized_texts,
            "source_names": source_names,
        }

    def _split_valid_text_refs(self, text_refs: List[str], text_corpus: Dict[str, Any]) -> tuple[List[str], List[str]]:
        valid: List[str] = []
        invalid: List[str] = []
        source_names = {str(item) for item in text_corpus.get("source_names") or []}
        normalized_texts = [str(item) for item in text_corpus.get("normalized_texts") or [] if item]
        for ref in [str(item).strip() for item in text_refs if str(item).strip()]:
            norm = self._normalize_text_ref(ref)
            if ref in source_names or (self._text_ref_long_enough(norm) and any(norm in text for text in normalized_texts)):
                if ref not in valid:
                    valid.append(ref)
            elif ref not in invalid:
                invalid.append(ref)
        return valid, invalid

    def _normalize_text_ref(self, text: str) -> str:
        return "".join(ch.lower() for ch in str(text) if ch.isalnum())

    def _text_ref_long_enough(self, normalized: str) -> bool:
        if len(normalized) >= 4:
            return True
        return len(normalized) >= 2 and any(ch.isalpha() for ch in normalized) and any(ch.isdigit() for ch in normalized)

    def _vlm_relation_candidate_id(self, idx: int, relation: Dict[str, Any], used_ids: set[str]) -> str:
        rtype = "".join(ch for ch in str(relation.get("type") or "relation").lower() if ch.isalnum()) or "relation"
        obj = self._candidate_object_from_corrected(relation)
        obj_text = "".join(ch for ch in str(obj or "").upper() if ch.isalnum())[:16]
        base = f"vlm_rel_{idx}_{rtype}"
        if obj_text:
            base = f"{base}_{obj_text}"
        candidate_id = base
        suffix = 1
        while candidate_id in used_ids:
            suffix += 1
            candidate_id = f"{base}_{suffix}"
        return candidate_id

    def _requirements_from_instructions(self, candidate_id: str, instructions: List[Dict[str, Any]], *, source: str) -> List[Dict[str, Any]]:
        requirements = []
        for idx, instruction in enumerate(instructions):
            action = str(instruction.get("action") or "")
            if not action:
                continue
            requirements.append({
                "requirement_id": f"{candidate_id}:{source}:{idx}:{action}",
                "action": action,
                "required_fields": [],
                "expected_instruction": instruction,
                "match_rule": {
                    "action": action,
                    "from_candidate": candidate_id,
                    "field_values": {
                        key: instruction.get(key)
                        for key in ["id", "points", "lines", "value", "target"]
                        if key in instruction
                    },
                },
                "source": source,
            })
        return requirements

    def _capability_guide(
        self,
        required_tasks: List[Dict[str, Any]],
        deferred_tasks: List[Dict[str, Any]],
        evidence_graph: Dict[str, Any],
    ) -> Dict[str, Any]:
        builder_contract = evidence_graph.get("builder_contract") or {}
        relation_rules = []
        for item in builder_contract.get("relation_mapping_contract") or []:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status") or "")
            relation_rules.append({
                "relation_type": item.get("relation_type"),
                "status": status,
                "buildability_mode": item.get("buildability_mode"),
                "allowed_builder_actions": item.get("builder_actions") or [],
                "core_constraint_action": item.get("core_constraint_action"),
                "stage2_policy": self._policy_for_mapping(item),
                "note": item.get("note") or "",
            })
        return {
            "purpose": "explicit decision guide for VLM stage-2; follow this before emitting GeoSym build_instructions",
            "global_rules": [
                "emit only actions listed in builder_contract.supported_actions",
                "copy response_template when it agrees with stage-1 relations and this guide",
                "do not emit fake coordinate constraints for theorem_only or not_buildable relations",
                "pixel/overlay ids are visual evidence only, never mathematical coordinates",
                "every required task needs one instruction_task_decisions entry",
                "canonical right-triangle role frame: vertical_leg/height/first_leg -> AB, horizontal_leg/base/second_leg -> BC, hypotenuse/unknown side x -> AC; emit AB/BC/AC instructions instead of role-derived fake point names",
            ],
            "canonical_role_frames": [
                {
                    "name": "right_triangle_unlabeled_roles",
                    "trigger": "relations mention vertical_leg/horizontal_leg/hypotenuse or equivalent role names and a right_angle/perpendicular relation between the legs",
                    "symbolic_frame": {
                        "vertical_leg": "AB",
                        "horizontal_leg": "BC",
                        "hypotenuse": "AC",
                        "right_angle": ["AB", "BC"],
                    },
                    "stage2_policy": "emit add_segment/set_length for AB and BC, set_perpendicular for AB/BC, and add_segment/set_target for AC when it is the requested target; do not emit VE/HO/HY or other role-derived fake point names",
                }
            ],
            "relation_to_instruction_rules": relation_rules,
            "task_decision_guide": [self._task_decision_guide_item(task, required=True) for task in required_tasks],
            "deferred_task_guide": [self._task_decision_guide_item(task, required=False) for task in deferred_tasks],
            "output_policy": {
                "required_tasks": "emit supported expected_instruction instances or explain missing_actions",
                "deferred_tasks": "do not emit builder instructions unless instruction_requirements explicitly list supported scaffolding",
                "instruction_task_decisions": "decision=emit only when emitted_actions cover required_actions; otherwise use skip, defer, or uncertain",
            },
        }

    def _policy_for_mapping(self, mapping: Dict[str, Any]) -> str:
        status = str(mapping.get("status") or "")
        mode = str(mapping.get("buildability_mode") or "")
        if status == "buildable":
            return "emit listed builder_actions when the relation is confirmed and required by the task packet"
        if status == "convertible":
            return "emit listed builder_actions only when required explicit objects are present"
        if status == "theorem_only" or mode == "theorem_only":
            return "extract as relation/formal graph but do not emit coordinate builder actions"
        if status == "not_buildable":
            return "do not emit builder actions; mark task defer or uncertain"
        return "follow required_tasks and response_template; do not invent unsupported actions"

    def _task_decision_guide_item(self, task: Dict[str, Any], *, required: bool) -> Dict[str, Any]:
        cid = str(task.get("candidate_id") or "")
        mapping = task.get("builder_mapping") or {}
        buildability = task.get("relation_buildability") or {}
        builder_decision = task.get("relation_builder_decision") or {}
        requirements = task.get("instruction_requirements") or []
        expected_instructions = [
            dict(item.get("expected_instruction") or {})
            for item in requirements
            if isinstance(item, dict) and item.get("expected_instruction")
        ]
        required_actions = [str(item) for item in task.get("required_actions") or [] if item]
        status = str(mapping.get("status") or "")
        mode = str(task.get("buildability_mode") or buildability.get("mode") or mapping.get("buildability_mode") or "")
        blocked_by_builder_decision = bool(task.get("blocked_by_relation_builder_decision"))
        if required and required_actions:
            recommended_decision = "emit"
            reason = "task is required and current GeoSym builder has expected supported instructions"
        elif blocked_by_builder_decision:
            recommended_decision = "defer"
            reason = "stage-1 relation_builder_decision says this relation should not become a current GeoSym builder constraint"
        elif status in {"theorem_only", "not_buildable"} or mode == "theorem_only":
            recommended_decision = "defer"
            reason = "candidate is theorem-level or not buildable by current canonical builder"
        else:
            recommended_decision = "uncertain"
            reason = "candidate is not selected as a required coordinate-build task"
        return {
            "candidate_id": cid,
            "candidate_type": task.get("candidate_type"),
            "candidate_object": task.get("effective_candidate_object") or task.get("candidate_object"),
            "required": bool(required),
            "relation_type": mapping.get("relation_type") or task.get("candidate_type"),
            "builder_mapping_status": status,
            "relation_builder_decision": builder_decision.get("decision"),
            "relation_builder_decision_reason": builder_decision.get("reason") or "",
            "buildability_mode": mode,
            "required_actions": required_actions,
            "expected_instructions": expected_instructions,
            "recommended_instruction_task_decision": recommended_decision,
            "decision_reason": reason,
            "must_not_emit_fake_constraints": bool(blocked_by_builder_decision or mapping.get("must_not_emit_fake_constraints") or status in {"theorem_only", "not_buildable"} or mode == "theorem_only"),
        }

    def _corrected_instruction_requirements(self, candidate_id: str, decision: Dict[str, Any], task: Dict[str, Any]) -> List[Dict[str, Any]]:
        if str(decision.get("decision") or "") != "correct":
            return []
        corrected = decision.get("corrected_relation")
        if not isinstance(corrected, dict) or not corrected:
            return []
        instructions = self._instructions_from_corrected(candidate_id, corrected, task)
        requirements = []
        for idx, instruction in enumerate(instructions):
            action = str(instruction.get("action") or "")
            requirements.append({
                "requirement_id": f"{candidate_id}:corrected:{idx}:{action}",
                "action": action,
                "required_fields": [],
                "expected_instruction": instruction,
                "match_rule": {
                    "action": action,
                    "from_candidate": candidate_id,
                    "field_values": {
                        key: instruction.get(key)
                        for key in ["id", "points", "lines", "value", "target"]
                        if key in instruction
                    },
                },
                "source": "candidate_decision.corrected_relation",
            })
        return requirements

    def _instructions_from_corrected(self, candidate_id: str, corrected: Dict[str, Any], task: Dict[str, Any]) -> List[Dict[str, Any]]:
        ctype = str(corrected.get("type") or task.get("candidate_type") or "")
        visual_ids = [str(item) for item in corrected.get("visual_ids") or [] if item]
        if task.get("candidate_type") == "target":
            target = dict(corrected)
            target.setdefault("from_candidate", candidate_id)
            target.setdefault("candidate_id", candidate_id)
            target.setdefault("value_symbol", "x")
            return [{"action": "set_target", "target": target, "from_candidate": candidate_id}]
        if ctype == "length":
            line = self._line_from_relation(corrected)
            if not line:
                return []
            a, b = line
            out = [
                {"action": "add_point", "id": a, "from_candidate": candidate_id},
                {"action": "add_point", "id": b, "from_candidate": candidate_id},
                {"action": "add_segment", "points": [a, b], "from_candidate": candidate_id},
            ]
            if corrected.get("value") not in {None, ""}:
                out.append({"action": "set_length", "points": [a, b], "value": str(corrected.get("value")), "from_candidate": candidate_id})
            return self._attach_visual_ids_to_instructions(out, visual_ids)
        if ctype in {"parallel", "perpendicular", "right_angle"}:
            lines = self._lines_from_relation(corrected)
            out = []
            seen_points = set()
            seen_segments = set()
            for line in lines:
                if len(line) < 2:
                    continue
                for point in line[:2]:
                    if point not in seen_points:
                        out.append({"action": "add_point", "id": point, "from_candidate": candidate_id})
                        seen_points.add(point)
                if line not in seen_segments:
                    out.append({"action": "add_segment", "points": [line[0], line[1]], "from_candidate": candidate_id})
                    seen_segments.add(line)
            if len(lines) >= 2:
                action = "set_perpendicular" if ctype == "right_angle" else f"set_{ctype}"
                out.append({"action": action, "lines": lines[:2], "from_candidate": candidate_id})
            return self._attach_visual_ids_to_instructions(out, visual_ids)
        return []

    def _template_target(self, required_tasks: List[Dict[str, Any]], relation_pass: Dict[str, Any]) -> Dict[str, Any]:
        candidates: List[Dict[str, Any]] = []
        if isinstance(relation_pass.get("target"), dict):
            candidates.append(dict(relation_pass["target"]))
        candidates.extend(dict(item) for item in relation_pass.get("targets") or [] if isinstance(item, dict))
        for task in required_tasks:
            if task.get("candidate_type") != "target":
                continue
            for requirement in task.get("instruction_requirements") or []:
                expected = requirement.get("expected_instruction") or {}
                if expected.get("action") != "set_target":
                    continue
                expected_target = dict(expected.get("target") or {})
                if expected_target:
                    expected_target.setdefault("from_candidate", task.get("candidate_id"))
                    candidates.append(expected_target)
        selected = max(candidates, key=self._target_rank) if candidates else {}
        return selected if self._target_is_emit_ready(selected) else {}

    def _target_from_requirements(self, requirements: List[Dict[str, Any]]) -> Dict[str, Any]:
        targets = []
        for requirement in requirements or []:
            if not isinstance(requirement, dict):
                continue
            expected = requirement.get("expected_instruction") or {}
            if isinstance(expected, dict) and expected.get("action") == "set_target" and isinstance(expected.get("target"), dict):
                targets.append(dict(expected["target"]))
        return max(targets, key=self._target_rank) if targets else {}

    def _target_instruction_requirements(self, candidate_id: str, target: Dict[str, Any], *, source: str) -> List[Dict[str, Any]]:
        clean_target = dict(target)
        clean_target.setdefault("from_candidate", candidate_id)
        clean_target.setdefault("candidate_id", candidate_id)
        clean_target.setdefault("value_symbol", "x")
        instruction = {"action": "set_target", "target": clean_target, "from_candidate": candidate_id}
        match_target = {
            key: clean_target.get(key)
            for key in ["type", "object", "value_symbol"]
            if clean_target.get(key) not in {None, ""}
        }
        return [{
            "requirement_id": f"{candidate_id}:{source}:0:set_target",
            "action": "set_target",
            "required_fields": ["target"],
            "expected_instruction": instruction,
            "match_rule": {
                "action": "set_target",
                "from_candidate": candidate_id,
                "field_values": {"target": match_target},
            },
            "source": source,
        }]

    def _target_rank(self, target: Dict[str, Any]) -> tuple[int, int, int, int, int]:
        if not isinstance(target, dict):
            return (0, 0, 0, 0, 0)
        target_type = str(target.get("type") or "").lower()
        line = self.object_normalizer.line_from_relation({"object": target.get("object")})
        source = str(target.get("source") or target.get("repair_source") or "").lower()
        metadata = target.get("metadata") or {}
        return (
            1 if target_type and target_type != "unknown" else 0,
            1 if target_type == "length" else 0,
            1 if len(line) >= 2 else 0,
            1 if metadata.get("semantic_role_canonicalized") else 0,
            0 if source in {"text_regex", "script_build_suggestion"} else 1,
        )

    def _target_is_emit_ready(self, target: Dict[str, Any]) -> bool:
        if not isinstance(target, dict) or not target:
            return False
        target_type = str(target.get("type") or "").lower()
        obj = str(target.get("object") or "").strip()
        if not target_type or target_type == "unknown" or not obj or obj.upper() == "X":
            return False
        if target_type == "length":
            return len(self.object_normalizer.line_from_relation({"object": obj})) >= 2
        return target_type in {"angle", "arc_measure", "area", "perimeter", "ratio"}

    def _template_targets(self, relation_pass: Dict[str, Any], target: Dict[str, Any]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        if target:
            items.append(dict(target))
        for item in relation_pass.get("targets") or []:
            if isinstance(item, dict) and self._target_is_emit_ready(item):
                items.append(dict(item))
        if not items and isinstance(relation_pass.get("target"), dict) and self._target_is_emit_ready(relation_pass.get("target")):
            items.append(dict(relation_pass["target"]))
        out: List[Dict[str, Any]] = []
        seen = set()
        for item in items:
            key = (
                str(item.get("candidate_id") or item.get("from_candidate") or ""),
                str(item.get("type") or ""),
                str(item.get("object") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    def _template_engine_assessment(self, required_tasks: List[Dict[str, Any]], evidence_graph: Dict[str, Any]) -> Dict[str, Any]:
        if required_tasks:
            return {
                "expected_mode": "coordinate_buildable",
                "reason": "required tasks are covered by current GeoSym builder instruction requirements",
            }
        builder_contract = evidence_graph.get("builder_contract") or {}
        hint = builder_contract.get("expected_mode_hint") or "failed"
        return {
            "expected_mode": hint,
            "reason": "no required coordinate-buildable tasks were selected by relation pass",
        }

    def _stable_key(self, item: Dict[str, Any]) -> str:
        return json.dumps(item, sort_keys=True, ensure_ascii=False)

    def _candidate_object_from_corrected(self, corrected: Dict[str, Any]) -> Any:
        if corrected.get("object") is not None:
            return corrected.get("object")
        if corrected.get("lines") is not None:
            return corrected.get("lines")
        if corrected.get("objects") is not None:
            return corrected.get("objects")
        return None

    def _buildability_from_corrected(self, corrected: Dict[str, Any]) -> Dict[str, Any]:
        ctype = str(corrected.get("type") or "")
        if ctype in {"length", "parallel", "perpendicular"}:
            return {
                "mode": "coordinate_constraint",
                "can_generate_build_instruction": True,
                "source": "candidate_decision.corrected_relation",
            }
        if ctype == "right_angle":
            return {
                "mode": "convertible_coordinate_constraint",
                "can_generate_build_instruction": True,
                "source": "candidate_decision.corrected_relation",
            }
        if ctype:
            return {
                "mode": "theorem_only",
                "can_generate_build_instruction": False,
                "source": "candidate_decision.corrected_relation",
            }
        return {}

    def _builder_mapping_from_corrected(self, corrected: Dict[str, Any]) -> Dict[str, Any]:
        ctype = str(corrected.get("type") or "")
        if ctype == "length":
            return {
                "status": "buildable",
                "relation_type": ctype,
                "buildability_mode": "coordinate_constraint",
                "builder_actions": ["add_point", "add_segment", "set_length"],
                "core_constraint_action": "set_length",
                "instruction_requirements_source": "candidate_decision.corrected_relation",
            }
        if ctype in {"parallel", "perpendicular", "right_angle"}:
            core = "set_perpendicular" if ctype == "right_angle" else f"set_{ctype}"
            mode = "convertible_coordinate_constraint" if ctype == "right_angle" else "coordinate_constraint"
            return {
                "status": "buildable",
                "relation_type": ctype,
                "buildability_mode": mode,
                "builder_actions": ["add_point", "add_segment", core],
                "core_constraint_action": core,
                "instruction_requirements_source": "candidate_decision.corrected_relation",
            }
        if ctype:
            return {
                "status": "not_buildable",
                "relation_type": ctype,
                "builder_actions": [],
                "must_not_emit_fake_constraints": True,
                "instruction_requirements_source": "candidate_decision.corrected_relation",
            }
        return {}

    def _line_from_relation(self, rel: Dict[str, Any]) -> List[str]:
        return self.object_normalizer.line_from_relation(rel)

    def _lines_from_relation(self, rel: Dict[str, Any]) -> List[str]:
        return self.object_normalizer.lines_from_relation(rel)

    def _point_id(self, value: Any) -> str:
        return self.object_normalizer.point_id(value)

    def _line_id(self, value: Any) -> str:
        return self.object_normalizer.line_id(value)
