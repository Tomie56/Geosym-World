"""Stage-level contract checks for script-led VLM extraction."""

from __future__ import annotations

from typing import Any, Dict, List, Set

from .builder_mapping_consistency import BuilderMappingConsistencyAnalyzer


class StageContractAnalyzer:
    """Check whether VLM stages honored the evidence and builder contracts."""

    def __init__(self) -> None:
        self.builder_mapping_consistency = BuilderMappingConsistencyAnalyzer()

    def analyze(
        self,
        *,
        evidence_graph: Dict[str, Any],
        instruction_task_packet: Dict[str, Any],
        semantic: Dict[str, Any],
    ) -> Dict[str, Any]:
        checklist_ids = self._checklist_ids(evidence_graph.get("candidate_review_checklist") or [])
        relation_ids = self._candidate_ids(evidence_graph.get("relation_candidates") or [])
        target_ids = self._candidate_ids(evidence_graph.get("target_candidates") or [])
        all_candidate_ids = relation_ids | target_ids
        if checklist_ids:
            all_candidate_ids = checklist_ids
        decisions = self._decision_index(semantic)
        reviewed_ids = set(decisions) & all_candidate_ids
        missing_decision_ids = sorted(all_candidate_ids - reviewed_ids)
        unknown_decision_ids = sorted(set(decisions) - all_candidate_ids)
        grounding_task_ids = self._grounding_task_ids(evidence_graph.get("grounding_tasks") or [])
        grounding_task_choices = self._grounding_task_choice_index(evidence_graph.get("grounding_tasks") or [])
        grounding_decisions = self._grounding_decision_index(semantic)
        covered_grounding_ids = set(grounding_decisions) & grounding_task_ids
        missing_grounding_ids = sorted(grounding_task_ids - covered_grounding_ids)
        unknown_grounding_ids = sorted(set(grounding_decisions) - grounding_task_ids)
        grounding_choice_reports = self._grounding_choice_reports(grounding_decisions, grounding_task_choices)
        invalid_grounding_choice_reports = [item for item in grounding_choice_reports if item.get("invalid_visual_ids")]
        named_proposal_ids = self._named_proposal_ids(((evidence_graph.get("named_object_proposals") or {}).get("items") or []))
        named_proposal_choices = self._named_proposal_choice_index(((evidence_graph.get("named_object_proposals") or {}).get("items") or []))
        named_bindings = self._named_object_binding_index(semantic)
        covered_named_ids = set(named_bindings) & named_proposal_ids
        missing_named_ids = sorted(named_proposal_ids - covered_named_ids)
        unknown_named_ids = sorted(set(named_bindings) - named_proposal_ids)
        named_choice_reports = self._named_binding_choice_reports(named_bindings, named_proposal_choices)
        invalid_named_choice_reports = [item for item in named_choice_reports if item.get("invalid_visual_ids")]
        template = evidence_graph.get("relation_pass_response_template") or {}
        expected_relation_ids = self._refs_from_items(template.get("relations") or [])
        actual_relation_ids = self._refs_from_items(semantic.get("relations") or [])
        missing_relation_ids = sorted(expected_relation_ids - actual_relation_ids)
        expected_buildability_ids = self._relation_ids_requiring_buildability(template.get("relations") or [])
        actual_buildability_ids = self._relation_ids_with_field(semantic.get("relations") or [], "buildability")
        actual_builder_mapping_ids = self._relation_ids_with_field(semantic.get("relations") or [], "builder_mapping")
        missing_buildability_ids = sorted(expected_buildability_ids - actual_buildability_ids)
        missing_builder_mapping_ids = sorted(expected_buildability_ids - actual_builder_mapping_ids)
        builder_mapping_consistency = self.builder_mapping_consistency.analyze(semantic=semantic)
        inconsistent_buildability_reports = builder_mapping_consistency.get("inconsistent_relation_buildability_reports") or []
        expected_target_ids = self._target_refs(template)
        actual_target_ids = self._target_refs(semantic)
        missing_target_ids = sorted(expected_target_ids - actual_target_ids)

        required_tasks = [
            task for task in instruction_task_packet.get("required_tasks") or []
            if isinstance(task, dict) and task.get("candidate_id")
        ]
        instructions = [
            instruction for instruction in semantic.get("build_instructions") or []
            if isinstance(instruction, dict)
        ]
        task_reports = [self._task_report(task, instructions) for task in required_tasks]
        missing_task_reports = [item for item in task_reports if not item.get("covered")]
        instruction_task_decisions = self._instruction_task_decision_index(semantic)
        required_task_ids = {
            str(task.get("candidate_id"))
            for task in required_tasks
            if task.get("candidate_id")
        }
        reviewed_task_decision_ids = set(instruction_task_decisions) & required_task_ids
        missing_task_decision_ids = sorted(required_task_ids - reviewed_task_decision_ids)
        unknown_task_decision_ids = sorted(set(instruction_task_decisions) - required_task_ids)
        task_decision_consistency_reports = self._task_decision_consistency_reports(
            instruction_task_decisions,
            task_reports,
            instructions,
        )
        inconsistent_task_decision_reports = [
            item for item in task_decision_consistency_reports
            if item.get("issues")
        ]

        return {
            "stage1": {
                "num_relation_candidates": len(relation_ids),
                "num_target_candidates": len(target_ids),
                "num_candidates": len(all_candidate_ids),
                "num_reviewed_candidates": len(reviewed_ids),
                "num_missing_candidate_decisions": len(missing_decision_ids),
                "num_unknown_candidate_decisions": len(unknown_decision_ids),
                "num_grounding_tasks": len(grounding_task_ids),
                "num_reviewed_grounding_tasks": len(covered_grounding_ids),
                "num_missing_grounding_decisions": len(missing_grounding_ids),
                "num_unknown_grounding_decisions": len(unknown_grounding_ids),
                "num_invalid_grounding_visual_ids": sum(len(item.get("invalid_visual_ids") or []) for item in invalid_grounding_choice_reports),
                "num_invalid_grounding_decisions": len(invalid_grounding_choice_reports),
                "grounding_decision_coverage": (len(covered_grounding_ids) / len(grounding_task_ids)) if grounding_task_ids else None,
                "num_named_object_proposals": len(named_proposal_ids),
                "num_reviewed_named_object_bindings": len(covered_named_ids),
                "num_missing_named_object_bindings": len(missing_named_ids),
                "num_unknown_named_object_bindings": len(unknown_named_ids),
                "num_invalid_named_object_visual_ids": sum(len(item.get("invalid_visual_ids") or []) for item in invalid_named_choice_reports),
                "num_invalid_named_object_bindings": len(invalid_named_choice_reports),
                "named_object_binding_coverage": (len(covered_named_ids) / len(named_proposal_ids)) if named_proposal_ids else None,
                "num_expected_relation_refs": len(expected_relation_ids),
                "num_covered_relation_refs": len(expected_relation_ids) - len(missing_relation_ids),
                "num_missing_relation_refs": len(missing_relation_ids),
                "relation_ref_coverage": ((len(expected_relation_ids) - len(missing_relation_ids)) / len(expected_relation_ids)) if expected_relation_ids else None,
                "num_expected_relation_buildability_refs": len(expected_buildability_ids),
                "num_missing_relation_buildability_refs": len(missing_buildability_ids),
                "num_missing_relation_builder_mapping_refs": len(missing_builder_mapping_ids),
                "num_inconsistent_relation_buildability": builder_mapping_consistency.get("num_inconsistent_relation_buildability", 0),
                "relation_buildability_coverage": ((len(expected_buildability_ids) - len(missing_buildability_ids)) / len(expected_buildability_ids)) if expected_buildability_ids else None,
                "relation_builder_mapping_coverage": ((len(expected_buildability_ids) - len(missing_builder_mapping_ids)) / len(expected_buildability_ids)) if expected_buildability_ids else None,
                "num_expected_target_refs": len(expected_target_ids),
                "num_covered_target_refs": len(expected_target_ids) - len(missing_target_ids),
                "num_missing_target_refs": len(missing_target_ids),
                "target_ref_coverage": ((len(expected_target_ids) - len(missing_target_ids)) / len(expected_target_ids)) if expected_target_ids else None,
                "candidate_decision_coverage": (len(reviewed_ids) / len(all_candidate_ids)) if all_candidate_ids else None,
                "missing_candidate_decision_ids": missing_decision_ids[:120],
                "unknown_candidate_decision_ids": unknown_decision_ids[:120],
                "missing_grounding_decision_ids": missing_grounding_ids[:120],
                "unknown_grounding_decision_ids": unknown_grounding_ids[:120],
                "invalid_grounding_choice_reports": invalid_grounding_choice_reports[:120],
                "missing_named_object_binding_ids": missing_named_ids[:120],
                "unknown_named_object_binding_ids": unknown_named_ids[:120],
                "invalid_named_object_binding_reports": invalid_named_choice_reports[:120],
                "missing_relation_ref_ids": missing_relation_ids[:120],
                "missing_relation_buildability_ids": missing_buildability_ids[:120],
                "missing_relation_builder_mapping_ids": missing_builder_mapping_ids[:120],
                "inconsistent_relation_buildability_reports": inconsistent_buildability_reports[:120],
                "missing_target_ref_ids": missing_target_ids[:120],
            },
            "stage2": {
                "num_required_tasks": len(required_tasks),
                "num_covered_required_tasks": len(task_reports) - len(missing_task_reports),
                "num_missing_required_tasks": len(missing_task_reports),
                "required_task_coverage": ((len(task_reports) - len(missing_task_reports)) / len(task_reports)) if task_reports else None,
                "num_instruction_task_decisions": len(instruction_task_decisions),
                "num_reviewed_instruction_task_decisions": len(reviewed_task_decision_ids),
                "num_missing_instruction_task_decisions": len(missing_task_decision_ids),
                "num_unknown_instruction_task_decisions": len(unknown_task_decision_ids),
                "num_inconsistent_instruction_task_decisions": len(inconsistent_task_decision_reports),
                "instruction_task_decision_coverage": (len(reviewed_task_decision_ids) / len(required_task_ids)) if required_task_ids else None,
                "missing_instruction_task_decision_ids": missing_task_decision_ids[:120],
                "unknown_instruction_task_decision_ids": unknown_task_decision_ids[:120],
                "inconsistent_instruction_task_decision_reports": inconsistent_task_decision_reports[:120],
                "task_reports": task_reports[:120],
                "missing_task_reports": missing_task_reports[:120],
            },
            "valid": (
                not missing_decision_ids
                and not unknown_decision_ids
                and not missing_grounding_ids
                and not unknown_grounding_ids
                and not invalid_grounding_choice_reports
                and not missing_named_ids
                and not unknown_named_ids
                and not invalid_named_choice_reports
                and not missing_relation_ids
                and not missing_buildability_ids
                and not missing_builder_mapping_ids
                and not inconsistent_buildability_reports
                and not missing_target_ids
                and not missing_task_reports
                and not missing_task_decision_ids
                and not unknown_task_decision_ids
                and not inconsistent_task_decision_reports
            ),
        }

    def _candidate_ids(self, items: List[Dict[str, Any]]) -> Set[str]:
        return {
            str(item.get("id"))
            for item in items
            if isinstance(item, dict) and item.get("id")
        }

    def _checklist_ids(self, items: List[Dict[str, Any]]) -> Set[str]:
        return {
            str(item.get("candidate_id"))
            for item in items
            if isinstance(item, dict) and item.get("candidate_id")
        }

    def _decision_index(self, semantic: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for item in semantic.get("candidate_decisions") or []:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("candidate_id") or "")
            if cid:
                out[cid] = item
        return out

    def _grounding_task_ids(self, items: List[Dict[str, Any]]) -> Set[str]:
        return {
            str(item.get("id"))
            for item in items
            if isinstance(item, dict) and item.get("id")
        }

    def _grounding_decision_index(self, semantic: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for item in semantic.get("grounding_decisions") or []:
            if not isinstance(item, dict):
                continue
            task_id = str(item.get("task_id") or "")
            if task_id:
                out[task_id] = item
        return out

    def _grounding_task_choice_index(self, tasks: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
        out: Dict[str, Set[str]] = {}
        for task in tasks:
            if not isinstance(task, dict):
                continue
            task_id = str(task.get("id") or "")
            if not task_id:
                continue
            choices: Set[str] = set()
            for choice in task.get("visual_choices") or []:
                if not isinstance(choice, dict):
                    continue
                if choice.get("visual_id"):
                    choices.add(str(choice.get("visual_id")))
                if choice.get("source_visual_id"):
                    choices.add(str(choice.get("source_visual_id")))
            out[task_id] = choices
        return out

    def _named_proposal_ids(self, items: List[Dict[str, Any]]) -> Set[str]:
        return {
            str(item.get("id"))
            for item in items
            if isinstance(item, dict) and item.get("id")
        }

    def _named_object_binding_index(self, semantic: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for item in semantic.get("named_object_bindings") or []:
            if not isinstance(item, dict):
                continue
            proposal_id = str(item.get("proposal_id") or "")
            if proposal_id:
                out[proposal_id] = item
        return out

    def _named_proposal_choice_index(self, proposals: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
        out: Dict[str, Set[str]] = {}
        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue
            proposal_id = str(proposal.get("id") or "")
            if not proposal_id:
                continue
            choices: Set[str] = set()
            for choice in proposal.get("visual_choices") or []:
                if not isinstance(choice, dict):
                    continue
                if choice.get("visual_id"):
                    choices.add(str(choice.get("visual_id")))
                if choice.get("source_visual_id"):
                    choices.add(str(choice.get("source_visual_id")))
                if choice.get("visual_source_id"):
                    choices.add(str(choice.get("visual_source_id")))
            out[proposal_id] = choices
        return out

    def _named_binding_choice_reports(
        self,
        bindings: Dict[str, Dict[str, Any]],
        proposal_choices: Dict[str, Set[str]],
    ) -> List[Dict[str, Any]]:
        reports: List[Dict[str, Any]] = []
        for proposal_id, binding in sorted(bindings.items()):
            if proposal_id not in proposal_choices:
                continue
            visual_ids = [str(item) for item in binding.get("visual_ids") or [] if item]
            allowed = proposal_choices.get(proposal_id) or set()
            invalid = sorted({vid for vid in visual_ids if vid not in allowed})
            reports.append({
                "proposal_id": proposal_id,
                "object": binding.get("object"),
                "kind": binding.get("kind"),
                "decision": binding.get("decision"),
                "visual_ids": visual_ids,
                "num_allowed_visual_ids": len(allowed),
                "invalid_visual_ids": invalid,
            })
        return reports

    def _grounding_choice_reports(
        self,
        decisions: Dict[str, Dict[str, Any]],
        task_choices: Dict[str, Set[str]],
    ) -> List[Dict[str, Any]]:
        reports: List[Dict[str, Any]] = []
        for task_id, decision in sorted(decisions.items()):
            if task_id not in task_choices:
                continue
            visual_ids = [str(item) for item in decision.get("visual_ids") or [] if item]
            allowed = task_choices.get(task_id) or set()
            invalid = sorted({vid for vid in visual_ids if vid not in allowed})
            reports.append({
                "task_id": task_id,
                "candidate_id": decision.get("candidate_id"),
                "decision": decision.get("decision"),
                "visual_ids": visual_ids,
                "num_allowed_visual_ids": len(allowed),
                "invalid_visual_ids": invalid,
            })
        return reports

    def _instruction_task_decision_index(self, semantic: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for item in semantic.get("instruction_task_decisions") or []:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("candidate_id") or "")
            if cid:
                out[cid] = item
        return out

    def _task_decision_consistency_reports(
        self,
        decisions: Dict[str, Dict[str, Any]],
        task_reports: List[Dict[str, Any]],
        instructions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        reports: List[Dict[str, Any]] = []
        task_by_id = {
            str(task.get("candidate_id")): task
            for task in task_reports
            if task.get("candidate_id")
        }
        for cid, decision in sorted(decisions.items()):
            if cid not in task_by_id:
                continue
            task = task_by_id[cid]
            decision_value = str(decision.get("decision") or "").lower()
            actual_actions = self._covered_actions_for_candidate(cid, instructions)
            emitted_actions = {
                str(item)
                for item in decision.get("emitted_actions") or []
                if item
            }
            missing_actions_claimed = {
                str(item)
                for item in decision.get("missing_actions") or []
                if item
            }
            task_missing_actions = {
                str(item)
                for item in task.get("missing_instruction_requirements") or []
                if item.get("action")
            }
            task_missing_actions.update(str(item) for item in task.get("missing_actions") or [] if item)
            issues: List[Dict[str, Any]] = []
            if decision_value == "emit" and not task.get("covered"):
                issues.append({
                    "code": "emit_but_task_not_covered",
                    "missing_actions": sorted(task_missing_actions),
                })
            if decision_value in {"skip", "defer", "uncertain"} and actual_actions:
                issues.append({
                    "code": "non_emit_but_instructions_present",
                    "decision": decision_value,
                    "actual_actions": sorted(actual_actions),
                })
            missing_emitted = sorted(emitted_actions - actual_actions)
            if missing_emitted:
                issues.append({
                    "code": "emitted_actions_not_present",
                    "emitted_actions": sorted(emitted_actions),
                    "actual_actions": sorted(actual_actions),
                    "missing_emitted_actions": missing_emitted,
                })
            if decision_value == "emit" and task.get("covered") and missing_actions_claimed:
                issues.append({
                    "code": "emit_covered_but_missing_actions_claimed",
                    "missing_actions": sorted(missing_actions_claimed),
                })
            reports.append({
                "candidate_id": cid,
                "decision": decision_value,
                "covered": bool(task.get("covered")),
                "actual_actions": sorted(actual_actions),
                "emitted_actions": sorted(emitted_actions),
                "missing_actions_claimed": sorted(missing_actions_claimed),
                "issues": issues,
            })
        return reports

    def _covered_actions_for_candidate(self, candidate_id: str, instructions: List[Dict[str, Any]]) -> Set[str]:
        actions: Set[str] = set()
        for instruction in instructions:
            if self._refs_candidate(instruction, candidate_id):
                action = str(instruction.get("action") or "")
                if action:
                    actions.add(action)
        return actions

    def _refs_candidate(self, item: Dict[str, Any], candidate_id: str) -> bool:
        if not isinstance(item, dict):
            return False
        for key in ["candidate_id", "from_candidate"]:
            if str(item.get(key) or "") == candidate_id:
                return True
        for key in ["candidate_ids", "from_candidates"]:
            value = item.get(key)
            if isinstance(value, list) and candidate_id in {str(part) for part in value}:
                return True
            if value and str(value) == candidate_id:
                return True
        target = item.get("target")
        if isinstance(target, dict):
            return self._refs_candidate(target, candidate_id)
        return False

    def _refs_from_items(self, items: List[Dict[str, Any]]) -> Set[str]:
        refs: Set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
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
                refs.update(self._refs_from_items([target]))
        return refs

    def _target_refs(self, semantic: Dict[str, Any]) -> Set[str]:
        items: List[Dict[str, Any]] = []
        target = semantic.get("target")
        if isinstance(target, dict):
            items.append(target)
        targets = semantic.get("targets")
        if isinstance(targets, list):
            items.extend(item for item in targets if isinstance(item, dict))
        return self._refs_from_items(items)

    def _relation_ids_requiring_buildability(self, items: List[Dict[str, Any]]) -> Set[str]:
        refs: Set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            cid = item.get("candidate_id") or item.get("from_candidate")
            if cid and (item.get("buildability") or item.get("builder_mapping")):
                refs.add(str(cid))
        return refs

    def _relation_ids_with_field(self, items: List[Dict[str, Any]], field: str) -> Set[str]:
        refs: Set[str] = set()
        for item in items:
            if not isinstance(item, dict) or not item.get(field):
                continue
            cid = item.get("candidate_id") or item.get("from_candidate")
            if cid:
                refs.add(str(cid))
        return refs

    def _task_report(self, task: Dict[str, Any], instructions: List[Dict[str, Any]]) -> Dict[str, Any]:
        cid = str(task.get("candidate_id") or "")
        expected_requirements = [
            requirement for requirement in task.get("instruction_requirements") or []
            if isinstance(requirement, dict)
        ]
        requirement_reports = [
            self._requirement_report(requirement, instructions)
            for requirement in expected_requirements
        ]
        missing_requirements = [item for item in requirement_reports if not item.get("covered")]
        return {
            "candidate_id": cid,
            "candidate_type": task.get("candidate_type"),
            "decision": task.get("decision"),
            "num_instruction_requirements": len(requirement_reports),
            "num_covered_instruction_requirements": len(requirement_reports) - len(missing_requirements),
            "num_missing_instruction_requirements": len(missing_requirements),
            "covered": not missing_requirements,
            "missing_instruction_requirements": missing_requirements[:50],
        }

    def _requirement_report(self, requirement: Dict[str, Any], instructions: List[Dict[str, Any]]) -> Dict[str, Any]:
        expected = requirement.get("expected_instruction") or {}
        matches = [
            idx for idx, instruction in enumerate(instructions)
            if self._instruction_matches(instruction, expected)
        ]
        return {
            "requirement_id": requirement.get("requirement_id"),
            "action": requirement.get("action") or expected.get("action"),
            "expected_instruction": expected,
            "covered": bool(matches),
            "matched_instruction_indices": matches[:10],
        }

    def _instruction_matches(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        if not expected:
            return False
        for key, value in expected.items():
            if key == "supported_by_current_builder":
                continue
            if actual.get(key) != value:
                return False
        return True
