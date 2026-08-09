"""Audit whether VLM output follows the script-led GeoSym build contract."""

from __future__ import annotations

from typing import Any, Dict, List

from .task_packet_coverage import InstructionTaskPacketCoverageAnalyzer


class ExtractionContractAuditor:
    """Combine candidate, instruction, and engine checks into one verdict."""

    def __init__(self):
        self.task_packet_coverage = InstructionTaskPacketCoverageAnalyzer()

    def audit(
        self,
        *,
        evidence_graph: Dict[str, Any],
        semantic_validation: Dict[str, Any],
        candidate_usage: Dict[str, Any],
        instruction_coverage: Dict[str, Any],
        instruction_plan: Dict[str, Any],
        engine_assessment: Dict[str, Any],
    ) -> Dict[str, Any]:
        warnings: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        relation_candidates = evidence_graph.get("relation_candidates") or []
        buildable_candidates = [
            item for item in relation_candidates
            if (item.get("buildability") or {}).get("can_generate_build_instruction")
        ]

        if semantic_validation.get("valid") is False:
            warnings.append(self._issue(
                "contract.semantic_schema_warning",
                "VLM output had schema warnings before grounding",
                count=(semantic_validation.get("metrics") or {}).get("semantic_schema_warnings", 0),
            ))
        if candidate_usage.get("num_unknown_candidate_refs", 0) > 0:
            warnings.append(self._issue(
                "contract.unknown_candidate_reference",
                "VLM referenced candidate ids that were not produced by scripts",
                refs=candidate_usage.get("unknown_candidate_refs") or [],
            ))
        if buildable_candidates and candidate_usage.get("num_confirmed_candidates", 0) == 0:
            warnings.append(self._issue(
                "contract.script_candidates_not_used",
                "Scripts proposed buildable semantic candidates, but VLM did not confirm any of them",
                num_buildable_candidates=len(buildable_candidates),
            ))
        if candidate_usage.get("num_unreviewed_candidates", 0) > 0:
            warnings.append(self._issue(
                "contract.candidate_unreviewed",
                "Some script candidates were neither confirmed, corrected, rejected, nor marked uncertain by the VLM",
                count=candidate_usage.get("num_unreviewed_candidates", 0),
                candidate_ids=candidate_usage.get("unreviewed_candidate_ids") or [],
            ))
        if candidate_usage.get("num_decision_conflicts", 0) > 0:
            warnings.append(self._issue(
                "contract.candidate_decision_conflict",
                "Some candidates were rejected but also used by VLM relations or instructions",
                candidate_ids=candidate_usage.get("decision_conflict_candidate_ids") or [],
            ))
        if instruction_coverage.get("num_missing_constraints", 0) > 0:
            errors.append(self._issue(
                "contract.missing_build_instruction",
                "Buildable relations were not covered by accepted GeoSym constraints",
                missing_constraints=instruction_coverage.get("missing_constraints") or [],
            ))
        if instruction_plan.get("num_rejected", 0) > 0:
            warnings.append(self._issue(
                "contract.rejected_build_instruction",
                "Some VLM build instructions were rejected by the GeoSym planner",
                num_rejected=instruction_plan.get("num_rejected", 0),
            ))
        if buildable_candidates and instruction_plan.get("num_synthesized", 0) > 0 and instruction_plan.get("num_instructions", 0) == 0:
            warnings.append(self._issue(
                "contract.vlm_instruction_missing_but_synthesized",
                "VLM did not emit build instructions; GeoSym synthesized them from grounded relations",
                num_synthesized=instruction_plan.get("num_synthesized", 0),
            ))
        if engine_assessment.get("consistent") is False:
            warnings.append(self._issue(
                "contract.engine_assessment_mismatch",
                "VLM expected mode differs from GeoSym formalizer/planner result",
                expected_mode=engine_assessment.get("expected_mode"),
                actual_mode=engine_assessment.get("actual_mode"),
            ))

        build_suggestions = evidence_graph.get("build_suggestions") or []
        unsupported_suggestions = [item for item in build_suggestions if not item.get("supported_by_current_builder", False)]
        if unsupported_suggestions:
            warnings.append(self._issue(
                "contract.unsupported_script_suggestion",
                "Script produced build suggestions outside the current builder contract",
                count=len(unsupported_suggestions),
            ))

        task_coverage = self._build_task_coverage(evidence_graph, candidate_usage, instruction_plan)
        for missing in task_coverage["missing_tasks"]:
            warnings.append(self._issue(
                "contract.build_task_missing_instruction",
                "A confirmed/corrected script candidate was not fully converted into candidate-linked build instructions",
                candidate_id=missing.get("candidate_id"),
                missing_actions=missing.get("missing_actions") or [],
                required_actions=missing.get("required_actions") or [],
                missing_instruction_requirements=missing.get("num_missing_instruction_requirements", 0),
            ))

        status = "pass"
        if errors:
            status = "fail"
        elif warnings:
            status = "warn"
        return {
            "status": status,
            "valid": not errors,
            "num_errors": len(errors),
            "num_warnings": len(warnings),
            "errors": errors,
            "warnings": warnings,
            "metrics": {
                "buildable_script_candidates": len(buildable_candidates),
                "confirmed_script_candidates": int(candidate_usage.get("num_confirmed_candidates") or 0),
                "corrected_script_candidates": int(candidate_usage.get("num_corrected_candidates") or 0),
                "rejected_script_candidates": int(candidate_usage.get("num_rejected_candidates") or 0),
                "unreviewed_script_candidates": int(candidate_usage.get("num_unreviewed_candidates") or 0),
                "candidate_decision_conflicts": int(candidate_usage.get("num_decision_conflicts") or 0),
                "unknown_candidate_refs": int(candidate_usage.get("num_unknown_candidate_refs") or 0),
                "missing_build_constraints": int(instruction_coverage.get("num_missing_constraints") or 0),
                "rejected_build_instructions": int(instruction_plan.get("num_rejected") or 0),
                "synthesized_build_instructions": int(instruction_plan.get("num_synthesized") or 0),
                "required_build_tasks": task_coverage["num_required_tasks"],
                "covered_build_tasks": task_coverage["num_covered_tasks"],
                "missing_build_tasks": task_coverage["num_missing_tasks"],
                "required_instruction_requirements": task_coverage.get("num_instruction_requirements", 0),
                "covered_instruction_requirements": task_coverage.get("num_covered_instruction_requirements", 0),
                "missing_instruction_requirements": task_coverage.get("num_missing_instruction_requirements", 0),
                "engine_assessment_consistent": int(engine_assessment.get("consistent") is True),
            },
        }

    def _build_task_coverage(
        self,
        evidence_graph: Dict[str, Any],
        candidate_usage: Dict[str, Any],
        instruction_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        decisions = {
            str(item.get("candidate_id")): str(item.get("decision") or "")
            for item in candidate_usage.get("candidate_decisions") or []
            if item.get("candidate_id")
        }
        required_tasks = []
        for task in evidence_graph.get("build_tasks") or []:
            cid = str(task.get("candidate_id") or "")
            if not cid:
                continue
            decision = decisions.get(cid)
            if decision not in set(task.get("must_emit_when_decision") or []):
                continue
            if not (task.get("required_actions") or task.get("instruction_requirements")):
                continue
            item = dict(task)
            item["decision"] = decision
            required_tasks.append(item)
        packet = {"required_tasks": required_tasks}
        return self.task_packet_coverage.analyze(
            packet=packet,
            semantic={"build_instructions": instruction_plan.get("accepted_instructions") or []},
        )

    def _issue(self, code: str, message: str, **metadata: Any) -> Dict[str, Any]:
        return {
            "code": code,
            "message": message,
            "metadata": metadata,
        }
