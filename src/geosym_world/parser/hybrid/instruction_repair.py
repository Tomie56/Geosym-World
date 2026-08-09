"""Repair missing builder instructions from confirmed script candidates."""

from __future__ import annotations

from typing import Any, Dict, List, Set
import json

from .builder_capabilities import GEOSYM_2D_BUILDER_CAPABILITIES, GeoSymBuilderCapabilities
from .relation_object_normalizer import RelationObjectNormalizer


class InstructionRepairer:
    """Fill build-task gaps with script suggestions after VLM semantic parsing.

    This is not a replacement for VLM extraction. It only adds builder
    instructions for candidates that the semantic output already confirmed,
    corrected, or used. The resulting instructions are marked as script repair so
    reports can distinguish VLM-native conversion from framework repair.
    """

    def __init__(self, capabilities: GeoSymBuilderCapabilities = GEOSYM_2D_BUILDER_CAPABILITIES):
        self.capabilities = capabilities
        self.object_normalizer = RelationObjectNormalizer()

    def repair(
        self,
        *,
        semantic: Dict[str, Any],
        evidence_graph: Dict[str, Any],
        candidate_usage: Dict[str, Any],
        instruction_task_packet: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        original = list(semantic.get("build_instructions") or [])
        confirmed_ids = self._repairable_candidate_ids(candidate_usage)
        corrected_ids = set(str(item) for item in candidate_usage.get("corrected_candidate_ids") or [] if item)
        task_decisions = self._instruction_task_decisions(semantic)
        relation_builder_blocked = self._relation_builder_blocked_tasks(instruction_task_packet or {})
        semantic_suggestions = self._semantic_build_suggestions(semantic)
        existing_by_candidate = self._existing_actions(original)
        additions: List[Dict[str, Any]] = []
        reasons: List[Dict[str, Any]] = []
        blocked: List[Dict[str, Any]] = []
        repaired_vlm_ids: Set[str] = set()

        for task in self._repair_tasks(evidence_graph=evidence_graph, instruction_task_packet=instruction_task_packet or {}):
            cid = str(task.get("candidate_id") or "")
            is_vlm_derived = self._is_vlm_derived_candidate(cid)
            if not cid or (cid not in confirmed_ids and not is_vlm_derived):
                continue
            relation_builder_decision = relation_builder_blocked.get(cid)
            if relation_builder_decision:
                blocked.append({
                    "candidate_id": cid,
                    "decision": relation_builder_decision.get("decision") or "defer",
                    "reason": relation_builder_decision.get("reason") or "relation_builder_decision blocked automatic repair",
                    "missing_actions": relation_builder_decision.get("missing_actions") or task.get("required_actions") or [],
                    "blocked_by_relation_builder_decision": True,
                    "relation_builder_decision": relation_builder_decision.get("relation_builder_decision") or {},
                })
                continue
            task_decision = task_decisions.get(cid)
            if task_decision and task_decision.get("decision") in {"skip", "defer", "uncertain"}:
                blocked.append({
                    "candidate_id": cid,
                    "decision": task_decision.get("decision"),
                    "reason": task_decision.get("reason") or "stage-2 instruction_task_decision blocked automatic repair",
                    "missing_actions": task_decision.get("missing_actions") or [],
                })
                continue
            semantic_suggestions_for_candidate = [] if task.get("candidate_type") == "target" else (semantic_suggestions.get(cid) or [])
            semantic_requirements = self._requirements_from_semantic_suggestions(cid, semantic_suggestions_for_candidate)
            requirements = semantic_requirements or [item for item in task.get("instruction_requirements") or [] if isinstance(item, dict)]
            required = [str(action) for action in task.get("required_actions") or [] if action]
            if not requirements and not required:
                continue
            task_additions = self._additions_for_requirements(requirements, original + additions, cid)
            if not task_additions:
                existing = existing_by_candidate.get(cid, set())
                missing = [action for action in required if action not in existing]
                suggestions = semantic_suggestions_for_candidate or task.get("suggested_instructions") or []
                for suggestion in suggestions:
                    action = str(suggestion.get("action") or "")
                    if action not in missing:
                        continue
                    repaired = self._normalize_repair_instruction(suggestion, cid)
                    if repaired:
                        task_additions.append(repaired)
            if task_additions:
                if is_vlm_derived:
                    repaired_vlm_ids.add(cid)
                missing = sorted({item.get("action") for item in task_additions if item.get("action")})
                additions.extend(task_additions)
                reasons.append({
                    "candidate_id": cid,
                    "source": "instruction_task_packet" if is_vlm_derived else "evidence_graph",
                    "missing_actions": missing,
                    "added_actions": [item["action"] for item in task_additions],
                    "added_instruction_requirements": [
                        item.get("repair_requirement_id")
                        for item in task_additions
                        if item.get("repair_requirement_id")
                    ],
                })

        repaired_semantic = dict(semantic)
        repaired_semantic["build_instructions"] = self._dedupe(original + additions)
        metadata = dict(repaired_semantic.get("metadata") or {})
        metadata["instruction_repair"] = {
            "used": bool(additions),
            "num_added": len(additions),
            "num_blocked_by_task_decision": len(blocked),
            "num_blocked_by_relation_builder_decision": sum(1 for item in blocked if item.get("blocked_by_relation_builder_decision")),
            "num_vlm_derived_candidates": len(repaired_vlm_ids),
            "source": "task_requirements_for_confirmed_or_vlm_derived_candidates",
        }
        repaired_semantic["metadata"] = metadata
        return {
            "used": bool(additions),
            "num_original_instructions": len(original),
            "num_added_instructions": len(additions),
            "num_final_instructions": len(repaired_semantic["build_instructions"]),
            "confirmed_candidate_ids": sorted(confirmed_ids),
            "vlm_derived_candidate_ids": sorted(repaired_vlm_ids),
            "additions": additions,
            "reasons": reasons,
            "blocked": blocked,
            "semantic": repaired_semantic,
        }

    def _relation_builder_blocked_tasks(self, instruction_task_packet: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for task in instruction_task_packet.get("deferred_tasks") or []:
            if not isinstance(task, dict):
                continue
            cid = str(task.get("candidate_id") or "")
            if not cid or not task.get("blocked_by_relation_builder_decision"):
                continue
            relation_builder_decision = dict(task.get("relation_builder_decision") or {})
            out[cid] = {
                "decision": "defer",
                "reason": (
                    relation_builder_decision.get("reason")
                    or task.get("reason")
                    or "relation_builder_decision marked this relation non-buildable for current GeoSym builder"
                ),
                "missing_actions": task.get("required_actions") or [],
                "relation_builder_decision": relation_builder_decision,
            }
        return out

    def _repair_tasks(self, *, evidence_graph: Dict[str, Any], instruction_task_packet: Dict[str, Any]) -> List[Dict[str, Any]]:
        tasks: List[Dict[str, Any]] = [
            item for item in evidence_graph.get("build_tasks") or []
            if isinstance(item, dict)
        ]
        seen = {str(item.get("candidate_id") or "") for item in tasks if item.get("candidate_id")}
        for task in instruction_task_packet.get("required_tasks") or []:
            if not isinstance(task, dict):
                continue
            cid = str(task.get("candidate_id") or "")
            if not self._is_vlm_derived_candidate(cid) or cid in seen:
                continue
            tasks.append(task)
            seen.add(cid)
        return tasks

    def _is_vlm_derived_candidate(self, candidate_id: str) -> bool:
        return str(candidate_id or "").startswith("vlm_rel_")

    def _instruction_task_decisions(self, semantic: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for item in semantic.get("instruction_task_decisions") or []:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("candidate_id") or "")
            if not cid:
                continue
            decision = dict(item)
            decision["decision"] = str(decision.get("decision") or "").lower()
            out[cid] = decision
        return out

    def _additions_for_requirements(
        self,
        requirements: List[Dict[str, Any]],
        instructions: List[Dict[str, Any]],
        candidate_id: str,
    ) -> List[Dict[str, Any]]:
        additions = []
        for requirement in requirements:
            expected = dict(requirement.get("expected_instruction") or {})
            if not expected:
                continue
            if any(self._matches_expected_instruction(inst, expected, candidate_id) for inst in instructions + additions):
                continue
            repaired = self._normalize_repair_instruction(expected, candidate_id)
            if repaired:
                repaired["repair_requirement_id"] = requirement.get("requirement_id")
                additions.append(repaired)
        return additions

    def _requirements_from_semantic_suggestions(self, candidate_id: str, suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        requirements = []
        for idx, suggestion in enumerate(suggestions):
            action = str(suggestion.get("action") or "")
            if action not in self.capabilities.supported_actions:
                continue
            requirements.append({
                "requirement_id": f"{candidate_id}:semantic:{idx}:{action}",
                "action": action,
                "expected_instruction": suggestion,
            })
        return requirements

    def _semantic_build_suggestions(self, semantic: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {}
        declared_points: Dict[str, Set[str]] = {}
        declared_segments: Dict[str, Set[str]] = {}

        def add(cid: str, instruction: Dict[str, Any]) -> None:
            if not cid:
                return
            inst = dict(instruction)
            inst.setdefault("from_candidate", cid)
            inst["repair_source"] = "semantic_relation"
            out.setdefault(cid, []).append(inst)

        def add_point(cid: str, point: str) -> None:
            point = self._point_id(point)
            if not point:
                return
            if point in declared_points.setdefault(cid, set()):
                return
            declared_points[cid].add(point)
            add(cid, {"action": "add_point", "id": point})

        def add_segment(cid: str, a: str, b: str, visual_ids: List[str] | None = None) -> None:
            a = self._point_id(a)
            b = self._point_id(b)
            if not a or not b:
                return
            key = "".join(sorted([a, b]))
            if key in declared_segments.setdefault(cid, set()):
                return
            declared_segments[cid].add(key)
            inst: Dict[str, Any] = {"action": "add_segment", "points": [a, b]}
            if visual_ids:
                inst["visual_ids"] = list(visual_ids)
            add(cid, inst)

        for rel in semantic.get("relations") or []:
            if not isinstance(rel, dict):
                continue
            cid = str(rel.get("candidate_id") or rel.get("from_candidate") or "")
            rtype = str(rel.get("type") or "")
            visual_ids = [str(item) for item in rel.get("visual_ids") or [] if item]
            if rtype == "length":
                line = self._line_from_relation(rel)
                if not line:
                    continue
                a, b = line
                add_point(cid, a)
                add_point(cid, b)
                add_segment(cid, a, b, visual_ids)
                if rel.get("value") not in {None, ""}:
                    inst = {"action": "set_length", "points": [a, b], "value": str(rel.get("value"))}
                    if visual_ids:
                        inst["visual_ids"] = visual_ids
                    add(cid, inst)
            elif rtype in {"parallel", "perpendicular", "right_angle"}:
                lines = self._lines_from_relation(rel)
                for line in lines:
                    if len(line) >= 2:
                        add_point(cid, line[0])
                        add_point(cid, line[1])
                        add_segment(cid, line[0], line[1], visual_ids)
                if len(lines) >= 2:
                    action = "set_perpendicular" if rtype == "right_angle" else f"set_{rtype}"
                    inst = {"action": action, "lines": lines[:2]}
                    if visual_ids:
                        inst["visual_ids"] = visual_ids
                    add(cid, inst)

        target = semantic.get("target") if isinstance(semantic.get("target"), dict) else {}
        target_cid = str(target.get("candidate_id") or target.get("from_candidate") or "")
        if target_cid:
            add(target_cid, {"action": "set_target", "target": dict(target)})
        return out

    def _line_from_relation(self, rel: Dict[str, Any]) -> List[str]:
        return self.object_normalizer.line_from_relation(rel)

    def _lines_from_relation(self, rel: Dict[str, Any]) -> List[str]:
        return self.object_normalizer.lines_from_relation(rel)

    def _matches_expected_instruction(self, instruction: Dict[str, Any], expected: Dict[str, Any], candidate_id: str) -> bool:
        if str(instruction.get("action") or "") != str(expected.get("action") or ""):
            return False
        if candidate_id not in self._instruction_candidate_refs(instruction):
            return False
        for key in ["id", "points", "lines", "value", "target"]:
            if key in expected and not self._same_value(instruction.get(key), expected.get(key)):
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

    def _repairable_candidate_ids(self, candidate_usage: Dict[str, Any]) -> Set[str]:
        out = set(str(item) for item in candidate_usage.get("confirmed_candidate_ids") or [] if item)
        out.update(str(item) for item in candidate_usage.get("corrected_candidate_ids") or [] if item)
        return out

    def _existing_actions(self, instructions: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
        out: Dict[str, Set[str]] = {}
        for inst in instructions:
            for cid in self._instruction_candidate_refs(inst):
                action = str(inst.get("action") or "")
                if action:
                    out.setdefault(cid, set()).add(action)
        return out

    def _instruction_candidate_refs(self, item: Dict[str, Any]) -> Set[str]:
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
            refs.update(self._instruction_candidate_refs(target))
        return refs

    def _normalize_repair_instruction(self, suggestion: Dict[str, Any], candidate_id: str) -> Dict[str, Any]:
        action = str(suggestion.get("action") or "")
        if action not in self.capabilities.supported_actions:
            return {}
        out: Dict[str, Any] = {
            "action": action,
            "from_candidate": candidate_id,
            "repair_source": suggestion.get("repair_source") or "script_build_suggestion",
        }
        if action == "add_point":
            point_id = self._point_id(suggestion.get("id"))
            if not point_id:
                return {}
            out["id"] = point_id
        elif action in {"add_segment", "set_length"}:
            points = [self._point_id(item) for item in (suggestion.get("points") or [])[:2]]
            points = [item for item in points if item]
            if len(points) < 2:
                return {}
            out["points"] = points[:2]
            if action == "set_length":
                if suggestion.get("value") in {None, ""}:
                    return {}
                out["value"] = str(suggestion.get("value"))
        elif action in {"set_parallel", "set_perpendicular"}:
            lines = [self._line_id(item) for item in (suggestion.get("lines") or [])[:2]]
            lines = [item for item in lines if len(item) >= 2]
            if len(lines) < 2:
                return {}
            out["lines"] = lines[:2]
        elif action == "set_target":
            target = dict(suggestion.get("target") or {})
            if not target:
                return {}
            if "object" in target:
                target["object"] = str(target.get("object") or "").upper()
            target.setdefault("type", "unknown")
            target.setdefault("value_symbol", "x")
            target.setdefault("from_candidate", candidate_id)
            out["target"] = target
        else:
            return {}
        visual_ids = [str(item) for item in suggestion.get("visual_ids") or [] if item]
        if visual_ids:
            out["visual_ids"] = visual_ids
        return out

    def _dedupe(self, instructions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        seen = set()
        for inst in instructions:
            key = json.dumps(inst, sort_keys=True, ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            out.append(inst)
        return out

    def _point_id(self, value: Any) -> str:
        return self.object_normalizer.point_id(value)

    def _line_id(self, value: Any) -> str:
        return self.object_normalizer.line_id(value)
