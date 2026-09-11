"""Build a compact evidence graph for VLM-first semantic extraction."""

from __future__ import annotations

from typing import Any, Dict, List

from .builder_capabilities import GEOSYM_2D_BUILDER_CAPABILITIES, GeoSymBuilderCapabilities


class EvidenceGraphBuilder:
    """Unify weak script hints and builder capability for the VLM.

    The evidence graph is not a geometry state and the visual primitive ids are
    not the main parsing substrate. The VLM reads the original image/text first;
    this graph gives it script hints, known dataset text, and the current
    GeoSym builder contract so later code can normalize and verify its output.
    """

    def __init__(self, capabilities: GeoSymBuilderCapabilities = GEOSYM_2D_BUILDER_CAPABILITIES):
        self.capabilities = capabilities

    def build(
        self,
        *,
        visual_state: Dict[str, Any],
        script_observations: Dict[str, Any],
        problem_text: str = "",
        answer: Any = None,
    ) -> Dict[str, Any]:
        semantic_candidates = script_observations.get("semantic_candidates") or {}
        engine_suggestions = script_observations.get("engine_build_suggestions") or {}
        visual_candidates = self._visual_candidates(visual_state, script_observations)
        text_evidence = self._text_evidence(script_observations, problem_text, answer)
        named_object_proposals = self._named_object_proposals(script_observations)
        relation_candidates = self._relation_candidates(semantic_candidates)
        target_candidates = self._target_candidates(semantic_candidates)
        grounding_tasks = self._grounding_tasks(script_observations)
        build_suggestions = self._build_suggestions(engine_suggestions)
        build_tasks = self._build_tasks(relation_candidates, target_candidates, build_suggestions)
        candidate_review_checklist = self._candidate_review_checklist(relation_candidates, target_candidates)
        relation_pass_response_template = self._relation_pass_response_template(
            candidate_review_checklist,
            grounding_tasks,
            engine_suggestions,
            named_object_proposals,
        )
        return {
            "version": "vlm-first-v0",
            "principle": "vlm-first semantic extraction with script-assisted normalization, conversion, and verification",
            "extraction_strategy": {
                "primary_extractor": "VLM should read named points, labels, relations, marks, and target directly from the original image/problem text",
                "script_hints": "relation_candidates, target_candidates, named_object_proposals, and visual_candidates are weak hints/checklists",
                "visual_candidates": "OpenCV primitive ids are optional review/alignment evidence; do not force every relation through them",
                "post_vlm_script": "script code normalizes relation schema, checks object validity, maps to builder capability, compiles build plans, and verifies",
            },
            "visual_candidates": visual_candidates,
            "text_evidence": text_evidence,
            "named_object_proposals": named_object_proposals,
            "relation_candidates": relation_candidates,
            "target_candidates": target_candidates,
            "grounding_tasks": grounding_tasks,
            "candidate_review_checklist": candidate_review_checklist,
            "relation_pass_response_template": relation_pass_response_template,
            "build_suggestions": build_suggestions,
            "build_tasks": build_tasks,
            "builder_contract": self._builder_contract(engine_suggestions),
            "vlm_contract": {
                "relation_pass": [
                    "first read the original image/problem text and extract named geometry semantics",
                    "use named_object_proposals such as A, AB, and ABC as weak hints; bind to visual evidence when helpful, but do not require OpenCV ids for clear named relations",
                    "confirm/correct/reject relation_candidates after comparing them with the original image/text",
                    "bind labels and relations to visual candidate ids only when those ids help support or review the semantic claim",
                    "include candidate_id when a relation uses a script candidate",
                    "for VLM-discovered relations without a script candidate, include evidence.visual_evidence_ids/visual_ids or evidence.text_refs/supporting_text",
                    "visual evidence ids must come from provided visual candidates or grounding tasks; text refs must quote or cite current problem/OCR/text evidence",
                    "VLM-discovered relations without valid evidence are deferred and cannot become GeoSym builder constraints",
                    "put the primary target in target and every target candidate in targets",
                    "keep theorem-level relations as relations; do not force them into build instructions",
                ],
                "instruction_pass": [
                    "emit only supported builder actions",
                    "cover build_tasks for confirmed/corrected coordinate-buildable candidates",
                    "include from_candidate when using script build_suggestions",
                    "do not output pixel coordinates or unsupported theorem instructions",
                    "set engine_assessment.expected_mode using builder_contract",
                ],
            },
        }

    def summary(self, evidence_graph: Dict[str, Any]) -> Dict[str, Any]:
        visual = evidence_graph.get("visual_candidates") or {}
        builder = evidence_graph.get("builder_contract") or {}
        named = evidence_graph.get("named_object_proposals") or {}
        named_summary = named.get("summary") or {}
        return {
            "num_visual_lines": len(visual.get("lines") or []),
            "num_visual_circles": len(visual.get("circles") or []),
            "num_visual_fill_regions": len(visual.get("fill_regions") or []),
            "num_visual_markers": len(visual.get("markers") or []),
            "num_visual_text_regions": len(visual.get("text_regions") or []),
            "num_visual_point_candidates": len((visual.get("visual_structure_candidates") or {}).get("point_candidates") or []),
            "num_visual_segment_candidates": len((visual.get("visual_structure_candidates") or {}).get("segment_candidates") or []),
            "num_visual_angle_candidates": len((visual.get("visual_structure_candidates") or {}).get("angle_candidates") or []),
            "num_relation_candidates": len(evidence_graph.get("relation_candidates") or []),
            "num_target_candidates": len(evidence_graph.get("target_candidates") or []),
            "num_grounding_tasks": len(evidence_graph.get("grounding_tasks") or []),
            "num_candidate_review_items": len(evidence_graph.get("candidate_review_checklist") or []),
            "num_build_suggestions": len(evidence_graph.get("build_suggestions") or []),
            "num_build_tasks": len(evidence_graph.get("build_tasks") or []),
            "num_named_object_proposals": int(named_summary.get("num_items") or len(named.get("items") or [])),
            "num_named_point_label_proposals": int(named_summary.get("num_point_labels") or 0),
            "num_named_segment_proposals": int(named_summary.get("num_segments") or 0),
            "expected_mode_hint": builder.get("expected_mode_hint"),
            "supported_actions": builder.get("supported_actions") or [],
            "supported_constraint_types": builder.get("supported_constraint_types") or [],
            "relation_mapping_contract": builder.get("relation_mapping_contract") or [],
            "unsupported_candidate_types": builder.get("unsupported_candidate_types") or [],
        }

    def _visual_candidates(self, visual_state: Dict[str, Any], script_observations: Dict[str, Any]) -> Dict[str, Any]:
        topology = visual_state.get("topology") or {}
        return {
            "image_size": visual_state.get("image_size"),
            "lines": self._take(visual_state.get("lines") or [], 60),
            "circles": self._take(visual_state.get("circles") or [], 20),
            "fill_regions": self._take(visual_state.get("fill_regions") or [], 40),
            "markers": self._take(visual_state.get("markers") or [], 40),
            "text_regions": self._take(visual_state.get("text_regions") or [], 60),
            "intersections": self._take(topology.get("line_intersections") or [], 80),
            "circle_line_hits": self._take(topology.get("circle_line_hits") or [], 80),
            "endpoints": self._take(topology.get("endpoints") or [], 80),
            "visual_structure_candidates": self._compact_visual_structure(
                script_observations.get("visual_structure_candidates") or {}
            ),
        }

    def _compact_visual_structure(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "point_candidates": self._take(structure.get("point_candidates") or [], 120),
            "segment_candidates": self._take(structure.get("segment_candidates") or [], 80),
            "angle_candidates": self._take(structure.get("angle_candidates") or [], 60),
            "label_region_candidates": self._take(structure.get("label_region_candidates") or [], 80),
            "summary": dict(structure.get("summary") or {}),
            "coordinate_policy": structure.get("coordinate_policy") or "visual candidates are evidence only",
        }

    def _text_evidence(self, script_observations: Dict[str, Any], problem_text: str, answer: Any) -> Dict[str, Any]:
        hints = script_observations.get("text_hints") or {}
        return {
            "problem_text": problem_text,
            "known_answer": answer,
            "text_sources": [
                {
                    "name": item.get("name"),
                    "confidence": item.get("confidence"),
                    "text": item.get("text"),
                }
                for item in (script_observations.get("text_sources") or [])[:20]
            ],
            "labels_mentioned": hints.get("labels_mentioned") or [],
            "segments_mentioned": hints.get("segments_mentioned") or [],
            "numbers_mentioned": hints.get("numbers_mentioned") or [],
            "length_claims": hints.get("length_claims") or [],
            "angle_claims": hints.get("angle_claims") or [],
            "area_claims": hints.get("area_claims") or [],
            "perimeter_claims": hints.get("perimeter_claims") or [],
            "ratio_claims": hints.get("ratio_claims") or [],
            "triangle_claims": hints.get("triangle_claims") or [],
            "target_hint": hints.get("target_hint"),
            "target_angle_hint": hints.get("target_angle_hint"),
            "target_area_hint": hints.get("target_area_hint"),
            "target_perimeter_hint": hints.get("target_perimeter_hint"),
            "target_ratio_hint": hints.get("target_ratio_hint"),
        }

    def _named_object_proposals(self, script_observations: Dict[str, Any]) -> Dict[str, Any]:
        proposals = script_observations.get("named_object_proposals") or {}
        return {
            "items": self._take(proposals.get("items") or [], 120),
            "summary": dict(proposals.get("summary") or {}),
            "vlm_instruction": proposals.get("vlm_instruction")
            or "Bind symbolic point/segment/angle objects to visual evidence before confirming buildable relations.",
            "policy": (
                "These are symbolic named objects from text/script candidates. VLM should use them as AB/CD/points/endpoints "
                "objects for relations, never generic descriptions such as first line or second line."
            ),
        }

    def _grounding_tasks(self, script_observations: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self._take(script_observations.get("grounding_tasks") or [], 120)

    def _relation_candidates(self, semantic_candidates: Dict[str, Any]) -> List[Dict[str, Any]]:
        out = []
        for item in semantic_candidates.get("relation_candidates") or []:
            candidate = dict(item)
            ctype = str(candidate.get("type") or "")
            candidate["buildability"] = self._candidate_buildability(ctype, candidate)
            candidate["builder_mapping"] = self._builder_mapping_for_candidate(candidate)
            out.append(candidate)
        return out

    def _target_candidates(self, semantic_candidates: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [dict(item) for item in semantic_candidates.get("target_candidates") or []]

    def _candidate_review_checklist(
        self,
        relation_candidates: List[Dict[str, Any]],
        target_candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        out = []
        for candidate in relation_candidates:
            cid = str(candidate.get("id") or "")
            if not cid:
                continue
            out.append({
                "candidate_id": cid,
                "candidate_kind": "relation",
                "candidate_type": candidate.get("type"),
                "candidate_object": candidate.get("object") or candidate.get("objects"),
                "candidate_value": candidate.get("value"),
                "buildability_mode": (candidate.get("buildability") or {}).get("mode"),
                "builder_mapping": candidate.get("builder_mapping") or {},
                "expected_relation": self._expected_relation(candidate),
                "expected_decision": "confirm|correct|reject|uncertain",
                "expected_output": {
                    "candidate_id": cid,
                    "decision": "confirm",
                    "corrected_relation": self._expected_relation(candidate),
                    "reason": "brief evidence from image/text",
                    "confidence": 0.0,
                },
                "candidate": candidate,
            })
        for candidate in target_candidates:
            cid = str(candidate.get("id") or "")
            if not cid:
                continue
            out.append({
                "candidate_id": cid,
                "candidate_kind": "target",
                "candidate_type": candidate.get("type"),
                "candidate_object": candidate.get("object"),
                "candidate_value": candidate.get("value_symbol"),
                "buildability_mode": "target",
                "expected_target": self._expected_target(candidate),
                "expected_decision": "confirm|correct|reject|uncertain",
                "expected_output": {
                    "candidate_id": cid,
                    "decision": "confirm",
                    "corrected_relation": self._expected_target(candidate),
                    "reason": "brief evidence from problem target",
                    "confidence": 0.0,
                },
                "candidate": candidate,
            })
        return out

    def _relation_pass_response_template(
        self,
        checklist: List[Dict[str, Any]],
        grounding_tasks: List[Dict[str, Any]],
        engine_suggestions: Dict[str, Any],
        named_object_proposals: Dict[str, Any],
    ) -> Dict[str, Any]:
        targets = [
            dict(item.get("expected_target") or {})
            for item in checklist
            if item.get("candidate_kind") == "target" and item.get("expected_target")
        ]
        return {
            "bindings": [],
            "named_object_bindings": [
                self._expected_named_object_binding(item)
                for item in (named_object_proposals.get("items") or [])
                if item.get("id")
            ],
            "relations": [
                dict(item.get("expected_relation") or {})
                for item in checklist
                if item.get("candidate_kind") == "relation" and item.get("expected_relation")
            ],
            "candidate_decisions": [
                dict(item.get("expected_output") or {})
                for item in checklist
            ],
            "grounding_decisions": [
                self._expected_grounding_decision(item)
                for item in grounding_tasks
                if item.get("id")
            ],
            "relation_builder_decisions": [
                self._expected_relation_builder_decision(item)
                for item in checklist
                if item.get("candidate_kind") == "relation" and item.get("expected_relation")
            ],
            "target": dict(targets[0]) if targets else {},
            "targets": targets,
            "engine_assessment": {
                "expected_mode": engine_suggestions.get("expected_mode_hint") or "failed",
                "reason": "initial stage-1 estimate from script candidates; revise if image/text contradicts it",
            },
            "notes": (
                "Draft generated from script candidates. Stage-1 VLM should fill candidate_decisions "
                "for every checklist item, fill grounding_decisions for every grounding task, add "
                "relations/target only when supported by image/text evidence, preserve all confirmed targets in targets[], "
                "fill named_object_bindings for named_object_proposals before confirming relations, "
                "fill relation_builder_decisions for every confirmed/corrected relation using the current builder capability, "
                "use relation_index in relation_builder_decisions for VLM-discovered relations without script candidate_id, "
                "and attach valid visual/text evidence to every VLM-discovered relation so the evidence gate can decide "
                "whether it may become a builder constraint."
            ),
        }

    def _expected_named_object_binding(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "proposal_id": str(proposal.get("id") or ""),
            "object": str(proposal.get("object") or ""),
            "kind": str(proposal.get("kind") or ""),
            "visual_ids": [],
            "decision": "bind|absent|uncertain|reject",
            "reason": "choose visual ids from this proposal's visual_choices when visible; otherwise use absent or uncertain",
            "confidence": 0.0,
        }

    def _expected_grounding_decision(self, task: Dict[str, Any]) -> Dict[str, Any]:
        choices = [
            str(item.get("visual_id"))
            for item in task.get("visual_choices") or []
            if isinstance(item, dict) and item.get("visual_id")
        ]
        return {
            "task_id": str(task.get("id") or ""),
            "candidate_id": str(task.get("candidate_id") or ""),
            "semantic_ref": str(task.get("semantic_ref") or ""),
            "visual_ids": [],
            "decision": "bind" if choices else "absent",
            "reason": "choose visual_ids from the task visual_choices when image evidence supports the binding; otherwise use absent or uncertain",
            "confidence": 0.0,
        }

    def _expected_relation_builder_decision(self, checklist_item: Dict[str, Any]) -> Dict[str, Any]:
        relation = checklist_item.get("expected_relation") or {}
        mapping = relation.get("builder_mapping") or checklist_item.get("builder_mapping") or {}
        buildability = relation.get("buildability") or {}
        status = str(mapping.get("status") or "")
        mode = str(buildability.get("mode") or mapping.get("buildability_mode") or "")
        if status == "buildable":
            decision = "convertible" if mode == "convertible_coordinate_constraint" else "buildable"
        elif status == "theorem_only":
            decision = "theorem_only"
        elif mode == "theorem_only":
            decision = "theorem_only"
        elif status == "not_buildable":
            decision = "unsupported"
        else:
            decision = "unsupported"
        return {
            "candidate_id": str(checklist_item.get("candidate_id") or relation.get("candidate_id") or ""),
            "relation_type": str(relation.get("type") or checklist_item.get("candidate_type") or ""),
            "decision": decision,
            "builder_actions": [str(item) for item in mapping.get("builder_actions") or []],
            "core_constraint_action": mapping.get("core_constraint_action"),
            "reason": "copy/revise from current GeoSym relation_mapping_contract after checking image/text evidence",
            "confidence": 0.0,
        }

    def _expected_relation(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        ctype = str(candidate.get("type") or "")
        cid = str(candidate.get("id") or "")
        if not ctype or not cid:
            return {}
        relation: Dict[str, Any] = {
            "type": ctype,
            "candidate_id": cid,
            "grounded": False,
            "source": candidate.get("source") or "script_candidate",
            "buildability": dict(candidate.get("buildability") or {}),
            "builder_mapping": dict(candidate.get("builder_mapping") or {}),
        }
        if candidate.get("source_text"):
            relation["source_text"] = candidate.get("source_text")
        objects = candidate.get("objects")
        if objects is not None:
            relation["objects"] = [str(item) for item in objects]
        elif candidate.get("object"):
            obj = str(candidate.get("object") or "")
            relation["objects"] = list(obj) if len(obj) == 2 and obj.isalpha() else [obj]
        if candidate.get("object") is not None:
            relation["object"] = str(candidate.get("object"))
        if candidate.get("value") is not None:
            relation["value"] = str(candidate.get("value"))
        if candidate.get("angle") is not None:
            relation["angle"] = str(candidate.get("angle"))
        if candidate.get("vertex") is not None:
            relation["vertex"] = str(candidate.get("vertex"))
        return relation

    def _expected_target(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        cid = str(candidate.get("id") or "")
        obj = str(candidate.get("object") or "")
        if not cid or not obj:
            return {}
        return {
            "type": str(candidate.get("type") or "unknown"),
            "object": obj,
            "value_symbol": str(candidate.get("value_symbol") or "x"),
            "candidate_id": cid,
            "from_candidate": cid,
            "source": candidate.get("source") or "script_candidate",
        }

    def _build_suggestions(self, engine_suggestions: Dict[str, Any]) -> List[Dict[str, Any]]:
        out = []
        for item in engine_suggestions.get("suggested_build_instructions") or []:
            suggestion = dict(item)
            action = str(suggestion.get("action") or "")
            suggestion["supported_by_current_builder"] = action in self.capabilities.supported_actions
            out.append(suggestion)
        return out

    def _build_tasks(
        self,
        relation_candidates: List[Dict[str, Any]],
        target_candidates: List[Dict[str, Any]],
        build_suggestions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        by_candidate: Dict[str, List[Dict[str, Any]]] = {}
        for suggestion in build_suggestions:
            cid = str(suggestion.get("from_candidate") or "")
            if not cid:
                continue
            by_candidate.setdefault(cid, []).append(suggestion)

        tasks: List[Dict[str, Any]] = []
        for candidate in relation_candidates:
            cid = str(candidate.get("id") or "")
            if not cid:
                continue
            buildability = candidate.get("buildability") or {}
            suggestions = by_candidate.get(cid) or []
            task = {
                "candidate_id": cid,
                "candidate_type": candidate.get("type"),
                "candidate_object": candidate.get("object") or candidate.get("objects"),
                "buildability_mode": buildability.get("mode"),
                "relation_buildability": buildability,
                "builder_mapping": candidate.get("builder_mapping") or {},
                "required_actions": self._required_actions_for_candidate(candidate, suggestions),
                "instruction_requirements": self._instruction_requirements_for_candidate(cid, suggestions),
                "optional_actions": self._optional_actions_for_candidate(candidate, suggestions),
                "suggested_instructions": suggestions,
                "must_emit_when_decision": ["confirm", "correct"] if buildability.get("can_generate_build_instruction") else [],
                "notes": buildability.get("required_confirmation") or "",
            }
            tasks.append(task)

        for candidate in target_candidates:
            cid = str(candidate.get("id") or "")
            if not cid:
                continue
            suggestions = by_candidate.get(cid) or []
            tasks.append({
                "candidate_id": cid,
                "candidate_type": "target",
                "candidate_object": candidate.get("object"),
                "buildability_mode": "target",
                "required_actions": ["set_target"],
                "instruction_requirements": self._instruction_requirements_for_candidate(cid, suggestions),
                "optional_actions": [],
                "suggested_instructions": suggestions,
                "must_emit_when_decision": ["confirm", "correct"],
                "notes": "confirmed target candidates should be converted to set_target",
            })
        return tasks

    def _builder_contract(self, engine_suggestions: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "capability_version": self.capabilities.version,
            "supported_actions": sorted(self.capabilities.supported_actions),
            "action_schema": self.capabilities.action_schema_contract(),
            "supported_constraint_types": sorted(self.capabilities.supported_constraint_types),
            "theorem_relation_types": sorted(self.capabilities.theorem_relation_types),
            "relation_mapping_contract": self.capabilities.relation_mapping_contract(),
            "unsupported_constructs": list(self.capabilities.unsupported_constructs),
            "expected_mode_hint": engine_suggestions.get("expected_mode_hint"),
            "unsupported_candidate_types": engine_suggestions.get("unsupported_candidate_types") or [],
            "rule": "build_instructions must use only supported_actions and must not encode theorem-only facts as fake coordinate constraints",
        }

    def _candidate_buildability(self, ctype: str, candidate: Dict[str, Any]) -> Dict[str, Any]:
        if ctype in self.capabilities.supported_constraint_types:
            return {
                "mode": "coordinate_constraint",
                "can_generate_build_instruction": True,
                "required_confirmation": "bind named objects to visible labels/segments when possible",
            }
        if ctype == "right_angle":
            return {
                "mode": "convertible_coordinate_constraint",
                "can_generate_build_instruction": True,
                "required_confirmation": "must identify two incident lines before converting to set_perpendicular",
            }
        if ctype in self.capabilities.theorem_relation_types:
            return {
                "mode": "theorem_only",
                "can_generate_build_instruction": False,
                "required_confirmation": candidate.get("engine_note") or "keep as semantic relation unless a supported low-level constraint is explicit",
            }
        return {
            "mode": "unsupported",
            "can_generate_build_instruction": False,
            "required_confirmation": "extract only if semantically important; current builder cannot build it",
        }

    def _builder_mapping_for_candidate(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Describe how a semantic relation can become builder instructions.

        This mapping is prompt/audit metadata. It is not a builder instruction
        itself, and it intentionally avoids mathematical coordinates.
        """
        ctype = str(candidate.get("type") or "")
        buildability = candidate.get("buildability") or {}
        if not buildability.get("can_generate_build_instruction"):
            return {
                "status": "not_buildable",
                "reason": buildability.get("required_confirmation") or "current builder does not support this relation type",
                "builder_actions": [],
                "must_not_emit_fake_constraints": True,
            }
        global_mapping = self._global_mapping_for_type(ctype)
        actions = list(global_mapping.get("builder_actions") or self._required_actions_for_candidate(candidate, []))
        return {
            "status": "buildable",
            "relation_type": ctype,
            "buildability_mode": buildability.get("mode"),
            "builder_actions": actions,
            "core_constraint_action": global_mapping.get("core_constraint_action"),
            "instruction_requirements_source": "engine_build_suggestions",
            "rule": "stage-2 must emit all required builder actions for confirmed/corrected candidates",
        }

    def _global_mapping_for_type(self, relation_type: str) -> Dict[str, Any]:
        for item in self.capabilities.relation_mapping_contract():
            if item.get("relation_type") == relation_type:
                return dict(item)
        return {}

    def _required_actions_for_candidate(self, candidate: Dict[str, Any], suggestions: List[Dict[str, Any]]) -> List[str]:
        ctype = str(candidate.get("type") or "")
        if ctype == "length":
            return ["add_point", "add_segment", "set_length"]
        if ctype in {"parallel", "perpendicular", "right_angle"}:
            return ["add_point", "add_segment", f"set_{'perpendicular' if ctype == 'right_angle' else ctype}"]
        if suggestions and (candidate.get("buildability") or {}).get("can_generate_build_instruction"):
            return sorted({str(item.get("action")) for item in suggestions if item.get("action")})
        return []

    def _optional_actions_for_candidate(self, candidate: Dict[str, Any], suggestions: List[Dict[str, Any]]) -> List[str]:
        required = set(self._required_actions_for_candidate(candidate, suggestions))
        optional = sorted({str(item.get("action")) for item in suggestions if item.get("action") and str(item.get("action")) not in required})
        return optional

    def _instruction_requirements_for_candidate(self, candidate_id: str, suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        requirements = []
        for idx, suggestion in enumerate(suggestions):
            action = str(suggestion.get("action") or "")
            spec = self.capabilities.supported_actions.get(action)
            if not spec:
                continue
            expected = dict(suggestion)
            expected.setdefault("from_candidate", candidate_id)
            requirements.append({
                "requirement_id": f"{candidate_id}:{idx}:{action}",
                "action": action,
                "required_fields": list(spec.required_fields),
                "expected_instruction": expected,
                "match_rule": {
                    "action": action,
                    "from_candidate": candidate_id,
                    "field_values": {
                        key: expected.get(key)
                        for key in ["id", "points", "lines", "value", "target"]
                        if key in expected
                    },
                },
                "source": "script_build_suggestion",
            })
        return requirements

    def _take(self, items: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        return [dict(item) for item in items[:limit] if isinstance(item, dict)]
