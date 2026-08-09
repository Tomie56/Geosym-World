"""Validate and normalize VLM semantic output before grounding."""

from __future__ import annotations

from typing import Any, Dict, List

from .builder_capabilities import GEOSYM_2D_BUILDER_CAPABILITIES, GeoSymBuilderCapabilities


class SemanticOutputValidator:
    """Keep malformed VLM JSON from leaking into later parser stages."""

    RELATION_TYPES = {
        "angle",
        "arc_measure",
        "area",
        "chord",
        "circle",
        "congruent",
        "incidence",
        "intersecting_chords",
        "length",
        "parallel",
        "perpendicular",
        "perimeter",
        "ratio",
        "right_angle",
        "right_triangle",
        "similar",
        "target",
        "triangle",
    }
    ENGINE_MODES = {"coordinate_buildable", "theorem_only", "unsupported", "failed"}
    CANDIDATE_DECISIONS = {"confirm", "correct", "reject", "uncertain"}
    GROUNDING_DECISIONS = {"bind", "absent", "uncertain", "reject"}
    INSTRUCTION_TASK_DECISIONS = {"emit", "skip", "defer", "uncertain"}
    RELATION_BUILDER_DECISIONS = {"buildable", "convertible", "theorem_only", "unsupported", "uncertain"}

    def __init__(self, capabilities: GeoSymBuilderCapabilities = GEOSYM_2D_BUILDER_CAPABILITIES):
        self.capabilities = capabilities

    def validate(self, semantic: Any, stage: str = "full") -> Dict[str, Any]:
        warnings: List[Dict[str, Any]] = []
        if not isinstance(semantic, dict):
            warnings.append(self._warning("semantic.not_object", "VLM semantic output is not a JSON object", "$"))
            semantic = {}

        if stage not in {"full", "relation_pass", "instruction_pass"}:
            warnings.append(self._warning("semantic.unknown_validation_stage", f"Unknown validation stage: {stage}", "$.stage"))
            stage = "full"

        normalized = dict(semantic)
        normalized["bindings"] = self._dict_list(normalized.get("bindings"), "bindings", warnings)
        normalized["named_object_bindings"] = self._normalize_named_object_bindings(normalized, warnings) if stage in {"full", "relation_pass"} else self._dict_list(normalized.get("named_object_bindings"), "named_object_bindings", warnings)
        normalized["relations"] = self._normalize_relations(normalized, warnings) if stage in {"full", "relation_pass"} else self._dict_list(normalized.get("relations"), "relations", warnings)
        normalized["build_instructions"] = self._normalize_instructions(normalized, warnings) if stage in {"full", "instruction_pass"} else self._dict_list(normalized.get("build_instructions"), "build_instructions", warnings)
        normalized["candidate_decisions"] = self._normalize_candidate_decisions(normalized, warnings) if stage in {"full", "relation_pass"} else self._dict_list(normalized.get("candidate_decisions"), "candidate_decisions", warnings)
        normalized["grounding_decisions"] = self._normalize_grounding_decisions(normalized, warnings) if stage in {"full", "relation_pass"} else self._dict_list(normalized.get("grounding_decisions"), "grounding_decisions", warnings)
        normalized["relation_builder_decisions"] = self._normalize_relation_builder_decisions(normalized, warnings) if stage in {"full", "relation_pass"} else self._dict_list(normalized.get("relation_builder_decisions"), "relation_builder_decisions", warnings)
        normalized["instruction_task_decisions"] = self._normalize_instruction_task_decisions(normalized, warnings) if stage in {"full", "instruction_pass"} else self._dict_list(normalized.get("instruction_task_decisions"), "instruction_task_decisions", warnings)
        normalized["target"] = self._dict_field(normalized.get("target"), "target", warnings)
        normalized["targets"] = self._normalize_targets(normalized.get("targets"), warnings)
        if normalized["target"]:
            normalized["targets"] = self._dedupe_targets([normalized["target"]] + normalized["targets"])
        normalized["engine_assessment"] = self._normalize_engine_assessment(
            normalized.get("engine_assessment"),
            warnings,
        )

        metadata = dict(normalized.get("metadata") or {})
        metadata["semantic_validation"] = {
            "valid": not warnings,
            "num_warnings": len(warnings),
            "stage": stage,
        }
        normalized["metadata"] = metadata
        return {
            "semantic": normalized,
            "valid": not warnings,
            "warnings": warnings,
            "metrics": {
                "semantic_schema_warnings": len(warnings),
                "semantic_relations": len(normalized["relations"]),
                "semantic_named_object_bindings": len(normalized["named_object_bindings"]),
                "semantic_build_instructions": len(normalized["build_instructions"]),
                "semantic_candidate_decisions": len(normalized["candidate_decisions"]),
                "semantic_grounding_decisions": len(normalized["grounding_decisions"]),
                "semantic_relation_builder_decisions": len(normalized["relation_builder_decisions"]),
                "semantic_instruction_task_decisions": len(normalized["instruction_task_decisions"]),
                "semantic_targets": len(normalized["targets"]),
            },
        }

    def _normalize_relations(self, semantic: Dict[str, Any], warnings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            self._normalize_relation(item, idx, warnings)
            for idx, item in enumerate(self._dict_list(semantic.get("relations"), "relations", warnings))
        ]

    def _normalize_instructions(self, semantic: Dict[str, Any], warnings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            self._normalize_instruction(item, idx, warnings)
            for idx, item in enumerate(self._dict_list(semantic.get("build_instructions"), "build_instructions", warnings))
        ]

    def _normalize_candidate_decisions(self, semantic: Dict[str, Any], warnings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            self._normalize_candidate_decision(item, idx, warnings)
            for idx, item in enumerate(self._dict_list(semantic.get("candidate_decisions"), "candidate_decisions", warnings))
        ]

    def _normalize_grounding_decisions(self, semantic: Dict[str, Any], warnings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            self._normalize_grounding_decision(item, idx, warnings)
            for idx, item in enumerate(self._dict_list(semantic.get("grounding_decisions"), "grounding_decisions", warnings))
        ]

    def _normalize_relation_builder_decisions(self, semantic: Dict[str, Any], warnings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            self._normalize_relation_builder_decision(item, idx, warnings)
            for idx, item in enumerate(self._dict_list(semantic.get("relation_builder_decisions"), "relation_builder_decisions", warnings))
        ]

    def _normalize_named_object_bindings(self, semantic: Dict[str, Any], warnings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            self._normalize_named_object_binding(item, idx, warnings)
            for idx, item in enumerate(self._dict_list(semantic.get("named_object_bindings"), "named_object_bindings", warnings))
        ]

    def _normalize_instruction_task_decisions(self, semantic: Dict[str, Any], warnings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            self._normalize_instruction_task_decision(item, idx, warnings)
            for idx, item in enumerate(self._dict_list(semantic.get("instruction_task_decisions"), "instruction_task_decisions", warnings))
        ]

    def _normalize_targets(self, value: Any, warnings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            self._normalize_target(item, idx, warnings)
            for idx, item in enumerate(self._dict_list(value, "targets", warnings))
        ]

    def _normalize_target(self, target: Dict[str, Any], idx: int, warnings: List[Dict[str, Any]]) -> Dict[str, Any]:
        out = dict(target)
        location = f"targets[{idx}]"
        if out.get("type") is not None:
            out["type"] = str(out.get("type"))
        if out.get("object") is not None:
            out["object"] = str(out.get("object"))
        if out.get("value_symbol") is not None:
            out["value_symbol"] = str(out.get("value_symbol"))
        for key in ["candidate_id", "from_candidate"]:
            if out.get(key) is not None:
                out[key] = str(out.get(key))
        if "target" in out and not isinstance(out.get("target"), dict):
            warnings.append(self._warning("semantic.bad_field_type", f"{location}.target must be an object", f"{location}.target"))
            out.pop("target", None)
        return out

    def _normalize_relation(self, rel: Dict[str, Any], idx: int, warnings: List[Dict[str, Any]]) -> Dict[str, Any]:
        rel = dict(rel)
        location = f"relations[{idx}]"
        rtype = rel.get("type")
        if not rtype:
            warnings.append(self._warning("semantic.relation_missing_type", "Relation is missing type", location))
            rel["type"] = "unknown"
        elif str(rtype) not in self.RELATION_TYPES:
            warnings.append(self._warning("semantic.unknown_relation_type", f"Unknown relation type: {rtype}", f"{location}.type"))
            rel["type"] = str(rtype)
        else:
            rel["type"] = str(rtype)
        if "objects" in rel:
            rel["objects"] = self._string_list(rel.get("objects"), f"{location}.objects", warnings)
        if "visual_ids" in rel:
            rel["visual_ids"] = self._string_list(rel.get("visual_ids"), f"{location}.visual_ids", warnings)
        self._normalize_relation_evidence(rel, location, warnings)
        if "grounded" in rel:
            rel["grounded"] = self._bool_field(rel.get("grounded"), f"{location}.grounded", warnings)
        if "buildability" in rel:
            rel["buildability"] = self._normalize_buildability(rel.get("buildability"), f"{location}.buildability", warnings)
        if "builder_mapping" in rel:
            rel["builder_mapping"] = self._normalize_builder_mapping(rel.get("builder_mapping"), f"{location}.builder_mapping", warnings)
        return rel

    def _normalize_relation_evidence(self, rel: Dict[str, Any], location: str, warnings: List[Dict[str, Any]]) -> None:
        visual_ids = list(rel.get("visual_ids") or []) if isinstance(rel.get("visual_ids"), list) else []
        text_refs = self._string_list_soft(rel.get("text_refs"))
        evidence = rel.get("evidence")
        if isinstance(evidence, dict):
            for key in ["visual_ids", "visual_evidence_ids", "visual_refs", "visual_evidence"]:
                visual_ids.extend(self._evidence_visual_ids(evidence.get(key)))
            for key in ["text_refs", "text_evidence", "textual_evidence", "supporting_text", "quote", "citations"]:
                text_refs.extend(self._string_list_soft(evidence.get(key)))
        elif evidence:
            text_refs.extend(self._string_list_soft(evidence))
        for key in ["visual_evidence_ids", "visual_refs", "visual_evidence"]:
            visual_ids.extend(self._evidence_visual_ids(rel.get(key)))
        for key in ["text_evidence", "textual_evidence", "supporting_text", "quote", "citations", "evidence_text"]:
            text_refs.extend(self._string_list_soft(rel.get(key)))
        visual_ids = self._dedupe_strings(visual_ids)
        text_refs = self._dedupe_strings(text_refs)
        if visual_ids:
            rel["visual_ids"] = visual_ids
        if text_refs:
            rel["text_refs"] = text_refs
        if "confidence" in rel:
            try:
                rel["confidence"] = float(rel["confidence"])
            except (TypeError, ValueError):
                warnings.append(self._warning("semantic.bad_field_type", f"{location}.confidence must be numeric", f"{location}.confidence"))
                rel["confidence"] = 0.0

    def _evidence_visual_ids(self, value: Any) -> List[str]:
        items = self._string_list_soft(value)
        return [
            item
            for item in items
            if item.startswith(("vl_", "vc_", "vt_", "vp_", "vs_", "va_", "vlabel_", "ve_", "vx_"))
        ]

    def _string_list_soft(self, value: Any) -> List[str]:
        if value is None or value == "":
            return []
        if isinstance(value, list):
            out: List[str] = []
            for item in value:
                out.extend(self._string_list_soft(item))
            return out
        if isinstance(value, dict):
            out: List[str] = []
            for key in ["id", "visual_id", "text", "quote", "ref", "value", "label"]:
                if value.get(key) is not None and value.get(key) != "":
                    out.append(str(value.get(key)))
            if not out:
                for item in value.values():
                    out.extend(self._string_list_soft(item))
            return out
        return [str(value)]

    def _dedupe_strings(self, values: List[str]) -> List[str]:
        out: List[str] = []
        for value in values:
            text = str(value)
            if not text or text in out:
                continue
            out.append(text)
        return out

    def _normalize_candidate_decision(self, item: Dict[str, Any], idx: int, warnings: List[Dict[str, Any]]) -> Dict[str, Any]:
        decision = dict(item)
        location = f"candidate_decisions[{idx}]"
        candidate_id = decision.get("candidate_id") or decision.get("id")
        if not candidate_id:
            warnings.append(self._warning("semantic.candidate_decision_missing_id", "Candidate decision is missing candidate_id", location))
            decision["candidate_id"] = ""
        else:
            decision["candidate_id"] = str(candidate_id)
        value = str(decision.get("decision") or "").lower()
        if value not in self.CANDIDATE_DECISIONS:
            warnings.append(self._warning("semantic.candidate_decision_unknown", f"Unknown candidate decision: {decision.get('decision')}", f"{location}.decision"))
            decision["decision"] = "uncertain"
        else:
            decision["decision"] = value
        if "corrected_relation" in decision:
            corrected = self._dict_field(decision.get("corrected_relation"), f"{location}.corrected_relation", warnings)
            if corrected:
                decision["corrected_relation"] = self._normalize_relation(corrected, idx, warnings)
            else:
                decision["corrected_relation"] = corrected
        if "reason" not in decision:
            decision["reason"] = ""
        if "confidence" in decision:
            try:
                decision["confidence"] = float(decision["confidence"])
            except (TypeError, ValueError):
                warnings.append(self._warning("semantic.bad_field_type", f"{location}.confidence must be numeric", f"{location}.confidence"))
                decision["confidence"] = 0.0
        return decision

    def _normalize_grounding_decision(self, item: Dict[str, Any], idx: int, warnings: List[Dict[str, Any]]) -> Dict[str, Any]:
        decision = dict(item)
        location = f"grounding_decisions[{idx}]"
        task_id = decision.get("task_id") or decision.get("id")
        if not task_id:
            warnings.append(self._warning("semantic.grounding_decision_missing_id", "Grounding decision is missing task_id", location))
            decision["task_id"] = ""
        else:
            decision["task_id"] = str(task_id)
        if decision.get("candidate_id") is not None:
            decision["candidate_id"] = str(decision.get("candidate_id"))
        if decision.get("semantic_ref") is not None:
            decision["semantic_ref"] = str(decision.get("semantic_ref"))
        if "visual_ids" in decision:
            decision["visual_ids"] = self._string_list(decision.get("visual_ids"), f"{location}.visual_ids", warnings)
        value = str(decision.get("decision") or "").lower()
        if value not in self.GROUNDING_DECISIONS:
            warnings.append(self._warning("semantic.grounding_decision_unknown", f"Unknown grounding decision: {decision.get('decision')}", f"{location}.decision"))
            decision["decision"] = "uncertain"
        else:
            decision["decision"] = value
        if "confidence" in decision:
            try:
                decision["confidence"] = float(decision["confidence"])
            except (TypeError, ValueError):
                warnings.append(self._warning("semantic.bad_field_type", f"{location}.confidence must be numeric", f"{location}.confidence"))
                decision["confidence"] = 0.0
        if "reason" not in decision:
            decision["reason"] = ""
        return decision

    def _normalize_named_object_binding(self, item: Dict[str, Any], idx: int, warnings: List[Dict[str, Any]]) -> Dict[str, Any]:
        binding = dict(item)
        location = f"named_object_bindings[{idx}]"
        proposal_id = binding.get("proposal_id") or binding.get("id")
        if not proposal_id:
            warnings.append(self._warning("semantic.named_object_binding_missing_id", "Named object binding is missing proposal_id", location))
            binding["proposal_id"] = ""
        else:
            binding["proposal_id"] = str(proposal_id)
        if binding.get("object") is not None:
            binding["object"] = str(binding.get("object")).upper()
        if binding.get("kind") is not None:
            binding["kind"] = str(binding.get("kind"))
        if "visual_ids" in binding:
            binding["visual_ids"] = self._string_list(binding.get("visual_ids"), f"{location}.visual_ids", warnings)
        value = str(binding.get("decision") or "").lower()
        if value not in self.GROUNDING_DECISIONS:
            warnings.append(self._warning("semantic.named_object_binding_unknown", f"Unknown named object binding decision: {binding.get('decision')}", f"{location}.decision"))
            binding["decision"] = "uncertain"
        else:
            binding["decision"] = value
        if "confidence" in binding:
            try:
                binding["confidence"] = float(binding["confidence"])
            except (TypeError, ValueError):
                warnings.append(self._warning("semantic.bad_field_type", f"{location}.confidence must be numeric", f"{location}.confidence"))
                binding["confidence"] = 0.0
        if "reason" not in binding:
            binding["reason"] = ""
        return binding

    def _normalize_relation_builder_decision(self, item: Dict[str, Any], idx: int, warnings: List[Dict[str, Any]]) -> Dict[str, Any]:
        decision = dict(item)
        location = f"relation_builder_decisions[{idx}]"
        candidate_id = decision.get("candidate_id") or decision.get("from_candidate") or decision.get("id")
        if not candidate_id:
            if decision.get("relation_index") is None:
                warnings.append(self._warning("semantic.relation_builder_decision_missing_id", "Relation builder decision is missing candidate_id", location))
            decision["candidate_id"] = ""
        else:
            decision["candidate_id"] = str(candidate_id)
        if "relation_index" in decision and decision.get("relation_index") is not None:
            try:
                decision["relation_index"] = int(decision.get("relation_index"))
            except (TypeError, ValueError):
                warnings.append(self._warning("semantic.bad_field_type", f"{location}.relation_index must be an integer", f"{location}.relation_index"))
                decision.pop("relation_index", None)
        value = str(decision.get("decision") or "").lower()
        if value not in self.RELATION_BUILDER_DECISIONS:
            warnings.append(self._warning("semantic.relation_builder_decision_unknown", f"Unknown relation builder decision: {decision.get('decision')}", f"{location}.decision"))
            decision["decision"] = "uncertain"
        else:
            decision["decision"] = value
        if "builder_actions" in decision:
            decision["builder_actions"] = self._string_list(decision.get("builder_actions"), f"{location}.builder_actions", warnings)
        if "core_constraint_action" in decision and decision.get("core_constraint_action") is not None:
            decision["core_constraint_action"] = str(decision.get("core_constraint_action"))
        if "relation_type" in decision and decision.get("relation_type") is not None:
            decision["relation_type"] = str(decision.get("relation_type"))
        if "confidence" in decision:
            try:
                decision["confidence"] = float(decision["confidence"])
            except (TypeError, ValueError):
                warnings.append(self._warning("semantic.bad_field_type", f"{location}.confidence must be numeric", f"{location}.confidence"))
                decision["confidence"] = 0.0
        if "reason" not in decision:
            decision["reason"] = ""
        return decision

    def _normalize_instruction_task_decision(self, item: Dict[str, Any], idx: int, warnings: List[Dict[str, Any]]) -> Dict[str, Any]:
        decision = dict(item)
        location = f"instruction_task_decisions[{idx}]"
        candidate_id = decision.get("candidate_id") or decision.get("from_candidate") or decision.get("id")
        if not candidate_id:
            warnings.append(self._warning("semantic.instruction_task_decision_missing_id", "Instruction task decision is missing candidate_id", location))
            decision["candidate_id"] = ""
        else:
            decision["candidate_id"] = str(candidate_id)
        value = str(decision.get("decision") or "").lower()
        if value not in self.INSTRUCTION_TASK_DECISIONS:
            warnings.append(self._warning("semantic.instruction_task_decision_unknown", f"Unknown instruction task decision: {decision.get('decision')}", f"{location}.decision"))
            decision["decision"] = "uncertain"
        else:
            decision["decision"] = value
        if "emitted_actions" in decision:
            decision["emitted_actions"] = self._string_list(decision.get("emitted_actions"), f"{location}.emitted_actions", warnings)
        if "missing_actions" in decision:
            decision["missing_actions"] = self._string_list(decision.get("missing_actions"), f"{location}.missing_actions", warnings)
        if "confidence" in decision:
            try:
                decision["confidence"] = float(decision["confidence"])
            except (TypeError, ValueError):
                warnings.append(self._warning("semantic.bad_field_type", f"{location}.confidence must be numeric", f"{location}.confidence"))
                decision["confidence"] = 0.0
        if "reason" not in decision:
            decision["reason"] = ""
        return decision

    def _normalize_instruction(self, inst: Dict[str, Any], idx: int, warnings: List[Dict[str, Any]]) -> Dict[str, Any]:
        inst = dict(inst)
        location = f"build_instructions[{idx}]"
        action = inst.get("action")
        if not action:
            warnings.append(self._warning("semantic.instruction_missing_action", "Build instruction is missing action", location))
            inst["action"] = "unknown"
        elif str(action) not in self.capabilities.supported_actions:
            warnings.append(self._warning("semantic.unsupported_instruction_action", f"Unsupported instruction action: {action}", f"{location}.action"))
            inst["action"] = str(action)
        else:
            inst["action"] = str(action)
        if "visual_ids" in inst:
            inst["visual_ids"] = self._string_list(inst.get("visual_ids"), f"{location}.visual_ids", warnings)
        if "points" in inst:
            inst["points"] = self._string_list(inst.get("points"), f"{location}.points", warnings)
        if "lines" in inst:
            inst["lines"] = self._string_list(inst.get("lines"), f"{location}.lines", warnings)
        return inst

    def _normalize_engine_assessment(self, value: Any, warnings: List[Dict[str, Any]]) -> Dict[str, Any]:
        assessment = self._dict_field(value, "engine_assessment", warnings)
        mode = assessment.get("expected_mode")
        if not mode:
            warnings.append(self._warning("semantic.engine_assessment_missing_mode", "engine_assessment.expected_mode is missing", "engine_assessment.expected_mode"))
            assessment["expected_mode"] = "failed"
        elif str(mode) not in self.ENGINE_MODES:
            warnings.append(self._warning("semantic.engine_assessment_unknown_mode", f"Unknown expected_mode: {mode}", "engine_assessment.expected_mode"))
            assessment["expected_mode"] = "failed"
        else:
            assessment["expected_mode"] = str(mode)
        if "reason" not in assessment:
            assessment["reason"] = ""
        return assessment

    def _dedupe_targets(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

    def _normalize_buildability(self, value: Any, location: str, warnings: List[Dict[str, Any]]) -> Dict[str, Any]:
        buildability = self._dict_field(value, location, warnings)
        mode = buildability.get("mode")
        allowed_modes = {"coordinate_constraint", "convertible_coordinate_constraint", "theorem_only", "unsupported", "target"}
        if mode is not None and str(mode) not in allowed_modes:
            warnings.append(self._warning("semantic.unknown_buildability_mode", f"Unknown buildability mode: {mode}", f"{location}.mode"))
            buildability["mode"] = str(mode)
        elif mode is not None:
            buildability["mode"] = str(mode)
        if "can_generate_build_instruction" in buildability:
            buildability["can_generate_build_instruction"] = self._bool_field(
                buildability.get("can_generate_build_instruction"),
                f"{location}.can_generate_build_instruction",
                warnings,
            )
        return buildability

    def _normalize_builder_mapping(self, value: Any, location: str, warnings: List[Dict[str, Any]]) -> Dict[str, Any]:
        mapping = self._dict_field(value, location, warnings)
        if "builder_actions" in mapping:
            mapping["builder_actions"] = self._string_list(mapping.get("builder_actions"), f"{location}.builder_actions", warnings)
        if "core_constraint_action" in mapping and mapping.get("core_constraint_action") is not None:
            mapping["core_constraint_action"] = str(mapping.get("core_constraint_action"))
        if "must_not_emit_fake_constraints" in mapping:
            mapping["must_not_emit_fake_constraints"] = self._bool_field(
                mapping.get("must_not_emit_fake_constraints"),
                f"{location}.must_not_emit_fake_constraints",
                warnings,
            )
        return mapping

    def _dict_list(self, value: Any, field: str, warnings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if value is None:
            return []
        if not isinstance(value, list):
            warnings.append(self._warning("semantic.bad_field_type", f"{field} must be a list", field))
            return []
        out = []
        for idx, item in enumerate(value):
            if isinstance(item, dict):
                out.append(item)
            else:
                warnings.append(self._warning("semantic.bad_item_type", f"{field}[{idx}] must be an object", f"{field}[{idx}]"))
        return out

    def _dict_field(self, value: Any, field: str, warnings: List[Dict[str, Any]]) -> Dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return dict(value)
        warnings.append(self._warning("semantic.bad_field_type", f"{field} must be an object", field))
        return {}

    def _string_list(self, value: Any, location: str, warnings: List[Dict[str, Any]]) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value if item is not None]
        warnings.append(self._warning("semantic.bad_field_type", f"{location} must be a list", location))
        return [str(value)]

    def _bool_field(self, value: Any, location: str, warnings: List[Dict[str, Any]]) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lower = value.strip().lower()
            if lower in {"true", "yes", "1"}:
                warnings.append(self._warning("semantic.coerced_bool", f"Coerced string bool at {location}", location))
                return True
            if lower in {"false", "no", "0"}:
                warnings.append(self._warning("semantic.coerced_bool", f"Coerced string bool at {location}", location))
                return False
        warnings.append(self._warning("semantic.bad_field_type", f"{location} must be boolean", location))
        return False

    def _warning(self, code: str, message: str, location: str) -> Dict[str, Any]:
        return {
            "severity": "warning",
            "code": code,
            "message": message,
            "location": location,
        }
