"""Hybrid semantic extraction with visual candidate context."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import base64
import json
import mimetypes
import os
import re
import socket
import time
import urllib.error
import urllib.request

from .builder_capabilities import GEOSYM_2D_BUILDER_CAPABILITIES, GeoSymBuilderCapabilities
from .build_plan_compiler import BuildPlanCompiler
from .instruction_pass_adapter import InstructionPassAdapter
from .instruction_task_packet import InstructionTaskPacketBuilder
from .relation_pass_adapter import RelationPassAdapter
from .semantic_role_canonicalizer import SemanticRoleCanonicalizer
from .semantic_validator import SemanticOutputValidator


DEFAULT_HYBRID_MODEL = os.environ.get("GEOSYM_OPENROUTER_MODEL", "qwen/qwen3.7-plus")
DEFAULT_OPENROUTER_TIMEOUT = int(os.environ.get("GEOSYM_OPENROUTER_TIMEOUT", "120"))
DEFAULT_OPENROUTER_MAX_RETRIES = int(os.environ.get("GEOSYM_OPENROUTER_MAX_RETRIES", "3"))


class HybridSemanticExtractor:
    def __init__(
        self,
        backend: str = "openrouter",
        model: Optional[str] = None,
        relation_model: Optional[str] = None,
        instruction_model: Optional[str] = None,
        api_key: Optional[str] = None,
        capabilities: GeoSymBuilderCapabilities = GEOSYM_2D_BUILDER_CAPABILITIES,
        extraction_mode: str = "single",
    ):
        self.backend = backend
        self.model = model or DEFAULT_HYBRID_MODEL
        self.relation_model = relation_model or self.model
        self.instruction_model = instruction_model or self.model
        self.api_key = api_key
        self.capabilities = capabilities
        self.task_packet_builder = InstructionTaskPacketBuilder()
        self.build_plan_compiler = BuildPlanCompiler()
        self.relation_pass_adapter = RelationPassAdapter()
        self.instruction_pass_adapter = InstructionPassAdapter()
        self.role_canonicalizer = SemanticRoleCanonicalizer()
        self.stage_validator = SemanticOutputValidator(capabilities=capabilities)
        if extraction_mode not in {"single", "staged"}:
            raise ValueError(f"Unsupported extraction_mode: {extraction_mode}")
        self.extraction_mode = extraction_mode

    def extract(
        self,
        *,
        image_path: str,
        problem_text: str,
        answer: Any = None,
        visual_state: Dict[str, Any],
        overlay_path: Optional[str] = None,
        script_observations: Optional[Dict[str, Any]] = None,
        evidence_graph: Optional[Dict[str, Any]] = None,
        evidence_board: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.backend == "mock":
            return self._mock(problem_text, answer, visual_state, overlay_path=overlay_path, script_observations=script_observations, evidence_graph=evidence_graph, evidence_board=evidence_board)
        if self.backend == "scripted":
            return self._scripted(problem_text, answer, visual_state, overlay_path=overlay_path, script_observations=script_observations, evidence_graph=evidence_graph, evidence_board=evidence_board)
        if self.backend == "openrouter":
            return self._openrouter(image_path, problem_text, answer, visual_state, overlay_path=overlay_path, script_observations=script_observations, evidence_graph=evidence_graph, evidence_board=evidence_board)
        raise NotImplementedError(f"Unsupported hybrid semantic backend: {self.backend}")

    def _mock(
        self,
        problem_text: str,
        answer: Any,
        visual_state: Dict[str, Any],
        overlay_path: Optional[str] = None,
        script_observations: Optional[Dict[str, Any]] = None,
        evidence_graph: Optional[Dict[str, Any]] = None,
        evidence_board: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "bindings": [],
            "relations": [],
            "build_instructions": [],
            "engine_assessment": {
                "expected_mode": "failed",
                "reason": "mock backend does not extract geometry",
            },
            "target": {},
            "metadata": {
                "backend": "mock",
                "problem_text": problem_text,
                "answer": answer,
                "overlay_path": overlay_path,
                "overlay_used": bool(overlay_path),
                "script_observations_used": bool(script_observations),
                "evidence_graph_used": bool(evidence_graph),
                "evidence_board_used": bool(evidence_board),
                "extraction_mode": self.extraction_mode,
            },
        }

    def _scripted(
        self,
        problem_text: str,
        answer: Any,
        visual_state: Dict[str, Any],
        overlay_path: Optional[str] = None,
        script_observations: Optional[Dict[str, Any]] = None,
        evidence_graph: Optional[Dict[str, Any]] = None,
        evidence_board: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Model-free baseline that converts script candidates to GeoSym JSON.

        This backend is intentionally conservative. It does not infer mathematical
        coordinates from pixels; it only turns script/text candidates into
        semantic relations and supported builder instructions. VLM backends can
        then be evaluated against this lower-bound conversion.
        """
        evidence = evidence_graph or {}
        relation_candidates = list(evidence.get("relation_candidates") or [])
        target_candidates = list(evidence.get("target_candidates") or [])
        build_suggestions = list(evidence.get("build_suggestions") or [])
        grounding_tasks = list(evidence.get("grounding_tasks") or [])
        named_object_proposals = list((evidence.get("named_object_proposals") or {}).get("items") or [])
        relations = []
        candidate_decisions = []
        grounding_decisions = self._scripted_grounding_decisions(grounding_tasks)
        named_object_bindings = self._scripted_named_object_bindings(named_object_proposals)

        for candidate in relation_candidates:
            cid = str(candidate.get("id") or "")
            ctype = str(candidate.get("type") or "")
            buildability = candidate.get("buildability") or {}
            can_emit = bool(buildability.get("can_generate_build_instruction"))
            theorem_only = buildability.get("mode") == "theorem_only"
            relation = self._relation_from_candidate(candidate)
            if relation:
                relations.append(relation)
            if cid:
                if can_emit:
                    decision = "confirm"
                    reason = "script candidate maps to supported GeoSym builder instructions"
                    confidence = 0.75
                elif theorem_only:
                    decision = "uncertain"
                    reason = "script candidate is theorem-level under current builder; extracted but not coordinate-built"
                    confidence = 0.55
                elif ctype:
                    decision = "uncertain"
                    reason = "script candidate is not supported by current GeoSym builder"
                    confidence = 0.4
                else:
                    decision = "reject"
                    reason = "script candidate has no usable relation type"
                    confidence = 0.2
                candidate_decisions.append({
                    "candidate_id": cid,
                    "decision": decision,
                    "reason": reason,
                    "confidence": confidence,
                })

        target = self._target_from_candidates(target_candidates)
        targets = self._targets_from_candidates(target_candidates)
        for candidate in target_candidates:
            cid = str(candidate.get("id") or "")
            if cid:
                candidate_decisions.append({
                    "candidate_id": cid,
                    "decision": "confirm" if target else "uncertain",
                    "reason": "script target candidate converted to set_target" if target else "script target candidate could not be normalized",
                    "confidence": 0.75 if target else 0.35,
                })

        build_instructions = self._scripted_build_instructions(build_suggestions, relation_candidates, target)
        expected_mode = self._scripted_expected_mode(relation_candidates, build_instructions)
        if target and not any(str(inst.get("action")) == "set_target" for inst in build_instructions):
            build_instructions.append({"action": "set_target", "target": target, "from_candidate": target.get("candidate_id")})
        instruction_task_decisions = self._scripted_instruction_task_decisions(evidence, build_instructions)

        return {
            "bindings": [],
            "named_object_bindings": named_object_bindings,
            "relations": relations,
            "candidate_decisions": candidate_decisions,
            "grounding_decisions": grounding_decisions,
            "instruction_task_decisions": instruction_task_decisions,
            "build_instructions": build_instructions,
            "engine_assessment": {
                "expected_mode": expected_mode,
                "reason": "model-free scripted conversion from evidence_graph candidates",
            },
            "target": target,
            "targets": targets,
            "notes": "scripted backend: no VLM; no pixel-to-coordinate reconstruction",
            "metadata": {
                "backend": "scripted",
                "problem_text": problem_text,
                "answer": answer,
                "overlay_path": overlay_path,
                "overlay_used": bool(overlay_path),
                "script_observations_used": bool(script_observations),
                "evidence_graph_used": bool(evidence_graph),
                "evidence_board_used": bool(evidence_board),
                "extraction_mode": self.extraction_mode,
                "num_relation_candidates": len(relation_candidates),
                "num_target_candidates": len(target_candidates),
                "num_grounding_tasks": len(grounding_tasks),
                "num_named_object_bindings": len(named_object_bindings),
                "num_instruction_task_decisions": len(instruction_task_decisions),
                "num_build_suggestions": len(build_suggestions),
            },
        }

    def _scripted_grounding_decisions(self, grounding_tasks: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        out: list[Dict[str, Any]] = []
        for task in grounding_tasks:
            if not isinstance(task, dict) or not task.get("id"):
                continue
            choices = [
                str(item.get("visual_id"))
                for item in task.get("visual_choices") or []
                if isinstance(item, dict) and item.get("visual_id")
            ]
            decision = "uncertain" if choices else "absent"
            out.append({
                "task_id": str(task.get("id")),
                "candidate_id": str(task.get("candidate_id") or ""),
                "semantic_ref": str(task.get("semantic_ref") or ""),
                "visual_ids": choices[:1] if choices else [],
                "decision": decision,
                "confidence": 0.35 if choices else 0.2,
                "reason": "scripted backend records visual candidates but does not verify image binding",
            })
        return out

    def _scripted_named_object_bindings(self, proposals: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        out: list[Dict[str, Any]] = []
        for proposal in proposals[:120]:
            if not isinstance(proposal, dict) or not proposal.get("id"):
                continue
            choices = [
                str(item.get("visual_id"))
                for item in proposal.get("visual_choices") or []
                if isinstance(item, dict) and item.get("visual_id")
            ]
            out.append({
                "proposal_id": str(proposal.get("id")),
                "object": str(proposal.get("object") or ""),
                "kind": str(proposal.get("kind") or ""),
                "visual_ids": choices[:1] if choices else [],
                "decision": "uncertain" if choices else "absent",
                "confidence": 0.35 if choices else 0.2,
                "reason": "scripted backend records named-object visual candidates but does not verify image binding",
            })
        return out

    def _scripted_instruction_task_decisions(self, evidence: Dict[str, Any], build_instructions: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        task_packet = self.task_packet_builder.build(
            relation_pass={
                "relations": [
                    self._relation_from_candidate(candidate)
                    for candidate in evidence.get("relation_candidates") or []
                    if self._relation_from_candidate(candidate)
                ],
                "target": self._target_from_candidates(evidence.get("target_candidates") or []),
                "targets": self._targets_from_candidates(evidence.get("target_candidates") or []),
                "candidate_decisions": [
                    {"candidate_id": candidate.get("id"), "decision": "confirm"}
                    for candidate in (evidence.get("relation_candidates") or []) + (evidence.get("target_candidates") or [])
                    if candidate.get("id")
                ],
            },
            evidence_graph=evidence,
        )
        emitted_by_candidate: dict[str, set[str]] = {}
        for instruction in build_instructions:
            if not isinstance(instruction, dict):
                continue
            cid = str(instruction.get("from_candidate") or "")
            action = str(instruction.get("action") or "")
            if cid and action:
                emitted_by_candidate.setdefault(cid, set()).add(action)
        out = []
        for task in task_packet.get("required_tasks") or []:
            cid = str(task.get("candidate_id") or "")
            if not cid:
                continue
            required_actions = [str(item) for item in task.get("required_actions") or [] if item]
            emitted_actions = sorted(emitted_by_candidate.get(cid) or set())
            missing_actions = sorted(set(required_actions) - set(emitted_actions))
            decision = "emit" if not missing_actions else "uncertain"
            out.append({
                "candidate_id": cid,
                "decision": decision,
                "emitted_actions": emitted_actions,
                "missing_actions": missing_actions,
                "reason": "scripted backend decision derived from generated build_instructions",
                "confidence": 0.75 if decision == "emit" else 0.35,
            })
        return out

    def _relation_from_candidate(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        ctype = str(candidate.get("type") or "")
        if not ctype:
            return {}
        relation: Dict[str, Any] = {
            "type": ctype,
            "candidate_id": candidate.get("id"),
            "grounded": False,
            "confidence": 0.65,
            "source": candidate.get("source") or "script",
            "raw": candidate.get("raw") or "",
        }
        if candidate.get("objects") is not None:
            relation["objects"] = [str(item) for item in (candidate.get("objects") or [])]
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

    def _target_from_candidates(self, target_candidates: list[Dict[str, Any]]) -> Dict[str, Any]:
        targets = self._targets_from_candidates(target_candidates)
        if not targets:
            return {}
        return dict(targets[0])

    def _targets_from_candidates(self, target_candidates: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        out: list[Dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for raw_candidate in target_candidates:
            if not isinstance(raw_candidate, dict):
                continue
            candidate = dict(raw_candidate)
            if not candidate.get("object"):
                continue
            cid = str(candidate.get("id") or "")
            target = {
                "type": str(candidate.get("type") or "unknown"),
                "object": str(candidate.get("object") or ""),
                "value_symbol": str(candidate.get("value_symbol") or "x"),
            }
            if cid:
                target["candidate_id"] = cid
                target["from_candidate"] = cid
            key = (cid, target["type"], target["object"])
            if key in seen:
                continue
            seen.add(key)
            out.append(target)
        return out

    def _scripted_build_instructions(
        self,
        build_suggestions: list[Dict[str, Any]],
        relation_candidates: list[Dict[str, Any]],
        target: Dict[str, Any],
    ) -> list[Dict[str, Any]]:
        supported_actions = set(self.capabilities.supported_actions)
        buildable_ids = {
            str(candidate.get("id"))
            for candidate in relation_candidates
            if (candidate.get("buildability") or {}).get("can_generate_build_instruction")
        }
        if target.get("candidate_id"):
            buildable_ids.add(str(target["candidate_id"]))
        out: list[Dict[str, Any]] = []
        seen: set[str] = set()
        for suggestion in build_suggestions:
            action = str(suggestion.get("action") or "")
            cid = str(suggestion.get("from_candidate") or "")
            if action not in supported_actions:
                continue
            if cid and cid not in buildable_ids:
                continue
            normalized = self._normalize_scripted_instruction(suggestion)
            if not normalized:
                continue
            key = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            out.append(normalized)
        return out

    def _normalize_scripted_instruction(self, suggestion: Dict[str, Any]) -> Dict[str, Any]:
        action = str(suggestion.get("action") or "")
        out: Dict[str, Any] = {"action": action}
        if suggestion.get("from_candidate"):
            out["from_candidate"] = str(suggestion.get("from_candidate"))
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
            if "type" not in target:
                target["type"] = "unknown"
            if "value_symbol" not in target:
                target["value_symbol"] = "x"
            out["target"] = target
        else:
            return {}
        return out

    def _scripted_expected_mode(self, relation_candidates: list[Dict[str, Any]], build_instructions: list[Dict[str, Any]]) -> str:
        if any((candidate.get("buildability") or {}).get("mode") == "theorem_only" for candidate in relation_candidates):
            return "theorem_only"
        if build_instructions:
            return "coordinate_buildable"
        if relation_candidates:
            return "unsupported"
        return "failed"

    def _point_id(self, value: Any) -> str:
        return "".join(ch for ch in str(value or "").upper() if ch.isalnum())[:4]

    def _line_id(self, value: Any) -> str:
        if isinstance(value, (list, tuple)):
            return "".join(self._point_id(part)[:1] for part in value[:2])
        clean = "".join(ch for ch in str(value or "").upper() if ch.isalnum())
        return clean[:2]

    def _openrouter(
        self,
        image_path: str,
        problem_text: str,
        answer: Any,
        visual_state: Dict[str, Any],
        overlay_path: Optional[str] = None,
        script_observations: Optional[Dict[str, Any]] = None,
        evidence_graph: Optional[Dict[str, Any]] = None,
        evidence_board: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        api_key = self.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for hybrid OpenRouter extraction")
        if self.extraction_mode == "staged":
            return self._openrouter_staged(image_path, problem_text, answer, visual_state, overlay_path=overlay_path, script_observations=script_observations, evidence_graph=evidence_graph, evidence_board=evidence_board)
        prompt = self._prompt(problem_text, answer, visual_state, script_observations, evidence_graph, evidence_board=evidence_board)
        content, request_audit = self._request_content(image_path=image_path, overlay_path=overlay_path, prompt=prompt)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        data, request_status = self._post_openrouter_json(
            payload=payload,
            api_key=api_key,
            stage="single",
        )
        parsed = self._parse_json_response(data["choices"][0]["message"]["content"])
        metadata = dict(parsed.get("metadata") or {})
        metadata.update({
            "backend": "openrouter",
            "model": self.model,
            "overlay_path": overlay_path,
            "overlay_used": bool(overlay_path and Path(overlay_path).exists()),
            "script_observations_used": bool(script_observations),
            "evidence_graph_used": bool(evidence_graph),
            "evidence_board_used": bool(evidence_board),
            "extraction_mode": self.extraction_mode,
            "request_audit": {**request_audit, **request_status},
        })
        parsed["metadata"] = metadata
        return parsed

    def _openrouter_staged(
        self,
        image_path: str,
        problem_text: str,
        answer: Any,
        visual_state: Dict[str, Any],
        overlay_path: Optional[str] = None,
        script_observations: Optional[Dict[str, Any]] = None,
        evidence_graph: Optional[Dict[str, Any]] = None,
        evidence_board: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        relation_prompt = self._relation_prompt(problem_text, answer, visual_state, script_observations, evidence_graph, evidence_board=evidence_board)
        relation_request = self._openrouter_request(
            image_path=image_path,
            overlay_path=overlay_path,
            prompt=relation_prompt,
            model=self.relation_model,
            stage="relation_pass",
        )
        raw_relation_pass = relation_request["parsed"]
        adapted_relation_pass = self.relation_pass_adapter.adapt(
            relation_pass=raw_relation_pass,
            evidence_graph=evidence_graph or {},
        )
        relation_validation = self.stage_validator.validate(adapted_relation_pass, stage="relation_pass")
        relation_pass = self.role_canonicalizer.canonicalize(relation_validation["semantic"])
        relation_validation = self.stage_validator.validate(relation_pass, stage="relation_pass")
        instruction_task_packet = self.task_packet_builder.build(
            relation_pass=relation_pass,
            evidence_graph=evidence_graph or {},
        )
        build_plan_seed = self.build_plan_compiler.compile(
            semantic=relation_pass,
            evidence_board=evidence_board or {},
            instruction_task_packet=instruction_task_packet,
        )
        instruction_prompt = self._instruction_prompt(problem_text, answer, visual_state, script_observations, relation_pass, evidence_graph, instruction_task_packet, evidence_board=evidence_board, build_plan_seed=build_plan_seed)
        instruction_error: Dict[str, Any] | None = None
        try:
            instruction_request = self._openrouter_request(
                image_path=image_path,
                overlay_path=overlay_path,
                prompt=instruction_prompt,
                model=self.instruction_model,
                stage="instruction_pass",
            )
            raw_instruction_pass = instruction_request["parsed"]
        except Exception as exc:
            instruction_error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "stage": "instruction_pass",
            }
            request_audit = {
                "model": self.instruction_model,
                "stage": "instruction_pass",
                "prompt_chars": len(instruction_prompt),
                "num_image_parts": relation_request["audit"].get("num_image_parts"),
                "input_images": relation_request["audit"].get("input_images") or [],
                "error": instruction_error,
            }
            instruction_request = {
                "parsed": {},
                "audit": request_audit,
            }
            raw_instruction_pass = {"error": instruction_error}
        adapted_instruction_pass = self.instruction_pass_adapter.adapt(
            instruction_pass={} if instruction_error else raw_instruction_pass,
            instruction_task_packet=instruction_task_packet,
            build_plan_seed=build_plan_seed,
        )
        instruction_validation = self.stage_validator.validate(adapted_instruction_pass, stage="instruction_pass")
        instruction_pass = instruction_validation["semantic"]
        if instruction_error:
            instruction_metadata = dict(instruction_pass.get("metadata") or {})
            instruction_metadata["instruction_pass_error"] = instruction_error
            instruction_pass["metadata"] = instruction_metadata
        merged = self._merge_staged_outputs(relation_pass, instruction_pass)
        metadata = dict(merged.get("metadata") or {})
        metadata.update({
            "backend": "openrouter",
            "model": self.model,
            "relation_model": self.relation_model,
            "instruction_model": self.instruction_model,
            "overlay_path": overlay_path,
            "overlay_used": bool(overlay_path and Path(overlay_path).exists()),
            "script_observations_used": bool(script_observations),
            "evidence_graph_used": bool(evidence_graph),
            "evidence_board_used": bool(evidence_board),
            "extraction_mode": "staged",
            "staged_relation_pass_available": bool(relation_pass),
            "staged_instruction_pass_available": bool(instruction_pass),
            "staged_instruction_pass_error": instruction_error,
            "staged_relation_pass_warnings": relation_validation["metrics"]["semantic_schema_warnings"],
            "staged_instruction_pass_warnings": instruction_validation["metrics"]["semantic_schema_warnings"],
            "relation_request_audit": relation_request["audit"],
            "instruction_request_audit": instruction_request["audit"],
        })
        merged["metadata"] = metadata
        merged["staged_outputs"] = {
            "relation_pass": relation_pass,
            "raw_relation_pass": raw_relation_pass,
            "relation_pass_validation": self._validation_summary(relation_validation),
            "instruction_task_packet": instruction_task_packet,
            "build_plan_seed": build_plan_seed,
            "instruction_pass": instruction_pass,
            "raw_instruction_pass": raw_instruction_pass,
            "instruction_pass_error": instruction_error or {},
            "instruction_pass_validation": self._validation_summary(instruction_validation),
        }
        merged["staged_transcript"] = {
            "version": "staged-transcript-v0",
            "backend": "openrouter",
            "model": self.model,
            "relation_model": self.relation_model,
            "instruction_model": self.instruction_model,
            "image_path": image_path,
            "overlay_path": overlay_path,
            "input_images": relation_request["audit"].get("input_images") or [],
            "stages": [
                {
                    "name": "relation_pass",
                    "prompt": relation_prompt,
                    "request_audit": relation_request["audit"],
                    "raw_response": raw_relation_pass,
                    "response": relation_pass,
                    "validation": self._validation_summary(relation_validation),
                },
                {
                    "name": "instruction_pass",
                    "prompt": instruction_prompt,
                    "request_audit": instruction_request["audit"],
                    "instruction_task_packet": instruction_task_packet,
                    "build_plan_seed": build_plan_seed,
                    "evidence_board": self._compact_evidence_board(evidence_board or {}, stage="instruction"),
                    "raw_response": raw_instruction_pass,
                    "response": instruction_pass,
                    "error": instruction_error or {},
                    "validation": self._validation_summary(instruction_validation),
                },
            ],
        }
        return merged

    def _validation_summary(self, validation: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "valid": validation.get("valid"),
            "warnings": validation.get("warnings") or [],
            "metrics": validation.get("metrics") or {},
        }

    def _openrouter_request(self, *, image_path: str, overlay_path: Optional[str], prompt: str, model: Optional[str] = None, stage: str = "") -> Dict[str, Any]:
        api_key = self.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for hybrid OpenRouter extraction")
        request_model = model or self.model
        content, audit = self._request_content(image_path=image_path, overlay_path=overlay_path, prompt=prompt)
        payload = {
            "model": request_model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        data, request_status = self._post_openrouter_json(
            payload=payload,
            api_key=api_key,
            stage=stage,
        )
        return {
            "parsed": self._parse_json_response(data["choices"][0]["message"]["content"]),
            "audit": {**audit, **request_status, "model": request_model, "stage": stage},
        }

    def _post_openrouter_json(self, *, payload: Dict[str, Any], api_key: str, stage: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
        url = "https://openrouter.ai/api/v1/chat/completions"
        max_retries = max(1, DEFAULT_OPENROUTER_MAX_RETRIES)
        timeout = max(1, DEFAULT_OPENROUTER_TIMEOUT)
        attempts = []
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            started = time.time()
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://geosym.local",
                    "X-Title": "GeoSym-Hybrid",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                attempts.append({
                    "attempt": attempt,
                    "ok": True,
                    "elapsed_sec": round(time.time() - started, 3),
                })
                return data, {
                    "openrouter_stage": stage,
                    "openrouter_attempts": attempts,
                    "openrouter_num_attempts": attempt,
                    "openrouter_retry_used": attempt > 1,
                    "openrouter_timeout_sec": timeout,
                    "openrouter_max_retries": max_retries,
                }
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                retryable = exc.code == 429 or 500 <= exc.code < 600
                attempts.append({
                    "attempt": attempt,
                    "ok": False,
                    "error_type": "HTTPError",
                    "status_code": exc.code,
                    "retryable": retryable,
                    "message": body[:500],
                    "elapsed_sec": round(time.time() - started, 3),
                })
                last_error = RuntimeError(f"OpenRouter HTTP {exc.code}: {body[:1000]}")
                if not retryable or attempt >= max_retries:
                    break
            except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                attempts.append({
                    "attempt": attempt,
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "retryable": True,
                    "message": str(exc)[:500],
                    "elapsed_sec": round(time.time() - started, 3),
                })
                last_error = exc
                if attempt >= max_retries:
                    break
            time.sleep(min(2 ** (attempt - 1), 8))
        message = str(last_error) if last_error else "unknown OpenRouter request failure"
        raise RuntimeError(f"OpenRouter request failed after {len(attempts)} attempt(s) at stage={stage}: {message}; attempts={attempts}") from last_error

    def _request_content(self, *, image_path: str, overlay_path: Optional[str], prompt: str) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
        content: list[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        input_images = []
        original = Path(image_path) if image_path else None
        if original and original.exists():
            content.append({"type": "image_url", "image_url": {"url": self._image_data_url(str(original))}})
            input_images.append(self._image_audit("original", original))
        elif image_path:
            input_images.append({"role": "original", "path": image_path, "exists": False})
        overlay = Path(overlay_path) if overlay_path else None
        if overlay and overlay.exists():
            content.append({"type": "image_url", "image_url": {"url": self._image_data_url(str(overlay))}})
            input_images.append(self._image_audit("overlay", overlay))
        elif overlay_path:
            input_images.append({"role": "overlay", "path": overlay_path, "exists": False})
        return content, {
            "prompt_chars": len(prompt),
            "num_content_parts": len(content),
            "num_image_parts": sum(1 for item in content if item.get("type") == "image_url"),
            "input_images": input_images,
        }

    def _image_audit(self, role: str, path: Path) -> Dict[str, Any]:
        return {
            "role": role,
            "path": str(path),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
            "mime": mimetypes.guess_type(path.name)[0] or "image/png",
        }

    def _merge_staged_outputs(self, relation_pass: Dict[str, Any], instruction_pass: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "bindings": relation_pass.get("bindings") or [],
            "named_object_bindings": relation_pass.get("named_object_bindings") or instruction_pass.get("named_object_bindings") or [],
            "relations": relation_pass.get("relations") or [],
            "candidate_decisions": relation_pass.get("candidate_decisions") or instruction_pass.get("candidate_decisions") or [],
            "grounding_decisions": relation_pass.get("grounding_decisions") or instruction_pass.get("grounding_decisions") or [],
            "relation_builder_decisions": relation_pass.get("relation_builder_decisions") or instruction_pass.get("relation_builder_decisions") or [],
            "instruction_task_decisions": instruction_pass.get("instruction_task_decisions") or [],
            "build_instructions": instruction_pass.get("build_instructions") or [],
            "engine_assessment": instruction_pass.get("engine_assessment") or relation_pass.get("engine_assessment") or {},
            "target": instruction_pass.get("target") or relation_pass.get("target") or {},
            "targets": self._merge_targets(relation_pass, instruction_pass),
            "notes": {
                "relation_pass_notes": relation_pass.get("notes") or "",
                "instruction_pass_notes": instruction_pass.get("notes") or "",
            },
        }

    def _merge_targets(self, relation_pass: Dict[str, Any], instruction_pass: Dict[str, Any]) -> list[Dict[str, Any]]:
        items: list[Dict[str, Any]] = []
        for semantic in [instruction_pass, relation_pass]:
            target = semantic.get("target")
            if isinstance(target, dict) and target:
                items.append(dict(target))
            for item in semantic.get("targets") or []:
                if isinstance(item, dict):
                    items.append(dict(item))
        out: list[Dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
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

    def _prompt(
        self,
        problem_text: str,
        answer: Any,
        visual_state: Dict[str, Any],
        script_observations: Optional[Dict[str, Any]] = None,
        evidence_graph: Optional[Dict[str, Any]] = None,
        evidence_board: Optional[Dict[str, Any]] = None,
    ) -> str:
        context = self._prompt_context(
            visual_state=visual_state,
            script_observations=script_observations,
            evidence_graph=evidence_graph,
            evidence_board=evidence_board,
            stage="single",
        )
        return f"""
You are a geometry diagram semantic parser for GeoSym.

Return JSON only. Do not output markdown.

Important:
- Your job has two stages:
  1. Read the original image/text and extract named geometry semantics.
  2. Convert buildable semantics into GeoSym builder instructions.
- VLM semantic extraction is primary. Script/OpenCV candidates are weak hints
  and review aids, not the source of truth.
- The first image is the original problem diagram. The optional second image is
  a script-generated overlay with visual candidate IDs.
- Use overlay IDs only when they help justify or review a semantic claim; do
  not treat overlay strokes, primitive ids, or pixel geometry as original facts.
- visual_ids inside build_instructions are also checked by GeoSym. If you cite
  an ID that is not in the visual candidates, that instruction will be rejected.
- Pixel coordinates are visual evidence only, not mathematical coordinates.
- If a named relation is clear from the original image/text but script visual
  candidates are poor, keep the relation named and explain the evidence instead
  of forcing a bad vp_/vs_/va_ binding.
- Prefer theorem-level relations when coordinate construction is not possible.
- Use script topology IDs such as vx_* for visible intersections and vcl_* for
  circle-line contacts only when they support a relation.
- Canonical role frame: when the image/text clearly describes an unlabeled
  right triangle using role names, convert roles into a consistent named frame
  before builder mapping: vertical_leg/height/first_leg -> AB,
  horizontal_leg/base/second_leg -> BC, hypotenuse/unknown side x -> AC, and
  right_angle(vertical_leg,horizontal_leg) -> lines ["AB","BC"]. Preserve the
  original role text in evidence/supporting_text or metadata, but relation
  objects for buildable constraints should be AB/BC/AC rather than role words.
- The script_observations block contains model-free hints. Treat them as
  weak candidate evidence, not guaranteed geometry facts. Prefer claims that
  are supported by the original image/text; use script candidate IDs to make
  later normalization and diagnosis easier.
- script_observations.semantic_candidates are text-derived relation/target
  candidates. Confirm, correct, or reject them using the original image,
  overlay ids, and problem text.
- script_observations.engine_build_suggestions are only suggested instructions.
  Copy them into build_instructions only if you can bind/confirm the required
  objects and they are supported by the current GeoSym builder capability.
- When you use a script candidate in a relation, include "candidate_id".
  When you use a script build suggestion in build_instructions, include
  "from_candidate". If you reject or change a candidate, explain in notes.
- For every evidence_graph relation/target candidate you inspect, add a
  candidate_decisions item with decision confirm|correct|reject|uncertain.
- For every evidence_graph.grounding_tasks item you inspect, add a
  grounding_decisions item with task_id, semantic_ref, candidate_id, visual_ids,
  decision bind|absent|uncertain|reject, reason, and confidence.
- For every evidence_graph.named_object_proposals item you inspect, add a
  named_object_bindings item with proposal_id, object, kind, visual_ids,
  decision bind|absent|uncertain|reject, reason, and confidence.
- If script_observations suggests the current builder cannot build a theorem
  problem, still extract relations but keep build_instructions empty or limited
  to supported scaffolding.
- Prefer the evidence_graph over raw script_observations when both are present.
- evidence_graph is the contract: confirm/correct/reject relation_candidates,
  then emit only build instructions allowed by builder_contract.supported_actions.

{self.capabilities.prompt_text()}

Output schema:
{{
  "bindings": [
    {{"label": "A", "visual_id": "vt_1|vl_2|vc_0", "kind": "point|line|circle|text", "confidence": 0.0}}
  ],
	  "relations": [
	    {{"type": "length|angle|right_angle|perpendicular|parallel|area|perimeter|ratio|triangle|circle|chord|arc_measure|intersecting_chords|similar|congruent|incidence|right_triangle|target", "objects": ["A","B"], "visual_ids": ["vl_1"], "value": "4", "candidate_id": "sc_len_0", "grounded": true, "confidence": 0.0, "source": "image|text|both", "raw": ""}}
	  ],
	  "candidate_decisions": [
	    {{"candidate_id": "sc_len_0", "decision": "confirm|correct|reject|uncertain", "corrected_relation": {{}}, "reason": "", "confidence": 0.0}}
	  ],
	  "grounding_decisions": [
	    {{"task_id": "gt_0", "candidate_id": "sc_len_0", "semantic_ref": "AB", "visual_ids": ["vl_1"], "decision": "bind|absent|uncertain|reject", "reason": "", "confidence": 0.0}}
	  ],
	  {self.capabilities.json_schema_hint()},
  "engine_assessment": {{"expected_mode": "coordinate_buildable|theorem_only|unsupported|failed", "reason": "why this sample can or cannot be built by current GeoSym 2D builder"}},
  "target": {{"type": "length|angle|arc_measure|area|perimeter|ratio|unknown", "object": "primary target object", "value_symbol": "x", "candidate_id": "sc_target_0"}},
  "targets": [
    {{"type": "length|angle|arc_measure|area|perimeter|ratio|unknown", "object": "each confirmed target object", "value_symbol": "x", "candidate_id": "sc_target_0"}}
  ],
  "notes": ""
}}

Problem text:
{problem_text}

Known dataset answer, for consistency checking only:
{answer}

Visual candidates:
{json.dumps(context["visual_candidates"], ensure_ascii=False)}

Script observations compact:
{json.dumps(context["script_observations"], ensure_ascii=False)}

Evidence graph compact:
{json.dumps(context["evidence_graph"], ensure_ascii=False)}

Geometry evidence board compact:
{json.dumps(context["evidence_board"], ensure_ascii=False)}
""".strip()

    def _relation_prompt(
        self,
        problem_text: str,
        answer: Any,
        visual_state: Dict[str, Any],
        script_observations: Optional[Dict[str, Any]] = None,
        evidence_graph: Optional[Dict[str, Any]] = None,
        evidence_board: Optional[Dict[str, Any]] = None,
    ) -> str:
        context = self._prompt_context(
            visual_state=visual_state,
            script_observations=script_observations,
            evidence_graph=evidence_graph,
            evidence_board=evidence_board,
            stage="relation",
        )
        return f"""
You are stage 1 of a GeoSym real-geometry parser.

Return JSON only. Do not output markdown.

Task:
- Read the original diagram and problem text first, then extract named geometry
  semantics: objects, labels, relations, diagram marks, givens, and target.
- Treat script/OpenCV evidence as auxiliary. It can help with recall and later
  verification, but it is not reliable enough to be the primary parser.
- Round 1 inside this response: produce named objects and relations from the
  original image/text. Use geometry_evidence_board.named_object_cards as weak
  hints; bind them to visual_choices only when helpful and credible.
- Round 2 inside this response: adjudicate every geometry_evidence_board.relation_cards
  and target_cards item against the original image/text; emit candidate_decisions,
  relations, target, and targets.
- Confirm, correct, or reject script_observations.semantic_candidates.
- Ground visible objects to overlay candidate IDs only when the overlay candidate
  clearly supports the named semantic object. Do not reject a clear named relation
  solely because OpenCV did not produce a good primitive.
- Use evidence_graph.grounding_tasks as the explicit label/segment/target
  binding checklist. For confirmed relations, include visual_ids when a
  grounding task can be resolved; if it cannot, mark the candidate uncertain
  or explain the original-image/text support instead of inventing visual ids.
- Use script_observations.visual_structure_candidates and
  evidence_graph.visual_candidates.visual_structure_candidates as the visual
  auxiliary candidate pool. vp_*/vs_*/va_*/vlabel_* ids are optional supporting
  evidence for review/alignment, never required geometry truth and never SymPy
  coordinates.
- When a text/image relation names a segment such as AB, preserve the named
  segment if it is clear from the diagram/text. Add vs_* support when available;
  if script candidates are poor, use evidence.supporting_text or a short
  diagram-mark description rather than fabricating coordinates.
- Emit one grounding_decisions item for each resolved or inspected
  evidence_graph.grounding_tasks item.
- Do not output build_instructions in this stage.
- Include candidate_id when a relation comes from a script candidate.
- Pixel coordinates are visual evidence only, not mathematical coordinates.
- visual_ids must refer to provided evidence ids such as vl_*, vc_*, vt_*,
  vx_*, vp_*, vs_*, va_*, or vlabel_*; do not invent raw coordinate arrays.
- Use evidence_graph.relation_candidates as a weak checklist. Confirm,
  correct, or reject those candidates, but also add image-derived relations
  that are visible in the original diagram even when script missed them.
- Use evidence_graph.named_object_proposals before inventing object names.
  Proposed symbols such as A, B, AB, CD, and ABC are hints from text/script.
  Emit named_object_bindings when useful, but the canonical relation objects
  should be named geometry ids read from image/text, not primitive ids.
- A VLM-discovered relation can use named_object_bindings as its visual
  grounding path: for example, bind proposal AB -> vs_0, then emit a relation
  whose object is "AB". Do not also fabricate pixel coordinates. The downstream
  task packet will verify that the bound visual_ids are in the evidence graph
  before allowing this relation to become a builder constraint.
- Extra relations that are discovered by the VLM, rather than copied from a
  script candidate, must carry verifiable evidence. Include
  evidence.visual_evidence_ids or evidence.visual_ids when the relation is
  supported by diagram candidates, and include evidence.text_refs or
  evidence.supporting_text when it is supported by problem/OCR text.
- If a VLM-discovered relation is supported by prior named_object_bindings,
  keep the relation object fields canonical and named, such as object="AB" or
  lines=["AB","CD"]. The evidence gate can inherit the valid visual support
  from named_object_bindings, but only when the object names match.
- The evidence gate accepts only provided visual ids and text references from
  the current evidence graph/problem text. Invalid visual_ids/text_refs are
  ignored; a VLM-discovered relation without remaining valid evidence will be
  deferred and cannot become a GeoSym builder constraint.
- Do not invent new geometry facts without evidence. If a relation is plausible
  but not grounded in image/text evidence, leave it out of relations or put it
  in notes instead of making it a buildable constraint.
- evidence_graph.candidate_review_checklist is the authoritative stage-1
  checklist. Emit one candidate_decisions item for every checklist item.
- evidence_graph.relation_pass_response_template is a draft valid stage-1
  response. Prefer copying its relations, target, targets, and candidate_decisions,
  then revise type, objects, values, grounding, decision, corrected_relation,
  reason, and confidence according to image/text evidence.
- Use "target" for the primary problem target and "targets" for every
  confirmed target candidate. Do not collapse multiple targets into one string.
- Each candidate_review_checklist item may include expected_relation or
  expected_target. Use these as GeoSym semantic JSON drafts; do not invent
  alternate field names when the draft already matches the schema.
- Each relation candidate includes buildability and builder_mapping metadata.
  Copy or revise these fields into relations/corrected_relation so stage 2 can
  see whether the relation is coordinate-buildable, convertible, theorem-only,
  or unsupported under the current GeoSym builder.
- Do not guess buildability. For each relation type, use
  evidence_graph.builder_contract.relation_mapping_contract as the source of
  truth:
  length/parallel/perpendicular are buildable coordinate constraints;
  right_angle is convertible only when two incident lines are explicit;
  angle/circle/area/perimeter/ratio/similar/congruent/triangle-style theorem
  relations are theorem_only and must not claim builder_actions.
- Emit candidate_decisions for all inspected relation_candidates and
  target_candidates. Use decision="correct" when the script candidate is close
  but needs different objects/type/value; put the corrected relation in
  corrected_relation.
- If a relation candidate is theorem_only under evidence_graph.buildability,
  keep it as a relation and do not imply it is coordinate-buildable.
- For buildable VLM-discovered relations, prefer canonical object fields that
  the stage-2 adapter can normalize:
  length -> object/segment like "AB" plus value;
  parallel/perpendicular -> lines like ["AB", "CD"] or objects containing two
  line/segment ids;
  right_angle -> two explicit incident lines like ["AB", "BC"].
  If you use alternative fields such as points, endpoints, segment, line1/line2,
  keep them textual/named rather than pixel coordinates.
- If the relation is a right-triangle role frame with no visible point labels,
  use the canonical symbolic frame AB/BC/AC described above instead of generic
  objects like "vertical_leg", "horizontal_leg", or "hypotenuse". Downstream
  can verify this canonical frame by answer consistency and image alignment.
- Do not use generic object descriptions such as "first line", "second line",
  "horizontal side", "visible edge", "left segment", or "the marked angle" as
  objects for buildable relations. A buildable relation must bind to named
  geometry ids such as AB/CD, points ["A","B"], endpoints, segment, line1/line2,
  or a script candidate id. If the image suggests a relation but the named
  endpoints/labels are not recoverable, leave it out of buildable relations or
  mark it uncertain in notes; otherwise downstream will record
  object_normalization_failed and defer it.
- Use geometry_evidence_board.builder_capabilities to understand what the current
  GeoSym data engine can build. Do not call theorem-only relations buildable.
- For every confirmed or corrected relation, emit one relation_builder_decisions
  item. This is the explicit bridge between visual/text geometry semantics and
  the current GeoSym builder. Copy the decision from builder_capabilities:
  length/parallel/perpendicular are buildable, right_angle is convertible only
  when incident named lines are explicit, theorem-only relation types stay
  theorem_only, and unsupported relation types stay unsupported.
- Downstream will turn your relation_builder_decisions into an
  engine_translation_summary. Missing or wrong builder decisions are treated as
  diagnosis failures even when the relation text looks geometrically plausible.
- For VLM-discovered relations that do not have a script candidate_id, set
  relation_builder_decisions[*].relation_index to the zero-based index of the
  relation in relations[]. Without this explicit builder decision, downstream
  will defer the relation even if visual/text evidence is present.
- Use geometry_evidence_board.relation_cards and target_cards as the board-aware
  checklist. If a relation appears only in the image, add it as a new relation
  candidate with named objects and explicit evidence; it is not benchmark truth
  until the downstream build plan and verifiers accept it.

Output schema:
{{
  "bindings": [
    {{"label": "A", "visual_id": "vt_1|vl_2|vc_0", "kind": "point|line|circle|text", "confidence": 0.0}}
  ],
  "named_object_bindings": [
    {{"proposal_id": "no_0", "object": "AB", "kind": "segment", "visual_ids": ["vs_0"], "decision": "bind|absent|uncertain|reject", "reason": "", "confidence": 0.0}}
  ],
	  "relations": [
	    {{"type": "length|angle|right_angle|perpendicular|parallel|area|perimeter|ratio|triangle|circle|chord|arc_measure|intersecting_chords|similar|congruent|incidence|right_triangle|target", "objects": ["A","B"], "visual_ids": ["vl_1"], "value": "4", "candidate_id": "sc_len_0", "grounded": true, "confidence": 0.0, "source": "image|text|both", "evidence": {{"visual_evidence_ids": ["vl_1"], "text_refs": ["problem_text"], "supporting_text": "AB = 4"}}, "buildability": {{"mode": "coordinate_constraint|convertible_coordinate_constraint|theorem_only|unsupported", "can_generate_build_instruction": true}}, "builder_mapping": {{"status": "buildable|convertible|theorem_only|unsupported|not_buildable", "builder_actions": ["add_point","add_segment","set_length"], "core_constraint_action": "set_length"}}, "raw": ""}}
	  ],
	  "candidate_decisions": [
	    {{"candidate_id": "sc_len_0", "decision": "confirm|correct|reject|uncertain", "corrected_relation": {{}}, "reason": "", "confidence": 0.0}}
	  ],
	  "grounding_decisions": [
	    {{"task_id": "gt_0", "candidate_id": "sc_len_0", "semantic_ref": "AB", "visual_ids": ["vl_1"], "decision": "bind|absent|uncertain|reject", "reason": "", "confidence": 0.0}}
	  ],
	  "relation_builder_decisions": [
	    {{"candidate_id": "sc_len_0", "relation_index": 0, "relation_type": "length", "decision": "buildable|convertible|theorem_only|unsupported|uncertain", "builder_actions": ["add_point","add_segment","set_length"], "core_constraint_action": "set_length", "reason": "copied from current GeoSym builder capability", "confidence": 0.0}}
	  ],
	  "target": {{"type": "length|angle|arc_measure|area|perimeter|ratio|unknown", "object": "primary target object", "value_symbol": "x", "candidate_id": "sc_target_0"}},
	  "targets": [
	    {{"type": "length|angle|arc_measure|area|perimeter|ratio|unknown", "object": "each confirmed target object", "value_symbol": "x", "candidate_id": "sc_target_0"}}
	  ],
  "engine_assessment": {{"expected_mode": "coordinate_buildable|theorem_only|unsupported|failed", "reason": "semantic-level expectation before instruction planning"}},
  "notes": ""
}}

Problem text:
{problem_text}

Known dataset answer, for consistency checking only:
{answer}

Visual candidates:
{json.dumps(context["visual_candidates"], ensure_ascii=False)}

Script observations compact:
{json.dumps(context["script_observations"], ensure_ascii=False)}

Evidence graph compact:
{json.dumps(context["evidence_graph"], ensure_ascii=False)}

Geometry evidence board compact:
{json.dumps(context["evidence_board"], ensure_ascii=False)}
""".strip()

    def _instruction_prompt(
        self,
        problem_text: str,
        answer: Any,
        visual_state: Dict[str, Any],
        script_observations: Optional[Dict[str, Any]],
        relation_pass: Dict[str, Any],
        evidence_graph: Optional[Dict[str, Any]] = None,
        instruction_task_packet: Optional[Dict[str, Any]] = None,
        evidence_board: Optional[Dict[str, Any]] = None,
        build_plan_seed: Optional[Dict[str, Any]] = None,
    ) -> str:
        context = self._prompt_context(
            visual_state=visual_state,
            script_observations=script_observations,
            evidence_graph=evidence_graph,
            evidence_board=evidence_board,
            stage="instruction",
        )
        return f"""
You are stage 2 of a GeoSym real-geometry parser.

Return JSON only. Do not output markdown.

Task:
- Convert confirmed stage-1 relations into GeoSym builder instructions.
- Use the current GeoSym builder capability exactly.
- Read geometry_evidence_board.builder_capabilities and
  instruction_task_packet.required_tasks before writing instructions. The
  buildable surface is the intersection of stage-1 confirmed relations and the
  current GeoSym data engine capability.
- Read build_plan_seed. It is a model-free compilation of stage-1 named
  relations into GeoSym actions. Prefer emitting instructions that correspond
  to build_plan_seed.tasks[*].expected_instructions. Do not emit instructions for
  build_plan_seed.blocked_relations unless you can fix the named object fields
  using explicit stage-1 evidence.
- Use stage-1 relation_builder_decisions as the capability bridge. Emit
  instructions only for relations whose decision is buildable or convertible and
  whose builder_actions are supported by builder_contract. Defer theorem_only
  relations and skip unsupported relations.
- Copy script_observations.engine_build_suggestions only after checking they
  are supported and consistent with stage-1 relations.
- Include from_candidate when an instruction comes from a script candidate.
- If relations require theorem-level solving, keep build_instructions empty or
  limited to supported scaffolding and set engine_assessment accordingly.
- Do not invent unsupported circle/arc/angle/similarity instructions.
	- Treat evidence_graph.builder_contract as the hard contract. If an action is
	  not listed in supported_actions, do not output it.
	- Use builder_contract.action_schema as the field-level schema for every
	  build instruction. Do not invent alternate fields when the schema gives an
	  exact field name such as points, lines, value, target, or from_candidate.
	- Use evidence_graph.build_tasks as the checklist for candidate-to-instruction
	  conversion. For each confirmed/corrected build task, emit all required_actions
	  when the geometry is supported.
	- instruction_task_packet.required_tasks is the authoritative checklist derived
	  from stage-1 candidate_decisions and relation/target candidate refs.
	- instruction_task_packet.response_template is a draft valid GeoSym stage-2
	  response generated from current builder capability. Prefer copying its
	  build_instructions, target, targets, and engine_assessment unless the draft conflicts
	  with stage-1 relations or the original image/text.
	- instruction_task_packet.capability_guide is the explicit relation-to-action
	  and task-decision guide. Use it to decide which candidates should be emit,
	  defer, skip, or uncertain before writing build_instructions.
	- instruction_task_packet.engine_translation_contract is the authoritative
	  stage-1-to-stage-2 conversion table. Emit build_instructions only for
	  items whose translation_decision is "emit". For every other item, copy the
	  block_reasons into instruction_task_decisions and do not fabricate
	  points, lines, lengths, or coordinates.
	- instruction_task_packet.engine_translation_summary is the compact audit of
	  that conversion table. Use it to notice missing relation_builder_decision,
	  theorem_only, unsupported, evidence-gate, and object-normalization blockers
	  before emitting any build_instructions.
	- For every required_task, emit all required_actions using supported builder
	  actions and action_schema examples unless the task is impossible; if
	  impossible, explain in instruction_task_decisions and notes.
	- Prefer instruction_task_packet.required_tasks[*].instruction_requirements:
	  each item is an expected instruction instance. Copy
	  expected_instruction exactly when it is consistent with stage-1 relations.
	  Do not collapse multiple add_point requirements into one instruction.
	- Emit one instruction_task_decisions item for every
	  instruction_task_packet.required_tasks item. Use decision="emit" when the
	  task is converted, "skip" when the stage-1 relation contradicts the task,
	  "defer" when it needs theorem-level solving, and "uncertain" when the task
	  cannot be decided from the image/text.
	- If capability_guide.task_decision_guide recommends emit, emitted_actions
	  should cover that task's required_actions. If it recommends defer, do not
	  emit fake coordinate constraints for that candidate.
	- For each relation from stage 1 that is coordinate-buildable or convertible,
	  either emit a supported instruction or explain why it cannot be built.
- Required tasks whose candidate_id starts with "vlm_rel_" come from VLM-discovered
  relations rather than script candidates. They may be emitted only if the
  relation is explicit in the image/text and builder_mapping allows it; keep
  the "vlm_rel_" from_candidate so downstream diagnosis can separate them.
- If required_tasks or relation_pass contain a right-triangle role frame, use
  canonical symbolic instructions for AB, BC, and AC:
  add_segment AB, set_length AB; add_segment BC, set_length BC;
  set_perpendicular ["AB","BC"]; add_segment AC and set_target AC when AC is the
  requested hypotenuse. Never emit role-derived fake point names such as VE,
  HO, HY, VERTICALLEG, or HORIZONTALLEG.
	- Check instruction_task_packet.vlm_derived_evidence before emitting any
	  "vlm_rel_" instruction. If passes_evidence_gate is false or no visual_ids/text_refs
	  support the relation, do not emit it; mark the task defer or uncertain.
	- instruction_task_packet.vlm_derived_evidence may include
	  named_object_binding_visual_ids. These ids mean stage 1 first bound a named
	  object such as AB to a script visual candidate such as vs_0, and the
	  VLM-derived relation then used that named object. Treat this as valid
	  explicit evidence only through the task packet; do not invent extra visual
	  ids or replace named objects with pixel coordinates.
	- Use required_tasks[*].builder_mapping and relation_buildability as the
	  contract connecting stage-1 geometry semantics to stage-2 GeoSym actions.
	  Do not emit instructions for tasks whose builder_mapping.status is
	  not_buildable except supported scaffolding explicitly listed in
	  instruction_requirements.
	- If engine_translation_contract.items[*].expected_instructions is present,
	  copy those instructions exactly for emit items. The contract already
	  encodes the current GeoSym builder capability, evidence gate, and object
	  normalization result.
	- The task packet already normalizes common VLM relation object fields
	  such as object, objects, lines, segments, points, endpoints, segment,
	  line1/line2 into named point/line ids. Use the expected_instruction
	  instances in instruction_requirements after this normalization instead of
	  re-parsing raw relation text yourself.
	- If instruction_task_packet.vlm_derived_evidence marks
	  object_normalization_failed, do not fabricate named points or lines to
	  force an instruction. Mark that task defer or uncertain and explain that
	  the relation lacks named geometry ids even if it looked visually plausible.
	- If a stage-1 relation claims buildability or builder_mapping that conflicts
	  with instruction_task_packet.required_tasks or builder_contract, follow the
	  instruction_task_packet and record skip/defer/uncertain in
	  instruction_task_decisions instead of inventing unsupported actions.

{self.capabilities.prompt_text()}

Output schema:
{{
  {self.capabilities.json_schema_hint()},
  "instruction_task_decisions": [
    {{"candidate_id": "sc_len_0", "decision": "emit|skip|defer|uncertain", "emitted_actions": ["add_point","add_segment","set_length"], "missing_actions": [], "reason": "", "confidence": 0.0}}
  ],
  "target": {{"type": "length|angle|arc_measure|area|perimeter|ratio|unknown", "object": "named object", "value_symbol": "x", "from_candidate": "sc_target_0"}},
  "targets": [
    {{"type": "length|angle|arc_measure|area|perimeter|ratio|unknown", "object": "each confirmed target object", "value_symbol": "x", "from_candidate": "sc_target_0"}}
  ],
  "engine_assessment": {{"expected_mode": "coordinate_buildable|theorem_only|unsupported|failed", "reason": "why this sample can or cannot be built by current GeoSym 2D builder"}},
  "notes": ""
}}

Problem text:
{problem_text}

Known dataset answer, for consistency checking only:
{answer}

Visual candidates:
{json.dumps(context["visual_candidates"], ensure_ascii=False)}

Script observations compact:
{json.dumps(context["script_observations"], ensure_ascii=False)}

	Stage-1 semantic extraction:
	{json.dumps(relation_pass, ensure_ascii=False)}

	Instruction task packet:
	{json.dumps(instruction_task_packet or {}, ensure_ascii=False)}

	Build plan seed:
	{json.dumps(build_plan_seed or {}, ensure_ascii=False)}

	Evidence graph compact:
	{json.dumps(context["evidence_graph"], ensure_ascii=False)}

	Geometry evidence board compact:
	{json.dumps(context["evidence_board"], ensure_ascii=False)}
""".strip()

    def _prompt_context(
        self,
        *,
        visual_state: Dict[str, Any],
        script_observations: Optional[Dict[str, Any]],
        evidence_graph: Optional[Dict[str, Any]],
        stage: str,
        evidence_board: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "visual_candidates": self._compact_visual(visual_state),
            "script_observations": self._compact_script_observations(script_observations or {}, stage=stage),
            "evidence_graph": self._compact_evidence_graph(evidence_graph or {}, stage=stage),
            "evidence_board": self._compact_evidence_board(evidence_board or {}, stage=stage),
        }

    def _compact_script_observations(self, observations: Dict[str, Any], *, stage: str) -> Dict[str, Any]:
        semantic_candidates = observations.get("semantic_candidates") or {}
        engine_suggestions = observations.get("engine_build_suggestions") or {}
        relation_hints = observations.get("relation_hints") or {}
        topology_hints = observations.get("topology_hints") or {}
        visual_structure = observations.get("visual_structure_candidates") or {}
        return {
            "source": observations.get("source"),
            "extraction_strategy": observations.get("extraction_strategy") or {},
            "image_size": observations.get("image_size"),
            "counts": observations.get("counts") or {},
            "visual_structure_candidates": {
                "summary": visual_structure.get("summary") or {},
                "point_candidates": self._take_dicts(visual_structure.get("point_candidates") or [], 80 if stage in {"single", "relation"} else 12),
                "segment_candidates": self._take_dicts(visual_structure.get("segment_candidates") or [], 60 if stage in {"single", "relation"} else 12),
                "angle_candidates": self._take_dicts(visual_structure.get("angle_candidates") or [], 40 if stage in {"single", "relation"} else 8),
                "label_region_candidates": self._take_dicts(visual_structure.get("label_region_candidates") or [], 60 if stage in {"single", "relation"} else 12),
                "coordinate_policy": visual_structure.get("coordinate_policy"),
            },
            "text_hints": observations.get("text_hints") or {},
            "semantic_candidates": {
                "relation_candidates": self._take_dicts(semantic_candidates.get("relation_candidates") or [], 80),
                "target_candidates": self._take_dicts(semantic_candidates.get("target_candidates") or [], 20),
            },
            "grounding_tasks": self._take_dicts(observations.get("grounding_tasks") or [], 80),
            "engine_build_suggestions": {
                "expected_mode_hint": engine_suggestions.get("expected_mode_hint"),
                "unsupported_candidate_types": engine_suggestions.get("unsupported_candidate_types") or [],
                "suggested_build_instructions": self._take_dicts(
                    engine_suggestions.get("suggested_build_instructions") or [],
                    100 if stage in {"single", "instruction"} else 30,
                ),
            },
            "relation_hints": {
                "parallel_line_candidates": self._take_dicts(relation_hints.get("parallel_line_candidates") or [], 8),
                "perpendicular_line_candidates": self._take_dicts(relation_hints.get("perpendicular_line_candidates") or [], 8),
                "similar_length_line_candidates": self._take_dicts(relation_hints.get("similar_length_line_candidates") or [], 8),
            },
            "topology_hints": {
                "visible_intersections": self._take_dicts(topology_hints.get("visible_intersections") or [], 12),
                "circle_line_contacts": self._take_dicts(topology_hints.get("circle_line_contacts") or [], 12),
            },
            "guidance": observations.get("guidance") or {},
        }

    def _compact_evidence_graph(self, evidence: Dict[str, Any], *, stage: str) -> Dict[str, Any]:
        builder_contract = evidence.get("builder_contract") or {}
        compact: Dict[str, Any] = {
            "version": evidence.get("version"),
            "principle": evidence.get("principle"),
            "extraction_strategy": evidence.get("extraction_strategy") or {},
            "text_evidence": evidence.get("text_evidence") or {},
            "relation_candidates": self._take_dicts(evidence.get("relation_candidates") or [], 80),
            "target_candidates": self._take_dicts(evidence.get("target_candidates") or [], 20),
            "grounding_tasks": self._take_dicts(evidence.get("grounding_tasks") or [], 100),
            "candidate_review_checklist": self._take_dicts(evidence.get("candidate_review_checklist") or [], 100),
            "builder_contract": {
                "capability_version": builder_contract.get("capability_version"),
                "supported_actions": builder_contract.get("supported_actions") or [],
                "action_schema": builder_contract.get("action_schema") or [],
                "supported_constraint_types": builder_contract.get("supported_constraint_types") or [],
                "theorem_relation_types": builder_contract.get("theorem_relation_types") or [],
                "relation_mapping_contract": builder_contract.get("relation_mapping_contract") or [],
                "unsupported_candidate_types": builder_contract.get("unsupported_candidate_types") or [],
                "rule": builder_contract.get("rule") or "",
            },
            "vlm_contract": evidence.get("vlm_contract") or {},
        }
        if stage in {"single", "relation"}:
            compact["visual_candidates"] = evidence.get("visual_candidates") or {}
        if stage in {"single", "relation"}:
            compact["relation_pass_response_template"] = evidence.get("relation_pass_response_template") or {}
        if stage in {"single", "instruction"}:
            compact["build_tasks"] = self._take_dicts(evidence.get("build_tasks") or [], 100)
            compact["build_suggestions"] = self._take_dicts(evidence.get("build_suggestions") or [], 120)
        return compact

    def _compact_evidence_board(self, board: Dict[str, Any], *, stage: str) -> Dict[str, Any]:
        if not board:
            return {}
        compact: Dict[str, Any] = {
            "version": board.get("version"),
            "principle": board.get("principle"),
            "parsing_policy": board.get("parsing_policy") or {},
            "image": board.get("image") or {},
            "visual_candidate_summary": board.get("visual_candidate_summary") or {},
            "named_object_cards": self._take_dicts(board.get("named_object_cards") or [], 140 if stage == "relation" else 40),
            "relation_cards": self._take_dicts(board.get("relation_cards") or [], 100 if stage in {"single", "relation"} else 40),
            "target_cards": self._take_dicts(board.get("target_cards") or [], 30),
            "builder_capabilities": board.get("builder_capabilities") or {},
            "vlm_contract": board.get("vlm_contract") or {},
        }
        if stage in {"single", "relation"}:
            compact["topology_cards"] = self._take_dicts(board.get("topology_cards") or [], 80)
            compact["text_evidence"] = board.get("text_evidence") or {}
        return compact

    def _compact_visual(self, visual_state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "image_size": visual_state.get("image_size"),
            "lines": (visual_state.get("lines") or [])[:40],
            "circles": (visual_state.get("circles") or [])[:10],
            "text_regions": (visual_state.get("text_regions") or [])[:40],
            "topology": {
                "line_intersections": ((visual_state.get("topology") or {}).get("line_intersections") or [])[:40],
                "circle_line_hits": ((visual_state.get("topology") or {}).get("circle_line_hits") or [])[:40],
            },
        }

    def _take_dicts(self, items: list[Dict[str, Any]], limit: int) -> list[Dict[str, Any]]:
        return [dict(item) for item in items[:limit] if isinstance(item, dict)]

    def _image_data_url(self, image_path: str) -> str:
        path = Path(image_path)
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise ValueError(f"Hybrid semantic response did not contain JSON: {text[:500]}")
        data = json.loads(match.group(0))
        if not isinstance(data, dict):
            raise ValueError("Hybrid semantic response JSON is not an object")
        return data
