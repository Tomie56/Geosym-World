"""Validate VLM-proposed GeoSym builder instructions.

The VLM proposes instructions; this module decides which instructions are
compatible with the current 2D GeoSym canonical builder. It deliberately keeps
the supported instruction set small and explicit.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .builder_capabilities import GEOSYM_2D_BUILDER_CAPABILITIES, GeoSymBuilderCapabilities
from .relation_object_normalizer import RelationObjectNormalizer


class BuildInstructionPlanner:
    def __init__(self, capabilities: GeoSymBuilderCapabilities = GEOSYM_2D_BUILDER_CAPABILITIES):
        self.capabilities = capabilities
        self.object_normalizer = RelationObjectNormalizer()

    def plan(self, semantic: Dict[str, Any], formal: Dict[str, Any], visual_state: Dict[str, Any] | None = None) -> Dict[str, Any]:
        instructions = list(semantic.get("build_instructions") or [])
        valid_visual_ids = self._valid_visual_ids(visual_state or {})
        accepted: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        synthesized: List[Dict[str, Any]] = []
        points: set[str] = set()
        lines: set[Tuple[str, str]] = set()
        constraints: List[Dict[str, Any]] = []
        target_candidates: List[Dict[str, Any]] = []
        initial_target = dict(semantic.get("target") or formal.get("target") or {})
        if initial_target:
            target_candidates.append(initial_target)
        target: Dict[str, Any] = dict(initial_target)

        for idx, raw in enumerate(instructions):
            inst = dict(raw)
            action = str(inst.get("action") or "")
            reason = self._reject_reason(action, inst, valid_visual_ids)
            if reason:
                inst["instruction_status"] = "rejected"
                inst["instruction_reason"] = reason
                inst["instruction_index"] = idx
                rejected.append(inst)
                continue
            normalized = self._normalize_instruction(inst)
            action = normalized["action"]
            if action == "add_point":
                points.add(normalized["id"])
            elif action == "add_segment":
                a, b = normalized["points"]
                points.update([a, b])
                lines.add(tuple(sorted((a, b))))
            elif action == "set_length":
                a, b = normalized["points"]
                points.update([a, b])
                lines.add(tuple(sorted((a, b))))
                constraints.append({"type": "length", "object": f"{a}{b}", "value": normalized["value"], "source": "vlm_instruction"})
            elif action in {"set_parallel", "set_perpendicular"}:
                line_ids = normalized["lines"]
                for line in line_ids:
                    if len(line) >= 2:
                        a, b = line[0], line[1]
                        points.update([a, b])
                        lines.add(tuple(sorted((a, b))))
                constraints.append({"type": action.replace("set_", ""), "lines": line_ids, "source": "vlm_instruction"})
            elif action == "set_target":
                target_candidates.append(dict(normalized["target"]))
            normalized["instruction_status"] = "accepted"
            normalized["instruction_index"] = idx
            accepted.append(normalized)

        relation_graph = self._relation_graph_fallback(formal)
        if not instructions and formal.get("mode") != "theorem_only":
            synthesized = self._synthesize_instructions_from_relation_graph(relation_graph, target)
            for idx, inst in enumerate(synthesized):
                inst["instruction_status"] = "accepted"
                inst["instruction_index"] = idx
                inst["instruction_source"] = "synthesized_from_relations"
            accepted.extend(synthesized)
        if relation_graph["constraints"] and not constraints:
            constraints = relation_graph["constraints"]
        if relation_graph["points"]:
            points.update(relation_graph["points"])
        if relation_graph["lines"]:
            lines.update(tuple(sorted(line)) for line in relation_graph["lines"])
        target = self._select_target(target_candidates, formal)
        self._add_target_scaffold(target, points, lines)

        constraint_graph = {
            "objects": {
                "points": sorted(points),
                "lines": [list(item) for item in sorted(lines)],
            },
            "constraints": constraints,
            "target": target,
            "answer": formal.get("answer"),
            "extraction": formal,
            "instruction_plan": {
                "accepted": accepted,
                "rejected": rejected,
                "synthesized": synthesized,
            },
        }
        unsupported_actions = sorted({str(inst.get("action")) for inst in instructions if str(inst.get("action")) not in self.capabilities.supported_actions})
        buildable = bool(lines) and all(c.get("type") in self.capabilities.supported_constraint_types for c in constraints)
        if formal.get("mode") == "theorem_only":
            buildable = False
            reason = "formal graph requires theorem-level solver"
        elif formal.get("requires_theorem_solver") and buildable:
            reason = "coordinate scaffold is compatible with current canonical builder; theorem relations remain for diagnosis"
        elif unsupported_actions:
            reason = f"unsupported instruction actions: {unsupported_actions}"
        elif not instructions and not relation_graph["constraints"]:
            reason = "no build instructions or coordinate constraints"
        elif not lines:
            reason = "no segment/line instructions"
        elif buildable:
            reason = "instructions are compatible with current canonical builder" if instructions else "synthesized build instructions from grounded relations"
        else:
            reason = "instructions contain unsupported constraints"

        return {
            "buildable": buildable,
            "reason": reason,
            "accepted_instructions": accepted,
            "rejected_instructions": rejected,
            "synthesized_instructions": synthesized,
            "constraint_graph": constraint_graph,
            "num_instructions": len(instructions),
            "num_accepted": len(accepted),
            "num_rejected": len(rejected),
            "num_synthesized": len(synthesized),
        }

    def _select_target(self, candidates: List[Dict[str, Any]], formal: Dict[str, Any]) -> Dict[str, Any]:
        all_candidates = [dict(item) for item in candidates if isinstance(item, dict) and item]
        formal_target = formal.get("target")
        if isinstance(formal_target, dict) and formal_target:
            all_candidates.append(dict(formal_target))
        if not all_candidates:
            return {}
        selected = max(all_candidates, key=self._target_rank)
        return selected if self._target_is_usable(selected) else {}

    def _target_rank(self, target: Dict[str, Any]) -> tuple[int, int, int, int, int]:
        target_type = str(target.get("type") or "").lower()
        obj = target.get("object")
        line = self.object_normalizer.line_from_relation({"object": obj})
        source = str(target.get("source") or target.get("repair_source") or "").lower()
        metadata = target.get("metadata") or {}
        return (
            1 if target_type and target_type != "unknown" else 0,
            1 if target_type == "length" else 0,
            1 if len(line) >= 2 else 0,
            1 if metadata.get("semantic_role_canonicalized") else 0,
            0 if source in {"text_regex", "script_build_suggestion"} else 1,
        )

    def _target_is_usable(self, target: Dict[str, Any]) -> bool:
        if not isinstance(target, dict) or not target:
            return False
        target_type = str(target.get("type") or "").lower()
        obj = str(target.get("object") or "").strip()
        if not target_type or target_type == "unknown" or not obj or obj.upper() == "X":
            return False
        if target_type == "length":
            return len(self.object_normalizer.line_from_relation({"object": obj})) >= 2
        return target_type in {"angle", "arc_measure", "area", "entity_area", "shadow_area", "perimeter", "ratio"}

    def _add_target_scaffold(self, target: Dict[str, Any], points: set[str], lines: set[Tuple[str, str]]) -> None:
        if str(target.get("type") or "").lower() != "length":
            return
        line = self.object_normalizer.line_from_relation({"object": target.get("object")})
        if len(line) < 2:
            return
        a, b = line[0], line[1]
        points.update([a, b])
        lines.add(tuple(sorted((a, b))))

    def _reject_reason(self, action: str, inst: Dict[str, Any], valid_visual_ids: set[str]) -> str:
        if action not in self.capabilities.supported_actions:
            return f"unsupported action: {action or '<missing>'}"
        visual_ids = [str(item) for item in inst.get("visual_ids") or []]
        missing = [item for item in visual_ids if item not in valid_visual_ids]
        if missing:
            return f"unknown visual ids: {missing}"
        if action == "add_point" and not inst.get("id"):
            return "add_point requires id"
        if action in {"add_segment", "set_length"} and len(inst.get("points") or []) < 2:
            return f"{action} requires two points"
        if action == "set_length" and inst.get("value") in {None, ""}:
            return "set_length requires value"
        if action in {"set_parallel", "set_perpendicular"} and len(self.object_normalizer.lines_from_relation({"lines": inst.get("lines") or []})) < 2:
            return f"{action} requires two lines"
        if action == "set_target" and not (inst.get("target") or inst.get("object")):
            return "set_target requires target or object"
        return ""

    def _valid_visual_ids(self, visual_state: Dict[str, Any]) -> set[str]:
        out: set[str] = set()
        for key in ["lines", "circles", "text_regions"]:
            for item in visual_state.get(key) or []:
                if item.get("id"):
                    out.add(str(item["id"]))
        topology = visual_state.get("topology") or {}
        for key in ["endpoints", "line_intersections", "circle_line_hits"]:
            for item in topology.get(key) or []:
                if item.get("id"):
                    out.add(str(item["id"]))
        structure = visual_state.get("visual_structure_candidates") or {}
        for key in ["point_candidates", "segment_candidates", "angle_candidates", "label_region_candidates"]:
            for item in structure.get(key) or []:
                if item.get("id"):
                    out.add(str(item["id"]))
                if item.get("visual_id"):
                    out.add(str(item["visual_id"]))
        return out

    def _normalize_instruction(self, inst: Dict[str, Any]) -> Dict[str, Any]:
        action = str(inst.get("action") or "")
        out = dict(inst)
        out["action"] = action
        if action == "add_point":
            out["id"] = self._point_id(inst.get("id"))
        elif action in {"add_segment", "set_length"}:
            out["points"] = [self._point_id(p) for p in (inst.get("points") or [])[:2]]
        elif action in {"set_parallel", "set_perpendicular"}:
            out["lines"] = self.object_normalizer.lines_from_relation({"lines": inst.get("lines") or []})[:2]
        elif action == "set_target":
            out["target"] = dict(inst.get("target") or {})
            if not out["target"]:
                out["target"] = {
                    "type": inst.get("target_type") or "unknown",
                    "object": inst.get("object"),
                    "value_symbol": inst.get("value_symbol") or "x",
                }
        return out

    def _relation_graph_fallback(self, formal: Dict[str, Any]) -> Dict[str, Any]:
        objects = formal.get("objects") or {}
        constraints = []
        for rel in formal.get("relations") or []:
            rtype = rel.get("type")
            if rtype == "length":
                line = self.object_normalizer.line_from_relation(rel)
                if line:
                    constraints.append({"type": "length", "object": "".join(line), "value": rel.get("value"), "source": "grounded_relation"})
            elif rtype in {"parallel", "perpendicular"}:
                lines = self.object_normalizer.lines_from_relation(rel)[:2]
                if len(lines) >= 2:
                    constraints.append({"type": rtype, "lines": lines, "source": "grounded_relation"})
            elif rtype == "right_angle":
                lines = self.object_normalizer.lines_from_relation(rel)[:2]
                if len(lines) >= 2:
                    constraints.append({"type": "perpendicular", "lines": lines, "source": "grounded_relation"})
        return {
            "points": list(objects.get("points") or []),
            "lines": [tuple(line) for line in objects.get("lines") or [] if len(line) >= 2],
            "constraints": constraints,
        }

    def _synthesize_instructions_from_relation_graph(self, relation_graph: Dict[str, Any], target: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for point in sorted(set(relation_graph.get("points") or [])):
            out.append({"action": "add_point", "id": point})
        for line in sorted(set(tuple(item) for item in relation_graph.get("lines") or [])):
            if len(line) >= 2:
                out.append({"action": "add_segment", "points": [line[0], line[1]]})
        for constraint in relation_graph.get("constraints") or []:
            ctype = constraint.get("type")
            if ctype == "length":
                line = self.object_normalizer.line_from_relation({"object": constraint.get("object")})
                if line:
                    out.append({"action": "set_length", "points": [line[0], line[1]], "value": constraint.get("value")})
            elif ctype in {"parallel", "perpendicular"}:
                out.append({"action": f"set_{ctype}", "lines": constraint.get("lines") or []})
        if target:
            out.append({"action": "set_target", "target": target})
        return out

    def _line_from_objects(self, objs: List[Any]) -> Tuple[str, str] | None:
        line = self.object_normalizer.line_from_relation({"objects": objs})
        return (line[0], line[1]) if len(line) >= 2 else None

    def _point_id(self, value: Any) -> str:
        return self.object_normalizer.point_id(value)

    def _line_id(self, value: Any) -> str:
        return self.object_normalizer.line_id(value)
