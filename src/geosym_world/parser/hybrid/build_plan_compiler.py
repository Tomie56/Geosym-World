"""Compile verified/adjudicated relations into model-free build-plan tasks."""

from __future__ import annotations

from typing import Any, Dict, List

from .relation_object_normalizer import RelationObjectNormalizer


class BuildPlanCompiler:
    """Turn relation evidence into builder-level tasks without using a VLM."""

    SUPPORTED = {"length", "parallel", "perpendicular", "right_angle"}

    def __init__(self) -> None:
        self.normalizer = RelationObjectNormalizer()

    def compile(
        self,
        *,
        semantic: Dict[str, Any],
        evidence_board: Dict[str, Any],
        instruction_task_packet: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        bindings = self._binding_index(semantic)
        tasks: List[Dict[str, Any]] = []
        blocked: List[Dict[str, Any]] = []
        for relation in semantic.get("relations") or []:
            if not isinstance(relation, dict):
                continue
            relation_id = str(relation.get("candidate_id") or relation.get("from_candidate") or f"rel_{len(tasks) + len(blocked)}")
            compiled = self._compile_relation(relation, relation_id, bindings)
            if compiled.get("task"):
                tasks.append(compiled["task"])
            else:
                blocked.append(compiled["blocked"])

        packet_metrics = (instruction_task_packet or {}).get("metrics") or {}
        return {
            "version": "build-plan-v0",
            "principle": "build plan is compiled model-free from adjudicated named relations; no pixel coordinates are used",
            "tasks": tasks,
            "blocked_relations": blocked,
            "source_metrics": {
                "num_semantic_relations": len(semantic.get("relations") or []),
                "num_named_object_bindings": len(semantic.get("named_object_bindings") or []),
                "num_evidence_board_named_cards": len(evidence_board.get("named_object_cards") or []),
                "num_instruction_packet_required_tasks": packet_metrics.get("num_required_tasks", 0),
            },
            "metrics": {
                "num_tasks": len(tasks),
                "num_expected_instructions": sum(len(item.get("expected_instructions") or []) for item in tasks),
                "num_blocked_relations": len(blocked),
                "num_blocked_unsupported": sum(1 for item in blocked if item.get("blocker") == "unsupported_relation_type"),
                "num_blocked_object_normalization": sum(1 for item in blocked if item.get("blocker") == "object_normalization_failed"),
            },
        }

    def _compile_relation(
        self,
        relation: Dict[str, Any],
        relation_id: str,
        bindings: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        rtype = str(relation.get("type") or "")
        if rtype not in self.SUPPORTED:
            return {"blocked": self._blocked(relation_id, relation, "unsupported_relation_type")}
        if rtype == "length":
            segment = self.normalizer.line_id_from_relation(relation)
            if not segment:
                return {"blocked": self._blocked(relation_id, relation, "object_normalization_failed")}
            return {"task": self._task(
                relation_id,
                relation,
                "set_length",
                {"points": [segment[0], segment[1]], "value": str(relation.get("value") or "")},
                bindings,
                [segment],
            )}
        if rtype in {"parallel", "perpendicular", "right_angle"}:
            lines = self.normalizer.lines_from_relation(relation)
            if len(lines) < 2:
                return {"blocked": self._blocked(relation_id, relation, "object_normalization_failed")}
            action = "set_perpendicular" if rtype in {"perpendicular", "right_angle"} else "set_parallel"
            return {"task": self._task(
                relation_id,
                relation,
                action,
                {"lines": lines[:2]},
                bindings,
                lines[:2],
            )}
        return {"blocked": self._blocked(relation_id, relation, "unsupported_relation_type")}

    def _task(
        self,
        relation_id: str,
        relation: Dict[str, Any],
        action: str,
        args: Dict[str, Any],
        bindings: Dict[str, Dict[str, Any]],
        objects: List[str],
    ) -> Dict[str, Any]:
        binding_refs = [bindings[obj] for obj in objects if obj in bindings]
        visual_ids: List[str] = []
        for binding in binding_refs:
            visual_ids.extend(str(item) for item in (binding.get("visual_ids") or []) if item)
        visual_ids = sorted(set(visual_ids))
        expected_instructions = self._expected_instructions(
            action=action,
            args=args,
            relation_id=relation_id,
            visual_ids=visual_ids,
        )
        return {
            "id": f"bt_{relation_id}",
            "from_relation": relation_id,
            "action": action,
            "args": args,
            "expected_instructions": expected_instructions,
            "required_actions": [item.get("action") for item in expected_instructions if item.get("action")],
            "review_required": str(relation_id).startswith("vlm_rel_") or relation.get("source") == "vlm",
            "evidence": {
                "named_objects": objects,
                "named_object_bindings": binding_refs,
                "visual_ids": visual_ids,
                "text_refs": ((relation.get("evidence") or {}).get("text_refs") or []),
            },
            "relation": relation,
        }

    def _expected_instructions(
        self,
        *,
        action: str,
        args: Dict[str, Any],
        relation_id: str,
        visual_ids: List[str],
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []

        def add_point(point: str) -> None:
            if point and not any(item.get("action") == "add_point" and item.get("id") == point for item in out):
                inst = {"action": "add_point", "id": point, "from_candidate": relation_id}
                if visual_ids:
                    inst["visual_ids"] = visual_ids
                out.append(inst)

        def add_segment(line: str) -> None:
            if len(line) < 2:
                return
            add_point(line[0])
            add_point(line[1])
            inst = {"action": "add_segment", "points": [line[0], line[1]], "from_candidate": relation_id}
            if visual_ids:
                inst["visual_ids"] = visual_ids
            if not any(item.get("action") == "add_segment" and item.get("points") == inst["points"] for item in out):
                out.append(inst)

        if action == "set_length":
            points = [str(item) for item in (args.get("points") or [])[:2]]
            if len(points) >= 2:
                line = "".join(points[:2])
                add_segment(line)
                inst = {
                    "action": "set_length",
                    "points": points[:2],
                    "value": str(args.get("value") or ""),
                    "from_candidate": relation_id,
                }
                if visual_ids:
                    inst["visual_ids"] = visual_ids
                out.append(inst)
        elif action in {"set_parallel", "set_perpendicular"}:
            lines = [str(item) for item in (args.get("lines") or [])[:2]]
            for line in lines:
                add_segment(line)
            out.append({"action": action, "lines": lines, "from_candidate": relation_id})
        return out

    def _blocked(self, relation_id: str, relation: Dict[str, Any], blocker: str) -> Dict[str, Any]:
        return {
            "id": f"blocked_{relation_id}",
            "from_relation": relation_id,
            "blocker": blocker,
            "relation": relation,
        }

    def _binding_index(self, semantic: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for item in semantic.get("named_object_bindings") or []:
            if not isinstance(item, dict):
                continue
            obj = str(item.get("object") or "").upper()
            if obj:
                out[obj] = item
        return out
