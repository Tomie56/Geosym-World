"""Ground VLM semantic relations against script-detected visual candidates."""

from __future__ import annotations

from typing import Any, Dict, List


class RelationGrounder:
    def ground(self, semantic: Dict[str, Any], visual_state: Dict[str, Any]) -> Dict[str, Any]:
        valid_ids = self._valid_visual_ids(visual_state)
        accepted: List[Dict[str, Any]] = []
        uncertain: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []

        for rel in semantic.get("relations") or []:
            rel = dict(rel)
            visual_ids = [str(v) for v in rel.get("visual_ids") or []]
            missing = [vid for vid in visual_ids if vid not in valid_ids]
            if missing:
                rel["grounding_status"] = "rejected"
                rel["grounding_reason"] = f"unknown visual ids: {missing}"
                rejected.append(rel)
                continue
            if rel.get("grounded") is True and visual_ids:
                rel["grounding_status"] = "accepted"
                accepted.append(rel)
            elif rel.get("type") in {"target", "intersecting_chords", "similar", "congruent"}:
                rel["grounding_status"] = "uncertain"
                rel["grounding_reason"] = "semantic/theorem relation without full visual binding"
                uncertain.append(rel)
            else:
                rel["grounding_status"] = "uncertain"
                rel["grounding_reason"] = "ungrounded or no visual candidate ids"
                uncertain.append(rel)

        bindings = []
        for binding in semantic.get("bindings") or []:
            binding = dict(binding)
            vid = str(binding.get("visual_id", ""))
            binding["grounding_status"] = "accepted" if vid in valid_ids else "rejected"
            bindings.append(binding)

        return {
            "bindings": bindings,
            "accepted_relations": accepted,
            "uncertain_relations": uncertain,
            "rejected_relations": rejected,
            "target": semantic.get("target") or {},
            "semantic": semantic,
        }

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
