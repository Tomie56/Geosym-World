"""Geometry evidence board for VLM-first, script-verified parsing."""

from __future__ import annotations

from typing import Any, Dict, List

from .builder_capabilities import GEOSYM_2D_BUILDER_CAPABILITIES, GeoSymBuilderCapabilities


class GeometryEvidenceBoardBuilder:
    """Create the stable board that VLM passes should consult.

    The board is not an OpenCV-to-geometry interface. It gives the VLM weak
    script hints and the current GeoSym builder contract; the VLM still reads
    the original image/text first and the script later verifies/converts.
    """

    def __init__(self, capabilities: GeoSymBuilderCapabilities = GEOSYM_2D_BUILDER_CAPABILITIES):
        self.capabilities = capabilities

    def build(
        self,
        *,
        visual_state: Dict[str, Any],
        script_observations: Dict[str, Any],
        text_claims: Dict[str, Any],
        evidence_graph: Dict[str, Any],
    ) -> Dict[str, Any]:
        visual_structure = script_observations.get("visual_structure_candidates") or {}
        named_cards = [
            self._named_object_card(item)
            for item in (text_claims.get("named_objects") or [])
            if isinstance(item, dict)
        ]
        relation_cards = [
            self._relation_card(item)
            for item in (text_claims.get("relation_claims") or [])
            if isinstance(item, dict)
        ]
        target_cards = [
            self._target_card(item)
            for item in (text_claims.get("target_claims") or [])
            if isinstance(item, dict)
        ]
        return {
            "version": "geometry-evidence-board-v0",
            "principle": "VLM-first semantic extraction; script candidates are weak hints and post-VLM normalization/verification inputs",
            "parsing_policy": {
                "primary_extractor": "VLM reads the original diagram/problem text for named geometry semantics",
                "script_evidence_role": "text regex, dataset metadata, and visual primitives provide recall-oriented hints, not hard geometry truth",
                "visual_primitive_role": "optional auxiliary evidence for review/alignment; do not require every relation to bind to vp_/vs_/va_ ids",
                "builder_role": "GeoSym builder capability decides which VLM relations can become canonical instructions",
            },
            "image": {
                "path": visual_state.get("image_path"),
                "size": visual_state.get("image_size"),
            },
            "visual_candidate_summary": self._visual_summary(visual_state, visual_structure),
            "fill_region_cards": [dict(item) for item in (visual_state.get("fill_regions") or [])[:40]],
            "marker_cards": [dict(item) for item in (visual_state.get("markers") or [])[:40]],
            "named_object_cards": named_cards,
            "relation_cards": relation_cards,
            "target_cards": target_cards,
            "topology_cards": self._topology_cards(visual_state),
            "text_evidence": {
                "text_sources": text_claims.get("text_sources") or [],
                "raw_text_hints": text_claims.get("raw_text_hints") or {},
            },
            "builder_capabilities": {
                "version": self.capabilities.version,
                "supported_actions": sorted(self.capabilities.supported_actions),
                "action_schema": self.capabilities.action_schema_contract(),
                "supported_constraint_types": sorted(self.capabilities.supported_constraint_types),
                "theorem_relation_types": sorted(self.capabilities.theorem_relation_types),
                "relation_mapping_contract": self.capabilities.relation_mapping_contract(),
            },
            "legacy_evidence_graph_ref": {
                "version": evidence_graph.get("version"),
                "num_relation_candidates": len(evidence_graph.get("relation_candidates") or []),
                "num_target_candidates": len(evidence_graph.get("target_candidates") or []),
            },
            "vlm_contract": {
                "round_1": "read the original image/text and extract named objects, labels, relations, diagram marks, and target",
                "round_2": "use named_object_cards/relation_cards as weak checklists: confirm/correct/reject them and add image-derived relations with named objects",
                "round_3": "for each relation, state whether current GeoSym can build it via relation_builder_decisions",
                "rule": "do not output GeoSym coordinates or builder instructions",
            },
            "metrics": {
                "num_named_object_cards": len(named_cards),
                "num_relation_cards": len(relation_cards),
                "num_target_cards": len(target_cards),
                "num_topology_cards": len(self._topology_cards(visual_state)),
            },
        }

    def _named_object_card(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(item.get("id") or ""),
            "kind": str(item.get("kind") or ""),
            "object": str(item.get("object") or ""),
            "source": item.get("source") or "",
            "visual_choices": list(item.get("visual_choices") or []),
            "candidate_ids": item.get("candidate_ids") or [],
            "expected_vlm_output": {
                "proposal_id": str(item.get("id") or ""),
                "object": str(item.get("object") or ""),
                "decision": "bind|absent|uncertain|reject",
                "visual_ids": [],
                "confidence": 0.0,
                "reason": "use visual_choices only as optional evidence; if the original image clearly shows the object but choices are poor, explain and keep named object semantic",
            },
        }

    def _relation_card(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(item.get("id") or ""),
            "type": str(item.get("type") or ""),
            "object": item.get("object"),
            "objects": item.get("objects") or [],
            "value": item.get("value"),
            "source": item.get("source") or "",
            "source_text": item.get("source_text") or "",
            "needs_visual_binding": bool(item.get("needs_visual_binding")),
            "buildability": item.get("buildability") or {},
            "builder_mapping": item.get("builder_mapping") or {},
            "expected_vlm_output": {
                "candidate_id": str(item.get("id") or ""),
                "decision": "confirm|correct|reject|uncertain",
                "relation": {
                    "type": str(item.get("type") or ""),
                    "object": item.get("object"),
                    "objects": item.get("objects") or [],
                    "value": item.get("value"),
                },
                "evidence": {
                    "named_object_ids": [],
                    "visual_ids": [],
                    "text_refs": [],
                },
            },
        }

    def _target_card(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(item.get("id") or ""),
            "type": str(item.get("type") or ""),
            "object": item.get("object"),
            "objects": item.get("objects") or [],
            "value_symbol": item.get("value_symbol"),
            "source": item.get("source") or "",
            "expected_vlm_output": {
                "candidate_id": str(item.get("id") or ""),
                "decision": "confirm|correct|reject|uncertain",
            },
        }

    def _topology_cards(self, visual_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        topology = visual_state.get("topology") or {}
        out: List[Dict[str, Any]] = []
        for item in (topology.get("line_intersections") or [])[:80]:
            out.append({
                "id": str(item.get("id") or ""),
                "type": "line_intersection",
                "visual_id": item.get("id"),
                "line_ids": item.get("line_ids") or [],
                "xy_px": item.get("xy"),
                "coordinate_policy": "visual evidence only",
            })
        for item in (topology.get("circle_line_hits") or [])[:80]:
            out.append({
                "id": str(item.get("id") or ""),
                "type": "circle_line_contact",
                "visual_id": item.get("id"),
                "circle_id": item.get("circle_id"),
                "line_id": item.get("line_id"),
                "coordinate_policy": "visual evidence only",
            })
        return out

    def _visual_summary(self, visual_state: Dict[str, Any], visual_structure: Dict[str, Any]) -> Dict[str, Any]:
        summary = visual_structure.get("summary") or {}
        return {
            "num_lines": len(visual_state.get("lines") or []),
            "num_circles": len(visual_state.get("circles") or []),
            "num_fill_regions": len(visual_state.get("fill_regions") or []),
            "num_markers": len(visual_state.get("markers") or []),
            "num_text_regions": len(visual_state.get("text_regions") or []),
            "num_point_candidates": summary.get("num_point_candidates", 0),
            "num_segment_candidates": summary.get("num_segment_candidates", 0),
            "num_angle_candidates": summary.get("num_angle_candidates", 0),
            "num_label_region_candidates": summary.get("num_label_region_candidates", 0),
        }
