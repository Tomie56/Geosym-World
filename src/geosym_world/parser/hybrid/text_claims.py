"""Model-free text claim artifact for real-geometry parsing."""

from __future__ import annotations

from typing import Any, Dict, List


class TextClaimsBuilder:
    """Convert script observations into a compact auditable text-claim file."""

    def build(self, *, script_observations: Dict[str, Any]) -> Dict[str, Any]:
        text_hints = script_observations.get("text_hints") or {}
        semantic = script_observations.get("semantic_candidates") or {}
        named = script_observations.get("named_object_proposals") or {}
        relation_claims = [
            self._relation_claim(item)
            for item in (semantic.get("relation_candidates") or [])
            if isinstance(item, dict)
        ]
        target_claims = [
            self._target_claim(item)
            for item in (semantic.get("target_candidates") or [])
            if isinstance(item, dict)
        ]
        return {
            "version": "text-claims-v0",
            "principle": "text claims are symbolic evidence; they are not canonical GeoSym truth until verified",
            "text_sources": script_observations.get("text_sources") or [],
            "named_objects": list(named.get("items") or []),
            "relation_claims": relation_claims,
            "target_claims": target_claims,
            "raw_text_hints": {
                "labels_mentioned": text_hints.get("labels_mentioned") or [],
                "segments_mentioned": text_hints.get("segments_mentioned") or [],
                "numbers_mentioned": text_hints.get("numbers_mentioned") or [],
                "length_claims": text_hints.get("length_claims") or [],
                "angle_claims": text_hints.get("angle_claims") or [],
                "area_claims": text_hints.get("area_claims") or [],
                "perimeter_claims": text_hints.get("perimeter_claims") or [],
                "ratio_claims": text_hints.get("ratio_claims") or [],
                "triangle_claims": text_hints.get("triangle_claims") or [],
            },
            "metrics": {
                "num_text_sources": len(script_observations.get("text_sources") or []),
                "num_named_objects": len(named.get("items") or []),
                "num_relation_claims": len(relation_claims),
                "num_target_claims": len(target_claims),
            },
        }

    def _relation_claim(self, item: Dict[str, Any]) -> Dict[str, Any]:
        out = {
            "id": str(item.get("id") or ""),
            "type": str(item.get("type") or ""),
            "object": item.get("object"),
            "objects": item.get("objects") or [],
            "value": item.get("value"),
            "source": item.get("source") or "",
            "source_text": item.get("source_text") or "",
            "needs_visual_binding": bool(item.get("needs_vlm_confirmation", True)),
            "buildability": item.get("buildability") or {},
            "builder_mapping": item.get("builder_mapping") or {},
        }
        return out

    def _target_claim(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(item.get("id") or ""),
            "type": str(item.get("type") or ""),
            "object": item.get("object"),
            "objects": item.get("objects") or [],
            "value_symbol": item.get("value_symbol"),
            "source": item.get("source") or "",
            "needs_visual_binding": bool(item.get("needs_vlm_confirmation", True)),
        }
