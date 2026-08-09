"""Apply explicit grounding decisions to existing semantic relations."""

from __future__ import annotations

from typing import Any, Dict, List


class GroundingDecisionRepairer:
    """Patch relation visual grounding from stage-1 grounding_decisions.

    This repair does not create relations or mathematical constraints. It only
    copies VLM-reviewed visual evidence onto relations that already reference
    the same script candidate, so the downstream grounder can use the explicit
    grounding checklist instead of relying on duplicated visual_ids.
    """

    def repair(self, *, semantic: Dict[str, Any]) -> Dict[str, Any]:
        decisions = self._decisions_by_candidate(semantic.get("grounding_decisions") or [])
        if not decisions:
            return self._result(False, semantic, [], 0)

        patches: List[Dict[str, Any]] = []
        patched_relations = []
        for idx, relation in enumerate(semantic.get("relations") or []):
            if not isinstance(relation, dict):
                continue
            patched = dict(relation)
            cid = str(patched.get("candidate_id") or patched.get("from_candidate") or "")
            candidate_decisions = decisions.get(cid) or []
            bind_decisions = [
                item for item in candidate_decisions
                if item.get("decision") == "bind" and item.get("visual_ids")
            ]
            if not cid or not bind_decisions:
                patched_relations.append(patched)
                continue
            existing_visual_ids = [str(item) for item in patched.get("visual_ids") or [] if item]
            merged_visual_ids = self._dedupe(existing_visual_ids + [
                str(vid)
                for decision in bind_decisions
                for vid in decision.get("visual_ids") or []
                if vid
            ])
            fields: List[str] = []
            if merged_visual_ids != existing_visual_ids:
                patched["visual_ids"] = merged_visual_ids
                fields.append("visual_ids")
            if patched.get("grounded") is not True and merged_visual_ids:
                patched["grounded"] = True
                fields.append("grounded")
            if fields:
                patched["grounding_repair_source"] = "grounding_decisions"
                patches.append({
                    "index": idx,
                    "candidate_id": cid,
                    "fields": fields,
                    "visual_ids": merged_visual_ids,
                    "decision_task_ids": [
                        str(item.get("task_id"))
                        for item in bind_decisions
                        if item.get("task_id")
                    ],
                })
            patched_relations.append(patched)

        if not patches:
            return self._result(False, semantic, [], len(decisions))

        repaired = dict(semantic)
        repaired["relations"] = patched_relations
        metadata = dict(repaired.get("metadata") or {})
        metadata["grounding_decision_repair"] = {
            "used": True,
            "num_relation_patches": len(patches),
            "source": "semantic.grounding_decisions",
        }
        repaired["metadata"] = metadata
        return self._result(True, repaired, patches, len(decisions))

    def _decisions_by_candidate(self, decisions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {}
        for item in decisions:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("candidate_id") or "")
            if not cid:
                continue
            decision = dict(item)
            decision["decision"] = str(decision.get("decision") or "").lower()
            decision["visual_ids"] = [str(vid) for vid in decision.get("visual_ids") or [] if vid]
            out.setdefault(cid, []).append(decision)
        return out

    def _result(self, used: bool, semantic: Dict[str, Any], patches: List[Dict[str, Any]], num_decision_candidates: int) -> Dict[str, Any]:
        return {
            "used": used,
            "num_decision_candidates": num_decision_candidates,
            "num_relation_patches": len(patches),
            "num_visual_id_patches": self._count_field(patches, "visual_ids"),
            "num_grounded_patches": self._count_field(patches, "grounded"),
            "patches": patches[:100],
            "semantic": semantic,
        }

    def _count_field(self, patches: List[Dict[str, Any]], field: str) -> int:
        return sum(1 for patch in patches if field in (patch.get("fields") or []))

    def _dedupe(self, items: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out
