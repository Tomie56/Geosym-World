"""Audit whether VLM output used script-proposed semantic candidates."""

from __future__ import annotations

from typing import Any, Dict, List, Set


class CandidateUsageAnalyzer:
    """Compare script candidates with VLM relations and build instructions."""

    def analyze(self, script_observations: Dict[str, Any], semantic: Dict[str, Any]) -> Dict[str, Any]:
        candidates = self._candidate_ids(script_observations)
        relation_refs = self._refs_from_items(semantic.get("relations") or [])
        instruction_refs = self._refs_from_items(semantic.get("build_instructions") or [])
        target_refs = self._target_refs(semantic)
        decision_refs = self._candidate_decisions(semantic)
        used_refs = relation_refs | instruction_refs | target_refs
        decision_ids = set(decision_refs)
        vlm_derived_refs = sorted(ref for ref in (used_refs | decision_ids) if self._is_vlm_derived_ref(ref))
        unknown_refs = sorted(ref for ref in (used_refs | decision_ids) if ref not in candidates["all"] and not self._is_vlm_derived_ref(ref))
        confirmed_relation_ids = self._semantic_match_candidates(script_observations, semantic)
        confirmed_instruction_ids = sorted(instruction_refs & candidates["all"])
        explicitly_confirmed = {cid for cid, item in decision_refs.items() if item.get("decision") == "confirm"}
        corrected_ids = {cid for cid, item in decision_refs.items() if item.get("decision") == "correct"}
        rejected_ids = {cid for cid, item in decision_refs.items() if item.get("decision") == "reject"}
        uncertain_ids = {cid for cid, item in decision_refs.items() if item.get("decision") == "uncertain"}
        confirmed_ids = sorted((set(confirmed_relation_ids) | set(confirmed_instruction_ids) | (target_refs & candidates["all"]) | explicitly_confirmed | corrected_ids) & candidates["all"])
        reviewed_ids = set(confirmed_ids) | rejected_ids | uncertain_ids
        unreviewed_ids = sorted(candidates["all"] - reviewed_ids)
        unconfirmed_ids = sorted(candidates["all"] - set(confirmed_ids))
        conflict_ids = sorted((set(confirmed_ids) & rejected_ids) & candidates["all"])
        conflict_warnings = [
            {
                "severity": "warning",
                "code": "candidate.decision_conflict",
                "message": f"Candidate {cid} was rejected but also appears confirmed by relation/instruction output",
                "location": "candidate_decisions/relations/build_instructions",
            }
            for cid in conflict_ids
        ]
        warnings = [
            {
                "severity": "warning",
                "code": "candidate.unknown_reference",
                "message": f"VLM referenced unknown script candidate id: {ref}",
                "location": "relations/build_instructions/target",
            }
            for ref in unknown_refs
        ] + conflict_warnings
        return {
            "num_relation_candidates": len(candidates["relation"]),
            "num_target_candidates": len(candidates["target"]),
            "num_script_candidates": len(candidates["all"]),
            "num_confirmed_candidates": len(confirmed_ids),
            "num_unconfirmed_candidates": len(unconfirmed_ids),
            "num_rejected_candidates": len(rejected_ids & candidates["all"]),
            "num_corrected_candidates": len(corrected_ids & candidates["all"]),
            "num_uncertain_candidates": len(uncertain_ids & candidates["all"]),
            "num_reviewed_candidates": len(reviewed_ids & candidates["all"]),
            "num_unreviewed_candidates": len(unreviewed_ids),
            "num_decision_conflicts": len(conflict_ids),
            "num_unknown_candidate_refs": len(unknown_refs),
            "num_vlm_derived_candidate_refs": len(vlm_derived_refs),
            "confirmed_candidate_ids": confirmed_ids[:80],
            "unconfirmed_candidate_ids": unconfirmed_ids[:80],
            "rejected_candidate_ids": sorted(rejected_ids & candidates["all"])[:80],
            "corrected_candidate_ids": sorted(corrected_ids & candidates["all"])[:80],
            "uncertain_candidate_ids": sorted(uncertain_ids & candidates["all"])[:80],
            "unreviewed_candidate_ids": unreviewed_ids[:80],
            "decision_conflict_candidate_ids": conflict_ids[:80],
            "unknown_candidate_refs": unknown_refs[:80],
            "vlm_derived_candidate_refs": vlm_derived_refs[:80],
            "relation_candidate_refs": sorted(relation_refs),
            "instruction_candidate_refs": sorted(instruction_refs),
            "target_candidate_refs": sorted(target_refs),
            "candidate_decisions": list(decision_refs.values())[:120],
            "warnings": warnings,
        }

    def _is_vlm_derived_ref(self, ref: str) -> bool:
        return str(ref or "").startswith("vlm_rel_")

    def _candidate_ids(self, script_observations: Dict[str, Any]) -> Dict[str, Set[str]]:
        semantic_candidates = script_observations.get("semantic_candidates") or {}
        relation = {
            str(item.get("id"))
            for item in semantic_candidates.get("relation_candidates") or []
            if item.get("id")
        }
        target = {
            str(item.get("id"))
            for item in semantic_candidates.get("target_candidates") or []
            if item.get("id")
        }
        suggestion = {
            str(item.get("from_candidate"))
            for item in (script_observations.get("engine_build_suggestions") or {}).get("suggested_build_instructions") or []
            if item.get("from_candidate")
        }
        return {
            "relation": relation,
            "target": target,
            "suggestion": suggestion,
            "all": relation | target | suggestion,
        }

    def _refs_from_items(self, items: List[Dict[str, Any]]) -> Set[str]:
        out: Set[str] = set()
        for item in items:
            out.update(self._refs_from_item(item))
        return out

    def _refs_from_item(self, item: Dict[str, Any]) -> Set[str]:
        if not isinstance(item, dict):
            return set()
        out: Set[str] = set()
        for key in ["candidate_id", "from_candidate"]:
            value = item.get(key)
            if value:
                out.add(str(value))
        for key in ["candidate_ids", "from_candidates"]:
            value = item.get(key)
            if isinstance(value, list):
                out.update(str(v) for v in value if v)
            elif value:
                out.add(str(value))
        target = item.get("target")
        if isinstance(target, dict):
            out.update(self._refs_from_item(target))
        return out

    def _semantic_match_candidates(self, script_observations: Dict[str, Any], semantic: Dict[str, Any]) -> List[str]:
        relations = [rel for rel in semantic.get("relations") or [] if isinstance(rel, dict)]
        targets = []
        if isinstance(semantic.get("target"), dict):
            targets.append(semantic.get("target"))
        targets.extend(item for item in semantic.get("targets") or [] if isinstance(item, dict))
        confirmed: Set[str] = set()
        semantic_candidates = script_observations.get("semantic_candidates") or {}
        for candidate in semantic_candidates.get("relation_candidates") or []:
            cid = str(candidate.get("id") or "")
            if cid and any(self._candidate_matches_relation(candidate, rel) for rel in relations):
                confirmed.add(cid)
        for candidate in semantic_candidates.get("target_candidates") or []:
            cid = str(candidate.get("id") or "")
            if cid and any(self._candidate_matches_target(candidate, target) for target in targets):
                confirmed.add(cid)
        return sorted(confirmed)

    def _candidate_matches_relation(self, candidate: Dict[str, Any], relation: Dict[str, Any]) -> bool:
        if str(candidate.get("type")) != str(relation.get("type")):
            return False
        cand_obj = self._object_key(candidate)
        rel_obj = self._object_key(relation)
        if cand_obj and rel_obj and cand_obj != rel_obj:
            return False
        cand_value = str(candidate.get("value") or "").strip().rstrip(".")
        rel_value = str(relation.get("value") or "").strip().rstrip(".")
        if cand_value and rel_value and cand_value != rel_value:
            return False
        return bool(cand_obj or cand_value)

    def _candidate_matches_target(self, candidate: Dict[str, Any], target: Dict[str, Any]) -> bool:
        cand_obj = self._object_key(candidate)
        target_obj = self._object_key(target)
        if cand_obj and target_obj and cand_obj == target_obj:
            return True
        return bool(candidate.get("type") and str(candidate.get("type")) == str(target.get("type")) and not cand_obj)

    def _object_key(self, item: Dict[str, Any]) -> str:
        if item.get("object"):
            return "".join(ch for ch in str(item.get("object")).upper() if ch.isalnum())
        objects = item.get("objects")
        if isinstance(objects, list):
            return "".join(str(obj).upper() for obj in objects if obj)
        return ""

    def _candidate_decisions(self, semantic: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for item in semantic.get("candidate_decisions") or []:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("candidate_id") or "")
            if not cid:
                continue
            decision = str(item.get("decision") or "uncertain").lower()
            out[cid] = {
                "candidate_id": cid,
                "decision": decision,
                "reason": item.get("reason") or "",
                "confidence": item.get("confidence"),
                "corrected_relation": item.get("corrected_relation"),
            }
        return out

    def _target_refs(self, semantic: Dict[str, Any]) -> Set[str]:
        refs = self._refs_from_item(semantic.get("target") or {})
        refs.update(self._refs_from_items(semantic.get("targets") or []))
        return refs
