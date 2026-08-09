"""Check VLM relation-to-builder capability decisions."""

from __future__ import annotations

from typing import Any, Dict, List, Set

from .builder_capabilities import GEOSYM_2D_BUILDER_CAPABILITIES, GeoSymBuilderCapabilities


class RelationBuilderDecisionAnalyzer:
    """Verify that stage-1 relations include explicit builder capability decisions."""

    def __init__(self, capabilities: GeoSymBuilderCapabilities = GEOSYM_2D_BUILDER_CAPABILITIES):
        self.capabilities = capabilities
        self._mapping_by_type = {
            str(item.get("relation_type")): item
            for item in self.capabilities.relation_mapping_contract()
            if item.get("relation_type")
        }

    def analyze(self, *, semantic: Dict[str, Any]) -> Dict[str, Any]:
        expected = self._expected_relations(semantic)
        decisions = self._decision_index(semantic)
        expected_ids = set(expected)
        decision_ids = set(decisions)
        missing_ids = sorted(expected_ids - decision_ids)
        unknown_ids = sorted(decision_ids - expected_ids)
        decision_reports = [
            self._decision_report(candidate_id, expected[candidate_id], decisions[candidate_id])
            for candidate_id in sorted(expected_ids & decision_ids)
        ]
        inconsistent = [item for item in decision_reports if item.get("issues")]
        return {
            "version": "relation-builder-decision-v0",
            "num_expected_relation_builder_decisions": len(expected_ids),
            "num_relation_builder_decisions": len(decision_ids),
            "num_missing_relation_builder_decisions": len(missing_ids),
            "num_unknown_relation_builder_decisions": len(unknown_ids),
            "num_inconsistent_relation_builder_decisions": len(inconsistent),
            "coverage": ((len(expected_ids) - len(missing_ids)) / len(expected_ids)) if expected_ids else 1.0,
            "missing_relation_builder_decision_ids": missing_ids[:120],
            "unknown_relation_builder_decision_ids": unknown_ids[:120],
            "inconsistent_relation_builder_decision_reports": inconsistent[:120],
        }

    def _expected_relations(self, semantic: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for idx, relation in enumerate(semantic.get("relations") or []):
            if not isinstance(relation, dict):
                continue
            cid = str(relation.get("candidate_id") or relation.get("from_candidate") or "")
            rtype = str(relation.get("type") or "")
            if not cid or not rtype or rtype == "target":
                if cid or not rtype or rtype == "target":
                    continue
                cid = self._vlm_relation_key(idx, relation)
            mapping = self._mapping_by_type.get(rtype) or {
                "relation_type": rtype,
                "status": "unsupported",
                "buildability_mode": "unsupported",
                "builder_actions": [],
                "core_constraint_action": None,
            }
            out[cid] = {
                "relation_index": idx,
                "candidate_id": cid,
                "relation_type": rtype,
                "expected_decision": self._expected_decision(mapping),
                "expected_actions": [str(item) for item in mapping.get("builder_actions") or []],
                "expected_core_action": mapping.get("core_constraint_action"),
                "expected_status": mapping.get("status"),
                "expected_buildability_mode": mapping.get("buildability_mode"),
            }
        return out

    def _decision_index(self, semantic: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        relation_keys = {
            idx: self._vlm_relation_key(idx, relation)
            for idx, relation in enumerate(semantic.get("relations") or [])
            if isinstance(relation, dict)
            and not (relation.get("candidate_id") or relation.get("from_candidate"))
            and relation.get("type")
            and relation.get("type") != "target"
        }
        signature_keys = {
            self._relation_signature(relation): self._vlm_relation_key(idx, relation)
            for idx, relation in enumerate(semantic.get("relations") or [])
            if isinstance(relation, dict)
            and not (relation.get("candidate_id") or relation.get("from_candidate"))
            and relation.get("type")
            and relation.get("type") != "target"
        }
        for item in semantic.get("relation_builder_decisions") or []:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("candidate_id") or item.get("from_candidate") or "")
            if not cid and item.get("relation_index") is not None:
                try:
                    cid = relation_keys.get(int(item.get("relation_index")), "")
                except (TypeError, ValueError):
                    cid = ""
            if not cid:
                cid = signature_keys.get(self._decision_signature(item), "")
            if cid:
                out[cid] = item
        return out

    def _decision_report(self, candidate_id: str, expected: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
        actual_decision = str(decision.get("decision") or "")
        actual_actions = [str(item) for item in decision.get("builder_actions") or []]
        actual_core = decision.get("core_constraint_action")
        issues: List[Dict[str, Any]] = []
        if actual_decision and actual_decision != expected["expected_decision"]:
            issues.append({
                "field": "decision",
                "actual": actual_decision,
                "expected": expected["expected_decision"],
            })
        if actual_actions and actual_actions != expected["expected_actions"]:
            issues.append({
                "field": "builder_actions",
                "actual": actual_actions,
                "expected": expected["expected_actions"],
            })
        if "core_constraint_action" in decision and actual_core != expected["expected_core_action"]:
            issues.append({
                "field": "core_constraint_action",
                "actual": actual_core,
                "expected": expected["expected_core_action"],
            })
        return {
            "candidate_id": candidate_id,
            "relation_type": expected["relation_type"],
            "expected": expected,
            "actual": {
                "decision": actual_decision,
                "builder_actions": actual_actions,
                "core_constraint_action": actual_core,
                "reason": decision.get("reason") or "",
            },
            "issues": issues,
        }

    def _expected_decision(self, mapping: Dict[str, Any]) -> str:
        status = str(mapping.get("status") or "")
        if status == "buildable":
            return "buildable"
        if status == "convertible":
            return "convertible"
        if status == "theorem_only":
            return "theorem_only"
        return "unsupported"

    def _vlm_relation_key(self, idx: int, relation: Dict[str, Any]) -> str:
        return f"vlm_relation[{idx}]::{self._relation_signature(relation)}"

    def _relation_signature(self, relation: Dict[str, Any]) -> str:
        rtype = str(relation.get("type") or "")
        obj = self._object_text(relation)
        value = str(relation.get("value") or "")
        return f"{rtype}|{obj}|{value}"

    def _decision_signature(self, decision: Dict[str, Any]) -> str:
        rtype = str(decision.get("relation_type") or "")
        obj = self._object_text(decision)
        value = str(decision.get("value") or "")
        return f"{rtype}|{obj}|{value}"

    def _object_text(self, item: Dict[str, Any]) -> str:
        if item.get("object") not in {None, ""}:
            return str(item.get("object"))
        for key in ["objects", "lines", "points"]:
            value = item.get(key)
            if isinstance(value, list):
                return "|".join(str(part) for part in value)
            if value not in {None, ""}:
                return str(value)
        return ""
