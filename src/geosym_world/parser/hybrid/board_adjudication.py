"""Validate that VLM passes used the geometry evidence board."""

from __future__ import annotations

from typing import Any, Dict, List, Set


class BoardAdjudicationAnalyzer:
    """Check coverage of named-object, relation, and target board cards."""

    def analyze(self, *, semantic: Dict[str, Any], evidence_board: Dict[str, Any]) -> Dict[str, Any]:
        named_ids = self._card_ids(evidence_board.get("named_object_cards") or [])
        relation_ids = self._card_ids(evidence_board.get("relation_cards") or [])
        target_ids = self._card_ids(evidence_board.get("target_cards") or [])
        reviewed_named = self._named_binding_ids(semantic)
        reviewed_relations = self._candidate_decision_ids(semantic) | self._relation_candidate_ids(semantic)
        reviewed_targets = self._target_candidate_ids(semantic) | self._candidate_decision_ids(semantic)
        missing_named = sorted(named_ids - reviewed_named)
        missing_relations = sorted(relation_ids - reviewed_relations)
        missing_targets = sorted(target_ids - reviewed_targets)
        return {
            "version": "board-adjudication-v0",
            "valid": not (missing_named or missing_relations or missing_targets),
            "metrics": {
                "num_named_object_cards": len(named_ids),
                "num_reviewed_named_object_cards": len(named_ids & reviewed_named),
                "num_missing_named_object_cards": len(missing_named),
                "named_object_card_coverage": self._ratio(len(named_ids & reviewed_named), len(named_ids)),
                "num_relation_cards": len(relation_ids),
                "num_reviewed_relation_cards": len(relation_ids & reviewed_relations),
                "num_missing_relation_cards": len(missing_relations),
                "relation_card_coverage": self._ratio(len(relation_ids & reviewed_relations), len(relation_ids)),
                "num_target_cards": len(target_ids),
                "num_reviewed_target_cards": len(target_ids & reviewed_targets),
                "num_missing_target_cards": len(missing_targets),
                "target_card_coverage": self._ratio(len(target_ids & reviewed_targets), len(target_ids)),
            },
            "missing_named_object_card_ids": missing_named,
            "missing_relation_card_ids": missing_relations,
            "missing_target_card_ids": missing_targets,
        }

    def _card_ids(self, items: List[Dict[str, Any]]) -> Set[str]:
        return {str(item.get("id")) for item in items if isinstance(item, dict) and item.get("id")}

    def _named_binding_ids(self, semantic: Dict[str, Any]) -> Set[str]:
        out: Set[str] = set()
        for item in semantic.get("named_object_bindings") or []:
            if isinstance(item, dict) and item.get("proposal_id"):
                out.add(str(item.get("proposal_id")))
        return out

    def _candidate_decision_ids(self, semantic: Dict[str, Any]) -> Set[str]:
        out: Set[str] = set()
        for item in semantic.get("candidate_decisions") or []:
            if isinstance(item, dict) and item.get("candidate_id"):
                out.add(str(item.get("candidate_id")))
        return out

    def _relation_candidate_ids(self, semantic: Dict[str, Any]) -> Set[str]:
        out: Set[str] = set()
        for item in semantic.get("relations") or []:
            if not isinstance(item, dict):
                continue
            for key in ["candidate_id", "from_candidate"]:
                if item.get(key):
                    out.add(str(item.get(key)))
        return out

    def _target_candidate_ids(self, semantic: Dict[str, Any]) -> Set[str]:
        out: Set[str] = set()
        for item in [semantic.get("target") or {}, *(semantic.get("targets") or [])]:
            if not isinstance(item, dict):
                continue
            for key in ["candidate_id", "from_candidate"]:
                if item.get(key):
                    out.add(str(item.get(key)))
        return out

    def _ratio(self, numerator: int, denominator: int) -> float:
        if denominator <= 0:
            return 1.0
        return round(numerator / denominator, 6)
