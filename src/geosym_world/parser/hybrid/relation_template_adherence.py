"""Compare stage-1 output with the relation-pass template."""

from __future__ import annotations

from typing import Any, Dict, List


class RelationTemplateAdherenceAnalyzer:
    """Measure whether stage-1 output followed the relation-pass template."""

    def analyze(self, *, evidence_graph: Dict[str, Any], semantic: Dict[str, Any]) -> Dict[str, Any]:
        template = evidence_graph.get("relation_pass_response_template") or {}
        expected = [
            item for item in template.get("candidate_decisions") or []
            if isinstance(item, dict) and item.get("candidate_id")
        ]
        actual = [
            item for item in semantic.get("candidate_decisions") or []
            if isinstance(item, dict) and item.get("candidate_id")
        ]
        actual_by_id: Dict[str, List[Dict[str, Any]]] = {}
        for item in actual:
            actual_by_id.setdefault(str(item.get("candidate_id")), []).append(item)

        expected_reports = []
        matched_ids: set[str] = set()
        for idx, expected_decision in enumerate(expected):
            cid = str(expected_decision.get("candidate_id"))
            matches = actual_by_id.get(cid) or []
            if matches:
                matched_ids.add(cid)
            expected_reports.append({
                "template_index": idx,
                "candidate_id": cid,
                "expected_decision": expected_decision,
                "matched": bool(matches),
                "actual_decisions": matches[:5],
            })

        missing = [item for item in expected_reports if not item.get("matched")]
        expected_ids = {str(item.get("candidate_id")) for item in expected}
        extra = [
            {
                "candidate_id": str(item.get("candidate_id")),
                "decision": item,
            }
            for item in actual
            if str(item.get("candidate_id")) not in expected_ids
        ]
        expected_relations = [
            item for item in template.get("relations") or []
            if isinstance(item, dict) and item.get("candidate_id")
        ]
        actual_relations = [
            item for item in semantic.get("relations") or []
            if isinstance(item, dict)
        ]
        actual_relation_refs = self._candidate_refs(actual_relations)
        missing_relations = [
            {
                "candidate_id": str(item.get("candidate_id")),
                "expected_relation": item,
            }
            for item in expected_relations
            if str(item.get("candidate_id")) not in actual_relation_refs
        ]
        expected_target_refs = self._target_refs(template)
        actual_target_refs = self._target_refs(semantic)
        missing_target_refs = sorted(expected_target_refs - actual_target_refs)
        expected_grounding = [
            item for item in template.get("grounding_decisions") or []
            if isinstance(item, dict) and item.get("task_id")
        ]
        actual_grounding = [
            item for item in semantic.get("grounding_decisions") or []
            if isinstance(item, dict) and item.get("task_id")
        ]
        actual_grounding_by_id: Dict[str, List[Dict[str, Any]]] = {}
        for item in actual_grounding:
            actual_grounding_by_id.setdefault(str(item.get("task_id")), []).append(item)
        grounding_reports = []
        for idx, expected_decision in enumerate(expected_grounding):
            task_id = str(expected_decision.get("task_id"))
            matches = actual_grounding_by_id.get(task_id) or []
            grounding_reports.append({
                "template_index": idx,
                "task_id": task_id,
                "expected_decision": expected_decision,
                "matched": bool(matches),
                "actual_decisions": matches[:5],
            })
        missing_grounding = [item for item in grounding_reports if not item.get("matched")]
        expected_grounding_ids = {str(item.get("task_id")) for item in expected_grounding}
        extra_grounding = [
            {
                "task_id": str(item.get("task_id")),
                "decision": item,
            }
            for item in actual_grounding
            if str(item.get("task_id")) not in expected_grounding_ids
        ]
        expected_builder_decisions = [
            item for item in template.get("relation_builder_decisions") or []
            if isinstance(item, dict) and item.get("candidate_id")
        ]
        actual_builder_decisions = [
            item for item in semantic.get("relation_builder_decisions") or []
            if isinstance(item, dict) and item.get("candidate_id")
        ]
        actual_builder_decision_by_id: Dict[str, List[Dict[str, Any]]] = {}
        for item in actual_builder_decisions:
            actual_builder_decision_by_id.setdefault(str(item.get("candidate_id")), []).append(item)
        builder_decision_reports = []
        for idx, expected_decision in enumerate(expected_builder_decisions):
            cid = str(expected_decision.get("candidate_id"))
            matches = actual_builder_decision_by_id.get(cid) or []
            builder_decision_reports.append({
                "template_index": idx,
                "candidate_id": cid,
                "expected_decision": expected_decision,
                "matched": bool(matches),
                "actual_decisions": matches[:5],
            })
        missing_builder_decisions = [item for item in builder_decision_reports if not item.get("matched")]
        expected_builder_decision_ids = {str(item.get("candidate_id")) for item in expected_builder_decisions}
        extra_builder_decisions = [
            {
                "candidate_id": str(item.get("candidate_id")),
                "decision": item,
            }
            for item in actual_builder_decisions
            if str(item.get("candidate_id")) not in expected_builder_decision_ids
        ]
        num_expected = len(expected)
        num_matched = len(expected_reports) - len(missing)
        num_expected_relations = len(expected_relations)
        num_matched_relations = num_expected_relations - len(missing_relations)
        num_expected_grounding = len(expected_grounding)
        num_matched_grounding = num_expected_grounding - len(missing_grounding)
        num_expected_builder_decisions = len(expected_builder_decisions)
        num_matched_builder_decisions = num_expected_builder_decisions - len(missing_builder_decisions)
        target_matched = not missing_target_refs
        return {
            "num_template_candidate_decisions": num_expected,
            "num_actual_candidate_decisions": len(actual),
            "num_matched_template_candidate_decisions": num_matched,
            "num_missing_template_candidate_decisions": len(missing),
            "num_extra_candidate_decisions": len(extra),
            "num_template_relations": num_expected_relations,
            "num_actual_relations": len(actual_relations),
            "num_matched_template_relations": num_matched_relations,
            "num_missing_template_relations": len(missing_relations),
            "template_relation_adherence": (num_matched_relations / num_expected_relations) if num_expected_relations else None,
            "template_target_expected": bool(expected_target_refs),
            "template_target_matched": target_matched if expected_target_refs else None,
            "num_missing_template_target_refs": len(missing_target_refs),
            "num_template_grounding_decisions": num_expected_grounding,
            "num_actual_grounding_decisions": len(actual_grounding),
            "num_matched_template_grounding_decisions": num_matched_grounding,
            "num_missing_template_grounding_decisions": len(missing_grounding),
            "num_extra_grounding_decisions": len(extra_grounding),
            "template_grounding_adherence": (num_matched_grounding / num_expected_grounding) if num_expected_grounding else None,
            "num_template_builder_decisions": num_expected_builder_decisions,
            "num_actual_builder_decisions": len(actual_builder_decisions),
            "num_matched_template_builder_decisions": num_matched_builder_decisions,
            "num_missing_template_builder_decisions": len(missing_builder_decisions),
            "num_extra_builder_decisions": len(extra_builder_decisions),
            "template_builder_decision_adherence": (num_matched_builder_decisions / num_expected_builder_decisions) if num_expected_builder_decisions else None,
            "template_adherence": (num_matched / num_expected) if num_expected else None,
            "missing_template_candidate_decisions": missing[:100],
            "extra_candidate_decisions": extra[:100],
            "missing_template_relations": missing_relations[:100],
            "missing_template_target_refs": missing_target_refs[:100],
            "missing_template_grounding_decisions": missing_grounding[:100],
            "extra_grounding_decisions": extra_grounding[:100],
            "missing_template_builder_decisions": missing_builder_decisions[:100],
            "extra_builder_decisions": extra_builder_decisions[:100],
        }

    def _candidate_refs(self, items: List[Dict[str, Any]]) -> set[str]:
        refs: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            for key in ["candidate_id", "from_candidate"]:
                value = item.get(key)
                if value:
                    refs.add(str(value))
            for key in ["candidate_ids", "from_candidates"]:
                value = item.get(key)
                if isinstance(value, list):
                    refs.update(str(v) for v in value if v)
                elif value:
                    refs.add(str(value))
        return refs

    def _target_refs(self, semantic: Dict[str, Any]) -> set[str]:
        items: List[Dict[str, Any]] = []
        target = semantic.get("target")
        if isinstance(target, dict):
            items.append(target)
        targets = semantic.get("targets")
        if isinstance(targets, list):
            items.extend(item for item in targets if isinstance(item, dict))
        return self._candidate_refs(items)
