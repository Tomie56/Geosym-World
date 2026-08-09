"""Repair missing relation/target drafts from confirmed script candidates."""

from __future__ import annotations

from typing import Any, Dict, List, Set
import json


class RelationDraftRepairer:
    """Fill relation-pass drafts only after candidate confirmation.

    This repair is intentionally narrower than semantic extraction. It copies
    relation_pass_response_template.relations/target only for candidates the
    model already confirmed, corrected, or referenced. Raw stage-contract
    metrics still expose whether the model omitted these drafts.
    """

    def repair(self, *, semantic: Dict[str, Any], evidence_graph: Dict[str, Any]) -> Dict[str, Any]:
        template = evidence_graph.get("relation_pass_response_template") or {}
        original_relations = [
            dict(item) for item in semantic.get("relations") or []
            if isinstance(item, dict)
        ]
        original_target = dict(semantic.get("target") or {}) if isinstance(semantic.get("target"), dict) else {}
        original_targets = [
            dict(item) for item in semantic.get("targets") or []
            if isinstance(item, dict)
        ]
        repairable_ids = self._repairable_candidate_ids(semantic)
        decisions = self._decision_index(semantic)
        template_relations_by_candidate = self._template_relations_by_candidate(template)
        existing_relation_refs = self._refs_from_items(original_relations)
        relation_additions: List[Dict[str, Any]] = []

        for relation in template.get("relations") or []:
            if not isinstance(relation, dict):
                continue
            refs = self._refs_from_item(relation)
            if not refs or not refs <= repairable_ids:
                continue
            if refs & existing_relation_refs:
                continue
            cid = sorted(refs)[0]
            repaired = self._relation_from_decision(decisions.get(cid), fallback=relation, candidate_id=cid)
            relation_additions.append(repaired)
            existing_relation_refs.update(refs)

        repaired_relations = self._dedupe(original_relations + relation_additions)
        metadata_patches = self._patch_relation_metadata(
            repaired_relations,
            repairable_ids=repairable_ids,
            template_relations_by_candidate=template_relations_by_candidate,
        )

        target_addition: Dict[str, Any] = {}
        target_additions: List[Dict[str, Any]] = []
        template_targets = self._template_targets(template)
        existing_all_target_refs = self._refs_from_item(original_target) | self._refs_from_items(original_targets)
        for template_target in template_targets:
            target_refs = self._refs_from_item(template_target)
            if not template_target or not target_refs or not target_refs <= repairable_ids:
                continue
            if target_refs <= existing_all_target_refs:
                continue
            cid = sorted(target_refs)[0]
            repaired_target = self._target_from_decision(decisions.get(cid), fallback=template_target, candidate_id=cid)
            target_additions.append(repaired_target)
            existing_all_target_refs.update(target_refs)
        target_addition = target_additions[0] if target_additions and not original_target else {}
        repaired_targets = self._dedupe(([original_target] if original_target else []) + original_targets + target_additions)

        if not relation_additions and not target_additions and not metadata_patches:
            return {
                "used": False,
                "num_original_relations": len(original_relations),
                "num_added_relations": 0,
                "num_metadata_patched_relations": 0,
                "num_added_buildability": 0,
                "num_added_builder_mapping": 0,
                "target_added": False,
                "num_added_targets": 0,
                "repairable_candidate_ids": sorted(repairable_ids),
                "additions": [],
                "metadata_patches": [],
                "semantic": semantic,
            }

        repaired_semantic = dict(semantic)
        repaired_semantic["relations"] = repaired_relations
        if target_addition:
            repaired_semantic["target"] = target_addition
        if repaired_targets:
            repaired_semantic["targets"] = repaired_targets
        metadata = dict(repaired_semantic.get("metadata") or {})
        metadata["relation_draft_repair"] = {
            "used": True,
            "num_added_relations": len(relation_additions),
            "num_metadata_patched_relations": len(metadata_patches),
            "num_added_buildability": self._count_metadata_patch_field(metadata_patches, "buildability"),
            "num_added_builder_mapping": self._count_metadata_patch_field(metadata_patches, "builder_mapping"),
            "target_added": bool(target_addition),
            "num_added_targets": len(target_additions),
            "source": "relation_pass_response_template_for_confirmed_candidates",
        }
        repaired_semantic["metadata"] = metadata
        return {
            "used": True,
            "num_original_relations": len(original_relations),
            "num_added_relations": len(relation_additions),
            "num_metadata_patched_relations": len(metadata_patches),
            "num_added_buildability": self._count_metadata_patch_field(metadata_patches, "buildability"),
            "num_added_builder_mapping": self._count_metadata_patch_field(metadata_patches, "builder_mapping"),
            "target_added": bool(target_addition),
            "num_added_targets": len(target_additions),
            "repairable_candidate_ids": sorted(repairable_ids),
            "additions": relation_additions,
            "metadata_patches": metadata_patches,
            "target_addition": target_addition,
            "target_additions": target_additions,
            "semantic": repaired_semantic,
        }

    def _repairable_candidate_ids(self, semantic: Dict[str, Any]) -> Set[str]:
        out: Set[str] = set()
        for decision in semantic.get("candidate_decisions") or []:
            if not isinstance(decision, dict):
                continue
            if str(decision.get("decision") or "") in {"confirm", "correct"} and decision.get("candidate_id"):
                out.add(str(decision.get("candidate_id")))
        out.update(self._refs_from_items(semantic.get("relations") or []))
        out.update(self._refs_from_items(semantic.get("build_instructions") or []))
        out.update(self._refs_from_item(semantic.get("target") or {}))
        out.update(self._refs_from_items(semantic.get("targets") or []))
        return out

    def _decision_index(self, semantic: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for decision in semantic.get("candidate_decisions") or []:
            if not isinstance(decision, dict):
                continue
            cid = str(decision.get("candidate_id") or "")
            if cid:
                out[cid] = decision
        return out

    def _relation_from_decision(self, decision: Dict[str, Any] | None, *, fallback: Dict[str, Any], candidate_id: str) -> Dict[str, Any]:
        if decision and str(decision.get("decision") or "") == "correct":
            corrected = decision.get("corrected_relation")
            if isinstance(corrected, dict) and corrected.get("type"):
                relation = dict(corrected)
                relation.setdefault("candidate_id", candidate_id)
                relation.setdefault("grounded", False)
                relation["repair_source"] = "candidate_decision.corrected_relation"
                return relation
        relation = dict(fallback)
        relation["repair_source"] = "relation_pass_response_template"
        return relation

    def _target_from_decision(self, decision: Dict[str, Any] | None, *, fallback: Dict[str, Any], candidate_id: str) -> Dict[str, Any]:
        if decision and str(decision.get("decision") or "") == "correct":
            corrected = decision.get("corrected_relation")
            if isinstance(corrected, dict) and (corrected.get("object") or corrected.get("target")):
                target = dict(corrected)
                target.setdefault("candidate_id", candidate_id)
                target.setdefault("from_candidate", candidate_id)
                target.setdefault("value_symbol", "x")
                target["repair_source"] = "candidate_decision.corrected_relation"
                return target
        target = dict(fallback)
        target["repair_source"] = "relation_pass_response_template"
        return target

    def _template_relations_by_candidate(self, template: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for relation in template.get("relations") or []:
            if not isinstance(relation, dict):
                continue
            for ref in self._refs_from_item(relation):
                out.setdefault(ref, relation)
        return out

    def _template_targets(self, template: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for item in template.get("targets") or []:
            if isinstance(item, dict):
                out.append(dict(item))
        target = template.get("target")
        if isinstance(target, dict) and target:
            target_refs = self._refs_from_item(target)
            if not target_refs or not any(target_refs <= self._refs_from_item(item) for item in out):
                out.insert(0, dict(target))
        return self._dedupe(out)

    def _patch_relation_metadata(
        self,
        relations: List[Dict[str, Any]],
        *,
        repairable_ids: Set[str],
        template_relations_by_candidate: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        patches: List[Dict[str, Any]] = []
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            refs = self._refs_from_item(relation)
            candidate_ids = sorted(refs & repairable_ids)
            if not candidate_ids:
                continue
            cid = candidate_ids[0]
            template_relation = template_relations_by_candidate.get(cid) or {}
            if not template_relation:
                continue
            if relation.get("type") and template_relation.get("type") and relation.get("type") != template_relation.get("type"):
                continue
            patched_fields: List[str] = []
            for field in ["buildability", "builder_mapping"]:
                if relation.get(field) or not isinstance(template_relation.get(field), dict):
                    continue
                relation[field] = dict(template_relation[field])
                patched_fields.append(field)
            if patched_fields:
                relation["metadata_repair_source"] = "relation_pass_response_template"
                patches.append({"candidate_id": cid, "fields": patched_fields})
        return patches

    def _count_metadata_patch_field(self, patches: List[Dict[str, Any]], field: str) -> int:
        count = 0
        for patch in patches:
            fields = patch.get("fields") or []
            if field in fields:
                count += 1
        return count

    def _refs_from_items(self, items: List[Dict[str, Any]]) -> Set[str]:
        refs: Set[str] = set()
        for item in items:
            refs.update(self._refs_from_item(item))
        return refs

    def _refs_from_item(self, item: Dict[str, Any]) -> Set[str]:
        if not isinstance(item, dict):
            return set()
        refs: Set[str] = set()
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
        target = item.get("target")
        if isinstance(target, dict):
            refs.update(self._refs_from_item(target))
        return refs

    def _dedupe(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        seen = set()
        for item in items:
            key = json.dumps(item, sort_keys=True, ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out
