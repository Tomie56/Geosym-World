"""Adapt stage-1 VLM relations back to script candidates and builder contract."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .builder_capabilities import GEOSYM_2D_BUILDER_CAPABILITIES, GeoSymBuilderCapabilities
from .relation_engine_metadata import RelationEngineMetadataAnnotator


class RelationPassAdapter:
    """Attach script candidate ids and builder metadata to VLM relation drafts.

    This adapter does not invent new geometry and does not add candidate
    decisions. It only normalizes relation/target drafts that are already
    present in the VLM output by matching them against evidence graph candidates.
    Raw pass diagnostics should still be computed from the pre-adapted output.
    """

    def __init__(self, capabilities: GeoSymBuilderCapabilities = GEOSYM_2D_BUILDER_CAPABILITIES) -> None:
        self.capabilities = capabilities
        self.engine_metadata = RelationEngineMetadataAnnotator(capabilities=capabilities)
        self._mapping_by_type = {
            str(item.get("relation_type")): dict(item)
            for item in self.capabilities.relation_mapping_contract()
            if item.get("relation_type")
        }

    def adapt(self, *, relation_pass: Dict[str, Any], evidence_graph: Dict[str, Any]) -> Dict[str, Any]:
        adapted = dict(relation_pass or {})
        relation_candidates = [
            item for item in evidence_graph.get("relation_candidates") or []
            if isinstance(item, dict)
        ]
        target_candidates = [
            item for item in evidence_graph.get("target_candidates") or []
            if isinstance(item, dict)
        ]
        relation_index = self._relation_index(relation_candidates)
        candidate_by_id = {
            str(item.get("id")): item
            for item in relation_candidates
            if item.get("id")
        }
        target_by_id = {
            str(item.get("id")): item
            for item in target_candidates
            if item.get("id")
        }

        relation_patches: List[Dict[str, Any]] = []
        adapted_relations = []
        for idx, relation in enumerate(adapted.get("relations") or []):
            if not isinstance(relation, dict):
                continue
            patched = dict(relation)
            patch_fields: List[str] = []
            cid = str(patched.get("candidate_id") or patched.get("from_candidate") or "")
            candidate = candidate_by_id.get(cid) if cid else None
            if not candidate:
                key = self._relation_key(patched)
                candidates = relation_index.get(key) or []
                if len(candidates) == 1:
                    candidate = candidates[0]
                    cid = str(candidate.get("id") or "")
                    patched["candidate_id"] = cid
                    patch_fields.append("candidate_id")
            if candidate:
                for field in ["buildability", "builder_mapping"]:
                    expected = candidate.get(field)
                    if not isinstance(expected, dict):
                        continue
                    actual = patched.get(field)
                    if not actual:
                        patched[field] = dict(expected)
                        patch_fields.append(field)
                    elif isinstance(actual, dict) and not self._same_dict(actual, expected):
                        patched[field] = dict(expected)
                        patch_fields.append(field)
                if not patched.get("object") and candidate.get("object"):
                    patched["object"] = str(candidate.get("object"))
                    patch_fields.append("object")
                if not patched.get("objects") and candidate.get("objects") is not None:
                    patched["objects"] = [str(item) for item in (candidate.get("objects") or [])]
                    patch_fields.append("objects")
                if patched.get("value") in {None, ""} and candidate.get("value") not in {None, ""}:
                    patched["value"] = str(candidate.get("value"))
                    patch_fields.append("value")
            if patch_fields:
                patched["adapter_source"] = "relation_pass_adapter"
                relation_patches.append({
                    "index": idx,
                    "candidate_id": cid,
                    "fields": patch_fields,
                })
            adapted_relations.append(patched)
        adapted["relations"] = adapted_relations

        target_patches: List[Dict[str, Any]] = []
        target = adapted.get("target")
        if isinstance(target, dict) and target:
            patched_target, target_patch = self._adapt_target(target, target_candidates, target_by_id, index="primary")
            if target_patch:
                target_patches.append(target_patch)
            adapted["target"] = patched_target

        adapted_targets = []
        for idx, item in enumerate(adapted.get("targets") or []):
            if not isinstance(item, dict):
                continue
            patched_target, target_patch = self._adapt_target(item, target_candidates, target_by_id, index=idx)
            if target_patch:
                target_patches.append(target_patch)
            adapted_targets.append(patched_target)
        if adapted_targets:
            adapted["targets"] = self._dedupe_targets(([adapted["target"]] if isinstance(adapted.get("target"), dict) and adapted.get("target") else []) + adapted_targets)

        metadata = dict(adapted.get("metadata") or {})
        metadata["relation_pass_adapter"] = {
            "used": bool(relation_patches or target_patches),
            "num_relation_patches": len(relation_patches),
            "num_candidate_id_patches": self._count_field(relation_patches, "candidate_id") + sum(1 for patch in target_patches if "candidate_id" in patch.get("fields", [])),
            "num_buildability_patches": self._count_field(relation_patches, "buildability"),
            "num_builder_mapping_patches": self._count_field(relation_patches, "builder_mapping"),
            "relation_patches": relation_patches[:100],
            "target_patches": target_patches[:100],
            "target_patch": target_patches[0] if target_patches else {},
        }
        adapted["metadata"] = metadata
        annotated = self.engine_metadata.annotate(semantic=adapted)
        annotated_metadata = dict(annotated.get("metadata") or {})
        adapter_metadata = dict(annotated_metadata.get("relation_pass_adapter") or {})
        engine_metadata = annotated_metadata.get("relation_engine_metadata_annotator") or {}
        adapter_metadata["used"] = bool(adapter_metadata.get("used") or engine_metadata.get("used"))
        adapter_metadata["engine_metadata_patches"] = int(engine_metadata.get("num_patches") or 0)
        adapter_metadata["engine_metadata_buildability_patches"] = int(engine_metadata.get("num_buildability_patches") or 0)
        adapter_metadata["engine_metadata_builder_mapping_patches"] = int(engine_metadata.get("num_builder_mapping_patches") or 0)
        annotated_metadata["relation_pass_adapter"] = adapter_metadata
        annotated["metadata"] = annotated_metadata
        return self._ensure_relation_builder_decisions(annotated)

    def _ensure_relation_builder_decisions(self, semantic: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure every stage-1 relation has a current-capability decision.

        The VLM still owns semantic extraction. This method only normalizes the
        explicit bridge from extracted semantics to the current GeoSym data
        engine. It never creates new relations or geometry facts.
        """
        adapted = dict(semantic or {})
        relations = [
            rel for rel in adapted.get("relations") or []
            if isinstance(rel, dict)
        ]
        existing = list(adapted.get("relation_builder_decisions") or [])
        decision_index: Dict[str, Dict[str, Any]] = {}
        decision_patches: List[Dict[str, Any]] = []
        added: List[Dict[str, Any]] = []
        corrected: List[Dict[str, Any]] = []

        for item in existing:
            if not isinstance(item, dict):
                continue
            key = self._decision_key(item, relations)
            if key:
                decision_index[key] = dict(item)

        for idx, relation in enumerate(relations):
            if str(relation.get("type") or "") == "target":
                continue
            key = self._relation_decision_key(idx, relation)
            expected = self._expected_decision(idx, relation, key)
            actual = dict(decision_index.get(key) or {})
            patched = dict(actual)
            patch_fields = []
            for field in ["candidate_id", "relation_index", "relation_type", "decision", "builder_actions", "core_constraint_action"]:
                if patched.get(field) != expected.get(field):
                    patched[field] = expected.get(field)
                    patch_fields.append(field)
            if not patched.get("reason"):
                patched["reason"] = expected["reason"]
                patch_fields.append("reason")
            if patched.get("confidence") in {None, ""}:
                patched["confidence"] = expected["confidence"]
                patch_fields.append("confidence")
            if patch_fields:
                patched["adapter_source"] = "relation_pass_adapter.capability_decision"
                decision_patches.append({
                    "relation_index": idx,
                    "candidate_id": expected.get("candidate_id"),
                    "relation_type": expected.get("relation_type"),
                    "fields": patch_fields,
                    "created": not bool(actual),
                })
                if actual:
                    corrected.append(patched)
                else:
                    added.append(patched)
            decision_index[key] = patched

        adapted["relation_builder_decisions"] = list(decision_index.values())
        metadata = dict(adapted.get("metadata") or {})
        adapter_metadata = dict(metadata.get("relation_pass_adapter") or {})
        adapter_metadata["used"] = bool(adapter_metadata.get("used") or decision_patches)
        adapter_metadata["capability_decision_patches"] = len(decision_patches)
        adapter_metadata["capability_decisions_added"] = len(added)
        adapter_metadata["capability_decisions_corrected"] = len(corrected)
        adapter_metadata["capability_decision_patch_items"] = decision_patches[:100]
        metadata["relation_pass_adapter"] = adapter_metadata
        adapted["metadata"] = metadata
        return adapted

    def _expected_decision(self, idx: int, relation: Dict[str, Any], key: str) -> Dict[str, Any]:
        rtype = str(relation.get("type") or "")
        mapping = self._mapping_by_type.get(rtype) or {
            "relation_type": rtype,
            "status": "unsupported",
            "builder_actions": [],
            "core_constraint_action": None,
            "note": "current GeoSym builder has no mapping for this relation type",
        }
        status = str(mapping.get("status") or "")
        if status == "buildable":
            decision = "buildable"
        elif status == "convertible":
            decision = "convertible"
        elif status == "theorem_only":
            decision = "theorem_only"
        else:
            decision = "unsupported"
        cid = str(relation.get("candidate_id") or relation.get("from_candidate") or "")
        return {
            "candidate_id": cid,
            "relation_index": idx,
            "relation_key": key,
            "relation_type": rtype,
            "decision": decision,
            "builder_actions": [str(item) for item in mapping.get("builder_actions") or []],
            "core_constraint_action": mapping.get("core_constraint_action"),
            "reason": f"normalized from GeoSym builder capability {self.capabilities.version}: {mapping.get('note') or status}",
            "confidence": relation.get("confidence", 0.8),
        }

    def _decision_key(self, decision: Dict[str, Any], relations: List[Dict[str, Any]]) -> str:
        if decision.get("relation_index") is not None:
            try:
                idx = int(decision.get("relation_index"))
            except (TypeError, ValueError):
                idx = -1
            if 0 <= idx < len(relations):
                return self._relation_decision_key(idx, relations[idx])
        cid = str(decision.get("candidate_id") or decision.get("from_candidate") or "")
        if cid:
            return f"candidate::{cid}"
        relation_key = str(decision.get("relation_key") or "")
        return relation_key

    def _relation_decision_key(self, idx: int, relation: Dict[str, Any]) -> str:
        cid = str(relation.get("candidate_id") or relation.get("from_candidate") or "")
        if cid:
            return f"candidate::{cid}"
        return f"relation::{idx}::{self._relation_signature(relation)}"

    def _relation_signature(self, relation: Dict[str, Any]) -> str:
        return "|".join([
            str(relation.get("type") or ""),
            self._object_key(relation),
            self._value_key(relation.get("value")),
        ])

    def _relation_index(self, candidates: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
        out: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
        for candidate in candidates:
            key = self._relation_key(candidate)
            if not key[0] or not key[1]:
                continue
            out.setdefault(key, []).append(candidate)
        return out

    def _relation_key(self, item: Dict[str, Any]) -> Tuple[str, str, str]:
        rtype = str(item.get("type") or "").strip().lower()
        obj = self._object_key(item)
        value = self._value_key(item.get("value"))
        return (rtype, obj, value)

    def _object_key(self, item: Dict[str, Any]) -> str:
        if item.get("object") not in {None, ""}:
            text = str(item.get("object") or "")
            clean = "".join(ch for ch in text.upper() if ch.isalnum())
            return clean
        objects = item.get("objects")
        if isinstance(objects, list):
            clean_objects = [
                "".join(ch for ch in str(obj).upper() if ch.isalnum())
                for obj in objects
                if obj is not None
            ]
            if len(clean_objects) == 2 and all(len(obj) == 1 for obj in clean_objects):
                return "".join(clean_objects)
            return "|".join(clean_objects)
        lines = item.get("lines")
        if isinstance(lines, list):
            return "|".join("".join(ch for ch in str(line).upper() if ch.isalnum()) for line in lines)
        return ""

    def _value_key(self, value: Any) -> str:
        if value in {None, ""}:
            return ""
        return str(value).replace(" ", "").lower()

    def _match_target(self, target: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        target_type = str(target.get("type") or "").lower()
        target_object = "".join(ch for ch in str(target.get("object") or "").upper() if ch.isalnum())
        for candidate in candidates:
            ctype = str(candidate.get("type") or "").lower()
            cobject = "".join(ch for ch in str(candidate.get("object") or "").upper() if ch.isalnum())
            if target_type and ctype and target_type != ctype:
                continue
            if target_object and cobject and target_object == cobject:
                return candidate
        if len(candidates) == 1:
            return candidates[0]
        return {}

    def _adapt_target(
        self,
        target: Dict[str, Any],
        target_candidates: List[Dict[str, Any]],
        target_by_id: Dict[str, Dict[str, Any]],
        *,
        index: Any,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        patched_target = dict(target)
        patch: Dict[str, Any] = {"index": index, "fields": []}
        cid = str(patched_target.get("candidate_id") or patched_target.get("from_candidate") or "")
        candidate = target_by_id.get(cid) if cid else None
        if not candidate:
            candidate = self._match_target(patched_target, target_candidates)
            if candidate:
                cid = str(candidate.get("id") or "")
                patched_target["candidate_id"] = cid
                patched_target["from_candidate"] = cid
                patch["fields"].extend(["candidate_id", "from_candidate"])
        if candidate:
            if not patched_target.get("type") and candidate.get("type"):
                patched_target["type"] = str(candidate.get("type"))
                patch["fields"].append("type")
            if not patched_target.get("object") and candidate.get("object"):
                patched_target["object"] = str(candidate.get("object"))
                patch["fields"].append("object")
            if not patched_target.get("value_symbol") and candidate.get("value_symbol"):
                patched_target["value_symbol"] = str(candidate.get("value_symbol"))
                patch["fields"].append("value_symbol")
        if patch["fields"]:
            patch["candidate_id"] = cid
            patched_target["adapter_source"] = "relation_pass_adapter"
            return patched_target, patch
        return patched_target, {}

    def _dedupe_targets(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen = set()
        for item in items:
            key = (str(item.get("candidate_id") or item.get("from_candidate") or ""), str(item.get("type") or ""), str(item.get("object") or ""))
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    def _count_field(self, patches: List[Dict[str, Any]], field: str) -> int:
        return sum(1 for patch in patches if field in (patch.get("fields") or []))

    def _same_dict(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        return self._canonical_dict(actual) == self._canonical_dict(expected)

    def _canonical_dict(self, value: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(item, dict):
                out[str(key)] = self._canonical_dict(item)
            elif isinstance(item, list):
                out[str(key)] = [self._canonical_value(part) for part in item]
            else:
                out[str(key)] = self._canonical_value(item)
        return out

    def _canonical_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return value
        return value
