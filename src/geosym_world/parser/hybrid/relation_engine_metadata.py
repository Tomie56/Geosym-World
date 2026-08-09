"""Attach GeoSym builder capability metadata to relation drafts."""

from __future__ import annotations

from typing import Any, Dict, List

from .builder_capabilities import GEOSYM_2D_BUILDER_CAPABILITIES, GeoSymBuilderCapabilities


class RelationEngineMetadataAnnotator:
    """Annotate relation drafts with current GeoSym buildability metadata.

    This is intentionally not a candidate linker and not a builder. It only
    explains whether the relation type is buildable by the current engine. A
    relation without a script candidate id remains evidence for diagnosis/formal
    graph, but it does not automatically become a required build task.
    """

    def __init__(self, capabilities: GeoSymBuilderCapabilities = GEOSYM_2D_BUILDER_CAPABILITIES):
        self.capabilities = capabilities
        self._mapping_by_type = {
            str(item.get("relation_type")): dict(item)
            for item in self.capabilities.relation_mapping_contract()
            if item.get("relation_type")
        }

    def annotate(self, *, semantic: Dict[str, Any]) -> Dict[str, Any]:
        annotated = dict(semantic or {})
        patches: List[Dict[str, Any]] = []
        relations = []
        for idx, relation in enumerate(annotated.get("relations") or []):
            if not isinstance(relation, dict):
                continue
            patched = dict(relation)
            patch_fields: List[str] = []
            rtype = str(patched.get("type") or "")
            if rtype:
                expected_buildability = self._buildability_for_type(rtype)
                expected_builder_mapping = self._builder_mapping_for_type(rtype)
                if self._needs_buildability_patch(patched.get("buildability"), expected_buildability):
                    patched["buildability"] = expected_buildability
                    patch_fields.append("buildability")
                if self._needs_builder_mapping_patch(patched.get("builder_mapping"), expected_builder_mapping):
                    patched["builder_mapping"] = expected_builder_mapping
                    patch_fields.append("builder_mapping")
            if patch_fields:
                patched["engine_metadata_source"] = "relation_engine_metadata_annotator"
                patches.append({
                    "index": idx,
                    "candidate_id": patched.get("candidate_id") or patched.get("from_candidate"),
                    "relation_type": rtype,
                    "fields": patch_fields,
                })
            relations.append(patched)
        annotated["relations"] = relations
        metadata = dict(annotated.get("metadata") or {})
        metadata["relation_engine_metadata_annotator"] = {
            "used": bool(patches),
            "num_patches": len(patches),
            "num_buildability_patches": self._count_patch_field(patches, "buildability"),
            "num_builder_mapping_patches": self._count_patch_field(patches, "builder_mapping"),
            "patches": patches[:100],
        }
        annotated["metadata"] = metadata
        return annotated

    def _buildability_for_type(self, relation_type: str) -> Dict[str, Any]:
        mapping = self._mapping_by_type.get(relation_type)
        if not mapping:
            return {
                "mode": "unsupported",
                "can_generate_build_instruction": False,
                "required_confirmation": "current GeoSym builder does not support this relation type",
            }
        mode = str(mapping.get("buildability_mode") or "unsupported")
        status = str(mapping.get("status") or "")
        return {
            "mode": mode,
            "can_generate_build_instruction": status in {"buildable", "convertible"} and bool(mapping.get("builder_actions") or []),
            "required_confirmation": mapping.get("note") or "follow current GeoSym builder contract",
        }

    def _builder_mapping_for_type(self, relation_type: str) -> Dict[str, Any]:
        mapping = dict(self._mapping_by_type.get(relation_type) or {})
        if not mapping:
            return {
                "status": "not_buildable",
                "relation_type": relation_type,
                "builder_actions": [],
                "must_not_emit_fake_constraints": True,
                "reason": "current GeoSym builder does not support this relation type",
            }
        status = str(mapping.get("status") or "")
        out = {
            "status": status or "not_buildable",
            "relation_type": relation_type,
            "buildability_mode": mapping.get("buildability_mode"),
            "builder_actions": list(mapping.get("builder_actions") or []),
            "core_constraint_action": mapping.get("core_constraint_action"),
            "instruction_requirements_source": "relation_engine_metadata_annotator",
            "rule": "metadata only; relation still needs candidate/task evidence before canonical build",
        }
        if status not in {"buildable", "convertible"}:
            out["must_not_emit_fake_constraints"] = True
            out["reason"] = mapping.get("note") or "current builder cannot encode this theorem-level relation"
        return out

    def _count_patch_field(self, patches: List[Dict[str, Any]], field: str) -> int:
        return sum(1 for item in patches if field in (item.get("fields") or []))

    def _needs_buildability_patch(self, actual: Any, expected: Dict[str, Any]) -> bool:
        if not isinstance(actual, dict) or not actual:
            return True
        for field in ["mode", "can_generate_build_instruction"]:
            if actual.get(field) != expected.get(field):
                return True
        return False

    def _needs_builder_mapping_patch(self, actual: Any, expected: Dict[str, Any]) -> bool:
        if not isinstance(actual, dict) or not actual:
            return True
        for field in ["status", "relation_type", "buildability_mode", "builder_actions", "core_constraint_action"]:
            if actual.get(field) != expected.get(field):
                return True
        return False
