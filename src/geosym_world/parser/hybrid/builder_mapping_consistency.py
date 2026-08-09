"""Check relation buildability metadata against current GeoSym capabilities."""

from __future__ import annotations

from typing import Any, Dict, List

from .builder_capabilities import GEOSYM_2D_BUILDER_CAPABILITIES, GeoSymBuilderCapabilities


class BuilderMappingConsistencyAnalyzer:
    """Validate VLM relation metadata against the builder capability contract."""

    def __init__(self, capabilities: GeoSymBuilderCapabilities = GEOSYM_2D_BUILDER_CAPABILITIES):
        self.capabilities = capabilities
        self._mapping_by_type = {
            str(item.get("relation_type")): item
            for item in self.capabilities.relation_mapping_contract()
            if item.get("relation_type")
        }

    def analyze(self, *, semantic: Dict[str, Any]) -> Dict[str, Any]:
        reports: List[Dict[str, Any]] = []
        for idx, relation in enumerate(semantic.get("relations") or []):
            if not isinstance(relation, dict):
                continue
            rtype = str(relation.get("type") or "")
            if not rtype:
                continue
            expected = self._mapping_by_type.get(rtype)
            if not expected:
                expected = {
                    "relation_type": rtype,
                    "status": "unsupported",
                    "buildability_mode": "unsupported",
                    "builder_actions": [],
                    "core_constraint_action": None,
                }
            report = self._relation_report(idx, relation, expected)
            if report.get("issues"):
                reports.append(report)
        return {
            "num_checked_relations": sum(1 for item in semantic.get("relations") or [] if isinstance(item, dict) and item.get("type")),
            "num_inconsistent_relation_buildability": len(reports),
            "inconsistent_relation_buildability_reports": reports[:120],
        }

    def _relation_report(self, idx: int, relation: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, Any]:
        buildability = relation.get("buildability") if isinstance(relation.get("buildability"), dict) else {}
        mapping = relation.get("builder_mapping") if isinstance(relation.get("builder_mapping"), dict) else {}
        expected_status = str(expected.get("status") or "")
        expected_mode = str(expected.get("buildability_mode") or "")
        expected_actions = [str(item) for item in expected.get("builder_actions") or []]
        expected_core = expected.get("core_constraint_action")
        expected_can_generate = expected_status in {"buildable", "convertible"} and bool(expected_actions)
        issues: List[Dict[str, Any]] = []

        actual_mode = str(buildability.get("mode") or "")
        if actual_mode and expected_mode and actual_mode != expected_mode:
            issues.append({
                "field": "buildability.mode",
                "actual": actual_mode,
                "expected": expected_mode,
            })

        if "can_generate_build_instruction" in buildability:
            actual_can_generate = bool(buildability.get("can_generate_build_instruction"))
            if actual_can_generate != expected_can_generate:
                issues.append({
                    "field": "buildability.can_generate_build_instruction",
                    "actual": actual_can_generate,
                    "expected": expected_can_generate,
                })

        actual_status = str(mapping.get("status") or "")
        if actual_status and expected_status and actual_status != expected_status:
            issues.append({
                "field": "builder_mapping.status",
                "actual": actual_status,
                "expected": expected_status,
            })

        actual_actions = [str(item) for item in mapping.get("builder_actions") or []]
        if actual_actions and actual_actions != expected_actions:
            issues.append({
                "field": "builder_mapping.builder_actions",
                "actual": actual_actions,
                "expected": expected_actions,
            })

        if "core_constraint_action" in mapping and mapping.get("core_constraint_action") != expected_core:
            issues.append({
                "field": "builder_mapping.core_constraint_action",
                "actual": mapping.get("core_constraint_action"),
                "expected": expected_core,
            })

        return {
            "relation_index": idx,
            "candidate_id": relation.get("candidate_id") or relation.get("from_candidate"),
            "relation_type": relation.get("type"),
            "expected": {
                "status": expected_status,
                "buildability_mode": expected_mode,
                "can_generate_build_instruction": expected_can_generate,
                "builder_actions": expected_actions,
                "core_constraint_action": expected_core,
            },
            "actual": {
                "buildability": buildability,
                "builder_mapping": mapping,
            },
            "issues": issues,
        }
