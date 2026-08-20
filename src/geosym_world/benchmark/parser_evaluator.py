"""Component metrics for Parser candidates against canonical reference states."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from geosym_world.schema.problem import GeoSymProblemState
from geosym_world.schema.problem_normalize import normalize_id, parse_expr
from geosym_world.schema.relation_normalization import normalize_relation_type, normalize_target_type


class ParserStateEvaluator:
    """Compare Parser v0 output without admitting the candidate as persistent state."""

    def evaluate(
        self,
        prediction: GeoSymProblemState | Mapping[str, Any],
        reference: GeoSymProblemState | Mapping[str, Any],
    ) -> dict[str, Any]:
        predicted = self._state_dict(prediction)
        expected = self._state_dict(reference)
        predicted_geometry = dict(predicted.get("geometry") or {})
        expected_geometry = dict(expected.get("geometry") or {})

        object_metric = self._set_metric(
            self._object_signatures(predicted_geometry),
            self._object_signatures(expected_geometry),
        )
        relation_metric = self._set_metric(
            self._relation_signatures(predicted_geometry),
            self._relation_signatures(expected_geometry),
        )
        quantity_metric = self._set_metric(
            self._quantity_signatures(predicted_geometry),
            self._quantity_signatures(expected_geometry),
        )
        target_metric = self._target_metric(predicted.get("target"), expected.get("target"))
        grounding_metric = self._grounding_metric(predicted_geometry, predicted.get("target"))
        component_scores = [
            object_metric["f1"],
            relation_metric["f1"],
            quantity_metric["f1"],
            target_metric["accuracy"],
            grounding_metric["accuracy"],
        ]
        return {
            "objects": object_metric,
            "relations": relation_metric,
            "quantities": quantity_metric,
            "target": target_metric,
            "grounding": grounding_metric,
            "macro_score": sum(component_scores) / len(component_scores),
        }

    def _state_dict(self, value: GeoSymProblemState | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(value, GeoSymProblemState):
            return value.to_dict()
        if isinstance(value, Mapping):
            return dict(value)
        raise TypeError("ParserStateEvaluator expects a GeoSymProblemState or mapping")

    def _object_signatures(self, geometry: Mapping[str, Any]) -> set[str]:
        signatures: set[str] = set()
        fields = (
            ("points", "point"),
            ("lines", "line"),
            ("segments", "segment"),
            ("rays", "ray"),
            ("circles", "circle"),
            ("arcs", "arc"),
            ("polygons", "polygon"),
            ("regions", "region"),
            ("composite_regions", "composite_region"),
            ("planes", "plane"),
            ("polyhedra", "polyhedron"),
            ("entities", "entity"),
        )
        for field, kind in fields:
            for item in geometry.get(field) or []:
                if isinstance(item, Mapping) and item.get("id"):
                    signatures.add(f"{kind}:{normalize_id(item['id'])}")
        return signatures

    def _relation_signatures(self, geometry: Mapping[str, Any]) -> set[str]:
        signatures = set()
        for relation in geometry.get("relations") or []:
            if not isinstance(relation, Mapping):
                continue
            relation_type = normalize_relation_type(str(relation.get("type") or ""))
            if relation_type == "length":
                continue
            references = self._relation_references(relation)
            signatures.add(f"{relation_type}:{','.join(sorted(references))}")
        return signatures

    def _quantity_signatures(self, geometry: Mapping[str, Any]) -> set[str]:
        signatures = set()
        candidates = list(geometry.get("quantities") or [])
        candidates.extend(
            relation
            for relation in geometry.get("relations") or []
            if isinstance(relation, Mapping) and normalize_relation_type(str(relation.get("type") or "")) == "length"
        )
        for quantity in candidates:
            if not isinstance(quantity, Mapping):
                continue
            quantity_type = normalize_relation_type(str(quantity.get("type") or quantity.get("kind") or "quantity"))
            object_id = normalize_id(quantity.get("object") or quantity.get("object_id"))
            value = self._canonical_value(quantity.get("value"))
            signatures.add(f"{quantity_type}:{object_id}:{value}")
        return signatures

    def _target_metric(self, prediction: Any, reference: Any) -> dict[str, Any]:
        predicted = self._target_signature(prediction)
        expected = self._target_signature(reference)
        return {
            "accuracy": float(predicted == expected),
            "predicted": predicted,
            "expected": expected,
        }

    def _grounding_metric(self, geometry: Mapping[str, Any], target: Any) -> dict[str, Any]:
        object_ids = {
            signature.split(":", 1)[1]
            for signature in self._object_signatures(geometry)
        }
        references: list[str] = []
        for relation in geometry.get("relations") or []:
            if isinstance(relation, Mapping):
                references.extend(self._relation_references(relation))
        if isinstance(target, Mapping) and target.get("object"):
            references.append(normalize_id(target["object"]))
        resolved = [reference for reference in references if reference in object_ids]
        total = len(references)
        return {
            "accuracy": len(resolved) / total if total else 1.0,
            "resolved": len(resolved),
            "total": total,
            "unresolved_ids": sorted(set(references).difference(object_ids)),
        }

    def _relation_references(self, relation: Mapping[str, Any]) -> list[str]:
        references: list[str] = []
        for key in ("lines", "points", "objects", "args"):
            value = relation.get(key)
            if isinstance(value, (list, tuple)):
                references.extend(normalize_id(item) for item in value if item is not None)
        object_id = relation.get("object") or relation.get("object_id")
        if object_id:
            references.append(normalize_id(object_id))
        return references

    def _target_signature(self, target: Any) -> str | None:
        if not isinstance(target, Mapping):
            return None
        target_type = normalize_target_type(str(target.get("type") or ""))
        object_id = normalize_id(target.get("object"))
        if not target_type or not object_id:
            return None
        return f"{target_type}:{object_id}"

    def _canonical_value(self, value: Any) -> str:
        if isinstance(value, Mapping):
            value = value.get("expr", value.get("value"))
        parsed = parse_expr(value)
        return str(parsed) if parsed is not None else str(value)

    def _set_metric(self, predicted: set[str], expected: set[str]) -> dict[str, Any]:
        true_positive = len(predicted & expected)
        precision = true_positive / len(predicted) if predicted else float(not expected)
        recall = true_positive / len(expected) if expected else float(not predicted)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positive": true_positive,
            "predicted": len(predicted),
            "expected": len(expected),
            "false_positive": sorted(predicted - expected),
            "false_negative": sorted(expected - predicted),
        }
