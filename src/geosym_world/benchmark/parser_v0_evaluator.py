"""Component diagnostics for the clean image Parser v0."""

from __future__ import annotations

from collections.abc import Mapping
from statistics import mean
from typing import Any


class CleanParserEvaluatorV0:
    def evaluate(self, prediction: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
        observable = reference.get("observable_state") or {}
        expected_objects = observable.get("objects") or {}
        observations = prediction.get("observations") or {}
        primitives = observations.get("primitives") or {}
        grounding = prediction.get("grounding") or {}

        auxiliary_lines = sum(
            1
            for item in observable.get("visual_evidence") or ()
            if item.get("role") == "auxiliary" and item.get("kind") in {"line", "ray", "segment"}
        )
        line_expected = sum(len(expected_objects.get(bucket) or {}) for bucket in ("lines", "segments", "rays")) + auxiliary_lines
        circle_expected = len(expected_objects.get("circles") or {})
        point_expected = len(expected_objects.get("points") or {})
        line_metric = _count_metric(len(primitives.get("lines") or ()), line_expected)
        circle_metric = _count_metric(len(primitives.get("circles") or ()), circle_expected)
        point_metric = _count_metric(len(grounding.get("points") or ()), point_expected)

        expected_labels = set(str(value) for value in (observable.get("point_labels") or {}).values())
        predicted_labels = {str(item["text"]) for item in observations.get("labels") or () if item.get("text")}
        label_metric = _set_metric(predicted_labels, expected_labels)
        expected_relations = {_visual_relation_type(str(item.get("type"))) for item in observable.get("relations") or ()}
        predicted_relations = {str(item.get("type")) for item in prediction.get("relation_hypotheses") or ()}
        relation_metric = _set_metric(predicted_relations, expected_relations)

        expected_target = (observable.get("target") or {}).get("type")
        predicted_target = (prediction.get("target_hypothesis") or {}).get("type")
        target_accuracy = float(expected_target is not None and predicted_target == expected_target)
        component_scores = [line_metric["f1"], circle_metric["f1"], point_metric["f1"], label_metric["f1"], relation_metric["f1"], target_accuracy]
        diagnostics = []
        if line_metric["precision"] < 0.7:
            diagnostics.append("line_oversegmentation")
        if line_metric["recall"] < 0.7:
            diagnostics.append("line_miss")
        if circle_metric["recall"] < 0.7:
            diagnostics.append("circle_miss")
        if label_metric["recall"] < 0.7:
            diagnostics.append("label_ocr_unresolved")
        if point_metric["f1"] < 0.7:
            diagnostics.append("topology_point_mismatch")
        if relation_metric["recall"] < 0.7:
            diagnostics.append("relation_miss")
        if target_accuracy == 0.0:
            diagnostics.append("target_miss")
        return {
            "schema_version": "1.0",
            "objects": {"linear": line_metric, "circle": circle_metric, "point": point_metric},
            "labels": label_metric,
            "relations": relation_metric,
            "target": {"accuracy": target_accuracy, "expected": expected_target, "predicted": predicted_target},
            "macro_score": mean(component_scores),
            "diagnostic_axes": diagnostics,
        }


def _count_metric(predicted: int, expected: int) -> dict[str, Any]:
    if predicted == expected == 0:
        precision = recall = 1.0
    else:
        matched = min(predicted, expected)
        precision = matched / predicted if predicted else 0.0
        recall = matched / expected if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"predicted": predicted, "expected": expected, "precision": precision, "recall": recall, "f1": f1}


def _set_metric(predicted: set[str], expected: set[str]) -> dict[str, Any]:
    matched = len(predicted & expected)
    precision = matched / len(predicted) if predicted else float(not expected)
    recall = matched / len(expected) if expected else float(not predicted)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "predicted": sorted(predicted),
        "expected": sorted(expected),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _visual_relation_type(relation_type: str) -> str:
    return {
        "foot_of_perpendicular": "perpendicular",
    }.get(relation_type, relation_type)
