"""Staged image-only baseline for clean synthetic geometry diagrams."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from geosym_world.errors import InvalidGeometryInput
from geosym_world.parser.adapters.hypothesis_record import HypothesisRecordAdapter
from geosym_world.parser.contracts import ParserInput, validate_parser_input
from geosym_world.parser.question import infer_target_type
from geosym_world.parser.vision import VisualPrimitiveDetector
from geosym_world.schema import GeometryRecord
from geosym_world.schema.canonical import stable_checksum


CLEAN_PARSER_SCHEMA_VERSION = "1.1"


class LabelRecognizer(Protocol):
    def recognize(self, image_path: str, text_regions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]: ...


class UnresolvedLabelRecognizer:
    """Preserve label regions without pretending OCR text is known."""

    def recognize(self, image_path: str, text_regions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "id": str(region["id"]),
                "bbox": list(region["bbox"]),
                "text": None,
                "confidence": 0.0,
                "status": "ocr_backend_unavailable",
                "provenance": {"source": image_path, "method": "connected_component"},
            }
            for region in text_regions
        ]


@dataclass(frozen=True)
class CleanParseResult:
    parser_input: ParserInput
    record: GeometryRecord
    stages: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "parser_input": self.parser_input.to_dict(),
            "record": self.record.to_dict(include_state=False),
            "stages": [dict(stage) for stage in self.stages],
        }


class CleanDiagramParserV0:
    """Parse clean diagrams into uncertain visual-geometry hypotheses."""

    parser_name = "clean_diagram_parser_v0"

    def __init__(
        self,
        *,
        detector: VisualPrimitiveDetector | None = None,
        label_recognizer: LabelRecognizer | None = None,
    ):
        self.detector = detector or VisualPrimitiveDetector()
        self.label_recognizer = label_recognizer or UnresolvedLabelRecognizer()
        self.record_adapter = HypothesisRecordAdapter()

    def parse(self, data: ParserInput | Mapping[str, Any]) -> CleanParseResult:
        parser_input = validate_parser_input(data)
        image_path = Path(parser_input.image)
        if not image_path.is_file():
            raise InvalidGeometryInput(f"Parser input image does not exist: {image_path}")

        visual = self.detector.detect(str(image_path))
        if visual.get("error"):
            raise InvalidGeometryInput(f"Parser could not read image {image_path}: {visual['error']}")
        labels = self.label_recognizer.recognize(str(image_path), visual.get("text_regions") or ())
        primitives = {
            "lines": [dict(item) for item in visual.get("lines") or ()],
            "circles": [dict(item) for item in visual.get("circles") or ()],
            "arcs": [dict(item) for item in visual.get("arcs") or ()],
            "point_candidates": [
                dict(item) for item in (visual.get("topology") or {}).get("point_candidates") or ()
            ],
            "fill_regions": [dict(item) for item in visual.get("fill_regions") or ()],
            "markers": [dict(item) for item in visual.get("markers") or ()],
        }
        topology = dict(visual.get("topology") or {})
        point_candidates = _ground_visual_points(topology)
        relation_hypotheses = _relation_hypotheses(primitives["lines"], topology)
        target_hypothesis = _target_hypothesis(parser_input.question)
        stages = (
            _stage("normalize", "complete", 1, {"image_size": visual.get("image_size")}),
            _stage("ocr_labels", "complete" if any(item.get("text") for item in labels) else "partial", len(labels), {"resolved": sum(bool(item.get("text")) for item in labels)}),
            _stage("primitive_detection", "complete", sum(len(items) for items in primitives.values())),
            _stage("topology", "complete", len(point_candidates)),
            _stage("relation_hypotheses", "complete", len(relation_hypotheses)),
            _stage("grounding", "partial" if labels else "complete", len(point_candidates), {"unresolved_labels": sum(not item.get("text") for item in labels)}),
            _stage("target", "complete" if target_hypothesis else "partial", int(target_hypothesis is not None)),
        )
        payload = {
            "schema_version": CLEAN_PARSER_SCHEMA_VERSION,
            "sample_id": parser_input.id,
            "observations": {
                "image_size": visual.get("image_size"),
                "labels": labels,
                "primitives": primitives,
                "topology": topology,
            },
            "grounding": {"points": point_candidates, "label_bindings": []},
            "relation_hypotheses": relation_hypotheses,
            "target_hypothesis": target_hypothesis,
            "stage_trace": list(stages),
            "input_checksum": stable_checksum(parser_input.to_dict()),
        }
        confidence = _overall_confidence(primitives, labels, target_hypothesis)
        record = self.record_adapter.convert(
            sample_id=parser_input.id,
            dataset_name="parser_benchmark",
            parser_name=self.parser_name,
            backend="opencv_clean_v0",
            candidate_kind="visual_geometry_graph",
            candidate_payload=payload,
            confidence=confidence,
            verification={"status": "not_admitted", "reason": "visual_hypothesis_requires_grounding_and_geometric_verification"},
            image_path=parser_input.image,
            metadata={"parser_schema_version": CLEAN_PARSER_SCHEMA_VERSION, "stage_count": len(stages)},
        )
        return CleanParseResult(parser_input=parser_input, record=record, stages=stages)


def _stage(name: str, status: str, output_count: int, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "status": status, "output_count": output_count, "metadata": dict(metadata or {})}


def _ground_visual_points(topology: Mapping[str, Any], tolerance: float = 8.0) -> list[dict[str, Any]]:
    observations = [
        {"xy": item["xy"], "source": item["id"], "kind": "endpoint"}
        for item in topology.get("endpoints") or ()
    ]
    observations.extend(
        {"xy": item["xy"], "source": item["id"], "kind": "intersection"}
        for item in topology.get("line_intersections") or ()
    )
    observations.extend(
        {"xy": item["xy"], "source": item["id"], "kind": "explicit_point_blob"}
        for item in topology.get("point_candidates") or ()
    )
    clusters: list[dict[str, Any]] = []
    for observation in observations:
        x, y = (float(value) for value in observation["xy"])
        match = next((item for item in clusters if math.hypot(x - item["center"][0], y - item["center"][1]) <= tolerance), None)
        if match is None:
            clusters.append({"center": [x, y], "evidence": [observation]})
            continue
        match["evidence"].append(observation)
        count = len(match["evidence"])
        match["center"] = [(match["center"][0] * (count - 1) + x) / count, (match["center"][1] * (count - 1) + y) / count]
    return [
        {
            "id": f"vp_{index}",
            "pixel": [round(value, 3) for value in item["center"]],
            "confidence": min(0.95, 0.55 + 0.08 * len(item["evidence"])),
            "evidence": item["evidence"],
            "status": "unlabeled_visual_point",
        }
        for index, item in enumerate(clusters)
    ]


def _relation_hypotheses(lines: Sequence[Mapping[str, Any]], topology: Mapping[str, Any]) -> list[dict[str, Any]]:
    relations = [
        {
            "id": f"rel_intersection_{index}",
            "type": "intersection",
            "objects": list(item["line_ids"]),
            "pixel": list(item["xy"]),
            "confidence": 0.8,
            "provenance": {"method": "bounded_segment_intersection", "evidence_id": item["id"]},
        }
        for index, item in enumerate(topology.get("line_intersections") or ())
    ]
    for index, left in enumerate(lines):
        for right in lines[index + 1 :]:
            delta = _angle_distance(float(left["angle_deg"]), float(right["angle_deg"]))
            relation_type = "parallel" if delta <= 2.5 else "perpendicular" if abs(delta - 90.0) <= 3.0 else None
            if relation_type is None:
                continue
            residual = delta if relation_type == "parallel" else abs(delta - 90.0)
            relations.append(
                {
                    "id": f"rel_{relation_type}_{len(relations)}",
                    "type": relation_type,
                    "objects": [str(left["id"]), str(right["id"])],
                    "confidence": round(max(0.5, 0.9 - residual / 10.0), 3),
                    "provenance": {"method": "line_angle_residual", "residual_deg": round(residual, 3)},
                }
            )
    return relations


def _angle_distance(left: float, right: float) -> float:
    delta = abs(left - right) % 180.0
    return min(delta, 180.0 - delta)


def _target_hypothesis(question: str) -> dict[str, Any] | None:
    match = infer_target_type(question)
    if match is None:
        return None
    return {
        "type": match.target_type,
        "object": None,
        "confidence": 0.8,
        "status": "type_resolved_object_unresolved",
        "provenance": {"method": "question_pattern", "pattern": match.pattern},
    }


def _overall_confidence(primitives: Mapping[str, Sequence[Any]], labels: Sequence[Mapping[str, Any]], target: Mapping[str, Any] | None) -> float:
    visual_score = 0.55 if primitives.get("lines") or primitives.get("circles") else 0.15
    target_score = 0.15 if target else 0.0
    label_score = 0.2 if any(item.get("text") for item in labels) else 0.0
    return min(0.95, visual_score + target_score + label_score)
