from pathlib import Path

from geosym_world.parser import ParserInput, TypedDiagramParserV0
from geosym_world.schema import AdmissionStatus, ParseOrigin
from geosym_world.verifier import ParseGraphAdmission


class SegmentDetector:
    def __init__(self, line_y_values):
        self.line_y_values = tuple(line_y_values)

    def detect(self, image_path: str):
        lines = [
            {
                "id": f"vl_{index}",
                "p1": [20, y],
                "p2": [180, y],
                "angle_deg": 0.0,
                "source_ids": [f"raw_{index}"],
            }
            for index, y in enumerate(self.line_y_values)
        ]
        return {
            "image_size": {"width": 200, "height": 200},
            "lines": lines,
            "circles": [],
            "text_regions": [{"id": "vt_value", "bbox": [94, 94, 12, 12]}],
            "topology": {
                "endpoints": [
                    endpoint
                    for index, y in enumerate(self.line_y_values)
                    for endpoint in (
                        {"id": f"vl_{index}_p1", "line_id": f"vl_{index}", "xy": [20, y]},
                        {"id": f"vl_{index}_p2", "line_id": f"vl_{index}", "xy": [180, y]},
                    )
                ],
                "line_intersections": [],
                "circle_line_intersections": [],
                "circle_circle_intersections": [],
            },
        }


class Recognizer:
    def __init__(self, text):
        self.text = text

    def recognize(self, image_path, text_regions):
        return [
            {
                "id": "vt_value",
                "bbox": [94, 94, 12, 12],
                "text": self.text,
                "confidence": 0.96,
                "status": "resolved",
            }
        ]


def test_image_fraction_becomes_a_conservative_segment_length_candidate(tmp_path):
    result = _parse(tmp_path, line_y_values=(120,), text=r"\frac{3}{2}")

    assert len(result.graph.quantities) == 1
    quantity = result.graph.quantities[0]
    assert quantity.origin is ParseOrigin.IMAGE
    assert quantity.kind == "length"
    assert quantity.arguments[0].ref == "segment:vl_0"
    assert quantity.dependencies == ("text:vt_value",)
    assert str(quantity.expression.to_sympy()) == "3/2"
    assert quantity.metadata["binding_status"] == "proposed"
    assert result.graph.metadata["image_quantity_issues"] == []
    assert _stage(result, "image_quantity_grounding")["output_count"] == 1

    admission = ParseGraphAdmission().admit(result.graph)
    decision = next(item for item in admission.decisions if item.candidate_id == quantity.id)
    assert decision.decision is AdmissionStatus.ABSTAINED
    assert decision.reason == "evidence_precision_below_bound"


def test_equally_near_segments_remain_an_explicit_quantity_conflict(tmp_path):
    result = _parse(tmp_path, line_y_values=(80, 120), text="7")

    assert len(result.graph.quantities) == 2
    conflict = result.graph.conflicts[0]
    assert set(conflict.members) == {quantity.id for quantity in result.graph.quantities}
    assert conflict.reason == "multiple_segments_within_spatial_margin"
    assert result.graph.metadata["image_quantity_issues"][0]["status"] == "object_ambiguous"
    assert _stage(result, "point_label_grounding")["status"] == "complete"
    assert _stage(result, "image_quantity_grounding")["status"] == "partial"

    admission = ParseGraphAdmission().admit(result.graph)
    decisions = [
        item
        for item in admission.decisions
        if item.candidate_id in {quantity.id for quantity in result.graph.quantities}
    ]
    assert {item.reason for item in decisions} == {"unresolved_conflict"}


def test_invalid_measurement_expression_is_preserved_as_an_issue(tmp_path):
    result = _parse(tmp_path, line_y_values=(120,), text="3.__class__")

    assert result.graph.quantities == ()
    issue = result.graph.metadata["image_quantity_issues"][0]
    assert issue["status"] == "invalid_expression"
    assert issue["raw_text"] == "3.__class__"
    assert _stage(result, "image_quantity_grounding")["status"] == "partial"


def test_point_names_are_not_misclassified_as_image_quantities(tmp_path):
    for point_name in ("A", "A1"):
        result = _parse(tmp_path, line_y_values=(120,), text=point_name)

        assert result.graph.quantities == ()
        assert result.graph.metadata["image_quantity_issues"] == []


def _parse(tmp_path: Path, *, line_y_values, text):
    image_path = tmp_path / "opaque.png"
    image_path.write_bytes(b"fixture")
    parser = TypedDiagramParserV0(
        detector=SegmentDetector(line_y_values),
        label_recognizer=Recognizer(text),
        infer_planar_faces=False,
    )
    return parser.parse(ParserInput(id="image_quantity", image=str(image_path), question=""))


def _stage(result, name):
    return next(stage for stage in result.stages if stage["name"] == name)
