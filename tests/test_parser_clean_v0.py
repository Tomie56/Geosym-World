import cv2
import json
import numpy as np
import pytest
import sys

from geosym_world.parser import CleanDiagramParserV0, ParserInput
from geosym_world.tools.parse_images_v0 import main as parse_images_v0_main


def _diagram(path):
    image = np.full((256, 256, 3), 255, dtype=np.uint8)
    cv2.line(image, (35, 210), (220, 210), (0, 0, 0), 3)
    cv2.line(image, (35, 210), (35, 40), (0, 0, 0), 3)
    cv2.line(image, (35, 40), (220, 210), (0, 0, 0), 3)
    cv2.putText(image, "A", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), image)


def test_clean_parser_uses_only_public_parser_input(tmp_path):
    image_path = tmp_path / "triangle.png"
    _diagram(image_path)

    result = CleanDiagramParserV0().parse(ParserInput(id="clean_triangle", image=str(image_path), question="Find the area."))
    hypothesis = result.record.hypotheses[0]

    assert result.record.state is None
    assert hypothesis.kind == "visual_geometry_graph"
    assert hypothesis.metadata["admission_required"] is True
    assert hypothesis.payload["observations"]["primitives"]["lines"]
    assert len(hypothesis.payload["observations"]["primitives"]["lines"]) <= 5
    assert hypothesis.payload["observations"]["primitives"]["circles"] == []
    assert hypothesis.payload["observations"]["primitives"]["markers"] == []
    assert hypothesis.payload["grounding"]["points"]
    assert hypothesis.payload["target_hypothesis"]["type"] == "area"
    assert {stage["name"] for stage in result.stages} == {
        "normalize",
        "ocr_labels",
        "primitive_detection",
        "topology",
        "relation_hypotheses",
        "grounding",
        "target",
    }


def test_clean_parser_rejects_private_runtime_fields(tmp_path):
    image_path = tmp_path / "triangle.png"
    _diagram(image_path)
    payload = ParserInput(id="clean_triangle", image=str(image_path), question="Find the area.").to_dict()

    with pytest.raises(ValueError, match="unsupported fields"):
        CleanDiagramParserV0().parse({**payload, "formal_state": {}})


def test_clean_parser_cli_reads_public_jsonl(tmp_path, monkeypatch, capsys):
    image_path = tmp_path / "triangle.png"
    _diagram(image_path)
    input_path = tmp_path / "parser-input.jsonl"
    output_path = tmp_path / "results.jsonl"
    input_path.write_text(json.dumps(ParserInput(id="clean_triangle", image=str(image_path), question="Find the area.").to_dict()) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["parse_images_v0", "--input", str(input_path), "--output", str(output_path), "--progress-every", "1"])

    parse_images_v0_main()

    result = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert result["record"]["state"] is None
    assert result["record"]["hypotheses"][0]["status"] == "proposed"
    assert "processed=1" in capsys.readouterr().err
