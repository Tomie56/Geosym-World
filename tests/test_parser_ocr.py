import json
import sys
import types

import pytest
from PIL import Image

from geosym_world.parser import (
    OCRImagePrediction,
    OCRRegionPrediction,
    ParserInput,
    RecordedLabelRecognizer,
    ReconstructionMode,
    TesseractLabelRecognizer,
    TypedDiagramParserV0,
    load_ocr_predictions,
)
from geosym_world.schema import ParseObjectKind
from geosym_world.tools.parse_reconstruct_v0 import run as run_workflow


class TextRegionDetector:
    def detect(self, image_path: str):
        return {
            "image_size": {"width": 100, "height": 80},
            "lines": [],
            "circles": [],
            "fill_regions": [],
            "markers": [],
            "text_regions": [{"id": "vt_a", "bbox": [10, 12, 8, 10], "area_px": 40}],
            "topology": {
                "endpoints": [],
                "line_intersections": [],
                "circle_line_intersections": [],
                "circle_circle_intersections": [],
            },
        }


def test_recorded_ocr_round_trip_preserves_traceable_label_provenance(tmp_path):
    image_path = tmp_path / "diagram.png"
    Image.new("RGB", (100, 80), "white").save(image_path)
    prediction = _prediction(image_path)
    sidecar_path = tmp_path / "ocr.jsonl"
    sidecar_path.write_text(json.dumps(prediction.to_dict()) + "\n", encoding="utf-8")

    recognizer = load_ocr_predictions(sidecar_path)
    result = TypedDiagramParserV0(
        detector=TextRegionDetector(),
        label_recognizer=recognizer,
    ).parse(ParserInput(id="ocr_sample", image=str(image_path), question=""))

    label = next(item for item in result.graph.objects if item.kind is ParseObjectKind.TEXT)
    evidence = next(item for item in result.graph.evidence if item.id == label.evidence_refs[0])
    assert label.visible_name == "A"
    assert label.metadata["recognizer_provenance"]["prediction_checksum"] == prediction.checksum()
    assert evidence.metadata["recognizer_provenance"]["image_sha256"] == prediction.image_sha256
    assert recognizer.configuration()["prediction_set_checksum"]


def test_recorded_ocr_rejects_tampering_and_detector_region_drift(tmp_path):
    image_path = tmp_path / "diagram.png"
    Image.new("RGB", (100, 80), "white").save(image_path)
    prediction = _prediction(image_path)
    payload = prediction.to_dict()
    payload["regions"][0]["text"] = "B"

    with pytest.raises(ValueError, match="checksum does not match"):
        OCRImagePrediction.from_dict(payload)

    recognizer = RecordedLabelRecognizer((prediction,))
    with pytest.raises(ValueError, match="region set does not match"):
        recognizer.recognize(
            str(image_path),
            [{"id": "vt_a", "bbox": [11, 12, 8, 10]}],
        )


def test_verified_parser_cli_records_ocr_sidecar_configuration(tmp_path):
    image_path = tmp_path / "blank.png"
    Image.new("RGB", (100, 80), "white").save(image_path)
    parser_input = ParserInput(id="blank", image=str(image_path), question="")
    input_path = tmp_path / "inputs.jsonl"
    input_path.write_text(json.dumps(parser_input.to_dict()) + "\n", encoding="utf-8")
    prediction = OCRImagePrediction.create(
        image_path,
        backend="fixture",
        model="fixture-ocr",
        model_revision="r1",
        regions=(),
    )
    sidecar_path = tmp_path / "ocr.jsonl"
    sidecar_path.write_text(json.dumps(prediction.to_dict()) + "\n", encoding="utf-8")

    manifest = run_workflow(
        input_path=input_path,
        output_path=tmp_path / "results.jsonl",
        render_dir=tmp_path / "renders",
        reconstruction_mode=ReconstructionMode.PIXEL_WITNESS,
        output_format="none",
        resynthesis_seeds=(),
        width=320,
        height=240,
        progress_every=0,
        ocr_predictions_path=sidecar_path,
    )

    assert manifest["schema_version"] == "geosym.parser_run_manifest.v8"
    assert manifest["ocr_predictions"] == str(sidecar_path)
    assert manifest["ocr_predictions_sha256"]
    assert manifest["configuration"]["parser"]["kind"] == "opencv_typed_v0"
    assert manifest["configuration"]["parser"]["ocr"]["model_revisions"] == ["r1"]


def test_tesseract_recognizer_runs_one_crop_per_visual_region(tmp_path, monkeypatch):
    image_path = tmp_path / "diagram.png"
    Image.new("RGB", (100, 80), "white").save(image_path)
    calls = []

    def image_to_data(image, *, lang, config, output_type):
        calls.append((image.size, lang, config, output_type))
        return {"text": ["A", ""], "conf": ["87", "-1"]}

    fake_pytesseract = types.SimpleNamespace(
        Output=types.SimpleNamespace(DICT="dict"),
        image_to_data=image_to_data,
    )
    monkeypatch.setitem(sys.modules, "pytesseract", fake_pytesseract)

    recognizer = TesseractLabelRecognizer(language="eng", config="--psm 10", padding_px=2)
    results = recognizer.recognize(
        str(image_path),
        [{"id": "vt_a", "bbox": [10, 12, 8, 10]}],
    )

    assert calls == [((12, 14), "eng", "--psm 10", "dict")]
    assert results[0]["text"] == "A"
    assert results[0]["status"] == "resolved"
    assert results[0]["confidence"] == 0.87
    assert results[0]["provenance"]["method"] == "tesseract_region_crop"
    assert recognizer.configuration()["kind"] == "tesseract_regions"


def test_verified_parser_workflow_records_tesseract_configuration(tmp_path, monkeypatch):
    image_path = tmp_path / "blank.png"
    Image.new("RGB", (100, 80), "white").save(image_path)
    input_path = tmp_path / "inputs.jsonl"
    input_path.write_text(
        json.dumps(ParserInput(id="blank", image=str(image_path), question="").to_dict()) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(
        sys.modules,
        "pytesseract",
        types.SimpleNamespace(Output=types.SimpleNamespace(DICT="dict")),
    )

    manifest = run_workflow(
        input_path=input_path,
        output_path=tmp_path / "results.jsonl",
        render_dir=tmp_path / "renders",
        reconstruction_mode=ReconstructionMode.PIXEL_WITNESS,
        output_format="none",
        resynthesis_seeds=(),
        width=320,
        height=240,
        progress_every=0,
        tesseract_ocr=True,
    )

    assert manifest["configuration"]["parser"]["ocr"]["kind"] == "tesseract_regions"


def _prediction(image_path) -> OCRImagePrediction:
    return OCRImagePrediction.create(
        image_path,
        backend="fixture",
        model="fixture-ocr",
        model_revision="r1",
        regions=(
            OCRRegionPrediction(
                region_id="vt_a",
                bbox=(10, 12, 8, 10),
                text="A",
                confidence=0.97,
                status="resolved",
            ),
        ),
    )
