import json
import sys
import types

import pytest
from PIL import Image

from geosym_world.parser import (
    OCRImagePrediction,
    OCRRegionPrediction,
    RecordedLabelRecognizer,
    TesseractLabelRecognizer,
    load_ocr_predictions,
)


def test_recorded_ocr_round_trip_preserves_traceable_label_provenance(tmp_path):
    image_path = tmp_path / "diagram.png"
    Image.new("RGB", (100, 80), "white").save(image_path)
    prediction = _prediction(image_path)
    sidecar_path = tmp_path / "ocr.jsonl"
    sidecar_path.write_text(json.dumps(prediction.to_dict()) + "\n", encoding="utf-8")

    recognizer = load_ocr_predictions(sidecar_path)
    result = recognizer.recognize(
        str(image_path),
        [{"id": "vt_a", "bbox": [10, 12, 8, 10]}],
    )

    assert result[0]["text"] == "A"
    assert result[0]["provenance"]["prediction_checksum"] == prediction.checksum()
    assert result[0]["provenance"]["image_sha256"] == prediction.image_sha256
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
