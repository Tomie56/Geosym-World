"""Checksum-bound OCR observations for reproducible Parser runs."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from geosym_world.errors import InvalidGeometryInput
from geosym_world.schema.canonical import stable_checksum


OCR_PREDICTION_SCHEMA_VERSION = "geosym.ocr_image_prediction.v1"
_OCR_STATUSES = frozenset({"resolved", "unresolved"})


@dataclass(frozen=True)
class OCRRegionPrediction:
    region_id: str
    bbox: tuple[float, float, float, float]
    text: str | None
    confidence: float
    status: str

    def __post_init__(self) -> None:
        if not isinstance(self.region_id, str) or not self.region_id.strip():
            raise InvalidGeometryInput("OCR region prediction requires a non-empty region_id")
        _validate_bbox(self.bbox, f"OCR region {self.region_id}")
        if not isinstance(self.confidence, (int, float)) or isinstance(self.confidence, bool):
            raise InvalidGeometryInput(f"OCR region {self.region_id} confidence must be numeric")
        if not math.isfinite(float(self.confidence)) or not 0.0 <= float(self.confidence) <= 1.0:
            raise InvalidGeometryInput(f"OCR region {self.region_id} confidence must be between zero and one")
        if not isinstance(self.status, str) or self.status not in _OCR_STATUSES:
            raise InvalidGeometryInput(f"OCR region {self.region_id} has unsupported status: {self.status!r}")
        if self.status == "resolved":
            if not isinstance(self.text, str) or not self.text.strip():
                raise InvalidGeometryInput(f"Resolved OCR region {self.region_id} requires non-empty text")
            if len(self.text) > 128 or any(ord(character) < 32 for character in self.text):
                raise InvalidGeometryInput(f"OCR region {self.region_id} contains invalid text")
            if self.confidence <= 0.0:
                raise InvalidGeometryInput(f"Resolved OCR region {self.region_id} requires positive confidence")
        elif self.text is not None or self.confidence != 0.0:
            raise InvalidGeometryInput(
                f"Unresolved OCR region {self.region_id} requires null text and zero confidence"
            )
        object.__setattr__(self, "bbox", tuple(float(value) for value in self.bbox))
        object.__setattr__(self, "confidence", float(self.confidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "bbox": list(self.bbox),
            "text": self.text,
            "confidence": float(self.confidence),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OCRRegionPrediction":
        _check_fields(
            data,
            {"region_id", "bbox", "text", "confidence", "status"},
            "OCR region prediction",
        )
        return cls(
            region_id=data["region_id"],
            bbox=_bbox(data["bbox"], f"OCR region {data['region_id']}"),
            text=data["text"],
            confidence=data["confidence"],
            status=data["status"],
        )


@dataclass(frozen=True)
class OCRImagePrediction:
    image_sha256: str
    region_set_checksum: str
    backend: str
    model: str
    model_revision: str
    regions: tuple[OCRRegionPrediction, ...]
    schema_version: str = OCR_PREDICTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_sha256(self.image_sha256, "OCR image SHA-256")
        _require_sha256(self.region_set_checksum, "OCR region-set checksum")
        for value, name in (
            (self.backend, "OCR backend"),
            (self.model, "OCR model"),
            (self.model_revision, "OCR model revision"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise InvalidGeometryInput(f"{name} must be non-empty")
        if self.schema_version != OCR_PREDICTION_SCHEMA_VERSION:
            raise InvalidGeometryInput(f"Unsupported OCR prediction schema version: {self.schema_version}")
        if not isinstance(self.regions, tuple) or any(
            not isinstance(region, OCRRegionPrediction) for region in self.regions
        ):
            raise InvalidGeometryInput("OCR image prediction regions must be OCRRegionPrediction tuples")
        region_ids = [region.region_id for region in self.regions]
        if len(region_ids) != len(set(region_ids)):
            raise InvalidGeometryInput("OCR image prediction contains duplicate region IDs")
        if self.region_set_checksum != _region_set_checksum(self.regions):
            raise InvalidGeometryInput("OCR region-set checksum does not match region IDs and boxes")

    @classmethod
    def create(
        cls,
        image_path: str | Path,
        *,
        backend: str,
        model: str,
        model_revision: str,
        regions: Sequence[OCRRegionPrediction],
    ) -> "OCRImagePrediction":
        normalized = tuple(regions)
        return cls(
            image_sha256=image_sha256(image_path),
            region_set_checksum=_region_set_checksum(normalized),
            backend=backend,
            model=model,
            model_revision=model_revision,
            regions=normalized,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "image_sha256": self.image_sha256,
            "region_set_checksum": self.region_set_checksum,
            "backend": self.backend,
            "model": self.model,
            "model_revision": self.model_revision,
            "regions": [region.to_dict() for region in self.regions],
        }

    def checksum(self) -> str:
        return stable_checksum(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["checksum"] = stable_checksum(payload)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OCRImagePrediction":
        _check_fields(
            data,
            {
                "schema_version",
                "image_sha256",
                "region_set_checksum",
                "backend",
                "model",
                "model_revision",
                "regions",
                "checksum",
            },
            "OCR image prediction",
        )
        payload = {key: value for key, value in data.items() if key != "checksum"}
        if data["checksum"] != stable_checksum(payload):
            raise InvalidGeometryInput("OCR image prediction checksum does not match payload")
        regions = data["regions"]
        if not isinstance(regions, Sequence) or isinstance(regions, (str, bytes)):
            raise InvalidGeometryInput("OCR image prediction regions must be a sequence")
        return cls(
            image_sha256=data["image_sha256"],
            region_set_checksum=data["region_set_checksum"],
            backend=data["backend"],
            model=data["model"],
            model_revision=data["model_revision"],
            regions=tuple(OCRRegionPrediction.from_dict(region) for region in regions),
            schema_version=data["schema_version"],
        )


class RecordedLabelRecognizer:
    """Replay OCR results only when image bytes and detector regions match."""

    def __init__(self, predictions: Sequence[OCRImagePrediction]):
        indexed = {}
        for prediction in predictions:
            if prediction.image_sha256 in indexed:
                raise InvalidGeometryInput(
                    f"Duplicate OCR prediction for image SHA-256: {prediction.image_sha256}"
                )
            indexed[prediction.image_sha256] = prediction
        if not indexed:
            raise InvalidGeometryInput("Recorded OCR recognizer requires at least one image prediction")
        self._predictions = indexed

    @property
    def image_hashes(self) -> frozenset[str]:
        return frozenset(self._predictions)

    def configuration(self) -> dict[str, Any]:
        predictions = tuple(self._predictions.values())
        return {
            "kind": "recorded_ocr_regions",
            "schema_version": OCR_PREDICTION_SCHEMA_VERSION,
            "num_images": len(predictions),
            "backends": sorted({prediction.backend for prediction in predictions}),
            "models": sorted({prediction.model for prediction in predictions}),
            "model_revisions": sorted({prediction.model_revision for prediction in predictions}),
            "prediction_set_checksum": stable_checksum(
                sorted(prediction.checksum() for prediction in predictions)
            ),
        }

    def recognize(
        self,
        image_path: str,
        text_regions: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        digest = image_sha256(image_path)
        prediction = self._predictions.get(digest)
        if prediction is None:
            raise InvalidGeometryInput(f"No recorded OCR prediction for image SHA-256 {digest}")
        observed_regions = _normalize_observed_regions(text_regions)
        if prediction.region_set_checksum != _region_set_checksum(observed_regions):
            raise InvalidGeometryInput(
                "Recorded OCR prediction region set does not match current visual detector output"
            )
        predictions_by_id = {region.region_id: region for region in prediction.regions}
        return [
            {
                "id": region.region_id,
                "bbox": list(region.bbox),
                "text": region.text,
                "confidence": region.confidence,
                "status": region.status,
                "provenance": {
                    "method": "recorded_ocr_region",
                    "image_sha256": digest,
                    "region_set_checksum": prediction.region_set_checksum,
                    "backend": prediction.backend,
                    "model": prediction.model,
                    "model_revision": prediction.model_revision,
                    "prediction_checksum": prediction.checksum(),
                },
            }
            for observed in observed_regions
            for region in (predictions_by_id[observed.region_id],)
        ]


class TesseractLabelRecognizer:
    """Run optional Tesseract OCR independently on detected text regions."""

    def __init__(
        self,
        *,
        language: str = "eng",
        config: str = "--psm 10",
        padding_px: int = 2,
    ) -> None:
        if not isinstance(language, str) or not language.strip():
            raise InvalidGeometryInput("Tesseract language must be non-empty")
        if not isinstance(config, str):
            raise InvalidGeometryInput("Tesseract config must be a string")
        if isinstance(padding_px, bool) or not isinstance(padding_px, int) or padding_px < 0:
            raise InvalidGeometryInput("Tesseract padding_px must be a non-negative integer")
        self.language = language.strip()
        self.config = config
        self.padding_px = padding_px

    def configuration(self) -> dict[str, Any]:
        return {
            "kind": "tesseract_regions",
            "backend": "pytesseract",
            "language": self.language,
            "config": self.config,
            "padding_px": self.padding_px,
        }

    def recognize(
        self,
        image_path: str,
        text_regions: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        try:
            import pytesseract
        except ImportError as exc:
            raise InvalidGeometryInput(
                "TesseractLabelRecognizer requires the optional 'pytesseract' package and a Tesseract binary"
            ) from exc

        source = Path(image_path)
        if not source.is_file():
            raise InvalidGeometryInput(f"OCR source image does not exist: {source}")
        try:
            from PIL import Image

            with Image.open(source) as image:
                grayscale = image.convert("L")
                return [
                    self._recognize_region(pytesseract, grayscale, region)
                    for region in text_regions
                ]
        except InvalidGeometryInput:
            raise
        except OSError as exc:
            raise InvalidGeometryInput(f"Tesseract OCR could not read image {source}: {exc}") from exc

    def _recognize_region(
        self,
        pytesseract: Any,
        image: Any,
        region: Mapping[str, Any],
    ) -> dict[str, Any]:
        region_id = region.get("id")
        if not isinstance(region_id, str) or not region_id.strip():
            raise InvalidGeometryInput("Visual text region requires a non-empty id")
        bbox = _bbox(region.get("bbox"), f"Visual text region {region_id}")
        left, top, right, bottom = _crop_bounds(bbox, image.size, self.padding_px)
        if right <= left or bottom <= top:
            return _unresolved_ocr_result(region_id, bbox, self.configuration())
        try:
            data = pytesseract.image_to_data(
                image.crop((left, top, right, bottom)),
                lang=self.language,
                config=self.config,
                output_type=pytesseract.Output.DICT,
            )
        except RuntimeError as exc:
            raise InvalidGeometryInput(
                f"Tesseract OCR failed for visual text region {region_id}: {exc}"
            ) from exc
        if not isinstance(data, Mapping):
            raise InvalidGeometryInput(f"Tesseract OCR returned invalid data for visual text region {region_id}")
        tokens = data.get("text") or ()
        confidences = data.get("conf") or ()
        if not isinstance(tokens, Sequence) or isinstance(tokens, (str, bytes)):
            raise InvalidGeometryInput(f"Tesseract OCR text output is invalid for visual text region {region_id}")
        recognized = []
        scores = []
        for index, token in enumerate(tokens):
            normalized = str(token).strip()
            if not normalized:
                continue
            recognized.append(normalized)
            if index < len(confidences):
                try:
                    confidence = float(confidences[index])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(confidence) and confidence > 0.0:
                    scores.append(min(100.0, confidence) / 100.0)
        text = "".join(recognized)
        confidence = sum(scores) / len(scores) if scores else 0.0
        if not text or confidence <= 0.0:
            return _unresolved_ocr_result(region_id, bbox, self.configuration())
        return {
            "id": region_id,
            "bbox": list(bbox),
            "text": text,
            "confidence": round(confidence, 6),
            "status": "resolved",
            "provenance": {
                "method": "tesseract_region_crop",
                **self.configuration(),
            },
        }


def _unresolved_ocr_result(
    region_id: str,
    bbox: Sequence[float],
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "id": region_id,
        "bbox": list(bbox),
        "text": None,
        "confidence": 0.0,
        "status": "unresolved",
        "provenance": {
            "method": "tesseract_region_crop",
            **dict(configuration),
        },
    }


def _crop_bounds(
    bbox: Sequence[float],
    image_size: tuple[int, int],
    padding_px: int,
) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    image_width, image_height = image_size
    left = max(0, int(math.floor(x - padding_px)))
    top = max(0, int(math.floor(y - padding_px)))
    right = min(image_width, int(math.ceil(x + width + padding_px)))
    bottom = min(image_height, int(math.ceil(y + height + padding_px)))
    return left, top, right, bottom


def load_ocr_predictions(path: str | Path) -> RecordedLabelRecognizer:
    source = Path(path)
    if not source.is_file():
        raise InvalidGeometryInput(f"OCR prediction JSONL does not exist: {source}")
    predictions = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InvalidGeometryInput(
                    f"Invalid OCR prediction JSON at {source}:{line_number}: {exc.msg}"
                ) from exc
            try:
                predictions.append(OCRImagePrediction.from_dict(payload))
            except (TypeError, ValueError) as exc:
                raise InvalidGeometryInput(
                    f"Invalid OCR prediction at {source}:{line_number}: {exc}"
                ) from exc
    return RecordedLabelRecognizer(predictions)


def image_sha256(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise InvalidGeometryInput(f"OCR source image does not exist: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_observed_regions(
    text_regions: Sequence[Mapping[str, Any]],
) -> tuple[OCRRegionPrediction, ...]:
    normalized = []
    for region in text_regions:
        if not isinstance(region, Mapping):
            raise InvalidGeometryInput("Visual text regions must be mappings")
        region_id = region.get("id")
        if not isinstance(region_id, str) or not region_id.strip():
            raise InvalidGeometryInput("Visual text region requires a non-empty id")
        normalized.append(
            OCRRegionPrediction(
                region_id=region_id,
                bbox=_bbox(region.get("bbox"), f"Visual text region {region_id}"),
                text=None,
                confidence=0.0,
                status="unresolved",
            )
        )
    if len({region.region_id for region in normalized}) != len(normalized):
        raise InvalidGeometryInput("Visual text regions contain duplicate IDs")
    return tuple(normalized)


def _region_set_checksum(regions: Sequence[OCRRegionPrediction]) -> str:
    return stable_checksum(
        sorted(
            ({"region_id": region.region_id, "bbox": list(region.bbox)} for region in regions),
            key=lambda item: item["region_id"],
        )
    )


def _bbox(value: Any, context: str) -> tuple[float, float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise InvalidGeometryInput(f"{context} bbox must contain four values")
    if any(isinstance(item, bool) for item in value):
        raise InvalidGeometryInput(f"{context} bbox must be numeric")
    try:
        bbox = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise InvalidGeometryInput(f"{context} bbox must be numeric") from exc
    _validate_bbox(bbox, context)
    return bbox


def _validate_bbox(bbox: Sequence[float], context: str) -> None:
    if len(bbox) != 4 or not all(math.isfinite(float(value)) for value in bbox):
        raise InvalidGeometryInput(f"{context} bbox must contain four finite values")
    if float(bbox[2]) <= 0.0 or float(bbox[3]) <= 0.0:
        raise InvalidGeometryInput(f"{context} bbox width and height must be positive")


def _check_fields(data: Mapping[str, Any], required: set[str], context: str) -> None:
    if not isinstance(data, Mapping):
        raise InvalidGeometryInput(f"{context} must be a mapping")
    missing = sorted(required - set(data))
    unknown = sorted(set(data) - required)
    if missing:
        raise InvalidGeometryInput(f"{context} is missing fields: {missing}")
    if unknown:
        raise InvalidGeometryInput(f"{context} has unknown fields: {unknown}")


def _require_sha256(value: str, context: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise InvalidGeometryInput(f"{context} must be a lowercase SHA-256 digest")
