"""Strict contracts for replaying Geoparsing/GDP-29K predictions."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from geosym_world.errors import InvalidGeometryInput
from geosym_world.schema.canonical import stable_checksum

from .contracts import ParserInput, validate_parser_input
from .ocr import image_sha256


GEOPARSING_PREDICTION_SCHEMA_VERSION = "geosym.geoparsing_prediction.v1"
GEOPARSING_ADAPTER_VERSION = "geosym.geoparsing_adapter.v0"
_FORMAL_FIELDS = frozenset({"points", "lines", "circles", "planes", "structure", "semantics"})


@dataclass(frozen=True)
class GeoparsingFormalDescription:
    points: tuple[str, ...]
    lines: tuple[str, ...]
    circles: tuple[str, ...]
    planes: tuple[str, ...] = ()
    structure: tuple[str, ...] = ()
    semantics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in _FORMAL_FIELDS:
            values = getattr(self, field_name)
            if not isinstance(values, tuple):
                raise InvalidGeometryInput(f"Geoparsing field {field_name} must be a tuple")
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise InvalidGeometryInput(f"Geoparsing field {field_name} contains an empty item")
            if len(set(values)) != len(values):
                raise InvalidGeometryInput(f"Geoparsing field {field_name} contains duplicate items")
        if not self.points:
            raise InvalidGeometryInput("Geoparsing formal description requires at least one point")

    def to_dict(self) -> dict[str, list[str]]:
        return {field_name: list(getattr(self, field_name)) for field_name in sorted(_FORMAL_FIELDS)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GeoparsingFormalDescription":
        if not isinstance(data, Mapping):
            raise InvalidGeometryInput("Geoparsing formal description must be a mapping")
        unknown = sorted(set(data) - _FORMAL_FIELDS)
        missing = sorted({"points", "lines", "circles", "semantics"} - set(data))
        if unknown:
            raise InvalidGeometryInput(f"Geoparsing formal description has unknown fields: {unknown}")
        if missing:
            raise InvalidGeometryInput(f"Geoparsing formal description is missing fields: {missing}")
        return cls(
            points=_string_items(data["points"], "points"),
            lines=_string_items(data["lines"], "lines"),
            circles=_string_items(data["circles"], "circles"),
            planes=_string_items(data.get("planes") or (), "planes"),
            structure=_string_items(data.get("structure") or (), "structure"),
            semantics=_string_items(data["semantics"], "semantics"),
        )


@dataclass(frozen=True)
class GeoparsingPrediction:
    id: str
    parser_input_checksum: str
    image_sha256: str
    backend: str
    model: str
    model_revision: str
    prompt_version: str
    formal_description: GeoparsingFormalDescription
    confidence: float | None = None
    request_id: str | None = None
    decoding: Mapping[str, Any] = field(default_factory=dict)
    usage: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = GEOPARSING_PREDICTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for value, name in (
            (self.id, "Geoparsing prediction id"),
            (self.backend, "Geoparsing backend"),
            (self.model, "Geoparsing model"),
            (self.model_revision, "Geoparsing model revision"),
            (self.prompt_version, "Geoparsing prompt version"),
        ):
            _require_text(value, name)
        _require_sha256(self.parser_input_checksum, "Geoparsing ParserInput checksum")
        _require_sha256(self.image_sha256, "Geoparsing image SHA-256")
        if self.schema_version != GEOPARSING_PREDICTION_SCHEMA_VERSION:
            raise InvalidGeometryInput(f"Unsupported Geoparsing prediction schema: {self.schema_version}")
        if not isinstance(self.formal_description, GeoparsingFormalDescription):
            raise InvalidGeometryInput("Geoparsing prediction requires a typed formal description")
        if self.confidence is not None:
            if (
                isinstance(self.confidence, bool)
                or not isinstance(self.confidence, (int, float))
                or not math.isfinite(float(self.confidence))
                or not 0.0 <= float(self.confidence) <= 1.0
            ):
                raise InvalidGeometryInput("Geoparsing confidence must be null or a finite value in [0, 1]")
        if self.request_id is not None:
            _require_text(self.request_id, "Geoparsing request id")
        _require_json_mapping(self.decoding, "Geoparsing decoding")
        _require_json_mapping(self.usage, "Geoparsing usage")

    @classmethod
    def create(
        cls,
        parser_input: ParserInput | Mapping[str, Any],
        *,
        backend: str,
        model: str,
        model_revision: str,
        prompt_version: str,
        formal_description: GeoparsingFormalDescription | Mapping[str, Any],
        confidence: float | None = None,
        request_id: str | None = None,
        decoding: Mapping[str, Any] | None = None,
        usage: Mapping[str, Any] | None = None,
    ) -> "GeoparsingPrediction":
        public_input = validate_parser_input(parser_input)
        description = (
            formal_description
            if isinstance(formal_description, GeoparsingFormalDescription)
            else GeoparsingFormalDescription.from_dict(formal_description)
        )
        return cls(
            id=public_input.id,
            parser_input_checksum=stable_checksum(public_input.to_dict()),
            image_sha256=image_sha256(public_input.image),
            backend=backend,
            model=model,
            model_revision=model_revision,
            prompt_version=prompt_version,
            formal_description=description,
            confidence=confidence,
            request_id=request_id,
            decoding=dict(decoding or {}),
            usage=dict(usage or {}),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "parser_input_checksum": self.parser_input_checksum,
            "image_sha256": self.image_sha256,
            "backend": self.backend,
            "model": self.model,
            "model_revision": self.model_revision,
            "prompt_version": self.prompt_version,
            "formal_description": self.formal_description.to_dict(),
            "confidence": self.confidence,
            "request_id": self.request_id,
            "decoding": dict(self.decoding),
            "usage": dict(self.usage),
        }

    def checksum(self) -> str:
        return stable_checksum(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["checksum"] = stable_checksum(payload)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GeoparsingPrediction":
        required = {
            "schema_version",
            "id",
            "parser_input_checksum",
            "image_sha256",
            "backend",
            "model",
            "model_revision",
            "prompt_version",
            "formal_description",
            "confidence",
            "request_id",
            "decoding",
            "usage",
            "checksum",
        }
        _check_exact_fields(data, required, "Geoparsing prediction")
        payload = {key: value for key, value in data.items() if key != "checksum"}
        if data["checksum"] != stable_checksum(payload):
            raise InvalidGeometryInput("Geoparsing prediction checksum does not match payload")
        decoding = data["decoding"]
        usage = data["usage"]
        _require_json_mapping(decoding, "Geoparsing decoding")
        _require_json_mapping(usage, "Geoparsing usage")
        return cls(
            id=data["id"],
            parser_input_checksum=data["parser_input_checksum"],
            image_sha256=data["image_sha256"],
            backend=data["backend"],
            model=data["model"],
            model_revision=data["model_revision"],
            prompt_version=data["prompt_version"],
            formal_description=GeoparsingFormalDescription.from_dict(data["formal_description"]),
            confidence=data["confidence"],
            request_id=data["request_id"],
            decoding=dict(decoding),
            usage=dict(usage),
            schema_version=data["schema_version"],
        )


def load_geoparsing_predictions(path: str | Path) -> tuple[GeoparsingPrediction, ...]:
    source = Path(path)
    if not source.is_file():
        raise InvalidGeometryInput(f"Geoparsing prediction JSONL does not exist: {source}")
    predictions = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InvalidGeometryInput(
                    f"Invalid Geoparsing prediction JSON at {source}:{line_number}: {exc.msg}"
                ) from exc
            try:
                predictions.append(GeoparsingPrediction.from_dict(payload))
            except (TypeError, ValueError) as exc:
                raise InvalidGeometryInput(
                    f"Invalid Geoparsing prediction at {source}:{line_number}: {exc}"
                ) from exc
    if not predictions:
        raise InvalidGeometryInput(f"Geoparsing prediction JSONL is empty: {source}")
    ids = [prediction.id for prediction in predictions]
    if len(ids) != len(set(ids)):
        raise InvalidGeometryInput("Geoparsing prediction JSONL contains duplicate ids")
    return tuple(predictions)


def _string_items(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise InvalidGeometryInput(f"Geoparsing field {field_name} must be a sequence")
    return tuple(item.strip() if isinstance(item, str) else item for item in value)


def _require_text(value: Any, context: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise InvalidGeometryInput(f"{context} must be non-empty")


def _require_sha256(value: Any, context: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise InvalidGeometryInput(f"{context} must be a lowercase SHA-256 digest")


def _require_json_mapping(value: Any, context: str) -> None:
    if not isinstance(value, Mapping):
        raise InvalidGeometryInput(f"{context} must be a mapping")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise InvalidGeometryInput(f"{context} must contain finite JSON values") from exc


def _check_exact_fields(data: Mapping[str, Any], required: set[str], context: str) -> None:
    if not isinstance(data, Mapping):
        raise InvalidGeometryInput(f"{context} must be a mapping")
    missing = sorted(required - set(data))
    unknown = sorted(set(data) - required)
    if missing:
        raise InvalidGeometryInput(f"{context} is missing fields: {missing}")
    if unknown:
        raise InvalidGeometryInput(f"{context} has unknown fields: {unknown}")
