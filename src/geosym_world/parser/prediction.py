"""Provider-neutral contract for externally generated ParseGraphs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from geosym_world.errors import InvalidGeometryInput
from geosym_world.schema import ParseGraph, ParseOrigin
from geosym_world.schema.canonical import stable_checksum

from .contracts import ParserInput, validate_parser_input


EXTERNAL_PARSE_GRAPH_SCHEMA_VERSION = "geosym.external_parse_graph_prediction.v1"
_FORBIDDEN_SCORE_KINDS = frozenset(
    {
        "human_annotation",
        "verified_public_given",
        "verified_render_binding",
        "verified_render_label",
        "verified_synthetic_state",
    }
)
_FORBIDDEN_METADATA_KEYS = frozenset(
    {
        "answer",
        "construction_trace",
        "derivation",
        "formal_state",
        "ground_truth",
        "private_supervision",
        "reference_kind",
        "scene",
        "solution",
        "solution_text",
        "source_manifest",
    }
)
_SENSITIVE_AUDIT_KEYS = frozenset({"access_token", "api_key", "authorization", "secret"})


@dataclass(frozen=True)
class ExternalParseGraphPrediction:
    id: str
    parser_input_checksum: str
    backend: str
    model: str
    model_revision: str
    prompt_version: str
    parse_graph: ParseGraph
    request_id: str | None = None
    decoding: Mapping[str, Any] = field(default_factory=dict)
    usage: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = EXTERNAL_PARSE_GRAPH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.id, "External prediction id")
        _require_checksum(self.parser_input_checksum, "External prediction input checksum")
        for value, name in (
            (self.backend, "External prediction backend"),
            (self.model, "External prediction model"),
            (self.model_revision, "External prediction model revision"),
            (self.prompt_version, "External prediction prompt version"),
        ):
            _require_text(value, name)
        if self.request_id is not None:
            _require_text(self.request_id, "External prediction request id")
        if not isinstance(self.decoding, Mapping) or not isinstance(self.usage, Mapping):
            raise InvalidGeometryInput("External prediction decoding and usage must be mappings")
        _validate_audit_mapping(self.decoding, "External prediction decoding")
        _validate_audit_mapping(self.usage, "External prediction usage")
        if self.schema_version != EXTERNAL_PARSE_GRAPH_SCHEMA_VERSION:
            raise InvalidGeometryInput(f"Unsupported external ParseGraph schema version: {self.schema_version}")
        _validate_external_graph(self.parse_graph, self.id)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "parser_input_checksum": self.parser_input_checksum,
            "backend": self.backend,
            "model": self.model,
            "model_revision": self.model_revision,
            "prompt_version": self.prompt_version,
            "request_id": self.request_id,
            "decoding": dict(self.decoding),
            "usage": dict(self.usage),
            "parse_graph": self.parse_graph.to_dict(),
        }

    def checksum(self) -> str:
        return stable_checksum(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["checksum"] = stable_checksum(payload)
        return payload

    @classmethod
    def create(
        cls,
        parser_input: ParserInput | Mapping[str, Any],
        *,
        backend: str,
        model: str,
        model_revision: str,
        prompt_version: str,
        parse_graph: ParseGraph,
        request_id: str | None = None,
        decoding: Mapping[str, Any] | None = None,
        usage: Mapping[str, Any] | None = None,
    ) -> "ExternalParseGraphPrediction":
        public_input = validate_parser_input(parser_input)
        return cls(
            id=public_input.id,
            parser_input_checksum=stable_checksum(public_input.to_dict()),
            backend=backend,
            model=model,
            model_revision=model_revision,
            prompt_version=prompt_version,
            parse_graph=parse_graph,
            request_id=request_id,
            decoding=dict(decoding or {}),
            usage=dict(usage or {}),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExternalParseGraphPrediction":
        required = {
            "schema_version", "id", "parser_input_checksum", "backend", "model",
            "model_revision", "prompt_version", "request_id", "decoding", "usage",
            "parse_graph", "checksum",
        }
        if not isinstance(data, Mapping):
            raise InvalidGeometryInput("External ParseGraph prediction must be a mapping")
        missing = sorted(required - set(data))
        unknown = sorted(set(data) - required)
        if missing:
            raise InvalidGeometryInput(f"External ParseGraph prediction is missing fields: {missing}")
        if unknown:
            raise InvalidGeometryInput(f"External ParseGraph prediction has unknown fields: {unknown}")
        payload = {key: value for key, value in data.items() if key != "checksum"}
        if data["checksum"] != stable_checksum(payload):
            raise InvalidGeometryInput("External ParseGraph prediction checksum does not match payload")
        return cls(
            id=str(data["id"]),
            parser_input_checksum=str(data["parser_input_checksum"]),
            backend=str(data["backend"]),
            model=str(data["model"]),
            model_revision=str(data["model_revision"]),
            prompt_version=str(data["prompt_version"]),
            parse_graph=ParseGraph.from_dict(data["parse_graph"]),
            request_id=str(data["request_id"]) if data["request_id"] is not None else None,
            decoding=dict(data["decoding"]),
            usage=dict(data["usage"]),
            schema_version=str(data["schema_version"]),
        )


def _validate_external_graph(graph: ParseGraph, sample_id: str) -> None:
    if graph.source_id != sample_id:
        raise InvalidGeometryInput("External ParseGraph source_id must equal the public ParserInput id")
    _reject_forbidden_metadata(graph.metadata, "ParseGraph metadata")
    for evidence in graph.evidence:
        _reject_forbidden_metadata(evidence.metadata, f"Evidence {evidence.id} metadata")
    candidates = (*graph.objects, *graph.relations, *graph.quantities, *graph.regions, *graph.targets)
    for candidate in candidates:
        _reject_forbidden_metadata(candidate.metadata, f"Candidate {candidate.id} metadata")
        if candidate.origin in {ParseOrigin.NATIVE, ParseOrigin.COMPILED}:
            raise InvalidGeometryInput(
                f"External model candidate {candidate.id} cannot use trusted origin={candidate.origin.value}"
            )
        if candidate.score_kind in _FORBIDDEN_SCORE_KINDS:
            raise InvalidGeometryInput(
                f"External model candidate {candidate.id} cannot use trusted score_kind={candidate.score_kind!r}"
            )
    allowed_sources = {sample_id, f"{sample_id}:question"}
    unexpected = sorted({item.source_id for item in graph.evidence} - allowed_sources)
    if unexpected:
        raise InvalidGeometryInput(f"External ParseGraph evidence uses non-public source IDs: {unexpected}")


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise InvalidGeometryInput(f"{name} must be non-empty")


def _require_checksum(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise InvalidGeometryInput(f"{name} must be a lowercase SHA-256 digest")


def _reject_forbidden_metadata(value: Mapping[str, Any], context: str) -> None:
    for key, item in value.items():
        normalized = str(key).strip().lower().replace("-", "_")
        if normalized in _FORBIDDEN_METADATA_KEYS:
            raise InvalidGeometryInput(f"{context} contains forbidden private field: {key}")
        if isinstance(item, Mapping):
            _reject_forbidden_metadata(item, context)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            for nested in item:
                if isinstance(nested, Mapping):
                    _reject_forbidden_metadata(nested, context)


def _validate_audit_mapping(value: Mapping[str, Any], context: str) -> None:
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise InvalidGeometryInput(f"{context} keys must be non-empty strings")
        normalized = key.strip().lower().replace("-", "_")
        if normalized in _SENSITIVE_AUDIT_KEYS:
            raise InvalidGeometryInput(f"{context} cannot contain sensitive field: {key}")
        if isinstance(item, Mapping):
            _validate_audit_mapping(item, context)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            for nested in item:
                if isinstance(nested, Mapping):
                    _validate_audit_mapping(nested, context)
