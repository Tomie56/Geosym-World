"""Public parser inputs and proposed-output contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from geosym_world.errors import InvalidGeometryInput
from geosym_world.schema.canonical import stable_checksum
from geosym_world.schema import (
    AdmissionStatus,
    GeometryHypothesis,
    GeometryRecord,
    PARSE_GRAPH_HYPOTHESIS_KIND,
    ParseGraph,
)


PARSER_INPUT_SCHEMA_VERSION = "1.0"
_PARSER_INPUT_KEYS = frozenset({"schema_version", "id", "image", "question"})


@dataclass(frozen=True)
class ParserInput:
    """The complete runtime input visible to a geometry parser."""

    id: str
    image: str
    question: str
    schema_version: str = PARSER_INPUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, str):
            raise InvalidGeometryInput("Parser input schema version must be a string")
        if self.schema_version != PARSER_INPUT_SCHEMA_VERSION:
            raise InvalidGeometryInput(f"Unsupported parser input schema version: {self.schema_version}")
        if not isinstance(self.id, str) or not self.id.strip():
            raise InvalidGeometryInput("Parser input id must be non-empty")
        if not isinstance(self.image, str) or not self.image.strip():
            raise InvalidGeometryInput(f"Parser input {self.id} requires one image path")
        if not isinstance(self.question, str):
            raise InvalidGeometryInput(f"Parser input {self.id} question must be a string")

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "image": self.image,
            "question": self.question,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParserInput":
        keys = set(data)
        unknown = sorted(keys - _PARSER_INPUT_KEYS)
        missing = sorted(_PARSER_INPUT_KEYS - keys)
        if unknown:
            raise InvalidGeometryInput(f"Parser input exposes unsupported fields: {', '.join(unknown)}")
        if missing:
            raise InvalidGeometryInput(f"Parser input is missing fields: {', '.join(missing)}")
        return cls(
            id=data["id"],
            image=data["image"],
            question=data["question"],
            schema_version=data["schema_version"],
        )


def validate_parser_input(data: ParserInput | Mapping[str, Any]) -> ParserInput:
    """Reject any runtime field beyond image, question, and record identity."""

    return data if isinstance(data, ParserInput) else ParserInput.from_dict(data)


def validate_proposed_parser_record(record: GeometryRecord | Mapping[str, Any]) -> GeometryRecord:
    """Require parser output to remain a proposed, non-persistent candidate."""

    parsed = record if isinstance(record, GeometryRecord) else GeometryRecord.from_dict(record)
    if parsed.state is not None:
        raise InvalidGeometryInput(f"Parser record {parsed.id} must not contain admitted state")
    if not parsed.hypotheses:
        raise InvalidGeometryInput(f"Parser record {parsed.id} must contain at least one hypothesis")
    for hypothesis in parsed.hypotheses:
        if hypothesis.status is not AdmissionStatus.PROPOSED:
            raise InvalidGeometryInput(f"Parser hypothesis {hypothesis.id} must remain proposed")
        if not hypothesis.provenance:
            raise InvalidGeometryInput(f"Parser hypothesis {hypothesis.id} requires provenance")
        if hypothesis.metadata.get("admission_required") is not True:
            raise InvalidGeometryInput(f"Parser hypothesis {hypothesis.id} must declare admission_required=true")
        if hypothesis.kind == PARSE_GRAPH_HYPOTHESIS_KIND:
            validate_parse_graph_hypothesis(hypothesis)
    if parsed.metadata.get("admission_status") not in {None, "proposed"}:
        raise InvalidGeometryInput(f"Parser record {parsed.id} has invalid admission_status")
    return parsed


def validate_parse_graph_hypothesis(hypothesis: GeometryHypothesis) -> ParseGraph:
    """Validate and recover a typed parse graph without admitting it."""

    if hypothesis.kind != PARSE_GRAPH_HYPOTHESIS_KIND:
        raise InvalidGeometryInput(
            f"Hypothesis {hypothesis.id} has kind {hypothesis.kind!r}, expected {PARSE_GRAPH_HYPOTHESIS_KIND!r}"
        )
    if hypothesis.status is not AdmissionStatus.PROPOSED:
        raise InvalidGeometryInput(f"Parse graph hypothesis {hypothesis.id} must remain proposed")
    graph = ParseGraph.from_dict(hypothesis.payload)
    candidate_checksum = hypothesis.metadata.get("candidate_checksum")
    if candidate_checksum is not None and candidate_checksum != stable_checksum(hypothesis.payload):
        raise InvalidGeometryInput(f"Parse graph hypothesis {hypothesis.id} checksum does not match graph")
    return graph


def parser_record_audit(record: GeometryRecord | Mapping[str, Any]) -> dict[str, Any]:
    """Return a structured contract audit without admitting the candidate."""

    try:
        parsed = validate_proposed_parser_record(record)
    except InvalidGeometryInput as exc:
        return {"valid": False, "error": str(exc)}
    return {
        "valid": True,
        "record_id": parsed.id,
        "num_hypotheses": len(parsed.hypotheses),
        "hypothesis_ids": [hypothesis.id for hypothesis in parsed.hypotheses],
        "record_checksum": parsed.checksum(),
        "state_present": False,
        "admission_required": True,
    }
