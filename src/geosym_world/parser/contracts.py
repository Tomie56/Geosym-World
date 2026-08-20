"""Contracts for parser outputs before hypothesis admission."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from geosym_world.errors import InvalidGeometryInput
from geosym_world.schema import AdmissionStatus, GeometryRecord


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
    if parsed.metadata.get("admission_status") not in {None, "proposed"}:
        raise InvalidGeometryInput(f"Parser record {parsed.id} has invalid admission_status")
    return parsed


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
