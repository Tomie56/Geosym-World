"""Convert parser candidates into non-admitted geometry records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import hashlib
import re

from geosym_world.schema import EvidenceKind, GeometryHypothesis, GeometryRecord, Provenance
from geosym_world.schema.canonical import stable_checksum
from geosym_world.parser.contracts import validate_proposed_parser_record


class HypothesisRecordAdapter:
    """Persist parser output without promoting a candidate to formal state."""

    def convert(
        self,
        *,
        sample_id: str,
        dataset_name: str,
        parser_name: str,
        backend: str,
        candidate_kind: str,
        candidate_payload: Mapping[str, Any],
        confidence: float,
        verification: Mapping[str, Any],
        image_path: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> GeometryRecord:
        record_id = self._stable_id(sample_id, prefix="record")
        hypothesis_id = self._stable_id(f"{sample_id}_{candidate_kind}", prefix="hyp")
        hypothesis = GeometryHypothesis(
            id=hypothesis_id,
            kind=candidate_kind,
            payload=dict(candidate_payload),
            confidence=max(0.0, min(1.0, float(confidence))),
            provenance=(
                Provenance(
                    source_id=record_id,
                    kind=EvidenceKind.PARSER,
                    event_id=f"{parser_name}:candidate",
                    metadata={"backend": backend},
                ),
            ),
            metadata={
                "verification": dict(verification),
                "admission_required": True,
                "candidate_checksum": stable_checksum(candidate_payload),
            },
        )
        record = GeometryRecord(
            id=record_id,
            source={
                "type": "parser",
                "dataset": dataset_name,
                "original_id": sample_id,
                "image_path": image_path,
            },
            state=None,
            hypotheses=[hypothesis],
            metadata={
                "parser": parser_name,
                "backend": backend,
                "admission_status": "proposed",
                "admission_required": True,
                **dict(metadata or {}),
            },
        )
        return validate_proposed_parser_record(record)

    def _stable_id(self, value: str, *, prefix: str) -> str:
        raw = str(value).strip()
        normalized = re.sub(r"[^A-Za-z0-9_.:-]+", "_", raw).strip("_")
        if not normalized:
            normalized = "sample"
        if normalized != raw:
            digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
            normalized = f"{normalized}_{digest}"
        return f"{prefix}_{normalized}"
