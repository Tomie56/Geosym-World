"""Offline syntax-coverage audit for Geoparsing/GDP formal descriptions."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from geosym_world.errors import InvalidGeometryInput
from geosym_world.parser.geoparsing import compile_geoparsing_formal_description
from geosym_world.parser.geoparsing_contracts import GeoparsingFormalDescription
from geosym_world.schema.canonical import stable_checksum


GEOPARSING_GRAMMAR_AUDIT_SCHEMA_VERSION = "geosym.geoparsing_grammar_audit.v1"
_FORMAL_FIELDS = ("points", "lines", "circles", "planes", "structure", "semantics")


@dataclass(frozen=True)
class GeoparsingCorpusRecord:
    id: str
    formal_description: GeoparsingFormalDescription

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip() or any(character.isspace() for character in self.id):
            raise InvalidGeometryInput("Geoparsing corpus record id must be a non-empty string without whitespace")


@dataclass(frozen=True)
class GeoparsingGrammarAudit:
    num_records: int
    num_complete: int
    formal_item_counts: Mapping[str, int]
    emitted_candidate_counts: Mapping[str, int]
    diagnostic_counts: Mapping[str, int]
    diagnostic_field_counts: Mapping[str, int]
    examples: tuple[Mapping[str, Any], ...]
    schema_version: str = GEOPARSING_GRAMMAR_AUDIT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "num_records": self.num_records,
            "num_complete": self.num_complete,
            "num_with_diagnostics": self.num_records - self.num_complete,
            "complete_rate": self.num_complete / self.num_records if self.num_records else 0.0,
            "formal_item_counts": dict(sorted(self.formal_item_counts.items())),
            "emitted_candidate_counts": dict(sorted(self.emitted_candidate_counts.items())),
            "diagnostic_counts": dict(sorted(self.diagnostic_counts.items())),
            "diagnostic_field_counts": dict(sorted(self.diagnostic_field_counts.items())),
            "examples": [dict(item) for item in self.examples],
            "interpretation": "syntax coverage only; this is not image parsing accuracy",
        }
        payload["checksum"] = stable_checksum(payload)
        return payload


class GeoparsingGrammarAuditor:
    def __init__(self, *, max_examples: int = 20):
        if isinstance(max_examples, bool) or not isinstance(max_examples, int) or max_examples < 0:
            raise InvalidGeometryInput("Geoparsing grammar max_examples must be a non-negative integer")
        self.max_examples = max_examples

    def audit(
        self,
        records: Sequence[GeoparsingCorpusRecord],
        *,
        on_progress: Callable[[int], None] | None = None,
    ) -> GeoparsingGrammarAudit:
        if not records:
            raise InvalidGeometryInput("Geoparsing grammar audit requires at least one record")
        record_ids = [record.id for record in records]
        if len(record_ids) != len(set(record_ids)):
            raise InvalidGeometryInput("Geoparsing grammar audit record ids must be unique")

        formal_counts: Counter[str] = Counter()
        candidate_counts: Counter[str] = Counter()
        diagnostic_counts: Counter[str] = Counter()
        diagnostic_field_counts: Counter[str] = Counter()
        examples = []
        complete = 0
        for index, record in enumerate(records, start=1):
            for field_name in _FORMAL_FIELDS:
                formal_counts[field_name] += len(getattr(record.formal_description, field_name))
            compiled = compile_geoparsing_formal_description(
                record.formal_description,
                source_id=record.id,
            )
            candidate_counts.update(
                {
                    "objects": len(compiled.graph.objects),
                    "relations": len(compiled.graph.relations),
                    "quantities": len(compiled.graph.quantities),
                    "regions": len(compiled.graph.regions),
                    "targets": len(compiled.graph.targets),
                }
            )
            if compiled.complete:
                complete += 1
            for diagnostic in compiled.diagnostics:
                diagnostic_counts[diagnostic.code] += 1
                diagnostic_field_counts[diagnostic.field] += 1
                if len(examples) < self.max_examples:
                    examples.append({"record_id": record.id, **diagnostic.to_dict()})
            if on_progress is not None:
                on_progress(index)
        return GeoparsingGrammarAudit(
            num_records=len(records),
            num_complete=complete,
            formal_item_counts=dict(formal_counts),
            emitted_candidate_counts=dict(candidate_counts),
            diagnostic_counts=dict(diagnostic_counts),
            diagnostic_field_counts=dict(diagnostic_field_counts),
            examples=tuple(examples),
        )


def load_geoparsing_corpus(
    path: str | Path,
    *,
    id_field: str = "id",
    formal_field: str | None = None,
) -> tuple[GeoparsingCorpusRecord, ...]:
    source = Path(path)
    if not source.is_file():
        raise InvalidGeometryInput(f"Geoparsing corpus JSONL does not exist: {source}")
    if not id_field.strip():
        raise InvalidGeometryInput("Geoparsing corpus id_field must be non-empty")
    if formal_field is not None and not formal_field.strip():
        raise InvalidGeometryInput("Geoparsing corpus formal_field must be null or non-empty")

    records = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InvalidGeometryInput(
                    f"Invalid Geoparsing corpus JSON at {source}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, Mapping):
                raise InvalidGeometryInput(f"Geoparsing corpus row at {source}:{line_number} must be an object")
            if id_field not in row:
                raise InvalidGeometryInput(
                    f"Geoparsing corpus row at {source}:{line_number} is missing id field {id_field!r}"
                )
            record_id = str(row[id_field])
            if formal_field is None:
                formal_data = {field_name: row[field_name] for field_name in _FORMAL_FIELDS if field_name in row}
            else:
                if formal_field not in row:
                    raise InvalidGeometryInput(
                        f"Geoparsing corpus row at {source}:{line_number} is missing formal field {formal_field!r}"
                    )
                formal_data = row[formal_field]
            try:
                description = GeoparsingFormalDescription.from_dict(formal_data)
                records.append(GeoparsingCorpusRecord(record_id, description))
            except (TypeError, ValueError) as exc:
                raise InvalidGeometryInput(
                    f"Invalid Geoparsing formal description at {source}:{line_number}: {exc}"
                ) from exc
    if not records:
        raise InvalidGeometryInput(f"Geoparsing corpus JSONL is empty: {source}")
    record_ids = [record.id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise InvalidGeometryInput("Geoparsing corpus JSONL contains duplicate ids")
    return tuple(records)
