"""Materialize trusted Parser references from private human annotation chains."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, TypeVar

from geosym_world.benchmark import (
    ParserAnnotationAdjudication,
    ParserAnnotationPolicy,
    ParserAnnotationRecord,
    ParserAnnotationReview,
    ParserSourceManifest,
    build_adjudicated_parser_reference,
)
from geosym_world.parser import ParserInput
from geosym_world.schema.canonical import stable_checksum


DEFAULT_OUTPUT = "artifacts/parser/datasets/adjudicated_v0/references.jsonl"
_Record = TypeVar("_Record")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, help="Prepared public ParserInput JSONL.")
    parser.add_argument("--source-audit", required=True, help="Private source-audit JSONL from dataset preparation.")
    parser.add_argument("--annotations", required=True, help="Private primary human annotation JSONL.")
    parser.add_argument("--reviews", required=True, help="Private independent review JSONL.")
    parser.add_argument("--adjudications", required=True, help="Private adjudication JSONL.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Evaluator-only typed reference JSONL.")
    parser.add_argument("--lineage-output", default=None, help="Private annotation lineage JSONL.")
    parser.add_argument("--require-distinct-adjudicator", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    if args.progress_every < 0:
        parser.error("--progress-every must be >= 0")
    try:
        manifest = run(
            inputs_path=Path(args.inputs),
            source_audit_path=Path(args.source_audit),
            annotations_path=Path(args.annotations),
            reviews_path=Path(args.reviews),
            adjudications_path=Path(args.adjudications),
            output_path=Path(args.output),
            lineage_output_path=Path(args.lineage_output) if args.lineage_output else None,
            require_distinct_adjudicator=args.require_distinct_adjudicator,
            progress_every=args.progress_every,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def run(
    *,
    inputs_path: Path,
    source_audit_path: Path,
    annotations_path: Path,
    reviews_path: Path,
    adjudications_path: Path,
    output_path: Path,
    lineage_output_path: Path | None = None,
    require_distinct_adjudicator: bool = False,
    progress_every: int = 25,
) -> dict[str, Any]:
    if progress_every < 0:
        raise ValueError("progress_every must be >= 0")
    lineage_path = lineage_output_path or output_path.with_name(f"{output_path.stem}.lineage.jsonl")
    input_paths = {
        inputs_path.resolve(),
        source_audit_path.resolve(),
        annotations_path.resolve(),
        reviews_path.resolve(),
        adjudications_path.resolve(),
    }
    if output_path.resolve() in input_paths or lineage_path.resolve() in input_paths:
        raise ValueError("Parser reference outputs must not overwrite input files")
    if output_path.resolve() == lineage_path.resolve():
        raise ValueError("Parser reference and lineage outputs must be different files")

    inputs = _index_records(_read_jsonl(inputs_path), inputs_path, ParserInput.from_dict)
    source_manifests = _index_source_audits(_read_jsonl(source_audit_path), source_audit_path, inputs)
    annotations = _index_records(
        _read_jsonl(annotations_path), annotations_path, ParserAnnotationRecord.from_dict
    )
    reviews = _index_records(_read_jsonl(reviews_path), reviews_path, ParserAnnotationReview.from_dict)
    adjudications = _index_records(
        _read_jsonl(adjudications_path), adjudications_path, ParserAnnotationAdjudication.from_dict
    )
    _require_same_ids(
        inputs=inputs,
        source_audit=source_manifests,
        annotations=annotations,
        reviews=reviews,
        adjudications=adjudications,
    )

    policy = ParserAnnotationPolicy(require_distinct_adjudicator=require_distinct_adjudicator)
    references = []
    lineage = []
    for index, sample_id in enumerate(sorted(inputs), start=1):
        reference, audit = build_adjudicated_parser_reference(
            inputs[sample_id],
            manifest=source_manifests[sample_id],
            annotation=annotations[sample_id],
            review=reviews[sample_id],
            adjudication=adjudications[sample_id],
            policy=policy,
        )
        references.append(reference)
        lineage.append(audit)
        if progress_every and (index % progress_every == 0 or index == len(inputs)):
            print(f"[parser-annotation-v0] materialized={index}/{len(inputs)}", file=sys.stderr, flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lineage_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_path, references)
    _write_jsonl(lineage_path, lineage)
    manifest = {
        "schema_version": "geosym.parser_annotation_manifest.v1",
        "inputs": _file_descriptor(inputs_path),
        "source_audit": _file_descriptor(source_audit_path),
        "annotations": _file_descriptor(annotations_path),
        "reviews": _file_descriptor(reviews_path),
        "adjudications": _file_descriptor(adjudications_path),
        "references": _file_descriptor(output_path),
        "lineage": _file_descriptor(lineage_path),
        "num_records": len(references),
        "policy": policy.to_dict(),
    }
    manifest["checksum"] = stable_checksum(manifest)
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "manifest": str(manifest_path)}


def _index_records(
    rows: list[dict[str, Any]],
    path: Path,
    parser: Callable[[dict[str, Any]], _Record],
) -> dict[str, _Record]:
    indexed: dict[str, _Record] = {}
    for line_number, row in enumerate(rows, start=1):
        try:
            record = parser(row)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid record at {path}:{line_number}: {exc}") from exc
        sample_id = getattr(record, "id", None)
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"Record is missing id at {path}:{line_number}")
        if sample_id in indexed:
            raise ValueError(f"Duplicate record id {sample_id!r} at {path}:{line_number}")
        indexed[sample_id] = record
    if not indexed:
        raise ValueError(f"JSONL input is empty: {path}")
    return indexed


def _index_source_audits(
    rows: list[dict[str, Any]],
    path: Path,
    inputs: dict[str, ParserInput],
) -> dict[str, ParserSourceManifest]:
    indexed = {}
    required = {"schema_version", "public_id", "source_image", "source_manifest", "materialization", "checksum"}
    for line_number, row in enumerate(rows, start=1):
        if set(row) != required:
            raise ValueError(f"Invalid source audit fields at {path}:{line_number}")
        if row["schema_version"] != "geosym.parser_source_audit.v1":
            raise ValueError(f"Unsupported source audit schema at {path}:{line_number}")
        payload = {key: value for key, value in row.items() if key != "checksum"}
        if row["checksum"] != stable_checksum(payload):
            raise ValueError(f"Source audit checksum mismatch at {path}:{line_number}")
        sample_id = row["public_id"]
        if not isinstance(sample_id, str) or sample_id not in inputs:
            raise ValueError(f"Source audit references unknown public input at {path}:{line_number}")
        if sample_id in indexed:
            raise ValueError(f"Duplicate source audit id {sample_id!r} at {path}:{line_number}")
        try:
            manifest = ParserSourceManifest.from_dict(row["source_manifest"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid source manifest at {path}:{line_number}: {exc}") from exc
        materialization = row["materialization"]
        if not isinstance(materialization, dict):
            raise ValueError(f"Invalid materialization audit at {path}:{line_number}")
        materialization_payload = {key: value for key, value in materialization.items() if key != "checksum"}
        if materialization.get("checksum") != stable_checksum(materialization_payload):
            raise ValueError(f"Materialization checksum mismatch at {path}:{line_number}")
        if materialization.get("public_id") != sample_id:
            raise ValueError(f"Materialization public id mismatch at {path}:{line_number}")
        if materialization.get("parser_input_checksum") != stable_checksum(inputs[sample_id].to_dict()):
            raise ValueError(f"Materialization input checksum mismatch at {path}:{line_number}")
        if materialization.get("source_manifest_checksum") != manifest.checksum():
            raise ValueError(f"Materialization source manifest checksum mismatch at {path}:{line_number}")
        indexed[sample_id] = manifest
    return indexed


def _require_same_ids(**collections: dict[str, Any]) -> None:
    reference_name, reference = next(iter(collections.items()))
    expected = set(reference)
    for name, records in collections.items():
        actual = set(records)
        if actual != expected:
            raise ValueError(
                f"Parser annotation dataset IDs differ for {name} versus {reference_name}; "
                f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
            )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"JSONL input does not exist: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _file_descriptor(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": _file_hash(path), "bytes": path.stat().st_size}


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
