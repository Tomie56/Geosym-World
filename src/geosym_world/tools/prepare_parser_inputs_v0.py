"""Materialize leak-resistant public ParserInput records from a private source catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from geosym_world.benchmark import ParserSourceCatalogEntry, materialize_parser_input
from geosym_world.schema.canonical import stable_checksum


DEFAULT_OUTPUT_DIR = "artifacts/parser/datasets/prepared_v0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, help="Private parser source catalog JSONL.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Prepared dataset directory.")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    if args.progress_every < 0:
        parser.error("--progress-every must be >= 0")
    try:
        manifest = run(
            catalog_path=Path(args.catalog),
            output_dir=Path(args.output_dir),
            progress_every=args.progress_every,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def run(*, catalog_path: Path, output_dir: Path, progress_every: int = 25) -> dict[str, Any]:
    if progress_every < 0:
        raise ValueError("progress_every must be >= 0")
    entries = _read_catalog(catalog_path)
    duplicate_source_ids = sorted(source_id for source_id, count in Counter(item.source_id for item in entries).items() if count > 1)
    if duplicate_source_ids:
        raise ValueError(f"Parser source catalog has duplicate source IDs: {duplicate_source_ids}")

    manifests = tuple(entry.build_manifest() for entry in entries)
    output_dir.mkdir(parents=True, exist_ok=True)
    parser_input_path = output_dir / "parser-input.jsonl"
    source_audit_path = output_dir / "source-audit.jsonl"
    parser_rows = []
    audit_rows = []
    public_ids = set()
    for index, (entry, source_manifest) in enumerate(zip(entries, manifests), start=1):
        parser_input, materialization = materialize_parser_input(
            entry.image,
            manifest=source_manifest,
            output_dir=output_dir,
            question=entry.question,
        )
        if parser_input.id in public_ids:
            raise ValueError(f"Prepared parser input ID collision: {parser_input.id}")
        public_ids.add(parser_input.id)
        parser_rows.append(parser_input.to_dict())
        audit = {
            "schema_version": "geosym.parser_source_audit.v1",
            "public_id": parser_input.id,
            "source_image": entry.image,
            "source_manifest": source_manifest.to_dict(),
            "materialization": materialization,
        }
        audit["checksum"] = stable_checksum(audit)
        audit_rows.append(audit)
        if progress_every and (index % progress_every == 0 or index == len(entries)):
            print(f"[parser-dataset-v0] prepared={index}/{len(entries)}", file=sys.stderr, flush=True)

    _write_jsonl(parser_input_path, parser_rows)
    _write_jsonl(source_audit_path, audit_rows)
    manifest = {
        "schema_version": "geosym.parser_dataset_manifest.v1",
        "catalog": str(catalog_path),
        "catalog_sha256": _file_hash(catalog_path),
        "parser_input": str(parser_input_path),
        "parser_input_sha256": _file_hash(parser_input_path),
        "source_audit": str(source_audit_path),
        "source_audit_sha256": _file_hash(source_audit_path),
        "num_records": len(entries),
        "num_source_ids": len({item.source_id for item in entries}),
        "num_public_ids": len(public_ids),
        "protocol_counts": dict(sorted(Counter(item.protocol.value for item in entries).items())),
        "license_ids": sorted({item.license_id for item in entries}),
        "split_groups": sorted({item.split_group for item in entries}),
    }
    manifest["checksum"] = stable_checksum(manifest)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**manifest, "manifest": str(manifest_path)}


def _read_catalog(path: Path) -> tuple[ParserSourceCatalogEntry, ...]:
    if not path.is_file():
        raise ValueError(f"Parser source catalog does not exist: {path}")
    entries = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                entries.append(ParserSourceCatalogEntry.from_dict(payload))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid parser source catalog entry at {path}:{line_number}: {exc}") from exc
    if not entries:
        raise ValueError("Parser source catalog is empty")
    return tuple(entries)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
