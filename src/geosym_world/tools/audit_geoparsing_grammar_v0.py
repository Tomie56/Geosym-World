"""Audit Geoparsing/GDP formal-language coverage without images or model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from geosym_world.benchmark import GeoparsingGrammarAuditor, load_geoparsing_corpus
from geosym_world.schema.canonical import stable_checksum


GEOPARSING_GRAMMAR_AUDIT_MANIFEST_VERSION = "geosym.geoparsing_grammar_audit_manifest.v1"
DEFAULT_OUTPUT = "artifacts/parser/geoparsing/grammar-audit.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="GDP-style JSONL containing formal-language fields.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Audit JSON output path.")
    parser.add_argument("--id-field", default="id", help="Top-level record identifier field.")
    parser.add_argument(
        "--formal-field",
        default=None,
        help="Optional top-level field containing the formal description; default reads formal fields directly.",
    )
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()
    if args.max_examples < 0:
        parser.error("--max-examples must be >= 0")
    if args.progress_every < 0:
        parser.error("--progress-every must be >= 0")
    manifest = run(
        input_path=Path(args.input),
        output_path=Path(args.output),
        id_field=args.id_field,
        formal_field=args.formal_field,
        max_examples=args.max_examples,
        progress_every=args.progress_every,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def run(
    *,
    input_path: Path,
    output_path: Path,
    id_field: str,
    formal_field: str | None,
    max_examples: int,
    progress_every: int,
) -> dict[str, Any]:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Geoparsing grammar audit output must not overwrite the input")
    if progress_every < 0:
        raise ValueError("Geoparsing grammar audit progress_every must be non-negative")
    records = load_geoparsing_corpus(input_path, id_field=id_field, formal_field=formal_field)
    auditor = GeoparsingGrammarAuditor(max_examples=max_examples)

    def report_progress(processed: int) -> None:
        if progress_every and processed % progress_every == 0:
            print(
                f"[geoparsing-grammar-audit] processed={processed}/{len(records)}",
                file=sys.stderr,
                flush=True,
            )

    audit = auditor.audit(records, on_progress=report_progress)
    audit_payload = audit.to_dict()
    manifest = {
        "schema_version": GEOPARSING_GRAMMAR_AUDIT_MANIFEST_VERSION,
        "input": str(input_path),
        "input_sha256": _file_hash(input_path),
        "output": str(output_path),
        "configuration": {
            "id_field": id_field,
            "formal_field": formal_field,
            "max_examples": max_examples,
            "progress_every": progress_every,
        },
        "audit": audit_payload,
    }
    manifest["checksum"] = stable_checksum(manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
