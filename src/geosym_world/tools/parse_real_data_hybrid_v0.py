"""Run the script-plus-semantic hybrid parser over public JSONL inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from geosym_world.parser.pipeline import HybridParsePipeline
from geosym_world.schema.canonical import stable_checksum


HYBRID_RUN_MANIFEST_SCHEMA_VERSION = "geosym.hybrid_parser_run_manifest.v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--records-jsonl", required=True, help="JSONL rows containing public image/question inputs.")
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--parsed-root", default=None)
    parser.add_argument("--backend", choices=("mock", "scripted", "openrouter"), default="mock")
    parser.add_argument("--model", default=None)
    parser.add_argument("--relation-model", default=None)
    parser.add_argument("--instruction-model", default=None)
    parser.add_argument("--extraction-mode", choices=("single", "staged"), default="single")
    parser.add_argument("--judge-backend", choices=("none", "openrouter"), default="none")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Write an explicit failure report and continue after a record error.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every must be >= 0")
    manifest = run(
        dataset_name=args.dataset_name,
        records_path=Path(args.records_jsonl),
        raw_root=Path(args.raw_root) if args.raw_root else None,
        parsed_root=Path(args.parsed_root) if args.parsed_root else None,
        backend=args.backend,
        model=args.model,
        relation_model=args.relation_model,
        instruction_model=args.instruction_model,
        extraction_mode=args.extraction_mode,
        judge_backend=args.judge_backend,
        judge_model=args.judge_model,
        limit=args.limit,
        progress_every=args.progress_every,
        continue_on_error=args.continue_on_error,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def run(
    *,
    dataset_name: str,
    records_path: Path,
    raw_root: Path | None,
    parsed_root: Path | None,
    backend: str,
    model: str | None,
    relation_model: str | None,
    instruction_model: str | None,
    extraction_mode: str,
    judge_backend: str,
    judge_model: str | None,
    limit: int | None,
    progress_every: int,
    continue_on_error: bool,
) -> dict[str, Any]:
    if not records_path.is_file():
        raise ValueError(f"Hybrid parser records JSONL does not exist: {records_path}")
    if limit is not None and limit <= 0:
        raise ValueError("Hybrid parser limit must be positive")
    if progress_every < 0:
        raise ValueError("Hybrid parser progress_every must be non-negative")

    pipeline = HybridParsePipeline(
        dataset_name=dataset_name,
        raw_root=raw_root,
        parsed_root=parsed_root,
        semantic_backend=backend,
        model=model,
        relation_model=relation_model,
        instruction_model=instruction_model,
        extraction_mode=extraction_mode,
        judge_backend=judge_backend,
        judge_model=judge_model,
    )
    selected = 0
    read = 0
    failed = 0
    decisions: Counter[str] = Counter()
    stages: Counter[str] = Counter()
    with records_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if limit is not None and selected >= limit:
                break
            read += 1
            row = _json_object(line, records_path, line_number)
            selected += 1
            try:
                report = pipeline.process_record(row)
            except Exception as exc:
                if not continue_on_error:
                    raise ValueError(
                        f"Hybrid parser failed for record at {records_path}:{line_number}: {exc}"
                    ) from exc
                failed += 1
                report = pipeline.write_error_report(row, exc)
            decisions[str(report.get("merged_decision") or "unknown")] += 1
            stage = (report.get("stage_diagnosis") or {}).get("primary_stage")
            if stage:
                stages[str(stage)] += 1
            if progress_every and selected % progress_every == 0:
                print(
                    f"[hybrid-parser-v0] processed={selected} failed={failed}",
                    file=sys.stderr,
                    flush=True,
                )

    manifest = {
        "schema_version": HYBRID_RUN_MANIFEST_SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "records_path": str(records_path),
        "records_sha256": _file_hash(records_path),
        "parsed_dataset_dir": str(pipeline.layout.parsed_dataset_dir),
        "configuration": {
            "backend": backend,
            "model": model,
            "relation_model": relation_model,
            "instruction_model": instruction_model,
            "extraction_mode": extraction_mode,
            "judge_backend": judge_backend,
            "judge_model": judge_model,
            "limit": limit,
            "progress_every": progress_every,
            "continue_on_error": continue_on_error,
            "runtime_input_fields": ["schema_version", "id", "image", "question"],
        },
        "counts": {
            "records_read": read,
            "records_selected": selected,
            "records_succeeded": selected - failed,
            "records_failed": failed,
        },
        "merged_decision_counts": dict(sorted(decisions.items())),
        "primary_stage_counts": dict(sorted(stages.items())),
    }
    manifest["checksum"] = stable_checksum(manifest)
    manifest_path = pipeline.layout.parsed_dataset_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "manifest": str(manifest_path)}


def _json_object(line: str, source: Path, line_number: int) -> dict[str, Any]:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {source}:{line_number}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object at {source}:{line_number}")
    return payload


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
