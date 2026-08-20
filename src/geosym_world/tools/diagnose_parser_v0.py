"""Diagnose Parser v0 records and aggregate deterministic benchmark slices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from geosym_world.benchmark import ParserBenchmarkSlicer
from geosym_world.parser import SyntheticParserDebugger, SyntheticParserV0


DEFAULT_OUTPUT = "artifacts/parser/synthetic_v0/diagnosis.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Canonical synthetic record JSONL path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Diagnosis JSONL path.")
    parser.add_argument("--admit", action="store_true", help="Run grounding and admission before diagnosis.")
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N records; 0 disables progress output.")
    args = parser.parse_args()
    if args.progress_every < 0:
        parser.error("--progress-every must be >= 0")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parser_v0 = SyntheticParserV0()
    debugger = SyntheticParserDebugger()
    rows: list[dict[str, Any]] = []
    with Path(args.input).open("r", encoding="utf-8") as input_file, output_path.open("w", encoding="utf-8") as output_file:
        processed = 0
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {args.input}:{line_number}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {args.input}:{line_number}")
            try:
                result = parser_v0.parse_and_admit(record) if args.admit else parser_v0.parse(record)
                diagnosis = debugger.diagnose(result, _reference_from_record(record))
            except Exception as exc:
                record_id = record.get("id", f"line-{line_number}")
                raise ValueError(f"Parser diagnosis failed for record {record_id!r} at line {line_number}: {exc}") from exc
            state = record.get("formal_state") or record.get("state") or {}
            target = state.get("target") or {}
            synthetic = record.get("synthetic") or {}
            row = {
                **diagnosis,
                "parser": "synthetic_parser_v0",
                "task_type": target.get("type"),
                "difficulty": synthetic.get("difficulty"),
                "diagnostic_axes": synthetic.get("diagnostic_axes") or (),
            }
            output_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            rows.append(row)
            processed += 1
            if args.progress_every and processed % args.progress_every == 0:
                print(f"[parser-diagnosis] processed={processed}", file=sys.stderr, flush=True)

    summary = ParserBenchmarkSlicer().summarize(rows)
    summary.update({"input": str(args.input), "output": str(output_path), "admission_enabled": args.admit})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _reference_from_record(record: dict[str, Any]) -> dict[str, Any]:
    state = record.get("formal_state") or record.get("state")
    if not isinstance(state, dict):
        raise ValueError("Parser diagnosis record requires formal_state or state")
    buckets = (
        "points",
        "lines",
        "segments",
        "rays",
        "circles",
        "arcs",
        "polygons",
        "regions",
        "composite_regions",
        "planes",
        "polyhedra",
    )
    geometry = {key: list((state.get(key) or {}).values()) for key in buckets}
    geometry["entities"] = []
    geometry["relations"] = list(state.get("relations") or [])
    geometry["quantities"] = []
    return {"geometry": geometry, "target": state.get("target")}


if __name__ == "__main__":
    main()
