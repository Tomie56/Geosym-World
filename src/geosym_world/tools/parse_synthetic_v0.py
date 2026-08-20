"""Run the deterministic synthetic Parser v0 over canonical JSONL records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from geosym_world.parser import SyntheticParserV0


DEFAULT_OUTPUT = "artifacts/parser/synthetic_v0/results.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Canonical synthetic record JSONL path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Parser result JSONL path.")
    parser.add_argument("--admit", action="store_true", help="Run hypothesis grounding and admission after parsing.")
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N records; 0 disables progress output.")
    args = parser.parse_args()
    if args.progress_every < 0:
        parser.error("--progress-every must be >= 0")

    parser_v0 = SyntheticParserV0()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    admitted = 0
    with Path(args.input).open("r", encoding="utf-8") as input_file, output_path.open("w", encoding="utf-8") as output_file:
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
            except Exception as exc:
                record_id = record.get("id", f"line-{line_number}")
                raise ValueError(f"Parser v0 failed for record {record_id!r} at line {line_number}: {exc}") from exc
            output_file.write(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            processed += 1
            admitted += int(result.admitted)
            _print_progress(processed, admitted, args.progress_every)

    summary = {
        "input": str(args.input),
        "output": str(output_path),
        "num_records": processed,
        "num_admitted": admitted if args.admit else None,
        "admission_enabled": args.admit,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _print_progress(processed: int, admitted: int, every: int) -> None:
    if every and processed % every == 0:
        print(f"[parser-v0] processed={processed} admitted={admitted}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
