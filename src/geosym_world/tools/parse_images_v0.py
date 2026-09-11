"""Run the clean image-only Parser v0 over public ParserInput JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from geosym_world.parser import CleanDiagramParserV0


DEFAULT_OUTPUT = "artifacts/parser/clean_v0/results.jsonl"


def main() -> None:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--input", required=True, help="Public parser-input JSONL path.")
    argument_parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Proposed Parser result JSONL path.")
    argument_parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N records; 0 disables output.")
    args = argument_parser.parse_args()
    if args.progress_every < 0:
        argument_parser.error("--progress-every must be >= 0")

    parser = CleanDiagramParserV0()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    with input_path.open("r", encoding="utf-8") as input_file, output_path.open("w", encoding="utf-8") as output_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {input_path}:{line_number}: {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected an object at {input_path}:{line_number}")
            try:
                result = parser.parse(payload)
            except Exception as exc:
                record_id = payload.get("id", f"line-{line_number}")
                raise ValueError(f"Image Parser v0 failed for record {record_id!r} at {input_path}:{line_number}: {exc}") from exc
            output_file.write(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            processed += 1
            if args.progress_every and processed % args.progress_every == 0:
                print(f"[image-parser-v0] processed={processed}", file=sys.stderr, flush=True)
    print(json.dumps({"input": str(input_path), "output": str(output_path), "num_records": processed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
