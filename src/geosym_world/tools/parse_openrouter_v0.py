"""Call an OpenRouter vision model and write proposal-only ParseGraph predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geosym_world.parser.contracts import validate_parser_input
from geosym_world.parser.openrouter import api_key_from_env, parse_one


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="ParserInput JSONL")
    parser.add_argument("--output", required=True, help="External predictions JSONL")
    parser.add_argument("--model", default="google/gemini-2.0-flash-001")
    parser.add_argument("--max-records", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--failures", default=None, help="JSONL diagnostics path; defaults to OUTPUT.failed.jsonl")
    args = parser.parse_args()
    if args.max_records <= 0 or args.progress_every < 0:
        parser.error("--max-records must be positive and --progress-every must be non-negative")
    input_path, output_path = Path(args.input), Path(args.output)
    if not input_path.is_file():
        parser.error(f"Parser input does not exist: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    api_key = api_key_from_env()
    count = 0
    failures_path = Path(args.failures) if args.failures else output_path.with_name(output_path.name + ".failed.jsonl")
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target, failures_path.open("w", encoding="utf-8") as failures:
        for line_number, line in enumerate(source, 1):
            if not line.strip() or count >= args.max_records:
                continue
            try:
                parser_input = validate_parser_input(json.loads(line))
                prediction = parse_one(parser_input, model=args.model, api_key=api_key)
                target.write(json.dumps(prediction.to_dict(), ensure_ascii=False) + "\n")
                target.flush()
                count += 1
                if args.progress_every and count % args.progress_every == 0:
                    print(f"[openrouter] completed={count} last_id={parser_input.id}", flush=True)
            except Exception as exc:
                failure = {"line": line_number, "id": parser_input.id if "parser_input" in locals() else None, "error_type": type(exc).__name__, "error": str(exc)}
                failures.write(json.dumps(failure, ensure_ascii=False) + "\n")
                failures.flush()
                print(f"[openrouter] failed line={line_number} error={type(exc).__name__}: {exc}", flush=True)
                continue
    print(f"[openrouter] wrote={count} output={output_path} failures={failures_path}")


if __name__ == "__main__":
    main()
