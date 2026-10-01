"""Extract geometry semantics; optionally realize, verify and redraw a batch."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys

from PIL import Image

from geosym_world.parser import ParserPipeline
from geosym_world.parser.contracts import validate_parser_input
from geosym_world.parser.openrouter import PROMPT_VERSION, api_key_from_env, parse_one
from geosym_world.parser.provider_config import load_project_env
from geosym_world.rendering import write_png
from geosym_world.tools.build_parser_review_sheet import build_contact_sheet


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    load_project_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="ParserInput JSONL")
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path, help="Proposal-only predictions JSONL")
    destination.add_argument("--run-dir", type=Path, help="New directory for JSON, images, manifest and contact sheet")
    parser.add_argument("--model", default=os.environ.get("GEOSYM_OPENROUTER_MODEL", "qwen/qwen3.7-flash"))
    parser.add_argument("--max-records", type=int, default=3, help="Maximum attempted inputs, including failures")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--failures", type=Path, help="Proposal-only failure JSONL path")
    args = parser.parse_args()
    if args.max_records <= 0 or args.progress_every < 0:
        parser.error("--max-records must be positive and --progress-every must be non-negative")
    if not args.input.is_file():
        parser.error(f"Parser input does not exist: {args.input}")
    if args.run_dir and args.failures:
        parser.error("--failures applies only to --output; run failures are included in summary.json")
    if args.run_dir and args.run_dir.exists():
        parser.error(f"Run directory already exists: {args.run_dir}")

    api_key = api_key_from_env()

    def extractor(parser_input):
        raw_path = args.run_dir / "json" / "raw" / f"{parser_input.id}.json" if args.run_dir else None
        return parse_one(parser_input, model=args.model, api_key=api_key, raw_output=raw_path)

    pipeline = ParserPipeline(extractor)
    rows = []
    seen_ids: set[str] = set()
    attempted = completed = 0
    if args.run_dir:
        args.run_dir.mkdir(parents=True)
        package = Path(__file__).resolve().parents[1]
        code_paths = [
            package / "parser" / name for name in (
                "openrouter.py", "semantic_compiler.py", "pipeline.py", "seed_realizer.py", "constraint_completion.py",
                "prompts/geometry_extraction_zh.txt",
            )
        ]
        code_paths.append(Path(__file__).resolve())
        _write_json(args.run_dir / "manifest.json", {
            "mode": "geometry_semantics_to_candidate_state", "model": args.model,
            "prompt_version": PROMPT_VERSION, "python": sys.executable,
            "command": [sys.executable, "-m", "geosym_world.tools.parse_openrouter_v0", *sys.argv[1:]],
            "input": str(args.input.resolve()), "input_sha256": _digest(args.input),
            "code_sha256": {str(path.relative_to(package)): _digest(path) for path in code_paths},
            "max_records": args.max_records, "workers": 1, "temperature": 0,
            "retries": 0, "random_seed": None, "pixel_topology": "not_applicable",
        })
    with ExitStack() as stack:
        source = stack.enter_context(args.input.open(encoding="utf-8"))
        target = failures = None
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            failure_path = args.failures or args.output.with_name(args.output.name + ".failed.jsonl")
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            target = stack.enter_context(args.output.open("w", encoding="utf-8"))
            failures = stack.enter_context(failure_path.open("w", encoding="utf-8"))
        for line_number, line in enumerate(source, 1):
            if attempted >= args.max_records:
                break
            if not line.strip():
                continue
            attempted += 1
            sample_id = None
            sample_dir = None
            phase = "input"
            row = {"line": line_number}
            try:
                parser_input = validate_parser_input(json.loads(line))
                sample_id = parser_input.id
                if sample_id in seen_ids:
                    raise ValueError(f"Duplicate sample ID: {sample_id}")
                if Path(sample_id).name != sample_id or sample_id in {".", ".."}:
                    raise ValueError(f"Sample ID is not a safe file name: {sample_id!r}")
                seen_ids.add(sample_id)
                row.update(id=sample_id, image_sha256=_digest(Path(parser_input.image)))
                if args.run_dir:
                    sample_dir = args.run_dir / "images" / sample_id
                    sample_dir.mkdir(parents=True)
                    with Image.open(parser_input.image) as image:
                        image.convert("RGB").save(sample_dir / "input.png")
                    phase = "pipeline"
                    result = pipeline.run(parser_input)
                    _write_json(args.run_dir / "json" / f"{sample_id}.json", result.to_dict())
                    if result.realization.scene is not None:
                        phase = "render"
                        write_png(result.realization.scene, sample_dir / "redraw.png")
                    row.update(
                        status=result.realization.status.value, quality=result.quality.to_dict(),
                        diagnostics=list(result.realization.diagnostics),
                        redraw=result.realization.scene is not None,
                    )
                else:
                    phase = "extraction"
                    prediction = extractor(parser_input)
                    target.write(json.dumps(prediction.to_dict(), ensure_ascii=False) + "\n")
                    target.flush()
                    row.update(status="compiled")
                completed += 1
            except Exception as error:
                row.update(id=sample_id, status="error", phase=phase, error_type=type(error).__name__, error=str(error))
                if failures is not None:
                    failures.write(json.dumps(row, ensure_ascii=False) + "\n")
                    failures.flush()
                if sample_dir is not None and phase != "render":
                    _write_json(args.run_dir / "json" / f"{sample_id}.json", row)
                print(f"[parser] failed line={line_number} phase={phase} error={type(error).__name__}: {error}", flush=True)
            rows.append(row)
            if args.run_dir:
                _write_json(args.run_dir / "summary.json", {"attempted": attempted, "completed": completed, "samples": rows})
            if args.progress_every and attempted % args.progress_every == 0:
                print(f"[parser] attempted={attempted} completed={completed} last_id={sample_id} status={row['status']}", flush=True)
    if args.run_dir and any((args.run_dir / "images").glob("*/input.png")):
        print(f"[parser-review] wrote={build_contact_sheet(args.run_dir)}", flush=True)
    print(f"[parser] attempted={attempted} completed={completed} output={args.run_dir or args.output}", flush=True)


if __name__ == "__main__":
    main()
