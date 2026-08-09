"""Run the complete original geometry synthesis tools pipeline through the optimized bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geosym_world.datasets.layout import artifact_root
from geosym_world.legacy import LegacyCompatiblePipeline, LegacyPipelineOptions, compatibility_report


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "legacy" / "config_easy.json"
DEFAULT_OUTPUT = artifact_root() / "generated" / "legacy_review"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--records", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--stages", default="template,builder,shader,gt,qa")
    parser.add_argument("--rounds-json", default=None, help='Enhancement distribution such as \'{"0":1,"1":1}\'.')
    parser.add_argument("--shader-attempts", type=int, default=1)
    parser.add_argument("--disable-shader", action="store_true")
    parser.add_argument("--questions-per-record", type=int, default=1)
    parser.add_argument("--question-types", default="", help="Comma-separated legacy QA types; empty uses the source config.")
    parser.add_argument(
        "--task-timeout",
        type=float,
        default=300.0,
        help="Kill a generation/GT/QA item after this many seconds; 0 disables isolation timeouts.",
    )
    parser.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=0,
        help="Recycle pooled workers after this many items on Python 3.11+ when task timeouts are disabled; 0 disables recycling.",
    )
    parser.add_argument("--input-base", default=None)
    parser.add_argument("--input-enhanced", default=None)
    parser.add_argument("--input-shaded", default=None)
    parser.add_argument("--input-gt", default=None)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--compatibility-report", action="store_true")
    args = parser.parse_args()

    if args.compatibility_report:
        print(json.dumps(compatibility_report(), ensure_ascii=False, indent=2))
        return

    rounds = None
    if args.rounds_json:
        rounds = {int(key): int(value) for key, value in json.loads(args.rounds_json).items()}
    options = LegacyPipelineOptions(
        output_root=Path(args.output_root),
        records=args.records,
        seed=args.seed,
        workers=args.workers,
        stages=tuple(item.strip() for item in args.stages.split(",") if item.strip()),
        rounds_distribution=rounds,
        shader_attempts=args.shader_attempts,
        shader_enabled=not args.disable_shader,
        questions_per_record=args.questions_per_record,
        question_types=tuple(item.strip() for item in args.question_types.split(",") if item.strip()),
        task_timeout_seconds=args.task_timeout or None,
        max_tasks_per_child=args.max_tasks_per_child or None,
        fail_fast=args.fail_fast,
        input_base=Path(args.input_base) if args.input_base else None,
        input_enhanced=Path(args.input_enhanced) if args.input_enhanced else None,
        input_shaded=Path(args.input_shaded) if args.input_shaded else None,
        input_gt=Path(args.input_gt) if args.input_gt else None,
    )
    manifest = LegacyCompatiblePipeline.from_file(args.config, options).run()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
