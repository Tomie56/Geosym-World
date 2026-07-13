"""Run deterministic GeoClaw pilot baselines and export a manifest."""

from __future__ import annotations

import argparse
import json

from geosym_world_geometry.geoclaw.datasets import built_in_pilot_tasks
from geosym_world_geometry.geoclaw.runners import run_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="geosym_world_geometry/outputs/geoclaw_batch")
    parser.add_argument("--baseline", choices=("direct", "oracle-final"), default="oracle-final")
    parser.add_argument("--hidden-feedback", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--allowed-tools")
    args = parser.parse_args()
    allowed_tools = tuple(item.strip() for item in args.allowed_tools.split(",") if item.strip()) if args.allowed_tools else None
    provider = _direct_provider if args.baseline == "direct" else _oracle_final_provider
    manifest = run_batch(
        built_in_pilot_tasks(), provider, args.output_dir,
        expose_feedback=not args.hidden_feedback, render_steps=args.render, allowed_tools=allowed_tools,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _direct_provider(task):
    return [{"action": "final", "arguments": {"answer": task.metadata.get("direct_answer", "unknown")}}]


def _oracle_final_provider(task):
    return [{"action": "final", "arguments": {"answer": task.answer}}]


if __name__ == "__main__":
    main()
