"""Replay a JSONL action sequence on a built-in GeoClaw task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geosym_world_geometry.geoclaw.datasets import built_in_pilot_tasks, pilot_task
from geosym_world_geometry.geoclaw.runners import load_actions, replay_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="construct_midpoint")
    parser.add_argument("--actions")
    parser.add_argument("--output-dir", default="geosym_world_geometry/outputs/geoclaw")
    parser.add_argument("--hidden-feedback", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--allowed-tools", help="Comma-separated tool allowlist; final is always enabled")
    parser.add_argument("--list-tasks", action="store_true")
    args = parser.parse_args()
    if args.list_tasks:
        print(json.dumps([{"id": task.id, "category": task.metadata.get("category"), "problem": task.problem_text} for task in built_in_pilot_tasks()], ensure_ascii=False, indent=2))
        return
    actions = load_actions(args.actions) if args.actions else _default_actions(args.task)
    allowed_tools = tuple(item.strip() for item in args.allowed_tools.split(",") if item.strip()) if args.allowed_tools else None
    environment = replay_task(pilot_task(args.task), actions, args.output_dir, expose_feedback=not args.hidden_feedback, render_steps=not args.no_render, allowed_tools=allowed_tools)
    trajectory_path = Path(args.output_dir) / args.task / "trajectory.json"
    environment.trajectory.write_json(trajectory_path)
    print(json.dumps({"observation": environment.observe().to_dict(), "metrics": environment.trajectory.metrics, "trajectory": str(trajectory_path)}, ensure_ascii=False, indent=2))


def _default_actions(task_id: str) -> list[dict]:
    if task_id == "construct_midpoint":
        return [
            {"action": "construct", "arguments": {"operation": "midpoint", "id": "M", "params": {"of": ["A", "B"]}}},
            {"action": "verify", "arguments": {"claim": {"type": "midpoint", "object": "M"}}},
            {"action": "final", "arguments": {"answer": "4"}},
        ]
    return [{"action": "final", "arguments": {"answer": pilot_task(task_id).answer}}]


if __name__ == "__main__":
    main()
