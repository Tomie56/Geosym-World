"""Batch replay and manifest export for controlled GeoClaw experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

from ..scoring import aggregate_scores
from ..task import GeoClawTask
from .replay import replay_task


ActionProvider = Callable[[GeoClawTask], list[dict]]


def run_batch(
    tasks: Iterable[GeoClawTask],
    action_provider: ActionProvider,
    output_dir: str | Path,
    *,
    expose_feedback: bool = True,
    render_steps: bool = False,
    allowed_tools: tuple[str, ...] | None = None,
) -> dict:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    episodes = []
    scores = []
    for task in tasks:
        actions = action_provider(task)
        environment = replay_task(
            task,
            actions,
            root,
            expose_feedback=expose_feedback,
            render_steps=render_steps,
            allowed_tools=allowed_tools,
        )
        if not environment.terminated:
            environment.finish("actions_exhausted")
        trajectory_path = root / task.id / "trajectory.json"
        environment.trajectory.write_json(trajectory_path)
        episodes.append({
            "task_id": task.id,
            "category": task.metadata.get("category"),
            "difficulty": task.difficulty.to_dict(),
            "required_tool_sets": [list(items) for items in task.required_tool_sets],
            "visual_regions": [item.to_dict() for item in task.visual_regions],
            "trajectory": str(trajectory_path),
            "termination": environment.trajectory.termination,
            "metrics": environment.trajectory.metrics,
        })
        scores.append(environment.trajectory.metrics)
    manifest = {
        "num_tasks": len(episodes),
        "config": {
            "expose_feedback": expose_feedback,
            "render_steps": render_steps,
            "allowed_tools": list(allowed_tools) if allowed_tools else "all",
        },
        "aggregate": aggregate_scores(scores),
        "episodes": episodes,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
