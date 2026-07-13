"""Replay action JSONL against one GeoClaw task."""

from __future__ import annotations

import json
from pathlib import Path

from ..environment import GeoClawEnvironment
from ..task import GeoClawTask


def load_actions(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def replay_task(task: GeoClawTask, actions: list[dict], output_dir: str | Path, *, expose_feedback: bool = True, render_steps: bool = True, allowed_tools: tuple[str, ...] | None = None) -> GeoClawEnvironment:
    environment = GeoClawEnvironment(task, output_dir=output_dir, expose_feedback=expose_feedback, render_steps=render_steps, allowed_tools=allowed_tools)
    environment.replay(actions)
    return environment
