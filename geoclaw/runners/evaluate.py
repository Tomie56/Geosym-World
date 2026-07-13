"""Score serialized GeoClaw trajectories."""

from __future__ import annotations

import json
from pathlib import Path

from ..scoring import score_trajectory
from ..task import GeoClawTask


def score_trajectory_file(task: GeoClawTask, path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return score_trajectory(task, payload.get("steps", []))
