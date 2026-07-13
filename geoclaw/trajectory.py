"""Replayable GeoClaw episode logging."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class TrajectoryStep:
    step_id: int
    raw_action: Dict[str, Any]
    parsed_action: Dict[str, Any] | None
    feedback: Dict[str, Any]
    state_before: Dict[str, Any]
    state_after: Dict[str, Any]
    audit: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "raw_action": self.raw_action,
            "parsed_action": self.parsed_action,
            "feedback": self.feedback,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "audit": self.audit,
        }


@dataclass
class EpisodeTrajectory:
    episode_id: str
    task_id: str
    environment_config: Dict[str, Any]
    steps: list[TrajectoryStep] = field(default_factory=list)
    final_result: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    termination: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_private: bool = True) -> Dict[str, Any]:
        steps = [step.to_dict() for step in self.steps]
        if not include_private:
            for step in steps:
                step.pop("state_before", None)
                step.pop("state_after", None)
                step.pop("audit", None)
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "environment_config": self.environment_config,
            "steps": steps,
            "final_result": self.final_result,
            "metrics": self.metrics,
            "termination": self.termination,
        }

    def write_json(self, path: str | Path, include_private: bool = True) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(include_private=include_private), ensure_ascii=False, indent=2), encoding="utf-8")
