"""Run a compact GeoClaw midpoint episode."""

from __future__ import annotations

import json

from geosym_world_geometry.geoclaw import GeoClawEnvironment
from geosym_world_geometry.geoclaw.datasets import pilot_task


def main() -> None:
    environment = GeoClawEnvironment(pilot_task("construct_midpoint"), output_dir="geosym_world_geometry/outputs/geoclaw_sample")
    actions = [
        {"thought": "Construct the midpoint.", "action": "construct", "arguments": {"operation": "midpoint", "id": "M", "params": {"of": ["A", "B"]}}},
        {"thought": "Check the construction.", "action": "verify", "arguments": {"claim": {"type": "midpoint", "object": "M"}}},
        {"action": "final", "arguments": {"answer": "4"}},
    ]
    environment.replay(actions)
    print(json.dumps(environment.trajectory.to_dict(include_private=False), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
