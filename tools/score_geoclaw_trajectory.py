"""Score one stored GeoClaw trajectory."""

from __future__ import annotations

import argparse
import json

from geosym_world_geometry.geoclaw.datasets import pilot_task
from geosym_world_geometry.geoclaw.runners import score_trajectory_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--trajectory", required=True)
    args = parser.parse_args()
    print(json.dumps(score_trajectory_file(pilot_task(args.task), args.trajectory), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
