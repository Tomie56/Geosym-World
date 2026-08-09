"""Export a tiny built-in pilot benchmark JSONL."""

from __future__ import annotations

from pathlib import Path
import argparse
import json

from geosym_world import PilotBenchmarkExporter
from geosym_world.datasets.layout import artifact_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Export built-in GeoSym world pilot records.")
    parser.add_argument(
        "--output",
        default=str(artifact_root() / "generated" / "pilot_v0.jsonl"),
        help="Output JSONL path.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    summary = PilotBenchmarkExporter().write_jsonl(output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
