"""Export a tiny built-in pilot benchmark JSONL."""

from __future__ import annotations

from pathlib import Path
import argparse
import json

from geosym_world_geometry import PilotBenchmarkExporter


def main() -> None:
    parser = argparse.ArgumentParser(description="Export built-in GeoSym world pilot records.")
    parser.add_argument(
        "--output",
        default=str(Path("geosym_world_geometry") / "outputs" / "pilot_v0.jsonl"),
        help="Output JSONL path. Keep it under geosym_world_geometry for this prototype.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    if "geosym_world_geometry" not in output.parts:
        raise SystemExit("Refusing to write outside geosym_world_geometry/")
    summary = PilotBenchmarkExporter().write_jsonl(output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
