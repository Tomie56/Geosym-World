"""Render built-in pilot samples to SVG and/or PNG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geosym_world.construction.builder import ConstructionBuilder
from geosym_world.datasets.layout import artifact_root
from geosym_world.rendering import state_to_render_scene, write_png, write_svg
from geosym_world.datasets.samples import built_in_pilot_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(artifact_root() / "renders" / "pilot"))
    parser.add_argument("--format", choices=["svg", "png", "both"], default="svg")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    samples = built_in_pilot_samples()
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    rendered: list[dict] = []
    for sample in samples:
        state = ConstructionBuilder().execute(sample.trace)
        scene = state_to_render_scene(state, scene_id=sample.id)
        item = {"id": sample.id, "formats": []}
        if args.format in {"svg", "both"}:
            svg_path = output_dir / f"{sample.id}.svg"
            write_svg(scene, svg_path, width=args.width, height=args.height)
            item["formats"].append({"format": "svg", "path": str(svg_path)})
        if args.format in {"png", "both"}:
            png_path = output_dir / f"{sample.id}.png"
            write_png(scene, png_path, width=args.width, height=args.height)
            item["formats"].append({"format": "png", "path": str(png_path)})
        rendered.append(item)
    print(json.dumps({"output_dir": str(output_dir), "num_rendered": len(rendered), "records": rendered}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
