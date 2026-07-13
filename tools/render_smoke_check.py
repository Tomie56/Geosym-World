"""Small render smoke check for SVG and OpenCV PNG outputs."""

from __future__ import annotations

import json
from pathlib import Path

from geosym_world_geometry.builder import ConstructionBuilder
from geosym_world_geometry.rendering import state_to_render_scene, write_png, write_svg
from geosym_world_geometry.samples import built_in_pilot_samples

SMOKE_SAMPLE_IDS = {
    "pilot_angle_triangle_area",
    "pilot_circle_sector_area",
    "pilot_3d_auto_section_area",
    "pilot_3d_space_angle",
    "pilot_tangent_chord_angle",
}


def main() -> None:
    output_dir = Path("geosym_world_geometry/outputs/render_smoke")
    samples = [sample for sample in built_in_pilot_samples() if sample.id in SMOKE_SAMPLE_IDS]
    if len(samples) != len(SMOKE_SAMPLE_IDS):
        found = {sample.id for sample in samples}
        raise AssertionError(f"Missing render smoke samples: {sorted(SMOKE_SAMPLE_IDS.difference(found))}")

    rendered: list[dict] = []
    for sample in samples:
        state = ConstructionBuilder().execute(sample.trace)
        scene = state_to_render_scene(state, scene_id=sample.id)
        if not scene.primitives:
            raise AssertionError(f"{sample.id} produced an empty render scene")
        marker_count = sum(1 for primitive in scene.primitives if primitive.role == "marker")
        svg_path = output_dir / f"{sample.id}.svg"
        png_path = output_dir / f"{sample.id}.png"
        write_svg(scene, svg_path)
        write_png(scene, png_path)
        _assert_svg(svg_path)
        _assert_png(png_path)
        rendered.append(
            {
                "id": sample.id,
                "svg": str(svg_path),
                "png": str(png_path),
                "num_primitives": len(scene.primitives),
                "num_labels": len(scene.labels),
                "num_markers": marker_count,
            }
        )
    if not any(item["num_markers"] for item in rendered):
        raise AssertionError("Render smoke did not exercise any geometry markers")
    print(json.dumps({"num_samples": len(rendered), "rendered": rendered}, ensure_ascii=False, indent=2))


def _assert_svg(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "<svg" not in text or "</svg>" not in text:
        raise AssertionError(f"Invalid SVG output: {path}")
    if "data-id=" not in text:
        raise AssertionError(f"SVG output has no rendered geometry ids: {path}")


def _assert_png(path: Path) -> None:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise AssertionError("OpenCV is required for PNG smoke check") from exc
    image = cv2.imread(str(path))
    if image is None:
        raise AssertionError(f"Invalid PNG output: {path}")
    if image.shape[0] < 100 or image.shape[1] < 100:
        raise AssertionError(f"PNG output is unexpectedly small: {path} -> {image.shape}")
    nonwhite = np.count_nonzero(np.any(image < 245, axis=2))
    if nonwhite < 200:
        raise AssertionError(f"PNG output appears blank: {path}, nonwhite={nonwhite}")


if __name__ == "__main__":
    main()
