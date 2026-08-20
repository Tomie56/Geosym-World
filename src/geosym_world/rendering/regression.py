"""Compact cross-backend signatures for deterministic visual regression."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

from .png_renderer import render_png_rgb
from .scene import RenderScene, scene_fingerprint
from .svg_renderer import render_svg


RENDER_REGRESSION_SCHEMA_VERSION = "geosym.render_regression.v1"


def render_regression_signature(
    scene: RenderScene,
    *,
    width: int = 768,
    height: int = 768,
    padding: int = 48,
) -> dict[str, Any]:
    """Hash scene, SVG output, PNG pixels, and PNG foreground occupancy."""

    if width <= 0 or height <= 0:
        raise ValueError("Render regression viewport dimensions must be positive")
    if padding < 0 or padding * 2 >= min(width, height):
        raise ValueError("Render regression padding must leave a non-empty viewport")

    svg = render_svg(scene, width=width, height=height, padding=padding).encode("utf-8")
    rgb = render_png_rgb(scene, width=width, height=height, padding=padding)
    foreground = (rgb != 255).any(axis=2)
    return {
        "schema_version": RENDER_REGRESSION_SCHEMA_VERSION,
        "scene_fingerprint": scene_fingerprint(scene),
        "viewport": {"width": width, "height": height, "padding": padding},
        "svg_sha256": sha256(svg).hexdigest(),
        "png_rgb_sha256": sha256(rgb.tobytes(order="C")).hexdigest(),
        "png_foreground_mask_sha256": sha256(foreground.astype("uint8").tobytes(order="C")).hexdigest(),
        "foreground_pixels": int(foreground.sum()),
        "primitive_count": len(scene.primitives),
        "label_count": len(scene.labels),
    }
