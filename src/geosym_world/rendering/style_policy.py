"""Deterministic render style randomization."""

from __future__ import annotations

import random
from dataclasses import replace

from .scene import RenderLabel, RenderPrimitive, RenderScene, RenderStyle


class DeterministicStylePolicy:
    """Small deterministic style perturbation policy for synthetic images."""

    def __init__(self, *, seed: int = 0, variant: str = "clean_jitter"):
        self.seed = seed
        self.variant = variant

    def apply(self, scene: RenderScene) -> RenderScene:
        if self.variant == "clean":
            return scene
        rng = random.Random(self.seed)
        primitives = tuple(self._primitive(primitive, rng) for primitive in scene.primitives)
        labels = tuple(self._label(label, rng) for label in scene.labels)
        metadata = {**scene.metadata, "style_policy": {"seed": self.seed, "variant": self.variant}}
        return replace(scene, primitives=primitives, labels=labels, metadata=metadata)

    def _primitive(self, primitive: RenderPrimitive, rng: random.Random) -> RenderPrimitive:
        style = primitive.style
        width_factor = rng.choice((0.9, 1.0, 1.08, 1.15))
        opacity = min(1.0, max(0.25, style.opacity * rng.choice((0.92, 1.0, 1.0, 1.08))))
        stroke = _jitter_color(style.stroke, rng) if style.stroke != "none" else style.stroke
        fill = _jitter_color(style.fill, rng, weak=True) if style.fill != "none" else style.fill
        new_style = replace(style, stroke=stroke, fill=fill, stroke_width=round(style.stroke_width * width_factor, 3), opacity=round(opacity, 3))
        return replace(primitive, style=new_style)

    def _label(self, label: RenderLabel, rng: random.Random) -> RenderLabel:
        dx = label.offset[0] + rng.choice((-2.0, 0.0, 2.0))
        dy = label.offset[1] + rng.choice((-2.0, 0.0, 2.0))
        return replace(label, offset=(dx, dy))


def _jitter_color(color: str, rng: random.Random, *, weak: bool = False) -> str:
    if not color.startswith("#") or len(color) != 7:
        return color
    amount = 8 if weak else 14
    channels = []
    for idx in (1, 3, 5):
        value = int(color[idx : idx + 2], 16)
        value = max(0, min(255, value + rng.randint(-amount, amount)))
        channels.append(f"{value:02x}")
    return "#" + "".join(channels)
