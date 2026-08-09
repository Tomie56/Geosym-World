"""Renderer-neutral scene schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

Point2D = Tuple[float, float]


@dataclass(frozen=True)
class RenderStyle:
    stroke: str = "#1f2937"
    fill: str = "none"
    stroke_width: float = 2.0
    opacity: float = 1.0
    dasharray: Tuple[float, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stroke": self.stroke,
            "fill": self.fill,
            "stroke_width": self.stroke_width,
            "opacity": self.opacity,
            "dasharray": list(self.dasharray),
        }


@dataclass(frozen=True)
class RenderPrimitive:
    id: str
    kind: str
    geometry: Dict[str, Any]
    style: RenderStyle = field(default_factory=RenderStyle)
    role: str = "object"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "geometry": self.geometry,
            "style": self.style.to_dict(),
            "role": self.role,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class RenderLabel:
    id: str
    text: str
    position: Point2D
    offset: Point2D = (8.0, -8.0)
    style: RenderStyle = field(default_factory=lambda: RenderStyle(stroke="none", fill="#111827", stroke_width=0.0))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "position": list(self.position),
            "offset": list(self.offset),
            "style": self.style.to_dict(),
        }


@dataclass(frozen=True)
class RenderScene:
    id: str
    primitives: Tuple[RenderPrimitive, ...] = ()
    labels: Tuple[RenderLabel, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "primitives": [primitive.to_dict() for primitive in self.primitives],
            "labels": [label.to_dict() for label in self.labels],
            "metadata": self.metadata,
        }
