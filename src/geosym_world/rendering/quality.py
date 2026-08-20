"""Auditable quality checks for renderer-neutral geometry scenes."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from .label_layout import Box, _obstacles, _overlap_area, _segment_intersects_box, position_labels
from .layout import ViewTransform, _primitive_points, build_view_transform
from .scene import RenderScene


@dataclass(frozen=True)
class RenderQualityIssue:
    code: str
    severity: str
    message: str
    object_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "object_ids": list(self.object_ids),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RenderQualityReport:
    valid: bool
    issues: tuple[RenderQualityIssue, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "issues": [issue.to_dict() for issue in self.issues],
            "metrics": dict(self.metrics),
        }


def inspect_render_scene(
    scene: RenderScene,
    *,
    width: int = 768,
    height: int = 768,
    padding: int = 48,
) -> RenderQualityReport:
    """Check scene geometry and the shared pixel-space label layout."""

    transform = build_view_transform(scene, width=width, height=height, padding=padding)
    positioned = position_labels(scene, transform)
    point_boxes, marker_boxes, hole_boxes, segments = _obstacles(scene.primitives, transform)
    issues: list[RenderQualityIssue] = []
    viewport = (0.0, 0.0, float(width), float(height))

    for primitive in scene.primitives:
        for point in _primitive_points(primitive):
            if not all(math.isfinite(float(value)) for value in point):
                issues.append(RenderQualityIssue("primitive.non_finite_coordinate", "error", "Primitive contains a non-finite coordinate", (primitive.id,)))
                break

    for label in positioned:
        if not _contains(viewport, label.bbox):
            issues.append(
                RenderQualityIssue(
                    "label.out_of_bounds",
                    "error",
                    "Label extends outside the render viewport",
                    (label.label.id,),
                    {"bbox": list(label.bbox), "viewport": list(viewport)},
                )
            )
        if any(_overlap_area(label.bbox, obstacle) > 0 for obstacle in point_boxes):
            issues.append(RenderQualityIssue("label.overlaps_point", "error", "Label obscures a point marker", (label.label.id,)))
        if any(_overlap_area(label.bbox, obstacle) > 0 for obstacle in marker_boxes):
            issues.append(RenderQualityIssue("label.overlaps_marker", "error", "Label obscures an angle or right-angle marker", (label.label.id,)))
        if any(_overlap_area(label.bbox, obstacle) > 0 for obstacle in hole_boxes):
            issues.append(RenderQualityIssue("label.overlaps_hole", "error", "Label obscures a transparent region hole", (label.label.id,)))
        if any(_segment_intersects_box(segment, label.bbox) for segment in segments):
            issues.append(RenderQualityIssue("label.crosses_geometry", "error", "Geometry crosses the label text region", (label.label.id,)))

    for index, left in enumerate(positioned):
        for right in positioned[index + 1 :]:
            overlap = _overlap_area(left.bbox, right.bbox)
            if overlap > 0:
                issues.append(
                    RenderQualityIssue(
                        "label.overlap",
                        "error",
                        "Labels occupy overlapping pixel regions",
                        (left.label.id, right.label.id),
                        {"overlap_area": round(overlap, 3)},
                    )
                )

    display_labels = [label.text for label in scene.labels if label.id.startswith("label:")]
    duplicates = sorted({text for text in display_labels if display_labels.count(text) > 1})
    if duplicates:
        issues.append(
            RenderQualityIssue(
                "label.duplicate_display_name",
                "error",
                "Point labels are not unique",
                metadata={"labels": duplicates},
            )
        )

    metrics = {
        "num_primitives": len(scene.primitives),
        "num_labels": len(positioned),
        "num_errors": sum(issue.severity == "error" for issue in issues),
        "num_warnings": sum(issue.severity == "warning" for issue in issues),
        "viewport": {"width": width, "height": height, "padding": padding},
    }
    return RenderQualityReport(valid=not any(issue.severity == "error" for issue in issues), issues=tuple(issues), metrics=metrics)


def _contains(outer: Box, inner: Box) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]
