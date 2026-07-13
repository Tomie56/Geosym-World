"""Image zoom/crop tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ..feedback import INVALID_ARGUMENT, VALID
from .base import ToolResult


class ZoomTool:
    name = "zoom"

    def execute(self, environment: Any, arguments: Dict[str, Any]) -> ToolResult:
        region = arguments.get("region")
        scale = arguments.get("scale", 2.0)
        if not isinstance(region, list) or len(region) != 4:
            return ToolResult(INVALID_ARGUMENT, errors=[{"code": "zoom.invalid_region"}])
        try:
            x1, y1, x2, y2 = (float(item) for item in region)
            scale = float(scale)
        except (TypeError, ValueError):
            return ToolResult(INVALID_ARGUMENT, errors=[{"code": "zoom.invalid_numeric_value"}])
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1) or (x2 - x1) * (y2 - y1) < 0.0025 or scale < 1:
            return ToolResult(INVALID_ARGUMENT, errors=[{"code": "zoom.out_of_bounds_or_too_small"}])
        source = environment.current_image
        if source is None:
            return ToolResult(INVALID_ARGUMENT, errors=[{"code": "zoom.image_unavailable"}])
        try:
            import cv2
        except ImportError as exc:
            return ToolResult(INVALID_ARGUMENT, errors=[{"code": "zoom.opencv_unavailable", "message": str(exc)}])
        image = cv2.imread(str(source))
        if image is None:
            return ToolResult(INVALID_ARGUMENT, errors=[{"code": "zoom.image_unreadable", "path": str(source)}])
        height, width = image.shape[:2]
        left, top, right, bottom = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
        crop = image[top:bottom, left:right]
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        crop_id = f"crop_{len(environment.crops) + 1:03d}"
        output = Path(environment.output_dir) / "crops" / f"{crop_id}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), crop)
        record = {"id": crop_id, "path": str(output), "region": region, "pixel_region": [left, top, right, bottom], "scale": scale}
        overlaps = [_region_overlap(region, item) for item in environment.task.visual_regions]
        record["region_matches"] = [item for item in overlaps if item["iou"] > 0]
        record["best_region_iou"] = max((item["iou"] for item in overlaps), default=None)
        required = [item for item in overlaps if item["required"]]
        record["required_region_covered"] = any(item["coverage"] >= 0.5 for item in required) if required else None
        environment.crops.append(record)
        return ToolResult(VALID, result=record, image_changed=False)


def _region_overlap(selected: list[float], annotated) -> Dict[str, Any]:
    ax1, ay1, ax2, ay2 = annotated.region
    bx1, by1, bx2, by2 = selected
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height
    annotated_area = (ax2 - ax1) * (ay2 - ay1)
    selected_area = (bx2 - bx1) * (by2 - by1)
    union = annotated_area + selected_area - intersection
    return {"id": annotated.id, "purpose": annotated.purpose, "objects": list(annotated.objects), "required": annotated.required, "iou": 0.0 if union <= 0 else intersection / union, "coverage": 0.0 if annotated_area <= 0 else intersection / annotated_area}
