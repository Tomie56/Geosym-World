"""Render script-detected visual candidates on top of source diagrams."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np


class VisualOverlayRenderer:
    def render(self, image_path: str, visual_state: Dict[str, Any], output_path: str) -> str:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            size = visual_state.get("image_size") or {}
            w = int(size.get("width") or 640)
            h = int(size.get("height") or 480)
            img = np.full((h, w, 3), 255, dtype=np.uint8)

        overlay = img.copy()
        for line in visual_state.get("lines") or []:
            p1 = tuple(int(v) for v in line.get("p1", [0, 0]))
            p2 = tuple(int(v) for v in line.get("p2", [0, 0]))
            cv2.line(overlay, p1, p2, (30, 120, 255), 2, cv2.LINE_AA)
            self._label(overlay, line.get("id", ""), p1, (30, 120, 255))

        for circle in visual_state.get("circles") or []:
            center = tuple(int(v) for v in circle.get("center", [0, 0]))
            radius = int(circle.get("radius_px") or 0)
            if radius > 0:
                cv2.circle(overlay, center, radius, (60, 170, 60), 2, cv2.LINE_AA)
                self._label(overlay, circle.get("id", ""), center, (60, 170, 60))

        for region in visual_state.get("text_regions") or []:
            x, y, w, h = [int(v) for v in region.get("bbox", [0, 0, 0, 0])]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (180, 80, 180), 1, cv2.LINE_AA)
            self._label(overlay, region.get("id", ""), (x, y), (180, 80, 180))

        topology = visual_state.get("topology") or {}
        for point in topology.get("line_intersections") or []:
            x, y = [int(round(float(v))) for v in point.get("xy", [0, 0])]
            cv2.drawMarker(overlay, (x, y), (0, 0, 220), cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
            self._label(overlay, point.get("id", ""), (x + 4, y + 4), (0, 0, 220))

        for hit in topology.get("circle_line_hits") or []:
            text = f"{hit.get('id')}:{hit.get('circle_id')}/{hit.get('line_id')}"
            self._label(overlay, text, (8, 22 + 18 * min(20, len(text))), (70, 70, 70))

        self._render_visual_structure(overlay, visual_state.get("visual_structure_candidates") or {})

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, overlay)
        return output_path

    def _render_visual_structure(self, overlay: np.ndarray, structure: Dict[str, Any]) -> None:
        for segment in structure.get("segment_candidates") or []:
            p1 = segment.get("p1_px")
            p2 = segment.get("p2_px")
            if not (isinstance(p1, list) and isinstance(p2, list) and len(p1) >= 2 and len(p2) >= 2):
                continue
            a = tuple(int(round(float(v))) for v in p1[:2])
            b = tuple(int(round(float(v))) for v in p2[:2])
            cv2.line(overlay, a, b, (255, 80, 30), 1, cv2.LINE_AA)
            mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
            self._label(overlay, segment.get("id", ""), (mid[0] + 4, mid[1] - 4), (255, 80, 30), scale=0.34)

        for point in structure.get("point_candidates") or []:
            xy = point.get("xy_px")
            if not (isinstance(xy, list) and len(xy) >= 2):
                continue
            pos = tuple(int(round(float(v))) for v in xy[:2])
            cv2.circle(overlay, pos, 3, (20, 20, 230), -1, cv2.LINE_AA)
            self._label(overlay, point.get("id", ""), (pos[0] + 4, pos[1] + 10), (20, 20, 230), scale=0.3)

        for angle in structure.get("angle_candidates") or []:
            xy = angle.get("xy_px")
            if not (isinstance(xy, list) and len(xy) >= 2):
                continue
            pos = tuple(int(round(float(v))) for v in xy[:2])
            cv2.drawMarker(overlay, pos, (40, 170, 210), cv2.MARKER_TILTED_CROSS, 10, 1, cv2.LINE_AA)
            self._label(overlay, angle.get("id", ""), (pos[0] + 8, pos[1] - 8), (40, 170, 210), scale=0.32)

        for region in structure.get("label_region_candidates") or []:
            bbox = region.get("bbox")
            if not (isinstance(bbox, list) and len(bbox) >= 4):
                continue
            x, y, w, h = [int(round(float(v))) for v in bbox[:4]]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (120, 40, 210), 1, cv2.LINE_AA)
            self._label(overlay, region.get("id", ""), (x + 2, y + h + 10), (120, 40, 210), scale=0.3)

    def _label(self, img: np.ndarray, text: Any, xy: tuple[int, int], color: tuple[int, int, int], scale: float = 0.42) -> None:
        label = str(text or "")
        if not label:
            return
        x, y = xy
        y = max(12, y)
        cv2.putText(img, label[:32], (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
