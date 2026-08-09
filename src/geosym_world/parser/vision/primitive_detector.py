"""Script-side visual primitive candidates for geometry diagrams.

These candidates use pixel coordinates only as visual evidence. They are not
GeoSym mathematical coordinates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import math

import cv2
import numpy as np


class VisualPrimitiveDetector:
    def __init__(self, max_lines: int = 80, max_circles: int = 20):
        self.max_lines = max_lines
        self.max_circles = max_circles

    def detect(self, image_path: str) -> Dict[str, Any]:
        path = Path(image_path)
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return {
                "image_path": image_path,
                "image_size": None,
                "lines": [],
                "circles": [],
                "text_regions": [],
                "error": "image_read_failed",
            }
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = self._detect_lines(edges)
        circles = self._detect_circles(gray)
        text_regions = self._detect_text_like_regions(gray)
        topology = self._derive_topology(lines, circles)
        return {
            "image_path": image_path,
            "image_size": {"width": int(img.shape[1]), "height": int(img.shape[0])},
            "lines": lines,
            "circles": circles,
            "text_regions": text_regions,
            "topology": topology,
        }

    def _detect_lines(self, edges: np.ndarray) -> List[Dict[str, Any]]:
        raw = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=35,
            minLineLength=25,
            maxLineGap=8,
        )
        if raw is None:
            return []
        candidates = []
        for idx, line in enumerate(raw[:, 0, :]):
            x1, y1, x2, y2 = [int(v) for v in line]
            length = math.hypot(x2 - x1, y2 - y1)
            if length < 20:
                continue
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            candidates.append({
                "id": f"vl_{idx}",
                "p1": [x1, y1],
                "p2": [x2, y2],
                "length_px": round(length, 3),
                "angle_deg": round(angle, 3),
            })
        candidates.sort(key=lambda item: item["length_px"], reverse=True)
        return candidates[: self.max_lines]

    def _detect_circles(self, gray: np.ndarray) -> List[Dict[str, Any]]:
        blurred = cv2.medianBlur(gray, 5)
        raw = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=80,
            param2=25,
            minRadius=12,
            maxRadius=0,
        )
        if raw is None:
            return []
        circles = []
        for idx, c in enumerate(np.round(raw[0, :]).astype("int")):
            x, y, r = [int(v) for v in c]
            circles.append({"id": f"vc_{idx}", "center": [x, y], "radius_px": r})
        circles.sort(key=lambda item: item["radius_px"], reverse=True)
        return circles[: self.max_circles]

    def _detect_text_like_regions(self, gray: np.ndarray) -> List[Dict[str, Any]]:
        # Conservative connected-component candidates. OCR is intentionally not
        # performed here; VLM binds labels/numbers to these visual regions.
        _, binary = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        regions = []
        h, w = gray.shape[:2]
        for idx in range(1, num):
            x, y, bw, bh, area = [int(v) for v in stats[idx]]
            if area < 12 or area > 0.08 * h * w:
                continue
            if bw < 3 or bh < 6:
                continue
            aspect = bw / max(bh, 1)
            if aspect > 12:
                continue
            regions.append({
                "id": f"vt_{idx}",
                "bbox": [x, y, bw, bh],
                "area_px": area,
            })
        regions.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
        return regions[:120]

    def _derive_topology(self, lines: List[Dict[str, Any]], circles: List[Dict[str, Any]]) -> Dict[str, Any]:
        endpoints = []
        for line in lines:
            endpoints.append({"id": f"{line['id']}_p1", "line_id": line["id"], "xy": line["p1"]})
            endpoints.append({"id": f"{line['id']}_p2", "line_id": line["id"], "xy": line["p2"]})

        intersections = []
        for i, a in enumerate(lines[:40]):
            for b in lines[i + 1 : 40]:
                point = self._segment_intersection(a["p1"], a["p2"], b["p1"], b["p2"])
                if point is None:
                    continue
                intersections.append({
                    "id": f"vx_{len(intersections)}",
                    "line_ids": [a["id"], b["id"]],
                    "xy": [round(point[0], 3), round(point[1], 3)],
                })
                if len(intersections) >= 80:
                    break
            if len(intersections) >= 80:
                break

        circle_line_hits = []
        for circle in circles[:10]:
            cx, cy = circle["center"]
            r = circle["radius_px"]
            for line in lines[:40]:
                dist = self._point_segment_distance((cx, cy), line["p1"], line["p2"])
                if dist <= max(4.0, 0.08 * r):
                    circle_line_hits.append({
                        "id": f"vcl_{len(circle_line_hits)}",
                        "circle_id": circle["id"],
                        "line_id": line["id"],
                        "distance_px": round(dist, 3),
                    })
                    if len(circle_line_hits) >= 80:
                        break
            if len(circle_line_hits) >= 80:
                break

        return {
            "endpoints": endpoints[:160],
            "line_intersections": intersections,
            "circle_line_hits": circle_line_hits,
        }

    def _segment_intersection(
        self,
        p1: List[int],
        p2: List[int],
        p3: List[int],
        p4: List[int],
        eps: float = 1e-6,
    ) -> tuple[float, float] | None:
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < eps:
            return None
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
        if not (min(x1, x2) - 3 <= px <= max(x1, x2) + 3):
            return None
        if not (min(y1, y2) - 3 <= py <= max(y1, y2) + 3):
            return None
        if not (min(x3, x4) - 3 <= px <= max(x3, x4) + 3):
            return None
        if not (min(y3, y4) - 3 <= py <= max(y3, y4) + 3):
            return None
        return px, py

    def _point_segment_distance(self, point: tuple[int, int], p1: List[int], p2: List[int]) -> float:
        px, py = point
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(px - x1, py - y1)
        t = ((px - x1) * dx + (py - y1) * dy) / float(dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        qx = x1 + t * dx
        qy = y1 + t * dy
        return math.hypot(px - qx, py - qy)
