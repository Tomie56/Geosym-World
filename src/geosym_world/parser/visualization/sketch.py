"""Simple model-free sketch renderer for GeoSymProblemState."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import math

import cv2
import numpy as np

from geosym_world.schema.problem_normalize import parse_expr
from geosym_world.schema.problem import GeoSymProblemState


class SketchRenderer:
    def __init__(self, canvas_size: tuple[int, int] = (640, 480), margin: int = 40):
        self.canvas_size = canvas_size
        self.margin = margin

    def render(self, state: GeoSymProblemState, output_path: str) -> str:
        width, height = self.canvas_size
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        points = self._point_values(state)
        if not points:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, canvas)
            return output_path
        normalize = self._normalizer(points)
        pixels = {pid: normalize(coord) for pid, coord in points.items()}
        for line in state.geometry.lines:
            start = line.get("start_point_id")
            end = line.get("end_point_id")
            if start in pixels and end in pixels:
                cv2.line(canvas, pixels[start], pixels[end], (0, 0, 0), 2, cv2.LINE_AA)
        for pid, coord in pixels.items():
            cv2.circle(canvas, coord, 4, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.putText(canvas, pid, (coord[0] + 6, coord[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, canvas)
        return output_path

    def _point_values(self, state: GeoSymProblemState) -> Dict[str, Tuple[float, float]]:
        out = {}
        for point in state.geometry.points:
            x = parse_expr((point.get("x") or {}).get("expr"))
            y = parse_expr((point.get("y") or {}).get("expr"))
            if x is not None and y is not None:
                try:
                    xf = float(x.evalf())
                    yf = float(y.evalf())
                except TypeError:
                    continue
                if math.isfinite(xf) and math.isfinite(yf):
                    out[point["id"]] = (xf, yf)
        return out

    def _normalizer(self, points: Dict[str, Tuple[float, float]]):
        width, height = self.canvas_size
        xs = [p[0] for p in points.values()]
        ys = [p[1] for p in points.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(max_x - min_x, 1e-6)
        span_y = max(max_y - min_y, 1e-6)
        scale = min((width - 2 * self.margin) / span_x, (height - 2 * self.margin) / span_y)
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2

        def normalize(point: Tuple[float, float]) -> Tuple[int, int]:
            return (
                int(width / 2 + (point[0] - cx) * scale),
                int(height / 2 - (point[1] - cy) * scale),
            )

        return normalize
