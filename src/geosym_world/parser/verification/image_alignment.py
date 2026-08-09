"""Model-free sketch-image alignment verification v0."""

from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np


class ImageAlignmentVerifier:
    """Compare two line-drawing images after bbox normalization."""

    def verify_paths(
        self,
        original_path: str,
        sketch_path: str,
        edge_iou_min: float = 0.65,
        chamfer_distance_max: float = 5.0,
    ) -> Dict[str, Any]:
        original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
        sketch = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)
        if original is None or sketch is None:
            return {"valid": False, "error": "image_read_failed"}
        return self.verify_arrays(original, sketch, edge_iou_min, chamfer_distance_max)

    def verify_arrays(
        self,
        original_gray: np.ndarray,
        sketch_gray: np.ndarray,
        edge_iou_min: float = 0.65,
        chamfer_distance_max: float = 5.0,
    ) -> Dict[str, Any]:
        original = self._binary_edges(original_gray)
        sketch = self._binary_edges(sketch_gray)
        sketch = cv2.resize(sketch, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
        inter = np.logical_and(original > 0, sketch > 0).sum()
        union = np.logical_or(original > 0, sketch > 0).sum()
        edge_iou = float(inter / union) if union else 0.0
        chamfer = self._chamfer(original, sketch)
        valid = edge_iou >= edge_iou_min or chamfer <= chamfer_distance_max
        return {
            "valid": valid,
            "metrics": {
                "edge_iou": round(edge_iou, 6),
                "chamfer_distance": round(chamfer, 6),
            },
            "thresholds": {
                "edge_iou_min": edge_iou_min,
                "chamfer_distance_max": chamfer_distance_max,
            },
        }

    def _binary_edges(self, gray: np.ndarray) -> np.ndarray:
        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        return binary

    def _chamfer(self, a: np.ndarray, b: np.ndarray) -> float:
        if not np.any(a) or not np.any(b):
            return float("inf")
        dist_b = cv2.distanceTransform((b == 0).astype(np.uint8), cv2.DIST_L2, 3)
        dist_a = cv2.distanceTransform((a == 0).astype(np.uint8), cv2.DIST_L2, 3)
        a_to_b = float(dist_b[a > 0].mean())
        b_to_a = float(dist_a[b > 0].mean())
        return (a_to_b + b_to_a) / 2.0
