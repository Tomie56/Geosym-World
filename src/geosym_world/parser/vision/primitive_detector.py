"""Script-side visual primitive candidates for geometry diagrams.

These candidates use pixel coordinates only as visual evidence. They are not
GeoSym mathematical coordinates.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from .marker_detector import VisualMarkerDetector


VISUAL_PRIMITIVE_DETECTOR_VERSION = "0.8"


class VisualPrimitiveDetector:
    def __init__(
        self,
        max_lines: int = 80,
        max_circles: int = 20,
        max_arcs: int = 20,
        max_points: int = 80,
        max_fill_regions: int = 40,
        minimum_line_length_px: int = 25,
        minimum_line_length_ratio: float = 0.1,
        minimum_fill_area_ratio: float = 0.001,
        curve_text_overlap_threshold: float = 0.25,
        marker_detector: VisualMarkerDetector | None = None,
    ):
        if max_lines <= 0 or max_circles <= 0 or max_arcs <= 0 or max_points <= 0 or max_fill_regions <= 0:
            raise ValueError("Visual candidate limits must be positive")
        if minimum_line_length_px <= 0:
            raise ValueError("minimum_line_length_px must be positive")
        if not 0.0 <= minimum_line_length_ratio < 1.0:
            raise ValueError("minimum_line_length_ratio must be in [0, 1)")
        if not 0.0 < minimum_fill_area_ratio < 1.0:
            raise ValueError("minimum_fill_area_ratio must be in (0, 1)")
        if not 0.0 <= curve_text_overlap_threshold <= 1.0:
            raise ValueError("curve_text_overlap_threshold must be in [0, 1]")
        self.max_lines = max_lines
        self.max_circles = max_circles
        self.max_arcs = max_arcs
        self.max_points = max_points
        self.max_fill_regions = max_fill_regions
        self.minimum_line_length_px = minimum_line_length_px
        self.minimum_line_length_ratio = minimum_line_length_ratio
        self.minimum_fill_area_ratio = minimum_fill_area_ratio
        self.curve_text_overlap_threshold = curve_text_overlap_threshold
        self.marker_detector = marker_detector or VisualMarkerDetector()

    def configuration(self) -> Dict[str, Any]:
        return {
            "backend": "opencv_hough",
            "version": VISUAL_PRIMITIVE_DETECTOR_VERSION,
            "max_lines": self.max_lines,
            "max_circles": self.max_circles,
            "max_arcs": self.max_arcs,
            "max_points": self.max_points,
            "max_fill_regions": self.max_fill_regions,
            "minimum_line_length_px": self.minimum_line_length_px,
            "minimum_line_length_ratio": self.minimum_line_length_ratio,
            "minimum_fill_area_ratio": self.minimum_fill_area_ratio,
            "curve_text_overlap_threshold": self.curve_text_overlap_threshold,
            "marker_detector": self.marker_detector.configuration(),
        }

    def detect(self, image_path: str) -> Dict[str, Any]:
        path = Path(image_path)
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return {
                "image_path": image_path,
                "image_size": None,
                "lines": [],
                "circles": [],
                "arcs": [],
                "fill_regions": [],
                "markers": [],
                "text_regions": [],
                "error": "image_read_failed",
            }
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = self._detect_lines(edges)
        circles = self._detect_circles(gray, edges)
        arcs = self._detect_arcs(edges, circles)
        fill_regions = self._detect_fill_regions(img)
        text_regions = self._detect_text_like_regions(gray)
        circles, arcs, curve_filter = self._filter_curves_by_text(circles, arcs, text_regions, gray.shape)
        topology = self._derive_topology(lines, circles, arcs)
        topology["point_candidates"] = self._detect_points(gray, topology, text_regions, circles, lines)
        markers = self.marker_detector.detect(gray, lines, topology)
        return {
            "image_path": image_path,
            "image_size": {"width": int(img.shape[1]), "height": int(img.shape[0])},
            "lines": lines,
            "circles": circles,
            "arcs": arcs,
            "fill_regions": fill_regions,
            "markers": markers,
            "text_regions": text_regions,
            "topology": topology,
            "curve_filter": curve_filter,
        }

    def _filter_curves_by_text(
        self,
        circles: Sequence[Mapping[str, Any]],
        arcs: Sequence[Mapping[str, Any]],
        text_regions: Sequence[Mapping[str, Any]],
        image_shape: Sequence[int],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        image_area = float(image_shape[0] * image_shape[1])
        usable_text_boxes = []
        for region in text_regions:
            bbox = _xywh_bbox(region.get("bbox"))
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            if (x2 - x1) * (y2 - y1) > 0.05 * image_area:
                continue
            usable_text_boxes.append(bbox)

        rejected: List[Dict[str, Any]] = []

        def keep(candidate: Mapping[str, Any], kind: str) -> bool:
            bbox = _curve_bbox(candidate, kind)
            if bbox is None:
                return True
            overlap = _bbox_union_intersection_area(bbox, usable_text_boxes)
            candidate_area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            ratio = overlap / candidate_area
            if ratio <= self.curve_text_overlap_threshold:
                return True
            rejected.append({
                "id": str(candidate.get("id") or ""),
                "kind": kind,
                "text_overlap_ratio": round(ratio, 6),
                "reason": "curve_bbox_overlaps_text_regions",
            })
            return False

        filtered_circles = [dict(item) for item in circles if keep(item, "circle")]
        filtered_arcs = [dict(item) for item in arcs if keep(item, "arc")]
        return filtered_circles, filtered_arcs, {
            "threshold": self.curve_text_overlap_threshold,
            "text_regions_considered": len(usable_text_boxes),
            "input_circles": len(circles),
            "output_circles": len(filtered_circles),
            "input_arcs": len(arcs),
            "output_arcs": len(filtered_arcs),
            "rejected": rejected,
        }

    def _detect_lines(self, edges: np.ndarray) -> List[Dict[str, Any]]:
        minimum_length = max(
            self.minimum_line_length_px,
            int(round(min(edges.shape[:2]) * self.minimum_line_length_ratio)),
        )
        raw = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=35,
            minLineLength=minimum_length,
            maxLineGap=8,
        )
        if raw is None:
            return []
        candidates = []
        for idx, line in enumerate(raw[:, 0, :]):
            x1, y1, x2, y2 = [int(v) for v in line]
            length = math.hypot(x2 - x1, y2 - y1)
            if length < minimum_length:
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
        return self._merge_collinear_lines(candidates)[: self.max_lines]

    def _detect_circles(self, gray: np.ndarray, edges: np.ndarray) -> List[Dict[str, Any]]:
        blurred = cv2.medianBlur(gray, 5)
        height, width = gray.shape[:2]
        minimum_radius = max(12, int(min(height, width) * 0.04))
        raw = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=80,
            param2=25,
            minRadius=minimum_radius,
            maxRadius=0,
        )
        if raw is None:
            return []
        circles = []
        for idx, c in enumerate(np.round(raw[0, :]).astype("int")):
            x, y, r = [int(v) for v in c]
            if x - r < 0 or y - r < 0 or x + r >= width or y + r >= height:
                continue
            support = self._circle_edge_support(edges, x, y, r)
            if support < 0.75:
                continue
            circles.append(
                {
                    "id": f"vc_{idx}",
                    "center": [x, y],
                    "radius_px": r,
                    "edge_support": round(support, 3),
                }
            )
        circles.sort(key=lambda item: item["radius_px"], reverse=True)
        return circles[: self.max_circles]

    def _detect_arcs(
        self,
        edges: np.ndarray,
        circles: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        candidates = []
        for contour in contours:
            candidate = _fit_arc_contour(contour, edges.shape)
            if candidate is None:
                continue
            support_circle = _match_support_circle(candidate, circles)
            if (
                support_circle is not None
                and float(candidate["sweep_deg"]) > 240.0
                and float(support_circle.get("edge_support", 0.0)) >= 0.85
            ):
                continue
            candidate["circle_id"] = support_circle["id"] if support_circle is not None else None
            candidates.append(candidate)
        candidates.sort(key=lambda item: (-float(item["edge_support"]), float(item["residual_px"]), item["id"]))
        return _deduplicate_arcs(candidates)[: self.max_arcs]

    def _detect_fill_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Fill-region detection requires a BGR image")
        color = image.astype(np.int16)
        channel_min = color.min(axis=2)
        channel_max = color.max(axis=2)
        chroma = channel_max - channel_min
        white_distance = (255 - color).max(axis=2)
        mask = ((channel_min >= 120) & (chroma >= 4) & (white_distance >= 8)).astype(np.uint8)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )
        minimum_area = max(
            64,
            int(round(image.shape[0] * image.shape[1] * self.minimum_fill_area_ratio)),
        )
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        candidates = []
        for component_index in range(1, component_count):
            x, y, width, height, area = [int(value) for value in stats[component_index]]
            bbox_area = width * height
            if area < minimum_area or min(width, height) < 8 or area / max(1, bbox_area) < 0.15:
                continue
            component_mask = np.where(labels == component_index, 255, 0).astype(np.uint8)
            contours, hierarchy = cv2.findContours(component_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if not contours or hierarchy is None:
                continue
            parents = hierarchy[0, :, 3]
            outer_indices = [index for index, parent in enumerate(parents) if parent == -1]
            if not outer_indices:
                continue
            outer_index = max(outer_indices, key=lambda index: cv2.contourArea(contours[index]))
            outer_contour = _simplify_contour(contours[outer_index])
            if len(outer_contour) < 3:
                continue
            minimum_gap_area = max(64.0, area * 0.02)
            interior_gaps = [
                _simplify_contour(contour)
                for index, contour in enumerate(contours)
                if parents[index] == outer_index and cv2.contourArea(contour) >= minimum_gap_area
            ]
            component_pixels = image[labels == component_index]
            mean_rgb = component_pixels[:, ::-1].mean(axis=0)
            moments = cv2.moments(contours[outer_index])
            centroid = (
                [x + width / 2.0, y + height / 2.0]
                if abs(moments["m00"]) <= 1e-9
                else [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]
            )
            density = area / bbox_area
            mean_chroma = float(chroma[labels == component_index].mean())
            confidence = min(
                0.95,
                0.5
                + 0.15 * min(1.0, density)
                + 0.15 * min(1.0, area / (4.0 * minimum_area))
                + 0.15 * min(1.0, mean_chroma / 30.0),
            )
            candidates.append(
                {
                    "id": f"vf_{component_index}",
                    "bbox": [x, y, width, height],
                    "outer_contour": outer_contour,
                    "interior_gaps": [gap for gap in interior_gaps if len(gap) >= 3],
                    "area_px": area,
                    "fill_fraction": round(density, 4),
                    "centroid": [round(value, 3) for value in centroid],
                    "mean_rgb": [int(round(value)) for value in mean_rgb],
                    "confidence": round(confidence, 3),
                    "score_kind": "light_chromatic_fill_component",
                    "appearance_semantics": "observed_fill_not_shading_label",
                }
            )
        candidates.sort(key=lambda item: item["area_px"], reverse=True)
        return candidates[: self.max_fill_regions]

    def _detect_points(
        self,
        gray: np.ndarray,
        topology: Mapping[str, Any],
        text_regions: Sequence[Mapping[str, Any]],
        circles: Sequence[Mapping[str, Any]],
        lines: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        blurred = cv2.medianBlur(gray, 5)
        height, width = gray.shape[:2]
        maximum_radius = max(10, min(14, int(round(min(height, width) * 0.025))))
        raw = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=10,
            param1=80,
            param2=8,
            minRadius=2,
            maxRadius=maximum_radius,
        )
        if raw is None:
            return []
        excluded = [
            *_topology_xy(topology),
            *(
                tuple(float(value) for value in circle["center"])
                for circle in circles
            ),
        ]
        candidates = []
        for values in np.round(raw[0, :], 2):
            x, y, radius = (float(value) for value in values)
            if not 0.0 <= x < width or not 0.0 <= y < height:
                continue
            if any(math.dist((x, y), point) <= max(7.0, 1.5 * radius) for point in excluded):
                continue
            if any(_bbox_distance((x, y), region.get("bbox")) <= max(3.0, radius) for region in text_regions):
                continue
            fill_support = _disk_ink_ratio(gray, (x, y), radius)
            if fill_support < 0.55:
                continue
            line_support = []
            for line in lines:
                projection, residual = _point_segment_projection((x, y), line["p1"], line["p2"])
                if 0.02 <= projection <= 0.98 and residual <= max(4.0, 1.25 * radius):
                    line_support.append(
                        {
                            "line_id": str(line["id"]),
                            "projection": round(projection, 6),
                            "residual_px": round(residual, 3),
                        }
                    )
            candidates.append(
                {
                    "id": "pending",
                    "xy": [round(x, 3), round(y, 3)],
                    "radius_px": round(radius, 3),
                    "fill_support": round(fill_support, 3),
                    "confidence": round(min(0.95, 0.55 + 0.4 * fill_support), 3),
                    "score_kind": "filled_point_blob",
                    "line_support": line_support,
                }
            )
        candidates.sort(key=lambda item: (-float(item["fill_support"]), item["xy"][1], item["xy"][0]))
        selected = []
        for candidate in candidates:
            if any(
                math.dist(candidate["xy"], existing["xy"]) <= max(5.0, candidate["radius_px"] * 1.5)
                for existing in selected
            ):
                continue
            selected.append(candidate)
        for index, candidate in enumerate(selected[: self.max_points]):
            candidate["id"] = f"vpt_{index}"
        return selected[: self.max_points]

    def _merge_collinear_lines(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        groups = _coalesce_line_groups(
            [
                _line_group([candidate], _canonical_line_angle(float(candidate["angle_deg"])))
                for candidate in candidates
            ]
        )
        merged = []
        for index, group in enumerate(groups):
            direction, normal = _line_frame(group["angle_deg"])
            start = [group["interval"][0] * direction[axis] + group["rho"] * normal[axis] for axis in range(2)]
            end = [group["interval"][1] * direction[axis] + group["rho"] * normal[axis] for axis in range(2)]
            merged.append({
                "id": f"vl_{index}",
                "p1": [int(round(value)) for value in start],
                "p2": [int(round(value)) for value in end],
                "length_px": round(math.dist(start, end), 3),
                "angle_deg": round(group["angle_deg"], 3),
                "source_ids": [str(member["id"]) for member in group["members"]],
            })
        merged.sort(key=lambda item: item["length_px"], reverse=True)
        for index, item in enumerate(merged):
            item["id"] = f"vl_{index}"
        return merged

    def _circle_edge_support(self, edges: np.ndarray, x: int, y: int, radius: int) -> float:
        height, width = edges.shape[:2]
        supported = 0
        samples = 96
        for index in range(samples):
            angle = 2.0 * math.pi * index / samples
            px = int(round(x + radius * math.cos(angle)))
            py = int(round(y + radius * math.sin(angle)))
            x0, x1 = max(0, px - 2), min(width, px + 3)
            y0, y1 = max(0, py - 2), min(height, py + 3)
            supported += int(bool(np.any(edges[y0:y1, x0:x1])))
        return supported / samples

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

    def _derive_topology(
        self,
        lines: List[Dict[str, Any]],
        circles: List[Dict[str, Any]],
        arcs: Sequence[Mapping[str, Any]] = (),
    ) -> Dict[str, Any]:
        endpoints = []
        for line in lines:
            endpoints.append({"id": f"{line['id']}_p1", "line_id": line["id"], "xy": line["p1"]})
            endpoints.append({"id": f"{line['id']}_p2", "line_id": line["id"], "xy": line["p2"]})

        intersections = []
        line_overlaps = []
        for i, a in enumerate(lines[:40]):
            for b in lines[i + 1 : 40]:
                point = self._segment_intersection(a["p1"], a["p2"], b["p1"], b["p2"])
                if point is not None:
                    intersections.append({
                        "id": f"vx_{len(intersections)}",
                        "line_ids": [a["id"], b["id"]],
                        "xy": [round(point[0], 3), round(point[1], 3)],
                    })
                else:
                    overlap = self._segment_overlap(a["p1"], a["p2"], b["p1"], b["p2"])
                    if overlap is not None:
                        line_overlaps.append(
                            {
                                "id": f"vo_{len(line_overlaps)}",
                                "line_ids": [a["id"], b["id"]],
                                "interval_px": [round(overlap[0], 3), round(overlap[1], 3)],
                                "overlap_length_px": round(overlap[1] - overlap[0], 3),
                                "residual_px": round(overlap[2], 3),
                            }
                        )
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

        circle_line_intersections = []
        for circle in circles[:10]:
            for line in lines[:40]:
                points = self._segment_circle_intersections(
                    line["p1"],
                    line["p2"],
                    circle["center"],
                    circle["radius_px"],
                )
                contact = "tangent" if len(points) == 1 else "crossing"
                for point in points:
                    circle_line_intersections.append(
                        {
                            "id": f"vcli_{len(circle_line_intersections)}",
                            "circle_id": circle["id"],
                            "line_id": line["id"],
                            "xy": [round(point[0], 3), round(point[1], 3)],
                            "contact": contact,
                        }
                    )
                    if len(circle_line_intersections) >= 80:
                        break
                if len(circle_line_intersections) >= 80:
                    break
            if len(circle_line_intersections) >= 80:
                break

        circle_circle_intersections = []
        for index, left in enumerate(circles[:10]):
            for right in circles[index + 1 : 10]:
                points = self._circle_circle_intersections(
                    left["center"],
                    left["radius_px"],
                    right["center"],
                    right["radius_px"],
                )
                contact = "tangent" if len(points) == 1 else "crossing"
                for point in points:
                    circle_circle_intersections.append(
                        {
                            "id": f"vcci_{len(circle_circle_intersections)}",
                            "circle_ids": [left["id"], right["id"]],
                            "xy": [round(point[0], 3), round(point[1], 3)],
                            "contact": contact,
                        }
                    )
                    if len(circle_circle_intersections) >= 80:
                        break
                if len(circle_circle_intersections) >= 80:
                    break
            if len(circle_circle_intersections) >= 80:
                break

        arc_line_intersections = []
        for arc in arcs[:20]:
            for line in lines[:40]:
                points = self._segment_circle_intersections(
                    line["p1"],
                    line["p2"],
                    arc["center"],
                    arc["radius_px"],
                )
                points = tuple(point for point in points if _point_on_visual_arc(point, arc))
                contact = "tangent" if len(points) == 1 else "crossing"
                for point in points:
                    arc_line_intersections.append(
                        {
                            "id": f"vali_{len(arc_line_intersections)}",
                            "arc_id": str(arc["id"]),
                            "line_id": str(line["id"]),
                            "xy": [round(point[0], 3), round(point[1], 3)],
                            "contact": contact,
                        }
                    )
                    if len(arc_line_intersections) >= 80:
                        break
                if len(arc_line_intersections) >= 80:
                    break
            if len(arc_line_intersections) >= 80:
                break

        arc_circle_intersections = []
        for arc in arcs[:20]:
            for circle in circles[:10]:
                if arc.get("circle_id") == circle.get("id"):
                    continue
                points = self._circle_circle_intersections(
                    arc["center"],
                    arc["radius_px"],
                    circle["center"],
                    circle["radius_px"],
                )
                points = tuple(point for point in points if _point_on_visual_arc(point, arc))
                contact = "tangent" if len(points) == 1 else "crossing"
                for point in points:
                    arc_circle_intersections.append(
                        {
                            "id": f"vaci_{len(arc_circle_intersections)}",
                            "arc_id": str(arc["id"]),
                            "circle_id": str(circle["id"]),
                            "xy": [round(point[0], 3), round(point[1], 3)],
                            "contact": contact,
                        }
                    )
                    if len(arc_circle_intersections) >= 80:
                        break
                if len(arc_circle_intersections) >= 80:
                    break
            if len(arc_circle_intersections) >= 80:
                break

        arc_arc_intersections = []
        for index, left in enumerate(arcs[:20]):
            for right in arcs[index + 1 : 20]:
                if left.get("circle_id") and left.get("circle_id") == right.get("circle_id"):
                    continue
                points = self._circle_circle_intersections(
                    left["center"],
                    left["radius_px"],
                    right["center"],
                    right["radius_px"],
                )
                points = tuple(
                    point
                    for point in points
                    if _point_on_visual_arc(point, left) and _point_on_visual_arc(point, right)
                )
                contact = "tangent" if len(points) == 1 else "crossing"
                for point in points:
                    arc_arc_intersections.append(
                        {
                            "id": f"vaai_{len(arc_arc_intersections)}",
                            "arc_ids": [str(left["id"]), str(right["id"])],
                            "xy": [round(point[0], 3), round(point[1], 3)],
                            "contact": contact,
                        }
                    )
                    if len(arc_arc_intersections) >= 80:
                        break
                if len(arc_arc_intersections) >= 80:
                    break
            if len(arc_arc_intersections) >= 80:
                break

        arc_endpoints = [
            {
                "id": f"{arc['id']}_{endpoint}",
                "arc_id": str(arc["id"]),
                "endpoint": endpoint,
                "xy": list(arc[endpoint]),
            }
            for arc in arcs
            for endpoint in ("start", "end")
        ]
        return {
            "endpoints": endpoints[:160],
            "arc_endpoints": arc_endpoints[:160],
            "line_intersections": intersections,
            "line_overlaps": line_overlaps[:80],
            "circle_line_hits": circle_line_hits,
            "circle_line_intersections": circle_line_intersections,
            "circle_circle_intersections": circle_circle_intersections,
            "arc_line_intersections": arc_line_intersections,
            "arc_circle_intersections": arc_circle_intersections,
            "arc_arc_intersections": arc_arc_intersections,
        }

    def _segment_circle_intersections(self, p1, p2, center, radius, eps: float = 1e-6):
        x1, y1 = (float(value) for value in p1)
        x2, y2 = (float(value) for value in p2)
        cx, cy = (float(value) for value in center)
        radius = float(radius)
        dx, dy = x2 - x1, y2 - y1
        a = dx * dx + dy * dy
        if a <= eps or radius <= 0.0:
            return ()
        fx, fy = x1 - cx, y1 - cy
        b = 2.0 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - radius * radius
        discriminant = b * b - 4.0 * a * c
        if discriminant < -eps:
            return ()
        roots = (-b / (2.0 * a),) if abs(discriminant) <= eps else (
            (-b - math.sqrt(discriminant)) / (2.0 * a),
            (-b + math.sqrt(discriminant)) / (2.0 * a),
        )
        points = [
            (x1 + parameter * dx, y1 + parameter * dy)
            for parameter in roots
            if -eps <= parameter <= 1.0 + eps
        ]
        return tuple(sorted(set(points), key=lambda point: (point[1], point[0])))

    def _circle_circle_intersections(self, center1, radius1, center2, radius2, eps: float = 1e-6):
        x1, y1 = (float(value) for value in center1)
        x2, y2 = (float(value) for value in center2)
        radius1, radius2 = float(radius1), float(radius2)
        if radius1 <= 0.0 or radius2 <= 0.0:
            return ()
        dx, dy = x2 - x1, y2 - y1
        distance = math.hypot(dx, dy)
        if distance <= eps or distance > radius1 + radius2 + eps or distance < abs(radius1 - radius2) - eps:
            return ()
        along = (radius1 * radius1 - radius2 * radius2 + distance * distance) / (2.0 * distance)
        height_squared = radius1 * radius1 - along * along
        if height_squared < -eps:
            return ()
        base_x = x1 + along * dx / distance
        base_y = y1 + along * dy / distance
        if abs(height_squared) <= eps:
            return ((base_x, base_y),)
        height = math.sqrt(height_squared)
        offset_x = -dy * height / distance
        offset_y = dx * height / distance
        points = ((base_x + offset_x, base_y + offset_y), (base_x - offset_x, base_y - offset_y))
        return tuple(sorted(points, key=lambda point: (point[1], point[0])))

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

    def _segment_overlap(
        self,
        p1: List[int],
        p2: List[int],
        p3: List[int],
        p4: List[int],
        tolerance: float = 3.0,
    ) -> tuple[float, float, float] | None:
        dx = float(p2[0] - p1[0])
        dy = float(p2[1] - p1[1])
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            return None
        direction = dx / length, dy / length
        offset = abs((p3[0] - p1[0]) * direction[1] - (p3[1] - p1[1]) * direction[0])
        end_offset = abs((p4[0] - p1[0]) * direction[1] - (p4[1] - p1[1]) * direction[0])
        residual = max(offset, end_offset)
        if residual > tolerance:
            return None
        first = (0.0, length)
        second = tuple(
            (point[0] - p1[0]) * direction[0] + (point[1] - p1[1]) * direction[1]
            for point in (p3, p4)
        )
        overlap_start = max(first[0], min(second))
        overlap_end = min(first[1], max(second))
        if overlap_end - overlap_start <= tolerance:
            return None
        return overlap_start, overlap_end, residual

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


def _orientation_delta(left: float, right: float) -> float:
    delta = abs(left - right) % 180.0
    return min(delta, 180.0 - delta)


def _simplify_contour(contour: np.ndarray) -> List[List[int]]:
    perimeter = cv2.arcLength(contour, True)
    simplified = cv2.approxPolyDP(contour, max(1.0, 0.004 * perimeter), True)
    return [[int(point[0][0]), int(point[0][1])] for point in simplified]


def _topology_xy(topology: Mapping[str, Any]) -> tuple[tuple[float, float], ...]:
    points = []
    for key in (
        "endpoints",
        "arc_endpoints",
        "line_intersections",
        "circle_line_intersections",
        "circle_circle_intersections",
        "arc_line_intersections",
        "arc_circle_intersections",
        "arc_arc_intersections",
    ):
        for item in topology.get(key) or ():
            values = item.get("xy")
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and len(values) == 2:
                points.append((float(values[0]), float(values[1])))
    return tuple(points)


def _bbox_distance(point: tuple[float, float], bbox: Sequence[Any] | None) -> float:
    if bbox is None or isinstance(bbox, (str, bytes)) or len(bbox) != 4:
        return float("inf")
    x, y, width, height = (float(value) for value in bbox)
    px, py = point
    dx = max(x - px, 0.0, px - (x + width))
    dy = max(y - py, 0.0, py - (y + height))
    return math.hypot(dx, dy)


def _xywh_bbox(value: Sequence[Any] | None) -> tuple[float, float, float, float] | None:
    if value is None or isinstance(value, (str, bytes)) or len(value) != 4:
        return None
    try:
        x, y, width, height = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (x, y, width, height)) or width <= 0.0 or height <= 0.0:
        return None
    return x, y, x + width, y + height


def _curve_bbox(candidate: Mapping[str, Any], kind: str) -> tuple[float, float, float, float] | None:
    if kind == "circle":
        center = candidate.get("center")
        if not isinstance(center, Sequence) or isinstance(center, (str, bytes)) or len(center) != 2:
            return None
        try:
            cx, cy, radius = float(center[0]), float(center[1]), float(candidate["radius_px"])
        except (KeyError, TypeError, ValueError):
            return None
        if not all(math.isfinite(item) for item in (cx, cy, radius)) or radius <= 0.0:
            return None
        return cx - radius, cy - radius, cx + radius, cy + radius
    value = candidate.get("bbox")
    if value is None or isinstance(value, (str, bytes)) or len(value) != 4:
        return None
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in values):
        return None
    return min(values[0], values[2]), min(values[1], values[3]), max(values[0], values[2]), max(values[1], values[3])


def _bbox_union_intersection_area(
    candidate: tuple[float, float, float, float],
    boxes: Sequence[tuple[float, float, float, float]],
) -> float:
    x1, y1, x2, y2 = candidate
    clipped = [
        (max(x1, left), max(y1, top), min(x2, right), min(y2, bottom))
        for left, top, right, bottom in boxes
        if min(x2, right) > max(x1, left) and min(y2, bottom) > max(y1, top)
    ]
    if not clipped:
        return 0.0
    x_edges = sorted({x1, x2, *(item[0] for item in clipped), *(item[2] for item in clipped)})
    area = 0.0
    for left, right in zip(x_edges, x_edges[1:]):
        if right <= left:
            continue
        intervals = sorted((top, bottom) for box_left, top, box_right, bottom in clipped if box_left < right and box_right > left)
        covered = 0.0
        current_start = current_end = None
        for top, bottom in intervals:
            if current_start is None:
                current_start, current_end = top, bottom
            elif top <= current_end:
                current_end = max(current_end, bottom)
            else:
                covered += current_end - current_start
                current_start, current_end = top, bottom
        if current_start is not None:
            covered += current_end - current_start
        area += (right - left) * covered
    return area


def _point_segment_projection(
    point: tuple[float, float],
    start: Sequence[float],
    end: Sequence[float],
) -> tuple[float, float]:
    x1, y1 = (float(value) for value in start)
    x2, y2 = (float(value) for value in end)
    dx, dy = x2 - x1, y2 - y1
    denominator = dx * dx + dy * dy
    if denominator <= 1e-9:
        return 0.0, float("inf")
    projection = ((point[0] - x1) * dx + (point[1] - y1) * dy) / denominator
    nearest = x1 + projection * dx, y1 + projection * dy
    return projection, math.dist(point, nearest)


def _disk_ink_ratio(gray: np.ndarray, center: tuple[float, float], radius: float) -> float:
    cx, cy = (int(round(value)) for value in center)
    extent = max(2, int(math.ceil(radius)))
    y0, y1 = max(0, cy - extent), min(gray.shape[0], cy + extent + 1)
    x0, x1 = max(0, cx - extent), min(gray.shape[1], cx + extent + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= max(1.5, 0.75 * radius) ** 2
    if not np.any(disk):
        return 0.0
    return float(np.mean(gray[y0:y1, x0:x1][disk] <= 190))


def _fit_arc_contour(
    contour: np.ndarray,
    image_shape: tuple[int, int],
) -> Dict[str, Any] | None:
    points = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if len(points) < 18:
        return None
    unique = np.unique(np.round(points, decimals=3), axis=0)
    if len(unique) < 18:
        return None
    x = points[:, 0]
    y = points[:, 1]
    design = np.column_stack((2.0 * x, 2.0 * y, np.ones(len(points))))
    values = x * x + y * y
    try:
        solution, _, rank, _ = np.linalg.lstsq(design, values, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 3:
        return None
    center = float(solution[0]), float(solution[1])
    radius_squared = float(solution[2] + center[0] * center[0] + center[1] * center[1])
    if radius_squared <= 0.0:
        return None
    radius = math.sqrt(radius_squared)
    minimum_radius = max(14.0, 0.025 * min(image_shape))
    maximum_radius = 2.0 * math.hypot(*image_shape)
    if not minimum_radius <= radius <= maximum_radius:
        return None
    distances = np.hypot(x - center[0], y - center[1])
    residual = float(np.sqrt(np.mean((distances - radius) ** 2)))
    if residual > max(2.5, 0.035 * radius):
        return None
    angles = np.mod(np.arctan2(y - center[1], x - center[0]), 2.0 * math.pi)
    order = np.argsort(angles)
    ordered_angles = angles[order]
    gaps = np.diff(np.concatenate((ordered_angles, ordered_angles[:1] + 2.0 * math.pi)))
    largest_gap_index = int(np.argmax(gaps))
    sweep = 2.0 * math.pi - float(gaps[largest_gap_index])
    sweep_degrees = math.degrees(sweep)
    if not 25.0 <= sweep_degrees <= 335.0:
        return None
    angular_density = sweep / max(1.0, float(np.ptp(ordered_angles)))
    if angular_density < 0.5:
        return None
    start_index = int(order[(largest_gap_index + 1) % len(order)])
    end_index = int(order[largest_gap_index])
    contour_angles = np.unwrap(np.arctan2(y - center[1], x - center[0]))
    direction = float(np.sum(np.diff(contour_angles)))
    orientation = "ccw" if direction >= 0.0 else "cw"
    bbox = [
        max(0.0, float(np.min(x) - 2.0)),
        max(0.0, float(np.min(y) - 2.0)),
        min(float(image_shape[1] - 1), float(np.max(x) + 2.0)),
        min(float(image_shape[0] - 1), float(np.max(y) + 2.0)),
    ]
    edge_support = max(0.5, min(0.99, 1.0 - residual / max(2.5, 0.035 * radius)))
    return {
        "id": "pending",
        "center": [round(center[0], 3), round(center[1], 3)],
        "radius_px": round(radius, 3),
        "start": [round(float(points[start_index, 0]), 3), round(float(points[start_index, 1]), 3)],
        "end": [round(float(points[end_index, 0]), 3), round(float(points[end_index, 1]), 3)],
        "orientation": orientation,
        "sweep_deg": round(sweep_degrees, 3),
        "bbox": [round(value, 3) for value in bbox],
        "residual_px": round(residual, 3),
        "edge_support": round(edge_support, 3),
        "score_kind": "fitted_open_circular_contour",
    }


def _match_support_circle(
    arc: Mapping[str, Any],
    circles: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    center = tuple(float(value) for value in arc["center"])
    radius = float(arc["radius_px"])
    candidates = []
    for circle in circles:
        circle_center = tuple(float(value) for value in circle["center"])
        circle_radius = float(circle["radius_px"])
        center_error = math.dist(center, circle_center) / max(8.0, radius * 0.08)
        radius_error = abs(radius - circle_radius) / max(8.0, radius * 0.08)
        if center_error > 1.0 or radius_error > 1.0:
            continue
        candidates.append((center_error + radius_error, str(circle["id"]), circle))
    candidates.sort(key=lambda item: (item[0], item[1]))
    if not candidates:
        return None
    if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < 0.15:
        return None
    return candidates[0][2]


def _deduplicate_arcs(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected: list[Dict[str, Any]] = []
    for candidate in candidates:
        center = tuple(float(value) for value in candidate["center"])
        radius = float(candidate["radius_px"])
        duplicate = any(
            math.dist(center, tuple(float(value) for value in existing["center"]))
            <= max(5.0, radius * 0.05)
            and abs(radius - float(existing["radius_px"])) <= max(5.0, radius * 0.05)
            and abs(float(candidate["sweep_deg"]) - float(existing["sweep_deg"])) <= 12.0
            for existing in selected
        )
        if not duplicate:
            selected.append(dict(candidate))
    for index, candidate in enumerate(selected):
        candidate["id"] = f"varc_{index}"
    return selected


def _point_on_visual_arc(
    point: Sequence[float],
    arc: Mapping[str, Any],
    *,
    tolerance_degrees: float = 2.0,
) -> bool:
    center = tuple(float(value) for value in arc["center"])
    radius = float(arc["radius_px"])
    point_radius = math.dist(center, tuple(float(value) for value in point))
    if abs(point_radius - radius) > max(3.0, 0.04 * radius):
        return False
    start = tuple(float(value) for value in arc["start"])
    start_angle = math.atan2(start[1] - center[1], start[0] - center[0])
    point_angle = math.atan2(float(point[1]) - center[1], float(point[0]) - center[0])
    if str(arc.get("orientation") or "ccw") == "cw":
        delta = (start_angle - point_angle) % (2.0 * math.pi)
    else:
        delta = (point_angle - start_angle) % (2.0 * math.pi)
    sweep = math.radians(float(arc["sweep_deg"]))
    tolerance = math.radians(tolerance_degrees)
    return delta <= sweep + tolerance


def _canonical_line_angle(angle: float) -> float:
    return (angle + 90.0) % 180.0 - 90.0


def _line_frame(angle: float) -> tuple[tuple[float, float], tuple[float, float]]:
    radians = math.radians(angle)
    direction = math.cos(radians), math.sin(radians)
    return direction, (-direction[1], direction[0])


def _coalesce_line_groups(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged = list(groups)
    changed = True
    while changed:
        changed = False
        for left_index, left in enumerate(merged):
            for right_index in range(left_index + 1, len(merged)):
                right = merged[right_index]
                if _orientation_delta(left["angle_deg"], right["angle_deg"]) > 2.0:
                    continue
                right_interval, right_rho = _project_line_members(right["members"], left["angle_deg"])
                if abs(left["rho"] - right_rho) > 12.0:
                    continue
                if _interval_gap(left["interval"], right_interval) > 12.0:
                    continue
                members = [*left["members"], *right["members"]]
                merged[left_index] = _line_group(members, left["angle_deg"])
                del merged[right_index]
                changed = True
                break
            if changed:
                break
    return merged


def _line_group(members: List[Dict[str, Any]], angle: float) -> Dict[str, Any]:
    interval, rho = _project_line_members(members, angle)
    return {"angle_deg": angle, "rho": rho, "interval": interval, "members": members}


def _project_line_members(
    members: List[Dict[str, Any]],
    angle: float,
) -> tuple[List[float], float]:
    direction, normal = _line_frame(angle)
    points = [point for member in members for point in (member["p1"], member["p2"])]
    parameters = [point[0] * direction[0] + point[1] * direction[1] for point in points]
    offsets = [point[0] * normal[0] + point[1] * normal[1] for point in points]
    return [min(parameters), max(parameters)], math.fsum(offsets) / len(offsets)


def _interval_gap(left: List[float], right: List[float]) -> float:
    return max(0.0, max(left[0], right[0]) - min(left[1], right[1]))
