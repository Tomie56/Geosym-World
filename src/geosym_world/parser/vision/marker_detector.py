"""Deterministic visual marker candidates grounded to detected primitives."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


VISUAL_MARKER_DETECTOR_VERSION = "0.2"
SUPPORTED_MARKER_KINDS = ("right_angle", "parallel", "equal_length_tick", "angle_arc")


class VisualMarkerDetector:
    """Detect explicit diagram glyphs without inferring relations from geometry alone."""

    def __init__(
        self,
        *,
        minimum_size_px: int = 6,
        minimum_size_ratio: float = 0.012,
        maximum_size_px: int = 36,
        maximum_size_ratio: float = 0.05,
        right_angle_tolerance_deg: float = 8.0,
        ink_threshold: int = 180,
        sampling_radius_px: int = 1,
        minimum_stroke_support: float = 0.78,
        parallel_tolerance_deg: float = 3.0,
        equal_angle_tolerance_deg: float = 8.0,
        max_markers: int = 40,
    ) -> None:
        if minimum_size_px <= 0 or maximum_size_px < minimum_size_px:
            raise ValueError("Marker pixel sizes must be positive and ordered")
        if not 0.0 <= minimum_size_ratio <= maximum_size_ratio < 1.0:
            raise ValueError("Marker size ratios must be ordered within [0, 1)")
        if not 0.0 <= right_angle_tolerance_deg < 45.0:
            raise ValueError("Right-angle tolerance must be in [0, 45)")
        if not 0 <= ink_threshold <= 255:
            raise ValueError("Marker ink threshold must be in [0, 255]")
        if sampling_radius_px < 0:
            raise ValueError("Marker sampling radius must be non-negative")
        if not 0.0 < minimum_stroke_support <= 1.0:
            raise ValueError("Minimum marker stroke support must be in (0, 1]")
        if not 0.0 <= parallel_tolerance_deg < 45.0:
            raise ValueError("Parallel marker tolerance must be in [0, 45)")
        if not 0.0 <= equal_angle_tolerance_deg < 45.0:
            raise ValueError("Equal-angle marker tolerance must be in [0, 45)")
        if max_markers <= 0:
            raise ValueError("max_markers must be positive")
        self.minimum_size_px = minimum_size_px
        self.minimum_size_ratio = minimum_size_ratio
        self.maximum_size_px = maximum_size_px
        self.maximum_size_ratio = maximum_size_ratio
        self.right_angle_tolerance_deg = right_angle_tolerance_deg
        self.ink_threshold = ink_threshold
        self.sampling_radius_px = sampling_radius_px
        self.minimum_stroke_support = minimum_stroke_support
        self.parallel_tolerance_deg = parallel_tolerance_deg
        self.equal_angle_tolerance_deg = equal_angle_tolerance_deg
        self.max_markers = max_markers

    def configuration(self) -> dict[str, Any]:
        return {
            "version": VISUAL_MARKER_DETECTOR_VERSION,
            "supported_kinds": list(SUPPORTED_MARKER_KINDS),
            "minimum_size_px": self.minimum_size_px,
            "minimum_size_ratio": self.minimum_size_ratio,
            "maximum_size_px": self.maximum_size_px,
            "maximum_size_ratio": self.maximum_size_ratio,
            "right_angle_tolerance_deg": self.right_angle_tolerance_deg,
            "ink_threshold": self.ink_threshold,
            "sampling_radius_px": self.sampling_radius_px,
            "minimum_stroke_support": self.minimum_stroke_support,
            "parallel_tolerance_deg": self.parallel_tolerance_deg,
            "equal_angle_tolerance_deg": self.equal_angle_tolerance_deg,
            "max_markers": self.max_markers,
        }

    def detect(
        self,
        gray: np.ndarray,
        lines: Sequence[Mapping[str, Any]],
        topology: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if gray.ndim != 2:
            raise ValueError("VisualMarkerDetector requires a grayscale image")
        line_index = {str(line["id"]): line for line in lines}
        ink = gray <= self.ink_threshold
        markers = []
        for intersection in topology.get("line_intersections") or ():
            candidate = self._right_angle_candidate(ink, line_index, intersection)
            if candidate is not None:
                markers.append(candidate)
            else:
                markers.extend(self._angle_arc_candidates(ink, line_index, intersection))
        occupied = tuple(_marker_center(marker) for marker in markers)
        junctions = _topology_points(topology)
        linear_candidates = []
        for line in lines:
            linear_candidates.extend(self._linear_marker_candidates(ink, line, junctions, occupied))
        markers.extend(_non_maximum_markers(linear_candidates))
        markers = _non_maximum_markers(markers)[: self.max_markers]
        markers.sort(key=lambda item: (_marker_center(item)[1], _marker_center(item)[0], item["kind"], item["line_ids"]))
        for index, marker in enumerate(markers):
            marker["id"] = f"vm_{index}"
        self._assign_relation_groups(markers, line_index)
        return markers

    def _linear_marker_candidates(
        self,
        ink: np.ndarray,
        line: Mapping[str, Any],
        junctions: Sequence[tuple[float, float]],
        occupied: Sequence[tuple[float, float]],
    ) -> list[dict[str, Any]]:
        direction = _line_direction(line)
        if direction is None:
            return []
        start = tuple(float(value) for value in line["p1"])
        end = tuple(float(value) for value in line["p2"])
        length = math.dist(start, end)
        minimum_size, maximum_size = self._linear_size_range(ink.shape, length)
        if maximum_size < minimum_size:
            return []
        normal = (-direction[1], direction[0])
        support_half_width = _estimate_support_half_width(ink, start, direction, normal, length)
        margin = max(2.2 * maximum_size, 0.08 * length)
        if length <= 2.0 * margin:
            return []
        excluded = (*junctions, *occupied)
        candidates = []
        for offset in range(int(math.ceil(margin)), int(math.floor(length - margin)) + 1):
            center = (start[0] + direction[0] * offset, start[1] + direction[1] * offset)
            if any(math.dist(center, point) <= maximum_size * 1.6 for point in excluded):
                continue
            tick = self._tick_candidate(
                ink,
                line,
                center,
                direction,
                normal,
                minimum_size,
                maximum_size,
                support_half_width,
            )
            parallel = self._parallel_candidate(
                ink,
                line,
                center,
                direction,
                normal,
                minimum_size,
                maximum_size,
                support_half_width,
            )
            if tick is not None:
                candidates.append(tick)
            elif parallel is not None:
                candidates.append(parallel)
        return candidates

    def _tick_candidate(
        self,
        ink: np.ndarray,
        line: Mapping[str, Any],
        center: tuple[float, float],
        direction: tuple[float, float],
        normal: tuple[float, float],
        minimum_size: int,
        maximum_size: int,
        support_half_width: float,
    ) -> dict[str, Any] | None:
        candidates = []
        first_size = max(minimum_size, int(math.ceil(support_half_width + 1.0)))
        for half_size in range(first_size, maximum_size + 1):
            start = (center[0] - normal[0] * half_size, center[1] - normal[1] * half_size)
            end = (center[0] + normal[0] * half_size, center[1] + normal[1] * half_size)
            if not _path_inside_image((start, end), ink.shape):
                continue
            gap = min(2.0, half_size * 0.35)
            negative_inner = (center[0] - normal[0] * gap, center[1] - normal[1] * gap)
            positive_inner = (center[0] + normal[0] * gap, center[1] + normal[1] * gap)
            supports = (
                _segment_ink_support(ink, start, negative_inner, self.sampling_radius_px),
                _segment_ink_support(ink, positive_inner, end, self.sampling_radius_px),
            )
            minimum_support = min(supports)
            if minimum_support < self.minimum_stroke_support or not _point_has_ink(ink, start) or not _point_has_ink(ink, end):
                continue
            candidates.append((minimum_support, sum(supports) / 2.0, half_size, start, end, supports))
        if not candidates:
            return None
        minimum_support, mean_support, half_size, start, end, supports = max(
            candidates,
            key=lambda item: (round(item[0], 6), round(item[1], 6), item[2]),
        )
        profile = _off_support_profile(
            ink,
            center,
            direction,
            normal,
            max(5.0, half_size * 1.5),
            half_size + 2.0,
            support_half_width,
        )
        if profile is None or profile[0] > max(4.0, float(half_size)):
            return None
        center_residual = abs(profile[2])
        template_score = max(0.0, minimum_support - center_residual / max(20.0, half_size * 20.0))
        confidence = max(0.0, min(0.97, 0.67 + 0.3 * mean_support))
        return {
            "id": "pending",
            "kind": "equal_length_tick",
            "line_ids": [str(line["id"])],
            "center": [round(value, 3) for value in center],
            "path": [[round(value, 3) for value in start], [round(value, 3) for value in end]],
            "bbox": [round(value, 3) for value in _path_bbox((start, end), padding=self.sampling_radius_px + 1)],
            "size_px": half_size,
            "confidence": round(confidence, 3),
            "score_kind": "transverse_tick_stroke_support",
            "stroke_support": [round(value, 3) for value in supports],
            "template_score": round(template_score, 3),
            "center_residual_px": round(center_residual, 3),
        }

    def _parallel_candidate(
        self,
        ink: np.ndarray,
        line: Mapping[str, Any],
        center: tuple[float, float],
        direction: tuple[float, float],
        normal: tuple[float, float],
        minimum_size: int,
        maximum_size: int,
        support_half_width: float,
    ) -> dict[str, Any] | None:
        candidates = []
        first_size = max(5, minimum_size, int(math.ceil((support_half_width + 0.1) / 0.48)))
        for size in range(first_size, maximum_size + 1):
            for sign in (-1, 1):
                along = (direction[0] * sign, direction[1] * sign)
                tail = (center[0] - along[0] * size * 0.55, center[1] - along[1] * size * 0.55)
                tip = (center[0] + along[0] * size * 0.55, center[1] + along[1] * size * 0.55)
                wings = (
                    (tail[0] + normal[0] * size * 0.48, tail[1] + normal[1] * size * 0.48),
                    (tail[0] - normal[0] * size * 0.48, tail[1] - normal[1] * size * 0.48),
                )
                path = (wings[0], tip, wings[1])
                if not _path_inside_image(path, ink.shape):
                    continue
                supports = tuple(
                    _segment_ink_support(ink, wing, tip, self.sampling_radius_px)
                    for wing in wings
                )
                minimum_support = min(supports)
                if (
                    minimum_support < self.minimum_stroke_support
                    or not all(_point_has_ink(ink, wing) for wing in wings)
                ):
                    continue
                candidates.append((minimum_support, sum(supports) / 2.0, size, path, supports))
        if not candidates:
            return None
        minimum_support, mean_support, size, path, supports = max(
            candidates,
            key=lambda item: (round(item[0], 6), round(item[1], 6), item[2]),
        )
        profile = _off_support_profile(
            ink,
            center,
            direction,
            normal,
            size * 1.5,
            size + 2.0,
            support_half_width,
        )
        if profile is None or profile[0] < size * 0.75:
            return None
        center_residual = abs(profile[2])
        template_score = max(0.0, minimum_support - center_residual / max(20.0, size * 20.0))
        confidence = max(0.0, min(0.98, 0.69 + 0.29 * mean_support))
        return {
            "id": "pending",
            "kind": "parallel",
            "line_ids": [str(line["id"])],
            "center": [round(value, 3) for value in center],
            "path": [[round(value, 3) for value in point] for point in path],
            "bbox": [round(value, 3) for value in _path_bbox(path, padding=self.sampling_radius_px + 1)],
            "size_px": size,
            "confidence": round(confidence, 3),
            "score_kind": "parallel_chevron_stroke_support",
            "stroke_support": [round(value, 3) for value in supports],
            "template_score": round(template_score, 3),
            "center_residual_px": round(center_residual, 3),
        }

    def _angle_arc_candidates(
        self,
        ink: np.ndarray,
        lines: Mapping[str, Mapping[str, Any]],
        intersection: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        line_ids = tuple(str(value) for value in intersection.get("line_ids") or ())
        if len(line_ids) != 2 or any(line_id not in lines for line_id in line_ids):
            return []
        directions = tuple(_line_direction(lines[line_id]) for line_id in line_ids)
        if any(direction is None for direction in directions):
            return []
        vertex = tuple(float(value) for value in intersection["xy"])
        support_half_widths = tuple(
            _line_support_half_width(ink, lines[line_id], direction)
            for line_id, direction in zip(line_ids, directions)
        )
        minimum_size, _ = self._size_range(ink.shape, lines[line_ids[0]], lines[line_ids[1]])
        maximum_size = min(
            self.maximum_size_px,
            int(round(min(ink.shape[:2]) * max(self.maximum_size_ratio, 0.1))),
            int(math.floor(min(float(lines[line_id]["length_px"]) for line_id in line_ids) * 0.3)),
        )
        candidates = []
        for left_sign in (-1, 1):
            for right_sign in (-1, 1):
                first = tuple(component * left_sign for component in directions[0])
                second = tuple(component * right_sign for component in directions[1])
                start_angle = math.atan2(first[1], first[0])
                delta = _signed_minor_delta(start_angle, math.atan2(second[1], second[0]))
                span_deg = abs(math.degrees(delta))
                if span_deg < 20.0 or span_deg > 165.0:
                    continue
                sector_candidates = []
                for radius in range(minimum_size, maximum_size + 1):
                    path = _arc_path(vertex, radius, start_angle, delta)
                    interior = path[2:-2]
                    core = tuple(
                        point
                        for point in interior
                        if all(
                            _point_line_distance(point, vertex, direction) > half_width + 0.75
                            for direction, half_width in zip(directions, support_half_widths)
                        )
                    )
                    if (
                        len(core) < 3
                        or len(core) < math.ceil(len(interior) * 0.25)
                        or not _path_inside_image(path, ink.shape)
                    ):
                        continue
                    support = _points_ink_support(ink, core, self.sampling_radius_px)
                    if support < self.minimum_stroke_support:
                        continue
                    sector_candidates.append((support, radius, span_deg, path, (left_sign, right_sign)))
                if len(sector_candidates) < 3:
                    continue
                best_support = max(item[0] for item in sector_candidates)
                stable = [item for item in sector_candidates if item[0] >= best_support - 0.05]
                median_radius = float(np.median([item[1] for item in stable]))
                candidates.append(
                    min(
                        stable,
                        key=lambda item: (abs(item[1] - median_radius), -item[0], item[1]),
                    )
                )
        selected = _non_maximum_markers(
            [
                {
                    "id": "pending",
                    "kind": "angle_arc",
                    "line_ids": list(line_ids),
                    "vertex": [round(value, 3) for value in vertex],
                    "center": [round(value, 3) for value in _arc_midpoint(path)],
                    "path": [[round(value, 3) for value in point] for point in path],
                    "bbox": [round(value, 3) for value in _path_bbox(path, padding=self.sampling_radius_px + 1)],
                    "size_px": radius,
                    "confidence": round(max(0.0, min(0.96, 0.66 + 0.3 * support)), 3),
                    "score_kind": "angle_arc_stroke_support",
                    "stroke_support": [round(support, 3)],
                    "template_score": round(support, 3),
                    "angle_span_deg": round(span_deg, 3),
                    "ray_signs": list(signs),
                }
                for support, radius, span_deg, path, signs in candidates
            ]
        )
        return selected[:1]

    def _linear_size_range(self, image_shape: tuple[int, ...], line_length: float) -> tuple[int, int]:
        image_extent = min(image_shape[:2])
        minimum = max(3, int(round(image_extent * 0.006)))
        maximum = min(18, int(round(image_extent * 0.027)), int(math.floor(line_length * 0.08)))
        return minimum, maximum

    def _assign_relation_groups(
        self,
        markers: list[dict[str, Any]],
        lines: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self._group_parallel_markers(markers, lines)
        self._group_equal_length_markers(markers)
        self._group_equal_angle_markers(markers)

    def _group_parallel_markers(
        self,
        markers: list[dict[str, Any]],
        lines: Mapping[str, Mapping[str, Any]],
    ) -> None:
        candidates = [marker for marker in markers if marker["kind"] == "parallel" and len(marker["line_ids"]) == 1]
        groups = _connected_marker_groups(
            candidates,
            lambda left, right: left["line_ids"] != right["line_ids"]
            and _orientation_delta(
                float(lines[left["line_ids"][0]]["angle_deg"]),
                float(lines[right["line_ids"][0]]["angle_deg"]),
            )
            <= self.parallel_tolerance_deg
            and _comparable_size(left, right),
        )
        _annotate_unambiguous_groups(groups, "parallel")

    def _group_equal_length_markers(self, markers: list[dict[str, Any]]) -> None:
        candidates = [marker for marker in markers if marker["kind"] == "equal_length_tick"]
        groups = _connected_marker_groups(candidates, _comparable_size)
        _annotate_unambiguous_groups(groups, "equal_length", permit_repeated_support=True)

    def _group_equal_angle_markers(self, markers: list[dict[str, Any]]) -> None:
        candidates = [marker for marker in markers if marker["kind"] == "angle_arc"]
        groups = _connected_marker_groups(
            candidates,
            lambda left, right: abs(float(left["angle_span_deg"]) - float(right["angle_span_deg"]))
            <= self.equal_angle_tolerance_deg
            and _comparable_size(left, right),
        )
        _annotate_unambiguous_groups(groups, "equal_angle", permit_repeated_support=True)

    def _right_angle_candidate(
        self,
        ink: np.ndarray,
        lines: Mapping[str, Mapping[str, Any]],
        intersection: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        line_ids = tuple(str(value) for value in intersection.get("line_ids") or ())
        if len(line_ids) != 2 or any(line_id not in lines for line_id in line_ids):
            return None
        left, right = (lines[line_id] for line_id in line_ids)
        angle_delta = _orientation_delta(float(left["angle_deg"]), float(right["angle_deg"]))
        residual = abs(angle_delta - 90.0)
        if residual > self.right_angle_tolerance_deg:
            return None
        directions = (_line_direction(left), _line_direction(right))
        if directions[0] is None or directions[1] is None:
            return None
        vertex = tuple(float(value) for value in intersection["xy"])
        minimum_size, maximum_size = self._size_range(ink.shape, left, right)
        if maximum_size < minimum_size:
            return None

        candidates = []
        for size in range(minimum_size, maximum_size + 1):
            for left_sign in (-1, 1):
                for right_sign in (-1, 1):
                    path = _right_angle_path(vertex, directions, size, left_sign, right_sign)
                    if not _path_inside_image(path, ink.shape):
                        continue
                    supports = (
                        _segment_ink_support(ink, path[0], path[1], self.sampling_radius_px),
                        _segment_ink_support(ink, path[1], path[2], self.sampling_radius_px),
                    )
                    minimum_support = min(supports)
                    if minimum_support < self.minimum_stroke_support:
                        continue
                    candidates.append((minimum_support, sum(supports) / 2.0, size, path, supports))
        if not candidates:
            return None
        minimum_support, mean_support, size, path, supports = max(
            candidates,
            key=lambda item: (round(item[0], 6), round(item[1], 6), item[2]),
        )
        confidence = max(0.0, min(0.99, 0.72 + 0.27 * mean_support - residual / 100.0))
        bbox = _path_bbox((vertex, *path), padding=self.sampling_radius_px + 1)
        return {
            "id": "pending",
            "kind": "right_angle",
            "semantic_relation": "perpendicular",
            "line_ids": list(line_ids),
            "vertex": [round(value, 3) for value in vertex],
            "path": [[round(value, 3) for value in point] for point in path],
            "bbox": [round(value, 3) for value in bbox],
            "size_px": size,
            "confidence": round(confidence, 3),
            "score_kind": "right_angle_l_stroke_support",
            "stroke_support": [round(value, 3) for value in supports],
            "angle_residual_deg": round(residual, 3),
        }

    def _size_range(
        self,
        image_shape: tuple[int, ...],
        left: Mapping[str, Any],
        right: Mapping[str, Any],
    ) -> tuple[int, int]:
        image_extent = min(image_shape[:2])
        minimum = max(self.minimum_size_px, int(round(image_extent * self.minimum_size_ratio)))
        support_extent = min(float(left["length_px"]), float(right["length_px"]))
        maximum = min(
            self.maximum_size_px,
            int(round(image_extent * self.maximum_size_ratio)),
            int(math.floor(support_extent * 0.3)),
        )
        return minimum, maximum


def _orientation_delta(left: float, right: float) -> float:
    delta = abs(left - right) % 180.0
    return min(delta, 180.0 - delta)


def _signed_minor_delta(start: float, end: float) -> float:
    return (end - start + math.pi) % (2.0 * math.pi) - math.pi


def _line_direction(line: Mapping[str, Any]) -> tuple[float, float] | None:
    dx = float(line["p2"][0]) - float(line["p1"][0])
    dy = float(line["p2"][1]) - float(line["p1"][1])
    length = math.hypot(dx, dy)
    return None if length <= 1e-9 else (dx / length, dy / length)


def _point_line_distance(
    point: tuple[float, float],
    origin: tuple[float, float],
    direction: tuple[float, float],
) -> float:
    dx, dy = point[0] - origin[0], point[1] - origin[1]
    return abs(dx * direction[1] - dy * direction[0])


def _right_angle_path(
    vertex: tuple[float, float],
    directions: tuple[tuple[float, float], tuple[float, float]],
    size: int,
    left_sign: int,
    right_sign: int,
) -> tuple[tuple[float, float], ...]:
    left = tuple(component * left_sign for component in directions[0])
    right = tuple(component * right_sign for component in directions[1])
    first = (vertex[0] + left[0] * size, vertex[1] + left[1] * size)
    corner = (first[0] + right[0] * size, first[1] + right[1] * size)
    last = (vertex[0] + right[0] * size, vertex[1] + right[1] * size)
    return first, corner, last


def _arc_path(
    vertex: tuple[float, float],
    radius: float,
    start_angle: float,
    delta: float,
) -> tuple[tuple[float, float], ...]:
    steps = max(9, int(math.ceil(abs(delta) * radius)))
    return tuple(
        (
            vertex[0] + radius * math.cos(start_angle + delta * index / steps),
            vertex[1] + radius * math.sin(start_angle + delta * index / steps),
        )
        for index in range(steps + 1)
    )


def _arc_midpoint(path: Sequence[tuple[float, float]]) -> tuple[float, float]:
    return path[len(path) // 2]


def _path_inside_image(path: Sequence[tuple[float, float]], image_shape: tuple[int, ...]) -> bool:
    height, width = image_shape[:2]
    return all(0.0 <= x < width and 0.0 <= y < height for x, y in path)


def _segment_ink_support(
    ink: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
    radius: int,
) -> float:
    sample_count = max(7, int(round(math.dist(start, end))))
    height, width = ink.shape
    supported = 0
    for index in range(1, sample_count):
        parameter = index / sample_count
        x = int(round(start[0] + (end[0] - start[0]) * parameter))
        y = int(round(start[1] + (end[1] - start[1]) * parameter))
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        supported += int(bool(np.any(ink[y0:y1, x0:x1])))
    return supported / (sample_count - 1)


def _points_ink_support(
    ink: np.ndarray,
    points: Sequence[tuple[float, float]],
    radius: int,
) -> float:
    height, width = ink.shape
    supported = 0
    for x_value, y_value in points:
        x = int(round(x_value))
        y = int(round(y_value))
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        supported += int(bool(np.any(ink[y0:y1, x0:x1])))
    return supported / len(points) if points else 0.0


def _point_has_ink(ink: np.ndarray, point: tuple[float, float]) -> bool:
    x = int(round(point[0]))
    y = int(round(point[1]))
    height, width = ink.shape
    return 0 <= x < width and 0 <= y < height and bool(ink[y, x])


def _off_support_profile(
    ink: np.ndarray,
    center: tuple[float, float],
    direction: tuple[float, float],
    normal: tuple[float, float],
    along_radius: float,
    normal_radius: float,
    support_half_width: float,
) -> tuple[float, int, float] | None:
    height, width = ink.shape
    radius = int(math.ceil(max(along_radius, normal_radius)))
    center_x, center_y = center
    along_values = []
    signs = set()
    for y in range(max(0, int(math.floor(center_y)) - radius), min(height, int(math.ceil(center_y)) + radius + 1)):
        for x in range(max(0, int(math.floor(center_x)) - radius), min(width, int(math.ceil(center_x)) + radius + 1)):
            if not ink[y, x]:
                continue
            dx, dy = x - center_x, y - center_y
            along = dx * direction[0] + dy * direction[1]
            across = dx * normal[0] + dy * normal[1]
            if abs(along) > along_radius or not support_half_width + 0.1 < abs(across) <= normal_radius:
                continue
            along_values.append(along)
            signs.add(1 if across > 0 else -1)
    if len(along_values) < 2 or len(signs) != 2:
        return None
    return max(along_values) - min(along_values), len(along_values), sum(along_values) / len(along_values)


def _estimate_support_half_width(
    ink: np.ndarray,
    start: tuple[float, float],
    direction: tuple[float, float],
    normal: tuple[float, float],
    length: float,
) -> float:
    extents = []
    height, width = ink.shape
    for fraction in (0.14, 0.24, 0.34, 0.44, 0.56, 0.66, 0.76, 0.86):
        center = (start[0] + direction[0] * length * fraction, start[1] + direction[1] * length * fraction)
        values = []
        center_x, center_y = center
        for y in range(max(0, int(math.floor(center_y)) - 7), min(height, int(math.ceil(center_y)) + 8)):
            for x in range(max(0, int(math.floor(center_x)) - 7), min(width, int(math.ceil(center_x)) + 8)):
                if not ink[y, x]:
                    continue
                dx, dy = x - center_x, y - center_y
                along = dx * direction[0] + dy * direction[1]
                across = dx * normal[0] + dy * normal[1]
                if abs(along) <= 1.25 and abs(across) <= 6.0:
                    values.append(abs(across))
        extents.append(max(values, default=0.0))
    return max(1.0, round(float(np.median(extents)), 3))


def _line_support_half_width(
    ink: np.ndarray,
    line: Mapping[str, Any],
    direction: tuple[float, float],
) -> float:
    start = tuple(float(value) for value in line["p1"])
    length = float(line.get("length_px", math.dist(line["p1"], line["p2"])))
    normal = (-direction[1], direction[0])
    return _estimate_support_half_width(ink, start, direction, normal, length)


def _path_bbox(points: Sequence[tuple[float, float]], *, padding: float) -> tuple[float, float, float, float]:
    return (
        min(point[0] for point in points) - padding,
        min(point[1] for point in points) - padding,
        max(point[0] for point in points) + padding,
        max(point[1] for point in points) + padding,
    )


def _marker_center(marker: Mapping[str, Any]) -> tuple[float, float]:
    values = marker.get("center") or marker.get("vertex")
    return float(values[0]), float(values[1])


def _topology_points(topology: Mapping[str, Any]) -> tuple[tuple[float, float], ...]:
    points = []
    for key in (
        "endpoints",
        "point_candidates",
        "arc_endpoints",
        "line_intersections",
        "circle_line_hits",
        "circle_line_intersections",
        "circle_circle_intersections",
        "arc_line_intersections",
        "arc_circle_intersections",
        "arc_arc_intersections",
    ):
        for item in topology.get(key) or ():
            values = item.get("xy")
            if values is not None and len(values) == 2:
                points.append((float(values[0]), float(values[1])))
    return tuple(points)


def _non_maximum_markers(markers: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    ordered = sorted(
        markers,
        key=lambda item: (
            -float(item.get("template_score", min(item.get("stroke_support") or (0.0,)))),
            -float(item.get("confidence", 0.0)),
            -float(item.get("size_px", 0.0)),
            item.get("kind", ""),
        ),
    )
    for marker in ordered:
        center = _marker_center(marker)
        radius = max(4.0, float(marker.get("size_px", 0.0)) * 2.8)
        if any(
            marker["kind"] == known["kind"]
            and set(marker.get("line_ids") or ()) == set(known.get("line_ids") or ())
            and math.dist(center, _marker_center(known)) <= max(
                radius,
                float(known.get("size_px", 0.0)) * 2.8,
            )
            for known in selected
        ):
            continue
        selected.append(marker)
    return selected


def _connected_marker_groups(markers: Sequence[dict[str, Any]], compatible) -> list[list[dict[str, Any]]]:
    pending = list(markers)
    groups = []
    while pending:
        group = [pending.pop(0)]
        changed = True
        while changed:
            changed = False
            for marker in tuple(pending):
                if any(compatible(marker, member) for member in group):
                    pending.remove(marker)
                    group.append(marker)
                    changed = True
        groups.append(group)
    return groups


def _annotate_unambiguous_groups(
    groups: Sequence[Sequence[dict[str, Any]]],
    semantic_relation: str,
    *,
    permit_repeated_support: bool = False,
) -> None:
    group_index = 0
    for group in groups:
        if len(group) < 2:
            continue
        support_ids = [line_id for marker in group for line_id in marker["line_ids"]]
        if not permit_repeated_support and len(set(support_ids)) != len(support_ids):
            continue
        group_id = f"vg_{semantic_relation}_{group_index}"
        group_index += 1
        for marker in group:
            marker["relation_group_id"] = group_id
            marker["semantic_relation"] = semantic_relation
            marker["group_line_ids"] = sorted(set(support_ids))


def _comparable_size(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_size = max(1.0, float(left.get("size_px", 1.0)))
    right_size = max(1.0, float(right.get("size_px", 1.0)))
    return max(left_size, right_size) / min(left_size, right_size) <= 1.6
