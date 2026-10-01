"""Pixel-grounded checks for structural ParseGraph candidates."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from geosym_world.errors import InvalidGeometryInput
from geosym_world.schema import ParseGraph, ParseObject, ParseObjectKind, ParseOrigin, ParseRelation
from geosym_world.schema.canonical import stable_checksum


PIXEL_TOPOLOGY_REPORT_SCHEMA_VERSION = "1.0"
PIXEL_TOPOLOGY_VERIFIER_VERSION = "0.2"


class PixelTopologyStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PixelTopologyConfig:
    ink_threshold: int = 210
    stroke_radius_px: float = 3.5
    sample_spacing_px: float = 1.5
    minimum_path_support: float = 0.65
    minimum_local_support: float = 0.55
    geometry_tolerance_px: float = 4.0
    local_extent_px: float = 8.0

    def __post_init__(self) -> None:
        if not 0 <= self.ink_threshold <= 255:
            raise InvalidGeometryInput("Pixel topology ink_threshold must be in [0, 255]")
        for name in ("stroke_radius_px", "sample_spacing_px", "geometry_tolerance_px", "local_extent_px"):
            if float(getattr(self, name)) <= 0.0:
                raise InvalidGeometryInput(f"Pixel topology {name} must be positive")
        for name in ("minimum_path_support", "minimum_local_support"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise InvalidGeometryInput(f"Pixel topology {name} must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ink_threshold": self.ink_threshold,
            "stroke_radius_px": self.stroke_radius_px,
            "sample_spacing_px": self.sample_spacing_px,
            "minimum_path_support": self.minimum_path_support,
            "minimum_local_support": self.minimum_local_support,
            "geometry_tolerance_px": self.geometry_tolerance_px,
            "local_extent_px": self.local_extent_px,
        }


@dataclass(frozen=True)
class PixelTopologyCheck:
    candidate_id: str
    candidate_kind: str
    status: PixelTopologyStatus
    code: str
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_kind": self.candidate_kind,
            "status": self.status.value,
            "code": self.code,
            "metrics": _finite_mapping(self.metrics),
        }


@dataclass(frozen=True)
class PixelTopologyReport:
    source_graph_checksum: str
    asset_sha256: str | None
    image_size: tuple[int, int] | None
    checks: tuple[PixelTopologyCheck, ...]
    config: PixelTopologyConfig
    schema_version: str = PIXEL_TOPOLOGY_REPORT_SCHEMA_VERSION
    verifier_version: str = PIXEL_TOPOLOGY_VERIFIER_VERSION

    def __post_init__(self) -> None:
        candidate_ids = [item.candidate_id for item in self.checks]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise InvalidGeometryInput("Pixel topology report contains duplicate candidate checks")

    @property
    def status(self) -> str:
        statuses = {item.status for item in self.checks}
        if PixelTopologyStatus.FAILED in statuses:
            return "failed"
        if PixelTopologyStatus.UNKNOWN in statuses:
            return "partial"
        return "passed" if statuses else "not_applicable"

    def check_for(self, candidate_id: str) -> PixelTopologyCheck | None:
        return next((item for item in self.checks if item.candidate_id == candidate_id), None)

    def to_dict(self) -> dict[str, Any]:
        counts = Counter(item.status.value for item in self.checks)
        payload = {
            "schema_version": self.schema_version,
            "verifier": "pixel_topology",
            "verifier_version": self.verifier_version,
            "source_graph_checksum": self.source_graph_checksum,
            "asset_sha256": self.asset_sha256,
            "image_size": list(self.image_size) if self.image_size is not None else None,
            "status": self.status,
            "config": self.config.to_dict(),
            "summary": dict(sorted(counts.items())),
            "checks": [item.to_dict() for item in self.checks],
        }
        payload["checksum"] = stable_checksum(payload)
        return payload


class PixelTopologyVerifier:
    """Verify visible paths and intersection topology against source pixels."""

    def __init__(self, config: PixelTopologyConfig | None = None):
        self.config = config or PixelTopologyConfig()

    def verify(self, graph: ParseGraph | Mapping[str, Any], image_path: str | Path) -> PixelTopologyReport:
        source = graph if isinstance(graph, ParseGraph) else ParseGraph.from_dict(graph)
        checkable = _checkable_candidates(source)
        if not checkable:
            return PixelTopologyReport(source.checksum(), None, None, (), self.config)

        path = Path(image_path)
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise InvalidGeometryInput(f"Pixel topology verifier could not read image: {path}")
        image_size = int(image.shape[1]), int(image.shape[0])
        expected_size = _parse_image_size(source.metadata.get("image_size"))
        if expected_size is not None and expected_size != image_size:
            raise InvalidGeometryInput(
                f"Pixel topology image size {image_size} does not match ParseGraph metadata {expected_size}"
            )

        ink = image <= self.config.ink_threshold
        objects = {item.id: item for item in source.objects}
        checks = []
        for candidate_kind, candidate in checkable:
            if candidate_kind == "object":
                checks.append(self._verify_object(candidate, objects, ink))
            elif candidate.predicate == "intersection":
                checks.append(self._verify_intersection(candidate, objects, ink))
            elif candidate.predicate == "point_on_line":
                checks.append(self._verify_point_on_line(candidate, objects, ink))
            else:
                checks.append(self._verify_ordered_points(candidate, objects))
        return PixelTopologyReport(
            source_graph_checksum=source.checksum(),
            asset_sha256=_file_sha256(path),
            image_size=image_size,
            checks=tuple(checks),
            config=self.config,
        )

    def _verify_object(
        self,
        candidate: ParseObject,
        objects: Mapping[str, ParseObject],
        ink: np.ndarray,
    ) -> PixelTopologyCheck:
        arc_geometry = None
        if candidate.kind in {ParseObjectKind.SEGMENT, ParseObjectKind.LINE, ParseObjectKind.RAY}:
            if not {"p1", "p2"} <= set(candidate.geometry):
                return PixelTopologyCheck(
                    candidate.id,
                    "object",
                    PixelTopologyStatus.UNKNOWN,
                    "linear_geometry_unavailable",
                )
            points = _sample_segment(candidate.geometry["p1"], candidate.geometry["p2"], self.config.sample_spacing_px)
        elif candidate.kind is ParseObjectKind.CIRCLE:
            if not {"center", "radius_px"} <= set(candidate.geometry):
                return PixelTopologyCheck(
                    candidate.id,
                    "object",
                    PixelTopologyStatus.UNKNOWN,
                    "circle_geometry_unavailable",
                )
            points = _sample_circle(
                candidate.geometry["center"],
                float(candidate.geometry["radius_px"]),
                self.config.sample_spacing_px,
            )
        elif candidate.kind is ParseObjectKind.ARC:
            arc_geometry = _arc_geometry(candidate, objects)
            if arc_geometry is None:
                return PixelTopologyCheck(
                    candidate.id,
                    "object",
                    PixelTopologyStatus.UNKNOWN,
                    "arc_geometry_unavailable",
                )
            points = _sample_arc(*arc_geometry, self.config.sample_spacing_px)
        else:
            raise AssertionError(f"Unsupported pixel topology object: {candidate.kind.value}")
        support = _object_support_ratio(
            candidate,
            ink,
            points,
            self.config.stroke_radius_px,
            arc_geometry=arc_geometry if candidate.kind is ParseObjectKind.ARC else None,
        )
        passed = support >= self.config.minimum_path_support
        return PixelTopologyCheck(
            candidate_id=candidate.id,
            candidate_kind="object",
            status=PixelTopologyStatus.PASSED if passed else PixelTopologyStatus.FAILED,
            code="stroke_path_supported" if passed else "stroke_path_missing",
            metrics={"support_ratio": round(support, 6), "sample_count": len(points)},
        )

    def _verify_intersection(
        self,
        relation: ParseRelation,
        objects: Mapping[str, ParseObject],
        ink: np.ndarray,
    ) -> PixelTopologyCheck:
        point_ref = next((item.ref for item in relation.arguments if item.role == "point"), None)
        support_refs = tuple(item.ref for item in relation.arguments if item.role == "support")
        point = _pixel_point(objects.get(point_ref)) if point_ref is not None else None
        supports = tuple(objects.get(ref) for ref in support_refs)
        if point is None or len(supports) < 2 or any(item is None for item in supports):
            return PixelTopologyCheck(
                relation.id,
                "relation",
                PixelTopologyStatus.UNKNOWN,
                "intersection_geometry_unavailable",
            )

        residuals = []
        local_support = []
        for support in supports:
            arc_geometry = _arc_geometry(support, objects) if support.kind is ParseObjectKind.ARC else None
            residual, samples = _support_residual_and_local_samples(
                support,
                point,
                self.config.local_extent_px,
                self.config.sample_spacing_px,
                objects,
            )
            if residual is None:
                return PixelTopologyCheck(
                    relation.id,
                    "relation",
                    PixelTopologyStatus.UNKNOWN,
                    "intersection_support_type_unsupported",
                    {"support_id": support.id},
                )
            residuals.append(residual)
            local_support.append(
                _object_support_ratio(
                    support,
                    ink,
                    samples,
                    self.config.stroke_radius_px,
                    arc_geometry=arc_geometry,
                )
            )

        geometry_passed = max(residuals) <= self.config.geometry_tolerance_px
        pixels_passed = min(local_support) >= self.config.minimum_local_support
        passed = geometry_passed and pixels_passed
        return PixelTopologyCheck(
            candidate_id=relation.id,
            candidate_kind="relation",
            status=PixelTopologyStatus.PASSED if passed else PixelTopologyStatus.FAILED,
            code="intersection_supported" if passed else "intersection_not_supported",
            metrics={
                "maximum_geometry_residual_px": round(max(residuals), 6),
                "minimum_local_support": round(min(local_support), 6),
                "support_count": len(supports),
            },
        )

    def _verify_ordered_points(
        self,
        relation: ParseRelation,
        objects: Mapping[str, ParseObject],
    ) -> PixelTopologyCheck:
        support_ref = next((item.ref for item in relation.arguments if item.role == "support"), None)
        support = objects.get(support_ref) if support_ref is not None else None
        point_refs = tuple(item.ref for item in relation.arguments if item.role.startswith("point_"))
        points = tuple(_pixel_point(objects.get(ref)) for ref in point_refs)
        if support is None or len(points) < 2 or any(point is None for point in points):
            return PixelTopologyCheck(
                relation.id,
                "relation",
                PixelTopologyStatus.UNKNOWN,
                "ordered_points_geometry_unavailable",
            )
        if support.kind not in {ParseObjectKind.SEGMENT, ParseObjectKind.LINE, ParseObjectKind.RAY}:
            return PixelTopologyCheck(
                relation.id,
                "relation",
                PixelTopologyStatus.UNKNOWN,
                "ordered_points_support_type_unsupported",
            )
        if not {"p1", "p2"} <= set(support.geometry):
            return PixelTopologyCheck(
                relation.id,
                "relation",
                PixelTopologyStatus.UNKNOWN,
                "ordered_points_geometry_unavailable",
            )

        parameters = []
        residuals = []
        for point in points:
            parameter, residual = _segment_projection(point, support.geometry["p1"], support.geometry["p2"])
            parameters.append(parameter)
            residuals.append(residual)
        monotonic = all(left <= right + 1e-9 for left, right in zip(parameters, parameters[1:]))
        bounded = min(parameters) >= -0.02 and max(parameters) <= 1.02
        geometry_passed = max(residuals) <= self.config.geometry_tolerance_px
        passed = monotonic and bounded and geometry_passed
        return PixelTopologyCheck(
            candidate_id=relation.id,
            candidate_kind="relation",
            status=PixelTopologyStatus.PASSED if passed else PixelTopologyStatus.FAILED,
            code="point_order_supported" if passed else "point_order_invalid",
            metrics={
                "maximum_geometry_residual_px": round(max(residuals), 6),
                "monotonic": monotonic,
                "bounded": bounded,
                "point_count": len(points),
            },
        )

    def _verify_point_on_line(
        self,
        relation: ParseRelation,
        objects: Mapping[str, ParseObject],
        ink: np.ndarray,
    ) -> PixelTopologyCheck:
        point_ref = next((item.ref for item in relation.arguments if item.role == "point"), None)
        support_ref = next(
            (item.ref for item in relation.arguments if item.role in {"support", "line", "object"}),
            None,
        )
        point = _pixel_point(objects.get(point_ref)) if point_ref is not None else None
        support = objects.get(support_ref) if support_ref is not None else None
        if point is None or support is None:
            return PixelTopologyCheck(
                relation.id,
                "relation",
                PixelTopologyStatus.UNKNOWN,
                "point_on_line_geometry_unavailable",
            )
        arc_geometry = _arc_geometry(support, objects) if support.kind is ParseObjectKind.ARC else None
        residual, samples = _support_residual_and_local_samples(
            support,
            point,
            self.config.local_extent_px,
            self.config.sample_spacing_px,
            objects,
        )
        if residual is None:
            return PixelTopologyCheck(
                relation.id,
                "relation",
                PixelTopologyStatus.UNKNOWN,
                "point_on_line_support_type_unsupported",
                {"support_id": support.id},
            )
        bounded = True
        if support.kind in {ParseObjectKind.SEGMENT, ParseObjectKind.LINE, ParseObjectKind.RAY}:
            if not {"p1", "p2"} <= set(support.geometry):
                return PixelTopologyCheck(
                    relation.id,
                    "relation",
                    PixelTopologyStatus.UNKNOWN,
                    "point_on_line_geometry_unavailable",
                )
            parameter, _ = _segment_projection(point, support.geometry["p1"], support.geometry["p2"])
            bounded = -0.02 <= parameter <= 1.02
        local_support = _object_support_ratio(
            support,
            ink,
            samples,
            self.config.stroke_radius_px,
            arc_geometry=arc_geometry,
        )
        passed = residual <= self.config.geometry_tolerance_px and bounded and local_support >= self.config.minimum_local_support
        return PixelTopologyCheck(
            relation.id,
            "relation",
            PixelTopologyStatus.PASSED if passed else PixelTopologyStatus.FAILED,
            "point_on_support" if passed else "point_on_support_not_supported",
            {
                "geometry_residual_px": round(residual, 6),
                "bounded": bounded,
                "local_support": round(local_support, 6),
            },
        )


def _checkable_candidates(graph: ParseGraph) -> tuple[tuple[str, Any], ...]:
    objects = tuple(
        ("object", item)
        for item in graph.objects
        if item.origin in {ParseOrigin.IMAGE, ParseOrigin.IMAGE_TEXT}
        and item.kind in {
            ParseObjectKind.SEGMENT,
            ParseObjectKind.LINE,
            ParseObjectKind.RAY,
            ParseObjectKind.CIRCLE,
            ParseObjectKind.ARC,
        }
        and _has_pixel_geometry(item)
    )
    relations = tuple(
        ("relation", item)
        for item in graph.relations
        if item.origin in {ParseOrigin.IMAGE, ParseOrigin.IMAGE_TEXT}
        and item.predicate in {"intersection", "ordered_points", "point_on_line"}
    )
    return (*objects, *relations)


def _has_pixel_geometry(candidate: ParseObject) -> bool:
    if candidate.kind in {ParseObjectKind.SEGMENT, ParseObjectKind.LINE, ParseObjectKind.RAY}:
        return "p1" in candidate.geometry and "p2" in candidate.geometry
    if candidate.kind is ParseObjectKind.CIRCLE:
        return "center" in candidate.geometry and "radius_px" in candidate.geometry
    if candidate.kind is ParseObjectKind.ARC:
        return (
            candidate.geometry.get("orientation") in {"ccw", "cw"}
            and "center" in candidate.metadata
            and "radius_px" in candidate.metadata
            and "sweep_deg" in candidate.metadata
        )
    return False


def _finite_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _finite_value(item) for key, item in value.items()}


def _finite_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _finite_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_finite_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _parse_image_size(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise InvalidGeometryInput("ParseGraph metadata image_size must be a [width, height] pair")
    try:
        width, height = (int(item) for item in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidGeometryInput("ParseGraph metadata image_size must contain two integers") from exc
    if width <= 0 or height <= 0:
        raise InvalidGeometryInput("ParseGraph metadata image_size must be positive")
    return width, height


def _sample_segment(p1: Sequence[float], p2: Sequence[float], spacing: float) -> tuple[tuple[float, float], ...]:
    start = np.asarray(p1, dtype=float)
    end = np.asarray(p2, dtype=float)
    length = float(np.linalg.norm(end - start))
    count = max(2, int(math.ceil(length / spacing)) + 1)
    return tuple(tuple(start + parameter * (end - start)) for parameter in np.linspace(0.0, 1.0, count))


def _sample_circle(center: Sequence[float], radius: float, spacing: float) -> tuple[tuple[float, float], ...]:
    count = max(24, int(math.ceil(2.0 * math.pi * radius / spacing)))
    cx, cy = (float(value) for value in center)
    return tuple(
        (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
        for angle in np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    )


def _sample_arc(
    center: Sequence[float],
    radius: float,
    start_angle: float,
    direction: float,
    sweep: float,
    spacing: float,
) -> tuple[tuple[float, float], ...]:
    count = max(3, int(math.ceil(radius * sweep / spacing)) + 1)
    cx, cy = (float(value) for value in center)
    return tuple(
        (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
        for angle in np.linspace(start_angle, start_angle + direction * sweep, count)
    )


def _object_support_ratio(
    candidate: ParseObject,
    ink: np.ndarray,
    points: Sequence[Sequence[float]],
    radius: float,
    *,
    arc_geometry: tuple[tuple[float, float], float, float, float, float] | None = None,
) -> float:
    if candidate.kind in {ParseObjectKind.SEGMENT, ParseObjectKind.LINE, ParseObjectKind.RAY}:
        start = np.asarray(candidate.geometry["p1"], dtype=float)
        end = np.asarray(candidate.geometry["p2"], dtype=float)
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length == 0.0:
            return 0.0
        normal = np.asarray((-direction[1], direction[0]), dtype=float) / length
        flags = (_cross_section_has_ink(ink, point, normal, radius) for point in points)
    elif candidate.kind is ParseObjectKind.CIRCLE:
        center = np.asarray(candidate.geometry["center"], dtype=float)
        flags = (
            _cross_section_has_ink(
                ink,
                point,
                _radial_normal(point, center),
                radius,
            )
            for point in points
        )
    elif candidate.kind is ParseObjectKind.ARC and arc_geometry is not None:
        center = np.asarray(arc_geometry[0], dtype=float)
        flags = (
            _cross_section_has_ink(
                ink,
                point,
                _radial_normal(point, center),
                radius,
            )
            for point in points
        )
    else:
        return 0.0
    values = tuple(flags)
    return float(sum(values) / len(values)) if values else 0.0


def _cross_section_has_ink(
    ink: np.ndarray,
    point: Sequence[float],
    normal: np.ndarray,
    radius: float,
) -> bool:
    height, width = ink.shape[:2]
    point_array = np.asarray(point, dtype=float)
    count = max(3, int(math.ceil(2.0 * radius)) + 1)
    for offset in np.linspace(-radius, radius, count):
        x, y = point_array + offset * normal
        column = int(round(float(x)))
        row = int(round(float(y)))
        if 0 <= row < height and 0 <= column < width and bool(ink[row, column]):
            return True
    return False


def _radial_normal(point: Sequence[float], center: np.ndarray) -> np.ndarray:
    vector = np.asarray(point, dtype=float) - center
    length = float(np.linalg.norm(vector))
    return vector / length if length else np.asarray((1.0, 0.0))


def _pixel_point(candidate: ParseObject | None) -> tuple[float, float] | None:
    if candidate is None or candidate.kind is not ParseObjectKind.POINT or "pixel" not in candidate.geometry:
        return None
    return tuple(float(value) for value in candidate.geometry["pixel"])


def _support_residual_and_local_samples(
    support: ParseObject,
    point: tuple[float, float],
    extent: float,
    spacing: float,
    objects: Mapping[str, ParseObject],
) -> tuple[float | None, tuple[tuple[float, float], ...]]:
    if support.kind in {ParseObjectKind.SEGMENT, ParseObjectKind.LINE, ParseObjectKind.RAY}:
        if not {"p1", "p2"} <= set(support.geometry):
            return None, ()
        parameter, residual = _segment_projection(point, support.geometry["p1"], support.geometry["p2"])
        start = np.asarray(support.geometry["p1"], dtype=float)
        end = np.asarray(support.geometry["p2"], dtype=float)
        length = float(np.linalg.norm(end - start))
        if length == 0.0:
            return None, ()
        delta = min(0.5, extent / length)
        lower = max(0.0, parameter - delta)
        upper = min(1.0, parameter + delta)
        count = max(3, int(math.ceil((upper - lower) * length / spacing)) + 1)
        samples = tuple(tuple(start + value * (end - start)) for value in np.linspace(lower, upper, count))
        return residual, samples
    if support.kind is ParseObjectKind.CIRCLE:
        if not {"center", "radius_px"} <= set(support.geometry):
            return None, ()
        center = np.asarray(support.geometry["center"], dtype=float)
        radius = float(support.geometry["radius_px"])
        vector = np.asarray(point, dtype=float) - center
        distance = float(np.linalg.norm(vector))
        if radius <= 0.0 or distance == 0.0:
            return None, ()
        angle = math.atan2(vector[1], vector[0])
        delta = min(math.pi, extent / radius)
        count = max(3, int(math.ceil(2.0 * extent / spacing)) + 1)
        samples = tuple(
            (center[0] + radius * math.cos(value), center[1] + radius * math.sin(value))
            for value in np.linspace(angle - delta, angle + delta, count)
        )
        return abs(distance - radius), samples
    if support.kind is ParseObjectKind.ARC:
        arc_geometry = _arc_geometry(support, objects)
        if arc_geometry is None:
            return None, ()
        center, radius, start_angle, direction, sweep = arc_geometry
        vector = np.asarray(point, dtype=float) - np.asarray(center, dtype=float)
        distance = float(np.linalg.norm(vector))
        if radius <= 0.0 or distance == 0.0:
            return None, ()
        point_angle = math.atan2(vector[1], vector[0])
        delta = ((point_angle - start_angle) * direction) % (2.0 * math.pi)
        if delta > sweep + 1e-6:
            return float("inf"), ()
        angular_extent = min(sweep, extent / radius)
        lower = max(0.0, delta - angular_extent)
        upper = min(sweep, delta + angular_extent)
        count = max(3, int(math.ceil((upper - lower) * radius / spacing)) + 1)
        samples = tuple(
            (
                center[0] + radius * math.cos(start_angle + direction * value),
                center[1] + radius * math.sin(start_angle + direction * value),
            )
            for value in np.linspace(lower, upper, count)
        )
        return abs(distance - radius), samples
    return None, ()


def _arc_geometry(
    candidate: ParseObject,
    objects: Mapping[str, ParseObject],
) -> tuple[tuple[float, float], float, float, float, float] | None:
    circle_ref = next((item.ref for item in candidate.references if item.role == "circle"), None)
    circle = objects.get(circle_ref) if circle_ref is not None else None
    metadata_center = candidate.metadata.get("center")
    metadata_radius = candidate.metadata.get("radius_px")
    if metadata_center is not None and metadata_radius is not None:
        center = tuple(float(value) for value in metadata_center)
        radius = float(metadata_radius)
    elif circle is not None and circle.kind is ParseObjectKind.CIRCLE:
        center = tuple(float(value) for value in circle.geometry["center"])
        radius = float(circle.geometry["radius_px"])
    else:
        return None
    start_ref = next((item.ref for item in candidate.references if item.role == "start"), None)
    end_ref = next((item.ref for item in candidate.references if item.role == "end"), None)
    start_object = objects.get(start_ref) if start_ref is not None else None
    end_object = objects.get(end_ref) if end_ref is not None else None
    if start_object is None or end_object is None:
        return None
    start_pixel = start_object.geometry.get("pixel")
    end_pixel = end_object.geometry.get("pixel")
    if start_pixel is None or end_pixel is None:
        return None
    start_angle = math.atan2(float(start_pixel[1]) - center[1], float(start_pixel[0]) - center[0])
    image_orientation = candidate.metadata.get("image_orientation", candidate.geometry.get("orientation"))
    direction = -1.0 if image_orientation == "cw" else 1.0
    sweep_value = candidate.metadata.get("sweep_deg")
    if sweep_value is None:
        end_angle = math.atan2(float(end_pixel[1]) - center[1], float(end_pixel[0]) - center[0])
        sweep = ((end_angle - start_angle) * direction) % (2.0 * math.pi)
    else:
        sweep = math.radians(float(sweep_value))
    if not math.isfinite(radius) or radius <= 0.0 or not 0.0 < sweep <= 2.0 * math.pi:
        return None
    return center, radius, start_angle, direction, sweep


def _segment_projection(
    point: Sequence[float],
    p1: Sequence[float],
    p2: Sequence[float],
) -> tuple[float, float]:
    start = np.asarray(p1, dtype=float)
    end = np.asarray(p2, dtype=float)
    vector = end - start
    denominator = float(np.dot(vector, vector))
    if denominator == 0.0:
        return 0.0, float("inf")
    parameter = float(np.dot(np.asarray(point, dtype=float) - start, vector) / denominator)
    projection = start + parameter * vector
    return parameter, float(np.linalg.norm(np.asarray(point, dtype=float) - projection))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
