"""Formal geometry state registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from .errors import DuplicateObjectError, UnknownObjectError
from .objects import Arc, Circle, Line, Plane, Point, Polygon, Polyhedron, Ray, Region, Segment, Target


@dataclass
class FormalGeometryState:
    """Executable geometry state produced by a construction trace."""

    points: Dict[str, Point] = field(default_factory=dict)
    lines: Dict[str, Line] = field(default_factory=dict)
    segments: Dict[str, Segment] = field(default_factory=dict)
    rays: Dict[str, Ray] = field(default_factory=dict)
    circles: Dict[str, Circle] = field(default_factory=dict)
    arcs: Dict[str, Arc] = field(default_factory=dict)
    polygons: Dict[str, Polygon] = field(default_factory=dict)
    regions: Dict[str, Region] = field(default_factory=dict)
    planes: Dict[str, Plane] = field(default_factory=dict)
    polyhedra: Dict[str, Polyhedron] = field(default_factory=dict)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    construction_trace: List[Dict[str, Any]] = field(default_factory=list)
    target: Target | None = None

    def add_point(self, point: Point) -> None:
        self._ensure_free(point.id)
        self.points[point.id] = point

    def add_line(self, line: Line) -> None:
        self._ensure_free(line.id)
        self.lines[line.id] = line

    def add_segment(self, segment: Segment) -> None:
        self.require_points([segment.start, segment.end])
        self._ensure_free(segment.id)
        self.segments[segment.id] = segment

    def add_ray(self, ray: Ray) -> None:
        self.require_point(ray.start)
        if ray.through:
            self.require_point(ray.through)
        self._ensure_free(ray.id)
        self.rays[ray.id] = ray

    def add_circle(self, circle: Circle) -> None:
        self.require_point(circle.center)
        self._ensure_free(circle.id)
        self.circles[circle.id] = circle

    def add_arc(self, arc: Arc) -> None:
        if arc.circle not in self.circles:
            raise UnknownObjectError(f"Unknown circle: {arc.circle}")
        self.require_points([arc.start, arc.end])
        self._ensure_free(arc.id)
        self.arcs[arc.id] = arc

    def add_polygon(self, polygon: Polygon) -> None:
        self.require_points(polygon.vertices)
        self._ensure_free(polygon.id)
        self.polygons[polygon.id] = polygon

    def add_region(self, region: Region) -> None:
        self._ensure_free(region.id)
        self.regions[region.id] = region

    def add_plane(self, plane: Plane) -> None:
        self._ensure_free(plane.id)
        self.planes[plane.id] = plane

    def add_polyhedron(self, polyhedron: Polyhedron) -> None:
        self.require_points(polyhedron.vertices)
        self._ensure_free(polyhedron.id)
        self.polyhedra[polyhedron.id] = polyhedron

    def add_relation(self, relation: Dict[str, Any]) -> None:
        self.relations.append(dict(relation))

    def set_target(self, target: Target) -> None:
        self.target = target

    def require_point(self, point_id: str) -> Point:
        if point_id not in self.points:
            raise UnknownObjectError(f"Unknown point: {point_id}")
        return self.points[point_id]

    def require_points(self, point_ids: Iterable[str]) -> None:
        for point_id in point_ids:
            self.require_point(point_id)

    def get_line_like(self, object_id: str) -> Line:
        if object_id in self.lines:
            return self.lines[object_id]
        if object_id in self.segments:
            seg = self.segments[object_id]
            a = self.require_point(seg.start)
            b = self.require_point(seg.end)
            from .geometry import sub

            return Line(id=f"line_of_{object_id}", point=a.coords, direction=sub(b.coords, a.coords), source_points=(a.id, b.id))
        raise UnknownObjectError(f"Unknown line-like object: {object_id}")

    def object_ids(self) -> set[str]:
        ids = set()
        for bucket in self._buckets():
            ids.update(bucket.keys())
        return ids

    def to_dict(self) -> Dict[str, Any]:
        return {
            "points": {k: v.to_dict() for k, v in self.points.items()},
            "lines": {k: v.to_dict() for k, v in self.lines.items()},
            "segments": {k: v.to_dict() for k, v in self.segments.items()},
            "rays": {k: v.to_dict() for k, v in self.rays.items()},
            "circles": {k: v.to_dict() for k, v in self.circles.items()},
            "arcs": {k: v.to_dict() for k, v in self.arcs.items()},
            "polygons": {k: v.to_dict() for k, v in self.polygons.items()},
            "regions": {k: v.to_dict() for k, v in self.regions.items()},
            "planes": {k: v.to_dict() for k, v in self.planes.items()},
            "polyhedra": {k: v.to_dict() for k, v in self.polyhedra.items()},
            "relations": self.relations,
            "construction_trace": self.construction_trace,
            "target": self.target.to_dict() if self.target else None,
        }

    def _ensure_free(self, object_id: str) -> None:
        if object_id in self.object_ids():
            raise DuplicateObjectError(f"Duplicate geometry object id: {object_id}")

    def _buckets(self):
        return [
            self.points,
            self.lines,
            self.segments,
            self.rays,
            self.circles,
            self.arcs,
            self.polygons,
            self.regions,
            self.planes,
            self.polyhedra,
        ]
