"""Formal geometry state registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping

import sympy as sp

from geosym_world.errors import (
    DegenerateGeometryError,
    DuplicateObjectError,
    InvalidGeometryInput,
    NumericalInstabilityError,
    UnknownObjectError,
)
from geosym_world.schema.entities import Arc, Circle, CompositeRegion, Line, Plane, Point, Polygon, Polyhedron, Ray, Region, Segment, Target
from geosym_world.schema.ids import validate_geometry_id
from geosym_world.schema.relation_normalization import normalize_relation, normalize_target_type
from geosym_world.kernel.geometry2d.analytic import distance

STATE_SCHEMA_VERSION = "1.1"
PROOF_TARGET_TYPES = frozenset({
    "parallel",
    "perpendicular",
    "collinear",
    "tangent_to_circle",
    "equal_length",
    "equal_angle",
    "similar",
    "congruent",
    "midpoint",
    "cyclic_quadrilateral",
})


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
    composite_regions: Dict[str, CompositeRegion] = field(default_factory=dict)
    planes: Dict[str, Plane] = field(default_factory=dict)
    polyhedra: Dict[str, Polyhedron] = field(default_factory=dict)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    construction_trace: List[Dict[str, Any]] = field(default_factory=list)
    target: Target | None = None
    schema_version: str = STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STATE_SCHEMA_VERSION:
            raise InvalidGeometryInput(f"Unsupported geometry state schema version: {self.schema_version}")

    def add_point(self, point: Point) -> None:
        self._validate_id(point.id)
        _validate_coords(point.coords, label=f"point {point.id}")
        self._ensure_free(point.id)
        self.points[point.id] = point

    def add_line(self, line: Line) -> None:
        self._validate_id(line.id)
        _validate_vector_pair(line.point, line.direction, label=f"line {line.id}")
        self._require_existing_points(line.source_points)
        self._ensure_free(line.id)
        self.lines[line.id] = line

    def add_segment(self, segment: Segment) -> None:
        self.require_points([segment.start, segment.end])
        if segment.start == segment.end or self.points[segment.start].coords == self.points[segment.end].coords:
            raise DegenerateGeometryError(f"Segment endpoints must be distinct: {segment.id}")
        self._validate_id(segment.id)
        self._ensure_free(segment.id)
        self.segments[segment.id] = segment

    def add_ray(self, ray: Ray) -> None:
        start = self.require_point(ray.start)
        if ray.through:
            through = self.require_point(ray.through)
            if through.coords == start.coords:
                raise DegenerateGeometryError(f"Ray start and through point must be distinct: {ray.id}")
        _validate_vector(ray.direction, dimension=start.dim, label=f"ray {ray.id} direction")
        self._validate_id(ray.id)
        self._ensure_free(ray.id)
        self.rays[ray.id] = ray

    def add_circle(self, circle: Circle) -> None:
        self.require_point(circle.center)
        self._validate_id(circle.id)
        _validate_positive_expr(circle.radius, label=f"circle {circle.id} radius")
        self._ensure_free(circle.id)
        self.circles[circle.id] = circle

    def add_arc(self, arc: Arc) -> None:
        self._validate_id(arc.id)
        self._validate_arc_reference(arc)
        self._ensure_free(arc.id)
        self.arcs[arc.id] = arc

    def add_polygon(self, polygon: Polygon) -> None:
        self.require_points(polygon.vertices)
        self._validate_id(polygon.id)
        if len(polygon.vertices) < 3 or len(set(polygon.vertices)) != len(polygon.vertices):
            raise DegenerateGeometryError(f"Polygon needs three distinct vertices: {polygon.id}")
        dimensions = {self.points[point_id].dim for point_id in polygon.vertices}
        if len(dimensions) != 1:
            raise InvalidGeometryInput(f"Polygon vertices must have one dimension: {polygon.id}")
        self._validate_polygon_area_formula(polygon)
        self._validate_polygon_perimeter_formula(polygon)
        self._ensure_free(polygon.id)
        self.polygons[polygon.id] = polygon

    def add_region(self, region: Region) -> None:
        self._validate_id(region.id)
        if not region.boundary:
            raise InvalidGeometryInput(f"Region boundary cannot be empty: {region.id}")
        self._require_existing_objects(region.boundary)
        self._require_existing_objects(region.holes)
        self._validate_sector_region(region)
        self._validate_circular_segment_region(region)
        self._validate_lens_region(region)
        self._validate_annulus_region(region)
        self._validate_region_area_formula(region)
        self._validate_region_perimeter_formula(region)
        self._ensure_free(region.id)
        self.regions[region.id] = region

    def add_composite_region(self, region: CompositeRegion) -> None:
        self._validate_id(region.id)
        if region.operation not in {"union", "intersection", "difference", "sum", "partition"}:
            raise InvalidGeometryInput(f"Unsupported composite region operation: {region.operation}")
        if not region.components:
            raise InvalidGeometryInput(f"Composite region needs components: {region.id}")
        if len(set(region.components)) != len(region.components):
            raise DegenerateGeometryError(f"Composite region repeats components: {region.id}")
        for component in region.components:
            if component not in self.regions and component not in self.composite_regions:
                raise UnknownObjectError(f"Unknown region component: {component}")
        self._validate_partition_region(region)
        self._validate_composite_area_formula(region)
        self._ensure_free(region.id)
        self.composite_regions[region.id] = region

    def add_plane(self, plane: Plane) -> None:
        self._validate_id(plane.id)
        _validate_vector(plane.point, dimension=3, label=f"plane {plane.id} point")
        _validate_vector(plane.normal, dimension=3, label=f"plane {plane.id} normal")
        _validate_nonzero(plane.normal, label=f"plane {plane.id} normal")
        self._require_existing_points(plane.source_points, dimension=3)
        self._ensure_free(plane.id)
        self.planes[plane.id] = plane

    def add_polyhedron(self, polyhedron: Polyhedron) -> None:
        self.require_points(polyhedron.vertices)
        self._validate_id(polyhedron.id)
        if len(polyhedron.vertices) < 4 or len(set(polyhedron.vertices)) != len(polyhedron.vertices):
            raise DegenerateGeometryError(f"Polyhedron needs four distinct vertices: {polyhedron.id}")
        if any(self.points[point_id].dim != 3 for point_id in polyhedron.vertices):
            raise InvalidGeometryInput(f"Polyhedron vertices must be 3D: {polyhedron.id}")
        vertex_ids = set(polyhedron.vertices)
        for face in polyhedron.faces:
            if len(face) < 3 or len(set(face)) != len(face) or not set(face).issubset(vertex_ids):
                raise InvalidGeometryInput(f"Polyhedron face references are invalid: {polyhedron.id}")
        self._ensure_free(polyhedron.id)
        self.polyhedra[polyhedron.id] = polyhedron

    def add_relation(self, relation: Dict[str, Any]) -> None:
        normalized = normalize_relation(relation)
        self._validate_relation_references(normalized)
        if any(existing.get("id") == normalized["id"] for existing in self.relations):
            raise DuplicateObjectError(f"Duplicate geometry relation id: {normalized['id']}")
        self.relations.append(normalized)

    def set_target(self, target: Target) -> None:
        if not isinstance(target, Target):
            raise InvalidGeometryInput("State target must be a Target")
        normalized_type = normalize_target_type(target.type)
        if not normalized_type:
            raise InvalidGeometryInput("Target type must be non-empty")
        normalized = Target(
            type=normalized_type,
            object=target.object,
            metadata=dict(target.metadata),
            mode=target.mode,
            expectation=dict(target.expectation) if target.expectation is not None else None,
        )
        self._validate_target_mode(normalized)
        self._validate_area_ratio_target(normalized)
        self.target = normalized

    def validate(self) -> None:
        """Validate references and invariants of an already assembled state.

        The mutating ``add_*`` methods validate incrementally.  This boundary
        check is also required for callers that assemble dataclass mappings
        directly or mutate a copied state during a transaction.
        """

        if self.schema_version != STATE_SCHEMA_VERSION:
            raise InvalidGeometryInput(f"Unsupported geometry state schema version: {self.schema_version}")
        ids: set[str] = set()
        for bucket in self._buckets():
            for object_id in bucket:
                self._validate_id(object_id)
                if object_id != getattr(bucket[object_id], "id", object_id):
                    raise InvalidGeometryInput(f"Object registry key does not match object id: {object_id}")
                if object_id in ids:
                    raise DuplicateObjectError(f"Duplicate geometry object id: {object_id}")
                ids.add(object_id)
        for point in self.points.values():
            _validate_coords(point.coords, label=f"point {point.id}")
        for line in self.lines.values():
            _validate_vector_pair(line.point, line.direction, label=f"line {line.id}")
            self._require_existing_points(line.source_points)
        for segment in self.segments.values():
            self._validate_segment_reference(segment)
        for ray in self.rays.values():
            self._validate_ray_reference(ray)
        for circle in self.circles.values():
            self.require_point(circle.center)
            _validate_positive_expr(circle.radius, label=f"circle {circle.id} radius")
        for arc in self.arcs.values():
            self._validate_arc_reference(arc)
        for polygon in self.polygons.values():
            self._validate_polygon_reference(polygon)
        for region in self.regions.values():
            self._validate_region_reference(region)
        for region in self.composite_regions.values():
            self._validate_composite_region_reference(region)
        for plane in self.planes.values():
            _validate_vector(plane.point, dimension=3, label=f"plane {plane.id} point")
            _validate_vector(plane.normal, dimension=3, label=f"plane {plane.id} normal")
            _validate_nonzero(plane.normal, label=f"plane {plane.id} normal")
            self._require_existing_points(plane.source_points, dimension=3)
        for polyhedron in self.polyhedra.values():
            self._validate_polyhedron_reference(polyhedron)
        normalized_relations = [normalize_relation(relation) for relation in self.relations]
        relation_ids = [relation["id"] for relation in normalized_relations]
        if len(set(relation_ids)) != len(relation_ids):
            raise DuplicateObjectError("Geometry relation IDs must be unique")
        for relation in normalized_relations:
            self._validate_relation_references(relation)
        self.relations[:] = normalized_relations
        if self.target is not None:
            target = Target(
                type=normalize_target_type(self.target.type),
                object=self.target.object,
                metadata=dict(self.target.metadata),
                mode=self.target.mode,
                expectation=dict(self.target.expectation) if self.target.expectation is not None else None,
            )
            self._validate_target(target)
            self.target = target

    def _validate_target(self, target: Target) -> None:
        if not isinstance(target, Target):
            raise InvalidGeometryInput("State target must be a Target")
        normalized_type = normalize_target_type(target.type)
        if not normalized_type:
            raise InvalidGeometryInput("Target type must be non-empty")
        validate_geometry_id(target.object)
        self._validate_target_mode(target)
        self._validate_area_ratio_target(target)

    def _validate_target_mode(self, target: Target) -> None:
        if target.mode == "proof" and target.type not in PROOF_TARGET_TYPES:
            raise InvalidGeometryInput(f"Unsupported proof target type: {target.type}")
        if target.mode == "constraint":
            try:
                sp.sympify(target.expectation["value"])
            except (TypeError, ValueError, SyntaxError) as exc:
                raise InvalidGeometryInput("Constraint target expected value must be a valid expression") from exc

    def _validate_area_ratio_target(self, target: Target) -> None:
        if normalize_target_type(target.type) != "ratio" or target.metadata.get("ratio_kind") != "area":
            return
        numerator = target.metadata.get("numerator")
        denominator = target.metadata.get("denominator")
        if not isinstance(numerator, str) or not isinstance(denominator, str):
            raise InvalidGeometryInput("Area-ratio target requires numerator and denominator region IDs")
        if numerator == denominator:
            raise InvalidGeometryInput("Area-ratio target requires distinct numerator and denominator regions")
        known_regions = set(self.regions) | set(self.composite_regions)
        missing = [region_id for region_id in (numerator, denominator) if region_id not in known_regions]
        if missing:
            raise UnknownObjectError(f"Unknown area-ratio region: {missing[0]}")

    def _validate_segment_reference(self, segment: Segment) -> None:
        self.require_points([segment.start, segment.end])
        if segment.start == segment.end or self.points[segment.start].coords == self.points[segment.end].coords:
            raise DegenerateGeometryError(f"Segment endpoints must be distinct: {segment.id}")

    def _validate_ray_reference(self, ray: Ray) -> None:
        start = self.require_point(ray.start)
        if ray.through:
            through = self.require_point(ray.through)
            if through.coords == start.coords:
                raise DegenerateGeometryError(f"Ray start and through point must be distinct: {ray.id}")
        _validate_vector(ray.direction, dimension=start.dim, label=f"ray {ray.id} direction")
        _validate_nonzero(ray.direction, label=f"ray {ray.id} direction")

    def _validate_arc_reference(self, arc: Arc) -> None:
        if arc.circle not in self.circles:
            raise UnknownObjectError(f"Unknown circle: {arc.circle}")
        self.require_points([arc.start, arc.end])
        if self.points[arc.start].dim != 2 or self.points[arc.end].dim != 2:
            raise InvalidGeometryInput(f"Arc endpoints must be 2D points: {arc.id}")
        if arc.orientation not in {"ccw", "cw"}:
            raise InvalidGeometryInput(f"Arc orientation must be 'ccw' or 'cw': {arc.id}")
        circle = self.circles[arc.circle]
        center = self.require_point(circle.center).coords
        start = self.require_point(arc.start).coords
        end = self.require_point(arc.end).coords
        from geosym_world.kernel.geometry2d.analytic import distance
        from geosym_world.kernel.geometry2d.intersections import directed_arc_angle_2d

        if sp.simplify(distance(center, start) - circle.radius) != 0 or sp.simplify(distance(center, end) - circle.radius) != 0:
            raise InvalidGeometryInput(f"Arc endpoints must lie on its circle: {arc.id}")
        allow_full_circle = bool(arc.metadata.get("allow_full_circle"))
        if arc.start == arc.end and not allow_full_circle:
            raise DegenerateGeometryError(f"Arc endpoints must be distinct unless full-circle metadata is set: {arc.id}")
        if arc.central_angle is not None:
            _validate_positive_expr(arc.central_angle, label=f"arc {arc.id} central angle")
            if sp.simplify(2 * sp.pi - arc.central_angle).is_nonnegative is not True:
                raise InvalidGeometryInput(f"Arc central angle must not exceed 2*pi: {arc.id}")
            expected = sp.Integer(2) * sp.pi if allow_full_circle and arc.start == arc.end else directed_arc_angle_2d(
                center,
                start,
                end,
                orientation=arc.orientation,
            )
            if sp.simplify(expected - arc.central_angle) != 0:
                raise InvalidGeometryInput(f"Arc central angle does not match its endpoints and orientation: {arc.id}")

    def _validate_polygon_reference(self, polygon: Polygon) -> None:
        self.require_points(polygon.vertices)
        if len(polygon.vertices) < 3 or len(set(polygon.vertices)) != len(polygon.vertices):
            raise DegenerateGeometryError(f"Polygon needs three distinct vertices: {polygon.id}")
        if len({self.points[point_id].dim for point_id in polygon.vertices}) != 1:
            raise InvalidGeometryInput(f"Polygon vertices must have one dimension: {polygon.id}")
        self._validate_polygon_area_formula(polygon)
        self._validate_polygon_perimeter_formula(polygon)

    def _validate_polygon_area_formula(self, polygon: Polygon) -> None:
        formula = polygon.metadata.get("area_formula")
        if formula is None:
            return
        if not isinstance(formula, Mapping):
            raise InvalidGeometryInput(f"Polygon area_formula must be a mapping: {polygon.id}")
        rule = formula.get("rule")
        if rule in {"triangle_base_height", "rectangle_base_height", "parallelogram_base_height"}:
            self._require_segment_reference(formula.get("base"), f"{rule} base", polygon.id)
            self._require_length_reference(formula.get("height"), f"{rule} height", polygon.id)
            return
        if rule == "trapezoid_parallel_bases_height":
            bases = formula.get("bases")
            if not isinstance(bases, (list, tuple)) or len(bases) != 2:
                raise InvalidGeometryInput(f"trapezoid_parallel_bases_height requires two bases: {polygon.id}")
            for base in bases:
                self._require_segment_reference(base, "trapezoid base", polygon.id)
            self._require_length_reference(formula.get("height"), "trapezoid height", polygon.id)
            return
        if rule == "kite_perpendicular_diagonals":
            diagonals = formula.get("diagonals")
            if not isinstance(diagonals, (list, tuple)) or len(diagonals) != 2:
                raise InvalidGeometryInput(f"kite_perpendicular_diagonals requires two diagonals: {polygon.id}")
            for diagonal in diagonals:
                self._require_segment_reference(diagonal, "kite diagonal", polygon.id)
            return
        if rule == "chord_triangle_radius_angle":
            if len(polygon.vertices) != 3:
                raise InvalidGeometryInput(f"chord_triangle_radius_angle requires a triangle: {polygon.id}")
            sector_id = formula.get("sector_region")
            if not isinstance(sector_id, str) or sector_id not in self.regions or self.regions[sector_id].kind != "sector_region":
                raise UnknownObjectError(f"chord_triangle_radius_angle requires an existing sector region: {polygon.id}")
            return
        raise InvalidGeometryInput(f"Unsupported polygon area formula: {polygon.id}")

    def _require_segment_reference(self, reference: Any, label: str, object_id: str) -> None:
        if not isinstance(reference, str) or reference not in self.segments:
            raise UnknownObjectError(f"{label} requires an existing segment: {object_id}")

    def _require_length_reference(self, reference: Any, label: str, object_id: str) -> None:
        if isinstance(reference, str):
            self._require_segment_reference(reference, label, object_id)
            return
        if not isinstance(reference, (list, tuple)) or len(reference) != 2:
            raise InvalidGeometryInput(f"{label} requires a segment or point pair: {object_id}")
        self.require_points(tuple(str(item) for item in reference))

    def _validate_polygon_perimeter_formula(self, polygon: Polygon) -> None:
        formula = polygon.metadata.get("perimeter_formula")
        if formula is None:
            return
        if not isinstance(formula, dict) or formula.get("rule") != "rectangle_side_perimeter":
            raise InvalidGeometryInput(f"Unsupported polygon perimeter formula: {polygon.id}")
        if len(polygon.vertices) != 4:
            raise InvalidGeometryInput(f"rectangle_side_perimeter requires four polygon vertices: {polygon.id}")
        references = (formula.get("width"), formula.get("height"))
        if any(not isinstance(reference, str) or reference not in self.segments for reference in references):
            raise UnknownObjectError(f"rectangle_side_perimeter requires existing width and height segments: {polygon.id}")
        if references[0] == references[1]:
            raise InvalidGeometryInput(f"rectangle_side_perimeter requires distinct segments: {polygon.id}")

    def _validate_region_reference(self, region: Region) -> None:
        if not region.boundary:
            raise InvalidGeometryInput(f"Region boundary cannot be empty: {region.id}")
        self._require_existing_objects(region.boundary)
        self._require_existing_objects(region.holes)
        self._validate_sector_region(region)
        self._validate_circular_segment_region(region)
        self._validate_lens_region(region)
        self._validate_annulus_region(region)
        self._validate_region_area_formula(region)
        self._validate_region_perimeter_formula(region)

    def _validate_annulus_region(self, region: Region) -> None:
        if region.kind != "annulus_region":
            return
        inner_id = region.metadata.get("inner_circle")
        outer_id = region.metadata.get("outer_circle")
        if not isinstance(inner_id, str) or inner_id not in self.circles:
            raise UnknownObjectError(f"Unknown inner circle for annulus region: {inner_id}")
        if not isinstance(outer_id, str) or outer_id not in self.circles:
            raise UnknownObjectError(f"Unknown outer circle for annulus region: {outer_id}")
        inner = self.circles[inner_id]
        outer = self.circles[outer_id]
        if inner.center != outer.center:
            raise InvalidGeometryInput(f"Annulus circles must be concentric: {region.id}")
        if region.boundary != (outer.center,):
            raise InvalidGeometryInput(f"Annulus boundary must reference its common center: {region.id}")
        if sp.simplify(outer.radius - inner.radius).is_positive is not True:
            raise DegenerateGeometryError(f"Annulus outer radius must exceed inner radius: {region.id}")

    def _validate_sector_region(self, region: Region) -> None:
        if region.kind != "sector_region":
            return
        circle_id = region.metadata.get("circle")
        if not isinstance(circle_id, str) or circle_id not in self.circles:
            raise UnknownObjectError(f"Unknown circle for sector region: {circle_id}")
        circle = self.circles[circle_id]
        center_id = region.metadata.get("center")
        start_id = region.metadata.get("start")
        end_id = region.metadata.get("end")
        if center_id != circle.center or region.boundary != (center_id, start_id, end_id):
            raise InvalidGeometryInput(f"Sector boundary must match its circle center and endpoints: {region.id}")
        self.require_points((center_id, start_id, end_id))
        if len({center_id, start_id, end_id}) != 3:
            raise DegenerateGeometryError(f"Sector requires distinct center and endpoints: {region.id}")
        theta = sp.sympify(region.metadata.get("central_angle"))
        if theta.is_positive is not True or sp.simplify(sp.pi - theta).is_nonnegative is not True:
            raise InvalidGeometryInput(f"Sector central angle must satisfy 0 < theta <= pi: {region.id}")
        radius_segment = region.metadata.get("radius_segment")
        if not isinstance(radius_segment, str) or radius_segment not in self.segments:
            raise UnknownObjectError(f"Sector region requires an existing radius segment: {region.id}")
        for key, rule in (
            ("area_formula", "sector_radius_angle_area"),
            ("arc_length_formula", "sector_radius_angle_arc_length"),
            ("perimeter_formula", "sector_radius_angle_perimeter"),
        ):
            formula = region.metadata.get(key)
            if not isinstance(formula, Mapping) or formula.get("rule") != rule or formula.get("radius_segment") != radius_segment:
                raise InvalidGeometryInput(f"Sector region requires {rule} with its radius segment: {region.id}")

    def _validate_circular_segment_region(self, region: Region) -> None:
        if region.kind != "circular_segment_region":
            return
        sector_id = region.metadata.get("sector_region")
        triangle_id = region.metadata.get("triangle_polygon")
        arc_id = region.metadata.get("arc")
        chord_id = region.metadata.get("chord")
        if not isinstance(sector_id, str) or sector_id not in self.regions or self.regions[sector_id].kind != "sector_region":
            raise UnknownObjectError(f"Circular segment requires an existing sector region: {region.id}")
        if not isinstance(triangle_id, str) or triangle_id not in self.polygons:
            raise UnknownObjectError(f"Circular segment requires an existing chord triangle: {region.id}")
        if not isinstance(arc_id, str) or arc_id not in self.arcs:
            raise UnknownObjectError(f"Circular segment requires an existing arc: {region.id}")
        if not isinstance(chord_id, str) or chord_id not in self.segments:
            raise UnknownObjectError(f"Circular segment requires an existing chord segment: {region.id}")
        sector = self.regions[sector_id]
        arc = self.arcs[arc_id]
        chord = self.segments[chord_id]
        endpoints = {sector.metadata["start"], sector.metadata["end"]}
        if arc.circle != sector.metadata["circle"] or {arc.start, arc.end} != endpoints or {chord.start, chord.end} != endpoints:
            raise InvalidGeometryInput(f"Circular segment arc and chord must match its sector endpoints: {region.id}")
        if region.boundary != (arc_id, chord_id):
            raise InvalidGeometryInput(f"Circular segment boundary must contain its arc then chord: {region.id}")
        formula = region.metadata.get("area_formula")
        expected = {"rule": "circular_segment_difference", "sector_region": sector_id, "triangle_polygon": triangle_id}
        if not isinstance(formula, Mapping) or any(formula.get(key) != value for key, value in expected.items()):
            raise InvalidGeometryInput(f"Circular segment requires its sector-minus-triangle area formula: {region.id}")
        perimeter_formula = region.metadata.get("perimeter_formula")
        expected_perimeter = {
            "rule": "circular_segment_arc_chord_perimeter",
            "sector_region": sector_id,
            "chord": chord_id,
        }
        if not isinstance(perimeter_formula, Mapping) or any(
            perimeter_formula.get(key) != value for key, value in expected_perimeter.items()
        ):
            raise InvalidGeometryInput(f"Circular segment requires its arc-plus-chord perimeter formula: {region.id}")

    def _validate_lens_region(self, region: Region) -> None:
        if region.kind != "lens_region":
            return
        circle_ids = tuple(region.metadata.get("circles") or ())
        arc_ids = tuple(region.metadata.get("arcs") or ())
        intersection_ids = tuple(region.metadata.get("intersections") or ())
        if len(circle_ids) != 2 or len(set(circle_ids)) != 2 or any(circle_id not in self.circles for circle_id in circle_ids):
            raise UnknownObjectError(f"Lens region requires two distinct circles: {region.id}")
        if len(arc_ids) != 2 or len(set(arc_ids)) != 2 or any(arc_id not in self.arcs for arc_id in arc_ids):
            raise UnknownObjectError(f"Lens region requires two distinct boundary arcs: {region.id}")
        if len(intersection_ids) != 2 or len(set(intersection_ids)) != 2:
            raise InvalidGeometryInput(f"Lens region requires two distinct intersection points: {region.id}")
        self.require_points(intersection_ids)
        first_circle, second_circle = (self.circles[circle_id] for circle_id in circle_ids)
        if first_circle.center == second_circle.center:
            raise DegenerateGeometryError(f"Lens region circles must have distinct centers: {region.id}")
        endpoints = set(intersection_ids)
        from geosym_world.kernel.geometry2d.analytic import distance
        from geosym_world.kernel.geometry2d.intersections import directed_arc_angle_2d

        center_distance = distance(self.require_point(first_circle.center).coords, self.require_point(second_circle.center).coords)
        if (
            sp.simplify(first_circle.radius + second_circle.radius - center_distance).is_positive is not True
            or sp.simplify(center_distance - sp.Abs(first_circle.radius - second_circle.radius)).is_positive is not True
        ):
            raise DegenerateGeometryError(f"Lens region circles must intersect at two points: {region.id}")
        for arc_id, circle_id in zip(arc_ids, circle_ids):
            arc = self.arcs[arc_id]
            if arc.circle != circle_id or {arc.start, arc.end} != endpoints:
                raise InvalidGeometryInput(f"Lens boundary arcs must match their circles and intersection points: {region.id}")
            circle = self.circles[circle_id]
            sweep = directed_arc_angle_2d(
                self.require_point(circle.center).coords,
                self.require_point(arc.start).coords,
                self.require_point(arc.end).coords,
                orientation=arc.orientation,
            )
            if sweep.is_positive is not True or sp.simplify(sp.pi - sweep).is_nonnegative is not True:
                raise InvalidGeometryInput(f"Lens boundary must use the minor arc from each circle: {region.id}")
        if region.boundary != arc_ids:
            raise InvalidGeometryInput(f"Lens boundary order must match its arc references: {region.id}")
        area_formula = region.metadata.get("area_formula")
        perimeter_formula = region.metadata.get("perimeter_formula")
        if area_formula is None and perimeter_formula is None:
            raise InvalidGeometryInput(f"Lens region requires an area or perimeter formula: {region.id}")
        if area_formula is not None:
            if not isinstance(area_formula, Mapping):
                raise InvalidGeometryInput(f"Lens area formula must be a mapping: {region.id}")
            self._validate_lens_measure_formula(region, area_formula, measure="area")
        if perimeter_formula is not None:
            if not isinstance(perimeter_formula, Mapping):
                raise InvalidGeometryInput(f"Lens perimeter formula must be a mapping: {region.id}")
            self._validate_lens_measure_formula(region, perimeter_formula, measure="perimeter")

    def _validate_lens_measure_formula(self, region: Region, formula: Mapping[str, Any], *, measure: str) -> None:
        circle_ids = tuple(region.metadata["circles"])
        first_circle, second_circle = (self.circles[circle_id] for circle_id in circle_ids)
        rule = formula.get("rule")
        prefix = f"lens_{measure}"
        if rule == f"{prefix}_equal_radius_area" or rule == "lens_equal_radius_area":
            if measure != "area":
                raise InvalidGeometryInput(f"lens_equal_radius_area is an area rule: {region.id}")
            if sp.simplify(first_circle.radius - second_circle.radius) != 0:
                raise InvalidGeometryInput(f"lens_equal_radius_area requires equal circle radii: {region.id}")
            radius_segment_ids = (formula.get("radius_segment"),)
            expected_radius_segments = (region.metadata.get("radius_segment"),)
            circles_for_segments = (first_circle,)
        elif rule == "lens_two_radius_area" or rule == "lens_two_radius_perimeter":
            if rule == "lens_two_radius_area" and measure != "area":
                raise InvalidGeometryInput(f"lens_two_radius_area is an area rule: {region.id}")
            if rule == "lens_two_radius_perimeter" and measure != "perimeter":
                raise InvalidGeometryInput(f"lens_two_radius_perimeter is a perimeter rule: {region.id}")
            radius_segment_ids = tuple(formula.get("radius_segments") or ())
            expected_radius_segments = tuple(region.metadata.get("radius_segments") or ())
            circles_for_segments = (first_circle, second_circle)
            if len(radius_segment_ids) != 2 or len(set(radius_segment_ids)) != 2:
                raise InvalidGeometryInput(f"{rule} requires two distinct radius segments: {region.id}")
        elif rule == "lens_equal_radius_perimeter":
            if measure != "perimeter":
                raise InvalidGeometryInput(f"lens_equal_radius_perimeter is a perimeter rule: {region.id}")
            if sp.simplify(first_circle.radius - second_circle.radius) != 0:
                raise InvalidGeometryInput(f"lens_equal_radius_perimeter requires equal circle radii: {region.id}")
            radius_segment_ids = (formula.get("radius_segment"),)
            expected_radius_segments = (region.metadata.get("radius_segment"),)
            circles_for_segments = (first_circle,)
        else:
            raise InvalidGeometryInput(f"Unsupported lens {measure} formula: {region.id}")
        center_segment_id = formula.get("center_segment")
        if radius_segment_ids != expected_radius_segments or center_segment_id != region.metadata.get("center_segment"):
            raise InvalidGeometryInput(f"Lens {measure} formula references must match region metadata: {region.id}")
        if any(segment_id not in self.segments for segment_id in (*radius_segment_ids, center_segment_id)):
            raise UnknownObjectError(f"Lens {measure} formula requires radius and center-distance segments: {region.id}")
        for radius_segment_id, circle in zip(radius_segment_ids, circles_for_segments):
            radius_segment = self.segments[radius_segment_id]
            if circle.center not in {radius_segment.start, radius_segment.end}:
                raise InvalidGeometryInput(f"Lens radius segment must meet its circle center: {region.id}")
            radius_endpoint = radius_segment.end if radius_segment.start == circle.center else radius_segment.start
            if sp.simplify(distance(self.require_point(circle.center).coords, self.require_point(radius_endpoint).coords) - circle.radius) != 0:
                raise InvalidGeometryInput(f"Lens radius segment must equal its circle radius: {region.id}")
        center_segment = self.segments[center_segment_id]
        if {center_segment.start, center_segment.end} != {first_circle.center, second_circle.center}:
            raise InvalidGeometryInput(f"Lens center segment must join the circle centers: {region.id}")

    def _validate_region_area_formula(self, region: Region) -> None:
        if len(set(region.holes)) != len(region.holes):
            raise InvalidGeometryInput(f"Region holes must be distinct: {region.id}")
        missing_holes = [hole_id for hole_id in region.holes if hole_id not in self.regions]
        if missing_holes:
            raise UnknownObjectError(f"Unknown region hole: {missing_holes[0]}")
        formula = region.metadata.get("area_formula")
        if formula is None:
            return
        if not isinstance(formula, dict):
            raise InvalidGeometryInput(f"Region area_formula must be a mapping: {region.id}")
        rule = formula.get("rule")
        if rule in {"lens_equal_radius_area", "lens_two_radius_area"}:
            if region.kind != "lens_region":
                raise InvalidGeometryInput(f"{rule} requires a lens region: {region.id}")
            return
        if rule == "circular_segment_difference":
            if region.kind != "circular_segment_region":
                raise InvalidGeometryInput(f"circular_segment_difference requires a circular segment region: {region.id}")
            return
        if rule == "sector_radius_angle_area":
            if region.kind != "sector_region":
                raise InvalidGeometryInput(f"sector_radius_angle_area requires a sector region: {region.id}")
            return
        if rule == "annulus_radius_area":
            self._validate_annulus_radius_formula(region, formula, rule)
            return
        if rule != "region_hole_difference":
            raise InvalidGeometryInput(f"Unsupported region area formula: {region.id}")
        if region.kind != "polygon_region" or not region.holes:
            raise InvalidGeometryInput(f"region_hole_difference requires a polygon region with holes: {region.id}")
        outer_id = formula.get("outer_region")
        if not isinstance(outer_id, str) or outer_id not in self.regions:
            raise UnknownObjectError(f"Unknown outer region for region_hole_difference: {outer_id}")
        formula_holes = formula.get("holes")
        if not isinstance(formula_holes, (list, tuple)) or tuple(str(item) for item in formula_holes) != region.holes:
            raise InvalidGeometryInput(f"region_hole_difference holes must match Region.holes order: {region.id}")
        if outer_id == region.id or outer_id in region.holes:
            raise InvalidGeometryInput(f"region_hole_difference references must be distinct: {region.id}")

    def _validate_region_perimeter_formula(self, region: Region) -> None:
        formula = region.metadata.get("perimeter_formula")
        if formula is None:
            return
        if not isinstance(formula, dict):
            raise InvalidGeometryInput(f"Region perimeter_formula must be a mapping: {region.id}")
        rule = formula.get("rule")
        if rule == "circular_segment_arc_chord_perimeter":
            if region.kind != "circular_segment_region":
                raise InvalidGeometryInput(f"circular_segment_arc_chord_perimeter requires a circular segment region: {region.id}")
            return
        if rule == "sector_radius_angle_perimeter":
            if region.kind != "sector_region":
                raise InvalidGeometryInput(f"sector_radius_angle_perimeter requires a sector region: {region.id}")
            return
        if rule == "annulus_radius_perimeter":
            self._validate_annulus_radius_formula(region, formula, rule)
            return
        if rule in {"lens_equal_radius_perimeter", "lens_two_radius_perimeter"}:
            if region.kind != "lens_region":
                raise InvalidGeometryInput(f"{rule} requires a lens region: {region.id}")
            circle_ids = tuple(region.metadata.get("circles") or ())
            if len(circle_ids) != 2 or any(circle_id not in self.circles for circle_id in circle_ids):
                raise UnknownObjectError(f"{rule} requires the lens circles: {region.id}")
            first_circle, second_circle = (self.circles[circle_id] for circle_id in circle_ids)
            if rule == "lens_equal_radius_perimeter":
                if sp.simplify(first_circle.radius - second_circle.radius) != 0:
                    raise InvalidGeometryInput(f"lens_equal_radius_perimeter requires equal circle radii: {region.id}")
                radius_segment_ids = (formula.get("radius_segment"),)
                expected_radius_segments = (region.metadata.get("radius_segment"),)
                circles_for_segments = (first_circle,)
            else:
                radius_segment_ids = tuple(formula.get("radius_segments") or ())
                expected_radius_segments = tuple(region.metadata.get("radius_segments") or ())
                circles_for_segments = (first_circle, second_circle)
                if len(radius_segment_ids) != 2 or len(set(radius_segment_ids)) != 2:
                    raise InvalidGeometryInput(f"lens_two_radius_perimeter requires two distinct radius segments: {region.id}")
            if radius_segment_ids != expected_radius_segments or formula.get("center_segment") != region.metadata.get("center_segment"):
                raise InvalidGeometryInput(f"{rule} references must match region metadata: {region.id}")
            center_segment_id = formula.get("center_segment")
            if any(segment_id not in self.segments for segment_id in (*radius_segment_ids, center_segment_id)):
                raise UnknownObjectError(f"{rule} requires existing measurement segments: {region.id}")
            for segment_id, circle in zip(radius_segment_ids, circles_for_segments):
                segment = self.segments[segment_id]
                if circle.center not in {segment.start, segment.end}:
                    raise InvalidGeometryInput(f"{rule} radius segment must meet its circle center: {region.id}")
                endpoint = segment.end if segment.start == circle.center else segment.start
                if sp.simplify(distance(self.require_point(circle.center).coords, self.require_point(endpoint).coords) - circle.radius) != 0:
                    raise InvalidGeometryInput(f"{rule} radius segment must equal its circle radius: {region.id}")
            center_segment = self.segments[center_segment_id]
            if {center_segment.start, center_segment.end} != {first_circle.center, second_circle.center}:
                raise InvalidGeometryInput(f"{rule} center segment must join the circle centers: {region.id}")
            return
        if rule != "region_hole_boundary_sum":
            raise InvalidGeometryInput(f"Unsupported region perimeter formula: {region.id}")
        if region.kind != "polygon_region" or not region.holes:
            raise InvalidGeometryInput(f"region_hole_boundary_sum requires a polygon region with holes: {region.id}")
        outer_id = formula.get("outer_region")
        if not isinstance(outer_id, str) or outer_id not in self.regions:
            raise UnknownObjectError(f"Unknown outer region for region_hole_boundary_sum: {outer_id}")
        formula_holes = formula.get("holes")
        if not isinstance(formula_holes, (list, tuple)) or tuple(str(item) for item in formula_holes) != region.holes:
            raise InvalidGeometryInput(f"region_hole_boundary_sum holes must match Region.holes order: {region.id}")
        if outer_id == region.id or outer_id in region.holes:
            raise InvalidGeometryInput(f"region_hole_boundary_sum references must be distinct: {region.id}")

    def _validate_annulus_radius_formula(self, region: Region, formula: Mapping[str, Any], rule: str) -> None:
        if region.kind != "annulus_region":
            raise InvalidGeometryInput(f"{rule} requires an annulus region: {region.id}")
        references = (formula.get("outer_radius_segment"), formula.get("inner_radius_segment"))
        if any(not isinstance(reference, str) or reference not in self.segments for reference in references):
            raise UnknownObjectError(f"{rule} requires existing outer and inner radius segments: {region.id}")
        if references[0] == references[1]:
            raise InvalidGeometryInput(f"{rule} requires distinct radius segments: {region.id}")

    def _validate_composite_region_reference(self, region: CompositeRegion) -> None:
        if region.operation not in {"union", "intersection", "difference", "sum", "partition"}:
            raise InvalidGeometryInput(f"Unsupported composite region operation: {region.operation}")
        if not region.components or len(set(region.components)) != len(region.components):
            raise InvalidGeometryInput(f"Composite region needs distinct components: {region.id}")
        for component in region.components:
            if component not in self.regions and component not in self.composite_regions:
                raise UnknownObjectError(f"Unknown region component: {component}")
        self._validate_partition_region(region)
        self._validate_composite_area_formula(region)

    def _validate_partition_region(self, region: CompositeRegion) -> None:
        if region.operation != "partition":
            return
        if len(region.components) < 2:
            raise InvalidGeometryInput(f"Partition region needs at least two components: {region.id}")
        parent_id = region.metadata.get("parent_region")
        if not isinstance(parent_id, str) or not parent_id:
            raise InvalidGeometryInput(f"Partition region requires metadata.parent_region: {region.id}")
        if parent_id not in self.regions:
            raise UnknownObjectError(f"Unknown partition parent region: {parent_id}")
        if parent_id in region.components:
            raise InvalidGeometryInput(f"Partition parent cannot also be a component: {region.id}")

    def _validate_composite_area_formula(self, region: CompositeRegion) -> None:
        formula = region.metadata.get("area_formula")
        if formula is None:
            return
        if not isinstance(formula, dict):
            raise InvalidGeometryInput(f"Composite region area_formula must be a mapping: {region.id}")
        if formula.get("rule") != "region_difference":
            raise InvalidGeometryInput(f"Unsupported composite region area formula: {region.id}")
        if region.operation != "difference" or len(region.components) != 2:
            raise InvalidGeometryInput(f"region_difference requires two difference components: {region.id}")
        references = (formula.get("minuend"), formula.get("subtrahend"))
        if references != region.components:
            raise InvalidGeometryInput(f"region_difference references must match ordered components: {region.id}")
        if formula.get("mode", "contained_hole") not in {"contained_hole", "boundary_notch"}:
            raise InvalidGeometryInput(f"Unsupported region_difference mode: {region.id}")

    def _validate_polyhedron_reference(self, polyhedron: Polyhedron) -> None:
        self.require_points(polyhedron.vertices)
        if len(polyhedron.vertices) < 4 or len(set(polyhedron.vertices)) != len(polyhedron.vertices):
            raise DegenerateGeometryError(f"Polyhedron needs four distinct vertices: {polyhedron.id}")
        if any(self.points[point_id].dim != 3 for point_id in polyhedron.vertices):
            raise InvalidGeometryInput(f"Polyhedron vertices must be 3D: {polyhedron.id}")
        vertex_ids = set(polyhedron.vertices)
        for face in polyhedron.faces:
            if len(face) < 3 or len(set(face)) != len(face) or not set(face).issubset(vertex_ids):
                raise InvalidGeometryInput(f"Polyhedron face references are invalid: {polyhedron.id}")

    def _validate_relation_references(self, relation: Mapping[str, Any]) -> None:
        relation_type = str(relation["type"])
        references: list[tuple[str, str]] = []

        def add(kind: str, value: Any, *, flatten: bool = False) -> None:
            if isinstance(value, (list, tuple)):
                for item in value:
                    if flatten and isinstance(item, (list, tuple)):
                        add(kind, item, flatten=True)
                    else:
                        references.append((kind, str(item)))
            elif value is not None:
                references.append((kind, str(value)))

        if relation_type in {"parallel", "perpendicular", "equal_length"}:
            add("line_like", relation.get("objects"))
            add("line_like", relation.get("segments"))
        elif relation_type == "intersection":
            for object_id in relation.get("objects", ()):
                object_id = str(object_id)
                if object_id in self.circles:
                    references.append(("circle", object_id))
                elif object_id in self.arcs:
                    references.append(("arc", object_id))
                elif object_id in self.lines or object_id in self.segments or object_id in self.rays:
                    references.append(("line_like", object_id))
                else:
                    raise UnknownObjectError(f"Unknown intersection relation reference: {object_id}")
        elif relation_type in {"midpoint", "collinear", "equal_angle", "angle_bisector", "point_on_line", "point_on_ray", "point_on_circle", "point_on_arc", "foot_of_perpendicular", "point_on_segment_ratio"}:
            add("point", relation.get("point"))
            add("point", relation.get("points"))
            add("point", relation.get("of"))
            add("point", relation.get("from"))
            if relation_type == "point_on_segment_ratio":
                add("point", relation.get("segment"))
            add("point", relation.get("angle"), flatten=True)
            add("point", relation.get("angles"), flatten=True)
        if relation_type in {"point_on_line", "point_on_ray", "foot_of_perpendicular", "angle_bisector"}:
            add("line_like", relation.get("line"))
            add("line_like", relation.get("segment"))
        if relation_type in {"point_on_circle", "tangent", "chord", "arc", "sector_region"}:
            add("circle", relation.get("circle"))
        if relation_type == "point_on_arc":
            add("arc", relation.get("arc"))
        if relation_type == "tangent":
            add("point", relation.get("point"))
            add("point", relation.get("line"), flatten=True)
        if relation_type == "chord":
            add("line_like", relation.get("object"))
            add("point", relation.get("points"))
        if relation_type == "arc":
            add("point", relation.get("start"))
            add("point", relation.get("end"))
        if relation_type in {"similar", "congruent", "triangle"}:
            add("polygon", relation.get("polygons"))
            add("polygon", relation.get("polygon"))
            mapping = relation.get("mapping")
            if isinstance(mapping, Mapping):
                add("point", list(mapping))
                add("point", list(mapping.values()))
        if relation_type in {"point_projection_plane", "line_projection_plane", "polyhedron_plane_section", "dihedral_angle"}:
            add("plane", relation.get("plane"))
            add("plane", relation.get("planes"))
            add("polyhedron", relation.get("polyhedron"))
        if relation_type == "line_projection_plane":
            add("line_like", relation.get("line"))
        if relation_type == "polyhedron_plane_section":
            add("point", relation.get("boundary"))
            add("polygon", relation.get("polygon"))
        for kind, object_id in references:
            if kind == "line_like":
                if object_id not in self.lines and object_id not in self.segments and object_id not in self.rays:
                    raise UnknownObjectError(f"Unknown line-like relation reference: {object_id}")
            elif kind == "polygon":
                if object_id not in self.polygons:
                    raise UnknownObjectError(f"Unknown polygon relation reference: {object_id}")
            elif kind == "plane":
                if object_id not in self.planes:
                    raise UnknownObjectError(f"Unknown plane relation reference: {object_id}")
            elif kind == "polyhedron":
                if object_id not in self.polyhedra:
                    raise UnknownObjectError(f"Unknown polyhedron relation reference: {object_id}")
            elif kind == "circle":
                if object_id not in self.circles:
                    raise UnknownObjectError(f"Unknown circle relation reference: {object_id}")
            elif kind == "arc":
                if object_id not in self.arcs:
                    raise UnknownObjectError(f"Unknown arc relation reference: {object_id}")
            elif kind == "point":
                if object_id not in self.points:
                    raise UnknownObjectError(f"Unknown point relation reference: {object_id}")

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
            from geosym_world.kernel.geometry2d.analytic import sub

            return Line(id=f"line_of_{object_id}", point=a.coords, direction=sub(b.coords, a.coords), source_points=(a.id, b.id))
        if object_id in self.rays:
            ray = self.rays[object_id]
            start = self.require_point(ray.start)
            source_points = (ray.start, ray.through) if ray.through is not None else (ray.start,)
            return Line(id=f"line_of_{object_id}", point=start.coords, direction=ray.direction, source_points=source_points)
        raise UnknownObjectError(f"Unknown line-like object: {object_id}")

    def line_like_domain(self, object_id: str) -> str:
        if object_id in self.lines:
            return "line"
        if object_id in self.segments:
            return "segment"
        if object_id in self.rays:
            return "ray"
        raise UnknownObjectError(f"Unknown line-like object: {object_id}")

    def object_ids(self) -> set[str]:
        ids = set()
        for bucket in self._buckets():
            ids.update(bucket.keys())
        return ids

    def checksum(self) -> str:
        from .canonical import state_checksum

        self.validate()
        return state_checksum(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "points": {k: v.to_dict() for k, v in self.points.items()},
            "lines": {k: v.to_dict() for k, v in self.lines.items()},
            "segments": {k: v.to_dict() for k, v in self.segments.items()},
            "rays": {k: v.to_dict() for k, v in self.rays.items()},
            "circles": {k: v.to_dict() for k, v in self.circles.items()},
            "arcs": {k: v.to_dict() for k, v in self.arcs.items()},
            "polygons": {k: v.to_dict() for k, v in self.polygons.items()},
            "regions": {k: v.to_dict() for k, v in self.regions.items()},
            "composite_regions": {k: v.to_dict() for k, v in self.composite_regions.items()},
            "planes": {k: v.to_dict() for k, v in self.planes.items()},
            "polyhedra": {k: v.to_dict() for k, v in self.polyhedra.items()},
            "relations": self.relations,
            "construction_trace": self.construction_trace,
            "target": self.target.to_dict() if self.target else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FormalGeometryState":
        version = str(data.get("schema_version", STATE_SCHEMA_VERSION))
        state = cls(schema_version=version)
        for item in data.get("points", {}).values():
            state.add_point(Point(id=str(item["id"]), coords=_coords(item["coords"]), metadata=dict(item.get("metadata", {}))))
        for item in data.get("lines", {}).values():
            state.add_line(
                Line(
                    id=str(item["id"]),
                    point=_coords(item["point"]),
                    direction=_coords(item["direction"]),
                    source_points=tuple(str(value) for value in item.get("source_points", ())),
                    metadata=dict(item.get("metadata", {})),
                )
            )
        for item in data.get("segments", {}).values():
            state.add_segment(Segment(id=str(item["id"]), start=str(item["start"]), end=str(item["end"]), metadata=dict(item.get("metadata", {}))))
        for item in data.get("rays", {}).values():
            state.add_ray(
                Ray(
                    id=str(item["id"]),
                    start=str(item["start"]),
                    direction=_coords(item["direction"]),
                    through=str(item["through"]) if item.get("through") is not None else None,
                    metadata=dict(item.get("metadata", {})),
                )
            )
        for item in data.get("circles", {}).values():
            state.add_circle(
                Circle(
                    id=str(item["id"]),
                    center=str(item["center"]),
                    radius=_expr(item["radius"]),
                    metadata=dict(item.get("metadata", {})),
                )
            )
        for item in data.get("arcs", {}).values():
            state.add_arc(
                Arc(
                    id=str(item["id"]),
                    circle=str(item["circle"]),
                    start=str(item["start"]),
                    end=str(item["end"]),
                    central_angle=_expr(item["central_angle"]) if item.get("central_angle") is not None else None,
                    orientation=str(item.get("orientation", "ccw")),
                    metadata=dict(item.get("metadata", {})),
                )
            )
        pending_polygons = {str(item["id"]): item for item in data.get("polygons", {}).values()}
        pending_regions = {str(item["id"]): item for item in data.get("regions", {}).values()}
        dependency_ids = set(pending_polygons) | set(pending_regions)
        while pending_polygons or pending_regions:
            progressed = False
            for polygon_id, item in tuple(pending_polygons.items()):
                dependencies = _serialized_polygon_dependencies(item) & dependency_ids
                if not dependencies.issubset(state.object_ids()):
                    continue
                state.add_polygon(
                    Polygon(
                        id=polygon_id,
                        vertices=tuple(str(value) for value in item["vertices"]),
                        metadata=dict(item.get("metadata", {})),
                    )
                )
                del pending_polygons[polygon_id]
                progressed = True
            for region_id, item in tuple(pending_regions.items()):
                dependencies = _serialized_region_dependencies(item) & dependency_ids
                if not dependencies.issubset(state.object_ids()):
                    continue
                state.add_region(
                    Region(
                        id=region_id,
                        boundary=tuple(str(value) for value in item["boundary"]),
                        kind=str(item.get("kind", "polygon_region")),
                        metadata=dict(item.get("metadata", {})),
                        holes=tuple(str(value) for value in item.get("holes", ())),
                    )
                )
                del pending_regions[region_id]
                progressed = True
            if not progressed:
                unresolved = sorted((*pending_polygons, *pending_regions))
                raise InvalidGeometryInput(f"Polygon/region dependency cycle: {unresolved}")
        for item in data.get("composite_regions", {}).values():
            state.add_composite_region(
                CompositeRegion(
                    id=str(item["id"]),
                    operation=str(item["operation"]),
                    components=tuple(str(value) for value in item["components"]),
                    metadata=dict(item.get("metadata", {})),
                )
            )
        for item in data.get("planes", {}).values():
            state.add_plane(
                Plane(
                    id=str(item["id"]),
                    point=_coords(item["point"]),
                    normal=_coords(item["normal"]),
                    source_points=tuple(str(value) for value in item.get("source_points", ())),
                    metadata=dict(item.get("metadata", {})),
                )
            )
        for item in data.get("polyhedra", {}).values():
            state.add_polyhedron(
                Polyhedron(
                    id=str(item["id"]),
                    vertices=tuple(str(value) for value in item["vertices"]),
                    faces=tuple(tuple(str(value) for value in face) for face in item["faces"]),
                    metadata=dict(item.get("metadata", {})),
                )
            )
        for relation in data.get("relations", ()):
            state.add_relation(relation)
        state.construction_trace = [dict(item) for item in data.get("construction_trace", ())]
        if data.get("target") is not None:
            target = data["target"]
            state.set_target(Target(
                type=str(target["type"]),
                object=str(target["object"]),
                metadata=dict(target.get("metadata", {})),
                mode=str(target.get("mode", "value")),
                expectation=dict(target["expectation"]) if target.get("expectation") is not None else None,
            ))
        state.validate()
        return state

    def _ensure_free(self, object_id: str) -> None:
        self._validate_id(object_id)
        if object_id in self.object_ids():
            raise DuplicateObjectError(f"Duplicate geometry object id: {object_id}")

    def _validate_id(self, object_id: str) -> None:
        validate_geometry_id(object_id)

    def _require_existing_points(self, point_ids: Iterable[str], *, dimension: int | None = None) -> None:
        for point_id in point_ids:
            point = self.require_point(point_id)
            if dimension is not None and point.dim != dimension:
                raise InvalidGeometryInput(f"Point {point_id} must be {dimension}D")

    def _require_existing_objects(self, object_ids: Iterable[str]) -> None:
        known = self.object_ids()
        for object_id in object_ids:
            if object_id not in known:
                raise UnknownObjectError(f"Unknown geometry object: {object_id}")

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
            self.composite_regions,
            self.planes,
            self.polyhedra,
        ]


def _expr(payload: Any) -> sp.Expr:
    value = payload.get("expr") if isinstance(payload, Mapping) else payload
    if value is None:
        raise ValueError("Missing exact expression payload")
    return sp.sympify(value)


def _serialized_region_dependencies(item: Mapping[str, Any]) -> set[str]:
    dependencies = {str(value) for value in item.get("holes", ())}
    metadata = item.get("metadata") or {}
    if isinstance(metadata, Mapping):
        if metadata.get("sector_region") is not None:
            dependencies.add(str(metadata["sector_region"]))
        if metadata.get("triangle_polygon") is not None:
            dependencies.add(str(metadata["triangle_polygon"]))
        for formula_key in ("area_formula", "perimeter_formula"):
            formula = metadata.get(formula_key)
            if isinstance(formula, Mapping) and formula.get("outer_region") is not None:
                dependencies.add(str(formula["outer_region"]))
            if isinstance(formula, Mapping):
                if formula.get("radius_segment") is not None:
                    dependencies.add(str(formula["radius_segment"]))
                dependencies.update(str(value) for value in formula.get("radius_segments", ()) if value is not None)
                if formula.get("center_segment") is not None:
                    dependencies.add(str(formula["center_segment"]))
    return dependencies


def _serialized_polygon_dependencies(item: Mapping[str, Any]) -> set[str]:
    metadata = item.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        return set()
    formula = metadata.get("area_formula")
    if isinstance(formula, Mapping) and formula.get("sector_region") is not None:
        return {str(formula["sector_region"])}
    return set()


def _coords(payloads: Iterable[Any]) -> tuple[sp.Expr, ...]:
    return tuple(_expr(payload) for payload in payloads)


def _validate_coords(values: Iterable[Any], *, label: str) -> None:
    coords = tuple(values)
    if len(coords) not in {2, 3}:
        raise InvalidGeometryInput(f"{label} must be 2D or 3D")
    for value in coords:
        simplified = sp.simplify(sp.sympify(value))
        if simplified.is_real is False:
            raise InvalidGeometryInput(f"{label} must contain real coordinates")
        if simplified.is_finite is False:
            raise NumericalInstabilityError(f"{label} contains a non-finite coordinate")


def _validate_vector(values: Iterable[Any], *, dimension: int, label: str) -> None:
    coords = tuple(values)
    if len(coords) != dimension:
        raise InvalidGeometryInput(f"{label} must have dimension {dimension}")
    _validate_coords(coords, label=label)


def _validate_vector_pair(point: Iterable[Any], direction: Iterable[Any], *, label: str) -> None:
    point_coords = tuple(point)
    direction_coords = tuple(direction)
    if len(point_coords) != len(direction_coords):
        raise InvalidGeometryInput(f"{label} point and direction dimensions differ")
    _validate_coords(point_coords, label=f"{label} point")
    _validate_coords(direction_coords, label=f"{label} direction")
    _validate_nonzero(direction_coords, label=f"{label} direction")


def _validate_nonzero(values: Iterable[Any], *, label: str) -> None:
    norm_squared = sp.simplify(sum(sp.sympify(value) ** 2 for value in values))
    if norm_squared == 0:
        raise DegenerateGeometryError(f"Zero {label}")


def _validate_positive_expr(value: Any, *, label: str) -> None:
    simplified = sp.simplify(sp.sympify(value))
    if simplified.is_real is False:
        raise InvalidGeometryInput(f"{label} must be real")
    if simplified.is_finite is False:
        raise NumericalInstabilityError(f"{label} is non-finite")
    if simplified.is_positive is not True:
        raise DegenerateGeometryError(f"{label} must be positive")
