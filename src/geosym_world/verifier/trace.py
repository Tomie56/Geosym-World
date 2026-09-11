"""Trace and state verification for the geometry world prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Set

import sympy as sp

from geosym_world.construction.builder import ConstructionBuilder
from geosym_world.construction.operations import ConstructionOp, normalize_trace, operation_references
from geosym_world.errors import GeometryWorldError
from geosym_world.kernel.geometry2d.analytic import add, angle_between, distance, dot, is_parallel, is_perpendicular, is_rectangle_2d, is_zero, line_tangent_to_circle_at_point_2d, midpoint, point_on_line, point_on_segment, polygon_area_2d, rotate2, sub, unit
from geosym_world.kernel.geometry2d.intersections import (
    arc_arc_intersection_result_2d,
    circle_circle_intersection_result_2d,
    circle_arc_intersection_result_2d,
    directed_arc_angle_2d,
    linear_arc_intersection_result_2d,
    linear_linear_intersection_result_2d,
    line_circle_intersection_result_2d,
    point_on_arc_domain_2d,
)
from geosym_world.kernel.geometry2d.regions import resolve_region_geometry_2d
from geosym_world.schema.state import FormalGeometryState


@dataclass(frozen=True)
class VerificationIssue:
    severity: str
    code: str
    message: str
    op_index: int | None = None
    op: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "op_index": self.op_index,
            "op": self.op,
            "metadata": self.metadata,
        }


@dataclass
class VerificationReport:
    static_valid: bool = True
    executable: bool = True
    state_valid: bool = True
    issues: List[VerificationIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.static_valid and self.executable and self.state_valid and not any(issue.severity == "error" for issue in self.issues)

    def add(self, severity: str, code: str, message: str, *, op_index: int | None = None,
            op: str | None = None, **metadata: Any) -> None:
        self.issues.append(
            VerificationIssue(
                severity=severity,
                code=code,
                message=message,
                op_index=op_index,
                op=op,
                metadata=metadata,
            )
        )
        if severity == "error":
            if code.startswith("trace."):
                self.static_valid = False
            elif code.startswith("execute."):
                self.executable = False
            else:
                self.state_valid = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "static_valid": self.static_valid,
            "executable": self.executable,
            "state_valid": self.state_valid,
            "valid": self.valid,
            "issues": [issue.to_dict() for issue in self.issues],
            "metrics": self.metrics,
        }


class TraceVerifier:
    """Checks construction traces before benchmark export."""

    def verify_operation(
        self,
        operation: ConstructionOp | Dict[str, Any],
        *,
        initial_state: FormalGeometryState | None = None,
        execute: bool = True,
    ) -> VerificationReport:
        """Verify one operation against the supplied persistent state."""

        return self.verify_trace([operation], execute=execute, initial_state=initial_state)

    def verify_trace(
        self,
        trace: Iterable[ConstructionOp | Dict[str, Any]],
        *,
        execute: bool = True,
        initial_state: FormalGeometryState | None = None,
    ) -> VerificationReport:
        ops = normalize_trace(trace)
        report = VerificationReport()
        self._static_trace_checks(ops, report, initial_state=initial_state)
        state = None
        if execute and report.static_valid:
            try:
                seed = FormalGeometryState.from_dict(initial_state.to_dict()) if initial_state is not None else None
                state = ConstructionBuilder(seed).execute(ops)
            except Exception as exc:
                code = "execute.geometry_world_error" if isinstance(exc, GeometryWorldError) else "execute.unexpected_error"
                report.add("error", code, str(exc), exception_type=type(exc).__name__)
        if state is not None:
            self.verify_state(state, report)
        report.metrics.update(
            {
                "num_ops": len(ops),
                "num_errors": sum(1 for issue in report.issues if issue.severity == "error"),
                "num_warnings": sum(1 for issue in report.issues if issue.severity == "warning"),
            }
        )
        return report

    def verify_state(self, state: FormalGeometryState, report: VerificationReport | None = None) -> VerificationReport:
        report = report or VerificationReport()
        self._state_checks(state, report)
        report.metrics.update(
            {
                "num_points": len(state.points),
                "num_lines": len(state.lines),
                "num_segments": len(state.segments),
                "num_circles": len(state.circles),
                "num_arcs": len(state.arcs),
                "num_regions": len(state.regions),
                "num_planes": len(state.planes),
                "num_polyhedra": len(state.polyhedra),
            }
        )
        return report

    def _static_trace_checks(
        self,
        ops: List[ConstructionOp],
        report: VerificationReport,
        *,
        initial_state: FormalGeometryState | None = None,
    ) -> None:
        created: Set[str] = set(initial_state.object_ids()) if initial_state is not None else set()
        point_ids: Set[str] = set(initial_state.points) if initial_state is not None else set()
        line_ids: Set[str] = set(initial_state.lines) if initial_state is not None else set()
        segment_ids: Set[str] = set(initial_state.segments) if initial_state is not None else set()
        circle_ids: Set[str] = set(initial_state.circles) if initial_state is not None else set()
        arc_ids: Set[str] = set(initial_state.arcs) if initial_state is not None else set()
        polygon_ids: Set[str] = set(initial_state.polygons) if initial_state is not None else set()
        region_ids: Set[str] = set(initial_state.regions) if initial_state is not None else set()
        composite_region_ids: Set[str] = set(initial_state.composite_regions) if initial_state is not None else set()
        ray_ids: Set[str] = set(initial_state.rays) if initial_state is not None else set()
        plane_ids: Set[str] = set(initial_state.planes) if initial_state is not None else set()
        polyhedron_ids: Set[str] = set(initial_state.polyhedra) if initial_state is not None else set()

        for idx, op in enumerate(ops):
            if op.id and op.id in created and op.op not in {"set_target", "add_relation"}:
                report.add("error", "trace.duplicate_id", f"Duplicate construction id: {op.id}", op_index=idx, op=op.op, object_id=op.id)

            refs = operation_references(op)
            for ref_kind, ref_id in refs:
                known = {
                    "point": point_ids,
                    "line": line_ids | segment_ids | ray_ids,
                    "segment": segment_ids,
                    "circle": circle_ids,
                    "arc": arc_ids,
                    "polygon": polygon_ids,
                    "ray": ray_ids,
                    "plane": plane_ids,
                    "polyhedron": polyhedron_ids,
                    "region": region_ids | composite_region_ids,
                }.get(ref_kind, created)
                if ref_id not in known:
                    report.add("error", "trace.unknown_reference", f"Unknown {ref_kind} reference: {ref_id}", op_index=idx, op=op.op, ref_kind=ref_kind, ref_id=ref_id)

            if op.op == "add_point3d":
                coord = op.params.get("coord", op.params.get("coords"))
                if coord is not None and len(coord) != 3:
                    report.add("error", "trace.dimension_mismatch", "add_point3d requires exactly three coordinates", op_index=idx, op=op.op, object_id=op.id)

            if op.op in {
                "intersection_line_circle",
                "intersection_circle_circle",
                "intersection_line_arc",
                "intersection_circle_arc",
                "intersection_arc_arc",
                "tangent_points_from_point",
            }:
                for point_id in _output_ids(op):
                    if point_id in created:
                        report.add("error", "trace.duplicate_id", f"Duplicate construction id: {point_id}", op_index=idx, op=op.op, object_id=point_id)
                    created.add(point_id)
                    point_ids.add(point_id)
                continue

            if op.id and op.op not in {"set_target", "add_relation"}:
                created.add(op.id)
                if op.op in {"add_point", "add_point3d", "midpoint", "intersection_line_line", "foot_of_perpendicular", "point_on_ray_distance", "point_on_segment_ratio", "project_point_to_plane"}:
                    point_ids.add(op.id)
                elif op.op in {"add_line", "parallel_through", "perpendicular_through", "line_through_angle", "angle_bisector", "project_line_to_plane"}:
                    line_ids.add(op.id)
                elif op.op in {"add_segment", "add_chord"}:
                    segment_ids.add(op.id)
                elif op.op == "add_circle":
                    circle_ids.add(op.id)
                elif op.op == "add_polygon":
                    polygon_ids.add(op.id)
                elif op.op in {"add_region", "mark_polygon_region", "mark_sector_region"}:
                    region_ids.add(op.id)
                elif op.op == "add_composite_region":
                    composite_region_ids.add(op.id)
                elif op.op in {"add_ray", "rotate_ray"}:
                    ray_ids.add(op.id)
                elif op.op == "add_arc":
                    arc_ids.add(op.id)
                elif op.op == "add_plane":
                    plane_ids.add(op.id)
                elif op.op == "add_polyhedron":
                    polyhedron_ids.add(op.id)

        if not any(op.op == "set_target" for op in ops):
            report.add("warning", "trace.missing_target", "Trace has no target; it can build a state but not a benchmark item")

    def _state_checks(self, state: FormalGeometryState, report: VerificationReport) -> None:
        for seg_id, seg in state.segments.items():
            a = state.require_point(seg.start)
            b = state.require_point(seg.end)
            if a.coords == b.coords:
                report.add("error", "state.degenerate_segment", f"Segment {seg_id} has identical endpoints", object_id=seg_id)

        for polygon_id, polygon in state.polygons.items():
            if len(set(polygon.vertices)) != len(polygon.vertices):
                report.add("error", "state.repeated_polygon_vertex", f"Polygon {polygon_id} repeats vertices", object_id=polygon_id)
            formula = polygon.metadata.get("perimeter_formula")
            if isinstance(formula, dict) and formula.get("rule") == "rectangle_side_perimeter":
                points = tuple(state.require_point(point_id).coords for point_id in polygon.vertices)
                width_id = str(formula.get("width", ""))
                height_id = str(formula.get("height", ""))
                if not is_rectangle_2d(points) or not _segments_are_adjacent_polygon_edges(state, polygon.vertices, width_id, height_id):
                    report.add(
                        "error",
                        "state.invalid_rectangle_perimeter_formula",
                        f"Polygon {polygon_id} does not support its rectangle perimeter formula",
                        object_id=polygon_id,
                        width=width_id,
                        height=height_id,
                    )
            area_formula = polygon.metadata.get("area_formula")
            if isinstance(area_formula, dict) and area_formula.get("rule") == "chord_triangle_radius_angle":
                sector_id = area_formula.get("sector_region")
                sector = state.regions.get(sector_id)
                valid = (
                    sector is not None
                    and sector.kind == "sector_region"
                    and set(polygon.vertices) == {sector.metadata.get("center"), sector.metadata.get("start"), sector.metadata.get("end")}
                    and _sector_region_is_valid(state, sector)
                )
                if not valid:
                    report.add(
                        "error",
                        "state.invalid_chord_triangle_area_formula",
                        f"Polygon {polygon_id} does not match its sector chord triangle",
                        object_id=polygon_id,
                        sector_region=sector_id,
                    )

        for circle_id, circle in state.circles.items():
            if circle.radius == 0:
                report.add("error", "state.degenerate_circle", f"Circle {circle_id} has zero radius", object_id=circle_id)

        for arc_id, arc in state.arcs.items():
            if arc.start == arc.end and not arc.metadata.get("allow_full_circle"):
                report.add("warning", "state.same_arc_endpoint", f"Arc {arc_id} has identical endpoints", object_id=arc_id)

        for poly_id, poly in state.polyhedra.items():
            if len(poly.vertices) < 4:
                report.add("error", "state.degenerate_polyhedron", f"Polyhedron {poly_id} needs at least four vertices", object_id=poly_id)
            for face in poly.faces:
                if len(face) < 3:
                    report.add("error", "state.degenerate_polyhedron_face", f"Polyhedron {poly_id} has a face with fewer than three vertices", object_id=poly_id, face=list(face))

        for region_id, region in state.regions.items():
            if region.kind == "sector_region" and not _sector_region_is_valid(state, region):
                report.add(
                    "error",
                    "state.invalid_sector_measure_formula",
                    f"Sector {region_id} does not match its circle, radius segment, and central angle",
                    object_id=region_id,
                    circle=region.metadata.get("circle"),
                    radius_segment=region.metadata.get("radius_segment"),
                    central_angle=region.metadata.get("central_angle"),
                )
            if region.kind == "circular_segment_region" and not _circular_segment_region_is_valid(state, region):
                report.add(
                    "error",
                    "state.invalid_circular_segment_region",
                    f"Circular segment {region_id} does not match its sector, arc, chord, and triangle",
                    object_id=region_id,
                    sector_region=region.metadata.get("sector_region"),
                    triangle_polygon=region.metadata.get("triangle_polygon"),
                )
            if region.kind == "lens_region" and not _lens_region_is_valid(state, region):
                report.add(
                    "error",
                    "state.invalid_lens_region",
                    f"Lens {region_id} does not match its circles, arcs, intersections, and public measurements",
                    object_id=region_id,
                    circles=region.metadata.get("circles"),
                    arcs=region.metadata.get("arcs"),
                    intersections=region.metadata.get("intersections"),
                )
            if region.kind == "lens_region":
                perimeter_formula = region.metadata.get("perimeter_formula")
                if isinstance(perimeter_formula, dict) and not _lens_perimeter_formula_is_valid(state, region, perimeter_formula):
                    report.add(
                        "error",
                        "state.invalid_lens_perimeter_formula",
                        f"Lens {region_id} does not support its public perimeter formula",
                        object_id=region_id,
                        perimeter_formula=perimeter_formula,
                    )
            if region.kind == "annulus_region":
                for formula_key, rule, issue_code in (
                    ("area_formula", "annulus_radius_area", "state.invalid_annulus_area_formula"),
                    ("perimeter_formula", "annulus_radius_perimeter", "state.invalid_annulus_perimeter_formula"),
                ):
                    formula = region.metadata.get(formula_key)
                    if isinstance(formula, dict) and formula.get("rule") == rule and not _annulus_radius_formula_is_valid(state, region, formula):
                        report.add(
                            "error",
                            issue_code,
                            f"Annulus {region_id} does not support its public-radius formula",
                            object_id=region_id,
                            outer_radius_segment=formula.get("outer_radius_segment"),
                            inner_radius_segment=formula.get("inner_radius_segment"),
                        )
            if region.kind != "polygon_region" or not _region_is_2d(state, region):
                continue
            geometry = resolve_region_geometry_2d(state, region_id)
            if geometry.status == "unsupported":
                report.add(
                    "error",
                    "state.invalid_region_boundary",
                    f"Region {region_id} cannot be admitted: {geometry.reason}",
                    object_id=region_id,
                    reason=geometry.reason,
                    source_ids=list(geometry.source_ids),
                )
            elif geometry.status == "empty":
                report.add(
                    "error",
                    "state.empty_region_boundary",
                    f"Region {region_id} resolves to an empty boundary",
                    object_id=region_id,
                    reason=geometry.reason,
                )
            formula = region.metadata.get("area_formula")
            if isinstance(formula, dict) and formula.get("rule") == "region_hole_difference":
                outer_id = formula.get("outer_region")
                outer_geometry = resolve_region_geometry_2d(state, outer_id) if outer_id in state.regions else None
                valid_geometry = (
                    geometry.status == "polygon_with_holes"
                    and len(geometry.holes) == len(region.holes)
                    and outer_geometry is not None
                    and outer_geometry.status == "polygon"
                    and geometry.boundary == outer_geometry.boundary
                )
                if not valid_geometry:
                    report.add(
                        "error",
                        "state.invalid_region_hole_difference",
                        f"Semantic holed region {region_id} does not match its outer region",
                        object_id=region_id,
                        outer_region=outer_id,
                        status=geometry.status,
                        reason=geometry.reason,
                    )
            perimeter_formula = region.metadata.get("perimeter_formula")
            if isinstance(perimeter_formula, dict) and perimeter_formula.get("rule") == "region_hole_boundary_sum":
                outer_id = perimeter_formula.get("outer_region")
                outer_geometry = resolve_region_geometry_2d(state, outer_id) if outer_id in state.regions else None
                valid_geometry = (
                    geometry.status == "polygon_with_holes"
                    and len(geometry.holes) == len(region.holes)
                    and outer_geometry is not None
                    and outer_geometry.status == "polygon"
                    and geometry.boundary == outer_geometry.boundary
                )
                if not valid_geometry:
                    report.add(
                        "error",
                        "state.invalid_region_hole_perimeter",
                        f"Semantic holed perimeter {region_id} does not match its outer region",
                        object_id=region_id,
                        outer_region=outer_id,
                        status=geometry.status,
                        reason=geometry.reason,
                    )

        for region_id, region in state.composite_regions.items():
            if region.operation != "partition":
                formula = region.metadata.get("area_formula")
                if not isinstance(formula, dict) or formula.get("rule") != "region_difference":
                    continue
                geometry = resolve_region_geometry_2d(state, region_id)
                mode = formula.get("mode", "contained_hole")
                if mode == "contained_hole":
                    valid_geometry = geometry.status == "polygon_with_holes" and len(geometry.holes) == 1
                elif mode == "boundary_notch":
                    valid_geometry = geometry.status == "polygon" and len(geometry.boundary) == 6
                else:
                    valid_geometry = False
                if not valid_geometry:
                    report.add(
                        "error",
                        "state.invalid_region_difference",
                        f"Semantic difference {region_id} violates mode {mode}",
                        object_id=region_id,
                        mode=mode,
                        status=geometry.status,
                        reason=geometry.reason,
                        source_ids=list(geometry.source_ids),
                    )
                continue
            geometry = resolve_region_geometry_2d(state, region_id)
            if not geometry.resolved:
                report.add(
                    "error",
                    "state.invalid_region_partition",
                    f"Partition {region_id} cannot be admitted: {geometry.reason}",
                    object_id=region_id,
                    reason=geometry.reason,
                    source_ids=list(geometry.source_ids),
                )

        self._relation_checks(state, report)

    def _relation_checks(self, state: FormalGeometryState, report: VerificationReport) -> None:
        for index, relation in enumerate(state.relations):
            relation_type = str(relation.get("type", ""))
            if relation_type in {"parallel", "perpendicular"}:
                self._linear_relation_check(state, report, relation, index, relation_type)
            elif relation_type == "equal_length":
                self._equal_length_relation_check(state, report, relation, index)
            elif relation_type == "equal_angle":
                self._equal_angle_relation_check(state, report, relation, index)
            elif relation_type in {"similar", "congruent"}:
                self._polygon_relation_check(state, report, relation, index, relation_type)
            elif relation_type == "midpoint":
                self._midpoint_relation_check(state, report, relation, index)
            elif relation_type == "intersection":
                self._intersection_relation_check(state, report, relation, index)
            elif relation_type == "tangent":
                self._tangent_relation_check(state, report, relation, index)
            elif relation_type in {"point_on_line", "point_on_ray", "point_on_circle", "point_on_arc"}:
                self._incidence_relation_check(state, report, relation, index, relation_type)
            elif relation_type == "collinear":
                self._collinear_relation_check(state, report, relation, index)
            elif relation_type == "foot_of_perpendicular":
                self._foot_relation_check(state, report, relation, index)
            elif relation_type == "angle_bisector":
                self._angle_bisector_relation_check(state, report, relation, index)
            elif relation_type == "angle_from_line":
                self._angle_from_line_relation_check(state, report, relation, index)
            elif relation_type == "point_on_segment_ratio":
                self._segment_ratio_relation_check(state, report, relation, index)
            elif relation_type == "point_on_ray_distance":
                self._ray_distance_relation_check(state, report, relation, index)
            elif relation_type == "proportional_lengths":
                self._proportional_lengths_relation_check(state, report, relation, index)

    def _linear_relation_check(
        self,
        state: FormalGeometryState,
        report: VerificationReport,
        relation: Dict[str, Any],
        index: int,
        relation_type: str,
    ) -> None:
        object_ids = tuple(str(item) for item in relation.get("objects", ()))
        if len(object_ids) != 2:
            report.add("error", "state.invalid_relation_payload", f"{relation_type} relation requires two objects", relation_index=index)
            return
        try:
            left = state.get_line_like(object_ids[0])
            right = state.get_line_like(object_ids[1])
        except GeometryWorldError as exc:
            report.add("error", "state.unknown_relation_object", str(exc), relation_index=index, relation_type=relation_type)
            return
        predicate = is_parallel if relation_type == "parallel" else is_perpendicular
        if len(left.direction) != len(right.direction) or not predicate(left.direction, right.direction):
            report.add(
                "error",
                f"state.false_{relation_type}",
                f"Relation {relation_type} is not satisfied by {object_ids[0]} and {object_ids[1]}",
                relation_index=index,
                objects=list(object_ids),
            )

    def _midpoint_relation_check(
        self,
        state: FormalGeometryState,
        report: VerificationReport,
        relation: Dict[str, Any],
        index: int,
    ) -> None:
        endpoints = tuple(str(item) for item in relation.get("of", ()))
        point_id = relation.get("point")
        if point_id is None or len(endpoints) != 2:
            report.add("error", "state.invalid_relation_payload", "midpoint relation requires point and two endpoints", relation_index=index)
            return
        try:
            point = state.require_point(str(point_id))
            left = state.require_point(endpoints[0])
            right = state.require_point(endpoints[1])
        except GeometryWorldError as exc:
            report.add("error", "state.unknown_relation_object", str(exc), relation_index=index, relation_type="midpoint")
            return
        expected = midpoint(left.coords, right.coords)
        if not _coords_equal(point.coords, expected):
            report.add("error", "state.false_midpoint", f"Point {point.id} is not the midpoint of {endpoints[0]} and {endpoints[1]}", relation_index=index)

    def _equal_length_relation_check(
        self,
        state: FormalGeometryState,
        report: VerificationReport,
        relation: Dict[str, Any],
        index: int,
    ) -> None:
        objects = relation.get("segments") or relation.get("objects") or ()
        if len(objects) != 2:
            report.add("error", "state.invalid_relation_payload", "equal_length relation requires two segments", relation_index=index)
            return
        try:
            lengths = tuple(_line_like_length(state, str(object_id)) for object_id in objects)
        except GeometryWorldError as exc:
            report.add("error", "state.unknown_relation_object", str(exc), relation_index=index, relation_type="equal_length")
            return
        if not _coords_equal((lengths[0],), (lengths[1],)):
            report.add("error", "state.false_equal_length", f"Segments {objects[0]} and {objects[1]} have different lengths", relation_index=index, values=[str(value) for value in lengths])

    def _equal_angle_relation_check(
        self,
        state: FormalGeometryState,
        report: VerificationReport,
        relation: Dict[str, Any],
        index: int,
    ) -> None:
        angles = relation.get("angles") or ()
        if len(angles) != 2 or any(len(angle) != 3 for angle in angles):
            report.add("error", "state.invalid_relation_payload", "equal_angle relation requires two angle triples", relation_index=index)
            return
        try:
            values = tuple(_angle_value(state, tuple(str(item) for item in angle)) for angle in angles)
        except GeometryWorldError as exc:
            report.add("error", "state.unknown_relation_object", str(exc), relation_index=index, relation_type="equal_angle")
            return
        if sp.simplify(values[0] - values[1]) != 0:
            report.add("error", "state.false_equal_angle", "Equal-angle relation has different angle values", relation_index=index, values=[str(value) for value in values])

    def _polygon_relation_check(
        self,
        state: FormalGeometryState,
        report: VerificationReport,
        relation: Dict[str, Any],
        index: int,
        relation_type: str,
    ) -> None:
        polygon_ids = tuple(str(item) for item in relation.get("polygons") or relation.get("objects") or ())
        mapping = {str(key): str(value) for key, value in (relation.get("mapping") or {}).items()}
        if len(polygon_ids) != 2 or not mapping:
            report.add("error", "state.invalid_relation_payload", f"{relation_type} relation requires two polygons and a mapping", relation_index=index)
            return
        try:
            left = state.polygons[polygon_ids[0]]
            right = state.polygons[polygon_ids[1]]
        except KeyError as exc:
            report.add("error", "state.unknown_relation_object", f"Unknown polygon: {exc.args[0]}", relation_index=index, relation_type=relation_type)
            return
        if len(left.vertices) != len(right.vertices) or set(mapping) != set(left.vertices) or set(mapping.values()) != set(right.vertices):
            report.add("error", "state.invalid_relation_mapping", f"{relation_type} mapping does not cover both polygon vertex sets", relation_index=index)
            return
        left_points = {point_id: state.require_point(point_id).coords for point_id in left.vertices}
        right_points = {point_id: state.require_point(point_id).coords for point_id in right.vertices}
        if any(len(point) != 2 for point in (*left_points.values(), *right_points.values())):
            report.add("error", "state.dimension_mismatch", f"{relation_type} currently requires 2D polygons", relation_index=index)
            return
        if polygon_area_2d([left_points[point_id] for point_id in left.vertices]) == 0 or polygon_area_2d([right_points[point_id] for point_id in right.vertices]) == 0:
            report.add("error", "state.degenerate_relation_polygon", f"{relation_type} cannot use a degenerate polygon", relation_index=index)
            return
        mapped_right = tuple(mapping[point_id] for point_id in left.vertices)
        left_sides = _polygon_side_squares(left_points, left.vertices)
        right_sides = _polygon_side_squares(right_points, mapped_right)
        if relation_type == "congruent":
            residual = sp.simplify(sum((left_value - right_value) ** 2 for left_value, right_value in zip(left_sides, right_sides)))
        else:
            residual = sp.simplify(sum((left_sides[index] * right_sides[0] - left_sides[0] * right_sides[index]) ** 2 for index in range(1, len(left_sides))))
        if residual != 0:
            report.add("error", f"state.false_{relation_type}", f"{relation_type} relation has nonzero side residual", relation_index=index, residual=str(residual))

    def _intersection_relation_check(
        self,
        state: FormalGeometryState,
        report: VerificationReport,
        relation: Dict[str, Any],
        index: int,
    ) -> None:
        object_ids = tuple(str(item) for item in relation.get("objects", ()))
        point_ids = tuple(str(item) for item in relation.get("points", ()))
        if relation.get("point") is not None:
            point_ids = (str(relation["point"]),)
        if len(object_ids) != 2 or not point_ids:
            report.add("error", "state.invalid_relation_payload", "intersection relation requires two objects and at least one point", relation_index=index)
            return
        try:
            expected = _intersection_points(state, object_ids)
            actual = tuple(state.require_point(point_id).coords for point_id in point_ids)
        except GeometryWorldError as exc:
            report.add("error", "state.unknown_relation_object", str(exc), relation_index=index, relation_type="intersection")
            return
        if not expected or any(not any(_coords_equal(point, candidate) for candidate in expected) for point in actual):
            report.add(
                "error",
                "state.false_intersection",
                f"Intersection points {list(point_ids)} are not valid for {object_ids[0]} and {object_ids[1]}",
                relation_index=index,
                objects=list(object_ids),
            )

    def _tangent_relation_check(
        self,
        state: FormalGeometryState,
        report: VerificationReport,
        relation: Dict[str, Any],
        index: int,
    ) -> None:
        point_id = relation.get("point")
        circle_id = relation.get("circle")
        line_ids = tuple(str(item) for item in relation.get("line", ()))
        if point_id is None or circle_id is None or len(line_ids) != 2 or line_ids[0] == line_ids[1]:
            report.add("error", "state.invalid_relation_payload", "tangent relation requires a point, circle, and two distinct line points", relation_index=index)
            return
        try:
            tangent = state.require_point(str(point_id))
            line_start = state.require_point(line_ids[0])
            line_end = state.require_point(line_ids[1])
            circle = state.circles[str(circle_id)]
            center = state.require_point(circle.center)
        except (GeometryWorldError, KeyError) as exc:
            report.add("error", "state.unknown_relation_object", str(exc), relation_index=index, relation_type="tangent")
            return
        if any(point.dim != 2 for point in (tangent, line_start, line_end, center)):
            report.add("error", "state.dimension_mismatch", "tangent relation currently requires 2D points", relation_index=index)
            return
        try:
            valid = line_tangent_to_circle_at_point_2d(tangent.coords, line_start.coords, line_end.coords, center.coords, circle.radius)
        except GeometryWorldError as exc:
            report.add("error", "state.degenerate_tangent", str(exc), relation_index=index)
            return
        if not valid:
            report.add(
                "error",
                "state.false_tangent",
                f"Line {line_ids[0]}-{line_ids[1]} is not tangent to {circle_id} at {point_id}",
                relation_index=index,
                point=str(point_id),
                circle=str(circle_id),
                line=list(line_ids),
            )

    def _incidence_relation_check(
        self,
        state: FormalGeometryState,
        report: VerificationReport,
        relation: Dict[str, Any],
        index: int,
        relation_type: str,
    ) -> None:
        point_id = relation.get("point")
        if point_id is None:
            report.add("error", "state.invalid_relation_payload", f"{relation_type} requires a point", relation_index=index)
            return
        point = state.require_point(str(point_id))
        valid = False
        if relation_type in {"point_on_line", "point_on_ray"}:
            line_id = relation.get("line")
            if line_id is None:
                report.add("error", "state.invalid_relation_payload", f"{relation_type} requires a line-like object", relation_index=index)
                return
            line = state.get_line_like(str(line_id))
            valid = point_on_line(point.coords, line.point, line.direction)
            if relation_type == "point_on_ray":
                valid = valid and not _is_negative(dot(sub(point.coords, line.point), line.direction))
        elif relation_type == "point_on_circle":
            circle_id = relation.get("circle")
            if circle_id is None:
                report.add("error", "state.invalid_relation_payload", "point_on_circle requires a circle", relation_index=index)
                return
            circle = state.circles[str(circle_id)]
            valid = is_zero(distance(point.coords, state.require_point(circle.center).coords) ** 2 - circle.radius ** 2)
        else:
            arc_id = relation.get("arc")
            if arc_id is None:
                report.add("error", "state.invalid_relation_payload", "point_on_arc requires an arc", relation_index=index)
                return
            arc = state.arcs[str(arc_id)]
            circle = state.circles[arc.circle]
            valid = point_on_arc_domain_2d(
                point.coords,
                state.require_point(circle.center).coords,
                circle.radius,
                state.require_point(arc.start).coords,
                state.require_point(arc.end).coords,
                orientation=arc.orientation,
                central_angle=arc.central_angle,
                allow_full_circle=bool(arc.metadata.get("allow_full_circle", False)),
            )
        if not valid:
            report.add("error", f"state.false_{relation_type}", f"Point {point_id} does not satisfy {relation_type}", relation_index=index)

    def _collinear_relation_check(self, state: FormalGeometryState, report: VerificationReport, relation: Dict[str, Any], index: int) -> None:
        point_ids = tuple(str(item) for item in relation.get("points", ()))
        if len(point_ids) < 3:
            report.add("error", "state.invalid_relation_payload", "collinear relation requires at least three points", relation_index=index)
            return
        points = tuple(state.require_point(point_id).coords for point_id in point_ids)
        direction = sub(points[1], points[0])
        if all(is_zero(value) for value in direction) or any(not point_on_line(point, points[0], direction) for point in points[2:]):
            report.add("error", "state.false_collinear", f"Points {list(point_ids)} are not collinear", relation_index=index)

    def _foot_relation_check(self, state: FormalGeometryState, report: VerificationReport, relation: Dict[str, Any], index: int) -> None:
        point_id, source_id, line_id = relation.get("point"), relation.get("from"), relation.get("line")
        if None in {point_id, source_id, line_id}:
            report.add("error", "state.invalid_relation_payload", "foot_of_perpendicular requires point, from, and line", relation_index=index)
            return
        foot = state.require_point(str(point_id))
        source = state.require_point(str(source_id))
        line = state.get_line_like(str(line_id))
        valid = point_on_line(foot.coords, line.point, line.direction) and is_zero(dot(sub(source.coords, foot.coords), line.direction))
        if not valid:
            report.add("error", "state.false_foot_of_perpendicular", f"Point {point_id} is not the perpendicular foot from {source_id}", relation_index=index)

    def _angle_bisector_relation_check(self, state: FormalGeometryState, report: VerificationReport, relation: Dict[str, Any], index: int) -> None:
        line_id = relation.get("line")
        angle = tuple(str(item) for item in relation.get("angle", ()))
        mode = str(relation.get("mode", "internal"))
        if line_id is None or len(angle) != 3 or mode not in {"internal", "external"}:
            report.add("error", "state.invalid_relation_payload", "angle_bisector requires line, angle triple, and valid mode", relation_index=index)
            return
        try:
            first = state.require_point(angle[0])
            vertex = state.require_point(angle[1])
            second = state.require_point(angle[2])
            line = state.get_line_like(str(line_id))
            first_direction = unit(sub(first.coords, vertex.coords))
            second_direction = unit(sub(second.coords, vertex.coords))
            expected_direction = add(first_direction, second_direction) if mode == "internal" else sub(first_direction, second_direction)
        except GeometryWorldError as exc:
            report.add("error", "state.invalid_relation_payload", str(exc), relation_index=index, relation_type="angle_bisector")
            return
        if not point_on_line(vertex.coords, line.point, line.direction) or not is_parallel(line.direction, expected_direction):
            report.add(
                "error",
                "state.false_angle_bisector",
                f"Line {line_id} is not the declared {mode} angle bisector",
                relation_index=index,
            )

    def _angle_from_line_relation_check(self, state: FormalGeometryState, report: VerificationReport, relation: Dict[str, Any], index: int) -> None:
        line_id = relation.get("line")
        base_line_id = relation.get("base_line")
        angle_value = relation.get("angle")
        if line_id is None or base_line_id is None or angle_value is None:
            report.add("error", "state.invalid_relation_payload", "angle_from_line requires line, base_line, and angle", relation_index=index)
            return
        try:
            line = state.get_line_like(str(line_id))
            base_line = state.get_line_like(str(base_line_id))
            angle = sp.sympify(angle_value)
        except (GeometryWorldError, TypeError, ValueError) as exc:
            report.add("error", "state.invalid_relation_payload", str(exc), relation_index=index, relation_type="angle_from_line")
            return
        valid = (
            angle.is_real is not False
            and len(line.direction) == 2
            and len(base_line.direction) == 2
            and is_parallel(line.direction, rotate2(base_line.direction, angle))
        )
        if not valid:
            report.add(
                "error",
                "state.false_angle_from_line",
                f"Line {line_id} does not have the declared angle from {base_line_id}",
                relation_index=index,
            )

    def _segment_ratio_relation_check(self, state: FormalGeometryState, report: VerificationReport, relation: Dict[str, Any], index: int) -> None:
        point_id = relation.get("point")
        endpoints = tuple(str(item) for item in relation.get("segment", ()))
        if point_id is None or len(endpoints) != 2 or relation.get("ratio") is None:
            report.add("error", "state.invalid_relation_payload", "point_on_segment_ratio requires point, two endpoints, and ratio", relation_index=index)
            return
        point = state.require_point(str(point_id)).coords
        start = state.require_point(endpoints[0]).coords
        end = state.require_point(endpoints[1]).coords
        ratio = sp.sympify(relation["ratio"])
        expected = tuple(sp.simplify(start[dim] + ratio * (end[dim] - start[dim])) for dim in range(len(start)))
        if _is_negative(ratio) or _is_negative(1 - ratio) or not point_on_segment(point, start, end) or not _coords_equal(point, expected):
            report.add("error", "state.false_point_on_segment_ratio", f"Point {point_id} does not match the declared segment ratio", relation_index=index)

    def _ray_distance_relation_check(self, state: FormalGeometryState, report: VerificationReport, relation: Dict[str, Any], index: int) -> None:
        point_id = relation.get("point")
        ray_id = relation.get("ray")
        distance_value = relation.get("distance")
        if point_id is None or ray_id is None or distance_value is None:
            report.add("error", "state.invalid_relation_payload", "point_on_ray_distance requires point, ray, and distance", relation_index=index)
            return
        try:
            point = state.require_point(str(point_id))
            ray = state.rays[str(ray_id)]
            start = state.require_point(ray.start)
            expected_distance = sp.sympify(distance_value)
        except (GeometryWorldError, KeyError, TypeError, ValueError) as exc:
            report.add("error", "state.invalid_relation_payload", str(exc), relation_index=index, relation_type="point_on_ray_distance")
            return
        displacement = sub(point.coords, start.coords)
        valid = (
            expected_distance.is_real is not False
            and not _is_negative(expected_distance)
            and point_on_line(point.coords, start.coords, ray.direction)
            and not _is_negative(dot(displacement, ray.direction))
            and is_zero(distance(point.coords, start.coords) - expected_distance)
        )
        if not valid:
            report.add(
                "error",
                "state.false_point_on_ray_distance",
                f"Point {point_id} is not at the declared distance on ray {ray_id}",
                relation_index=index,
            )

    def _proportional_lengths_relation_check(
        self,
        state: FormalGeometryState,
        report: VerificationReport,
        relation: Dict[str, Any],
        index: int,
    ) -> None:
        ratios = tuple(tuple(str(item) for item in pair) for pair in relation.get("ratios", ()))
        if len(ratios) < 2 or any(len(pair) != 2 for pair in ratios) or relation.get("value") is None:
            report.add(
                "error",
                "state.invalid_relation_payload",
                "proportional_lengths requires at least two segment pairs and one ratio value",
                relation_index=index,
            )
            return
        expected = sp.sympify(relation["value"])
        for numerator_id, denominator_id in ratios:
            if numerator_id not in state.segments or denominator_id not in state.segments:
                report.add(
                    "error",
                    "state.unknown_relation_object",
                    f"Unknown segment in proportional_lengths: {numerator_id}/{denominator_id}",
                    relation_index=index,
                )
                return
            numerator = state.segments[numerator_id]
            denominator = state.segments[denominator_id]
            numerator_length = distance(state.require_point(numerator.start).coords, state.require_point(numerator.end).coords)
            denominator_length = distance(state.require_point(denominator.start).coords, state.require_point(denominator.end).coords)
            if sp.simplify(denominator_length) == 0 or sp.simplify(numerator_length / denominator_length - expected) != 0:
                report.add(
                    "error",
                    "state.false_proportional_lengths",
                    f"Segment ratio {numerator_id}/{denominator_id} does not equal {expected}",
                    relation_index=index,
                )
                return



def _region_is_2d(state: FormalGeometryState, region) -> bool:
    references = (*region.boundary, *region.holes)
    point_ids = [item for item in references if item in state.points]
    if point_ids:
        return all(state.points[item].dim == 2 for item in point_ids)
    if len(region.boundary) == 1 and region.boundary[0] in state.polygons:
        return all(state.points[item].dim == 2 for item in state.polygons[region.boundary[0]].vertices)
    if all(item in state.segments for item in region.boundary):
        endpoints = {endpoint for segment_id in region.boundary for endpoint in (state.segments[segment_id].start, state.segments[segment_id].end)}
        return all(state.points[item].dim == 2 for item in endpoints)
    return False

def _intersection_points(state: FormalGeometryState, object_ids: tuple[str, str]) -> tuple[tuple[sp.Expr, ...], ...]:
    left_id, right_id = object_ids
    left_linear = _is_line_like(state, left_id)
    right_linear = _is_line_like(state, right_id)
    if left_linear and right_linear:
        left = state.get_line_like(left_id)
        right = state.get_line_like(right_id)
        result = linear_linear_intersection_result_2d(
            left.point,
            left.direction,
            right.point,
            right.direction,
            first_domain=state.line_like_domain(left_id),
            second_domain=state.line_like_domain(right_id),
            objects=object_ids,
        )
        return result.points if result.is_constructible else ()
    if left_linear and right_id in state.circles:
        return _linear_circle_points(state, left_id, right_id, object_ids)
    if right_linear and left_id in state.circles:
        return _linear_circle_points(state, right_id, left_id, object_ids)
    if left_id in state.circles and right_id in state.circles:
        left = state.circles[left_id]
        right = state.circles[right_id]
        left_center = state.require_point(left.center)
        right_center = state.require_point(right.center)
        result = circle_circle_intersection_result_2d(
            left_center.coords,
            left.radius,
            right_center.coords,
            right.radius,
            objects=object_ids,
        )
        return result.points if result.is_constructible else ()
    if left_linear and right_id in state.arcs:
        return _linear_arc_points(state, left_id, right_id, object_ids)
    if right_linear and left_id in state.arcs:
        return _linear_arc_points(state, right_id, left_id, object_ids)
    if left_id in state.circles and right_id in state.arcs:
        return _circle_arc_points(state, left_id, right_id, object_ids)
    if right_id in state.circles and left_id in state.arcs:
        return _circle_arc_points(state, right_id, left_id, object_ids)
    if left_id in state.arcs and right_id in state.arcs:
        first = _arc_domain(state, left_id)
        second = _arc_domain(state, right_id)
        result = arc_arc_intersection_result_2d(
            first[0], first[1], first[2], first[3],
            second[0], second[1], second[2], second[3],
            first_orientation=first[4],
            second_orientation=second[4],
            first_central_angle=first[5],
            second_central_angle=second[5],
            first_allow_full_circle=first[6],
            second_allow_full_circle=second[6],
            objects=object_ids,
        )
        return result.points if result.is_constructible else ()
    return ()


def _linear_circle_points(
    state: FormalGeometryState,
    linear_id: str,
    circle_id: str,
    object_ids: tuple[str, str],
) -> tuple[tuple[sp.Expr, ...], ...]:
    linear = state.get_line_like(linear_id)
    circle = state.circles[circle_id]
    center = state.require_point(circle.center)
    result = line_circle_intersection_result_2d(
        linear.point,
        linear.direction,
        center.coords,
        circle.radius,
        domain=state.line_like_domain(linear_id),
        objects=object_ids,
    )
    return result.points if result.is_constructible else ()


def _linear_arc_points(
    state: FormalGeometryState,
    linear_id: str,
    arc_id: str,
    object_ids: tuple[str, str],
) -> tuple[tuple[sp.Expr, ...], ...]:
    linear = state.get_line_like(linear_id)
    arc_domain = _arc_domain(state, arc_id)
    result = linear_arc_intersection_result_2d(
        linear.point,
        linear.direction,
        arc_domain[0],
        arc_domain[1],
        arc_domain[2],
        arc_domain[3],
        domain=state.line_like_domain(linear_id),
        orientation=arc_domain[4],
        central_angle=arc_domain[5],
        allow_full_circle=arc_domain[6],
        objects=object_ids,
    )
    return result.points if result.is_constructible else ()


def _circle_arc_points(
    state: FormalGeometryState,
    circle_id: str,
    arc_id: str,
    object_ids: tuple[str, str],
) -> tuple[tuple[sp.Expr, ...], ...]:
    circle = state.circles[circle_id]
    arc_domain = _arc_domain(state, arc_id)
    result = circle_arc_intersection_result_2d(
        state.require_point(circle.center).coords,
        circle.radius,
        arc_domain[0],
        arc_domain[1],
        arc_domain[2],
        arc_domain[3],
        orientation=arc_domain[4],
        central_angle=arc_domain[5],
        allow_full_circle=arc_domain[6],
        objects=object_ids,
    )
    return result.points if result.is_constructible else ()


def _arc_domain(state: FormalGeometryState, arc_id: str) -> tuple:
    arc = state.arcs[arc_id]
    circle = state.circles[arc.circle]
    return (
        state.require_point(circle.center).coords,
        circle.radius,
        state.require_point(arc.start).coords,
        state.require_point(arc.end).coords,
        arc.orientation,
        arc.central_angle,
        bool(arc.metadata.get("allow_full_circle")),
    )


def _is_line_like(state: FormalGeometryState, object_id: str) -> bool:
    return object_id in state.lines or object_id in state.segments or object_id in state.rays


def _coords_equal(left, right) -> bool:
    return len(left) == len(right) and all(sp.simplify(a - b) == 0 for a, b in zip(left, right))


def _is_negative(value: sp.Expr) -> bool:
    simplified = sp.simplify(value)
    if simplified.is_negative is not None:
        return bool(simplified.is_negative)
    return float(sp.N(simplified)) < 0


def _line_like_length(state: FormalGeometryState, object_id: str) -> sp.Expr:
    if object_id in state.segments:
        segment = state.segments[object_id]
        return distance(state.require_point(segment.start).coords, state.require_point(segment.end).coords)
    if object_id in state.lines or object_id in state.rays:
        line = state.get_line_like(object_id)
        return sp.sqrt(sum(value ** 2 for value in line.direction))
    raise GeometryWorldError(f"Unknown length object: {object_id}")


def _segments_are_adjacent_polygon_edges(
    state: FormalGeometryState,
    vertices: tuple[str, ...],
    first_segment_id: str,
    second_segment_id: str,
) -> bool:
    if first_segment_id not in state.segments or second_segment_id not in state.segments:
        return False
    polygon_edges = {
        frozenset((vertices[index], vertices[(index + 1) % len(vertices)]))
        for index in range(len(vertices))
    }
    first = state.segments[first_segment_id]
    second = state.segments[second_segment_id]
    first_edge = frozenset((first.start, first.end))
    second_edge = frozenset((second.start, second.end))
    return first_edge in polygon_edges and second_edge in polygon_edges and len(first_edge & second_edge) == 1


def _annulus_radius_formula_is_valid(state: FormalGeometryState, region, formula: dict) -> bool:
    pairs = (
        (formula.get("outer_radius_segment"), region.metadata.get("outer_circle")),
        (formula.get("inner_radius_segment"), region.metadata.get("inner_circle")),
    )
    for segment_id, circle_id in pairs:
        if segment_id not in state.segments or circle_id not in state.circles:
            return False
        segment = state.segments[segment_id]
        circle = state.circles[circle_id]
        if circle.center not in {segment.start, segment.end}:
            return False
        endpoint_id = segment.end if segment.start == circle.center else segment.start
        if sp.simplify(distance(state.require_point(circle.center).coords, state.require_point(endpoint_id).coords) - circle.radius) != 0:
            return False
    return True


def _sector_region_is_valid(state: FormalGeometryState, region) -> bool:
    circle_id = region.metadata.get("circle")
    segment_id = region.metadata.get("radius_segment")
    if circle_id not in state.circles or segment_id not in state.segments:
        return False
    circle = state.circles[circle_id]
    center_id = circle.center
    start_id = region.metadata.get("start")
    end_id = region.metadata.get("end")
    segment = state.segments[segment_id]
    if center_id not in {segment.start, segment.end} or start_id not in state.points or end_id not in state.points:
        return False
    center = state.require_point(center_id).coords
    start = state.require_point(start_id).coords
    end = state.require_point(end_id).coords
    radius_endpoint_id = segment.end if segment.start == center_id else segment.start
    if sp.simplify(distance(center, state.require_point(radius_endpoint_id).coords) - circle.radius) != 0:
        return False
    if sp.simplify(distance(center, start) - circle.radius) != 0 or sp.simplify(distance(center, end) - circle.radius) != 0:
        return False
    return sp.simplify(angle_between(sub(start, center), sub(end, center)) - sp.sympify(region.metadata.get("central_angle"))) == 0


def _circular_segment_region_is_valid(state: FormalGeometryState, region) -> bool:
    sector = state.regions.get(region.metadata.get("sector_region"))
    triangle = state.polygons.get(region.metadata.get("triangle_polygon"))
    arc = state.arcs.get(region.metadata.get("arc"))
    chord = state.segments.get(region.metadata.get("chord"))
    if sector is None or triangle is None or arc is None or chord is None or not _sector_region_is_valid(state, sector):
        return False
    endpoints = {sector.metadata.get("start"), sector.metadata.get("end")}
    return (
        set(triangle.vertices) == {sector.metadata.get("center"), *endpoints}
        and arc.circle == sector.metadata.get("circle")
        and {arc.start, arc.end} == endpoints
        and {chord.start, chord.end} == endpoints
    )


def _lens_region_is_valid(state: FormalGeometryState, region) -> bool:
    circle_ids = tuple(region.metadata.get("circles") or ())
    arc_ids = tuple(region.metadata.get("arcs") or ())
    intersection_ids = tuple(region.metadata.get("intersections") or ())
    formula = region.metadata.get("area_formula")
    if len(circle_ids) != 2 or len(arc_ids) != 2 or len(intersection_ids) != 2:
        return False
    if formula is not None and not isinstance(formula, dict):
        return False
    if any(circle_id not in state.circles for circle_id in circle_ids) or any(arc_id not in state.arcs for arc_id in arc_ids):
        return False
    if any(point_id not in state.points for point_id in intersection_ids):
        return False
    first_circle, second_circle = (state.circles[circle_id] for circle_id in circle_ids)
    center_distance = distance(
        state.require_point(first_circle.center).coords,
        state.require_point(second_circle.center).coords,
    )
    if (
        sp.simplify(first_circle.radius + second_circle.radius - center_distance).is_positive is not True
        or sp.simplify(center_distance - sp.Abs(first_circle.radius - second_circle.radius)).is_positive is not True
    ):
        return False
    endpoints = set(intersection_ids)
    if any(state.arcs[arc_id].circle != circle_id or {state.arcs[arc_id].start, state.arcs[arc_id].end} != endpoints for arc_id, circle_id in zip(arc_ids, circle_ids)):
        return False
    for arc_id, circle_id in zip(arc_ids, circle_ids):
        arc = state.arcs[arc_id]
        circle = state.circles[circle_id]
        sweep = directed_arc_angle_2d(
            state.require_point(circle.center).coords,
            state.require_point(arc.start).coords,
            state.require_point(arc.end).coords,
            orientation=arc.orientation,
        )
        if sweep.is_positive is not True or sp.simplify(sp.pi - sweep).is_nonnegative is not True:
            return False
    if formula is None:
        return region.boundary == arc_ids and isinstance(region.metadata.get("perimeter_formula"), dict)
    if formula.get("rule") == "lens_equal_radius_area":
        if sp.simplify(first_circle.radius - second_circle.radius) != 0:
            return False
        radius_segment_ids = (formula.get("radius_segment"),)
        circles_for_segments = (first_circle,)
    elif formula.get("rule") == "lens_two_radius_area":
        radius_segment_ids = tuple(formula.get("radius_segments") or ())
        if len(radius_segment_ids) != 2 or len(set(radius_segment_ids)) != 2:
            return False
        circles_for_segments = (first_circle, second_circle)
    else:
        return False
    center_segment_id = formula.get("center_segment")
    if any(segment_id not in state.segments for segment_id in (*radius_segment_ids, center_segment_id)):
        return False
    center_segment = state.segments[center_segment_id]
    for radius_segment_id, circle in zip(radius_segment_ids, circles_for_segments):
        radius_segment = state.segments[radius_segment_id]
        if circle.center not in {radius_segment.start, radius_segment.end}:
            return False
        radius_endpoint = radius_segment.end if radius_segment.start == circle.center else radius_segment.start
        if sp.simplify(distance(state.require_point(circle.center).coords, state.require_point(radius_endpoint).coords) - circle.radius) != 0:
            return False
    return region.boundary == arc_ids and {center_segment.start, center_segment.end} == {first_circle.center, second_circle.center}


def _lens_perimeter_formula_is_valid(state: FormalGeometryState, region, formula: dict) -> bool:
    circle_ids = tuple(region.metadata.get("circles") or ())
    if len(circle_ids) != 2 or any(circle_id not in state.circles for circle_id in circle_ids):
        return False
    first_circle, second_circle = (state.circles[circle_id] for circle_id in circle_ids)
    rule = formula.get("rule")
    if rule == "lens_equal_radius_perimeter":
        if sp.simplify(first_circle.radius - second_circle.radius) != 0:
            return False
        radius_segment_ids = (formula.get("radius_segment"),)
        circles_for_segments = (first_circle,)
    elif rule == "lens_two_radius_perimeter":
        radius_segment_ids = tuple(formula.get("radius_segments") or ())
        if len(radius_segment_ids) != 2 or len(set(radius_segment_ids)) != 2:
            return False
        circles_for_segments = (first_circle, second_circle)
    else:
        return False
    if any(segment_id not in state.segments for segment_id in (*radius_segment_ids, formula.get("center_segment"))):
        return False
    expected_radius_segment_ids = (
        (region.metadata.get("radius_segment"),)
        if rule == "lens_equal_radius_perimeter"
        else tuple(region.metadata.get("radius_segments") or ())
    )
    if tuple(radius_segment_ids) != expected_radius_segment_ids:
        return False
    if formula.get("center_segment") != region.metadata.get("center_segment"):
        return False
    for segment_id, circle in zip(radius_segment_ids, circles_for_segments):
        segment = state.segments[segment_id]
        if circle.center not in {segment.start, segment.end}:
            return False
        endpoint = segment.end if segment.start == circle.center else segment.start
        if sp.simplify(distance(state.require_point(circle.center).coords, state.require_point(endpoint).coords) - circle.radius) != 0:
            return False
    center_segment = state.segments[formula["center_segment"]]
    return {center_segment.start, center_segment.end} == {first_circle.center, second_circle.center}


def _angle_value(state: FormalGeometryState, angle: tuple[str, str, str]) -> sp.Expr:
    left, vertex, right = (state.require_point(point_id) for point_id in angle)
    return angle_between(sub(left.coords, vertex.coords), sub(right.coords, vertex.coords))


def _polygon_side_squares(points: Dict[str, tuple], vertices: tuple[str, ...]) -> tuple[sp.Expr, ...]:
    values = []
    for index, vertex in enumerate(vertices):
        next_vertex = vertices[(index + 1) % len(vertices)]
        delta = sub(points[next_vertex], points[vertex])
        values.append(sp.simplify(sum(value ** 2 for value in delta)))
    return tuple(values)


def _output_ids(op: ConstructionOp) -> List[str]:
    if "ids" in op.params:
        return [str(item) for item in op.params["ids"]]
    if "point_ids" in op.params:
        return [str(item) for item in op.params["point_ids"]]
    return [op.id] if op.id else []
