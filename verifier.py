"""Trace and state verification for the geometry world prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Set

from .builder import ConstructionBuilder
from .construction import ConstructionOp, normalize_trace
from .errors import GeometryWorldError
from .state import FormalGeometryState


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

    def verify_trace(self, trace: Iterable[ConstructionOp | Dict[str, Any]], *, execute: bool = True) -> VerificationReport:
        ops = normalize_trace(trace)
        report = VerificationReport()
        self._static_trace_checks(ops, report)
        state = None
        if execute and report.static_valid:
            try:
                state = ConstructionBuilder().execute(ops)
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

    def _static_trace_checks(self, ops: List[ConstructionOp], report: VerificationReport) -> None:
        created: Set[str] = set()
        point_ids: Set[str] = set()
        line_ids: Set[str] = set()
        segment_ids: Set[str] = set()
        circle_ids: Set[str] = set()
        polygon_ids: Set[str] = set()
        ray_ids: Set[str] = set()
        plane_ids: Set[str] = set()
        polyhedron_ids: Set[str] = set()

        for idx, op in enumerate(ops):
            if op.id and op.id in created and op.op not in {"set_target"}:
                report.add("error", "trace.duplicate_id", f"Duplicate construction id: {op.id}", op_index=idx, op=op.op, object_id=op.id)

            refs = _references_for_op(op)
            for ref_kind, ref_id in refs:
                known = {
                    "point": point_ids,
                    "line": line_ids | segment_ids,
                    "segment": segment_ids,
                    "circle": circle_ids,
                    "polygon": polygon_ids,
                    "ray": ray_ids,
                    "plane": plane_ids,
                    "polyhedron": polyhedron_ids,
                }.get(ref_kind, created)
                if ref_id not in known:
                    report.add("error", "trace.unknown_reference", f"Unknown {ref_kind} reference: {ref_id}", op_index=idx, op=op.op, ref_kind=ref_kind, ref_id=ref_id)

            if op.op == "add_point3d":
                coord = op.params.get("coord", op.params.get("coords"))
                if coord is not None and len(coord) != 3:
                    report.add("error", "trace.dimension_mismatch", "add_point3d requires exactly three coordinates", op_index=idx, op=op.op, object_id=op.id)

            if op.op in {"intersection_line_circle", "intersection_circle_circle", "tangent_points_from_point"}:
                for point_id in _output_ids(op):
                    if point_id in created:
                        report.add("error", "trace.duplicate_id", f"Duplicate construction id: {point_id}", op_index=idx, op=op.op, object_id=point_id)
                    created.add(point_id)
                    point_ids.add(point_id)
                continue

            if op.id and op.op not in {"set_target"}:
                created.add(op.id)
                if op.op in {"add_point", "add_point3d", "midpoint", "intersection_line_line", "foot_of_perpendicular", "point_on_ray_distance", "point_on_segment_ratio", "project_point_to_plane"}:
                    point_ids.add(op.id)
                elif op.op in {"add_line", "parallel_through", "perpendicular_through", "line_through_angle", "project_line_to_plane"}:
                    line_ids.add(op.id)
                elif op.op in {"add_segment", "add_chord"}:
                    segment_ids.add(op.id)
                elif op.op == "add_circle":
                    circle_ids.add(op.id)
                elif op.op == "add_polygon":
                    polygon_ids.add(op.id)
                elif op.op in {"add_ray", "rotate_ray"}:
                    ray_ids.add(op.id)
                elif op.op == "add_arc":
                    pass
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


def _references_for_op(op: ConstructionOp) -> List[tuple[str, str]]:
    p = op.params
    refs: List[tuple[str, str]] = []
    if op.op == "add_segment":
        refs.extend(("point", item) for item in p.get("points", [p.get("start"), p.get("end")]) if item)
    elif op.op == "add_line":
        refs.extend(("point", item) for item in p.get("points", []))
        if "point" in p:
            refs.append(("point", p["point"]))
    elif op.op == "add_ray":
        refs.append(("point", p["start"]))
        if "through" in p:
            refs.append(("point", p["through"]))
    elif op.op == "add_circle":
        refs.append(("point", p["center"]))
    elif op.op == "add_arc":
        refs.extend([("circle", p["circle"]), ("point", p["start"]), ("point", p["end"])])
    elif op.op == "add_polygon":
        refs.extend(("point", item) for item in p.get("vertices", []))
    elif op.op == "mark_polygon_region":
        if "polygon" in p:
            refs.append(("polygon", p["polygon"]))
        refs.extend(("point", item) for item in p.get("vertices", []))
    elif op.op == "mark_sector_region":
        refs.extend([("circle", p["circle"]), ("point", p["start"]), ("point", p["end"])])
    elif op.op == "midpoint":
        refs.extend(("point", item) for item in (p.get("of") or p.get("points") or []))
    elif op.op == "intersection_line_line":
        refs.extend([("line", p["line1"]), ("line", p["line2"])])
    elif op.op == "intersection_line_circle":
        refs.extend([("line", p["line"]), ("circle", p["circle"])])
    elif op.op == "intersection_circle_circle":
        refs.extend([("circle", p["circle1"]), ("circle", p["circle2"])])
    elif op.op == "tangent_points_from_point":
        refs.extend([("point", p["point"]), ("circle", p["circle"])])
    elif op.op == "add_chord":
        refs.append(("circle", p["circle"]))
        refs.extend(("point", item) for item in p.get("points", []))
    elif op.op in {"parallel_through", "perpendicular_through"}:
        refs.extend([("point", p["point"]), ("line", p["line"])])
    elif op.op == "line_through_angle":
        refs.append(("point", p["point"]))
        if "base_line" in p:
            refs.append(("line", p["base_line"]))
    elif op.op == "rotate_ray":
        refs.append(("ray", p["ray"]))
    elif op.op == "angle_bisector":
        refs.append(("point", p["vertex"]))
        refs.extend(("point", item) for item in p.get("ray1", []))
        refs.extend(("point", item) for item in p.get("ray2", []))
    elif op.op == "foot_of_perpendicular":
        refs.extend([("point", p["point"]), ("line", p["line"])])
    elif op.op == "point_on_ray_distance":
        if "ray" in p:
            refs.append(("ray", p["ray"]))
        if "start" in p:
            refs.append(("point", p["start"]))
    elif op.op == "point_on_segment_ratio":
        if "segment" in p:
            refs.append(("segment", p["segment"]))
        refs.extend(("point", item) for item in p.get("points", []))
    elif op.op == "project_point_to_plane":
        refs.extend([("point", p["point"]), ("plane", p["plane"])])
    elif op.op == "project_line_to_plane":
        refs.extend([("line", p["line"]), ("plane", p["plane"])])
    elif op.op == "add_plane":
        refs.extend(("point", item) for item in p.get("points", []))
        if "point" in p:
            refs.append(("point", p["point"]))
    elif op.op == "add_polyhedron":
        refs.extend(("point", item) for item in p.get("vertices", []))
    elif op.op == "dihedral_angle":
        refs.extend(("plane", item) for item in p.get("planes", []))
        if "polyhedron" in p:
            refs.append(("polyhedron", p["polyhedron"]))
    elif op.op == "section_polyhedron_with_plane":
        refs.extend([("polyhedron", p["polyhedron"]), ("plane", p["plane"])])
        refs.extend(("point", item) for item in p.get("boundary", []))
    return [(kind, str(ref)) for kind, ref in refs if ref is not None]


def _output_ids(op: ConstructionOp) -> List[str]:
    if "ids" in op.params:
        return [str(item) for item in op.params["ids"]]
    if "point_ids" in op.params:
        return [str(item) for item in op.params["point_ids"]]
    return [op.id] if op.id else []
