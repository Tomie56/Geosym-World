"""Replay construction traces into exact formal geometry states."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Sequence

import sympy as sp

from geosym_world.construction.operations import ConstructionOp, normalize_trace
from geosym_world.errors import DegenerateGeometryError, UnsupportedConstructionError
from geosym_world.kernel.geometry2d.analytic import (
    add,
    circle_circle_intersections_2d,
    coords,
    cross3,
    dot,
    expr,
    foot_point_2d,
    line_circle_intersections_2d,
    line_intersection_2d,
    midpoint,
    mul,
    norm_sq,
    project_point_to_plane,
    project_vector_to_plane,
    rotate2,
    segment_plane_intersection_3d,
    simplify_coords,
    sub,
    tangent_points_from_point_to_circle_2d,
    unit,
)
from geosym_world.schema.entities import Arc, Circle, Line, Plane, Point, Polygon, Polyhedron, Ray, Region, Segment, Target
from geosym_world.schema.state import FormalGeometryState


class ConstructionBuilder:
    """Executes replayable construction traces."""

    def __init__(self, state: FormalGeometryState | None = None):
        self.state = state or FormalGeometryState()

    def execute(self, trace: Iterable[ConstructionOp | Dict[str, Any]]) -> FormalGeometryState:
        for op in normalize_trace(trace):
            self.execute_op(op)
        return self.state

    def execute_op(self, op: ConstructionOp) -> None:
        handler = getattr(self, f"_op_{op.op}", None)
        if handler is None:
            raise UnsupportedConstructionError(f"Unsupported construction op: {op.op}")
        handler(op.id, op.params)
        self.state.construction_trace.append(op.to_dict())

    def _op_add_point(self, object_id: str | None, params: Dict[str, Any]) -> None:
        point_id = _required_id(object_id)
        raw = params.get("coord", params.get("coords"))
        if raw is None:
            raw = [params["x"], params["y"]] if "z" not in params else [params["x"], params["y"], params["z"]]
        self.state.add_point(Point(id=point_id, coords=coords(raw), metadata=_metadata(params)))

    def _op_add_point3d(self, object_id: str | None, params: Dict[str, Any]) -> None:
        self._op_add_point(object_id, params)
        point = self.state.require_point(_required_id(object_id))
        if point.dim != 3:
            raise DegenerateGeometryError("add_point3d requires three coordinates")

    def _op_add_segment(self, object_id: str | None, params: Dict[str, Any]) -> None:
        start, end = _point_pair(params)
        seg_id = object_id or f"{start}{end}"
        self._ensure_distinct_points(start, end)
        self.state.add_segment(Segment(id=seg_id, start=start, end=end, metadata=_metadata(params)))

    def _op_add_line(self, object_id: str | None, params: Dict[str, Any]) -> None:
        line_id = _required_id(object_id)
        if "points" in params:
            a_id, b_id = _as_pair(params["points"], "points")
            self._ensure_distinct_points(a_id, b_id)
            a = self.state.require_point(a_id)
            b = self.state.require_point(b_id)
            line = Line(id=line_id, point=a.coords, direction=sub(b.coords, a.coords), source_points=(a_id, b_id), metadata=_metadata(params))
        else:
            point_id = params["point"]
            point = self.state.require_point(point_id)
            direction = coords(params["direction"])
            _ensure_nonzero(direction, "line direction")
            line = Line(id=line_id, point=point.coords, direction=direction, source_points=(point_id,), metadata=_metadata(params))
        self.state.add_line(line)

    def _op_add_ray(self, object_id: str | None, params: Dict[str, Any]) -> None:
        ray_id = _required_id(object_id)
        start = params["start"]
        self.state.require_point(start)
        if "through" in params:
            through = params["through"]
            self._ensure_distinct_points(start, through)
            direction = sub(self.state.require_point(through).coords, self.state.require_point(start).coords)
        else:
            through = None
            direction = coords(params["direction"])
            _ensure_nonzero(direction, "ray direction")
        self.state.add_ray(Ray(id=ray_id, start=start, direction=direction, through=through, metadata=_metadata(params)))

    def _op_add_circle(self, object_id: str | None, params: Dict[str, Any]) -> None:
        circle_id = _required_id(object_id)
        radius = expr(params["radius"])
        if sp.simplify(radius) == 0:
            raise DegenerateGeometryError("Circle radius cannot be zero")
        self.state.add_circle(Circle(id=circle_id, center=params["center"], radius=radius, metadata=_metadata(params)))

    def _op_add_arc(self, object_id: str | None, params: Dict[str, Any]) -> None:
        arc_id = _required_id(object_id)
        angle = expr(params["central_angle"]) if "central_angle" in params else None
        self.state.add_arc(
            Arc(
                id=arc_id,
                circle=params["circle"],
                start=params["start"],
                end=params["end"],
                central_angle=angle,
                orientation=params.get("orientation", "ccw"),
                metadata=_metadata(params),
            )
        )
        self.state.add_relation(
            {
                "type": "arc",
                "object": arc_id,
                "circle": params["circle"],
                "start": params["start"],
                "end": params["end"],
                "central_angle": str(angle) if angle is not None else None,
                "orientation": params.get("orientation", "ccw"),
            }
        )

    def _op_add_polygon(self, object_id: str | None, params: Dict[str, Any]) -> None:
        polygon_id = _required_id(object_id)
        vertices = tuple(params["vertices"])
        if len(vertices) < 3:
            raise DegenerateGeometryError("A polygon needs at least three vertices")
        self.state.add_polygon(Polygon(id=polygon_id, vertices=vertices, metadata=_metadata(params)))

    def _op_add_region(self, object_id: str | None, params: Dict[str, Any]) -> None:
        region_id = _required_id(object_id)
        kind = params.get("kind", "polygon_region")
        metadata = _metadata(params)
        if kind == "annulus_region":
            inner = metadata.get("inner_circle")
            outer = metadata.get("outer_circle")
            if inner not in self.state.circles or outer not in self.state.circles:
                raise ValueError("annulus_region requires existing inner_circle and outer_circle")
            if self.state.circles[inner].center != self.state.circles[outer].center:
                raise ValueError("annulus_region circles must be concentric")
            radius_difference = sp.simplify(self.state.circles[outer].radius - self.state.circles[inner].radius)
            if radius_difference.is_positive is not True:
                raise DegenerateGeometryError("annulus_region outer radius must exceed inner radius")
        self.state.add_region(
            Region(
                id=region_id,
                boundary=tuple(params["boundary"]),
                kind=kind,
                metadata=metadata,
            )
        )

    def _op_mark_polygon_region(self, object_id: str | None, params: Dict[str, Any]) -> None:
        region_id = _required_id(object_id)
        if "polygon" in params:
            polygon = self.state.polygons[params["polygon"]]
            boundary = polygon.vertices
            metadata = {"polygon": params["polygon"], **_metadata(params)}
        else:
            boundary = tuple(params["vertices"])
            metadata = _metadata(params)
        self.state.add_region(Region(id=region_id, boundary=tuple(boundary), kind="polygon_region", metadata=metadata))

    def _op_mark_sector_region(self, object_id: str | None, params: Dict[str, Any]) -> None:
        region_id = _required_id(object_id)
        circle = self.state.circles[params["circle"]]
        center = circle.center
        start = params["start"]
        end = params["end"]
        self.state.require_points([center, start, end])
        central_angle = expr(params["central_angle"])
        self.state.add_region(
            Region(
                id=region_id,
                boundary=(center, start, end),
                kind="sector_region",
                metadata={
                    "circle": params["circle"],
                    "center": center,
                    "start": start,
                    "end": end,
                    "central_angle": str(central_angle),
                    **_metadata(params),
                },
            )
        )
        self.state.add_relation({"type": "sector_region", "object": region_id, "circle": params["circle"], "central_angle": str(central_angle)})

    def _op_set_target(self, object_id: str | None, params: Dict[str, Any]) -> None:
        target_type = params.get("type") or params.get("target_type")
        target_object = params.get("object") or object_id
        if not target_type or not target_object:
            raise ValueError("set_target requires type and object")
        self.state.set_target(Target(type=str(target_type), object=str(target_object), metadata=_metadata(params)))

    def _op_midpoint(self, object_id: str | None, params: Dict[str, Any]) -> None:
        mid_id = _required_id(object_id)
        a_id, b_id = _as_pair(params.get("of") or params.get("points"), "of")
        self._ensure_distinct_points(a_id, b_id)
        a = self.state.require_point(a_id)
        b = self.state.require_point(b_id)
        point = Point(id=mid_id, coords=midpoint(a.coords, b.coords), metadata={"constructed_by": "midpoint", **_metadata(params)})
        self.state.add_point(point)
        self.state.add_relation({"type": "midpoint", "point": mid_id, "of": [a_id, b_id]})

    def _op_intersection_line_line(self, object_id: str | None, params: Dict[str, Any]) -> None:
        point_id = _required_id(object_id)
        l1 = self.state.get_line_like(params["line1"])
        l2 = self.state.get_line_like(params["line2"])
        if l1.dim != 2 or l2.dim != 2:
            raise UnsupportedConstructionError("line-line intersection is currently implemented for 2D only")
        point = line_intersection_2d(l1.point, l1.direction, l2.point, l2.direction)
        self.state.add_point(Point(id=point_id, coords=point, metadata={"constructed_by": "intersection_line_line", **_metadata(params)}))
        self.state.add_relation({"type": "intersection", "point": point_id, "objects": [params["line1"], params["line2"]]})

    def _op_intersection_line_circle(self, object_id: str | None, params: Dict[str, Any]) -> None:
        point_ids = _point_output_ids(object_id, params)
        line = self.state.get_line_like(params["line"])
        circle = self.state.circles[params["circle"]]
        center = self.state.require_point(circle.center)
        if line.dim != 2 or center.dim != 2:
            raise UnsupportedConstructionError("line-circle intersection is currently implemented for 2D only")
        points = line_circle_intersections_2d(line.point, line.direction, center.coords, circle.radius)
        if len(point_ids) != len(points):
            raise ValueError(f"Expected {len(points)} output point ids, got {len(point_ids)}")
        for point_id, point in zip(point_ids, points):
            self.state.add_point(Point(id=point_id, coords=point, metadata={"constructed_by": "intersection_line_circle", **_metadata(params)}))
        self.state.add_relation({"type": "intersection", "points": list(point_ids), "objects": [params["line"], params["circle"]]})

    def _op_intersection_circle_circle(self, object_id: str | None, params: Dict[str, Any]) -> None:
        point_ids = _point_output_ids(object_id, params)
        c1 = self.state.circles[params["circle1"]]
        c2 = self.state.circles[params["circle2"]]
        p1 = self.state.require_point(c1.center)
        p2 = self.state.require_point(c2.center)
        if p1.dim != 2 or p2.dim != 2:
            raise UnsupportedConstructionError("circle-circle intersection is currently implemented for 2D only")
        points = circle_circle_intersections_2d(p1.coords, c1.radius, p2.coords, c2.radius)
        if len(point_ids) != len(points):
            raise ValueError(f"Expected {len(points)} output point ids, got {len(point_ids)}")
        for point_id, point in zip(point_ids, points):
            self.state.add_point(Point(id=point_id, coords=point, metadata={"constructed_by": "intersection_circle_circle", **_metadata(params)}))
        self.state.add_relation({"type": "intersection", "points": list(point_ids), "objects": [params["circle1"], params["circle2"]]})

    def _op_tangent_points_from_point(self, object_id: str | None, params: Dict[str, Any]) -> None:
        point_ids = _point_output_ids(object_id, params)
        source = self.state.require_point(params["point"])
        circle = self.state.circles[params["circle"]]
        center = self.state.require_point(circle.center)
        if source.dim != 2 or center.dim != 2:
            raise UnsupportedConstructionError("circle tangent points are currently implemented for 2D only")
        points = tangent_points_from_point_to_circle_2d(source.coords, center.coords, circle.radius)
        if len(point_ids) != len(points):
            raise ValueError(f"Expected {len(points)} output point ids, got {len(point_ids)}")
        for point_id, point in zip(point_ids, points):
            self.state.add_point(Point(id=point_id, coords=point, metadata={"constructed_by": "tangent_points_from_point", **_metadata(params)}))
            self.state.add_relation({"type": "tangent", "point": point_id, "line": [params["point"], point_id], "circle": params["circle"]})

    def _op_add_chord(self, object_id: str | None, params: Dict[str, Any]) -> None:
        chord_id = _required_id(object_id)
        start, end = _as_pair(params["points"], "points")
        self._ensure_distinct_points(start, end)
        circle_id = params["circle"]
        if circle_id not in self.state.circles:
            raise ValueError(f"Unknown circle: {circle_id}")
        self.state.add_segment(Segment(id=chord_id, start=start, end=end, metadata={"circle": circle_id, **_metadata(params)}))
        self.state.add_relation({"type": "chord", "object": chord_id, "circle": circle_id, "points": [start, end]})

    def _op_parallel_through(self, object_id: str | None, params: Dict[str, Any]) -> None:
        line_id = _required_id(object_id)
        through = self.state.require_point(params["point"])
        base = self.state.get_line_like(params["line"])
        self.state.add_line(
            Line(id=line_id, point=through.coords, direction=base.direction, source_points=(params["point"],), metadata={"constructed_by": "parallel_through", **_metadata(params)})
        )
        self.state.add_relation({"type": "parallel", "objects": [line_id, params["line"]]})

    def _op_perpendicular_through(self, object_id: str | None, params: Dict[str, Any]) -> None:
        line_id = _required_id(object_id)
        through = self.state.require_point(params["point"])
        base = self.state.get_line_like(params["line"])
        if base.dim != 2:
            raise UnsupportedConstructionError("perpendicular_through is currently implemented for 2D lines")
        direction = simplify_coords((-base.direction[1], base.direction[0]))
        self.state.add_line(
            Line(id=line_id, point=through.coords, direction=direction, source_points=(params["point"],), metadata={"constructed_by": "perpendicular_through", **_metadata(params)})
        )
        self.state.add_relation({"type": "perpendicular", "objects": [line_id, params["line"]]})

    def _op_line_through_angle(self, object_id: str | None, params: Dict[str, Any]) -> None:
        line_id = _required_id(object_id)
        point_id = params["point"]
        point = self.state.require_point(point_id)
        if point.dim != 2:
            raise UnsupportedConstructionError("line_through_angle is currently implemented for 2D")
        if "base_line" in params:
            base_dir = self.state.get_line_like(params["base_line"]).direction
        else:
            base_dir = coords((1, 0))
        direction = rotate2(base_dir, params["angle"])
        self.state.add_line(
            Line(id=line_id, point=point.coords, direction=direction, source_points=(point_id,), metadata={"constructed_by": "line_through_angle", **_metadata(params)})
        )
        self.state.add_relation({"type": "angle_from_line", "line": line_id, "base_line": params.get("base_line"), "angle": str(expr(params["angle"]))})

    def _op_rotate_ray(self, object_id: str | None, params: Dict[str, Any]) -> None:
        ray_id = _required_id(object_id)
        base_ray = self.state.rays[params["ray"]]
        start = params.get("start", base_ray.start)
        direction = rotate2(base_ray.direction, params["angle"])
        self.state.add_ray(Ray(id=ray_id, start=start, direction=direction, metadata={"constructed_by": "rotate_ray", **_metadata(params)}))

    def _op_angle_bisector(self, object_id: str | None, params: Dict[str, Any]) -> None:
        line_id = _required_id(object_id)
        vertex_id = params["vertex"]
        vertex = self.state.require_point(vertex_id)
        ray1 = _as_pair(params["ray1"], "ray1")
        ray2 = _as_pair(params["ray2"], "ray2")
        if ray1[0] != vertex_id or ray2[0] != vertex_id:
            raise ValueError("angle_bisector expects rays to start at the vertex")
        a = self.state.require_point(ray1[1])
        c = self.state.require_point(ray2[1])
        u = unit(sub(a.coords, vertex.coords))
        v = unit(sub(c.coords, vertex.coords))
        direction = add(u, v) if params.get("mode", "internal") == "internal" else sub(u, v)
        _ensure_nonzero(direction, "angle bisector direction")
        self.state.add_line(
            Line(id=line_id, point=vertex.coords, direction=direction, source_points=(vertex_id,), metadata={"constructed_by": "angle_bisector", **_metadata(params)})
        )
        self.state.add_relation({"type": "angle_bisector", "line": line_id, "angle": [ray1[1], vertex_id, ray2[1]], "mode": params.get("mode", "internal")})

    def _op_foot_of_perpendicular(self, object_id: str | None, params: Dict[str, Any]) -> None:
        foot_id = _required_id(object_id)
        point = self.state.require_point(params["point"])
        line = self.state.get_line_like(params["line"])
        if point.dim != 2 or line.dim != 2:
            raise UnsupportedConstructionError("foot_of_perpendicular is currently implemented for 2D")
        foot = foot_point_2d(point.coords, line.point, line.direction)
        self.state.add_point(Point(id=foot_id, coords=foot, metadata={"constructed_by": "foot_of_perpendicular", **_metadata(params)}))
        self.state.add_relation({"type": "foot_of_perpendicular", "point": foot_id, "from": params["point"], "line": params["line"]})

    def _op_point_on_ray_distance(self, object_id: str | None, params: Dict[str, Any]) -> None:
        point_id = _required_id(object_id)
        distance = expr(params["distance"])
        if "ray" in params:
            ray = self.state.rays[params["ray"]]
            start = self.state.require_point(ray.start)
            direction = unit(ray.direction)
        else:
            start = self.state.require_point(params["start"])
            direction = unit(coords(params["direction"]))
        self.state.add_point(Point(id=point_id, coords=add(start.coords, mul(direction, distance)), metadata={"constructed_by": "point_on_ray_distance", **_metadata(params)}))

    def _op_point_on_segment_ratio(self, object_id: str | None, params: Dict[str, Any]) -> None:
        point_id = _required_id(object_id)
        if "segment" in params:
            seg = self.state.segments[params["segment"]]
            a_id, b_id = seg.start, seg.end
        else:
            a_id, b_id = _as_pair(params["points"], "points")
        ratio = expr(params["ratio"])
        a = self.state.require_point(a_id)
        b = self.state.require_point(b_id)
        point = add(a.coords, mul(sub(b.coords, a.coords), ratio))
        self.state.add_point(Point(id=point_id, coords=point, metadata={"constructed_by": "point_on_segment_ratio", **_metadata(params)}))
        self.state.add_relation({"type": "point_on_segment_ratio", "point": point_id, "segment": [a_id, b_id], "ratio": str(ratio)})

    def _op_project_point_to_plane(self, object_id: str | None, params: Dict[str, Any]) -> None:
        point_id = _required_id(object_id)
        source = self.state.require_point(params["point"])
        plane = self.state.planes[params["plane"]]
        if source.dim != 3:
            raise UnsupportedConstructionError("project_point_to_plane requires a 3D source point")
        projected = project_point_to_plane(source.coords, plane.point, plane.normal)
        self.state.add_point(Point(id=point_id, coords=projected, metadata={"constructed_by": "project_point_to_plane", **_metadata(params)}))
        self.state.add_relation({"type": "point_projection_plane", "point": point_id, "from": params["point"], "plane": params["plane"]})

    def _op_project_line_to_plane(self, object_id: str | None, params: Dict[str, Any]) -> None:
        line_id = _required_id(object_id)
        source = self.state.get_line_like(params["line"])
        plane = self.state.planes[params["plane"]]
        if source.dim != 3:
            raise UnsupportedConstructionError("project_line_to_plane requires a 3D source line")
        projected_point = project_point_to_plane(source.point, plane.point, plane.normal)
        projected_direction = project_vector_to_plane(source.direction, plane.normal)
        _ensure_nonzero(projected_direction, "projected line direction")
        self.state.add_line(
            Line(
                id=line_id,
                point=projected_point,
                direction=projected_direction,
                source_points=source.source_points,
                metadata={"constructed_by": "project_line_to_plane", **_metadata(params)},
            )
        )
        self.state.add_relation({"type": "line_projection_plane", "line": line_id, "from": params["line"], "plane": params["plane"]})

    def _op_add_plane(self, object_id: str | None, params: Dict[str, Any]) -> None:
        plane_id = _required_id(object_id)
        if "points" in params:
            a_id, b_id, c_id = _as_triple(params["points"], "points")
            a = self.state.require_point(a_id)
            b = self.state.require_point(b_id)
            c = self.state.require_point(c_id)
            if a.dim != 3 or b.dim != 3 or c.dim != 3:
                raise DegenerateGeometryError("Plane source points must be 3D")
            normal = cross3(sub(b.coords, a.coords), sub(c.coords, a.coords))
            _ensure_nonzero(normal, "plane normal")
            plane = Plane(id=plane_id, point=a.coords, normal=normal, source_points=(a_id, b_id, c_id), metadata=_metadata(params))
        else:
            point = self.state.require_point(params["point"])
            normal = coords(params["normal"])
            if point.dim != 3 or len(normal) != 3:
                raise DegenerateGeometryError("add_plane requires 3D point and normal")
            _ensure_nonzero(normal, "plane normal")
            plane = Plane(id=plane_id, point=point.coords, normal=normal, source_points=(params["point"],), metadata=_metadata(params))
        self.state.add_plane(plane)

    def _op_add_polyhedron(self, object_id: str | None, params: Dict[str, Any]) -> None:
        poly_id = _required_id(object_id)
        vertices = tuple(params["vertices"])
        faces = tuple(tuple(face) for face in params["faces"])
        for vertex in vertices:
            if self.state.require_point(vertex).dim != 3:
                raise DegenerateGeometryError("Polyhedron vertices must be 3D")
        self.state.add_polyhedron(Polyhedron(id=poly_id, vertices=vertices, faces=faces, metadata=_metadata(params)))

    def _op_dihedral_angle(self, object_id: str | None, params: Dict[str, Any]) -> None:
        angle_id = _required_id(object_id)
        relation = {"type": "dihedral_angle", "object": angle_id}
        if "planes" in params:
            plane1, plane2 = _as_pair(params["planes"], "planes")
            if plane1 not in self.state.planes or plane2 not in self.state.planes:
                raise ValueError("dihedral_angle plane references must exist")
            relation.update({"planes": [plane1, plane2]})
        elif "faces" in params:
            face1, face2 = _as_pair(params["faces"], "faces")
            relation.update({"polyhedron": params["polyhedron"], "faces": [face1, face2]})
        else:
            raise ValueError("dihedral_angle requires planes or faces")
        self.state.add_relation({**relation, **_metadata(params)})

    def _op_section_polyhedron_with_plane(self, object_id: str | None, params: Dict[str, Any]) -> None:
        section_id = _required_id(object_id)
        polyhedron_id = params["polyhedron"]
        plane_id = params["plane"]
        if polyhedron_id not in self.state.polyhedra:
            raise ValueError(f"Unknown polyhedron: {polyhedron_id}")
        if plane_id not in self.state.planes:
            raise ValueError(f"Unknown plane: {plane_id}")
        boundary = tuple(params.get("boundary", ()))
        if boundary:
            self.state.require_points(boundary)
        else:
            boundary = self._auto_section_boundary(section_id, polyhedron_id, plane_id, params)
        polygon_id = None
        if len(boundary) >= 3 and params.get("create_polygon", True):
            polygon_id = params.get("polygon_id", f"{section_id}_polygon")
            ordered_boundary = _sort_points_on_plane(boundary, self.state.planes[plane_id], self.state)
            self.state.add_polygon(
                Polygon(
                    id=polygon_id,
                    vertices=tuple(ordered_boundary),
                    metadata={
                        "constructed_by": "section_polyhedron_with_plane",
                        "section": section_id,
                        "polyhedron": polyhedron_id,
                        "plane": plane_id,
                    },
                )
            )
            boundary = tuple(ordered_boundary)
        self.state.add_relation(
            {
                "type": "polyhedron_plane_section",
                "object": section_id,
                "polyhedron": polyhedron_id,
                "plane": plane_id,
                "boundary": list(boundary),
                "polygon": polygon_id,
                **_metadata(params),
            }
        )

    def _auto_section_boundary(self, section_id: str, polyhedron_id: str, plane_id: str, params: Dict[str, Any]) -> tuple[str, ...]:
        polyhedron = self.state.polyhedra[polyhedron_id]
        plane = self.state.planes[plane_id]
        seen_coords: set[tuple[str, ...]] = set()
        boundary: list[str] = []
        prefix = params.get("point_prefix", f"{section_id}_p")
        for left_id, right_id in _polyhedron_edges(polyhedron.faces):
            left = self.state.require_point(left_id)
            right = self.state.require_point(right_id)
            point = segment_plane_intersection_3d(left.coords, right.coords, plane.point, plane.normal)
            if point is None:
                continue
            key = tuple(str(value) for value in point)
            if key in seen_coords:
                continue
            seen_coords.add(key)
            point_id = f"{prefix}{len(boundary)}"
            self.state.add_point(
                Point(
                    id=point_id,
                    coords=point,
                    metadata={
                        "constructed_by": "section_polyhedron_with_plane",
                        "polyhedron": polyhedron_id,
                        "plane": plane_id,
                        "edge": [left_id, right_id],
                    },
                )
            )
            boundary.append(point_id)
        return tuple(boundary)

    def _ensure_distinct_points(self, a_id: str, b_id: str) -> None:
        a = self.state.require_point(a_id)
        b = self.state.require_point(b_id)
        if a.coords == b.coords:
            raise DegenerateGeometryError(f"Points {a_id} and {b_id} have identical coordinates")


def _required_id(object_id: str | None) -> str:
    if not object_id:
        raise ValueError("Construction op requires an id")
    return object_id


def _as_pair(values: Sequence[str], name: str) -> tuple[str, str]:
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two ids")
    return str(values[0]), str(values[1])


def _as_triple(values: Sequence[str], name: str) -> tuple[str, str, str]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three ids")
    return str(values[0]), str(values[1]), str(values[2])


def _point_pair(params: Dict[str, Any]) -> tuple[str, str]:
    if "points" in params:
        return _as_pair(params["points"], "points")
    return str(params["start"]), str(params["end"])


def _point_output_ids(object_id: str | None, params: Dict[str, Any]) -> tuple[str, ...]:
    if "ids" in params:
        return tuple(str(item) for item in params["ids"])
    if "point_ids" in params:
        return tuple(str(item) for item in params["point_ids"])
    return (_required_id(object_id),)


def _ensure_nonzero(vector, label: str) -> None:
    if sp.simplify(norm_sq(tuple(vector))) == 0:
        raise DegenerateGeometryError(f"Zero {label}")


def _metadata(params: Dict[str, Any]) -> Dict[str, Any]:
    return dict(params.get("metadata") or {})


def _polyhedron_edges(faces: Sequence[Sequence[str]]) -> tuple[tuple[str, str], ...]:
    edges = set()
    for face in faces:
        for index, left in enumerate(face):
            right = face[(index + 1) % len(face)]
            edges.add(tuple(sorted((str(left), str(right)))))
    return tuple(sorted(edges))


def _sort_points_on_plane(point_ids: Sequence[str], plane: Plane, state: FormalGeometryState) -> tuple[str, ...]:
    if len(point_ids) <= 2:
        return tuple(point_ids)
    points = [state.require_point(point_id).coords for point_id in point_ids]
    centroid = tuple(sp.simplify(sum(point[index] for point in points) / len(points)) for index in range(3))
    u = _plane_basis_u(plane.normal)
    v = cross3(plane.normal, u)

    def key(point_id: str) -> float:
        vec = sub(state.require_point(point_id).coords, centroid)
        x = float(sp.N(dot(vec, u)))
        y = float(sp.N(dot(vec, v)))
        return math.atan2(y, x)

    return tuple(sorted(point_ids, key=key))


def _plane_basis_u(normal) -> tuple:
    axes = ((sp.Integer(1), sp.Integer(0), sp.Integer(0)), (sp.Integer(0), sp.Integer(1), sp.Integer(0)), (sp.Integer(0), sp.Integer(0), sp.Integer(1)))
    for axis in axes:
        candidate = cross3(normal, axis)
        if sp.simplify(norm_sq(candidate)) != 0:
            return unit(candidate)
    raise DegenerateGeometryError("Cannot build plane basis from zero normal")
