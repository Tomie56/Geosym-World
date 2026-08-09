"""Coordinate fact closure for formal geometry states."""

from __future__ import annotations

from itertools import combinations, permutations
from typing import Dict, Tuple

import sympy as sp

from geosym_world.schema.facts import FactStore
from geosym_world.kernel.geometry2d.analytic import (
    angle_between,
    cross3,
    distance,
    dot,
    is_parallel,
    is_perpendicular,
    line_intersection_2d,
    line_parallel_to_plane,
    norm,
    point_on_line,
    point_on_plane,
    point_on_segment,
    polygon_area_3d,
    polygon_area_2d,
    polyhedron_volume_by_origin,
    sub,
    undirected_angle_between,
)
from geosym_world.schema.state import FormalGeometryState


class FactClosureSolver:
    """Derives exact coordinate-backed facts from a formal state."""

    PROFILES = {"minimal", "auto", "full"}

    def __init__(self, max_combinatorial_points: int = 20, *, profile: str = "full"):
        if profile not in self.PROFILES:
            raise ValueError(f"Unknown solver profile: {profile}")
        self.max_combinatorial_points = max_combinatorial_points
        self.profile = profile

    def solve(self, state: FormalGeometryState) -> FactStore:
        facts = FactStore()
        self._object_facts(state, facts)
        target_type = state.target.type if state.target else None
        include_pairwise = self.profile == "full" or target_type == "distance"
        self._metric_facts(state, facts, include_pairwise=include_pairwise)
        self._construction_relation_facts(state, facts)
        target_resolved = bool(state.target and facts.find(state.target.type, state.target.object))
        if self.profile == "full" or (self.profile == "auto" and not target_resolved):
            if not include_pairwise:
                self._pairwise_distance_facts(state, facts)
            self._segment_angle_facts(state, facts)
            self._topology_facts(state, facts)
            self._theorem_lite_facts(state, facts)
        self._target_fact(state, facts)
        return facts

    def _object_facts(self, state: FormalGeometryState, facts: FactStore) -> None:
        for point_id in state.points:
            facts.add("exists", point_id, source="constructed", rule="state_registry")
        for object_id in state.lines:
            facts.add("exists", object_id, source="constructed", rule="state_registry")
        for object_id in state.segments:
            facts.add("exists", object_id, source="constructed", rule="state_registry")
        for object_id in state.rays:
            facts.add("exists", object_id, source="constructed", rule="state_registry")
        for object_id in state.circles:
            facts.add("exists", object_id, source="constructed", rule="state_registry")
        for object_id in state.arcs:
            facts.add("exists", object_id, source="constructed", rule="state_registry")
        for object_id in state.polygons:
            facts.add("exists", object_id, source="constructed", rule="state_registry")
        for object_id in state.regions:
            facts.add("exists", object_id, source="constructed", rule="state_registry")
        for object_id in state.planes:
            facts.add("exists", object_id, source="constructed", rule="state_registry")
        for object_id in state.polyhedra:
            facts.add("exists", object_id, source="constructed", rule="state_registry")

    def _metric_facts(self, state: FormalGeometryState, facts: FactStore, *, include_pairwise: bool = True) -> None:
        if include_pairwise:
            self._pairwise_distance_facts(state, facts)

        for seg_id, seg in state.segments.items():
            a = state.require_point(seg.start)
            b = state.require_point(seg.end)
            facts.add("length", seg_id, distance(a.coords, b.coords), rule="distance_formula", dependencies=(f"point:{a.id}", f"point:{b.id}"))

        for circle_id, circle in state.circles.items():
            facts.add("radius", circle_id, circle.radius, rule="circle_definition", dependencies=(f"point:{circle.center}",))
            facts.add("circumference", circle_id, 2 * sp.pi * circle.radius, rule="circle_circumference_formula", dependencies=(f"circle:{circle_id}",))
            facts.add("area", circle_id, sp.pi * circle.radius ** 2, rule="circle_area_formula", dependencies=(f"circle:{circle_id}",))

        for arc_id, arc in state.arcs.items():
            circle = state.circles[arc.circle]
            if arc.central_angle is not None:
                facts.add("arc_length", arc_id, circle.radius * sp.Abs(arc.central_angle), rule="arc_length_formula", dependencies=(f"circle:{arc.circle}", f"arc:{arc_id}"))
                facts.add("sector_area", arc_id, circle.radius ** 2 * sp.Abs(arc.central_angle) / 2, rule="sector_area_formula", dependencies=(f"circle:{arc.circle}", f"arc:{arc_id}"))

        for polygon_id, polygon in state.polygons.items():
            points = [state.require_point(pid).coords for pid in polygon.vertices]
            perimeter = _closed_polyline_length(points)
            facts.add("perimeter", polygon_id, perimeter, rule="closed_polyline_perimeter", dependencies=tuple(f"point:{pid}" for pid in polygon.vertices))
            if points and len(points[0]) == 2:
                facts.add("area", polygon_id, polygon_area_2d(points), rule="shoelace_formula", dependencies=tuple(f"point:{pid}" for pid in polygon.vertices))
            if points and len(points[0]) == 3:
                facts.add("area", polygon_id, polygon_area_3d(points), rule="triangulated_3d_polygon_area", dependencies=tuple(f"point:{pid}" for pid in polygon.vertices))

        for region_id, region in state.regions.items():
            if region.kind == "polygon_region":
                if "polygon" in region.metadata and region.metadata["polygon"] in state.polygons:
                    polygon = state.polygons[region.metadata["polygon"]]
                    points = [state.require_point(pid).coords for pid in polygon.vertices]
                else:
                    points = [state.require_point(pid).coords for pid in region.boundary]
                facts.add("perimeter", region_id, _closed_polyline_length(points), rule="polygon_region_perimeter", dependencies=tuple(f"point:{pid}" for pid in region.boundary))
                if points and len(points[0]) == 2:
                    facts.add("area", region_id, polygon_area_2d(points), rule="polygon_region_area", dependencies=tuple(f"point:{pid}" for pid in region.boundary))
                elif points and len(points[0]) == 3:
                    facts.add("area", region_id, polygon_area_3d(points), rule="polygon_region_area_3d", dependencies=tuple(f"point:{pid}" for pid in region.boundary))
            elif region.kind == "sector_region":
                circle_id = region.metadata["circle"]
                circle = state.circles[circle_id]
                theta = sp.sympify(region.metadata["central_angle"])
                facts.add("area", region_id, circle.radius ** 2 * sp.Abs(theta) / 2, rule="sector_region_area", dependencies=(f"circle:{circle_id}",))
                facts.add("arc_length", region_id, circle.radius * sp.Abs(theta), rule="sector_region_arc_length", dependencies=(f"circle:{circle_id}",))
            elif region.kind == "annulus_region":
                inner_id = region.metadata["inner_circle"]
                outer_id = region.metadata["outer_circle"]
                inner = state.circles[inner_id]
                outer = state.circles[outer_id]
                facts.add(
                    "area",
                    region_id,
                    sp.pi * (outer.radius ** 2 - inner.radius ** 2),
                    rule="annulus_area_difference",
                    dependencies=(f"circle:{outer_id}", f"circle:{inner_id}"),
                )

        for poly_id, poly in state.polyhedra.items():
            vertex_coords = {pid: state.require_point(pid).coords for pid in poly.vertices}
            surface_area = sp.Integer(0)
            for face in poly.faces:
                points = [state.require_point(pid).coords for pid in face]
                surface_area += polygon_area_3d(points)
            facts.add("surface_area", poly_id, sp.simplify(surface_area), rule="polyhedron_face_area_sum", dependencies=tuple(f"point:{pid}" for pid in poly.vertices))
            facts.add("volume", poly_id, polyhedron_volume_by_origin(vertex_coords, poly.faces), rule="oriented_face_volume", dependencies=tuple(f"point:{pid}" for pid in poly.vertices))

    def _pairwise_distance_facts(self, state: FormalGeometryState, facts: FactStore) -> None:
        if len(state.points) > self.max_combinatorial_points:
            return
        for (left_id, left), (right_id, right) in combinations(state.points.items(), 2):
            if left.dim != right.dim:
                continue
            facts.add(
                "distance",
                f"{left_id}|{right_id}",
                distance(left.coords, right.coords),
                rule="point_distance_formula",
                dependencies=(f"point:{left_id}", f"point:{right_id}"),
            )

    def _segment_angle_facts(self, state: FormalGeometryState, facts: FactStore) -> None:
        segment_dirs: Dict[str, Tuple[str, str, tuple]] = {}
        for seg_id, seg in state.segments.items():
            a = state.require_point(seg.start)
            b = state.require_point(seg.end)
            segment_dirs[seg_id] = (seg.start, seg.end, sub(b.coords, a.coords))

        for left_id, right_id in combinations(segment_dirs.keys(), 2):
            a1, b1, d1 = segment_dirs[left_id]
            a2, b2, d2 = segment_dirs[right_id]
            shared = set((a1, b1)).intersection((a2, b2))
            if len(shared) == 1:
                vertex = next(iter(shared))
                p_left = b1 if a1 == vertex else a1
                p_right = b2 if a2 == vertex else a2
                v1 = sub(state.require_point(p_left).coords, state.require_point(vertex).coords)
                v2 = sub(state.require_point(p_right).coords, state.require_point(vertex).coords)
                angle_id = f"{p_left}{vertex}{p_right}"
                facts.add("angle", angle_id, angle_between(v1, v2), rule="vector_angle", dependencies=(f"segment:{left_id}", f"segment:{right_id}"))

    def _topology_facts(self, state: FormalGeometryState, facts: FactStore) -> None:
        line_like = {}
        for line_id, line in state.lines.items():
            line_like[line_id] = (line.point, line.direction)
        for ray_id, ray in state.rays.items():
            start = state.require_point(ray.start)
            line_like[ray_id] = (start.coords, ray.direction)
        for seg_id, seg in state.segments.items():
            a = state.require_point(seg.start)
            b = state.require_point(seg.end)
            line_like[seg_id] = (a.coords, sub(b.coords, a.coords))

        for left_id, right_id in combinations(line_like.keys(), 2):
            left_anchor, left = line_like[left_id]
            right_anchor, right = line_like[right_id]
            pair_id = f"{left_id}|{right_id}"
            if len(left) != len(right):
                continue
            facts.add(
                "angle",
                pair_id,
                undirected_angle_between(left, right),
                source="derived",
                rule="line_like_undirected_angle",
                dependencies=(left_id, right_id),
            )
            if len(left) == 3:
                facts.add(
                    "space_angle",
                    pair_id,
                    undirected_angle_between(left, right),
                    source="derived",
                    rule="space_line_line_angle",
                    dependencies=(left_id, right_id),
                    metadata={"kind": "line_line"},
                )
            if is_parallel(left, right):
                facts.add("parallel", pair_id, source="derived", rule="direction_cross_zero", dependencies=(left_id, right_id))
            elif len(left) == 2:
                intersection = line_intersection_2d(left_anchor, left, right_anchor, right)
                facts.add(
                    "line_intersection",
                    pair_id,
                    source="derived",
                    rule="line_line_intersection_formula",
                    dependencies=(left_id, right_id),
                    metadata={"coords": [str(value) for value in intersection]},
                )
            if is_perpendicular(left, right):
                facts.add("perpendicular", pair_id, source="derived", rule="direction_dot_zero", dependencies=(left_id, right_id))

        if len(state.points) <= self.max_combinatorial_points:
            point_items = list(state.points.items())
            for point_id, point in point_items:
                for line_id, (anchor, direction) in line_like.items():
                    if len(point.coords) != len(anchor):
                        continue
                    if point_on_line(point.coords, anchor, direction):
                        facts.add("point_on_line", f"{point_id}|{line_id}", source="derived", rule="collinearity_cross_zero", dependencies=(f"point:{point_id}", line_id))
                for seg_id, seg in state.segments.items():
                    a = state.require_point(seg.start)
                    b = state.require_point(seg.end)
                    if len(point.coords) != len(a.coords):
                        continue
                    if point_on_segment(point.coords, a.coords, b.coords):
                        facts.add("point_on_segment", f"{point_id}|{seg_id}", source="derived", rule="segment_bounds", dependencies=(f"point:{point_id}", f"segment:{seg_id}"))
                for circle_id, circle in state.circles.items():
                    center = state.require_point(circle.center)
                    if len(point.coords) != len(center.coords):
                        continue
                    if sp.simplify(distance(point.coords, center.coords) - circle.radius) == 0:
                        facts.add("point_on_circle", f"{point_id}|{circle_id}", source="derived", rule="distance_equals_radius", dependencies=(f"point:{point_id}", f"circle:{circle_id}"))

            for (a_id, a), (b_id, b), (c_id, c) in combinations(point_items, 3):
                if len(a.coords) != len(b.coords) or len(a.coords) != len(c.coords):
                    continue
                if point_on_line(c.coords, a.coords, sub(b.coords, a.coords)):
                    facts.add("collinear", f"{a_id}{b_id}{c_id}", source="derived", rule="three_point_collinearity", dependencies=(f"point:{a_id}", f"point:{b_id}", f"point:{c_id}"))

        self._plane_topology_facts(state, facts, line_like)

    def _plane_topology_facts(self, state: FormalGeometryState, facts: FactStore, line_like: Dict[str, Tuple[tuple, tuple]]) -> None:
        if not state.planes:
            return

        if len(state.points) <= self.max_combinatorial_points:
            for point_id, point in state.points.items():
                if point.dim != 3:
                    continue
                for plane_id, plane in state.planes.items():
                    if point_on_plane(point.coords, plane.point, plane.normal):
                        facts.add(
                            "point_on_plane",
                            f"{point_id}|{plane_id}",
                            source="derived",
                            rule="plane_equation_zero",
                            dependencies=(f"point:{point_id}", f"plane:{plane_id}"),
                        )

        for line_id, (anchor, direction) in line_like.items():
            if len(anchor) != 3 or len(direction) != 3:
                continue
            for plane_id, plane in state.planes.items():
                pair_id = f"{line_id}|{plane_id}"
                if line_parallel_to_plane(direction, plane.normal):
                    facts.add("line_parallel_plane", pair_id, source="derived", rule="line_direction_dot_plane_normal_zero", dependencies=(line_id, f"plane:{plane_id}"))
                    if point_on_plane(anchor, plane.point, plane.normal):
                        facts.add("line_on_plane", pair_id, source="derived", rule="line_parallel_plane_and_anchor_on_plane", dependencies=(line_id, f"plane:{plane_id}"))
                if is_parallel(direction, plane.normal):
                    facts.add("line_perpendicular_plane", pair_id, source="derived", rule="line_direction_parallel_plane_normal", dependencies=(line_id, f"plane:{plane_id}"))
                normal_angle = undirected_angle_between(direction, plane.normal)
                facts.add(
                    "line_plane_angle",
                    pair_id,
                    sp.simplify(sp.pi / 2 - normal_angle),
                    source="derived",
                    rule="line_plane_angle_from_normal",
                    dependencies=(line_id, f"plane:{plane_id}"),
                )
                facts.add(
                    "space_angle",
                    pair_id,
                    sp.simplify(sp.pi / 2 - normal_angle),
                    source="derived",
                    rule="space_line_plane_angle",
                    dependencies=(line_id, f"plane:{plane_id}"),
                    metadata={"kind": "line_plane"},
                )
                line_plane_distance = sp.Integer(0)
                if line_parallel_to_plane(direction, plane.normal):
                    line_plane_distance = sp.simplify(sp.Abs(dot(sub(anchor, plane.point), plane.normal)) / norm(plane.normal))
                facts.add(
                    "line_plane_distance",
                    pair_id,
                    line_plane_distance,
                    source="derived",
                    rule="line_plane_distance",
                    dependencies=(line_id, f"plane:{plane_id}"),
                )
                facts.add(
                    "line_plane_normal_dot",
                    pair_id,
                    dot(direction, plane.normal),
                    source="derived",
                    rule="line_plane_direction_normal_dot",
                    dependencies=(line_id, f"plane:{plane_id}"),
                )

        for left_id, right_id in combinations(state.planes.keys(), 2):
            left = state.planes[left_id]
            right = state.planes[right_id]
            pair_id = f"{left_id}|{right_id}"
            facts.add(
                "angle",
                pair_id,
                undirected_angle_between(left.normal, right.normal),
                source="derived",
                rule="plane_normal_undirected_angle",
                dependencies=(f"plane:{left_id}", f"plane:{right_id}"),
            )
            facts.add(
                "space_angle",
                pair_id,
                undirected_angle_between(left.normal, right.normal),
                source="derived",
                rule="space_plane_plane_angle",
                dependencies=(f"plane:{left_id}", f"plane:{right_id}"),
                metadata={"kind": "plane_plane"},
            )
            if is_parallel(left.normal, right.normal):
                facts.add("plane_parallel", pair_id, source="derived", rule="plane_normals_parallel", dependencies=(f"plane:{left_id}", f"plane:{right_id}"))
            if is_perpendicular(left.normal, right.normal):
                facts.add("plane_perpendicular", pair_id, source="derived", rule="plane_normals_perpendicular", dependencies=(f"plane:{left_id}", f"plane:{right_id}"))

    def _construction_relation_facts(self, state: FormalGeometryState, facts: FactStore) -> None:
        for index, relation in enumerate(state.relations):
            rel_type = str(relation.get("type", "relation"))
            object_id = _relation_object_id(relation, index)
            facts.add(rel_type, object_id, source="constructed", rule="construction_trace", metadata=relation)
            if rel_type == "midpoint":
                mid = relation["point"]
                a, b = relation["of"]
                facts.add("equal_length", f"{a}{mid}|{mid}{b}", source="derived", rule="midpoint_definition", dependencies=(f"point:{mid}", f"point:{a}", f"point:{b}"))
            if rel_type == "angle_bisector":
                line = relation["line"]
                a, vertex, c = relation["angle"]
                left_angle_id = f"{a}{vertex}|{line}"
                right_angle_id = f"{line}|{vertex}{c}"
                facts.add("equal_angle", f"{left_angle_id}|{right_angle_id}", source="derived", rule="angle_bisector_definition", dependencies=(f"line:{line}", f"point:{vertex}"))
                if line in state.lines and a in state.points and vertex in state.points and c in state.points:
                    bisector = state.lines[line]
                    vertex_point = state.require_point(vertex)
                    left_vec = sub(state.require_point(a).coords, vertex_point.coords)
                    right_vec = sub(state.require_point(c).coords, vertex_point.coords)
                    left_angle = angle_between(left_vec, bisector.direction)
                    right_angle = angle_between(bisector.direction, right_vec)
                    facts.add(
                        "angle",
                        left_angle_id,
                        left_angle,
                        source="derived",
                        rule="angle_bisector_left_angle",
                        dependencies=(f"point:{a}", f"point:{vertex}", f"line:{line}"),
                    )
                    facts.add(
                        "angle",
                        right_angle_id,
                        right_angle,
                        source="derived",
                        rule="angle_bisector_right_angle",
                        dependencies=(f"line:{line}", f"point:{vertex}", f"point:{c}"),
                    )
                    facts.add(
                        "angle_bisector_residual",
                        line,
                        sp.simplify(left_angle - right_angle),
                        source="derived",
                        rule="angle_bisector_numeric_check",
                        dependencies=(f"point:{a}", f"point:{vertex}", f"point:{c}", f"line:{line}"),
                        metadata={"expected_zero": True, "left_angle": str(left_angle), "right_angle": str(right_angle), "mode": relation.get("mode")},
                    )
            if rel_type == "chord":
                chord = relation["object"]
                circle = relation["circle"]
                facts.add("chord_of_circle", f"{chord}|{circle}", source="constructed", rule="chord_definition", metadata=relation)
            if rel_type == "tangent":
                point = relation["point"]
                circle = relation["circle"]
                facts.add("tangent_to_circle", f"{point}|{circle}", source="constructed", rule="tangent_construction", metadata=relation)
            if rel_type == "point_projection_plane":
                self._point_projection_relation_facts(state, facts, relation)
            if rel_type == "line_projection_plane":
                self._line_projection_relation_facts(state, facts, relation)
            if rel_type == "dihedral_angle":
                self._dihedral_relation_facts(state, facts, relation)
            if rel_type == "polyhedron_plane_section":
                self._section_relation_facts(state, facts, relation)

    def _point_projection_relation_facts(self, state: FormalGeometryState, facts: FactStore, relation: dict) -> None:
        point_id = relation["point"]
        source_id = relation["from"]
        plane_id = relation["plane"]
        if point_id not in state.points or source_id not in state.points or plane_id not in state.planes:
            return
        projected = state.require_point(point_id)
        source = state.require_point(source_id)
        plane = state.planes[plane_id]
        residual = dot(sub(projected.coords, plane.point), plane.normal)
        facts.add(
            "point_projection_plane",
            f"{source_id}->{point_id}|{plane_id}",
            source="derived",
            rule="orthogonal_projection_to_plane",
            dependencies=(f"point:{source_id}", f"point:{point_id}", f"plane:{plane_id}"),
            metadata={"projected_point": point_id},
        )
        facts.add(
            "projection_distance",
            f"{source_id}->{point_id}",
            distance(source.coords, projected.coords),
            source="derived",
            rule="point_plane_projection_distance",
            dependencies=(f"point:{source_id}", f"point:{point_id}"),
            metadata={"plane": plane_id},
        )
        facts.add(
            "projection_plane_residual",
            f"{point_id}|{plane_id}",
            residual,
            source="derived",
            rule="projected_point_plane_equation_check",
            dependencies=(f"point:{point_id}", f"plane:{plane_id}"),
            metadata={"expected_zero": True},
        )

    def _line_projection_relation_facts(self, state: FormalGeometryState, facts: FactStore, relation: dict) -> None:
        line_id = relation["line"]
        source_id = relation["from"]
        plane_id = relation["plane"]
        if line_id not in state.lines or plane_id not in state.planes:
            return
        projected = state.lines[line_id]
        plane = state.planes[plane_id]
        anchor_residual = dot(sub(projected.point, plane.point), plane.normal)
        direction_residual = dot(projected.direction, plane.normal)
        facts.add(
            "line_projection_plane",
            f"{source_id}->{line_id}|{plane_id}",
            source="derived",
            rule="line_projection_to_plane",
            dependencies=(source_id, line_id, f"plane:{plane_id}"),
            metadata={"projected_line": line_id},
        )
        facts.add(
            "line_projection_plane_residual",
            f"{line_id}|{plane_id}",
            sp.simplify(anchor_residual ** 2 + direction_residual ** 2),
            source="derived",
            rule="projected_line_plane_check",
            dependencies=(line_id, f"plane:{plane_id}"),
            metadata={"expected_zero": True, "anchor_residual": str(anchor_residual), "direction_residual": str(direction_residual)},
        )

    def _dihedral_relation_facts(self, state: FormalGeometryState, facts: FactStore, relation: dict) -> None:
        object_id = relation["object"]
        normal_pair = _dihedral_normals_from_relation(state, relation)
        if normal_pair is None:
            return
        left_normal, right_normal = normal_pair
        angle = undirected_angle_between(left_normal, right_normal)
        shared_edge = _dihedral_shared_edge_from_relation(state, relation)
        metadata = {key: relation[key] for key in ("planes", "polyhedron", "faces") if key in relation}
        if shared_edge:
            metadata["shared_edge"] = list(shared_edge)
        facts.add(
            "dihedral_angle",
            object_id,
            angle,
            source="derived",
            rule="dihedral_angle_from_normals",
            dependencies=tuple(_dihedral_dependencies(relation)),
            metadata=metadata,
        )
        if shared_edge:
            facts.add(
                "dihedral_shared_edge",
                object_id,
                source="derived",
                rule="polyhedron_faces_shared_edge",
                dependencies=tuple(_dihedral_dependencies(relation)),
                metadata={"shared_edge": list(shared_edge)},
            )

    def _section_relation_facts(self, state: FormalGeometryState, facts: FactStore, relation: dict) -> None:
        object_id = relation["object"]
        polyhedron_id = relation["polyhedron"]
        plane_id = relation["plane"]
        boundary = tuple(relation.get("boundary") or ())
        polygon_id = relation.get("polygon")
        if polyhedron_id not in state.polyhedra or plane_id not in state.planes:
            return
        facts.add(
            "section_interface",
            object_id,
            source="derived",
            rule="polyhedron_plane_section_interface",
            dependencies=(f"polyhedron:{polyhedron_id}", f"plane:{plane_id}"),
            metadata={"boundary": list(boundary), "polygon": polygon_id},
        )
        if polygon_id and polygon_id in state.polygons:
            polygon = state.polygons[polygon_id]
            points = [state.require_point(point_id).coords for point_id in polygon.vertices]
            area = polygon_area_3d(points)
            facts.add(
                "section_polygon",
                object_id,
                source="derived",
                rule="section_polygon_materialized",
                dependencies=(f"polygon:{polygon_id}", f"polyhedron:{polyhedron_id}", f"plane:{plane_id}"),
                metadata={"polygon": polygon_id},
            )
            facts.add(
                "section_area",
                object_id,
                area,
                source="derived",
                rule="section_polygon_area",
                dependencies=tuple(f"point:{point_id}" for point_id in polygon.vertices),
                metadata={"polygon": polygon_id},
            )
            facts.add(
                "section_perimeter",
                object_id,
                _closed_polyline_length(points),
                source="derived",
                rule="section_polygon_perimeter",
                dependencies=tuple(f"point:{point_id}" for point_id in polygon.vertices),
                metadata={"polygon": polygon_id},
            )
            facts.add(
                "section_degeneracy_measure",
                object_id,
                area,
                source="derived",
                rule="section_area_nondegeneracy_measure",
                dependencies=tuple(f"point:{point_id}" for point_id in polygon.vertices),
                metadata={"expected_nonzero": True, "polygon": polygon_id},
            )
        facts.add(
            "section_boundary_size",
            object_id,
            len(boundary),
            source="derived",
            rule="section_boundary_count",
            dependencies=tuple(f"point:{point_id}" for point_id in boundary),
            metadata={"polyhedron": polyhedron_id, "plane": plane_id},
        )
        if boundary:
            plane = state.planes[plane_id]
            residual = sp.Integer(0)
            for point_id in boundary:
                if point_id not in state.points:
                    return
                residual += dot(sub(state.require_point(point_id).coords, plane.point), plane.normal) ** 2
            facts.add(
                "section_boundary_plane_residual",
                object_id,
                sp.simplify(residual),
                source="derived",
                rule="section_boundary_on_plane_check",
                dependencies=tuple(f"point:{point_id}" for point_id in boundary) + (f"plane:{plane_id}",),
                metadata={"expected_zero": True},
            )

    def _theorem_lite_facts(self, state: FormalGeometryState, facts: FactStore) -> None:
        self._right_triangle_facts(state, facts)
        self._isosceles_triangle_facts(state, facts)
        self._rhombus_facts(state, facts)
        self._cyclic_quadrilateral_facts(state, facts)
        self._tangential_quadrilateral_facts(state, facts)
        self._similar_triangle_facts(state, facts)
        self._aa_similarity_facts(state, facts)
        self._parallel_cut_similarity_facts(state, facts)
        self._circle_theorem_lite_facts(state, facts)
        self._inscribed_angle_facts(state, facts)
        self._tangent_chord_angle_facts(state, facts)

    def _right_triangle_facts(self, state: FormalGeometryState, facts: FactStore) -> None:
        for polygon_id, polygon in state.polygons.items():
            if len(polygon.vertices) != 3:
                continue
            vertices = list(polygon.vertices)
            points = {pid: state.require_point(pid).coords for pid in vertices}
            if any(len(point) != 2 for point in points.values()):
                continue
            for vertex in vertices:
                others = [pid for pid in vertices if pid != vertex]
                v1 = sub(points[others[0]], points[vertex])
                v2 = sub(points[others[1]], points[vertex])
                if is_perpendicular(v1, v2):
                    hyp = "".join(sorted(others))
                    legs = [f"{vertex}{others[0]}", f"{vertex}{others[1]}"]
                    facts.add(
                        "right_triangle",
                        polygon_id,
                        source="derived",
                        rule="perpendicular_sides",
                        dependencies=tuple(f"point:{pid}" for pid in vertices),
                        metadata={"right_angle_vertex": vertex, "legs": legs, "hypotenuse_points": others, "hypotenuse_key": hyp},
                    )
                    a = distance(points[vertex], points[others[0]])
                    b = distance(points[vertex], points[others[1]])
                    c = distance(points[others[0]], points[others[1]])
                    facts.add(
                        "pythagorean_relation",
                        polygon_id,
                        sp.simplify(c ** 2 - a ** 2 - b ** 2),
                        source="derived",
                        rule="right_triangle_pythagorean_check",
                        dependencies=tuple(f"point:{pid}" for pid in vertices),
                        metadata={"expected_zero": True, "right_angle_vertex": vertex},
                    )

    def _isosceles_triangle_facts(self, state: FormalGeometryState, facts: FactStore) -> None:
        for polygon_id, polygon in state.polygons.items():
            if len(polygon.vertices) != 3:
                continue
            vertices = list(polygon.vertices)
            points = {pid: state.require_point(pid).coords for pid in vertices}
            if any(len(point) != 2 for point in points.values()):
                continue
            for apex in vertices:
                base_left, base_right = [pid for pid in vertices if pid != apex]
                left_len = distance(points[apex], points[base_left])
                right_len = distance(points[apex], points[base_right])
                if sp.simplify(left_len - right_len) != 0:
                    continue
                object_id = f"{polygon_id}|apex:{apex}"
                left_base_angle = angle_between(sub(points[apex], points[base_left]), sub(points[base_right], points[base_left]))
                right_base_angle = angle_between(sub(points[apex], points[base_right]), sub(points[base_left], points[base_right]))
                facts.add(
                    "isosceles_triangle",
                    object_id,
                    source="derived",
                    rule="equal_sides",
                    dependencies=tuple(f"point:{pid}" for pid in vertices),
                    metadata={
                        "polygon": polygon_id,
                        "apex": apex,
                        "base": [base_left, base_right],
                        "equal_sides": [f"{apex}{base_left}", f"{apex}{base_right}"],
                        "side_length": str(left_len),
                    },
                )
                facts.add(
                    "equal_base_angles",
                    object_id,
                    source="derived",
                    rule="isosceles_triangle_base_angles",
                    dependencies=tuple(f"point:{pid}" for pid in vertices),
                    metadata={
                        "left_base_angle": f"{apex}{base_left}{base_right}",
                        "right_base_angle": f"{apex}{base_right}{base_left}",
                    },
                )
                facts.add(
                    "equal_base_angle_residual",
                    object_id,
                    sp.simplify(left_base_angle - right_base_angle),
                    source="derived",
                    rule="isosceles_triangle_base_angle_check",
                    dependencies=tuple(f"point:{pid}" for pid in vertices),
                    metadata={
                        "expected_zero": True,
                        "left_base_angle": str(left_base_angle),
                        "right_base_angle": str(right_base_angle),
                    },
                )

    def _rhombus_facts(self, state: FormalGeometryState, facts: FactStore) -> None:
        for polygon_id, polygon in state.polygons.items():
            if len(polygon.vertices) != 4:
                continue
            vertices = tuple(polygon.vertices)
            points = {point_id: state.require_point(point_id).coords for point_id in vertices}
            if any(len(point) != 2 for point in points.values()):
                continue
            side_lengths = [
                distance(points[vertices[index]], points[vertices[(index + 1) % 4]])
                for index in range(4)
            ]
            equal_side_residual = sp.simplify(sum((length - side_lengths[0]) ** 2 for length in side_lengths[1:]))
            if equal_side_residual != 0:
                continue
            first_diagonal = distance(points[vertices[0]], points[vertices[2]])
            second_diagonal = distance(points[vertices[1]], points[vertices[3]])
            area = polygon_area_2d([points[point_id] for point_id in vertices])
            dependencies = tuple(f"point:{point_id}" for point_id in vertices)
            facts.add(
                "rhombus",
                polygon_id,
                source="derived",
                rule="four_equal_sides",
                dependencies=dependencies,
                metadata={"vertices": list(vertices), "side_length": str(side_lengths[0])},
            )
            facts.add(
                "rhombus_equal_side_residual",
                polygon_id,
                equal_side_residual,
                rule="rhombus_equal_side_check",
                dependencies=dependencies,
                metadata={"expected_zero": True},
            )
            facts.add(
                "rhombus_diagonal_area_residual",
                polygon_id,
                sp.simplify(area - first_diagonal * second_diagonal / 2),
                rule="rhombus_diagonal_area_formula",
                dependencies=dependencies,
                metadata={"expected_zero": True, "diagonals": [f"{vertices[0]}{vertices[2]}", f"{vertices[1]}{vertices[3]}"], "area": str(area)},
            )

    def _cyclic_quadrilateral_facts(self, state: FormalGeometryState, facts: FactStore) -> None:
        for polygon_id, polygon in state.polygons.items():
            if len(polygon.vertices) != 4:
                continue
            vertices = tuple(polygon.vertices)
            points = [state.require_point(point_id).coords for point_id in vertices]
            if any(len(point) != 2 for point in points) or sp.simplify(polygon_area_2d(points)) == 0:
                continue
            determinant = sp.simplify(
                sp.Matrix([[point[0], point[1], point[0] ** 2 + point[1] ** 2, 1] for point in points]).det()
            )
            if determinant != 0:
                continue
            dependencies = tuple(f"point:{point_id}" for point_id in vertices)
            facts.add(
                "cyclic_quadrilateral",
                polygon_id,
                source="derived",
                rule="concyclic_determinant_zero",
                dependencies=dependencies,
                metadata={"vertices": list(vertices), "circle": polygon.metadata.get("circle")},
            )
            facts.add(
                "cyclicity_residual",
                polygon_id,
                determinant,
                rule="concyclic_determinant_check",
                dependencies=dependencies,
                metadata={"expected_zero": True},
            )

    def _tangential_quadrilateral_facts(self, state: FormalGeometryState, facts: FactStore) -> None:
        for polygon_id, polygon in state.polygons.items():
            if len(polygon.vertices) != 4:
                continue
            circle_id = polygon.metadata.get("incircle")
            if circle_id not in state.circles:
                continue
            circle = state.circles[circle_id]
            center = state.require_point(circle.center).coords
            vertices = tuple(polygon.vertices)
            points = {point_id: state.require_point(point_id).coords for point_id in vertices}
            if len(center) != 2 or any(len(point) != 2 for point in points.values()):
                continue
            side_lengths = []
            tangent_sides = []
            for index in range(4):
                start_id = vertices[index]
                end_id = vertices[(index + 1) % 4]
                start = points[start_id]
                end = points[end_id]
                delta_x = sp.simplify(end[0] - start[0])
                delta_y = sp.simplify(end[1] - start[1])
                length_squared = sp.simplify(delta_x ** 2 + delta_y ** 2)
                cross_value = sp.simplify(delta_x * (center[1] - start[1]) - delta_y * (center[0] - start[0]))
                if sp.simplify(cross_value ** 2 - circle.radius ** 2 * length_squared) != 0:
                    break
                side_lengths.append(sp.sqrt(length_squared))
                tangent_sides.append(f"{start_id}{end_id}")
            if len(tangent_sides) != 4:
                continue
            dependencies = tuple(f"point:{point_id}" for point_id in vertices) + (f"circle:{circle_id}",)
            pitot_residual = sp.simplify(side_lengths[0] + side_lengths[2] - side_lengths[1] - side_lengths[3])
            facts.add(
                "tangential_quadrilateral",
                polygon_id,
                source="derived",
                rule="four_sides_tangent_to_one_circle",
                dependencies=dependencies,
                metadata={"vertices": list(vertices), "incircle": circle_id, "tangent_sides": tangent_sides},
            )
            for side in tangent_sides:
                facts.add(
                    "tangent_to_circle",
                    f"{side}|{circle_id}",
                    source="derived",
                    rule="point_line_distance_equals_radius",
                    dependencies=dependencies,
                )
            facts.add(
                "pitot_residual",
                polygon_id,
                pitot_residual,
                rule="pitot_tangential_quadrilateral",
                dependencies=dependencies,
                metadata={"expected_zero": True, "opposite_side_sums": [[tangent_sides[0], tangent_sides[2]], [tangent_sides[1], tangent_sides[3]]]},
            )

    def _similar_triangle_facts(self, state: FormalGeometryState, facts: FactStore) -> None:
        triangle_items = []
        for polygon_id, polygon in state.polygons.items():
            if len(polygon.vertices) != 3:
                continue
            vertices = tuple(polygon.vertices)
            points = {pid: state.require_point(pid).coords for pid in vertices}
            if any(len(point) != 2 for point in points.values()):
                continue
            if sp.simplify(polygon_area_2d([points[pid] for pid in vertices])) == 0:
                continue
            triangle_items.append((polygon_id, vertices, points))

        for left, right in combinations(triangle_items, 2):
            left_id, left_vertices, left_points = left
            right_id, right_vertices, right_points = right
            left_sides = _triangle_side_squares(left_points, left_vertices)
            if any(sp.simplify(side) == 0 for side in left_sides):
                continue
            for right_perm in permutations(right_vertices):
                right_sides = _triangle_side_squares(right_points, right_perm)
                if any(sp.simplify(side) == 0 for side in right_sides):
                    continue
                residuals = [sp.simplify(left_sides[idx] * right_sides[0] - left_sides[0] * right_sides[idx]) for idx in range(1, 3)]
                if any(residual != 0 for residual in residuals):
                    continue

                mapping = dict(zip(left_vertices, right_perm))
                object_id = _similarity_object_id(left_id, right_id, left_vertices, right_perm)
                scale_squared = sp.simplify(right_sides[0] / left_sides[0])
                facts.add(
                    "similar_triangle",
                    object_id,
                    source="derived",
                    rule="sss_similarity",
                    dependencies=tuple(f"point:{pid}" for pid in (*left_vertices, *right_perm)),
                    metadata={
                        "left_polygon": left_id,
                        "right_polygon": right_id,
                        "mapping": mapping,
                        "scale_squared": str(scale_squared),
                    },
                )
                facts.add(
                    "similarity_ratio_residual",
                    object_id,
                    sp.simplify(sum(residual ** 2 for residual in residuals)),
                    source="derived",
                    rule="sss_similarity_ratio_check",
                    dependencies=tuple(f"point:{pid}" for pid in (*left_vertices, *right_perm)),
                    metadata={"expected_zero": True, "side_square_residuals": [str(residual) for residual in residuals]},
                )
                facts.add(
                    "similarity_scale_squared",
                    object_id,
                    scale_squared,
                    source="derived",
                    rule="sss_similarity_scale",
                    dependencies=tuple(f"point:{pid}" for pid in (*left_vertices, *right_perm)),
                    metadata={"left_polygon": left_id, "right_polygon": right_id},
                )

                angle_residuals = []
                for left_vertex, right_vertex in zip(left_vertices, right_perm):
                    left_others = [pid for pid in left_vertices if pid != left_vertex]
                    right_others = [pid for pid in right_perm if pid != right_vertex]
                    left_angle = angle_between(
                        sub(left_points[left_others[0]], left_points[left_vertex]),
                        sub(left_points[left_others[1]], left_points[left_vertex]),
                    )
                    right_angle = angle_between(
                        sub(right_points[right_others[0]], right_points[right_vertex]),
                        sub(right_points[right_others[1]], right_points[right_vertex]),
                    )
                    angle_residuals.append(sp.simplify(left_angle - right_angle))
                facts.add(
                    "equal_corresponding_angles",
                    object_id,
                    source="derived",
                    rule="similar_triangles_corresponding_angles",
                    dependencies=tuple(f"point:{pid}" for pid in (*left_vertices, *right_perm)),
                    metadata={"mapping": mapping},
                )
                facts.add(
                    "similar_angle_residual",
                    object_id,
                    sp.simplify(sum(residual ** 2 for residual in angle_residuals)),
                    source="derived",
                    rule="similar_triangle_angle_check",
                    dependencies=tuple(f"point:{pid}" for pid in (*left_vertices, *right_perm)),
                    metadata={"expected_zero": True, "angle_residuals": [str(residual) for residual in angle_residuals]},
                )
                break

    def _aa_similarity_facts(self, state: FormalGeometryState, facts: FactStore) -> None:
        triangle_items = _triangle_items_2d(state)
        for left, right in combinations(triangle_items, 2):
            left_id, left_vertices, left_points = left
            right_id, right_vertices, right_points = right
            left_angles = _triangle_angles(left_points, left_vertices)
            for right_perm in permutations(right_vertices):
                right_angles = _triangle_angles(right_points, right_perm)
                angle_residuals = [sp.simplify(left_angles[idx] - right_angles[idx]) for idx in range(3)]
                if sum(1 for residual in angle_residuals if residual == 0) < 2:
                    continue
                object_id = _similarity_object_id(left_id, right_id, left_vertices, right_perm)
                mapping = dict(zip(left_vertices, right_perm))
                facts.add(
                    "aa_similarity",
                    object_id,
                    source="derived",
                    rule="aa_similarity",
                    dependencies=tuple(f"point:{pid}" for pid in (*left_vertices, *right_perm)),
                    metadata={"left_polygon": left_id, "right_polygon": right_id, "mapping": mapping},
                )
                facts.add(
                    "aa_similarity_residual",
                    object_id,
                    sp.simplify(sum(residual ** 2 for residual in angle_residuals)),
                    source="derived",
                    rule="aa_similarity_angle_check",
                    dependencies=tuple(f"point:{pid}" for pid in (*left_vertices, *right_perm)),
                    metadata={"expected_zero": True, "angle_residuals": [str(residual) for residual in angle_residuals]},
                )
                facts.add(
                    "similar_triangle",
                    object_id,
                    source="derived",
                    rule="aa_similarity",
                    dependencies=tuple(f"point:{pid}" for pid in (*left_vertices, *right_perm)),
                    metadata={"left_polygon": left_id, "right_polygon": right_id, "mapping": mapping},
                )
                break

    def _parallel_cut_similarity_facts(self, state: FormalGeometryState, facts: FactStore) -> None:
        triangle_items = _triangle_items_2d(state)
        for big_id, big_vertices, big_points in triangle_items:
            for small_id, small_vertices, small_points in triangle_items:
                if big_id == small_id:
                    continue
                shared = set(big_vertices).intersection(small_vertices)
                if len(shared) != 1:
                    continue
                apex = next(iter(shared))
                big_base = [pid for pid in big_vertices if pid != apex]
                small_cut = [pid for pid in small_vertices if pid != apex]
                found = False
                for base_left, base_right in (tuple(big_base), tuple(reversed(big_base))):
                    for cut_left, cut_right in (tuple(small_cut), tuple(reversed(small_cut))):
                        if not point_on_segment(small_points[cut_left], big_points[apex], big_points[base_left]):
                            continue
                        if not point_on_segment(small_points[cut_right], big_points[apex], big_points[base_right]):
                            continue
                        cut_vec = sub(small_points[cut_right], small_points[cut_left])
                        base_vec = sub(big_points[base_right], big_points[base_left])
                        if not is_parallel(cut_vec, base_vec):
                            continue
                        object_id = _parallel_cut_object_id(big_id, small_id, apex, cut_left, cut_right, base_left, base_right)
                        left_small = _squared_distance(big_points[apex], small_points[cut_left])
                        left_big = _squared_distance(big_points[apex], big_points[base_left])
                        right_small = _squared_distance(big_points[apex], small_points[cut_right])
                        right_big = _squared_distance(big_points[apex], big_points[base_right])
                        residual = sp.simplify(left_small * right_big - right_small * left_big)
                        mapping = {apex: apex, base_left: cut_left, base_right: cut_right}
                        facts.add(
                            "parallel_cut_similarity",
                            object_id,
                            source="derived",
                            rule="parallel_cut_similarity",
                            dependencies=tuple(f"point:{pid}" for pid in (apex, base_left, base_right, cut_left, cut_right)),
                            metadata={
                                "big_polygon": big_id,
                                "small_polygon": small_id,
                                "apex": apex,
                                "base": [base_left, base_right],
                                "cut": [cut_left, cut_right],
                                "mapping": mapping,
                            },
                        )
                        facts.add(
                            "parallel_cut_ratio_residual",
                            object_id,
                            residual,
                            source="derived",
                            rule="parallel_cut_ratio_check",
                            dependencies=tuple(f"point:{pid}" for pid in (apex, base_left, base_right, cut_left, cut_right)),
                            metadata={"expected_zero": True, "ratio": "apex_cut/apex_base"},
                        )
                        facts.add(
                            "similar_triangle",
                            object_id,
                            source="derived",
                            rule="parallel_cut_similarity",
                            dependencies=tuple(f"point:{pid}" for pid in (apex, base_left, base_right, cut_left, cut_right)),
                            metadata={"big_polygon": big_id, "small_polygon": small_id, "mapping": mapping},
                        )
                        found = True
                        break
                    if found:
                        break

    def _circle_theorem_lite_facts(self, state: FormalGeometryState, facts: FactStore) -> None:
        chord_items = []
        for relation in state.relations:
            if relation.get("type") != "chord":
                continue
            chord_id = relation["object"]
            circle_id = relation["circle"]
            if chord_id not in state.segments or circle_id not in state.circles:
                continue
            seg = state.segments[chord_id]
            circle = state.circles[circle_id]
            center = state.require_point(circle.center)
            a = state.require_point(seg.start)
            b = state.require_point(seg.end)
            if len(center.coords) != 2 or len(a.coords) != 2 or len(b.coords) != 2:
                continue
            v1 = sub(a.coords, center.coords)
            v2 = sub(b.coords, center.coords)
            theta = angle_between(v1, v2)
            chord_len = distance(a.coords, b.coords)
            facts.add(
                "central_angle",
                f"{seg.start}{circle.center}{seg.end}",
                theta,
                source="derived",
                rule="chord_central_angle",
                dependencies=(f"point:{seg.start}", f"point:{circle.center}", f"point:{seg.end}", f"circle:{circle_id}"),
                metadata={"chord": chord_id, "circle": circle_id},
            )
            facts.add(
                "chord_length_formula_check",
                chord_id,
                sp.trigsimp(sp.simplify(chord_len ** 2 - 2 * circle.radius ** 2 * (1 - sp.cos(theta)))),
                source="derived",
                rule="chord_length_formula",
                dependencies=(f"segment:{chord_id}", f"circle:{circle_id}"),
                metadata={"expected_zero": True, "central_angle": str(theta), "formula": "chord^2 = 2*r^2*(1-cos(theta))"},
            )
            chord_items.append((chord_id, circle_id, chord_len, theta))

        for left, right in combinations(chord_items, 2):
            left_chord, left_circle, left_len, left_theta = left
            right_chord, right_circle, right_len, right_theta = right
            if left_circle != right_circle:
                continue
            if sp.simplify(left_len - right_len) == 0:
                facts.add(
                    "equal_chord_equal_central_angle",
                    f"{left_chord}|{right_chord}",
                    sp.simplify(left_theta - right_theta),
                    source="derived",
                    rule="equal_chords_subtend_equal_angles",
                    dependencies=(f"segment:{left_chord}", f"segment:{right_chord}", f"circle:{left_circle}"),
                    metadata={"expected_zero": True},
                )

    def _inscribed_angle_facts(self, state: FormalGeometryState, facts: FactStore) -> None:
        for circle_id, circle in state.circles.items():
            center = state.require_point(circle.center)
            if center.dim != 2:
                continue
            circle_points = []
            for point_id, point in state.points.items():
                if point.dim != 2 or point_id == circle.center:
                    continue
                if sp.simplify(distance(point.coords, center.coords) - circle.radius) == 0:
                    circle_points.append(point_id)
            if len(circle_points) < 3:
                continue

            inscribed_items = []
            for a_id, b_id in combinations(circle_points, 2):
                a = state.require_point(a_id)
                b = state.require_point(b_id)
                central_angle = angle_between(sub(a.coords, center.coords), sub(b.coords, center.coords))
                for observer_id in circle_points:
                    if observer_id in (a_id, b_id):
                        continue
                    observer = state.require_point(observer_id)
                    inscribed_angle = angle_between(sub(a.coords, observer.coords), sub(b.coords, observer.coords))
                    minor_residual = sp.simplify(inscribed_angle - central_angle / 2)
                    major_residual = sp.simplify(inscribed_angle + central_angle / 2 - sp.pi)
                    if minor_residual == 0:
                        residual = minor_residual
                        arc_mode = "minor_arc"
                    elif major_residual == 0:
                        residual = major_residual
                        arc_mode = "major_arc_complement"
                    else:
                        continue
                    object_id = f"{a_id}{observer_id}{b_id}|{circle_id}"
                    facts.add(
                        "inscribed_angle",
                        object_id,
                        inscribed_angle,
                        source="derived",
                        rule="circle_inscribed_angle",
                        dependencies=(f"point:{a_id}", f"point:{observer_id}", f"point:{b_id}", f"circle:{circle_id}"),
                        metadata={"circle": circle_id, "chord_endpoints": [a_id, b_id], "central_angle": str(central_angle), "arc_mode": arc_mode},
                    )
                    facts.add(
                        "inscribed_angle_central_residual",
                        object_id,
                        residual,
                        source="derived",
                        rule="inscribed_angle_half_central_angle_check",
                        dependencies=(f"point:{a_id}", f"point:{observer_id}", f"point:{b_id}", f"circle:{circle_id}"),
                        metadata={"expected_zero": True, "central_angle": str(central_angle), "arc_mode": arc_mode},
                    )
                    for arc_id, arc_angle in _matching_arcs(state, circle_id, a_id, b_id):
                        arc_object_id = f"{a_id}{observer_id}{b_id}|{arc_id}"
                        arc_minor_residual = sp.simplify(inscribed_angle - sp.Abs(arc_angle) / 2)
                        arc_major_residual = sp.simplify(inscribed_angle + sp.Abs(arc_angle) / 2 - sp.pi)
                        if arc_minor_residual == 0:
                            arc_residual = arc_minor_residual
                            arc_mode_from_relation = "arc_half_angle"
                        elif arc_major_residual == 0:
                            arc_residual = arc_major_residual
                            arc_mode_from_relation = "arc_complement"
                        else:
                            continue
                        facts.add(
                            "arc_subtends_inscribed_angle",
                            arc_object_id,
                            source="derived",
                            rule="arc_subtends_inscribed_angle",
                            dependencies=(f"arc:{arc_id}", f"point:{observer_id}", f"circle:{circle_id}"),
                            metadata={"circle": circle_id, "arc": arc_id, "angle": object_id, "mode": arc_mode_from_relation},
                        )
                        facts.add(
                            "inscribed_angle_arc_residual",
                            arc_object_id,
                            arc_residual,
                            source="derived",
                            rule="inscribed_angle_arc_check",
                            dependencies=(f"arc:{arc_id}", f"point:{observer_id}", f"circle:{circle_id}"),
                            metadata={"expected_zero": True, "arc_angle": str(arc_angle), "mode": arc_mode_from_relation},
                        )
                    inscribed_items.append((a_id, b_id, observer_id, object_id, inscribed_angle, arc_mode))

            for left, right in combinations(inscribed_items, 2):
                left_a, left_b, left_observer, left_object, left_angle, left_mode = left
                right_a, right_b, right_observer, right_object, right_angle, right_mode = right
                if (left_a, left_b, left_mode) != (right_a, right_b, right_mode):
                    continue
                facts.add(
                    "equal_inscribed_angles",
                    f"{left_object}|{right_object}",
                    sp.simplify(left_angle - right_angle),
                    source="derived",
                    rule="same_chord_same_arc_inscribed_angles",
                    dependencies=(f"point:{left_a}", f"point:{left_b}", f"point:{left_observer}", f"point:{right_observer}", f"circle:{circle_id}"),
                    metadata={"expected_zero": True, "chord_endpoints": [left_a, left_b], "arc_mode": left_mode},
                )

    def _tangent_chord_angle_facts(self, state: FormalGeometryState, facts: FactStore) -> None:
        tangent_items = []
        for relation in state.relations:
            if relation.get("type") != "tangent":
                continue
            circle_id = relation["circle"]
            tangent_point = relation["point"]
            line_points = relation.get("line") or []
            if len(line_points) != 2 or circle_id not in state.circles:
                continue
            external = line_points[0] if line_points[1] == tangent_point else line_points[1]
            if external not in state.points or tangent_point not in state.points:
                continue
            tangent_items.append((circle_id, external, tangent_point))

        chord_items = []
        for relation in state.relations:
            if relation.get("type") != "chord":
                continue
            chord_id = relation["object"]
            circle_id = relation["circle"]
            if chord_id not in state.segments:
                continue
            seg = state.segments[chord_id]
            chord_items.append((circle_id, chord_id, seg.start, seg.end))

        for circle_id, external, tangent_point in tangent_items:
            circle = state.circles[circle_id]
            center = state.require_point(circle.center)
            if center.dim != 2:
                continue
            tangent = state.require_point(tangent_point)
            external_point = state.require_point(external)
            tangent_vec = sub(external_point.coords, tangent.coords)
            circle_points = [
                point_id
                for point_id, point in state.points.items()
                if point.dim == 2
                and point_id != circle.center
                and sp.simplify(distance(point.coords, center.coords) - circle.radius) == 0
            ]
            for chord_circle, chord_id, start, end in chord_items:
                if chord_circle != circle_id or tangent_point not in (start, end):
                    continue
                chord_other = end if start == tangent_point else start
                chord_other_point = state.require_point(chord_other)
                chord_vec = sub(chord_other_point.coords, tangent.coords)
                tangent_angle = angle_between(tangent_vec, chord_vec)
                for observer_id in circle_points:
                    if observer_id in (tangent_point, chord_other):
                        continue
                    observer = state.require_point(observer_id)
                    alternate_angle = angle_between(
                        sub(state.require_point(tangent_point).coords, observer.coords),
                        sub(chord_other_point.coords, observer.coords),
                    )
                    residual = sp.simplify(tangent_angle - alternate_angle)
                    if residual != 0:
                        continue
                    object_id = _tangent_chord_object_id(tangent_point, chord_other, external, observer_id, circle_id)
                    facts.add(
                        "tangent_chord_angle",
                        object_id,
                        tangent_angle,
                        source="derived",
                        rule="tangent_chord_angle",
                        dependencies=(f"point:{external}", f"point:{tangent_point}", f"segment:{chord_id}", f"circle:{circle_id}"),
                        metadata={"circle": circle_id, "tangent_point": tangent_point, "external_point": external, "chord": chord_id},
                    )
                    facts.add(
                        "alternate_segment_angle",
                        object_id,
                        alternate_angle,
                        source="derived",
                        rule="alternate_segment_angle",
                        dependencies=(f"point:{tangent_point}", f"point:{chord_other}", f"point:{observer_id}", f"circle:{circle_id}"),
                        metadata={"angle": f"{tangent_point}{observer_id}{chord_other}", "observer": observer_id},
                    )
                    facts.add(
                        "tangent_chord_angle_residual",
                        object_id,
                        residual,
                        source="derived",
                        rule="tangent_chord_theorem_check",
                        dependencies=(f"point:{external}", f"point:{tangent_point}", f"segment:{chord_id}", f"point:{observer_id}", f"circle:{circle_id}"),
                        metadata={"expected_zero": True},
                    )

    def _target_fact(self, state: FormalGeometryState, facts: FactStore) -> None:
        if not state.target:
            return
        matches = facts.find(state.target.type, state.target.object)
        if matches:
            valued_matches = [match for match in matches if match.value is not None]
            match = valued_matches[0] if valued_matches else matches[0]
            facts.add("target", state.target.object, match.value, source="derived", rule="target_lookup", dependencies=(match.id,), metadata=state.target.metadata)
        else:
            facts.add("target_unresolved", state.target.object, source="derived", rule="target_lookup_failed", metadata=state.target.metadata)


def _relation_object_id(relation: dict, index: int) -> str:
    for key in ("point", "line", "object"):
        if key in relation:
            return str(relation[key])
    if "objects" in relation:
        return "|".join(str(item) for item in relation["objects"])
    return f"relation_{index}"


def _closed_polyline_length(points) -> sp.Expr:
    if len(points) < 2:
        return sp.Integer(0)
    total = sp.Integer(0)
    for index, point in enumerate(points):
        total += distance(point, points[(index + 1) % len(points)])
    return sp.simplify(total)


def _triangle_items_2d(state: FormalGeometryState) -> list[tuple[str, tuple[str, str, str], Dict[str, tuple]]]:
    triangle_items = []
    for polygon_id, polygon in state.polygons.items():
        if len(polygon.vertices) != 3:
            continue
        vertices = tuple(polygon.vertices)
        points = {pid: state.require_point(pid).coords for pid in vertices}
        if any(len(point) != 2 for point in points.values()):
            continue
        if sp.simplify(polygon_area_2d([points[pid] for pid in vertices])) == 0:
            continue
        triangle_items.append((polygon_id, vertices, points))
    return triangle_items


def _triangle_angles(points: Dict[str, tuple], vertices: tuple[str, str, str]) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    angles = []
    for vertex in vertices:
        others = [pid for pid in vertices if pid != vertex]
        angles.append(
            angle_between(
                sub(points[others[0]], points[vertex]),
                sub(points[others[1]], points[vertex]),
            )
        )
    return tuple(angles)


def _squared_distance(a, b) -> sp.Expr:
    delta = sub(a, b)
    return sp.simplify(sum(value ** 2 for value in delta))


def _triangle_side_squares(points: Dict[str, tuple], vertices: tuple[str, str, str]) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    a, b, c = vertices
    return (
        _squared_distance(points[a], points[b]),
        _squared_distance(points[b], points[c]),
        _squared_distance(points[c], points[a]),
    )


def _similarity_object_id(left_id: str, right_id: str, left_vertices: tuple[str, str, str], right_vertices: tuple[str, str, str]) -> str:
    mapping = ",".join(f"{left}-{right}" for left, right in zip(left_vertices, right_vertices))
    return f"{left_id}|{right_id}|map:{mapping}"


def _parallel_cut_object_id(big_id: str, small_id: str, apex: str, cut_left: str, cut_right: str, base_left: str, base_right: str) -> str:
    return f"{small_id}|{big_id}|apex:{apex}|cut:{cut_left}-{cut_right}|base:{base_left}-{base_right}"


def _tangent_chord_object_id(tangent_point: str, chord_other: str, external: str, observer: str, circle_id: str) -> str:
    return f"{tangent_point}{chord_other}|tan:{external}-{tangent_point}|obs:{observer}|{circle_id}"


def _dihedral_normals_from_relation(state: FormalGeometryState, relation: dict) -> tuple[tuple, tuple] | None:
    if "planes" in relation:
        left_id, right_id = relation["planes"]
        if left_id not in state.planes or right_id not in state.planes:
            return None
        return state.planes[left_id].normal, state.planes[right_id].normal
    if "polyhedron" in relation and "faces" in relation:
        poly_id = relation["polyhedron"]
        if poly_id not in state.polyhedra:
            return None
        left = _polyhedron_face_normal(state, poly_id, relation["faces"][0])
        right = _polyhedron_face_normal(state, poly_id, relation["faces"][1])
        if left is None or right is None:
            return None
        return left, right
    return None


def _dihedral_shared_edge_from_relation(state: FormalGeometryState, relation: dict) -> tuple[str, str] | None:
    if "polyhedron" not in relation or "faces" not in relation:
        return None
    poly_id = relation["polyhedron"]
    if poly_id not in state.polyhedra:
        return None
    left = _polyhedron_face_vertices(state, poly_id, relation["faces"][0])
    right = _polyhedron_face_vertices(state, poly_id, relation["faces"][1])
    if left is None or right is None:
        return None
    shared = [vertex for vertex in left if vertex in set(right)]
    if len(shared) != 2:
        return None
    return tuple(shared)


def _polyhedron_face_vertices(state: FormalGeometryState, poly_id: str, face_ref) -> tuple[str, ...] | None:
    poly = state.polyhedra[poly_id]
    if isinstance(face_ref, int) or str(face_ref).isdigit():
        index = int(face_ref)
        if index < 0 or index >= len(poly.faces):
            return None
        return tuple(poly.faces[index])
    return tuple(face_ref)


def _polyhedron_face_normal(state: FormalGeometryState, poly_id: str, face_ref) -> tuple | None:
    face = _polyhedron_face_vertices(state, poly_id, face_ref)
    if face is None:
        return None
    if len(face) < 3:
        return None
    a = state.require_point(face[0])
    b = state.require_point(face[1])
    c = state.require_point(face[2])
    return cross3(sub(b.coords, a.coords), sub(c.coords, a.coords))


def _dihedral_dependencies(relation: dict) -> tuple[str, ...]:
    if "planes" in relation:
        return tuple(f"plane:{plane_id}" for plane_id in relation["planes"])
    if "polyhedron" in relation:
        return (f"polyhedron:{relation['polyhedron']}",)
    return ()


def _matching_arcs(state: FormalGeometryState, circle_id: str, a_id: str, b_id: str) -> list[tuple[str, sp.Expr]]:
    arcs = []
    endpoints = {a_id, b_id}
    for arc_id, arc in state.arcs.items():
        if arc.circle != circle_id or {arc.start, arc.end} != endpoints or arc.central_angle is None:
            continue
        arcs.append((arc_id, arc.central_angle))
    return arcs
