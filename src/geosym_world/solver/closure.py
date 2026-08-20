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
    is_zero,
    is_rectangle_2d,
    line_tangent_to_circle_at_point_2d,
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
from geosym_world.kernel.geometry2d.intersections import linear_linear_intersection_result_2d, point_on_arc_domain_2d
from geosym_world.kernel.geometry2d.regions import boolean_polygon_area_2d, region_geometry_area_2d, resolve_region_geometry_2d
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
        self._semantic_area_facts(state, facts)
        self._semantic_perimeter_facts(state, facts)
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
        for object_id in state.composite_regions:
            facts.add("exists", object_id, source="constructed", rule="state_registry")
        for object_id in state.planes:
            facts.add("exists", object_id, source="constructed", rule="state_registry")
        for object_id in state.polyhedra:
            facts.add("exists", object_id, source="constructed", rule="state_registry")

    def _object_facts_for_ids(self, state: FormalGeometryState, facts: FactStore, object_ids: set[str]) -> None:
        for bucket in (
            state.points,
            state.lines,
            state.segments,
            state.rays,
            state.circles,
            state.arcs,
            state.polygons,
            state.regions,
            state.composite_regions,
            state.planes,
            state.polyhedra,
        ):
            for object_id in bucket:
                if object_id in object_ids:
                    facts.add("exists", object_id, source="constructed", rule="state_registry")

    def _metric_facts(
        self,
        state: FormalGeometryState,
        facts: FactStore,
        *,
        include_pairwise: bool = True,
        object_ids: set[str] | None = None,
    ) -> None:
        if include_pairwise:
            self._pairwise_distance_facts(state, facts, object_ids=object_ids)

        for seg_id, seg in state.segments.items():
            if object_ids is not None and seg_id not in object_ids:
                continue
            a = state.require_point(seg.start)
            b = state.require_point(seg.end)
            facts.add("length", seg_id, distance(a.coords, b.coords), rule="distance_formula", dependencies=(f"point:{a.id}", f"point:{b.id}"))

        for circle_id, circle in state.circles.items():
            if object_ids is not None and circle_id not in object_ids:
                continue
            facts.add("radius", circle_id, circle.radius, rule="circle_definition", dependencies=(f"point:{circle.center}",))
            facts.add("circumference", circle_id, 2 * sp.pi * circle.radius, rule="circle_circumference_formula", dependencies=(f"circle:{circle_id}",))
            facts.add("area", circle_id, sp.pi * circle.radius ** 2, rule="circle_area_formula", dependencies=(f"circle:{circle_id}",))

        for arc_id, arc in state.arcs.items():
            if object_ids is not None and arc_id not in object_ids:
                continue
            circle = state.circles[arc.circle]
            if arc.central_angle is not None:
                facts.add("arc_length", arc_id, circle.radius * sp.Abs(arc.central_angle), rule="arc_length_formula", dependencies=(f"circle:{arc.circle}", f"arc:{arc_id}"))
                facts.add("sector_area", arc_id, circle.radius ** 2 * sp.Abs(arc.central_angle) / 2, rule="sector_area_formula", dependencies=(f"circle:{arc.circle}", f"arc:{arc_id}"))

        for polygon_id, polygon in state.polygons.items():
            if object_ids is not None and polygon_id not in object_ids:
                continue
            points = [state.require_point(pid).coords for pid in polygon.vertices]
            perimeter = _closed_polyline_length(points)
            facts.add("perimeter", polygon_id, perimeter, rule="closed_polyline_perimeter", dependencies=tuple(f"point:{pid}" for pid in polygon.vertices))
            if points and len(points[0]) == 2:
                facts.add("area", polygon_id, polygon_area_2d(points), rule="shoelace_formula", dependencies=tuple(f"point:{pid}" for pid in polygon.vertices))
            if points and len(points[0]) == 3:
                facts.add("area", polygon_id, polygon_area_3d(points), rule="triangulated_3d_polygon_area", dependencies=tuple(f"point:{pid}" for pid in polygon.vertices))

        for region_id, region in state.regions.items():
            if object_ids is not None and region_id not in object_ids:
                continue
            if region.kind == "polygon_region":
                resolved_geometry = resolve_region_geometry_2d(state, region_id)
                if "polygon" in region.metadata and region.metadata["polygon"] in state.polygons:
                    polygon = state.polygons[region.metadata["polygon"]]
                    points = [state.require_point(pid).coords for pid in polygon.vertices]
                elif len(region.boundary) == 1 and region.boundary[0] in state.polygons:
                    polygon = state.polygons[region.boundary[0]]
                    points = [state.require_point(pid).coords for pid in polygon.vertices]
                elif all(point_id in state.points for point_id in region.boundary):
                    points = [state.require_point(pid).coords for pid in region.boundary]
                else:
                    points = []
                if not points and resolved_geometry.resolved:
                    points = list(resolved_geometry.boundary)
                if not points:
                    if resolved_geometry.status not in {"parametric", "unsupported"}:
                        facts.add(
                            "region_boundary_unresolved",
                            region_id,
                            source="derived",
                            rule="region_boundary_resolution",
                            dependencies=tuple(f"region:{item}" for item in resolved_geometry.source_ids),
                            metadata=resolved_geometry.to_dict(),
                        )
                    continue
                dependencies = (
                    (f"polygon:{region.boundary[0]}",)
                    if len(region.boundary) == 1 and region.boundary[0] in state.polygons
                    else tuple(f"point:{pid}" for pid in region.boundary)
                )
                if points and len(points[0]) == 2:
                    if resolved_geometry.resolved:
                        points = list(resolved_geometry.boundary)
                        hole_points = [list(hole) for hole in resolved_geometry.holes]
                        area = polygon_area_2d(points) - sum((polygon_area_2d(hole) for hole in hole_points), sp.Integer(0))
                        perimeter = _closed_polyline_length(points) + sum((_closed_polyline_length(hole) for hole in hole_points), sp.Integer(0))
                        dependencies = tuple(dict.fromkeys((*dependencies, *(f"region:{item}" for item in resolved_geometry.source_ids))))
                        facts.add(
                            "region_boundary",
                            region_id,
                            source="derived",
                            rule="region_boundary_resolution",
                            dependencies=dependencies,
                            metadata=resolved_geometry.to_dict(),
                        )
                        boundary_fact = facts.find("region_boundary", region_id)[-1]
                        dependencies = tuple(dict.fromkeys((*dependencies, boundary_fact.id)))
                        facts.add("perimeter", region_id, perimeter, rule="polygon_region_perimeter_with_holes" if hole_points else "polygon_region_perimeter", dependencies=dependencies)
                    else:
                        facts.add("perimeter", region_id, _closed_polyline_length(points), rule="polygon_region_perimeter", dependencies=dependencies)
                        area = polygon_area_2d(points)
                elif points and len(points[0]) == 3:
                    facts.add("perimeter", region_id, _closed_polyline_length(points), rule="polygon_region_perimeter", dependencies=dependencies)
                    area = polygon_area_3d(points)
                else:
                    area = None
                if area is not None:
                    if not resolved_geometry.resolved:
                        hole_areas = [facts.find("area", hole_id) for hole_id in region.holes]
                        if all(items for items in hole_areas):
                            area -= sum(sp.sympify(items[0].value) for items in hole_areas)
                    facts.add("area", region_id, area, rule="polygon_region_area_with_holes" if region.holes else "polygon_region_area", dependencies=dependencies + tuple(f"region:{hole_id}" for hole_id in region.holes))
            elif region.kind == "sector_region":
                circle_id = region.metadata["circle"]
                circle = state.circles[circle_id]
                center = state.require_point(circle.center)
                start = state.require_point(region.metadata["start"])
                end = state.require_point(region.metadata["end"])
                coordinate_theta = angle_between(sub(start.coords, center.coords), sub(end.coords, center.coords))
                given_theta = sp.sympify(region.metadata["central_angle"])
                dependencies = (f"circle:{circle_id}", f"point:{start.id}", f"point:{end.id}")
                facts.add("area", region_id, circle.radius ** 2 * coordinate_theta / 2, rule="sector_region_area", dependencies=dependencies)
                facts.add("arc_length", region_id, circle.radius * coordinate_theta, rule="sector_region_arc_length", dependencies=dependencies)
                facts.add("perimeter", region_id, circle.radius * (coordinate_theta + 2), rule="sector_region_perimeter", dependencies=dependencies)
                facts.add(
                    "central_angle",
                    region_id,
                    given_theta,
                    source="constructed",
                    rule="sector_central_angle_label",
                    dependencies=(f"region:{region_id}",),
                    metadata={"unit": "radian", "visible": bool(region.metadata.get("show_central_angle"))},
                )
                facts.add(
                    "central_angle_residual",
                    region_id,
                    sp.simplify(coordinate_theta - given_theta),
                    rule="sector_central_angle_check",
                    dependencies=dependencies,
                    metadata={"expected_zero": True, "unit": "radian"},
                )
            elif region.kind == "circular_segment_region":
                sector = state.regions[region.metadata["sector_region"]]
                circle = state.circles[sector.metadata["circle"]]
                center = state.require_point(circle.center)
                start = state.require_point(sector.metadata["start"])
                end = state.require_point(sector.metadata["end"])
                theta = angle_between(sub(start.coords, center.coords), sub(end.coords, center.coords))
                chord_length = distance(start.coords, end.coords)
                dependencies = (f"circle:{circle.id}", f"point:{start.id}", f"point:{end.id}")
                facts.add(
                    "area",
                    region_id,
                    circle.radius ** 2 * (theta - sp.sin(theta)) / 2,
                    rule="circular_segment_coordinate_area",
                    dependencies=dependencies,
                )
                facts.add(
                    "perimeter",
                    region_id,
                    circle.radius * theta + chord_length,
                    rule="circular_segment_coordinate_perimeter",
                    dependencies=dependencies,
                )
            elif region.kind == "lens_region":
                first_circle_id, second_circle_id = tuple(region.metadata["circles"])
                first_circle = state.circles[first_circle_id]
                second_circle = state.circles[second_circle_id]
                first_center = state.require_point(first_circle.center)
                second_center = state.require_point(second_circle.center)
                first_intersection_id, second_intersection_id = tuple(region.metadata["intersections"])
                center_distance = distance(first_center.coords, second_center.coords)
                first_radius = first_circle.radius
                second_radius = second_circle.radius
                first_projection = sp.simplify(
                    (center_distance ** 2 + first_radius ** 2 - second_radius ** 2) / (2 * center_distance)
                )
                second_projection = sp.simplify(center_distance - first_projection)
                half_chord = sp.simplify(sp.sqrt(first_radius ** 2 - first_projection ** 2))
                value = sp.simplify(
                    first_radius ** 2 * sp.acos(first_projection / first_radius)
                    + second_radius ** 2 * sp.acos(second_projection / second_radius)
                    - center_distance * half_chord
                )
                facts.add(
                    "area",
                    region_id,
                    value,
                    rule="lens_coordinate_area",
                    dependencies=(
                        f"circle:{first_circle_id}",
                        f"circle:{second_circle_id}",
                        f"point:{first_intersection_id}",
                        f"point:{second_intersection_id}",
                    ),
                )
                first_arc_id, second_arc_id = tuple(region.metadata["arcs"])
                first_arc = state.arcs[first_arc_id]
                second_arc = state.arcs[second_arc_id]
                first_arc_angle = angle_between(
                    sub(state.require_point(first_arc.start).coords, first_center.coords),
                    sub(state.require_point(first_arc.end).coords, first_center.coords),
                )
                second_arc_angle = angle_between(
                    sub(state.require_point(second_arc.start).coords, second_center.coords),
                    sub(state.require_point(second_arc.end).coords, second_center.coords),
                )
                facts.add(
                    "perimeter",
                    region_id,
                    sp.simplify(first_radius * first_arc_angle + second_radius * second_arc_angle),
                    rule="lens_coordinate_perimeter",
                    dependencies=(
                        f"circle:{first_circle_id}",
                        f"circle:{second_circle_id}",
                        f"arc:{first_arc_id}",
                        f"arc:{second_arc_id}",
                    ),
                )
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
                facts.add(
                    "perimeter",
                    region_id,
                    2 * sp.pi * (outer.radius + inner.radius),
                    rule="annulus_boundary_perimeter",
                    dependencies=(f"circle:{outer_id}", f"circle:{inner_id}"),
                )

        for region_id, region in state.composite_regions.items():
            if object_ids is not None and region_id not in object_ids:
                continue
            component_facts = [facts.find("area", component) for component in region.components]
            if not all(items for items in component_facts):
                continue
            values = [sp.sympify(items[0].value) for items in component_facts]
            boundary_dependencies: tuple[str, ...] = ()
            if region.operation == "sum":
                area = sum(values)
            elif region.operation in {"union", "intersection", "difference"}:
                resolved_geometry = resolve_region_geometry_2d(state, region_id)
                if resolved_geometry.resolved:
                    facts.add(
                        "region_boundary",
                        region_id,
                        source="derived",
                        rule="region_boundary_resolution",
                        dependencies=tuple(f"region:{item}" for item in resolved_geometry.source_ids),
                        metadata=resolved_geometry.to_dict(),
                    )
                    boundary_fact = facts.find("region_boundary", region_id)[-1]
                    boundary_dependencies = (boundary_fact.id,)
                else:
                    boundary_dependencies = ()
                boolean_area = region_geometry_area_2d(resolved_geometry) if resolved_geometry.resolved else _composite_polygon_boolean_area(state, region)
                if boolean_area is None:
                    facts.add(
                        "area_unresolved",
                        region_id,
                        source="derived",
                        rule="composite_region_boolean_requires_simple_polygons",
                        dependencies=tuple(f"region:{component}" for component in region.components),
                        metadata={"operation": region.operation, "reason": "unsupported_region_boundary"},
                    )
                    continue
                else:
                    area = boolean_area
            else:
                continue
            facts.add(
                "area",
                region_id,
                sp.simplify(area),
                rule=f"composite_region_{region.operation}_area",
                dependencies=tuple(dict.fromkeys((*boundary_dependencies, *(f"region:{component}" for component in region.components)))),
                metadata={"operation": region.operation},
            )

        for poly_id, poly in state.polyhedra.items():
            if object_ids is not None and poly_id not in object_ids:
                continue
            vertex_coords = {pid: state.require_point(pid).coords for pid in poly.vertices}
            surface_area = sp.Integer(0)
            for face in poly.faces:
                points = [state.require_point(pid).coords for pid in face]
                surface_area += polygon_area_3d(points)
            facts.add("surface_area", poly_id, sp.simplify(surface_area), rule="polyhedron_face_area_sum", dependencies=tuple(f"point:{pid}" for pid in poly.vertices))
            facts.add("volume", poly_id, polyhedron_volume_by_origin(vertex_coords, poly.faces), rule="oriented_face_volume", dependencies=tuple(f"point:{pid}" for pid in poly.vertices))

    def _pairwise_distance_facts(
        self,
        state: FormalGeometryState,
        facts: FactStore,
        *,
        object_ids: set[str] | None = None,
    ) -> None:
        if len(state.points) > self.max_combinatorial_points:
            return
        for (left_id, left), (right_id, right) in combinations(state.points.items(), 2):
            if object_ids is not None and left_id not in object_ids and right_id not in object_ids:
                continue
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
            line_like[line_id] = (line.point, line.direction, "line")
        for ray_id, ray in state.rays.items():
            start = state.require_point(ray.start)
            line_like[ray_id] = (start.coords, ray.direction, "ray")
        for seg_id, seg in state.segments.items():
            a = state.require_point(seg.start)
            b = state.require_point(seg.end)
            line_like[seg_id] = (a.coords, sub(b.coords, a.coords), "segment")

        for left_id, right_id in combinations(line_like.keys(), 2):
            left_anchor, left, left_domain = line_like[left_id]
            right_anchor, right, right_domain = line_like[right_id]
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
                intersection = linear_linear_intersection_result_2d(
                    left_anchor,
                    left,
                    right_anchor,
                    right,
                    first_domain=left_domain,
                    second_domain=right_domain,
                    objects=(left_id, right_id),
                )
                if intersection.status == "unique":
                    facts.add(
                        "line_intersection",
                        pair_id,
                        source="derived",
                        rule="linear_domain_intersection",
                        dependencies=(left_id, right_id),
                        metadata={"coords": [str(value) for value in intersection.points[0]], "domains": [left_domain, right_domain]},
                    )
                elif intersection.status == "overlap":
                    facts.add(
                        "linear_overlap",
                        pair_id,
                        source="derived",
                        rule="collinear_domain_overlap",
                        dependencies=(left_id, right_id),
                        metadata=intersection.to_dict(),
                    )
            if is_perpendicular(left, right):
                facts.add("perpendicular", pair_id, source="derived", rule="direction_dot_zero", dependencies=(left_id, right_id))

        if len(state.points) <= self.max_combinatorial_points:
            point_items = list(state.points.items())
            for point_id, point in point_items:
                for line_id, (anchor, direction, domain) in line_like.items():
                    if len(point.coords) != len(anchor):
                        continue
                    if domain != "segment" and _point_on_linear_domain(point.coords, anchor, direction, domain):
                        fact_type = "point_on_line" if domain == "line" else "point_on_ray"
                        facts.add(fact_type, f"{point_id}|{line_id}", source="derived", rule=f"{domain}_domain_membership", dependencies=(f"point:{point_id}", line_id))
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
                for arc_id, arc in state.arcs.items():
                    circle = state.circles[arc.circle]
                    center = state.require_point(circle.center)
                    if len(point.coords) != len(center.coords):
                        continue
                    if point_on_arc_domain_2d(
                        point.coords,
                        center.coords,
                        circle.radius,
                        state.require_point(arc.start).coords,
                        state.require_point(arc.end).coords,
                        orientation=arc.orientation,
                        central_angle=arc.central_angle,
                        allow_full_circle=bool(arc.metadata.get("allow_full_circle")),
                    ):
                        facts.add(
                            "point_on_arc",
                            f"{point_id}|{arc_id}",
                            source="derived",
                            rule="directed_arc_domain_membership",
                            dependencies=(f"point:{point_id}", f"arc:{arc_id}", f"circle:{arc.circle}"),
                        )

            for (a_id, a), (b_id, b), (c_id, c) in combinations(point_items, 3):
                if len(a.coords) != len(b.coords) or len(a.coords) != len(c.coords):
                    continue
                if point_on_line(c.coords, a.coords, sub(b.coords, a.coords)):
                    facts.add("collinear", f"{a_id}{b_id}{c_id}", source="derived", rule="three_point_collinearity", dependencies=(f"point:{a_id}", f"point:{b_id}", f"point:{c_id}"))

        self._plane_topology_facts(state, facts, {object_id: (anchor, direction) for object_id, (anchor, direction, _) in line_like.items()})

    def _curve_intersection_fact(self, state: FormalGeometryState, facts: FactStore, relation: dict, relation_fact) -> None:
        object_ids = tuple(str(item) for item in relation.get("objects", ()))
        point_ids = tuple(str(item) for item in relation.get("points", ()))
        if relation.get("point") is not None:
            point_ids = (str(relation["point"]),)
        if len(object_ids) != 2 or not point_ids:
            return
        if not all(
            point_id in state.points
            and all(_point_on_intersection_object(state, state.points[point_id].coords, object_id) for object_id in object_ids)
            for point_id in point_ids
        ):
            return
        facts.add(
            "curve_intersection",
            "|".join(object_ids),
            source="derived",
            rule="exact_curve_domain_intersection_check",
            dependencies=(
                relation_fact.id,
                *(f"point:{point_id}" for point_id in point_ids),
                *(f"object:{object_id}" for object_id in object_ids),
            ),
            metadata={"objects": list(object_ids), "points": list(point_ids), "selection": relation.get("selection", "all")},
        )

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

    def _construction_relation_facts(
        self,
        state: FormalGeometryState,
        facts: FactStore,
        *,
        relation_ids: set[str] | None = None,
    ) -> None:
        for index, relation in enumerate(state.relations):
            if relation_ids is not None and str(relation["id"]) not in relation_ids:
                continue
            rel_type = str(relation.get("type", "relation"))
            object_id = _relation_object_id(relation, index)
            relation_fact = facts.add(
                rel_type,
                object_id,
                source="constructed",
                rule="construction_trace",
                dependencies=_relation_dependencies(relation),
                metadata=relation,
            )
            if rel_type == "intersection" and any(str(item) in state.arcs for item in relation.get("objects", ())):
                self._curve_intersection_fact(state, facts, relation, relation_fact)
            if rel_type == "equal_length":
                self._explicit_equal_length_facts(state, facts, relation)
            if rel_type == "equal_angle":
                self._explicit_equal_angle_facts(state, facts, relation)
            if rel_type in {"similar", "congruent"}:
                self._explicit_polygon_relation_facts(state, facts, relation, rel_type)
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
                line_points = tuple(str(item) for item in relation.get("line", ()))
                circle_object = state.circles[circle]
                if len(line_points) == 2 and line_tangent_to_circle_at_point_2d(
                    state.require_point(point).coords,
                    state.require_point(line_points[0]).coords,
                    state.require_point(line_points[1]).coords,
                    state.require_point(circle_object.center).coords,
                    circle_object.radius,
                ):
                    facts.add("tangent_to_circle", f"{point}|{circle}", source="constructed", rule="tangent_construction", metadata=relation)
            if rel_type == "point_projection_plane":
                self._point_projection_relation_facts(state, facts, relation)
            if rel_type == "line_projection_plane":
                self._line_projection_relation_facts(state, facts, relation)
            if rel_type == "dihedral_angle":
                self._dihedral_relation_facts(state, facts, relation)
            if rel_type == "polyhedron_plane_section":
                self._section_relation_facts(state, facts, relation)

    def _semantic_area_facts(self, state: FormalGeometryState, facts: FactStore) -> None:
        for polygon_id, polygon in state.polygons.items():
            formula = polygon.metadata.get("area_formula")
            if not isinstance(formula, dict):
                continue
            rule = str(formula.get("rule", ""))
            if rule == "triangle_base_height":
                self._base_height_area_fact(state, facts, polygon_id, formula, divisor=sp.Integer(2), rule=rule)
            elif rule in {"rectangle_base_height", "parallelogram_base_height"}:
                self._base_height_area_fact(state, facts, polygon_id, formula, divisor=sp.Integer(1), rule=rule)
            elif rule == "trapezoid_parallel_bases_height":
                self._trapezoid_area_fact(state, facts, polygon_id, formula)
            elif rule == "kite_perpendicular_diagonals":
                self._kite_area_fact(state, facts, polygon_id, formula)
            elif rule == "chord_triangle_radius_angle":
                self._chord_triangle_area_fact(state, facts, polygon_id, formula)
        self._semantic_region_area_facts(state, facts)
        self._semantic_sector_measure_facts(state, facts)
        self._semantic_annulus_area_facts(state, facts)
        self._semantic_circular_segment_area_facts(state, facts)
        self._semantic_lens_area_facts(state, facts)
        self._semantic_hole_area_facts(state, facts)
        self._semantic_difference_area_facts(state, facts)
        self._partition_area_facts(state, facts)
        self._area_ratio_facts(state, facts)

    def _semantic_perimeter_facts(self, state: FormalGeometryState, facts: FactStore) -> None:
        for polygon_id, polygon in state.polygons.items():
            formula = polygon.metadata.get("perimeter_formula")
            if not isinstance(formula, dict) or formula.get("rule") != "rectangle_side_perimeter":
                continue
            self._rectangle_perimeter_fact(state, facts, polygon_id, formula)
        for region_id, region in state.regions.items():
            preferred_rule = region.metadata.get("preferred_perimeter_rule")
            polygon_id = region.metadata.get("polygon")
            if preferred_rule and polygon_id in state.polygons:
                polygon_fact = _valued_fact(facts, "perimeter", polygon_id, rule=str(preferred_rule))
                if polygon_fact is not None:
                    facts.add(
                        "perimeter",
                        region_id,
                        polygon_fact.value,
                        rule="region_from_semantic_polygon_perimeter",
                        dependencies=(polygon_fact.id,),
                        metadata={"polygon": polygon_id, "polygon_perimeter_rule": preferred_rule},
                    )
        self._semantic_hole_perimeter_facts(state, facts)
        self._semantic_sector_perimeter_facts(state, facts)
        self._semantic_annulus_perimeter_facts(state, facts)
        self._semantic_circular_segment_perimeter_facts(state, facts)
        self._semantic_lens_perimeter_facts(state, facts)

    def _semantic_sector_measure_facts(self, state: FormalGeometryState, facts: FactStore) -> None:
        for region_id, region in state.regions.items():
            if region.kind != "sector_region":
                continue
            inputs = self._sector_public_measure_facts(state, facts, region_id)
            if inputs is None:
                continue
            radius_fact, angle_fact = inputs
            radius = sp.sympify(radius_fact.value)
            theta = sp.sympify(angle_fact.value)
            area_value = sp.simplify(radius ** 2 * theta / 2)
            arc_value = sp.simplify(radius * theta)
            self._add_sector_semantic_fact(
                facts,
                region_id,
                area_value,
                coordinate_rule="sector_region_area",
                inputs=inputs,
                fact_type="area",
                rule="sector_radius_angle_area",
                residual_type="sector_area_residual",
            )
            self._add_sector_semantic_fact(
                facts,
                region_id,
                arc_value,
                coordinate_rule="sector_region_arc_length",
                inputs=inputs,
                fact_type="arc_length",
                rule="sector_radius_angle_arc_length",
                residual_type="sector_arc_length_residual",
            )

    def _semantic_sector_perimeter_facts(self, state: FormalGeometryState, facts: FactStore) -> None:
        for region_id, region in state.regions.items():
            if region.kind != "sector_region":
                continue
            inputs = self._sector_public_measure_facts(state, facts, region_id)
            if inputs is None:
                continue
            radius_fact, angle_fact = inputs
            value = sp.simplify(sp.sympify(radius_fact.value) * (sp.sympify(angle_fact.value) + 2))
            self._add_sector_semantic_fact(
                facts,
                region_id,
                value,
                coordinate_rule="sector_region_perimeter",
                inputs=inputs,
                fact_type="perimeter",
                rule="sector_radius_angle_perimeter",
                residual_type="sector_perimeter_residual",
            )

    def _sector_public_measure_facts(self, state: FormalGeometryState, facts: FactStore, region_id: str):
        region = state.regions[region_id]
        segment_id = str(region.metadata.get("radius_segment", ""))
        circle_id = str(region.metadata.get("circle", ""))
        if not _segment_is_circle_radius(state, segment_id, circle_id):
            return None
        radius_fact = _valued_fact(facts, "length", segment_id)
        angle_fact = _valued_fact(facts, "central_angle", region_id, rule="sector_central_angle_label")
        return (radius_fact, angle_fact) if radius_fact is not None and angle_fact is not None else None

    def _add_sector_semantic_fact(
        self,
        facts: FactStore,
        region_id: str,
        value: sp.Expr,
        *,
        coordinate_rule: str,
        inputs,
        fact_type: str,
        rule: str,
        residual_type: str,
    ) -> None:
        coordinate_fact = _valued_fact(facts, fact_type, region_id, rule=coordinate_rule)
        if coordinate_fact is None:
            return
        dependencies = tuple(fact.id for fact in inputs)
        residual = sp.simplify(sp.sympify(coordinate_fact.value) - value)
        residual_fact = facts.add(
            residual_type,
            region_id,
            residual,
            rule=f"{rule}_check",
            dependencies=(*dependencies, coordinate_fact.id),
            metadata={"expected_zero": True, "radius_segment": inputs[0].object, "angle_object": inputs[1].object},
        )
        if residual != 0:
            return
        facts.add(
            fact_type,
            region_id,
            value,
            rule=rule,
            dependencies=dependencies,
            metadata={
                "admission": "coordinate_verified",
                "admission_fact_id": residual_fact.id,
                "radius_segment": inputs[0].object,
                "angle_object": inputs[1].object,
            },
        )

    def _semantic_annulus_area_facts(self, state: FormalGeometryState, facts: FactStore) -> None:
        for region_id, region in state.regions.items():
            formula = region.metadata.get("area_formula")
            if not isinstance(formula, dict) or formula.get("rule") != "annulus_radius_area":
                continue
            radius_facts = self._annulus_radius_facts(state, facts, region_id, formula)
            coordinate_fact = _valued_fact(facts, "area", region_id, rule="annulus_area_difference")
            if radius_facts is None or coordinate_fact is None:
                continue
            outer_fact, inner_fact = radius_facts
            value = sp.simplify(sp.pi * (sp.sympify(outer_fact.value) ** 2 - sp.sympify(inner_fact.value) ** 2))
            self._add_annulus_semantic_fact(
                facts,
                region_id,
                value,
                coordinate_fact=coordinate_fact,
                radius_facts=radius_facts,
                formula=formula,
                fact_type="area",
                rule="annulus_radius_area",
                residual_type="annulus_area_residual",
            )

    def _chord_triangle_area_fact(self, state: FormalGeometryState, facts: FactStore, polygon_id: str, formula: dict) -> None:
        sector_id = str(formula.get("sector_region", ""))
        if sector_id not in state.regions or state.regions[sector_id].kind != "sector_region":
            return
        inputs = self._sector_public_measure_facts(state, facts, sector_id)
        if inputs is None:
            return
        sector = state.regions[sector_id]
        polygon = state.polygons[polygon_id]
        if set(polygon.vertices) != {sector.metadata["center"], sector.metadata["start"], sector.metadata["end"]}:
            return
        radius_fact, angle_fact = inputs
        value = sp.simplify(sp.sympify(radius_fact.value) ** 2 * sp.sin(sp.sympify(angle_fact.value)) / 2)
        self._add_area_formula_facts(
            facts,
            polygon_id,
            value,
            rule="chord_triangle_radius_angle",
            dependencies=(radius_fact.id, angle_fact.id),
            metadata={"sector_region": sector_id, "radius_segment": radius_fact.object, "angle_object": angle_fact.object},
        )

    def _semantic_circular_segment_area_facts(self, state: FormalGeometryState, facts: FactStore) -> None:
        for region_id, region in state.regions.items():
            formula = region.metadata.get("area_formula")
            if region.kind != "circular_segment_region" or not isinstance(formula, dict) or formula.get("rule") != "circular_segment_difference":
                continue
            sector_id = str(formula.get("sector_region", ""))
            triangle_id = str(formula.get("triangle_polygon", ""))
            sector_fact = _valued_fact(facts, "area", sector_id, rule="sector_radius_angle_area")
            triangle_fact = _valued_fact(facts, "area", triangle_id, rule="chord_triangle_radius_angle")
            coordinate_fact = _valued_fact(facts, "area", region_id, rule="circular_segment_coordinate_area")
            if sector_fact is None or triangle_fact is None or coordinate_fact is None:
                continue
            value = sp.simplify(sp.sympify(sector_fact.value) - sp.sympify(triangle_fact.value))
            residual = sp.simplify(sp.sympify(coordinate_fact.value) - value)
            residual_fact = facts.add(
                "circular_segment_area_residual",
                region_id,
                residual,
                rule="circular_segment_difference_check",
                dependencies=(sector_fact.id, triangle_fact.id, coordinate_fact.id),
                metadata={"expected_zero": True, "sector_region": sector_id, "triangle_polygon": triangle_id},
            )
            if residual != 0:
                continue
            facts.add(
                "area",
                region_id,
                value,
                rule="circular_segment_difference",
                dependencies=(sector_fact.id, triangle_fact.id),
                metadata={
                    "admission": "coordinate_verified",
                    "admission_fact_id": residual_fact.id,
                    "sector_region": sector_id,
                    "triangle_polygon": triangle_id,
                },
            )

    def _semantic_lens_area_facts(self, state: FormalGeometryState, facts: FactStore) -> None:
        for region_id, region in state.regions.items():
            formula = region.metadata.get("area_formula")
            if region.kind != "lens_region" or not isinstance(formula, dict):
                continue
            if formula.get("rule") == "lens_two_radius_area":
                self._semantic_two_radius_lens_area_facts(region_id, formula, facts)
                continue
            if formula.get("rule") != "lens_equal_radius_area":
                continue
            radius_fact = _valued_fact(facts, "length", str(formula.get("radius_segment", "")), rule="distance_formula")
            center_distance_fact = _valued_fact(facts, "length", str(formula.get("center_segment", "")), rule="distance_formula")
            coordinate_fact = _valued_fact(facts, "area", region_id, rule="lens_coordinate_area")
            if radius_fact is None or center_distance_fact is None or coordinate_fact is None:
                continue
            radius = sp.sympify(radius_fact.value)
            center_distance = sp.sympify(center_distance_fact.value)
            half_chord_value = sp.simplify(sp.sqrt(radius ** 2 - center_distance ** 2 / 4))
            half_chord_fact = facts.add(
                "lens_half_chord",
                region_id,
                half_chord_value,
                rule="lens_half_chord_from_radius_distance",
                dependencies=(radius_fact.id, center_distance_fact.id),
                metadata={"radius_segment": radius_fact.object, "center_segment": center_distance_fact.object},
            )
            angle_value = sp.simplify(2 * sp.acos(center_distance / (2 * radius)))
            angle_fact = facts.add(
                "lens_central_angle",
                region_id,
                angle_value,
                rule="lens_angle_from_radius_distance",
                dependencies=(radius_fact.id, center_distance_fact.id),
                metadata={"unit": "radian"},
            )
            sectors_value = sp.simplify(radius ** 2 * angle_value)
            sectors_fact = facts.add(
                "lens_sector_area_sum",
                region_id,
                sectors_value,
                rule="lens_two_sector_area",
                dependencies=(radius_fact.id, angle_fact.id),
            )
            triangles_value = sp.simplify(center_distance * half_chord_value)
            triangles_fact = facts.add(
                "lens_triangle_area_sum",
                region_id,
                triangles_value,
                rule="lens_two_triangle_area",
                dependencies=(center_distance_fact.id, half_chord_fact.id),
            )
            value = sp.simplify(sectors_value - triangles_value)
            residual = sp.trigsimp(sp.simplify(sp.sympify(coordinate_fact.value) - value))
            residual_fact = facts.add(
                "lens_area_residual",
                region_id,
                residual,
                rule="lens_equal_radius_area_check",
                dependencies=(sectors_fact.id, triangles_fact.id, coordinate_fact.id),
                metadata={"expected_zero": True},
            )
            if residual != 0:
                continue
            facts.add(
                "area",
                region_id,
                value,
                rule="lens_equal_radius_area",
                dependencies=(sectors_fact.id, triangles_fact.id),
                metadata={
                    "admission": "coordinate_verified",
                    "admission_fact_id": residual_fact.id,
                    "radius_segment": radius_fact.object,
                    "center_segment": center_distance_fact.object,
                },
            )

    def _semantic_two_radius_lens_area_facts(self, region_id: str, formula: dict, facts: FactStore) -> None:
        radius_segment_ids = tuple(formula.get("radius_segments") or ())
        if len(radius_segment_ids) != 2:
            return
        first_radius_fact = _valued_fact(facts, "length", str(radius_segment_ids[0]), rule="distance_formula")
        second_radius_fact = _valued_fact(facts, "length", str(radius_segment_ids[1]), rule="distance_formula")
        center_distance_fact = _valued_fact(facts, "length", str(formula.get("center_segment", "")), rule="distance_formula")
        coordinate_fact = _valued_fact(facts, "area", region_id, rule="lens_coordinate_area")
        if any(item is None for item in (first_radius_fact, second_radius_fact, center_distance_fact, coordinate_fact)):
            return
        first_radius = sp.sympify(first_radius_fact.value)
        second_radius = sp.sympify(second_radius_fact.value)
        center_distance = sp.sympify(center_distance_fact.value)
        first_projection_value = sp.simplify(
            (center_distance ** 2 + first_radius ** 2 - second_radius ** 2) / (2 * center_distance)
        )
        first_projection_fact = facts.add(
            "lens_first_projection",
            region_id,
            first_projection_value,
            rule="lens_first_projection_from_radii_distance",
            dependencies=(first_radius_fact.id, second_radius_fact.id, center_distance_fact.id),
        )
        second_projection_value = sp.simplify(center_distance - first_projection_value)
        second_projection_fact = facts.add(
            "lens_second_projection",
            region_id,
            second_projection_value,
            rule="lens_second_projection_from_distance",
            dependencies=(center_distance_fact.id, first_projection_fact.id),
        )
        half_chord_value = sp.simplify(sp.sqrt(first_radius ** 2 - first_projection_value ** 2))
        half_chord_fact = facts.add(
            "lens_half_chord",
            region_id,
            half_chord_value,
            rule="lens_half_chord_from_radius_projection",
            dependencies=(first_radius_fact.id, first_projection_fact.id),
        )
        first_angle_value = sp.simplify(2 * sp.acos(first_projection_value / first_radius))
        second_angle_value = sp.simplify(2 * sp.acos(second_projection_value / second_radius))
        first_angle_fact = facts.add(
            "lens_first_central_angle",
            region_id,
            first_angle_value,
            rule="lens_first_angle_from_projection_radius",
            dependencies=(first_projection_fact.id, first_radius_fact.id),
            metadata={"unit": "radian"},
        )
        second_angle_fact = facts.add(
            "lens_second_central_angle",
            region_id,
            second_angle_value,
            rule="lens_second_angle_from_projection_radius",
            dependencies=(second_projection_fact.id, second_radius_fact.id),
            metadata={"unit": "radian"},
        )
        first_sector_fact = facts.add(
            "lens_first_sector_area",
            region_id,
            sp.simplify(first_radius ** 2 * first_angle_value / 2),
            rule="lens_first_sector_area",
            dependencies=(first_radius_fact.id, first_angle_fact.id),
        )
        second_sector_fact = facts.add(
            "lens_second_sector_area",
            region_id,
            sp.simplify(second_radius ** 2 * second_angle_value / 2),
            rule="lens_second_sector_area",
            dependencies=(second_radius_fact.id, second_angle_fact.id),
        )
        first_triangle_fact = facts.add(
            "lens_first_triangle_area",
            region_id,
            sp.simplify(first_projection_value * half_chord_value),
            rule="lens_first_triangle_area",
            dependencies=(first_projection_fact.id, half_chord_fact.id),
        )
        second_triangle_fact = facts.add(
            "lens_second_triangle_area",
            region_id,
            sp.simplify(second_projection_value * half_chord_value),
            rule="lens_second_triangle_area",
            dependencies=(second_projection_fact.id, half_chord_fact.id),
        )
        sectors_fact = facts.add(
            "lens_sector_area_sum",
            region_id,
            sp.simplify(sp.sympify(first_sector_fact.value) + sp.sympify(second_sector_fact.value)),
            rule="lens_sector_area_sum",
            dependencies=(first_sector_fact.id, second_sector_fact.id),
        )
        triangles_fact = facts.add(
            "lens_triangle_area_sum",
            region_id,
            sp.simplify(sp.sympify(first_triangle_fact.value) + sp.sympify(second_triangle_fact.value)),
            rule="lens_triangle_area_sum",
            dependencies=(first_triangle_fact.id, second_triangle_fact.id),
        )
        value = sp.simplify(sp.sympify(sectors_fact.value) - sp.sympify(triangles_fact.value))
        residual = sp.trigsimp(sp.simplify(sp.sympify(coordinate_fact.value) - value))
        residual_fact = facts.add(
            "lens_area_residual",
            region_id,
            residual,
            rule="lens_two_radius_area_check",
            dependencies=(sectors_fact.id, triangles_fact.id, coordinate_fact.id),
            metadata={"expected_zero": True},
        )
        if residual != 0:
            return
        facts.add(
            "area",
            region_id,
            value,
            rule="lens_two_radius_area",
            dependencies=(sectors_fact.id, triangles_fact.id),
            metadata={
                "admission": "coordinate_verified",
                "admission_fact_id": residual_fact.id,
                "radius_segments": list(radius_segment_ids),
                "center_segment": center_distance_fact.object,
            },
        )

    def _semantic_circular_segment_perimeter_facts(self, state: FormalGeometryState, facts: FactStore) -> None:
        for region_id, region in state.regions.items():
            formula = region.metadata.get("perimeter_formula")
            if (
                region.kind != "circular_segment_region"
                or not isinstance(formula, dict)
                or formula.get("rule") != "circular_segment_arc_chord_perimeter"
            ):
                continue
            sector_id = str(formula.get("sector_region", ""))
            chord_id = str(formula.get("chord", ""))
            inputs = self._sector_public_measure_facts(state, facts, sector_id) if sector_id in state.regions else None
            coordinate_chord = _valued_fact(facts, "length", chord_id, rule="distance_formula")
            coordinate_perimeter = _valued_fact(facts, "perimeter", region_id, rule="circular_segment_coordinate_perimeter")
            arc_fact = _valued_fact(facts, "arc_length", sector_id, rule="sector_radius_angle_arc_length")
            if inputs is None or coordinate_chord is None or coordinate_perimeter is None or arc_fact is None:
                continue
            radius_fact, angle_fact = inputs
            chord_value = sp.simplify(
                2 * sp.sympify(radius_fact.value) * sp.sin(sp.sympify(angle_fact.value) / 2)
            )
            chord_residual = sp.simplify(sp.sympify(coordinate_chord.value) - chord_value)
            chord_residual_fact = facts.add(
                "chord_length_residual",
                chord_id,
                chord_residual,
                rule="chord_radius_angle_length_check",
                dependencies=(radius_fact.id, angle_fact.id, coordinate_chord.id),
                metadata={"expected_zero": True, "sector_region": sector_id},
            )
            if chord_residual != 0:
                continue
            chord_fact = facts.add(
                "length",
                chord_id,
                chord_value,
                rule="chord_radius_angle_length",
                dependencies=(radius_fact.id, angle_fact.id),
                metadata={
                    "admission": "coordinate_verified",
                    "admission_fact_id": chord_residual_fact.id,
                    "sector_region": sector_id,
                },
            )
            value = sp.simplify(sp.sympify(arc_fact.value) + chord_value)
            perimeter_residual = sp.simplify(sp.sympify(coordinate_perimeter.value) - value)
            perimeter_residual_fact = facts.add(
                "circular_segment_perimeter_residual",
                region_id,
                perimeter_residual,
                rule="circular_segment_arc_chord_perimeter_check",
                dependencies=(arc_fact.id, chord_fact.id, coordinate_perimeter.id),
                metadata={"expected_zero": True, "sector_region": sector_id, "chord": chord_id},
            )
            if perimeter_residual != 0:
                continue
            facts.add(
                "perimeter",
                region_id,
                value,
                rule="circular_segment_arc_chord_perimeter",
                dependencies=(arc_fact.id, chord_fact.id),
                metadata={
                    "admission": "coordinate_verified",
                    "admission_fact_id": perimeter_residual_fact.id,
                    "sector_region": sector_id,
                    "chord": chord_id,
                },
            )

    def _semantic_lens_perimeter_facts(self, state: FormalGeometryState, facts: FactStore) -> None:
        for region_id, region in state.regions.items():
            formula = region.metadata.get("perimeter_formula")
            if region.kind != "lens_region" or not isinstance(formula, dict):
                continue
            rule = formula.get("rule")
            if rule == "lens_equal_radius_perimeter":
                radius_segment_ids = (str(formula.get("radius_segment", "")),)
                semantic_rule = "lens_equal_radius_perimeter"
                residual_rule = "lens_equal_radius_perimeter_check"
            elif rule == "lens_two_radius_perimeter":
                radius_segment_ids = tuple(str(item) for item in (formula.get("radius_segments") or ()))
                semantic_rule = "lens_two_radius_perimeter"
                residual_rule = "lens_two_radius_perimeter_check"
            else:
                continue
            if len(radius_segment_ids) not in {1, 2}:
                continue
            radius_facts = tuple(
                _valued_fact(facts, "length", segment_id, rule="distance_formula")
                for segment_id in radius_segment_ids
            )
            center_fact = _valued_fact(facts, "length", str(formula.get("center_segment", "")), rule="distance_formula")
            coordinate_fact = _valued_fact(facts, "perimeter", region_id, rule="lens_coordinate_perimeter")
            if any(fact is None for fact in (*radius_facts, center_fact, coordinate_fact)):
                continue
            first_radius = sp.sympify(radius_facts[0].value)
            center_distance = sp.sympify(center_fact.value)
            if rule == "lens_equal_radius_perimeter":
                angle_value = sp.simplify(2 * sp.acos(center_distance / (2 * first_radius)))
                value = sp.simplify(2 * first_radius * angle_value)
                dependencies = (radius_facts[0].id, center_fact.id)
            else:
                second_radius = sp.sympify(radius_facts[1].value)
                first_projection = sp.simplify(
                    (center_distance ** 2 + first_radius ** 2 - second_radius ** 2) / (2 * center_distance)
                )
                second_projection = sp.simplify(center_distance - first_projection)
                first_angle = sp.simplify(2 * sp.acos(first_projection / first_radius))
                second_angle = sp.simplify(2 * sp.acos(second_projection / second_radius))
                value = sp.simplify(first_radius * first_angle + second_radius * second_angle)
                dependencies = (radius_facts[0].id, radius_facts[1].id, center_fact.id)
            residual = sp.trigsimp(sp.simplify(sp.sympify(coordinate_fact.value) - value))
            residual = _canonical_zero_residual(residual)
            residual_fact = facts.add(
                "lens_perimeter_residual",
                region_id,
                residual,
                rule=residual_rule,
                dependencies=(*dependencies, coordinate_fact.id),
                metadata={"expected_zero": True},
            )
            if residual != 0:
                continue
            facts.add(
                "perimeter",
                region_id,
                value,
                rule=semantic_rule,
                dependencies=dependencies,
                metadata={"admission": "coordinate_verified", "admission_fact_id": residual_fact.id},
            )

    def _semantic_annulus_perimeter_facts(self, state: FormalGeometryState, facts: FactStore) -> None:
        for region_id, region in state.regions.items():
            formula = region.metadata.get("perimeter_formula")
            if not isinstance(formula, dict) or formula.get("rule") != "annulus_radius_perimeter":
                continue
            radius_facts = self._annulus_radius_facts(state, facts, region_id, formula)
            coordinate_fact = _valued_fact(facts, "perimeter", region_id, rule="annulus_boundary_perimeter")
            if radius_facts is None or coordinate_fact is None:
                continue
            outer_fact, inner_fact = radius_facts
            value = sp.simplify(2 * sp.pi * (sp.sympify(outer_fact.value) + sp.sympify(inner_fact.value)))
            self._add_annulus_semantic_fact(
                facts,
                region_id,
                value,
                coordinate_fact=coordinate_fact,
                radius_facts=radius_facts,
                formula=formula,
                fact_type="perimeter",
                rule="annulus_radius_perimeter",
                residual_type="annulus_perimeter_residual",
            )

    def _annulus_radius_facts(self, state: FormalGeometryState, facts: FactStore, region_id: str, formula: dict):
        region = state.regions[region_id]
        outer_segment_id = str(formula.get("outer_radius_segment", ""))
        inner_segment_id = str(formula.get("inner_radius_segment", ""))
        outer_circle_id = str(region.metadata.get("outer_circle", ""))
        inner_circle_id = str(region.metadata.get("inner_circle", ""))
        if not _segment_is_circle_radius(state, outer_segment_id, outer_circle_id):
            return None
        if not _segment_is_circle_radius(state, inner_segment_id, inner_circle_id):
            return None
        outer_fact = _valued_fact(facts, "length", outer_segment_id)
        inner_fact = _valued_fact(facts, "length", inner_segment_id)
        return (outer_fact, inner_fact) if outer_fact is not None and inner_fact is not None else None

    def _add_annulus_semantic_fact(
        self,
        facts: FactStore,
        region_id: str,
        value: sp.Expr,
        *,
        coordinate_fact,
        radius_facts,
        formula: dict,
        fact_type: str,
        rule: str,
        residual_type: str,
    ) -> None:
        dependencies = tuple(fact.id for fact in radius_facts)
        metadata = {
            "outer_radius_segment": formula["outer_radius_segment"],
            "inner_radius_segment": formula["inner_radius_segment"],
        }
        residual = sp.simplify(sp.sympify(coordinate_fact.value) - value)
        residual_fact = facts.add(
            residual_type,
            region_id,
            residual,
            rule=f"{rule}_check",
            dependencies=(*dependencies, coordinate_fact.id),
            metadata={"expected_zero": True, **metadata},
        )
        if residual != 0:
            return
        facts.add(
            fact_type,
            region_id,
            value,
            rule=rule,
            dependencies=dependencies,
            metadata={"admission": "coordinate_verified", "admission_fact_id": residual_fact.id, **metadata},
        )

    def _rectangle_perimeter_fact(self, state: FormalGeometryState, facts: FactStore, polygon_id: str, formula: dict) -> None:
        polygon = state.polygons[polygon_id]
        width_id = str(formula.get("width", ""))
        height_id = str(formula.get("height", ""))
        if not _segments_are_adjacent_polygon_edges(state, polygon.vertices, width_id, height_id):
            return
        points = tuple(state.require_point(point_id).coords for point_id in polygon.vertices)
        if not is_rectangle_2d(points):
            return
        width_fact = _valued_fact(facts, "length", width_id)
        height_fact = _valued_fact(facts, "length", height_id)
        coordinate_fact = _valued_fact(facts, "perimeter", polygon_id, rule="closed_polyline_perimeter")
        if width_fact is None or height_fact is None or coordinate_fact is None:
            return
        dependencies = (width_fact.id, height_fact.id)
        value = sp.simplify(2 * (sp.sympify(width_fact.value) + sp.sympify(height_fact.value)))
        residual = sp.simplify(sp.sympify(coordinate_fact.value) - value)
        residual_fact = facts.add(
            "perimeter_formula_residual",
            f"{polygon_id}|rectangle_side_perimeter",
            residual,
            rule="rectangle_side_perimeter_check",
            dependencies=(*dependencies, coordinate_fact.id),
            metadata={"expected_zero": True, "width": width_id, "height": height_id},
        )
        if residual != 0:
            return
        facts.add(
            "perimeter",
            polygon_id,
            value,
            rule="rectangle_side_perimeter",
            dependencies=dependencies,
            metadata={"admission": "coordinate_verified", "admission_fact_id": residual_fact.id, "width": width_id, "height": height_id},
        )

    def _semantic_hole_perimeter_facts(self, state: FormalGeometryState, facts: FactStore) -> None:
        for region_id, region in state.regions.items():
            formula = region.metadata.get("perimeter_formula")
            if not isinstance(formula, dict) or formula.get("rule") != "region_hole_boundary_sum":
                continue
            resolved = resolve_region_geometry_2d(state, region_id)
            outer_id = formula.get("outer_region")
            component_ids = (outer_id, *region.holes)
            component_facts = tuple(_preferred_region_perimeter_fact(state, facts, component_id) for component_id in component_ids)
            coordinate_fact = _valued_fact(facts, "perimeter", region_id, rule="polygon_region_perimeter_with_holes")
            if not resolved.resolved or coordinate_fact is None or any(fact is None for fact in component_facts):
                continue
            value = sp.simplify(sum((sp.sympify(fact.value) for fact in component_facts), sp.Integer(0)))
            dependencies = tuple(fact.id for fact in component_facts)
            residual = sp.simplify(sp.sympify(coordinate_fact.value) - value)
            residual_fact = facts.add(
                "perimeter_hole_sum_residual",
                region_id,
                residual,
                rule="region_hole_boundary_sum_check",
                dependencies=(*dependencies, coordinate_fact.id),
                metadata={"expected_zero": True, "outer_region": outer_id, "holes": list(region.holes)},
            )
            if residual != 0:
                continue
            facts.add(
                "perimeter",
                region_id,
                value,
                rule="region_hole_boundary_sum",
                dependencies=dependencies,
                metadata={
                    "admission": "materialized_holes_verified",
                    "admission_fact_id": residual_fact.id,
                    "outer_region": outer_id,
                    "holes": list(region.holes),
                },
            )

    def _semantic_region_area_facts(self, state: FormalGeometryState, facts: FactStore) -> None:
        for region_id, region in state.regions.items():
            preferred_rule = region.metadata.get("preferred_area_rule")
            polygon_id = region.metadata.get("polygon")
            if not preferred_rule or polygon_id not in state.polygons:
                continue
            polygon_fact = _valued_fact(facts, "area", polygon_id, rule=str(preferred_rule))
            if polygon_fact is None:
                continue
            facts.add(
                "area",
                region_id,
                polygon_fact.value,
                rule="region_from_semantic_polygon_area",
                dependencies=(polygon_fact.id,),
                metadata={"polygon": polygon_id, "polygon_area_rule": preferred_rule},
            )

    def _partition_area_facts(self, state: FormalGeometryState, facts: FactStore) -> None:
        for region_id, region in state.composite_regions.items():
            if region.operation != "partition":
                continue
            parent_id = region.metadata.get("parent_region")
            resolved = resolve_region_geometry_2d(state, region_id)
            if not resolved.resolved or parent_id not in state.regions:
                facts.add(
                    "area_unresolved",
                    region_id,
                    rule="region_partition_validation_failed",
                    dependencies=tuple(f"region:{component}" for component in region.components),
                    metadata={"operation": "partition", "reason": resolved.reason, "parent_region": parent_id},
                )
                continue
            component_facts = tuple(self._preferred_region_area_fact(state, facts, component) for component in region.components)
            parent_fact = _valued_fact(facts, "area", parent_id)
            if parent_fact is None or any(fact is None for fact in component_facts):
                continue
            value = sp.simplify(sum((sp.sympify(fact.value) for fact in component_facts), sp.Integer(0)))
            residual = sp.simplify(sp.sympify(parent_fact.value) - value)
            dependencies = tuple(fact.id for fact in component_facts)
            residual_fact = facts.add(
                "area_partition_residual",
                f"{region_id}|{parent_id}",
                residual,
                rule="region_partition_area_check",
                dependencies=(*dependencies, parent_fact.id),
                metadata={"expected_zero": True, "parent_region": parent_id, "components": list(region.components)},
            )
            if residual != 0:
                continue
            facts.add(
                "area",
                region_id,
                value,
                rule="region_partition_area_sum",
                dependencies=dependencies,
                metadata={
                    "admission": "parent_area_verified",
                    "admission_fact_id": residual_fact.id,
                    "parent_region": parent_id,
                    "components": list(region.components),
                },
            )

    def _semantic_hole_area_facts(self, state: FormalGeometryState, facts: FactStore) -> None:
        for region_id, region in state.regions.items():
            formula = region.metadata.get("area_formula")
            if not isinstance(formula, dict) or formula.get("rule") != "region_hole_difference":
                continue
            resolved = resolve_region_geometry_2d(state, region_id)
            outer_id = formula.get("outer_region")
            component_ids = (outer_id, *region.holes)
            component_facts = tuple(self._preferred_region_area_fact(state, facts, component_id) for component_id in component_ids)
            coordinate_fact = _valued_fact(facts, "area", region_id, rule="polygon_region_area_with_holes")
            if not resolved.resolved or coordinate_fact is None or any(fact is None for fact in component_facts):
                continue
            value = sp.simplify(
                sp.sympify(component_facts[0].value)
                - sum((sp.sympify(fact.value) for fact in component_facts[1:]), sp.Integer(0))
            )
            dependencies = tuple(fact.id for fact in component_facts)
            residual = sp.simplify(sp.sympify(coordinate_fact.value) - value)
            residual_fact = facts.add(
                "area_hole_difference_residual",
                region_id,
                residual,
                rule="region_hole_difference_area_check",
                dependencies=(*dependencies, coordinate_fact.id),
                metadata={"expected_zero": True, "outer_region": outer_id, "holes": list(region.holes)},
            )
            if residual != 0:
                continue
            facts.add(
                "area",
                region_id,
                value,
                rule="region_hole_difference_area",
                dependencies=dependencies,
                metadata={
                    "admission": "materialized_holes_verified",
                    "admission_fact_id": residual_fact.id,
                    "outer_region": outer_id,
                    "holes": list(region.holes),
                },
            )

    def _semantic_difference_area_facts(self, state: FormalGeometryState, facts: FactStore) -> None:
        for region_id, region in state.composite_regions.items():
            formula = region.metadata.get("area_formula")
            if not isinstance(formula, dict) or formula.get("rule") != "region_difference":
                continue
            resolved = resolve_region_geometry_2d(state, region_id)
            component_facts = tuple(self._preferred_region_area_fact(state, facts, component) for component in region.components)
            coordinate_fact = _valued_fact(facts, "area", region_id, rule="composite_region_difference_area")
            if not resolved.resolved or coordinate_fact is None or any(fact is None for fact in component_facts):
                continue
            value = sp.simplify(sp.sympify(component_facts[0].value) - sp.sympify(component_facts[1].value))
            dependencies = tuple(fact.id for fact in component_facts)
            residual = sp.simplify(sp.sympify(coordinate_fact.value) - value)
            residual_fact = facts.add(
                "area_difference_residual",
                region_id,
                residual,
                rule="region_difference_area_check",
                dependencies=(*dependencies, coordinate_fact.id),
                metadata={"expected_zero": True, "components": list(region.components)},
            )
            if residual != 0:
                continue
            facts.add(
                "area",
                region_id,
                value,
                rule="region_difference_area",
                dependencies=dependencies,
                metadata={
                    "admission": "boolean_area_verified",
                    "admission_fact_id": residual_fact.id,
                    "components": list(region.components),
                },
            )

    def _preferred_region_area_fact(self, state: FormalGeometryState, facts: FactStore, region_id: str):
        region = state.regions.get(region_id)
        if region is None:
            return _valued_fact(facts, "area", region_id)
        preferred_rule = region.metadata.get("preferred_area_rule")
        if preferred_rule:
            direct = _valued_fact(facts, "area", region_id, rule=str(preferred_rule))
            if direct is not None:
                return direct
            semantic = _valued_fact(facts, "area", region_id, rule="region_from_semantic_polygon_area")
            if semantic is not None and semantic.metadata.get("polygon_area_rule") == preferred_rule:
                return semantic
        return _valued_fact(facts, "area", region_id)

    def _area_ratio_facts(self, state: FormalGeometryState, facts: FactStore) -> None:
        target = state.target
        if target is None or target.type != "ratio" or target.metadata.get("ratio_kind") != "area":
            return
        numerator_id = target.metadata.get("numerator")
        denominator_id = target.metadata.get("denominator")
        numerator = self._preferred_region_area_fact(state, facts, numerator_id)
        denominator = self._preferred_region_area_fact(state, facts, denominator_id)
        if numerator is None or denominator is None:
            return
        denominator_value = sp.sympify(denominator.value)
        if denominator_value == 0:
            facts.add(
                "ratio_unresolved",
                target.object,
                rule="region_area_ratio_zero_denominator",
                dependencies=(numerator.id, denominator.id),
                metadata={"numerator": numerator_id, "denominator": denominator_id},
            )
            return
        facts.add(
            "ratio",
            target.object,
            sp.simplify(sp.sympify(numerator.value) / denominator_value),
            rule="region_area_ratio",
            dependencies=(numerator.id, denominator.id),
            metadata={"ratio_kind": "area", "numerator": numerator_id, "denominator": denominator_id, "unit": "dimensionless"},
        )

    def _base_height_area_fact(
        self,
        state: FormalGeometryState,
        facts: FactStore,
        polygon_id: str,
        formula: dict,
        *,
        divisor: sp.Expr,
        rule: str,
    ) -> None:
        base_id = str(formula.get("base", ""))
        height_reference = formula.get("height")
        if base_id not in state.segments:
            return
        base_segment = state.segments[base_id]
        base_direction = sub(state.require_point(base_segment.end).coords, state.require_point(base_segment.start).coords)
        base_fact = _valued_fact(facts, "length", base_id)
        height_fact, height_pair = self._height_fact(state, facts, height_reference, base_directions=(base_direction,))
        if base_fact is None or height_fact is None:
            return
        value = sp.simplify(sp.sympify(base_fact.value) * sp.sympify(height_fact.value) / divisor)
        self._add_area_formula_facts(
            facts,
            polygon_id,
            value,
            rule=rule,
            dependencies=(base_fact.id, height_fact.id),
            metadata={"base": base_id, "height": list(height_pair), "divisor": str(divisor)},
        )

    def _trapezoid_area_fact(self, state: FormalGeometryState, facts: FactStore, polygon_id: str, formula: dict) -> None:
        bases = tuple(str(item) for item in formula.get("bases", ()))
        height_reference = formula.get("height")
        if len(bases) != 2 or any(base_id not in state.segments for base_id in bases):
            return
        base_directions = tuple(
            sub(state.require_point(state.segments[base_id].end).coords, state.require_point(state.segments[base_id].start).coords)
            for base_id in bases
        )
        if not is_parallel(base_directions[0], base_directions[1]):
            return
        base_facts = tuple(_valued_fact(facts, "length", base_id) for base_id in bases)
        height_fact, height_pair = self._height_fact(state, facts, height_reference, base_directions=base_directions)
        if any(fact is None for fact in base_facts) or height_fact is None:
            return
        value = sp.simplify(
            (sp.sympify(base_facts[0].value) + sp.sympify(base_facts[1].value))
            * sp.sympify(height_fact.value)
            / 2
        )
        self._add_area_formula_facts(
            facts,
            polygon_id,
            value,
            rule="trapezoid_parallel_bases_height",
            dependencies=(base_facts[0].id, base_facts[1].id, height_fact.id),
            metadata={"bases": list(bases), "height": list(height_pair)},
        )

    def _kite_area_fact(self, state: FormalGeometryState, facts: FactStore, polygon_id: str, formula: dict) -> None:
        diagonals = tuple(str(item) for item in formula.get("diagonals", ()))
        if len(diagonals) != 2 or any(diagonal not in state.segments for diagonal in diagonals):
            return
        diagonal_facts = tuple(_valued_fact(facts, "length", diagonal) for diagonal in diagonals)
        if any(fact is None for fact in diagonal_facts):
            return
        first = state.segments[diagonals[0]]
        second = state.segments[diagonals[1]]
        first_direction = sub(state.require_point(first.end).coords, state.require_point(first.start).coords)
        second_direction = sub(state.require_point(second.end).coords, state.require_point(second.start).coords)
        if not is_perpendicular(first_direction, second_direction):
            return
        value = sp.simplify(sp.sympify(diagonal_facts[0].value) * sp.sympify(diagonal_facts[1].value) / 2)
        self._add_area_formula_facts(
            facts,
            polygon_id,
            value,
            rule="kite_perpendicular_diagonals",
            dependencies=(diagonal_facts[0].id, diagonal_facts[1].id),
            metadata={"diagonals": list(diagonals)},
        )

    def _distance_fact(
        self,
        state: FormalGeometryState,
        facts: FactStore,
        pair: tuple[str, str],
        *,
        rule: str,
    ):
        left_id, right_id = pair
        if left_id not in state.points or right_id not in state.points:
            return None
        object_id = f"{left_id}|{right_id}"
        existing = _valued_fact(facts, "distance", object_id) or _valued_fact(facts, "distance", f"{right_id}|{left_id}")
        if existing is not None:
            return existing
        return facts.add(
            "distance",
            object_id,
            distance(state.require_point(left_id).coords, state.require_point(right_id).coords),
            rule=rule,
            dependencies=(f"point:{left_id}", f"point:{right_id}"),
        )

    def _height_fact(
        self,
        state: FormalGeometryState,
        facts: FactStore,
        reference,
        *,
        base_directions: tuple[tuple, ...],
    ):
        if isinstance(reference, str):
            if reference not in state.segments:
                return None, None
            segment = state.segments[reference]
            pair = (segment.start, segment.end)
            height_direction = sub(state.require_point(segment.end).coords, state.require_point(segment.start).coords)
            fact = _valued_fact(facts, "length", reference)
        else:
            pair = _as_point_pair(reference)
            if pair is None or any(point_id not in state.points for point_id in pair):
                return None, None
            height_direction = sub(state.require_point(pair[1]).coords, state.require_point(pair[0]).coords)
            matching_segment = next(
                (
                    segment_id
                    for segment_id, segment in state.segments.items()
                    if {segment.start, segment.end} == set(pair)
                ),
                None,
            )
            fact = _valued_fact(facts, "length", matching_segment) if matching_segment is not None else None
            if fact is None:
                fact = self._distance_fact(state, facts, pair, rule="perpendicular_height")
        if fact is None or any(not is_perpendicular(height_direction, base_direction) for base_direction in base_directions):
            return None, None
        return fact, pair

    def _add_area_formula_facts(
        self,
        facts: FactStore,
        polygon_id: str,
        value: sp.Expr,
        *,
        rule: str,
        dependencies: tuple[str, ...],
        metadata: dict,
    ) -> None:
        coordinate_fact = _valued_fact(facts, "area", polygon_id, rule="shoelace_formula")
        if coordinate_fact is None:
            return
        residual = sp.simplify(sp.sympify(coordinate_fact.value) - value)
        facts.add(
            "area_formula_residual",
            f"{polygon_id}|{rule}",
            residual,
            rule=f"{rule}_check",
            dependencies=(*dependencies, coordinate_fact.id),
            metadata={"expected_zero": True, "area_rule": rule, **metadata},
        )
        if residual != 0:
            return
        facts.add(
            "area",
            polygon_id,
            value,
            rule=rule,
            dependencies=dependencies,
            metadata={"admission": "coordinate_verified", **metadata},
        )

    def _explicit_equal_length_facts(self, state: FormalGeometryState, facts: FactStore, relation: dict) -> None:
        objects = tuple(str(item) for item in (relation.get("segments") or relation.get("objects") or ()))
        if len(objects) != 2:
            return
        left, right = objects
        try:
            left_value = _line_like_length(state, left)
            right_value = _line_like_length(state, right)
        except Exception:
            return
        object_id = "|".join(sorted(objects))
        facts.add("equal_length", object_id, source="asserted", rule="explicit_relation", dependencies=objects, metadata={"residual": str(sp.simplify(left_value - right_value)), "relation_id": relation["id"]})
        facts.add("equal_length_residual", object_id, sp.simplify(left_value - right_value), source="derived", rule="explicit_equal_length_check", dependencies=objects, metadata={"expected_zero": True})

    def _explicit_equal_angle_facts(self, state: FormalGeometryState, facts: FactStore, relation: dict) -> None:
        angles = tuple(tuple(str(item) for item in angle) for angle in relation.get("angles") or ())
        if len(angles) != 2 or any(len(angle) != 3 for angle in angles):
            return
        try:
            left_value = _angle_value(state, angles[0])
            right_value = _angle_value(state, angles[1])
        except Exception:
            return
        object_id = f"{''.join(angles[0])}|{''.join(angles[1])}"
        dependencies = tuple(f"point:{item}" for angle in angles for item in angle)
        facts.add("equal_angle", object_id, source="asserted", rule="explicit_relation", dependencies=dependencies, metadata={"residual": str(sp.simplify(left_value - right_value)), "relation_id": relation["id"]})
        facts.add("equal_angle_residual", object_id, sp.simplify(left_value - right_value), source="derived", rule="explicit_equal_angle_check", dependencies=dependencies, metadata={"expected_zero": True})

    def _explicit_polygon_relation_facts(self, state: FormalGeometryState, facts: FactStore, relation: dict, relation_type: str) -> None:
        polygon_ids = tuple(str(item) for item in relation.get("polygons") or relation.get("objects") or ())
        mapping = {str(key): str(value) for key, value in (relation.get("mapping") or {}).items()}
        if len(polygon_ids) != 2 or not mapping or polygon_ids[0] not in state.polygons or polygon_ids[1] not in state.polygons:
            return
        left = state.polygons[polygon_ids[0]]
        right = state.polygons[polygon_ids[1]]
        if len(left.vertices) != len(right.vertices) or set(mapping) != set(left.vertices) or set(mapping.values()) != set(right.vertices):
            return
        left_points = {point_id: state.require_point(point_id).coords for point_id in left.vertices}
        right_points = {point_id: state.require_point(point_id).coords for point_id in right.vertices}
        mapped_right = tuple(mapping[point_id] for point_id in left.vertices)
        left_sides = _polygon_side_squares(left_points, left.vertices)
        right_sides = _polygon_side_squares(right_points, mapped_right)
        if relation_type == "congruent":
            residual = sp.simplify(sum((left_value - right_value) ** 2 for left_value, right_value in zip(left_sides, right_sides)))
        else:
            residual = sp.simplify(sum((left_sides[index] * right_sides[0] - left_sides[0] * right_sides[index]) ** 2 for index in range(1, len(left_sides))))
        object_id = f"{polygon_ids[0]}|{polygon_ids[1]}"
        dependencies = tuple(f"point:{point_id}" for point_id in (*left.vertices, *mapped_right))
        facts.add(relation_type, object_id, source="asserted", rule="explicit_relation", dependencies=dependencies, metadata={"mapping": mapping, "relation_id": relation["id"]})
        facts.add(f"{relation_type}_residual", object_id, residual, source="derived", rule=f"explicit_{relation_type}_check", dependencies=dependencies, metadata={"expected_zero": True, "mapping": mapping})

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
                if a.coords == b.coords:
                    continue
                central_angle = angle_between(sub(a.coords, center.coords), sub(b.coords, center.coords))
                for observer_id in circle_points:
                    if observer_id in (a_id, b_id):
                        continue
                    observer = state.require_point(observer_id)
                    if observer.coords == a.coords or observer.coords == b.coords:
                        continue
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
        preferred_rule = state.target.metadata.get("preferred_rule")
        if preferred_rule:
            matches = [match for match in matches if match.rule == preferred_rule]
        if matches:
            if state.target.mode == "proof":
                match = matches[0]
                facts.add(
                    "target",
                    state.target.object,
                    sp.Integer(1),
                    source="derived",
                    rule="proof_target_lookup",
                    dependencies=(match.id,),
                    metadata={
                        **state.target.metadata,
                        "target_mode": "proof",
                        "target_type": state.target.type,
                        "target_reference": state.target.object,
                        "resolved_fact_id": match.id,
                        "claim_status": "proved",
                        "preferred_rule": preferred_rule,
                    },
                )
                return
            valued_matches = [match for match in matches if match.value is not None]
            if state.target.mode == "constraint":
                if not valued_matches:
                    matches = []
                else:
                    match = valued_matches[0]
                    outcome, outcome_value, residual = _constraint_outcome(match.value, state.target.expectation or {})
                    facts.add(
                        "target",
                        state.target.object,
                        outcome_value,
                        source="derived",
                        rule="constraint_target_check",
                        dependencies=(match.id,),
                        metadata={
                            **state.target.metadata,
                            "target_mode": "constraint",
                            "target_type": state.target.type,
                            "target_reference": state.target.object,
                            "resolved_fact_id": match.id,
                            "constraint_status": outcome,
                            "constraint_residual": str(residual),
                            "expectation": dict(state.target.expectation or {}),
                            "preferred_rule": preferred_rule,
                        },
                    )
                    return
            if not matches:
                pass
            else:
                match = valued_matches[0] if valued_matches else matches[0]
                facts.add(
                    "target",
                    state.target.object,
                    match.value,
                    source="derived",
                    rule="target_lookup",
                    dependencies=(match.id,),
                    metadata={
                        **state.target.metadata,
                        "target_mode": "value",
                        "target_type": state.target.type,
                        "target_reference": state.target.object,
                        "resolved_fact_id": match.id,
                        "preferred_rule": preferred_rule,
                    },
                )
                return
        facts.add(
            "target_unresolved",
            state.target.object,
            source="derived",
            rule="target_lookup_failed",
            metadata={
                **state.target.metadata,
                "target_mode": state.target.mode,
                "target_type": state.target.type,
                "target_reference": state.target.object,
                "available_fact_types": sorted({fact.type for fact in facts}),
                "available_rules": sorted({fact.rule for fact in facts.find(state.target.type, state.target.object)}),
                "preferred_rule": preferred_rule,
            },
        )


def _constraint_outcome(actual_value, expectation: dict) -> tuple[str, sp.Integer, sp.Expr]:
    actual = sp.sympify(actual_value)
    expected = sp.sympify(expectation["value"])
    residual = sp.simplify(actual - expected)
    operator = str(expectation["operator"])
    truth: bool | None
    if operator == "eq":
        truth = True if residual == 0 else False if residual.is_zero is False else None
    elif operator == "ne":
        truth = False if residual == 0 else True if residual.is_zero is False else None
    elif operator == "lt":
        truth = True if residual.is_negative else False if residual.is_nonnegative else None
    elif operator == "le":
        truth = True if residual.is_nonpositive else False if residual.is_positive else None
    elif operator == "gt":
        truth = True if residual.is_positive else False if residual.is_nonpositive else None
    elif operator == "ge":
        truth = True if residual.is_nonnegative else False if residual.is_negative else None
    else:
        raise ValueError(f"Unsupported constraint operator: {operator}")
    if truth is True:
        return "satisfied", sp.Integer(1), residual
    if truth is False:
        return "violated", sp.Integer(0), residual
    return "undetermined", sp.Integer(-1), residual


def _relation_object_id(relation: dict, index: int) -> str:
    for key in ("point", "line", "object"):
        if key in relation:
            return str(relation[key])
    if "objects" in relation:
        return "|".join(str(item) for item in relation["objects"])
    return str(relation.get("id", f"relation_{index}"))


def _relation_dependencies(relation: dict) -> tuple[str, ...]:
    """Extract stable object tokens used by a relation's derived facts."""

    keys = {
        "point", "points", "line", "base_line", "circle", "circle1", "circle2",
        "segment", "segments", "polygon", "polygons", "plane", "planes",
        "polyhedron", "from", "of", "objects", "boundary", "chord",
    }
    values: list[str] = []
    for key in sorted(keys):
        value = relation.get(key)
        if isinstance(value, (list, tuple)):
            values.extend(str(item) for item in value)
        elif value is not None:
            values.append(str(value))
    mapping = relation.get("mapping")
    if isinstance(mapping, dict):
        values.extend(str(item) for item in mapping)
        values.extend(str(item) for item in mapping.values())
    return tuple(dict.fromkeys(values))


def _point_on_linear_domain(point, anchor, direction, domain: str) -> bool:
    if not point_on_line(point, anchor, direction):
        return False
    if domain == "line":
        return True
    parameter = sp.simplify(dot(sub(point, anchor), direction) / dot(direction, direction))
    return parameter.is_nonnegative is True or bool(parameter.is_number and sp.N(parameter) >= 0)


def _point_on_intersection_object(state: FormalGeometryState, point, object_id: str) -> bool:
    if object_id in state.lines or object_id in state.rays or object_id in state.segments:
        linear = state.get_line_like(object_id)
        domain = state.line_like_domain(object_id)
        if domain == "segment":
            segment = state.segments[object_id]
            return point_on_segment(point, state.require_point(segment.start).coords, state.require_point(segment.end).coords)
        return _point_on_linear_domain(point, linear.point, linear.direction, domain)
    if object_id in state.circles:
        circle = state.circles[object_id]
        return sp.simplify(distance(point, state.require_point(circle.center).coords) - circle.radius) == 0
    if object_id in state.arcs:
        arc = state.arcs[object_id]
        circle = state.circles[arc.circle]
        return point_on_arc_domain_2d(
            point,
            state.require_point(circle.center).coords,
            circle.radius,
            state.require_point(arc.start).coords,
            state.require_point(arc.end).coords,
            orientation=arc.orientation,
            central_angle=arc.central_angle,
            allow_full_circle=bool(arc.metadata.get("allow_full_circle")),
        )
    return False


def _closed_polyline_length(points) -> sp.Expr:
    if len(points) < 2:
        return sp.Integer(0)
    total = sp.Integer(0)
    for index, point in enumerate(points):
        total += distance(point, points[(index + 1) % len(points)])
    return sp.simplify(total)


def _as_point_pair(value) -> tuple[str, str] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    return str(value[0]), str(value[1])


def _valued_fact(facts: FactStore, fact_type: str, object_id: str, *, rule: str | None = None):
    matches = facts.find(fact_type, object_id)
    if rule is not None:
        matches = [fact for fact in matches if fact.rule == rule]
    valued = [fact for fact in matches if fact.value is not None]
    return sorted(valued, key=lambda fact: fact.id)[0] if valued else None


def _canonical_zero_residual(value: sp.Expr) -> sp.Expr:
    simplified = sp.simplify(value)
    if simplified == 0:
        return sp.Integer(0)
    certified = sp.nsimplify(simplified, [sp.pi])
    if certified == 0 and abs(float(sp.N(simplified, 80))) < 1e-70:
        return sp.Integer(0)
    return simplified


def _preferred_region_perimeter_fact(state: FormalGeometryState, facts: FactStore, region_id: str):
    region = state.regions.get(region_id)
    if region is None:
        return _valued_fact(facts, "perimeter", region_id)
    preferred_rule = region.metadata.get("preferred_perimeter_rule")
    if preferred_rule:
        direct = _valued_fact(facts, "perimeter", region_id, rule=str(preferred_rule))
        if direct is not None:
            return direct
        semantic = _valued_fact(facts, "perimeter", region_id, rule="region_from_semantic_polygon_perimeter")
        if semantic is not None and semantic.metadata.get("polygon_perimeter_rule") == preferred_rule:
            return semantic
    return _valued_fact(facts, "perimeter", region_id)


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


def _segment_is_circle_radius(state: FormalGeometryState, segment_id: str, circle_id: str) -> bool:
    if segment_id not in state.segments or circle_id not in state.circles:
        return False
    segment = state.segments[segment_id]
    circle = state.circles[circle_id]
    if circle.center not in {segment.start, segment.end}:
        return False
    endpoint_id = segment.end if segment.start == circle.center else segment.start
    center = state.require_point(circle.center)
    endpoint = state.require_point(endpoint_id)
    return sp.simplify(distance(center.coords, endpoint.coords) - circle.radius) == 0


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


def _composite_polygon_boolean_area(state: FormalGeometryState, region) -> sp.Expr | None:
    if len(region.components) != 2:
        return None
    polygons = []
    for component_id in region.components:
        component = state.regions.get(component_id)
        if component is None or component.kind != "polygon_region" or component.holes:
            return None
        polygon_id = component.metadata.get("polygon")
        if polygon_id in state.polygons:
            vertices = state.polygons[polygon_id].vertices
        elif len(component.boundary) == 1 and component.boundary[0] in state.polygons:
            vertices = state.polygons[component.boundary[0]].vertices
        elif all(point_id in state.points for point_id in component.boundary):
            vertices = component.boundary
        else:
            return None
        points = tuple(state.require_point(point_id).coords for point_id in vertices)
        if any(len(point) != 2 for point in points):
            return None
        polygons.append(points)
    try:
        return boolean_polygon_area_2d(polygons[0], polygons[1], region.operation)
    except Exception:
        return None


def _line_like_length(state: FormalGeometryState, object_id: str) -> sp.Expr:
    if object_id in state.segments:
        segment = state.segments[object_id]
        return distance(state.require_point(segment.start).coords, state.require_point(segment.end).coords)
    if object_id in state.lines or object_id in state.rays:
        line = state.get_line_like(object_id)
        return norm(line.direction)
    raise ValueError(f"Unknown length object: {object_id}")


def _angle_value(state: FormalGeometryState, angle: tuple[str, str, str]) -> sp.Expr:
    left, vertex, right = (state.require_point(point_id) for point_id in angle)
    return angle_between(sub(left.coords, vertex.coords), sub(right.coords, vertex.coords))


def _polygon_side_squares(points: Dict[str, tuple], vertices: tuple[str, ...]) -> tuple[sp.Expr, ...]:
    return tuple(
        sp.simplify(sum(value ** 2 for value in sub(points[vertices[(index + 1) % len(vertices)]], points[vertex])))
        for index, vertex in enumerate(vertices)
    )
