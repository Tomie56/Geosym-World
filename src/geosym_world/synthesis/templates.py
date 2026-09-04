"""Template families for procedural synthetic trace construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Dict, Iterable, List
import json
from copy import deepcopy


Trace = List[Dict[str, Any]]


@dataclass(frozen=True)
class SyntheticSpec:
    id: str
    family: str
    trace: Trace
    question: str
    answer_hint: str = ""
    difficulty: str = "medium"
    diagnostic_axes: tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TemplateFamily:
    name: str
    parameter_grid: Dict[str, tuple[Any, ...]]
    builder: Callable[[Dict[str, Any]], SyntheticSpec]
    description: str = ""

    def variants(self) -> Iterable[SyntheticSpec]:
        keys = tuple(self.parameter_grid)
        for values in product(*(self.parameter_grid[key] for key in keys)):
            yield self.builder(dict(zip(keys, values)))


def built_in_template_families() -> list[TemplateFamily]:
    """Return compact families that expand into many controlled trace variants."""

    return [
        TemplateFamily(
            name="triangle_area_from_height",
            description="Triangle with explicit altitude and midpoint markers.",
            parameter_grid={"base": (4, 5, 6, 7), "height": (2, 3, 4), "skew": (1, 2, 3)},
            builder=_triangle_area,
        ),
        TemplateFamily(
            name="triangle_partition_area",
            description="An altitude partitions a triangle into two regions whose areas sum to the target.",
            parameter_grid={"left_base": (2, 3, 4), "right_base": (3, 4, 5), "height": (2, 3, 4)},
            builder=_triangle_partition_area,
        ),
        TemplateFamily(
            name="triangle_partition_area_ratio",
            description="An altitude partitions a triangle and targets the exact ratio of admitted subregion areas.",
            parameter_grid={"left_base": (2, 3, 4), "right_base": (3, 4, 5), "height": (2, 3, 4)},
            builder=_triangle_partition_area_ratio,
        ),
        TemplateFamily(
            name="angled_line_intersection",
            description="Two lines through endpoints create a triangle by angle intersection.",
            parameter_grid={"base": (4, 5, 6), "left_angle": ("pi/6", "pi/4", "pi/3"), "right_angle": ("2*pi/3", "3*pi/4")},
            builder=_angled_intersection,
        ),
        TemplateFamily(
            name="circle_sector",
            description="Sector area from a rendered radius and exact central angle.",
            parameter_grid={"radius": (3, 4, 5, 6), "central_angle": ("pi/3", "pi/2", "2*pi/3", "pi")},
            builder=_circle_sector,
        ),
        TemplateFamily(
            name="sector_arc_length",
            description="Sector arc length from a rendered radius and exact central angle.",
            parameter_grid={"radius": (3, 4, 5, 6), "central_angle": ("pi/3", "pi/2", "2*pi/3", "pi")},
            builder=_sector_arc_length,
        ),
        TemplateFamily(
            name="sector_perimeter",
            description="Total sector boundary length from a rendered radius and exact central angle.",
            parameter_grid={"radius": (3, 4, 5, 6), "central_angle": ("pi/3", "pi/2", "2*pi/3", "pi")},
            builder=_sector_perimeter,
        ),
        TemplateFamily(
            name="circular_segment_area",
            description="Circular-segment shaded area as a semantic sector-minus-chord-triangle derivation.",
            parameter_grid={"radius": (3, 4, 5, 6), "central_angle": ("pi/3", "pi/2", "2*pi/3")},
            builder=_circular_segment_area,
        ),
        TemplateFamily(
            name="circular_segment_perimeter",
            description="Circular-segment perimeter as a semantic arc-plus-derived-chord derivation.",
            parameter_grid={"radius": (3, 4, 5, 6), "central_angle": ("pi/3", "pi/2", "2*pi/3")},
            builder=_circular_segment_perimeter,
        ),
        TemplateFamily(
            name="circular_segment_area_ratio",
            description="Exact ratio of a shaded circular segment to its containing sector.",
            parameter_grid={"radius": (3, 4, 5, 6), "central_angle": ("pi/3", "pi/2", "2*pi/3")},
            builder=_circular_segment_area_ratio,
        ),
        TemplateFamily(
            name="equal_circle_lens_area",
            description="Equal-circle lens area from public common radius and center distance with constructed intersections.",
            parameter_grid={"radius": (3, 4, 5, 6), "center_distance": (2, 3, 4)},
            builder=_equal_circle_lens_area,
        ),
        TemplateFamily(
            name="equal_circle_lens_perimeter",
            description="Equal-circle lens perimeter from the two admitted minor arc lengths.",
            parameter_grid={"radius": (3, 4, 5, 6), "center_distance": (2, 3, 4)},
            builder=_equal_circle_lens_perimeter,
        ),
        TemplateFamily(
            name="circle_lens_area",
            description="Two-circle lens area from public radii and center distance with constructed intersections.",
            parameter_grid={
                "case": (
                    (4, 3, 5),
                    (4, 3, 6),
                    (4, 4, 5),
                    (4, 4, 6),
                    (5, 3, 6),
                    (5, 4, 6),
                    (5, 4, 7),
                    (5, 5, 6),
                    (6, 4, 7),
                    (6, 5, 7),
                    (6, 5, 8),
                    (6, 6, 7),
                )
            },
            builder=_circle_lens_area,
        ),
        TemplateFamily(
            name="circle_lens_perimeter",
            description="Unequal-radius lens perimeter from two public radii and center distance.",
            parameter_grid={
                "case": (
                    (4, 3, 5),
                    (4, 3, 6),
                    (4, 4, 5),
                    (4, 4, 6),
                    (5, 3, 6),
                    (5, 4, 6),
                    (5, 4, 7),
                    (5, 5, 6),
                    (6, 4, 7),
                    (6, 5, 7),
                    (6, 5, 8),
                    (6, 6, 7),
                )
            },
            builder=_circle_lens_perimeter,
        ),
        TemplateFamily(
            name="parallel_cut_similarity",
            description="Triangle with a parallel interior cut producing similar triangles.",
            parameter_grid={"base": (6, 8, 10), "height": (4, 6), "ratio": ("1/3", "1/2", "2/3")},
            builder=_parallel_cut,
        ),
        TemplateFamily(
            name="similarity_dual_proof",
            description="Prove one pair of scaled triangles similar through independent SSS and AA fact paths.",
            parameter_grid={"base": (6, 8, 10), "height": (4, 6), "ratio": ("1/3", "1/2", "2/3")},
            builder=_similarity_dual_proof,
        ),
        TemplateFamily(
            name="rectangle_area_diagonal",
            description="Rectangle with diagonal and area target.",
            parameter_grid={"width": (3, 4, 5, 6), "height": (2, 3, 4)},
            builder=_rectangle_area,
        ),
        TemplateFamily(
            name="rectangular_frame_area",
            description="A rectangular frame uses outer and inner dimensions to derive shaded area by difference.",
            parameter_grid={"outer_width": (8, 10, 12), "outer_height": (6, 8), "border": (1, 2)},
            builder=_rectangular_frame_area,
        ),
        TemplateFamily(
            name="double_rectangular_hole_area",
            description="A rectangular plate subtracts two disjoint rectangular holes from public dimensions.",
            parameter_grid={"outer_width": (12, 14), "outer_height": (7, 8), "left_width": (2, 3), "left_height": (2, 3), "right_width": (2, 3), "right_height": (2, 3)},
            builder=_double_rectangular_hole_area,
        ),
        TemplateFamily(
            name="double_rectangular_hole_perimeter",
            description="A rectangular plate sums its outer and two inner boundary perimeters from public dimensions.",
            parameter_grid={"outer_width": (12, 14), "outer_height": (7, 8), "left_width": (2, 3), "left_height": (2, 3), "right_width": (2, 3), "right_height": (2, 3)},
            builder=_double_rectangular_hole_perimeter,
        ),
        TemplateFamily(
            name="parallelogram_area",
            description="Parallelogram with explicit height marker and area target.",
            parameter_grid={"base": (4, 5, 6, 7), "height": (2, 3, 4), "skew": (1, 2, 3)},
            builder=_parallelogram_area,
        ),
        TemplateFamily(
            name="trapezoid_area",
            description="Trapezoid with bases and height marker.",
            parameter_grid={"bottom": (6, 7, 8), "top": (2, 3, 4), "height": (2, 3, 4), "offset": (1, 2)},
            builder=_trapezoid_area,
        ),
        TemplateFamily(
            name="kite_area_diagonals",
            description="Kite with perpendicular diagonals and an explicit crossing.",
            parameter_grid={"horizontal": (6, 8, 10), "upper": (3, 4, 5), "lower": (2, 3, 4)},
            builder=_kite_area,
        ),
        TemplateFamily(
            name="rhombus_diagonal_theorem",
            description="Rhombus with perpendicular bisecting diagonals and an exact area residual.",
            parameter_grid={"horizontal": (6, 8, 10), "vertical": (4, 6, 8)},
            builder=_rhombus_diagonal,
        ),
        TemplateFamily(
            name="cyclic_quadrilateral",
            description="Four points on one circle with a coordinate-backed cyclicity residual.",
            parameter_grid={"radius": (4, 5, 6), "lower_angle": ("4*pi/3", "3*pi/2", "5*pi/3")},
            builder=_cyclic_quadrilateral,
        ),
        TemplateFamily(
            name="tangential_quadrilateral",
            description="Tangential isosceles trapezoid with an incircle and Pitot residual.",
            parameter_grid={"bottom": (8, 9, 12), "top": (2, 3, 4)},
            builder=_tangential_quadrilateral,
        ),
        TemplateFamily(
            name="concave_composite_area",
            description="Concave L-shaped polygon requiring subtraction of a rectangular notch.",
            parameter_grid={"width": (6, 8, 10), "height": (5, 7), "notch_width": (2, 3), "notch_height": (2, 3)},
            builder=_concave_composite_area,
        ),
        TemplateFamily(
            name="annulus_area",
            description="Concentric circles with a shaded annular target region.",
            parameter_grid={"outer_radius": (5, 6, 8), "inner_radius": (2, 3, 4)},
            builder=_annulus_area,
        ),
        TemplateFamily(
            name="annulus_perimeter",
            description="Total inner and outer boundary length of a shaded annulus from public radii.",
            parameter_grid={"outer_radius": (5, 6, 8), "inner_radius": (2, 3, 4)},
            builder=_annulus_perimeter,
        ),
        TemplateFamily(
            name="tetrahedron_section",
            description="Tetrahedron section at z=h with section area or perimeter target.",
            parameter_grid={"scale": (1, 2, 3), "height": ("1/3", "1/2", "2/3"), "target": ("section_area", "section_perimeter")},
            builder=_tetra_section,
        ),
    ]


def select_template_families(names: Iterable[str] | None = None, *, grid_overrides: Dict[str, Dict[str, tuple[Any, ...]]] | None = None) -> list[TemplateFamily]:
    families = built_in_template_families()
    if names:
        wanted = {name for name in names}
        families = [family for family in families if family.name in wanted]
        missing = wanted.difference({family.name for family in families})
        if missing:
            raise ValueError(f"Unknown template families: {sorted(missing)}")
    if grid_overrides:
        families = [_override_family_grid(family, grid_overrides.get(family.name)) for family in families]
    return families


def load_grid_overrides(path: str | None = None, inline_json: str | None = None) -> Dict[str, Dict[str, tuple[Any, ...]]]:
    if path and inline_json:
        raise ValueError("Use either path or inline_json, not both")
    if not path and not inline_json:
        return {}
    payload = json.loads(inline_json) if inline_json else json.loads(open(path, encoding="utf-8").read())
    if not isinstance(payload, dict):
        raise ValueError("Grid override payload must be an object")
    result: Dict[str, Dict[str, tuple[Any, ...]]] = {}
    for family_name, family_grid in payload.items():
        if not isinstance(family_grid, dict):
            raise ValueError(f"Grid override for {family_name} must be an object")
        result[str(family_name)] = {str(key): tuple(values) for key, values in family_grid.items()}
    return result


def _override_family_grid(family: TemplateFamily, override: Dict[str, tuple[Any, ...]] | None) -> TemplateFamily:
    if not override:
        return family
    unknown = set(override).difference(family.parameter_grid)
    if unknown:
        raise ValueError(f"Unknown parameters for {family.name}: {sorted(unknown)}")
    return TemplateFamily(
        name=family.name,
        parameter_grid={**family.parameter_grid, **override},
        builder=family.builder,
        description=family.description,
    )


def _triangle_area(params: Dict[str, Any]) -> SyntheticSpec:
    base = params["base"]
    height = params["height"]
    skew = params["skew"]
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": [str(base), "0"]},
        {"op": "add_point", "id": "C", "coord": [str(skew), str(height)]},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"], "metadata": {"show_length": True, "label_position": 0.63}},
        {"op": "add_segment", "id": "AC", "points": ["A", "C"]},
        {"op": "add_segment", "id": "BC", "points": ["B", "C"]},
        {"op": "midpoint", "id": "M", "of": ["A", "B"]},
        {"op": "foot_of_perpendicular", "id": "H", "point": "C", "line": "AB", "metadata": {"show_length": True}},
        {"op": "add_polygon", "id": "tri_ABC", "vertices": ["A", "B", "C"], "metadata": {"area_formula": {"rule": "triangle_base_height", "base": "AB", "height": ["C", "H"]}}},
        {"op": "set_target", "type": "area", "object": "tri_ABC", "metadata": {"preferred_rule": "triangle_base_height"}},
    ]
    return SyntheticSpec(
        id=f"tri_area_b{base}_h{height}_s{skew}",
        family="triangle_area_from_height",
        trace=trace,
        question="Find the area of triangle ABC.",
        difficulty="easy" if height <= 2 else "medium",
        diagnostic_axes=("triangle", "area", "altitude", "midpoint"),
    )


def _triangle_partition_area(params: Dict[str, Any]) -> SyntheticSpec:
    left_base = params["left_base"]
    right_base = params["right_base"]
    height = params["height"]
    total_base = left_base + right_base
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": [str(total_base), "0"]},
        {"op": "add_point", "id": "C", "coord": [str(left_base), str(height)]},
        {"op": "add_line", "id": "base_line", "points": ["A", "B"], "metadata": {"render": False}},
        {"op": "foot_of_perpendicular", "id": "H", "point": "C", "line": "base_line"},
        {"op": "add_segment", "id": "AC", "points": ["A", "C"]},
        {"op": "add_segment", "id": "BC", "points": ["B", "C"]},
        {"op": "add_segment", "id": "AH", "points": ["A", "H"], "metadata": {"show_length": True}},
        {"op": "add_segment", "id": "HB", "points": ["H", "B"], "metadata": {"show_length": True}},
        {"op": "add_segment", "id": "CH", "points": ["C", "H"], "metadata": {"show_length": True}},
        {"op": "add_polygon", "id": "tri_ABC", "vertices": ["A", "B", "C"], "metadata": {"render": False}},
        {"op": "add_polygon", "id": "tri_ACH", "vertices": ["A", "C", "H"], "metadata": {"render": False, "area_formula": {"rule": "triangle_base_height", "base": "AH", "height": "CH"}}},
        {"op": "add_polygon", "id": "tri_HBC", "vertices": ["H", "B", "C"], "metadata": {"render": False, "area_formula": {"rule": "triangle_base_height", "base": "HB", "height": "CH"}}},
        {"op": "mark_polygon_region", "id": "region_ABC", "polygon": "tri_ABC", "metadata": {"render": False}},
        {"op": "mark_polygon_region", "id": "region_ACH", "polygon": "tri_ACH", "metadata": {"preferred_area_rule": "triangle_base_height"}},
        {"op": "mark_polygon_region", "id": "region_HBC", "polygon": "tri_HBC", "metadata": {"preferred_area_rule": "triangle_base_height", "fill_variant": "alternate"}},
        {"op": "add_composite_region", "id": "partition_ABC", "operation": "partition", "components": ["region_ACH", "region_HBC"], "metadata": {"parent_region": "region_ABC"}},
        {"op": "set_target", "type": "area", "object": "partition_ABC", "metadata": {"preferred_rule": "region_partition_area_sum"}},
    ]
    return SyntheticSpec(
        id=f"tri_partition_l{left_base}_r{right_base}_h{height}",
        family="triangle_partition_area",
        trace=trace,
        question="Altitude CH partitions triangle ABC into triangles ACH and BCH. Find the area of triangle ABC.",
        difficulty="hard",
        diagnostic_axes=("triangle", "altitude", "region_partition", "composite_area", "multi_step_area"),
    )


def _triangle_partition_area_ratio(params: Dict[str, Any]) -> SyntheticSpec:
    area_spec = _triangle_partition_area(params)
    trace = [*area_spec.trace[:-2]]
    partition = dict(area_spec.trace[-2])
    partition["metadata"] = {**partition["metadata"], "render": False}
    trace.extend(
        [
            partition,
            {
                "op": "set_target",
                "type": "ratio",
                "object": "ratio_ACH_HBC",
                "metadata": {
                    "ratio_kind": "area",
                    "numerator": "region_ACH",
                    "denominator": "region_HBC",
                    "preferred_rule": "region_area_ratio",
                },
            },
        ]
    )
    return SyntheticSpec(
        id=area_spec.id.replace("tri_partition_", "tri_partition_ratio_"),
        family="triangle_partition_area_ratio",
        trace=trace,
        question="Altitude CH partitions triangle ABC. Find the ratio [ACH]:[BCH].",
        difficulty="hard",
        diagnostic_axes=("triangle", "altitude", "region_partition", "area_ratio", "multi_step_area"),
    )


def _angled_intersection(params: Dict[str, Any]) -> SyntheticSpec:
    base = params["base"]
    left_angle = params["left_angle"]
    right_angle = params["right_angle"]
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": [str(base), "0"]},
        {"op": "line_through_angle", "id": "l_AC", "point": "A", "angle": left_angle},
        {"op": "line_through_angle", "id": "l_BC", "point": "B", "angle": right_angle},
        {"op": "intersection_line_line", "id": "C", "line1": "l_AC", "line2": "l_BC"},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"], "metadata": {"show_length": True}},
        {"op": "add_segment", "id": "AC", "points": ["A", "C"]},
        {"op": "add_segment", "id": "BC", "points": ["B", "C"]},
        {"op": "add_polygon", "id": "tri_ABC", "vertices": ["A", "B", "C"]},
        {"op": "angle_bisector", "id": "bis_ABC", "vertex": "B", "ray1": ["B", "A"], "ray2": ["B", "C"]},
        {"op": "set_target", "type": "area", "object": "tri_ABC"},
    ]
    token = f"{left_angle}_{right_angle}".replace("/", "_").replace("*", "")
    return SyntheticSpec(
        id=f"angled_intersection_b{base}_{token}",
        family="angled_line_intersection",
        trace=trace,
        question="Two angled rays meet at C. Find the area of triangle ABC.",
        difficulty="medium",
        diagnostic_axes=("angle_construction", "intersection", "area", "angle_bisector"),
    )


def _sector_trace(params: Dict[str, Any]) -> Trace:
    radius = params["radius"]
    central_angle = str(params["central_angle"])
    return [
        {"op": "add_point", "id": "O", "coord": ["0", "0"]},
        {"op": "add_point", "id": "A", "coord": [str(radius), "0"]},
        {"op": "add_point", "id": "B", "coord": [f"{radius}*cos({central_angle})", f"{radius}*sin({central_angle})"]},
        {"op": "add_circle", "id": "circle_O", "center": "O", "radius": str(radius)},
        {"op": "add_segment", "id": "OA", "points": ["O", "A"], "metadata": {"show_length": True}},
        {"op": "add_segment", "id": "OB", "points": ["O", "B"]},
        {"op": "add_arc", "id": "arc_AB", "circle": "circle_O", "start": "A", "end": "B", "central_angle": central_angle},
        {
            "op": "mark_sector_region",
            "id": "sector_AOB",
            "circle": "circle_O",
            "start": "A",
            "end": "B",
            "central_angle": central_angle,
            "metadata": {"radius_segment": "OA", "show_central_angle": True},
        },
    ]


def _sector_token(value: Any) -> str:
    return str(value).replace("*", "").replace("/", "_")


def _circle_sector(params: Dict[str, Any]) -> SyntheticSpec:
    radius = params["radius"]
    central_angle = params["central_angle"]
    trace = _sector_trace(params)
    trace.append({"op": "set_target", "type": "area", "object": "sector_AOB", "metadata": {"preferred_rule": "sector_radius_angle_area"}})
    return SyntheticSpec(
        id=f"circle_sector_r{radius}_a{_sector_token(central_angle)}",
        family="circle_sector",
        trace=trace,
        question="Find the area of sector AOB.",
        difficulty="medium",
        diagnostic_axes=("circle", "sector_area", "central_angle", "curved_boundary"),
    )


def _sector_arc_length(params: Dict[str, Any]) -> SyntheticSpec:
    radius = params["radius"]
    central_angle = params["central_angle"]
    trace = _sector_trace(params)
    trace.append({"op": "set_target", "type": "arc_length", "object": "sector_AOB", "metadata": {"preferred_rule": "sector_radius_angle_arc_length"}})
    return SyntheticSpec(
        id=f"sector_arc_length_r{radius}_a{_sector_token(central_angle)}",
        family="sector_arc_length",
        trace=trace,
        question="Find the length of arc AB.",
        difficulty="medium",
        diagnostic_axes=("circle", "arc_length", "central_angle", "curved_boundary"),
    )


def _sector_perimeter(params: Dict[str, Any]) -> SyntheticSpec:
    radius = params["radius"]
    central_angle = params["central_angle"]
    trace = _sector_trace(params)
    trace.append({"op": "set_target", "type": "perimeter", "object": "sector_AOB", "metadata": {"preferred_rule": "sector_radius_angle_perimeter"}})
    return SyntheticSpec(
        id=f"sector_perimeter_r{radius}_a{_sector_token(central_angle)}",
        family="sector_perimeter",
        trace=trace,
        question="Find the total perimeter of sector AOB.",
        difficulty="medium",
        diagnostic_axes=("circle", "sector_perimeter", "central_angle", "curved_boundary"),
    )


def _circular_segment_area(params: Dict[str, Any]) -> SyntheticSpec:
    radius = params["radius"]
    central_angle = params["central_angle"]
    trace = _circular_segment_trace(params)
    trace.append({"op": "set_target", "type": "area", "object": "segment_AB", "metadata": {"preferred_rule": "circular_segment_difference"}})
    return SyntheticSpec(
        id=f"circular_segment_area_r{radius}_a{_sector_token(central_angle)}",
        family="circular_segment_area",
        trace=trace,
        question="Find the area of the shaded circular segment bounded by arc AB and chord AB.",
        difficulty="hard",
        diagnostic_axes=("circle", "circular_segment", "shaded_area", "sector_minus_triangle", "multi_step_area"),
    )


def _circular_segment_perimeter(params: Dict[str, Any]) -> SyntheticSpec:
    radius = params["radius"]
    central_angle = params["central_angle"]
    trace = _circular_segment_trace(params)
    trace.append({"op": "set_target", "type": "perimeter", "object": "segment_AB", "metadata": {"preferred_rule": "circular_segment_arc_chord_perimeter"}})
    return SyntheticSpec(
        id=f"circular_segment_perimeter_r{radius}_a{_sector_token(central_angle)}",
        family="circular_segment_perimeter",
        trace=trace,
        question="Find the total boundary length of the shaded circular segment.",
        difficulty="hard",
        diagnostic_axes=("circle", "circular_segment", "perimeter", "arc_plus_chord", "multi_step_length"),
    )


def _circular_segment_area_ratio(params: Dict[str, Any]) -> SyntheticSpec:
    radius = params["radius"]
    central_angle = params["central_angle"]
    trace = _circular_segment_trace(params)
    trace.append(
        {
            "op": "set_target",
            "type": "ratio",
            "object": "segment_AB|sector_AOB",
            "metadata": {
                "ratio_kind": "area",
                "numerator": "segment_AB",
                "denominator": "sector_AOB",
                "preferred_rule": "region_area_ratio",
            },
        }
    )
    return SyntheticSpec(
        id=f"circular_segment_area_ratio_r{radius}_a{_sector_token(central_angle)}",
        family="circular_segment_area_ratio",
        trace=trace,
        question="Find the ratio of the shaded circular-segment area to the area of sector AOB.",
        difficulty="expert",
        diagnostic_axes=(
            "circle",
            "circular_segment",
            "shaded_area",
            "sector_minus_triangle",
            "area_ratio",
            "multi_step_area",
        ),
    )


def _circular_segment_trace(params: Dict[str, Any]) -> Trace:
    trace = _sector_trace(params)
    sector_step = next(step for step in trace if step["op"] == "mark_sector_region")
    sector_step["metadata"]["render"] = False
    trace.extend(
        [
            {"op": "add_chord", "id": "AB", "circle": "circle_O", "points": ["A", "B"]},
            {
                "op": "add_polygon",
                "id": "tri_OAB",
                "vertices": ["O", "A", "B"],
                "metadata": {
                    "render": False,
                    "area_formula": {"rule": "chord_triangle_radius_angle", "sector_region": "sector_AOB"},
                },
            },
            {
                "op": "add_region",
                "id": "segment_AB",
                "kind": "circular_segment_region",
                "boundary": ["arc_AB", "AB"],
                "metadata": {
                    "sector_region": "sector_AOB",
                    "triangle_polygon": "tri_OAB",
                    "arc": "arc_AB",
                    "chord": "AB",
                    "area_formula": {
                        "rule": "circular_segment_difference",
                        "sector_region": "sector_AOB",
                        "triangle_polygon": "tri_OAB",
                    },
                    "perimeter_formula": {
                        "rule": "circular_segment_arc_chord_perimeter",
                        "sector_region": "sector_AOB",
                        "chord": "AB",
                    },
                    "preferred_area_rule": "circular_segment_difference",
                    "preferred_perimeter_rule": "circular_segment_arc_chord_perimeter",
                },
            },
        ]
    )
    return trace


def _equal_circle_lens_area(params: Dict[str, Any]) -> SyntheticSpec:
    radius = int(params["radius"])
    center_distance = int(params["center_distance"])
    if not 0 < center_distance < 2 * radius:
        raise ValueError("equal_circle_lens_area requires 0 < center_distance < 2*radius")
    trace = [
        {"op": "add_point", "id": "O1", "coord": ["0", "0"]},
        {"op": "add_point", "id": "O2", "coord": [str(center_distance), "0"]},
        {"op": "add_point", "id": "R", "coord": ["0", str(radius)]},
        {"op": "add_segment", "id": "O1O2", "points": ["O1", "O2"], "metadata": {"show_length": True}},
        {"op": "add_segment", "id": "O1R", "points": ["O1", "R"], "metadata": {"show_length": True}},
        {"op": "add_circle", "id": "circle_1", "center": "O1", "radius": str(radius)},
        {"op": "add_circle", "id": "circle_2", "center": "O2", "radius": str(radius)},
        {"op": "intersection_circle_circle", "ids": ["P", "Q"], "circle1": "circle_1", "circle2": "circle_2"},
        {"op": "add_arc", "id": "arc_1", "circle": "circle_1", "start": "P", "end": "Q", "metadata": {"render": False}},
        {"op": "add_arc", "id": "arc_2", "circle": "circle_2", "start": "Q", "end": "P", "metadata": {"render": False}},
        {
            "op": "add_region",
            "id": "lens_12",
            "kind": "lens_region",
            "boundary": ["arc_1", "arc_2"],
            "metadata": {
                "circles": ["circle_1", "circle_2"],
                "arcs": ["arc_1", "arc_2"],
                "intersections": ["P", "Q"],
                "radius_segment": "O1R",
                "center_segment": "O1O2",
                "area_formula": {
                    "rule": "lens_equal_radius_area",
                    "radius_segment": "O1R",
                    "center_segment": "O1O2",
                },
                "preferred_area_rule": "lens_equal_radius_area",
            },
        },
        {"op": "set_target", "type": "area", "object": "lens_12", "metadata": {"preferred_rule": "lens_equal_radius_area"}},
    ]
    return SyntheticSpec(
        id=f"equal_circle_lens_area_r{radius}_d{center_distance}",
        family="equal_circle_lens_area",
        trace=trace,
        question="Two equal circles overlap. Find the area of their shaded common lens.",
        difficulty="hard",
        diagnostic_axes=("circle", "lens", "circle_circle_intersection", "curved_region", "multi_step_area"),
    )


def _circle_lens_area(params: Dict[str, Any]) -> SyntheticSpec:
    if "case" in params:
        first_radius, second_radius, center_distance = (int(value) for value in params["case"])
    else:
        first_radius = int(params["first_radius"])
        second_radius = int(params["second_radius"])
        center_distance = int(params["center_distance"])
    if not abs(first_radius - second_radius) < center_distance < first_radius + second_radius:
        raise ValueError("circle_lens_area requires |r1-r2| < center_distance < r1+r2")
    trace = [
        {"op": "add_point", "id": "O1", "coord": ["0", "0"]},
        {"op": "add_point", "id": "O2", "coord": [str(center_distance), "0"]},
        {"op": "add_point", "id": "R1", "coord": ["0", str(first_radius)]},
        {"op": "add_point", "id": "R2", "coord": [str(center_distance), str(second_radius)]},
        {"op": "add_segment", "id": "O1O2", "points": ["O1", "O2"], "metadata": {"show_length": True}},
        {"op": "add_segment", "id": "O1R1", "points": ["O1", "R1"], "metadata": {"show_length": True}},
        {"op": "add_segment", "id": "O2R2", "points": ["O2", "R2"], "metadata": {"show_length": True}},
        {"op": "add_circle", "id": "circle_1", "center": "O1", "radius": str(first_radius)},
        {"op": "add_circle", "id": "circle_2", "center": "O2", "radius": str(second_radius)},
        {"op": "intersection_circle_circle", "ids": ["P", "Q"], "circle1": "circle_1", "circle2": "circle_2"},
        {"op": "add_arc", "id": "arc_1", "circle": "circle_1", "start": "P", "end": "Q", "metadata": {"render": False}},
        {"op": "add_arc", "id": "arc_2", "circle": "circle_2", "start": "Q", "end": "P", "metadata": {"render": False}},
        {
            "op": "add_region",
            "id": "lens_12",
            "kind": "lens_region",
            "boundary": ["arc_1", "arc_2"],
            "metadata": {
                "circles": ["circle_1", "circle_2"],
                "arcs": ["arc_1", "arc_2"],
                "intersections": ["P", "Q"],
                "radius_segments": ["O1R1", "O2R2"],
                "center_segment": "O1O2",
                "area_formula": {
                    "rule": "lens_two_radius_area",
                    "radius_segments": ["O1R1", "O2R2"],
                    "center_segment": "O1O2",
                },
                "preferred_area_rule": "lens_two_radius_area",
            },
        },
        {"op": "set_target", "type": "area", "object": "lens_12", "metadata": {"preferred_rule": "lens_two_radius_area"}},
    ]
    return SyntheticSpec(
        id=f"circle_lens_area_r{first_radius}_r{second_radius}_d{center_distance}",
        family="circle_lens_area",
        trace=trace,
        question="Two intersecting circles have radii and center distance shown. Find the area of their common lens.",
        difficulty="expert",
        diagnostic_axes=("circle", "lens", "unequal_radii", "circle_circle_intersection", "curved_region", "multi_step_area"),
    )


def _equal_circle_lens_perimeter(params: Dict[str, Any]) -> SyntheticSpec:
    base = _equal_circle_lens_area(params)
    trace = deepcopy(base.trace)
    region_step = next(step for step in trace if step.get("id") == "lens_12")
    region_step["metadata"]["perimeter_formula"] = {
        "rule": "lens_equal_radius_perimeter",
        "radius_segment": "O1R",
        "center_segment": "O1O2",
    }
    region_step["metadata"]["preferred_perimeter_rule"] = "lens_equal_radius_perimeter"
    target_step = next(step for step in trace if step.get("op") == "set_target")
    target_step.update({"type": "perimeter", "metadata": {"preferred_rule": "lens_equal_radius_perimeter"}})
    return SyntheticSpec(
        id=base.id.replace("_area_", "_perimeter_"),
        family="equal_circle_lens_perimeter",
        trace=trace,
        question="Two equal circles overlap. Find the perimeter of their shaded common lens.",
        difficulty="hard",
        diagnostic_axes=("circle", "lens", "perimeter", "two_minor_arcs", "multi_step_length"),
    )


def _circle_lens_perimeter(params: Dict[str, Any]) -> SyntheticSpec:
    base = _circle_lens_area(params)
    trace = deepcopy(base.trace)
    region_step = next(step for step in trace if step.get("id") == "lens_12")
    region_step["metadata"]["perimeter_formula"] = {
        "rule": "lens_two_radius_perimeter",
        "radius_segments": ["O1R1", "O2R2"],
        "center_segment": "O1O2",
    }
    region_step["metadata"]["preferred_perimeter_rule"] = "lens_two_radius_perimeter"
    target_step = next(step for step in trace if step.get("op") == "set_target")
    target_step.update({"type": "perimeter", "metadata": {"preferred_rule": "lens_two_radius_perimeter"}})
    return SyntheticSpec(
        id=base.id.replace("_area_", "_perimeter_"),
        family="circle_lens_perimeter",
        trace=trace,
        question="Two intersecting circles have radii and center distance shown. Find the perimeter of their common lens.",
        difficulty="expert",
        diagnostic_axes=("circle", "lens", "unequal_radii", "perimeter", "two_minor_arcs", "multi_step_length"),
    )


def _parallel_cut(params: Dict[str, Any]) -> SyntheticSpec:
    base = params["base"]
    height = params["height"]
    ratio = params["ratio"]
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", str(height)]},
        {"op": "add_point", "id": "B", "coord": ["0", "0"]},
        {"op": "add_point", "id": "C", "coord": [str(base), "0"]},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
        {"op": "add_segment", "id": "AC", "points": ["A", "C"]},
        {"op": "add_segment", "id": "BC", "points": ["B", "C"]},
        {"op": "point_on_segment_ratio", "id": "D", "points": ["A", "B"], "ratio": ratio},
        {"op": "point_on_segment_ratio", "id": "E", "points": ["A", "C"], "ratio": ratio},
        {"op": "add_segment", "id": "DE", "points": ["D", "E"]},
        {
            "op": "add_relation",
            "id": "given_parallel_DE_BC",
            "type": "parallel",
            "objects": ["DE", "BC"],
            "metadata": {
                "annotation_role": "public_given",
                "annotation_channel": "relation_marker",
                "annotation_evidence": "visible_marker",
            },
        },
        {"op": "add_polygon", "id": "tri_ABC", "vertices": ["A", "B", "C"]},
        {"op": "add_polygon", "id": "tri_ADE", "vertices": ["A", "D", "E"]},
        {
            "op": "set_target",
            "type": "parallel_cut_ratio_residual",
            "object": "tri_ADE|tri_ABC|apex:A|cut:D-E|base:B-C",
            "metadata": {"annotation_mode": "minimal_public_givens", "allow_inferred_markers": False},
        },
    ]
    ratio_token = str(ratio).replace("/", "_")
    return SyntheticSpec(
        id=f"parallel_cut_b{base}_h{height}_r{ratio_token}",
        family="parallel_cut_similarity",
        trace=trace,
        question="DE is parallel to BC. Verify the similarity ratio residual.",
        difficulty="hard",
        diagnostic_axes=("similar_triangle", "parallel_cut_similarity", "side_ratio"),
    )


def _similarity_dual_proof(params: Dict[str, Any]) -> SyntheticSpec:
    base = _parallel_cut(params)
    body = [deepcopy(step) for step in base.trace if step.get("op") != "set_target"]
    segment_ids = {str(step.get("id")) for step in body if step.get("op") == "add_segment"}
    insert_at = next(index for index, step in enumerate(body) if step.get("op") == "add_polygon")
    for segment_id, points in (("AD", ["A", "D"]), ("AE", ["A", "E"])):
        if segment_id not in segment_ids:
            body.insert(insert_at, {"op": "add_segment", "id": segment_id, "points": points})
            insert_at += 1
    target_object = "tri_ABC|tri_ADE|map:A-A,B-D,C-E"
    relation_insert = next(index for index, step in enumerate(body) if step.get("op") == "add_polygon")
    body[relation_insert:relation_insert] = [
        {
            "op": "add_relation",
            "id": "given_D_on_AB",
            "type": "point_on_line",
            "point": "D",
            "line": "AB",
            "metadata": {"annotation_role": "public_given", "annotation_evidence": "diagram_incidence"},
        },
        {
            "op": "add_relation",
            "id": "given_E_on_AC",
            "type": "point_on_line",
            "point": "E",
            "line": "AC",
            "metadata": {"annotation_role": "public_given", "annotation_evidence": "diagram_incidence"},
        },
        {
            "op": "add_relation",
            "id": "given_corresponding_side_ratio",
            "type": "proportional_lengths",
            "object": target_object,
            "ratios": [["AD", "AB"], ["AE", "AC"], ["DE", "BC"]],
            "value": str(params["ratio"]),
            "metadata": {
                "annotation_role": "public_given",
                "annotation_channel": "relation_value",
                "annotation_evidence": "visible_relation_value",
                "label_anchor": "A",
                "label_offset": [36.0, -34.0],
            },
        },
    ]
    body.append({
        "op": "set_target",
        "mode": "proof",
        "type": "similar_triangle",
        "object": target_object,
        "metadata": {
            "solution_rules": ["sss_similarity", "aa_similarity"],
            "annotation_mode": "minimal_public_givens",
            "allow_inferred_markers": False,
        },
    })
    ratio_token = str(params["ratio"]).replace("/", "_")
    return SyntheticSpec(
        id=f"similarity_dual_proof_b{params['base']}_h{params['height']}_r{ratio_token}",
        family="similarity_dual_proof",
        trace=body,
        question="Prove that triangle ABC is similar to triangle ADE in two ways using the marked conditions.",
        answer_hint="proved",
        difficulty="expert",
        diagnostic_axes=("proof_target", "similar_triangle", "multiple_solution_paths", "sss", "aa", "auxiliary_line"),
    )


def _rectangle_area(params: Dict[str, Any]) -> SyntheticSpec:
    width = params["width"]
    height = params["height"]
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": [str(width), "0"]},
        {"op": "add_point", "id": "C", "coord": [str(width), str(height)]},
        {"op": "add_point", "id": "D", "coord": ["0", str(height)]},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"], "metadata": {"show_length": True}},
        {"op": "add_segment", "id": "BC", "points": ["B", "C"], "metadata": {"show_length": True}},
        {"op": "add_segment", "id": "CD", "points": ["C", "D"]},
        {"op": "add_segment", "id": "DA", "points": ["D", "A"]},
        {"op": "add_segment", "id": "AC", "points": ["A", "C"]},
        {"op": "add_polygon", "id": "rect_ABCD", "vertices": ["A", "B", "C", "D"], "metadata": {"area_formula": {"rule": "rectangle_base_height", "base": "AB", "height": "BC"}}},
        {"op": "set_target", "type": "area", "object": "rect_ABCD", "metadata": {"preferred_rule": "rectangle_base_height"}},
    ]
    return SyntheticSpec(
        id=f"rect_area_w{width}_h{height}",
        family="rectangle_area_diagonal",
        trace=trace,
        question="Find the area of rectangle ABCD.",
        difficulty="easy",
        diagnostic_axes=("rectangle", "area", "diagonal"),
    )


def _rectangular_frame_area(params: Dict[str, Any]) -> SyntheticSpec:
    outer_width = params["outer_width"]
    outer_height = params["outer_height"]
    border = params["border"]
    inner_width = outer_width - 2 * border
    inner_height = outer_height - 2 * border
    if inner_width <= 0 or inner_height <= 0:
        raise ValueError("Frame border must leave a positive inner rectangle")
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": [str(outer_width), "0"]},
        {"op": "add_point", "id": "C", "coord": [str(outer_width), str(outer_height)]},
        {"op": "add_point", "id": "D", "coord": ["0", str(outer_height)]},
        {"op": "add_point", "id": "E", "coord": [str(border), str(border)]},
        {"op": "add_point", "id": "F", "coord": [str(border + inner_width), str(border)]},
        {"op": "add_point", "id": "G", "coord": [str(border + inner_width), str(border + inner_height)]},
        {"op": "add_point", "id": "H", "coord": [str(border), str(border + inner_height)]},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"], "metadata": {"show_length": True}},
        {"op": "add_segment", "id": "BC", "points": ["B", "C"], "metadata": {"show_length": True}},
        {"op": "add_segment", "id": "CD", "points": ["C", "D"]},
        {"op": "add_segment", "id": "DA", "points": ["D", "A"]},
        {"op": "add_segment", "id": "EF", "points": ["E", "F"], "metadata": {"show_length": True}},
        {"op": "add_segment", "id": "FG", "points": ["F", "G"], "metadata": {"show_length": True}},
        {"op": "add_segment", "id": "GH", "points": ["G", "H"]},
        {"op": "add_segment", "id": "HE", "points": ["H", "E"]},
        {"op": "add_polygon", "id": "outer_rect", "vertices": ["A", "B", "C", "D"], "metadata": {"render": False, "area_formula": {"rule": "rectangle_base_height", "base": "AB", "height": "BC"}}},
        {"op": "add_polygon", "id": "inner_rect", "vertices": ["E", "F", "G", "H"], "metadata": {"render": False, "area_formula": {"rule": "rectangle_base_height", "base": "EF", "height": "FG"}}},
        {"op": "mark_polygon_region", "id": "outer_region", "polygon": "outer_rect", "metadata": {"preferred_area_rule": "rectangle_base_height"}},
        {"op": "mark_polygon_region", "id": "inner_region", "polygon": "inner_rect", "metadata": {"preferred_area_rule": "rectangle_base_height"}},
        {
            "op": "add_composite_region",
            "id": "shaded_frame",
            "operation": "difference",
            "components": ["outer_region", "inner_region"],
            "metadata": {"area_formula": {"rule": "region_difference", "minuend": "outer_region", "subtrahend": "inner_region"}},
        },
        {"op": "set_target", "type": "area", "object": "shaded_frame", "metadata": {"preferred_rule": "region_difference_area"}},
    ]
    return SyntheticSpec(
        id=f"rect_frame_ow{outer_width}_oh{outer_height}_b{border}",
        family="rectangular_frame_area",
        trace=trace,
        question="Find the area of the shaded rectangular frame.",
        difficulty="medium",
        diagnostic_axes=("rectangle", "shaded_area", "region_difference", "composite_area", "multi_step_area"),
    )


def _double_rectangular_hole_area(params: Dict[str, Any]) -> SyntheticSpec:
    trace, suffix = _double_rectangular_hole_trace(params)
    trace.append({"op": "set_target", "type": "area", "object": "perforated_plate", "metadata": {"preferred_rule": "region_hole_difference_area"}})
    return SyntheticSpec(
        id=f"double_hole_area_{suffix}",
        family="double_rectangular_hole_area",
        trace=trace,
        question="Find the area of the shaded rectangular plate after the two rectangular holes are removed.",
        difficulty="hard",
        diagnostic_axes=("rectangle", "multi_hole_region", "shaded_area", "region_difference", "multi_step_area"),
    )


def _double_rectangular_hole_perimeter(params: Dict[str, Any]) -> SyntheticSpec:
    trace, suffix = _double_rectangular_hole_trace(params)
    trace.append({"op": "set_target", "type": "perimeter", "object": "perforated_plate", "metadata": {"preferred_rule": "region_hole_boundary_sum"}})
    return SyntheticSpec(
        id=f"double_hole_perimeter_{suffix}",
        family="double_rectangular_hole_perimeter",
        trace=trace,
        question="Find the total boundary length of the rectangular plate, including both rectangular holes.",
        difficulty="hard",
        diagnostic_axes=("rectangle", "multi_hole_region", "perimeter", "inner_boundary", "multi_step_perimeter"),
    )


def _double_rectangular_hole_trace(params: Dict[str, Any]) -> tuple[Trace, str]:
    outer_width = params["outer_width"]
    outer_height = params["outer_height"]
    left_width = params["left_width"]
    left_height = params["left_height"]
    right_width = params["right_width"]
    right_height = params["right_height"]
    if max(left_height, right_height) + 2 >= outer_height:
        raise ValueError("Rectangular holes must leave vertical plate material")
    if left_width + right_width + 3 >= outer_width:
        raise ValueError("Rectangular holes must be separated and leave horizontal plate material")

    left_y = 1
    right_x = outer_width - right_width - 1
    right_y = outer_height - right_height - 1
    points = {
        "A": (0, 0), "B": (outer_width, 0), "C": (outer_width, outer_height), "D": (0, outer_height),
        "E": (1, left_y), "F": (1 + left_width, left_y), "G": (1 + left_width, left_y + left_height), "H": (1, left_y + left_height),
        "I": (right_x, right_y), "J": (right_x + right_width, right_y), "K": (right_x + right_width, right_y + right_height), "L": (right_x, right_y + right_height),
    }
    trace: Trace = [
        {
            "op": "add_point",
            "id": point_id,
            "coord": [str(x), str(y)],
            "metadata": {} if point_id in {"A", "B", "C", "D"} else {"show_label": False, "show_marker": False},
        }
        for point_id, (x, y) in points.items()
    ]
    for left, right, metadata in (
        ("A", "B", {"show_length": True}),
        ("B", "C", {"show_length": True, "label_side": -1.0}),
        ("C", "D", {}),
        ("D", "A", {}),
        ("E", "F", {"show_length": True}),
        ("F", "G", {}),
        ("G", "H", {}),
        ("H", "E", {"show_length": True}),
        ("I", "J", {"show_length": True}),
        ("J", "K", {}),
        ("K", "L", {}),
        ("L", "I", {"show_length": True}),
    ):
        trace.append({"op": "add_segment", "id": f"{left}{right}", "points": [left, right], "metadata": metadata})
    trace.extend(
        [
            {"op": "add_polygon", "id": "outer_rect", "vertices": ["A", "B", "C", "D"], "metadata": {"render": False, "area_formula": {"rule": "rectangle_base_height", "base": "AB", "height": "BC"}, "perimeter_formula": {"rule": "rectangle_side_perimeter", "width": "AB", "height": "BC"}}},
            {"op": "add_polygon", "id": "left_hole_rect", "vertices": ["E", "F", "G", "H"], "metadata": {"render": False, "area_formula": {"rule": "rectangle_base_height", "base": "EF", "height": "HE"}, "perimeter_formula": {"rule": "rectangle_side_perimeter", "width": "EF", "height": "HE"}}},
            {"op": "add_polygon", "id": "right_hole_rect", "vertices": ["I", "J", "K", "L"], "metadata": {"render": False, "area_formula": {"rule": "rectangle_base_height", "base": "IJ", "height": "LI"}, "perimeter_formula": {"rule": "rectangle_side_perimeter", "width": "IJ", "height": "LI"}}},
            {"op": "mark_polygon_region", "id": "outer_region", "polygon": "outer_rect", "metadata": {"preferred_area_rule": "rectangle_base_height", "preferred_perimeter_rule": "rectangle_side_perimeter", "render": False}},
            {"op": "mark_polygon_region", "id": "left_hole", "polygon": "left_hole_rect", "metadata": {"preferred_area_rule": "rectangle_base_height", "preferred_perimeter_rule": "rectangle_side_perimeter"}},
            {"op": "mark_polygon_region", "id": "right_hole", "polygon": "right_hole_rect", "metadata": {"preferred_area_rule": "rectangle_base_height", "preferred_perimeter_rule": "rectangle_side_perimeter"}},
            {
                "op": "add_region",
                "id": "perforated_plate",
                "boundary": ["A", "B", "C", "D"],
                "holes": ["left_hole", "right_hole"],
                "metadata": {
                    "area_formula": {
                        "rule": "region_hole_difference",
                        "outer_region": "outer_region",
                        "holes": ["left_hole", "right_hole"],
                    },
                    "perimeter_formula": {
                        "rule": "region_hole_boundary_sum",
                        "outer_region": "outer_region",
                        "holes": ["left_hole", "right_hole"],
                    },
                },
            },
        ]
    )
    suffix = f"ow{outer_width}_oh{outer_height}_lw{left_width}_lh{left_height}_rw{right_width}_rh{right_height}"
    return trace, suffix


def _parallelogram_area(params: Dict[str, Any]) -> SyntheticSpec:
    base = params["base"]
    height = params["height"]
    skew = params["skew"]
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": [str(base), "0"]},
        {"op": "add_point", "id": "C", "coord": [str(base + skew), str(height)]},
        {"op": "add_point", "id": "D", "coord": [str(skew), str(height)]},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"], "metadata": {"show_length": True}},
        {"op": "add_segment", "id": "BC", "points": ["B", "C"]},
        {"op": "add_segment", "id": "CD", "points": ["C", "D"]},
        {"op": "add_segment", "id": "DA", "points": ["D", "A"]},
        {"op": "foot_of_perpendicular", "id": "H", "point": "D", "line": "AB", "metadata": {"show_length": True}},
        {"op": "add_polygon", "id": "para_ABCD", "vertices": ["A", "B", "C", "D"], "metadata": {"area_formula": {"rule": "parallelogram_base_height", "base": "AB", "height": ["D", "H"]}}},
        {"op": "set_target", "type": "area", "object": "para_ABCD", "metadata": {"preferred_rule": "parallelogram_base_height"}},
    ]
    return SyntheticSpec(
        id=f"para_area_b{base}_h{height}_s{skew}",
        family="parallelogram_area",
        trace=trace,
        question="Find the area of parallelogram ABCD.",
        difficulty="medium",
        diagnostic_axes=("parallelogram", "area", "altitude"),
    )


def _trapezoid_area(params: Dict[str, Any]) -> SyntheticSpec:
    bottom = params["bottom"]
    top = params["top"]
    height = params["height"]
    offset = params["offset"]
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": [str(bottom), "0"]},
        {"op": "add_point", "id": "C", "coord": [str(offset + top), str(height)]},
        {"op": "add_point", "id": "D", "coord": [str(offset), str(height)]},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"], "metadata": {"show_length": True}},
        {"op": "add_segment", "id": "BC", "points": ["B", "C"]},
        {"op": "add_segment", "id": "CD", "points": ["C", "D"], "metadata": {"show_length": True}},
        {"op": "add_segment", "id": "DA", "points": ["D", "A"]},
        {"op": "foot_of_perpendicular", "id": "H", "point": "D", "line": "AB", "metadata": {"show_length": True}},
        {"op": "add_polygon", "id": "trap_ABCD", "vertices": ["A", "B", "C", "D"], "metadata": {"area_formula": {"rule": "trapezoid_parallel_bases_height", "bases": ["AB", "CD"], "height": ["D", "H"]}}},
        {"op": "set_target", "type": "area", "object": "trap_ABCD", "metadata": {"preferred_rule": "trapezoid_parallel_bases_height"}},
    ]
    return SyntheticSpec(
        id=f"trap_area_b{bottom}_t{top}_h{height}_o{offset}",
        family="trapezoid_area",
        trace=trace,
        question="Find the area of trapezoid ABCD.",
        difficulty="medium",
        diagnostic_axes=("trapezoid", "area", "parallel_bases", "altitude"),
    )


def _kite_area(params: Dict[str, Any]) -> SyntheticSpec:
    horizontal = params["horizontal"]
    upper = params["upper"]
    lower = params["lower"]
    half = horizontal // 2
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", str(upper)]},
        {"op": "add_point", "id": "B", "coord": [str(half), "0"]},
        {"op": "add_point", "id": "C", "coord": ["0", str(-lower)]},
        {"op": "add_point", "id": "D", "coord": [str(-half), "0"]},
        {"op": "add_line", "id": "diag_AC_line", "points": ["A", "C"], "metadata": {"render": False}},
        {"op": "add_line", "id": "diag_BD_line", "points": ["B", "D"], "metadata": {"render": False}},
        {"op": "intersection_line_line", "id": "O", "line1": "diag_AC_line", "line2": "diag_BD_line"},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
        {"op": "add_segment", "id": "BC", "points": ["B", "C"]},
        {"op": "add_segment", "id": "CD", "points": ["C", "D"]},
        {"op": "add_segment", "id": "DA", "points": ["D", "A"]},
        {"op": "add_segment", "id": "AC", "points": ["A", "C"], "metadata": {"show_length": True, "label_position": 0.32}},
        {"op": "add_segment", "id": "BD", "points": ["B", "D"], "metadata": {"show_length": True, "label_position": 0.7}},
        {"op": "add_polygon", "id": "kite_ABCD", "vertices": ["A", "B", "C", "D"], "metadata": {"area_formula": {"rule": "kite_perpendicular_diagonals", "diagonals": ["AC", "BD"]}}},
        {"op": "set_target", "type": "area", "object": "kite_ABCD", "metadata": {"preferred_rule": "kite_perpendicular_diagonals"}},
    ]
    return SyntheticSpec(
        id=f"kite_area_h{horizontal}_u{upper}_l{lower}",
        family="kite_area_diagonals",
        trace=trace,
        question="The diagonals AC and BD of kite ABCD are perpendicular. Find its area.",
        difficulty="hard",
        diagnostic_axes=("kite", "area", "intersection", "multi_step_area"),
    )


def _rhombus_diagonal(params: Dict[str, Any]) -> SyntheticSpec:
    horizontal = params["horizontal"]
    vertical = params["vertical"]
    trace = [
        {"op": "add_point", "id": "A", "coord": [f"-{horizontal}/2", "0"]},
        {"op": "add_point", "id": "B", "coord": ["0", f"{vertical}/2"]},
        {"op": "add_point", "id": "C", "coord": [f"{horizontal}/2", "0"]},
        {"op": "add_point", "id": "D", "coord": ["0", f"-{vertical}/2"]},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
        {"op": "add_segment", "id": "BC", "points": ["B", "C"]},
        {"op": "add_segment", "id": "CD", "points": ["C", "D"]},
        {"op": "add_segment", "id": "DA", "points": ["D", "A"]},
        {"op": "add_segment", "id": "AC", "points": ["A", "C"], "metadata": {"show_length": True, "label_position": 0.25}},
        {"op": "add_segment", "id": "BD", "points": ["B", "D"], "metadata": {"show_length": True, "label_position": 0.75}},
        {"op": "add_polygon", "id": "rhombus_ABCD", "vertices": ["A", "B", "C", "D"]},
        {"op": "set_target", "type": "rhombus_diagonal_area_residual", "object": "rhombus_ABCD"},
    ]
    return SyntheticSpec(
        id=f"rhombus_diag_{horizontal}_{vertical}",
        family="rhombus_diagonal_theorem",
        trace=trace,
        question="Verify the rhombus diagonal-area residual.",
        difficulty="medium",
        diagnostic_axes=("rhombus", "perpendicular_diagonals", "diagonal_area"),
    )


def _cyclic_quadrilateral(params: Dict[str, Any]) -> SyntheticSpec:
    radius = params["radius"]
    lower_angle = params["lower_angle"]
    angle_token = str(lower_angle).replace("*", "").replace("/", "_")
    trace = [
        {"op": "add_point", "id": "O", "coord": ["0", "0"]},
        {"op": "add_point", "id": "A", "coord": [str(radius), "0"]},
        {"op": "add_point", "id": "B", "coord": ["0", str(radius)]},
        {"op": "add_point", "id": "C", "coord": [f"-{radius}", "0"]},
        {"op": "add_point", "id": "D", "coord": [f"{radius}*cos({lower_angle})", f"{radius}*sin({lower_angle})"]},
        {"op": "add_circle", "id": "circle_O", "center": "O", "radius": str(radius)},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
        {"op": "add_segment", "id": "BC", "points": ["B", "C"]},
        {"op": "add_segment", "id": "CD", "points": ["C", "D"]},
        {"op": "add_segment", "id": "DA", "points": ["D", "A"]},
        {"op": "add_segment", "id": "AC", "points": ["A", "C"]},
        {"op": "add_polygon", "id": "cyclic_ABCD", "vertices": ["A", "B", "C", "D"], "metadata": {"circle": "circle_O"}},
        {"op": "set_target", "type": "cyclicity_residual", "object": "cyclic_ABCD"},
    ]
    return SyntheticSpec(
        id=f"cyclic_quad_r{radius}_a{angle_token}",
        family="cyclic_quadrilateral",
        trace=trace,
        question="Verify that ABCD is cyclic by finding its cyclicity residual.",
        difficulty="hard",
        diagnostic_axes=("cyclic_quadrilateral", "circle", "concyclic_points"),
    )


def _tangential_quadrilateral(params: Dict[str, Any]) -> SyntheticSpec:
    bottom = params["bottom"]
    top = params["top"]
    if top >= bottom:
        raise ValueError("top must be shorter than bottom")
    height = f"sqrt({bottom}*{top})"
    trace = [
        {"op": "add_point", "id": "A", "coord": [f"-{bottom}/2", f"-{height}/2"]},
        {"op": "add_point", "id": "B", "coord": [f"{bottom}/2", f"-{height}/2"]},
        {"op": "add_point", "id": "C", "coord": [f"{top}/2", f"{height}/2"]},
        {"op": "add_point", "id": "D", "coord": [f"-{top}/2", f"{height}/2"]},
        {"op": "add_point", "id": "O", "coord": ["0", "0"]},
        {"op": "add_point", "id": "T", "coord": ["0", f"-{height}/2"]},
        {"op": "add_circle", "id": "incircle_O", "center": "O", "radius": f"{height}/2"},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"], "metadata": {"show_length": True, "label_position": 0.25}},
        {"op": "add_segment", "id": "BC", "points": ["B", "C"]},
        {"op": "add_segment", "id": "CD", "points": ["C", "D"], "metadata": {"show_length": True}},
        {"op": "add_segment", "id": "DA", "points": ["D", "A"]},
        {"op": "add_segment", "id": "OT", "points": ["O", "T"], "metadata": {"show_length": True, "label_position": 0.6}},
        {"op": "add_polygon", "id": "tangential_ABCD", "vertices": ["A", "B", "C", "D"], "metadata": {"incircle": "incircle_O"}},
        {"op": "set_target", "type": "pitot_residual", "object": "tangential_ABCD"},
    ]
    return SyntheticSpec(
        id=f"tangential_quad_b{bottom}_t{top}",
        family="tangential_quadrilateral",
        trace=trace,
        question="Verify Pitot's relation for tangential quadrilateral ABCD.",
        difficulty="hard",
        diagnostic_axes=("tangential_quadrilateral", "incircle", "pitot_theorem"),
    )


def _concave_composite_area(params: Dict[str, Any]) -> SyntheticSpec:
    width = params["width"]
    height = params["height"]
    notch_width = params["notch_width"]
    notch_height = params["notch_height"]
    if notch_width >= width or notch_height >= height:
        raise ValueError("The notch must fit inside the outer rectangle")
    inner_x = width - notch_width
    inner_y = height - notch_height
    points = {
        "A": (0, 0),
        "B": (width, 0),
        "C": (width, inner_y),
        "D": (inner_x, inner_y),
        "E": (inner_x, height),
        "F": (0, height),
        "G": (width, height),
    }
    trace: Trace = [
        {"op": "add_point", "id": point_id, "coord": [str(x), str(y)], "metadata": {"render": False} if point_id == "G" else {}}
        for point_id, (x, y) in points.items()
    ]
    shown = {"AB", "CD", "DE", "FA"}
    visible_boundary = ("A", "B", "C", "D", "E", "F")
    for left, right in zip(visible_boundary, (*visible_boundary[1:], "A")):
        segment_id = f"{left}{right}"
        metadata = {"show_length": True} if segment_id in shown else {}
        trace.append({"op": "add_segment", "id": segment_id, "points": [left, right], "metadata": metadata})
    trace.extend(
        [
            {"op": "add_polygon", "id": "outer_rect", "vertices": ["A", "B", "G", "F"], "metadata": {"render": False, "area_formula": {"rule": "rectangle_base_height", "base": "AB", "height": "FA"}}},
            {"op": "add_polygon", "id": "notch_rect", "vertices": ["D", "C", "G", "E"], "metadata": {"render": False, "area_formula": {"rule": "rectangle_base_height", "base": "CD", "height": "DE"}}},
            {"op": "mark_polygon_region", "id": "outer_region", "polygon": "outer_rect", "metadata": {"preferred_area_rule": "rectangle_base_height"}},
            {"op": "mark_polygon_region", "id": "notch_region", "polygon": "notch_rect", "metadata": {"preferred_area_rule": "rectangle_base_height"}},
            {
                "op": "add_composite_region",
                "id": "concave_ABCDEF",
                "operation": "difference",
                "components": ["outer_region", "notch_region"],
                "metadata": {"area_formula": {"rule": "region_difference", "mode": "boundary_notch", "minuend": "outer_region", "subtrahend": "notch_region"}},
            },
            {"op": "set_target", "type": "area", "object": "concave_ABCDEF", "metadata": {"preferred_rule": "region_difference_area"}},
        ]
    )
    return SyntheticSpec(
        id=f"concave_area_w{width}_h{height}_nw{notch_width}_nh{notch_height}",
        family="concave_composite_area",
        trace=trace,
        question="Find the area of the concave polygon ABCDEF.",
        difficulty="hard",
        diagnostic_axes=("concave_polygon", "composite_area", "multi_step_area"),
    )


def _annulus_trace(params: Dict[str, Any]) -> list[dict[str, Any]]:
    outer_radius = params["outer_radius"]
    inner_radius = params["inner_radius"]
    if inner_radius >= outer_radius:
        raise ValueError("inner_radius must be smaller than outer_radius")
    return [
        {"op": "add_point", "id": "O", "coord": ["0", "0"]},
        {"op": "add_point", "id": "A", "coord": [str(outer_radius), "0"]},
        {"op": "add_point", "id": "B", "coord": ["0", str(inner_radius)]},
        {"op": "add_circle", "id": "outer_circle", "center": "O", "radius": str(outer_radius)},
        {"op": "add_circle", "id": "inner_circle", "center": "O", "radius": str(inner_radius)},
        {"op": "add_segment", "id": "OA", "points": ["O", "A"], "metadata": {"show_length": True}},
        {"op": "add_segment", "id": "OB", "points": ["O", "B"], "metadata": {"show_length": True}},
        {
            "op": "add_region",
            "id": "annulus_O",
            "kind": "annulus_region",
            "boundary": ["O"],
            "metadata": {
                "inner_circle": "inner_circle",
                "outer_circle": "outer_circle",
                "area_formula": {
                    "rule": "annulus_radius_area",
                    "outer_radius_segment": "OA",
                    "inner_radius_segment": "OB",
                },
                "perimeter_formula": {
                    "rule": "annulus_radius_perimeter",
                    "outer_radius_segment": "OA",
                    "inner_radius_segment": "OB",
                },
            },
        },
    ]


def _annulus_area(params: Dict[str, Any]) -> SyntheticSpec:
    outer_radius = params["outer_radius"]
    inner_radius = params["inner_radius"]
    trace = _annulus_trace(params)
    trace.append({"op": "set_target", "type": "area", "object": "annulus_O", "metadata": {"preferred_rule": "annulus_radius_area"}})
    return SyntheticSpec(
        id=f"annulus_area_R{outer_radius}_r{inner_radius}",
        family="annulus_area",
        trace=trace,
        question="Find the area of the shaded annulus.",
        difficulty="medium",
        diagnostic_axes=("annulus", "circle_difference", "multi_step_area"),
    )


def _annulus_perimeter(params: Dict[str, Any]) -> SyntheticSpec:
    outer_radius = params["outer_radius"]
    inner_radius = params["inner_radius"]
    trace = _annulus_trace(params)
    trace.append({"op": "set_target", "type": "perimeter", "object": "annulus_O", "metadata": {"preferred_rule": "annulus_radius_perimeter"}})
    return SyntheticSpec(
        id=f"annulus_perimeter_R{outer_radius}_r{inner_radius}",
        family="annulus_perimeter",
        trace=trace,
        question="Find the total length of the inner and outer boundaries of the shaded annulus.",
        difficulty="medium",
        diagnostic_axes=("annulus", "curved_boundary", "multi_step_perimeter"),
    )


def _tetra_section(params: Dict[str, Any]) -> SyntheticSpec:
    scale = params["scale"]
    height = params["height"]
    target = params["target"]
    object_id = "section_mid"
    trace = [
        {"op": "add_point3d", "id": "O", "coord": ["0", "0", "0"]},
        {"op": "add_point3d", "id": "X", "coord": [str(scale), "0", "0"]},
        {"op": "add_point3d", "id": "Y", "coord": ["0", str(scale), "0"]},
        {"op": "add_point3d", "id": "Z", "coord": ["0", "0", str(scale)]},
        {"op": "add_point3d", "id": "M", "coord": ["0", "0", f"{scale}*{height}"]},
        {"op": "add_plane", "id": "section_plane", "point": "M", "normal": ["0", "0", "1"]},
        {"op": "add_polyhedron", "id": "tetra_OXYZ", "vertices": ["O", "X", "Y", "Z"], "faces": [["O", "X", "Y"], ["O", "X", "Z"], ["O", "Y", "Z"], ["X", "Y", "Z"]]},
        {"op": "section_polyhedron_with_plane", "id": object_id, "polyhedron": "tetra_OXYZ", "plane": "section_plane"},
        {"op": "set_target", "type": target, "object": object_id},
    ]
    height_token = str(height).replace("/", "_")
    return SyntheticSpec(
        id=f"tetra_section_s{scale}_h{height_token}_{target}",
        family="tetrahedron_section",
        trace=trace,
        question="A horizontal plane cuts tetrahedron OXYZ. Find the requested section measure.",
        difficulty="hard",
        diagnostic_axes=("3d_interface", "solid_section", "auto_section", target),
    )
