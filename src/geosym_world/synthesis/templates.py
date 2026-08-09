"""Template families for procedural synthetic trace construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Dict, Iterable, List
import json


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
            name="angled_line_intersection",
            description="Two lines through endpoints create a triangle by angle intersection.",
            parameter_grid={"base": (4, 5, 6), "left_angle": ("pi/6", "pi/4", "pi/3"), "right_angle": ("2*pi/3", "3*pi/4")},
            builder=_angled_intersection,
        ),
        TemplateFamily(
            name="circle_sector",
            description="Circle diameter, tangent points, chord, and sector target.",
            parameter_grid={"radius": (3, 4, 5, 6), "external_x": (9, 11, 13)},
            builder=_circle_sector,
        ),
        TemplateFamily(
            name="parallel_cut_similarity",
            description="Triangle with a parallel interior cut producing similar triangles.",
            parameter_grid={"base": (6, 8, 10), "height": (4, 6), "ratio": ("1/3", "1/2", "2/3")},
            builder=_parallel_cut,
        ),
        TemplateFamily(
            name="rectangle_area_diagonal",
            description="Rectangle with diagonal and area target.",
            parameter_grid={"width": (3, 4, 5, 6), "height": (2, 3, 4)},
            builder=_rectangle_area,
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
        {"op": "add_polygon", "id": "tri_ABC", "vertices": ["A", "B", "C"]},
        {"op": "midpoint", "id": "M", "of": ["A", "B"]},
        {"op": "foot_of_perpendicular", "id": "H", "point": "C", "line": "AB", "metadata": {"show_length": True}},
        {"op": "set_target", "type": "area", "object": "tri_ABC"},
    ]
    return SyntheticSpec(
        id=f"tri_area_b{base}_h{height}_s{skew}",
        family="triangle_area_from_height",
        trace=trace,
        question="Find the area of triangle ABC.",
        difficulty="easy" if height <= 2 else "medium",
        diagnostic_axes=("triangle", "area", "altitude", "midpoint"),
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


def _circle_sector(params: Dict[str, Any]) -> SyntheticSpec:
    radius = params["radius"]
    external_x = params["external_x"]
    trace = [
        {"op": "add_point", "id": "O", "coord": ["0", "0"]},
        {"op": "add_circle", "id": "circle_O", "center": "O", "radius": str(radius)},
        {"op": "add_line", "id": "diameter_line", "point": "O", "direction": ["1", "0"]},
        {"op": "intersection_line_circle", "ids": ["A", "B"], "line": "diameter_line", "circle": "circle_O"},
        {"op": "add_segment", "id": "OA", "points": ["O", "A"], "metadata": {"show_length": True}},
        {"op": "add_point", "id": "P", "coord": [str(external_x), "0"]},
        {"op": "tangent_points_from_point", "ids": ["T1", "T2"], "point": "P", "circle": "circle_O"},
        {"op": "add_chord", "id": "T1T2", "circle": "circle_O", "points": ["T1", "T2"]},
        {"op": "add_arc", "id": "arc_AB", "circle": "circle_O", "start": "A", "end": "B", "central_angle": "pi"},
        {"op": "mark_sector_region", "id": "sector_AOB", "circle": "circle_O", "start": "A", "end": "B", "central_angle": "pi"},
        {"op": "set_target", "type": "area", "object": "sector_AOB"},
    ]
    return SyntheticSpec(
        id=f"circle_sector_r{radius}_x{external_x}",
        family="circle_sector",
        trace=trace,
        question="Find the area of sector AOB.",
        difficulty="medium",
        diagnostic_axes=("circle", "sector_area", "tangent", "chord"),
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
        {"op": "add_polygon", "id": "tri_ABC", "vertices": ["A", "B", "C"]},
        {"op": "add_polygon", "id": "tri_ADE", "vertices": ["A", "D", "E"]},
        {"op": "set_target", "type": "parallel_cut_ratio_residual", "object": "tri_ADE|tri_ABC|apex:A|cut:D-E|base:B-C"},
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
        {"op": "add_polygon", "id": "rect_ABCD", "vertices": ["A", "B", "C", "D"]},
        {"op": "set_target", "type": "area", "object": "rect_ABCD"},
    ]
    return SyntheticSpec(
        id=f"rect_area_w{width}_h{height}",
        family="rectangle_area_diagonal",
        trace=trace,
        question="Find the area of rectangle ABCD.",
        difficulty="easy",
        diagnostic_axes=("rectangle", "area", "diagonal"),
    )


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
        {"op": "add_polygon", "id": "para_ABCD", "vertices": ["A", "B", "C", "D"]},
        {"op": "foot_of_perpendicular", "id": "H", "point": "D", "line": "AB", "metadata": {"show_length": True}},
        {"op": "set_target", "type": "area", "object": "para_ABCD"},
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
        {"op": "add_polygon", "id": "trap_ABCD", "vertices": ["A", "B", "C", "D"]},
        {"op": "foot_of_perpendicular", "id": "H", "point": "D", "line": "AB", "metadata": {"show_length": True}},
        {"op": "set_target", "type": "area", "object": "trap_ABCD"},
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
        {"op": "add_polygon", "id": "kite_ABCD", "vertices": ["A", "B", "C", "D"]},
        {"op": "set_target", "type": "area", "object": "kite_ABCD"},
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
    }
    trace: Trace = [
        {"op": "add_point", "id": point_id, "coord": [str(x), str(y)]}
        for point_id, (x, y) in points.items()
    ]
    shown = {"AB", "CD", "DE", "FA"}
    for left, right in zip(points, (*tuple(points)[1:], "A")):
        segment_id = f"{left}{right}"
        metadata = {"show_length": True} if segment_id in shown else {}
        trace.append({"op": "add_segment", "id": segment_id, "points": [left, right], "metadata": metadata})
    trace.extend(
        [
            {"op": "add_polygon", "id": "concave_ABCDEF", "vertices": list(points)},
            {"op": "set_target", "type": "area", "object": "concave_ABCDEF"},
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


def _annulus_area(params: Dict[str, Any]) -> SyntheticSpec:
    outer_radius = params["outer_radius"]
    inner_radius = params["inner_radius"]
    if inner_radius >= outer_radius:
        raise ValueError("inner_radius must be smaller than outer_radius")
    trace = [
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
            "metadata": {"inner_circle": "inner_circle", "outer_circle": "outer_circle"},
        },
        {"op": "set_target", "type": "area", "object": "annulus_O"},
    ]
    return SyntheticSpec(
        id=f"annulus_area_R{outer_radius}_r{inner_radius}",
        family="annulus_area",
        trace=trace,
        question="Find the area of the shaded annulus.",
        difficulty="medium",
        diagnostic_axes=("annulus", "circle_difference", "multi_step_area"),
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
