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
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
        {"op": "add_segment", "id": "AC", "points": ["A", "C"]},
        {"op": "add_segment", "id": "BC", "points": ["B", "C"]},
        {"op": "add_polygon", "id": "tri_ABC", "vertices": ["A", "B", "C"]},
        {"op": "midpoint", "id": "M", "of": ["A", "B"]},
        {"op": "foot_of_perpendicular", "id": "H", "point": "C", "line": "AB"},
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
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
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
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
        {"op": "add_segment", "id": "BC", "points": ["B", "C"]},
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
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
        {"op": "add_segment", "id": "BC", "points": ["B", "C"]},
        {"op": "add_segment", "id": "CD", "points": ["C", "D"]},
        {"op": "add_segment", "id": "DA", "points": ["D", "A"]},
        {"op": "add_polygon", "id": "para_ABCD", "vertices": ["A", "B", "C", "D"]},
        {"op": "foot_of_perpendicular", "id": "H", "point": "D", "line": "AB"},
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
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
        {"op": "add_segment", "id": "BC", "points": ["B", "C"]},
        {"op": "add_segment", "id": "CD", "points": ["C", "D"]},
        {"op": "add_segment", "id": "DA", "points": ["D", "A"]},
        {"op": "add_polygon", "id": "trap_ABCD", "vertices": ["A", "B", "C", "D"]},
        {"op": "foot_of_perpendicular", "id": "H", "point": "D", "line": "AB"},
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
