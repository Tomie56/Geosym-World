"""Built-in pilot samples for exporter validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from geosym_world.datasets.adapters.legacy import LegacyGeometryAdapter


@dataclass(frozen=True)
class PilotSample:
    id: str
    question: str
    trace: List[Dict[str, Any]]
    answer: str = ""
    difficulty: str = "pilot"
    task_type: str = "geometry"
    diagnostic_axes: tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)


def built_in_pilot_samples() -> List[PilotSample]:
    return [
        _angle_triangle_sample(),
        _angled_line_intersection_sample(),
        _angle_bisector_sample(),
        _isosceles_triangle_sample(),
        _plane_3d_sample(),
        _projection_dihedral_3d_sample(),
        _solid_section_3d_sample(),
        _line_plane_face_dihedral_3d_sample(),
        _auto_section_3d_sample(),
        _auto_section_area_3d_sample(),
        _auto_section_perimeter_3d_sample(),
        _space_angle_3d_sample(),
        _circle_sector_sample(),
        _equal_chords_sample(),
        _similar_triangles_sample(),
        _aa_similar_triangles_sample(),
        _parallel_cut_similarity_sample(),
        _inscribed_angle_sample(),
        _tangent_chord_angle_sample(),
        _legacy_adapter_sample(),
    ]


def _angle_triangle_sample() -> PilotSample:
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": ["4", "0"]},
        {"op": "line_through_angle", "id": "l_AC", "point": "A", "angle": "pi/6"},
        {"op": "line_through_angle", "id": "l_BC", "point": "B", "angle": "2*pi/3"},
        {"op": "intersection_line_line", "id": "C", "line1": "l_AC", "line2": "l_BC"},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
        {"op": "add_segment", "id": "AC", "points": ["A", "C"]},
        {"op": "add_segment", "id": "BC", "points": ["B", "C"]},
        {"op": "add_polygon", "id": "tri_ABC", "vertices": ["A", "B", "C"]},
        {"op": "midpoint", "id": "M", "of": ["A", "B"]},
        {"op": "angle_bisector", "id": "bis_ABC", "vertex": "B", "ray1": ["B", "A"], "ray2": ["B", "C"]},
        {"op": "foot_of_perpendicular", "id": "H", "point": "C", "line": "AB"},
        {"op": "add_point3d", "id": "P", "coord": ["0", "0", "0"]},
        {"op": "add_point3d", "id": "Q", "coord": ["1", "0", "0"]},
        {"op": "add_point3d", "id": "R", "coord": ["0", "1", "0"]},
        {"op": "add_point3d", "id": "S", "coord": ["0", "0", "1"]},
        {"op": "add_plane", "id": "plane_PQR", "points": ["P", "Q", "R"]},
        {"op": "add_polyhedron", "id": "tetra_PQRS", "vertices": ["P", "Q", "R", "S"], "faces": [["P", "Q", "R"], ["P", "Q", "S"], ["P", "R", "S"], ["Q", "R", "S"]]},
        {"op": "set_target", "type": "area", "object": "tri_ABC"},
    ]
    return PilotSample(
        id="pilot_angle_triangle_area",
        question="Find the area of triangle ABC.",
        trace=trace,
        answer="2*sqrt(3)",
        difficulty="medium",
        task_type="area",
        diagnostic_axes=("angle_construction", "intersection", "area", "auxiliary_construction", "3d_interface"),
    )


def _circle_sector_sample() -> PilotSample:
    trace = [
        {"op": "add_point", "id": "O", "coord": ["0", "0"]},
        {"op": "add_point", "id": "A", "coord": ["5", "0"]},
        {"op": "add_point", "id": "B", "coord": ["-5", "0"]},
        {"op": "add_circle", "id": "circle_O", "center": "O", "radius": "5"},
        {"op": "add_segment", "id": "OA", "points": ["O", "A"], "metadata": {"show_length": True}},
        {"op": "add_segment", "id": "OB", "points": ["O", "B"]},
        {"op": "add_arc", "id": "arc_AB", "circle": "circle_O", "start": "A", "end": "B", "central_angle": "pi"},
        {"op": "mark_sector_region", "id": "sector_AOB", "circle": "circle_O", "start": "A", "end": "B", "central_angle": "pi", "metadata": {"radius_segment": "OA", "show_central_angle": True}},
        {"op": "add_point3d", "id": "U", "coord": ["0", "0", "0"]},
        {"op": "add_point3d", "id": "V", "coord": ["1", "0", "0"]},
        {"op": "add_point3d", "id": "W", "coord": ["0", "1", "0"]},
        {"op": "add_point3d", "id": "Z", "coord": ["0", "0", "1"]},
        {"op": "add_polyhedron", "id": "tetra_UVWZ", "vertices": ["U", "V", "W", "Z"], "faces": [["U", "W", "V"], ["U", "V", "Z"], ["V", "W", "Z"], ["W", "U", "Z"]]},
        {"op": "set_target", "type": "area", "object": "sector_AOB", "metadata": {"preferred_rule": "sector_radius_angle_area"}},
    ]
    return PilotSample(
        id="pilot_circle_sector_area",
        question="Find the area of sector AOB.",
        trace=trace,
        answer="25*pi/2",
        difficulty="medium",
        task_type="sector_area",
        diagnostic_axes=("circle", "arc", "sector_area", "central_angle", "3d_interface"),
    )


def _plane_3d_sample() -> PilotSample:
    trace = [
        {"op": "add_point3d", "id": "O", "coord": ["0", "0", "0"]},
        {"op": "add_point3d", "id": "X", "coord": ["1", "0", "0"]},
        {"op": "add_point3d", "id": "Y", "coord": ["0", "1", "0"]},
        {"op": "add_point3d", "id": "Z", "coord": ["0", "0", "1"]},
        {"op": "add_line", "id": "vertical_OZ", "points": ["O", "Z"]},
        {"op": "add_line", "id": "axis_OX", "points": ["O", "X"]},
        {"op": "add_plane", "id": "base_plane", "points": ["O", "X", "Y"]},
        {"op": "add_plane", "id": "wall_plane", "points": ["O", "Y", "Z"]},
        {"op": "add_polygon", "id": "tri_OXY", "vertices": ["O", "X", "Y"]},
        {"op": "add_polyhedron", "id": "tetra_OXYZ", "vertices": ["O", "X", "Y", "Z"], "faces": [["O", "X", "Y"], ["O", "X", "Z"], ["O", "Y", "Z"], ["X", "Y", "Z"]]},
        {"op": "set_target", "type": "angle", "object": "base_plane|wall_plane"},
    ]
    return PilotSample(
        id="pilot_3d_plane_relations",
        question="Find the angle between the base plane OXY and the wall plane OYZ.",
        trace=trace,
        answer="pi/2",
        difficulty="easy",
        task_type="plane_angle",
        diagnostic_axes=("3d_interface", "plane", "line_plane_relation", "polyhedron", "surface_area", "volume"),
    )


def _projection_dihedral_3d_sample() -> PilotSample:
    trace = [
        {"op": "add_point3d", "id": "O", "coord": ["0", "0", "0"]},
        {"op": "add_point3d", "id": "X", "coord": ["1", "0", "0"]},
        {"op": "add_point3d", "id": "Y", "coord": ["0", "1", "0"]},
        {"op": "add_point3d", "id": "Z", "coord": ["0", "0", "1"]},
        {"op": "add_point3d", "id": "T", "coord": ["1", "1", "1"]},
        {"op": "add_line", "id": "top_line", "points": ["Z", "T"]},
        {"op": "add_plane", "id": "base_plane", "points": ["O", "X", "Y"]},
        {"op": "add_plane", "id": "wall_plane", "points": ["O", "Y", "Z"]},
        {"op": "project_point_to_plane", "id": "H", "point": "T", "plane": "base_plane"},
        {"op": "project_line_to_plane", "id": "top_line_on_base", "line": "top_line", "plane": "base_plane"},
        {"op": "dihedral_angle", "id": "dih_base_wall", "planes": ["base_plane", "wall_plane"]},
        {"op": "set_target", "type": "dihedral_angle", "object": "dih_base_wall"},
    ]
    return PilotSample(
        id="pilot_3d_projection_dihedral",
        question="Find the dihedral angle between the base plane and wall plane.",
        trace=trace,
        answer="pi/2",
        difficulty="medium",
        task_type="dihedral_angle",
        diagnostic_axes=("3d_interface", "projection_3d", "plane", "line_plane_relation", "dihedral_angle"),
    )


def _solid_section_3d_sample() -> PilotSample:
    trace = [
        {"op": "add_point3d", "id": "O", "coord": ["0", "0", "0"]},
        {"op": "add_point3d", "id": "X", "coord": ["1", "0", "0"]},
        {"op": "add_point3d", "id": "Y", "coord": ["0", "1", "0"]},
        {"op": "add_point3d", "id": "Z", "coord": ["0", "0", "1"]},
        {"op": "add_plane", "id": "base_plane", "points": ["O", "X", "Y"]},
        {"op": "add_polyhedron", "id": "tetra_OXYZ", "vertices": ["O", "X", "Y", "Z"], "faces": [["O", "X", "Y"], ["O", "X", "Z"], ["O", "Y", "Z"], ["X", "Y", "Z"]]},
        {"op": "section_polyhedron_with_plane", "id": "section_base", "polyhedron": "tetra_OXYZ", "plane": "base_plane", "boundary": ["O", "X", "Y"]},
        {"op": "set_target", "type": "section_boundary_plane_residual", "object": "section_base"},
    ]
    return PilotSample(
        id="pilot_3d_solid_section_interface",
        question="The base plane cuts tetrahedron OXYZ along triangle OXY. Find the boundary-plane residual.",
        trace=trace,
        answer="0",
        difficulty="medium",
        task_type="solid_section",
        diagnostic_axes=("3d_interface", "polyhedron", "plane", "solid_section", "section_boundary"),
    )


def _line_plane_face_dihedral_3d_sample() -> PilotSample:
    trace = [
        {"op": "add_point3d", "id": "O", "coord": ["0", "0", "0"]},
        {"op": "add_point3d", "id": "X", "coord": ["1", "0", "0"]},
        {"op": "add_point3d", "id": "Y", "coord": ["0", "1", "0"]},
        {"op": "add_point3d", "id": "Z", "coord": ["0", "0", "1"]},
        {"op": "add_point3d", "id": "D", "coord": ["1", "0", "1"]},
        {"op": "add_line", "id": "diag_OD", "points": ["O", "D"]},
        {"op": "add_plane", "id": "base_plane", "points": ["O", "X", "Y"]},
        {"op": "add_polyhedron", "id": "tetra_OXYZ", "vertices": ["O", "X", "Y", "Z"], "faces": [["O", "X", "Y"], ["O", "X", "Z"], ["O", "Y", "Z"], ["X", "Y", "Z"]]},
        {"op": "dihedral_angle", "id": "dih_faces_0_1", "polyhedron": "tetra_OXYZ", "faces": [0, 1]},
        {"op": "set_target", "type": "line_plane_angle", "object": "diag_OD|base_plane"},
    ]
    return PilotSample(
        id="pilot_3d_line_plane_face_dihedral",
        question="Find the angle between line OD and the base plane.",
        trace=trace,
        answer="pi/4",
        difficulty="medium",
        task_type="line_plane_angle",
        diagnostic_axes=("3d_interface", "line_plane_angle", "line_plane_relation", "dihedral_angle", "polyhedron_face"),
    )


def _auto_section_3d_sample() -> PilotSample:
    trace = [
        {"op": "add_point3d", "id": "O", "coord": ["0", "0", "0"]},
        {"op": "add_point3d", "id": "X", "coord": ["1", "0", "0"]},
        {"op": "add_point3d", "id": "Y", "coord": ["0", "1", "0"]},
        {"op": "add_point3d", "id": "Z", "coord": ["0", "0", "1"]},
        {"op": "add_point3d", "id": "M", "coord": ["0", "0", "1/2"]},
        {"op": "add_plane", "id": "section_plane", "point": "M", "normal": ["0", "0", "1"]},
        {"op": "add_polyhedron", "id": "tetra_OXYZ", "vertices": ["O", "X", "Y", "Z"], "faces": [["O", "X", "Y"], ["O", "X", "Z"], ["O", "Y", "Z"], ["X", "Y", "Z"]]},
        {"op": "section_polyhedron_with_plane", "id": "section_mid", "polyhedron": "tetra_OXYZ", "plane": "section_plane"},
        {"op": "set_target", "type": "section_boundary_size", "object": "section_mid"},
    ]
    return PilotSample(
        id="pilot_3d_auto_section",
        question="The plane z=1/2 cuts tetrahedron OXYZ. Find the number of generated section boundary points.",
        trace=trace,
        answer="3",
        difficulty="medium",
        task_type="solid_section",
        diagnostic_axes=("3d_interface", "polyhedron", "plane", "solid_section", "auto_section"),
    )


def _auto_section_area_3d_sample() -> PilotSample:
    trace = [
        {"op": "add_point3d", "id": "O", "coord": ["0", "0", "0"]},
        {"op": "add_point3d", "id": "X", "coord": ["1", "0", "0"]},
        {"op": "add_point3d", "id": "Y", "coord": ["0", "1", "0"]},
        {"op": "add_point3d", "id": "Z", "coord": ["0", "0", "1"]},
        {"op": "add_point3d", "id": "M", "coord": ["0", "0", "1/2"]},
        {"op": "add_plane", "id": "section_plane", "point": "M", "normal": ["0", "0", "1"]},
        {"op": "add_polyhedron", "id": "tetra_OXYZ", "vertices": ["O", "X", "Y", "Z"], "faces": [["O", "X", "Y"], ["O", "X", "Z"], ["O", "Y", "Z"], ["X", "Y", "Z"]]},
        {"op": "section_polyhedron_with_plane", "id": "section_mid_area", "polyhedron": "tetra_OXYZ", "plane": "section_plane"},
        {"op": "set_target", "type": "section_area", "object": "section_mid_area"},
    ]
    return PilotSample(
        id="pilot_3d_auto_section_area",
        question="The plane z=1/2 cuts tetrahedron OXYZ. Find the area of the generated section triangle.",
        trace=trace,
        answer="1/8",
        difficulty="medium",
        task_type="solid_section_area",
        diagnostic_axes=("3d_interface", "polyhedron", "plane", "solid_section", "auto_section", "section_area"),
    )


def _auto_section_perimeter_3d_sample() -> PilotSample:
    trace = [
        {"op": "add_point3d", "id": "O", "coord": ["0", "0", "0"]},
        {"op": "add_point3d", "id": "X", "coord": ["1", "0", "0"]},
        {"op": "add_point3d", "id": "Y", "coord": ["0", "1", "0"]},
        {"op": "add_point3d", "id": "Z", "coord": ["0", "0", "1"]},
        {"op": "add_point3d", "id": "M", "coord": ["0", "0", "1/2"]},
        {"op": "add_plane", "id": "section_plane", "point": "M", "normal": ["0", "0", "1"]},
        {"op": "add_polyhedron", "id": "tetra_OXYZ", "vertices": ["O", "X", "Y", "Z"], "faces": [["O", "X", "Y"], ["O", "X", "Z"], ["O", "Y", "Z"], ["X", "Y", "Z"]]},
        {"op": "section_polyhedron_with_plane", "id": "section_mid_perim", "polyhedron": "tetra_OXYZ", "plane": "section_plane"},
        {"op": "set_target", "type": "section_perimeter", "object": "section_mid_perim"},
    ]
    return PilotSample(
        id="pilot_3d_auto_section_perimeter",
        question="The plane z=1/2 cuts tetrahedron OXYZ. Find the perimeter of the generated section triangle.",
        trace=trace,
        answer="1 + sqrt(2)/2",
        difficulty="medium",
        task_type="solid_section_perimeter",
        diagnostic_axes=("3d_interface", "polyhedron", "plane", "solid_section", "auto_section", "section_perimeter"),
    )


def _space_angle_3d_sample() -> PilotSample:
    trace = [
        {"op": "add_point3d", "id": "O", "coord": ["0", "0", "0"]},
        {"op": "add_point3d", "id": "X", "coord": ["1", "0", "0"]},
        {"op": "add_point3d", "id": "Y", "coord": ["0", "1", "0"]},
        {"op": "add_point3d", "id": "D", "coord": ["1", "0", "1"]},
        {"op": "add_line", "id": "diag_OD", "points": ["O", "D"]},
        {"op": "add_plane", "id": "base_plane", "points": ["O", "X", "Y"]},
        {"op": "set_target", "type": "space_angle", "object": "diag_OD|base_plane"},
    ]
    return PilotSample(
        id="pilot_3d_space_angle",
        question="Find the unified space-angle fact between line OD and the base plane.",
        trace=trace,
        answer="pi/4",
        difficulty="medium",
        task_type="space_angle",
        diagnostic_axes=("3d_interface", "space_angle", "line_plane_angle", "line_plane_relation"),
    )


def _isosceles_triangle_sample() -> PilotSample:
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "2"]},
        {"op": "add_point", "id": "B", "coord": ["-1", "0"]},
        {"op": "add_point", "id": "C", "coord": ["1", "0"]},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
        {"op": "add_segment", "id": "AC", "points": ["A", "C"]},
        {"op": "add_segment", "id": "BC", "points": ["B", "C"]},
        {"op": "add_polygon", "id": "tri_ABC", "vertices": ["A", "B", "C"]},
        {"op": "set_target", "type": "equal_base_angle_residual", "object": "tri_ABC|apex:A"},
    ]
    return PilotSample(
        id="pilot_isosceles_base_angles",
        question="Triangle ABC is isosceles with AB = AC. Find the base-angle residual.",
        trace=trace,
        answer="0",
        difficulty="easy",
        task_type="isosceles_triangle",
        diagnostic_axes=("triangle", "isosceles", "equal_sides", "equal_base_angles", "theorem_lite"),
    )


def _angle_bisector_sample() -> PilotSample:
    trace = [
        {"op": "add_point", "id": "O", "coord": ["0", "0"]},
        {"op": "add_point", "id": "A", "coord": ["1", "0"]},
        {"op": "add_point", "id": "C", "coord": ["0", "1"]},
        {"op": "add_segment", "id": "OA", "points": ["O", "A"]},
        {"op": "add_segment", "id": "OC", "points": ["O", "C"]},
        {"op": "angle_bisector", "id": "bis_AOC", "vertex": "O", "ray1": ["O", "A"], "ray2": ["O", "C"]},
        {"op": "set_target", "type": "angle_bisector_residual", "object": "bis_AOC"},
    ]
    return PilotSample(
        id="pilot_angle_bisector_residual",
        question="Line bis_AOC bisects angle AOC. Find the left-right angle residual.",
        trace=trace,
        answer="0",
        difficulty="easy",
        task_type="angle_bisector",
        diagnostic_axes=("angle_bisector", "equal_angle", "fact_closure"),
    )


def _angled_line_intersection_sample() -> PilotSample:
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "P", "coord": ["2", "0"]},
        {"op": "add_line", "id": "x_axis", "point": "A", "direction": ["1", "0"]},
        {"op": "line_through_angle", "id": "tilt_60", "point": "P", "base_line": "x_axis", "angle": "pi/3"},
        {"op": "intersection_line_line", "id": "X", "line1": "x_axis", "line2": "tilt_60"},
        {"op": "midpoint", "id": "M", "of": ["A", "P"]},
        {"op": "perpendicular_through", "id": "perp_at_P", "point": "P", "line": "x_axis"},
        {"op": "set_target", "type": "angle", "object": "x_axis|tilt_60"},
    ]
    return PilotSample(
        id="pilot_angled_line_intersection",
        question="Two lines intersect at a 60 degree angle. Find the smaller angle between them.",
        trace=trace,
        answer="pi/3",
        difficulty="easy",
        task_type="angle",
        diagnostic_axes=("angle_construction", "line_intersection", "midpoint", "perpendicular", "fact_closure"),
    )


def _legacy_adapter_sample() -> PilotSample:
    legacy_like = {
        "points": {
            "A": {"x": "0", "y": "0"},
            "B": {"x": "4", "y": "0"},
            "C": {"x": "4", "y": "3"},
            "O": {"x": "0", "y": "0"},
        },
        "segments": [
            {"id": "AB", "points": ["A", "B"]},
            {"id": "BC", "points": ["B", "C"]},
            {"id": "AC", "points": ["A", "C"]},
        ],
        "circles": [{"id": "circle_O", "center": "O", "radius": "5"}],
        "polygons": [{"id": "tri_ABC", "vertices": ["A", "B", "C"]}],
        "regions": [{"id": "region_ABC", "polygon": "tri_ABC"}],
        "target": {"type": "length", "object": "AC"},
    }
    return PilotSample(
        id="pilot_legacy_adapter_length",
        question="Find the length of AC.",
        trace=LegacyGeometryAdapter().to_trace(legacy_like),
        answer="5",
        difficulty="easy",
        task_type="length",
        diagnostic_axes=("legacy_adapter", "distance", "verification"),
        metadata={"source": "legacy_like_dict"},
    )


def _equal_chords_sample() -> PilotSample:
    trace = [
        {"op": "add_point", "id": "O", "coord": ["0", "0"]},
        {"op": "add_circle", "id": "circle_O", "center": "O", "radius": "5"},
        {"op": "add_point", "id": "A", "coord": ["3", "4"]},
        {"op": "add_point", "id": "B", "coord": ["3", "-4"]},
        {"op": "add_point", "id": "C", "coord": ["4", "3"]},
        {"op": "add_point", "id": "D", "coord": ["-4", "3"]},
        {"op": "add_chord", "id": "AB", "circle": "circle_O", "points": ["A", "B"]},
        {"op": "add_chord", "id": "CD", "circle": "circle_O", "points": ["C", "D"]},
        {"op": "set_target", "type": "length", "object": "AB"},
    ]
    return PilotSample(
        id="pilot_equal_chords_central_angle",
        question="In circle O, chords AB and CD are equal. Find the length of AB.",
        trace=trace,
        answer="8",
        difficulty="medium",
        task_type="circle_chord",
        diagnostic_axes=("circle", "chord", "equal_chord", "central_angle", "theorem_lite"),
    )


def _similar_triangles_sample() -> PilotSample:
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": ["3", "0"]},
        {"op": "add_point", "id": "C", "coord": ["0", "4"]},
        {"op": "add_point", "id": "D", "coord": ["0", "0"]},
        {"op": "add_point", "id": "E", "coord": ["6", "0"]},
        {"op": "add_point", "id": "F", "coord": ["0", "8"]},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
        {"op": "add_segment", "id": "BC", "points": ["B", "C"]},
        {"op": "add_segment", "id": "CA", "points": ["C", "A"]},
        {"op": "add_segment", "id": "DE", "points": ["D", "E"]},
        {"op": "add_segment", "id": "EF", "points": ["E", "F"]},
        {"op": "add_segment", "id": "FD", "points": ["F", "D"]},
        {"op": "add_polygon", "id": "tri_ABC", "vertices": ["A", "B", "C"]},
        {"op": "add_polygon", "id": "tri_DEF", "vertices": ["D", "E", "F"]},
        {"op": "set_target", "type": "similarity_ratio_residual", "object": "tri_ABC|tri_DEF|map:A-D,B-E,C-F"},
    ]
    return PilotSample(
        id="pilot_similar_triangles_sss",
        question="Triangles ABC and DEF are SSS-similar. Find their side-ratio residual.",
        trace=trace,
        answer="0",
        difficulty="medium",
        task_type="similar_triangle",
        diagnostic_axes=("triangle", "similar_triangle", "side_ratio", "corresponding_angles", "theorem_lite"),
    )


def _aa_similar_triangles_sample() -> PilotSample:
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": ["2", "0"]},
        {"op": "add_point", "id": "C", "coord": ["0", "2*sqrt(3)"]},
        {"op": "add_point", "id": "D", "coord": ["5", "1"]},
        {"op": "add_point", "id": "E", "coord": ["8", "1"]},
        {"op": "add_point", "id": "F", "coord": ["5", "1 + 3*sqrt(3)"]},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
        {"op": "add_segment", "id": "BC", "points": ["B", "C"]},
        {"op": "add_segment", "id": "CA", "points": ["C", "A"]},
        {"op": "add_segment", "id": "DE", "points": ["D", "E"]},
        {"op": "add_segment", "id": "EF", "points": ["E", "F"]},
        {"op": "add_segment", "id": "FD", "points": ["F", "D"]},
        {"op": "add_polygon", "id": "tri_ABC", "vertices": ["A", "B", "C"]},
        {"op": "add_polygon", "id": "tri_DEF", "vertices": ["D", "E", "F"]},
        {"op": "set_target", "type": "aa_similarity_residual", "object": "tri_ABC|tri_DEF|map:A-D,B-E,C-F"},
    ]
    return PilotSample(
        id="pilot_similar_triangles_aa",
        question="Triangles ABC and DEF have two equal corresponding angles. Find their AA-similarity residual.",
        trace=trace,
        answer="0",
        difficulty="medium",
        task_type="similar_triangle",
        diagnostic_axes=("triangle", "similar_triangle", "aa_similarity", "corresponding_angles", "theorem_lite"),
    )


def _parallel_cut_similarity_sample() -> PilotSample:
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": ["4", "0"]},
        {"op": "add_point", "id": "C", "coord": ["0", "4"]},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
        {"op": "add_segment", "id": "AC", "points": ["A", "C"]},
        {"op": "add_segment", "id": "BC", "points": ["B", "C"]},
        {"op": "add_line", "id": "line_AC", "points": ["A", "C"]},
        {"op": "point_on_segment_ratio", "id": "D", "segment": "AB", "ratio": "1/2"},
        {"op": "parallel_through", "id": "line_DE", "point": "D", "line": "BC"},
        {"op": "intersection_line_line", "id": "E", "line1": "line_DE", "line2": "line_AC"},
        {"op": "add_segment", "id": "AD", "points": ["A", "D"]},
        {"op": "add_segment", "id": "AE", "points": ["A", "E"]},
        {"op": "add_segment", "id": "DE", "points": ["D", "E"]},
        {"op": "add_polygon", "id": "tri_ABC", "vertices": ["A", "B", "C"]},
        {"op": "add_polygon", "id": "tri_ADE", "vertices": ["A", "D", "E"]},
        {"op": "set_target", "type": "parallel_cut_ratio_residual", "object": "tri_ADE|tri_ABC|apex:A|cut:D-E|base:B-C"},
    ]
    return PilotSample(
        id="pilot_parallel_cut_similarity",
        question="In triangle ABC, DE is parallel to BC with D on AB and E on AC. Find the parallel-cut ratio residual.",
        trace=trace,
        answer="0",
        difficulty="medium",
        task_type="similar_triangle",
        diagnostic_axes=("triangle", "parallel", "parallel_cut_similarity", "side_ratio", "theorem_lite"),
    )


def _inscribed_angle_sample() -> PilotSample:
    trace = [
        {"op": "add_point", "id": "O", "coord": ["0", "0"]},
        {"op": "add_circle", "id": "circle_O", "center": "O", "radius": "1"},
        {"op": "add_point", "id": "A", "coord": ["1", "0"]},
        {"op": "add_point", "id": "B", "coord": ["0", "1"]},
        {"op": "add_point", "id": "P", "coord": ["-1", "0"]},
        {"op": "add_point", "id": "Q", "coord": ["0", "-1"]},
        {"op": "add_chord", "id": "AB", "circle": "circle_O", "points": ["A", "B"]},
        {"op": "add_arc", "id": "arc_AB", "circle": "circle_O", "start": "A", "end": "B", "central_angle": "pi/2"},
        {"op": "set_target", "type": "inscribed_angle_arc_residual", "object": "APB|arc_AB"},
    ]
    return PilotSample(
        id="pilot_inscribed_angle_same_chord",
        question="A and B lie on circle O, and arc AB is known. Find the arc-backed inscribed-angle residual for angle APB.",
        trace=trace,
        answer="0",
        difficulty="medium",
        task_type="inscribed_angle",
        diagnostic_axes=("circle", "chord", "arc", "inscribed_angle", "central_angle", "equal_inscribed_angles", "theorem_lite"),
    )


def _tangent_chord_angle_sample() -> PilotSample:
    trace = [
        {"op": "add_point", "id": "O", "coord": ["0", "0"]},
        {"op": "add_circle", "id": "circle_O", "center": "O", "radius": "1"},
        {"op": "add_point", "id": "P", "coord": ["1", "2"]},
        {"op": "tangent_points_from_point", "ids": ["T_alt", "A"], "point": "P", "circle": "circle_O"},
        {"op": "add_point", "id": "B", "coord": ["0", "1"]},
        {"op": "add_point", "id": "C", "coord": ["-1", "0"]},
        {"op": "add_chord", "id": "AB", "circle": "circle_O", "points": ["A", "B"]},
        {"op": "set_target", "type": "tangent_chord_angle_residual", "object": "AB|tan:P-A|obs:C|circle_O"},
    ]
    return PilotSample(
        id="pilot_tangent_chord_angle",
        question="A tangent at A and chord AB determine an angle equal to the alternate segment angle. Find the residual.",
        trace=trace,
        answer="0",
        difficulty="medium",
        task_type="circle_tangent_angle",
        diagnostic_axes=("circle", "tangent", "chord", "alternate_segment", "theorem_lite"),
    )
