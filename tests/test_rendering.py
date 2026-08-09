"""Unit tests for render style policy and scene generation."""

from geosym_world.construction.builder import ConstructionBuilder
from geosym_world.rendering import DeterministicStylePolicy, format_math_expression, position_labels, render_svg, state_to_render_scene, write_png
from geosym_world.rendering.layout import build_view_transform


def test_style_policy_is_deterministic():
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": ["1", "0"]},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
        {"op": "set_target", "type": "length", "object": "AB"},
    ]
    state = ConstructionBuilder().execute(trace)
    left = state_to_render_scene(state, style_policy=DeterministicStylePolicy(seed=11)).to_dict()
    right = state_to_render_scene(state, style_policy=DeterministicStylePolicy(seed=11)).to_dict()
    assert left == right
    assert left["metadata"]["style_policy"]["seed"] == 11


def test_foot_of_perpendicular_renders_auxiliary_segment():
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": ["4", "0"]},
        {"op": "add_point", "id": "C", "coord": ["1", "3"]},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
        {"op": "foot_of_perpendicular", "id": "H", "point": "C", "line": "AB"},
        {"op": "set_target", "type": "length", "object": "AB"},
    ]
    state = ConstructionBuilder().execute(trace)
    scene = state_to_render_scene(state)
    ids = {primitive.id for primitive in scene.primitives}
    assert "altitude:C-H" in ids
    assert "right_angle:H" in ids


def test_polyhedron_hidden_edges_are_dashed():
    trace = [
        {"op": "add_point3d", "id": "O", "coord": ["0", "0", "0"]},
        {"op": "add_point3d", "id": "X", "coord": ["1", "0", "0"]},
        {"op": "add_point3d", "id": "Y", "coord": ["0", "1", "0"]},
        {"op": "add_point3d", "id": "Z", "coord": ["0", "0", "1"]},
        {"op": "add_polyhedron", "id": "tetra", "vertices": ["O", "X", "Y", "Z"], "faces": [["O", "X", "Y"], ["O", "X", "Z"], ["O", "Y", "Z"], ["X", "Y", "Z"]]},
        {"op": "set_target", "type": "volume", "object": "tetra"},
    ]
    state = ConstructionBuilder().execute(trace)
    scene = state_to_render_scene(state)
    hidden = [primitive for primitive in scene.primitives if primitive.metadata.get("hidden")]
    assert hidden
    assert all(primitive.style.dasharray for primitive in hidden)


def test_implicit_segment_intersection_is_rendered():
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": ["4", "4"]},
        {"op": "add_point", "id": "C", "coord": ["0", "4"]},
        {"op": "add_point", "id": "D", "coord": ["4", "0"]},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
        {"op": "add_segment", "id": "CD", "points": ["C", "D"]},
        {"op": "set_target", "type": "length", "object": "AB"},
    ]
    scene = state_to_render_scene(ConstructionBuilder().execute(trace))
    intersections = [primitive for primitive in scene.primitives if primitive.role == "implicit_intersection"]
    assert len(intersections) == 1
    assert intersections[0].geometry["point"] == (2.0, 2.0)
    assert intersections[0].style.fill == "#fb923c"
    assert scene.metadata["point_labels"][intersections[0].id] == "E"
    assert next(label.text for label in scene.labels if label.id == f"label:{intersections[0].id}") == "E"


def test_auxiliary_line_crossings_are_not_promoted_to_named_points():
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": ["4", "0"]},
        {"op": "add_point", "id": "C", "coord": ["2", "-2"]},
        {"op": "add_point", "id": "D", "coord": ["2", "2"]},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
        {"op": "add_line", "id": "CD_line", "points": ["C", "D"], "metadata": {"constructed_by": "perpendicular_through"}},
        {"op": "set_target", "type": "length", "object": "AB"},
    ]
    scene = state_to_render_scene(ConstructionBuilder().execute(trace))
    assert not [primitive for primitive in scene.primitives if primitive.role == "implicit_intersection"]


def test_far_field_implicit_intersections_do_not_shrink_the_subject():
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": ["1", "0"]},
        {"op": "add_point", "id": "C", "coord": ["0", "1"]},
        {"op": "add_point", "id": "D", "coord": ["1", "101/100"]},
        {"op": "add_line", "id": "AB_line", "points": ["A", "B"]},
        {"op": "add_line", "id": "CD_line", "points": ["C", "D"]},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
        {"op": "set_target", "type": "length", "object": "AB"},
    ]
    scene = state_to_render_scene(ConstructionBuilder().execute(trace))
    assert not [primitive for primitive in scene.primitives if primitive.role == "implicit_intersection"]


def test_measurement_labels_include_segment_and_altitude_lengths():
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": ["4", "0"]},
        {"op": "add_point", "id": "C", "coord": ["1", "3"]},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"], "metadata": {"show_length": True}},
        {"op": "foot_of_perpendicular", "id": "H", "point": "C", "line": "AB", "metadata": {"show_length": True}},
        {"op": "set_target", "type": "length", "object": "AB"},
    ]
    scene = state_to_render_scene(ConstructionBuilder().execute(trace))
    labels = {label.id: label.text for label in scene.labels}
    assert labels["measure:AB"] == "4"
    assert labels["measure:altitude:C-H"] == "3"


def test_measurement_label_position_can_avoid_geometry_labels():
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": ["10", "0"]},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"], "metadata": {"show_length": True, "label_position": 0.7}},
        {"op": "set_target", "type": "length", "object": "AB"},
    ]
    scene = state_to_render_scene(ConstructionBuilder().execute(trace))
    label = next(label for label in scene.labels if label.id == "measure:AB")
    assert label.position == (7.0, 0.0)


def test_non_rendered_construction_line_is_omitted():
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": ["4", "0"]},
        {"op": "add_line", "id": "construction", "points": ["A", "B"], "metadata": {"render": False}},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
        {"op": "set_target", "type": "length", "object": "AB"},
    ]
    scene = state_to_render_scene(ConstructionBuilder().execute(trace))
    assert "construction" not in {primitive.id for primitive in scene.primitives}


def test_construction_extensions_render_behind_finite_segments():
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": ["4", "0"]},
        {"op": "add_line", "id": "line_AB", "points": ["A", "B"], "metadata": {"constructed_by": "extension"}},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
        {"op": "set_target", "type": "length", "object": "AB"},
    ]
    scene = state_to_render_scene(ConstructionBuilder().execute(trace))
    primitive_ids = [primitive.id for primitive in scene.primitives]
    assert primitive_ids.index("line_AB") < primitive_ids.index("AB")


def test_perpendicular_marker_outside_segment_gets_visible_extension():
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": ["1", "0"]},
        {"op": "add_point", "id": "C", "coord": ["2", "1"]},
        {"op": "add_segment", "id": "BC", "points": ["B", "C"]},
        {"op": "perpendicular_through", "id": "perp_A_BC", "point": "A", "line": "BC"},
        {"op": "set_target", "type": "length", "object": "BC"},
    ]
    scene = state_to_render_scene(ConstructionBuilder().execute(trace))
    extension = next(primitive for primitive in scene.primitives if primitive.id.startswith("extension:BC:"))
    marker = next(primitive for primitive in scene.primitives if primitive.id.startswith("right_angle:perp_A_BC"))
    assert extension.metadata["relation"] == "segment_extension"
    assert marker.geometry["vertex"] == (0.5, -0.5)


def test_perpendicular_relation_intersection_gets_sequential_point_label():
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": ["1", "0"]},
        {"op": "add_point", "id": "C", "coord": ["2", "1"]},
        {"op": "add_segment", "id": "BC", "points": ["B", "C"]},
        {"op": "perpendicular_through", "id": "perp_A_BC", "point": "A", "line": "BC"},
        {"op": "set_target", "type": "length", "object": "BC"},
    ]
    scene = state_to_render_scene(ConstructionBuilder().execute(trace))
    intersection = next(primitive for primitive in scene.primitives if primitive.role == "relation_intersection")
    assert intersection.geometry["point"] == (0.5, -0.5)
    assert scene.metadata["point_labels"][intersection.id] == "D"
    assert next(label.text for label in scene.labels if label.id == f"label:{intersection.id}") == "D"


def test_external_perpendicular_foot_gets_base_extension():
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": ["1", "0"]},
        {"op": "add_point", "id": "C", "coord": ["2", "2"]},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
        {"op": "foot_of_perpendicular", "id": "H", "point": "C", "line": "AB"},
        {"op": "set_target", "type": "length", "object": "AB"},
    ]
    scene = state_to_render_scene(ConstructionBuilder().execute(trace))
    extension = next(primitive for primitive in scene.primitives if primitive.id == "extension:AB:foot:H")
    assert extension.geometry["points"][0] == (1.0, 0.0)
    assert extension.geometry["points"][1][0] > 2.0


def test_generated_point_ids_receive_unique_sequential_display_labels():
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": ["4", "0"]},
        {"op": "add_point", "id": "C", "coord": ["0", "4"]},
        {"op": "add_point", "id": "H_A_BC_e0_0", "coord": ["2", "0"], "metadata": {"constructed_by": "foot_of_perpendicular"}},
        {"op": "add_point", "id": "X_AB_CD_e0_1", "coord": ["2", "2"], "metadata": {"constructed_by": "intersection_lines"}},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
        {"op": "set_target", "type": "length", "object": "AB"},
    ]
    scene = state_to_render_scene(ConstructionBuilder().execute(trace))
    assert scene.metadata["point_labels"] == {
        "A": "A",
        "B": "B",
        "C": "C",
        "H_A_BC_e0_0": "D",
        "X_AB_CD_e0_1": "E",
    }
    labels = {label.id: label.text for label in scene.labels}
    assert labels["label:H_A_BC_e0_0"] == "D"
    assert labels["label:X_AB_CD_e0_1"] == "E"


def test_exact_math_labels_use_radicals_and_fractions(tmp_path):
    assert format_math_expression("sqrt(13)") == "√13"
    assert format_math_expression("2*sqrt(2)") == "2√2"
    assert format_math_expression("7/2") == "⁷⁄₂"
    assert format_math_expression("3*sqrt(5)/2") == "3√5⁄₂"
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": ["2", "3"]},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"], "metadata": {"show_length": True}},
        {"op": "set_target", "type": "length", "object": "AB"},
    ]
    scene = state_to_render_scene(ConstructionBuilder().execute(trace))
    assert next(label.text for label in scene.labels if label.id == "measure:AB") == "√13"
    output = tmp_path / "math_label.png"
    write_png(scene, output, width=320, height=240)
    assert output.stat().st_size > 1000


def test_polygon_right_angle_is_inferred_for_rendering():
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": ["4", "0"]},
        {"op": "add_point", "id": "C", "coord": ["3", "sqrt(3)"]},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
        {"op": "add_segment", "id": "AC", "points": ["A", "C"]},
        {"op": "add_segment", "id": "BC", "points": ["B", "C"]},
        {"op": "add_polygon", "id": "tri_ABC", "vertices": ["A", "B", "C"]},
        {"op": "set_target", "type": "area", "object": "tri_ABC"},
    ]
    scene = state_to_render_scene(ConstructionBuilder().execute(trace))
    marker = next(primitive for primitive in scene.primitives if primitive.id == "right_angle:tri_ABC:C")
    assert marker.metadata["relation"] == "inferred_polygon_right_angle"


def test_foot_marker_uses_longer_interior_base_direction():
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": ["4", "0"]},
        {"op": "add_point", "id": "C", "coord": ["1", "3"]},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
        {"op": "foot_of_perpendicular", "id": "H", "point": "C", "line": "AB"},
        {"op": "set_target", "type": "length", "object": "AB"},
    ]
    scene = state_to_render_scene(ConstructionBuilder().execute(trace))
    marker = next(primitive for primitive in scene.primitives if primitive.id == "right_angle:H")
    assert marker.geometry["direction2"][0] > 0


def test_shared_label_layout_avoids_label_collisions():
    trace = [
        {"op": "add_point", "id": "A", "coord": ["-3", "0"]},
        {"op": "add_point", "id": "B", "coord": ["3", "0"]},
        {"op": "add_point", "id": "C", "coord": ["0", "-2"]},
        {"op": "add_point", "id": "D", "coord": ["0", "2"]},
        {"op": "add_point", "id": "O", "coord": ["0", "0"]},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"], "metadata": {"show_length": True}},
        {"op": "add_segment", "id": "CD", "points": ["C", "D"], "metadata": {"show_length": True}},
        {"op": "set_target", "type": "length", "object": "AB"},
    ]
    scene = state_to_render_scene(ConstructionBuilder().execute(trace))
    positioned = position_labels(scene, build_view_transform(scene))
    for index, left in enumerate(positioned):
        for right in positioned[index + 1:]:
            assert _box_overlap_area(left.bbox, right.bbox) == 0
    svg = render_svg(scene)
    assert 'data-id="label:O"' in svg
    assert 'data-id="measure:AB"' in svg


def _box_overlap_area(left, right):
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    return width * height
