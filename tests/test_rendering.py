"""Unit tests for render style policy and scene generation."""

import json
from pathlib import Path

import pytest

from geosym_world.construction.builder import ConstructionBuilder
from geosym_world.rendering import DeterministicStylePolicy, format_math_expression, inspect_render_scene, position_labels, render_regression_signature, render_svg, scene_fingerprint, state_to_render_scene, write_png
from geosym_world.rendering.layout import build_view_transform
from geosym_world.synthesis import select_template_families


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
    assert "render_quality" in left["metadata"]
    assert set(left["metadata"]["render_quality"]) == {"valid", "issues", "metrics"}
    assert left["metadata"]["scene_fingerprint"] == scene_fingerprint(state_to_render_scene(state, style_policy=DeterministicStylePolicy(seed=11)))


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


def test_region_holes_are_preserved_in_scene_and_svg():
    state = ConstructionBuilder().execute([
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": ["4", "0"]},
        {"op": "add_point", "id": "C", "coord": ["4", "3"]},
        {"op": "add_point", "id": "D", "coord": ["0", "3"]},
        {"op": "add_point", "id": "E", "coord": ["1", "1"]},
        {"op": "add_point", "id": "F", "coord": ["2", "1"]},
        {"op": "add_point", "id": "G", "coord": ["2", "2"]},
        {"op": "add_point", "id": "H", "coord": ["1", "2"]},
        {"op": "add_region", "id": "hole", "boundary": ["E", "F", "G", "H"]},
        {"op": "add_region", "id": "outer", "boundary": ["A", "B", "C", "D"], "holes": ["hole"]},
        {"op": "set_target", "type": "area", "object": "outer"},
    ])
    scene = state_to_render_scene(state)
    primitive = next(item for item in scene.primitives if item.id == "outer")
    assert len(primitive.geometry["holes"]) == 1
    svg = render_svg(scene)
    assert 'fill-rule="evenodd"' in svg
    output = Path("/tmp/geosym_region_hole.png")
    write_png(scene, output, width=320, height=240)
    from PIL import Image

    image = Image.open(output).convert("RGB")
    transform = build_view_transform(scene, width=320, height=240)
    hole_pixel = tuple(int(value) for value in image.getpixel(tuple(round(value) for value in transform.point((1.5, 1.5)))))
    outer_pixel = tuple(int(value) for value in image.getpixel(tuple(round(value) for value in transform.point((0.5, 0.5)))))
    assert hole_pixel == (255, 255, 255)
    assert outer_pixel != (255, 255, 255)


def test_sector_renders_exact_central_angle_marker_and_fill(tmp_path):
    from PIL import Image

    spec = next(iter(select_template_families(["circle_sector"])[0].variants()))
    scene = state_to_render_scene(ConstructionBuilder().execute(spec.trace))
    sector = next(item for item in scene.primitives if item.id == "sector_AOB")
    marker = next(item for item in scene.primitives if item.id == "central_angle:sector_AOB")
    angle_label = next(label for label in scene.labels if label.id == "measure:central_angle:sector_AOB")
    output = tmp_path / "sector.png"
    write_png(scene, output, width=360, height=300)

    assert sector.kind == "sector"
    assert sector.role == "target"
    assert marker.kind == "angle_marker"
    assert marker.geometry["orientation"] == "ccw"
    assert angle_label.text == "π⁄₃"
    svg = render_svg(scene, width=360, height=300)
    assert 'data-id="central_angle:sector_AOB"' in svg
    image = Image.open(output).convert("RGB")
    transform = build_view_transform(scene, width=360, height=300)
    sector_pixel = tuple(int(value) for value in image.getpixel(tuple(round(value) for value in transform.point((2.0, 0.8)))))
    outside_pixel = tuple(int(value) for value in image.getpixel(tuple(round(value) for value in transform.point((-1.2, -0.7)))))
    assert sector_pixel != (255, 255, 255)
    assert outside_pixel == (255, 255, 255)


def test_circular_segment_renders_arc_chord_shading(tmp_path):
    from PIL import Image

    spec = next(iter(select_template_families(["circular_segment_area"])[0].variants()))
    scene = state_to_render_scene(ConstructionBuilder().execute(spec.trace), include_labels=False)
    target = next(item for item in scene.primitives if item.id == "segment_AB")
    output = tmp_path / "circular_segment.png"
    write_png(scene, output, width=360, height=300)

    assert target.kind == "circular_segment"
    assert target.role == "target"
    assert "sector_AOB" not in {item.id for item in scene.primitives}
    assert "tri_OAB" not in {item.id for item in scene.primitives}
    svg = render_svg(scene, width=360, height=300)
    assert 'data-id="segment_AB"' in svg
    assert " A " in svg
    image = Image.open(output).convert("RGB")
    transform = build_view_transform(scene, width=360, height=300)
    segment_pixel = tuple(int(value) for value in image.getpixel(tuple(round(value) for value in transform.point((2.7, 0.9)))))
    triangle_pixel = tuple(int(value) for value in image.getpixel(tuple(round(value) for value in transform.point((1.5, 0.7)))))
    assert segment_pixel != (255, 255, 255)
    assert triangle_pixel == (255, 255, 255)


def test_circular_segment_perimeter_reuses_arc_chord_scene():
    spec = next(iter(select_template_families(["circular_segment_perimeter"])[0].variants()))
    scene = state_to_render_scene(ConstructionBuilder().execute(spec.trace), include_labels=False)
    target = next(item for item in scene.primitives if item.id == "segment_AB")

    assert target.kind == "circular_segment"
    assert target.role == "target"
    assert target.metadata["arc"] == "arc_AB"
    assert target.metadata["chord"] == "AB"


def test_equal_circle_lens_renders_two_arc_boundary(tmp_path):
    from PIL import Image

    spec = next(iter(select_template_families(["equal_circle_lens_area"])[0].variants()))
    scene = state_to_render_scene(ConstructionBuilder().execute(spec.trace), include_labels=False)
    target = next(item for item in scene.primitives if item.id == "lens_12")
    output = tmp_path / "lens.png"
    write_png(scene, output, width=420, height=320)

    assert target.kind == "lens"
    assert target.role == "target"
    assert len(target.geometry["arcs"]) == 2
    assert "arc_1" not in {item.id for item in scene.primitives}
    assert "arc_2" not in {item.id for item in scene.primitives}
    svg = render_svg(scene, width=420, height=320)
    assert 'data-id="lens_12"' in svg
    assert svg.count(" A ") >= 2
    image = Image.open(output).convert("RGB")
    transform = build_view_transform(scene, width=420, height=320)
    lens_pixel = tuple(int(value) for value in image.getpixel(tuple(round(value) for value in transform.point((1.0, 0.0)))))
    outside_pixel = tuple(int(value) for value in image.getpixel(tuple(round(value) for value in transform.point((-2.5, 0.0)))))
    assert lens_pixel != (255, 255, 255)
    assert outside_pixel == (255, 255, 255)


def test_circle_lens_renders_unequal_radius_two_arc_boundary(tmp_path):
    from PIL import Image

    spec = next(iter(select_template_families(["circle_lens_area"])[0].variants()))
    scene = state_to_render_scene(ConstructionBuilder().execute(spec.trace), include_labels=False)
    target = next(item for item in scene.primitives if item.id == "lens_12")
    output = tmp_path / "circle_lens.png"
    write_png(scene, output, width=420, height=320)

    assert target.kind == "lens"
    assert len(target.geometry["arcs"]) == 2
    assert target.geometry["arcs"][0]["radius"] != target.geometry["arcs"][1]["radius"]
    svg = render_svg(scene, width=420, height=320)
    assert svg.count(" A ") >= 2
    image = Image.open(output).convert("RGB")
    transform = build_view_transform(scene, width=420, height=320)
    lens_pixel = tuple(int(value) for value in image.getpixel(tuple(round(value) for value in transform.point((1.0, 0.0)))))
    outside_pixel = tuple(int(value) for value in image.getpixel(tuple(round(value) for value in transform.point((-2.5, 0.0)))))
    assert lens_pixel != (255, 255, 255)
    assert outside_pixel == (255, 255, 255)


@pytest.mark.parametrize("family_name", ["equal_circle_lens_perimeter", "circle_lens_perimeter"])
def test_lens_perimeter_reuses_curved_lens_scene(family_name):
    spec = next(iter(select_template_families([family_name])[0].variants()))
    scene = state_to_render_scene(ConstructionBuilder().execute(spec.trace), include_labels=False)
    target = next(item for item in scene.primitives if item.id == "lens_12")
    assert target.kind == "lens"
    assert target.metadata["perimeter_formula"]["rule"].endswith("_perimeter")


def test_disjoint_composite_region_renders_all_components():
    state = ConstructionBuilder().execute([
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": ["2", "0"]},
        {"op": "add_point", "id": "C", "coord": ["2", "2"]},
        {"op": "add_point", "id": "D", "coord": ["0", "2"]},
        {"op": "add_point", "id": "E", "coord": ["4", "0"]},
        {"op": "add_point", "id": "F", "coord": ["6", "0"]},
        {"op": "add_point", "id": "G", "coord": ["6", "2"]},
        {"op": "add_point", "id": "H", "coord": ["4", "2"]},
        {"op": "add_region", "id": "left", "boundary": ["A", "B", "C", "D"]},
        {"op": "add_region", "id": "right", "boundary": ["E", "F", "G", "H"]},
        {"op": "add_composite_region", "id": "combined", "operation": "union", "components": ["left", "right"]},
        {"op": "set_target", "type": "area", "object": "combined"},
    ])
    primitive = next(item for item in state_to_render_scene(state).primitives if item.id == "combined")
    assert primitive.geometry["components"]


def test_partition_target_renders_parent_once_and_keeps_partition_segment():
    spec = next(iter(select_template_families(["triangle_partition_area"])[0].variants()))
    scene = state_to_render_scene(ConstructionBuilder().execute(spec.trace))
    target_regions = [primitive for primitive in scene.primitives if primitive.role == "target"]
    polygon_ids = {primitive.id for primitive in scene.primitives if primitive.kind == "polygon"}

    assert [primitive.id for primitive in target_regions] == ["partition_ABC"]
    assert not {"tri_ABC", "tri_ACH", "tri_HBC", "region_ABC", "region_ACH", "region_HBC"}.intersection(polygon_ids)
    assert "CH" in {primitive.id for primitive in scene.primitives}


def test_area_ratio_scene_distinguishes_partition_regions():
    spec = next(iter(select_template_families(["triangle_partition_area_ratio"])[0].variants()))
    scene = state_to_render_scene(ConstructionBuilder().execute(spec.trace))
    regions = {primitive.id: primitive for primitive in scene.primitives if primitive.id in {"region_ACH", "region_HBC"}}

    assert set(regions) == {"region_ACH", "region_HBC"}
    assert regions["region_ACH"].style.fill != regions["region_HBC"].style.fill
    assert "partition_ABC" not in {primitive.id for primitive in scene.primitives}


def test_rectangular_frame_target_preserves_transparent_hole(tmp_path):
    from PIL import Image

    spec = next(iter(select_template_families(["rectangular_frame_area"])[0].variants()))
    scene = state_to_render_scene(ConstructionBuilder().execute(spec.trace))
    primitive = next(item for item in scene.primitives if item.id == "shaded_frame")
    output = tmp_path / "rectangular_frame.png"
    write_png(scene, output, width=360, height=280)

    assert primitive.role == "target"
    assert len(primitive.geometry["holes"]) == 1
    image = Image.open(output).convert("RGB")
    transform = build_view_transform(scene, width=360, height=280)
    frame_pixel = tuple(int(value) for value in image.getpixel(tuple(round(value) for value in transform.point((0.5, 0.5)))))
    hole_pixel = tuple(int(value) for value in image.getpixel(tuple(round(value) for value in transform.point((4.0, 3.0)))))
    assert frame_pixel != (255, 255, 255)
    assert hole_pixel == (255, 255, 255)


def test_concave_area_target_renders_boundary_notch_without_hidden_point(tmp_path):
    from PIL import Image

    spec = next(iter(select_template_families(["concave_composite_area"])[0].variants()))
    scene = state_to_render_scene(ConstructionBuilder().execute(spec.trace))
    primitive = next(item for item in scene.primitives if item.id == "concave_ABCDEF")
    output = tmp_path / "concave_area.png"
    write_png(scene, output, width=360, height=280)

    assert primitive.role == "target"
    assert len(primitive.geometry["points"]) == 6
    assert primitive.geometry["holes"] == []
    assert "G" not in {item.id for item in scene.primitives if item.kind == "point"}
    assert "label:G" not in {label.id for label in scene.labels}
    image = Image.open(output).convert("RGB")
    transform = build_view_transform(scene, width=360, height=280)
    region_pixel = tuple(int(value) for value in image.getpixel(tuple(round(value) for value in transform.point((1.0, 4.0)))))
    notch_pixel = tuple(int(value) for value in image.getpixel(tuple(round(value) for value in transform.point((5.0, 4.0)))))
    assert region_pixel != (255, 255, 255)
    assert notch_pixel == (255, 255, 255)


def test_double_rectangular_hole_target_preserves_both_transparent_holes(tmp_path):
    from PIL import Image

    spec = next(iter(select_template_families(["double_rectangular_hole_area"])[0].variants()))
    scene = state_to_render_scene(ConstructionBuilder().execute(spec.trace))
    primitive = next(item for item in scene.primitives if item.id == "perforated_plate")
    output = tmp_path / "double_rectangular_hole.png"
    write_png(scene, output, width=420, height=300)

    assert primitive.role == "target"
    assert len(primitive.geometry["holes"]) == 2
    assert {item.id for item in scene.primitives if item.role == "region"} == set()
    assert {label.id for label in scene.labels if label.id.startswith("label:")} == {"label:A", "label:B", "label:C", "label:D"}
    assert not {point_id for point_id in "EFGHIJKL" if point_id in {item.id for item in scene.primitives if item.kind == "point"}}
    assert {label.id for label in scene.labels if label.id.startswith("measure:")} == {
        "measure:AB",
        "measure:BC",
        "measure:EF",
        "measure:HE",
        "measure:IJ",
        "measure:LI",
    }
    assert scene.metadata["render_quality"]["valid"]
    image = Image.open(output).convert("RGB")
    transform = build_view_transform(scene, width=420, height=300)
    plate_pixel = tuple(int(value) for value in image.getpixel(tuple(round(value) for value in transform.point((5.0, 2.0)))))
    left_hole_pixel = tuple(int(value) for value in image.getpixel(tuple(round(value) for value in transform.point((2.0, 2.0)))))
    right_hole_pixels = [
        tuple(int(value) for value in image.getpixel(tuple(round(value) for value in transform.point(point))))
        for point in ((9.5, 4.5), (9.5, 5.5), (10.5, 4.5), (10.5, 5.5))
    ]
    assert plate_pixel != (255, 255, 255)
    assert left_hole_pixel == (255, 255, 255)
    assert all(pixel == (255, 255, 255) for pixel in right_hole_pixels)


def test_annulus_target_preserves_transparent_inner_disk(tmp_path):
    from PIL import Image

    spec = next(iter(select_template_families(["annulus_area"])[0].variants()))
    scene = state_to_render_scene(ConstructionBuilder().execute(spec.trace))
    primitive = next(item for item in scene.primitives if item.id == "annulus_O")
    output = tmp_path / "annulus.png"
    write_png(scene, output, width=360, height=300)

    assert primitive.role == "target"
    assert primitive.geometry["inner_radius"] == 2.0
    assert primitive.geometry["outer_radius"] == 5.0
    svg = render_svg(scene, width=360, height=300)
    assert 'data-id="annulus_O"' in svg
    assert 'fill-rule="evenodd"' in svg
    assert 'data-id="annulus_O:hole"' not in svg
    image = Image.open(output).convert("RGB")
    transform = build_view_transform(scene, width=360, height=300)
    inner_pixel = tuple(int(value) for value in image.getpixel(tuple(round(value) for value in transform.point((1.0, 1.0)))))
    annulus_pixel = tuple(int(value) for value in image.getpixel(tuple(round(value) for value in transform.point((3.0, 1.0)))))
    assert inner_pixel == (255, 255, 255)
    assert annulus_pixel != (255, 255, 255)


def test_annulus_inner_disk_preserves_underlying_region(tmp_path):
    from PIL import Image

    annulus = next(iter(select_template_families(["annulus_area"])[0].variants()))
    background = [
        {"op": "add_point", "id": "P", "coord": ["-6", "-6"], "metadata": {"render": False}},
        {"op": "add_point", "id": "Q", "coord": ["6", "-6"], "metadata": {"render": False}},
        {"op": "add_point", "id": "R", "coord": ["6", "6"], "metadata": {"render": False}},
        {"op": "add_point", "id": "S", "coord": ["-6", "6"], "metadata": {"render": False}},
        {"op": "add_polygon", "id": "background_polygon", "vertices": ["P", "Q", "R", "S"], "metadata": {"render": False}},
        {"op": "mark_polygon_region", "id": "background_region", "polygon": "background_polygon"},
    ]
    scene = state_to_render_scene(ConstructionBuilder().execute([*background, *annulus.trace]), include_labels=False)
    output = tmp_path / "annulus_overlay.png"
    write_png(scene, output, width=360, height=300)

    image = Image.open(output).convert("RGB")
    transform = build_view_transform(scene, width=360, height=300)
    inner_pixel = tuple(int(value) for value in image.getpixel(tuple(round(value) for value in transform.point((1.0, 1.0)))))
    background_pixel = tuple(int(value) for value in image.getpixel(tuple(round(value) for value in transform.point((5.5, 5.5)))))
    annulus_pixel = tuple(int(value) for value in image.getpixel(tuple(round(value) for value in transform.point((3.0, 1.0)))))
    assert inner_pixel == background_pixel
    assert annulus_pixel != background_pixel


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


def test_non_rendered_point_does_not_suppress_visible_implicit_intersection():
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": ["4", "4"]},
        {"op": "add_point", "id": "C", "coord": ["0", "4"]},
        {"op": "add_point", "id": "D", "coord": ["4", "0"]},
        {"op": "add_point", "id": "X", "coord": ["2", "2"], "metadata": {"render": False}},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
        {"op": "add_segment", "id": "CD", "points": ["C", "D"]},
        {"op": "set_target", "type": "length", "object": "AB"},
    ]

    scene = state_to_render_scene(ConstructionBuilder().execute(trace))

    assert "X" not in {primitive.id for primitive in scene.primitives}
    assert "implicit_intersection:AB|CD" in {primitive.id for primitive in scene.primitives}


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
    assert format_math_expression("x/(y + 1)") == "x⁄(y + 1)"
    assert format_math_expression("(sqrt(3) + 1)/2") == "(1 + √3)⁄₂"
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


def test_render_quality_report_accepts_shared_layout_for_auxiliary_geometry():
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": ["4", "0"]},
        {"op": "add_point", "id": "C", "coord": ["1", "3"]},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"], "metadata": {"show_length": True}},
        {"op": "foot_of_perpendicular", "id": "H", "point": "C", "line": "AB"},
        {"op": "set_target", "type": "area", "object": "ABC"},
    ]
    scene = state_to_render_scene(ConstructionBuilder().execute(trace))
    report = inspect_render_scene(scene)
    assert report.valid
    assert report.metrics["num_errors"] == 0


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
    assert f'data-scene-fingerprint="{scene.metadata["scene_fingerprint"]}"' in svg


def test_render_scene_fixture_has_stable_fingerprint():
    fixture = json.loads((Path(__file__).parent / "fixtures" / "render_scene_baseline.json").read_text(encoding="utf-8"))
    scene = state_to_render_scene(ConstructionBuilder().execute(fixture["trace"]), scene_id=fixture["id"])
    repeated = state_to_render_scene(ConstructionBuilder().execute(fixture["trace"]), scene_id=fixture["id"])
    assert scene.metadata["scene_fingerprint"] == repeated.metadata["scene_fingerprint"]
    assert scene.metadata["render_quality"]["metrics"]["num_primitives"] == len(scene.primitives)
    assert render_regression_signature(scene, **fixture["viewport"]) == fixture["signature"]


def test_render_regression_signature_detects_geometry_change():
    fixture = json.loads((Path(__file__).parent / "fixtures" / "render_scene_baseline.json").read_text(encoding="utf-8"))
    changed_trace = json.loads(json.dumps(fixture["trace"]))
    point_c = next(step for step in changed_trace if step.get("id") == "C")
    point_c["coord"] = ["2", "3"]
    changed_scene = state_to_render_scene(ConstructionBuilder().execute(changed_trace), scene_id=fixture["id"])
    changed = render_regression_signature(changed_scene, **fixture["viewport"])

    assert changed["scene_fingerprint"] != fixture["signature"]["scene_fingerprint"]
    assert changed["svg_sha256"] != fixture["signature"]["svg_sha256"]
    assert changed["png_rgb_sha256"] != fixture["signature"]["png_rgb_sha256"]
    assert changed["png_foreground_mask_sha256"] != fixture["signature"]["png_foreground_mask_sha256"]


def test_render_regression_signature_rejects_invalid_viewport():
    fixture = json.loads((Path(__file__).parent / "fixtures" / "render_scene_baseline.json").read_text(encoding="utf-8"))
    scene = state_to_render_scene(ConstructionBuilder().execute(fixture["trace"]), scene_id=fixture["id"])

    import pytest

    with pytest.raises(ValueError, match="dimensions must be positive"):
        render_regression_signature(scene, width=0, height=768)
    with pytest.raises(ValueError, match="non-empty viewport"):
        render_regression_signature(scene, width=100, height=100, padding=50)


def test_representative_render_regression_matrix():
    fixture = json.loads((Path(__file__).parent / "fixtures" / "render_regression_matrix.json").read_text(encoding="utf-8"))

    assert fixture["schema_version"] == "geosym.render_regression_matrix.v1"
    assert len({case["family"] for case in fixture["cases"]}) == len(fixture["cases"])
    for case in fixture["cases"]:
        spec = next(iter(select_template_families([case["family"]])[0].variants()))
        assert spec.id == case["spec_id"]
        scene = state_to_render_scene(ConstructionBuilder().execute(spec.trace), scene_id=spec.id)
        quality = inspect_render_scene(scene, **fixture["viewport"])
        assert quality.valid, [issue.to_dict() for issue in quality.issues]
        assert render_regression_signature(scene, **fixture["viewport"]) == case["signature"]


def _box_overlap_area(left, right):
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    return width * height
