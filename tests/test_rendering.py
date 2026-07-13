"""Unit tests for render style policy and scene generation."""

from geosym_world_geometry.builder import ConstructionBuilder
from geosym_world_geometry.rendering import DeterministicStylePolicy, state_to_render_scene


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
