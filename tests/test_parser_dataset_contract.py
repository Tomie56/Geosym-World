import pytest

from geosym_world.benchmark import build_parser_benchmark_pair, synthetic_state_to_parse_graph, validate_parser_reference
from geosym_world.construction.builder import ConstructionBuilder
from geosym_world.parser import CompletionStatus, ParserInput, build_constraint_completion_plan, validate_parser_input
from geosym_world.rendering import RenderPrimitive, RenderScene, state_to_render_scene
from geosym_world.schema import ParseGraph, ParseObjectKind, stable_checksum
from geosym_world.synthesis import select_template_families


def _fixture():
    trace = [
        {"op": "add_point", "id": "A", "coords": [0, 0]},
        {"op": "add_point", "id": "B", "coords": [4, 0]},
        {"op": "add_segment", "id": "AB", "start": "A", "end": "B", "metadata": {"show_length": True}},
        {"op": "set_target", "type": "length", "object": "AB"},
    ]
    state = ConstructionBuilder().execute(trace)
    scene = state_to_render_scene(state, scene_id="parser_contract")
    record = {
        "id": "parser_contract",
        "question": "Find AB.",
        "answer": "4",
        "image": {"formats": {"png": "/tmp/parser_contract.png"}},
        "public_givens": [],
        "facts": [],
        "derivation": {"steps": []},
        "construction_trace": trace,
    }
    return record, state, scene


def test_parser_input_rejects_private_fields():
    payload = ParserInput(id="sample", image="sample.png", question="Find x.").to_dict()
    assert validate_parser_input(payload).id == "sample"

    for forbidden in ("formal_state", "scene", "answer", "facts", "construction_trace", "target"):
        with pytest.raises(ValueError, match="unsupported fields"):
            validate_parser_input({**payload, forbidden: {}})
    with pytest.raises(ValueError, match="image path"):
        validate_parser_input({**payload, "image": 3})


def test_parser_benchmark_pair_separates_public_input_and_private_reference():
    record, state, scene = _fixture()
    parser_input, reference = build_parser_benchmark_pair(record, state=state, scene=scene)

    assert set(parser_input) == {"schema_version", "id", "image", "question"}
    assert parser_input["image"].endswith(".png")
    assert reference["private_supervision"]["answer"] == "4"
    assert reference["private_supervision"]["formal_state"]["segments"]["AB"]["start"] == "A"
    assert set(reference["observable_state"]["objects"]["points"]) == {"A", "B"}
    assert set(reference["observable_state"]["objects"]["segments"]) == {"AB"}
    assert reference["parser_input_checksum"] == stable_checksum(parser_input)
    validate_parser_reference(reference, parser_input)


def test_parser_reference_detects_input_or_reference_tampering():
    record, state, scene = _fixture()
    parser_input, reference = build_parser_benchmark_pair(record, state=state, scene=scene)

    with pytest.raises(ValueError, match="input checksum"):
        validate_parser_reference(reference, {**parser_input, "question": "Changed"})
    with pytest.raises(ValueError, match="reference checksum"):
        validate_parser_reference({**reference, "observable_state": {}}, parser_input)


def test_synthetic_typed_reference_round_trips_and_reconstructs_exact_state():
    record, state, scene = _fixture()

    graph = synthetic_state_to_parse_graph(
        record,
        state=state,
        scene=scene,
        source_id="source_parser_contract",
    )
    restored = ParseGraph.from_dict(graph.to_dict())
    completion = build_constraint_completion_plan(restored)

    assert restored.point_names() == {"A": "A", "B": "B"}
    assert {item.predicate for item in restored.relations} == {"binds"}
    assert restored.targets[0].arguments[0].ref == "AB"
    assert all(item.geometry.get("exact_coords") for item in restored.objects if item.id in {"A", "B"})
    assert completion.status is CompletionStatus.EXACT
    completed_state = ConstructionBuilder().execute(completion.trace)
    assert set(completed_state.points) == {"A", "B"}
    assert set(completed_state.segments) == {"AB"}


def test_synthetic_typed_reference_keeps_visible_unnamed_points_visible():
    trace = [
        {"op": "add_point", "id": "U", "coords": [0, 0], "metadata": {"show_label": False}},
        {"op": "add_point", "id": "V", "coords": [3, 0]},
        {"op": "add_segment", "id": "UV", "start": "U", "end": "V"},
    ]
    state = ConstructionBuilder().execute(trace)
    scene = state_to_render_scene(state, scene_id="unnamed_point")
    graph = synthetic_state_to_parse_graph(
        {"id": "unnamed_point", "question": "", "public_givens": []},
        state=state,
        scene=scene,
        source_id="source_unnamed_point",
    )

    point = next(item for item in graph.objects if item.id == "U")
    assert point.visible is True
    assert point.visible_name is None
    assert "U" not in graph.point_names()


def test_synthetic_typed_reference_projects_visible_marker_truth():
    trace = [
        {"op": "add_point", "id": "A", "coords": [0, 0]},
        {"op": "add_point", "id": "B", "coords": [4, 0]},
        {"op": "add_point", "id": "C", "coords": [1, 3]},
        {"op": "add_segment", "id": "AB", "start": "A", "end": "B"},
        {"op": "foot_of_perpendicular", "id": "H", "point": "C", "line": "AB"},
    ]
    state = ConstructionBuilder().execute(trace)
    scene = state_to_render_scene(state, scene_id="marker_truth")

    graph = synthetic_state_to_parse_graph(
        {"id": "marker_truth", "question": "", "public_givens": []},
        state=state,
        scene=scene,
        source_id="source_marker_truth",
    )

    markers = [item for item in graph.objects if item.kind is ParseObjectKind.MARKER]
    assert len(markers) == 1
    marker = markers[0]
    assert marker.metadata == {
        "marker_kind": "right_angle",
        "render_primitive_id": "right_angle:H",
        "relation": "foot_of_perpendicular",
    }
    assert {(reference.role, reference.ref) for reference in marker.references} == {
        ("vertex", "H"),
        ("from", "C"),
        ("line", "AB"),
    }
    marker_evidence = next(item for item in graph.evidence if item.id == marker.evidence_refs[0])
    assert marker_evidence.bbox is not None
    assert marker_evidence.raw_content == "right_angle"
    assert graph.metadata["num_visible_markers"] == 1
    ParseGraph.from_dict(graph.to_dict())


def test_synthetic_typed_reference_projects_all_canonical_marker_kinds():
    state = ConstructionBuilder().execute([])
    scene = RenderScene(
        id="marker_truth_kinds",
        primitives=(
            RenderPrimitive(
                id="angle",
                kind="angle_marker",
                role="marker",
                geometry={"vertex": (0.0, 0.0), "arm1": (1.0, 0.0), "arm2": (0.0, 1.0)},
            ),
            RenderPrimitive(
                id="right_angle",
                kind="right_angle_marker",
                role="marker",
                geometry={"vertex": (1.0, 1.0), "arm1": (2.0, 1.0), "arm2": (1.0, 2.0)},
            ),
            RenderPrimitive(
                id="tick",
                kind="tick_marker",
                role="marker",
                geometry={"start": (2.0, 0.0), "end": (3.0, 0.0)},
            ),
            RenderPrimitive(
                id="parallel",
                kind="parallel_marker",
                role="marker",
                geometry={"points": [(3.0, 1.0), (3.2, 1.2), (3.4, 1.0)]},
            ),
        ),
        metadata={"render_dim": 2},
    )

    graph = synthetic_state_to_parse_graph(
        {"id": "marker_truth_kinds", "question": "", "public_givens": []},
        state=state,
        scene=scene,
        source_id="source_marker_truth_kinds",
    )

    assert sorted(
        item.metadata["marker_kind"]
        for item in graph.objects
        if item.kind is ParseObjectKind.MARKER
    ) == ["angle_arc", "equal_length_tick", "parallel", "right_angle"]


def test_synthetic_typed_reference_rejects_3d_render_scenes():
    spec = next(iter(select_template_families(["tetrahedron_section"])[0].variants()))
    state = ConstructionBuilder().execute(spec.trace)
    scene = state_to_render_scene(state, scene_id=spec.id)

    with pytest.raises(ValueError, match="require a 2D render scene"):
        synthetic_state_to_parse_graph(
            {"id": spec.id, "question": spec.question, "public_givens": []},
            state=state,
            scene=scene,
            source_id=f"source_{spec.id}",
        )


def test_synthetic_typed_reference_projects_executable_relations_canonically():
    trace = [
        {"op": "add_point", "id": "A", "coords": [0, 0]},
        {"op": "add_point", "id": "B", "coords": [6, 0]},
        {"op": "add_point", "id": "C", "coords": [2, 4]},
        {"op": "add_segment", "id": "AB", "start": "A", "end": "B"},
        {"op": "add_ray", "id": "ray_AB", "start": "A", "through": "B"},
        {"op": "foot_of_perpendicular", "id": "H", "point": "C", "line": "AB"},
        {"op": "point_on_segment_ratio", "id": "M", "segment": "AB", "ratio": "1/3"},
        {"op": "point_on_ray_distance", "id": "D", "ray": "ray_AB", "distance": "1"},
        {"op": "parallel_through", "id": "parallel_C", "point": "C", "line": "AB"},
        {
            "op": "line_through_angle",
            "id": "angle_C",
            "point": "C",
            "base_line": "AB",
            "angle": "pi/4",
        },
        {
            "op": "angle_bisector",
            "id": "bisector_A",
            "vertex": "A",
            "ray1": ["A", "B"],
            "ray2": ["A", "C"],
        },
    ]
    state = ConstructionBuilder().execute(trace)
    scene = state_to_render_scene(state, scene_id="canonical_relations")
    public_givens = [
        {
            "kind": "relation",
            "relation_id": relation["id"],
            "type": relation["type"],
            "payload": {
                key: value
                for key, value in relation.items()
                if key not in {"id", "result", "metadata"}
            },
        }
        for relation in state.relations
    ]

    graph = synthetic_state_to_parse_graph(
        {"id": "canonical_relations", "question": "", "public_givens": public_givens},
        state=state,
        scene=scene,
        source_id="source_canonical_relations",
    )
    relations = {relation.predicate: relation for relation in graph.relations if relation.predicate != "binds"}

    assert [argument.role for argument in relations["foot_of_perpendicular"].arguments] == [
        "point",
        "from",
        "line",
    ]
    assert [argument.role for argument in relations["point_on_segment_ratio"].arguments] == [
        "point",
        "segment",
    ]
    assert relations["point_on_segment_ratio"].metadata["ratio"]["operator"] == "div"
    assert [argument.role for argument in relations["point_on_ray_distance"].arguments] == ["point", "ray"]
    assert relations["point_on_ray_distance"].metadata["distance"]["operator"] == "literal"
    assert [argument.role for argument in relations["parallel"].arguments] == ["left", "right"]
    assert [argument.role for argument in relations["angle_from_line"].arguments] == [
        "line",
        "point",
        "base_line",
    ]
    assert relations["angle_from_line"].metadata["angle"]["operator"] == "div"
    assert [argument.role for argument in relations["angle_bisector"].arguments] == [
        "line",
        "arm_0",
        "vertex",
        "arm_1",
    ]
    assert all("source_relation_id" in relation.metadata for relation in relations.values())



def test_synthetic_typed_reference_preserves_intersections_and_segment_orientation():
    trace = [
        {"op": "add_point", "id": "A", "coords": [0, 0]},
        {"op": "add_point", "id": "B", "coords": [4, 0]},
        {"op": "add_point", "id": "C", "coords": [2, -2]},
        {"op": "add_point", "id": "D", "coords": [2, 2]},
        {"op": "add_segment", "id": "BA", "start": "B", "end": "A"},
        {"op": "add_segment", "id": "CD", "start": "C", "end": "D"},
        {"op": "intersection_line_line", "id": "X", "line1": "BA", "line2": "CD"},
        {"op": "point_on_segment_ratio", "id": "R", "points": ["A", "B"], "ratio": "1/4"},
    ]
    state = ConstructionBuilder().execute(trace)
    scene = state_to_render_scene(state, scene_id="intersection_orientation")
    public_givens = [
        {
            "kind": "relation",
            "relation_id": relation["id"],
            "type": relation["type"],
            "payload": {
                key: value
                for key, value in relation.items()
                if key not in {"id", "result", "metadata"}
            },
        }
        for relation in state.relations
    ]

    graph = synthetic_state_to_parse_graph(
        {"id": "intersection_orientation", "question": "", "public_givens": public_givens},
        state=state,
        scene=scene,
        source_id="source_intersection_orientation",
    )
    relations = {relation.predicate: relation for relation in graph.relations if relation.predicate != "binds"}

    assert [argument.role for argument in relations["intersection"].arguments] == [
        "support",
        "support",
        "point",
    ]
    assert [argument.ref for argument in relations["point_on_segment_ratio"].arguments] == ["R", "BA"]
    assert relations["point_on_segment_ratio"].metadata["ratio"] == {
        "operator": "div",
        "value": None,
        "operands": [
            {"operator": "literal", "value": "3", "operands": []},
            {"operator": "literal", "value": "4", "operands": []},
        ],
    }

    restored = ParseGraph.from_dict(graph.to_dict())
    assert next(item for item in restored.objects if item.id == "R").geometry["exact_coords"]
