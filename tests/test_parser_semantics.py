from copy import deepcopy

import pytest
from PIL import Image

from geosym_world.construction import ConstructionBuilder
from geosym_world.errors import InvalidGeometryInput
from geosym_world.parser import ParserInput, ParserPipeline, parse_one_openrouter
from geosym_world.parser import openrouter
from geosym_world.parser.semantic_compiler import compile_semantic_payload


def _parse(payload, tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    Image.new("RGB", (120, 80), "white").save(source)
    monkeypatch.setattr(openrouter, "_request_json_completion", lambda **kwargs: ({"usage": {}}, deepcopy(payload)))
    parser_input = ParserInput(id="semantics", image=str(source), question="")
    prediction = parse_one_openrouter(parser_input, model="fixture", api_key="fixture", base_url="https://fixture.invalid/v1")
    return parser_input, prediction


def test_coordinate_free_parallel_triangle_reaches_verified_replayable_seed(tmp_path, monkeypatch):
    payload = {
        "points": ["P", "Q", "R", "S", "T"],
        "strokes": [{"kind": "segment", "points": points} for points in [["P", "S", "R"], ["P", "T", "Q"], ["R", "Q"], ["S", "T"]]],
        "relations": [{"type": "parallel", "on": [{"segment": ["S", "T"]}, {"segment": ["R", "Q"]}], "basis": "mark"}],
    }
    parser_input, prediction = _parse(payload, tmp_path, monkeypatch)
    graph = prediction.parse_graph
    assert all(not item.geometry for item in graph.objects)
    assert sum(item.kind.value == "segment" for item in graph.objects) == 4
    assert sum(item.predicate == "point_on_segment" for item in graph.relations) == 2
    result = ParserPipeline(lambda _: prediction).run(parser_input)
    assert result.quality.seed_realized and result.quality.constraint_satisfied
    assert result.realization.metadata["coordinate_source"] in {"constraint_seed", "refined_constraint_seed"}
    assert result.realization.metadata["source_facts_admitted"] is False
    assert result.quality.source_reconstruction_passed is None
    state = result.realization.geometry_state
    assert ConstructionBuilder().execute(state.construction_trace).checksum() == state.checksum()
    reversed_payload = deepcopy(payload)
    reversed_payload["points"].reverse()
    reversed_payload["strokes"].append({"kind": "segment", "points": ["Q", "R"]})
    _, second = _parse(reversed_payload, tmp_path, monkeypatch)
    assert {item.id for item in graph.objects} == {item.id for item in second.parse_graph.objects}


def test_semantic_seed_preserves_polygon_shape_when_ordered_relations_overlap(tmp_path, monkeypatch):
    payload = {
        "points": ["A", "B", "C", "D"],
        "strokes": [
            {"kind": "segment", "points": ["A", "B"]},
            {"kind": "segment", "points": ["B", "C"]},
            {"kind": "segment", "points": ["C", "D"]},
            {"kind": "segment", "points": ["D", "A"]},
            {"kind": "segment", "points": ["A", "D", "C"]},
        ],
        "relations": [],
    }
    parser_input, prediction = _parse(payload, tmp_path, monkeypatch)
    result = ParserPipeline(lambda _: prediction).run(parser_input)

    positions = result.realization.point_positions
    point_ids = {
        item.visible_name: item.id
        for item in prediction.parse_graph.objects
        if item.kind.value == "point"
    }
    first, second, third, fourth = (positions[point_ids[name]] for name in "ABCD")
    area = sum(
        left[0] * right[1] - right[0] * left[1]
        for left, right in zip((first, second, third, fourth), (second, third, fourth, first))
    )
    assert result.quality.skeleton_redraw
    assert result.realization.metadata["seed_refinement"] == "shape_preserving"
    assert abs(area) > 1e-6


def test_curve_semantics_keep_membership_tangency_and_typed_values(tmp_path, monkeypatch):
    payload = {
        "points": ["O", "A", "B", "C", "D"], "strokes": [{"kind": "line", "points": ["A", "D"]}],
        "circles": [{"name": "ring", "center": "O", "through": ["A", "B"]}],
        "arcs": [{"name": "curve", "circle": "ring", "ends": ["A", "B"], "through": ["C"]}],
        "relations": [{"type": "tangent", "on": [{"circle": "ring"}, {"line": ["A", "D"]}], "basis": "text"}],
        "measurements": [{"type": "radius", "on": {"circle": "ring"}, "value": "sqrt(3)", "basis": "text"}],
        "target": {"type": "arc_length", "on": {"arc": "curve"}, "mode": "value"},
    }
    _, prediction = _parse(payload, tmp_path, monkeypatch)
    graph = prediction.parse_graph
    assert {item.predicate for item in graph.relations} == {"point_on_circle", "point_on_arc", "tangent"}
    assert graph.metadata["compilation"]["dropped_candidates"] == 0
    assert graph.quantities[0].kind == "radius"
    assert str(graph.quantities[0].expression.to_sympy()) == "sqrt(3)"
    assert graph.quantities[0].metadata["show_label"] is False
    assert graph.targets[0].kind == "arc_length"
    assert all(not item.geometry for item in graph.objects)


def test_region_quantity_and_ratio_target_use_declared_regions(tmp_path, monkeypatch):
    payload = {
        "points": ["A", "B", "C", "D"], "strokes": [],
        "regions": [{"name": "left", "boundary": ["A", "B", "C"], "shaded": True}, {"name": "right", "boundary": ["A", "C", "D"]}],
        "measurements": [{"type": "area", "on": {"region": "left"}, "value": "2/3"}],
        "target": {"type": "ratio", "on": [{"region": "left"}, {"region": "right"}], "mode": "value"},
    }
    _, prediction = _parse(payload, tmp_path, monkeypatch)
    graph = prediction.parse_graph
    region_ids = {item.id for item in graph.regions}
    assert graph.quantities[0].arguments[0].ref in region_ids
    assert [item.role for item in graph.targets[0].arguments] == ["numerator", "denominator"]
    assert {item.ref for item in graph.targets[0].arguments} == region_ids
    assert graph.metadata["compilation"]["dropped_candidates"] == 0
    assert not any(item.kind.value == "unknown" for item in graph.objects)


def test_external_semantics_quarantine_malformed_advanced_candidates(tmp_path, monkeypatch):
    payload = {
        "points": ["A", "B", "C"],
        "strokes": [{"kind": "segment", "points": ["A", "B", "C"]}],
        "regions": [{
            "name": "outer",
            "boundary": [{"segment": ["A", "B"]}, {"segment": ["B", "C"]}],
            "holes": [{"polygon": ["A", "B", "C"]}],
        }],
        "measurements": [{
            "type": "angle", "on": {"arc": "missing_arc"}, "value": "30",
        }],
        "target": {"type": "angle", "on": {"polygon": ["A", "B", "C"]}},
    }
    _, prediction = _parse(payload, tmp_path, monkeypatch)
    graph = prediction.parse_graph

    assert {item.visible_name for item in graph.objects if item.visible_name} >= {"A", "B", "C"}
    assert len(graph.regions) == 2
    assert not graph.targets
    audit = graph.metadata["normalization_audit"]
    reasons = {item["reason"] for item in audit["dropped_candidates"]}
    assert "angle_requires_three_point_reference" in reasons
    assert any(item["from"] == "polygon_hole" for item in audit["field_aliases"])


@pytest.mark.parametrize("mutation", [
    {"points": [{"name": "A", "x": 10, "y": 20}]},
    {"strokes": [{"kind": "segment", "points": ["A", "missing"]}]},
    {"strokes": [{"kind": "segment", "points": ["A", "A"]}]},
    {"layout": {"A": [0, 0]}},
    {"relations": [{"type": "perpendicular", "on": ["A", "B"]}]},
    {"measurements": [{"type": "radius", "on": {"segment": ["A", "B"]}, "value": "3"}]},
])
def test_semantic_contract_rejects_coordinates_and_invalid_references(mutation):
    with pytest.raises(InvalidGeometryInput):
        compile_semantic_payload({"points": ["A", "B"], "strokes": [], **mutation})


def test_unknown_circle_center_is_preserved_without_invented_points():
    result = compile_semantic_payload({"points": ["A", "B"], "strokes": [], "circles": [{"name": "c1", "through": ["A", "B"]}], "uncertain": ["circle center not given"]})
    assert len(result["objects"]) == 3
    circle = next(item for item in result["objects"] if item["type"] == "circle")
    assert circle["references"] == []
    assert result["metadata"]["ambiguities"] == ["circle center not given"]


def test_anonymous_intersection_and_angle_bindings_need_no_coordinates(tmp_path, monkeypatch):
    payload = {
        "points": ["A", "B", "C", "D", "_1"],
        "strokes": [{"kind": "segment", "points": ["A", "_1", "B"]}, {"kind": "segment", "points": ["C", "_1", "D"]}],
        "relations": [{"type": "equal_angle", "on": [{"angle": ["A", "_1", "C"]}, {"angle": ["B", "_1", "D"]}], "basis": "mark"}],
        "measurements": [{"type": "angle", "on": {"angle": ["A", "_1", "C"]}, "value": "x+10", "unit": "degree"}],
    }
    parser_input, prediction = _parse(payload, tmp_path, monkeypatch)
    graph = prediction.parse_graph
    anonymous = next(item for item in graph.objects if item.metadata.get("anonymous"))
    assert anonymous.visible_name is None
    incidence = [item for item in graph.relations if item.predicate == "point_on_segment"]
    assert len(incidence) == 2
    assert all(item.arguments[0].ref == anonymous.id for item in incidence)
    equal_angle = next(item for item in graph.relations if item.predicate == "equal_angle")
    assert len(equal_angle.arguments) == 6
    assert graph.quantities[0].arguments[1].ref == anonymous.id
    result = ParserPipeline(lambda _: prediction).run(parser_input)
    assert result.realization.geometry_state is not None
    assert result.realization.geometry_state.points[anonymous.id].metadata["show_label"] is False
    assert not any(label.text == anonymous.id for label in result.realization.scene.labels)


def test_polygon_rotation_and_reverse_keep_one_object():
    result = compile_semantic_payload({
        "points": ["U", "V", "W"], "strokes": [],
        "measurements": [{"type": "area", "on": {"polygon": ["U", "V", "W"]}, "value": "6"}],
        "target": {"type": "perimeter", "on": {"polygon": ["W", "V", "U"]}, "mode": "value"},
    })
    assert sum(item["type"] == "polygon" for item in result["objects"]) == 1
    assert result["quantities"][0]["arguments"][0]["ref"] == result["targets"][0]["arguments"][0]["ref"]


def test_semantic_pipeline_never_uses_legacy_repairs_or_pixel_checks(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Coordinate-free semantics must not enter legacy repair or pixel verification")

    monkeypatch.setattr(openrouter, "_augment_explicit_text_claims", forbidden)
    payload = {
        "points": ["A", "B"], "strokes": [{"kind": "segment", "points": ["A", "B"]}],
        "measurements": [{"type": "length", "on": {"segment": ["A", "B"]}, "value": "3", "unit": "", "basis": "text"}],
        "uncertain": ["Only this side is specified"],
    }
    parser_input, prediction = _parse(payload, tmp_path, monkeypatch)
    assert prediction.parse_graph.quantities[0].origin.value == "text"
    assert prediction.parse_graph.quantities[0].unit is None
    assert prediction.parse_graph.quantities[0].score_kind == "uncalibrated_model"
    assert prediction.parse_graph.metadata["semantic_payload"] == payload

    class ForbiddenPixels:
        verify = staticmethod(forbidden)

    result = ParserPipeline(lambda _: prediction, pixel_topology_verifier=ForbiddenPixels()).run(parser_input)
    assert result.pixel_topology is None
    assert result.quality.pixel_topology_passed is None
    assert "pixel_topology_not_applicable" in result.quality.reasons
    assert result.quality.metrics["uncertain_statements"] == 1
    assert result.quality.constraint_satisfied
