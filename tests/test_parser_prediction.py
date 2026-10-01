import json
from importlib.resources import files

import pytest
from PIL import Image

from geosym_world.errors import InvalidGeometryInput
from geosym_world.parser import ExternalParseGraphPrediction, ParserInput, parse_one_openrouter
from geosym_world.parser import openrouter
from geosym_world.schema import ParseGraph


def test_region_target_binding_does_not_report_a_declared_region_as_missing(tmp_path, monkeypatch):
    image = tmp_path / "image.png"
    Image.new("RGB", (100, 100), "white").save(image)
    payload = {
        "objects": [
            {"id": name, "type": "point", "label": name, "x": coords[0], "y": coords[1]}
            for name, coords in [("A", (10, 10)), ("B", (90, 10)), ("C", (10, 90))]
        ],
        "regions": [{"id": "region", "boundary": ["A", "B", "C"]}],
        "targets": [{"id": "ask", "type": "area", "region": "region"}],
    }
    monkeypatch.setattr(openrouter, "_request_json_completion", lambda **_: ({"usage": {}}, payload))

    graph = parse_one_openrouter(
        ParserInput(id="sample", image=str(image), question=""), model="fixture", api_key="offline",
        base_url="https://offline.invalid/v1",
    ).parse_graph

    assert graph.targets[0].mode == "value"
    assert graph.targets[0].arguments[0].ref == "region"
    assert not graph.metadata["normalization_audit"]["unresolved_references"]


def test_external_prediction_round_trips_without_private_state(tmp_path):
    image = tmp_path / "diagram.png"
    image.write_bytes(b"image")
    parser_input = ParserInput(id="sample", image=str(image), question="Find AB.")
    prediction = ExternalParseGraphPrediction.create(
        parser_input,
        backend="fixture",
        model="fixture-model",
        model_revision="v1",
        prompt_version="fixture-prompt",
        parse_graph=ParseGraph(id="graph_sample", source_id="sample"),
    )

    restored = ExternalParseGraphPrediction.from_dict(prediction.to_dict())

    assert restored.to_dict() == prediction.to_dict()


def test_external_prediction_rejects_tampering(tmp_path):
    image = tmp_path / "diagram.png"
    image.write_bytes(b"image")
    parser_input = ParserInput(id="sample", image=str(image), question="")
    prediction = ExternalParseGraphPrediction.create(
        parser_input,
        backend="fixture",
        model="fixture-model",
        model_revision="v1",
        prompt_version="fixture-prompt",
        parse_graph=ParseGraph(id="graph_sample", source_id="sample"),
    ).to_dict()
    prediction["model"] = "tampered"

    with pytest.raises(InvalidGeometryInput, match="checksum"):
        ExternalParseGraphPrediction.from_dict(prediction)


def test_openrouter_adapter_preserves_explicit_text_constraints(tmp_path, monkeypatch):
    image = tmp_path / "diagram.png"
    Image.new("RGB", (320, 240), "white").save(image)
    parser_input = ParserInput(
        id="sample",
        image=str(image),
        question="$A B C D$ is a rectangle, AB=16, BC=12, and AC perpendicular to CE.",
    )
    calls = []

    def request(**kwargs):
        calls.append(kwargs)
        return {"id": "parse", "usage": {}}, ParseGraph(id="graph_sample", source_id="sample").to_dict()

    monkeypatch.setattr(openrouter, "_request_json_completion", request)

    prediction = parse_one_openrouter(
        parser_input,
        model="fixture-model",
        api_key="fixture-key",
        base_url="https://provider.invalid/v1",
    )

    graph = prediction.parse_graph
    assert {item.predicate for item in graph.relations} == {"perpendicular", "rectangle"}
    assert {(item.kind, item.raw_text) for item in graph.quantities} == {
        ("length", "AB=16"),
        ("length", "BC=12"),
    }
    assert any(item.kind.value == "polygon" and item.visible_name == "ABCD" for item in graph.objects)
    assert len(calls) == 1
    prompt, context = calls[0]["instruction"].rsplit("\n\nINPUT\n", 1)
    assert prompt == files("geosym_world.parser").joinpath("prompts", "geometry_extraction_zh.txt").read_text(encoding="utf-8").rstrip()
    assert json.loads(context) == {"question": parser_input.question}
    assert graph.metadata["extraction_policy"]["mode"] == "constraint_only"
    assert graph.metadata["image_size"] == [320, 240]


def test_openrouter_adapter_normalizes_top_level_linear_endpoints(tmp_path, monkeypatch):
    image = tmp_path / "diagram.png"
    Image.new("RGB", (320, 240), "white").save(image)
    payload = ParseGraph(id="graph_sample", source_id="sample").to_dict()
    payload["objects"] = [
        {"id": "P", "type": "point", "label": "P", "x": 100, "y": 80, "confidence": 0.9},
        {"id": "Q", "type": "point", "label": "Q", "x": 220, "y": 180, "confidence": 0.9},
        {
            "id": "segment_PQ",
            "type": "segment",
            "start": "P",
            "end": "Q",
            "confidence": 0.9,
        },
    ]

    def request(**kwargs):
        return {"id": "parse", "usage": {}}, payload

    monkeypatch.setattr(openrouter, "_request_json_completion", request)

    prediction = parse_one_openrouter(
        ParserInput(id="sample", image=str(image), question=""),
        model="fixture-model",
        api_key="fixture-key",
        base_url="https://provider.invalid/v1",
    )

    segment = next(item for item in prediction.parse_graph.objects if item.kind.value == "segment")
    assert [(reference.role, reference.ref) for reference in segment.references] == [
        ("start", "P"),
        ("end", "Q"),
    ]
    assert segment.geometry == {"p1": [100.0, 80.0], "p2": [220.0, 180.0]}
    assert segment.evidence_refs == ("pixel_evidence_segment_PQ",)


def test_openrouter_adapter_does_not_infer_references_from_object_ids(tmp_path, monkeypatch):
    image = tmp_path / "diagram.png"
    Image.new("RGB", (200, 100), "white").save(image)
    payload = {
        "objects": [
            {"id": "poly_quad", "type": "polygon", "confidence": 0.8},
            {"id": "s0", "type": "segment", "confidence": 0.8},
        ]
    }
    monkeypatch.setattr(
        openrouter,
        "_request_json_completion",
        lambda **_: ({"id": "parse", "usage": {}}, payload),
    )

    prediction = parse_one_openrouter(
        ParserInput(id="sample", image=str(image), question=""),
        model="fixture-model",
        api_key="fixture-key",
        base_url="https://provider.invalid/v1",
    )

    assert all(not item.references for item in prediction.parse_graph.objects)


def test_openrouter_adapter_compiles_compact_polygon_references(tmp_path, monkeypatch):
    image = tmp_path / "diagram.png"
    Image.new("RGB", (200, 100), "white").save(image)
    payload = {
        "objects": [
            {"id": "A", "type": "point", "label": "A", "x": 10, "y": 10},
            {"id": "B", "type": "point", "label": "B", "x": 90, "y": 10},
            {"id": "C", "type": "point", "label": "C", "x": 50, "y": 80},
            {"id": "triangle", "type": "polygon", "references": ["A", "B", "C"]},
        ]
    }
    monkeypatch.setattr(
        openrouter,
        "_request_json_completion",
        lambda **_: ({"id": "parse", "usage": {}}, payload),
    )

    prediction = parse_one_openrouter(
        ParserInput(id="sample", image=str(image), question=""),
        model="fixture-model",
        api_key="fixture-key",
        base_url="https://provider.invalid/v1",
    )

    polygon = next(item for item in prediction.parse_graph.objects if item.kind.value == "polygon")
    assert [(reference.role, reference.ref) for reference in polygon.references] == [
        ("vertex_0", "A"),
        ("vertex_1", "B"),
        ("vertex_2", "C"),
    ]
    assert polygon.evidence_refs == ("pixel_evidence_triangle",)


def test_openrouter_adapter_compiles_compact_non_point_locations(tmp_path, monkeypatch):
    image = tmp_path / "diagram.png"
    Image.new("RGB", (200, 100), "white").save(image)
    payload = {
        "objects": [
            {"id": "A", "type": "point", "x": 10, "y": 10},
            {"id": "B", "type": "point", "x": 30, "y": 10},
            {"id": "circle", "type": "circle", "x": 50, "y": 40, "radius": 20},
            {"id": "right", "type": "right-angle-marker", "x": 80, "y": 60},
            {"id": "label", "type": "label", "label": "4", "x": 120, "y": 30},
            {"id": "side", "type": "edge", "start": "A", "end": "B"},
        ]
    }
    monkeypatch.setattr(
        openrouter,
        "_request_json_completion",
        lambda **_: ({"id": "parse", "usage": {}}, payload),
    )

    prediction = parse_one_openrouter(
        ParserInput(id="sample", image=str(image), question=""),
        model="fixture-model",
        api_key="fixture-key",
        base_url="https://provider.invalid/v1",
    )

    objects = {item.id: item for item in prediction.parse_graph.objects}
    assert objects["circle"].geometry == {"center": [50.0, 40.0], "radius_px": 20.0}
    assert objects["right"].kind.value == "marker"
    assert objects["right"].geometry == {"bbox": [74.0, 54.0, 86.0, 66.0]}
    assert objects["label"].kind.value == "text"
    assert objects["label"].visible_name == "4"
    assert objects["side"].kind.value == "segment"


def test_openrouter_compiles_short_named_support_references(tmp_path, monkeypatch):
    image = tmp_path / "diagram.png"
    Image.new("RGB", (200, 100), "white").save(image)
    payload = {
        "objects": [
            {"id": "point_p", "type": "point", "label": "P", "x": 20, "y": 20},
            {"id": "point_q", "type": "point", "label": "Q", "x": 180, "y": 20},
            {"id": "point_s", "type": "point", "label": "S", "x": 90, "y": 20},
        ],
        "relations": [
            {"id": "s_on_pq", "type": "point_on_segment", "objects": ["S", "PQ"]},
        ],
    }
    monkeypatch.setattr(
        openrouter,
        "_request_json_completion",
        lambda **_: ({"id": "parse", "usage": {}}, payload),
    )

    prediction = parse_one_openrouter(
        ParserInput(id="sample", image=str(image), question=""),
        model="fixture-model",
        api_key="fixture-key",
        base_url="https://provider.invalid/v1",
    )

    objects = {item.id: item for item in prediction.parse_graph.objects}
    assert "segment_P_Q" in objects
    relation = prediction.parse_graph.relations[0]
    assert [argument.ref for argument in relation.arguments] == ["point_s", "segment_P_Q"]
    assert objects["segment_P_Q"].references[0].ref == "point_p"


@pytest.mark.parametrize("mode", ["find", "query", "value"])
def test_openrouter_compiles_common_vlm_relation_quantity_and_target_shorthands(tmp_path, monkeypatch, mode):
    image = tmp_path / "image.png"
    Image.new("RGB", (240, 180), "white").save(image)
    payload = {
        "objects": [
            {"id": name, "type": "point", "label": name, "x": x, "y": y}
            for name, x, y in (
                ("J", 20, 150), ("K", 80, 20), ("L", 220, 80),
                ("R", 50, 85), ("S", 150, 50), ("T", 120, 115), ("P", 105, 80),
            )
        ],
        "relations": [
            {"type": "collinear", "point_0": "J", "point_1": "R", "point_2": "K"},
            {"type": "collinear", "point_0": "K", "point_1": "S", "point_2": "L"},
        ],
        "quantities": [
            {"id": "ticks", "type": "equal_length", "arguments": ["KR", "RJ"]},
            {"id": "given", "type": "length", "object": "RP", "value": 3.5},
        ],
        "targets": [{"id": "ask", "type": "length", "object": "PS", "mode": mode}],
    }
    monkeypatch.setattr(
        openrouter,
        "_request_json_completion",
        lambda **_: ({"id": "parse", "usage": {}}, payload),
    )

    prediction = parse_one_openrouter(
        ParserInput(id="sample", image=str(image), question="Find PS."),
        model="fixture-model",
        api_key="fixture-key",
        base_url="https://provider.invalid/v1",
    )

    graph = prediction.parse_graph
    assert {relation.predicate for relation in graph.relations} == {"collinear", "equal_length"}
    assert {quantity.kind for quantity in graph.quantities} == {"length"}
    assert graph.targets[0].mode == "value"
    segment_ids = {item.id for item in graph.objects if item.kind.value == "segment"}
    assert {"segment_K_R", "segment_R_J", "segment_R_P", "segment_P_S"} <= segment_ids
    assert graph.metadata["normalization_audit"]["dropped"] == {
        "relations": [], "quantities": [], "regions": [], "targets": [],
    }


def test_openrouter_flattens_grouped_object_payload(tmp_path, monkeypatch):
    image = tmp_path / "diagram.png"
    Image.new("RGB", (200, 100), "white").save(image)
    payload = {
        "objects": {
            "points": [
                {"id": "A", "label": "A", "x": 20, "y": 20},
                {"id": "B", "label": "B", "x": 180, "y": 20},
            ],
            "segments": [{"id": "AB", "label": "AB", "start": "A", "end": "B"}],
        }
    }
    monkeypatch.setattr(
        openrouter,
        "_request_json_completion",
        lambda **_: ({"id": "parse", "usage": {}}, payload),
    )

    prediction = parse_one_openrouter(
        ParserInput(id="sample", image=str(image), question=""),
        model="fixture-model",
        api_key="fixture-key",
        base_url="https://provider.invalid/v1",
    )

    objects = {item.id: item for item in prediction.parse_graph.objects}
    assert objects["A"].kind.value == "point"
    assert objects["AB"].kind.value == "segment"


def test_openrouter_expands_multi_support_incidence_and_ordered_points(tmp_path, monkeypatch):
    image = tmp_path / "diagram.png"
    Image.new("RGB", (200, 100), "white").save(image)
    payload = {
        "objects": [
            {"id": "A", "type": "point", "label": "A", "x": 10, "y": 20},
            {"id": "B", "type": "point", "label": "B", "x": 100, "y": 20},
            {"id": "C", "type": "point", "label": "C", "x": 170, "y": 20},
            {"id": "AB", "type": "segment", "start": "A", "end": "B"},
            {"id": "AC", "type": "segment", "start": "A", "end": "C"},
        ],
        "relations": [
            {"id": "inc_A", "type": "incidence", "objects": ["A", "AB", "AC"]},
            {"id": "order", "type": "ordered_points", "objects": ["A", "B", "C"]},
        ],
    }
    monkeypatch.setattr(
        openrouter,
        "_request_json_completion",
        lambda **_: ({"id": "parse", "usage": {}}, payload),
    )

    prediction = parse_one_openrouter(
        ParserInput(id="sample", image=str(image), question=""),
        model="fixture-model",
        api_key="fixture-key",
        base_url="https://provider.invalid/v1",
    )

    relations = {item.id: item for item in prediction.parse_graph.relations}
    incidence = [item for item in relations.values() if item.predicate == "incidence"]
    ordered = next(item for item in relations.values() if item.predicate == "ordered_points")
    assert len(incidence) == 2
    assert [item.role for item in ordered.arguments] == ["support", "point_0", "point_1", "point_2"]
    audit = prediction.parse_graph.metadata["normalization_audit"]
    assert audit["dropped"] == {"relations": [], "quantities": [], "regions": [], "targets": []}
