import json
from dataclasses import replace

import pytest
from PIL import Image

from geosym_world.parser import (
    GeoparsingAdapter,
    GeoparsingFormalDescription,
    GeoparsingPrediction,
    OCRImagePrediction,
    ParserInput,
    ReconstructionMode,
    RecordedParseGraphParser,
    load_geoparsing_predictions,
)
from geosym_world.schema import AdmissionStatus, EvidencePrecision, ParseObjectKind
from geosym_world.verifier import ParseGraphAdmission
from geosym_world.tools.parse_reconstruct_v0 import run as run_workflow


def test_geoparsing_adapter_maps_official_2d_language_as_global_hypotheses(tmp_path):
    parser_input = _parser_input(tmp_path)
    prediction = GeoparsingPrediction.create(
        parser_input,
        backend="offline",
        model="PeijieWang/GDP-4B",
        model_revision="revision-sha",
        prompt_version="official-hierarchical-v1",
        formal_description={
            "points": ["A", "B", "C", "D", "E", "O"],
            "lines": ["line A E B", "line C E D", "line q lieson A E B"],
            "circles": [r"\odot O lieson A B C"],
            "semantics": [
                r"AB \perp CD on E",
                r"line q \parallel CD",
                r"AB = CD = 4\sqrt{3}",
                r"m \angle AEC = m \angle BED = 60",
                r"m \widehat{AB} = 90",
            ],
        },
    )

    result = GeoparsingAdapter().convert(parser_input, prediction)
    graph = result.prediction.parse_graph

    assert result.complete
    assert {item.visible_name for item in graph.objects if item.kind is ParseObjectKind.POINT} == {
        "A",
        "B",
        "C",
        "D",
        "E",
        "O",
    }
    assert {item.predicate for item in graph.relations} >= {
        "ordered_points",
        "point_on_circle",
        "perpendicular",
        "intersection",
        "parallel",
        "equal_length",
        "equal_angle",
    }
    assert {(item.kind, str(item.expression.to_sympy())) for item in graph.quantities} >= {
        ("length", "4*sqrt(3)"),
        ("angle", "60"),
        ("angle", "90"),
    }
    assert all(item.precision is EvidencePrecision.IMAGE_ONLY for item in graph.evidence)
    assert all(
        item.confidence == 0.0 and item.score_kind == "unscored_external_sequence"
        for item in (*graph.objects, *graph.relations, *graph.quantities)
    )
    assert graph.metadata["grounding_scope"] == "global_image_only"


def test_geoparsing_global_output_remains_proposed_and_is_not_admitted(tmp_path):
    parser_input = _parser_input(tmp_path)
    prediction = _prediction(parser_input)
    imported = GeoparsingAdapter().convert(parser_input, prediction).prediction

    parsed = RecordedParseGraphParser((imported,)).parse(parser_input)
    admission = ParseGraphAdmission().admit(parsed.graph)

    assert parsed.record.state is None
    assert parsed.record.hypotheses[0].status is AdmissionStatus.PROPOSED
    assert not admission.admitted_ids
    assert set(admission.abstained_ids) == {
        item.id
        for item in (*parsed.graph.objects, *parsed.graph.relations, *parsed.graph.quantities)
    }


def test_geoparsing_adapter_maps_multichar_3d_vertices_without_guessing(tmp_path):
    parser_input = _parser_input(tmp_path)
    prediction = GeoparsingPrediction.create(
        parser_input,
        backend="offline",
        model="GDP-4B",
        model_revision="revision-sha",
        prompt_version="solid-v1",
        confidence=0.7,
        formal_description={
            "points": ["A", "B", "C", "D", "A1", "B1", "C1", "D1"],
            "lines": ["line A B", "line A1 B1"],
            "circles": [],
            "planes": ["plane A B C D", "plane A1 B1 C1 D1"],
            "structure": ["Cube ABCD-A1B1C1D1"],
            "semantics": [],
        },
    )

    graph = GeoparsingAdapter().convert(parser_input, prediction).prediction.parse_graph
    solid = next(item for item in graph.objects if item.kind is ParseObjectKind.POLYHEDRON)

    assert len(solid.references) == 8
    assert solid.metadata["solid_kind"] == "cube"
    assert sum(item.kind is ParseObjectKind.PLANE for item in graph.objects) == 2
    assert all(item.confidence == 0.7 for item in graph.objects)


def test_geoparsing_adapter_reports_unsupported_syntax_or_fails_strictly(tmp_path):
    parser_input = _parser_input(tmp_path)
    prediction = replace(
        _prediction(parser_input),
        formal_description=GeoparsingFormalDescription.from_dict(
            {
                "points": ["A", "B"],
                "lines": ["line A B"],
                "circles": [],
                "semantics": ["A is somewhere near B"],
            }
        ),
    )

    with pytest.raises(ValueError, match="unsupported_semantic"):
        GeoparsingAdapter(strict=True).convert(parser_input, prediction)

    result = GeoparsingAdapter(strict=False).convert(parser_input, prediction)
    assert not result.complete
    assert result.diagnostics[0].code == "unsupported_semantic"
    assert result.prediction.parse_graph.metadata["import_complete"] is False


def test_geoparsing_adapter_refuses_to_guess_an_arc_circle(tmp_path):
    parser_input = _parser_input(tmp_path)
    prediction = GeoparsingPrediction.create(
        parser_input,
        backend="offline",
        model="GDP-4B",
        model_revision="revision-sha",
        prompt_version="official-v1",
        formal_description={
            "points": ["A", "B", "O", "P"],
            "lines": ["line A B"],
            "circles": [r"\odot O lieson A B", r"\odot P lieson A B"],
            "semantics": [r"m \widehat{AB} = 90"],
        },
    )

    with pytest.raises(ValueError, match="arc_circle_ambiguous"):
        GeoparsingAdapter().convert(parser_input, prediction)


def test_geoparsing_prediction_rejects_input_and_image_tampering(tmp_path):
    parser_input = _parser_input(tmp_path)
    prediction = _prediction(parser_input)

    changed_question = replace(parser_input, question="A different public question")
    with pytest.raises(ValueError, match="ParserInput checksum"):
        GeoparsingAdapter().convert(changed_question, prediction)

    Image.new("RGB", (64, 64), "black").save(parser_input.image)
    with pytest.raises(ValueError, match="image SHA-256"):
        GeoparsingAdapter().convert(parser_input, prediction)


def test_geoparsing_prediction_jsonl_round_trip_is_checksum_bound(tmp_path):
    parser_input = _parser_input(tmp_path)
    prediction = _prediction(parser_input)
    path = tmp_path / "geoparsing.jsonl"
    path.write_text(json.dumps(prediction.to_dict()) + "\n", encoding="utf-8")

    assert load_geoparsing_predictions(path) == (prediction,)

    payload = prediction.to_dict()
    payload["model_revision"] = "changed"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_geoparsing_predictions(path)


def test_verified_workflow_imports_geoparsing_predictions_and_audits_manifest(tmp_path):
    parser_input = _parser_input(tmp_path)
    prediction = _prediction(parser_input)
    input_path = tmp_path / "inputs.jsonl"
    prediction_path = tmp_path / "geoparsing.jsonl"
    output_path = tmp_path / "results.jsonl"
    input_path.write_text(json.dumps(parser_input.to_dict()) + "\n", encoding="utf-8")
    prediction_path.write_text(json.dumps(prediction.to_dict()) + "\n", encoding="utf-8")

    manifest = run_workflow(
        input_path=input_path,
        output_path=output_path,
        render_dir=tmp_path / "renders",
        reconstruction_mode=ReconstructionMode.PIXEL_WITNESS,
        output_format="none",
        resynthesis_seeds=(),
        width=320,
        height=240,
        progress_every=0,
        geoparsing_predictions_path=prediction_path,
    )

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert manifest["geoparsing_predictions_sha256"]
    assert manifest["geoparsing_import"] == {
        "strict": True,
        "num_predictions": 1,
        "num_complete": 1,
        "num_with_diagnostics": 0,
        "diagnostic_counts": {},
    }
    assert manifest["configuration"]["geoparsing_strict"] is True
    assert row["result"]["parser"]["record"]["state"] is None
    assert row["result"]["admission"]["admitted_ids"] == []


def test_verified_workflow_can_fuse_geoparsing_with_the_visual_baseline(tmp_path):
    parser_input = _parser_input(tmp_path)
    prediction = _prediction(parser_input)
    input_path = tmp_path / "inputs.jsonl"
    prediction_path = tmp_path / "geoparsing.jsonl"
    ocr_path = tmp_path / "ocr.jsonl"
    output_path = tmp_path / "results.jsonl"
    input_path.write_text(json.dumps(parser_input.to_dict()) + "\n", encoding="utf-8")
    prediction_path.write_text(json.dumps(prediction.to_dict()) + "\n", encoding="utf-8")
    ocr_prediction = OCRImagePrediction.create(
        parser_input.image,
        backend="fixture",
        model="fixture-ocr",
        model_revision="r1",
        regions=(),
    )
    ocr_path.write_text(json.dumps(ocr_prediction.to_dict()) + "\n", encoding="utf-8")

    manifest = run_workflow(
        input_path=input_path,
        output_path=output_path,
        render_dir=tmp_path / "renders",
        reconstruction_mode=ReconstructionMode.PIXEL_WITNESS,
        output_format="none",
        resynthesis_seeds=(),
        width=320,
        height=240,
        progress_every=0,
        ocr_predictions_path=ocr_path,
        geoparsing_predictions_path=prediction_path,
        fuse_geoparsing_visual=True,
    )

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "geosym.parser_run_manifest.v8"
    assert manifest["configuration"]["parser"]["kind"] == "fused_parse_graph_v0"
    assert manifest["configuration"]["parser"]["visual_parser"]["ocr"]["models"] == [
        "fixture-ocr"
    ]
    assert manifest["configuration"]["fuse_geoparsing_visual"] is True
    assert row["result"]["parser"]["parse_graph"]["metadata"]["producer"] == (
        "geosym.parse_graph_fusion.v0"
    )
    assert "compatibility_certificate" in row["result"]["parser"]["parse_graph"]["metadata"]
    assert "compatibility_status_counts" in row["result"]["parser"]["stages"][2]["metadata"]
    assert row["result"]["parser"]["record"]["state"] is None


def test_geoparsing_formal_description_rejects_dataset_metadata_leakage():
    with pytest.raises(ValueError, match="unknown fields"):
        GeoparsingFormalDescription.from_dict(
            {
                "diagram": "private-or-gt-relative-path.png",
                "points": ["A", "B"],
                "lines": ["line A B"],
                "circles": [],
                "semantics": [],
            }
        )


def _parser_input(tmp_path):
    image_path = tmp_path / "diagram.png"
    Image.new("RGB", (64, 64), "white").save(image_path)
    return ParserInput(id="sample_gdp", image=str(image_path), question="Find the requested value.")


def _prediction(parser_input):
    return GeoparsingPrediction.create(
        parser_input,
        backend="offline",
        model="GDP-4B",
        model_revision="revision-sha",
        prompt_version="official-v1",
        formal_description={
            "points": ["A", "B"],
            "lines": ["line A B"],
            "circles": [],
            "semantics": ["AB = 5"],
        },
    )
