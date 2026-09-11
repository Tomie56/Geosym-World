import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from geosym_world.parser import (
    ExternalParseGraphPrediction,
    ParserInput,
    RecordedParseGraphParser,
    ReconstructionMode,
)
from geosym_world.schema import (
    EvidencePrecision,
    MathExpression,
    ParseEvidence,
    ParseEvidenceKind,
    ParseGraph,
    ParseObject,
    ParseObjectKind,
    ParseOrigin,
    ParseQuantity,
    ParseReference,
)
from geosym_world.tools.parse_reconstruct_v0 import run as run_workflow


def test_recorded_external_parse_graph_round_trips_as_proposed_hypothesis(tmp_path):
    parser_input, prediction = _prediction(tmp_path)
    restored = ExternalParseGraphPrediction.from_dict(prediction.to_dict())
    result = RecordedParseGraphParser((restored,)).parse(parser_input)

    assert restored == prediction
    assert result.graph == prediction.parse_graph
    assert result.record.state is None
    assert result.record.metadata["admission_required"] is True
    assert result.record.metadata["model"] == "fixture-model"
    assert result.stages[0]["metadata"]["prediction_checksum"] == prediction.checksum()


def test_external_prediction_rejects_tampering_and_trusted_provenance(tmp_path):
    parser_input, prediction = _prediction(tmp_path)
    payload = prediction.to_dict()
    payload["model_revision"] = "changed"
    with pytest.raises(ValueError, match="checksum"):
        ExternalParseGraphPrediction.from_dict(payload)

    native_point = replace(prediction.parse_graph.objects[0], origin=ParseOrigin.NATIVE)
    with pytest.raises(ValueError, match="origin=native"):
        replace(prediction, parse_graph=replace(prediction.parse_graph, objects=(native_point,)))

    compiled_point = replace(prediction.parse_graph.objects[0], origin=ParseOrigin.COMPILED)
    with pytest.raises(ValueError, match="origin=compiled"):
        replace(prediction, parse_graph=replace(prediction.parse_graph, objects=(compiled_point,)))

    human_point = replace(prediction.parse_graph.objects[0], score_kind="human_annotation")
    with pytest.raises(ValueError, match="trusted score_kind"):
        replace(prediction, parse_graph=replace(prediction.parse_graph, objects=(human_point,)))

    with pytest.raises(ValueError, match="forbidden private field"):
        replace(prediction, parse_graph=replace(prediction.parse_graph, metadata={"formal_state": {}}))
    with pytest.raises(ValueError, match="sensitive field"):
        replace(prediction, usage={"api_key": "must-not-be-recorded"})

    changed_input = replace(parser_input, question="Changed public question")
    with pytest.raises(ValueError, match="input checksum"):
        RecordedParseGraphParser((prediction,)).parse(changed_input)


def test_verified_workflow_replays_external_prediction_and_records_configuration(tmp_path):
    parser_input, prediction = _prediction(tmp_path)
    input_path = tmp_path / "inputs.jsonl"
    prediction_path = tmp_path / "external.jsonl"
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
        external_predictions_path=prediction_path,
    )

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert manifest["configuration"]["parser"]["kind"] == "recorded_external_parse_graph"
    assert manifest["configuration"]["parser"]["models"] == ["fixture-model"]
    assert manifest["external_predictions_sha256"]
    assert row["result"]["parser"]["parse_graph"] == prediction.parse_graph.to_dict()
    assert row["result"]["parser"]["record"]["state"] is None


def test_external_prediction_ids_are_checked_before_output(tmp_path):
    parser_input, prediction = _prediction(tmp_path)
    input_path = tmp_path / "inputs.jsonl"
    prediction_path = tmp_path / "external.jsonl"
    output_path = tmp_path / "results.jsonl"
    input_path.write_text(json.dumps(parser_input.to_dict()) + "\n", encoding="utf-8")
    extra_graph = replace(
        prediction.parse_graph,
        source_id="sample_extra",
        evidence=tuple(replace(item, source_id="sample_extra") for item in prediction.parse_graph.evidence),
    )
    extra = replace(prediction, id="sample_extra", parse_graph=extra_graph)
    prediction_path.write_text(
        json.dumps(prediction.to_dict()) + "\n" + json.dumps(extra.to_dict()) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"unexpected=\['sample_extra'\]"):
        run_workflow(
            input_path=input_path,
            output_path=output_path,
            render_dir=tmp_path / "renders",
            reconstruction_mode=ReconstructionMode.PIXEL_WITNESS,
            output_format="none",
            resynthesis_seeds=(),
            width=320,
            height=240,
            progress_every=0,
            external_predictions_path=prediction_path,
        )
    assert not output_path.exists()


def test_verified_workflow_exports_exact_branch_set_without_selecting_a_branch(tmp_path):
    parser_input, prediction = _sss_prediction(tmp_path)
    input_path = tmp_path / "inputs.jsonl"
    prediction_path = tmp_path / "external.jsonl"
    output_path = tmp_path / "results.jsonl"
    input_path.write_text(json.dumps(parser_input.to_dict()) + "\n", encoding="utf-8")
    prediction_path.write_text(json.dumps(prediction.to_dict()) + "\n", encoding="utf-8")

    manifest = run_workflow(
        input_path=input_path,
        output_path=output_path,
        render_dir=tmp_path / "renders",
        reconstruction_mode=ReconstructionMode.EXACT,
        enumerate_branches=True,
        max_branches=4,
        output_format="svg",
        resynthesis_seeds=(),
        width=320,
        height=240,
        progress_every=0,
        external_predictions_path=prediction_path,
    )

    row = json.loads(output_path.read_text(encoding="utf-8"))
    branch_set = row["branch_set"]
    branches = branch_set["branches"]
    assert manifest["schema_version"] == "geosym.parser_run_manifest.v8"
    assert manifest["configuration"]["enumerate_branches"] is True
    assert manifest["configuration"]["max_branches"] == 4
    assert manifest["summary"]["num_branch_sets"] == 1
    assert manifest["summary"]["num_candidate_branches"] == 2
    assert manifest["summary"]["num_enumerated_branches"] == 2
    assert manifest["summary"]["num_rejected_branches"] == 0
    assert manifest["summary"]["num_records_with_multiple_branches"] == 1
    assert manifest["summary"]["branch_reconstruction_status_counts"] == {"exact": 2}
    assert manifest["summary"]["candidate_branch_status_counts"] == {"exact": 2}
    assert manifest["summary"]["branch_rejection_code_counts"] == {}
    assert manifest["summary"]["num_branch_semantic_redraws"] == 2
    assert manifest["summary"]["branch_redraw_status_counts"] == {"valid": 2}
    assert row["result"]["reconstruction"]["status"] == "partial"
    assert row["result"]["reconstruction"]["branch_choices"] == {}
    assert branch_set["ambiguous_candidates"] == ["point_c"]
    assert branch_set["num_branches"] == 2
    assert branch_set["num_candidate_branches"] == 2
    assert branch_set["num_rejected_branches"] == 0
    assert branch_set["rejected_branches"] == []
    assert {branch["branch_choices"]["point_c"] for branch in branches} == {"first", "second"}
    assert all(branch["status"] == "exact" for branch in branches)
    assert all(branch["state"] is not None and branch["trace"] for branch in branches)
    assert len({branch["checksum"] for branch in branches}) == 2
    assert len(row["branch_redraws"]) == 2
    assert all(
        branch_redraw["result"]["branch_checksum"]
        == branch_redraw["result"]["redraw"]["lineage"]["reconstruction_checksum"]
        for branch_redraw in row["branch_redraws"]
    )
    assert all(
        branch_redraw["artifacts"]["semantic_redraw_svg"]
        for branch_redraw in row["branch_redraws"]
    )
    assert all(
        Path(branch_redraw["artifacts"]["semantic_redraw_svg"]).is_file()
        for branch_redraw in row["branch_redraws"]
    )


def test_verified_workflow_rejects_invalid_branch_enumeration_configuration(tmp_path):
    parser_input, prediction = _sss_prediction(tmp_path)
    input_path = tmp_path / "inputs.jsonl"
    prediction_path = tmp_path / "external.jsonl"
    input_path.write_text(json.dumps(parser_input.to_dict()) + "\n", encoding="utf-8")
    prediction_path.write_text(json.dumps(prediction.to_dict()) + "\n", encoding="utf-8")
    arguments = {
        "input_path": input_path,
        "output_path": tmp_path / "results.jsonl",
        "render_dir": tmp_path / "renders",
        "output_format": "none",
        "width": 320,
        "height": 240,
        "progress_every": 0,
        "external_predictions_path": prediction_path,
    }

    with pytest.raises(ValueError, match="requires exact reconstruction mode"):
        run_workflow(
            **arguments,
            reconstruction_mode=ReconstructionMode.PIXEL_WITNESS,
            enumerate_branches=True,
            resynthesis_seeds=(),
        )
    with pytest.raises(ValueError, match="max_branches must be a positive integer"):
        run_workflow(
            **arguments,
            reconstruction_mode=ReconstructionMode.EXACT,
            enumerate_branches=True,
            max_branches=0,
            resynthesis_seeds=(),
        )
    with pytest.raises(ValueError, match="cannot use resynthesis seeds"):
        run_workflow(
            **arguments,
            reconstruction_mode=ReconstructionMode.EXACT,
            enumerate_branches=True,
            resynthesis_seeds=(3,),
        )


def test_verified_workflow_manifest_counts_redundant_constraint_branch_rejections(tmp_path):
    parser_input, prediction = _redundant_distance_prediction(tmp_path)
    input_path = tmp_path / "inputs.jsonl"
    prediction_path = tmp_path / "external.jsonl"
    output_path = tmp_path / "results.jsonl"
    input_path.write_text(json.dumps(parser_input.to_dict()) + "\n", encoding="utf-8")
    prediction_path.write_text(json.dumps(prediction.to_dict()) + "\n", encoding="utf-8")

    manifest = run_workflow(
        input_path=input_path,
        output_path=output_path,
        render_dir=tmp_path / "renders",
        reconstruction_mode=ReconstructionMode.EXACT,
        enumerate_branches=True,
        max_branches=4,
        output_format="none",
        resynthesis_seeds=(),
        width=320,
        height=240,
        progress_every=0,
        external_predictions_path=prediction_path,
    )

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert manifest["summary"]["num_candidate_branches"] == 4
    assert manifest["summary"]["num_enumerated_branches"] == 2
    assert manifest["summary"]["num_rejected_branches"] == 2
    assert manifest["summary"]["candidate_branch_status_counts"] == {
        "exact": 2,
        "partial": 2,
    }
    assert manifest["summary"]["branch_rejection_code_counts"] == {
        "reconstruction.condition_conflict": 2,
    }
    assert manifest["summary"]["num_branch_semantic_redraws"] == 2
    assert row["branch_set"]["num_branches"] == 2
    assert row["branch_set"]["num_rejected_branches"] == 2
    assert len(row["branch_redraws"]) == 2


def _prediction(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    parser_input = ParserInput(id="sample_public_1", image=str(image_path), question="Name point A.")
    evidence = ParseEvidence(
        id="evidence_point_a",
        source_id=parser_input.id,
        kind=ParseEvidenceKind.PIXEL_POINT,
        precision=EvidencePrecision.LOCALIZED,
        point=(20.0, 30.0),
    )
    point = ParseObject(
        id="point_a",
        kind=ParseObjectKind.POINT,
        origin=ParseOrigin.IMAGE,
        confidence=0.9,
        score_kind="model_probability",
        evidence_refs=(evidence.id,),
        visible_name="A",
        visible=True,
        geometry={"pixel": [20.0, 30.0]},
    )
    graph = ParseGraph(
        id="external_graph_1",
        source_id=parser_input.id,
        evidence=(evidence,),
        objects=(point,),
        metadata={"image_size": [100, 100], "producer": "external_model"},
    )
    prediction = ExternalParseGraphPrediction.create(
        parser_input,
        backend="offline_fixture",
        model="fixture-model",
        model_revision="revision-1",
        prompt_version="parse-graph-v1",
        parse_graph=graph,
        request_id="request-1",
        decoding={"temperature": 0},
        usage={"input_tokens": 10, "output_tokens": 20},
    )
    return parser_input, prediction


def _sss_prediction(tmp_path):
    return _distance_prediction(
        tmp_path,
        sample_id="sample_sss",
        coordinates={
            "point_a": (50.0, 20.0),
            "point_b": (20.0, 60.0),
            "point_c": (50.0, 60.0),
        },
        segment_specs=(
            ("segment_ab", "point_a", "point_b", 5),
            ("segment_ac", "point_a", "point_c", 4),
            ("segment_bc", "point_b", "point_c", 3),
        ),
    )


def _redundant_distance_prediction(tmp_path):
    return _distance_prediction(
        tmp_path,
        sample_id="sample_redundant_distance",
        coordinates={
            "point_a": (50.0, 20.0),
            "point_b": (20.0, 60.0),
            "point_c": (50.0, 60.0),
            "point_d": (20.0, 20.0),
        },
        segment_specs=(
            ("segment_ab", "point_a", "point_b", 5),
            ("segment_ac", "point_a", "point_c", 4),
            ("segment_ad", "point_a", "point_d", 3),
            ("segment_bc", "point_b", "point_c", 3),
            ("segment_bd", "point_b", "point_d", 4),
            ("segment_cd", "point_c", "point_d", 5),
        ),
    )


def _distance_prediction(tmp_path, *, sample_id, coordinates, segment_specs):
    image_path = tmp_path / f"{sample_id}.png"
    image = Image.new("RGB", (100, 100), "white")
    draw = ImageDraw.Draw(image)
    for _, start_id, end_id, _ in segment_specs:
        draw.line((*coordinates[start_id], *coordinates[end_id]), fill="black", width=4)
    image.save(image_path)
    parser_input = ParserInput(
        id=sample_id,
        image=str(image_path),
        question="Reconstruct the diagram from its displayed side lengths.",
    )
    point_evidence = tuple(
        ParseEvidence(
            id=f"evidence_{point_id}",
            source_id=parser_input.id,
            kind=ParseEvidenceKind.PIXEL_POINT,
            precision=EvidencePrecision.BOUND,
            point=coordinates[point_id],
        )
        for point_id in coordinates
    )
    segment_evidence = tuple(
        ParseEvidence(
            id=f"evidence_{segment_id}",
            source_id=parser_input.id,
            kind=ParseEvidenceKind.IMAGE_REGION,
            precision=EvidencePrecision.BOUND,
            bbox=(
                min(coordinates[start_id][0], coordinates[end_id][0]),
                min(coordinates[start_id][1], coordinates[end_id][1]),
                max(coordinates[start_id][0], coordinates[end_id][0]),
                max(coordinates[start_id][1], coordinates[end_id][1]),
            ),
        )
        for segment_id, start_id, end_id, _ in segment_specs
    )
    points = tuple(
        ParseObject(
            id=point_id,
            kind=ParseObjectKind.POINT,
            origin=ParseOrigin.IMAGE,
            confidence=0.95,
            score_kind="model_probability",
            evidence_refs=(f"evidence_{point_id}",),
            geometry={"pixel": list(coordinates[point_id])},
        )
        for point_id in coordinates
    )
    segments = tuple(
        ParseObject(
            id=segment_id,
            kind=ParseObjectKind.SEGMENT,
            origin=ParseOrigin.IMAGE,
            confidence=0.95,
            score_kind="model_probability",
            references=(ParseReference("start", start_id), ParseReference("end", end_id)),
            evidence_refs=(f"evidence_{segment_id}",),
            geometry={"p1": list(coordinates[start_id]), "p2": list(coordinates[end_id])},
        )
        for segment_id, start_id, end_id, _ in segment_specs
    )
    quantities = tuple(
        ParseQuantity(
            id=f"quantity_{segment_id}",
            kind="length",
            arguments=(ParseReference("object", segment_id),),
            expression=MathExpression("literal", value=str(length)),
            origin=ParseOrigin.IMAGE_TEXT,
            confidence=0.95,
            score_kind="model_probability",
            raw_text=str(length),
            evidence_refs=(f"evidence_{segment_id}",),
        )
        for segment_id, _, _, length in segment_specs
    )
    graph = ParseGraph(
        id=f"external_graph_{sample_id}",
        source_id=parser_input.id,
        evidence=(*point_evidence, *segment_evidence),
        objects=(*points, *segments),
        quantities=quantities,
        metadata={"image_size": [100, 100], "producer": "external_model"},
    )
    prediction = ExternalParseGraphPrediction.create(
        parser_input,
        backend="offline_fixture",
        model="fixture-model",
        model_revision="revision-1",
        prompt_version="parse-graph-v1",
        parse_graph=graph,
        request_id=f"request-{sample_id}",
        decoding={"temperature": 0},
        usage={"input_tokens": 20, "output_tokens": 80},
    )
    return parser_input, prediction
