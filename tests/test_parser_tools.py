import json

from PIL import Image, ImageDraw

from geosym_world.benchmark import (
    ParserProtocol,
    ParserSourceManifest,
    build_typed_parser_reference,
    hash_asset,
)
from geosym_world.parser import ParserInput, ReconstructionMode
from geosym_world.schema import (
    EvidencePrecision,
    ParseEvidence,
    ParseEvidenceKind,
    ParseGraph,
    ParseObject,
    ParseObjectKind,
    ParseOrigin,
)
from geosym_world.tools.evaluate_parse_graph_v0 import run as run_evaluation
from geosym_world.tools.parse_reconstruct_v0 import run as run_workflow


def test_parse_reconstruct_tool_writes_auditable_results_and_manifest(tmp_path):
    image_path = tmp_path / "opaque.png"
    image = Image.new("RGB", (180, 140), "white")
    ImageDraw.Draw(image).line((20, 100, 160, 40), fill="black", width=4)
    image.save(image_path)
    input_path = tmp_path / "inputs.jsonl"
    input_path.write_text(
        json.dumps(ParserInput(id="opaque_1", image=str(image_path), question="").to_dict()) + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "results.jsonl"

    manifest = run_workflow(
        input_path=input_path,
        output_path=output_path,
        render_dir=tmp_path / "renders",
        reconstruction_mode=ReconstructionMode.PIXEL_WITNESS,
        output_format="svg",
        resynthesis_seeds=(),
        width=320,
        height=240,
        progress_every=1,
        permit_unique_fill_shading=True,
    )

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert manifest["summary"]["num_records"] == 1
    assert manifest["output_sha256"] == hash_asset(output_path)
    assert manifest["configuration"]["admission_policy"]["require_pixel_topology"] is True
    assert manifest["configuration"]["admission_policy"]["permit_unique_fill_shading"] is True
    assert manifest["configuration"]["admission_policy"]["policy_id"] == "verified_pixels_unique_fill_shading_v1"
    assert manifest["configuration"]["pixel_topology"]["minimum_local_support"] > 0.0
    assert manifest["configuration"]["resynthesis_transform"] == "rigid"
    assert manifest["configuration"]["enumerate_branches"] is False
    assert manifest["summary"]["num_branch_sets"] == 0
    assert manifest["summary"]["num_candidate_branches"] == 0
    assert manifest["summary"]["num_enumerated_branches"] == 0
    assert manifest["summary"]["num_rejected_branches"] == 0
    assert manifest["summary"]["candidate_branch_status_counts"] == {}
    assert manifest["summary"]["branch_rejection_code_counts"] == {}
    assert manifest["summary"]["num_branch_semantic_redraws"] == 0
    assert manifest["summary"]["branch_redraw_status_counts"] == {}
    assert row["checksum"]
    assert row["branch_set"] is None
    assert row["branch_redraws"] == []
    assert row["result"]["schema_version"] == "1.1"
    assert row["result"]["pixel_topology"]["source_graph_checksum"]
    assert row["result"]["parser"]["record"]["state"] is None
    assert row["artifacts"]
    assert (tmp_path / "results.jsonl.manifest.json").is_file()


def test_parse_graph_evaluation_tool_validates_private_reference_binding(tmp_path):
    image_path = tmp_path / "asset.png"
    image_path.write_bytes(b"fixture")
    parser_input = ParserInput(id="opaque_1", image=str(image_path), question="")
    graph = _point_graph()
    prediction_path = tmp_path / "predictions.jsonl"
    prediction_path.write_text(
        json.dumps({"parser_input": parser_input.to_dict(), "parse_graph": graph.to_dict()}) + "\n",
        encoding="utf-8",
    )
    manifest = ParserSourceManifest(
        source_id=graph.source_id,
        protocol=ParserProtocol.IMAGE_ONLY,
        source_version="fixture-v1",
        license_id="test-only",
        asset_sha256=hash_asset(image_path),
        split_group="fixture-family",
        public_fields=("image",),
    )
    reference = build_typed_parser_reference(parser_input, graph=graph, manifest=manifest)
    reference_path = tmp_path / "references.jsonl"
    reference_path.write_text(json.dumps(reference) + "\n", encoding="utf-8")
    output_path = tmp_path / "metrics.jsonl"

    result = run_evaluation(
        predictions_path=prediction_path,
        references_path=reference_path,
        output_path=output_path,
    )

    metric = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["num_records"] == 1
    assert result["component_macro_f1"]["objects"] == 1.0
    assert result["component_sample_accuracy"]["objects"] == 1.0
    assert result["perfect_parsing_rate"] == 1.0
    assert result["core_perfect_parsing_rate"] == 1.0
    assert metric["metrics"]["macro_score"] == 1.0
    assert metric["metrics"]["perfect_parsing"] is True
    assert (tmp_path / "metrics.jsonl.manifest.json").is_file()


def _point_graph() -> ParseGraph:
    evidence = ParseEvidence(
        id="evidence_a",
        source_id="source_a",
        kind=ParseEvidenceKind.PIXEL_POINT,
        precision=EvidencePrecision.BOUND,
        point=(20.0, 30.0),
    )
    point = ParseObject(
        id="point_a",
        kind=ParseObjectKind.POINT,
        origin=ParseOrigin.IMAGE,
        confidence=1.0,
        score_kind="fixture",
        evidence_refs=(evidence.id,),
        visible_name="A",
        visible=True,
        geometry={"pixel": [20.0, 30.0]},
    )
    return ParseGraph(
        id="graph_a",
        source_id="source_a",
        evidence=(evidence,),
        objects=(point,),
        metadata={"image_size": [100, 100]},
    )
