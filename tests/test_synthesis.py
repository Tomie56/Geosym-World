"""Unit tests for synthetic sampler and exporter."""

import json
from argparse import Namespace

from geosym_world_geometry.synthesis import GeometryGenerationPipeline, PipelineRunConfig, SyntheticRecordExporter, SyntheticSampler, load_grid_overrides, select_template_families
from geosym_world_geometry.synthesis.enhancer import FormalTraceEnhancer
from geosym_world_geometry.synthesis.filters import DegeneracyRejector
from geosym_world_geometry.tools.synthesize_records import _apply_preset


def test_sampler_produces_valid_specs():
    specs = SyntheticSampler(seed=3).sample(max_records=5)
    assert len(specs) == 5
    assert {spec.family for spec in specs}
    assert all(spec.trace for spec in specs)


def test_exporter_writes_image_qa_records(tmp_path):
    specs = SyntheticSampler(seed=5, enable_mutation=False).sample(max_records=3)
    path = tmp_path / "synthetic.jsonl"
    manifest = SyntheticRecordExporter().write_jsonl(specs, path, image_format="svg", style_seed=7)
    assert manifest["num_records"] == 3
    assert manifest["num_valid"] == 3
    assert manifest["image_formats"] == {"svg": 3}
    assert manifest["num_images"] == 3
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    record = json.loads(lines[0])
    assert record["image"]["formats"]["svg"].endswith(".svg")
    assert record["synthetic"]["target_value"] is not None
    assert record["answer"] == record["synthetic"]["target_value"]
    qa = json.loads(path.with_suffix(".jsonl.qa.jsonl").read_text(encoding="utf-8").splitlines()[0])
    gt = json.loads(path.with_suffix(".jsonl.gt.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert qa["question"] == record["question"]
    assert qa["answer"] == record["answer"]
    assert gt["target"]["value"] == record["synthetic"]["target_value"]


def test_exporter_require_exact_count_fails_before_writing(tmp_path):
    specs = SyntheticSampler(seed=5, enable_mutation=False).sample(max_records=1)
    path = tmp_path / "synthetic.jsonl"
    try:
        SyntheticRecordExporter().write_jsonl(specs, path, image_format="none", require_count=2)
    except ValueError as exc:
        assert "Expected 2 accepted specs" in str(exc)
    else:  # pragma: no cover - assertion guard.
        raise AssertionError("expected exact-count failure")
    assert not path.exists()


def test_parameterized_family_selection_and_limits():
    overrides = load_grid_overrides(inline_json='{"triangle_area_from_height": {"base": [3], "height": [2], "skew": [1]}}')
    families = select_template_families(["triangle_area_from_height"], grid_overrides=overrides)
    specs = SyntheticSampler(families=families, seed=1, mutations_per_variant=2, per_family_limit=2).sample(max_records=10)
    assert len(specs) == 2
    assert {spec.family for spec in specs} == {"triangle_area_from_height"}


def test_new_2d_template_families_are_valid():
    families = select_template_families(["rectangle_area_diagonal", "parallelogram_area", "trapezoid_area"])
    specs = SyntheticSampler(families=families, seed=4, enable_mutation=False).sample(max_records=3)
    assert {spec.family for spec in specs} == {"rectangle_area_diagonal", "parallelogram_area", "trapezoid_area"}
    assert all(DegeneracyRejector().check(spec).valid for spec in specs)


def test_formal_trace_enhancer_expands_valid_specs():
    base = SyntheticSampler(seed=2, enable_mutation=False).sample(max_records=1)[0]
    enhanced = FormalTraceEnhancer(seed=4, rounds_distribution={0: 1, 1: 2}).enhance(base)
    assert len(enhanced) == 3
    assert all(spec.id.startswith(base.id) for spec in enhanced)
    assert all(DegeneracyRejector().check(spec).valid for spec in enhanced)


def test_staged_generation_pipeline_writes_intermediate_files(tmp_path):
    output = tmp_path / "records.jsonl"
    manifest = GeometryGenerationPipeline(
        PipelineRunConfig(
            target_records=3,
            base_records=2,
            seed=9,
            image_format="none",
            require_exact_count=True,
            base_mutations_per_variant=0,
            enhancement_rounds={0: 1, 1: 2},
        )
    ).run(output)
    assert manifest["num_records"] == 3
    assert manifest["num_images"] == 0
    assert manifest["run_config"]["pipeline"] == "staged_formal_geosym"
    assert manifest["run_config"]["num_base_specs"] == 2
    assert (tmp_path / "json" / "base" / "records.base.jsonl").exists()
    enhanced_path = tmp_path / "json" / "enhanced" / "records.enhanced.jsonl"
    assert enhanced_path.exists()
    enhanced_lines = [json.loads(line) for line in enhanced_path.read_text(encoding="utf-8").splitlines()]
    assert len({line["metadata"]["base_spec_id"] for line in enhanced_lines[:2]}) == 2


def test_large_preset_preserves_explicit_cli_overrides():
    args = Namespace(
        preset="large-500",
        output="custom.jsonl",
        max_records=20,
        format="both",
        families="",
        per_family=None,
        mutations_per_variant=1,
        require_exact_count=False,
        pipeline=False,
        base_records=None,
        base_mutations_per_variant=1,
        enhancement_rounds_json=None,
        enhancement_operations=None,
        max_enhancements_per_base=None,
        grid_json=None,
    )
    result = _apply_preset(args, {"output", "format"})
    assert result.output == "custom.jsonl"
    assert result.format == "both"
    assert result.max_records == 500
    assert result.per_family == 72
    assert "tetrahedron_section" not in result.families
    assert "trapezoid_area" in result.families
    assert result.pipeline is True
    assert result.base_records == 500
    assert result.base_mutations_per_variant == 0
    assert result.max_enhancements_per_base == 1
    assert result.enhancement_rounds_json == '{"1": 1}'
    assert "triangle_area_from_height" in result.grid_json
