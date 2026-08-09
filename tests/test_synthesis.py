"""Unit tests for synthetic sampler and exporter."""

import json
from argparse import Namespace
from pathlib import Path

from geosym_world.synthesis import GeometryGenerationPipeline, PipelineRunConfig, SyntheticRecordExporter, SyntheticSampler, SyntheticSpec, load_grid_overrides, select_template_families
from geosym_world.synthesis.enhancer import FormalTraceEnhancer
from geosym_world.synthesis.filters import DegeneracyRejector
from geosym_world.tools.synthesize_records import _apply_preset


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


def test_exporter_rejects_duplicate_ids_before_images_can_be_overwritten(tmp_path):
    spec = SyntheticSampler(seed=5, enable_mutation=False).sample(max_records=1)[0]
    path = tmp_path / "synthetic.jsonl"
    try:
        SyntheticRecordExporter().write_jsonl([spec, spec], path, image_format="none")
    except ValueError as exc:
        assert "Duplicate synthetic record IDs" in str(exc)
    else:  # pragma: no cover - assertion guard.
        raise AssertionError("expected duplicate-ID failure")
    assert not path.exists()


def test_parameterized_family_selection_and_limits():
    overrides = load_grid_overrides(inline_json='{"triangle_area_from_height": {"base": [3], "height": [2], "skew": [1]}}')
    families = select_template_families(["triangle_area_from_height"], grid_overrides=overrides)
    specs = SyntheticSampler(families=families, seed=1, mutations_per_variant=2, per_family_limit=2).sample(max_records=10)
    assert len(specs) == 2
    assert {spec.family for spec in specs} == {"triangle_area_from_height"}


def test_new_2d_template_families_are_valid(tmp_path):
    expected = {
        "rectangle_area_diagonal",
        "parallelogram_area",
        "trapezoid_area",
        "kite_area_diagonals",
        "rhombus_diagonal_theorem",
        "cyclic_quadrilateral",
        "tangential_quadrilateral",
        "concave_composite_area",
        "annulus_area",
    }
    families = select_template_families(expected)
    specs = SyntheticSampler(families=families, seed=4, enable_mutation=False).sample(max_records=len(expected))
    assert {spec.family for spec in specs} == expected
    assert all(DegeneracyRejector().check(spec).valid for spec in specs)

    annulus = next(spec for spec in specs if spec.family == "annulus_area")
    record = SyntheticRecordExporter(solver_profile="auto").build_record(annulus, output_dir=tmp_path, image_format="none")
    assert record["synthetic"]["target_value"] == "21*pi"

    theorem_types = {
        "rhombus_diagonal_theorem": {"rhombus", "rhombus_diagonal_area_residual"},
        "cyclic_quadrilateral": {"cyclic_quadrilateral", "cyclicity_residual"},
        "tangential_quadrilateral": {"tangential_quadrilateral", "pitot_residual"},
    }
    for family_name, expected_types in theorem_types.items():
        spec = next(spec for spec in specs if spec.family == family_name)
        assert spec.difficulty in {"medium", "hard", "expert"}
        record = SyntheticRecordExporter(solver_profile="auto").build_record(spec, output_dir=tmp_path, image_format="none")
        assert record["synthetic"]["target_value"] == "0"
        assert expected_types.issubset({fact["type"] for fact in record["facts"]})


def test_kite_parameter_variants_have_unique_record_ids():
    family = select_template_families(["kite_area_diagonals"])[0]
    specs = list(family.variants())
    assert len({spec.id for spec in specs}) == len(specs)


def test_formal_trace_enhancer_expands_valid_specs():
    base = SyntheticSampler(seed=2, enable_mutation=False).sample(max_records=1)[0]
    enhanced = FormalTraceEnhancer(seed=4, rounds_distribution={0: 1, 1: 2}).enhance(base)
    assert len(enhanced) == 3
    assert all(spec.id.startswith(base.id) for spec in enhanced)
    reports = [DegeneracyRejector().check(spec) for spec in enhanced]
    assert reports[0].valid
    assert all(report.valid or report.reason == "coincident_points" for report in reports)


def test_formal_trace_enhancer_uses_extended_constructions():
    base = SyntheticSampler(
        families=select_template_families(["rectangle_area_diagonal"]),
        seed=2,
        enable_mutation=False,
    ).sample(max_records=1)[0]
    for operation in ("parallel_through", "perpendicular_through", "point_on_segment_ratio"):
        enhanced = FormalTraceEnhancer(seed=7, rounds_distribution={1: 1}, operation_types=(operation,)).enhance(base)[0]
        assert operation in {item["op"] for item in enhanced.trace}
        assert DegeneracyRejector().check(enhanced).valid


def test_multi_round_enhancement_does_not_repeat_operation_types():
    base = SyntheticSampler(
        families=select_template_families(["rectangle_area_diagonal"]),
        seed=2,
        enable_mutation=False,
    ).sample(max_records=1)[0]
    enhanced = FormalTraceEnhancer(
        seed=8,
        rounds_distribution={3: 1},
        operation_types=("parallel_through", "perpendicular_through", "angle_bisector", "point_on_segment_ratio"),
    ).enhance(base)[0]
    operations = enhanced.metadata["enhancement_operations"]
    assert len(operations) == 3
    assert len(operations) == len(set(operations))


def test_rejector_drops_points_coincident_with_existing_points():
    spec = SyntheticSpec(
        id="coincident_enhancement",
        family="test",
        trace=[
            {"op": "add_point", "id": "A", "coord": ["0", "0"]},
            {"op": "add_point", "id": "B", "coord": ["2", "0"]},
            {"op": "add_point", "id": "O", "coord": ["1", "0"]},
            {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
            {"op": "midpoint", "id": "M_AB_e1_0", "of": ["A", "B"]},
            {"op": "set_target", "type": "length", "object": "AB"},
        ],
        question="Find AB.",
    )
    report = DegeneracyRejector().check(spec)
    assert not report.valid
    assert report.reason == "coincident_points"
    assert report.metadata == {"point": "M_AB_e1_0", "coincident_with": "O"}


def test_rejector_drops_template_foot_coincident_with_endpoint():
    family = select_template_families(
        ["triangle_area_from_height"],
        grid_overrides={"triangle_area_from_height": {"base": (3,), "height": (2,), "skew": (0,)}},
    )[0]
    report = DegeneracyRejector().check(next(family.variants()))

    assert not report.valid
    assert report.reason == "coincident_points"
    assert report.metadata == {"point": "A", "coincident_with": "H"}


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


def test_pipeline_resumes_export_from_enhanced_specs(tmp_path):
    initial_output = tmp_path / "initial" / "records.jsonl"
    initial = GeometryGenerationPipeline(
        PipelineRunConfig(
            target_records=2,
            base_records=2,
            seed=13,
            image_format="none",
            require_exact_count=True,
            base_mutations_per_variant=0,
            enhancement_rounds={0: 1},
        )
    ).run(initial_output)
    enhanced_path = Path(initial["run_config"]["enhanced_path"])

    resumed_output = tmp_path / "resumed" / "records.jsonl"
    resumed = GeometryGenerationPipeline(
        PipelineRunConfig(
            target_records=2,
            seed=13,
            image_format="none",
            require_exact_count=True,
            workers=2,
        )
    ).run(resumed_output, input_enhanced_path=enhanced_path)

    assert resumed["num_records"] == 2
    assert resumed["run_config"]["pipeline"] == "resumed_enhanced_export"
    assert resumed["run_config"]["input_enhanced_path"] == str(enhanced_path)
    assert not (resumed_output.parent / "json" / "base").exists()


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
        base_reserve_records=None,
        base_mutations_per_variant=1,
        enhancement_rounds_json=None,
        enhancement_operations=None,
        max_enhancements_per_base=None,
        grid_json=None,
        progress_every=25,
    )
    result = _apply_preset(args, {"output", "format"})
    assert result.output == "custom.jsonl"
    assert result.format == "both"
    assert result.max_records == 500
    assert result.per_family == 50
    assert "tetrahedron_section" not in result.families
    assert "trapezoid_area" in result.families
    assert "rhombus_diagonal_theorem" in result.families
    assert "cyclic_quadrilateral" in result.families
    assert "tangential_quadrilateral" in result.families
    assert "kite_area_diagonals" in result.families
    assert "concave_composite_area" in result.families
    assert "annulus_area" in result.families
    assert result.pipeline is True
    assert result.base_records == 500
    assert result.base_reserve_records == 200
    assert result.base_mutations_per_variant == 0
    assert result.max_enhancements_per_base == 3
    assert result.enhancement_rounds_json == '{"1": 1, "2": 1, "3": 1}'
    assert "parallel_through" in result.enhancement_operations
    assert "triangle_area_from_height" in result.grid_json


def test_exact_pipeline_backfills_rejected_enhancements(tmp_path):
    class RejectFirst:
        def __init__(self):
            self.calls = 0

        def check(self, spec):
            self.calls += 1
            return type("Report", (), {"valid": self.calls > 1})()

    pipeline = GeometryGenerationPipeline(
        PipelineRunConfig(
            target_records=4,
            base_records=4,
            base_reserve_records=2,
            seed=3,
            image_format="none",
            require_exact_count=True,
            families=("triangle_area_from_height",),
            per_family_limit=4,
            base_mutations_per_variant=0,
            enhancement_rounds={1: 1},
            max_enhancements_per_base=1,
        )
    )
    pipeline.rejector = RejectFirst()
    manifest = pipeline.run(tmp_path / "records.jsonl")
    assert manifest["num_records"] == 4
    assert manifest["run_config"]["num_base_specs"] == 6
    assert manifest["run_config"]["num_rejected_enhancements"] == 1


def test_pipeline_rotates_enhancement_depths_across_base_specs(tmp_path):
    manifest = GeometryGenerationPipeline(
        PipelineRunConfig(
            target_records=6,
            base_records=6,
            seed=11,
            image_format="none",
            require_exact_count=True,
            families=("triangle_area_from_height",),
            base_mutations_per_variant=0,
            enhancement_rounds={1: 1, 2: 1, 3: 1},
            max_enhancements_per_base=3,
            progress_every=0,
        )
    ).run(tmp_path / "records.jsonl")
    assert set(manifest["enhancement_rounds"]) == {"1", "2", "3"}
    assert sum(manifest["enhancement_rounds"].values()) == 6
