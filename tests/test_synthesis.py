"""Unit tests for synthetic sampler and exporter."""

import json
from copy import deepcopy
from argparse import Namespace
from pathlib import Path

import pytest

from geosym_world.construction import normalize_trace, operation_references
from geosym_world.construction.builder import ConstructionBuilder
from geosym_world.errors import UnknownObjectError
from geosym_world.schema.derivations import DerivationTracer
from geosym_world.schema.solutions import build_solution_paths, validate_solution_paths
from geosym_world.synthesis import CandidateDiagnostics, GeometryGenerationPipeline, PipelineRunConfig, SyntheticRecordExporter, SyntheticSampler, SyntheticSpec, assign_split_labels, assign_splits, lineage_root, load_grid_overrides, select_template_families, split_group_key, validate_split_disjoint, validate_task_contract
from geosym_world.synthesis.enhancer import FormalTraceEnhancer
from geosym_world.synthesis.difficulty import score_task, score_trace
from geosym_world.synthesis.filters import DegeneracyRejector
from geosym_world.solver import FactClosureSolver
from geosym_world.tools.synthesize_records import _apply_preset
from geosym_world.verifier import TraceVerifier


def test_sampler_produces_valid_specs():
    sampler = SyntheticSampler(seed=3)
    specs = sampler.sample(max_records=5)
    assert len(specs) == 5
    assert {spec.family for spec in specs}
    assert all(spec.trace for spec in specs)
    assert sampler.diagnostics.accepted == 5
    assert sampler.diagnostics.checked == sampler.diagnostics.accepted + sampler.diagnostics.rejected
    assert all(spec.metadata["candidate_validation"]["status"] == "accepted" for spec in specs)
    assert all(spec.metadata["candidate_validation"]["reason"] == "validated" for spec in specs)
    assert all(spec.metadata["candidate_validation"]["derivation_checksum"] for spec in specs)


def test_similarity_proof_exports_two_independent_solution_paths(tmp_path):
    spec = next(iter(select_template_families(["similarity_dual_proof"])[0].variants()))
    state = ConstructionBuilder().execute(spec.trace)
    facts = FactClosureSolver(profile="auto").solve(state)
    solutions = build_solution_paths(facts, state.target)

    assert solutions["validation"]["valid"]
    assert solutions["num_paths"] == 2
    assert solutions["alternative_count"] == 1
    assert solutions["rules"] == ["sss_similarity", "aa_similarity"]
    assert all(path["integrity"]["valid"] for path in solutions["paths"])
    assert solutions["paths"][0]["integrity"]["max_depth"] >= 2
    assert DegeneracyRejector().check(spec).valid

    record = SyntheticRecordExporter().build_record(spec, output_dir=tmp_path, image_format="none", scene_payload="none")
    assert record["solution_paths"] == record["gt"]["solution_paths"]
    assert record["task"]["solutions"]["num_paths"] == 2
    assert record["task"]["solutions"]["alternative_count"] == 1
    assert record["formal_state"]["target"]["mode"] == "proof"
    assert record["answer"] == "proved"


def test_similarity_proof_uses_minimal_public_annotations(tmp_path):
    spec = next(iter(select_template_families(["similarity_dual_proof"])[0].variants()))
    record = SyntheticRecordExporter().build_record(spec, output_dir=tmp_path, image_format="none", scene_payload="full")

    assert record["task"]["annotation"] == {
        "schema_version": "1.2",
        "mode": "minimal_public_givens",
        "profile": {"schema_version": "1.0", "level": "givens", "relation_markers": True},
        "visible_measurements": [],
        "visible_relations": ["given_parallel_DE_BC"],
        "visible_relation_values": ["given_corresponding_side_ratio"],
        "textual_relations": ["given_D_on_AB", "given_E_on_AC"],
        "allow_inferred_markers": False,
    }
    assert not [given for given in record["public_givens"] if given["kind"] == "quantity"]
    assert {given["type"] for given in record["public_givens"]} == {
        "parallel",
        "point_on_line",
        "proportional_lengths",
    }
    paths = {path["root_fact"]["rule"]: path["root_fact"] for path in record["solution_paths"]["paths"]}
    assert paths["sss_similarity"]["metadata"]["proof_basis"] == "public_proportional_lengths"
    assert paths["aa_similarity"]["metadata"]["proof_basis"] == "public_parallel_incidence"
    assert all(path["dependencies"] for path in paths.values())


def test_points_only_export_suppresses_values_and_relation_markers(tmp_path):
    spec = next(iter(select_template_families(["similarity_dual_proof"])[0].variants()))
    record = SyntheticRecordExporter(annotation_level="points", relation_markers=False).build_record(
        spec,
        output_dir=tmp_path,
        image_format="none",
        scene_payload="full",
    )

    annotation = record["task"]["annotation"]
    assert annotation["profile"] == {"schema_version": "1.0", "level": "points", "relation_markers": False}
    assert annotation["visible_measurements"] == []
    assert annotation["visible_relations"] == []
    assert annotation["visible_relation_values"] == []
    assert set(annotation["textual_relations"]) == {
        "given_D_on_AB",
        "given_E_on_AC",
        "given_corresponding_side_ratio",
        "given_parallel_DE_BC",
    }
    scene = record["image"]["scene"]
    assert [label["text"] for label in scene["labels"]] == ["A", "B", "C", "D", "E"]
    assert not [primitive for primitive in scene["primitives"] if primitive["role"] == "marker"]


def test_solution_path_validation_rejects_tampered_rule():
    spec = next(iter(select_template_families(["similarity_dual_proof"])[0].variants()))
    state = ConstructionBuilder().execute(spec.trace)
    solutions = build_solution_paths(FactClosureSolver(profile="auto").solve(state), state.target)
    tampered = deepcopy(solutions)
    tampered["paths"][0]["root_fact"]["rule"] = "coordinate_guess"
    assert not validate_solution_paths(tampered).valid


def test_difficulty_policy_exposes_construction_derivation_and_solution_features():
    simple = next(iter(select_template_families(["triangle_area_from_height"])[0].variants()))
    proof = next(iter(select_template_families(["similarity_dual_proof"])[0].variants()))
    estimated_label, estimated = score_trace(proof.trace, proof.diagnostic_axes)
    state = ConstructionBuilder().execute(proof.trace)
    facts = FactClosureSolver(profile="auto").solve(state)
    derivation = DerivationTracer().trace_target(facts)
    solutions = build_solution_paths(facts, state.target)
    actual_label, actual = score_task(proof.trace, proof.diagnostic_axes, derivation=derivation, solution_paths=solutions)
    _, simple_policy = score_trace(simple.trace, simple.diagnostic_axes)

    assert estimated["schema_version"] == "2.0"
    assert estimated["features"]["construction"]["effective_depth"] > 0
    assert estimated["features"]["construction"]["dependency_graph_checksum"]
    assert estimated_label in {"easy", "medium", "hard", "expert"}
    assert actual_label == actual["label"]
    assert actual["features"]["derivation"]["available"]
    assert actual["features"]["solutions"]["alternative_count"] == 1
    assert actual["score"] > estimated["score"] > simple_policy["score"]


def test_candidate_diagnostics_bounds_rejection_examples():
    diagnostics = CandidateDiagnostics(max_examples_per_reason=2)
    for index in range(4):
        diagnostics.record_rejection("duplicate_content", candidate_id=f"candidate_{index}", family="fixture")
    payload = diagnostics.to_dict()
    assert payload["checked"] == 4
    assert payload["rejected"] == 4
    assert payload["rejection_counts"] == {"duplicate_content": 4}
    assert payload["rejection_code_counts"] == {"policy.duplicate.content": 4}
    assert payload["rejection_stage_counts"] == {"policy": 4}
    assert payload["rejection_component_counts"] == {"duplicate": 4}
    assert payload["rejection_family_counts"] == {"fixture": 4}
    assert len(payload["rejection_examples"]["duplicate_content"]) == 2


def test_candidate_diagnostics_separates_policy_skips():
    diagnostics = CandidateDiagnostics(max_examples_per_reason=1)
    diagnostics.record_accept()
    diagnostics.record_rejection("invalid", candidate_id="rejected", family="fixture")
    diagnostics.record_skip("quota_reached", candidate_id="skipped", family="fixture")
    payload = diagnostics.to_dict()
    assert payload["considered"] == 3
    assert payload["checked"] == 2
    assert payload["accepted"] == 1
    assert payload["rejected"] == 1
    assert payload["skipped"] == 1
    assert payload["skip_counts"] == {"quota_reached": 1}


def test_lineage_aware_split_keeps_variants_together():
    specs = [
        SyntheticSpec("base_a", "fixture", [], "", metadata={"lineage_root": "base_a"}),
        SyntheticSpec("base_a_e1", "fixture", [], "", metadata={"lineage_root": "base_a", "base_spec_id": "base_a"}),
        SyntheticSpec("base_b", "fixture", [], "", metadata={"lineage_root": "base_b"}),
        SyntheticSpec("base_c", "fixture", [], "", metadata={"lineage_root": "base_c"}),
    ]
    splits = assign_splits(specs, seed=7)
    validate_split_disjoint(splits)
    variants = [spec for group in splits.values() for spec in group if lineage_root(spec) == "base_a"]
    assert len(variants) == 2
    assert {spec.metadata["split"] for spec in variants}.__len__() == 1


def test_split_labels_preserve_record_order():
    specs = [SyntheticSpec(f"base_{index}", "fixture", [], "") for index in range(5)]
    assigned = assign_split_labels(specs, seed=4)
    assert [spec.id for spec in assigned] == [spec.id for spec in specs]
    assert all(spec.metadata["split"] in {"train", "dev", "test"} for spec in assigned)


def test_split_labels_preserve_existing_assignment_and_reject_conflicts():
    existing = SyntheticSpec("base_a", "fixture", [], "", metadata={"lineage_root": "root_a", "split": "test"})
    assert assign_split_labels([existing], seed=999)[0].metadata["split"] == "test"
    conflicting = SyntheticSpec("base_a_variant", "fixture", [], "", metadata={"lineage_root": "root_a", "split": "dev"})
    with pytest.raises(ValueError, match="conflicting split labels"):
        assign_split_labels([existing, conflicting], seed=0)


def test_topology_split_keeps_parameter_variants_together():
    trace_a = [
        {"op": "point", "id": "A", "coords": [0, 0]},
        {"op": "set_target", "type": "area", "object": "shape_a"},
    ]
    trace_b = [
        {"op": "point", "id": "P", "coords": [17, -4]},
        {"op": "set_target", "type": "area", "object": "shape_b"},
    ]
    specs = [
        SyntheticSpec("variant_a", "area_fixture", trace_a, "", diagnostic_axes=("area",)),
        SyntheticSpec("variant_b", "area_fixture", trace_b, "", diagnostic_axes=("area",)),
    ]

    assigned = assign_split_labels(specs, seed=3, group_mode="topology")

    assert split_group_key(specs[0], mode="topology") == split_group_key(specs[1], mode="topology")
    assert {spec.metadata["split"] for spec in assigned}.__len__() == 1
    assert all(spec.metadata["split_group_mode"] == "topology" for spec in assigned)


def test_topology_split_rejects_conflicting_existing_labels():
    trace = [{"op": "set_target", "type": "area", "object": "shape"}]
    specs = [
        SyntheticSpec("variant_a", "area_fixture", trace, "", metadata={"split": "dev"}),
        SyntheticSpec("variant_b", "area_fixture", trace, "", metadata={"split": "test"}),
    ]

    with pytest.raises(ValueError, match="conflicting split labels"):
        assign_split_labels(specs, group_mode="topology")


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
    assert record["schema_version"] == "1.0"
    assert record["image"]["formats"]["svg"].endswith(".svg")
    assert record["synthetic"]["target_value"] is not None
    assert record["construction_trace"] == specs[0].trace
    assert len(record["construction_steps"]) == len(record["construction_trace"])
    assert record["construction_steps"][-1]["transition"]["after_checksum"] == record["formal_state_checksum"]
    assert record["synthetic"]["lineage"]["root"] == specs[0].metadata["lineage_root"]
    assert record["synthetic"]["split"] in {"train", "dev", "test"}
    assert record["synthetic"]["metadata"]["split_group_mode"] == "lineage"
    assert record["render_quality"] is not None
    assert record["derivation"]["integrity"]["valid"]
    assert record["derivation"]["validation"]["valid"]
    assert record["derivation"]["steps"][-1]["type"] == "target"
    assert record["derivation"]["checksum"] == record["gt"]["derivation"]["checksum"]
    assert record["task"] == record["gt"]["task"]
    assert record["task"]["target"] == record["formal_state"]["target"]
    assert record["task"]["construction"]["num_steps"] == len(record["construction_trace"])
    assert record["task"]["derivation"]["checksum"] == record["derivation"]["checksum"]
    assert record["task"]["validation"]["valid"]
    assert validate_task_contract(record["task"]).valid
    assert record["task"]["public_givens"]
    assert record["public_givens"] == record["task"]["public_givens"]
    assert all("coords" not in given for given in record["task"]["public_givens"])
    tampered_task = {**record["task"], "structure_signature": "0" * 64}
    assert not validate_task_contract(tampered_task).valid
    assert record["answer"] == record["synthetic"]["target_value"]
    qa = json.loads(path.with_suffix(".jsonl.qa.jsonl").read_text(encoding="utf-8").splitlines()[0])
    gt = json.loads(path.with_suffix(".jsonl.gt.jsonl").read_text(encoding="utf-8").splitlines()[0])
    parser_input = json.loads(path.with_suffix(".jsonl.parser-input.jsonl").read_text(encoding="utf-8").splitlines()[0])
    parser_reference = json.loads(path.with_suffix(".jsonl.parser-reference.jsonl").read_text(encoding="utf-8").splitlines()[0])
    typed_parser_reference = json.loads(path.with_suffix(".jsonl.typed-parser-reference.jsonl").read_text(encoding="utf-8").splitlines()[0])
    parser_source_audit = json.loads(path.with_suffix(".jsonl.parser-source-audit.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert qa["question"] == record["question"]
    assert qa["answer"] == record["answer"]
    assert gt["target"]["value"] == record["synthetic"]["target_value"]
    assert gt["derivation"]["integrity"]["valid"]
    assert gt["derivation"]["validation"]["valid"]
    assert set(parser_input) == {"schema_version", "id", "image", "question"}
    assert parser_input["id"].startswith("sample_")
    assert parser_input["id"] != record["id"]
    assert record["id"] not in Path(parser_input["image"]).name
    assert Path(parser_input["image"]).suffix == ".png"
    assert Path(parser_input["image"]).is_file()
    assert parser_reference["parser_input_checksum"]
    assert parser_reference["id"] == parser_input["id"]
    assert parser_reference["private_supervision"]["source_record_id"] == record["id"]
    assert parser_reference["private_supervision"]["formal_state_checksum"] == record["formal_state_checksum"]
    assert typed_parser_reference["id"] == parser_input["id"]
    assert typed_parser_reference["source_manifest"]["metadata"]["formal_state_checksum"] == record["formal_state_checksum"]
    assert typed_parser_reference["parse_graph"]["targets"]
    assert typed_parser_reference["parse_graph"]["checksum"]
    assert parser_source_audit["public_id"] == parser_input["id"]
    assert parser_source_audit["source_record_id"] == record["id"]
    assert parser_source_audit["checksum"]
    assert manifest["parser_input_path"].endswith(".parser-input.jsonl")
    assert manifest["parser_reference_path"].endswith(".parser-reference.jsonl")
    assert manifest["typed_parser_reference_path"].endswith(".typed-parser-reference.jsonl")
    assert manifest["parser_source_audit_path"].endswith(".parser-source-audit.jsonl")
    assert manifest["num_parser_benchmark_records"] == manifest["num_records"]
    assert manifest["parser_benchmark_skip_reasons"] == {}
    assert manifest["invalid_derivations"] == 0
    assert sum(manifest["derivation_depths"].values()) == manifest["num_records"]
    assert manifest["derivation_rules"]["target_lookup"] == manifest["num_records"]
    assert sum(manifest["target_types"].values()) == manifest["num_records"]
    assert sum(manifest["construction_depths"].values()) == manifest["num_records"]
    assert manifest["num_structure_signatures"] >= 1
    assert manifest["num_split_groups"] == manifest["num_lineage_roots"]
    assert manifest["split_group_modes"] == ["lineage"]


def test_exporter_excludes_3d_scenes_from_parser_v0_sidecars(tmp_path):
    spec = next(iter(select_template_families(["tetrahedron_section"])[0].variants()))
    path = tmp_path / "synthetic_3d.jsonl"

    manifest = SyntheticRecordExporter().write_jsonl([spec], path, image_format="png")

    assert manifest["num_records"] == 1
    assert manifest["num_parser_benchmark_records"] == 0
    assert manifest["parser_benchmark_skip_reasons"] == {"unsupported_render_dim:3": 1}
    assert path.with_suffix(".jsonl.parser-input.jsonl").read_text(encoding="utf-8") == ""
    assert path.with_suffix(".jsonl.typed-parser-reference.jsonl").read_text(encoding="utf-8") == ""


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
        "triangle_partition_area",
        "triangle_partition_area_ratio",
        "rectangular_frame_area",
        "double_rectangular_hole_area",
        "double_rectangular_hole_perimeter",
        "rectangle_area_diagonal",
        "parallelogram_area",
        "trapezoid_area",
        "kite_area_diagonals",
        "rhombus_diagonal_theorem",
        "cyclic_quadrilateral",
        "tangential_quadrilateral",
        "concave_composite_area",
        "annulus_area",
        "annulus_perimeter",
    }
    families = select_template_families(expected)
    specs = SyntheticSampler(families=families, seed=4, enable_mutation=False).sample(max_records=len(expected))
    assert {spec.family for spec in specs} == expected
    assert all(DegeneracyRejector().check(spec).valid for spec in specs)

    annulus = next(spec for spec in specs if spec.family == "annulus_area")
    record = SyntheticRecordExporter(solver_profile="auto").build_record(annulus, output_dir=tmp_path, image_format="none")
    assert record["synthetic"]["target_value"] == "21*pi"
    assert {step["rule"] for step in record["derivation"]["steps"]}.issuperset({"distance_formula", "annulus_radius_area", "target_lookup"})
    assert len([given for given in record["public_givens"] if given["kind"] == "quantity"]) == 2

    annulus_perimeter = next(spec for spec in specs if spec.family == "annulus_perimeter")
    perimeter_record = SyntheticRecordExporter(solver_profile="auto").build_record(annulus_perimeter, output_dir=tmp_path, image_format="none")
    assert perimeter_record["synthetic"]["target_value"] == "14*pi"
    assert {step["rule"] for step in perimeter_record["derivation"]["steps"]}.issuperset({"distance_formula", "annulus_radius_perimeter", "target_lookup"})
    assert len([given for given in perimeter_record["public_givens"] if given["kind"] == "quantity"]) == 2

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
    partition = next(spec for spec in specs if spec.family == "triangle_partition_area")
    record = SyntheticRecordExporter(solver_profile="auto").build_record(
        partition,
        output_dir=tmp_path,
        image_format="none",
        scene_payload="metadata",
    )
    rules = {step["rule"] for step in record["derivation"]["steps"]}
    public_ids = {given["source_fact_id"] for given in record["public_givens"] if given["kind"] == "quantity"}
    semantic_polygon_areas = [step for step in record["derivation"]["steps"] if step["rule"] == "triangle_base_height"]
    assert len(semantic_polygon_areas) == 2
    assert {"region_from_semantic_polygon_area", "region_partition_area_sum"}.issubset(rules)
    assert "region_partition_area_check" not in rules
    assert any(fact["rule"] == "region_partition_area_check" and fact["value"] == "0" for fact in record["facts"])
    assert all(set(step["dependencies"]).issubset(public_ids) for step in semantic_polygon_areas)
    assert record["derivation"]["integrity"]["max_depth"] >= 5
    assert record["image"]["scene"]["metadata"]["region_geometry"]["partition_ABC"]["status"] == "polygon"

    ratio_spec = next(spec for spec in specs if spec.family == "triangle_partition_area_ratio")
    record = SyntheticRecordExporter(solver_profile="auto").build_record(
        ratio_spec,
        output_dir=tmp_path,
        image_format="none",
        scene_payload="full",
    )
    ratio_step = next(step for step in record["derivation"]["steps"] if step["rule"] == "region_area_ratio")
    assert record["formal_state"]["target"]["type"] == "ratio"
    assert record["synthetic"]["target_value"] == "2/3"
    assert ratio_step["metadata"]["unit"] == "dimensionless"
    assert len(ratio_step["dependencies"]) == 2
    assert record["derivation"]["integrity"]["max_depth"] >= 5

    frame = next(spec for spec in specs if spec.family == "rectangular_frame_area")
    record = SyntheticRecordExporter(solver_profile="auto").build_record(
        frame,
        output_dir=tmp_path,
        image_format="none",
        scene_payload="metadata",
    )
    public_ids = {given["source_fact_id"] for given in record["public_givens"] if given["kind"] == "quantity"}
    area_steps = [step for step in record["derivation"]["steps"] if step["type"] == "area"]
    difference_step = next(step for step in area_steps if step["rule"] == "region_difference_area")
    assert len(public_ids) == 4
    assert len([step for step in area_steps if step["rule"] == "rectangle_base_height"]) == 2
    assert len(difference_step["dependencies"]) == 2
    assert record["image"]["scene"]["metadata"]["region_geometry"]["shaded_frame"]["status"] == "polygon_with_holes"
    assert any(fact["rule"] == "region_difference_area_check" and fact["value"] == "0" for fact in record["facts"])
    double_hole = next(spec for spec in specs if spec.family == "double_rectangular_hole_area")
    record = SyntheticRecordExporter(solver_profile="auto").build_record(
        double_hole,
        output_dir=tmp_path,
        image_format="none",
        scene_payload="metadata",
    )
    public_ids = {given["source_fact_id"] for given in record["public_givens"] if given["kind"] == "quantity"}
    area_steps = [step for step in record["derivation"]["steps"] if step["type"] == "area"]
    difference_step = next(step for step in area_steps if step["rule"] == "region_hole_difference_area")
    geometry = record["image"]["scene"]["metadata"]["region_geometry"]["perforated_plate"]
    assert record["synthetic"]["target_value"] == "76"
    assert len(public_ids) == 6
    assert len([step for step in area_steps if step["rule"] == "rectangle_base_height"]) == 3
    assert len(difference_step["dependencies"]) == 3
    assert record["derivation"]["integrity"]["max_depth"] >= 5
    assert geometry["status"] == "polygon_with_holes"
    assert len(geometry["holes"]) == 2
    assert any(fact["rule"] == "region_hole_difference_area_check" and fact["value"] == "0" for fact in record["facts"])

    double_hole_perimeter = next(spec for spec in specs if spec.family == "double_rectangular_hole_perimeter")
    record = SyntheticRecordExporter(solver_profile="auto").build_record(
        double_hole_perimeter,
        output_dir=tmp_path,
        image_format="none",
        scene_payload="metadata",
    )
    public_ids = {given["source_fact_id"] for given in record["public_givens"] if given["kind"] == "quantity"}
    perimeter_steps = [step for step in record["derivation"]["steps"] if step["type"] == "perimeter"]
    boundary_step = next(step for step in perimeter_steps if step["rule"] == "region_hole_boundary_sum")
    geometry = record["image"]["scene"]["metadata"]["region_geometry"]["perforated_plate"]
    assert record["formal_state"]["target"]["type"] == "perimeter"
    assert record["synthetic"]["target_value"] == "54"
    assert len(public_ids) == 6
    assert len([step for step in perimeter_steps if step["rule"] == "rectangle_side_perimeter"]) == 3
    assert len(boundary_step["dependencies"]) == 3
    assert record["derivation"]["integrity"]["max_depth"] >= 5
    assert geometry["status"] == "polygon_with_holes"
    assert len(geometry["holes"]) == 2
    assert any(fact["rule"] == "region_hole_boundary_sum_check" and fact["value"] == "0" for fact in record["facts"])

    concave = next(spec for spec in specs if spec.family == "concave_composite_area")
    record = SyntheticRecordExporter(solver_profile="auto").build_record(
        concave,
        output_dir=tmp_path,
        image_format="none",
        scene_payload="metadata",
    )
    public_ids = {given["source_fact_id"] for given in record["public_givens"] if given["kind"] == "quantity"}
    area_steps = [step for step in record["derivation"]["steps"] if step["type"] == "area"]
    difference_step = next(step for step in area_steps if step["rule"] == "region_difference_area")
    geometry = record["image"]["scene"]["metadata"]["region_geometry"]["concave_ABCDEF"]
    assert record["synthetic"]["target_value"] == "26"
    assert len(public_ids) == 4
    assert len([step for step in area_steps if step["rule"] == "rectangle_base_height"]) == 2
    assert len(difference_step["dependencies"]) == 2
    assert record["derivation"]["integrity"]["max_depth"] >= 5
    assert geometry["status"] == "polygon"
    assert len(geometry["boundary"]) == 6
    assert any(fact["rule"] == "region_difference_area_check" and fact["value"] == "0" for fact in record["facts"])


def test_annulus_formula_rejects_missing_and_false_radius_segments():
    spec = next(iter(select_template_families(["annulus_area"])[0].variants()))
    region_operation = next(operation for operation in normalize_trace(spec.trace) if operation.op == "add_region")
    assert operation_references(region_operation) == (
        ("object", "O"),
        ("circle", "inner_circle"),
        ("circle", "outer_circle"),
        ("segment", "OA"),
        ("segment", "OB"),
    )
    missing_reference = deepcopy(spec.trace)
    region_step = next(step for step in missing_reference if step["op"] == "add_region")
    region_step["metadata"]["area_formula"]["outer_radius_segment"] = "missing_radius"
    with pytest.raises(UnknownObjectError, match="requires existing outer and inner radius segments"):
        ConstructionBuilder().execute(missing_reference)

    false_radius = deepcopy(spec.trace)
    outer_segment = next(step for step in false_radius if step.get("id") == "OA")
    outer_segment["points"] = ["O", "B"]
    report = TraceVerifier().verify_trace(false_radius)
    assert not report.valid
    assert {issue.code for issue in report.issues}.issuperset(
        {"state.invalid_annulus_area_formula", "state.invalid_annulus_perimeter_formula"}
    )


@pytest.mark.parametrize(
    ("family_name", "target_type", "expected_value", "expected_rule"),
    (
        ("circle_sector", "area", "3*pi/2", "sector_radius_angle_area"),
        ("sector_arc_length", "arc_length", "pi", "sector_radius_angle_arc_length"),
        ("sector_perimeter", "perimeter", "pi + 6", "sector_radius_angle_perimeter"),
    ),
)
def test_sector_targets_derive_from_public_radius_and_angle(tmp_path, family_name, target_type, expected_value, expected_rule):
    spec = next(iter(select_template_families([family_name])[0].variants()))
    record = SyntheticRecordExporter(solver_profile="auto").build_record(spec, output_dir=tmp_path, image_format="none", scene_payload="full")
    quantities = [given for given in record["public_givens"] if given["kind"] == "quantity"]
    target = record["derivation"]["target_fact"]
    resolved = next(fact for fact in record["facts"] if fact["id"] == target["metadata"]["resolved_fact_id"])

    assert record["formal_state"]["target"]["type"] == target_type
    assert record["synthetic"]["target_value"] == expected_value
    assert resolved["rule"] == expected_rule
    assert {(given["type"], given["value"]) for given in quantities} == {("length", "3"), ("central_angle", "pi/3")}
    assert set(resolved["dependencies"]) == {given["source_fact_id"] for given in quantities}
    assert {label["text"] for label in record["image"]["scene"]["labels"] if label["id"].startswith("measure:")} == {"3", "π⁄₃"}
    assert all(fact["value"] == "0" for fact in record["facts"] if fact["type"].startswith("sector_") and fact["type"].endswith("_residual"))


def test_sector_rejects_false_central_angle_and_missing_radius_segment():
    spec = next(iter(select_template_families(["circle_sector"])[0].variants()))
    missing_radius = deepcopy(spec.trace)
    region_step = next(step for step in missing_radius if step["op"] == "mark_sector_region")
    region_step["metadata"]["radius_segment"] = "missing_radius"
    with pytest.raises(UnknownObjectError, match="requires an existing radius segment"):
        ConstructionBuilder().execute(missing_radius)

    false_angle = deepcopy(spec.trace)
    region_step = next(step for step in false_angle if step["op"] == "mark_sector_region")
    region_step["central_angle"] = "pi/2"
    report = TraceVerifier().verify_trace(false_angle)
    assert not report.valid
    assert "state.invalid_sector_measure_formula" in {issue.code for issue in report.issues}


def test_circular_segment_area_derives_sector_minus_chord_triangle(tmp_path):
    spec = next(iter(select_template_families(["circular_segment_area"])[0].variants()))
    state = ConstructionBuilder().execute(spec.trace)
    assert state.checksum() == type(state).from_dict(state.to_dict()).checksum()
    record = SyntheticRecordExporter(solver_profile="auto").build_record(spec, output_dir=tmp_path, image_format="none", scene_payload="full")
    rules = [step["rule"] for step in record["derivation"]["steps"]]
    quantities = [given for given in record["public_givens"] if given["kind"] == "quantity"]
    target_step = next(step for step in record["derivation"]["steps"] if step["rule"] == "circular_segment_difference")

    assert record["synthetic"]["target_value"] == "-9*sqrt(3)/4 + 3*pi/2"
    assert rules.index("sector_radius_angle_area") < rules.index("circular_segment_difference")
    assert rules.index("chord_triangle_radius_angle") < rules.index("circular_segment_difference")
    assert len(target_step["dependencies"]) == 2
    assert {(given["type"], given["value"]) for given in quantities} == {("length", "3"), ("central_angle", "pi/3")}
    assert any(fact["type"] == "circular_segment_area_residual" and fact["value"] == "0" for fact in record["facts"])
    assert [(primitive["id"], primitive["kind"], primitive["role"]) for primitive in record["image"]["scene"]["primitives"] if primitive["role"] in {"target", "region"}] == [("segment_AB", "circular_segment", "target")]


def test_circular_segment_rejects_mismatched_chord():
    spec = next(iter(select_template_families(["circular_segment_area"])[0].variants()))
    trace = deepcopy(spec.trace)
    trace.insert(3, {"op": "add_point", "id": "C", "coord": ["0", "3"]})
    chord = next(step for step in trace if step.get("id") == "AB")
    chord["points"] = ["A", "C"]
    report = TraceVerifier().verify_trace(trace)
    assert not report.valid
    assert "execute.geometry_world_error" in {issue.code for issue in report.issues}


def test_circular_segment_perimeter_derives_arc_plus_chord(tmp_path):
    spec = next(iter(select_template_families(["circular_segment_perimeter"])[0].variants()))
    state = ConstructionBuilder().execute(spec.trace)
    assert state.checksum() == type(state).from_dict(state.to_dict()).checksum()
    record = SyntheticRecordExporter(solver_profile="auto").build_record(
        spec,
        output_dir=tmp_path,
        image_format="none",
        scene_payload="full",
    )
    rules = [step["rule"] for step in record["derivation"]["steps"]]
    quantities = [given for given in record["public_givens"] if given["kind"] == "quantity"]
    chord_step = next(step for step in record["derivation"]["steps"] if step["rule"] == "chord_radius_angle_length")
    target_step = next(step for step in record["derivation"]["steps"] if step["rule"] == "circular_segment_arc_chord_perimeter")

    assert record["synthetic"]["target_value"] == "3 + pi"
    assert rules.index("sector_radius_angle_arc_length") < rules.index("circular_segment_arc_chord_perimeter")
    assert rules.index("chord_radius_angle_length") < rules.index("circular_segment_arc_chord_perimeter")
    assert len(chord_step["dependencies"]) == 2
    assert len(target_step["dependencies"]) == 2
    assert {(given["type"], given["value"]) for given in quantities} == {("length", "3"), ("central_angle", "pi/3")}
    assert "AB" not in {given["display_object"] for given in quantities}
    assert any(fact["type"] == "chord_length_residual" and fact["value"] == "0" for fact in record["facts"])
    assert any(fact["type"] == "circular_segment_perimeter_residual" and fact["value"] == "0" for fact in record["facts"])


def test_circular_segment_area_ratio_uses_admitted_region_areas(tmp_path):
    spec = next(iter(select_template_families(["circular_segment_area_ratio"])[0].variants()))
    record = SyntheticRecordExporter(solver_profile="auto").build_record(
        spec,
        output_dir=tmp_path,
        image_format="none",
        scene_payload="full",
    )

    assert record["formal_state"]["target"]["type"] == "ratio"
    assert record["formal_state"]["target"]["metadata"]["ratio_kind"] == "area"
    rules = {step["rule"] for step in record["derivation"]["steps"]}
    assert "circular_segment_difference" in rules
    assert "region_area_ratio" in rules
    assert record["solution_paths"]["validation"]["valid"]


def test_circular_segment_rejects_invalid_perimeter_formula():
    spec = next(iter(select_template_families(["circular_segment_perimeter"])[0].variants()))
    trace = deepcopy(spec.trace)
    region_step = next(step for step in trace if step.get("id") == "segment_AB")
    region_step["metadata"]["perimeter_formula"]["chord"] = "OA"
    report = TraceVerifier().verify_trace(trace)
    assert not report.valid
    assert "execute.geometry_world_error" in {issue.code for issue in report.issues}


def test_equal_circle_lens_area_derives_public_measurement_chain(tmp_path):
    spec = next(iter(select_template_families(["equal_circle_lens_area"])[0].variants()))
    state = ConstructionBuilder().execute(spec.trace)
    assert state.checksum() == type(state).from_dict(state.to_dict()).checksum()
    record = SyntheticRecordExporter(solver_profile="auto").build_record(
        spec,
        output_dir=tmp_path,
        image_format="none",
        scene_payload="full",
    )
    rules = [step["rule"] for step in record["derivation"]["steps"]]
    quantities = [given for given in record["public_givens"] if given["kind"] == "quantity"]

    assert record["synthetic"]["target_value"] == "-4*sqrt(2) + 18*acos(1/3)"
    assert {(given["display_object"], given["value"]) for given in quantities} == {("O1O2", "2"), ("O1R", "3")}
    assert rules.index("lens_angle_from_radius_distance") < rules.index("lens_two_sector_area")
    assert rules.index("lens_half_chord_from_radius_distance") < rules.index("lens_two_triangle_area")
    assert rules.index("lens_two_sector_area") < rules.index("lens_equal_radius_area")
    assert rules.index("lens_two_triangle_area") < rules.index("lens_equal_radius_area")
    assert any(fact["type"] == "lens_area_residual" and fact["value"] == "0" for fact in record["facts"])
    assert [(primitive["id"], primitive["kind"], primitive["role"]) for primitive in record["image"]["scene"]["primitives"] if primitive["role"] in {"target", "region"}] == [("lens_12", "lens", "target")]


def test_equal_circle_lens_rejects_mismatched_boundary_arc():
    spec = next(iter(select_template_families(["equal_circle_lens_area"])[0].variants()))
    trace = deepcopy(spec.trace)
    arc_step = next(step for step in trace if step.get("id") == "arc_2")
    arc_step["circle"] = "circle_1"
    report = TraceVerifier().verify_trace(trace)
    assert not report.valid
    assert "execute.geometry_world_error" in {issue.code for issue in report.issues}


def test_circle_lens_area_derives_two_public_radii(tmp_path):
    spec = next(iter(select_template_families(["circle_lens_area"])[0].variants()))
    state = ConstructionBuilder().execute(spec.trace)
    assert state.checksum() == type(state).from_dict(state.to_dict()).checksum()
    record = SyntheticRecordExporter(solver_profile="auto").build_record(
        spec,
        output_dir=tmp_path,
        image_format="none",
        scene_payload="full",
    )
    rules = [step["rule"] for step in record["derivation"]["steps"]]
    quantities = [given for given in record["public_givens"] if given["kind"] == "quantity"]

    assert record["synthetic"]["target_value"] == "-12 + 9*acos(3/5) + 16*acos(4/5)"
    assert {(given["display_object"], given["value"]) for given in quantities} == {
        ("O1O2", "5"),
        ("O1R1", "4"),
        ("O2R2", "3"),
    }
    assert rules.index("lens_first_projection_from_radii_distance") < rules.index("lens_first_angle_from_projection_radius")
    assert rules.index("lens_second_projection_from_distance") < rules.index("lens_second_angle_from_projection_radius")
    assert rules.index("lens_first_sector_area") < rules.index("lens_sector_area_sum")
    assert rules.index("lens_second_sector_area") < rules.index("lens_sector_area_sum")
    assert rules.index("lens_sector_area_sum") < rules.index("lens_two_radius_area")
    assert rules.index("lens_triangle_area_sum") < rules.index("lens_two_radius_area")
    assert any(fact["type"] == "lens_area_residual" and fact["value"] == "0" for fact in record["facts"])


def test_circle_lens_rejects_equal_radius_formula_for_unequal_circles():
    spec = next(iter(select_template_families(["circle_lens_area"])[0].variants()))
    trace = deepcopy(spec.trace)
    region_step = next(step for step in trace if step.get("id") == "lens_12")
    region_step["metadata"]["area_formula"] = {
        "rule": "lens_equal_radius_area",
        "radius_segment": "O1R1",
        "center_segment": "O1O2",
    }
    report = TraceVerifier().verify_trace(trace)
    assert not report.valid
    assert "execute.geometry_world_error" in {issue.code for issue in report.issues}


def test_circle_lens_rejects_major_boundary_arc():
    spec = next(iter(select_template_families(["circle_lens_area"])[0].variants()))
    trace = deepcopy(spec.trace)
    arc_step = next(step for step in trace if step.get("id") == "arc_2")
    arc_step["orientation"] = "cw"
    report = TraceVerifier().verify_trace(trace)
    assert not report.valid
    assert "execute.geometry_world_error" in {issue.code for issue in report.issues}


@pytest.mark.parametrize(
    ("family_name", "expected_rule"),
    (
        ("equal_circle_lens_perimeter", "lens_equal_radius_perimeter"),
        ("circle_lens_perimeter", "lens_two_radius_perimeter"),
    ),
)
def test_lens_perimeters_derive_from_public_measurements(tmp_path, family_name, expected_rule):
    spec = next(iter(select_template_families([family_name])[0].variants()))
    record = SyntheticRecordExporter(solver_profile="auto").build_record(
        spec,
        output_dir=tmp_path,
        image_format="none",
        scene_payload="full",
    )
    perimeter_steps = [step for step in record["derivation"]["steps"] if step["type"] == "perimeter"]
    quantities = [given for given in record["public_givens"] if given["kind"] == "quantity"]

    assert perimeter_steps[-1]["rule"] == expected_rule
    assert len(quantities) == (2 if family_name == "equal_circle_lens_perimeter" else 3)
    assert any(fact["type"] == "lens_perimeter_residual" and fact["value"] == "0" for fact in record["facts"])
    target_fact = next(fact for fact in record["facts"] if fact["type"] == "target")
    assert target_fact["value"] == record["synthetic"]["target_value"]


def test_lens_perimeter_rejects_wrong_radius_reference():
    spec = next(iter(select_template_families(["circle_lens_perimeter"])[0].variants()))
    trace = deepcopy(spec.trace)
    region_step = next(step for step in trace if step.get("id") == "lens_12")
    region_step["metadata"]["perimeter_formula"]["radius_segments"] = ["O1O2", "O2R2"]
    report = TraceVerifier().verify_trace(trace)
    assert not report.valid
    assert "execute.geometry_world_error" in {issue.code for issue in report.issues}


def test_lens_perimeter_can_be_admitted_without_area_formula():
    spec = next(iter(select_template_families(["circle_lens_perimeter"])[0].variants()))
    trace = deepcopy(spec.trace)
    region_step = next(step for step in trace if step.get("id") == "lens_12")
    region_step["metadata"].pop("area_formula")
    region_step["metadata"].pop("preferred_area_rule")
    state = ConstructionBuilder().execute(trace)
    facts = FactClosureSolver(profile="minimal").solve(state)
    state_round_trip = type(state).from_dict(state.to_dict())
    assert state_round_trip.checksum() == state.checksum()
    assert facts.find("perimeter", "lens_12")
    assert facts.find("target", "lens_12")
    assert not [fact for fact in facts.find("area", "lens_12") if fact.rule in {"lens_equal_radius_area", "lens_two_radius_area"}]


@pytest.mark.parametrize(
    ("family_name", "expected_rule"),
    (
        ("triangle_area_from_height", "triangle_base_height"),
        ("rectangle_area_diagonal", "rectangle_base_height"),
        ("parallelogram_area", "parallelogram_base_height"),
        ("trapezoid_area", "trapezoid_parallel_bases_height"),
        ("kite_area_diagonals", "kite_perpendicular_diagonals"),
    ),
)
def test_area_targets_derive_from_public_measurements(tmp_path, family_name, expected_rule):
    family = select_template_families([family_name])[0]
    spec = next(iter(family.variants()))
    record = SyntheticRecordExporter(solver_profile="auto").build_record(
        spec,
        output_dir=tmp_path,
        image_format="none",
        scene_payload="none",
    )

    target = record["derivation"]["target_fact"]
    area_fact = next(fact for fact in record["facts"] if fact["id"] == target["metadata"]["resolved_fact_id"])
    public_fact_ids = {
        given["source_fact_id"]
        for given in record["public_givens"]
        if given["kind"] == "quantity"
    }
    residual = next(
        fact
        for fact in record["facts"]
        if fact["type"] == "area_formula_residual" and fact["object"] == f"{area_fact['object']}|{expected_rule}"
    )

    assert area_fact["rule"] == expected_rule
    assert area_fact["metadata"]["admission"] == "coordinate_verified"
    assert set(area_fact["dependencies"]).issubset(public_fact_ids)
    assert residual["value"] == "0"
    assert expected_rule in {step["rule"] for step in record["derivation"]["steps"]}


def test_area_formula_with_nonzero_coordinate_residual_is_rejected(tmp_path):
    family = select_template_families(["trapezoid_area"])[0]
    base_spec = next(iter(family.variants()))
    trace = deepcopy(base_spec.trace)
    polygon_step = next(step for step in trace if step["op"] == "add_polygon")
    polygon_step["metadata"]["area_formula"] = {
        "rule": "rectangle_base_height",
        "base": "AB",
        "height": ["D", "H"],
    }
    target_step = next(step for step in trace if step["op"] == "set_target")
    target_step["metadata"] = {"preferred_rule": "rectangle_base_height"}
    spec = SyntheticSpec(
        id="invalid_trapezoid_rectangle_formula",
        family="fixture",
        trace=trace,
        question="Find the area.",
    )

    report = DegeneracyRejector().check(spec)
    assert not report.valid
    assert report.reason == "nonzero_verification_residual"
    assert report.metadata["facts"][0]["type"] == "area_formula_residual"
    with pytest.raises(ValueError, match="Unresolved target"):
        SyntheticRecordExporter(solver_profile="auto").build_record(
            spec,
            output_dir=tmp_path,
            image_format="none",
            scene_payload="none",
        )


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
    base = SyntheticSpec(
        id="triangle_with_interior_point",
        family="fixture",
        trace=[
            {"op": "add_point", "id": "A", "coord": ["0", "0"]},
            {"op": "add_point", "id": "B", "coord": ["6", "0"]},
            {"op": "add_point", "id": "C", "coord": ["0", "4"]},
            {"op": "add_point", "id": "D", "coord": ["3/2", "0"]},
            {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
            {"op": "add_segment", "id": "BC", "points": ["B", "C"]},
            {"op": "add_segment", "id": "CA", "points": ["C", "A"]},
            {"op": "add_polygon", "id": "tri_ABC", "vertices": ["A", "B", "C"]},
            {"op": "set_target", "type": "area", "object": "tri_ABC"},
        ],
        question="Find the area of triangle ABC.",
    )
    for operation in ("parallel_through", "perpendicular_through", "point_on_segment_ratio"):
        enhanced = FormalTraceEnhancer(seed=7, rounds_distribution={1: 1}, operation_types=(operation,)).enhance(base)[0]
        assert operation in {item["op"] for item in enhanced.trace}
        if operation in {"parallel_through", "perpendicular_through"}:
            line_index = next(index for index, item in enumerate(enhanced.trace) if item["op"] == operation)
            intersection = enhanced.trace[line_index + 1]
            assert intersection["op"] == "intersection_line_line"
            assert intersection["line1"] == enhanced.trace[line_index]["id"]
            state = ConstructionBuilder().execute(enhanced.trace)
            point = state.points[intersection["id"]]
            relation = next(item for item in state.relations if item.get("point") == point.id and item["type"] == "intersection")
            assert point.metadata["enhancement_role"] == "intended_intersection"
            assert relation["dependencies"] == [intersection["line1"], intersection["line2"]]
        assert DegeneracyRejector().check(enhanced).valid


def test_angle_bisector_enhancement_materializes_opposite_edge_intersection():
    base = SyntheticSpec(
        id="triangle_bisector_fixture",
        family="fixture",
        trace=[
            {"op": "add_point", "id": "A", "coord": ["0", "0"]},
            {"op": "add_point", "id": "B", "coord": ["6", "0"]},
            {"op": "add_point", "id": "C", "coord": ["0", "4"]},
            {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
            {"op": "add_segment", "id": "BC", "points": ["B", "C"]},
            {"op": "add_segment", "id": "CA", "points": ["C", "A"]},
            {"op": "add_polygon", "id": "tri_ABC", "vertices": ["A", "B", "C"]},
            {"op": "set_target", "type": "area", "object": "tri_ABC"},
        ],
        question="Find the area of triangle ABC.",
    )

    enhanced = FormalTraceEnhancer(seed=3, rounds_distribution={1: 1}, operation_types=("angle_bisector",)).enhance(base)[0]
    line_index = next(index for index, item in enumerate(enhanced.trace) if item["op"] == "angle_bisector")
    intersection_ops = []
    for item in enhanced.trace[line_index + 1 :]:
        if item["op"] != "intersection_line_line":
            break
        intersection_ops.append(item)

    assert intersection_ops
    assert all(item["line1"] == enhanced.trace[line_index]["id"] for item in intersection_ops)
    state = ConstructionBuilder().execute(enhanced.trace)
    assert all(item["id"] in state.points for item in intersection_ops)
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
    assert enhanced.metadata["task_graph_audit"]["dead_enhancement_steps"] == []


def test_task_dependency_graph_audits_dead_enhancement_operations():
    from geosym_world.synthesis import build_task_dependency_graph, validate_task_dependency_graph

    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": ["4", "0"]},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"], "metadata": {"show_length": True}},
        {
            "op": "point_on_segment_ratio",
            "id": "R",
            "segment": "AB",
            "ratio": "1/3",
            "metadata": {"enhancement_role": "ratio_point"},
        },
        {"op": "set_target", "type": "length", "object": "AB"},
    ]

    graph = build_task_dependency_graph(trace)

    assert validate_task_dependency_graph(graph)
    assert graph["dead_enhancement_steps"] == [3]
    assert graph["nodes"][4]["dependencies"] == [2]
    assert graph["effective_depth"] == 3


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
    assert {key: report.metadata[key] for key in ("point", "coincident_with")} == {"point": "M_AB_e1_0", "coincident_with": "O"}
    assert report.metadata["diagnostic"]["code"] == "geometry.point.coincident"


def test_enhancement_preflight_rejects_coincident_points_and_overlapping_segments():
    from geosym_world.synthesis import EnhancementPreflight

    state = ConstructionBuilder().execute([
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": ["4", "0"]},
        {"op": "add_point", "id": "M", "coord": ["2", "0"]},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
    ])
    preflight = EnhancementPreflight()

    midpoint = preflight.check(state, {"op": "midpoint", "id": "X", "of": ["A", "B"]})
    overlap = preflight.check(state, {"op": "add_segment", "id": "AM", "points": ["A", "M"]})

    assert not midpoint.valid
    assert midpoint.reason == "coincident_point"
    assert midpoint.metadata == {"point": "X", "coincident_with": "M"}
    assert not overlap.valid
    assert overlap.reason == "overlapping_segment"


def test_triangle_area_template_reuses_endpoint_when_altitude_lands_on_vertex():
    family = select_template_families(
        ["triangle_area_from_height"],
        grid_overrides={"triangle_area_from_height": {"base": (3,), "height": (2,), "skew": (0,)}},
    )[0]
    spec = next(family.variants())
    report = DegeneracyRejector().check(spec)

    assert report.valid
    assert not [operation for operation in spec.trace if operation.get("op") == "foot_of_perpendicular"]
    polygon = next(operation for operation in spec.trace if operation.get("op") == "add_polygon")
    assert polygon["metadata"]["area_formula"]["height"] == ["C", "A"]


def test_rejector_classifies_domain_execution_errors(monkeypatch):
    spec = next(iter(select_template_families(["triangle_area_from_height"])[0].variants()))
    rejector = DegeneracyRejector()

    def fail_with_unknown_reference(_state):
        raise UnknownObjectError("missing fixture object")

    monkeypatch.setattr(rejector.solver, "solve", fail_with_unknown_reference)
    report = rejector.check(spec)

    assert not report.valid
    assert report.reason == "execution_failed"
    assert report.metadata["error_type"] == "UnknownObjectError"
    assert report.metadata["diagnostic"]["code"] == "construction.reference.unknown"


def test_staged_generation_pipeline_writes_intermediate_files(tmp_path):
    output = tmp_path / "records.jsonl"
    manifest = GeometryGenerationPipeline(
        PipelineRunConfig(
            target_records=3,
            base_records=2,
            seed=9,
            image_format="none",
            scene_payload="full",
            annotation_level="points",
            relation_markers=False,
            require_exact_count=True,
            base_mutations_per_variant=0,
            enhancement_rounds={0: 1, 1: 2},
        )
    ).run(output)
    assert manifest["num_records"] == 3
    assert manifest["num_images"] == 0
    assert manifest["run_config"]["pipeline"] == "staged_formal_geosym"
    assert manifest["run_config"]["num_base_specs"] == 2
    assert manifest["run_config"]["annotation_level"] == "points"
    assert manifest["run_config"]["relation_markers"] is False
    assert sum(manifest["split_counts"].values()) == 3
    assert manifest["difficulty_audit"]["schema_versions"] == ["2.0"]
    assert manifest["difficulty_audit"]["minimum_score"] <= manifest["difficulty_audit"]["maximum_score"]
    assert (tmp_path / "json" / "base" / "records.base.jsonl").exists()
    enhanced_path = tmp_path / "json" / "enhanced" / "records.enhanced.jsonl"
    assert enhanced_path.exists()
    enhanced_lines = [json.loads(line) for line in enhanced_path.read_text(encoding="utf-8").splitlines()]
    assert len({line["metadata"]["base_spec_id"] for line in enhanced_lines[:2]}) == 2
    first_record = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert first_record["task"]["annotation"]["profile"] == {
        "schema_version": "1.0",
        "level": "points",
        "relation_markers": False,
    }


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
            seed=999,
            image_format="none",
            require_exact_count=True,
            workers=2,
        )
    ).run(resumed_output, input_enhanced_path=enhanced_path)

    assert resumed["num_records"] == 2
    assert resumed["run_config"]["pipeline"] == "resumed_enhanced_export"
    assert resumed["run_config"]["input_enhanced_path"] == str(enhanced_path)
    assert resumed["split_counts"] == initial["split_counts"]
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
    assert "triangle_partition_area" in result.families
    assert "triangle_partition_area_ratio" in result.families
    assert "rectangular_frame_area" in result.families
    assert "double_rectangular_hole_area" in result.families
    assert "double_rectangular_hole_perimeter" in result.families
    assert "sector_arc_length" in result.families
    assert "sector_perimeter" in result.families
    assert "circular_segment_area" in result.families
    assert "circular_segment_perimeter" in result.families
    assert "equal_circle_lens_area" in result.families
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
            label, policy = score_trace(spec.trace, spec.diagnostic_axes)
            return type(
                "Report",
                (),
                {
                    "valid": self.calls > 1,
                    "metadata": {"difficulty_label": label, "difficulty_policy": policy},
                },
            )()

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
    assert manifest["candidate_diagnostics"]["rejection_counts"] == {"candidate_validation_failed": 1}
    assert manifest["candidate_diagnostics"]["accepted"] == 4


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
