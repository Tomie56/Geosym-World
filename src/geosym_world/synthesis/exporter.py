"""Image + QA record exporter for synthetic geometry traces."""

from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, Iterable

from geosym_world.benchmark.parser_dataset import build_parser_benchmark_pair
from geosym_world.benchmark.parse_graph_reference import build_synthetic_typed_parser_reference
from geosym_world.benchmark.parser_sources import (
    ParserProtocol,
    ParserSourceManifest,
    hash_asset,
    materialize_parser_input,
)
from geosym_world.construction.builder import ConstructionBuilder
from geosym_world.construction.trace import record_construction_steps
from geosym_world.datasets.export import export_record
from geosym_world.rendering import AnnotationPolicy, RenderScene, state_to_render_scene, write_png, write_svg
from geosym_world.rendering.style_policy import DeterministicStylePolicy
from geosym_world.schema.solutions import build_solution_paths
from geosym_world.schema.canonical import stable_checksum
from geosym_world.schema.state import FormalGeometryState
from geosym_world.solver.closure import FactClosureSolver
from geosym_world.verifier.trace import TraceVerifier

from .difficulty import score_task
from .templates import SyntheticSpec
from .splits import assign_split_labels
from .task_contract import build_task_contract, validate_task_contract


class SyntheticRecordExporter:
    def __init__(
        self,
        *,
        solver_profile: str = "auto",
        annotation_level: str = "givens",
        relation_markers: bool = True,
    ):
        self.verifier = TraceVerifier()
        self.solver = FactClosureSolver(profile=solver_profile)
        self.annotation_policy = AnnotationPolicy(level=annotation_level, relation_markers=relation_markers)

    def build_record(
        self,
        spec: SyntheticSpec,
        *,
        output_dir: Path,
        image_format: str = "both",
        style_seed: int = 0,
        scene_payload: str = "full",
        parser_benchmark: bool = False,
    ) -> Dict[str, Any]:
        if image_format not in {"none", "svg", "png", "both"}:
            raise ValueError(f"Unsupported image format: {image_format}")
        if scene_payload not in {"full", "metadata", "none"}:
            raise ValueError(f"Unsupported scene payload: {scene_payload}")
        verification = self.verifier.verify_trace(spec.trace, execute=False)
        if not verification.valid:
            raise ValueError(f"Invalid construction trace for {spec.id}: {verification.to_dict()}")
        state = ConstructionBuilder().execute(spec.trace)
        self.verifier.verify_state(state, verification)
        if not verification.valid:
            raise ValueError(f"Invalid formal state for {spec.id}: {verification.to_dict()}")
        facts = self.solver.solve(state)
        if state.target is None:
            raise ValueError(f"Missing target for {spec.id}")
        record = export_record(state, facts, record_id=spec.id, question=spec.question, answer=spec.answer_hint, verification=verification)
        if record["derivation"]["target_fact"] is None:
            unresolved = [fact.to_dict() for fact in facts.find("target_unresolved", state.target.object if state.target else "")]
            raise ValueError(f"Unresolved target for {spec.id}: {unresolved}")
        if not record["derivation"]["integrity"]["valid"] or not record["derivation"]["validation"]["valid"]:
            raise ValueError(
                f"Invalid target derivation for {spec.id}: "
                f"integrity={record['derivation']['integrity']}, validation={record['derivation']['validation']}"
            )
        solution_paths = build_solution_paths(facts, state.target)
        if not solution_paths["validation"]["valid"]:
            raise ValueError(f"Invalid solution paths for {spec.id}: {solution_paths['validation']}")
        difficulty_label, difficulty_policy = score_task(
            spec.trace,
            spec.diagnostic_axes,
            derivation=record["derivation"],
            solution_paths=solution_paths,
        )
        image_dir = output_dir / "images"
        image_payload: Dict[str, Any] = {"scene": None, "formats": {}}
        scene = None
        if image_format != "none" or scene_payload != "none":
            scene = state_to_render_scene(
                state,
                scene_id=spec.id,
                style_policy=DeterministicStylePolicy(seed=style_seed),
                annotation_policy=self.annotation_policy,
            )
        if scene_payload == "full" and scene is not None:
            image_payload["scene"] = scene.to_dict()
        elif scene_payload == "metadata" and scene is not None:
            image_payload["scene"] = {"id": scene.id, "metadata": scene.metadata}
        if image_format in {"svg", "both"}:
            svg_path = image_dir / f"{spec.id}.svg"
            assert scene is not None
            write_svg(scene, svg_path)
            image_payload["formats"]["svg"] = str(svg_path)
        if image_format in {"png", "both"}:
            png_path = image_dir / f"{spec.id}.png"
            assert scene is not None
            write_png(scene, png_path)
            image_payload["formats"]["png"] = str(png_path)
        target = record["derivation"]["target_fact"]
        answer = spec.answer_hint or (target["value"] if target else "")
        record["answer"] = answer
        record["synthetic"] = {
            "family": spec.family,
            "difficulty": difficulty_label,
            "difficulty_policy": difficulty_policy,
            "diagnostic_axes": list(spec.diagnostic_axes),
            "target_value": target["value"] if target else None,
            "metadata": spec.metadata,
            "lineage": {
                "root": str(spec.metadata.get("lineage_root") or spec.metadata.get("base_spec_id") or spec.id),
                "role": str(spec.metadata.get("lineage_role", "base")),
                "base_spec_id": spec.metadata.get("base_spec_id"),
                "mutated_from": spec.metadata.get("mutated_from"),
                "enhancement_rounds": spec.metadata.get("enhancement_rounds", 0),
                "enhancement_operations": list(spec.metadata.get("enhancement_operations", ())),
            },
            "split": spec.metadata.get("split"),
        }
        record["task"] = build_task_contract(
            state,
            facts,
            record["derivation"],
            family=spec.family,
            difficulty=difficulty_label,
            diagnostic_axes=spec.diagnostic_axes,
            solution_paths=solution_paths,
            annotation_policy=self.annotation_policy,
        )
        record["task"]["validation"] = validate_task_contract(record["task"]).to_dict()
        if not record["task"]["validation"]["valid"]:
            raise ValueError(f"Invalid task contract for {spec.id}: {record['task']['validation']}")
        record["public_givens"] = record["task"]["public_givens"]
        record["construction_trace"] = spec.trace
        record["solution_paths"] = solution_paths
        record["construction_steps"] = record_construction_steps(spec.trace)
        record["formal_state_checksum"] = state.checksum()
        record["render_quality"] = scene.metadata.get("render_quality") if scene is not None else None
        record["image"] = image_payload
        record["qa"] = _qa_payload(record)
        record["gt"] = _gt_payload(record)
        if parser_benchmark:
            if scene is None:
                raise ValueError(f"Parser benchmark export requires a rendered image for {spec.id}")
            skip_reason = _parser_benchmark_skip_reason(scene)
            if skip_reason is None:
                record["_parser_benchmark"] = _build_parser_benchmark_bundle(
                    record,
                    state=state,
                    scene=scene,
                    output_dir=output_dir,
                )
            else:
                record["_parser_benchmark_skip_reason"] = skip_reason
        return record

    def write_jsonl(
        self,
        specs: Iterable[SyntheticSpec],
        path: str | Path,
        *,
        image_format: str = "both",
        style_seed: int = 0,
        qa_path: str | Path | None = None,
        gt_path: str | Path | None = None,
        parser_input_path: str | Path | None = None,
        parser_reference_path: str | Path | None = None,
        typed_parser_reference_path: str | Path | None = None,
        parser_source_audit_path: str | Path | None = None,
        require_count: int | None = None,
        run_config: Dict[str, Any] | None = None,
        workers: int = 1,
        scene_payload: str = "full",
        progress_every: int = 0,
    ) -> Dict[str, Any]:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_dir = output_path.parent
        spec_list = list(specs)
        config = run_config or {}
        spec_list = assign_split_labels(
            spec_list,
            seed=int(config.get("seed", 0)),
            ratios=config.get("split_ratios"),
            group_mode=str(config.get("split_group_mode", "lineage")),
        )
        duplicate_ids = sorted(spec_id for spec_id, count in Counter(spec.id for spec in spec_list).items() if count > 1)
        if duplicate_ids:
            preview = ", ".join(duplicate_ids[:5])
            raise ValueError(f"Duplicate synthetic record IDs would overwrite outputs: {preview}")
        if require_count is not None and len(spec_list) != require_count:
            raise ValueError(f"Expected {require_count} accepted specs, got {len(spec_list)}")
        qa_output_path = Path(qa_path) if qa_path else output_path.with_suffix(output_path.suffix + ".qa.jsonl")
        gt_output_path = Path(gt_path) if gt_path else output_path.with_suffix(output_path.suffix + ".gt.jsonl")
        parser_input_output_path = Path(parser_input_path) if parser_input_path else output_path.with_suffix(output_path.suffix + ".parser-input.jsonl")
        parser_reference_output_path = Path(parser_reference_path) if parser_reference_path else output_path.with_suffix(output_path.suffix + ".parser-reference.jsonl")
        typed_parser_reference_output_path = (
            Path(typed_parser_reference_path)
            if typed_parser_reference_path
            else output_path.with_suffix(output_path.suffix + ".typed-parser-reference.jsonl")
        )
        parser_source_audit_output_path = (
            Path(parser_source_audit_path)
            if parser_source_audit_path
            else output_path.with_suffix(output_path.suffix + ".parser-source-audit.jsonl")
        )
        qa_output_path.parent.mkdir(parents=True, exist_ok=True)
        gt_output_path.parent.mkdir(parents=True, exist_ok=True)
        parser_benchmark_enabled = image_format != "none"
        if parser_benchmark_enabled:
            parser_input_output_path.parent.mkdir(parents=True, exist_ok=True)
            parser_reference_output_path.parent.mkdir(parents=True, exist_ok=True)
            typed_parser_reference_output_path.parent.mkdir(parents=True, exist_ok=True)
            parser_source_audit_output_path.parent.mkdir(parents=True, exist_ok=True)
        workers = max(1, int(workers))

        def build(item: tuple[int, SyntheticSpec]) -> Dict[str, Any]:
            index, spec = item
            return self.build_record(
                spec,
                output_dir=output_dir,
                image_format=image_format,
                style_seed=style_seed + index,
                scene_payload=scene_payload,
                parser_benchmark=parser_benchmark_enabled,
            )

        indexed_specs = enumerate(spec_list)
        executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        records = executor.map(build, indexed_specs) if executor else map(build, indexed_specs)
        stats = _empty_manifest_stats()
        try:
            with ExitStack() as stack:
                record_file = stack.enter_context(output_path.open("w", encoding="utf-8"))
                qa_file = stack.enter_context(qa_output_path.open("w", encoding="utf-8"))
                gt_file = stack.enter_context(gt_output_path.open("w", encoding="utf-8"))
                parser_input_file = stack.enter_context(parser_input_output_path.open("w", encoding="utf-8")) if parser_benchmark_enabled else None
                parser_reference_file = stack.enter_context(parser_reference_output_path.open("w", encoding="utf-8")) if parser_benchmark_enabled else None
                typed_parser_reference_file = (
                    stack.enter_context(typed_parser_reference_output_path.open("w", encoding="utf-8"))
                    if parser_benchmark_enabled
                    else None
                )
                parser_source_audit_file = (
                    stack.enter_context(parser_source_audit_output_path.open("w", encoding="utf-8"))
                    if parser_benchmark_enabled
                    else None
                )
                for index, record in enumerate(records, start=1):
                    parser_benchmark = record.pop("_parser_benchmark", None)
                    parser_benchmark_skip_reason = record.pop("_parser_benchmark_skip_reason", None)
                    _write_jsonl_row(record_file, record)
                    _write_jsonl_row(qa_file, record["qa"])
                    _write_jsonl_row(gt_file, record["gt"])
                    if parser_benchmark is not None:
                        assert (
                            parser_input_file is not None
                            and parser_reference_file is not None
                            and typed_parser_reference_file is not None
                            and parser_source_audit_file is not None
                        )
                        _write_jsonl_row(parser_input_file, parser_benchmark["input"])
                        _write_jsonl_row(parser_reference_file, parser_benchmark["reference"])
                        _write_jsonl_row(typed_parser_reference_file, parser_benchmark["typed_reference"])
                        _write_jsonl_row(parser_source_audit_file, parser_benchmark["source_audit"])
                    _update_manifest_stats(
                        stats,
                        record,
                        parser_benchmark_exported=parser_benchmark is not None,
                        parser_benchmark_skip_reason=parser_benchmark_skip_reason,
                    )
                    if progress_every > 0 and (index % progress_every == 0 or index == len(spec_list)):
                        print(f"[geosym] export progress: {index}/{len(spec_list)} records", file=sys.stderr, flush=True)
        finally:
            if executor:
                executor.shutdown(wait=True, cancel_futures=True)
        manifest = _manifest(
            output_path,
            stats,
            qa_path=qa_output_path,
            gt_path=gt_output_path,
            parser_input_path=parser_input_output_path if parser_benchmark_enabled else None,
            parser_reference_path=parser_reference_output_path if parser_benchmark_enabled else None,
            typed_parser_reference_path=typed_parser_reference_output_path if parser_benchmark_enabled else None,
            parser_source_audit_path=parser_source_audit_output_path if parser_benchmark_enabled else None,
            run_config=config,
        )
        manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest


def _build_parser_benchmark_bundle(
    record: Dict[str, Any],
    *,
    state: FormalGeometryState,
    scene: RenderScene,
    output_dir: Path,
) -> Dict[str, Any]:
    temporary_path: Path | None = None
    source_image = (record.get("image") or {}).get("formats", {}).get("png")
    if source_image is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.NamedTemporaryFile(prefix=".parser-source-", suffix=".png", dir=output_dir, delete=False)
        temporary.close()
        temporary_path = Path(temporary.name)
        write_png(scene, temporary_path)
        source_image = str(temporary_path)
    try:
        question = str(record.get("question", ""))
        protocol = ParserProtocol.IMAGE_QUESTION if question else ParserProtocol.IMAGE_ONLY
        lineage = (record.get("synthetic") or {}).get("lineage") or {}
        source_manifest = ParserSourceManifest(
            source_id=f"synthetic:{record['id']}",
            protocol=protocol,
            source_version=str((record.get("task") or {}).get("checksum") or state.checksum()),
            license_id="geosym-world-synthetic",
            asset_sha256=hash_asset(source_image),
            split_group=str(lineage.get("root") or record["id"]),
            public_fields=("image", "question") if question else ("image",),
            metadata={
                "generator": "geosym-world",
                "formal_state_checksum": state.checksum(),
                "scene_id": scene.id,
            },
        )
        parser_input_object, materialization = materialize_parser_input(
            source_image,
            manifest=source_manifest,
            output_dir=output_dir,
            question=question,
        )
        parser_input, parser_reference = build_parser_benchmark_pair(
            record,
            state=state,
            scene=scene,
            parser_input=parser_input_object,
        )
        typed_reference = build_synthetic_typed_parser_reference(
            record,
            parser_input_object,
            state=state,
            scene=scene,
            manifest=source_manifest,
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    source_audit = {
        "schema_version": "geosym.synthetic_parser_source_audit.v1",
        "public_id": parser_input_object.id,
        "source_record_id": record["id"],
        "source_images": dict((record.get("image") or {}).get("formats") or {}),
        "source_manifest": source_manifest.to_dict(),
        "materialization": materialization,
    }
    source_audit["checksum"] = stable_checksum(source_audit)
    return {
        "input": parser_input,
        "reference": parser_reference,
        "typed_reference": typed_reference,
        "source_audit": source_audit,
    }


def _parser_benchmark_skip_reason(scene: RenderScene) -> str | None:
    render_dim = scene.metadata.get("render_dim")
    if render_dim != 2:
        return f"unsupported_render_dim:{render_dim}"
    return None


def _qa_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": record["id"],
        "image": record["image"]["formats"],
        "question": record["question"],
        "answer": record["answer"],
        "family": record["synthetic"]["family"],
        "difficulty": record["synthetic"]["difficulty"],
    }


def _gt_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    target = record["formal_state"].get("target") or {}
    target_fact = record["derivation"]["target_fact"]
    return {
        "id": record["id"],
        "target": {
            "type": target.get("type"),
            "object": target.get("object"),
            "value": target_fact["value"] if target_fact else None,
            "numeric_value": target_fact["numeric_value"] if target_fact else None,
            "rule": target_fact["rule"] if target_fact else None,
        },
        "derivation_fact_ids": [fact["id"] for fact in record["derivation"]["facts"]],
        "solution_paths": record["solution_paths"],
        "derivation": {
            "schema_version": record["derivation"]["schema_version"],
            "checksum": record["derivation"]["checksum"],
            "steps": record["derivation"]["steps"],
            "integrity": record["derivation"]["integrity"],
            "validation": record["derivation"]["validation"],
        },
        "task": record["task"],
        "diagnostic_axes": record["synthetic"]["diagnostic_axes"],
        "verification": record["verification"],
    }


def _write_jsonl_row(file, row: Dict[str, Any]) -> None:
    json.dump(row, file, ensure_ascii=False)
    file.write("\n")


def _empty_manifest_stats() -> Dict[str, Any]:
    return {
        "num_records": 0,
        "num_valid": 0,
        "missing_target_values": 0,
        "invalid_derivations": 0,
        "derivation_depth_counts": {},
        "derivation_rule_counts": {},
        "target_type_counts": {},
        "construction_depth_counts": {},
        "structure_signatures": set(),
        "family_counts": {},
        "difficulty_counts": {},
        "difficulty_scores": [],
        "difficulty_schema_versions": set(),
        "image_counts": {},
        "enhancement_round_counts": {},
        "enhancement_operation_counts": {},
        "split_counts": {},
        "split_groups": {},
        "split_group_modes": set(),
        "lineage_roots": [],
        "record_ids": [],
        "parser_benchmark_records": 0,
        "parser_benchmark_skip_reasons": {},
    }


def _update_manifest_stats(
    stats: Dict[str, Any],
    record: Dict[str, Any],
    *,
    parser_benchmark_exported: bool,
    parser_benchmark_skip_reason: str | None,
) -> None:
    stats["num_records"] += 1
    stats["num_valid"] += int(record["verification"]["valid"])
    stats["missing_target_values"] += int(record["synthetic"]["target_value"] is None)
    derivation = record["derivation"]
    stats["invalid_derivations"] += int(not derivation["integrity"]["valid"])
    depth_key = str(derivation["integrity"]["max_depth"])
    stats["derivation_depth_counts"][depth_key] = stats["derivation_depth_counts"].get(depth_key, 0) + 1
    for step in derivation["steps"]:
        rule = step["rule"] or "unspecified"
        stats["derivation_rule_counts"][rule] = stats["derivation_rule_counts"].get(rule, 0) + 1
    task = record["task"]
    target_type = str((task.get("target") or {}).get("type", "missing"))
    stats["target_type_counts"][target_type] = stats["target_type_counts"].get(target_type, 0) + 1
    construction_depth = str(task["construction"]["num_steps"])
    stats["construction_depth_counts"][construction_depth] = stats["construction_depth_counts"].get(construction_depth, 0) + 1
    stats["structure_signatures"].add(task["structure_signature"])
    stats["record_ids"].append(record["id"])
    stats["parser_benchmark_records"] += int(parser_benchmark_exported)
    if parser_benchmark_skip_reason is not None:
        skipped = stats["parser_benchmark_skip_reasons"]
        skipped[parser_benchmark_skip_reason] = skipped.get(parser_benchmark_skip_reason, 0) + 1
    lineage = record["synthetic"].get("lineage", {})
    if lineage.get("root") is not None:
        stats["lineage_roots"].append(lineage["root"])
    split = record["synthetic"].get("split")
    if split is not None:
        stats["split_counts"][split] = stats["split_counts"].get(split, 0) + 1
    metadata = record["synthetic"].get("metadata", {})
    split_group_key = metadata.get("split_group_key")
    if split_group_key is not None and split is not None:
        previous = stats["split_groups"].setdefault(str(split_group_key), str(split))
        if previous != split:
            raise ValueError(f"Split group {split_group_key!r} appears in both {previous} and {split}")
    split_group_mode = metadata.get("split_group_mode")
    if split_group_mode is not None:
        stats["split_group_modes"].add(str(split_group_mode))
    for key, value in (("family_counts", record["synthetic"]["family"]), ("difficulty_counts", record["synthetic"]["difficulty"])):
        stats[key][value] = stats[key].get(value, 0) + 1
    difficulty_policy = record["synthetic"]["difficulty_policy"]
    stats["difficulty_scores"].append(int(difficulty_policy["score"]))
    stats["difficulty_schema_versions"].add(str(difficulty_policy["schema_version"]))
    for image_format in record["image"]["formats"]:
        stats["image_counts"][image_format] = stats["image_counts"].get(image_format, 0) + 1
    rounds = metadata.get("enhancement_rounds")
    if rounds is not None:
        key = str(rounds)
        stats["enhancement_round_counts"][key] = stats["enhancement_round_counts"].get(key, 0) + 1
    for operation in metadata.get("enhancement_operations", []):
        stats["enhancement_operation_counts"][operation] = stats["enhancement_operation_counts"].get(operation, 0) + 1


def _manifest(
    path: Path,
    stats: Dict[str, Any],
    *,
    qa_path: Path,
    gt_path: Path,
    parser_input_path: Path | None,
    parser_reference_path: Path | None,
    typed_parser_reference_path: Path | None,
    parser_source_audit_path: Path | None,
    run_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    image_counts = stats["image_counts"]
    config = run_config or {}
    difficulty_scores = stats["difficulty_scores"]
    return {
        "path": str(path),
        "qa_path": str(qa_path),
        "gt_path": str(gt_path),
        "parser_input_path": str(parser_input_path) if parser_input_path is not None else None,
        "parser_reference_path": str(parser_reference_path) if parser_reference_path is not None else None,
        "typed_parser_reference_path": str(typed_parser_reference_path) if typed_parser_reference_path is not None else None,
        "parser_source_audit_path": str(parser_source_audit_path) if parser_source_audit_path is not None else None,
        "num_records": stats["num_records"],
        "num_valid": stats["num_valid"],
        "num_parser_benchmark_records": stats["parser_benchmark_records"],
        "parser_benchmark_skip_reasons": dict(sorted(stats["parser_benchmark_skip_reasons"].items())),
        "num_images": sum(image_counts.values()),
        "image_formats": dict(sorted(image_counts.items())),
        "missing_target_values": stats["missing_target_values"],
        "invalid_derivations": stats["invalid_derivations"],
        "derivation_depths": dict(sorted(stats["derivation_depth_counts"].items(), key=lambda item: int(item[0]))),
        "derivation_rules": dict(sorted(stats["derivation_rule_counts"].items())),
        "target_types": dict(sorted(stats["target_type_counts"].items())),
        "construction_depths": dict(sorted(stats["construction_depth_counts"].items(), key=lambda item: int(item[0]))),
        "num_structure_signatures": len(stats["structure_signatures"]),
        "families": dict(sorted(stats["family_counts"].items())),
        "difficulties": dict(sorted(stats["difficulty_counts"].items())),
        "difficulty_audit": {
            "schema_versions": sorted(stats["difficulty_schema_versions"]),
            "minimum_score": min(difficulty_scores) if difficulty_scores else None,
            "maximum_score": max(difficulty_scores) if difficulty_scores else None,
            "mean_score": sum(difficulty_scores) / len(difficulty_scores) if difficulty_scores else None,
        },
        "enhancement_rounds": dict(sorted(stats["enhancement_round_counts"].items())),
        "enhancement_operations": dict(sorted(stats["enhancement_operation_counts"].items())),
        "split_counts": dict(sorted(stats["split_counts"].items())),
        "num_split_groups": len(stats["split_groups"]),
        "split_group_modes": sorted(stats["split_group_modes"]),
        "num_lineage_roots": len(set(stats["lineage_roots"])),
        "record_ids": stats["record_ids"],
        "candidate_diagnostics": config.get("candidate_diagnostics", {}),
        "run_config": config,
    }
