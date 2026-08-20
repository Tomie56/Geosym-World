"""Image + QA record exporter for synthetic geometry traces."""

from __future__ import annotations

import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable

from geosym_world.construction.builder import ConstructionBuilder
from geosym_world.construction.trace import record_construction_steps
from geosym_world.datasets.export import export_record
from geosym_world.rendering import state_to_render_scene, write_png, write_svg
from geosym_world.rendering.style_policy import DeterministicStylePolicy
from geosym_world.solver.closure import FactClosureSolver
from geosym_world.verifier.trace import TraceVerifier

from .templates import SyntheticSpec
from .splits import assign_split_labels
from .task_contract import build_task_contract, validate_task_contract


class SyntheticRecordExporter:
    def __init__(self, *, solver_profile: str = "auto"):
        self.verifier = TraceVerifier()
        self.solver = FactClosureSolver(profile=solver_profile)

    def build_record(
        self,
        spec: SyntheticSpec,
        *,
        output_dir: Path,
        image_format: str = "both",
        style_seed: int = 0,
        scene_payload: str = "full",
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
        record = export_record(state, facts, record_id=spec.id, question=spec.question, answer=spec.answer_hint, verification=verification)
        if record["derivation"]["target_fact"] is None:
            unresolved = [fact.to_dict() for fact in facts.find("target_unresolved", state.target.object if state.target else "")]
            raise ValueError(f"Unresolved target for {spec.id}: {unresolved}")
        if not record["derivation"]["integrity"]["valid"] or not record["derivation"]["validation"]["valid"]:
            raise ValueError(
                f"Invalid target derivation for {spec.id}: "
                f"integrity={record['derivation']['integrity']}, validation={record['derivation']['validation']}"
            )
        image_dir = output_dir / "images"
        image_payload: Dict[str, Any] = {"scene": None, "formats": {}}
        scene = None
        if image_format != "none" or scene_payload != "none":
            scene = state_to_render_scene(state, scene_id=spec.id, style_policy=DeterministicStylePolicy(seed=style_seed))
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
            "difficulty": spec.difficulty,
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
            difficulty=spec.difficulty,
            diagnostic_axes=spec.diagnostic_axes,
        )
        record["task"]["validation"] = validate_task_contract(record["task"]).to_dict()
        if not record["task"]["validation"]["valid"]:
            raise ValueError(f"Invalid task contract for {spec.id}: {record['task']['validation']}")
        record["public_givens"] = record["task"]["public_givens"]
        record["construction_trace"] = spec.trace
        record["construction_steps"] = record_construction_steps(spec.trace)
        record["formal_state_checksum"] = state.checksum()
        record["render_quality"] = scene.metadata.get("render_quality") if scene is not None else None
        record["image"] = image_payload
        record["qa"] = _qa_payload(record)
        record["gt"] = _gt_payload(record)
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
        qa_output_path.parent.mkdir(parents=True, exist_ok=True)
        gt_output_path.parent.mkdir(parents=True, exist_ok=True)
        workers = max(1, int(workers))

        def build(item: tuple[int, SyntheticSpec]) -> Dict[str, Any]:
            index, spec = item
            return self.build_record(
                spec,
                output_dir=output_dir,
                image_format=image_format,
                style_seed=style_seed + index,
                scene_payload=scene_payload,
            )

        indexed_specs = enumerate(spec_list)
        executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        records = executor.map(build, indexed_specs) if executor else map(build, indexed_specs)
        stats = _empty_manifest_stats()
        try:
            with (
                output_path.open("w", encoding="utf-8") as record_file,
                qa_output_path.open("w", encoding="utf-8") as qa_file,
                gt_output_path.open("w", encoding="utf-8") as gt_file,
            ):
                for index, record in enumerate(records, start=1):
                    _write_jsonl_row(record_file, record)
                    _write_jsonl_row(qa_file, record["qa"])
                    _write_jsonl_row(gt_file, record["gt"])
                    _update_manifest_stats(stats, record)
                    if progress_every > 0 and (index % progress_every == 0 or index == len(spec_list)):
                        print(f"[geosym] export progress: {index}/{len(spec_list)} records", file=sys.stderr, flush=True)
        finally:
            if executor:
                executor.shutdown(wait=True, cancel_futures=True)
        manifest = _manifest(output_path, stats, qa_path=qa_output_path, gt_path=gt_output_path, run_config=config)
        manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest


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
        "image_counts": {},
        "enhancement_round_counts": {},
        "enhancement_operation_counts": {},
        "split_counts": {},
        "split_groups": {},
        "split_group_modes": set(),
        "lineage_roots": [],
        "record_ids": [],
    }


def _update_manifest_stats(stats: Dict[str, Any], record: Dict[str, Any]) -> None:
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
    run_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    image_counts = stats["image_counts"]
    config = run_config or {}
    return {
        "path": str(path),
        "qa_path": str(qa_path),
        "gt_path": str(gt_path),
        "num_records": stats["num_records"],
        "num_valid": stats["num_valid"],
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
