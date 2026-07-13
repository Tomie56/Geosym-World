"""Image + QA record exporter for synthetic geometry traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

from geosym_world_geometry.builder import ConstructionBuilder
from geosym_world_geometry.export import export_record
from geosym_world_geometry.rendering import state_to_render_scene, write_png, write_svg
from geosym_world_geometry.rendering.style_policy import DeterministicStylePolicy
from geosym_world_geometry.solver import FactClosureSolver
from geosym_world_geometry.verifier import TraceVerifier

from .templates import SyntheticSpec


class SyntheticRecordExporter:
    def __init__(self):
        self.verifier = TraceVerifier()
        self.solver = FactClosureSolver()

    def build_record(self, spec: SyntheticSpec, *, output_dir: Path, image_format: str = "both", style_seed: int = 0) -> Dict[str, Any]:
        if image_format not in {"none", "svg", "png", "both"}:
            raise ValueError(f"Unsupported image format: {image_format}")
        verification = self.verifier.verify_trace(spec.trace)
        state = ConstructionBuilder().execute(spec.trace)
        facts = self.solver.solve(state)
        record = export_record(state, facts, record_id=spec.id, question=spec.question, answer=spec.answer_hint, verification=verification)
        scene = state_to_render_scene(state, scene_id=spec.id, style_policy=DeterministicStylePolicy(seed=style_seed))
        image_dir = output_dir / "images"
        image_payload: Dict[str, Any] = {"scene": scene.to_dict(), "formats": {}}
        if image_format in {"svg", "both"}:
            svg_path = image_dir / f"{spec.id}.svg"
            write_svg(scene, svg_path)
            image_payload["formats"]["svg"] = str(svg_path)
        if image_format in {"png", "both"}:
            png_path = image_dir / f"{spec.id}.png"
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
        }
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
    ) -> Dict[str, Any]:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_dir = output_path.parent
        spec_list = list(specs)
        if require_count is not None and len(spec_list) != require_count:
            raise ValueError(f"Expected {require_count} accepted specs, got {len(spec_list)}")
        records = [self.build_record(spec, output_dir=output_dir, image_format=image_format, style_seed=style_seed + idx) for idx, spec in enumerate(spec_list)]
        with output_path.open("w", encoding="utf-8") as f:
            for record in records:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")
        qa_output_path = Path(qa_path) if qa_path else output_path.with_suffix(output_path.suffix + ".qa.jsonl")
        gt_output_path = Path(gt_path) if gt_path else output_path.with_suffix(output_path.suffix + ".gt.jsonl")
        _write_sidecar_jsonl(qa_output_path, (record["qa"] for record in records))
        _write_sidecar_jsonl(gt_output_path, (record["gt"] for record in records))
        manifest = _manifest(output_path, records, qa_path=qa_output_path, gt_path=gt_output_path, run_config=run_config)
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
        "diagnostic_axes": record["synthetic"]["diagnostic_axes"],
        "verification": record["verification"],
    }


def _write_sidecar_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


def _manifest(
    path: Path,
    records: list[Dict[str, Any]],
    *,
    qa_path: Path,
    gt_path: Path,
    run_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    family_counts: Dict[str, int] = {}
    difficulty_counts: Dict[str, int] = {}
    image_counts: Dict[str, int] = {}
    for record in records:
        family = record["synthetic"]["family"]
        difficulty = record["synthetic"]["difficulty"]
        family_counts[family] = family_counts.get(family, 0) + 1
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        for image_format in record["image"]["formats"]:
            image_counts[image_format] = image_counts.get(image_format, 0) + 1
    return {
        "path": str(path),
        "qa_path": str(qa_path),
        "gt_path": str(gt_path),
        "num_records": len(records),
        "num_valid": sum(1 for record in records if record["verification"]["valid"]),
        "num_images": sum(image_counts.values()),
        "image_formats": dict(sorted(image_counts.items())),
        "missing_target_values": sum(1 for record in records if record["synthetic"]["target_value"] is None),
        "families": dict(sorted(family_counts.items())),
        "difficulties": dict(sorted(difficulty_counts.items())),
        "record_ids": [record["id"] for record in records],
        "run_config": run_config or {},
    }
