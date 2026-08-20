"""Staged synthetic generation pipeline inspired by the original GeoSym flow."""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable

from .enhancer import DEFAULT_ENHANCEMENT_OPERATIONS, FormalTraceEnhancer
from .difficulty import difficulty_in_range
from .diagnostics import CandidateDiagnostics, accepted_validation_metadata
from .exporter import SyntheticRecordExporter
from .filters import DegeneracyRejector, DuplicateFilter
from .sampler import SyntheticSampler
from .splits import assign_split_labels
from .templates import SyntheticSpec, load_grid_overrides, select_template_families


@dataclass(frozen=True)
class PipelineRunConfig:
    target_records: int = 20
    base_records: int | None = None
    base_reserve_records: int | None = None
    seed: int = 0
    image_format: str = "both"
    require_exact_count: bool = False
    families: tuple[str, ...] = ()
    per_family_limit: int | None = None
    base_mutations_per_variant: int = 1
    enhancement_rounds: Dict[int, int] | None = None
    enhancement_operations: tuple[str, ...] = DEFAULT_ENHANCEMENT_OPERATIONS
    max_enhancements_per_base: int | None = None
    grid_json: str | None = None
    grid_json_file: str | None = None
    minimum_difficulty: str | None = None
    maximum_difficulty: str | None = None
    solver_profile: str = "auto"
    workers: int = 1
    scene_payload: str = "metadata"
    progress_every: int = 25
    split_ratios: Dict[str, float] | None = None
    split_group_mode: str = "topology"


class GeometryGenerationPipeline:
    """Run base generation, enhancement, filtering, render export, QA, and GT."""

    def __init__(self, config: PipelineRunConfig):
        self.config = config
        self.rejector = DegeneracyRejector(solver_profile=config.solver_profile)
        self.duplicates = DuplicateFilter()

    def run(
        self,
        output_path: str | Path,
        *,
        input_enhanced_path: str | Path | None = None,
        base_path: str | Path | None = None,
        enhanced_path: str | Path | None = None,
        qa_path: str | Path | None = None,
        gt_path: str | Path | None = None,
    ) -> Dict[str, Any]:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if input_enhanced_path is not None:
            return self._export_existing_enhanced(
                Path(input_enhanced_path),
                output,
                qa_path=qa_path,
                gt_path=gt_path,
            )
        base_output = Path(base_path) if base_path else output.parent / "json" / "base" / f"{output.stem}.base.jsonl"
        enhanced_output = Path(enhanced_path) if enhanced_path else output.parent / "json" / "enhanced" / f"{output.stem}.enhanced.jsonl"

        self._progress(f"sampling base specs (target={self.config.base_records or _default_base_records(self.config.target_records)}, reserve={max(0, self.config.base_reserve_records or 0)})")
        base_specs = self._generate_base_specs()
        _write_specs(base_output, base_specs)
        self._progress(f"base specs ready: {len(base_specs)} -> {base_output}")

        self._progress(f"enhancing and validating candidates (target={self.config.target_records})")
        enhanced_specs, diagnostics = self._generate_enhanced_specs(base_specs)
        if self.config.require_exact_count and len(enhanced_specs) != self.config.target_records:
            raise ValueError(f"Expected {self.config.target_records} enhanced specs, got {len(enhanced_specs)}")
        enhanced_specs = assign_split_labels(
            enhanced_specs,
            seed=self.config.seed,
            ratios=self.config.split_ratios,
            group_mode=self.config.split_group_mode,
        )
        _write_specs(enhanced_output, enhanced_specs)
        self._progress(f"enhanced specs ready: {len(enhanced_specs)} accepted, {diagnostics.rejected} rejected -> {enhanced_output}")

        run_config = {
            **asdict(self.config),
            "families": list(self.config.families) or "all",
            "enhancement_operations": list(self.config.enhancement_operations),
            "base_path": str(base_output),
            "enhanced_path": str(enhanced_output),
            "num_base_specs": len(base_specs),
            "num_enhanced_specs": len(enhanced_specs),
            "num_checked_enhancements": diagnostics.checked,
            "num_rejected_enhancements": diagnostics.rejected,
            "num_skipped_enhancements": diagnostics.skipped,
            "candidate_diagnostics": diagnostics.to_dict(),
            "pipeline": "staged_formal_geosym",
        }
        self._progress(f"exporting {len(enhanced_specs)} records ({self.config.image_format}) with {max(1, self.config.workers)} worker(s)")
        manifest = SyntheticRecordExporter(solver_profile=self.config.solver_profile).write_jsonl(
            enhanced_specs,
            output,
            image_format=self.config.image_format,
            style_seed=self.config.seed,
            qa_path=qa_path,
            gt_path=gt_path,
            require_count=self.config.target_records if self.config.require_exact_count else None,
            run_config=run_config,
            workers=self.config.workers,
            scene_payload=self.config.scene_payload,
            progress_every=self.config.progress_every,
        )
        self._progress(f"generation complete: {manifest['num_records']} records -> {output}")
        return manifest

    def _export_existing_enhanced(
        self,
        input_path: Path,
        output_path: Path,
        *,
        qa_path: str | Path | None,
        gt_path: str | Path | None,
    ) -> Dict[str, Any]:
        specs = _read_specs(input_path)
        if self.config.require_exact_count and len(specs) != self.config.target_records:
            raise ValueError(f"Expected {self.config.target_records} enhanced specs, got {len(specs)} from {input_path}")
        specs = assign_split_labels(
            specs,
            seed=self.config.seed,
            ratios=self.config.split_ratios,
            group_mode=self.config.split_group_mode,
        )
        run_config = {
            **asdict(self.config),
            "families": list(self.config.families) or "from_input_enhanced",
            "enhancement_operations": list(self.config.enhancement_operations),
            "input_enhanced_path": str(input_path),
            "num_enhanced_specs": len(specs),
            "pipeline": "resumed_enhanced_export",
        }
        self._progress(f"loading {len(specs)} accepted specs from {input_path}")
        self._progress(f"exporting {len(specs)} records ({self.config.image_format}) with {max(1, self.config.workers)} worker(s)")
        manifest = SyntheticRecordExporter(solver_profile=self.config.solver_profile).write_jsonl(
            specs,
            output_path,
            image_format=self.config.image_format,
            style_seed=self.config.seed,
            qa_path=qa_path,
            gt_path=gt_path,
            require_count=self.config.target_records if self.config.require_exact_count else None,
            run_config=run_config,
            workers=self.config.workers,
            scene_payload=self.config.scene_payload,
            progress_every=self.config.progress_every,
        )
        self._progress(f"resumed export complete: {manifest['num_records']} records -> {output_path}")
        return manifest

    def _generate_base_specs(self) -> list[SyntheticSpec]:
        overrides = load_grid_overrides(self.config.grid_json_file, self.config.grid_json)
        families = select_template_families(self.config.families, grid_overrides=overrides)
        requested = self.config.base_records or _default_base_records(self.config.target_records)
        reserve = max(0, self.config.base_reserve_records or 0)
        per_family_limit = self.config.per_family_limit
        if reserve and per_family_limit is not None and families:
            per_family_limit += (reserve + len(families) - 1) // len(families)
        sampler = SyntheticSampler(
            families=families,
            seed=self.config.seed,
            enable_mutation=self.config.base_mutations_per_variant > 0,
            mutations_per_variant=self.config.base_mutations_per_variant,
            per_family_limit=per_family_limit,
            minimum_difficulty=self.config.minimum_difficulty,
            maximum_difficulty=self.config.maximum_difficulty,
            validate_candidates=False,
            solver_profile=self.config.solver_profile,
        )
        specs = sampler.sample(max_records=requested + reserve)
        if not specs:
            raise ValueError("No base specs generated")
        if self.config.require_exact_count and len(specs) < requested:
            raise ValueError(f"Expected at least {requested} base specs, got {len(specs)}; expand the selected family grids")
        return specs

    def _generate_enhanced_specs(self, base_specs: Iterable[SyntheticSpec]) -> tuple[list[SyntheticSpec], CandidateDiagnostics]:
        base_specs = list(base_specs)
        enhancer = FormalTraceEnhancer(
            seed=self.config.seed + 2000,
            rounds_distribution=self.config.enhancement_rounds,
            operation_types=self.config.enhancement_operations,
        )
        accepted: list[SyntheticSpec] = []
        accepted_ids: set[str] = set()
        accepted_per_family: dict[str, int] = {}
        family_targets = Counter(spec.family for spec in base_specs[: self.config.target_records]) if len(base_specs) >= self.config.target_records else Counter()
        diagnostics = CandidateDiagnostics()
        per_base_candidates: list[list[SyntheticSpec]] = []
        for base_index, base_spec in enumerate(base_specs):
            candidates = enhancer.enhance(base_spec, base_index=base_index)
            if self.config.max_enhancements_per_base is not None:
                candidates = candidates[: max(0, self.config.max_enhancements_per_base)]
            if candidates:
                per_base_candidates.append(candidates)
        max_depth = max((len(candidates) for candidates in per_base_candidates), default=0)
        checked = 0
        for wave in range(max_depth):
            for base_index, candidates in enumerate(per_base_candidates):
                if wave >= len(candidates):
                    continue
                candidate = candidates[(base_index + wave) % len(candidates)]
                if family_targets and accepted_per_family.get(candidate.family, 0) >= family_targets[candidate.family]:
                    diagnostics.record_skip("family_target_reached", candidate_id=candidate.id, family=candidate.family)
                    continue
                if self.config.per_family_limit is not None and accepted_per_family.get(candidate.family, 0) >= self.config.per_family_limit:
                    diagnostics.record_skip("per_family_limit", candidate_id=candidate.id, family=candidate.family)
                    continue
                checked += 1
                if candidate.id in accepted_ids:
                    diagnostics.record_rejection("duplicate_id", candidate_id=candidate.id, family=candidate.family)
                    self._enhancement_progress(checked, len(accepted), diagnostics.rejected)
                    continue
                if not difficulty_in_range(
                    candidate.difficulty,
                    minimum=self.config.minimum_difficulty,
                    maximum=self.config.maximum_difficulty,
                ):
                    diagnostics.record_rejection(
                        "difficulty_out_of_range",
                        candidate_id=candidate.id,
                        family=candidate.family,
                        metadata={"difficulty": candidate.difficulty},
                    )
                    self._enhancement_progress(checked, len(accepted), diagnostics.rejected)
                    continue
                if not self.duplicates.accept(candidate):
                    diagnostics.record_rejection("duplicate_content", candidate_id=candidate.id, family=candidate.family)
                    self._enhancement_progress(checked, len(accepted), diagnostics.rejected)
                    continue
                report = self.rejector.check(candidate)
                if not report.valid:
                    diagnostics.record_rejection(
                        str(getattr(report, "reason", None) or "candidate_validation_failed"),
                        candidate_id=candidate.id,
                        family=candidate.family,
                        metadata=getattr(report, "metadata", {}),
                    )
                    self._enhancement_progress(checked, len(accepted), diagnostics.rejected)
                    continue
                candidate = replace(
                    candidate,
                    metadata={
                        **candidate.metadata,
                        "candidate_validation": accepted_validation_metadata(report),
                    },
                )
                accepted.append(candidate)
                diagnostics.record_accept()
                accepted_ids.add(candidate.id)
                accepted_per_family[candidate.family] = accepted_per_family.get(candidate.family, 0) + 1
                self._enhancement_progress(checked, len(accepted), diagnostics.rejected)
                if len(accepted) >= self.config.target_records:
                    return accepted, diagnostics
        return accepted, diagnostics

    def _enhancement_progress(self, checked: int, accepted: int, rejected: int) -> None:
        every = max(0, self.config.progress_every)
        if every and (checked % every == 0 or accepted == self.config.target_records):
            self._progress(f"enhancement progress: checked={checked}, accepted={accepted}/{self.config.target_records}, rejected={rejected}")

    def _progress(self, message: str) -> None:
        if self.config.progress_every > 0:
            print(f"[geosym] {message}", file=sys.stderr, flush=True)


def parse_rounds_distribution(inline_json: str | None) -> Dict[int, int] | None:
    if not inline_json:
        return None
    payload = json.loads(inline_json)
    if not isinstance(payload, dict):
        raise ValueError("Enhancement rounds must be a JSON object")
    return {int(key): int(value) for key, value in payload.items()}


def parse_operation_list(value: str | None) -> tuple[str, ...]:
    if not value:
        return DEFAULT_ENHANCEMENT_OPERATIONS
    operations = tuple(item.strip() for item in value.split(",") if item.strip())
    return operations or DEFAULT_ENHANCEMENT_OPERATIONS


def _default_base_records(target_records: int) -> int:
    # Default enhancement distribution emits five candidates per base. Add slack
    # because verifier/rejection filtering can discard operation combinations.
    return max(1, (target_records + 3) // 4)


def _write_specs(path: Path, specs: Iterable[SyntheticSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for spec in specs:
            json.dump(_spec_to_dict(spec), f, ensure_ascii=False)
            f.write("\n")


def _read_specs(path: Path) -> list[SyntheticSpec]:
    specs: list[SyntheticSpec] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                specs.append(_spec_from_dict(payload))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid enhanced spec at {path}:{line_number}: {exc}") from exc
    if not specs:
        raise ValueError(f"Enhanced spec input is empty: {path}")
    return specs


def _spec_from_dict(payload: Dict[str, Any]) -> SyntheticSpec:
    if not isinstance(payload, dict):
        raise TypeError("spec must be a JSON object")
    trace = payload["trace"]
    if not isinstance(trace, list):
        raise TypeError("trace must be a list")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be an object")
    return SyntheticSpec(
        id=str(payload["id"]),
        family=str(payload["family"]),
        trace=trace,
        question=str(payload["question"]),
        answer_hint=str(payload.get("answer_hint", "")),
        difficulty=str(payload.get("difficulty", "medium")),
        diagnostic_axes=tuple(str(item) for item in payload.get("diagnostic_axes", ())),
        metadata=metadata,
    )


def _spec_to_dict(spec: SyntheticSpec) -> Dict[str, Any]:
    return {
        "id": spec.id,
        "family": spec.family,
        "trace": spec.trace,
        "question": spec.question,
        "answer_hint": spec.answer_hint,
        "difficulty": spec.difficulty,
        "diagnostic_axes": list(spec.diagnostic_axes),
        "metadata": spec.metadata,
    }
