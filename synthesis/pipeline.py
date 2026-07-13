"""Staged synthetic generation pipeline inspired by the original GeoSym flow."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

from .enhancer import DEFAULT_ENHANCEMENT_OPERATIONS, FormalTraceEnhancer
from .exporter import SyntheticRecordExporter
from .filters import DegeneracyRejector, DuplicateFilter
from .sampler import SyntheticSampler
from .templates import SyntheticSpec, load_grid_overrides, select_template_families


@dataclass(frozen=True)
class PipelineRunConfig:
    target_records: int = 20
    base_records: int | None = None
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


class GeometryGenerationPipeline:
    """Run base generation, enhancement, filtering, render export, QA, and GT."""

    def __init__(self, config: PipelineRunConfig):
        self.config = config
        self.rejector = DegeneracyRejector()
        self.duplicates = DuplicateFilter()

    def run(
        self,
        output_path: str | Path,
        *,
        base_path: str | Path | None = None,
        enhanced_path: str | Path | None = None,
        qa_path: str | Path | None = None,
        gt_path: str | Path | None = None,
    ) -> Dict[str, Any]:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        base_output = Path(base_path) if base_path else output.parent / "json" / "base" / f"{output.stem}.base.jsonl"
        enhanced_output = Path(enhanced_path) if enhanced_path else output.parent / "json" / "enhanced" / f"{output.stem}.enhanced.jsonl"

        base_specs = self._generate_base_specs()
        _write_specs(base_output, base_specs)

        enhanced_specs, rejected = self._generate_enhanced_specs(base_specs)
        if self.config.require_exact_count and len(enhanced_specs) != self.config.target_records:
            raise ValueError(f"Expected {self.config.target_records} enhanced specs, got {len(enhanced_specs)}")
        _write_specs(enhanced_output, enhanced_specs)

        run_config = {
            **asdict(self.config),
            "families": list(self.config.families) or "all",
            "enhancement_operations": list(self.config.enhancement_operations),
            "base_path": str(base_output),
            "enhanced_path": str(enhanced_output),
            "num_base_specs": len(base_specs),
            "num_enhanced_specs": len(enhanced_specs),
            "num_rejected_enhancements": rejected,
            "pipeline": "staged_formal_geosym",
        }
        manifest = SyntheticRecordExporter().write_jsonl(
            enhanced_specs,
            output,
            image_format=self.config.image_format,
            style_seed=self.config.seed,
            qa_path=qa_path,
            gt_path=gt_path,
            require_count=self.config.target_records if self.config.require_exact_count else None,
            run_config=run_config,
        )
        return manifest

    def _generate_base_specs(self) -> list[SyntheticSpec]:
        overrides = load_grid_overrides(self.config.grid_json_file, self.config.grid_json)
        families = select_template_families(self.config.families, grid_overrides=overrides)
        requested = self.config.base_records or _default_base_records(self.config.target_records)
        sampler = SyntheticSampler(
            families=families,
            seed=self.config.seed,
            enable_mutation=self.config.base_mutations_per_variant > 0,
            mutations_per_variant=self.config.base_mutations_per_variant,
            per_family_limit=self.config.per_family_limit,
        )
        specs = sampler.sample(max_records=requested)
        if not specs:
            raise ValueError("No base specs generated")
        return specs

    def _generate_enhanced_specs(self, base_specs: Iterable[SyntheticSpec]) -> tuple[list[SyntheticSpec], int]:
        enhancer = FormalTraceEnhancer(
            seed=self.config.seed + 2000,
            rounds_distribution=self.config.enhancement_rounds,
            operation_types=self.config.enhancement_operations,
        )
        accepted: list[SyntheticSpec] = []
        rejected = 0
        per_base_candidates = []
        for base_index, base_spec in enumerate(base_specs):
            candidates = enhancer.enhance(base_spec, base_index=base_index)
            if self.config.max_enhancements_per_base is not None:
                candidates = candidates[: max(0, self.config.max_enhancements_per_base)]
            if candidates:
                per_base_candidates.append(candidates)
        max_depth = max((len(candidates) for candidates in per_base_candidates), default=0)
        for candidate_idx in range(max_depth):
            for candidates in per_base_candidates:
                if candidate_idx >= len(candidates):
                    continue
                candidate = candidates[candidate_idx]
                if not self.duplicates.accept(candidate):
                    rejected += 1
                    continue
                report = self.rejector.check(candidate)
                if not report.valid:
                    rejected += 1
                    continue
                accepted.append(candidate)
                if len(accepted) >= self.config.target_records:
                    return accepted, rejected
        return accepted, rejected


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
