"""Trace sampler for procedural synthetic geometry records."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from .difficulty import difficulty_in_range, score_trace
from .filters import DegeneracyRejector, DuplicateFilter
from .mutators import mutate_trace
from .templates import SyntheticSpec, TemplateFamily, built_in_template_families


class SyntheticSampler:
    def __init__(
        self,
        families: Iterable[TemplateFamily] | None = None,
        *,
        seed: int = 0,
        enable_mutation: bool = True,
        mutations_per_variant: int = 1,
        per_family_limit: int | None = None,
        minimum_difficulty: str | None = None,
        maximum_difficulty: str | None = None,
        validate_candidates: bool = True,
        solver_profile: str = "auto",
    ):
        self.families = list(families or built_in_template_families())
        self.seed = seed
        self.enable_mutation = enable_mutation
        self.mutations_per_variant = max(0, mutations_per_variant)
        self.per_family_limit = per_family_limit
        self.minimum_difficulty = minimum_difficulty
        self.maximum_difficulty = maximum_difficulty
        self.validate_candidates = validate_candidates
        self.rejector = DegeneracyRejector(solver_profile=solver_profile)
        self.duplicates = DuplicateFilter()

    def sample(self, max_records: int = 20) -> list[SyntheticSpec]:
        accepted: list[SyntheticSpec] = []
        accepted_ids: set[str] = set()
        counter = 0
        variants = [iter(family.variants()) for family in self.families]
        active = [True] * len(variants)
        per_family_counts = {family.name: 0 for family in self.families}
        while any(active) and len(accepted) < max_records:
            for idx, iterator in enumerate(variants):
                if not active[idx]:
                    continue
                family_name = self.families[idx].name
                if self.per_family_limit is not None and per_family_counts[family_name] >= self.per_family_limit:
                    active[idx] = False
                    continue
                try:
                    spec = next(iterator)
                except StopIteration:
                    active[idx] = False
                    continue
                for candidate in self._candidates(spec, counter):
                    counter += 1
                    if candidate.id in accepted_ids:
                        continue
                    if not self.duplicates.accept(candidate):
                        continue
                    if not difficulty_in_range(candidate.difficulty, minimum=self.minimum_difficulty, maximum=self.maximum_difficulty):
                        continue
                    if self.validate_candidates:
                        report = self.rejector.check(candidate)
                        if not report.valid:
                            continue
                    accepted.append(candidate)
                    accepted_ids.add(candidate.id)
                    per_family_counts[family_name] += 1
                    if len(accepted) >= max_records:
                        return accepted
                    if self.per_family_limit is not None and per_family_counts[family_name] >= self.per_family_limit:
                        break
        return accepted

    def _candidates(self, spec: SyntheticSpec, counter: int):
        yield _with_scored_difficulty(spec)
        if self.enable_mutation:
            for mutation_idx in range(self.mutations_per_variant):
                mutation_seed = self.seed + counter * 997 + mutation_idx
                mutated_trace = mutate_trace(spec.trace, seed=mutation_seed)
                if mutated_trace != spec.trace:
                    yield _with_scored_difficulty(
                        replace(
                            spec,
                            id=f"{spec.id}_m{counter}_{mutation_idx}",
                            trace=mutated_trace,
                            metadata={**spec.metadata, "mutated_from": spec.id, "mutation_seed": mutation_seed},
                        )
                    )


def _with_scored_difficulty(spec: SyntheticSpec) -> SyntheticSpec:
    label, policy = score_trace(spec.trace, spec.diagnostic_axes)
    return replace(spec, difficulty=label, metadata={**spec.metadata, "difficulty_policy": policy})
