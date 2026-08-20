"""Deterministic grouped dataset splitting for synthetic specs."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from typing import Iterable, Mapping

from .templates import SyntheticSpec


SPLIT_NAMES = ("train", "dev", "test")
SPLIT_GROUP_MODES = ("lineage", "topology")


def lineage_root(spec: SyntheticSpec) -> str:
    return str(spec.metadata.get("lineage_root") or spec.metadata.get("base_spec_id") or spec.metadata.get("mutated_from") or spec.id)


def topology_key(spec: SyntheticSpec) -> str:
    """Return an answer-free key shared by parameter variants of one task topology."""

    target_type = next(
        (str(step.get("type")) for step in spec.trace if step.get("op") == "set_target" and step.get("type") is not None),
        None,
    )
    payload = {
        "family": spec.family,
        "operations": [str(step.get("op", "")) for step in spec.trace],
        "target_type": target_type,
        "diagnostic_axes": sorted(spec.diagnostic_axes),
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def split_group_key(spec: SyntheticSpec, *, mode: str = "lineage") -> str:
    if mode == "lineage":
        return lineage_root(spec)
    if mode == "topology":
        return topology_key(spec)
    raise ValueError(f"Unknown split group mode {mode!r}; expected one of {SPLIT_GROUP_MODES}")


def assign_splits(
    specs: Iterable[SyntheticSpec],
    *,
    seed: int = 0,
    ratios: Mapping[str, float] | None = None,
    group_mode: str = "lineage",
) -> dict[str, list[SyntheticSpec]]:
    assigned: dict[str, list[SyntheticSpec]] = {name: [] for name in SPLIT_NAMES}
    for spec in assign_split_labels(specs, seed=seed, ratios=ratios, group_mode=group_mode):
        assigned[str(spec.metadata["split"])].append(spec)
    return assigned


def assign_split_labels(
    specs: Iterable[SyntheticSpec],
    *,
    seed: int = 0,
    ratios: Mapping[str, float] | None = None,
    group_mode: str = "lineage",
) -> list[SyntheticSpec]:
    """Assign group-stable split labels while preserving input order."""

    if group_mode not in SPLIT_GROUP_MODES:
        raise ValueError(f"Unknown split group mode {group_mode!r}; expected one of {SPLIT_GROUP_MODES}")
    weights = _validate_ratios(ratios or {"train": 0.8, "dev": 0.1, "test": 0.1})
    buckets: dict[str, str] = {}
    assigned: list[SyntheticSpec] = []
    for spec in specs:
        root = lineage_root(spec)
        group_key = split_group_key(spec, mode=group_mode)
        existing = spec.metadata.get("split")
        if existing is not None and existing not in SPLIT_NAMES:
            raise ValueError(f"Unknown split label for {spec.id}: {existing!r}")
        bucket = str(existing) if existing is not None else _bucket(group_key, seed=seed, weights=weights)
        previous = buckets.setdefault(group_key, bucket)
        if previous != bucket:
            raise ValueError(f"Split group {group_key!r} has conflicting split labels: {previous} and {bucket}")
        assigned.append(replace(spec, metadata={
            **spec.metadata,
            "split": bucket,
            "lineage_root": root,
            "split_group_mode": group_mode,
            "split_group_key": group_key,
        }))
    return assigned


def validate_split_disjoint(
    splits: Mapping[str, Iterable[SyntheticSpec]],
    *,
    group_mode: str = "lineage",
) -> None:
    seen: dict[str, str] = {}
    for split, specs in splits.items():
        for spec in specs:
            group_key = split_group_key(spec, mode=group_mode)
            previous = seen.setdefault(group_key, split)
            if previous != split:
                raise ValueError(f"Split group {group_key!r} appears in both {previous} and {split}")


def _validate_ratios(ratios: Mapping[str, float]) -> dict[str, float]:
    if set(ratios) != set(SPLIT_NAMES):
        raise ValueError(f"Split ratios must contain exactly {SPLIT_NAMES}")
    normalized = {name: float(ratios[name]) for name in SPLIT_NAMES}
    if any(value < 0 for value in normalized.values()) or abs(sum(normalized.values()) - 1.0) > 1e-9:
        raise ValueError("Split ratios must be non-negative and sum to one")
    if not any(normalized.values()):
        raise ValueError("At least one split ratio must be positive")
    return normalized


def _bucket(root: str, *, seed: int, weights: Mapping[str, float]) -> str:
    digest = hashlib.sha256(f"{seed}:{root}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    cursor = 0.0
    for split in SPLIT_NAMES:
        cursor += weights[split]
        if value < cursor:
            return split
    return SPLIT_NAMES[-1]
