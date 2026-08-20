"""Configurable trace enhancement for staged synthetic generation."""

from __future__ import annotations

import copy
import random
from dataclasses import replace
from typing import Any, Dict, Iterable, List

from .difficulty import score_trace
from .mutators import mutate_trace
from .templates import SyntheticSpec


Trace = List[Dict[str, Any]]


DEFAULT_ENHANCEMENT_OPERATIONS = (
    "coordinate_mutation",
    "add_midpoint",
    "connect_points",
    "foot_of_perpendicular",
    "angle_bisector",
    "parallel_through",
    "perpendicular_through",
    "point_on_segment_ratio",
)


class FormalTraceEnhancer:
    """Generate multiple enhanced traces from one base synthetic spec.

    This mirrors the original GeoSym builder stage at the formal-trace level:
    each base sample is expanded into several independent variants according to
    a rounds distribution. Operations are conservative and verifier-filtered by
    the caller, so a failed enhancement does not poison the whole run.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        rounds_distribution: Dict[int | str, int] | None = None,
        operation_types: Iterable[str] = DEFAULT_ENHANCEMENT_OPERATIONS,
    ):
        self.seed = seed
        self.rounds_distribution = _normalize_rounds(rounds_distribution or {0: 1, 1: 3, 2: 1})
        self.operation_types = tuple(operation_types)

    @property
    def enhancements_per_base(self) -> int:
        return sum(self.rounds_distribution.values())

    def enhance(self, spec: SyntheticSpec, *, base_index: int = 0) -> list[SyntheticSpec]:
        enhanced: list[SyntheticSpec] = []
        for enhance_idx, rounds in enumerate(_rounds_list(self.rounds_distribution)):
            trace = copy.deepcopy(spec.trace)
            applied: list[str] = []
            rng = random.Random(self.seed + base_index * 1009 + enhance_idx * 9176)
            for round_idx in range(rounds):
                op_name = self._choose_operation(rng, trace, excluded=set(applied))
                if op_name is None:
                    break
                next_trace = self._apply_operation(op_name, trace, rng, round_idx=round_idx, enhance_idx=enhance_idx)
                if next_trace == trace:
                    break
                trace = next_trace
                applied.append(op_name)

            suffix = f"e{enhance_idx:03d}"
            label, policy = score_trace(trace, spec.diagnostic_axes)
            enhanced.append(
                replace(
                    spec,
                    id=f"{spec.id}_{suffix}",
                    trace=trace,
                    difficulty=label,
                    metadata={
                        **spec.metadata,
                        "base_spec_id": spec.id,
                        "lineage_root": spec.metadata.get("lineage_root", spec.id),
                        "lineage_role": "enhanced",
                        "enhance_idx": enhance_idx,
                        "enhancement_rounds": rounds,
                        "enhancement_operations": applied,
                        "difficulty_policy": policy,
                    },
                )
            )
        return enhanced

    def _choose_operation(self, rng: random.Random, trace: Trace, *, excluded: set[str] | None = None) -> str | None:
        excluded = excluded or set()
        candidates = [name for name in self.operation_types if name not in excluded and self._can_apply(name, trace)]
        if not candidates:
            return None
        return rng.choice(candidates)

    def _can_apply(self, op_name: str, trace: Trace) -> bool:
        inventory = _inventory(trace)
        if op_name == "coordinate_mutation":
            return True
        if op_name == "add_midpoint":
            return bool(inventory["segments"])
        if op_name == "connect_points":
            return len(inventory["points2d"]) >= 2
        if op_name == "foot_of_perpendicular":
            return bool(inventory["segments"]) and bool(inventory["points2d"])
        if op_name == "angle_bisector":
            return any(len(neighbors) >= 2 for neighbors in inventory["adjacency"].values())
        if op_name in {"parallel_through", "perpendicular_through"}:
            return _has_external_point_for_segment(inventory)
        if op_name == "point_on_segment_ratio":
            return bool(inventory["segments"])
        return False

    def _apply_operation(self, op_name: str, trace: Trace, rng: random.Random, *, round_idx: int, enhance_idx: int) -> Trace:
        if op_name == "coordinate_mutation":
            mutated = mutate_trace(trace, seed=self.seed + enhance_idx * 10007 + round_idx)
            return mutated if mutated != trace else trace

        body, target = _split_target(trace)
        inventory = _inventory(trace)
        token = f"e{enhance_idx}_{round_idx}"
        if op_name == "add_midpoint":
            segment_id, endpoints = rng.choice(list(inventory["segments"].items()))
            new_id = _unique_id(f"M_{segment_id}_{token}", inventory["ids"])
            body.append({"op": "midpoint", "id": new_id, "of": list(endpoints)})
        elif op_name == "connect_points":
            op = _connect_points_op(inventory, rng, token)
            if op is None:
                return trace
            body.append(op)
        elif op_name == "foot_of_perpendicular":
            op = _foot_op(inventory, rng, token)
            if op is None:
                return trace
            body.append(op)
        elif op_name == "angle_bisector":
            op = _angle_bisector_op(inventory, rng, token)
            if op is None:
                return trace
            body.append(op)
        elif op_name in {"parallel_through", "perpendicular_through"}:
            op = _through_line_op(op_name, inventory, rng, token)
            if op is None:
                return trace
            body.append(op)
        elif op_name == "point_on_segment_ratio":
            segment_id = rng.choice(sorted(inventory["segments"]))
            ratio = rng.choice(("1/3", "1/2", "2/3"))
            new_id = _unique_id(f"R_{segment_id}_{token}", inventory["ids"])
            body.append({"op": "point_on_segment_ratio", "id": new_id, "segment": segment_id, "ratio": ratio})
        else:
            return trace
        return body + ([target] if target else [])


def _normalize_rounds(rounds_distribution: Dict[int | str, int]) -> Dict[int, int]:
    normalized = {int(key): int(value) for key, value in rounds_distribution.items()}
    if any(key < 0 or value < 0 for key, value in normalized.items()):
        raise ValueError("Enhancement rounds and counts must be non-negative")
    if not any(normalized.values()):
        raise ValueError("Enhancement rounds distribution must request at least one output")
    return dict(sorted(normalized.items()))


def _rounds_list(rounds_distribution: Dict[int, int]) -> list[int]:
    rounds: list[int] = []
    for round_count, num_outputs in rounds_distribution.items():
        rounds.extend([round_count] * num_outputs)
    return rounds


def _split_target(trace: Trace) -> tuple[Trace, Dict[str, Any] | None]:
    body = [copy.deepcopy(op) for op in trace if op.get("op") != "set_target"]
    targets = [copy.deepcopy(op) for op in trace if op.get("op") == "set_target"]
    return body, targets[-1] if targets else None


def _inventory(trace: Trace) -> Dict[str, Any]:
    ids = {str(op.get("id")) for op in trace if op.get("id")}
    points2d: list[str] = []
    segments: Dict[str, tuple[str, str]] = {}
    adjacency: Dict[str, set[str]] = {}
    for op in trace:
        name = op.get("op")
        if name == "add_point":
            points2d.append(str(op["id"]))
        elif name == "midpoint" or name == "foot_of_perpendicular" or name == "intersection_line_line":
            if op.get("id"):
                points2d.append(str(op["id"]))
        elif name == "intersection_line_circle" or name == "intersection_circle_circle" or name == "tangent_points_from_point":
            points2d.extend(str(item) for item in op.get("ids", []))
        elif name in {"add_segment", "add_chord"}:
            segment_id = str(op["id"])
            points = tuple(str(item) for item in op.get("points", ()))
            if len(points) == 2:
                segments[segment_id] = points
                adjacency.setdefault(points[0], set()).add(points[1])
                adjacency.setdefault(points[1], set()).add(points[0])
    return {"ids": ids, "points2d": points2d, "segments": segments, "adjacency": adjacency}


def _connect_points_op(inventory: Dict[str, Any], rng: random.Random, token: str) -> Dict[str, Any] | None:
    points = list(dict.fromkeys(inventory["points2d"]))
    existing_pairs = {tuple(sorted(pair)) for pair in inventory["segments"].values()}
    pairs = [(a, b) for idx, a in enumerate(points) for b in points[idx + 1 :] if tuple(sorted((a, b))) not in existing_pairs]
    if not pairs:
        return None
    a, b = rng.choice(pairs)
    return {"op": "add_segment", "id": _unique_id(f"{a}{b}_{token}", inventory["ids"]), "points": [a, b]}


def _foot_op(inventory: Dict[str, Any], rng: random.Random, token: str) -> Dict[str, Any] | None:
    segment_id, endpoints = rng.choice(list(inventory["segments"].items()))
    candidates = [point for point in inventory["points2d"] if point not in endpoints]
    if not candidates:
        return None
    point = rng.choice(candidates)
    return {"op": "foot_of_perpendicular", "id": _unique_id(f"H_{point}_{segment_id}_{token}", inventory["ids"]), "point": point, "line": segment_id}


def _angle_bisector_op(inventory: Dict[str, Any], rng: random.Random, token: str) -> Dict[str, Any] | None:
    vertices = [vertex for vertex, neighbors in inventory["adjacency"].items() if len(neighbors) >= 2]
    if not vertices:
        return None
    vertex = rng.choice(vertices)
    a, b = rng.sample(sorted(inventory["adjacency"][vertex]), 2)
    return {"op": "angle_bisector", "id": _unique_id(f"bis_{vertex}_{token}", inventory["ids"]), "vertex": vertex, "ray1": [vertex, a], "ray2": [vertex, b]}


def _has_external_point_for_segment(inventory: Dict[str, Any]) -> bool:
    points = set(inventory["points2d"])
    return any(points.difference(endpoints) for endpoints in inventory["segments"].values())


def _through_line_op(op_name: str, inventory: Dict[str, Any], rng: random.Random, token: str) -> Dict[str, Any] | None:
    candidates = []
    points = list(dict.fromkeys(inventory["points2d"]))
    for segment_id, endpoints in inventory["segments"].items():
        for point_id in points:
            if point_id not in endpoints:
                candidates.append((segment_id, point_id))
    if not candidates:
        return None
    segment_id, point_id = rng.choice(candidates)
    prefix = "par" if op_name == "parallel_through" else "perp"
    return {
        "op": op_name,
        "id": _unique_id(f"{prefix}_{point_id}_{segment_id}_{token}", inventory["ids"]),
        "point": point_id,
        "line": segment_id,
    }


def _unique_id(base: str, existing: set[str]) -> str:
    normalized = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in base)
    candidate = normalized
    counter = 1
    while candidate in existing:
        candidate = f"{normalized}_{counter}"
        counter += 1
    existing.add(candidate)
    return candidate
