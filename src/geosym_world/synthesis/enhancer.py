"""Configurable trace enhancement for staged synthetic generation."""

from __future__ import annotations

import copy
import random
from dataclasses import replace
from typing import Any, Dict, Iterable, List

import sympy as sp

from geosym_world.construction.builder import ConstructionBuilder
from geosym_world.construction.operations import ConstructionOp, operation_references
from geosym_world.kernel.geometry2d.intersections import linear_linear_intersection_result_2d

from .difficulty import score_trace
from .mutators import mutate_trace
from .preflight import EnhancementPreflight, PreflightDiagnostics
from .task_graph import build_task_dependency_graph
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
        self.preflight = EnhancementPreflight()

    @property
    def enhancements_per_base(self) -> int:
        return sum(self.rounds_distribution.values())

    def enhance(self, spec: SyntheticSpec, *, base_index: int = 0) -> list[SyntheticSpec]:
        enhanced: list[SyntheticSpec] = []
        for enhance_idx, rounds in enumerate(_rounds_list(self.rounds_distribution)):
            trace = copy.deepcopy(spec.trace)
            applied: list[str] = []
            frontier: set[str] = set()
            preflight_diagnostics = PreflightDiagnostics()
            rng = random.Random(self.seed + base_index * 1009 + enhance_idx * 9176)
            for round_idx in range(rounds):
                attempted = set(applied)
                while True:
                    op_name = self._choose_operation(
                        rng,
                        trace,
                        excluded=attempted,
                        terminal=round_idx == rounds - 1,
                        frontier=frontier,
                    )
                    if op_name is None:
                        next_trace = trace
                        break
                    attempted.add(op_name)
                    next_trace = self._apply_operation(
                        op_name,
                        trace,
                        rng,
                        round_idx=round_idx,
                        enhance_idx=enhance_idx,
                        diagnostics=preflight_diagnostics,
                        required_inputs=frontier,
                    )
                    if next_trace != trace:
                        break
                if next_trace == trace or op_name is None:
                    break
                trace = next_trace
                applied.append(op_name)
                frontier = _next_frontier(trace, op_name)

            suffix = f"e{enhance_idx:03d}"
            label, policy = score_trace(trace, spec.diagnostic_axes)
            task_graph = build_task_dependency_graph(trace)
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
                        "enhancement_preflight": preflight_diagnostics.to_dict(),
                        "task_graph_audit": {
                            "checksum": task_graph["checksum"],
                            "effective_depth": task_graph["effective_depth"],
                            "branch_points": task_graph["branch_points"],
                            "dead_enhancement_steps": task_graph["dead_enhancement_steps"],
                        },
                        "difficulty_policy": policy,
                    },
                )
            )
        return enhanced

    def _choose_operation(
        self,
        rng: random.Random,
        trace: Trace,
        *,
        excluded: set[str] | None = None,
        terminal: bool = False,
        frontier: set[str] | None = None,
    ) -> str | None:
        excluded = excluded or set()
        candidates = [name for name in self.operation_types if name not in excluded and self._can_apply(name, trace)]
        if frontier:
            candidates = [name for name in candidates if name != "coordinate_mutation"]
        if terminal:
            terminal_candidates = [name for name in candidates if name not in {"connect_points", "point_on_segment_ratio"}]
            if terminal_candidates:
                candidates = terminal_candidates
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

    def _apply_operation(
        self,
        op_name: str,
        trace: Trace,
        rng: random.Random,
        *,
        round_idx: int,
        enhance_idx: int,
        diagnostics: PreflightDiagnostics,
        required_inputs: set[str],
    ) -> Trace:
        if op_name == "coordinate_mutation":
            mutated = mutate_trace(trace, seed=self.seed + enhance_idx * 10007 + round_idx)
            return mutated if mutated != trace else trace

        body, target = _split_target(trace)
        inventory = _inventory(trace)
        token = f"e{enhance_idx}_{round_idx}"
        provenance = {"source": "synthetic_enhancement", "operation": op_name, "round": round_idx}
        base_state = ConstructionBuilder().execute(body)
        if op_name == "add_midpoint":
            candidates = [
                {
                    "op": "midpoint",
                    "id": _candidate_id(f"M_{segment_id}_{token}", inventory["ids"]),
                    "of": list(endpoints),
                    "metadata": {"enhancement_role": "auxiliary_point", "provenance": provenance},
                }
                for segment_id, endpoints in sorted(inventory["segments"].items())
            ]
            op = self._select_candidate(base_state, candidates, rng, diagnostics, required_inputs)
            if op is None:
                return trace
            body.append(op)
        elif op_name == "connect_points":
            candidates = _connect_point_candidates(inventory, token, provenance)
            op = self._select_candidate(base_state, candidates, rng, diagnostics, required_inputs)
            if op is None:
                return trace
            body.append(op)
        elif op_name == "foot_of_perpendicular":
            candidates = _foot_candidates(inventory, token, provenance)
            op = self._select_candidate(base_state, candidates, rng, diagnostics, required_inputs)
            if op is None:
                return trace
            body.append(op)
        elif op_name == "angle_bisector":
            ops = _line_with_intended_intersections(op_name, base_state, inventory, rng, token, round_idx, self.preflight, diagnostics, required_inputs)
            if ops is None:
                return trace
            body.extend(ops)
        elif op_name in {"parallel_through", "perpendicular_through"}:
            ops = _line_with_intended_intersections(op_name, base_state, inventory, rng, token, round_idx, self.preflight, diagnostics, required_inputs)
            if ops is None:
                return trace
            body.extend(ops)
        elif op_name == "point_on_segment_ratio":
            candidates = [
                {
                    "op": "point_on_segment_ratio",
                    "id": _candidate_id(f"R_{segment_id}_{token}", inventory["ids"]),
                    "segment": segment_id,
                    "ratio": ratio,
                    "metadata": {"enhancement_role": "ratio_point", "provenance": provenance},
                }
                for segment_id in sorted(inventory["segments"])
                for ratio in ("1/3", "1/2", "2/3")
            ]
            op = self._select_candidate(base_state, candidates, rng, diagnostics, required_inputs)
            if op is None:
                return trace
            body.append(op)
        else:
            return trace
        return body + ([target] if target else [])

    def _select_candidate(
        self,
        state,
        candidates: list[Dict[str, Any]],
        rng: random.Random,
        diagnostics: PreflightDiagnostics,
        required_inputs: set[str],
    ) -> Dict[str, Any] | None:
        if required_inputs:
            candidates = [operation for operation in candidates if _operation_inputs(operation) & required_inputs]
        rng.shuffle(candidates)
        for operation in candidates:
            result = self.preflight.check(state, operation)
            diagnostics.record(operation, result)
            if result.valid:
                return operation
        return None


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
        elif name in {"midpoint", "foot_of_perpendicular", "point_on_segment_ratio", "intersection_line_line"}:
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


def _connect_point_candidates(inventory: Dict[str, Any], token: str, provenance: Dict[str, Any]) -> list[Dict[str, Any]]:
    points = list(dict.fromkeys(inventory["points2d"]))
    existing_pairs = {tuple(sorted(pair)) for pair in inventory["segments"].values()}
    pairs = [(a, b) for idx, a in enumerate(points) for b in points[idx + 1 :] if tuple(sorted((a, b))) not in existing_pairs]
    return [
        {
            "op": "add_segment",
            "id": _candidate_id(f"{a}{b}_{token}", inventory["ids"]),
            "points": [a, b],
            "metadata": {"enhancement_role": "auxiliary_segment", "provenance": provenance},
        }
        for a, b in pairs
    ]


def _foot_candidates(inventory: Dict[str, Any], token: str, provenance: Dict[str, Any]) -> list[Dict[str, Any]]:
    return [
        {
            "op": "foot_of_perpendicular",
            "id": _candidate_id(f"H_{point}_{segment_id}_{token}", inventory["ids"]),
            "point": point,
            "line": segment_id,
            "metadata": {"enhancement_role": "altitude_foot", "provenance": provenance},
        }
        for segment_id, endpoints in sorted(inventory["segments"].items())
        for point in dict.fromkeys(inventory["points2d"])
        if point not in endpoints
    ]


def _has_external_point_for_segment(inventory: Dict[str, Any]) -> bool:
    points = set(inventory["points2d"])
    return any(points.difference(endpoints) for endpoints in inventory["segments"].values())


def _through_line_candidates(op_name: str, inventory: Dict[str, Any], token: str) -> list[Dict[str, Any]]:
    candidates = []
    points = list(dict.fromkeys(inventory["points2d"]))
    for segment_id, endpoints in inventory["segments"].items():
        for point_id in points:
            if point_id not in endpoints:
                prefix = "par" if op_name == "parallel_through" else "perp"
                candidates.append(
                    {
                        "op": op_name,
                        "id": _candidate_id(f"{prefix}_{point_id}_{segment_id}_{token}", inventory["ids"]),
                        "point": point_id,
                        "line": segment_id,
                    }
                )
    return candidates


def _angle_bisector_candidates(inventory: Dict[str, Any], token: str) -> list[Dict[str, Any]]:
    candidates = []
    for vertex, neighbors in sorted(inventory["adjacency"].items()):
        ordered = sorted(neighbors)
        for index, first in enumerate(ordered):
            for second in ordered[index + 1 :]:
                candidates.append(
                    {
                        "op": "angle_bisector",
                        "id": _candidate_id(f"bis_{vertex}_{token}_{index}", inventory["ids"]),
                        "vertex": vertex,
                        "ray1": [vertex, first],
                        "ray2": [vertex, second],
                    }
                )
    return candidates


def _line_with_intended_intersections(
    op_name: str,
    base_state,
    inventory: Dict[str, Any],
    rng: random.Random,
    token: str,
    round_idx: int,
    preflight: EnhancementPreflight,
    diagnostics: PreflightDiagnostics,
    required_inputs: set[str],
) -> list[Dict[str, Any]] | None:
    if op_name == "angle_bisector":
        candidates = _angle_bisector_candidates(inventory, token)
    else:
        candidates = _through_line_candidates(op_name, inventory, token)
    if required_inputs:
        candidates = [operation for operation in candidates if _operation_inputs(operation) & required_inputs]
    rng.shuffle(candidates)
    for line_op in candidates:
        result = preflight.check(base_state, line_op)
        diagnostics.record(line_op, result)
        if not result.valid:
            continue
        candidate_state = result.state
        assert candidate_state is not None
        intersections = _new_segment_intersections(candidate_state, line_op["id"], sorted(inventory["segments"]))
        if not intersections:
            continue
        provenance = {
            "source": "synthetic_enhancement",
            "operation": op_name,
            "round": round_idx,
        }
        line_op["metadata"] = {
            "enhancement_role": "auxiliary_line",
            "materialized_intersections": len(intersections),
            "provenance": provenance,
        }
        result = [line_op]
        used_ids = set(inventory["ids"])
        used_ids.add(line_op["id"])
        for index, (segment_id, _) in enumerate(intersections, start=1):
            point_id = _unique_id(f"X_{token}_{index}", used_ids)
            result.append(
                {
                    "op": "intersection_line_line",
                    "id": point_id,
                    "line1": line_op["id"],
                    "line2": segment_id,
                    "metadata": {
                        "enhancement_role": "intended_intersection",
                        "dependencies": [line_op["id"], segment_id],
                        "provenance": provenance,
                    },
                }
            )
        return result
    return None


def _operation_inputs(operation: Dict[str, Any]) -> set[str]:
    return {reference for _, reference in operation_references(ConstructionOp.from_dict(operation))}


def _next_frontier(trace: Trace, operation_name: str) -> set[str]:
    if operation_name not in {"connect_points", "point_on_segment_ratio"}:
        return set()
    body, _ = _split_target(trace)
    expected = "add_segment" if operation_name == "connect_points" else operation_name
    for operation in reversed(body):
        if operation.get("op") == expected and (operation.get("metadata") or {}).get("enhancement_role"):
            return {str(operation["id"])}
    return set()


def _new_segment_intersections(state, line_id: str, segment_ids: list[str]) -> list[tuple[str, tuple[sp.Expr, ...]]]:
    line = state.get_line_like(line_id)
    intersections: list[tuple[str, tuple[sp.Expr, ...]]] = []
    for segment_id in segment_ids:
        segment_line = state.get_line_like(segment_id)
        result = linear_linear_intersection_result_2d(
            line.point,
            line.direction,
            segment_line.point,
            segment_line.direction,
            first_domain="line",
            second_domain="segment",
            objects=(line_id, segment_id),
        )
        if not result.is_constructible:
            continue
        point = result.points[0]
        if _matches_existing_point(point, state.points.values()):
            continue
        if any(_same_coords(point, existing) for _, existing in intersections):
            continue
        intersections.append((segment_id, point))
    return intersections


def _matches_existing_point(coords, points) -> bool:
    return any(point.dim == len(coords) and _same_coords(coords, point.coords) for point in points)


def _same_coords(left, right) -> bool:
    return len(left) == len(right) and all(sp.simplify(a - b) == 0 for a, b in zip(left, right))


def _candidate_id(base: str, existing: set[str]) -> str:
    normalized = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in base)
    if normalized not in existing:
        return normalized
    counter = 1
    while f"{normalized}_{counter}" in existing:
        counter += 1
    return f"{normalized}_{counter}"


def _unique_id(base: str, existing: set[str]) -> str:
    normalized = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in base)
    candidate = normalized
    counter = 1
    while candidate in existing:
        candidate = f"{normalized}_{counter}"
        counter += 1
    existing.add(candidate)
    return candidate
