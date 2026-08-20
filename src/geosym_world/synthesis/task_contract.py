"""Benchmark task supervision derived from formal state and verified facts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

from geosym_world.schema.canonical import stable_checksum
from geosym_world.schema.facts import Fact, FactStore
from geosym_world.schema.state import FormalGeometryState


TASK_SCHEMA_VERSION = "1.1"
PUBLIC_RELATION_TYPES = {
    "angle_bisector",
    "foot_of_perpendicular",
    "midpoint",
    "point_on_segment_ratio",
    "perpendicular",
}


@dataclass(frozen=True)
class TaskContractValidation:
    valid: bool
    errors: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "errors": [dict(item) for item in self.errors]}


def build_task_contract(
    state: FormalGeometryState,
    facts: FactStore,
    derivation: Mapping[str, Any],
    *,
    family: str,
    difficulty: str,
    diagnostic_axes: tuple[str, ...],
    solution_paths: Mapping[str, Any],
) -> dict[str, Any]:
    """Build non-answer task metadata and explicit public givens."""

    operations = [str(item.get("op", "")) for item in state.construction_trace]
    operation_counts = Counter(operations)
    derivation_rules = Counter(str(step.get("rule") or "unspecified") for step in derivation.get("steps", ()))
    public_givens = _public_givens(state, facts)
    structure_payload = {
        "operations": operations,
        "target_type": state.target.type if state.target else None,
        "target_mode": state.target.mode if state.target else None,
        "alternative_solution_count": int(solution_paths["alternative_count"]),
        "object_kinds": _object_kind_counts(state),
        "diagnostic_axes": sorted(diagnostic_axes),
    }
    payload = {
        "schema_version": TASK_SCHEMA_VERSION,
        "family": family,
        "difficulty": difficulty,
        "target": state.target.to_dict() if state.target else None,
        "public_givens": public_givens,
        "construction": {
            "num_steps": len(state.construction_trace),
            "operations": operations,
            "operation_counts": dict(sorted(operation_counts.items())),
        },
        "derivation": {
            "num_steps": int(derivation["integrity"]["num_steps"]),
            "max_depth": int(derivation["integrity"]["max_depth"]),
            "rule_counts": dict(sorted(derivation_rules.items())),
            "checksum": derivation["checksum"],
        },
        "solutions": {
            "num_paths": int(solution_paths["num_paths"]),
            "alternative_count": int(solution_paths["alternative_count"]),
            "rules": list(solution_paths["rules"]),
            "checksum": str(solution_paths["checksum"]),
        },
        "object_kinds": _object_kind_counts(state),
        "dimensions": sorted({point.dim for point in state.points.values()}),
        "diagnostic_axes": sorted(diagnostic_axes),
        "structure_signature": stable_checksum(structure_payload),
    }
    payload["checksum"] = stable_checksum(payload)
    return payload


def validate_task_contract(payload: Mapping[str, Any]) -> TaskContractValidation:
    errors: list[dict[str, str]] = []

    def add(code: str, message: str) -> None:
        errors.append({"code": code, "message": message})

    if payload.get("schema_version") != TASK_SCHEMA_VERSION:
        add("task.schema_version", f"Expected {TASK_SCHEMA_VERSION}")
    supplied_checksum = payload.get("checksum")
    checksum_payload = {key: value for key, value in payload.items() if key not in {"checksum", "validation"}}
    if supplied_checksum != stable_checksum(checksum_payload):
        add("task.checksum", "Task checksum does not match payload")

    construction = payload.get("construction") or {}
    operations = [str(item) for item in construction.get("operations") or ()]
    operation_counts = dict(sorted(Counter(operations).items()))
    if construction.get("num_steps") != len(operations):
        add("task.construction_steps", "Construction step count does not match operations")
    if construction.get("operation_counts") != operation_counts:
        add("task.operation_counts", "Operation histogram does not match operations")

    derivation = payload.get("derivation") or {}
    rule_counts = derivation.get("rule_counts") or {}
    if any(not isinstance(count, int) or count < 0 for count in rule_counts.values()):
        add("task.derivation_rules", "Derivation rule counts must be non-negative integers")
    if sum(rule_counts.values()) != derivation.get("num_steps"):
        add("task.derivation_steps", "Derivation step count does not match rule histogram")
    if not isinstance(derivation.get("max_depth"), int) or derivation.get("max_depth", -1) < 0:
        add("task.derivation_depth", "Derivation depth must be a non-negative integer")
    if not isinstance(derivation.get("checksum"), str) or not derivation.get("checksum"):
        add("task.derivation_checksum", "Derivation checksum is required")

    solutions = payload.get("solutions") or {}
    rules = [str(item) for item in solutions.get("rules") or ()]
    if not isinstance(solutions.get("num_paths"), int) or solutions.get("num_paths", 0) < 1:
        add("task.solution_count", "At least one solution path is required")
    if solutions.get("alternative_count") != max(0, solutions.get("num_paths", 0) - 1):
        add("task.solution_count", "Alternative solution count is inconsistent")
    if len(rules) != solutions.get("num_paths"):
        add("task.solution_rules", "Each solution path requires one rule")
    if not isinstance(solutions.get("checksum"), str) or not solutions.get("checksum"):
        add("task.solution_checksum", "Solution-path checksum is required")

    object_kinds = payload.get("object_kinds") or {}
    if any(not isinstance(count, int) or count < 0 for count in object_kinds.values()):
        add("task.object_kinds", "Object-kind counts must be non-negative integers")
    dimensions = payload.get("dimensions") or []
    if dimensions != sorted(set(dimensions)) or any(dimension not in {2, 3} for dimension in dimensions):
        add("task.dimensions", "Dimensions must be sorted unique values from {2, 3}")

    for index, given in enumerate(payload.get("public_givens") or ()):
        if not isinstance(given, Mapping):
            add("task.public_given", f"Public given {index} must be a mapping")
            continue
        if "coords" in given or "target" in given:
            add("task.public_leak", f"Public given {index} exposes hidden state")
        if given.get("kind") not in {"quantity", "relation"}:
            add("task.public_kind", f"Public given {index} has an unknown kind")

    target = payload.get("target") or {}
    structure_payload = {
        "operations": operations,
        "target_type": target.get("type"),
        "target_mode": target.get("mode"),
        "alternative_solution_count": solutions.get("alternative_count"),
        "object_kinds": object_kinds,
        "diagnostic_axes": sorted(str(item) for item in payload.get("diagnostic_axes") or ()),
    }
    if payload.get("structure_signature") != stable_checksum(structure_payload):
        add("task.structure_signature", "Structure signature does not match task topology")
    return TaskContractValidation(valid=not errors, errors=tuple(errors))


def _public_givens(state: FormalGeometryState, facts: FactStore) -> list[dict[str, Any]]:
    givens: list[dict[str, Any]] = []
    for segment in state.segments.values():
        if segment.metadata.get("show_length"):
            fact = _first_fact(facts.find("length", segment.id))
            if fact is not None:
                givens.append(_measurement_given(fact, display_object=segment.id))
    for region in state.regions.values():
        if region.kind == "sector_region" and region.metadata.get("show_central_angle"):
            fact = _first_fact(facts.find("central_angle", region.id))
            if fact is not None:
                givens.append(_measurement_given(fact, display_object=f"angle:{region.id}"))
    for relation in state.relations:
        if relation.get("type") == "foot_of_perpendicular":
            foot_id = str(relation.get("point", ""))
            source_id = str(relation.get("from", ""))
            if foot_id in state.points and state.points[foot_id].metadata.get("show_length"):
                fact = _first_fact(facts.find("distance", f"{source_id}|{foot_id}")) or _first_fact(facts.find("distance", f"{foot_id}|{source_id}"))
                if fact is not None:
                    givens.append(_measurement_given(fact, display_object=f"{source_id}{foot_id}"))
        if relation.get("type") in PUBLIC_RELATION_TYPES:
            givens.append({
                "kind": "relation",
                "relation_id": relation["id"],
                "type": relation["type"],
                "payload": {key: value for key, value in relation.items() if key not in {"id", "result"}},
                "evidence": "visible_marker_or_construction",
            })
    return sorted(givens, key=lambda item: (item["kind"], item.get("display_object", ""), item.get("relation_id", "")))


def _measurement_given(fact: Fact, *, display_object: str) -> dict[str, Any]:
    payload = fact.to_dict()
    return {
        "kind": "quantity",
        "type": fact.type,
        "object": fact.object,
        "display_object": display_object,
        "value": payload["value"],
        "numeric_value": payload["numeric_value"],
        "source_fact_id": fact.id,
        "evidence": "rendered_measurement_label",
    }


def _first_fact(facts: list[Fact]) -> Fact | None:
    return sorted(facts, key=lambda fact: fact.id)[0] if facts else None


def _object_kind_counts(state: FormalGeometryState) -> dict[str, int]:
    return {
        "arc": len(state.arcs),
        "circle": len(state.circles),
        "composite_region": len(state.composite_regions),
        "line": len(state.lines),
        "plane": len(state.planes),
        "point": len(state.points),
        "polygon": len(state.polygons),
        "polyhedron": len(state.polyhedra),
        "ray": len(state.rays),
        "region": len(state.regions),
        "relation": len(state.relations),
        "segment": len(state.segments),
    }
