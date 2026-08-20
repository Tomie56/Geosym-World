"""Versioned target-derivation supervision extracted from canonical facts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping

from geosym_world.schema.canonical import stable_checksum
from geosym_world.schema.facts import Fact, FactStore


DERIVATION_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class DerivationStep:
    index: int
    fact: Fact
    input_fact_ids: tuple[str, ...]
    input_references: tuple[dict[str, Any], ...]
    unresolved_references: tuple[str, ...]
    depth: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "fact_id": self.fact.id,
            "type": self.fact.type,
            "object": self.fact.object,
            "value": self.fact.value,
            "numeric_value": self.fact.to_dict()["numeric_value"],
            "source": self.fact.source,
            "rule": self.fact.rule,
            "dependencies": list(self.fact.dependencies),
            "input_fact_ids": list(self.input_fact_ids),
            "input_references": [dict(item) for item in self.input_references],
            "unresolved_references": list(self.unresolved_references),
            "depth": self.depth,
            "metadata": dict(self.fact.metadata),
            "provenance": [item.to_dict() for item in self.fact.provenance],
        }


@dataclass(frozen=True)
class DerivationValidation:
    valid: bool
    errors: tuple[dict[str, str], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {"valid": self.valid, "errors": [dict(item) for item in self.errors]}


class DerivationTracer:
    """Extract a deterministic, auditable dependency program for the target fact."""

    def trace_target(self, facts: FactStore) -> Dict[str, Any]:
        all_facts = list(facts)
        targets = sorted((fact for fact in all_facts if fact.type == "target"), key=lambda fact: fact.id)
        if not targets:
            return _finalize_derivation(
                target=None,
                ordered=(),
                edges=(),
                dependency_inputs={},
                unresolved=(),
                cycles=(),
                target_count=0,
            )

        target = targets[0]
        visited: set[str] = set()
        visiting: list[str] = []
        ordered: list[Fact] = []
        edges: list[Dict[str, str]] = []
        edge_keys: set[tuple[str, str, str]] = set()
        dependency_inputs: dict[str, list[dict[str, Any]]] = {}
        unresolved: list[Dict[str, str]] = []
        cycles: list[list[str]] = []

        def visit(fact: Fact) -> None:
            if fact.id in visiting:
                start = visiting.index(fact.id)
                cycle = visiting[start:] + [fact.id]
                if cycle not in cycles:
                    cycles.append(cycle)
                return
            if fact.id in visited:
                return
            visiting.append(fact.id)
            inputs = dependency_inputs.setdefault(fact.id, [])
            for reference in fact.dependencies:
                resolved = tuple(sorted(facts.resolve_dependency(reference), key=lambda item: item.id))
                inputs.append({"reference": reference, "fact_ids": [item.id for item in resolved]})
                if not resolved:
                    unresolved.append({"from": fact.id, "dependency": reference})
                    continue
                for dependency_fact in resolved:
                    edge_key = (dependency_fact.id, fact.id, reference)
                    if edge_key not in edge_keys:
                        edges.append({"from": dependency_fact.id, "to": fact.id, "reference": reference})
                        edge_keys.add(edge_key)
                    visit(dependency_fact)
            visiting.pop()
            visited.add(fact.id)
            ordered.append(fact)

        visit(target)
        return _finalize_derivation(
            target=target,
            ordered=ordered,
            edges=edges,
            dependency_inputs=dependency_inputs,
            unresolved=unresolved,
            cycles=cycles,
            target_count=len(targets),
        )


def validate_derivation_payload(payload: Mapping[str, Any]) -> DerivationValidation:
    """Validate a serialized derivation without access to its original FactStore."""

    errors: list[dict[str, str]] = []

    def add(code: str, message: str) -> None:
        errors.append({"code": code, "message": message})

    if payload.get("schema_version") != DERIVATION_SCHEMA_VERSION:
        add("derivation.schema_version", f"Expected {DERIVATION_SCHEMA_VERSION}")
    supplied_checksum = payload.get("checksum")
    checksum_payload = {key: value for key, value in payload.items() if key not in {"checksum", "validation"}}
    if supplied_checksum != stable_checksum(checksum_payload):
        add("derivation.checksum", "Derivation checksum does not match payload")

    facts = list(payload.get("facts") or ())
    steps = list(payload.get("steps") or ())
    edges = list(payload.get("edges") or ())
    target = payload.get("target_fact")
    fact_ids = [str(fact.get("id", "")) for fact in facts]
    if not all(fact_ids) or len(set(fact_ids)) != len(fact_ids):
        add("derivation.fact_ids", "Fact IDs must be non-empty and unique")
    if len(steps) != len(facts):
        add("derivation.step_count", "Each derivation fact requires exactly one step")

    step_fact_ids = [str(step.get("fact_id", "")) for step in steps]
    if step_fact_ids != fact_ids:
        add("derivation.step_order", "Step fact order must match topological fact order")
    if [step.get("index") for step in steps] != list(range(len(steps))):
        add("derivation.step_indices", "Step indices must be contiguous from zero")

    target_ids = [fact_id for fact_id, fact in zip(fact_ids, facts) if fact.get("type") == "target"]
    target_id = str(target.get("id", "")) if isinstance(target, Mapping) else ""
    if len(target_ids) != 1 or target_id != target_ids[0]:
        add("derivation.target", "Derivation requires exactly one matching target fact")
    if step_fact_ids and step_fact_ids[-1] != target_id:
        add("derivation.target_order", "Target fact must be the final derivation step")

    positions = {fact_id: index for index, fact_id in enumerate(step_fact_ids)}
    depths: dict[str, int] = {}
    expected_edges: set[tuple[str, str, str]] = set()
    for step in steps:
        fact_id = str(step.get("fact_id", ""))
        input_fact_ids = [str(item) for item in step.get("input_fact_ids") or ()]
        if any(input_id not in positions or positions[input_id] >= positions.get(fact_id, -1) for input_id in input_fact_ids):
            add("derivation.dependency_order", f"Step {fact_id} depends on a missing or later fact")
        expected_depth = max((depths.get(input_id, -1) for input_id in input_fact_ids), default=-1) + 1
        if step.get("depth") != expected_depth:
            add("derivation.depth", f"Step {fact_id} has an invalid dependency depth")
        depths[fact_id] = expected_depth
        for reference in step.get("input_references") or ():
            reference_name = str(reference.get("reference", ""))
            for input_id in reference.get("fact_ids") or ():
                expected_edges.add((str(input_id), fact_id, reference_name))

    actual_edges = {
        (str(edge.get("from", "")), str(edge.get("to", "")), str(edge.get("reference", "")))
        for edge in edges
    }
    if actual_edges != expected_edges:
        add("derivation.edges", "Edges must exactly match step input references")

    unresolved = list(payload.get("unresolved_dependencies") or ())
    cycles = list(payload.get("cycles") or ())
    if unresolved:
        add("derivation.unresolved", "Derivation contains unresolved dependencies")
    if cycles:
        add("derivation.cycle", "Derivation contains dependency cycles")
    integrity = payload.get("integrity") or {}
    expected_integrity = {
        "acyclic": not cycles,
        "target_reachable": bool(step_fact_ids) and step_fact_ids[-1] == target_id,
        "target_fact_count": len(target_ids),
        "num_facts": len(facts),
        "num_steps": len(steps),
        "num_edges": len(edges),
        "unresolved_count": len(unresolved),
        "cycle_count": len(cycles),
        "max_depth": max(depths.values(), default=0),
    }
    for field, expected in expected_integrity.items():
        if integrity.get(field) != expected:
            add("derivation.integrity_summary", f"Integrity field {field} does not match payload")
    computed_valid = len(target_ids) == 1 and not unresolved and not cycles and not errors
    expected_status = "complete" if computed_valid else "missing_target" if not target_ids else "ambiguous_target" if len(target_ids) > 1 else "incomplete"
    if integrity.get("status") != expected_status:
        add("derivation.integrity_status", "Integrity status does not match payload")
    if bool(integrity.get("valid")) != computed_valid:
        add("derivation.integrity", "Integrity summary disagrees with serialized derivation")
    return DerivationValidation(valid=computed_valid and not errors, errors=tuple(errors))


def _finalize_derivation(
    *,
    target: Fact | None,
    ordered: Iterable[Fact],
    edges: Iterable[Dict[str, str]],
    dependency_inputs: dict[str, list[dict[str, Any]]],
    unresolved: Iterable[Dict[str, str]],
    cycles: Iterable[list[str]],
    target_count: int,
) -> Dict[str, Any]:
    ordered_facts = list(ordered)
    edge_list = list(edges)
    unresolved_list = list(unresolved)
    cycle_list = list(cycles)
    depths: dict[str, int] = {}
    steps: list[Dict[str, Any]] = []
    for index, fact in enumerate(ordered_facts):
        inputs = dependency_inputs.get(fact.id, [])
        input_fact_ids = tuple(dict.fromkeys(fact_id for item in inputs for fact_id in item["fact_ids"]))
        resolved_depths = [depths[fact_id] for fact_id in input_fact_ids if fact_id in depths]
        depth = max(resolved_depths, default=-1) + 1
        depths[fact.id] = depth
        unresolved_references = tuple(item["reference"] for item in inputs if not item["fact_ids"])
        steps.append(
            DerivationStep(
                index=index,
                fact=fact,
                input_fact_ids=input_fact_ids,
                input_references=tuple(inputs),
                unresolved_references=unresolved_references,
                depth=depth,
            ).to_dict()
        )

    has_target = target is not None
    target_reachable = bool(steps) and steps[-1]["fact_id"] == target.id if target is not None else False
    valid = target_count == 1 and target_reachable and not unresolved_list and not cycle_list
    status = "complete" if valid else "missing_target" if not has_target else "ambiguous_target" if target_count > 1 else "incomplete"
    integrity = {
        "valid": valid,
        "status": status,
        "acyclic": not cycle_list,
        "target_reachable": target_reachable,
        "target_fact_count": target_count,
        "num_facts": len(ordered_facts),
        "num_steps": len(steps),
        "num_edges": len(edge_list),
        "unresolved_count": len(unresolved_list),
        "cycle_count": len(cycle_list),
        "max_depth": max((step["depth"] for step in steps), default=0),
    }
    payload: Dict[str, Any] = {
        "schema_version": DERIVATION_SCHEMA_VERSION,
        "target_fact": target.to_dict() if target is not None else None,
        "facts": [fact.to_dict() for fact in ordered_facts],
        "edges": edge_list,
        "steps": steps,
        "unresolved_dependencies": unresolved_list,
        "cycles": cycle_list,
        "integrity": integrity,
    }
    payload["checksum"] = stable_checksum(payload)
    return payload
