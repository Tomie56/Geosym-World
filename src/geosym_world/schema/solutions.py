"""Validated alternative fact paths for one geometry target."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .canonical import stable_checksum
from .entities import Target
from .facts import Fact, FactStore


SOLUTION_PATH_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class SolutionPathValidation:
    valid: bool
    errors: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "errors": [dict(item) for item in self.errors]}


def build_solution_paths(facts: FactStore, target: Target) -> dict[str, Any]:
    requested_rules = tuple(str(item) for item in target.metadata.get("solution_rules", ()))
    preferred_rule = target.metadata.get("preferred_rule")
    candidates = facts.find(target.type, target.object)
    if requested_rules:
        candidates = [fact for rule in requested_rules for fact in candidates if fact.rule == rule]
    elif preferred_rule:
        candidates = [fact for fact in candidates if fact.rule == preferred_rule]
    else:
        target_facts = facts.find("target", target.object)
        resolved_id = target_facts[-1].metadata.get("resolved_fact_id") if target_facts else None
        candidates = [fact for fact in candidates if fact.id == resolved_id]

    paths = [_trace_path(facts, fact, index=index) for index, fact in enumerate(candidates)]
    payload = {
        "schema_version": SOLUTION_PATH_SCHEMA_VERSION,
        "target": target.to_dict(),
        "requested_rules": list(requested_rules),
        "num_paths": len(paths),
        "alternative_count": max(0, len(paths) - 1),
        "rules": [path["root_fact"]["rule"] for path in paths],
        "paths": paths,
    }
    payload["checksum"] = stable_checksum(payload)
    payload["validation"] = validate_solution_paths(payload).to_dict()
    return payload


def validate_solution_paths(payload: Mapping[str, Any]) -> SolutionPathValidation:
    errors: list[dict[str, str]] = []

    def add(code: str, message: str) -> None:
        errors.append({"code": code, "message": message})

    if payload.get("schema_version") != SOLUTION_PATH_SCHEMA_VERSION:
        add("solutions.schema_version", f"Expected {SOLUTION_PATH_SCHEMA_VERSION}")
    checksum_payload = {key: value for key, value in payload.items() if key not in {"checksum", "validation"}}
    if payload.get("checksum") != stable_checksum(checksum_payload):
        add("solutions.checksum", "Solution-path checksum does not match payload")
    paths = list(payload.get("paths") or ())
    if payload.get("num_paths") != len(paths) or payload.get("alternative_count") != max(0, len(paths) - 1):
        add("solutions.count", "Solution-path counts are inconsistent")
    target = payload.get("target") or {}
    requested_rules = [str(item) for item in payload.get("requested_rules") or ()]
    roots: list[str] = []
    rules: list[str] = []
    for index, path in enumerate(paths):
        path_checksum_payload = {key: value for key, value in path.items() if key != "checksum"}
        if path.get("checksum") != stable_checksum(path_checksum_payload):
            add("solutions.path_checksum", f"Path {index} checksum does not match")
        root = path.get("root_fact") or {}
        roots.append(str(root.get("id", "")))
        rules.append(str(root.get("rule", "")))
        if root.get("type") != target.get("type") or root.get("object") != target.get("object"):
            add("solutions.root_target", f"Path {index} root does not match target")
        if requested_rules and root.get("rule") not in requested_rules:
            add("solutions.root_rule", f"Path {index} uses an unrequested rule")
        steps = list(path.get("steps") or ())
        if not steps or steps[-1].get("fact_id") != root.get("id"):
            add("solutions.path_order", f"Path {index} does not terminate at its root fact")
        if path.get("unresolved_dependencies") or path.get("cycles"):
            add("solutions.path_integrity", f"Path {index} is incomplete")
        if not (path.get("integrity") or {}).get("valid"):
            add("solutions.path_integrity", f"Path {index} integrity is invalid")
    if len(set(roots)) != len(roots):
        add("solutions.duplicate_root", "Solution paths must use distinct root facts")
    if requested_rules and sorted(rules) != sorted(requested_rules):
        add("solutions.requested_rules", "Every requested solution rule must resolve exactly once")
    if payload.get("rules") != rules:
        add("solutions.rules", "Solution rule summary does not match paths")
    if not paths:
        add("solutions.missing", "At least one valid solution path is required")
    return SolutionPathValidation(valid=not errors, errors=tuple(errors))


def _trace_path(facts: FactStore, root: Fact, *, index: int) -> dict[str, Any]:
    visited: set[str] = set()
    visiting: list[str] = []
    ordered: list[Fact] = []
    inputs: dict[str, list[dict[str, Any]]] = {}
    edges: list[dict[str, str]] = []
    unresolved: list[dict[str, str]] = []
    cycles: list[list[str]] = []

    def visit(fact: Fact) -> None:
        if fact.id in visiting:
            cycles.append(visiting[visiting.index(fact.id):] + [fact.id])
            return
        if fact.id in visited:
            return
        visiting.append(fact.id)
        fact_inputs = inputs.setdefault(fact.id, [])
        for reference in fact.dependencies:
            resolved = tuple(sorted(facts.resolve_dependency(reference), key=lambda item: item.id))
            fact_inputs.append({"reference": reference, "fact_ids": [item.id for item in resolved]})
            if not resolved:
                unresolved.append({"from": fact.id, "dependency": reference})
            for dependency in resolved:
                edges.append({"from": dependency.id, "to": fact.id, "reference": reference})
                visit(dependency)
        visiting.pop()
        visited.add(fact.id)
        ordered.append(fact)

    visit(root)
    depths: dict[str, int] = {}
    steps: list[dict[str, Any]] = []
    for step_index, fact in enumerate(ordered):
        input_ids = tuple(dict.fromkeys(item for group in inputs.get(fact.id, ()) for item in group["fact_ids"]))
        depth = max((depths[item] for item in input_ids if item in depths), default=-1) + 1
        depths[fact.id] = depth
        steps.append({
            "index": step_index,
            "fact_id": fact.id,
            "type": fact.type,
            "object": fact.object,
            "rule": fact.rule,
            "input_fact_ids": list(input_ids),
            "depth": depth,
            "provenance": [item.to_dict() for item in fact.provenance],
        })
    path = {
        "index": index,
        "root_fact": root.to_dict(),
        "steps": steps,
        "edges": edges,
        "unresolved_dependencies": unresolved,
        "cycles": cycles,
        "integrity": {
            "valid": bool(steps) and steps[-1]["fact_id"] == root.id and not unresolved and not cycles,
            "num_steps": len(steps),
            "num_edges": len(edges),
            "max_depth": max(depths.values(), default=0),
        },
    }
    path["checksum"] = stable_checksum(path)
    return path
