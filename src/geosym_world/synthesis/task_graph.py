"""Replayable dependency graph and dead-operation audit for synthetic tasks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from geosym_world.construction.operations import normalize_trace, operation_outputs, operation_references
from geosym_world.schema.canonical import stable_checksum


TASK_GRAPH_SCHEMA_VERSION = "1.0"
_PUBLIC_RELATION_OPERATIONS = {
    "angle_bisector",
    "foot_of_perpendicular",
    "midpoint",
    "parallel_through",
    "perpendicular_through",
}


def build_task_dependency_graph(trace: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    operations = normalize_trace(dict(item) for item in trace)
    producer: dict[str, int] = {}
    nodes: list[dict[str, Any]] = []
    roots: set[int] = set()

    for index, operation in enumerate(operations):
        inputs = [reference for _, reference in operation_references(operation)]
        if operation.op == "set_target" and str(operation.params.get("object", "")) in producer:
            inputs.append(str(operation.params["object"]))
        dependencies = sorted({producer[reference] for reference in inputs if reference in producer})
        outputs = list(operation_outputs(operation))
        metadata = dict(operation.params.get("metadata") or {})
        role = _semantic_role(operation.op, metadata)
        node = {
            "index": index,
            "operation": operation.op,
            "operation_id": operation.id,
            "inputs": inputs,
            "outputs": outputs,
            "dependencies": dependencies,
            "consumers": [],
            "semantic_role": role,
            "enhancement": bool(metadata.get("enhancement_role")),
        }
        nodes.append(node)
        for output in outputs:
            producer[output] = index
        if role in {"target", "public_given", "intended_intersection"}:
            roots.add(index)

    for node in nodes:
        for dependency in node["dependencies"]:
            nodes[dependency]["consumers"].append(node["index"])

    live = set(roots)
    pending = list(roots)
    while pending:
        index = pending.pop()
        for dependency in nodes[index]["dependencies"]:
            if dependency not in live:
                live.add(dependency)
                pending.append(dependency)

    depth: dict[int, int] = {}
    for node in nodes:
        index = node["index"]
        node["live"] = index in live
        node["consumers"] = sorted(node["consumers"])
        depth[index] = 1 + max((depth[item] for item in node["dependencies"]), default=0)
        node["depth"] = depth[index]

    dead_enhancement_steps = [node["index"] for node in nodes if node["enhancement"] and not node["live"]]
    payload = {
        "schema_version": TASK_GRAPH_SCHEMA_VERSION,
        "nodes": nodes,
        "root_steps": sorted(roots),
        "live_steps": sorted(live),
        "dead_enhancement_steps": dead_enhancement_steps,
        "effective_depth": max((depth[index] for index in live), default=0),
        "branch_points": sum(1 for node in nodes if node["live"] and len(node["consumers"]) > 1),
    }
    payload["checksum"] = stable_checksum(payload)
    return payload


def validate_task_dependency_graph(payload: Mapping[str, Any]) -> bool:
    if payload.get("schema_version") != TASK_GRAPH_SCHEMA_VERSION:
        return False
    checksum_payload = {key: value for key, value in payload.items() if key != "checksum"}
    return payload.get("checksum") == stable_checksum(checksum_payload)


def _semantic_role(operation: str, metadata: Mapping[str, Any]) -> str:
    if operation == "set_target":
        return "target"
    if metadata.get("annotation_role") == "public_given" or metadata.get("show_length"):
        return "public_given"
    if metadata.get("enhancement_role") == "intended_intersection":
        return "intended_intersection"
    if operation in _PUBLIC_RELATION_OPERATIONS and metadata.get("annotation_role") != "internal":
        return "public_given"
    if metadata.get("enhancement_role"):
        return "enhancement"
    return "internal"
