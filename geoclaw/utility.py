"""Goal-aware utility classification without exact trace matching."""

from __future__ import annotations

from typing import Any, Iterable

from geosym_world_geometry.fact import FactStore

from .feedback import HARMFUL, PROGRESS, REDUNDANT, UNKNOWN


def fact_keys(facts: FactStore) -> set[tuple[str, str, str | None]]:
    return {(fact.type, fact.object, fact.value) for fact in facts}


def relevant_fact_keys(environment: Any, facts: FactStore) -> set[tuple[str, str, str | None]]:
    target = environment.state.target
    accepted = {
        (str(item.get("type")), str(item.get("object")), None if item.get("value") is None else str(item.get("value")))
        for item in environment.task.acceptable_intermediate_facts
    }
    if target is None:
        return accepted
    relevant_objects = {target.object}
    relevant_ids = set()
    changed = True
    while changed:
        changed = False
        for fact in facts:
            if fact.object in relevant_objects or fact.id in relevant_ids or any(dep in relevant_ids or _dependency_object(dep) in relevant_objects for dep in fact.dependencies):
                if fact.id not in relevant_ids:
                    relevant_ids.add(fact.id)
                    changed = True
                for dependency in fact.dependencies:
                    dep_object = _dependency_object(dependency)
                    if dep_object and dep_object not in relevant_objects:
                        relevant_objects.add(dep_object)
                        changed = True
    derived = {(fact.type, fact.object, fact.value) for fact in facts if fact.id in relevant_ids or fact.object in relevant_objects}
    return derived | accepted


def classify_utility(environment: Any, status: str, before: set[tuple[str, str, str | None]], after: set[tuple[str, str, str | None]], action_name: str, result: dict) -> str:
    if action_name == "final":
        return "terminal"
    if status != "valid":
        return HARMFUL
    if action_name == "verify":
        return PROGRESS if result.get("truth") == "true" else UNKNOWN
    if action_name in {"zoom", "calculate", "measure"}:
        signature = (action_name, str(result))
        if signature in environment.information_signatures:
            return REDUNDANT
        environment.information_signatures.add(signature)
        return PROGRESS
    relevant = relevant_fact_keys(environment, environment.facts)
    new_facts = after - before
    if new_facts & relevant:
        return PROGRESS
    return REDUNDANT if new_facts else REDUNDANT


def _dependency_object(dependency: str) -> str | None:
    if ":" in dependency:
        return dependency.split(":", 1)[1]
    return None
