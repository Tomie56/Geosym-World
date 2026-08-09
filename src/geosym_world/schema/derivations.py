"""Minimal target derivation extraction."""

from __future__ import annotations

from typing import Any, Dict, List

from geosym_world.schema.facts import Fact, FactStore


class DerivationTracer:
    """Extracts a dependency subgraph for target facts."""

    def trace_target(self, facts: FactStore) -> Dict[str, Any]:
        all_facts = list(facts)
        by_id = {fact.id: fact for fact in all_facts}
        targets = [fact for fact in all_facts if fact.type == "target"]
        if not targets:
            return {"target_fact": None, "facts": [], "edges": []}
        target = targets[0]
        visited: set[str] = set()
        ordered: List[Fact] = []
        edges: List[Dict[str, str]] = []

        def visit(fact: Fact) -> None:
            if fact.id in visited:
                return
            visited.add(fact.id)
            for dep in fact.dependencies:
                if dep in by_id:
                    edges.append({"from": dep, "to": fact.id})
                    visit(by_id[dep])
            ordered.append(fact)

        visit(target)
        return {
            "target_fact": target.to_dict(),
            "facts": [fact.to_dict() for fact in ordered],
            "edges": edges,
        }
