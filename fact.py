"""Fact schemas and fact store."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

import sympy as sp


@dataclass(frozen=True)
class Fact:
    id: str
    type: str
    object: str
    value: str | None = None
    source: str = "derived"
    rule: str = ""
    dependencies: tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "object": self.object,
            "value": self.value,
            "numeric_value": _numeric(self.value),
            "source": self.source,
            "rule": self.rule,
            "dependencies": list(self.dependencies),
            "metadata": self.metadata,
        }


class FactStore:
    """Deduplicated fact collection."""

    def __init__(self):
        self._facts: List[Fact] = []
        self._keys: set[tuple] = set()

    def add(self, fact_type: str, object_id: str, value=None, *, source: str = "derived",
            rule: str = "", dependencies: Iterable[str] = (), metadata: Dict[str, Any] | None = None) -> Fact:
        value_str = None if value is None else str(sp.simplify(value))
        key = (fact_type, object_id, value_str, source, rule)
        if key in self._keys:
            for fact in self._facts:
                if (fact.type, fact.object, fact.value, fact.source, fact.rule) == key:
                    return fact
        fact = Fact(
            id=f"fact_{len(self._facts):05d}",
            type=fact_type,
            object=object_id,
            value=value_str,
            source=source,
            rule=rule,
            dependencies=tuple(dependencies),
            metadata=metadata or {},
        )
        self._facts.append(fact)
        self._keys.add(key)
        return fact

    def find(self, fact_type: str | None = None, object_id: str | None = None) -> List[Fact]:
        results = self._facts
        if fact_type is not None:
            results = [fact for fact in results if fact.type == fact_type]
        if object_id is not None:
            results = [fact for fact in results if fact.object == object_id]
        return list(results)

    def to_list(self) -> List[Dict[str, Any]]:
        return [fact.to_dict() for fact in self._facts]

    def __iter__(self):
        return iter(self._facts)

    def __len__(self) -> int:
        return len(self._facts)


def _numeric(value: str | None):
    if value is None:
        return None
    try:
        return float(sp.N(sp.sympify(value)))
    except Exception:
        return None
