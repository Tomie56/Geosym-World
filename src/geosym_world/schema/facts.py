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
        self._by_key: Dict[tuple, Fact] = {}
        self._by_type: Dict[str, List[Fact]] = {}
        self._by_object: Dict[str, List[Fact]] = {}

    def add(self, fact_type: str, object_id: str, value=None, *, source: str = "derived",
            rule: str = "", dependencies: Iterable[str] = (), metadata: Dict[str, Any] | None = None) -> Fact:
        value_str = None if value is None else str(sp.simplify(value))
        key = (fact_type, object_id, value_str, source, rule)
        existing = self._by_key.get(key)
        if existing is not None:
            return existing
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
        self._by_key[key] = fact
        self._by_type.setdefault(fact_type, []).append(fact)
        self._by_object.setdefault(object_id, []).append(fact)
        return fact

    def find(self, fact_type: str | None = None, object_id: str | None = None) -> List[Fact]:
        if fact_type is not None and object_id is not None:
            candidates = self._by_type.get(fact_type, ())
            return [fact for fact in candidates if fact.object == object_id]
        if fact_type is not None:
            return list(self._by_type.get(fact_type, ()))
        if object_id is not None:
            return list(self._by_object.get(object_id, ()))
        return list(self._facts)

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
