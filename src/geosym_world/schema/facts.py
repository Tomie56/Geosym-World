"""Fact schemas and fact store."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

import sympy as sp

from .canonical import stable_checksum
from .ids import validate_geometry_id
from .provenance import EvidenceKind, Provenance
from geosym_world.errors import InvalidGeometryInput


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
    provenance: tuple[Provenance, ...] = ()

    def __post_init__(self) -> None:
        validate_geometry_id(self.id)
        if not isinstance(self.type, str) or not self.type.strip():
            raise InvalidGeometryInput("Fact type must be non-empty")
        validate_geometry_id(self.object)
        if not isinstance(self.source, str) or not self.source.strip():
            raise InvalidGeometryInput("Fact source must be non-empty")
        if not isinstance(self.rule, str):
            raise InvalidGeometryInput("Fact rule must be a string")
        for dependency in self.dependencies:
            validate_geometry_id(dependency)
        if not self.provenance:
            raise InvalidGeometryInput(f"Fact {self.id} requires provenance")

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
            "provenance": [item.to_dict() for item in self.provenance],
        }


class FactStore:
    """Deduplicated fact collection."""

    def __init__(self):
        self._facts: List[Fact] = []
        self._by_key: Dict[tuple, Fact] = {}
        self._by_type: Dict[str, List[Fact]] = {}
        self._by_object: Dict[str, List[Fact]] = {}

    def add(self, fact_type: str, object_id: str, value=None, *, source: str = "derived",
            rule: str = "", dependencies: Iterable[str] = (), metadata: Dict[str, Any] | None = None,
            provenance: Iterable[Provenance] | None = None) -> Fact:
        value_str = None if value is None else str(sp.simplify(value))
        dependency_ids = tuple(str(item) for item in dependencies)
        key = (fact_type, object_id, value_str, source, rule, dependency_ids)
        existing = self._by_key.get(key)
        if existing is not None:
            return existing
        fact = Fact(
            id=_fact_id(fact_type, object_id, value_str, source, rule, dependency_ids),
            type=fact_type,
            object=object_id,
            value=value_str,
            source=source,
            rule=rule,
            dependencies=dependency_ids,
            metadata=metadata or {},
            provenance=tuple(provenance or (_default_provenance(fact_type, source, rule, dependency_ids),)),
        )
        self._facts.append(fact)
        self._by_key[key] = fact
        self._by_type.setdefault(fact_type, []).append(fact)
        self._by_object.setdefault(object_id, []).append(fact)
        return fact

    def add_fact(self, fact: Fact) -> Fact:
        """Insert an already materialized fact while preserving its identity."""

        key = (fact.type, fact.object, fact.value, fact.source, fact.rule, fact.dependencies)
        existing = self._by_key.get(key)
        if existing is not None:
            return existing
        self._facts.append(fact)
        self._by_key[key] = fact
        self._by_type.setdefault(fact.type, []).append(fact)
        self._by_object.setdefault(fact.object, []).append(fact)
        return fact

    def copy(self) -> "FactStore":
        copied = FactStore()
        for fact in self._facts:
            copied.add_fact(fact)
        return copied

    def remove_where(self, predicate) -> tuple[Fact, ...]:
        """Remove facts matching predicate and return the removed facts."""

        removed = tuple(fact for fact in self._facts if predicate(fact))
        if not removed:
            return ()
        kept = tuple(fact for fact in self._facts if not predicate(fact))
        rebuilt = FactStore()
        for fact in kept:
            rebuilt.add_fact(fact)
        self._facts = rebuilt._facts
        self._by_key = rebuilt._by_key
        self._by_type = rebuilt._by_type
        self._by_object = rebuilt._by_object
        return removed

    def find(self, fact_type: str | None = None, object_id: str | None = None) -> List[Fact]:
        if fact_type is not None and object_id is not None:
            candidates = self._by_type.get(fact_type, ())
            return [fact for fact in candidates if fact.object == object_id]
        if fact_type is not None:
            return list(self._by_type.get(fact_type, ()))
        if object_id is not None:
            return list(self._by_object.get(object_id, ()))
        return list(self._facts)

    def resolve_dependency(self, dependency: str) -> tuple[Fact, ...]:
        """Resolve a fact ID or an object dependency token to canonical facts."""

        exact = tuple(fact for fact in self._facts if fact.id == dependency)
        if exact:
            return exact
        object_id = dependency.split(":", 1)[1] if ":" in dependency else dependency
        candidates = self.find(object_id=object_id)
        existence = tuple(fact for fact in candidates if fact.type == "exists")
        return existence or candidates

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


def _fact_id(
    fact_type: str,
    object_id: str,
    value: str | None,
    source: str,
    rule: str,
    dependencies: Iterable[str],
) -> str:
    digest = stable_checksum(
        {
            "type": fact_type,
            "object": object_id,
            "value": value,
            "source": source,
            "rule": rule,
            "dependencies": list(dependencies),
        }
    )[:16]
    return f"fact_{digest}"


def _default_provenance(
    fact_type: str,
    source: str,
    rule: str,
    dependencies: Iterable[str],
) -> Provenance:
    kind_by_source = {
        "constructed": EvidenceKind.CONSTRUCTION,
        "asserted": EvidenceKind.VERIFIER,
        "derived": EvidenceKind.SOLVER,
        "given": EvidenceKind.SOURCE,
        "parser": EvidenceKind.PARSER,
    }
    return Provenance(
        source_id=f"fact:{fact_type}",
        kind=kind_by_source.get(source, EvidenceKind.SOLVER),
        event_id=f"rule:{rule or 'unspecified'}",
        dependencies=tuple(dependencies),
        metadata={"fact_source": source, "rule": rule},
    )
