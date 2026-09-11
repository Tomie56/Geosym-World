"""Fact schemas and fact store."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping

import sympy as sp

from .canonical import stable_checksum
from .ids import validate_geometry_id
from .provenance import EvidenceKind, Provenance
from geosym_world.errors import InvalidGeometryInput


class FactScope(str, Enum):
    """Controls where a fact may be consumed."""

    SOURCE_ASSERTION = "source_assertion"
    ENTAILED_FACT = "entailed_fact"
    REALIZATION_FACT = "realization_fact"
    GENERATION_PARAMETER = "generation_parameter"


SOURCE_REASONING_SCOPES = frozenset({FactScope.SOURCE_ASSERTION, FactScope.ENTAILED_FACT})
REALIZATION_SCOPES = frozenset(FactScope)


@dataclass(frozen=True)
class Fact:
    id: str
    type: str
    object: str
    value: str | None = None
    source: str = "derived"
    scope: FactScope = FactScope.REALIZATION_FACT
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
        if not isinstance(self.scope, FactScope):
            raise InvalidGeometryInput(f"Unknown fact scope: {self.scope}")
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
            "scope": self.scope.value,
            "rule": self.rule,
            "dependencies": list(self.dependencies),
            "metadata": self.metadata,
            "provenance": [item.to_dict() for item in self.provenance],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Fact":
        source = str(data.get("source", "derived"))
        return cls(
            id=str(data["id"]),
            type=str(data["type"]),
            object=str(data["object"]),
            value=str(data["value"]) if data.get("value") is not None else None,
            source=source,
            scope=_resolve_scope(data.get("scope"), source),
            rule=str(data.get("rule", "")),
            dependencies=tuple(str(item) for item in data.get("dependencies", ())),
            metadata=dict(data.get("metadata", {})),
            provenance=tuple(Provenance.from_dict(item) for item in data.get("provenance", ())),
        )


class FactStore:
    """Deduplicated fact collection."""

    def __init__(self):
        self._facts: List[Fact] = []
        self._by_key: Dict[tuple, Fact] = {}
        self._by_type: Dict[str, List[Fact]] = {}
        self._by_object: Dict[str, List[Fact]] = {}

    def add(
        self,
        fact_type: str,
        object_id: str,
        value=None,
        *,
        source: str = "derived",
        scope: FactScope | str | None = None,
        rule: str = "",
        dependencies: Iterable[str] = (),
        metadata: Dict[str, Any] | None = None,
        provenance: Iterable[Provenance] | None = None,
    ) -> Fact:
        value_str = None if value is None else str(sp.simplify(value))
        dependency_ids = tuple(str(item) for item in dependencies)
        resolved_scope = _resolve_scope(scope, source)
        key = (fact_type, object_id, value_str, source, resolved_scope.value, rule, dependency_ids)
        existing = self._by_key.get(key)
        if existing is not None:
            return existing
        fact = Fact(
            id=_fact_id(fact_type, object_id, value_str, source, resolved_scope, rule, dependency_ids),
            type=fact_type,
            object=object_id,
            value=value_str,
            source=source,
            scope=resolved_scope,
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

        key = (fact.type, fact.object, fact.value, fact.source, fact.scope.value, fact.rule, fact.dependencies)
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

    def find(
        self,
        fact_type: str | None = None,
        object_id: str | None = None,
        *,
        scopes: Iterable[FactScope | str] | None = None,
    ) -> List[Fact]:
        if fact_type is not None and object_id is not None:
            candidates = self._by_type.get(fact_type, ())
            facts = [fact for fact in candidates if fact.object == object_id]
        elif fact_type is not None:
            facts = list(self._by_type.get(fact_type, ()))
        elif object_id is not None:
            facts = list(self._by_object.get(object_id, ()))
        else:
            facts = list(self._facts)
        if scopes is None:
            return facts
        allowed = _normalize_scopes(scopes)
        return [fact for fact in facts if fact.scope in allowed]

    def scoped(self, scopes: Iterable[FactScope | str]) -> "FactStore":
        """Return a store containing only facts consumable in the requested scopes."""

        allowed = _normalize_scopes(scopes)
        selected = FactStore()
        for fact in self._facts:
            if fact.scope in allowed:
                selected.add_fact(fact)
        return selected

    def for_source_reasoning(self) -> "FactStore":
        """Exclude coordinate witnesses and generation-only parameters."""

        return self.scoped(SOURCE_REASONING_SCOPES)

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

    @classmethod
    def from_list(cls, data: Iterable[Mapping[str, Any]]) -> "FactStore":
        store = cls()
        for item in data:
            store.add_fact(Fact.from_dict(item))
        return store

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
    scope: FactScope,
    rule: str,
    dependencies: Iterable[str],
) -> str:
    digest = stable_checksum(
        {
            "type": fact_type,
            "object": object_id,
            "value": value,
            "source": source,
            "scope": scope.value,
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


def _resolve_scope(scope: FactScope | str | None, source: str) -> FactScope:
    if scope is not None:
        try:
            return scope if isinstance(scope, FactScope) else FactScope(scope)
        except ValueError as exc:
            raise InvalidGeometryInput(f"Unknown fact scope: {scope}") from exc
    return {
        "asserted": FactScope.SOURCE_ASSERTION,
        "given": FactScope.SOURCE_ASSERTION,
        "parser": FactScope.SOURCE_ASSERTION,
        "entailed": FactScope.ENTAILED_FACT,
        "generation": FactScope.GENERATION_PARAMETER,
    }.get(source, FactScope.REALIZATION_FACT)


def _normalize_scopes(scopes: Iterable[FactScope | str]) -> frozenset[FactScope]:
    normalized: set[FactScope] = set()
    for scope in scopes:
        try:
            normalized.add(scope if isinstance(scope, FactScope) else FactScope(scope))
        except ValueError as exc:
            raise InvalidGeometryInput(f"Unknown fact scope: {scope}") from exc
    if not normalized:
        raise InvalidGeometryInput("At least one fact scope is required")
    return frozenset(normalized)
