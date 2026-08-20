"""Dependency-aware incremental fact closure for monotonic state updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from geosym_world.schema.facts import FactStore
from geosym_world.schema.state import FormalGeometryState

from .closure import FactClosureSolver


@dataclass(frozen=True)
class IncrementalSolveReport:
    mode: str
    changed_object_ids: tuple[str, ...]
    invalidated_fact_ids: tuple[str, ...]
    recomputed_fact_ids: tuple[str, ...]
    facts: FactStore
    previous_checksum: str | None = None
    current_checksum: str | None = None
    changed_relation_ids: tuple[str, ...] = ()
    target_changed: bool = False
    provenance: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "changed_object_ids": list(self.changed_object_ids),
            "invalidated_fact_ids": list(self.invalidated_fact_ids),
            "recomputed_fact_ids": list(self.recomputed_fact_ids),
            "num_facts": len(self.facts),
            "previous_checksum": self.previous_checksum,
            "current_checksum": self.current_checksum,
            "changed_relation_ids": list(self.changed_relation_ids),
            "target_changed": self.target_changed,
            "provenance": dict(self.provenance or {}),
        }


class IncrementalFactClosure:
    """Maintain a fact store across verified, mostly additive state updates."""

    def __init__(self, solver: FactClosureSolver | None = None):
        self.solver = solver or FactClosureSolver(profile="minimal")

    def initialize(self, state: FormalGeometryState) -> IncrementalSolveReport:
        state.validate()
        facts = self.solver.solve(state)
        checksum = state.checksum()
        return IncrementalSolveReport(
            "full_recompute",
            tuple(sorted(state.object_ids())),
            (),
            tuple(fact.id for fact in facts),
            facts,
            current_checksum=checksum,
            provenance={"solver_profile": self.solver.profile, "reason": "initialize"},
        )

    def update(
        self,
        previous_state: FormalGeometryState,
        previous_facts: FactStore,
        current_state: FormalGeometryState,
        *,
        changed_object_ids: Iterable[str] | None = None,
    ) -> IncrementalSolveReport:
        previous_state.validate()
        current_state.validate()
        changed = set(changed_object_ids or _state_diff_ids(previous_state, current_state))
        changed_relation_ids = _relation_diff_ids(previous_state, current_state)
        target_changed = previous_state.target != current_state.target
        previous_checksum = previous_state.checksum()
        current_checksum = current_state.checksum()
        if not changed and not changed_relation_ids and not target_changed:
            return IncrementalSolveReport(
                "no_op", (), (), (), previous_facts.copy(),
                previous_checksum=previous_checksum,
                current_checksum=current_checksum,
                provenance={"solver_profile": self.solver.profile, "reason": "state_unchanged"},
            )
        if _requires_full_recompute(self.solver, previous_state, current_state, changed, target_changed):
            facts = self.solver.solve(current_state)
            previous_ids = {fact.id for fact in previous_facts}
            recomputed = tuple(fact.id for fact in facts if fact.id not in previous_ids)
            return IncrementalSolveReport(
                "full_recompute", tuple(sorted(changed)), tuple(fact.id for fact in previous_facts), recomputed, facts,
                previous_checksum=previous_checksum,
                current_checksum=current_checksum,
                changed_relation_ids=tuple(sorted(changed_relation_ids)),
                target_changed=target_changed,
                provenance={"solver_profile": self.solver.profile, "reason": "global_solver_profile_or_target_policy"},
            )

        affected = _related_object_ids(current_state, changed)
        affected_relations = changed_relation_ids | _related_relation_ids(current_state, affected)
        facts = previous_facts.copy()
        invalidated = _invalidate(facts, affected, affected_relations)
        self.solver._object_facts_for_ids(current_state, facts, affected)
        include_pairwise = current_state.target is not None and current_state.target.type == "distance"
        self.solver._metric_facts(current_state, facts, include_pairwise=include_pairwise, object_ids=affected)
        self.solver._construction_relation_facts(current_state, facts, relation_ids=affected_relations)
        self.solver._semantic_area_facts(current_state, facts)
        self.solver._semantic_perimeter_facts(current_state, facts)
        self.solver._target_fact(current_state, facts)
        recomputed_ids = tuple(fact.id for fact in facts if fact.id not in {item.id for item in previous_facts})
        return IncrementalSolveReport(
            "local_recompute", tuple(sorted(changed)), tuple(fact.id for fact in invalidated), recomputed_ids, facts,
            previous_checksum=previous_checksum,
            current_checksum=current_checksum,
            changed_relation_ids=tuple(sorted(changed_relation_ids)),
            target_changed=target_changed,
            provenance={
                "solver_profile": self.solver.profile,
                "reason": "dependency_indexed_local_recompute",
                "affected_object_ids": sorted(affected),
                "affected_relation_ids": sorted(affected_relations),
            },
        )


def _state_diff_ids(previous: FormalGeometryState, current: FormalGeometryState) -> set[str]:
    previous_payload = previous.to_dict()
    current_payload = current.to_dict()
    changed: set[str] = set()
    for bucket in ("points", "lines", "segments", "rays", "circles", "arcs", "polygons", "regions", "composite_regions", "planes", "polyhedra"):
        left = previous_payload.get(bucket, {})
        right = current_payload.get(bucket, {})
        for object_id in set(left) | set(right):
            if left.get(object_id) != right.get(object_id):
                changed.add(str(object_id))
    return changed


def _requires_full_recompute(
    solver: FactClosureSolver,
    previous: FormalGeometryState,
    current: FormalGeometryState,
    changed: set[str],
    target_changed: bool,
) -> bool:
    if solver.profile != "minimal":
        return True
    if target_changed:
        return True
    return False


def _related_object_ids(state: FormalGeometryState, changed: set[str]) -> set[str]:
    related = set(changed)
    changed_again = True
    while changed_again:
        changed_again = False
        for object_id, dependencies in _object_dependencies(state).items():
            if object_id not in related and dependencies & related:
                related.add(object_id)
                changed_again = True
    return related


def _object_dependencies(state: FormalGeometryState) -> dict[str, set[str]]:
    dependencies: dict[str, set[str]] = {}
    dependencies.update({item_id: set(item.source_points) for item_id, item in state.lines.items()})
    dependencies.update({item_id: {item.start, item.end} for item_id, item in state.segments.items()})
    dependencies.update({item_id: {item.start} | ({item.through} if item.through else set()) for item_id, item in state.rays.items()})
    dependencies.update({item_id: {item.center} for item_id, item in state.circles.items()})
    dependencies.update({item_id: {item.circle, item.start, item.end} for item_id, item in state.arcs.items()})
    dependencies.update({
        item_id: set(item.vertices)
        | {
            str(reference)
            for reference in (
                (item.metadata.get("area_formula") or {}).get("base"),
                (item.metadata.get("area_formula") or {}).get("height"),
                (item.metadata.get("area_formula") or {}).get("sector_region"),
            )
            if isinstance(reference, str)
        }
        | {str(reference) for reference in (item.metadata.get("area_formula") or {}).get("bases", ())}
        | {str(reference) for reference in (item.metadata.get("area_formula") or {}).get("diagonals", ())}
        | {
            str(reference)
            for reference in ((item.metadata.get("area_formula") or {}).get("height") or ())
            if isinstance((item.metadata.get("area_formula") or {}).get("height"), (list, tuple))
        }
        | {
            str(reference)
            for reference in (
                (item.metadata.get("perimeter_formula") or {}).get("width"),
                (item.metadata.get("perimeter_formula") or {}).get("height"),
            )
            if reference
        }
        for item_id, item in state.polygons.items()
    })
    dependencies.update({
        item_id: set(item.boundary)
        | set(item.holes)
        | {str(reference) for reference in (item.metadata.get("outer_circle"), item.metadata.get("inner_circle")) if reference}
        | {str(reference) for reference in item.metadata.get("circles", ())}
        | {str(reference) for reference in item.metadata.get("arcs", ())}
        | {str(reference) for reference in item.metadata.get("intersections", ())}
        | {str(reference) for reference in (item.metadata.get("circle"), item.metadata.get("radius_segment")) if reference}
        | {
            str(reference)
            for reference in (
                item.metadata.get("sector_region"),
                item.metadata.get("triangle_polygon"),
                item.metadata.get("arc"),
                item.metadata.get("chord"),
            )
            if reference
        }
        | ({str(item.metadata["area_formula"]["outer_region"])} if isinstance(item.metadata.get("area_formula"), dict) and item.metadata["area_formula"].get("outer_region") else set())
        | ({str(item.metadata["perimeter_formula"]["outer_region"])} if isinstance(item.metadata.get("perimeter_formula"), dict) and item.metadata["perimeter_formula"].get("outer_region") else set())
        | {
            str(reference)
            for formula_key in ("area_formula", "perimeter_formula")
            for reference in (
                (item.metadata.get(formula_key) or {}).get("outer_radius_segment"),
                (item.metadata.get(formula_key) or {}).get("inner_radius_segment"),
                (item.metadata.get(formula_key) or {}).get("radius_segment"),
                (item.metadata.get(formula_key) or {}).get("center_segment"),
            )
            if reference
        }
        | {
            str(reference)
            for formula_key in ("area_formula", "perimeter_formula")
            for reference in (item.metadata.get(formula_key) or {}).get("radius_segments", ())
            if reference
        }
        for item_id, item in state.regions.items()
    })
    dependencies.update({
        item_id: set(item.components) | ({str(item.metadata["parent_region"])} if item.metadata.get("parent_region") else set())
        for item_id, item in state.composite_regions.items()
    })
    dependencies.update({item_id: set(item.source_points) for item_id, item in state.planes.items()})
    dependencies.update({item_id: set(item.vertices) for item_id, item in state.polyhedra.items()})
    return dependencies


def _relation_diff_ids(previous: FormalGeometryState, current: FormalGeometryState) -> set[str]:
    previous_by_id = {str(item["id"]): item for item in previous.relations}
    current_by_id = {str(item["id"]): item for item in current.relations}
    return {
        relation_id
        for relation_id in set(previous_by_id) | set(current_by_id)
        if previous_by_id.get(relation_id) != current_by_id.get(relation_id)
    }


def _related_relation_ids(state: FormalGeometryState, affected_objects: set[str]) -> set[str]:
    related: set[str] = set()
    for relation in state.relations:
        references = set(_relation_reference_values(relation))
        if references & affected_objects:
            related.add(str(relation["id"]))
    return related


def _relation_reference_values(relation: dict) -> tuple[str, ...]:
    keys = {
        "point", "points", "line", "base_line", "circle", "circle1", "circle2",
        "segment", "segments", "polygon", "polygons", "plane", "planes",
        "polyhedron", "from", "of", "objects", "boundary", "chord",
    }
    values: list[str] = []
    for key in keys:
        value = relation.get(key)
        if isinstance(value, (list, tuple)):
            values.extend(str(item) for item in value)
        elif value is not None:
            values.append(str(value))
    mapping = relation.get("mapping")
    if isinstance(mapping, dict):
        values.extend(str(item) for item in mapping)
        values.extend(str(item) for item in mapping.values())
    return tuple(values)


def _invalidate(facts: FactStore, affected_objects: set[str], affected_relations: set[str] | None = None):
    tokens = set(affected_objects)
    tokens.update(f"{prefix}:{object_id}" for prefix in ("point", "line", "segment", "ray", "circle", "arc", "polygon", "region", "plane", "polyhedron") for object_id in affected_objects)
    relation_ids = affected_relations or set()
    impacted = {
        fact.id
        for fact in facts
        if fact.object in affected_objects
        or tokens.intersection(fact.dependencies)
        or str(fact.metadata.get("relation_id", "")) in relation_ids
        or str(fact.metadata.get("id", "")) in relation_ids
    }
    changed = True
    while changed:
        changed = False
        for fact in facts:
            if fact.id not in impacted and set(fact.dependencies) & impacted:
                impacted.add(fact.id)
                changed = True
    return facts.remove_where(lambda fact: fact.id in impacted)
