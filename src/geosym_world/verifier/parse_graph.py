"""Evidence-gated admission for typed geometry ParseGraphs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from geosym_world.errors import InvalidGeometryInput
from geosym_world.schema import (
    AdmissionStatus,
    EvidencePrecision,
    ParseConflictGroup,
    ParseEvidenceKind,
    ParseGraph,
    ParseObject,
    ParseOrigin,
    ParseQuantity,
    ParseRegion,
    ParseRelation,
    ParseTarget,
)
from geosym_world.schema.canonical import stable_checksum

from .pixel_topology import PixelTopologyReport, PixelTopologyStatus


_STRUCTURAL_IMAGE_RELATIONS = frozenset(
    {
        "boundary_of",
        "endpoint_of",
        "hidden_edge",
        "intersection",
        "ordered_points",
        "point_on_line",
        "support_of",
        "visible_extension",
    }
)
_PRECISION_RANK = {
    EvidencePrecision.IMAGE_ONLY: 0,
    EvidencePrecision.LOCALIZED: 1,
    EvidencePrecision.BOUND: 2,
}


@dataclass(frozen=True)
class ParseAdmissionPolicy:
    minimum_confidence: float = 0.0
    permit_localized_structure: bool = True
    semantic_precision: EvidencePrecision = EvidencePrecision.BOUND
    require_pixel_topology: bool = False
    policy_id: str = "strict_v1"
    permit_unique_fill_shading: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise InvalidGeometryInput("Parse admission minimum_confidence must be in [0, 1]")
        if not isinstance(self.semantic_precision, EvidencePrecision):
            raise InvalidGeometryInput(f"Unknown semantic evidence precision: {self.semantic_precision}")
        if not isinstance(self.permit_unique_fill_shading, bool):
            raise InvalidGeometryInput("Parse admission permit_unique_fill_shading must be boolean")
        if not isinstance(self.require_pixel_topology, bool):
            raise InvalidGeometryInput("Parse admission require_pixel_topology must be boolean")
        if not self.policy_id.strip():
            raise InvalidGeometryInput("Parse admission policy_id must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "minimum_confidence": self.minimum_confidence,
            "permit_localized_structure": self.permit_localized_structure,
            "permit_unique_fill_shading": self.permit_unique_fill_shading,
            "semantic_precision": self.semantic_precision.value,
            "require_pixel_topology": self.require_pixel_topology,
        }


@dataclass(frozen=True)
class ParseCandidateDecision:
    candidate_id: str
    candidate_kind: str
    decision: AdmissionStatus
    reason: str
    dependencies: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_kind": self.candidate_kind,
            "decision": self.decision.value,
            "reason": self.reason,
            "dependencies": list(self.dependencies),
            "evidence_refs": list(self.evidence_refs),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ParseGraphAdmissionReport:
    source_graph_checksum: str
    policy: ParseAdmissionPolicy
    decisions: tuple[ParseCandidateDecision, ...]
    admitted_graph: ParseGraph

    @property
    def admitted_ids(self) -> tuple[str, ...]:
        return tuple(item.candidate_id for item in self.decisions if item.decision is AdmissionStatus.ADMITTED)

    @property
    def rejected_ids(self) -> tuple[str, ...]:
        return tuple(item.candidate_id for item in self.decisions if item.decision is AdmissionStatus.REJECTED)

    @property
    def abstained_ids(self) -> tuple[str, ...]:
        return tuple(item.candidate_id for item in self.decisions if item.decision is AdmissionStatus.ABSTAINED)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": "1.0",
            "source_graph_checksum": self.source_graph_checksum,
            "policy": self.policy.to_dict(),
            "decisions": [item.to_dict() for item in self.decisions],
            "admitted_ids": list(self.admitted_ids),
            "rejected_ids": list(self.rejected_ids),
            "abstained_ids": list(self.abstained_ids),
            "admitted_graph": self.admitted_graph.to_dict(),
        }
        payload["checksum"] = stable_checksum(payload)
        return payload


class ParseGraphAdmission:
    """Admit evidence-backed candidates without mutating FormalGeometryState."""

    def __init__(self, policy: ParseAdmissionPolicy | None = None):
        self.policy = policy or ParseAdmissionPolicy()

    def admit(
        self,
        graph: ParseGraph | Mapping[str, Any],
        *,
        pixel_topology: PixelTopologyReport | None = None,
    ) -> ParseGraphAdmissionReport:
        source = graph if isinstance(graph, ParseGraph) else ParseGraph.from_dict(graph)
        if pixel_topology is not None and pixel_topology.source_graph_checksum != source.checksum():
            raise InvalidGeometryInput("Pixel topology report does not match the ParseGraph checksum")
        evidence = {item.id: item for item in source.evidence}
        candidates = _candidate_sequence(source)
        conflicts = _conflict_index(source.conflicts)
        admitted: set[str] = set()
        decisions: dict[str, ParseCandidateDecision] = {}
        pending = list(candidates)

        while pending:
            progressed = False
            next_pending = []
            for kind, candidate in pending:
                dependencies = _candidate_dependencies(candidate)
                conflict = conflicts.get(candidate.id)
                if conflict is not None and conflict.status.value == "unresolved":
                    decisions[candidate.id] = _decision(
                        candidate,
                        kind,
                        AdmissionStatus.ABSTAINED,
                        "unresolved_conflict",
                        dependencies,
                    )
                    progressed = True
                    continue
                if conflict is not None and conflict.selected != candidate.id:
                    decisions[candidate.id] = _decision(
                        candidate,
                        kind,
                        AdmissionStatus.REJECTED,
                        "conflict_not_selected",
                        dependencies,
                    )
                    progressed = True
                    continue
                if candidate.confidence < self.policy.minimum_confidence:
                    decisions[candidate.id] = _decision(
                        candidate,
                        kind,
                        AdmissionStatus.ABSTAINED,
                        "below_confidence_threshold",
                        dependencies,
                    )
                    progressed = True
                    continue
                unresolved = tuple(sorted(set(dependencies) - admitted))
                if unresolved:
                    if any(dependency in decisions for dependency in unresolved):
                        decisions[candidate.id] = _decision(
                            candidate,
                            kind,
                            AdmissionStatus.ABSTAINED,
                            "dependency_not_admitted",
                            dependencies,
                            {"unresolved_dependencies": list(unresolved)},
                        )
                        progressed = True
                    else:
                        next_pending.append((kind, candidate))
                    continue
                status, reason, metadata = self._evidence_decision(
                    source,
                    kind,
                    candidate,
                    evidence,
                    pixel_topology,
                )
                decisions[candidate.id] = _decision(candidate, kind, status, reason, dependencies, metadata)
                if status is AdmissionStatus.ADMITTED:
                    admitted.add(candidate.id)
                progressed = True
            if not progressed:
                for kind, candidate in next_pending:
                    decisions[candidate.id] = _decision(
                        candidate,
                        kind,
                        AdmissionStatus.ABSTAINED,
                        "dependency_cycle_or_unresolved",
                        _candidate_dependencies(candidate),
                    )
                break
            pending = next_pending

        _enforce_unique_point_names(candidates, decisions, admitted)
        ordered_decisions = tuple(decisions[candidate.id] for _, candidate in candidates)
        admitted_graph = _select_admitted_graph(source, admitted, self.policy)
        return ParseGraphAdmissionReport(source.checksum(), self.policy, ordered_decisions, admitted_graph)

    def _evidence_decision(
        self,
        graph: ParseGraph,
        kind: str,
        candidate: Any,
        evidence: Mapping[str, Any],
        pixel_topology: PixelTopologyReport | None,
    ) -> tuple[AdmissionStatus, str, Mapping[str, Any] | None]:
        if candidate.origin is ParseOrigin.COMPILED:
            if not _candidate_dependencies(candidate):
                return AdmissionStatus.ABSTAINED, "compiled_dependencies_required", None
            return AdmissionStatus.ADMITTED, "compiled_from_admitted_dependencies", None
        candidate_evidence = [evidence[evidence_id] for evidence_id in candidate.evidence_refs]
        if candidate.origin is ParseOrigin.NATIVE:
            if candidate_evidence and all(item.kind.value == "native_entity" for item in candidate_evidence):
                return AdmissionStatus.ADMITTED, "native_entity_evidence", None
            return AdmissionStatus.ABSTAINED, "native_evidence_required", None
        if not candidate_evidence:
            return AdmissionStatus.ABSTAINED, "missing_evidence", None
        if kind == "target" and not candidate.arguments:
            return AdmissionStatus.ABSTAINED, "target_object_unresolved", None
        if kind == "relation" and candidate.predicate == "shaded_region":
            certificate = _unique_fill_shading_certificate(graph, candidate, evidence)
            if not self.policy.permit_unique_fill_shading:
                return AdmissionStatus.ABSTAINED, "semantic_shading_unverified", certificate
            if not certificate["passed"]:
                return AdmissionStatus.REJECTED, "unique_fill_shading_verification_failed", certificate
            return AdmissionStatus.ADMITTED, "unique_fill_shading_verified", certificate
        if kind == "target" and candidate.metadata.get("semantic_shading_unverified") is True:
            certificate = _shaded_target_certificate(graph, candidate, evidence)
            if not self.policy.permit_unique_fill_shading:
                return AdmissionStatus.ABSTAINED, "semantic_shading_unverified", certificate
            if not certificate["passed"]:
                return AdmissionStatus.REJECTED, "shaded_target_verification_failed", certificate
            return AdmissionStatus.ADMITTED, "shaded_target_grounded_by_verified_relation", certificate
        topology_check = pixel_topology.check_for(candidate.id) if pixel_topology is not None else None
        if topology_check is not None:
            metadata = {"pixel_topology": topology_check.to_dict()}
            if topology_check.status is PixelTopologyStatus.FAILED:
                return AdmissionStatus.REJECTED, "pixel_topology_verification_failed", metadata
            if topology_check.status is PixelTopologyStatus.UNKNOWN:
                return AdmissionStatus.ABSTAINED, "pixel_topology_verification_unknown", metadata
        elif self.policy.require_pixel_topology and _requires_pixel_topology(kind, candidate):
            return AdmissionStatus.ABSTAINED, "pixel_topology_verification_required", None
        required = self.policy.semantic_precision
        if kind == "object" or (kind == "relation" and candidate.predicate in _STRUCTURAL_IMAGE_RELATIONS):
            if self.policy.permit_localized_structure:
                required = EvidencePrecision.LOCALIZED
        if kind == "relation" and candidate.predicate == "binds":
            precisions = {_PRECISION_RANK[item.precision] for item in candidate_evidence}
            localized = _PRECISION_RANK[EvidencePrecision.LOCALIZED]
            bound = _PRECISION_RANK[EvidencePrecision.BOUND]
            if precisions and min(precisions) >= localized and max(precisions) >= bound:
                return AdmissionStatus.ADMITTED, "bound_label_grounding", None
        minimum_precision = min(_PRECISION_RANK[item.precision] for item in candidate_evidence)
        if minimum_precision < _PRECISION_RANK[required]:
            return AdmissionStatus.ABSTAINED, f"evidence_precision_below_{required.value}", None
        if kind == "relation" and candidate.metadata.get("exact_relation_not_admitted") is True:
            return AdmissionStatus.ABSTAINED, "visual_metric_is_not_an_exact_relation", None
        return AdmissionStatus.ADMITTED, "evidence_policy_satisfied", None


def _unique_fill_shading_certificate(
    graph: ParseGraph,
    relation: ParseRelation,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    region_arguments = tuple(argument.ref for argument in relation.arguments if argument.role == "region")
    observed_regions = tuple(
        region for region in graph.regions if region.metadata.get("fill_status") == "observed"
    )
    selected_region = next(
        (region for region in observed_regions if region_arguments == (region.id,)),
        None,
    )
    relation_evidence = tuple(evidence[evidence_id] for evidence_id in relation.evidence_refs)
    text_cue = any(
        item.kind is ParseEvidenceKind.TEXT_SPAN and _contains_shaded_area_cue(item.raw_content)
        for item in relation_evidence
    )
    fill_binding = selected_region is not None and any(
        item.kind is ParseEvidenceKind.IMAGE_REGION
        and item.precision is EvidencePrecision.BOUND
        and item.metadata.get("appearance_semantics") == "observed_fill_not_shading_label"
        and selected_region.id
        in {
            item.metadata.get("bound_face_id"),
            item.metadata.get("bound_curved_region_id"),
        }
        for item in relation_evidence
    )
    checks = {
        "image_text_origin": relation.origin is ParseOrigin.IMAGE_TEXT,
        "one_region_argument": len(relation.arguments) == 1 and len(region_arguments) == 1,
        "one_observed_fill_region": len(observed_regions) == 1,
        "argument_matches_observed_region": selected_region is not None,
        "explicit_shaded_area_text": text_cue,
        "bound_fill_evidence": fill_binding,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed_region_ids": [region.id for region in observed_regions],
        "argument_region_ids": list(region_arguments),
    }


def _shaded_target_certificate(
    graph: ParseGraph,
    target: ParseTarget,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    target_regions = tuple(argument.ref for argument in target.arguments if argument.role == "region")
    shading_relations = tuple(
        relation
        for relation in graph.relations
        if relation.id in target.dependencies and relation.predicate == "shaded_region"
    )
    relation_certificates = tuple(
        _unique_fill_shading_certificate(graph, relation, evidence)
        for relation in shading_relations
    )
    target_evidence = tuple(evidence[evidence_id] for evidence_id in target.evidence_refs)
    checks = {
        "shadow_area_target": target.kind == "shadow_area" and target.mode == "value",
        "image_text_origin": target.origin is ParseOrigin.IMAGE_TEXT,
        "one_region_argument": len(target.arguments) == 1 and len(target_regions) == 1,
        "one_shading_dependency": len(shading_relations) == 1,
        "dependency_matches_target": bool(shading_relations)
        and tuple(argument.ref for argument in shading_relations[0].arguments if argument.role == "region")
        == target_regions,
        "verified_shading_dependency": len(relation_certificates) == 1
        and relation_certificates[0]["passed"],
        "explicit_shaded_area_text": any(
            item.kind is ParseEvidenceKind.TEXT_SPAN and _contains_shaded_area_cue(item.raw_content)
            for item in target_evidence
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "target_region_ids": list(target_regions),
        "shading_dependency_ids": [relation.id for relation in shading_relations],
    }


def _contains_shaded_area_cue(raw_content: Any) -> bool:
    if not isinstance(raw_content, str):
        return False
    normalized = raw_content.casefold()
    return ("shaded" in normalized and "area" in normalized) or ("阴影" in normalized and "面积" in normalized)


def _requires_pixel_topology(kind: str, candidate: Any) -> bool:
    if candidate.origin not in {ParseOrigin.IMAGE, ParseOrigin.IMAGE_TEXT}:
        return False
    if kind == "object":
        return candidate.kind.value in {"segment", "line", "ray", "circle", "arc"}
    return kind == "relation" and candidate.predicate in {"intersection", "ordered_points"}


def _candidate_sequence(graph: ParseGraph) -> tuple[tuple[str, Any], ...]:
    return (
        *(("object", item) for item in graph.objects),
        *(("quantity", item) for item in graph.quantities),
        *(("region", item) for item in graph.regions),
        *(("relation", item) for item in graph.relations),
        *(("target", item) for item in graph.targets),
    )


def _candidate_dependencies(candidate: Any) -> tuple[str, ...]:
    if isinstance(candidate, ParseObject):
        return tuple(reference.ref for reference in candidate.references)
    if isinstance(candidate, ParseQuantity):
        return tuple(dict.fromkeys((*candidate.dependencies, *(reference.ref for reference in candidate.arguments))))
    if isinstance(candidate, ParseRegion):
        region_refs = candidate.expression.references() if candidate.expression is not None else ()
        return tuple(
            dict.fromkeys(
                (
                    *(reference.ref for reference in candidate.boundary),
                    *candidate.holes,
                    *candidate.components,
                    *region_refs,
                )
            )
        )
    if isinstance(candidate, ParseRelation):
        return tuple(dict.fromkeys((*candidate.dependencies, *(argument.ref for argument in candidate.arguments))))
    if isinstance(candidate, ParseTarget):
        return tuple(dict.fromkeys((*candidate.dependencies, *(argument.ref for argument in candidate.arguments))))
    raise TypeError(f"Unsupported parse candidate: {type(candidate).__name__}")


def _decision(
    candidate: Any,
    kind: str,
    status: AdmissionStatus,
    reason: str,
    dependencies: Sequence[str],
    metadata: Mapping[str, Any] | None = None,
) -> ParseCandidateDecision:
    return ParseCandidateDecision(
        candidate_id=candidate.id,
        candidate_kind=kind,
        decision=status,
        reason=reason,
        dependencies=tuple(dependencies),
        evidence_refs=tuple(candidate.evidence_refs),
        metadata=dict(metadata or {}),
    )


def _conflict_index(conflicts: Sequence[ParseConflictGroup]) -> dict[str, ParseConflictGroup]:
    index = {}
    for conflict in conflicts:
        for member in conflict.members:
            if member in index:
                raise InvalidGeometryInput(f"Parse candidate {member} belongs to multiple conflict groups")
            index[member] = conflict
    return index


def _enforce_unique_point_names(candidates, decisions, admitted: set[str]) -> None:
    candidate_by_id = {candidate.id: (kind, candidate) for kind, candidate in candidates}
    bindings = [
        candidate
        for kind, candidate in candidates
        if kind == "relation"
        and candidate.id in admitted
        and candidate.predicate == "binds"
        and candidate.metadata.get("binding_kind") == "point_name"
    ]
    by_key: dict[tuple[str, str], list[str]] = {}
    for binding in bindings:
        label_id = next((item.ref for item in binding.arguments if item.role == "label"), None)
        point_id = next((item.ref for item in binding.arguments if item.role == "object"), None)
        visible_name = binding.metadata.get("label")
        keys = (
            ("label_object", label_id),
            ("point", point_id),
            ("visible_name", visible_name.strip().upper() if isinstance(visible_name, str) else None),
        )
        for key_kind, key_value in keys:
            if key_value is not None:
                by_key.setdefault((key_kind, key_value), []).append(binding.id)
    revoked = {
        candidate_id
        for candidate_ids in by_key.values()
        if len(candidate_ids) > 1
        for candidate_id in candidate_ids
    }
    for candidate_id in sorted(revoked):
        kind, candidate = candidate_by_id[candidate_id]
        decisions[candidate_id] = _decision(
            candidate,
            kind,
            AdmissionStatus.ABSTAINED,
            "non_unique_point_name_binding",
            _candidate_dependencies(candidate),
        )
        admitted.remove(candidate_id)

    changed = True
    while changed:
        changed = False
        for candidate_id in tuple(sorted(admitted)):
            kind, candidate = candidate_by_id[candidate_id]
            dependencies = _candidate_dependencies(candidate)
            unresolved = tuple(sorted(set(dependencies) - admitted))
            if not unresolved:
                continue
            decisions[candidate_id] = _decision(
                candidate,
                kind,
                AdmissionStatus.ABSTAINED,
                "dependency_not_admitted",
                dependencies,
                {"unresolved_dependencies": list(unresolved)},
            )
            admitted.remove(candidate_id)
            changed = True


def _select_admitted_graph(source: ParseGraph, admitted: set[str], policy: ParseAdmissionPolicy) -> ParseGraph:
    return ParseGraph(
        id=f"{source.id}:admitted",
        source_id=source.source_id,
        evidence=source.evidence,
        objects=tuple(item for item in source.objects if item.id in admitted),
        relations=tuple(item for item in source.relations if item.id in admitted),
        quantities=tuple(item for item in source.quantities if item.id in admitted),
        regions=tuple(item for item in source.regions if item.id in admitted),
        targets=tuple(item for item in source.targets if item.id in admitted),
        metadata={
            **dict(source.metadata),
            "admission": {
                "source_graph_checksum": source.checksum(),
                "policy": policy.to_dict(),
                "admitted_ids": sorted(admitted),
            },
        },
    )
