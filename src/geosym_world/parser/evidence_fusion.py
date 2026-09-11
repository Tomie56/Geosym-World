"""Conservative fusion of semantic ParseGraphs with localized visual evidence."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from geosym_world.errors import InvalidGeometryInput
from geosym_world.schema import (
    ConflictStatus,
    ParseConflictGroup,
    ParseGraph,
    ParseObject,
    ParseObjectKind,
    ParseQuantity,
    ParseReference,
    ParseRelation,
    ParseTarget,
    math_expression_from_text,
)
from geosym_world.schema.canonical import stable_checksum


PARSE_GRAPH_FUSION_VERSION = "geosym.parse_graph_fusion.v0"
PARSE_GRAPH_COMPATIBILITY_VERSION = "geosym.parse_graph_compatibility.v0"
_ANGLE_COMPATIBILITY_TOLERANCE_DEG = 5.0
_POINT_ON_CIRCLE_TOLERANCE_RATIO = 0.08
_EQUIVALENT_RELATIONS = frozenset(
    {"equal_angle", "equal_length", "intersection", "parallel", "perpendicular", "point_on_circle"}
)
_MARKER_RELATION_KINDS = {
    "equal_angle": "angle_arc",
    "equal_length": "equal_length_tick",
    "parallel": "parallel",
    "perpendicular": "right_angle",
}


@dataclass(frozen=True)
class ParseFusionDecision:
    candidate_id: str
    candidate_kind: str
    status: str
    mapped_to: str | None
    mapped_references: tuple[tuple[str, str], ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_kind": self.candidate_kind,
            "status": self.status,
            "mapped_to": self.mapped_to,
            "mapped_references": [
                {"source": source, "target": target}
                for source, target in self.mapped_references
            ],
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ParseCompatibilityCheck:
    candidate_id: str
    candidate_kind: str
    status: str
    compared_ids: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"compatible", "incompatible", "unknown"}:
            raise InvalidGeometryInput(f"Unknown parse compatibility status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_kind": self.candidate_kind,
            "status": self.status,
            "compared_ids": list(self.compared_ids),
            "metrics": dict(self.metrics),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ParseGraphCompatibilityCertificate:
    visual_graph_checksum: str
    semantic_graph_checksum: str
    checks: tuple[ParseCompatibilityCheck, ...]
    version: str = PARSE_GRAPH_COMPATIBILITY_VERSION

    def status_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(check.status for check in self.checks).items()))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "version": self.version,
            "visual_graph_checksum": self.visual_graph_checksum,
            "semantic_graph_checksum": self.semantic_graph_checksum,
            "checks": [check.to_dict() for check in self.checks],
            "status_counts": self.status_counts(),
        }
        payload["checksum"] = stable_checksum(payload)
        return payload


@dataclass(frozen=True)
class ParseGraphFusionResult:
    graph: ParseGraph
    decisions: tuple[ParseFusionDecision, ...]
    compatibility: ParseGraphCompatibilityCertificate

    def status_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(decision.status for decision in self.decisions).items()))


class ParseGraphEvidenceFusion:
    """Ground global semantic references without promoting unsupported claims."""

    def fuse(self, visual_graph: ParseGraph, semantic_graph: ParseGraph) -> ParseGraphFusionResult:
        if visual_graph.source_id != semantic_graph.source_id:
            raise InvalidGeometryInput("Visual and semantic ParseGraphs must share one source_id")
        _require_disjoint_evidence(visual_graph, semantic_graph)

        object_mapping, mapping_decisions = _map_semantic_objects(visual_graph, semantic_graph)
        local_candidates = _candidate_ids(visual_graph)
        decisions = list(mapping_decisions)

        objects = list(visual_graph.objects)
        for candidate in semantic_graph.objects:
            mapped_to = object_mapping.get(candidate.id)
            if mapped_to is not None:
                continue
            rewritten = _rewrite_object(candidate, object_mapping)
            _append_unique(objects, rewritten, local_candidates)
            decisions.append(_retained_decision(candidate.id, "object", object_mapping, candidate.references))

        relations, relation_decisions, candidate_mapping = _merge_relations(
            visual_graph.relations,
            semantic_graph.relations,
            object_mapping,
            local_candidates,
        )
        decisions.extend(relation_decisions)

        quantities, quantity_decisions, quantity_mapping = _merge_quantities(
            visual_graph.quantities,
            semantic_graph.quantities,
            object_mapping,
            candidate_mapping,
            local_candidates,
        )
        decisions.extend(quantity_decisions)
        candidate_mapping.update(quantity_mapping)

        regions = list(visual_graph.regions)
        for candidate in semantic_graph.regions:
            rewritten = _rewrite_region(candidate, object_mapping, candidate_mapping)
            _append_unique(regions, rewritten, local_candidates)
            decisions.append(
                _retained_decision(candidate.id, "region", {**object_mapping, **candidate_mapping}, ())
            )

        targets, target_decisions, target_mapping = _merge_targets(
            visual_graph.targets,
            semantic_graph.targets,
            object_mapping,
            candidate_mapping,
            local_candidates,
        )
        decisions.extend(target_decisions)
        candidate_mapping.update(target_mapping)

        conflicts = _merge_conflicts(
            visual_graph.conflicts,
            semantic_graph.conflicts,
            {**object_mapping, **candidate_mapping},
            local_candidates,
        )
        compatibility = _build_compatibility_certificate(
            visual_graph,
            semantic_graph,
            {**object_mapping, **candidate_mapping},
        )
        metadata = {
            "producer": PARSE_GRAPH_FUSION_VERSION,
            "visual_graph_checksum": visual_graph.checksum(),
            "semantic_graph_checksum": semantic_graph.checksum(),
            "policy": {
                "point_mapping": "unique_nonconflicting_visible_name_binding",
                "object_mapping": "unique_endpoint_or_ordered_support_match",
                "semantic_admission": "preserve_original_evidence_precision",
                "equivalent_candidates": "reuse_local_candidate_without_duplication",
            },
            "status_counts": dict(sorted(Counter(item.status for item in decisions).items())),
            "decisions": [item.to_dict() for item in decisions],
            "compatibility_certificate": compatibility.to_dict(),
        }
        graph = ParseGraph(
            id=f"fusion:{stable_checksum(metadata)[:20]}",
            source_id=visual_graph.source_id,
            evidence=(*visual_graph.evidence, *semantic_graph.evidence),
            objects=tuple(objects),
            relations=tuple(relations),
            quantities=tuple(quantities),
            regions=tuple(regions),
            targets=tuple(targets),
            conflicts=tuple(conflicts),
            metadata=metadata,
        )
        return ParseGraphFusionResult(graph, tuple(decisions), compatibility)


def _build_compatibility_certificate(
    visual_graph: ParseGraph,
    semantic_graph: ParseGraph,
    mapping: Mapping[str, str],
) -> ParseGraphCompatibilityCertificate:
    local_objects = {candidate.id: candidate for candidate in visual_graph.objects}
    local_relations = tuple(visual_graph.relations)
    local_quantities = tuple(visual_graph.quantities)
    local_regions = tuple(visual_graph.regions)
    local_targets = tuple(visual_graph.targets)
    checks: list[ParseCompatibilityCheck] = []

    for candidate in semantic_graph.objects:
        local_id = mapping.get(candidate.id)
        if local_id is not None:
            label_check = _label_binding_compatibility(
                candidate,
                local_id,
                local_objects,
                local_relations,
            )
            if label_check is not None:
                checks.append(label_check)
                continue
            checks.append(
                _compatibility_check(
                    candidate.id,
                    "object",
                    "compatible",
                    (local_id,),
                    "semantic_object_mapped_to_local_candidate",
                )
            )
        else:
            checks.append(
                _compatibility_check(
                    candidate.id,
                    "object",
                    "unknown",
                    _mapped_ids(candidate.references, mapping),
                    "no_unique_local_object_mapping",
                )
            )

    for candidate in semantic_graph.relations:
        checks.append(_check_relation_compatibility(candidate, local_objects, local_relations, mapping))
    for candidate in semantic_graph.quantities:
        checks.append(_check_quantity_compatibility(candidate, local_objects, local_quantities, mapping))
    for candidate in semantic_graph.regions:
        checks.append(_check_region_compatibility(candidate, local_regions, mapping))
    for candidate in semantic_graph.targets:
        checks.append(_check_target_compatibility(candidate, local_targets, mapping))

    return ParseGraphCompatibilityCertificate(
        visual_graph_checksum=visual_graph.checksum(),
        semantic_graph_checksum=semantic_graph.checksum(),
        checks=tuple(checks),
    )


def _label_binding_compatibility(
    candidate: ParseObject,
    local_id: str,
    local_objects: Mapping[str, ParseObject],
    local_relations: Sequence[ParseRelation],
) -> ParseCompatibilityCheck | None:
    if candidate.kind not in {ParseObjectKind.POINT, ParseObjectKind.LINE} or candidate.visible_name is None:
        return None
    binding_kind = "point_name" if candidate.kind is ParseObjectKind.POINT else "line_name"
    candidate_kind = "point" if candidate.kind is ParseObjectKind.POINT else "line"
    bindings = []
    for relation in local_relations:
        if relation.predicate != "binds" or relation.metadata.get("binding_kind") != binding_kind:
            continue
        object_id = next((item.ref for item in relation.arguments if item.role == "object"), None)
        label_id = next((item.ref for item in relation.arguments if item.role == "label"), None)
        if object_id != local_id:
            continue
        label = local_objects.get(label_id)
        if label is not None and label.kind is ParseObjectKind.TEXT:
            bindings.append((relation, label))
    compared_ids = [local_id, *(relation.id for relation, _ in bindings)]
    if len(bindings) != 1:
        return _compatibility_check(
            candidate.id,
            "object",
            "unknown",
            compared_ids,
            f"mapped_{candidate_kind}_lacks_unique_visual_label_binding",
            {"expected_label": candidate.visible_name, "binding_count": len(bindings)},
        )
    binding, label = bindings[0]
    expected = candidate.visible_name.strip().upper()
    observed = (label.visible_name or binding.metadata.get("label") or "").strip().upper()
    compared_ids.append(label.id)
    if expected != observed:
        return _compatibility_check(
            candidate.id,
            "object",
            "incompatible",
            compared_ids,
            f"visual_{candidate_kind}_label_text_does_not_match_semantic_name",
            {"expected_label": candidate.visible_name, "observed_label": label.visible_name},
        )
    return _compatibility_check(
        candidate.id,
        "object",
        "compatible",
        compared_ids,
        f"semantic_{candidate_kind}_matches_grounded_visual_label",
        {"label": label.visible_name, "label_binding_id": binding.id},
    )


def _compatibility_check(
    candidate_id: str,
    candidate_kind: str,
    status: str,
    compared_ids: Sequence[str],
    reason: str,
    metrics: Mapping[str, Any] | None = None,
) -> ParseCompatibilityCheck:
    return ParseCompatibilityCheck(
        candidate_id=candidate_id,
        candidate_kind=candidate_kind,
        status=status,
        compared_ids=tuple(dict.fromkeys(compared_ids)),
        metrics=dict(metrics or {}),
        reason=reason,
    )


def _check_relation_compatibility(
    candidate: ParseRelation,
    local_objects: Mapping[str, ParseObject],
    local_relations: Sequence[ParseRelation],
    mapping: Mapping[str, str],
) -> ParseCompatibilityCheck:
    mapped_arguments = _mapped_argument_ids(candidate.arguments, mapping)
    if candidate.id in mapping:
        local_match = next(
            (relation for relation in local_relations if relation.id == mapping[candidate.id]),
            None,
        )
        marker_check = _marker_relation_compatibility(
            candidate,
            local_match,
            local_objects,
            mapped_arguments,
        )
        if marker_check is not None:
            return marker_check
        return _compatibility_check(
            candidate.id,
            "relation",
            "compatible",
            (mapping[candidate.id],),
            "semantic_relation_matched_existing_local_candidate",
        )

    if mapped_arguments is None:
        return _compatibility_check(
            candidate.id,
            "relation",
            "unknown",
            (),
            "relation_arguments_are_not_uniquely_grounded",
        )
    local_match = next(
        (
            relation
            for relation in local_relations
            if relation.predicate == candidate.predicate
            and sorted(argument.ref for argument in relation.arguments) == sorted(mapped_arguments)
        ),
        None,
    )
    if local_match is not None:
        marker_check = _marker_relation_compatibility(
            candidate,
            local_match,
            local_objects,
            mapped_arguments,
        )
        if marker_check is not None:
            return marker_check
        return _compatibility_check(
            candidate.id,
            "relation",
            "compatible",
            (local_match.id,),
            "same_typed_local_relation_after_reference_mapping",
        )

    if candidate.predicate in {"parallel", "perpendicular"}:
        return _linear_relation_compatibility(candidate, mapped_arguments, local_objects)
    if candidate.predicate == "point_on_circle":
        return _point_on_circle_compatibility(candidate.id, mapped_arguments, local_objects)
    return _compatibility_check(
        candidate.id,
        "relation",
        "unknown",
        mapped_arguments,
        "no_local_relation_or_supported_geometric_certificate",
    )


def _marker_relation_compatibility(
    candidate: ParseRelation,
    local_relation: ParseRelation | None,
    local_objects: Mapping[str, ParseObject],
    mapped_arguments: Sequence[str] | None,
) -> ParseCompatibilityCheck | None:
    if local_relation is None or candidate.predicate not in _MARKER_RELATION_KINDS:
        return None
    marker_ids = _relation_marker_ids(local_relation)
    if not marker_ids:
        return None
    markers = tuple(
        local_objects[marker_id]
        for marker_id in marker_ids
        if marker_id in local_objects and local_objects[marker_id].kind is ParseObjectKind.MARKER
    )
    expected_kind = _MARKER_RELATION_KINDS[candidate.predicate]
    metrics = {
        "marker_count": len(markers),
        "marker_kinds": sorted(
            {str(marker.metadata.get("marker_kind")) for marker in markers}
        ),
    }
    if len(markers) != len(marker_ids):
        return _compatibility_check(
            candidate.id,
            "relation",
            "unknown",
            (local_relation.id, *marker_ids),
            "local_marker_dependency_is_missing",
            metrics,
        )
    if any(marker.metadata.get("marker_kind") != expected_kind for marker in markers):
        return _compatibility_check(
            candidate.id,
            "relation",
            "incompatible",
            (local_relation.id, *marker_ids),
            "local_marker_kind_does_not_match_relation",
            metrics,
        )
    if not _marker_supports_relation(mapped_arguments, local_relation, markers):
        return _compatibility_check(
            candidate.id,
            "relation",
            "unknown",
            (local_relation.id, *marker_ids),
            "local_marker_support_is_not_fully_grounded",
            metrics,
        )
    return _compatibility_check(
        candidate.id,
        "relation",
        "compatible",
        (local_relation.id, *marker_ids),
        "semantic_relation_matches_grounded_visual_marker",
        metrics,
    )


def _relation_marker_ids(relation: ParseRelation) -> tuple[str, ...]:
    metadata_ids = relation.metadata.get("marker_ids")
    if isinstance(metadata_ids, Sequence) and not isinstance(metadata_ids, (str, bytes)):
        ids = tuple(str(marker_id) for marker_id in metadata_ids if str(marker_id))
    else:
        ids = ()
    metadata_id = relation.metadata.get("marker_id")
    if isinstance(metadata_id, str) and metadata_id:
        ids += (metadata_id,)
    ids += tuple(str(item) for item in relation.dependencies if str(item).startswith("marker:"))
    return tuple(dict.fromkeys(ids))


def _marker_supports_relation(
    mapped_arguments: Sequence[str] | None,
    local_relation: ParseRelation,
    markers: Sequence[ParseObject],
) -> bool:
    if mapped_arguments is None:
        return False
    candidate_refs = set(mapped_arguments)
    local_refs = {argument.ref for argument in local_relation.arguments}
    if candidate_refs != local_refs:
        return False
    if local_relation.predicate in {"parallel", "perpendicular"}:
        support_refs = set()
        for marker in markers:
            for line_id in marker.metadata.get("support_line_ids") or ():
                raw_ref = str(line_id)
                segment_ref = f"segment:{raw_ref}"
                if raw_ref in candidate_refs:
                    support_refs.add(raw_ref)
                elif segment_ref in candidate_refs:
                    support_refs.add(segment_ref)
                else:
                    return False
        return support_refs == candidate_refs
    if local_relation.predicate == "equal_length":
        marked_refs = {
            str(marker.metadata.get("marked_segment_id"))
            for marker in markers
            if marker.metadata.get("marked_segment_id")
        }
        return len(marked_refs) == len(markers) == 2 and marked_refs == candidate_refs
    if local_relation.predicate == "equal_angle":
        angle_refs = {
            str(point_id)
            for marker in markers
            for point_id in marker.metadata.get("ordered_angle") or ()
        }
        return len(markers) == 2 and angle_refs == candidate_refs
    return False


def _linear_relation_compatibility(
    candidate: ParseRelation,
    mapped_arguments: Sequence[str],
    local_objects: Mapping[str, ParseObject],
) -> ParseCompatibilityCheck:
    supports = [
        local_objects[object_id]
        for object_id in mapped_arguments
        if object_id in local_objects and local_objects[object_id].kind in {
            ParseObjectKind.LINE,
            ParseObjectKind.SEGMENT,
            ParseObjectKind.RAY,
        }
    ]
    angles = [_object_angle(item) for item in supports]
    if len(supports) != 2 or any(angle is None for angle in angles):
        return _compatibility_check(
            candidate.id,
            "relation",
            "unknown",
            mapped_arguments,
            "local_linear_support_lacks_pixel_direction",
        )
    delta = _undirected_angle_distance(angles[0], angles[1])
    expected = 0.0 if candidate.predicate == "parallel" else 90.0
    residual = abs(delta - expected)
    status = "compatible" if residual <= _ANGLE_COMPATIBILITY_TOLERANCE_DEG else "incompatible"
    return _compatibility_check(
        candidate.id,
        "relation",
        status,
        mapped_arguments,
        f"local_{candidate.predicate}_angle_residual",
        {"angle_delta_deg": round(delta, 6), "residual_deg": round(residual, 6)},
    )


def _point_on_circle_compatibility(
    candidate_id: str,
    mapped_arguments: Sequence[str],
    local_objects: Mapping[str, ParseObject],
) -> ParseCompatibilityCheck:
    point = next(
        (
            local_objects[object_id]
            for object_id in mapped_arguments
            if object_id in local_objects and local_objects[object_id].kind is ParseObjectKind.POINT
        ),
        None,
    )
    circle = next(
        (
            local_objects[object_id]
            for object_id in mapped_arguments
            if object_id in local_objects and local_objects[object_id].kind is ParseObjectKind.CIRCLE
        ),
        None,
    )
    point_xy = _pixel_point(point) if point is not None else None
    circle_data = circle.geometry if circle is not None else {}
    center = _finite_pair(circle_data.get("center"))
    radius = _finite_scalar(circle_data.get("radius_px"))
    if point_xy is None or center is None or radius is None or radius <= 0.0:
        return _compatibility_check(
            candidate_id,
            "relation",
            "unknown",
            mapped_arguments,
            "point_or_circle_lacks_pixel_geometry",
        )
    residual = abs(math.dist(point_xy, center) - radius)
    tolerance = max(1.0, radius * _POINT_ON_CIRCLE_TOLERANCE_RATIO)
    status = "compatible" if residual <= tolerance else "incompatible"
    return _compatibility_check(
        candidate_id,
        "relation",
        status,
        mapped_arguments,
        "local_point_circle_residual",
        {"radius_residual_px": round(residual, 6), "tolerance_px": round(tolerance, 6)},
    )


def _check_quantity_compatibility(
    candidate: ParseQuantity,
    local_objects: Mapping[str, ParseObject],
    local_quantities: Sequence[ParseQuantity],
    mapping: Mapping[str, str],
) -> ParseCompatibilityCheck:
    if candidate.id in mapping:
        local_match = next(
            (quantity for quantity in local_quantities if quantity.id == mapping[candidate.id]),
            None,
        )
        label_check = _quantity_label_compatibility(candidate, local_match, local_objects)
        if label_check is not None:
            return label_check
        return _compatibility_check(
            candidate.id,
            "quantity",
            "compatible",
            (mapping[candidate.id],),
            "semantic_quantity_matched_existing_local_candidate",
        )
    mapped_arguments = _mapped_argument_ids(candidate.arguments, mapping)
    if mapped_arguments is None:
        return _compatibility_check(
            candidate.id,
            "quantity",
            "unknown",
            (),
            "quantity_arguments_are_not_uniquely_grounded",
        )
    match = next(
        (
            quantity
            for quantity in local_quantities
            if quantity.kind == candidate.kind
            and quantity.unit == candidate.unit
            and str(quantity.expression.to_sympy()) == str(candidate.expression.to_sympy())
            and sorted(argument.ref for argument in quantity.arguments) == sorted(mapped_arguments)
        ),
        None,
    )
    if match is not None:
        label_check = _quantity_label_compatibility(candidate, match, local_objects)
        if label_check is not None:
            return label_check
        return _compatibility_check(
            candidate.id,
            "quantity",
            "compatible",
            (match.id,),
            "same_typed_local_quantity_after_reference_mapping",
        )
    return _compatibility_check(
        candidate.id,
        "quantity",
        "unknown",
        mapped_arguments,
        "no_local_quantity_match_without_scale_assumption",
    )


def _quantity_label_compatibility(
    candidate: ParseQuantity,
    local_quantity: ParseQuantity | None,
    local_objects: Mapping[str, ParseObject],
) -> ParseCompatibilityCheck | None:
    if local_quantity is None:
        return _compatibility_check(
            candidate.id,
            "quantity",
            "unknown",
            (),
            "local_quantity_mapping_is_missing",
        )
    label_id = local_quantity.metadata.get("label_object_id")
    if not isinstance(label_id, str) or not label_id:
        return None
    label = local_objects.get(label_id)
    if label is None or label.kind is not ParseObjectKind.TEXT:
        return _compatibility_check(
            candidate.id,
            "quantity",
            "unknown",
            (local_quantity.id, label_id),
            "local_quantity_label_object_is_missing",
        )
    raw_text = label.visible_name
    if raw_text is None:
        return _compatibility_check(
            candidate.id,
            "quantity",
            "unknown",
            (local_quantity.id, label.id),
            "local_quantity_label_text_is_unresolved",
        )
    try:
        label_expression = math_expression_from_text(raw_text)
    except InvalidGeometryInput:
        return _compatibility_check(
            candidate.id,
            "quantity",
            "unknown",
            (local_quantity.id, label.id),
            "local_quantity_label_expression_is_invalid",
            {"label_text": raw_text},
        )
    if str(label_expression.to_sympy()) != str(candidate.expression.to_sympy()):
        return _compatibility_check(
            candidate.id,
            "quantity",
            "incompatible",
            (local_quantity.id, label.id),
            "visual_quantity_label_expression_does_not_match_semantic_expression",
            {
                "label_text": raw_text,
                "visual_expression": str(label_expression.to_sympy()),
                "semantic_expression": str(candidate.expression.to_sympy()),
            },
        )
    return _compatibility_check(
        candidate.id,
        "quantity",
        "compatible",
        (local_quantity.id, label.id),
        "semantic_quantity_matches_grounded_visual_expression",
        {"label_text": raw_text, "label_id": label.id},
    )


def _check_region_compatibility(
    candidate: Any,
    local_regions: Sequence[Any],
    mapping: Mapping[str, str],
) -> ParseCompatibilityCheck:
    if candidate.id in mapping:
        return _compatibility_check(
            candidate.id,
            "region",
            "compatible",
            (mapping[candidate.id],),
            "semantic_region_matched_existing_local_candidate",
        )
    refs = set(candidate.references())
    mapped_refs = {mapping.get(ref) for ref in refs}
    if refs and None not in mapped_refs:
        for local in local_regions:
            if set(local.references()) == mapped_refs:
                return _compatibility_check(
                    candidate.id,
                    "region",
                    "compatible",
                    tuple(sorted(mapped_refs)),
                    "same_region_references_after_mapping",
                )
        return _compatibility_check(
            candidate.id,
            "region",
            "unknown",
            tuple(sorted(mapped_refs)),
            "region_references_grounded_without_local_region_match",
        )
    return _compatibility_check(candidate.id, "region", "unknown", (), "region_references_are_not_grounded")


def _check_target_compatibility(
    candidate: Any,
    local_targets: Sequence[Any],
    mapping: Mapping[str, str],
) -> ParseCompatibilityCheck:
    if candidate.id in mapping:
        return _compatibility_check(
            candidate.id,
            "target",
            "compatible",
            (mapping[candidate.id],),
            "semantic_target_matched_existing_local_candidate",
        )
    mapped_arguments = _mapped_argument_ids(candidate.arguments, mapping)
    if mapped_arguments is None:
        return _compatibility_check(candidate.id, "target", "unknown", (), "target_object_is_not_grounded")
    for local in local_targets:
        if local.kind == candidate.kind and sorted(argument.ref for argument in local.arguments) == sorted(mapped_arguments):
            return _compatibility_check(
                candidate.id,
                "target",
                "compatible",
                (local.id,),
                "same_target_after_reference_mapping",
            )
    return _compatibility_check(
        candidate.id,
        "target",
        "unknown",
        mapped_arguments,
        "target_object_references_are_grounded_without_local_target_match",
    )


def _mapped_argument_ids(
    arguments: Sequence[ParseReference],
    mapping: Mapping[str, str],
) -> tuple[str, ...] | None:
    mapped = tuple(mapping.get(argument.ref) for argument in arguments)
    if not mapped or any(item is None for item in mapped):
        return None
    return tuple(item for item in mapped if item is not None)


def _mapped_ids(references: Sequence[ParseReference], mapping: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(mapping[reference.ref] for reference in references if reference.ref in mapping)


def _object_angle(candidate: ParseObject) -> float | None:
    angle = _finite_scalar(candidate.geometry.get("angle_deg"))
    if angle is not None:
        return angle % 180.0
    first = _finite_pair(candidate.geometry.get("p1"))
    second = _finite_pair(candidate.geometry.get("p2"))
    if first is None or second is None or first == second:
        return None
    return math.degrees(math.atan2(second[1] - first[1], second[0] - first[0])) % 180.0


def _pixel_point(candidate: ParseObject | None) -> tuple[float, float] | None:
    if candidate is None:
        return None
    return _finite_pair(candidate.geometry.get("pixel"))


def _finite_pair(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return None
    try:
        pair = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    return pair if all(math.isfinite(item) for item in pair) else None


def _finite_scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return scalar if math.isfinite(scalar) else None


def _undirected_angle_distance(left: float | None, right: float | None) -> float:
    if left is None or right is None:
        raise ValueError("Angle residual requires two finite angles")
    delta = abs(left - right) % 180.0
    return min(delta, 180.0 - delta)


def _map_semantic_objects(
    visual_graph: ParseGraph,
    semantic_graph: ParseGraph,
) -> tuple[dict[str, str], tuple[ParseFusionDecision, ...]]:
    mapping: dict[str, str] = {}
    decisions: list[ParseFusionDecision] = []
    local_names = _unique_local_point_names(visual_graph)
    for candidate in semantic_graph.objects:
        if candidate.kind is not ParseObjectKind.POINT or candidate.visible_name is None:
            continue
        local_id = local_names.get(candidate.visible_name.strip().upper())
        if local_id is None:
            continue
        mapping[candidate.id] = local_id
        decisions.append(
            ParseFusionDecision(
                candidate_id=candidate.id,
                candidate_kind="object",
                status="mapped_to_local",
                mapped_to=local_id,
                mapped_references=(),
                reason="unique_nonconflicting_point_name_binding",
            )
        )

    local_objects = {candidate.id: candidate for candidate in visual_graph.objects}
    ordered_supports = _ordered_support_index(visual_graph)
    pending = [candidate for candidate in semantic_graph.objects if candidate.id not in mapping]
    for candidate in pending:
        local_id = _map_object(candidate, mapping, local_objects, ordered_supports)
        if local_id is None:
            continue
        mapping[candidate.id] = local_id
        decisions.append(
            ParseFusionDecision(
                candidate_id=candidate.id,
                candidate_kind="object",
                status="mapped_to_local",
                mapped_to=local_id,
                mapped_references=_mapped_references(candidate.references, mapping),
                reason=f"unique_{candidate.kind.value}_support_match",
            )
        )
    return mapping, tuple(decisions)


def _unique_local_point_names(graph: ParseGraph) -> dict[str, str]:
    unresolved_members = {
        member
        for conflict in graph.conflicts
        if conflict.status is ConflictStatus.UNRESOLVED
        for member in conflict.members
    }
    objects = {candidate.id: candidate for candidate in graph.objects}
    bindings: list[tuple[str, str]] = []
    for relation in graph.relations:
        if (
            relation.id in unresolved_members
            or relation.predicate != "binds"
            or relation.metadata.get("binding_kind") != "point_name"
        ):
            continue
        point_id = next((item.ref for item in relation.arguments if item.role == "object"), None)
        label_id = next((item.ref for item in relation.arguments if item.role == "label"), None)
        point = objects.get(point_id)
        label = objects.get(label_id)
        name = relation.metadata.get("label")
        if not isinstance(name, str) and label is not None:
            name = label.visible_name
        if point is None or point.kind is not ParseObjectKind.POINT or not isinstance(name, str):
            continue
        bindings.append((name.strip().upper(), point.id))

    name_counts = Counter(name for name, _ in bindings)
    point_counts = Counter(point_id for _, point_id in bindings)
    return {
        name: point_id
        for name, point_id in bindings
        if name and name_counts[name] == 1 and point_counts[point_id] == 1
    }


def _map_object(
    candidate: ParseObject,
    mapping: Mapping[str, str],
    local_objects: Mapping[str, ParseObject],
    ordered_supports: Sequence[tuple[str, tuple[str, ...]]],
) -> str | None:
    mapped_refs = tuple(mapping.get(reference.ref) for reference in candidate.references)
    if not mapped_refs or any(reference is None for reference in mapped_refs):
        return None
    refs = tuple(reference for reference in mapped_refs if reference is not None)
    if candidate.kind in {ParseObjectKind.SEGMENT, ParseObjectKind.RAY} and len(refs) == 2:
        matches = [
            local.id
            for local in local_objects.values()
            if local.kind in {ParseObjectKind.SEGMENT, candidate.kind}
            and set(_endpoint_refs(local)) == set(refs)
        ]
        return matches[0] if len(matches) == 1 else None
    if candidate.kind is ParseObjectKind.LINE:
        matches = [
            support_id
            for support_id, ordered_points in ordered_supports
            if _is_ordered_subsequence(refs, ordered_points)
            or _is_ordered_subsequence(tuple(reversed(refs)), ordered_points)
        ]
        return matches[0] if len(matches) == 1 else None
    if candidate.kind is ParseObjectKind.CIRCLE:
        center = next(
            (mapping.get(reference.ref) for reference in candidate.references if reference.role == "center"),
            None,
        )
        matches = [
            local.id
            for local in local_objects.values()
            if local.kind is ParseObjectKind.CIRCLE
            and next((item.ref for item in local.references if item.role == "center"), None) == center
        ]
        return matches[0] if len(matches) == 1 else None
    return None


def _ordered_support_index(graph: ParseGraph) -> tuple[tuple[str, tuple[str, ...]], ...]:
    out = []
    for relation in graph.relations:
        if relation.predicate != "ordered_points":
            continue
        support = next((item.ref for item in relation.arguments if item.role == "support"), None)
        points = tuple(item.ref for item in relation.arguments if item.role.startswith("point_"))
        if support is not None and len(points) >= 2:
            out.append((support, points))
    return tuple(out)


def _merge_relations(
    visual: Sequence[ParseRelation],
    semantic: Sequence[ParseRelation],
    object_mapping: Mapping[str, str],
    local_candidates: set[str],
) -> tuple[list[ParseRelation], list[ParseFusionDecision], dict[str, str]]:
    merged = list(visual)
    decisions = []
    mapping: dict[str, str] = {}
    local_index = _unique_signature_index(visual, _relation_signature)
    for candidate in semantic:
        rewritten = _rewrite_relation(candidate, object_mapping, mapping)
        signature = _relation_signature(rewritten)
        equivalent = local_index.get(signature) if signature is not None else None
        if equivalent is not None:
            mapping[candidate.id] = equivalent
    for candidate in semantic:
        equivalent = mapping.get(candidate.id)
        if equivalent is not None:
            decisions.append(
                _equivalent_decision(candidate.id, "relation", equivalent, object_mapping, candidate.arguments)
            )
            continue
        rewritten = _rewrite_relation(candidate, object_mapping, mapping)
        _append_unique(merged, rewritten, local_candidates)
        decisions.append(_retained_decision(candidate.id, "relation", object_mapping, candidate.arguments))
    return merged, decisions, mapping


def _merge_quantities(
    visual: Sequence[ParseQuantity],
    semantic: Sequence[ParseQuantity],
    object_mapping: Mapping[str, str],
    candidate_mapping: Mapping[str, str],
    local_candidates: set[str],
) -> tuple[list[ParseQuantity], list[ParseFusionDecision], dict[str, str]]:
    merged = list(visual)
    decisions = []
    mapping: dict[str, str] = {}
    local_index = _unique_signature_index(visual, _quantity_signature)
    for candidate in semantic:
        rewritten = _rewrite_quantity(candidate, object_mapping, candidate_mapping)
        equivalent = local_index.get(_quantity_signature(rewritten))
        if equivalent is not None:
            mapping[candidate.id] = equivalent
    full_mapping = {**candidate_mapping, **mapping}
    for candidate in semantic:
        equivalent = mapping.get(candidate.id)
        if equivalent is not None:
            decisions.append(
                _equivalent_decision(candidate.id, "quantity", equivalent, object_mapping, candidate.arguments)
            )
            continue
        rewritten = _rewrite_quantity(candidate, object_mapping, full_mapping)
        _append_unique(merged, rewritten, local_candidates)
        decisions.append(_retained_decision(candidate.id, "quantity", object_mapping, candidate.arguments))
    return merged, decisions, mapping


def _merge_targets(
    visual: Sequence[ParseTarget],
    semantic: Sequence[ParseTarget],
    object_mapping: Mapping[str, str],
    candidate_mapping: Mapping[str, str],
    local_candidates: set[str],
) -> tuple[list[ParseTarget], list[ParseFusionDecision], dict[str, str]]:
    merged = list(visual)
    decisions = []
    mapping: dict[str, str] = {}
    local_index = _unique_signature_index(visual, _target_signature)
    for candidate in semantic:
        rewritten = _rewrite_target(candidate, object_mapping, candidate_mapping)
        equivalent = local_index.get(_target_signature(rewritten))
        if equivalent is not None:
            mapping[candidate.id] = equivalent
    full_mapping = {**candidate_mapping, **mapping}
    for candidate in semantic:
        equivalent = mapping.get(candidate.id)
        if equivalent is not None:
            decisions.append(
                _equivalent_decision(candidate.id, "target", equivalent, object_mapping, candidate.arguments)
            )
            continue
        rewritten = _rewrite_target(candidate, object_mapping, full_mapping)
        _append_unique(merged, rewritten, local_candidates)
        decisions.append(_retained_decision(candidate.id, "target", object_mapping, candidate.arguments))
    return merged, decisions, mapping


def _rewrite_object(candidate: ParseObject, mapping: Mapping[str, str]) -> ParseObject:
    return replace(
        candidate,
        references=_rewrite_references(candidate.references, mapping),
        metadata={**dict(candidate.metadata), "fusion_reference_mapping": _mapping_metadata(candidate.references, mapping)},
    )


def _rewrite_relation(
    candidate: ParseRelation,
    object_mapping: Mapping[str, str],
    candidate_mapping: Mapping[str, str],
) -> ParseRelation:
    mapping = {**object_mapping, **candidate_mapping}
    return replace(
        candidate,
        arguments=_rewrite_references(candidate.arguments, mapping),
        dependencies=tuple(mapping.get(item, item) for item in candidate.dependencies),
        metadata={**dict(candidate.metadata), "fusion_reference_mapping": _mapping_metadata(candidate.arguments, mapping)},
    )


def _rewrite_quantity(
    candidate: ParseQuantity,
    object_mapping: Mapping[str, str],
    candidate_mapping: Mapping[str, str],
) -> ParseQuantity:
    mapping = {**object_mapping, **candidate_mapping}
    return replace(
        candidate,
        arguments=_rewrite_references(candidate.arguments, mapping),
        dependencies=tuple(mapping.get(item, item) for item in candidate.dependencies),
        metadata={**dict(candidate.metadata), "fusion_reference_mapping": _mapping_metadata(candidate.arguments, mapping)},
    )


def _rewrite_region(candidate, object_mapping: Mapping[str, str], candidate_mapping: Mapping[str, str]):
    mapping = {**object_mapping, **candidate_mapping}
    boundary = tuple(replace(reference, ref=mapping.get(reference.ref, reference.ref)) for reference in candidate.boundary)
    return replace(
        candidate,
        boundary=boundary,
        holes=tuple(mapping.get(item, item) for item in candidate.holes),
        components=tuple(mapping.get(item, item) for item in candidate.components),
        metadata={**dict(candidate.metadata), "fusion_reference_mapping": {}},
    )


def _rewrite_target(
    candidate: ParseTarget,
    object_mapping: Mapping[str, str],
    candidate_mapping: Mapping[str, str],
) -> ParseTarget:
    mapping = {**object_mapping, **candidate_mapping}
    return replace(
        candidate,
        arguments=_rewrite_references(candidate.arguments, mapping),
        dependencies=tuple(mapping.get(item, item) for item in candidate.dependencies),
        metadata={**dict(candidate.metadata), "fusion_reference_mapping": _mapping_metadata(candidate.arguments, mapping)},
    )


def _rewrite_references(
    references: Sequence[ParseReference],
    mapping: Mapping[str, str],
) -> tuple[ParseReference, ...]:
    return tuple(replace(reference, ref=mapping.get(reference.ref, reference.ref)) for reference in references)


def _mapping_metadata(
    references: Sequence[ParseReference],
    mapping: Mapping[str, str],
) -> dict[str, str]:
    return {
        reference.ref: mapping[reference.ref]
        for reference in references
        if reference.ref in mapping and mapping[reference.ref] != reference.ref
    }


def _mapped_references(
    references: Sequence[ParseReference],
    mapping: Mapping[str, str],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (reference.ref, mapping[reference.ref])
        for reference in references
        if reference.ref in mapping and mapping[reference.ref] != reference.ref
    )


def _retained_decision(
    candidate_id: str,
    candidate_kind: str,
    mapping: Mapping[str, str],
    references: Sequence[ParseReference],
) -> ParseFusionDecision:
    mapped = _mapped_references(references, mapping)
    return ParseFusionDecision(
        candidate_id=candidate_id,
        candidate_kind=candidate_kind,
        status="retained_with_grounded_references" if mapped else "retained_unresolved",
        mapped_to=None,
        mapped_references=mapped,
        reason=(
            "semantic_candidate_preserves_original_evidence_precision"
            if mapped
            else "no_unique_local_equivalent"
        ),
    )


def _equivalent_decision(
    candidate_id: str,
    candidate_kind: str,
    local_id: str,
    mapping: Mapping[str, str],
    references: Sequence[ParseReference],
) -> ParseFusionDecision:
    return ParseFusionDecision(
        candidate_id=candidate_id,
        candidate_kind=candidate_kind,
        status="matched_existing_local_candidate",
        mapped_to=local_id,
        mapped_references=_mapped_references(references, mapping),
        reason="same_typed_signature_after_reference_grounding",
    )


def _relation_signature(candidate: ParseRelation) -> tuple[Any, ...] | None:
    if candidate.predicate not in _EQUIVALENT_RELATIONS:
        return None
    return candidate.predicate, tuple(sorted(argument.ref for argument in candidate.arguments))


def _quantity_signature(candidate: ParseQuantity) -> tuple[Any, ...]:
    return (
        candidate.kind,
        tuple(sorted(argument.ref for argument in candidate.arguments)),
        str(candidate.expression.to_sympy()),
        candidate.unit,
    )


def _target_signature(candidate: ParseTarget) -> tuple[Any, ...]:
    return (
        candidate.kind,
        candidate.mode,
        tuple(sorted(argument.ref for argument in candidate.arguments)),
        str(candidate.expression.to_sympy()) if candidate.expression is not None else None,
    )


def _unique_signature_index(candidates: Sequence[Any], signature_fn) -> dict[tuple[Any, ...], str]:
    grouped: dict[tuple[Any, ...], list[str]] = {}
    for candidate in candidates:
        signature = signature_fn(candidate)
        if signature is not None:
            grouped.setdefault(signature, []).append(candidate.id)
    return {signature: ids[0] for signature, ids in grouped.items() if len(ids) == 1}


def _endpoint_refs(candidate: ParseObject) -> tuple[str, ...]:
    return tuple(reference.ref for reference in candidate.references if reference.role in {"start", "end"})


def _is_ordered_subsequence(needle: Sequence[str], haystack: Sequence[str]) -> bool:
    cursor = iter(haystack)
    return all(any(value == expected for value in cursor) for expected in needle)


def _merge_conflicts(
    visual: Sequence[ParseConflictGroup],
    semantic: Sequence[ParseConflictGroup],
    mapping: Mapping[str, str],
    local_candidates: set[str],
) -> list[ParseConflictGroup]:
    merged = list(visual)
    for conflict in semantic:
        members = tuple(dict.fromkeys(mapping.get(member, member) for member in conflict.members))
        if len(members) < 2:
            continue
        selected = mapping.get(conflict.selected, conflict.selected) if conflict.selected is not None else None
        rewritten = replace(conflict, members=members, selected=selected)
        _append_unique(merged, rewritten, local_candidates)
    return merged


def _append_unique(candidates: list[Any], candidate: Any, known_ids: set[str]) -> None:
    if candidate.id in known_ids:
        raise InvalidGeometryInput(f"Fusion candidate id collides with an existing id: {candidate.id}")
    candidates.append(candidate)
    known_ids.add(candidate.id)


def _candidate_ids(graph: ParseGraph) -> set[str]:
    return {
        candidate.id
        for candidates in (
            graph.objects,
            graph.relations,
            graph.quantities,
            graph.regions,
            graph.targets,
            graph.conflicts,
        )
        for candidate in candidates
    }


def _require_disjoint_evidence(visual_graph: ParseGraph, semantic_graph: ParseGraph) -> None:
    overlap = {item.id for item in visual_graph.evidence} & {item.id for item in semantic_graph.evidence}
    if overlap:
        raise InvalidGeometryInput(f"Visual and semantic evidence ids overlap: {sorted(overlap)}")
