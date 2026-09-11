"""Structure-aware metrics for typed geometry ParseGraphs."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import sympy as sp

from geosym_world.schema import ParseGraph, ParseObject, ParseObjectKind, ParseReference, ParseRegion
from geosym_world.schema.canonical import canonical_json
from geosym_world.schema.relation_normalization import SYMMETRIC_RELATIONS


_UNMATCHED_COST = 100.0
_INVALID_COST = 1_000_000.0
_UNORDERED_ARGUMENT_RELATIONS = frozenset(
    {*SYMMETRIC_RELATIONS, "intersection", "disjoint", "overlaps", "touches", "linear_overlap"}
)
_TOPOLOGY_RELATIONS = frozenset(
    {
        "boundary_of",
        "contains",
        "disjoint",
        "endpoint_of",
        "inside",
        "intersection",
        "ordered_points",
        "overlaps",
        "linear_overlap",
        "support_of",
        "touches",
        "visible_extension",
    }
)
_CORE_COMPONENTS = ("objects", "relations", "quantities", "regions", "targets")
_ALL_COMPONENTS = ("objects", "relations", "labels", "markers", "bindings", "topology", "quantities", "regions", "targets")


class ParseGraphEvaluator:
    """Match objects independently, then score relations and bindings."""

    def __init__(self, *, point_tolerance: float = 0.035):
        if not 0.0 < point_tolerance <= 1.0:
            raise ValueError("point_tolerance must be in (0, 1]")
        self.point_tolerance = point_tolerance

    def evaluate(
        self,
        prediction: ParseGraph | Mapping[str, Any],
        reference: ParseGraph | Mapping[str, Any],
    ) -> dict[str, Any]:
        predicted = _as_graph(prediction)
        expected = _as_graph(reference)
        object_mapping, object_pairs = self._match_objects(predicted, expected)
        object_metric = _mapping_metric(predicted.objects, expected.objects, object_pairs)

        quantity_mapping, quantity_pairs = _match_by_signature(
            predicted.quantities,
            expected.quantities,
            lambda item: _quantity_signature(item, object_mapping),
            lambda item: _quantity_signature(item, None),
        )
        quantity_metric = _mapping_metric(predicted.quantities, expected.quantities, quantity_pairs)

        region_mapping, region_pairs = _match_regions(predicted.regions, expected.regions, object_mapping)
        region_metric = _mapping_metric(predicted.regions, expected.regions, region_pairs)

        relation_metric = _signature_metric(
            [_relation_signature(item, object_mapping, quantity_mapping, region_mapping) for item in predicted.relations],
            [_relation_signature(item, None, None, None) for item in expected.relations],
        )
        label_metric = _label_metric(predicted, expected, object_pairs)
        marker_metric = _object_kind_metric(predicted, expected, object_pairs, ParseObjectKind.MARKER)
        binding_metric = _relation_subset_metric(
            predicted,
            expected,
            object_mapping,
            quantity_mapping,
            region_mapping,
            lambda item: item.predicate == "binds",
        )
        topology_metric = _relation_subset_metric(
            predicted,
            expected,
            object_mapping,
            quantity_mapping,
            region_mapping,
            lambda item: item.predicate in _TOPOLOGY_RELATIONS,
        )
        target_metric = _signature_metric(
            [_target_signature(item, object_mapping, quantity_mapping, region_mapping) for item in predicted.targets],
            [_target_signature(item, None, None, None) for item in expected.targets],
        )
        scores = (
            object_metric["f1"],
            relation_metric["f1"],
            quantity_metric["f1"],
            region_metric["f1"],
            target_metric["f1"],
        )
        diagnostics = []
        if object_metric["recall"] < 1.0:
            diagnostics.append("object_mismatch")
        if relation_metric["recall"] < 1.0:
            diagnostics.append("relation_or_binding_mismatch")
        if label_metric["recall"] < 1.0:
            diagnostics.append("label_text_mismatch")
        if marker_metric["f1"] < 1.0:
            diagnostics.append("marker_mismatch")
        if binding_metric["recall"] < 1.0:
            diagnostics.append("label_binding_mismatch")
        if topology_metric["recall"] < 1.0:
            diagnostics.append("topology_mismatch")
        if quantity_metric["recall"] < 1.0:
            diagnostics.append("quantity_binding_mismatch")
        if region_metric["recall"] < 1.0:
            diagnostics.append("region_topology_mismatch")
        if target_metric["recall"] < 1.0:
            diagnostics.append("target_mismatch")
        components = {
            "objects": object_metric,
            "relations": relation_metric,
            "labels": label_metric,
            "markers": marker_metric,
            "bindings": binding_metric,
            "topology": topology_metric,
            "quantities": quantity_metric,
            "regions": region_metric,
            "targets": target_metric,
        }
        exact_match = {
            name: metric["true_positive"] == metric["predicted"] == metric["expected"]
            for name, metric in components.items()
        }
        return {
            "schema_version": "1.3",
            **components,
            "object_mapping": dict(sorted(object_mapping.items())),
            "quantity_mapping": dict(sorted(quantity_mapping.items())),
            "region_mapping": dict(sorted(region_mapping.items())),
            "macro_score": sum(scores) / len(scores),
            "component_exact_match": exact_match,
            "core_perfect_parsing": all(exact_match[name] for name in _CORE_COMPONENTS),
            "perfect_parsing": all(exact_match[name] for name in _ALL_COMPONENTS),
            "diagnostic_axes": diagnostics,
        }

    def _match_objects(
        self,
        predicted: ParseGraph,
        expected: ParseGraph,
    ) -> tuple[dict[str, str], list[tuple[str, str]]]:
        predicted_by_kind = _group_by_kind(predicted.objects)
        expected_by_kind = _group_by_kind(expected.objects)
        predicted_point_names = predicted.point_names()
        expected_point_names: Mapping[str, str] = {}
        mapping: dict[str, str] = {}
        pairs: list[tuple[str, str]] = []

        for kind in ParseObjectKind:
            left = predicted_by_kind.get(kind, ())
            right = expected_by_kind.get(kind, ())
            if not left or not right:
                continue
            unmatched_left = list(left)
            unmatched_right = list(right)
            while unmatched_left and unmatched_right:
                matched = _minimum_cost_pairs(
                    unmatched_left,
                    unmatched_right,
                    lambda candidate, truth: self._object_cost(
                        candidate,
                        truth,
                        predicted,
                        expected,
                        mapping,
                        predicted_point_names,
                        expected_point_names,
                    ),
                )
                if not matched:
                    break
                new_left_ids = {candidate.id for candidate, _ in matched}
                new_right_ids = {truth.id for _, truth in matched}
                for candidate, truth in matched:
                    mapping[candidate.id] = truth.id
                    pairs.append((candidate.id, truth.id))
                unmatched_left = [item for item in unmatched_left if item.id not in new_left_ids]
                unmatched_right = [item for item in unmatched_right if item.id not in new_right_ids]
        return mapping, pairs

    def _object_cost(
        self,
        predicted: ParseObject,
        expected: ParseObject,
        predicted_graph: ParseGraph,
        expected_graph: ParseGraph,
        mapping: Mapping[str, str],
        predicted_point_names: Mapping[str, str],
        expected_point_names: Mapping[str, str],
    ) -> float | None:
        if predicted.kind is not expected.kind:
            return None
        if predicted.kind is ParseObjectKind.MARKER and _marker_kind(predicted) != _marker_kind(expected):
            return None
        predicted_name = _normalized_name(predicted.visible_name or predicted_point_names.get(predicted.id))
        expected_name = _normalized_name(expected.visible_name or expected_point_names.get(expected.id))
        if predicted_name and expected_name and predicted_name != expected_name:
            return None

        reference_cost = _reference_cost(predicted, expected, mapping)
        if reference_cost is not None:
            return reference_cost + (0.0 if predicted_name == expected_name and predicted_name else 0.1)

        predicted_point = _normalized_position(predicted, predicted_graph)
        expected_point = _normalized_position(expected, expected_graph)
        if predicted_point is not None and expected_point is not None:
            distance = math.dist(predicted_point, expected_point)
            if distance > self.point_tolerance:
                return None
            return distance / self.point_tolerance + (0.0 if predicted_name == expected_name and predicted_name else 0.1)
        if predicted_name and predicted_name == expected_name:
            return 0.0
        if predicted.id == expected.id:
            return 0.25
        return None


def _as_graph(value: ParseGraph | Mapping[str, Any]) -> ParseGraph:
    if isinstance(value, ParseGraph):
        return value
    return ParseGraph.from_dict(value)


def _label_metric(
    predicted: ParseGraph,
    expected: ParseGraph,
    object_pairs: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    predicted_labels = tuple(item for item in predicted.objects if item.kind is ParseObjectKind.TEXT)
    expected_labels = tuple(item for item in expected.objects if item.kind is ParseObjectKind.TEXT)
    predicted_by_id = {item.id: item for item in predicted_labels}
    expected_by_id = {item.id: item for item in expected_labels}
    label_pairs = [
        (predicted_id, expected_id)
        for predicted_id, expected_id in object_pairs
        if predicted_id in predicted_by_id
        and expected_id in expected_by_id
        and _normalized_name(predicted_by_id[predicted_id].visible_name)
        and _normalized_name(predicted_by_id[predicted_id].visible_name)
        == _normalized_name(expected_by_id[expected_id].visible_name)
    ]
    return _mapping_metric(predicted_labels, expected_labels, label_pairs)


def _object_kind_metric(
    predicted: ParseGraph,
    expected: ParseGraph,
    object_pairs: Sequence[tuple[str, str]],
    kind: ParseObjectKind,
) -> dict[str, Any]:
    predicted_objects = tuple(item for item in predicted.objects if item.kind is kind)
    expected_objects = tuple(item for item in expected.objects if item.kind is kind)
    predicted_ids = {item.id for item in predicted_objects}
    expected_ids = {item.id for item in expected_objects}
    pairs = [
        (predicted_id, expected_id)
        for predicted_id, expected_id in object_pairs
        if predicted_id in predicted_ids and expected_id in expected_ids
    ]
    return _mapping_metric(predicted_objects, expected_objects, pairs)


def _marker_kind(item: ParseObject) -> str | None:
    value = item.metadata.get("marker_kind")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().lower()


def _relation_subset_metric(
    predicted: ParseGraph,
    expected: ParseGraph,
    object_mapping: Mapping[str, str],
    quantity_mapping: Mapping[str, str],
    region_mapping: Mapping[str, str],
    predicate: Callable[[Any], bool],
) -> dict[str, Any]:
    predicted_relations = [item for item in predicted.relations if predicate(item)]
    expected_relations = [item for item in expected.relations if predicate(item)]
    return _signature_metric(
        [_relation_signature(item, object_mapping, quantity_mapping, region_mapping) for item in predicted_relations],
        [_relation_signature(item, None, None, None) for item in expected_relations],
    )


def _group_by_kind(objects: Sequence[ParseObject]) -> dict[ParseObjectKind, tuple[ParseObject, ...]]:
    grouped: dict[ParseObjectKind, list[ParseObject]] = {}
    for item in objects:
        grouped.setdefault(item.kind, []).append(item)
    return {kind: tuple(sorted(items, key=lambda item: item.id)) for kind, items in grouped.items()}


def _reference_cost(predicted: ParseObject, expected: ParseObject, mapping: Mapping[str, str]) -> float | None:
    if not predicted.references and not expected.references:
        return None
    if len(predicted.references) != len(expected.references):
        return None
    mapped = [(reference.role, mapping.get(reference.ref)) for reference in predicted.references]
    if any(ref is None for _, ref in mapped):
        return None
    if predicted.kind is ParseObjectKind.SEGMENT:
        return 0.0 if sorted(ref for _, ref in mapped) == sorted(reference.ref for reference in expected.references) else None
    expected_refs = sorted((reference.role, reference.ref) for reference in expected.references)
    return 0.0 if sorted(mapped) == expected_refs else None


def _normalized_position(item: ParseObject, graph: ParseGraph) -> tuple[float, float] | None:
    position = _position_from_geometry(item.geometry)
    if position is None:
        evidence = {evidence.id: evidence for evidence in graph.evidence}
        for evidence_id in item.evidence_refs:
            candidate = evidence[evidence_id]
            if candidate.point is not None:
                position = candidate.point
                break
            if candidate.bbox is not None:
                position = ((candidate.bbox[0] + candidate.bbox[2]) / 2.0, (candidate.bbox[1] + candidate.bbox[3]) / 2.0)
                break
    if position is None:
        return None
    size = graph.metadata.get("image_size")
    if not isinstance(size, Sequence) or isinstance(size, (str, bytes)) or len(size) != 2:
        return None
    width, height = (float(value) for value in size)
    if width <= 0.0 or height <= 0.0:
        return None
    return position[0] / width, position[1] / height


def _position_from_geometry(geometry: Mapping[str, Any]) -> tuple[float, float] | None:
    for key in ("pixel", "center"):
        value = geometry.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
            try:
                point = float(value[0]), float(value[1])
            except (TypeError, ValueError):
                continue
            if all(math.isfinite(coordinate) for coordinate in point):
                return point
    bbox = geometry.get("bbox")
    if isinstance(bbox, Sequence) and not isinstance(bbox, (str, bytes)) and len(bbox) == 4:
        try:
            left, top, right, bottom = (float(value) for value in bbox)
        except (TypeError, ValueError):
            return None
        if all(math.isfinite(value) for value in (left, top, right, bottom)) and left <= right and top <= bottom:
            return (left + right) / 2.0, (top + bottom) / 2.0
    return None


def _match_by_signature(
    predicted: Sequence[Any],
    expected: Sequence[Any],
    predicted_signature: Callable[[Any], str],
    expected_signature: Callable[[Any], str],
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    predicted_by_signature: dict[str, list[Any]] = {}
    expected_by_signature: dict[str, list[Any]] = {}
    for item in predicted:
        predicted_by_signature.setdefault(predicted_signature(item), []).append(item)
    for item in expected:
        expected_by_signature.setdefault(expected_signature(item), []).append(item)
    mapping: dict[str, str] = {}
    pairs = []
    for key in sorted(set(predicted_by_signature) & set(expected_by_signature)):
        left = sorted(predicted_by_signature[key], key=lambda item: item.id)
        right = sorted(expected_by_signature[key], key=lambda item: item.id)
        for candidate, truth in zip(left, right):
            mapping[candidate.id] = truth.id
            pairs.append((candidate.id, truth.id))
    return mapping, pairs


def _match_regions(
    predicted: Sequence[ParseRegion],
    expected: Sequence[ParseRegion],
    object_mapping: Mapping[str, str],
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    mapping: dict[str, str] = {}
    pairs: list[tuple[str, str]] = []
    remaining_predicted = list(predicted)
    remaining_expected = list(expected)
    while remaining_predicted and remaining_expected:
        ready_predicted = [
            candidate
            for candidate in remaining_predicted
            if all(reference in mapping for reference in _nested_region_references(candidate))
        ]
        if not ready_predicted:
            break
        matched = _minimum_cost_pairs(
            ready_predicted,
            remaining_expected,
            lambda candidate, truth: _region_cost(candidate, truth, object_mapping, mapping),
        )
        if not matched:
            break
        left_ids = {candidate.id for candidate, _ in matched}
        right_ids = {truth.id for _, truth in matched}
        for candidate, truth in matched:
            mapping[candidate.id] = truth.id
            pairs.append((candidate.id, truth.id))
        remaining_predicted = [item for item in remaining_predicted if item.id not in left_ids]
        remaining_expected = [item for item in remaining_expected if item.id not in right_ids]
    return mapping, pairs


def _nested_region_references(region: ParseRegion) -> set[str]:
    references = {*region.holes, *region.components}
    if region.expression is not None:
        references.update(region.expression.references())
    return references


def _region_cost(
    predicted: ParseRegion,
    expected: ParseRegion,
    object_mapping: Mapping[str, str],
    region_mapping: Mapping[str, str],
) -> float | None:
    predicted_boundary = [
        (object_mapping.get(reference.ref), reference.orientation)
        for reference in predicted.boundary
    ]
    if any(ref is None for ref, _ in predicted_boundary):
        return None
    expected_boundary = [(reference.ref, reference.orientation) for reference in expected.boundary]
    if not _cyclic_equal(predicted_boundary, expected_boundary):
        return None
    mapped_holes = [region_mapping.get(ref) for ref in predicted.holes]
    mapped_components = [region_mapping.get(ref) for ref in predicted.components]
    if any(ref is None for ref in (*mapped_holes, *mapped_components)):
        return None
    if Counter(mapped_holes) != Counter(expected.holes) or Counter(mapped_components) != Counter(expected.components):
        return None
    predicted_expression = _mapped_region_expression(predicted.expression, region_mapping)
    expected_expression = _mapped_region_expression(expected.expression, None)
    if predicted_expression != expected_expression:
        return None
    return 0.0


def _mapped_region_expression(
    expression: Any,
    mapping: Mapping[str, str] | None,
) -> dict[str, Any] | None:
    if expression is None:
        return None
    if expression.ref is not None:
        mapped = expression.ref if mapping is None else mapping.get(expression.ref)
        if mapped is None:
            return {"unresolved": expression.ref}
        return {"operator": expression.operator, "ref": mapped, "operands": []}
    operands = [_mapped_region_expression(operand, mapping) for operand in expression.operands]
    if expression.operator in {"union", "intersection"}:
        operands.sort(key=canonical_json)
    return {
        "operator": expression.operator,
        "ref": None,
        "operands": operands,
    }


def _relation_signature(
    relation: Any,
    object_mapping: Mapping[str, str] | None,
    quantity_mapping: Mapping[str, str] | None,
    region_mapping: Mapping[str, str] | None,
) -> str:
    arguments = _mapped_arguments(relation.arguments, object_mapping, quantity_mapping, region_mapping)
    if relation.predicate in _UNORDERED_ARGUMENT_RELATIONS:
        arguments = [("object", reference) for _, reference in sorted(arguments, key=lambda item: item[1])]
    return canonical_json({"predicate": relation.predicate, "arguments": arguments})


def _quantity_signature(quantity: Any, object_mapping: Mapping[str, str] | None) -> str:
    arguments = _mapped_arguments(quantity.arguments, object_mapping, None, None)
    expression, unit = _normalized_expression(quantity.expression, quantity.unit, kind=quantity.kind)
    return canonical_json(
        {
            "kind": quantity.kind,
            "arguments": arguments,
            "expression": expression,
            "unit": unit,
        }
    )


def _target_signature(
    target: Any,
    object_mapping: Mapping[str, str] | None,
    quantity_mapping: Mapping[str, str] | None,
    region_mapping: Mapping[str, str] | None,
) -> str:
    expression = None
    if target.expression is not None:
        expression, _ = _normalized_expression(target.expression, None, kind=target.kind)
    return canonical_json(
        {
            "kind": target.kind,
            "mode": target.mode,
            "arguments": _mapped_arguments(target.arguments, object_mapping, quantity_mapping, region_mapping),
            "expression": expression,
        }
    )


def _normalized_expression(expression: Any, unit: str | None, *, kind: str) -> tuple[str, str | None]:
    value = sp.simplify(expression.to_sympy())
    normalized_unit = unit.strip().lower() if isinstance(unit, str) else None
    if kind == "angle" and normalized_unit in {"degree", "degrees", "deg", "°"}:
        value = sp.simplify(value * sp.pi / 180)
        normalized_unit = "radian"
    elif kind == "angle" and normalized_unit in {"radian", "radians", "rad"}:
        normalized_unit = "radian"
    return sp.srepr(value), normalized_unit


def _mapped_arguments(
    arguments: Sequence[ParseReference],
    object_mapping: Mapping[str, str] | None,
    quantity_mapping: Mapping[str, str] | None,
    region_mapping: Mapping[str, str] | None,
) -> list[tuple[str, str]]:
    mappings = tuple(mapping for mapping in (object_mapping, quantity_mapping, region_mapping) if mapping is not None)
    mapped = []
    for argument in arguments:
        reference = argument.ref
        if mappings:
            reference = next((mapping[reference] for mapping in mappings if reference in mapping), f"<unmatched:{reference}>")
        mapped.append((argument.role, reference))
    return mapped


def _mapping_metric(predicted: Sequence[Any], expected: Sequence[Any], pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    matched = len(pairs)
    precision, recall, f1 = _precision_recall_f1(matched, len(predicted), len(expected))
    matched_predicted = {left for left, _ in pairs}
    matched_expected = {right for _, right in pairs}
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": matched,
        "predicted": len(predicted),
        "expected": len(expected),
        "pairs": [{"predicted": left, "expected": right} for left, right in sorted(pairs)],
        "matched_predicted_ids": sorted(matched_predicted),
        "matched_expected_ids": sorted(matched_expected),
        "false_positive_ids": sorted(item.id for item in predicted if item.id not in matched_predicted),
        "false_negative_ids": sorted(item.id for item in expected if item.id not in matched_expected),
    }


def _cyclic_equal(left: Sequence[Any], right: Sequence[Any]) -> bool:
    if len(left) != len(right):
        return False
    if not left:
        return True
    doubled = [*right, *right]
    return any(list(left) == doubled[offset : offset + len(left)] for offset in range(len(right)))


def _signature_metric(predicted: Sequence[str], expected: Sequence[str]) -> dict[str, Any]:
    predicted_counter = Counter(predicted)
    expected_counter = Counter(expected)
    matched = sum((predicted_counter & expected_counter).values())
    precision, recall, f1 = _precision_recall_f1(matched, len(predicted), len(expected))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": matched,
        "predicted": len(predicted),
        "expected": len(expected),
        "false_positive": sorted((predicted_counter - expected_counter).elements()),
        "false_negative": sorted((expected_counter - predicted_counter).elements()),
    }


def _precision_recall_f1(matched: int, predicted: int, expected: int) -> tuple[float, float, float]:
    precision = matched / predicted if predicted else float(expected == 0)
    recall = matched / expected if expected else float(predicted == 0)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _minimum_cost_pairs(
    predicted: Sequence[Any],
    expected: Sequence[Any],
    cost: Callable[[Any, Any], float | None],
) -> list[tuple[Any, Any]]:
    if not predicted or not expected:
        return []
    columns = len(expected) + len(predicted)
    matrix = []
    for candidate in predicted:
        row = []
        for truth in expected:
            value = cost(candidate, truth)
            row.append(_INVALID_COST if value is None else float(value))
        row.extend(_UNMATCHED_COST if index == len(matrix) else _INVALID_COST for index in range(len(predicted)))
        matrix.append(row)
    assignment = _hungarian(matrix)
    return [
        (predicted[row], expected[column])
        for row, column in enumerate(assignment)
        if 0 <= column < len(expected) and matrix[row][column] < _UNMATCHED_COST
    ]


def _hungarian(costs: Sequence[Sequence[float]]) -> list[int]:
    """Return a minimum-cost assignment for a matrix with rows <= columns."""

    row_count = len(costs)
    column_count = len(costs[0])
    if row_count > column_count or any(len(row) != column_count for row in costs):
        raise ValueError("Hungarian assignment requires a rectangular rows <= columns matrix")
    row_potential = [0.0] * (row_count + 1)
    column_potential = [0.0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    predecessor = [0] * (column_count + 1)
    for row in range(1, row_count + 1):
        matched_row[0] = row
        min_value = [math.inf] * (column_count + 1)
        used = [False] * (column_count + 1)
        column = 0
        while True:
            used[column] = True
            current_row = matched_row[column]
            delta = math.inf
            next_column = 0
            for candidate_column in range(1, column_count + 1):
                if used[candidate_column]:
                    continue
                reduced = costs[current_row - 1][candidate_column - 1] - row_potential[current_row] - column_potential[candidate_column]
                if reduced < min_value[candidate_column]:
                    min_value[candidate_column] = reduced
                    predecessor[candidate_column] = column
                if min_value[candidate_column] < delta:
                    delta = min_value[candidate_column]
                    next_column = candidate_column
            for candidate_column in range(column_count + 1):
                if used[candidate_column]:
                    row_potential[matched_row[candidate_column]] += delta
                    column_potential[candidate_column] -= delta
                else:
                    min_value[candidate_column] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous_column = predecessor[column]
            matched_row[column] = matched_row[previous_column]
            column = previous_column
            if column == 0:
                break
    assignment = [-1] * row_count
    for column in range(1, column_count + 1):
        if matched_row[column] != 0:
            assignment[matched_row[column] - 1] = column - 1
    return assignment


def _normalized_name(value: str | None) -> str:
    return "" if value is None else "".join(value.split()).casefold()
