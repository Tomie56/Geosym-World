from dataclasses import replace

from geosym_world.benchmark import ParseGraphEvaluator
from geosym_world.schema import (
    BoundaryReference,
    MathExpression,
    ParseEvidence,
    ParseEvidenceKind,
    ParseGraph,
    ParseObject,
    ParseObjectKind,
    ParseOrigin,
    ParseQuantity,
    ParseReference,
    ParseRegion,
    ParseRelation,
    ParseTarget,
    EvidencePrecision,
    RegionExpression,
)


def test_evaluator_matches_renamed_objects_before_relations_and_bindings():
    reference = _graph(prefix="gt", offset=0.0)
    prediction = _graph(prefix="pred", offset=1.0)

    metrics = ParseGraphEvaluator(point_tolerance=0.05).evaluate(prediction, reference)

    assert metrics["objects"]["f1"] == 1.0
    assert metrics["relations"]["f1"] == 1.0
    assert metrics["quantities"]["f1"] == 1.0
    assert metrics["regions"]["f1"] == 1.0
    assert metrics["targets"]["f1"] == 1.0
    assert metrics["macro_score"] == 1.0
    assert all(metrics["component_exact_match"].values())
    assert metrics["core_perfect_parsing"] is True
    assert metrics["perfect_parsing"] is True


def test_equal_object_counts_do_not_hide_wrong_geometry_or_quantity_binding():
    reference = _graph(prefix="gt", offset=0.0)
    prediction = _graph(prefix="pred", offset=1.0)
    moved_objects = tuple(
        replace(item, geometry={"pixel": [180.0, 20.0]}) if item.visible_name == "A" else item
        for item in prediction.objects
    )
    wrong_quantity = replace(
        prediction.quantities[0],
        arguments=(ParseReference("object", "pred_segment_ac"),),
    )
    prediction = replace(prediction, objects=moved_objects, quantities=(wrong_quantity,))

    metrics = ParseGraphEvaluator(point_tolerance=0.05).evaluate(prediction, reference)

    assert metrics["objects"]["predicted"] == metrics["objects"]["expected"]
    assert metrics["objects"]["recall"] < 1.0
    assert metrics["quantities"]["f1"] == 0.0
    assert "quantity_binding_mismatch" in metrics["diagnostic_axes"]


def test_quantity_metric_compares_normalized_expression_values():
    reference = _graph(prefix="gt", offset=0.0)
    prediction = _graph(prefix="pred", offset=1.0)
    equivalent_four = MathExpression(
        "add",
        operands=(
            MathExpression("sqrt", operands=(MathExpression("literal", value="4"),)),
            MathExpression("literal", value="2"),
        ),
    )
    prediction = replace(prediction, quantities=(replace(prediction.quantities[0], expression=equivalent_four),))

    metrics = ParseGraphEvaluator(point_tolerance=0.05).evaluate(prediction, reference)

    assert metrics["quantities"]["f1"] == 1.0


def test_evaluator_reports_label_text_and_binding_as_separate_components():
    reference = _with_point_label(_graph(prefix="gt", offset=0.0), prefix="gt", point_name="A")
    prediction = _with_point_label(_graph(prefix="pred", offset=1.0), prefix="pred", point_name="B")

    metrics = ParseGraphEvaluator(point_tolerance=0.05).evaluate(prediction, reference)

    assert metrics["labels"]["f1"] == 1.0
    assert metrics["bindings"]["f1"] == 0.0
    assert metrics["topology"]["f1"] == 1.0
    assert metrics["component_exact_match"]["bindings"] is False
    assert metrics["perfect_parsing"] is False
    assert "label_binding_mismatch" in metrics["diagnostic_axes"]


def test_evaluator_scores_marker_type_independently_from_location():
    reference = _with_marker(_graph(prefix="gt", offset=0.0), prefix="gt", marker_kind="right_angle")
    matching = _with_marker(_graph(prefix="pred", offset=1.0), prefix="pred", marker_kind="right_angle")
    wrong_type = _with_marker(_graph(prefix="pred", offset=1.0), prefix="pred", marker_kind="parallel")

    matching_metrics = ParseGraphEvaluator(point_tolerance=0.05).evaluate(matching, reference)
    metrics = ParseGraphEvaluator(point_tolerance=0.05).evaluate(wrong_type, reference)

    assert matching_metrics["markers"]["f1"] == 1.0
    assert metrics["schema_version"] == "1.3"
    assert metrics["markers"]["predicted"] == 1
    assert metrics["markers"]["expected"] == 1
    assert metrics["markers"]["f1"] == 0.0
    assert metrics["component_exact_match"]["markers"] is False
    assert metrics["perfect_parsing"] is False
    assert "marker_mismatch" in metrics["diagnostic_axes"]


def test_object_matching_treats_binding_relation_as_a_point_name_projection():
    reference_graph = _graph(prefix="gt", offset=0.0)
    reference_point = replace(
        next(item for item in reference_graph.objects if item.visible_name == "A"),
        evidence_refs=(),
    )
    predicted_point = replace(
        reference_point,
        id="predicted_point_a",
        visible_name=None,
        evidence_refs=(),
    )
    predicted_text = ParseObject(
        id="predicted_text_a",
        kind=ParseObjectKind.TEXT,
        origin=ParseOrigin.IMAGE_TEXT,
        confidence=1.0,
        score_kind="fixture",
        visible_name="A",
        geometry={"bbox": [12.0, 170.0, 18.0, 180.0]},
    )
    binding = ParseRelation(
        id="predicted_binding_a",
        predicate="binds",
        arguments=(
            ParseReference("label", predicted_text.id),
            ParseReference("object", predicted_point.id),
        ),
        origin=ParseOrigin.IMAGE_TEXT,
        confidence=1.0,
        score_kind="fixture",
        metadata={"binding_kind": "point_name"},
    )
    prediction = ParseGraph(
        id="predicted_name_projection",
        source_id="predicted_name_source",
        objects=(predicted_point, predicted_text),
        relations=(binding,),
        metadata=reference_graph.metadata,
    )
    reference = ParseGraph(
        id="reference_name_projection",
        source_id="reference_name_source",
        objects=(reference_point,),
        metadata=reference_graph.metadata,
    )

    metrics = ParseGraphEvaluator(point_tolerance=0.05).evaluate(prediction, reference)

    assert metrics["object_mapping"][predicted_point.id] == reference_point.id


def test_region_metric_detects_missing_hole_despite_equal_outer_boundary():
    reference = _graph(prefix="gt", offset=0.0)
    prediction = _graph(prefix="pred", offset=1.0)
    outer_only = prediction.regions[0]
    prediction = replace(prediction, regions=(outer_only,))

    metrics = ParseGraphEvaluator().evaluate(prediction, reference)

    assert metrics["objects"]["f1"] == 1.0
    assert metrics["regions"]["recall"] < 1.0
    assert "region_topology_mismatch" in metrics["diagnostic_axes"]


def test_region_metric_matches_outer_hole_dependencies_before_parent_region():
    reference = _graph(prefix="gt", offset=0.0)
    prediction = _graph(prefix="pred", offset=1.0)
    reference_outer = replace(reference.regions[0], holes=(reference.regions[1].id,))
    predicted_outer = replace(prediction.regions[0], holes=(prediction.regions[1].id,))
    reference = replace(reference, regions=(reference_outer, *reference.regions[1:]))
    prediction = replace(prediction, regions=(predicted_outer, *prediction.regions[1:]))

    metrics = ParseGraphEvaluator().evaluate(prediction, reference)

    assert metrics["regions"]["f1"] == 1.0
    assert metrics["region_mapping"][predicted_outer.id] == reference_outer.id
    assert metrics["perfect_parsing"] is True


def test_region_metric_accepts_cyclic_boundary_and_commutative_union_order():
    reference = _graph(prefix="gt", offset=0.0)
    prediction = _graph(prefix="pred", offset=1.0)
    predicted_outer = replace(
        prediction.regions[0],
        boundary=prediction.regions[0].boundary[1:] + prediction.regions[0].boundary[:1],
    )
    predicted_shaded = replace(
        prediction.regions[2],
        expression=RegionExpression(
            "union",
            operands=(
                RegionExpression("reference", ref=prediction.regions[1].id),
                RegionExpression("reference", ref=prediction.regions[0].id),
            ),
        ),
    )
    reference_shaded = replace(
        reference.regions[2],
        expression=RegionExpression(
            "union",
            operands=(
                RegionExpression("reference", ref=reference.regions[0].id),
                RegionExpression("reference", ref=reference.regions[1].id),
            ),
        ),
    )
    prediction = replace(prediction, regions=(predicted_outer, prediction.regions[1], predicted_shaded))
    reference = replace(reference, regions=(reference.regions[0], reference.regions[1], reference_shaded))

    metrics = ParseGraphEvaluator().evaluate(prediction, reference)

    assert metrics["regions"]["f1"] == 1.0
    assert metrics["regions"]["false_positive_ids"] == []


def test_evaluator_scores_linear_overlap_as_symmetric_topology():
    reference = _graph(prefix="gt", offset=0.0)
    prediction = _graph(prefix="pred", offset=1.0)
    reference_relation = ParseRelation(
        id="gt_linear_overlap",
        predicate="linear_overlap",
        arguments=(
            ParseReference("support", "gt_segment_ab"),
            ParseReference("support", "gt_segment_ac"),
        ),
        origin=ParseOrigin.IMAGE,
        confidence=1.0,
        score_kind="fixture",
    )
    predicted_relation = replace(
        reference_relation,
        id="pred_linear_overlap",
        arguments=(
            ParseReference("support", "pred_segment_ac"),
            ParseReference("support", "pred_segment_ab"),
        ),
    )
    reference = replace(reference, relations=(*reference.relations, reference_relation))
    prediction = replace(prediction, relations=(*prediction.relations, predicted_relation))

    metrics = ParseGraphEvaluator().evaluate(prediction, reference)

    assert metrics["relations"]["f1"] == 1.0
    assert metrics["topology"]["f1"] == 1.0


def _graph(*, prefix: str, offset: float) -> ParseGraph:
    source_id = f"{prefix}_source"
    point_ids = {name: f"{prefix}_point_{name.lower()}" for name in "ABC"}
    positions = {"A": (20.0 + offset, 180.0), "B": (180.0 + offset, 180.0), "C": (20.0 + offset, 20.0)}
    evidence = tuple(
        ParseEvidence(
            id=f"{prefix}_evidence_{name.lower()}",
            source_id=source_id,
            kind=ParseEvidenceKind.PIXEL_POINT,
            precision=EvidencePrecision.BOUND,
            point=positions[name],
        )
        for name in "ABC"
    )
    points = tuple(
        ParseObject(
            id=point_ids[name],
            kind=ParseObjectKind.POINT,
            origin=ParseOrigin.IMAGE,
            confidence=1.0,
            score_kind="fixture",
            evidence_refs=(f"{prefix}_evidence_{name.lower()}",),
            visible_name=name,
            visible=True,
            geometry={"pixel": list(positions[name])},
        )
        for name in "ABC"
    )
    segment_ab = _segment(prefix, "ab", point_ids["A"], point_ids["B"])
    segment_ac = _segment(prefix, "ac", point_ids["A"], point_ids["C"])
    polygon = ParseObject(
        id=f"{prefix}_polygon_abc",
        kind=ParseObjectKind.POLYGON,
        origin=ParseOrigin.COMPILED,
        confidence=1.0,
        score_kind="fixture",
        references=tuple(ParseReference(f"vertex_{index}", point_ids[name]) for index, name in enumerate("ABC")),
        visible=True,
    )
    hole_polygon = ParseObject(
        id=f"{prefix}_hole_polygon",
        kind=ParseObjectKind.POLYGON,
        origin=ParseOrigin.COMPILED,
        confidence=1.0,
        score_kind="fixture",
        references=tuple(ParseReference(f"vertex_{index}", point_ids[name]) for index, name in enumerate("ABC")),
        visible=True,
    )
    objects = (*points, segment_ab, segment_ac, polygon, hole_polygon)
    quantity = ParseQuantity(
        id=f"{prefix}_quantity_ab",
        kind="length",
        arguments=(ParseReference("object", segment_ab.id),),
        expression=MathExpression("literal", value="4"),
        origin=ParseOrigin.TEXT,
        confidence=1.0,
        score_kind="fixture",
    )
    relation = ParseRelation(
        id=f"{prefix}_perpendicular",
        predicate="perpendicular",
        arguments=(ParseReference("left", segment_ab.id), ParseReference("right", segment_ac.id)),
        origin=ParseOrigin.IMAGE,
        confidence=1.0,
        score_kind="fixture",
    )
    outer = ParseRegion(
        id=f"{prefix}_outer",
        boundary=(BoundaryReference(polygon.id),),
        origin=ParseOrigin.IMAGE,
        confidence=1.0,
        score_kind="fixture",
    )
    hole = ParseRegion(
        id=f"{prefix}_hole",
        boundary=(BoundaryReference(hole_polygon.id),),
        origin=ParseOrigin.IMAGE,
        confidence=1.0,
        score_kind="fixture",
    )
    shaded = ParseRegion(
        id=f"{prefix}_shaded",
        boundary=(),
        holes=(hole.id,),
        components=(outer.id,),
        expression=RegionExpression(
            "difference",
            operands=(RegionExpression("reference", ref=outer.id), RegionExpression("reference", ref=hole.id)),
        ),
        origin=ParseOrigin.COMPILED,
        confidence=1.0,
        score_kind="fixture",
    )
    target = ParseTarget(
        id=f"{prefix}_target",
        kind="length",
        mode="value",
        arguments=(ParseReference("object", segment_ab.id),),
        origin=ParseOrigin.TEXT,
        confidence=1.0,
        score_kind="fixture",
    )
    return ParseGraph(
        id=f"{prefix}_graph",
        source_id=source_id,
        evidence=evidence,
        objects=objects,
        relations=(relation,),
        quantities=(quantity,),
        regions=(outer, hole, shaded),
        targets=(target,),
        metadata={"image_size": [200, 200]},
    )


def _with_point_label(graph: ParseGraph, *, prefix: str, point_name: str) -> ParseGraph:
    label = ParseObject(
        id=f"{prefix}_text_a",
        kind=ParseObjectKind.TEXT,
        origin=ParseOrigin.IMAGE_TEXT,
        confidence=1.0,
        score_kind="fixture",
        visible_name="A",
        geometry={"bbox": [8.0, 170.0, 16.0, 182.0]},
    )
    binding = ParseRelation(
        id=f"{prefix}_binding_a",
        predicate="binds",
        arguments=(
            ParseReference("label", label.id),
            ParseReference("object", f"{prefix}_point_{point_name.lower()}"),
        ),
        origin=ParseOrigin.IMAGE_TEXT,
        confidence=1.0,
        score_kind="fixture",
        metadata={"binding_kind": "point_name", "label": "A"},
    )
    return replace(graph, objects=(*graph.objects, label), relations=(*graph.relations, binding))


def _with_marker(graph: ParseGraph, *, prefix: str, marker_kind: str) -> ParseGraph:
    marker = ParseObject(
        id=f"{prefix}_marker",
        kind=ParseObjectKind.MARKER,
        origin=ParseOrigin.IMAGE,
        confidence=1.0,
        score_kind="fixture",
        visible=True,
        geometry={"bbox": [42.0, 42.0, 56.0, 56.0]},
        metadata={"marker_kind": marker_kind},
    )
    return replace(graph, objects=(*graph.objects, marker))


def _segment(prefix: str, suffix: str, start: str, end: str) -> ParseObject:
    return ParseObject(
        id=f"{prefix}_segment_{suffix}",
        kind=ParseObjectKind.SEGMENT,
        origin=ParseOrigin.IMAGE,
        confidence=1.0,
        score_kind="fixture",
        references=(ParseReference("start", start), ParseReference("end", end)),
        visible=True,
    )
