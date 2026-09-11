import json

import pytest

from geosym_world.errors import InvalidGeometryInput
from geosym_world.parser import validate_parse_graph_hypothesis
from geosym_world.parser.adapters.hypothesis_record import HypothesisRecordAdapter
from geosym_world.schema import (
    BoundaryReference,
    ConflictStatus,
    EvidencePrecision,
    MathExpression,
    math_expression_from_sympy,
    math_expression_from_text,
    ParseConflictGroup,
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
    RegionExpression,
)


def test_parse_graph_round_trips_anonymous_intersection_and_exact_expression():
    graph = _triangle_graph()

    restored = ParseGraph.from_dict(json.loads(json.dumps(graph.to_dict())))

    assert restored.to_dict() == graph.to_dict()
    assert restored.checksum() == graph.checksum()
    anonymous = next(item for item in restored.objects if item.id == "point_intersection_1")
    assert anonymous.visible_name is None
    assert anonymous.visible is True
    quantity = restored.quantities[0]
    assert quantity.expression.operator == "div"
    assert quantity.expression.operands[0].operator == "sqrt"


def test_supported_sympy_values_round_trip_through_safe_expression_ast():
    expression = math_expression_from_text(r"\frac{\sqrt{13}}{3} + x")

    assert expression.to_sympy().equals(MathExpression.from_dict(expression.to_dict()).to_sympy())
    assert str(expression.to_sympy()) == "x + sqrt(13)/3"


def test_source_math_text_parser_supports_common_notation_without_code_execution():
    expression = math_expression_from_text("2√13/3 + ½")

    assert str(expression.to_sympy()) == "1/2 + 2*sqrt(13)/3"
    with pytest.raises(InvalidGeometryInput):
        math_expression_from_text("__import__('os').system('echo unsafe')")
    with pytest.raises(InvalidGeometryInput, match="Expected a numeric or SymPy expression"):
        math_expression_from_sympy("sqrt(13)")


def test_parse_graph_checksum_and_unknown_fields_are_rejected():
    payload = _triangle_graph().to_dict()
    payload["unexpected"] = True
    with pytest.raises(InvalidGeometryInput, match="unknown fields"):
        ParseGraph.from_dict(payload)


@pytest.mark.parametrize(
    ("kind", "geometry", "message"),
    [
        (ParseObjectKind.POINT, {"pixel": [float("nan"), 1.0]}, "finite numbers"),
        (ParseObjectKind.CIRCLE, {"radius_px": -1.0}, "must be positive"),
        (ParseObjectKind.ARC, {"orientation": "clockwise"}, "orientation"),
        (ParseObjectKind.SEGMENT, {"pixel": [1.0, 2.0]}, "incompatible"),
        (ParseObjectKind.TEXT, {"bbox": [5.0, 5.0, 4.0, 6.0]}, "inverted bbox"),
    ],
)
def test_parse_object_rejects_invalid_geometry_at_schema_boundary(kind, geometry, message):
    with pytest.raises(InvalidGeometryInput, match=message):
        ParseObject(
            id="candidate_invalid",
            kind=kind,
            origin=ParseOrigin.IMAGE,
            confidence=0.8,
            score_kind="fixture",
            geometry=geometry,
        )


def test_parse_object_rejects_malformed_exact_expression_at_schema_boundary():
    with pytest.raises(InvalidGeometryInput, match="exact_coords is invalid"):
        ParseObject(
            id="point_invalid_exact",
            kind=ParseObjectKind.POINT,
            origin=ParseOrigin.NATIVE,
            confidence=1.0,
            score_kind="fixture",
            geometry={
                "exact_coords": [
                    {"operator": "sqrt", "operands": []},
                    {"operator": "literal", "value": "0", "operands": []},
                ]
            },
        )


def test_direct_parse_contract_requires_mapping_metadata_and_geometry():
    with pytest.raises(InvalidGeometryInput, match="geometry must be a mapping"):
        ParseObject(
            id="point_invalid_geometry",
            kind=ParseObjectKind.POINT,
            origin=ParseOrigin.IMAGE,
            confidence=0.8,
            score_kind="fixture",
            geometry=[],
        )
    with pytest.raises(InvalidGeometryInput, match="metadata must be a mapping"):
        ParseGraph(id="parse_invalid_metadata", source_id="source_invalid_metadata", metadata=[])


def test_parse_graph_is_persisted_as_a_proposed_geometry_hypothesis():
    graph = _triangle_graph()
    record = HypothesisRecordAdapter().convert_parse_graph(
        sample_id="triangle_sample",
        dataset_name="contract_fixture",
        parser_name="manual_contract",
        backend="manual",
        graph=graph,
        confidence=0.9,
        verification={"status": "not_admitted"},
    )

    restored = validate_parse_graph_hypothesis(record.hypotheses[0])

    assert record.state is None
    assert restored.to_dict() == graph.to_dict()

    payload = _triangle_graph().to_dict()
    payload["objects"][0]["visible_name"] = "Z"
    with pytest.raises(InvalidGeometryInput, match="checksum"):
        ParseGraph.from_dict(payload)


def test_parse_graph_rejects_dangling_bindings_and_invalid_conflicts():
    graph = _triangle_graph()
    relation = ParseRelation(
        id="binding_bad",
        predicate="binds",
        arguments=(ParseReference("label", "text_missing"), ParseReference("quantity", "quantity_cd")),
        origin=ParseOrigin.IMAGE,
        confidence=0.7,
        score_kind="model_score",
    )
    with pytest.raises(InvalidGeometryInput, match="unknown candidates"):
        ParseGraph(
            id=graph.id,
            source_id=graph.source_id,
            evidence=graph.evidence,
            objects=graph.objects,
            relations=(*graph.relations, relation),
            quantities=graph.quantities,
            targets=graph.targets,
        )

    with pytest.raises(InvalidGeometryInput, match="must select one member"):
        ParseConflictGroup(
            id="conflict_binding",
            members=("binding_left", "binding_right"),
            status=ConflictStatus.RESOLVED,
        )


def test_region_set_expression_preserves_holes_and_rejects_cycles():
    objects = (
        _object("outer_polygon", ParseObjectKind.POLYGON),
        _object("hole_polygon", ParseObjectKind.POLYGON),
    )
    outer = ParseRegion(
        id="outer_region",
        boundary=(BoundaryReference("outer_polygon", "forward"),),
        origin=ParseOrigin.IMAGE_TEXT,
        confidence=1.0,
        score_kind="human_annotation",
    )
    hole = ParseRegion(
        id="hole_region",
        boundary=(BoundaryReference("hole_polygon", "reverse"),),
        origin=ParseOrigin.IMAGE_TEXT,
        confidence=1.0,
        score_kind="human_annotation",
    )
    shaded = ParseRegion(
        id="shaded_region",
        boundary=(),
        holes=("hole_region",),
        components=("outer_region",),
        expression=RegionExpression(
            "difference",
            operands=(RegionExpression("reference", ref="outer_region"), RegionExpression("reference", ref="hole_region")),
        ),
        origin=ParseOrigin.COMPILED,
        confidence=1.0,
        score_kind="deterministic_rule",
    )

    graph = ParseGraph(id="parse_regions", source_id="source_regions", objects=objects, regions=(outer, hole, shaded))
    restored = ParseGraph.from_dict(graph.to_dict())
    assert restored.regions[-1].expression.references() == {"outer_region", "hole_region"}
    assert restored.regions[-1].holes == ("hole_region",)

    cyclic_outer = ParseRegion(
        id="outer_region",
        boundary=(BoundaryReference("outer_polygon"),),
        components=("shaded_region",),
        origin=ParseOrigin.COMPILED,
        confidence=1.0,
        score_kind="deterministic_rule",
    )
    with pytest.raises(InvalidGeometryInput, match="dependency cycle"):
        ParseGraph(
            id="parse_regions",
            source_id="source_regions",
            objects=objects,
            regions=(cyclic_outer, hole, shaded),
        )


def _triangle_graph() -> ParseGraph:
    evidence = (
        ParseEvidence(
            id="pixel_crossing_1",
            source_id="source_triangle",
            kind=ParseEvidenceKind.PIXEL_POINT,
            precision=EvidencePrecision.LOCALIZED,
            point=(120.0, 80.0),
        ),
        ParseEvidence(
            id="text_span_target",
            source_id="question_triangle",
            kind=ParseEvidenceKind.TEXT_SPAN,
            precision=EvidencePrecision.BOUND,
            span=(0, 16),
            raw_content="Find length CD.",
        ),
    )
    objects = (
        _object("A", ParseObjectKind.POINT, visible_name="A"),
        _object("B", ParseObjectKind.POINT, visible_name="B"),
        _object("C", ParseObjectKind.POINT, visible_name="C"),
        _object("D", ParseObjectKind.POINT, visible_name="D"),
        ParseObject(
            id="point_intersection_1",
            kind=ParseObjectKind.POINT,
            origin=ParseOrigin.IMAGE,
            confidence=0.82,
            score_kind="detector_score",
            evidence_refs=("pixel_crossing_1",),
            visible=True,
            geometry={"pixel": [120.0, 80.0]},
        ),
        ParseObject(
            id="segment_cd",
            kind=ParseObjectKind.SEGMENT,
            origin=ParseOrigin.IMAGE,
            confidence=0.95,
            score_kind="detector_score",
            references=(ParseReference("start", "C"), ParseReference("end", "D")),
            visible=True,
        ),
    )
    quantity = ParseQuantity(
        id="quantity_cd",
        kind="length",
        arguments=(ParseReference("object", "segment_cd"),),
        expression=MathExpression(
            "div",
            operands=(
                MathExpression("sqrt", operands=(MathExpression("literal", value="13"),)),
                MathExpression("literal", value="3"),
            ),
        ),
        origin=ParseOrigin.TEXT,
        confidence=0.9,
        score_kind="human_annotation",
        raw_text="sqrt(13)/3",
        evidence_refs=("text_span_target",),
    )
    ordered = ParseRelation(
        id="relation_order_ab",
        predicate="ordered_points",
        arguments=(
            ParseReference("point_0", "A"),
            ParseReference("point_1", "D"),
            ParseReference("point_2", "B"),
        ),
        origin=ParseOrigin.IMAGE,
        confidence=0.88,
        score_kind="detector_score",
    )
    target = ParseTarget(
        id="target_cd",
        kind="length",
        mode="value",
        arguments=(ParseReference("object", "segment_cd"),),
        origin=ParseOrigin.TEXT,
        confidence=0.95,
        score_kind="human_annotation",
        evidence_refs=("text_span_target",),
    )
    return ParseGraph(
        id="parse_triangle",
        source_id="source_triangle",
        evidence=evidence,
        objects=objects,
        relations=(ordered,),
        quantities=(quantity,),
        targets=(target,),
        metadata={"protocol": "image_question"},
    )


def _object(object_id: str, kind: ParseObjectKind, *, visible_name: str | None = None) -> ParseObject:
    return ParseObject(
        id=object_id,
        kind=kind,
        origin=ParseOrigin.IMAGE,
        confidence=1.0,
        score_kind="human_annotation",
        visible_name=visible_name,
        visible=True,
    )
