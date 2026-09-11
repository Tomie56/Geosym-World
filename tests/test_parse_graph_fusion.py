from dataclasses import replace

from PIL import Image

from geosym_world.parser import (
    GeoparsingAdapter,
    GeoparsingPrediction,
    ParseGraphEvidenceFusion,
    ParserInput,
)
from geosym_world.schema import (
    ConflictStatus,
    EvidencePrecision,
    ParseConflictGroup,
    ParseEvidence,
    ParseEvidenceKind,
    ParseGraph,
    ParseObject,
    ParseObjectKind,
    ParseOrigin,
    ParseQuantity,
    ParseReference,
    ParseRelation,
    ParseTarget,
    math_expression_from_text,
)
from geosym_world.verifier import ParseGraphAdmission


def test_fusion_grounds_unique_geoparsing_references_without_promoting_semantics(tmp_path):
    parser_input = _parser_input(tmp_path)
    visual = _visual_graph(parser_input.id)
    semantic = _semantic_graph(parser_input)

    result = ParseGraphEvidenceFusion().fuse(visual, semantic)
    graph = result.graph
    semantic_quantity = next(item for item in graph.quantities if item.id.startswith("gdp:"))

    assert tuple(item.ref for item in semantic_quantity.arguments) == ("visual:segment:AB",)
    assert all(not item.id.startswith("gdp:point") for item in graph.objects)
    assert all(not item.id.startswith("gdp:line") for item in graph.objects)
    assert all(not item.id.startswith("segment:") for item in graph.objects)
    assert result.status_counts()["mapped_to_local"] == 4
    assert result.status_counts()["retained_with_grounded_references"] == 2
    point_check = next(
        item for item in result.compatibility.checks if item.candidate_id.startswith("gdp:point:0:")
    )
    assert point_check.reason == "semantic_point_matches_grounded_visual_label"
    assert point_check.metrics["label"] == "A"
    quantity_check = next(item for item in result.compatibility.checks if item.candidate_id == semantic_quantity.id)
    assert quantity_check.status == "unknown"
    assert quantity_check.reason == "no_local_quantity_match_without_scale_assumption"
    assert result.graph.metadata["compatibility_certificate"]["status_counts"]["unknown"] >= 1

    admission = ParseGraphAdmission().admit(graph)
    assert semantic_quantity.id in admission.abstained_ids
    assert next(item for item in admission.decisions if item.candidate_id == semantic_quantity.id).reason == (
        "evidence_precision_below_bound"
    )


def test_fusion_reuses_an_equivalent_local_quantity_and_records_the_match(tmp_path):
    parser_input = _parser_input(tmp_path)
    visual = _visual_graph(parser_input.id, include_quantity=True)
    quantity = visual.quantities[0]
    label_evidence = ParseEvidence(
        id="visual:evidence:text:length:AB",
        source_id=parser_input.id,
        kind=ParseEvidenceKind.IMAGE_REGION,
        precision=EvidencePrecision.BOUND,
        bbox=(46.0, 24.0, 54.0, 32.0),
        raw_content="5",
    )
    label = ParseObject(
        id="visual:text:length:AB",
        kind=ParseObjectKind.TEXT,
        origin=ParseOrigin.IMAGE,
        confidence=0.95,
        score_kind="fixture_ocr",
        evidence_refs=(label_evidence.id,),
        visible_name="5",
        visible=True,
        geometry={"bbox": [46.0, 24.0, 54.0, 32.0]},
    )
    visual = replace(
        visual,
        evidence=(*visual.evidence, label_evidence),
        objects=(*visual.objects, label),
        quantities=(replace(quantity, metadata={"label_object_id": label.id}),),
    )
    semantic = _semantic_graph(parser_input)

    result = ParseGraphEvidenceFusion().fuse(visual, semantic)

    assert [item.id for item in result.graph.quantities] == ["visual:quantity:AB"]
    semantic_quantity = next(item for item in semantic.quantities if item.kind == "length")
    decision = next(item for item in result.decisions if item.candidate_id == semantic_quantity.id)
    assert decision.status == "matched_existing_local_candidate"
    assert decision.mapped_to == "visual:quantity:AB"
    compatibility = next(item for item in result.compatibility.checks if item.candidate_id == semantic_quantity.id)
    assert compatibility.status == "compatible"
    assert compatibility.reason == "semantic_quantity_matches_grounded_visual_expression"
    assert compatibility.compared_ids == ("visual:quantity:AB", "visual:text:length:AB")
    assert compatibility.metrics["label_text"] == "5"
    assert result.graph.metadata["semantic_graph_checksum"] == semantic.checksum()


def test_fusion_preserves_semantic_objects_when_point_name_binding_is_ambiguous(tmp_path):
    parser_input = _parser_input(tmp_path)
    visual = _visual_graph(parser_input.id, ambiguous_a=True)
    semantic = _semantic_graph(parser_input)

    result = ParseGraphEvidenceFusion().fuse(visual, semantic)
    semantic_a = next(
        item for item in semantic.objects if item.kind is ParseObjectKind.POINT and item.visible_name == "A"
    )
    semantic_b = next(
        item for item in semantic.objects if item.kind is ParseObjectKind.POINT and item.visible_name == "B"
    )

    assert any(item.id == semantic_a.id for item in result.graph.objects)
    assert all(item.id != semantic_b.id for item in result.graph.objects)
    decision = next(item for item in result.decisions if item.candidate_id == semantic_a.id)
    assert decision.status == "retained_unresolved"
    assert result.graph.conflicts[0].status is ConflictStatus.UNRESOLVED


def test_fusion_certificate_checks_mapped_perpendicular_geometry(tmp_path):
    parser_input = _parser_input(tmp_path)
    visual = _visual_graph_with_vertical_segment(_visual_graph(parser_input.id))
    prediction = GeoparsingPrediction.create(
        parser_input,
        backend="offline",
        model="GDP-4B",
        model_revision="revision-sha",
        prompt_version="official-v1",
        confidence=0.8,
        formal_description={
            "points": ["A", "B", "C"],
            "lines": ["line A B", "line B C"],
            "circles": [],
            "semantics": [r"AB \perp BC on B"],
        },
    )
    semantic = GeoparsingAdapter().convert(parser_input, prediction).prediction.parse_graph

    result = ParseGraphEvidenceFusion().fuse(visual, semantic)
    relation = next(item for item in semantic.relations if item.predicate == "perpendicular")
    check = next(item for item in result.compatibility.checks if item.candidate_id == relation.id)

    assert check.status == "compatible"
    assert check.metrics["residual_deg"] == 0.0


def test_fusion_certificate_records_grounded_right_angle_marker(tmp_path):
    parser_input = _parser_input(tmp_path)
    base = _visual_graph_with_vertical_segment(_visual_graph(parser_input.id))
    marker = ParseObject(
        id="visual:marker:right_angle",
        kind=ParseObjectKind.MARKER,
        origin=ParseOrigin.IMAGE,
        confidence=0.98,
        score_kind="fixture_right_angle_glyph",
        references=(
            ParseReference("support", "visual:segment:AB"),
            ParseReference("support", "visual:segment:BC"),
        ),
        evidence_refs=("visual:evidence:marker:right_angle",),
        visible=True,
        geometry={"bbox": [85.0, 25.0, 95.0, 35.0]},
        metadata={
            "marker_kind": "right_angle",
            "support_line_ids": ["visual:segment:AB", "visual:segment:BC"],
        },
    )
    marker_relation = ParseRelation(
        id="visual:perpendicular:AB:BC",
        predicate="perpendicular",
        arguments=(
            ParseReference("left", "visual:segment:AB"),
            ParseReference("right", "visual:segment:BC"),
        ),
        origin=ParseOrigin.IMAGE,
        confidence=0.98,
        score_kind="explicit_right_angle_marker",
        evidence_refs=("visual:evidence:marker:right_angle",),
        dependencies=(marker.id,),
        metadata={"marker_id": marker.id},
    )
    visual = replace(
        base,
        evidence=(*base.evidence, ParseEvidence(
            id="visual:evidence:marker:right_angle",
            source_id=parser_input.id,
            kind=ParseEvidenceKind.IMAGE_REGION,
            precision=EvidencePrecision.BOUND,
            bbox=(85.0, 25.0, 95.0, 35.0),
        )),
        objects=(*base.objects, marker),
        relations=(*base.relations, marker_relation),
    )
    prediction = GeoparsingPrediction.create(
        parser_input,
        backend="offline",
        model="GDP-4B",
        model_revision="revision-sha",
        prompt_version="official-v1",
        confidence=0.8,
        formal_description={
            "points": ["A", "B", "C"],
            "lines": ["line A B", "line B C"],
            "circles": [],
            "semantics": [r"AB \perp BC on B"],
        },
    )
    semantic = GeoparsingAdapter().convert(parser_input, prediction).prediction.parse_graph

    result = ParseGraphEvidenceFusion().fuse(visual, semantic)
    relation = next(item for item in semantic.relations if item.predicate == "perpendicular")
    check = next(item for item in result.compatibility.checks if item.candidate_id == relation.id)

    assert check.status == "compatible"
    assert check.reason == "semantic_relation_matches_grounded_visual_marker"
    assert check.metrics == {"marker_count": 1, "marker_kinds": ["right_angle"]}
    assert "visual:marker:right_angle" in check.compared_ids


def test_fusion_certificate_does_not_call_a_grounded_target_compatible(tmp_path):
    parser_input = _parser_input(tmp_path)
    visual = _visual_graph(parser_input.id)
    semantic_point = ParseObject(
        id="semantic:point:A",
        kind=ParseObjectKind.POINT,
        origin=ParseOrigin.IMAGE,
        confidence=0.8,
        score_kind="fixture_semantic_point",
        visible_name="A",
    )
    semantic_target = ParseTarget(
        id="semantic:target:ab",
        kind="length",
        mode="value",
        arguments=(ParseReference("object", semantic_point.id),),
        origin=ParseOrigin.TEXT,
        confidence=0.8,
        score_kind="fixture_semantic_target",
    )
    semantic = ParseGraph(
        id="semantic:target_graph",
        source_id=parser_input.id,
        objects=(semantic_point,),
        targets=(semantic_target,),
    )

    result = ParseGraphEvidenceFusion().fuse(visual, semantic)
    check = next(item for item in result.compatibility.checks if item.candidate_id == semantic_target.id)

    assert check.status == "unknown"
    assert check.reason == "target_object_references_are_grounded_without_local_target_match"


def _visual_graph_with_vertical_segment(graph: ParseGraph) -> ParseGraph:
    source_id = graph.source_id
    evidence = (
        _point_evidence(source_id, "C", (90.0, 0.0)),
        _label_evidence(source_id, "C", (92.0, 0.0, 98.0, 8.0)),
        ParseEvidence(
            id="visual:evidence:segment:BC",
            source_id=source_id,
            kind=ParseEvidenceKind.IMAGE_REGION,
            precision=EvidencePrecision.LOCALIZED,
            bbox=(89.0, 0.0, 91.0, 30.0),
        ),
    )
    objects = (
        _point("visual:point:C", "visual:evidence:point:C", (90.0, 0.0)),
        _label("visual:text:C", "visual:evidence:text:C", "C", (92.0, 0.0, 98.0, 8.0)),
        ParseObject(
            id="visual:segment:BC",
            kind=ParseObjectKind.SEGMENT,
            origin=ParseOrigin.IMAGE,
            confidence=0.9,
            score_kind="fixture_segment",
            references=(
                ParseReference("start", "visual:point:B"),
                ParseReference("end", "visual:point:C"),
            ),
            evidence_refs=("visual:evidence:segment:BC",),
            visible=True,
            geometry={"p1": [90.0, 30.0], "p2": [90.0, 0.0], "angle_deg": 90.0},
        ),
    )
    relations = (
        _binding("C", "visual:text:C", "visual:point:C"),
        ParseRelation(
            id="visual:ordered:BC",
            predicate="ordered_points",
            arguments=(
                ParseReference("support", "visual:segment:BC"),
                ParseReference("point_0", "visual:point:B"),
                ParseReference("point_1", "visual:point:C"),
            ),
            origin=ParseOrigin.IMAGE,
            confidence=0.9,
            score_kind="fixture_order",
            evidence_refs=("visual:evidence:segment:BC",),
        ),
    )
    return replace(
        graph,
        evidence=(*graph.evidence, *evidence),
        objects=(*graph.objects, *objects),
        relations=(*graph.relations, *relations),
    )


def _parser_input(tmp_path):
    image_path = tmp_path / "diagram.png"
    Image.new("RGB", (100, 60), "white").save(image_path)
    return ParserInput(id="fusion_sample", image=str(image_path), question="Find AB.")


def _semantic_graph(parser_input):
    prediction = GeoparsingPrediction.create(
        parser_input,
        backend="offline",
        model="GDP-4B",
        model_revision="revision-sha",
        prompt_version="official-v1",
        confidence=0.8,
        formal_description={
            "points": ["A", "B"],
            "lines": ["line A B"],
            "circles": [],
            "semantics": ["AB = 5"],
        },
    )
    return GeoparsingAdapter().convert(parser_input, prediction).prediction.parse_graph


def _visual_graph(source_id: str, *, include_quantity: bool = False, ambiguous_a: bool = False) -> ParseGraph:
    evidence = [
        _point_evidence(source_id, "A", (10.0, 30.0)),
        _point_evidence(source_id, "B", (90.0, 30.0)),
        _label_evidence(source_id, "A", (2.0, 34.0, 8.0, 44.0)),
        _label_evidence(source_id, "B", (92.0, 34.0, 98.0, 44.0)),
        ParseEvidence(
            id="visual:evidence:segment:AB",
            source_id=source_id,
            kind=ParseEvidenceKind.IMAGE_REGION,
            precision=EvidencePrecision.LOCALIZED,
            bbox=(10.0, 29.0, 90.0, 31.0),
        ),
    ]
    objects = [
        _point("visual:point:A", "visual:evidence:point:A", (10.0, 30.0)),
        _point("visual:point:B", "visual:evidence:point:B", (90.0, 30.0)),
        _label("visual:text:A", "visual:evidence:text:A", "A", (2.0, 34.0, 8.0, 44.0)),
        _label("visual:text:B", "visual:evidence:text:B", "B", (92.0, 34.0, 98.0, 44.0)),
        ParseObject(
            id="visual:segment:AB",
            kind=ParseObjectKind.SEGMENT,
            origin=ParseOrigin.IMAGE,
            confidence=0.9,
            score_kind="fixture_segment",
            references=(
                ParseReference("start", "visual:point:A"),
                ParseReference("end", "visual:point:B"),
            ),
            evidence_refs=("visual:evidence:segment:AB",),
            visible=True,
            geometry={"p1": [10.0, 30.0], "p2": [90.0, 30.0], "angle_deg": 0.0},
        ),
    ]
    relations = [
        _binding("A", "visual:text:A", "visual:point:A"),
        _binding("B", "visual:text:B", "visual:point:B"),
        ParseRelation(
            id="visual:ordered:AB",
            predicate="ordered_points",
            arguments=(
                ParseReference("support", "visual:segment:AB"),
                ParseReference("point_0", "visual:point:A"),
                ParseReference("point_1", "visual:point:B"),
            ),
            origin=ParseOrigin.IMAGE,
            confidence=0.9,
            score_kind="fixture_order",
            evidence_refs=("visual:evidence:segment:AB",),
        ),
    ]
    conflicts = []
    if ambiguous_a:
        evidence.append(_point_evidence(source_id, "A2", (14.0, 30.0)))
        objects.append(_point("visual:point:A2", "visual:evidence:point:A2", (14.0, 30.0)))
        alternate = _binding("A", "visual:text:A", "visual:point:A2")
        relations.append(alternate)
        conflicts.append(
            ParseConflictGroup(
                id="visual:conflict:A",
                members=("visual:binding:A", alternate.id),
                status=ConflictStatus.UNRESOLVED,
                reason="fixture_ambiguity",
            )
        )
    quantities = []
    if include_quantity:
        evidence.append(
            ParseEvidence(
                id="visual:evidence:quantity:AB",
                source_id=source_id,
                kind=ParseEvidenceKind.SOURCE_DOCUMENT,
                precision=EvidencePrecision.BOUND,
                raw_content="AB = 5",
            )
        )
        quantities.append(
            ParseQuantity(
                id="visual:quantity:AB",
                kind="length",
                arguments=(ParseReference("segment", "visual:segment:AB"),),
                expression=math_expression_from_text("5"),
                origin=ParseOrigin.IMAGE_TEXT,
                confidence=0.95,
                score_kind="fixture_bound_quantity",
                evidence_refs=("visual:evidence:quantity:AB",),
            )
        )
    return ParseGraph(
        id="visual:graph",
        source_id=source_id,
        evidence=tuple(evidence),
        objects=tuple(objects),
        relations=tuple(relations),
        quantities=tuple(quantities),
        conflicts=tuple(conflicts),
    )


def _point_evidence(source_id: str, suffix: str, point: tuple[float, float]) -> ParseEvidence:
    return ParseEvidence(
        id=f"visual:evidence:point:{suffix}",
        source_id=source_id,
        kind=ParseEvidenceKind.PIXEL_POINT,
        precision=EvidencePrecision.LOCALIZED,
        point=point,
    )


def _label_evidence(
    source_id: str,
    suffix: str,
    bbox: tuple[float, float, float, float],
) -> ParseEvidence:
    return ParseEvidence(
        id=f"visual:evidence:text:{suffix}",
        source_id=source_id,
        kind=ParseEvidenceKind.IMAGE_REGION,
        precision=EvidencePrecision.BOUND,
        bbox=bbox,
        raw_content=suffix,
    )


def _point(candidate_id: str, evidence_id: str, point: tuple[float, float]) -> ParseObject:
    return ParseObject(
        id=candidate_id,
        kind=ParseObjectKind.POINT,
        origin=ParseOrigin.IMAGE,
        confidence=0.9,
        score_kind="fixture_point",
        evidence_refs=(evidence_id,),
        geometry={"pixel": list(point)},
    )


def _label(
    candidate_id: str,
    evidence_id: str,
    name: str,
    bbox: tuple[float, float, float, float],
) -> ParseObject:
    return ParseObject(
        id=candidate_id,
        kind=ParseObjectKind.TEXT,
        origin=ParseOrigin.IMAGE,
        confidence=0.95,
        score_kind="fixture_ocr",
        evidence_refs=(evidence_id,),
        visible_name=name,
        visible=True,
        geometry={"bbox": list(bbox)},
    )


def _binding(name: str, label_id: str, point_id: str) -> ParseRelation:
    suffix = point_id.rsplit(":", 1)[-1]
    return ParseRelation(
        id=f"visual:binding:{name}" if suffix == name else f"visual:binding:{name}:{suffix}",
        predicate="binds",
        arguments=(ParseReference("label", label_id), ParseReference("object", point_id)),
        origin=ParseOrigin.IMAGE_TEXT,
        confidence=0.9,
        score_kind="fixture_binding",
        evidence_refs=(f"visual:evidence:text:{name}", f"visual:evidence:point:{suffix}"),
        metadata={"binding_kind": "point_name", "label": name},
    )
