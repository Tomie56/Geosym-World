from dataclasses import replace

from geosym_world.schema import (
    AdmissionStatus,
    EvidencePrecision,
    MathExpression,
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
)
from geosym_world.verifier import ParseAdmissionPolicy, ParseGraphAdmission


def test_admission_accepts_localized_topology_but_not_visual_metric_relation():
    graph = _graph()

    report = ParseGraphAdmission().admit(graph)

    assert "point_a" in report.admitted_ids
    assert "segment_ab" in report.admitted_ids
    assert "relation_order" in report.admitted_ids
    visual_metric = next(item for item in report.decisions if item.candidate_id == "relation_parallel")
    assert visual_metric.decision is AdmissionStatus.ABSTAINED
    assert visual_metric.reason == "evidence_precision_below_bound"
    assert report.admitted_graph.metadata["admission"]["source_graph_checksum"] == graph.checksum()


def test_bound_text_quantity_is_admitted_only_after_its_object():
    report = ParseGraphAdmission().admit(_graph())
    quantity = next(item for item in report.decisions if item.candidate_id == "quantity_ab")

    assert quantity.decision is AdmissionStatus.ADMITTED
    assert quantity.dependencies == ("segment_ab",)
    assert "quantity_ab" in report.admitted_ids


def test_model_structural_hypothesis_can_admit_without_pixel_localization():
    evidence = ParseEvidence(
        id="model_evidence",
        source_id="semantic_source",
        kind=ParseEvidenceKind.SOURCE_DOCUMENT,
        precision=EvidencePrecision.IMAGE_ONLY,
    )
    points = tuple(
        ParseObject(
            id=f"point_{name.lower()}",
            kind=ParseObjectKind.POINT,
            origin=ParseOrigin.IMAGE_TEXT,
            confidence=0.8,
            score_kind="model",
            visible_name=name,
            evidence_refs=(evidence.id,),
        )
        for name in "PQS"
    )
    segment = ParseObject(
        id="segment_pq",
        kind=ParseObjectKind.SEGMENT,
        origin=ParseOrigin.IMAGE_TEXT,
        confidence=0.8,
        score_kind="model",
        references=(ParseReference("start", "point_p"), ParseReference("end", "point_q")),
        evidence_refs=(evidence.id,),
    )
    relation = ParseRelation(
        id="s_on_pq",
        predicate="point_on_segment",
        arguments=(ParseReference("point", "point_s"), ParseReference("support", "segment_pq")),
        origin=ParseOrigin.IMAGE_TEXT,
        confidence=0.8,
        score_kind="model",
        evidence_refs=(evidence.id,),
    )
    graph = ParseGraph(
        id="semantic_graph",
        source_id="semantic_source",
        evidence=(evidence,),
        objects=(*points, segment),
        relations=(relation,),
    )

    report = ParseGraphAdmission().admit(graph)

    assert {item.id for item in report.admitted_graph.objects} == {
        "point_p", "point_q", "point_s", "segment_pq"
    }
    assert "s_on_pq" in report.admitted_ids
    decision = next(item for item in report.decisions if item.candidate_id == "s_on_pq")
    assert decision.reason == "model_structural_hypothesis"


def test_admission_abstains_on_dependency_rejected_by_confidence_policy():
    graph = _graph()
    objects = tuple(replace(item, confidence=0.2) if item.id == "point_a" else item for item in graph.objects)
    graph = replace(graph, objects=objects)

    report = ParseGraphAdmission(ParseAdmissionPolicy(minimum_confidence=0.5)).admit(graph)

    decisions = {item.candidate_id: item for item in report.decisions}
    assert decisions["point_a"].reason == "below_confidence_threshold"
    assert decisions["segment_ab"].reason == "dependency_not_admitted"
    assert decisions["relation_order"].reason == "dependency_not_admitted"
    assert {item.id for item in report.admitted_graph.objects} == {"point_b"}


def test_compiled_candidate_requires_admitted_dependencies():
    graph = _graph()
    fabricated = ParseObject(
        id="point_compiled_without_source",
        kind=ParseObjectKind.POINT,
        origin=ParseOrigin.COMPILED,
        confidence=1.0,
        score_kind="deterministic_rule",
        geometry={
            "exact_coords": [
                {"operator": "literal", "value": "0", "operands": []},
                {"operator": "literal", "value": "0", "operands": []},
            ]
        },
    )

    report = ParseGraphAdmission().admit(replace(graph, objects=(*graph.objects, fabricated)))

    decision = next(item for item in report.decisions if item.candidate_id == fabricated.id)
    assert decision.decision is AdmissionStatus.ABSTAINED
    assert decision.reason == "compiled_dependencies_required"
    assert fabricated.id not in report.admitted_ids


def test_observation_only_relation_is_retained_as_proposal_but_not_admitted():
    graph = _graph()
    relation = replace(
        graph.relations[0],
        id="relation_visual_observation",
        metadata={
            "observation_only": True,
            "coordinate_policy": "pixel_observation_only",
            "source_detector": "VisualPrimitiveDetector",
        },
    )
    graph = replace(graph, relations=(*graph.relations, relation))

    report = ParseGraphAdmission().admit(graph)

    decision = next(item for item in report.decisions if item.candidate_id == relation.id)
    assert decision.decision is AdmissionStatus.ABSTAINED
    assert decision.reason == "observation_only_candidate"
    assert relation.id not in report.admitted_ids
    assert any(item.id == relation.id for item in graph.relations)
    assert all(item.id != relation.id for item in report.admitted_graph.relations)
    assert any(item.id == relation.id for item in report.observation_graph.relations)


def test_admission_revokes_non_unique_point_names_and_dependent_candidates():
    graph = _graph()
    label_evidence = tuple(
        ParseEvidence(
            id=f"evidence_label_{name}",
            source_id=graph.source_id,
            kind=ParseEvidenceKind.IMAGE_REGION,
            precision=EvidencePrecision.BOUND,
            bbox=(8.0 + index * 4.0, 2.0, 11.0 + index * 4.0, 7.0),
            raw_content=name,
        )
        for index, name in enumerate(("A", "B"))
    )
    labels = tuple(
        ParseObject(
            id=f"label_{name}",
            kind=ParseObjectKind.TEXT,
            origin=ParseOrigin.IMAGE_TEXT,
            confidence=1.0,
            score_kind="fixture",
            evidence_refs=(item.id,),
            visible_name=name,
            geometry={"bbox": list(item.bbox)},
        )
        for name, item in zip(("A", "B"), label_evidence)
    )
    bindings = tuple(
        ParseRelation(
            id=f"binding_{name}",
            predicate="binds",
            arguments=(ParseReference("label", f"label_{name}"), ParseReference("object", "point_a")),
            origin=ParseOrigin.IMAGE_TEXT,
            confidence=1.0,
            score_kind="fixture",
            evidence_refs=(item.id, "evidence_point_a"),
            metadata={"binding_kind": "point_name", "label": name},
        )
        for name, item in zip(("A", "B"), label_evidence)
    )
    target = ParseTarget(
        id="target_with_name_dependencies",
        kind="length",
        mode="value",
        arguments=(ParseReference("object", "segment_ab"),),
        origin=ParseOrigin.TEXT,
        confidence=1.0,
        score_kind="fixture",
        evidence_refs=(label_evidence[0].id,),
        dependencies=tuple(item.id for item in bindings),
    )
    graph = replace(
        graph,
        evidence=(*graph.evidence, *label_evidence),
        objects=(*graph.objects, *labels),
        relations=(*graph.relations, *bindings),
        targets=(target,),
    )

    report = ParseGraphAdmission().admit(graph)
    decisions = {item.candidate_id: item for item in report.decisions}

    assert {decisions[item.id].reason for item in bindings} == {"non_unique_point_name_binding"}
    assert decisions[target.id].reason == "dependency_not_admitted"
    assert all(item.id not in report.admitted_ids for item in (*bindings, target))


def _graph() -> ParseGraph:
    point_evidence = ParseEvidence(
        id="evidence_point_a",
        source_id="source_1",
        kind=ParseEvidenceKind.PIXEL_POINT,
        precision=EvidencePrecision.LOCALIZED,
        point=(10.0, 10.0),
    )
    point_b_evidence = replace(point_evidence, id="evidence_point_b", point=(20.0, 10.0))
    line_evidence = ParseEvidence(
        id="evidence_line_ab",
        source_id="source_1",
        kind=ParseEvidenceKind.IMAGE_REGION,
        precision=EvidencePrecision.LOCALIZED,
        bbox=(10.0, 9.0, 20.0, 11.0),
    )
    text_evidence = ParseEvidence(
        id="evidence_text_4",
        source_id="source_1",
        kind=ParseEvidenceKind.IMAGE_REGION,
        precision=EvidencePrecision.BOUND,
        bbox=(14.0, 12.0, 16.0, 16.0),
        raw_content="4",
    )
    point_a = _point("point_a", point_evidence.id)
    point_b = _point("point_b", point_b_evidence.id)
    segment = ParseObject(
        id="segment_ab",
        kind=ParseObjectKind.SEGMENT,
        origin=ParseOrigin.IMAGE,
        confidence=0.9,
        score_kind="fixture",
        references=(ParseReference("start", point_a.id), ParseReference("end", point_b.id)),
        evidence_refs=(line_evidence.id,),
        visible=True,
    )
    quantity = ParseQuantity(
        id="quantity_ab",
        kind="length",
        arguments=(ParseReference("object", segment.id),),
        expression=MathExpression("literal", value="4"),
        origin=ParseOrigin.IMAGE_TEXT,
        confidence=0.9,
        score_kind="fixture",
        evidence_refs=(text_evidence.id,),
    )
    ordered = ParseRelation(
        id="relation_order",
        predicate="ordered_points",
        arguments=(
            ParseReference("support", segment.id),
            ParseReference("point_0", point_a.id),
            ParseReference("point_1", point_b.id),
        ),
        origin=ParseOrigin.IMAGE,
        confidence=0.9,
        score_kind="fixture",
        evidence_refs=(line_evidence.id,),
    )
    parallel = ParseRelation(
        id="relation_parallel",
        predicate="parallel",
        arguments=(ParseReference("left", segment.id), ParseReference("right", segment.id)),
        origin=ParseOrigin.IMAGE,
        confidence=0.9,
        score_kind="pixel_angle_residual",
        evidence_refs=(line_evidence.id,),
        metadata={"exact_relation_not_admitted": True},
    )
    return ParseGraph(
        id="graph_1",
        source_id="source_1",
        evidence=(point_evidence, point_b_evidence, line_evidence, text_evidence),
        objects=(point_a, point_b, segment),
        relations=(ordered, parallel),
        quantities=(quantity,),
    )


def _point(point_id: str, evidence_id: str) -> ParseObject:
    return ParseObject(
        id=point_id,
        kind=ParseObjectKind.POINT,
        origin=ParseOrigin.IMAGE,
        confidence=0.9,
        score_kind="fixture",
        evidence_refs=(evidence_id,),
        visible=True,
    )
