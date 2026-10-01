from dataclasses import replace

import pytest

from geosym_world.errors import InvalidGeometryInput
from geosym_world.parser import (
    ExternalParseGraphPrediction,
    ParserInput,
    ParserPipeline,
    SeedRealizationStatus,
)
from geosym_world.schema import (
    EvidencePrecision,
    ParseEvidence,
    ParseEvidenceKind,
    ParseGraph,
    ParseObject,
    ParseObjectKind,
    ParseOrigin,
    ParseReference,
    ParseRelation,
    ParseQuantity,
    MathExpression,
)


def test_parser_pipeline_preserves_every_stage_and_materializes_state(tmp_path):
    image = tmp_path / "diagram.png"
    image.write_bytes(b"fixture")
    parser_input = ParserInput(id="sample", image=str(image), question="")
    prediction = _prediction(parser_input)

    result = ParserPipeline(lambda _: prediction).run(parser_input)

    assert result.realization.status is SeedRealizationStatus.REALIZED
    assert result.realization.geometry_state is not None
    assert set(result.realization.geometry_state.points) == {"point_a", "point_b"}
    assert set(result.realization.geometry_state.segments) == {"segment_ab"}
    assert result.prediction.parse_graph.checksum() == result.admission.source_graph_checksum
    assert prediction.parse_graph.checksum() == result.realization.admitted_graph_checksum
    assert result.realization.metadata["state_scope"] == "candidate_instance"
    assert result.realization.metadata["source_facts_admitted"] is False
    assert all(point.metadata["parser_state_scope"] == "candidate_instance" for point in result.realization.geometry_state.points.values())
    lifecycle = {item.candidate_id: item for item in result.candidate_lifecycle}
    assert lifecycle["point_a"].admission_status == "admitted"
    assert lifecycle["point_a"].realization_status is None
    assert lifecycle["relation_order_ab"].admission_status == "admitted"
    assert lifecycle["relation_order_ab"].realization_status == "satisfied"
    assert lifecycle["relation_order_ab"].realization_usage == "construct_and_verify"
    quality = result.quality
    assert quality.payload_compiled
    assert quality.semantic_graph_complete is None
    assert quality.metrics["graph_integrity_passed"]
    assert quality.seed_realized
    assert quality.skeleton_built
    assert quality.skeleton_redraw
    assert quality.constraint_satisfied
    assert quality.target_bound
    assert not quality.pixel_topology_passed
    assert not quality.source_reconstruction_passed
    assert result.to_dict()["quality"]["source_reconstruction_passed"] is None
    assert result.to_dict()["checksum"] == result.checksum()


def test_pipeline_preserves_unadmitted_quantity_as_candidate_constraint(tmp_path):
    image = tmp_path / "diagram.png"
    image.write_bytes(b"fixture")
    parser_input = ParserInput(id="sample", image=str(image), question="")
    prediction = _prediction(parser_input)
    quantity = ParseQuantity(
        id="length_ab", kind="length", expression=MathExpression("literal", value="5"),
        arguments=(ParseReference("object", "segment_ab"),),
        origin=ParseOrigin.IMAGE, confidence=0.9, score_kind="model",
        evidence_refs=("evidence_segment_ab",),
    )
    graph = replace(prediction.parse_graph, quantities=(quantity,))
    prediction = replace(prediction, parse_graph=graph)
    original_checksum = graph.checksum()

    result = ParserPipeline(lambda _: prediction).run(parser_input)

    decisions = {item.candidate_id: item for item in result.candidate_lifecycle}
    assert decisions[quantity.id].admission_status == "abstained"
    assert decisions[quantity.id].realization_status == "satisfied"
    assert quantity.id in result.realization.metadata["selected_assumption_ids"]
    assert not result.admission.admitted_graph.quantities
    assert graph.checksum() == original_checksum
    assert result.quality.source_reconstruction_passed is None


def test_quality_counts_missing_constraints_even_if_remaining_subset_satisfied(tmp_path):
    image = tmp_path / "diagram.png"
    image.write_bytes(b"fixture")
    parser_input = ParserInput(id="sample", image=str(image), question="")
    prediction = _prediction(parser_input)
    result = ParserPipeline(lambda _: prediction).run(parser_input)
    incomplete = replace(result, realization=replace(result.realization, constraint_outcomes=()))

    assert not incomplete.quality.constraint_satisfied
    assert incomplete.quality.metrics["missing_constraint_outcomes"] == ["relation_order_ab"]


def test_parser_pipeline_rejects_prediction_for_another_input(tmp_path):
    image = tmp_path / "diagram.png"
    image.write_bytes(b"fixture")
    first = ParserInput(id="sample", image=str(image), question="first")
    second = ParserInput(id="sample", image=str(image), question="second")

    with pytest.raises(InvalidGeometryInput, match="does not match"):
        ParserPipeline(lambda _: _prediction(first)).run(second)


def test_parser_pipeline_requires_typed_external_prediction(tmp_path):
    image = tmp_path / "diagram.png"
    image.write_bytes(b"fixture")
    parser_input = ParserInput(id="sample", image=str(image), question="")

    with pytest.raises(TypeError, match="ExternalParseGraphPrediction"):
        ParserPipeline(lambda _: _prediction(parser_input).to_dict()).run(parser_input)


def _prediction(parser_input: ParserInput) -> ExternalParseGraphPrediction:
    point_a_evidence = ParseEvidence(
        id="evidence_point_a",
        source_id=parser_input.id,
        kind=ParseEvidenceKind.PIXEL_POINT,
        precision=EvidencePrecision.LOCALIZED,
        point=(10.0, 10.0),
    )
    point_b_evidence = ParseEvidence(
        id="evidence_point_b",
        source_id=parser_input.id,
        kind=ParseEvidenceKind.PIXEL_POINT,
        precision=EvidencePrecision.LOCALIZED,
        point=(20.0, 10.0),
    )
    segment_evidence = ParseEvidence(
        id="evidence_segment_ab",
        source_id=parser_input.id,
        kind=ParseEvidenceKind.IMAGE_REGION,
        precision=EvidencePrecision.LOCALIZED,
        bbox=(10.0, 9.0, 20.0, 11.0),
    )
    point_a = ParseObject(
        id="point_a",
        kind=ParseObjectKind.POINT,
        origin=ParseOrigin.IMAGE,
        confidence=0.9,
        score_kind="model",
        evidence_refs=(point_a_evidence.id,),
        visible_name="A",
    )
    point_b = ParseObject(
        id="point_b",
        kind=ParseObjectKind.POINT,
        origin=ParseOrigin.IMAGE,
        confidence=0.9,
        score_kind="model",
        evidence_refs=(point_b_evidence.id,),
        visible_name="B",
    )
    segment = ParseObject(
        id="segment_ab",
        kind=ParseObjectKind.SEGMENT,
        origin=ParseOrigin.IMAGE,
        confidence=0.9,
        score_kind="model",
        references=(ParseReference("start", point_a.id), ParseReference("end", point_b.id)),
        evidence_refs=(segment_evidence.id,),
    )
    order = ParseRelation(
        id="relation_order_ab",
        predicate="ordered_points",
        arguments=(
            ParseReference("support", segment.id),
            ParseReference("point_0", point_a.id),
            ParseReference("point_1", point_b.id),
        ),
        origin=ParseOrigin.IMAGE,
        confidence=0.9,
        score_kind="model",
        evidence_refs=(segment_evidence.id,),
    )
    graph = ParseGraph(
        id="graph_sample",
        source_id=parser_input.id,
        evidence=(point_a_evidence, point_b_evidence, segment_evidence),
        objects=(point_a, point_b, segment),
        relations=(order,),
    )
    return ExternalParseGraphPrediction.create(
        parser_input,
        backend="fixture",
        model="fixture-model",
        model_revision="v1",
        prompt_version="fixture-prompt",
        parse_graph=graph,
    )
