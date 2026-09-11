import json
from dataclasses import replace
from pathlib import Path

import pytest

from geosym_world.benchmark import (
    AdjudicationStatus,
    DisagreementCategory,
    PARSER_GT_GUIDELINE_VERSION,
    ParserAdjudicationResolution,
    ParserAnnotationAdjudication,
    ParserAnnotationDisagreement,
    ParserAnnotationPolicy,
    ParserAnnotationRecord,
    ParserAnnotationReview,
    ParserProtocol,
    ParserSourceManifest,
    ResolutionChoice,
    ReviewDecision,
    build_adjudicated_parser_reference,
    hash_asset,
    materialize_parser_input,
    validate_parser_annotation_chain,
    validate_typed_parser_reference,
)
from geosym_world.parser import ParserInput
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
)
from geosym_world.schema.canonical import stable_checksum
from geosym_world.tools.materialize_parser_references_v0 import run


def test_adjudicated_human_chain_materializes_checksum_bound_reference(tmp_path):
    parser_input, manifest, _, annotation, review, adjudication = _chain(tmp_path)

    reference, lineage = build_adjudicated_parser_reference(
        parser_input,
        manifest=manifest,
        annotation=annotation,
        review=review,
        adjudication=adjudication,
    )

    restored_manifest, restored_graph = validate_typed_parser_reference(reference, parser_input)
    assert restored_manifest == manifest
    assert restored_graph == adjudication.final_graph
    assert lineage["annotation_checksum"] == annotation.checksum()
    assert lineage["review_checksum"] == review.checksum()
    assert lineage["adjudication_checksum"] == adjudication.checksum()
    assert lineage["reference_checksum"] == reference["checksum"]


def test_annotation_factories_derive_chain_bindings(tmp_path):
    parser_input, manifest, _, _, _, _ = _chain(tmp_path)
    graph = _graph(manifest.source_id)
    annotation = ParserAnnotationRecord.create(
        parser_input,
        manifest=manifest,
        annotation_id="annotation_factory",
        annotator_id="annotator_alpha",
        guideline_version=PARSER_GT_GUIDELINE_VERSION,
        parse_graph=graph,
    )
    review = ParserAnnotationReview.create(
        annotation,
        review_id="review_factory",
        reviewer_id="annotator_beta",
        independent_graph=graph,
        decision=ReviewDecision.AGREE,
    )
    adjudication = ParserAnnotationAdjudication.create(
        annotation,
        review,
        adjudication_id="adjudication_factory",
        adjudicator_id="adjudicator_gamma",
        status=AdjudicationStatus.ACCEPTED,
        final_graph=graph,
    )

    assert annotation.parser_input_checksum == stable_checksum(parser_input.to_dict())
    assert review.annotation_checksum == annotation.checksum()
    assert adjudication.review_checksum == review.checksum()
    assert validate_parser_annotation_chain(
        parser_input,
        manifest=manifest,
        annotation=annotation,
        review=review,
        adjudication=adjudication,
    ) == graph


def test_annotation_record_rejects_unknown_guideline_version(tmp_path):
    _, _, _, annotation, _, _ = _chain(tmp_path)

    with pytest.raises(ValueError, match="Unsupported parser GT guideline version"):
        replace(annotation, guideline_version="parser-gt-draft")


def test_annotation_records_reject_tampering_and_model_derived_candidates(tmp_path):
    parser_input, manifest, _, annotation, _, _ = _chain(tmp_path)
    payload = annotation.to_dict()
    payload["notes"] = "changed after signing"
    with pytest.raises(ValueError, match="checksum"):
        ParserAnnotationRecord.from_dict(payload)

    model_graph = _graph(manifest.source_id, score_kind="model_score", confidence=0.99)
    with pytest.raises(ValueError, match="human_annotation"):
        replace(annotation, parse_graph=model_graph)
    with pytest.raises(ValueError, match="model assistance"):
        replace(annotation, model_assistance=True)
    assert annotation.parser_input_checksum == stable_checksum(parser_input.to_dict())


def test_human_annotation_requires_canonical_executable_relation_roles(tmp_path):
    parser_input, manifest, _, annotation, _, _ = _chain(tmp_path)
    point_b = replace(annotation.parse_graph.objects[0], id="point_b", visible_name="B")
    point_m = replace(annotation.parse_graph.objects[0], id="point_m", visible_name="M")
    malformed = ParseRelation(
        id="relation_midpoint",
        predicate="midpoint",
        arguments=(
            ParseReference("point", "point_m"),
            ParseReference("endpoint_0", "point_a"),
            ParseReference("endpoint_1", "point_b"),
        ),
        origin=ParseOrigin.IMAGE_TEXT,
        confidence=1.0,
        score_kind="human_annotation",
        evidence_refs=(annotation.parse_graph.evidence[0].id,),
    )
    graph = replace(
        annotation.parse_graph,
        objects=(*annotation.parse_graph.objects, point_b, point_m),
        relations=(malformed,),
    )

    with pytest.raises(ValueError, match="canonical roles"):
        ParserAnnotationRecord.create(
            parser_input,
            manifest=manifest,
            annotation_id="annotation_bad_roles",
            annotator_id="annotator_alpha",
            guideline_version=PARSER_GT_GUIDELINE_VERSION,
            parse_graph=graph,
        )


def test_human_annotation_rejects_missing_evidence_native_origin_and_exact_geometry(tmp_path):
    parser_input, manifest, _, annotation, _, _ = _chain(tmp_path)
    point = annotation.parse_graph.objects[0]

    cases = (
        (replace(point, evidence_refs=()), "requires source evidence"),
        (replace(point, origin=ParseOrigin.NATIVE), "cannot use origin=native"),
        (
            replace(
                point,
                geometry={
                    **point.geometry,
                    "exact_coords": [
                        {"operator": "literal", "value": "0", "operands": []},
                        {"operator": "literal", "value": "0", "operands": []},
                    ],
                },
            ),
            "cannot contain exact geometry fields",
        ),
    )
    for index, (invalid_point, message) in enumerate(cases):
        graph = replace(annotation.parse_graph, objects=(invalid_point,))
        with pytest.raises(ValueError, match=message):
            ParserAnnotationRecord.create(
                parser_input,
                manifest=manifest,
                annotation_id=f"annotation_invalid_evidence_{index}",
                annotator_id="annotator_alpha",
                guideline_version=PARSER_GT_GUIDELINE_VERSION,
                parse_graph=graph,
            )


def test_annotation_chain_requires_independent_reviewer(tmp_path):
    parser_input, manifest, _, annotation, review, adjudication = _chain(tmp_path)
    same_person_review = replace(review, reviewer_id=annotation.annotator_id)
    rebound = replace(adjudication, review_checksum=same_person_review.checksum())

    with pytest.raises(ValueError, match="reviewer must differ"):
        validate_parser_annotation_chain(
            parser_input,
            manifest=manifest,
            annotation=annotation,
            review=same_person_review,
            adjudication=rebound,
        )


def test_agreeing_review_requires_identical_canonical_graph(tmp_path):
    parser_input, manifest, _, annotation, review, adjudication = _chain(tmp_path)
    changed_graph = replace(
        review.independent_graph,
        objects=(replace(review.independent_graph.objects[0], geometry={"pixel": [21.0, 30.0]}),),
    )
    changed_review = replace(review, independent_graph=changed_graph)
    rebound = replace(adjudication, review_checksum=changed_review.checksum())

    with pytest.raises(ValueError, match="identical canonical ParseGraphs"):
        validate_parser_annotation_chain(
            parser_input,
            manifest=manifest,
            annotation=annotation,
            review=changed_review,
            adjudication=rebound,
        )


def test_annotation_chain_rejects_unresolved_or_partially_resolved_disagreement(tmp_path):
    parser_input, manifest, _, annotation, review, adjudication = _chain(tmp_path, disagree=True)

    with pytest.raises(ValueError, match="leaves disagreements unresolved"):
        validate_parser_annotation_chain(
            parser_input,
            manifest=manifest,
            annotation=annotation,
            review=review,
            adjudication=adjudication,
        )

    unresolved = replace(
        adjudication,
        status=AdjudicationStatus.UNRESOLVED,
        final_graph=None,
        resolutions=(),
    )
    with pytest.raises(ValueError, match="unresolved and cannot become trusted GT"):
        validate_parser_annotation_chain(
            parser_input,
            manifest=manifest,
            annotation=annotation,
            review=review,
            adjudication=unresolved,
        )


def test_annotation_policy_can_require_third_person_adjudicator(tmp_path):
    parser_input, manifest, _, annotation, review, adjudication = _chain(tmp_path)
    same_person_adjudication = replace(adjudication, adjudicator_id=review.reviewer_id)

    with pytest.raises(ValueError, match="adjudicator must differ"):
        validate_parser_annotation_chain(
            parser_input,
            manifest=manifest,
            annotation=annotation,
            review=review,
            adjudication=same_person_adjudication,
            policy=ParserAnnotationPolicy(require_distinct_adjudicator=True),
        )


def test_annotation_materialization_cli_writes_reference_lineage_and_manifest(tmp_path):
    parser_input, _, source_audit, annotation, review, adjudication = _chain(tmp_path)
    inputs_path = tmp_path / "parser-input.jsonl"
    source_audit_path = tmp_path / "source-audit.jsonl"
    annotations_path = tmp_path / "annotations.jsonl"
    reviews_path = tmp_path / "reviews.jsonl"
    adjudications_path = tmp_path / "adjudications.jsonl"
    output_path = tmp_path / "references.jsonl"
    _write_jsonl(inputs_path, [parser_input.to_dict()])
    _write_jsonl(source_audit_path, [source_audit])
    _write_jsonl(annotations_path, [annotation.to_dict()])
    _write_jsonl(reviews_path, [review.to_dict()])
    _write_jsonl(adjudications_path, [adjudication.to_dict()])

    result = run(
        inputs_path=inputs_path,
        source_audit_path=source_audit_path,
        annotations_path=annotations_path,
        reviews_path=reviews_path,
        adjudications_path=adjudications_path,
        output_path=output_path,
        progress_every=0,
    )

    lineage_path = tmp_path / "references.lineage.jsonl"
    reference = json.loads(output_path.read_text(encoding="utf-8"))
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    assert result["num_records"] == 1
    assert result["references"]["path"] == str(output_path)
    assert reference["id"] == parser_input.id
    assert lineage["reference_checksum"] == reference["checksum"]
    assert Path(result["manifest"]).is_file()


def _chain(tmp_path, *, disagree: bool = False):
    source = tmp_path / "source.png"
    source.write_bytes(b"stable-parser-annotation-fixture")
    manifest = ParserSourceManifest(
        source_id="source_real_1",
        protocol=ParserProtocol.IMAGE_QUESTION,
        source_version="fixture-v1",
        license_id="test-only",
        asset_sha256=hash_asset(source),
        split_group="fixture-family-1",
        public_fields=("image", "question"),
    )
    parser_input, materialization = materialize_parser_input(
        source,
        manifest=manifest,
        output_dir=tmp_path / "prepared",
        question="Name the visible point.",
    )
    source_audit = {
        "schema_version": "geosym.parser_source_audit.v1",
        "public_id": parser_input.id,
        "source_image": str(source),
        "source_manifest": manifest.to_dict(),
        "materialization": materialization,
    }
    source_audit["checksum"] = stable_checksum(source_audit)
    graph = _graph(manifest.source_id)
    annotation = ParserAnnotationRecord(
        id=parser_input.id,
        annotation_id="annotation_1",
        parser_input_checksum=stable_checksum(parser_input.to_dict()),
        source_manifest_checksum=manifest.checksum(),
        annotator_id="annotator_alpha",
        guideline_version="parser-gt-v1",
        parse_graph=graph,
    )
    disagreements = (
        ParserAnnotationDisagreement(
            id="difference_point_location",
            category=DisagreementCategory.GEOMETRY,
            path="objects/point_a/geometry/pixel",
            description="Review position differs from the primary annotation.",
        ),
    ) if disagree else ()
    review = ParserAnnotationReview(
        id=parser_input.id,
        review_id="review_1",
        parser_input_checksum=stable_checksum(parser_input.to_dict()),
        source_manifest_checksum=manifest.checksum(),
        annotation_checksum=annotation.checksum(),
        reviewer_id="annotator_beta",
        guideline_version="parser-gt-v1",
        independent_graph=graph,
        decision=ReviewDecision.DISAGREE if disagree else ReviewDecision.AGREE,
        disagreements=disagreements,
    )
    adjudication = ParserAnnotationAdjudication(
        id=parser_input.id,
        adjudication_id="adjudication_1",
        parser_input_checksum=stable_checksum(parser_input.to_dict()),
        source_manifest_checksum=manifest.checksum(),
        annotation_checksum=annotation.checksum(),
        review_checksum=review.checksum(),
        adjudicator_id="adjudicator_gamma",
        guideline_version="parser-gt-v1",
        status=AdjudicationStatus.ACCEPTED,
        final_graph=graph,
    )
    return parser_input, manifest, source_audit, annotation, review, adjudication


def _graph(source_id: str, *, score_kind: str = "human_annotation", confidence: float = 1.0) -> ParseGraph:
    evidence = ParseEvidence(
        id="evidence_point_a",
        source_id=source_id,
        kind=ParseEvidenceKind.PIXEL_POINT,
        precision=EvidencePrecision.BOUND,
        point=(20.0, 30.0),
    )
    return ParseGraph(
        id="human_graph_1",
        source_id=source_id,
        evidence=(evidence,),
        objects=(
            ParseObject(
                id="point_a",
                kind=ParseObjectKind.POINT,
                origin=ParseOrigin.IMAGE,
                confidence=confidence,
                score_kind=score_kind,
                evidence_refs=(evidence.id,),
                visible_name="A",
                visible=True,
                geometry={"pixel": [20.0, 30.0]},
            ),
        ),
        metadata={"reference_kind": "human_annotation"},
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
