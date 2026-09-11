"""Private human annotation, review, and adjudication contracts for Parser GT."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from geosym_world.errors import InvalidGeometryInput
from geosym_world.parser.contracts import ParserInput
from geosym_world.schema import (
    ConflictStatus,
    MathExpression,
    ParseEvidenceKind,
    ParseGraph,
    ParseObject,
    ParseOrigin,
    ParseRelation,
)
from geosym_world.schema.canonical import stable_checksum
from geosym_world.schema.ids import validate_geometry_id

from .parser_sources import ParserSourceManifest, build_typed_parser_reference


PARSER_ANNOTATION_SCHEMA_VERSION = "geosym.parser_annotation.v1"
PARSER_ANNOTATION_REVIEW_SCHEMA_VERSION = "geosym.parser_annotation_review.v1"
PARSER_ANNOTATION_ADJUDICATION_SCHEMA_VERSION = "geosym.parser_annotation_adjudication.v1"
PARSER_ANNOTATION_LINEAGE_SCHEMA_VERSION = "geosym.parser_annotation_lineage.v1"
PARSER_GT_GUIDELINE_VERSION = "parser-gt-v1"


class AnnotationStage(str, Enum):
    DRAFT = "draft"
    REVIEWED = "reviewed"
    ADJUDICATED = "adjudicated"


class ReviewDecision(str, Enum):
    AGREE = "agree"
    DISAGREE = "disagree"


class AdjudicationStatus(str, Enum):
    ACCEPTED = "accepted"
    UNRESOLVED = "unresolved"


class DisagreementCategory(str, Enum):
    MISSING_CANDIDATE = "missing_candidate"
    EXTRA_CANDIDATE = "extra_candidate"
    TYPE = "type"
    GEOMETRY = "geometry"
    BINDING = "binding"
    TOPOLOGY = "topology"
    QUANTITY = "quantity"
    REGION = "region"
    TARGET = "target"
    EVIDENCE = "evidence"
    OTHER = "other"


class ResolutionChoice(str, Enum):
    PRIMARY = "primary"
    REVIEW = "review"
    MERGED = "merged"
    EXCLUDED = "excluded"
    OTHER = "other"


@dataclass(frozen=True)
class ParserAnnotationDisagreement:
    id: str
    category: DisagreementCategory
    path: str
    description: str

    def __post_init__(self) -> None:
        validate_geometry_id(self.id)
        if not isinstance(self.category, DisagreementCategory):
            raise InvalidGeometryInput(f"Unknown annotation disagreement category: {self.category}")
        _require_text(self.path, "Annotation disagreement path")
        _require_text(self.description, "Annotation disagreement description")

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "category": self.category.value,
            "path": self.path,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParserAnnotationDisagreement":
        _check_fields(data, {"id", "category", "path", "description"}, "annotation disagreement")
        return cls(
            id=str(data["id"]),
            category=DisagreementCategory(data["category"]),
            path=str(data["path"]),
            description=str(data["description"]),
        )


@dataclass(frozen=True)
class ParserAdjudicationResolution:
    disagreement_id: str
    choice: ResolutionChoice
    rationale: str

    def __post_init__(self) -> None:
        validate_geometry_id(self.disagreement_id)
        if not isinstance(self.choice, ResolutionChoice):
            raise InvalidGeometryInput(f"Unknown annotation resolution choice: {self.choice}")
        _require_text(self.rationale, "Annotation resolution rationale")

    def to_dict(self) -> dict[str, str]:
        return {
            "disagreement_id": self.disagreement_id,
            "choice": self.choice.value,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParserAdjudicationResolution":
        _check_fields(data, {"disagreement_id", "choice", "rationale"}, "annotation resolution")
        return cls(
            disagreement_id=str(data["disagreement_id"]),
            choice=ResolutionChoice(data["choice"]),
            rationale=str(data["rationale"]),
        )


@dataclass(frozen=True)
class ParserAnnotationRecord:
    id: str
    annotation_id: str
    parser_input_checksum: str
    source_manifest_checksum: str
    annotator_id: str
    guideline_version: str
    parse_graph: ParseGraph
    notes: str = ""
    model_assistance: bool = False
    stage: AnnotationStage = AnnotationStage.DRAFT
    schema_version: str = PARSER_ANNOTATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_private_record_identity(self.id, self.annotation_id, self.annotator_id)
        _require_checksum(self.parser_input_checksum, "Parser input checksum")
        _require_checksum(self.source_manifest_checksum, "Source manifest checksum")
        _require_text(self.guideline_version, "Annotation guideline version")
        _validate_guideline_version(self.guideline_version)
        _require_text_or_empty(self.notes, "Annotation notes")
        if self.model_assistance:
            raise InvalidGeometryInput("Trusted human annotations cannot declare model assistance")
        if self.stage is not AnnotationStage.DRAFT:
            raise InvalidGeometryInput("Parser annotation records must have stage=draft")
        if self.schema_version != PARSER_ANNOTATION_SCHEMA_VERSION:
            raise InvalidGeometryInput(f"Unsupported parser annotation schema version: {self.schema_version}")
        _validate_human_reference_graph(self.parse_graph)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stage": self.stage.value,
            "id": self.id,
            "annotation_id": self.annotation_id,
            "parser_input_checksum": self.parser_input_checksum,
            "source_manifest_checksum": self.source_manifest_checksum,
            "annotator_id": self.annotator_id,
            "guideline_version": self.guideline_version,
            "model_assistance": self.model_assistance,
            "parse_graph": self.parse_graph.to_dict(),
            "notes": self.notes,
        }

    def checksum(self) -> str:
        return stable_checksum(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["checksum"] = stable_checksum(payload)
        return payload

    @classmethod
    def create(
        cls,
        parser_input: ParserInput | Mapping[str, Any],
        *,
        manifest: ParserSourceManifest,
        annotation_id: str,
        annotator_id: str,
        guideline_version: str,
        parse_graph: ParseGraph,
        notes: str = "",
    ) -> "ParserAnnotationRecord":
        public_input = parser_input if isinstance(parser_input, ParserInput) else ParserInput.from_dict(parser_input)
        build_typed_parser_reference(public_input, graph=parse_graph, manifest=manifest)
        return cls(
            id=public_input.id,
            annotation_id=annotation_id,
            parser_input_checksum=stable_checksum(public_input.to_dict()),
            source_manifest_checksum=manifest.checksum(),
            annotator_id=annotator_id,
            guideline_version=guideline_version,
            parse_graph=parse_graph,
            notes=notes,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParserAnnotationRecord":
        required = {
            "schema_version",
            "stage",
            "id",
            "annotation_id",
            "parser_input_checksum",
            "source_manifest_checksum",
            "annotator_id",
            "guideline_version",
            "model_assistance",
            "parse_graph",
            "notes",
            "checksum",
        }
        _check_fields(data, required, "parser annotation")
        _validate_record_checksum(data, "Parser annotation")
        return cls(
            id=str(data["id"]),
            annotation_id=str(data["annotation_id"]),
            parser_input_checksum=str(data["parser_input_checksum"]),
            source_manifest_checksum=str(data["source_manifest_checksum"]),
            annotator_id=str(data["annotator_id"]),
            guideline_version=str(data["guideline_version"]),
            parse_graph=ParseGraph.from_dict(data["parse_graph"]),
            notes=str(data["notes"]),
            model_assistance=_strict_bool(data["model_assistance"], "Parser annotation model_assistance"),
            stage=AnnotationStage(data["stage"]),
            schema_version=str(data["schema_version"]),
        )


@dataclass(frozen=True)
class ParserAnnotationReview:
    id: str
    review_id: str
    parser_input_checksum: str
    source_manifest_checksum: str
    annotation_checksum: str
    reviewer_id: str
    guideline_version: str
    independent_graph: ParseGraph
    decision: ReviewDecision
    disagreements: tuple[ParserAnnotationDisagreement, ...] = ()
    notes: str = ""
    model_assistance: bool = False
    stage: AnnotationStage = AnnotationStage.REVIEWED
    schema_version: str = PARSER_ANNOTATION_REVIEW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_private_record_identity(self.id, self.review_id, self.reviewer_id)
        _require_checksum(self.parser_input_checksum, "Parser input checksum")
        _require_checksum(self.source_manifest_checksum, "Source manifest checksum")
        _require_checksum(self.annotation_checksum, "Annotation checksum")
        _require_text(self.guideline_version, "Annotation guideline version")
        _validate_guideline_version(self.guideline_version)
        _require_text_or_empty(self.notes, "Annotation review notes")
        if self.model_assistance:
            raise InvalidGeometryInput("Trusted human reviews cannot declare model assistance")
        if self.stage is not AnnotationStage.REVIEWED:
            raise InvalidGeometryInput("Parser annotation reviews must have stage=reviewed")
        if self.schema_version != PARSER_ANNOTATION_REVIEW_SCHEMA_VERSION:
            raise InvalidGeometryInput(f"Unsupported parser annotation review schema version: {self.schema_version}")
        if not isinstance(self.decision, ReviewDecision):
            raise InvalidGeometryInput(f"Unknown parser annotation review decision: {self.decision}")
        disagreement_ids = [item.id for item in self.disagreements]
        if len(disagreement_ids) != len(set(disagreement_ids)):
            raise InvalidGeometryInput("Parser annotation review disagreement IDs must be unique")
        if self.decision is ReviewDecision.AGREE and self.disagreements:
            raise InvalidGeometryInput("An agreeing parser annotation review cannot contain disagreements")
        if self.decision is ReviewDecision.DISAGREE and not self.disagreements:
            raise InvalidGeometryInput("A disagreeing parser annotation review requires disagreements")
        _validate_human_reference_graph(self.independent_graph)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stage": self.stage.value,
            "id": self.id,
            "review_id": self.review_id,
            "parser_input_checksum": self.parser_input_checksum,
            "source_manifest_checksum": self.source_manifest_checksum,
            "annotation_checksum": self.annotation_checksum,
            "reviewer_id": self.reviewer_id,
            "guideline_version": self.guideline_version,
            "model_assistance": self.model_assistance,
            "independent_graph": self.independent_graph.to_dict(),
            "decision": self.decision.value,
            "disagreements": [item.to_dict() for item in self.disagreements],
            "notes": self.notes,
        }

    def checksum(self) -> str:
        return stable_checksum(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["checksum"] = stable_checksum(payload)
        return payload

    @classmethod
    def create(
        cls,
        annotation: ParserAnnotationRecord,
        *,
        review_id: str,
        reviewer_id: str,
        independent_graph: ParseGraph,
        decision: ReviewDecision,
        disagreements: tuple[ParserAnnotationDisagreement, ...] = (),
        notes: str = "",
    ) -> "ParserAnnotationReview":
        if independent_graph.source_id != annotation.parse_graph.source_id:
            raise InvalidGeometryInput("Independent review graph source_id does not match primary annotation")
        return cls(
            id=annotation.id,
            review_id=review_id,
            parser_input_checksum=annotation.parser_input_checksum,
            source_manifest_checksum=annotation.source_manifest_checksum,
            annotation_checksum=annotation.checksum(),
            reviewer_id=reviewer_id,
            guideline_version=annotation.guideline_version,
            independent_graph=independent_graph,
            decision=decision,
            disagreements=disagreements,
            notes=notes,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParserAnnotationReview":
        required = {
            "schema_version",
            "stage",
            "id",
            "review_id",
            "parser_input_checksum",
            "source_manifest_checksum",
            "annotation_checksum",
            "reviewer_id",
            "guideline_version",
            "model_assistance",
            "independent_graph",
            "decision",
            "disagreements",
            "notes",
            "checksum",
        }
        _check_fields(data, required, "parser annotation review")
        _validate_record_checksum(data, "Parser annotation review")
        return cls(
            id=str(data["id"]),
            review_id=str(data["review_id"]),
            parser_input_checksum=str(data["parser_input_checksum"]),
            source_manifest_checksum=str(data["source_manifest_checksum"]),
            annotation_checksum=str(data["annotation_checksum"]),
            reviewer_id=str(data["reviewer_id"]),
            guideline_version=str(data["guideline_version"]),
            independent_graph=ParseGraph.from_dict(data["independent_graph"]),
            decision=ReviewDecision(data["decision"]),
            disagreements=tuple(ParserAnnotationDisagreement.from_dict(item) for item in data["disagreements"]),
            notes=str(data["notes"]),
            model_assistance=_strict_bool(data["model_assistance"], "Parser annotation review model_assistance"),
            stage=AnnotationStage(data["stage"]),
            schema_version=str(data["schema_version"]),
        )


@dataclass(frozen=True)
class ParserAnnotationAdjudication:
    id: str
    adjudication_id: str
    parser_input_checksum: str
    source_manifest_checksum: str
    annotation_checksum: str
    review_checksum: str
    adjudicator_id: str
    guideline_version: str
    status: AdjudicationStatus
    final_graph: ParseGraph | None = None
    resolutions: tuple[ParserAdjudicationResolution, ...] = ()
    notes: str = ""
    model_assistance: bool = False
    stage: AnnotationStage = AnnotationStage.ADJUDICATED
    schema_version: str = PARSER_ANNOTATION_ADJUDICATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_private_record_identity(self.id, self.adjudication_id, self.adjudicator_id)
        _require_checksum(self.parser_input_checksum, "Parser input checksum")
        _require_checksum(self.source_manifest_checksum, "Source manifest checksum")
        _require_checksum(self.annotation_checksum, "Annotation checksum")
        _require_checksum(self.review_checksum, "Annotation review checksum")
        _require_text(self.guideline_version, "Annotation guideline version")
        _validate_guideline_version(self.guideline_version)
        _require_text_or_empty(self.notes, "Annotation adjudication notes")
        if self.model_assistance:
            raise InvalidGeometryInput("Trusted human adjudications cannot declare model assistance")
        if self.stage is not AnnotationStage.ADJUDICATED:
            raise InvalidGeometryInput("Parser annotation adjudications must have stage=adjudicated")
        if self.schema_version != PARSER_ANNOTATION_ADJUDICATION_SCHEMA_VERSION:
            raise InvalidGeometryInput(f"Unsupported parser annotation adjudication schema version: {self.schema_version}")
        if not isinstance(self.status, AdjudicationStatus):
            raise InvalidGeometryInput(f"Unknown parser annotation adjudication status: {self.status}")
        resolution_ids = [item.disagreement_id for item in self.resolutions]
        if len(resolution_ids) != len(set(resolution_ids)):
            raise InvalidGeometryInput("Parser annotation adjudication resolution IDs must be unique")
        if self.status is AdjudicationStatus.ACCEPTED and self.final_graph is None:
            raise InvalidGeometryInput("Accepted parser annotation adjudication requires a final graph")
        if self.status is AdjudicationStatus.UNRESOLVED and self.final_graph is not None:
            raise InvalidGeometryInput("Unresolved parser annotation adjudication cannot contain a final graph")
        if self.final_graph is not None:
            _validate_human_reference_graph(self.final_graph)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stage": self.stage.value,
            "id": self.id,
            "adjudication_id": self.adjudication_id,
            "parser_input_checksum": self.parser_input_checksum,
            "source_manifest_checksum": self.source_manifest_checksum,
            "annotation_checksum": self.annotation_checksum,
            "review_checksum": self.review_checksum,
            "adjudicator_id": self.adjudicator_id,
            "guideline_version": self.guideline_version,
            "model_assistance": self.model_assistance,
            "status": self.status.value,
            "final_graph": self.final_graph.to_dict() if self.final_graph is not None else None,
            "resolutions": [item.to_dict() for item in self.resolutions],
            "notes": self.notes,
        }

    def checksum(self) -> str:
        return stable_checksum(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["checksum"] = stable_checksum(payload)
        return payload

    @classmethod
    def create(
        cls,
        annotation: ParserAnnotationRecord,
        review: ParserAnnotationReview,
        *,
        adjudication_id: str,
        adjudicator_id: str,
        status: AdjudicationStatus,
        final_graph: ParseGraph | None = None,
        resolutions: tuple[ParserAdjudicationResolution, ...] = (),
        notes: str = "",
    ) -> "ParserAnnotationAdjudication":
        if review.annotation_checksum != annotation.checksum():
            raise InvalidGeometryInput("Parser annotation review does not reference the supplied annotation")
        return cls(
            id=annotation.id,
            adjudication_id=adjudication_id,
            parser_input_checksum=annotation.parser_input_checksum,
            source_manifest_checksum=annotation.source_manifest_checksum,
            annotation_checksum=annotation.checksum(),
            review_checksum=review.checksum(),
            adjudicator_id=adjudicator_id,
            guideline_version=annotation.guideline_version,
            status=status,
            final_graph=final_graph,
            resolutions=resolutions,
            notes=notes,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParserAnnotationAdjudication":
        required = {
            "schema_version",
            "stage",
            "id",
            "adjudication_id",
            "parser_input_checksum",
            "source_manifest_checksum",
            "annotation_checksum",
            "review_checksum",
            "adjudicator_id",
            "guideline_version",
            "model_assistance",
            "status",
            "final_graph",
            "resolutions",
            "notes",
            "checksum",
        }
        _check_fields(data, required, "parser annotation adjudication")
        _validate_record_checksum(data, "Parser annotation adjudication")
        final_graph = data["final_graph"]
        return cls(
            id=str(data["id"]),
            adjudication_id=str(data["adjudication_id"]),
            parser_input_checksum=str(data["parser_input_checksum"]),
            source_manifest_checksum=str(data["source_manifest_checksum"]),
            annotation_checksum=str(data["annotation_checksum"]),
            review_checksum=str(data["review_checksum"]),
            adjudicator_id=str(data["adjudicator_id"]),
            guideline_version=str(data["guideline_version"]),
            status=AdjudicationStatus(data["status"]),
            final_graph=ParseGraph.from_dict(final_graph) if final_graph is not None else None,
            resolutions=tuple(ParserAdjudicationResolution.from_dict(item) for item in data["resolutions"]),
            notes=str(data["notes"]),
            model_assistance=_strict_bool(data["model_assistance"], "Parser adjudication model_assistance"),
            stage=AnnotationStage(data["stage"]),
            schema_version=str(data["schema_version"]),
        )


@dataclass(frozen=True)
class ParserAnnotationPolicy:
    require_independent_reviewer: bool = True
    require_distinct_adjudicator: bool = False
    require_resolved_graph_conflicts: bool = True
    policy_id: str = "human_double_annotation_v1"

    def __post_init__(self) -> None:
        for name, value in (
            ("require_independent_reviewer", self.require_independent_reviewer),
            ("require_distinct_adjudicator", self.require_distinct_adjudicator),
            ("require_resolved_graph_conflicts", self.require_resolved_graph_conflicts),
        ):
            if not isinstance(value, bool):
                raise InvalidGeometryInput(f"Parser annotation policy {name} must be boolean")
        _require_text(self.policy_id, "Parser annotation policy id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "require_independent_reviewer": self.require_independent_reviewer,
            "require_distinct_adjudicator": self.require_distinct_adjudicator,
            "require_resolved_graph_conflicts": self.require_resolved_graph_conflicts,
        }


def build_adjudicated_parser_reference(
    parser_input: ParserInput | Mapping[str, Any],
    *,
    manifest: ParserSourceManifest,
    annotation: ParserAnnotationRecord,
    review: ParserAnnotationReview,
    adjudication: ParserAnnotationAdjudication,
    policy: ParserAnnotationPolicy | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one private human chain and materialize evaluator GT plus lineage."""

    public_input = parser_input if isinstance(parser_input, ParserInput) else ParserInput.from_dict(parser_input)
    active_policy = policy or ParserAnnotationPolicy()
    graph = validate_parser_annotation_chain(
        public_input,
        manifest=manifest,
        annotation=annotation,
        review=review,
        adjudication=adjudication,
        policy=active_policy,
    )
    reference = build_typed_parser_reference(public_input, graph=graph, manifest=manifest)
    lineage = {
        "schema_version": PARSER_ANNOTATION_LINEAGE_SCHEMA_VERSION,
        "id": public_input.id,
        "parser_input_checksum": stable_checksum(public_input.to_dict()),
        "source_manifest_checksum": manifest.checksum(),
        "annotation_checksum": annotation.checksum(),
        "review_checksum": review.checksum(),
        "adjudication_checksum": adjudication.checksum(),
        "final_graph_checksum": graph.checksum(),
        "reference_checksum": reference["checksum"],
        "annotator_id": annotation.annotator_id,
        "reviewer_id": review.reviewer_id,
        "adjudicator_id": adjudication.adjudicator_id,
        "guideline_version": adjudication.guideline_version,
        "policy": active_policy.to_dict(),
    }
    lineage["checksum"] = stable_checksum(lineage)
    return reference, lineage


def validate_parser_annotation_chain(
    parser_input: ParserInput | Mapping[str, Any],
    *,
    manifest: ParserSourceManifest,
    annotation: ParserAnnotationRecord,
    review: ParserAnnotationReview,
    adjudication: ParserAnnotationAdjudication,
    policy: ParserAnnotationPolicy | None = None,
) -> ParseGraph:
    """Return final trusted graph only when all private records form one valid chain."""

    public_input = parser_input if isinstance(parser_input, ParserInput) else ParserInput.from_dict(parser_input)
    active_policy = policy or ParserAnnotationPolicy()
    parser_input_checksum = stable_checksum(public_input.to_dict())
    manifest_checksum = manifest.checksum()
    records = (annotation, review, adjudication)
    for record in records:
        if record.id != public_input.id:
            raise InvalidGeometryInput("Parser annotation chain IDs do not match public input")
        if record.parser_input_checksum != parser_input_checksum:
            raise InvalidGeometryInput("Parser annotation chain input checksum does not match public input")
        if record.source_manifest_checksum != manifest_checksum:
            raise InvalidGeometryInput("Parser annotation chain source manifest checksum does not match")
    for graph in (annotation.parse_graph, review.independent_graph):
        if graph.source_id != manifest.source_id:
            raise InvalidGeometryInput("Parser annotation graph source_id does not match source manifest")
    if review.annotation_checksum != annotation.checksum():
        raise InvalidGeometryInput("Parser annotation review does not reference the supplied annotation")
    if adjudication.annotation_checksum != annotation.checksum():
        raise InvalidGeometryInput("Parser adjudication does not reference the supplied annotation")
    if adjudication.review_checksum != review.checksum():
        raise InvalidGeometryInput("Parser adjudication does not reference the supplied review")
    if len({annotation.guideline_version, review.guideline_version, adjudication.guideline_version}) != 1:
        raise InvalidGeometryInput("Parser annotation chain uses different guideline versions")
    if active_policy.require_independent_reviewer and annotation.annotator_id == review.reviewer_id:
        raise InvalidGeometryInput("Parser annotation reviewer must differ from the annotator")
    if active_policy.require_distinct_adjudicator and adjudication.adjudicator_id in {
        annotation.annotator_id,
        review.reviewer_id,
    }:
        raise InvalidGeometryInput("Parser annotation adjudicator must differ from annotator and reviewer")
    if adjudication.status is not AdjudicationStatus.ACCEPTED or adjudication.final_graph is None:
        raise InvalidGeometryInput("Parser annotation adjudication is unresolved and cannot become trusted GT")
    if adjudication.final_graph.source_id != manifest.source_id:
        raise InvalidGeometryInput("Adjudicated ParseGraph source_id does not match source manifest")
    if review.decision is ReviewDecision.AGREE:
        if annotation.parse_graph.checksum() != review.independent_graph.checksum():
            raise InvalidGeometryInput("Agreeing parser annotations must have identical canonical ParseGraphs")
        if adjudication.final_graph.checksum() != annotation.parse_graph.checksum():
            raise InvalidGeometryInput("Adjudication cannot alter a canonically agreed ParseGraph")
    disagreement_ids = {item.id for item in review.disagreements}
    resolution_ids = {item.disagreement_id for item in adjudication.resolutions}
    if resolution_ids - disagreement_ids:
        raise InvalidGeometryInput("Parser adjudication resolves unknown disagreement IDs")
    if disagreement_ids != resolution_ids:
        missing = sorted(disagreement_ids - resolution_ids)
        raise InvalidGeometryInput(f"Parser adjudication leaves disagreements unresolved: {missing}")
    if active_policy.require_resolved_graph_conflicts:
        unresolved = [
            conflict.id
            for conflict in adjudication.final_graph.conflicts
            if conflict.status is ConflictStatus.UNRESOLVED
        ]
        if unresolved:
            raise InvalidGeometryInput(f"Adjudicated ParseGraph contains unresolved conflicts: {unresolved}")
    return adjudication.final_graph


def _validate_human_reference_graph(graph: ParseGraph) -> None:
    if graph.metadata.get("reference_kind") != "human_annotation":
        raise InvalidGeometryInput("Human annotation ParseGraph requires metadata.reference_kind=human_annotation")
    candidates = (*graph.objects, *graph.relations, *graph.quantities, *graph.regions, *graph.targets)
    evidence_by_id = {item.id: item for item in graph.evidence}
    for candidate in candidates:
        if candidate.origin is ParseOrigin.NATIVE:
            raise InvalidGeometryInput(
                f"Human annotation candidate {candidate.id} cannot use origin=native"
            )
        expected_score = "deterministic_rule" if candidate.origin is ParseOrigin.COMPILED else "human_annotation"
        if candidate.score_kind != expected_score:
            raise InvalidGeometryInput(
                f"Human annotation candidate {candidate.id} requires score_kind={expected_score!r}"
            )
        if candidate.confidence != 1.0:
            raise InvalidGeometryInput(f"Human annotation candidate {candidate.id} requires confidence=1.0")
        if candidate.origin is not ParseOrigin.COMPILED and not candidate.evidence_refs:
            raise InvalidGeometryInput(
                f"Human annotation candidate {candidate.id} requires source evidence"
            )
        for evidence_id in candidate.evidence_refs:
            evidence = evidence_by_id[evidence_id]
            if evidence.source_id != graph.source_id:
                raise InvalidGeometryInput(
                    f"Human annotation candidate {candidate.id} references evidence from another source"
                )
            _validate_human_evidence_kind(candidate.id, candidate.origin, evidence.kind)
        if isinstance(candidate, ParseObject) and candidate.origin is not ParseOrigin.COMPILED:
            forbidden = {"exact_coords", "exact_radius", "exact_central_angle"} & set(candidate.geometry)
            if forbidden:
                raise InvalidGeometryInput(
                    f"Human annotation object {candidate.id} cannot contain exact geometry fields: {sorted(forbidden)}"
                )
    for relation in graph.relations:
        _validate_gt_relation_signature(relation)


def _validate_human_evidence_kind(
    candidate_id: str,
    origin: ParseOrigin,
    evidence_kind: ParseEvidenceKind,
) -> None:
    allowed = {
        ParseOrigin.IMAGE: {ParseEvidenceKind.IMAGE_REGION, ParseEvidenceKind.PIXEL_POINT},
        ParseOrigin.TEXT: {ParseEvidenceKind.TEXT_SPAN, ParseEvidenceKind.SOURCE_DOCUMENT},
        ParseOrigin.IMAGE_TEXT: {
            ParseEvidenceKind.IMAGE_REGION,
            ParseEvidenceKind.PIXEL_POINT,
            ParseEvidenceKind.TEXT_SPAN,
            ParseEvidenceKind.SOURCE_DOCUMENT,
        },
        ParseOrigin.COMPILED: set(ParseEvidenceKind),
    }[origin]
    if evidence_kind not in allowed:
        raise InvalidGeometryInput(
            f"Human annotation candidate {candidate_id} has evidence incompatible with origin={origin.value}"
        )


def _validate_gt_relation_signature(relation: ParseRelation) -> None:
    roles = tuple(argument.role for argument in relation.arguments)
    exact_roles = {
        "binds": ("label", "object"),
        "midpoint": ("point", "of_0", "of_1"),
        "foot_of_perpendicular": ("point", "from", "line"),
        "point_on_segment_ratio": ("point", "segment"),
        "point_on_ray_distance": ("point", "ray"),
        "parallel": ("object_0", "object_1"),
        "perpendicular": ("object_0", "object_1"),
        "angle_bisector": ("line", "arm_0", "vertex", "arm_1"),
        "angle_from_line": ("line", "point", "base_line"),
        "point_on_line": ("point", "line"),
        "point_on_ray": ("point", "line"),
        "point_on_circle": ("point", "circle"),
        "point_on_arc": ("point", "arc"),
    }
    expected = exact_roles.get(relation.predicate)
    if expected is not None and roles != expected:
        raise InvalidGeometryInput(
            f"Human annotation relation {relation.id} requires canonical roles {expected}; got {roles}"
        )
    if relation.predicate == "intersection":
        _validate_intersection_roles(relation, roles)
    elif relation.predicate == "ordered_points":
        _validate_ordered_point_roles(relation, roles)
    elif relation.predicate == "collinear":
        _validate_indexed_roles(relation, roles, "point", minimum=2)
    elif relation.predicate in {"point_on_segment_ratio", "point_on_ray_distance", "angle_from_line"}:
        field = {
            "point_on_segment_ratio": "ratio",
            "point_on_ray_distance": "distance",
            "angle_from_line": "angle",
        }[relation.predicate]
        expression = _required_relation_expression(relation, field)
        if expression.is_number is not True or expression.is_real is not True:
            raise InvalidGeometryInput(
                f"Human annotation relation {relation.id} metadata.{field} must be a real numeric MathExpression"
            )
        if field == "ratio" and not (
            expression.is_positive is True and (1 - expression).is_positive is True
        ):
            raise InvalidGeometryInput(
                f"Human annotation relation {relation.id} metadata.ratio must be strictly between 0 and 1"
            )
        if field == "distance" and expression.is_positive is not True:
            raise InvalidGeometryInput(
                f"Human annotation relation {relation.id} metadata.distance must be positive"
            )
    if relation.predicate == "angle_bisector" and relation.metadata.get("mode", "internal") not in {
        "internal",
        "external",
    }:
        raise InvalidGeometryInput(
            f"Human annotation relation {relation.id} metadata.mode must be internal or external"
        )


def _validate_intersection_roles(relation: ParseRelation, roles: tuple[str, ...]) -> None:
    if roles[:2] != ("support", "support"):
        raise InvalidGeometryInput(
            f"Human annotation relation {relation.id} requires two leading support roles"
        )
    point_roles = roles[2:]
    if point_roles == ("point",):
        return
    _validate_indexed_roles(relation, point_roles, "point", minimum=1)


def _validate_ordered_point_roles(relation: ParseRelation, roles: tuple[str, ...]) -> None:
    if not roles or roles[0] != "support":
        raise InvalidGeometryInput(
            f"Human annotation relation {relation.id} requires one leading support role"
        )
    _validate_indexed_roles(relation, roles[1:], "point", minimum=2)


def _validate_indexed_roles(
    relation: ParseRelation,
    roles: Sequence[str],
    prefix: str,
    *,
    minimum: int,
) -> None:
    expected = tuple(f"{prefix}_{index}" for index in range(len(roles)))
    if len(roles) < minimum or tuple(roles) != expected:
        raise InvalidGeometryInput(
            f"Human annotation relation {relation.id} requires contiguous {prefix}_0... roles"
        )


def _required_relation_expression(relation: ParseRelation, field: str):
    value = relation.metadata.get(field)
    if not isinstance(value, Mapping):
        raise InvalidGeometryInput(
            f"Human annotation relation {relation.id} metadata.{field} must use MathExpression"
        )
    try:
        return MathExpression.from_dict(value).to_sympy()
    except (InvalidGeometryInput, TypeError, ValueError) as exc:
        raise InvalidGeometryInput(
            f"Human annotation relation {relation.id} metadata.{field} must use MathExpression"
        ) from exc


def _validate_private_record_identity(sample_id: str, record_id: str, actor_id: str) -> None:
    _require_text(sample_id, "Parser annotation sample id")
    validate_geometry_id(record_id)
    validate_geometry_id(actor_id)


def _require_checksum(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise InvalidGeometryInput(f"{name} must be a lowercase SHA-256 digest")


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise InvalidGeometryInput(f"{name} must be non-empty")


def _require_text_or_empty(value: str, name: str) -> None:
    if not isinstance(value, str):
        raise InvalidGeometryInput(f"{name} must be a string")


def _validate_guideline_version(value: str) -> None:
    if value != PARSER_GT_GUIDELINE_VERSION:
        raise InvalidGeometryInput(
            f"Unsupported parser GT guideline version: {value}; "
            f"expected {PARSER_GT_GUIDELINE_VERSION}"
        )


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise InvalidGeometryInput(f"{name} must be boolean")
    return value


def _check_fields(data: Mapping[str, Any], required: set[str], context: str) -> None:
    if not isinstance(data, Mapping):
        raise InvalidGeometryInput(f"{context.capitalize()} must be a mapping")
    missing = sorted(required - set(data))
    unknown = sorted(set(data) - required)
    if missing:
        raise InvalidGeometryInput(f"{context.capitalize()} is missing fields: {missing}")
    if unknown:
        raise InvalidGeometryInput(f"{context.capitalize()} has unknown fields: {unknown}")


def _validate_record_checksum(data: Mapping[str, Any], context: str) -> None:
    payload = {key: value for key, value in data.items() if key != "checksum"}
    if data["checksum"] != stable_checksum(payload):
        raise InvalidGeometryInput(f"{context} checksum does not match payload")
