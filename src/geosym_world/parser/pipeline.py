"""Single orchestration path from public Parser input to verified geometry seed."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from geosym_world.errors import InvalidGeometryInput
from geosym_world.schema.canonical import stable_checksum
from geosym_world.verifier import ParseGraphAdmission, PixelTopologyReport, PixelTopologyVerifier
from geosym_world.verifier.parse_graph import ParseGraphAdmissionReport

from .contracts import ParserInput, validate_parser_input
from .prediction import ExternalParseGraphPrediction
from .seed_realizer import SeedRealization, SeedRealizer


PARSER_PIPELINE_RESULT_SCHEMA_VERSION = "geosym.parser.pipeline_result.v3"
PARSER_QUALITY_REPORT_SCHEMA_VERSION = "geosym.parser.quality_report.v3"
ParseGraphExtractor = Callable[[ParserInput], ExternalParseGraphPrediction]


@dataclass(frozen=True)
class CandidateLifecycle:
    candidate_id: str
    candidate_kind: str
    admission_status: str
    admission_reason: str
    realization_status: str | None = None
    realization_usage: str | None = None
    realization_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_kind": self.candidate_kind,
            "admission_status": self.admission_status,
            "admission_reason": self.admission_reason,
            "realization_status": self.realization_status,
            "realization_usage": self.realization_usage,
            "realization_reason": self.realization_reason,
        }


@dataclass(frozen=True)
class ParserQualityReport:
    """Evidence-based stage gates; a valid state is not source reconstruction."""

    payload_compiled: bool
    skeleton_built: bool
    skeleton_redraw: bool
    semantic_graph_complete: bool | None
    seed_realized: bool
    constraint_satisfied: bool
    target_bound: bool
    pixel_topology_passed: bool | None
    source_reconstruction_passed: bool | None
    reasons: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = PARSER_QUALITY_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PARSER_QUALITY_REPORT_SCHEMA_VERSION:
            raise InvalidGeometryInput(
                f"Unsupported parser quality report schema version: {self.schema_version}"
            )
        for name in (
            "payload_compiled",
            "skeleton_built",
            "skeleton_redraw",
            "seed_realized",
            "constraint_satisfied",
            "target_bound",
        ):
            if not isinstance(getattr(self, name), bool):
                raise InvalidGeometryInput(f"Parser quality field {name} must be boolean")
        for name in (
            "semantic_graph_complete",
            "pixel_topology_passed",
            "source_reconstruction_passed",
        ):
            if getattr(self, name) is not None and not isinstance(getattr(self, name), bool):
                raise InvalidGeometryInput(f"Parser quality field {name} must be boolean or null")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "payload_compiled": self.payload_compiled,
            "skeleton_built": self.skeleton_built,
            "skeleton_redraw": self.skeleton_redraw,
            "semantic_graph_complete": self.semantic_graph_complete,
            "seed_realized": self.seed_realized,
            "constraint_satisfied": self.constraint_satisfied,
            "target_bound": self.target_bound,
            "pixel_topology_passed": self.pixel_topology_passed,
            "source_reconstruction_passed": self.source_reconstruction_passed,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class ParserPipelineResult:
    parser_input: ParserInput
    prediction: ExternalParseGraphPrediction
    admission: ParseGraphAdmissionReport
    realization: SeedRealization
    pixel_topology: PixelTopologyReport | None = None
    schema_version: str = PARSER_PIPELINE_RESULT_SCHEMA_VERSION

    @property
    def candidate_lifecycle(self) -> tuple[CandidateLifecycle, ...]:
        realization = {
            item.candidate_id: item
            for item in self.realization.constraint_outcomes
        }
        return tuple(
            CandidateLifecycle(
                candidate_id=decision.candidate_id,
                candidate_kind=decision.candidate_kind,
                admission_status=decision.decision.value,
                admission_reason=decision.reason,
                realization_status=(
                    realization[decision.candidate_id].status.value
                    if decision.candidate_id in realization
                    else None
                ),
                realization_usage=(
                    realization[decision.candidate_id].usage
                    if decision.candidate_id in realization
                    else None
                ),
                realization_reason=(
                    realization[decision.candidate_id].reason
                    if decision.candidate_id in realization
                    else None
                ),
            )
            for decision in self.admission.decisions
        )

    @property
    def quality(self) -> ParserQualityReport:
        graph = self.prediction.parse_graph
        audit = graph.metadata.get("normalization_audit", {})
        if not isinstance(audit, Mapping):
            audit = {}
        dropped = audit.get("dropped", {})
        if not isinstance(dropped, Mapping):
            dropped = {}
        dropped_count = sum(
            len(value) if isinstance(value, (list, tuple, set)) else int(value or 0)
            for value in dropped.values()
        )
        unresolved_references = audit.get("unresolved_references", ())
        if not isinstance(unresolved_references, (list, tuple, set)):
            unresolved_references = ()
        unknown_objects = [item.id for item in graph.objects if item.kind.value == "unknown"]
        graph_integrity_passed = (
            bool(graph.objects)
            and dropped_count == 0
            and not audit.get("uncompiled_conflicts")
            and not unresolved_references
            and not unknown_objects
        )
        seed_realized = (
            self.realization.geometry_state is not None
            and self.realization.scene is not None
        )
        skeleton_built = bool(self.realization.geometry_state and self.realization.geometry_state.points)
        skeleton_redraw = self.realization.scene is not None
        outcomes = self.realization.constraint_outcomes
        required_ids = {item.id for item in (*graph.relations, *graph.quantities)}
        outcome_ids = {item.candidate_id for item in outcomes}
        missing_outcomes = sorted(required_ids - outcome_ids)
        constraint_satisfied = (
            seed_realized
            and bool(self.realization.verification_report.get("valid"))
            and not missing_outcomes
            and all(item.status.value == "satisfied" for item in outcomes)
            and not self.realization.unresolved_constraints
        )
        target_outcomes = self.realization.metadata.get("target_outcomes", ())
        target_bound = not graph.targets or all(
            item.get("status") == "bound" for item in target_outcomes
        )
        pixel_topology_passed = None
        if self.pixel_topology is not None and self.pixel_topology.status in {"passed", "failed"}:
            pixel_topology_passed = self.pixel_topology.status == "passed"
        reasons = ["source_reconstruction_not_assessed"]
        materialization = self.realization.metadata.get("materialization", {})
        if materialization.get("complete") is False:
            reasons.append("materialization_incomplete")
        if not graph_integrity_passed:
            reasons.append("semantic_graph_integrity_failed")
            if unresolved_references:
                reasons.append("semantic_graph_unresolved_references")
        if not seed_realized:
            reasons.append("seed_not_realized")
        if not skeleton_built:
            reasons.append("skeleton_not_built")
        if not skeleton_redraw:
            reasons.append("skeleton_redraw_unavailable")
        if not constraint_satisfied:
            reasons.append("constraints_not_satisfied")
        if not target_bound:
            reasons.append("target_not_bound")
        semantic_only = graph.metadata.get("extraction_contract") == "geometry_semantics_v1"
        if semantic_only:
            reasons.append("pixel_topology_not_applicable")
        elif self.pixel_topology is None:
            reasons.append("pixel_topology_not_run")
        elif not pixel_topology_passed:
            reasons.append(f"pixel_topology_{self.pixel_topology.status}")
        return ParserQualityReport(
            payload_compiled=True,
            skeleton_built=skeleton_built,
            skeleton_redraw=skeleton_redraw,
            semantic_graph_complete=None if graph_integrity_passed else False,
            seed_realized=seed_realized,
            constraint_satisfied=constraint_satisfied,
            target_bound=target_bound,
            pixel_topology_passed=pixel_topology_passed,
            source_reconstruction_passed=None,
            reasons=tuple(reasons),
            metrics={
                "objects": len(graph.objects),
                "relations": len(graph.relations),
                "quantities": len(graph.quantities),
                "regions": len(graph.regions),
                "targets": len(graph.targets),
                "admitted": len(self.admission.admitted_ids),
                "abstained": len(self.admission.abstained_ids),
                "rejected": len(self.admission.rejected_ids),
                "constraint_outcomes": len(outcomes),
                "satisfied_constraints": len(self.realization.satisfied_constraints),
                "unresolved_constraints": len(self.realization.unresolved_constraints),
                "pixel_checks": len(self.pixel_topology.checks) if self.pixel_topology is not None else 0,
                "unresolved_references": len(unresolved_references),
                "dropped_candidates": dropped_count,
                "unknown_object_ids": unknown_objects,
                "graph_integrity_passed": graph_integrity_passed,
                "skeleton_object_count": len(self.realization.geometry_state.object_ids()) if self.realization.geometry_state else 0,
                "skeleton_skipped_operations": len(self.realization.metadata.get("skeleton", {}).get("skipped_operations", ())),
                "required_constraints": len(required_ids),
                "missing_constraint_outcomes": missing_outcomes,
                "targets_are_assumptions": False,
                "materialization_complete": materialization.get("complete", False),
                "missing_object_ids": materialization.get("missing_object_ids", []),
                "missing_region_ids": materialization.get("missing_region_ids", []),
                "missing_target_ids": materialization.get("missing_target_ids", []),
                "bound_targets": sum(item.get("status") == "bound" for item in self.realization.metadata.get("target_outcomes", [])),
                "pixel_topology_scope": "input_proposal",
                "realization_scope": "candidate_instance",
                "extraction_contract": graph.metadata.get("extraction_contract", "parse_graph"),
                "uncertain_statements": len(graph.metadata.get("ambiguities", [])),
                "pixel_topology_applicable": not semantic_only,
            },
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.realization.status.value,
            "parser_input": self.parser_input.to_dict(),
            "prediction": self.prediction.to_dict(),
            "pixel_topology": self.pixel_topology.to_dict() if self.pixel_topology is not None else None,
            "admission": self.admission.to_dict(),
            "realization": self.realization.to_dict(),
            "quality": self.quality.to_dict(),
            "candidate_lifecycle": [item.to_dict() for item in self.candidate_lifecycle],
        }

    def checksum(self) -> str:
        return stable_checksum(self._payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "checksum": self.checksum()}


class ParserPipeline:
    """Realize a candidate interpretation while reporting admission independently."""

    def __init__(
        self,
        extractor: ParseGraphExtractor,
        *,
        admission: ParseGraphAdmission | None = None,
        realizer: SeedRealizer | None = None,
        pixel_topology_verifier: PixelTopologyVerifier | None = None,
    ) -> None:
        if not callable(extractor):
            raise TypeError("ParserPipeline extractor must be callable")
        self.extractor = extractor
        self.admission = admission or ParseGraphAdmission()
        self.realizer = realizer or SeedRealizer()
        self.pixel_topology_verifier = pixel_topology_verifier

    def run(self, parser_input: ParserInput | dict[str, Any]) -> ParserPipelineResult:
        public_input = validate_parser_input(parser_input)
        prediction = self.extractor(public_input)
        if not isinstance(prediction, ExternalParseGraphPrediction):
            raise TypeError("ParserPipeline extractor must return ExternalParseGraphPrediction")
        expected_checksum = stable_checksum(public_input.to_dict())
        if prediction.id != public_input.id or prediction.parser_input_checksum != expected_checksum:
            raise InvalidGeometryInput("External prediction does not match the ParserInput identity and checksum")

        pixel_topology = None
        semantic_only = prediction.parse_graph.metadata.get("extraction_contract") == "geometry_semantics_v1"
        if self.pixel_topology_verifier is not None and not semantic_only:
            pixel_topology = self.pixel_topology_verifier.verify(
                prediction.parse_graph,
                public_input.image,
            )
        admission = self.admission.admit(
            prediction.parse_graph,
            pixel_topology=pixel_topology,
        )
        realization = self.realizer.realize_proposal(prediction.parse_graph)
        return ParserPipelineResult(
            parser_input=public_input,
            prediction=prediction,
            pixel_topology=pixel_topology,
            admission=admission,
            realization=realization,
        )
