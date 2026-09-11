"""Restricted reconstruction from admitted ParseGraphs into formal states."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations, product
from typing import Any

import sympy as sp

from geosym_world.construction import ConstructionBuilder, ConstructionOp
from geosym_world.errors import InvalidGeometryInput
from geosym_world.schema import (
    FactScope,
    FactStore,
    FormalGeometryState,
    MathExpression,
    ParseGraph,
    ParseObject,
    ParseObjectKind,
    ParseQuantity,
    ParseReference,
    ParseRegion,
    ParseRelation,
    ParseTarget,
    Provenance,
)
from geosym_world.schema.canonical import stable_checksum
from geosym_world.schema.provenance import EvidenceKind
from geosym_world.schema.relation_normalization import RELATION_TYPES, TARGET_TYPES, normalize_target_type
from geosym_world.solver import FactClosureSolver
from geosym_world.verifier.parse_graph import ParseGraphAdmissionReport
from geosym_world.verifier.trace import TraceVerifier


RECONSTRUCTION_SCHEMA_VERSION = "1.1"
RECONSTRUCTION_BRANCH_SET_SCHEMA_VERSION = "1.1"
_BRANCH_SELECTIONS = frozenset({"all", "first", "second", "upper", "lower", "left", "right"})
_RUNTIME_BRANCH_SELECTIONS = frozenset({"first", "second"})


class ReconstructionMode(str, Enum):
    EXACT = "exact"
    PIXEL_WITNESS = "pixel_witness"


class ReconstructionStatus(str, Enum):
    EXACT = "exact"
    NUMERIC_WITNESS = "numeric_witness"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class ReconstructionIssue:
    code: str
    message: str
    candidate_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "candidate_id": self.candidate_id}


@dataclass(frozen=True)
class ReconstructionAnalysis:
    coordinate_basis: str
    free_parameter_status: str
    free_parameters: tuple[str, ...]
    unplaced_points: tuple[str, ...]
    branch_status: str
    ambiguous_candidates: tuple[str, ...]
    unsupported_candidates: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinate_basis": self.coordinate_basis,
            "free_parameter_status": self.free_parameter_status,
            "free_parameters": list(self.free_parameters),
            "unplaced_points": list(self.unplaced_points),
            "branch_status": self.branch_status,
            "ambiguous_candidates": list(self.ambiguous_candidates),
            "unsupported_candidates": list(self.unsupported_candidates),
        }


@dataclass(frozen=True)
class ReconstructionReport:
    source_graph_checksum: str
    admitted_graph_checksum: str
    mode: ReconstructionMode
    status: ReconstructionStatus
    trace: tuple[ConstructionOp, ...]
    state: FormalGeometryState | None
    source_facts: FactStore
    realization_facts: FactStore
    verification: Mapping[str, Any]
    analysis: ReconstructionAnalysis
    issues: tuple[ReconstructionIssue, ...] = ()
    candidate_outcomes: Mapping[str, str] = field(default_factory=dict)
    branch_choices: Mapping[str, str] = field(default_factory=dict)
    schema_version: str = RECONSTRUCTION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "source_graph_checksum": self.source_graph_checksum,
            "admitted_graph_checksum": self.admitted_graph_checksum,
            "mode": self.mode.value,
            "status": self.status.value,
            "trace": [_json_value(operation.to_dict()) for operation in self.trace],
            "state": _json_value(self.state.to_dict()) if self.state is not None else None,
            "source_facts": self.source_facts.to_list(),
            "realization_facts": self.realization_facts.to_list(),
            "verification": dict(self.verification),
            "analysis": self.analysis.to_dict(),
            "issues": [issue.to_dict() for issue in self.issues],
            "candidate_outcomes": dict(sorted(self.candidate_outcomes.items())),
            "branch_choices": dict(sorted(self.branch_choices.items())),
        }
        payload["checksum"] = stable_checksum(payload)
        return payload


@dataclass(frozen=True)
class _IntersectionPointPlan:
    relation: ParseRelation
    support_ids: tuple[str, str]


@dataclass(frozen=True)
class _DerivedPointPlan:
    relation: ParseRelation
    operation: str
    dependency_ids: tuple[str, ...]
    params: Mapping[str, Any]


@dataclass(frozen=True)
class _DerivedLinePlan:
    relation: ParseRelation
    operation: str
    dependency_ids: tuple[str, ...]
    params: Mapping[str, Any]


@dataclass(frozen=True)
class _DistanceEdge:
    segment_id: str
    start_id: str
    end_id: str
    length: sp.Expr
    quantity_ids: tuple[str, ...]


@dataclass(frozen=True)
class _DistancePlacementPlan:
    point_id: str
    support_start_id: str
    support_end_id: str
    support_length: sp.Expr
    start_length: sp.Expr
    end_length: sp.Expr
    segment_ids: tuple[str, str, str]
    quantity_ids: tuple[str, ...]
    status: str


@dataclass(frozen=True)
class _DistanceGaugePlan:
    base_start_id: str
    base_end_id: str
    base_length: sp.Expr
    placements: tuple[_DistancePlacementPlan, ...]
    edges: tuple[_DistanceEdge, ...]
    segment_ids: tuple[str, ...]
    quantity_ids: tuple[str, ...]
    x_offset: sp.Expr
    status: str
    gauge_kind: str


@dataclass(frozen=True)
class ReconstructionBranchSet:
    source_graph_checksum: str
    ambiguous_candidates: tuple[str, ...]
    branches: tuple[ReconstructionReport, ...]
    rejected_branches: tuple[ReconstructionReport, ...]
    max_branches: int
    schema_version: str = RECONSTRUCTION_BRANCH_SET_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        status_counts = Counter(branch.status.value for branch in self.branches)
        candidate_status_counts = Counter(
            branch.status.value for branch in (*self.branches, *self.rejected_branches)
        )
        payload = {
            "schema_version": self.schema_version,
            "source_graph_checksum": self.source_graph_checksum,
            "ambiguous_candidates": list(self.ambiguous_candidates),
            "max_branches": self.max_branches,
            "num_branches": len(self.branches),
            "num_candidate_branches": len(self.branches) + len(self.rejected_branches),
            "num_rejected_branches": len(self.rejected_branches),
            "status_counts": dict(sorted(status_counts.items())),
            "candidate_status_counts": dict(sorted(candidate_status_counts.items())),
            "branches": [branch.to_dict() for branch in self.branches],
            "rejected_branches": [branch.to_dict() for branch in self.rejected_branches],
        }
        payload["checksum"] = stable_checksum(payload)
        return payload


class ParseGraphReconstructor:
    """Compile only admitted candidates into an auditable construction trace."""

    def reconstruct(
        self,
        admission: ParseGraphAdmissionReport,
        *,
        mode: ReconstructionMode = ReconstructionMode.EXACT,
        branch_choices: Mapping[str, str] | None = None,
    ) -> ReconstructionReport:
        if not isinstance(admission, ParseGraphAdmissionReport):
            raise TypeError("ParseGraphReconstructor requires a ParseGraphAdmissionReport")
        if not isinstance(mode, ReconstructionMode):
            raise InvalidGeometryInput(f"Unknown reconstruction mode: {mode}")
        graph = admission.admitted_graph
        _validate_admission_binding(admission, graph)
        choices = _validate_branch_choices(graph, branch_choices or {})

        compiler = _TraceCompiler(graph, mode, choices)
        trace, source_facts, issues, outcomes = compiler.compile()
        if not trace:
            return _report(
                admission,
                mode,
                ReconstructionStatus.UNSUPPORTED,
                trace,
                None,
                source_facts,
                FactStore(),
                {"valid": False, "reason": "no_executable_geometry"},
                issues,
                outcomes,
                choices,
            )

        verification = TraceVerifier().verify_trace(trace)
        if not verification.valid:
            issues.append(ReconstructionIssue("reconstruction.trace_invalid", "Compiled trace failed verification"))
            return _report(
                admission,
                mode,
                ReconstructionStatus.PARTIAL,
                trace,
                None,
                source_facts,
                FactStore(),
                verification.to_dict(),
                issues,
                outcomes,
                choices,
            )

        state = ConstructionBuilder().execute(trace)
        realization_facts = FactClosureSolver(profile="auto").solve(state)
        status = _completion_status(mode, issues)
        return _report(
            admission,
            mode,
            status,
            trace,
            state,
            source_facts,
            realization_facts,
            verification.to_dict(),
            issues,
            outcomes,
            choices,
        )

    def enumerate_branches(
        self,
        admission: ParseGraphAdmissionReport,
        *,
        max_branches: int = 16,
    ) -> ReconstructionBranchSet:
        if not isinstance(admission, ParseGraphAdmissionReport):
            raise TypeError("ParseGraphReconstructor requires a ParseGraphAdmissionReport")
        if isinstance(max_branches, bool) or not isinstance(max_branches, int) or max_branches <= 0:
            raise InvalidGeometryInput("max_branches must be a positive integer")
        graph = admission.admitted_graph
        _validate_admission_binding(admission, graph)
        choice_groups = _branch_choice_groups(graph)
        branch_count = math.prod(len(group) for group in choice_groups) if choice_groups else 1
        if branch_count > max_branches:
            raise InvalidGeometryInput(
                f"Exact reconstruction requires {branch_count} branches, exceeding max_branches={max_branches}"
            )
        assignments = product(*choice_groups) if choice_groups else ((),)
        choices = tuple(
            {
                point_id: selection
                for assignment in grouped_assignments
                for point_id, selection in assignment.items()
            }
            for grouped_assignments in assignments
        )
        candidate_branches = tuple(
            self.reconstruct(admission, mode=ReconstructionMode.EXACT, branch_choices=assignment)
            for assignment in choices
        )
        branches = tuple(
            branch for branch in candidate_branches if branch.status is ReconstructionStatus.EXACT
        )
        rejected_branches = tuple(
            branch for branch in candidate_branches if branch.status is not ReconstructionStatus.EXACT
        )
        ambiguous_candidates = tuple(sorted({point_id for assignment in choices for point_id in assignment}))
        return ReconstructionBranchSet(
            source_graph_checksum=admission.source_graph_checksum,
            ambiguous_candidates=ambiguous_candidates,
            branches=branches,
            rejected_branches=rejected_branches,
            max_branches=max_branches,
        )


class _TraceCompiler:
    def __init__(
        self,
        graph: ParseGraph,
        mode: ReconstructionMode,
        branch_choices: Mapping[str, str],
    ):
        self.graph = graph
        self.mode = mode
        self.branch_choices = dict(branch_choices)
        self.trace: list[ConstructionOp] = []
        self.issues: list[ReconstructionIssue] = []
        self.outcomes: dict[str, str] = {}
        self.source_facts = FactStore()
        self.point_names = graph.point_names()
        self.objects_by_id = {candidate.id: candidate for candidate in graph.objects}
        self.intersection_plans = _intersection_point_plans(graph)
        self.derived_point_plans = _derived_point_plans(graph)
        self.derived_line_plans = _derived_line_plans(graph)
        self.distance_gauge_plans = _distance_gauge_plans(graph)
        self.constructed_relation_ids: set[str] = set()
        self.quantities_by_object: dict[str, list[ParseQuantity]] = {}
        for quantity in graph.quantities:
            if quantity.arguments:
                self.quantities_by_object.setdefault(quantity.arguments[0].ref, []).append(quantity)

    def compile(self) -> tuple[list[ConstructionOp], FactStore, list[ReconstructionIssue], dict[str, str]]:
        self._compile_exact_distance_gauges()
        self._compile_objects()
        self._compile_regions()
        for relation in self.graph.relations:
            self._compile_relation(relation)
        for quantity in self.graph.quantities:
            self._compile_quantity(quantity)
        for target in self.graph.targets:
            self._compile_target(target)
        return self.trace, self.source_facts, self.issues, self.outcomes

    def _compile_objects(self) -> None:
        pending = list(self.graph.objects)
        while pending:
            progressed = False
            deferred = []
            for candidate in pending:
                if candidate.id in self.outcomes:
                    progressed = True
                    continue
                dependency_ids = {reference.ref for reference in candidate.references}
                dependency_ids.update(self._planned_object_dependencies(candidate))
                dependency_outcomes = {
                    dependency_id: self.outcomes.get(dependency_id)
                    for dependency_id in dependency_ids
                }
                unavailable = sorted(
                    reference
                    for reference, outcome in dependency_outcomes.items()
                    if outcome is not None and outcome != "materialized"
                )
                if unavailable:
                    self._unsupported(
                        candidate.id,
                        "object",
                        f"Object dependencies were not materialized: {', '.join(unavailable)}",
                    )
                    progressed = True
                    continue
                if any(outcome is None for outcome in dependency_outcomes.values()):
                    deferred.append(candidate)
                    continue
                self._compile_object(candidate)
                progressed = True
            if not progressed:
                for candidate in deferred:
                    dependencies = sorted(
                        {
                            *(reference.ref for reference in candidate.references),
                            *self._planned_object_dependencies(candidate),
                        }
                    )
                    self._issue(
                        candidate.id,
                        "reconstruction.reference_unresolved",
                        f"Object dependency cycle or unresolved references: {', '.join(dependencies)}",
                    )
                break
            pending = deferred

    def _compile_exact_distance_gauges(self) -> None:
        if self.mode is not ReconstructionMode.EXACT:
            return
        for plan in self.distance_gauge_plans:
            first_point_id = plan.placements[0].point_id
            if plan.status != "valid":
                code = (
                    "reconstruction.condition_conflict"
                    if plan.status == "condition_conflict"
                    else "reconstruction.numeric_unstable"
                )
                self._issue(
                    first_point_id,
                    code,
                    f"Distance gauge containing {first_point_id} is {plan.status.replace('_', ' ')}",
                )
                continue
            start = self.objects_by_id[plan.base_start_id]
            end = self.objects_by_id[plan.base_end_id]
            common_metadata = {
                "reconstruction_gauge": plan.gauge_kind,
                "source_segment_ids": list(plan.segment_ids),
                "parse_quantity_ids": list(plan.quantity_ids),
            }
            self.trace.extend(
                (
                    ConstructionOp(
                        "add_point",
                        start.id,
                        {
                            "coords": [plan.x_offset, sp.Integer(0)],
                            "metadata": {
                                **self._object_metadata(start),
                                **common_metadata,
                                "gauge_role": "base_start",
                            },
                        },
                    ),
                    ConstructionOp(
                        "add_point",
                        end.id,
                        {
                            "coords": [plan.x_offset + plan.base_length, sp.Integer(0)],
                            "metadata": {
                                **self._object_metadata(end),
                                **common_metadata,
                                "gauge_role": "base_end",
                            },
                        },
                    ),
                )
            )
            self.outcomes[start.id] = "materialized"
            self.outcomes[end.id] = "materialized"
            missing_choices = tuple(
                placement.point_id
                for placement in plan.placements
                if placement.point_id not in self.branch_choices
            )
            if missing_choices:
                for point_id in missing_choices:
                    self._ambiguous(
                        point_id,
                        f"Distance gauge point {point_id} requires an explicit first/second mirror branch",
                    )
                continue

            coordinates = {
                start.id: (plan.x_offset, sp.Integer(0)),
                end.id: (plan.x_offset + plan.base_length, sp.Integer(0)),
            }
            for placement_index, placement in enumerate(plan.placements):
                point = self.objects_by_id[placement.point_id]
                selection = self.branch_choices[point.id]
                try:
                    point_coordinates = _trilateration_coordinates(
                        coordinates[placement.support_start_id],
                        coordinates[placement.support_end_id],
                        placement.support_length,
                        placement.start_length,
                        placement.end_length,
                        selection,
                    )
                except _ConditionConflictError as exc:
                    self._issue(point.id, "reconstruction.condition_conflict", str(exc))
                    break
                coordinates[point.id] = point_coordinates
                self.trace.append(
                    ConstructionOp(
                        "add_point",
                        point.id,
                        {
                            "coords": list(point_coordinates),
                            "metadata": {
                                **self._object_metadata(point),
                                **common_metadata,
                                "gauge_role": (
                                    "apex" if placement_index == 0 else "trilaterated_point"
                                ),
                                "support_points": [
                                    placement.support_start_id,
                                    placement.support_end_id,
                                ],
                                "support_segment_ids": list(placement.segment_ids),
                                "support_quantity_ids": list(placement.quantity_ids),
                                "branch_selection": selection,
                                "branch_selection_source": "runtime_choice",
                            },
                        },
                    )
                )
                self.outcomes[point.id] = "materialized"
            else:
                self._validate_distance_edges(plan, coordinates)

    def _validate_distance_edges(
        self,
        plan: _DistanceGaugePlan,
        coordinates: Mapping[str, tuple[sp.Expr, sp.Expr]],
    ) -> None:
        for edge in plan.edges:
            start = coordinates[edge.start_id]
            end = coordinates[edge.end_id]
            residual = sp.simplify(
                (end[0] - start[0]) ** 2
                + (end[1] - start[1]) ** 2
                - edge.length**2
            )
            if residual.is_zero is True:
                continue
            if residual.is_zero is False:
                self._issue(
                    edge.segment_id,
                    "reconstruction.condition_conflict",
                    f"Length constraint for {edge.segment_id} conflicts with the selected distance branch",
                )
            else:
                self._issue(
                    edge.segment_id,
                    "reconstruction.numeric_unstable",
                    f"Length residual for {edge.segment_id} could not be decided exactly",
                )

    def _compile_object(self, candidate: ParseObject) -> None:
        if candidate.kind in {ParseObjectKind.TEXT, ParseObjectKind.MARKER}:
            self.outcomes[candidate.id] = "evidence_only"
            return
        try:
            operation = self._object_operation(candidate)
        except _BranchAmbiguousError as exc:
            self._ambiguous(candidate.id, str(exc))
            return
        except _ConstructionAmbiguousError as exc:
            self.outcomes[candidate.id] = "ambiguous"
            self.issues.append(ReconstructionIssue("reconstruction.construction_ambiguous", str(exc), candidate.id))
            return
        except _ConditionConflictError as exc:
            self._issue(candidate.id, "reconstruction.condition_conflict", str(exc))
            return
        except _UnderconstrainedError as exc:
            self._issue(candidate.id, "reconstruction.underconstrained", str(exc))
            return
        except (InvalidGeometryInput, KeyError, TypeError, ValueError) as exc:
            self._unsupported(candidate.id, "object", str(exc))
            return
        if operation is None:
            self._unsupported(candidate.id, "object", f"Unsupported object kind: {candidate.kind.value}")
            return
        self.trace.append(operation)
        self.outcomes[candidate.id] = "materialized"

    def _planned_object_dependencies(self, candidate: ParseObject) -> tuple[str, ...]:
        if self.mode is not ReconstructionMode.EXACT:
            return ()
        if candidate.kind is ParseObjectKind.POINT and candidate.geometry.get("exact_coords") is None:
            return tuple(dict.fromkeys((
                *(
                    support_id
                    for plan in self.intersection_plans.get(candidate.id, ())
                    for support_id in plan.support_ids
                ),
                *(
                    dependency_id
                    for plan in self.derived_point_plans.get(candidate.id, ())
                    for dependency_id in plan.dependency_ids
                ),
            )))
        if candidate.kind is ParseObjectKind.LINE:
            return tuple(dict.fromkeys(
                dependency_id
                for plan in self.derived_line_plans.get(candidate.id, ())
                for dependency_id in plan.dependency_ids
            ))
        return ()

    def _object_operation(self, candidate: ParseObject) -> ConstructionOp | None:
        metadata = self._object_metadata(candidate)
        refs = _references_by_role(candidate.references)
        if candidate.kind is ParseObjectKind.POINT:
            if self.mode is ReconstructionMode.EXACT and candidate.geometry.get("exact_coords") is None:
                return self._exact_point_operation(candidate, metadata)
            return ConstructionOp(
                "add_point",
                candidate.id,
                {"coords": self._point_coords(candidate), "metadata": metadata},
            )
        if candidate.kind is ParseObjectKind.SEGMENT:
            return ConstructionOp(
                "add_segment",
                candidate.id,
                {"start": _one_ref(refs, "start"), "end": _one_ref(refs, "end"), "metadata": metadata},
            )
        if candidate.kind is ParseObjectKind.LINE:
            points = _ordered_refs(refs, "point", minimum=1)
            if len(points) >= 2:
                return ConstructionOp("add_line", candidate.id, {"points": points[:2], "metadata": metadata})
            if self.mode is ReconstructionMode.EXACT:
                return self._exact_line_operation(candidate, points[0], metadata)
            raise _UnderconstrainedError(f"Line {candidate.id} requires two points in pixel witness mode")
        if candidate.kind is ParseObjectKind.RAY:
            return ConstructionOp(
                "add_ray",
                candidate.id,
                {"start": _one_ref(refs, "start"), "through": _one_ref(refs, "through"), "metadata": metadata},
            )
        if candidate.kind is ParseObjectKind.CIRCLE:
            radius = self._geometry_expression(candidate, "exact_radius")
            radius_quantity_ids: tuple[str, ...] = ()
            if self.mode is ReconstructionMode.EXACT:
                radius, radius_quantity_ids = self._exact_circle_radius(candidate, radius)
            if self.mode is ReconstructionMode.PIXEL_WITNESS and radius is None:
                radius = _finite_number(candidate.geometry.get("radius_px"), f"circle {candidate.id} radius_px")
            if radius is None:
                raise _UnderconstrainedError(f"Circle {candidate.id} is missing a radius for {self.mode.value}")
            if radius_quantity_ids:
                metadata["parse_quantity_ids"] = list(radius_quantity_ids)
            return ConstructionOp(
                "add_circle",
                candidate.id,
                {"center": _one_ref(refs, "center"), "radius": radius, "metadata": metadata},
            )
        if candidate.kind is ParseObjectKind.ARC:
            params: dict[str, Any] = {
                "circle": _one_ref(refs, "circle"),
                "start": _one_ref(refs, "start"),
                "end": _one_ref(refs, "end"),
                "orientation": str(candidate.geometry.get("orientation", "ccw")),
                "metadata": metadata,
            }
            angle = self._geometry_expression(candidate, "exact_central_angle")
            if angle is not None:
                params["central_angle"] = angle
            return ConstructionOp("add_arc", candidate.id, params)
        if candidate.kind is ParseObjectKind.POLYGON:
            vertices = _ordered_refs(refs, "vertex", minimum=3)
            return ConstructionOp("add_polygon", candidate.id, {"vertices": vertices, "metadata": metadata})
        return None

    def _exact_point_operation(
        self,
        candidate: ParseObject,
        metadata: Mapping[str, Any],
    ) -> ConstructionOp:
        intersection_plans = self.intersection_plans.get(candidate.id, ())
        derived_plans = self.derived_point_plans.get(candidate.id, ())
        plan_count = len(intersection_plans) + len(derived_plans)
        if plan_count == 0:
            raise _UnderconstrainedError(f"Point {candidate.id} has no exact_coords or supported construction")
        if plan_count > 1:
            relation_ids = sorted(
                [plan.relation.id for plan in intersection_plans]
                + [plan.relation.id for plan in derived_plans]
            )
            raise _ConstructionAmbiguousError(
                f"Point {candidate.id} has multiple exact constructions: {', '.join(relation_ids)}"
            )
        if intersection_plans:
            return self._intersection_point_operation(candidate, metadata)
        return self._derived_point_operation(candidate, derived_plans[0], metadata)

    def _exact_line_operation(
        self,
        candidate: ParseObject,
        through_point_id: str,
        metadata: Mapping[str, Any],
    ) -> ConstructionOp:
        plans = self.derived_line_plans.get(candidate.id, ())
        if not plans:
            raise _UnderconstrainedError(
                f"Line {candidate.id} has one point but no supported exact direction construction"
            )
        if len(plans) > 1:
            relation_ids = ", ".join(sorted(plan.relation.id for plan in plans))
            raise _ConstructionAmbiguousError(
                f"Line {candidate.id} has multiple exact direction constructions: {relation_ids}"
            )
        plan = plans[0]
        operation_metadata = {
            **dict(metadata),
            "parse_relation_id": plan.relation.id,
        }
        if plan.operation in {"parallel_through", "perpendicular_through"}:
            planned_point_id, base_line_id = plan.dependency_ids
            if planned_point_id != through_point_id:
                raise InvalidGeometryInput(
                    f"Line construction {plan.relation.id} does not use its declared through point"
                )
            base = self.objects_by_id[base_line_id]
            if base.kind not in {
                ParseObjectKind.LINE,
                ParseObjectKind.RAY,
                ParseObjectKind.SEGMENT,
            }:
                raise InvalidGeometryInput(
                    f"Line construction {plan.relation.id} requires a linear base object"
                )
            params = {
                "point": through_point_id,
                "line": base_line_id,
                "metadata": operation_metadata,
            }
        elif plan.operation == "angle_bisector":
            vertex_id, first_arm_id, second_arm_id = plan.dependency_ids
            if vertex_id != through_point_id:
                raise InvalidGeometryInput(
                    f"Angle-bisector relation {plan.relation.id} must construct a line through its vertex"
                )
            if any(
                self.objects_by_id[point_id].kind is not ParseObjectKind.POINT
                for point_id in plan.dependency_ids
            ):
                raise InvalidGeometryInput(
                    f"Angle-bisector relation {plan.relation.id} requires three point arguments"
                )
            params = {
                "vertex": vertex_id,
                "ray1": [vertex_id, first_arm_id],
                "ray2": [vertex_id, second_arm_id],
                "mode": plan.params["mode"],
                "metadata": operation_metadata,
            }
        elif plan.operation == "line_through_angle":
            planned_point_id, base_line_id = plan.dependency_ids
            if planned_point_id != through_point_id:
                raise InvalidGeometryInput(
                    f"Angle-from-line relation {plan.relation.id} does not use its declared through point"
                )
            if self.objects_by_id[base_line_id].kind not in {
                ParseObjectKind.LINE,
                ParseObjectKind.RAY,
                ParseObjectKind.SEGMENT,
            }:
                raise InvalidGeometryInput(
                    f"Angle-from-line relation {plan.relation.id} requires a linear base object"
                )
            angle = _expression(
                plan.params.get("angle"),
                f"relation {plan.relation.id} angle",
            )
            if angle.is_number is not True or angle.is_real is not True:
                raise InvalidGeometryInput(
                    f"Angle-from-line relation {plan.relation.id} requires a real numeric angle"
                )
            params = {
                "point": through_point_id,
                "base_line": base_line_id,
                "angle": angle,
                "metadata": operation_metadata,
            }
        else:
            raise AssertionError(f"Unhandled exact line operation: {plan.operation}")
        params.update(_constructed_relation_params(plan.relation))
        self.constructed_relation_ids.add(plan.relation.id)
        return ConstructionOp(plan.operation, candidate.id, params)

    def _derived_point_operation(
        self,
        candidate: ParseObject,
        plan: _DerivedPointPlan,
        metadata: Mapping[str, Any],
    ) -> ConstructionOp:
        operation_metadata = {
            **dict(metadata),
            "parse_relation_id": plan.relation.id,
        }
        if plan.operation == "midpoint":
            endpoints = plan.dependency_ids
            if len(endpoints) != 2 or any(
                self.objects_by_id[point_id].kind is not ParseObjectKind.POINT
                for point_id in endpoints
            ):
                raise InvalidGeometryInput(f"Midpoint relation {plan.relation.id} requires two points")
            params = {"of": list(endpoints), "metadata": operation_metadata}
        elif plan.operation == "foot_of_perpendicular":
            source_id, support_id = plan.dependency_ids
            if self.objects_by_id[source_id].kind is not ParseObjectKind.POINT:
                raise InvalidGeometryInput(f"Perpendicular foot relation {plan.relation.id} requires a source point")
            if self.objects_by_id[support_id].kind not in {
                ParseObjectKind.LINE,
                ParseObjectKind.RAY,
                ParseObjectKind.SEGMENT,
            }:
                raise InvalidGeometryInput(f"Perpendicular foot relation {plan.relation.id} requires a linear support")
            params = {"point": source_id, "line": support_id, "metadata": operation_metadata}
        elif plan.operation == "point_on_segment_ratio":
            (segment_id,) = plan.dependency_ids
            if self.objects_by_id[segment_id].kind is not ParseObjectKind.SEGMENT:
                raise InvalidGeometryInput(f"Segment-ratio relation {plan.relation.id} requires a segment")
            ratio = _expression(plan.params.get("ratio"), f"relation {plan.relation.id} ratio")
            if ratio.is_number is not True or ratio.is_real is not True:
                raise InvalidGeometryInput(f"Segment-ratio relation {plan.relation.id} requires a real numeric ratio")
            if ratio.is_positive is not True or (sp.Integer(1) - ratio).is_positive is not True:
                raise InvalidGeometryInput(f"Segment-ratio relation {plan.relation.id} ratio must be strictly between 0 and 1")
            params = {"segment": segment_id, "ratio": ratio, "metadata": operation_metadata}
        elif plan.operation == "point_on_ray_distance":
            (ray_id,) = plan.dependency_ids
            if self.objects_by_id[ray_id].kind is not ParseObjectKind.RAY:
                raise InvalidGeometryInput(f"Ray-distance relation {plan.relation.id} requires a ray")
            distance = _expression(plan.params.get("distance"), f"relation {plan.relation.id} distance")
            if distance.is_number is not True or distance.is_real is not True:
                raise InvalidGeometryInput(f"Ray-distance relation {plan.relation.id} requires a real numeric distance")
            if distance.is_positive is not True:
                raise InvalidGeometryInput(f"Ray-distance relation {plan.relation.id} distance must be positive")
            params = {"ray": ray_id, "distance": distance, "metadata": operation_metadata}
        else:
            raise AssertionError(f"Unhandled exact point operation: {plan.operation}")
        params.update(_constructed_relation_params(plan.relation))
        self.constructed_relation_ids.add(plan.relation.id)
        return ConstructionOp(plan.operation, candidate.id, params)

    def _intersection_point_operation(
        self,
        candidate: ParseObject,
        metadata: Mapping[str, Any],
    ) -> ConstructionOp:
        plans = self.intersection_plans.get(candidate.id, ())
        if not plans:
            raise _UnderconstrainedError(f"Point {candidate.id} has no exact_coords or supported construction")
        if len(plans) != 1:
            relation_ids = ", ".join(sorted(plan.relation.id for plan in plans))
            raise _BranchAmbiguousError(
                f"Point {candidate.id} has multiple intersection constructions: {relation_ids}"
            )
        plan = plans[0]
        supports = [self.objects_by_id[support_id] for support_id in plan.support_ids]
        linear = {ParseObjectKind.LINE, ParseObjectKind.RAY, ParseObjectKind.SEGMENT}
        kinds = {support.kind for support in supports}
        params = {
            "metadata": {
                **dict(metadata),
                "parse_relation_id": plan.relation.id,
            },
            **_constructed_relation_params(plan.relation),
        }
        if kinds.issubset(linear):
            params.update({"line1": supports[0].id, "line2": supports[1].id})
            operation = ConstructionOp("intersection_line_line", candidate.id, params)
        elif len(kinds & linear) == 1 and len(kinds & {ParseObjectKind.CIRCLE}) == 1:
            line = next(support for support in supports if support.kind in linear)
            circle = next(support for support in supports if support.kind is ParseObjectKind.CIRCLE)
            selection, selection_source = _intersection_selection(
                plan.relation,
                candidate.id,
                self.branch_choices,
            )
            params["metadata"]["branch_selection_source"] = selection_source
            params.update(
                {
                    "line": line.id,
                    "circle": circle.id,
                    "select": selection,
                }
            )
            operation = ConstructionOp("intersection_line_circle", candidate.id, params)
        elif kinds == {ParseObjectKind.CIRCLE}:
            selection, selection_source = _intersection_selection(
                plan.relation,
                candidate.id,
                self.branch_choices,
            )
            params["metadata"]["branch_selection_source"] = selection_source
            params.update(
                {
                    "circle1": supports[0].id,
                    "circle2": supports[1].id,
                    "select": selection,
                }
            )
            operation = ConstructionOp("intersection_circle_circle", candidate.id, params)
        elif len(kinds & linear) == 1 and len(kinds & {ParseObjectKind.ARC}) == 1:
            line = next(support for support in supports if support.kind in linear)
            arc = next(support for support in supports if support.kind is ParseObjectKind.ARC)
            selection, selection_source = _intersection_selection(
                plan.relation,
                candidate.id,
                self.branch_choices,
            )
            params["metadata"]["branch_selection_source"] = selection_source
            params.update({"line": line.id, "arc": arc.id, "select": selection})
            operation = ConstructionOp("intersection_line_arc", candidate.id, params)
        elif kinds == {ParseObjectKind.CIRCLE, ParseObjectKind.ARC}:
            circle = next(support for support in supports if support.kind is ParseObjectKind.CIRCLE)
            arc = next(support for support in supports if support.kind is ParseObjectKind.ARC)
            selection, selection_source = _intersection_selection(
                plan.relation,
                candidate.id,
                self.branch_choices,
            )
            params["metadata"]["branch_selection_source"] = selection_source
            params.update({"circle": circle.id, "arc": arc.id, "select": selection})
            operation = ConstructionOp("intersection_circle_arc", candidate.id, params)
        elif kinds == {ParseObjectKind.ARC}:
            selection, selection_source = _intersection_selection(
                plan.relation,
                candidate.id,
                self.branch_choices,
            )
            params["metadata"]["branch_selection_source"] = selection_source
            params.update({"arc1": supports[0].id, "arc2": supports[1].id, "select": selection})
            operation = ConstructionOp("intersection_arc_arc", candidate.id, params)
        else:
            support_types = ", ".join(support.kind.value for support in supports)
            raise _UnderconstrainedError(
                f"Point {candidate.id} uses unsupported intersection supports: {support_types}"
            )
        self.constructed_relation_ids.add(plan.relation.id)
        return operation

    def _object_metadata(self, candidate: ParseObject) -> dict[str, Any]:
        metadata = _reconstruction_metadata(candidate, self.mode)
        display_name = candidate.visible_name or self.point_names.get(candidate.id)
        metadata.update(
            {
                "display_name": display_name,
                "show_label": display_name is not None,
                "show_marker": candidate.visible is not False,
            }
        )
        length_quantities = [
            quantity
            for quantity in self.quantities_by_object.get(candidate.id, ())
            if quantity.kind == "length"
        ]
        if candidate.kind is ParseObjectKind.SEGMENT and len(length_quantities) == 1:
            metadata.update(
                {
                    "show_length": True,
                    "display_value": str(length_quantities[0].expression.to_sympy()),
                    "annotation_scope": FactScope.SOURCE_ASSERTION.value,
                }
            )
        return metadata

    def _point_coords(self, candidate: ParseObject) -> list[sp.Expr | float]:
        exact = candidate.geometry.get("exact_coords")
        if exact is not None:
            if not isinstance(exact, Sequence) or isinstance(exact, (str, bytes)) or len(exact) not in {2, 3}:
                raise InvalidGeometryInput(f"Point {candidate.id} exact_coords must contain two or three expressions")
            return [_expression(item, f"point {candidate.id} exact_coords") for item in exact]
        if self.mode is ReconstructionMode.EXACT:
            raise _UnderconstrainedError(f"Point {candidate.id} has no exact_coords")
        pixel = candidate.geometry.get("pixel")
        if not isinstance(pixel, Sequence) or isinstance(pixel, (str, bytes)) or len(pixel) != 2:
            raise InvalidGeometryInput(f"Point {candidate.id} has no two-dimensional pixel witness")
        x = _finite_number(pixel[0], f"point {candidate.id} pixel x")
        y = _finite_number(pixel[1], f"point {candidate.id} pixel y")
        image_size = self.graph.metadata.get("image_size")
        if not isinstance(image_size, Sequence) or isinstance(image_size, (str, bytes)) or len(image_size) != 2:
            raise InvalidGeometryInput("Pixel witness reconstruction requires graph metadata.image_size")
        height = _finite_number(image_size[1], "image height")
        if height <= 0:
            raise InvalidGeometryInput("Pixel witness reconstruction requires a positive image height")
        return [x, height - y]

    def _geometry_expression(self, candidate: ParseObject, key: str) -> sp.Expr | None:
        value = candidate.geometry.get(key)
        return None if value is None else _expression(value, f"object {candidate.id} {key}")

    def _exact_circle_radius(
        self,
        candidate: ParseObject,
        geometry_radius: sp.Expr | None,
    ) -> tuple[sp.Expr | None, tuple[str, ...]]:
        constraints: list[tuple[str, sp.Expr]] = []
        if geometry_radius is not None:
            constraints.append((f"object:{candidate.id}", geometry_radius))
        for quantity in self.quantities_by_object.get(candidate.id, ()):
            if quantity.kind == "radius":
                constraints.append((quantity.id, quantity.expression.to_sympy()))
            elif quantity.kind == "diameter":
                constraints.append((quantity.id, sp.simplify(quantity.expression.to_sympy() / 2)))
        if not constraints:
            return None, ()
        for source_id, value in constraints:
            if value.is_number is not True or value.is_real is not True or value.is_positive is not True:
                raise InvalidGeometryInput(
                    f"Circle radius constraint {source_id} must be a positive real numeric value"
                )
        radius = constraints[0][1]
        conflicts = [
            source_id
            for source_id, value in constraints[1:]
            if sp.simplify(value - radius) != 0
        ]
        if conflicts:
            sources = ", ".join(source_id for source_id, _ in constraints)
            raise _ConditionConflictError(
                f"Circle {candidate.id} has conflicting radius constraints: {sources}"
            )
        return radius, tuple(
            source_id
            for source_id, _ in constraints
            if not source_id.startswith("object:")
        )

    def _compile_regions(self) -> None:
        pending = list(self.graph.regions)
        materialized = set()
        while pending:
            progressed = False
            deferred = []
            for region in pending:
                dependencies = set(region.holes) | set(region.components)
                if region.expression is not None:
                    dependencies.update(region.expression.references())
                region_dependencies = dependencies & {item.id for item in self.graph.regions}
                if not region_dependencies.issubset(materialized):
                    deferred.append(region)
                    continue
                boundary_dependencies = {reference.ref for reference in region.boundary}
                unavailable = sorted(
                    dependency
                    for dependency in boundary_dependencies
                    if self.outcomes.get(dependency) != "materialized"
                )
                if unavailable:
                    self._unsupported(
                        region.id,
                        "region",
                        f"Boundary objects were not materialized: {', '.join(unavailable)}",
                    )
                    progressed = True
                    continue
                try:
                    operation = _region_operation(region)
                except (InvalidGeometryInput, KeyError, TypeError, ValueError) as exc:
                    self._unsupported(region.id, "region", str(exc))
                    progressed = True
                    continue
                self.trace.append(operation)
                self.outcomes[region.id] = "materialized"
                materialized.add(region.id)
                progressed = True
            if not progressed:
                for region in deferred:
                    self._unsupported(region.id, "region", "Region dependencies were not materialized")
                break
            pending = deferred

    def _compile_relation(self, relation: ParseRelation) -> None:
        self._add_source_fact(relation.predicate, relation.id, relation, relation.arguments)
        if relation.id in self.constructed_relation_ids:
            self.outcomes[relation.id] = "materialized"
            return
        if relation.predicate not in RELATION_TYPES:
            self.outcomes[relation.id] = "source_fact_only"
            return
        materialized = _materialized_ids(self.trace)
        unavailable = sorted(
            argument.ref
            for argument in relation.arguments
            if argument.ref not in materialized
        )
        if unavailable:
            self._unsupported(
                relation.id,
                "relation",
                f"Relation arguments were not materialized: {', '.join(unavailable)}",
            )
            return
        try:
            payload = _state_relation_payload(relation, self.objects_by_id)
        except InvalidGeometryInput as exc:
            self._unsupported(relation.id, "relation", str(exc))
            return
        self.trace.append(ConstructionOp("add_relation", relation.id, payload))
        self.outcomes[relation.id] = "materialized"

    def _compile_quantity(self, quantity: ParseQuantity) -> None:
        object_id = quantity.arguments[0].ref
        value = quantity.expression.to_sympy()
        self.source_facts.add(
            quantity.kind,
            object_id,
            value,
            source="parser",
            scope=FactScope.SOURCE_ASSERTION,
            rule="admitted_parse_quantity",
            dependencies=tuple(argument.ref for argument in quantity.arguments),
            metadata={"candidate_id": quantity.id, "unit": quantity.unit, "raw_text": quantity.raw_text},
            provenance=(_provenance(self.graph, quantity.id, quantity.evidence_refs),),
        )
        self.outcomes[quantity.id] = "source_fact_only"

    def _compile_target(self, target: ParseTarget) -> None:
        if target.mode == "proof":
            if not target.arguments:
                self._unsupported(target.id, "target", "Proof targets require at least one object reference")
                return
        elif len(target.arguments) != 1:
            self._unsupported(target.id, "target", "Formal value targets require exactly one object reference")
            return
        target_type = normalize_target_type(target.kind)
        if target_type not in TARGET_TYPES:
            self._unsupported(target.id, "target", f"Unsupported formal target type: {target.kind}")
            return
        argument_ids = tuple(argument.ref for argument in target.arguments)
        unavailable = tuple(sorted(set(argument_ids) - _materialized_ids(self.trace)))
        if unavailable:
            self._unsupported(
                target.id,
                "target",
                f"Target objects were not materialized: {', '.join(unavailable)}",
            )
            return
        object_id = str(target.metadata.get("object_key") or _target_object_key(target))
        params: dict[str, Any] = {
            "type": target_type,
            "object": object_id,
            "mode": target.mode,
            "metadata": {"parse_candidate_id": target.id, "evidence_refs": list(target.evidence_refs)},
        }
        if target.mode == "constraint":
            if target.expression is None or target.metadata.get("operator") not in {
                "eq",
                "ne",
                "lt",
                "le",
                "gt",
                "ge",
            }:
                self._unsupported(target.id, "target", "Constraint target requires an operator and expression")
                return
            params["expectation"] = {
                "operator": target.metadata["operator"],
                "value": str(target.expression.to_sympy()),
            }
        self.trace.append(ConstructionOp("set_target", None, params))
        self.outcomes[target.id] = "materialized"

    def _add_source_fact(
        self,
        fact_type: str,
        object_id: str,
        candidate: ParseRelation,
        arguments: Sequence[ParseReference],
    ) -> None:
        self.source_facts.add(
            fact_type,
            object_id,
            source="parser",
            scope=FactScope.SOURCE_ASSERTION,
            rule="admitted_parse_relation",
            dependencies=tuple(argument.ref for argument in arguments),
            metadata={"candidate_id": candidate.id, "arguments": [item.to_dict() for item in arguments]},
            provenance=(_provenance(self.graph, candidate.id, candidate.evidence_refs),),
        )

    def _unsupported(self, candidate_id: str, kind: str, message: str) -> None:
        self._issue(candidate_id, f"reconstruction.unsupported_{kind}", message)

    def _ambiguous(self, candidate_id: str, message: str) -> None:
        self.outcomes[candidate_id] = "ambiguous"
        self.issues.append(ReconstructionIssue("reconstruction.branch_ambiguous", message, candidate_id))

    def _issue(self, candidate_id: str, code: str, message: str) -> None:
        self.outcomes[candidate_id] = "unsupported"
        self.issues.append(ReconstructionIssue(code, message, candidate_id))


class _UnderconstrainedError(InvalidGeometryInput):
    pass


class _BranchAmbiguousError(InvalidGeometryInput):
    pass


class _ConstructionAmbiguousError(InvalidGeometryInput):
    pass


class _ConditionConflictError(InvalidGeometryInput):
    pass


def _expression(value: Any, context: str) -> sp.Expr:
    if isinstance(value, MathExpression):
        return value.to_sympy()
    if isinstance(value, Mapping):
        return MathExpression.from_dict(value).to_sympy()
    raise InvalidGeometryInput(f"{context} must use the typed MathExpression schema")


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidGeometryInput(f"{context} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise InvalidGeometryInput(f"{context} must be finite")
    return number


def _references_by_role(references: Sequence[ParseReference]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for reference in references:
        grouped.setdefault(reference.role, []).append(reference.ref)
    return grouped


def _intersection_point_plans(
    graph: ParseGraph,
) -> dict[str, tuple[_IntersectionPointPlan, ...]]:
    plans: dict[str, list[_IntersectionPointPlan]] = {}
    for relation in graph.relations:
        if relation.predicate != "intersection":
            continue
        point_ids = tuple(
            argument.ref
            for argument in relation.arguments
            if argument.role == "point" or argument.role.startswith("point_")
        )
        support_ids = tuple(
            argument.ref
            for argument in relation.arguments
            if argument.role in {"support", "object"}
        )
        if len(point_ids) != 1 or len(support_ids) != 2:
            continue
        plans.setdefault(point_ids[0], []).append(
            _IntersectionPointPlan(relation, (support_ids[0], support_ids[1]))
        )
    return {
        point_id: tuple(sorted(items, key=lambda item: item.relation.id))
        for point_id, items in plans.items()
    }


def _derived_point_plans(graph: ParseGraph) -> dict[str, tuple[_DerivedPointPlan, ...]]:
    plans: dict[str, list[_DerivedPointPlan]] = {}
    for relation in graph.relations:
        refs = _references_by_role(relation.arguments)
        point_ids = refs.get("point", ())
        if len(point_ids) != 1:
            continue
        point_id = point_ids[0]
        try:
            if relation.predicate == "midpoint":
                endpoints = tuple(_ordered_refs(refs, "of", minimum=2))
                if len(endpoints) != 2:
                    continue
                plan = _DerivedPointPlan(relation, "midpoint", endpoints, {})
            elif relation.predicate == "foot_of_perpendicular":
                source_id = _one_of(refs, "from", "source")
                support_id = _one_of(refs, "line", "support")
                plan = _DerivedPointPlan(
                    relation,
                    "foot_of_perpendicular",
                    (source_id, support_id),
                    {},
                )
            elif relation.predicate == "point_on_segment_ratio":
                segment_id = _one_of(refs, "segment", "support")
                plan = _DerivedPointPlan(
                    relation,
                    "point_on_segment_ratio",
                    (segment_id,),
                    {"ratio": relation.metadata.get("ratio")},
                )
            elif relation.predicate == "point_on_ray_distance":
                ray_id = _one_of(refs, "ray", "support")
                plan = _DerivedPointPlan(
                    relation,
                    "point_on_ray_distance",
                    (ray_id,),
                    {"distance": relation.metadata.get("distance")},
                )
            else:
                continue
        except InvalidGeometryInput:
            continue
        plans.setdefault(point_id, []).append(plan)
    return {
        point_id: tuple(sorted(items, key=lambda item: item.relation.id))
        for point_id, items in plans.items()
    }


def _derived_line_plans(graph: ParseGraph) -> dict[str, tuple[_DerivedLinePlan, ...]]:
    objects = {candidate.id: candidate for candidate in graph.objects}
    linear_kinds = {
        ParseObjectKind.LINE,
        ParseObjectKind.RAY,
        ParseObjectKind.SEGMENT,
    }
    through_points: dict[str, str] = {}
    for candidate in graph.objects:
        if candidate.kind is not ParseObjectKind.LINE:
            continue
        try:
            point_ids = _ordered_refs(
                _references_by_role(candidate.references),
                "point",
                minimum=1,
            )
        except InvalidGeometryInput:
            continue
        if len(point_ids) == 1:
            through_points[candidate.id] = point_ids[0]

    plans: dict[str, list[_DerivedLinePlan]] = {}
    for relation in graph.relations:
        if relation.predicate not in {"parallel", "perpendicular"}:
            continue
        linear_ids = tuple(
            argument.ref
            for argument in relation.arguments
            if objects.get(argument.ref) is not None
            and objects[argument.ref].kind in linear_kinds
        )
        if len(linear_ids) != 2 or len(set(linear_ids)) != 2 or len(relation.arguments) != 2:
            continue
        for line_id in linear_ids:
            through_point_id = through_points.get(line_id)
            if through_point_id is None:
                continue
            base_line_id = linear_ids[1] if linear_ids[0] == line_id else linear_ids[0]
            operation = "parallel_through" if relation.predicate == "parallel" else "perpendicular_through"
            plans.setdefault(line_id, []).append(
                _DerivedLinePlan(
                    relation=relation,
                    operation=operation,
                    dependency_ids=(through_point_id, base_line_id),
                    params={},
                )
            )
    for relation in graph.relations:
        if relation.predicate != "angle_bisector":
            continue
        refs = _references_by_role(relation.arguments)
        try:
            line_id = _one_ref(refs, "line")
            vertex_id = _one_ref(refs, "vertex")
            arm_ids = _ordered_refs(refs, "arm", minimum=2)
        except InvalidGeometryInput:
            continue
        if len(relation.arguments) != 4 or len(arm_ids) != 2:
            continue
        through_point_id = through_points.get(line_id)
        if through_point_id is None:
            continue
        mode = str(relation.metadata.get("mode", "internal"))
        if mode not in {"internal", "external"}:
            continue
        plans.setdefault(line_id, []).append(
            _DerivedLinePlan(
                relation=relation,
                operation="angle_bisector",
                dependency_ids=(vertex_id, arm_ids[0], arm_ids[1]),
                params={"mode": mode},
            )
        )
    for relation in graph.relations:
        if relation.predicate != "angle_from_line":
            continue
        refs = _references_by_role(relation.arguments)
        try:
            line_id = _one_ref(refs, "line")
            through_point_id = _one_ref(refs, "point")
            base_line_id = _one_ref(refs, "base_line")
        except InvalidGeometryInput:
            continue
        if len(relation.arguments) != 3 or through_points.get(line_id) != through_point_id:
            continue
        plans.setdefault(line_id, []).append(
            _DerivedLinePlan(
                relation=relation,
                operation="line_through_angle",
                dependency_ids=(through_point_id, base_line_id),
                params={"angle": relation.metadata.get("angle")},
            )
        )
    return {
        line_id: tuple(sorted(items, key=lambda item: item.relation.id))
        for line_id, items in plans.items()
    }


def _distance_gauge_plans(graph: ParseGraph) -> tuple[_DistanceGaugePlan, ...]:
    points = {
        candidate.id: candidate
        for candidate in graph.objects
        if candidate.kind is ParseObjectKind.POINT
    }
    segments = {
        candidate.id: candidate
        for candidate in graph.objects
        if candidate.kind is ParseObjectKind.SEGMENT
    }
    lengths: dict[str, tuple[sp.Expr, tuple[str, ...]]] = {}
    quantity_groups: dict[str, list[ParseQuantity]] = {}
    for quantity in graph.quantities:
        if quantity.kind != "length" or len(quantity.arguments) != 1:
            continue
        segment_id = quantity.arguments[0].ref
        if segment_id in segments:
            quantity_groups.setdefault(segment_id, []).append(quantity)
    for segment_id, quantities in quantity_groups.items():
        values = tuple(quantity.expression.to_sympy() for quantity in quantities)
        if any(value.is_number is not True or value.is_real is not True for value in values):
            continue
        first = sp.simplify(values[0])
        if any(sp.simplify(value - first).is_zero is not True for value in values[1:]):
            continue
        lengths[segment_id] = first, tuple(sorted(quantity.id for quantity in quantities))

    all_edges = []
    for segment in segments.values():
        refs = _references_by_role(segment.references)
        starts = refs.get("start", ())
        ends = refs.get("end", ())
        if len(starts) != 1 or len(ends) != 1 or starts[0] not in points or ends[0] not in points:
            continue
        all_edges.append((segment, starts[0], ends[0]))
    distance_edges = tuple(
        _DistanceEdge(
            segment_id=segment.id,
            start_id=start_id,
            end_id=end_id,
            length=lengths[segment.id][0],
            quantity_ids=lengths[segment.id][1],
        )
        for segment, start_id, end_id in all_edges
        if segment.id in lengths
    )
    adjacency: dict[str, set[str]] = {}
    for edge in distance_edges:
        adjacency.setdefault(edge.start_id, set()).add(edge.end_id)
        adjacency.setdefault(edge.end_id, set()).add(edge.start_id)

    intersection_points = set(_intersection_point_plans(graph))
    derived_points = set(_derived_point_plans(graph))
    raw_plans = []
    remaining = set(adjacency)
    while remaining:
        root = min(remaining)
        component = set()
        stack = [root]
        while stack:
            point_id = stack.pop()
            if point_id in component:
                continue
            component.add(point_id)
            stack.extend(sorted(adjacency.get(point_id, ()), reverse=True))
        remaining.difference_update(component)
        if len(component) < 3:
            continue
        if any(points[point_id].geometry.get("exact_coords") is not None for point_id in component):
            continue
        if component & (intersection_points | derived_points):
            continue
        component_edges = [
            edge
            for edge in all_edges
            if edge[1] in component or edge[2] in component
        ]
        if any(
            start_id not in component or end_id not in component
            for _, start_id, end_id in component_edges
        ):
            continue
        by_pair: dict[frozenset[str], _DistanceEdge] = {}
        duplicate_pair = False
        for segment, start_id, end_id in component_edges:
            if segment.id not in lengths:
                duplicate_pair = True
                break
            pair = frozenset((start_id, end_id))
            if len(pair) != 2 or pair in by_pair:
                duplicate_pair = True
                break
            value, quantity_ids = lengths[segment.id]
            by_pair[pair] = _DistanceEdge(
                segment_id=segment.id,
                start_id=start_id,
                end_id=end_id,
                length=value,
                quantity_ids=quantity_ids,
            )
        if duplicate_pair or len(by_pair) < 2 * len(component) - 3:
            continue
        scaffold = _distance_scaffold(component, by_pair)
        if scaffold is None:
            continue
        base_points, attachment_points = scaffold
        base_edges = tuple(
            by_pair[frozenset(pair)]
            for pair in (
                (base_points[0], base_points[1]),
                (base_points[0], base_points[2]),
                (base_points[1], base_points[2]),
            )
        )
        base = min(base_edges, key=lambda edge: edge.segment_id)
        base_start_id = base.start_id
        base_end_id = base.end_id
        base_apex_id = next(
            point_id
            for point_id in base_points
            if point_id not in {base_start_id, base_end_id}
        )
        placements = [
            _distance_placement_plan(base_apex_id, base_start_id, base_end_id, by_pair)
        ]
        placements.extend(
            _distance_placement_plan(point_id, support_ids[0], support_ids[1], by_pair)
            for point_id, support_ids in attachment_points
        )
        status = next(
            (placement.status for placement in placements if placement.status != "valid"),
            "valid",
        )
        component_distance_edges = tuple(
            sorted(
                (edge for edge in distance_edges if edge.start_id in component and edge.end_id in component),
                key=lambda edge: edge.segment_id,
            )
        )
        side_lengths = tuple(edge.length for edge in component_distance_edges)
        raw_plans.append(
            {
                "base_start_id": base_start_id,
                "base_end_id": base_end_id,
                "base_length": base.length,
                "placements": tuple(placements),
                "edges": component_distance_edges,
                "segment_ids": tuple(edge.segment_id for edge in component_distance_edges),
                "quantity_ids": tuple(
                    quantity_id
                    for edge in component_distance_edges
                    for quantity_id in edge.quantity_ids
                ),
                "status": status,
                "gauge_kind": (
                    "sss_triangle_v1"
                    if len(component) == 3
                    else (
                        "triangulated_distance_graph_v1"
                        if len(by_pair) == 2 * len(component) - 3
                        else "redundant_triangulated_distance_graph_v1"
                    )
                ),
                "maximum_side": max(
                    (float(sp.N(value)) for value in side_lengths if value.is_positive is True),
                    default=1.0,
                ),
            }
        )

    plans = []
    x_offset = sp.Integer(0)
    ordered_plans = sorted(
        raw_plans,
        key=lambda item: (
            item["base_start_id"],
            item["base_end_id"],
            tuple(placement.point_id for placement in item["placements"]),
        ),
    )
    for raw in ordered_plans:
        maximum_side = raw.pop("maximum_side")
        plans.append(_DistanceGaugePlan(**raw, x_offset=x_offset))
        x_offset += sp.Integer(max(5, math.ceil(2.0 * maximum_side) + 5))
    return tuple(plans)


def _distance_scaffold(
    component: set[str],
    edges: Mapping[frozenset[str], _DistanceEdge],
) -> tuple[tuple[str, str, str], tuple[tuple[str, tuple[str, str]], ...]] | None:
    if len(edges) == 2 * len(component) - 3:
        decomposition = _two_tree_decomposition(component, edges)
        if decomposition is None:
            return None
        base_points, removed_points = decomposition
        return base_points, tuple(reversed(removed_points))

    ordered_points = tuple(sorted(component))
    for base_points in combinations(ordered_points, 3):
        if any(frozenset(pair) not in edges for pair in combinations(base_points, 2)):
            continue
        placed = set(base_points)
        attachments = []
        while len(placed) < len(component):
            candidates = []
            for point_id in sorted(component - placed):
                neighbors = tuple(
                    sorted(
                        candidate_id
                        for candidate_id in placed
                        if frozenset((point_id, candidate_id)) in edges
                    )
                )
                support_pairs = tuple(
                    pair
                    for pair in combinations(neighbors, 2)
                    if frozenset(pair) in edges
                )
                if support_pairs:
                    candidates.append((point_id, support_pairs[0]))
            if not candidates:
                break
            point_id, support_ids = min(candidates)
            attachments.append((point_id, support_ids))
            placed.add(point_id)
        if len(placed) == len(component):
            return base_points, tuple(attachments)
    return None


def _two_tree_decomposition(
    component: set[str],
    edges: Mapping[frozenset[str], _DistanceEdge],
) -> tuple[tuple[str, str, str], tuple[tuple[str, tuple[str, str]], ...]] | None:
    adjacency = {point_id: set() for point_id in component}
    for pair in edges:
        start_id, end_id = tuple(pair)
        adjacency[start_id].add(end_id)
        adjacency[end_id].add(start_id)
    removed = []
    while len(adjacency) > 3:
        removable = tuple(
            sorted(
                point_id
                for point_id, neighbors in adjacency.items()
                if len(neighbors) == 2 and frozenset(neighbors) in edges
            )
        )
        if not removable:
            return None
        point_id = removable[0]
        support_ids = tuple(sorted(adjacency[point_id]))
        removed.append((point_id, support_ids))
        for support_id in support_ids:
            adjacency[support_id].remove(point_id)
        del adjacency[point_id]
    base_points = tuple(sorted(adjacency))
    if len(base_points) != 3 or any(
        frozenset(pair) not in edges
        for pair in (
            (base_points[0], base_points[1]),
            (base_points[0], base_points[2]),
            (base_points[1], base_points[2]),
        )
    ):
        return None
    return base_points, tuple(removed)


def _distance_placement_plan(
    point_id: str,
    support_start_id: str,
    support_end_id: str,
    edges: Mapping[frozenset[str], _DistanceEdge],
) -> _DistancePlacementPlan:
    support = edges[frozenset((support_start_id, support_end_id))]
    start = edges[frozenset((support_start_id, point_id))]
    end = edges[frozenset((support_end_id, point_id))]
    return _DistancePlacementPlan(
        point_id=point_id,
        support_start_id=support_start_id,
        support_end_id=support_end_id,
        support_length=support.length,
        start_length=start.length,
        end_length=end.length,
        segment_ids=(support.segment_id, start.segment_id, end.segment_id),
        quantity_ids=(*support.quantity_ids, *start.quantity_ids, *end.quantity_ids),
        status=_triangle_length_status(support.length, start.length, end.length),
    )


def _triangle_length_status(base: sp.Expr, first: sp.Expr, second: sp.Expr) -> str:
    side_lengths = (base, first, second)
    if any(value.is_finite is not True for value in side_lengths):
        return "numeric_unstable"
    if any(value.is_positive is not True for value in side_lengths):
        return (
            "condition_conflict"
            if any(value.is_nonpositive is True for value in side_lengths)
            else "numeric_unstable"
        )
    projection = sp.simplify((first**2 - second**2 + base**2) / (2 * base))
    height_squared = sp.simplify(first**2 - projection**2)
    if height_squared.is_positive is True:
        return "valid"
    return "condition_conflict" if height_squared.is_nonpositive is True else "numeric_unstable"


def _trilateration_coordinates(
    support_start: tuple[sp.Expr, sp.Expr],
    support_end: tuple[sp.Expr, sp.Expr],
    support_length: sp.Expr,
    start_length: sp.Expr,
    end_length: sp.Expr,
    selection: str,
) -> tuple[sp.Expr, sp.Expr]:
    dx = sp.simplify(support_end[0] - support_start[0])
    dy = sp.simplify(support_end[1] - support_start[1])
    support_squared = sp.simplify(dx**2 + dy**2)
    if sp.simplify(support_squared - support_length**2).is_zero is not True:
        raise _ConditionConflictError("Trilateration support length conflicts with the current realization")
    projection_ratio = sp.simplify(
        (start_length**2 - end_length**2 + support_squared) / (2 * support_squared)
    )
    foot = (
        sp.simplify(support_start[0] + projection_ratio * dx),
        sp.simplify(support_start[1] + projection_ratio * dy),
    )
    height_squared = sp.simplify(start_length**2 - projection_ratio**2 * support_squared)
    if height_squared.is_positive is not True:
        raise _ConditionConflictError("Trilateration requires a nondegenerate triangle")
    perpendicular_scale = sp.sqrt(sp.simplify(height_squared / support_squared))
    sign = sp.Integer(1) if selection == "first" else sp.Integer(-1)
    return (
        sp.simplify(foot[0] - sign * perpendicular_scale * dy),
        sp.simplify(foot[1] + sign * perpendicular_scale * dx),
    )


def _validate_branch_choices(
    graph: ParseGraph,
    branch_choices: Mapping[str, str],
) -> dict[str, str]:
    if not isinstance(branch_choices, Mapping):
        raise InvalidGeometryInput("branch_choices must be a mapping")
    valid_points = {
        point_id
        for group in _branch_choice_groups(graph)
        for assignment in group
        for point_id in assignment
    }
    normalized = {}
    for point_id, selection in branch_choices.items():
        if not isinstance(point_id, str) or point_id not in valid_points:
            raise InvalidGeometryInput(f"Unknown or non-branching reconstruction point: {point_id!r}")
        if not isinstance(selection, str) or selection not in _RUNTIME_BRANCH_SELECTIONS:
            raise InvalidGeometryInput(
                f"Invalid reconstruction branch selection for {point_id}: {selection!r}"
            )
        normalized[point_id] = selection
    return dict(sorted(normalized.items()))


def _branch_choice_groups(graph: ParseGraph) -> tuple[tuple[dict[str, str], ...], ...]:
    objects = {candidate.id: candidate for candidate in graph.objects}
    plans = _intersection_point_plans(graph)
    grouped: dict[tuple[str, str], list[str]] = {}
    for point_id, point_plans in plans.items():
        point = objects.get(point_id)
        if point is None or point.geometry.get("exact_coords") is not None or len(point_plans) != 1:
            continue
        plan = point_plans[0]
        if plan.relation.metadata.get("selection") is not None:
            continue
        if plan.relation.metadata.get("contact") == "tangent":
            continue
        supports = tuple(objects.get(support_id) for support_id in plan.support_ids)
        if any(support is None for support in supports):
            continue
        if not _is_branching_curve_pair(supports):
            continue
        grouped.setdefault(tuple(sorted(plan.support_ids)), []).append(point_id)

    choice_groups = []
    for support_ids, point_ids in sorted(grouped.items()):
        ordered_points = tuple(sorted(point_ids))
        if len(ordered_points) > 2:
            raise InvalidGeometryInput(
                f"Intersection supports {support_ids} have more than two unresolved point candidates"
            )
        if len(ordered_points) == 1:
            point_id = ordered_points[0]
            choice_groups.append((
                {point_id: "first"},
                {point_id: "second"},
            ))
        else:
            first, second = ordered_points
            choice_groups.append((
                {first: "first", second: "second"},
                {first: "second", second: "first"},
            ))
    used_points = {
        point_id
        for group in choice_groups
        for assignment in group
        for point_id in assignment
    }
    for plan in _distance_gauge_plans(graph):
        if plan.status != "valid":
            continue
        for placement in plan.placements:
            if placement.point_id in used_points:
                raise InvalidGeometryInput(
                    f"Point {placement.point_id} has both a distance gauge branch and another branch construction"
                )
            choice_groups.append(
                (
                    {placement.point_id: "first"},
                    {placement.point_id: "second"},
                )
            )
            used_points.add(placement.point_id)
    return tuple(choice_groups)


def _is_branching_curve_pair(supports: Sequence[ParseObject | None]) -> bool:
    if len(supports) != 2:
        return False
    if any(support is None for support in supports):
        return False
    linear = {ParseObjectKind.LINE, ParseObjectKind.RAY, ParseObjectKind.SEGMENT}
    curves = {ParseObjectKind.CIRCLE, ParseObjectKind.ARC}
    kinds = {support.kind for support in supports}
    return kinds.issubset(curves) or (len(kinds & linear) == 1 and len(kinds & curves) == 1)


def _intersection_selection(
    relation: ParseRelation,
    point_id: str,
    branch_choices: Mapping[str, str],
) -> tuple[str, str]:
    if point_id in branch_choices:
        return branch_choices[point_id], "runtime_choice"
    selection = relation.metadata.get("selection")
    if selection is None and relation.metadata.get("contact") == "tangent":
        return "all", "single_tangent"
    if isinstance(selection, str) and selection in _BRANCH_SELECTIONS:
        return selection, "source_metadata"
    raise _BranchAmbiguousError(
        f"Point {point_id} requires an explicit intersection selection; got {selection!r}"
    )


def _one_ref(grouped: Mapping[str, list[str]], role: str) -> str:
    values = grouped.get(role, ())
    if len(values) != 1:
        raise InvalidGeometryInput(f"Expected exactly one {role} reference")
    return values[0]


def _ordered_refs(grouped: Mapping[str, list[str]], prefix: str, *, minimum: int) -> list[str]:
    pairs = []
    for role, refs in grouped.items():
        if role == prefix:
            pairs.extend((0, ref) for ref in refs)
        elif role.startswith(f"{prefix}_") and role[len(prefix) + 1 :].isdigit():
            pairs.extend((int(role[len(prefix) + 1 :]), ref) for ref in refs)
    pairs.sort(key=lambda item: (item[0], item[1]))
    values = [ref for _, ref in pairs]
    if len(values) < minimum:
        raise InvalidGeometryInput(f"Expected at least {minimum} ordered {prefix} references")
    return values


def _region_operation(region: ParseRegion) -> ConstructionOp:
    metadata = {"parse_candidate_id": region.id, "evidence_refs": list(region.evidence_refs), **dict(region.metadata)}
    if region.expression is not None:
        if any(operand.operator != "reference" for operand in region.expression.operands):
            raise InvalidGeometryInput("Nested region expressions require explicit intermediate ParseRegions")
        if region.expression.operator == "reference":
            raise InvalidGeometryInput("A reference-only ParseRegion cannot create a new formal region")
        components = [operand.ref for operand in region.expression.operands]
        return ConstructionOp(
            "add_composite_region",
            region.id,
            {"operation": region.expression.operator, "components": components, "metadata": metadata},
        )
    return ConstructionOp(
        "add_region",
        region.id,
        {
            "boundary": [reference.ref for reference in region.boundary],
            "holes": list(region.holes),
            "kind": str(region.metadata.get("kind", "polygon_region")),
            "metadata": metadata,
        },
    )


def _state_relation_payload(
    relation: ParseRelation,
    objects_by_id: Mapping[str, ParseObject],
) -> dict[str, Any]:
    refs = _references_by_role(relation.arguments)
    payload: dict[str, Any] = {
        "type": relation.predicate,
        "metadata": {
            "parse_candidate_id": relation.id,
            "evidence_refs": list(relation.evidence_refs),
            **dict(relation.metadata),
        },
    }
    if relation.predicate in {"parallel", "perpendicular", "equal_length"}:
        payload["objects"] = [argument.ref for argument in relation.arguments]
    elif relation.predicate == "equal_angle":
        payload["angles"] = [
            [
                _one_ref(refs, f"angle_{angle_index}_arm_0"),
                _one_ref(refs, f"angle_{angle_index}_vertex"),
                _one_ref(refs, f"angle_{angle_index}_arm_1"),
            ]
            for angle_index in range(2)
        ]
    elif relation.predicate == "collinear":
        payload["points"] = [argument.ref for argument in relation.arguments]
    elif relation.predicate == "intersection":
        payload["objects"] = [
            argument.ref
            for argument in relation.arguments
            if argument.role in {"support", "object"}
        ]
        payload["points"] = [argument.ref for argument in relation.arguments if argument.role.startswith("point")]
    elif relation.predicate in {"point_on_line", "point_on_ray"}:
        payload.update({"point": _one_ref(refs, "point"), "line": _one_of(refs, "line", "support")})
    elif relation.predicate == "point_on_circle":
        payload.update({"point": _one_ref(refs, "point"), "circle": _one_ref(refs, "circle")})
    elif relation.predicate == "point_on_arc":
        payload.update({"point": _one_ref(refs, "point"), "arc": _one_ref(refs, "arc")})
    elif relation.predicate == "tangent":
        line_id = _one_of(refs, "line", "support")
        line = objects_by_id.get(line_id)
        if line is None or line.kind not in {
            ParseObjectKind.LINE,
            ParseObjectKind.SEGMENT,
            ParseObjectKind.RAY,
        }:
            raise InvalidGeometryInput(
                f"Tangent relation {relation.id} requires a line-like object"
            )
        line_refs = _references_by_role(line.references)
        line_points = [
            _one_ref(line_refs, "start"),
            _one_ref(line_refs, "end"),
        ]
        payload.update(
            {
                "point": _one_ref(refs, "point"),
                "line": line_points,
                "circle": _one_ref(refs, "circle"),
            }
        )
    elif relation.predicate == "midpoint":
        payload.update({"point": _one_ref(refs, "point"), "of": _ordered_refs(refs, "of", minimum=2)[:2]})
    elif relation.predicate == "foot_of_perpendicular":
        payload.update(
            {
                "point": _one_ref(refs, "point"),
                "from": _one_of(refs, "from", "source"),
                "line": _one_of(refs, "line", "support"),
            }
        )
    elif relation.predicate == "point_on_segment_ratio":
        segment_id = _one_of(refs, "segment", "support")
        segment = objects_by_id.get(segment_id)
        if segment is None or segment.kind is not ParseObjectKind.SEGMENT:
            raise InvalidGeometryInput(
                f"Segment-ratio relation {relation.id} requires a segment object"
            )
        segment_refs = _references_by_role(segment.references)
        ratio = _expression(relation.metadata.get("ratio"), f"relation {relation.id} ratio")
        payload.update(
            {
                "point": _one_ref(refs, "point"),
                "segment": [
                    _one_ref(segment_refs, "start"),
                    _one_ref(segment_refs, "end"),
                ],
                "ratio": str(ratio),
            }
        )
    elif relation.predicate == "point_on_ray_distance":
        distance = _expression(
            relation.metadata.get("distance"),
            f"relation {relation.id} distance",
        )
        payload.update(
            {
                "point": _one_ref(refs, "point"),
                "ray": _one_of(refs, "ray", "support"),
                "distance": str(distance),
            }
        )
    elif relation.predicate == "angle_bisector":
        arms = _ordered_refs(refs, "arm", minimum=2)
        if len(arms) != 2:
            raise InvalidGeometryInput(
                f"Angle-bisector relation {relation.id} requires two arm points"
            )
        mode = str(relation.metadata.get("mode", "internal"))
        if mode not in {"internal", "external"}:
            raise InvalidGeometryInput(
                f"Angle-bisector relation {relation.id} has invalid mode: {mode}"
            )
        payload.update(
            {
                "line": _one_ref(refs, "line"),
                "angle": [arms[0], _one_ref(refs, "vertex"), arms[1]],
                "mode": mode,
            }
        )
    elif relation.predicate == "angle_from_line":
        angle = _expression(
            relation.metadata.get("angle"),
            f"relation {relation.id} angle",
        )
        payload.update(
            {
                "line": _one_ref(refs, "line"),
                "base_line": _one_ref(refs, "base_line"),
                "angle": str(angle),
            }
        )
    elif relation.predicate in {"similar", "congruent", "triangle"}:
        payload["polygons"] = [argument.ref for argument in relation.arguments]
    else:
        raise InvalidGeometryInput(f"No deterministic state mapping for relation {relation.predicate}")
    return payload


def _one_of(grouped: Mapping[str, list[str]], *roles: str) -> str:
    values = [ref for role in roles for ref in grouped.get(role, ())]
    if len(values) != 1:
        raise InvalidGeometryInput(f"Expected exactly one of references: {', '.join(roles)}")
    return values[0]


def _target_object_key(target: ParseTarget) -> str:
    if normalize_target_type(target.kind) == "tangent_to_circle":
        references = [
            argument.ref
            for argument in target.arguments
            if argument.role in {"point", "circle"}
        ]
        if len(references) != 2:
            raise InvalidGeometryInput(
                f"Tangent proof target {target.id} requires a point and a circle"
            )
        return "|".join(references)
    return "|".join(argument.ref for argument in target.arguments)


def _reconstruction_metadata(candidate: ParseObject, mode: ReconstructionMode) -> dict[str, Any]:
    return {
        "parse_candidate_id": candidate.id,
        "evidence_refs": list(candidate.evidence_refs),
        "reconstruction_mode": mode.value,
        "fact_scope": FactScope.REALIZATION_FACT.value,
        **dict(candidate.metadata),
    }


def _provenance(graph: ParseGraph, candidate_id: str, evidence_refs: Sequence[str]) -> Provenance:
    return Provenance(
        source_id=graph.source_id,
        kind=EvidenceKind.PARSER,
        event_id=candidate_id,
        dependencies=tuple(evidence_refs),
        metadata={"parse_graph_checksum": graph.checksum()},
    )


def _constructed_relation_params(relation: ParseRelation) -> dict[str, Any]:
    return {
        "relation_id": relation.id,
        "relation_dependencies": list(
            dict.fromkeys(
                [argument.ref for argument in relation.arguments]
                + list(relation.dependencies)
            )
        ),
        "relation_metadata": {
            **dict(relation.metadata),
            "parse_candidate_id": relation.id,
            "evidence_refs": list(relation.evidence_refs),
            "origin": relation.origin.value,
            "confidence": relation.confidence,
            "score_kind": relation.score_kind,
            "candidate_dependencies": list(relation.dependencies),
        },
    }


def _materialized_ids(trace: Sequence[ConstructionOp]) -> set[str]:
    return {operation.id for operation in trace if operation.id is not None}


def _validate_admission_binding(admission: ParseGraphAdmissionReport, graph: ParseGraph) -> None:
    metadata = graph.metadata.get("admission")
    if not isinstance(metadata, Mapping):
        raise InvalidGeometryInput("Admitted ParseGraph is missing admission metadata")
    if metadata.get("source_graph_checksum") != admission.source_graph_checksum:
        raise InvalidGeometryInput("Admission report does not match admitted ParseGraph")


def _completion_status(mode: ReconstructionMode, issues: Sequence[ReconstructionIssue]) -> ReconstructionStatus:
    if issues:
        return ReconstructionStatus.PARTIAL
    if mode is ReconstructionMode.EXACT:
        return ReconstructionStatus.EXACT
    return ReconstructionStatus.NUMERIC_WITNESS


def _report(
    admission: ParseGraphAdmissionReport,
    mode: ReconstructionMode,
    status: ReconstructionStatus,
    trace: Sequence[ConstructionOp],
    state: FormalGeometryState | None,
    source_facts: FactStore,
    realization_facts: FactStore,
    verification: Mapping[str, Any],
    issues: Sequence[ReconstructionIssue],
    outcomes: Mapping[str, str],
    branch_choices: Mapping[str, str],
) -> ReconstructionReport:
    return ReconstructionReport(
        source_graph_checksum=admission.source_graph_checksum,
        admitted_graph_checksum=admission.admitted_graph.checksum(),
        mode=mode,
        status=status,
        trace=tuple(trace),
        state=state,
        source_facts=source_facts,
        realization_facts=realization_facts,
        verification=dict(verification),
        analysis=_reconstruction_analysis(admission, mode, outcomes, branch_choices),
        issues=tuple(issues),
        candidate_outcomes=dict(outcomes),
        branch_choices=dict(branch_choices),
    )


def _reconstruction_analysis(
    admission: ParseGraphAdmissionReport,
    mode: ReconstructionMode,
    outcomes: Mapping[str, str],
    branch_choices: Mapping[str, str],
) -> ReconstructionAnalysis:
    graph = admission.admitted_graph
    unplaced_points = tuple(
        sorted(
            candidate.id
            for candidate in graph.objects
            if candidate.kind is ParseObjectKind.POINT and outcomes.get(candidate.id) != "materialized"
        )
    )
    free_symbols = set()
    for candidate in graph.objects:
        exact_coords = candidate.geometry.get("exact_coords")
        if not isinstance(exact_coords, Sequence) or isinstance(exact_coords, (str, bytes)):
            continue
        for value in exact_coords:
            try:
                free_symbols.update(str(symbol) for symbol in _expression(value, "exact coordinate").free_symbols)
            except InvalidGeometryInput:
                continue
    ambiguous = tuple(
        sorted(
            {
                *(
                    decision.candidate_id
                    for decision in admission.decisions
                    if decision.reason in {"unresolved_conflict", "dependency_cycle_or_unresolved"}
                ),
                *(candidate_id for candidate_id, outcome in outcomes.items() if outcome == "ambiguous"),
            }
        )
    )
    unsupported = tuple(sorted(candidate_id for candidate_id, outcome in outcomes.items() if outcome == "unsupported"))
    if mode is ReconstructionMode.PIXEL_WITNESS:
        coordinate_basis = "observed_pixel_witness"
        freedom_status = "not_estimated_from_pixels"
    elif unplaced_points:
        coordinate_basis = "partial_exact_coordinates"
        freedom_status = "underconstrained"
    else:
        coordinate_basis = (
            "constructed_exact_coordinates"
            if any(
                candidate.kind is ParseObjectKind.POINT
                and candidate.geometry.get("exact_coords") is None
                for candidate in graph.objects
            )
            else "explicit_exact_coordinates"
        )
        freedom_status = "parametric" if free_symbols else "fully_instantiated"
    return ReconstructionAnalysis(
        coordinate_basis=coordinate_basis,
        free_parameter_status=freedom_status,
        free_parameters=tuple(sorted(free_symbols)),
        unplaced_points=unplaced_points,
        branch_status="ambiguous" if ambiguous else "selected" if branch_choices else "resolved",
        ambiguous_candidates=ambiguous,
        unsupported_candidates=unsupported,
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, sp.Basic):
        return str(sp.simplify(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value
