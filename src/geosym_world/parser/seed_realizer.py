"""Generate a legal, non-exact scene from an admitted structural graph."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping

import numpy as np

from geosym_world.construction.builder import ConstructionBuilder
from geosym_world.errors import GeometryWorldError, InvalidGeometryInput
from geosym_world.parser.constraint_completion import CompletionStatus, build_constraint_completion_plan
from geosym_world.rendering import state_to_render_scene
from geosym_world.rendering.annotation_policy import AnnotationPolicy
from geosym_world.rendering.scene import RenderScene
from geosym_world.schema import (
    ConflictStatus,
    FormalGeometryState,
    MathExpression,
    ParseGraph,
    ParseObject,
    ParseObjectKind,
)
from geosym_world.schema.canonical import stable_checksum
from geosym_world.verifier import TraceVerifier
from geosym_world.verifier.parse_graph import ParseGraphAdmissionReport


SEED_REALIZATION_SCHEMA_VERSION = "geosym.parser.seed_realization.v2"
_POINT_ON_SUPPORT = frozenset({"incidence", "point_on_line", "point_on_segment", "point_on_ray"})
_POINT_ON_SUPPORT_LAYOUT = _POINT_ON_SUPPORT | {"point_on_segment_ratio"}
_LINE_KINDS = frozenset({ParseObjectKind.LINE, ParseObjectKind.RAY, ParseObjectKind.SEGMENT})


def _relation_refs(relation: Any) -> list[str]:
    roles = {argument.role: argument.ref for argument in relation.arguments}
    if relation.predicate in _POINT_ON_SUPPORT:
        return [roles["point"], roles["support"]]
    if relation.predicate in {"parallel", "perpendicular"}:
        return [roles["left"], roles["right"]]
    return [argument.ref for argument in relation.arguments]


def _midpoint_refs(relation: Any, edges: Mapping[str, tuple[str, str]]) -> tuple[str, ...]:
    roles = {argument.role: argument.ref for argument in relation.arguments}
    point = roles.get("point")
    if point is None:
        return ()
    if "support" in roles:
        endpoints = edges.get(roles["support"], ())
    else:
        endpoints = tuple(argument.ref for argument in relation.arguments if argument.role != "point")
    if len(endpoints) != 2 or len({point, *endpoints}) != 3:
        return ()
    return (point, *endpoints)


class SeedRealizationStatus(str, Enum):
    REALIZED = "realized"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"


class ConstraintStatus(str, Enum):
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class SkeletonBuildReport:
    state: FormalGeometryState
    skipped_operations: tuple[Mapping[str, Any], ...] = ()
    applied_constraints: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConstraintApplicationReport:
    state: FormalGeometryState
    applied_ids: tuple[str, ...] = ()
    skipped_operations: tuple[Mapping[str, Any], ...] = ()


class ConstraintApplier:
    """Apply relation operations independently after the drawable state exists."""

    def apply(
        self,
        state: FormalGeometryState,
        operations: Any,
        *,
        allow_partial: bool,
    ) -> ConstraintApplicationReport:
        builder = ConstructionBuilder(state)
        applied: list[str] = []
        skipped: list[Mapping[str, Any]] = []
        for operation in operations:
            try:
                builder.execute([operation])
                applied.append(str(operation.get("id", "")))
            except GeometryWorldError as error:
                if not allow_partial:
                    raise
                skipped.append({
                    "operation": operation.get("op"),
                    "id": operation.get("id"),
                    "reason": str(error),
                })
        return ConstraintApplicationReport(
            state=builder.state,
            applied_ids=tuple(applied),
            skipped_operations=tuple(skipped),
        )


class SkeletonSeedBuilder:
    """Build drawable geometry while quarantining optional construction failures."""

    def __init__(self, graph: ParseGraph, input_status: str) -> None:
        self.graph = graph
        self.input_status = input_status

    def build(self, trace: Any, *, allow_partial: bool) -> SkeletonBuildReport:
        graph_checksum = self.graph.checksum()
        scoped_trace = [
            {
                **operation,
                "metadata": {
                    **operation.get("metadata", {}),
                    "parser_state_scope": (
                        "candidate_instance"
                        if self.input_status == "proposal"
                        else "admitted_instance"
                    ),
                    "source_parse_graph_checksum": graph_checksum,
                    "parser_input_status": self.input_status,
                },
            }
            for operation in trace
        ]
        builder = ConstructionBuilder()
        skipped: list[Mapping[str, Any]] = []
        base_operations = []
        constraint_operations = []
        for operation in scoped_trace:
            if operation["op"] == "set_target":
                continue
            if operation["op"] == "add_relation":
                constraint_operations.append(operation)
            else:
                base_operations.append(operation)
        for operation in base_operations:
            snapshot = FormalGeometryState.from_dict(builder.state.to_dict())
            try:
                builder.execute([operation])
                if operation["op"] == "add_region":
                    verification = TraceVerifier().verify_state(builder.state)
                    invalid_region = next(
                        (
                            issue.to_dict() if hasattr(issue, "to_dict") else issue
                            for issue in verification.issues
                            if (
                                issue.to_dict() if hasattr(issue, "to_dict") else issue
                            ).get("code") == "state.invalid_region_boundary"
                        ),
                        None,
                    )
                    if invalid_region is not None:
                        builder.state = snapshot
                        skipped.append({
                            "operation": operation["op"],
                            "id": operation.get("id"),
                            "reason": invalid_region.get("message", "invalid_region_boundary"),
                        })
            except GeometryWorldError as error:
                if not allow_partial:
                    raise
                skipped.append({
                    "operation": operation["op"],
                    "id": operation.get("id"),
                    "reason": str(error),
                })
        constraints = ConstraintApplier().apply(
            builder.state,
            constraint_operations,
            allow_partial=allow_partial,
        )
        skipped.extend(constraints.skipped_operations)
        return SkeletonBuildReport(
            state=constraints.state,
            skipped_operations=tuple(skipped),
            applied_constraints=constraints.applied_ids,
        )


@dataclass(frozen=True)
class ConstraintOutcome:
    candidate_id: str
    status: ConstraintStatus
    usage: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "candidate_id": self.candidate_id,
            "status": self.status.value,
            "usage": self.usage,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SeedRealization:
    source_graph_checksum: str
    admitted_graph_checksum: str
    status: SeedRealizationStatus
    scene: RenderScene | None
    point_positions: Mapping[str, tuple[float, float]]
    free_parameters: tuple[str, ...] = ()
    satisfied_constraints: tuple[str, ...] = ()
    unresolved_constraints: tuple[str, ...] = ()
    constraint_outcomes: tuple[ConstraintOutcome, ...] = ()
    diagnostics: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    geometry_state: FormalGeometryState | None = None
    verification_report: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SEED_REALIZATION_SCHEMA_VERSION

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_graph_checksum": self.source_graph_checksum,
            "admitted_graph_checksum": self.admitted_graph_checksum,
            "status": self.status.value,
            "scene": self.scene.to_dict() if self.scene is not None else None,
            "point_positions": {
                key: [float(value[0]), float(value[1])]
                for key, value in sorted(self.point_positions.items())
            },
            "free_parameters": list(self.free_parameters),
            "satisfied_constraints": list(self.satisfied_constraints),
            "unresolved_constraints": list(self.unresolved_constraints),
            "constraint_outcomes": [item.to_dict() for item in self.constraint_outcomes],
            "diagnostics": list(self.diagnostics),
            "metadata": dict(self.metadata),
            "geometry_state": self.geometry_state.to_dict() if self.geometry_state is not None else None,
            "verification_report": dict(self.verification_report),
        }

    def checksum(self) -> str:
        return stable_checksum(self._payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "checksum": self.checksum()}


class SeedRealizer:
    """Realize admitted structure without pretending it is exact geometry."""

    def __init__(self, *, point_fraction: float = 0.55):
        if not 0.0 < point_fraction < 1.0:
            raise InvalidGeometryInput("Seed point_fraction must be between zero and one")
        self.point_fraction = float(point_fraction)

    def realize(self, admission: ParseGraphAdmissionReport) -> SeedRealization:
        if not isinstance(admission, ParseGraphAdmissionReport):
            raise TypeError("SeedRealizer requires a ParseGraphAdmissionReport")
        return self._realize_graph(
            admission.admitted_graph,
            source_graph_checksum=admission.source_graph_checksum,
            input_status="admitted",
        )

    def realize_proposal(self, graph: ParseGraph) -> SeedRealization:
        if not isinstance(graph, ParseGraph):
            raise TypeError("SeedRealizer proposal input must be a ParseGraph")
        if any(item.status is ConflictStatus.UNRESOLVED for item in graph.conflicts):
            return self._unsupported(graph, "unresolved_conflicts", input_status="proposal")
        result = self._realize_graph(
            graph,
            source_graph_checksum=graph.checksum(),
            input_status="proposal",
        )
        return replace(result, metadata={
            **result.metadata,
            "state_scope": "candidate_instance",
            "source_facts_admitted": False,
            "selected_assumption_ids": [item.id for item in (*graph.relations, *graph.quantities)],
        })

    def _realize_graph(
        self,
        graph: ParseGraph,
        *,
        source_graph_checksum: str,
        input_status: str,
    ) -> SeedRealization:
        result = self._realize_geometry(
            graph, source_graph_checksum=source_graph_checksum, input_status=input_status,
        )
        state = result.geometry_state
        bindings = self._state_bindings(graph, state) if state is not None else {}
        missing_objects = [
            item.id for item in graph.objects
            if item.kind not in {ParseObjectKind.TEXT, ParseObjectKind.MARKER} and item.id not in bindings
        ]
        missing_regions = [item.id for item in graph.regions if item.id not in bindings]
        target_outcomes = []
        for target in graph.targets:
            refs = [bindings.get(argument.ref) for argument in target.arguments]
            reason = "unsupported_target_contract"
            if len(graph.targets) > 1:
                reason = "state_supports_one_target"
            elif not refs or any(ref is None for ref in refs):
                reason = "target_object_not_materialized"
            elif target.mode == "value" and len(refs) == 1 and state is not None:
                supported = {
                    "length": set(state.segments),
                    "arc_length": set(state.arcs),
                    "radius": set(state.circles),
                    "diameter": set(state.circles),
                    "area": set(state.polygons) | set(state.regions) | set(state.composite_regions) | set(state.circles),
                    "perimeter": set(state.polygons) | set(state.regions),
                }
                if refs[0] in supported.get(target.kind, set()):
                    ConstructionBuilder(state).execute([{
                        "op": "set_target", "type": target.kind, "object": refs[0], "mode": target.mode,
                        "metadata": {
                            "source_parse_target_id": target.id,
                            "parse_target": target.to_dict(),
                            "parser_state_scope": "candidate_instance" if input_status == "proposal" else "admitted_instance",
                            "source_parse_graph_checksum": graph.checksum(),
                        },
                    }])
                    reason = "bound_to_geometry_state"
            target_outcomes.append({
                "candidate_id": target.id,
                "status": "bound" if reason == "bound_to_geometry_state" else "unsupported",
                "reason": reason,
            })
        missing_targets = [item["candidate_id"] for item in target_outcomes if item["status"] != "bound"]
        incomplete = bool(missing_objects or missing_regions or missing_targets)
        return replace(
            result,
            status=SeedRealizationStatus.PARTIAL if incomplete and state is not None else result.status,
            diagnostics=result.diagnostics + (("materialization_incomplete",) if incomplete and state is not None else ()),
            metadata={
                **result.metadata,
                "materialization": {
                    "object_bindings": bindings,
                    "missing_object_ids": missing_objects,
                    "missing_region_ids": missing_regions,
                    "missing_target_ids": missing_targets,
                    "complete": not incomplete,
                },
                "task_targets": [item.to_dict() for item in graph.targets],
                "target_outcomes": target_outcomes,
            },
        )

    @classmethod
    def _state_bindings(cls, graph: ParseGraph, state: FormalGeometryState) -> dict[str, str]:
        state_ids = state.object_ids()
        bindings = {item.id: item.id for item in (*graph.objects, *graph.regions) if item.id in state_ids}
        for item in graph.objects:
            if item.id in bindings:
                continue
            if item.kind is ParseObjectKind.POINT:
                name = cls._point_name(item, graph)
                if name in state.points:
                    bindings[item.id] = name
        for item in graph.objects:
            if item.kind is not ParseObjectKind.SEGMENT or item.id in bindings:
                continue
            endpoints = [bindings.get(ref.ref) for ref in item.references if ref.role in {"start", "end"}]
            matches = [
                segment.id for segment in state.segments.values()
                if len(endpoints) == 2 and None not in endpoints and set(endpoints) == {segment.start, segment.end}
            ]
            if len(matches) == 1:
                bindings[item.id] = matches[0]
        return bindings

    def _realize_geometry(
        self,
        graph: ParseGraph,
        *,
        source_graph_checksum: str,
        input_status: str,
    ) -> SeedRealization:
        supported_proposal_points = self._supported_proposal_point_ids(graph)
        points = tuple(
            item for item in graph.objects
            if item.kind is ParseObjectKind.POINT
            and (input_status != "proposal" or item.id in supported_proposal_points)
        )
        if not points:
            return self._unsupported(graph, "no_supported_points", input_status=input_status)

        names = {item.id: self._point_name(item, graph) for item in points}
        edges = self._line_edges(graph, set(names))
        has_curved_or_region_object = any(
            item.kind in {ParseObjectKind.CIRCLE, ParseObjectKind.ARC, ParseObjectKind.POLYGON}
            for item in graph.objects
        ) or bool(graph.regions)
        if not edges and not has_curved_or_region_object:
            return self._unsupported(graph, "no_structural_edges", input_status=input_status)
        domains = self._line_domains(graph, edges)
        constraints = self._constraints(graph, edges)
        conflicts = self._structural_conflicts(graph, edges)
        if conflicts:
            result = self._unsupported(graph, "inconsistent_structural_constraints", input_status=input_status)
            return replace(result, metadata={**result.metadata, "constraint_conflicts": conflicts})
        completion = build_constraint_completion_plan(graph)
        if completion.status in {CompletionStatus.EXACT, CompletionStatus.DETERMINISTIC_COMPLETION} and completion.trace:
            state = self._build_state(completion.trace, graph, input_status)
            exact_positions = {
                point_id: tuple(float(value) for value in state.points[name].coords)
                for point_id, name in names.items()
                if name in state.points
            }
            if len(exact_positions) == len(points):
                numeric_targets = self._numeric_length_targets(graph, edges)
                satisfied = [
                    relation.id for relation in constraints
                    if self._relation_holds(relation, exact_positions, edges, domains)
                ]
                violated_ids = [relation.id for relation in constraints if relation.id not in satisfied]
                satisfied.extend(
                    relation.id for relation in graph.relations
                    if relation.predicate == "rectangle"
                )
                for quantity_id, edge_id, target in numeric_targets:
                    start, end = edges[edge_id]
                    actual = math.dist(exact_positions[start], exact_positions[end])
                    if abs(actual - target) <= max(1e-5, target * 1e-4):
                        satisfied.append(quantity_id)
                    else:
                        violated_ids.append(quantity_id)
                bindings = self._state_bindings(graph, state)
                objects = {item.id: item for item in graph.objects}
                existing_relations = {
                    item.get("metadata", {}).get("source_parse_relation_id", item.get("id"))
                    for item in state.relations
                }
                for relation in constraints:
                    if relation.id not in satisfied or relation.id in existing_relations:
                        continue
                    payload = self._state_relation_payload(relation, objects)
                    if payload is None:
                        continue
                    payload = {
                        key: [bindings.get(ref, ref) for ref in value] if isinstance(value, list)
                        else bindings.get(value, value) if key != "type" else value
                        for key, value in payload.items()
                    }
                    ConstructionBuilder(state).execute([{
                        "op": "add_relation", "id": relation.id, **payload,
                        "metadata": {
                            "source_parse_relation_id": relation.id,
                            "source_origin": relation.origin.value,
                            "parser_state_scope": "candidate_instance" if input_status == "proposal" else "admitted_instance",
                            "parser_input_status": input_status,
                            "source_parse_graph_checksum": graph.checksum(),
                        },
                    }])
                scene = state_to_render_scene(state, scene_id=f"constraint-seed:{graph.id}", annotation_policy=AnnotationPolicy(inferred_markers=False))
                scene = replace(scene, metadata={
                    **scene.metadata,
                    "output_mode": "constraint_completion",
                    "constraint_exact": True,
                    "source_coordinate_recovery": False,
                })
                verification = TraceVerifier().verify_state(state)
                outcomes = self._apply_verification_outcomes(
                    self._constraint_outcomes(graph, satisfied, violated_ids),
                    state,
                    verification.to_dict(),
                    graph,
                )
                unsupported = tuple(
                    item.candidate_id for item in outcomes
                    if item.status is ConstraintStatus.UNSUPPORTED
                )
                violated = tuple(
                    item.candidate_id for item in outcomes
                    if item.status is ConstraintStatus.VIOLATED
                )
                unresolved = tuple(dict.fromkeys((*violated, *unsupported)))
                return SeedRealization(
                    source_graph_checksum=source_graph_checksum,
                    admitted_graph_checksum=graph.checksum(),
                    status=(
                        SeedRealizationStatus.PARTIAL
                        if unresolved or not verification.valid
                        else SeedRealizationStatus.REALIZED
                    ),
                    scene=scene,
                    point_positions=exact_positions,
                    satisfied_constraints=tuple(item.candidate_id for item in outcomes if item.status is ConstraintStatus.SATISFIED),
                    unresolved_constraints=unresolved,
                    constraint_outcomes=outcomes,
                    diagnostics=tuple(
                        item for item in (
                            "unsupported_constraints_preserved" if unsupported else None,
                            "geometry_state_verification_failed" if not verification.valid else None,
                        )
                        if item is not None
                    ),
                    metadata={
                        "realization_mode": "constraint_completion",
                        "coordinate_source": completion.metadata.get("coordinate_source"),
                        "constraint_exact": True,
                        "source_coordinate_recovery": False,
                        "input_status": input_status,
                        "completion_status": completion.status.value,
                        "completion_method": completion.metadata.get("method"),
                        "completion_plan_checksum": completion.checksum(),
                        "numeric_length_constraints": [item[0] for item in numeric_targets],
                    },
                    geometry_state=state,
                    verification_report=verification.to_dict(),
                )
        positions = self._observed_positions(graph, points)
        position_mode = "observed_pixels"
        if len(positions) != len(points):
            positions = self._semantic_scaffold_positions(points)
            if len(positions) == len(points):
                positions = self._place_incident_points(
                    positions,
                    self._support_edges_by_point(constraints, edges),
                    edges,
                )
                positions = self._place_ordered_points(positions, constraints, edges)
                position_mode = "semantic_scaffold"
            else:
                positions = {}
        if not positions:
            positions = self._seed_positions(graph, points, edges, constraints)
            position_mode = "constraint_seed"

        numeric_targets = self._numeric_length_targets(graph, edges)
        layout_scale = self._layout_scale(graph, positions, edges, numeric_targets)
        positions = {point_id: (position[0] * layout_scale, position[1] * layout_scale) for point_id, position in positions.items()}
        curves = self._curve_geometry(graph, positions, layout_scale)
        positions, curves = self._complete_tangent_circles(graph, positions, edges, curves)
        radial_constraints = self._radial_constraints(graph, curves)
        preserve_seed_shape = (
            graph.metadata.get("extraction_contract") == "geometry_semantics_v1"
            and position_mode in {"constraint_seed", "semantic_scaffold"}
        )
        if not preserve_seed_shape and (
            numeric_targets or radial_constraints or not self._constraints_hold(positions, constraints, edges, domains)
        ):
            positions = self._refine_numeric_witness(
                positions,
                edges,
                domains,
                constraints,
                numeric_targets,
                radial_constraints,
                anchor_weight=(
                    0.12
                    if graph.metadata.get("extraction_contract") == "geometry_semantics_v1"
                    else 1e-6
                ),
            )
            position_mode = f"refined_{position_mode}"
            positions, curves = self._complete_tangent_circles(graph, positions, edges, curves)
        unresolved = [
            relation.id for relation in constraints
            if not self._relation_holds(relation, positions, edges, domains)
        ]
        satisfied = [relation.id for relation in constraints if relation.id not in unresolved]
        for relation in graph.relations:
            if relation.predicate != "point_on_circle":
                continue
            roles = {argument.role: argument.ref for argument in relation.arguments}
            curve = curves.get(roles.get("circle", roles.get("support")))
            point_id = roles.get("point")
            if curve is not None and point_id in positions:
                center, radius = curve
                (satisfied if abs(math.dist(positions[center], positions[point_id]) - radius) <= 1e-5 else unresolved).append(relation.id)
        for quantity in graph.quantities:
            if quantity.kind in {"radius", "diameter"} and len(quantity.arguments) == 1:
                curve = curves.get(quantity.arguments[0].ref)
                value = quantity.expression.to_sympy()
                if curve is not None and not value.free_symbols:
                    actual = curve[1] * (2 if quantity.kind == "diameter" else 1)
                    (satisfied if abs(actual - float(value)) <= 1e-5 else unresolved).append(quantity.id)
        for quantity_id, edge_id, target in numeric_targets:
            start, end = edges[edge_id]
            actual = math.dist(positions[start], positions[end])
            if abs(actual - target) <= max(1e-5, target * 1e-4):
                satisfied.append(quantity_id)
            else:
                unresolved.append(quantity_id)
        for relation in graph.relations:
            if relation.predicate != "tangent":
                continue
            (satisfied if self._tangent_relation_holds(relation, positions, edges, curves) else unresolved).append(relation.id)
        build_report = self._materialize_state(
            graph,
            points,
            positions,
            edges,
            satisfied_ids=set(satisfied),
            position_mode=position_mode,
            input_status=input_status,
            curves=curves,
        )
        state = build_report.state
        verification = TraceVerifier().verify_state(state)
        outcomes = self._apply_verification_outcomes(
            self._constraint_outcomes(graph, satisfied, unresolved),
            state,
            verification.to_dict(),
            graph,
        )
        unsupported = [
            item.candidate_id for item in outcomes
            if item.status is ConstraintStatus.UNSUPPORTED
        ]
        violated = [
            item.candidate_id for item in outcomes
            if item.status is ConstraintStatus.VIOLATED
        ]
        unresolved = list(dict.fromkeys((*unresolved, *violated, *unsupported)))
        satisfied = [item.candidate_id for item in outcomes if item.status is ConstraintStatus.SATISFIED]
        scene = state_to_render_scene(state, scene_id=f"seed:{graph.id}", annotation_policy=AnnotationPolicy(inferred_markers=False))
        scene = replace(scene, metadata={
            **scene.metadata,
            "output_mode": "seed",
            "layout_scope": position_mode,
            "parse_graph_checksum": graph.checksum(),
            "exact": False,
        })
        status = SeedRealizationStatus.REALIZED if len(positions) == len(points) and not unresolved and verification.valid else SeedRealizationStatus.PARTIAL
        diagnostic_items = []
        if any(item.status is ConstraintStatus.VIOLATED for item in outcomes):
            diagnostic_items.append("admitted_constraints_not_realized")
        if unsupported:
            diagnostic_items.append("unsupported_constraints_preserved")
        if not verification.valid:
            diagnostic_items.append("geometry_state_verification_failed")
        return SeedRealization(
            source_graph_checksum=source_graph_checksum,
            admitted_graph_checksum=graph.checksum(),
            status=status,
            scene=scene,
            point_positions=positions,
            free_parameters=(
                ("triangle_shape", "point_fraction")
                if position_mode == "constraint_seed"
                else ("visual_scaffold",) if position_mode == "semantic_scaffold" else ()
            ),
            satisfied_constraints=tuple(satisfied),
            unresolved_constraints=tuple(unresolved),
            constraint_outcomes=outcomes,
            diagnostics=tuple(diagnostic_items),
            metadata={
                "realization_mode": "observation" if position_mode == "observed_pixels" else "seed",
                "coordinate_source": position_mode,
                "seed_refinement": "shape_preserving" if preserve_seed_shape else "numeric_witness",
                "exact": False,
                "point_fraction": self.point_fraction,
                "input_status": input_status,
                "numeric_length_constraints": [item[0] for item in numeric_targets],
                "layout_scale": layout_scale,
                "curve_radii": {curve_id: radius for curve_id, (_, radius) in curves.items()},
                "skeleton": {
                    "built": bool(state.points),
                    "skipped_operations": list(build_report.skipped_operations),
                    "applied_constraints": list(build_report.applied_constraints),
                },
            },
            geometry_state=state,
            verification_report=verification.to_dict(),
        )

    @staticmethod
    def _unsupported(graph: ParseGraph, reason: str, *, input_status: str) -> SeedRealization:
        return SeedRealization(
            source_graph_checksum=graph.metadata.get("source_graph_checksum", graph.checksum()),
            admitted_graph_checksum=graph.checksum(),
            status=SeedRealizationStatus.UNSUPPORTED,
            scene=None,
            point_positions={},
            diagnostics=(reason,),
            metadata={"realization_mode": "seed", "exact": False, "input_status": input_status},
        )

    @staticmethod
    def _constraint_outcomes(
        graph: ParseGraph,
        satisfied_ids: list[str] | tuple[str, ...],
        violated_ids: list[str] | tuple[str, ...],
    ) -> tuple[ConstraintOutcome, ...]:
        satisfied = set(satisfied_ids)
        violated = set(violated_ids)
        outcomes = []
        for candidate in (*graph.relations, *graph.quantities):
            if candidate.id in satisfied:
                outcomes.append(ConstraintOutcome(
                    candidate.id,
                    ConstraintStatus.SATISFIED,
                    "construct_and_verify",
                    "constraint_holds_in_geometry_state",
                ))
            elif candidate.id in violated:
                outcomes.append(ConstraintOutcome(
                    candidate.id,
                    ConstraintStatus.VIOLATED,
                    "construct_and_verify",
                    "constraint_does_not_hold_in_geometry_state",
                ))
            else:
                outcomes.append(ConstraintOutcome(
                    candidate.id,
                    ConstraintStatus.UNSUPPORTED,
                    "preserved_only",
                    "no_generic_realization_rule",
                ))
        return tuple(outcomes)

    @staticmethod
    def _apply_verification_outcomes(
        outcomes: tuple[ConstraintOutcome, ...],
        state: FormalGeometryState,
        verification: Mapping[str, Any],
        graph: ParseGraph,
    ) -> tuple[ConstraintOutcome, ...]:
        failed_ids = set()
        for issue in verification.get("issues", ()):
            if not isinstance(issue, Mapping) or issue.get("severity") != "error":
                continue
            metadata = issue.get("metadata")
            relation_index = metadata.get("relation_index") if isinstance(metadata, Mapping) else None
            if isinstance(relation_index, int) and 0 <= relation_index < len(state.relations):
                relation = state.relations[relation_index]
                failed_ids.add(str(relation.get("metadata", {}).get("source_parse_relation_id", relation.get("id", ""))))
        relation_ids = {item.id for item in graph.relations}
        materialized_ids = {
            item.get("metadata", {}).get("source_parse_relation_id", item.get("id"))
            for item in state.relations
        }
        return tuple(
            ConstraintOutcome(
                item.candidate_id,
                ConstraintStatus.VIOLATED,
                "verify",
                "independent_geometry_verifier_rejected_constraint",
            )
            if item.candidate_id in failed_ids
            else ConstraintOutcome(
                item.candidate_id, ConstraintStatus.UNSUPPORTED, "preserved_only",
                "relation_not_materialized_for_independent_verification",
            )
            if item.status is ConstraintStatus.SATISFIED and item.candidate_id in relation_ids - materialized_ids
            else item
            for item in outcomes
        )

    @classmethod
    def _materialize_state(
        cls,
        graph: ParseGraph,
        points: tuple[ParseObject, ...],
        positions: Mapping[str, tuple[float, float]],
        edges: Mapping[str, tuple[str, str]],
        *,
        satisfied_ids: set[str],
        position_mode: str,
        input_status: str,
        curves: Mapping[str, tuple[str, float]] | None = None,
    ) -> SkeletonBuildReport:
        objects = {item.id: item for item in graph.objects}
        curves = curves or {}
        point_names = graph.point_names()
        quantities_by_edge: dict[str, list[Any]] = {}
        for quantity in graph.quantities:
            if quantity.id not in satisfied_ids or quantity.kind != "length" or len(quantity.arguments) != 1:
                continue
            quantities_by_edge.setdefault(quantity.arguments[0].ref, []).append(quantity)

        trace: list[dict[str, Any]] = []
        for point in points:
            display_name = None if point.metadata.get("anonymous") else point.visible_name or point_names.get(point.id)
            metadata = {
                "source_parse_object_id": point.id,
                "coordinate_source": position_mode,
                "show_label": bool(display_name),
            }
            if display_name:
                metadata["display_name"] = display_name
            if point.metadata:
                metadata["parse_metadata"] = dict(point.metadata)
            trace.append({
                "op": "add_point",
                "id": point.id,
                "coord": list(positions[point.id]),
                "metadata": metadata,
            })

        for edge_id, (start, end) in edges.items():
            source = objects.get(edge_id)
            kind = source.kind if source is not None else ParseObjectKind.SEGMENT
            metadata: dict[str, Any] = {
                "source_parse_object_id": edge_id,
                "coordinate_source": position_mode,
            }
            if source is not None and source.metadata:
                metadata["parse_metadata"] = dict(source.metadata)
            display = graph.metadata.get("display_annotations", {})
            line_styles = display.get("line_styles", {}) if isinstance(display, Mapping) else {}
            if isinstance(line_styles, Mapping) and source is not None:
                line_style = source.metadata.get("line_style", line_styles.get(source.id, line_styles.get(source.visible_name)))
                if line_style in {"dashed", "solid"}:
                    metadata["line_style"] = line_style
            quantities = quantities_by_edge.get(edge_id, ())
            if len(quantities) == 1:
                metadata.update({
                    "show_length": cls._quantity_visible(quantities[0], graph),
                    "display_value": str(quantities[0].expression.to_sympy()),
                    "source_parse_quantity_id": quantities[0].id,
                })
            if kind is ParseObjectKind.LINE:
                trace.append({"op": "add_line", "id": edge_id, "points": [start, end], "metadata": metadata})
            elif kind is ParseObjectKind.RAY:
                trace.append({"op": "add_ray", "id": edge_id, "start": start, "through": end, "metadata": metadata})
            else:
                trace.append({"op": "add_segment", "id": edge_id, "points": [start, end], "metadata": metadata})

        for source in graph.objects:
            if source.kind is ParseObjectKind.CIRCLE:
                center = cls._reference(source, "center", fallback_kind=ParseObjectKind.POINT)
                radius = curves[source.id][1] if source.id in curves else cls._circle_radius(source)
                if center in positions and radius is not None:
                    trace.append({
                        "op": "add_circle",
                        "id": source.id,
                        "center": center,
                        "radius": radius,
                        "metadata": {
                            "source_parse_object_id": source.id,
                            "coordinate_source": position_mode,
                            "parse_metadata": dict(source.metadata),
                            "render": not any(
                                cls._reference(arc, "circle") == source.id
                                for arc in graph.objects
                                if arc.kind is ParseObjectKind.ARC
                            ),
                        },
                    })

        for source in graph.objects:
            if source.kind is ParseObjectKind.ARC:
                circle = cls._reference(source, "circle")
                start = cls._reference(source, "start", fallback_kind=ParseObjectKind.POINT)
                end = cls._reference(source, "end", fallback_kind=ParseObjectKind.POINT)
                curve = curves.get(source.id)
                if curve is None or start not in positions or end not in positions:
                    continue
                center, radius = curve
                if any(abs(math.dist(positions[center], positions[point]) - radius) > 1e-5 for point in (start, end)):
                    continue
                if circle is None:
                    circle = f"seed_circle:{source.id}"
                    trace.append({
                        "op": "add_circle", "id": circle, "center": center, "radius": radius,
                        "metadata": {"render": False, "source_parse_object_id": source.id},
                    })
                if (
                    circle is not None
                    and start in positions
                    and end in positions
                ):
                    metadata = dict(source.metadata)
                    start_vector = np.asarray(positions[start]) - positions[center]
                    end_vector = np.asarray(positions[end]) - positions[center]
                    cross = start_vector[0] * end_vector[1] - start_vector[1] * end_vector[0]
                    orientation = str(source.geometry.get("orientation", metadata.get("orientation", "ccw" if cross >= 0 else "cw")))
                    central_angle = metadata.get("central_angle")
                    operation = {
                        "op": "add_arc",
                        "id": source.id,
                        "circle": circle,
                        "start": start,
                        "end": end,
                        "orientation": orientation,
                        "metadata": {
                            "source_parse_object_id": source.id,
                            "coordinate_source": position_mode,
                            "parse_metadata": metadata,
                            "verification_mode": "numeric_witness",
                            "numeric_tolerance": 1e-5,
                        },
                    }
                    if central_angle is not None:
                        operation["central_angle"] = central_angle
                    trace.append(operation)

        trace.extend(cls._tangent_trace_operations(graph, positions, edges, curves, satisfied_ids))

        for polygon in graph.objects:
            if polygon.kind is not ParseObjectKind.POLYGON:
                continue
            vertices = [
                reference.ref for reference in polygon.references
                if reference.role.startswith("vertex") and reference.ref in positions
            ]
            if len(vertices) >= 3 and len(set(vertices)) == len(vertices):
                trace.append({
                    "op": "add_polygon",
                    "id": polygon.id,
                    "vertices": vertices,
                    "metadata": {
                        "source_parse_object_id": polygon.id,
                        "parse_metadata": dict(polygon.metadata),
                        "render_fill": polygon.metadata.get("shaded") is True,
                    },
                })

        for region in graph.regions:
            boundary = [reference.ref for reference in region.boundary]
            if not boundary or not all(item in objects for item in boundary):
                continue
            trace.append({
                "op": "add_region",
                "id": region.id,
                "boundary": boundary,
                "holes": list(region.holes),
                "kind": str(region.metadata.get("kind", "polygon_region")),
                "metadata": {
                    **dict(region.metadata),
                    "source_parse_region_id": region.id,
                    "coordinate_source": position_mode,
                    "render_fill": region.metadata.get("shaded") is True,
                },
            })

        for relation in graph.relations:
            if relation.id not in satisfied_ids:
                continue
            payload = cls._state_relation_payload(relation, objects)
            if payload is not None:
                trace.append({
                    "op": "add_relation",
                    "id": relation.id,
                    **payload,
                    "metadata": {
                        **dict(relation.metadata),
                        "source_parse_relation_id": relation.id,
                        "source_origin": relation.origin.value,
                        "verification_mode": "numeric_witness",
                        "numeric_tolerance": 1e-5,
                        "show_marker": relation.metadata.get("show_marker", relation.predicate != "parallel"),
                    },
                })
        return SkeletonSeedBuilder(graph, input_status).build(trace, allow_partial=True)

    @staticmethod
    def _build_state(
        trace: Any,
        graph: ParseGraph,
        input_status: str,
        *,
        allow_partial: bool = False,
    ) -> FormalGeometryState:
        return SkeletonSeedBuilder(graph, input_status).build(
            trace, allow_partial=allow_partial
        ).state

    @staticmethod
    def _state_relation_payload(relation: Any, objects: Mapping[str, ParseObject]) -> dict[str, Any] | None:
        refs = _relation_refs(relation)
        if relation.predicate in {"parallel", "perpendicular"} and len(refs) >= 2:
            return {"type": relation.predicate, "objects": refs[:2]}
        if relation.predicate == "collinear" and len(refs) >= 3:
            return {"type": "collinear", "points": refs}
        if relation.predicate == "equal_length" and len(refs) >= 2:
            return {"type": "equal_length", "segments": refs[:2]}
        if relation.predicate == "midpoint":
            edges = {
                item.id: tuple(reference.ref for reference in item.references if reference.role in {"start", "end"})
                for item in objects.values() if item.kind is ParseObjectKind.SEGMENT
            }
            midpoint = _midpoint_refs(relation, edges)
            if midpoint:
                return {"type": "midpoint", "point": midpoint[0], "of": list(midpoint[1:])}
            return None
        if relation.predicate == "point_on_circle" and len(refs) >= 2:
            roles = {argument.role: argument.ref for argument in relation.arguments}
            return {"type": "point_on_circle", "point": roles["point"], "circle": roles.get("circle", roles.get("support"))}
        if relation.predicate == "point_on_arc" and len(refs) >= 2:
            return {"type": "point_on_arc", "point": refs[0], "arc": refs[1]}
        if relation.predicate == "ordered_points":
            support = next((argument.ref for argument in relation.arguments if argument.role == "support"), None)
            points = [argument.ref for argument in relation.arguments if argument.role != "support"]
            if support is not None and len(points) >= 2:
                return {"type": "ordered_points", "support": support, "points": points}
            return None
        if relation.predicate in _POINT_ON_SUPPORT and len(refs) >= 2:
            point_id, support_id = refs[:2]
            support = objects.get(support_id)
            relation_type = relation.predicate
            if relation_type == "incidence":
                relation_type = {
                    ParseObjectKind.LINE: "point_on_line",
                    ParseObjectKind.RAY: "point_on_ray",
                    ParseObjectKind.SEGMENT: "point_on_segment",
                }.get(support.kind if support is not None else None, "point_on_line")
            key = {
                "point_on_line": "line",
                "point_on_ray": "line",
                "point_on_segment": "segment",
            }[relation_type]
            return {"type": relation_type, "point": point_id, key: support_id}
        return None

    @staticmethod
    def _reference(source: ParseObject, role: str, *, fallback_kind: ParseObjectKind | None = None) -> str | None:
        for reference in source.references:
            if reference.role == role:
                return reference.ref
        if fallback_kind is not None:
            for reference in source.references:
                if reference.role in {"object", "center", "start", "end"}:
                    return reference.ref
        return None

    @staticmethod
    def _circle_radius(source: ParseObject, scale: float = 1.0) -> float | str | dict[str, Any] | None:
        geometry = source.geometry
        value = geometry.get("exact_radius")
        if value is None and "radius_px" in geometry:
            value = float(geometry["radius_px"]) * scale
        if value is None:
            value = source.metadata.get("radius")
        if isinstance(value, Mapping) and "operator" in value:
            return MathExpression.from_dict(value).to_sympy()
        if isinstance(value, (int, float, str)):
            return value
        return None

    @staticmethod
    def _supported_proposal_point_ids(graph: ParseGraph) -> set[str]:
        objects = {item.id: item for item in graph.objects}
        if graph.metadata.get("extraction_contract") == "geometry_semantics_v1":
            referenced = {
                reference.ref for item in graph.objects for reference in item.references
            } | {
                argument.ref for relation in graph.relations for argument in relation.arguments
            } | {
                argument.ref for quantity in graph.quantities for argument in quantity.arguments
            } | {
                boundary.ref for region in graph.regions for boundary in region.boundary
            }
            return {point_id for point_id in referenced if point_id in objects and objects[point_id].kind is ParseObjectKind.POINT}
        source_text = "\n".join(
            item.raw_content for item in graph.evidence
            if isinstance(item.raw_content, str) and item.raw_content.strip()
        )
        supported = {
            item.id
            for item in graph.objects
            if item.kind is ParseObjectKind.POINT
            and (
                item.origin.value in {"native", "compiled"}
                or str(item.metadata.get("binding_status") or "").strip().lower()
                in {"bind", "bound", "confirmed"}
                or SeedRealizer._name_in_source_text(SeedRealizer._point_name(item, graph), source_text)
                or any(
                    evidence.id in item.evidence_refs
                    and evidence.precision.value in {"localized", "bound"}
                    for evidence in graph.evidence
                )
            )
        }
        for relation in graph.relations:
            source_type = str(relation.metadata.get("source_type") or "").strip().lower()
            if source_type not in {"text", "both"}:
                continue
            for argument in relation.arguments:
                item = objects.get(argument.ref)
                if item is None:
                    continue
                if item.kind is ParseObjectKind.POINT:
                    supported.add(item.id)
                elif item.kind in _LINE_KINDS | {ParseObjectKind.POLYGON}:
                    supported.update(
                        reference.ref
                        for reference in item.references
                        if objects.get(reference.ref) is not None
                        and objects[reference.ref].kind is ParseObjectKind.POINT
                    )
        return supported

    @staticmethod
    def _name_in_source_text(name: str, source_text: str) -> bool:
        if not name or not source_text:
            return False
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", source_text) is not None:
            return True
        if len(name) == 1 and name.isupper():
            identifiers = re.findall(r"(?<![A-Za-z0-9])([A-Z]{2,12})(?![A-Za-z0-9])", source_text)
            return any(name in identifier for identifier in identifiers)
        return False

    @staticmethod
    def _point_name(point: ParseObject, graph: ParseGraph) -> str:
        if point.visible_name:
            return point.visible_name.strip()
        binding_name = graph.point_names().get(point.id)
        if binding_name:
            return binding_name
        return point.id.rsplit(":", 1)[-1]

    @staticmethod
    def _line_edges(graph: ParseGraph, point_ids: set[str]) -> dict[str, tuple[str, str]]:
        edges: dict[str, tuple[str, str]] = {}
        for item in graph.objects:
            if item.kind is ParseObjectKind.POLYGON:
                vertices = [
                    reference.ref for reference in item.references
                    if reference.role.startswith("vertex_") and reference.ref in point_ids
                ]
                for index, start in enumerate(vertices):
                    end = vertices[(index + 1) % len(vertices)] if vertices else start
                    if start != end:
                        edges[f"{item.id}:edge:{index}"] = (start, end)
                continue
            if item.kind not in _LINE_KINDS:
                continue
            refs = [reference.ref for reference in item.references if reference.role in {"start", "end", "through"}]
            if len(refs) < 2:
                refs = [reference.ref for reference in item.references]
            if len(refs) >= 2 and refs[0] != refs[1] and refs[0] in point_ids and refs[1] in point_ids:
                edges[item.id] = (refs[0], refs[1])
        return edges

    @staticmethod
    def _constraints(graph: ParseGraph, edges: Mapping[str, tuple[str, str]]) -> tuple[Any, ...]:
        return tuple(
            relation for relation in graph.relations
            if relation.predicate in _POINT_ON_SUPPORT | {
                "ordered_points",
                "parallel",
                "perpendicular",
                "collinear",
                "equal_length",
                "midpoint",
            }
            and all(argument.ref in edges or argument.ref in {item.id for item in graph.objects} for argument in relation.arguments)
        )

    @staticmethod
    def _line_domains(
        graph: ParseGraph,
        edges: Mapping[str, tuple[str, str]],
    ) -> dict[str, str]:
        objects = {item.id: item for item in graph.objects}
        return {
            edge_id: (
                objects[edge_id].kind.value
                if edge_id in objects and objects[edge_id].kind in _LINE_KINDS
                else "segment"
            )
            for edge_id in edges
        }

    @staticmethod
    def _quantity_visible(quantity: Any, graph: ParseGraph) -> bool:
        if "show_label" in quantity.metadata:
            return quantity.metadata["show_label"] is True
        if quantity.origin.value in {"native", "image"}:
            return True
        annotations = graph.metadata.get("display_annotations", {})
        labels = annotations.get("labels", []) if isinstance(annotations, Mapping) else []
        if isinstance(labels, Mapping):
            labels = list(labels.values())
        values = [item.get("text", item.get("value", "")) if isinstance(item, Mapping) else item for item in labels]
        if isinstance(annotations, list):
            values.extend(item.get("value", item.get("text", "")) for item in annotations if isinstance(item, Mapping) and item.get("type") in {"text_label", "dimension_line"})
        expected = str(quantity.expression.to_sympy())
        pattern = rf"(?<![\w.]){re.escape(expected)}(?![\w.])"
        return any(re.search(pattern, str(value)) for value in values)

    @staticmethod
    def _structural_conflicts(graph: ParseGraph, edges: Mapping[str, tuple[str, str]]) -> list[dict[str, Any]]:
        supports = {edge_id: {point: [] for point in endpoints} for edge_id, endpoints in edges.items()}
        for relation in graph.relations:
            if relation.predicate in _POINT_ON_SUPPORT:
                point, support = _relation_refs(relation)
                if support in supports:
                    supports[support].setdefault(point, []).append(relation.id)
            elif relation.predicate == "collinear":
                points = {argument.ref for argument in relation.arguments}
                for edge_id, endpoints in edges.items():
                    if set(endpoints) <= points:
                        for point in points:
                            supports[edge_id].setdefault(point, []).append(relation.id)
        conflicts = []
        for relation in graph.relations:
            if relation.predicate != "perpendicular":
                continue
            left, right = _relation_refs(relation)
            if left not in edges or right not in edges:
                continue
            for support, members in supports.items():
                endpoints = set(edges[left]) | set(edges[right])
                if endpoints <= set(members):
                    conflicts.append({
                        "reason": "perpendicular_nonzero_edges_share_a_line",
                        "candidate_ids": sorted({relation.id, *(source for point in endpoints for source in members[point])}),
                        "objects": [left, right], "support": support,
                    })
                    break
        measurements: dict[tuple[str, str], list[tuple[str, float]]] = {}
        for quantity in graph.quantities:
            if quantity.kind not in {"length", "radius", "diameter"} or len(quantity.arguments) != 1:
                continue
            value = quantity.expression.to_sympy()
            if value.free_symbols or value.is_real is not True or value.is_positive is not True:
                continue
            kind = "radius" if quantity.kind == "diameter" else quantity.kind
            numeric = float(value) / (2 if quantity.kind == "diameter" else 1)
            key = (quantity.arguments[0].ref, kind)
            measurements.setdefault(key, []).append((quantity.id, numeric))
        for (object_id, kind), values in measurements.items():
            if max(value for _, value in values) - min(value for _, value in values) > 1e-5:
                conflicts.append({"reason": "incompatible_measurements", "candidate_ids": [item[0] for item in values], "object": object_id, "kind": kind})
        return conflicts

    @staticmethod
    def _layout_scale(
        graph: ParseGraph,
        positions: Mapping[str, tuple[float, float]],
        edges: Mapping[str, tuple[str, str]],
        numeric_targets: tuple[tuple[str, str, float], ...],
    ) -> float:
        ratios = []
        for _, edge_id, value in numeric_targets:
            start, end = edges[edge_id]
            observed = math.dist(positions[start], positions[end])
            if observed > 1e-9:
                ratios.append(value / observed)
        objects = {item.id: item for item in graph.objects}
        for quantity in graph.quantities:
            if quantity.kind not in {"radius", "diameter"} or len(quantity.arguments) != 1:
                continue
            source = objects.get(quantity.arguments[0].ref)
            value = quantity.expression.to_sympy()
            if source is not None and source.geometry.get("radius_px") and value.is_positive is True and not value.free_symbols:
                ratios.append(float(value) / (2 if quantity.kind == "diameter" else 1) / float(source.geometry["radius_px"]))
        return float(np.median(ratios)) if ratios else 1.0

    @classmethod
    def _curve_geometry(
        cls,
        graph: ParseGraph,
        positions: Mapping[str, tuple[float, float]],
        scale: float,
    ) -> dict[str, tuple[str, float]]:
        measured = {}
        for quantity in graph.quantities:
            if quantity.kind in {"radius", "diameter"} and len(quantity.arguments) == 1:
                value = quantity.expression.to_sympy()
                if value.is_positive is True and not value.free_symbols:
                    measured[quantity.arguments[0].ref] = float(value) / (2 if quantity.kind == "diameter" else 1)
        curves = {}
        for kind in (ParseObjectKind.CIRCLE, ParseObjectKind.ARC):
            for source in graph.objects:
                if source.kind is not kind:
                    continue
                circle_id = cls._reference(source, "circle")
                if circle_id in curves:
                    curves[source.id] = curves[circle_id]
                    continue
                center = cls._reference(source, "center")
                if center not in positions:
                    continue
                radius = measured.get(source.id, cls._circle_radius(source, scale))
                if radius is None and kind is ParseObjectKind.ARC:
                    endpoints = [cls._reference(source, role) for role in ("start", "end")]
                    if all(point in positions for point in endpoints):
                        radius = sum(math.dist(positions[center], positions[point]) for point in endpoints) / 2
                if radius is not None and math.isfinite(float(radius)) and float(radius) > 0:
                    curves[source.id] = (center, float(radius))
        return curves

    @classmethod
    def _complete_tangent_circles(
        cls,
        graph: ParseGraph,
        positions: Mapping[str, tuple[float, float]],
        edges: Mapping[str, tuple[str, str]],
        curves: Mapping[str, tuple[str, float]],
    ) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[str, float]]]:
        """Complete an unknown circle tangent to two lines and a known circle.

        This is the common quarter-circle seed: the unknown center is constrained
        by two tangent carriers and its radius by internal tangency to a known
        support circle. The semantic graph remains coordinate-free; coordinates
        are only created in the candidate seed instance.
        """
        output_positions = dict(positions)
        output_curves = dict(curves)
        objects = {item.id: item for item in graph.objects}
        tangent_relations = [item for item in graph.relations if item.predicate == "tangent"]
        circle_ids = {item.id for item in graph.objects if item.kind is ParseObjectKind.CIRCLE}
        for inner in sorted(circle_ids):
            center_id = cls._reference(objects[inner], "center")
            if center_id is None or center_id not in output_positions:
                continue
            if inner in output_curves and cls._circle_radius(objects[inner]) is not None:
                continue
            line_ids = []
            outer_id = None
            for relation in tangent_relations:
                roles = {argument.role: argument.ref for argument in relation.arguments}
                left, right = roles.get("left"), roles.get("right")
                if left == inner and right in edges:
                    line_ids.append(right)
                elif right == inner and left in edges:
                    line_ids.append(left)
                elif left == inner and right in circle_ids:
                    outer_id = right
                elif right == inner and left in circle_ids:
                    outer_id = left
            line_ids = list(dict.fromkeys(line_ids))
            if len(line_ids) < 2 or outer_id not in output_curves:
                continue
            first, second = (edges[line_id] for line_id in line_ids[:2])
            intersection = cls._carrier_intersection(output_positions, first, second)
            if intersection is None:
                continue
            outer_center_id, outer_radius = output_curves[outer_id]
            if outer_center_id not in output_positions:
                continue
            initial_center = np.asarray(output_positions[center_id], dtype=float)
            origin = np.asarray(intersection, dtype=float)
            outer_center = np.asarray(output_positions[outer_center_id], dtype=float)
            directions = []
            ray_directions = []
            for start, end in (first, second):
                start_xy = np.asarray(output_positions[start], dtype=float)
                end_xy = np.asarray(output_positions[end], dtype=float)
                vector = end_xy - start_xy
                length = float(np.linalg.norm(vector))
                if length <= 1e-9:
                    directions = []
                    break
                normal = np.asarray([-vector[1], vector[0]], dtype=float) / length
                directions.append(normal)
                ray = end_xy - origin if np.linalg.norm(end_xy - origin) >= np.linalg.norm(start_xy - origin) else start_xy - origin
                ray_length = float(np.linalg.norm(ray))
                ray_directions.append(ray / ray_length if ray_length > 1e-9 else None)
            if len(directions) != 2:
                continue
            candidates = []
            for sign_first in (-1.0, 1.0):
                for sign_second in (-1.0, 1.0):
                    normal_first = sign_first * directions[0]
                    normal_second = sign_second * directions[1]
                    denominator = 1.0 + float(np.dot(normal_first, normal_second))
                    if denominator <= 1e-9:
                        continue
                    bisector = (normal_first + normal_second) / denominator
                    offset = origin - outer_center
                    a = float(np.dot(bisector, bisector) - 1.0)
                    b = float(2.0 * np.dot(offset, bisector) + 2.0 * float(outer_radius))
                    c = float(np.dot(offset, offset) - float(outer_radius) ** 2)
                    roots = np.roots([a, b, c]) if abs(a) > 1e-9 else np.roots([b, c])
                    for root in roots:
                        if abs(float(np.imag(root))) > 1e-7:
                            continue
                        radius = float(np.real(root))
                        candidate_center = origin + radius * bisector
                        if radius <= 1e-7 or radius >= float(outer_radius):
                            continue
                        if any(
                            ray is None or float(np.dot(candidate_center - origin, ray)) <= 1e-7
                            for ray in ray_directions
                        ):
                            continue
                        residual = abs(float(np.linalg.norm(candidate_center - outer_center)) + radius - float(outer_radius))
                        if residual > 1e-4:
                            continue
                        score = float(np.linalg.norm(candidate_center - initial_center))
                        candidates.append((score, candidate_center, radius))
            if not candidates:
                continue
            _, candidate_center, radius = min(candidates, key=lambda item: item[0])
            output_positions[center_id] = (float(candidate_center[0]), float(candidate_center[1]))
            output_curves[inner] = (center_id, radius)
        return output_positions, output_curves

    @staticmethod
    def _carrier_intersection(
        positions: Mapping[str, tuple[float, float]],
        first: tuple[str, str],
        second: tuple[str, str],
    ) -> tuple[float, float] | None:
        p = np.asarray(positions[first[0]], dtype=float)
        r = np.asarray(positions[first[1]], dtype=float) - p
        q = np.asarray(positions[second[0]], dtype=float)
        s = np.asarray(positions[second[1]], dtype=float) - q
        denominator = float(r[0] * s[1] - r[1] * s[0])
        if abs(denominator) <= 1e-9:
            return None
        t = float((q[0] - p[0]) * s[1] - (q[1] - p[1]) * s[0]) / denominator
        point = p + t * r
        return float(point[0]), float(point[1])

    @classmethod
    def _tangent_relation_holds(
        cls,
        relation: Any,
        positions: Mapping[str, tuple[float, float]],
        edges: Mapping[str, tuple[str, str]],
        curves: Mapping[str, tuple[str, float]],
    ) -> bool:
        roles = {argument.role: argument.ref for argument in relation.arguments}
        left, right = roles.get("left"), roles.get("right")
        if left in curves and right in edges:
            circle_id, edge_id = left, right
        elif right in curves and left in edges:
            circle_id, edge_id = right, left
        elif left in curves and right in curves:
            first_center, first_radius = curves[left]
            second_center, second_radius = curves[right]
            if first_center not in positions or second_center not in positions:
                return False
            distance = math.dist(positions[first_center], positions[second_center])
            return min(
                abs(distance - first_radius - second_radius),
                abs(distance - abs(first_radius - second_radius)),
            ) <= 1e-5
        else:
            return False
        center, radius = curves[circle_id]
        if center not in positions:
            return False
        start, end = edges[edge_id]
        if start not in positions or end not in positions:
            return False
        start_xy, end_xy = np.asarray(positions[start]), np.asarray(positions[end])
        vector = end_xy - start_xy
        length = float(np.linalg.norm(vector))
        if length <= 1e-9:
            return False
        offset = np.asarray(positions[center]) - start_xy
        distance = abs(float(vector[0] * offset[1] - vector[1] * offset[0])) / length
        projection = float(np.dot(np.asarray(positions[center]) - start_xy, vector) / np.dot(vector, vector))
        return abs(distance - radius) <= 1e-5 and -1e-5 <= projection <= 1.0 + 1e-5

    @classmethod
    def _tangent_trace_operations(
        cls,
        graph: ParseGraph,
        positions: Mapping[str, tuple[float, float]],
        edges: Mapping[str, tuple[str, str]],
        curves: Mapping[str, tuple[str, float]],
        satisfied_ids: set[str],
    ) -> list[dict[str, Any]]:
        operations: list[dict[str, Any]] = []
        for relation in graph.relations:
            if relation.predicate != "tangent" or relation.id not in satisfied_ids:
                continue
            roles = {argument.role: argument.ref for argument in relation.arguments}
            left, right = roles.get("left"), roles.get("right")
            if left in curves and right in curves:
                operations.append({
                    "op": "add_relation",
                    "id": f"state_{relation.id}",
                    "type": "tangent_to_circle",
                    "circles": [left, right],
                    "metadata": {
                        "source_parse_relation_id": relation.id,
                        "verification_mode": "numeric_witness",
                        "numeric_tolerance": 1e-5,
                    },
                })
                continue
            circle_id, edge_id = (left, right) if left in curves and right in edges else (right, left)
            if circle_id not in curves or edge_id not in edges:
                continue
            center_id, _ = curves[circle_id]
            start, end = edges[edge_id]
            if center_id not in positions or start not in positions or end not in positions:
                continue
            start_xy, end_xy = np.asarray(positions[start]), np.asarray(positions[end])
            vector = end_xy - start_xy
            denominator = float(np.dot(vector, vector))
            if denominator <= 1e-9:
                continue
            fraction = float(np.dot(np.asarray(positions[center_id]) - start_xy, vector) / denominator)
            contact = start_xy + fraction * vector
            contact_id = f"tangent_point_{relation.id}"
            operations.extend([
                {
                    "op": "add_point",
                    "id": contact_id,
                    "coord": [float(contact[0]), float(contact[1])],
                    "metadata": {"show_label": False, "source_parse_relation_id": relation.id},
                },
                {
                    "op": "add_relation",
                    "id": f"state_{relation.id}",
                    "type": "tangent",
                    "point": contact_id,
                    "line": [start, end],
                    "circle": circle_id,
                    "metadata": {
                        "source_parse_relation_id": relation.id,
                        "verification_mode": "numeric_witness",
                        "numeric_tolerance": 1e-5,
                    },
                },
            ])
        return operations

    @classmethod
    def _radial_constraints(
        cls,
        graph: ParseGraph,
        curves: Mapping[str, tuple[str, float]],
    ) -> tuple[tuple[str, str, float], ...]:
        constraints = []
        for source in graph.objects:
            if source.kind is ParseObjectKind.ARC and source.id in curves:
                center, radius = curves[source.id]
                for role in ("start", "end"):
                    point = cls._reference(source, role)
                    if point is not None:
                        constraints.append((center, point, radius))
        for relation in graph.relations:
            if relation.predicate == "point_on_circle":
                roles = {argument.role: argument.ref for argument in relation.arguments}
                curve = curves.get(roles.get("circle", roles.get("support")))
                if curve is not None and roles.get("point") is not None:
                    constraints.append((curve[0], roles["point"], curve[1]))
        return tuple(dict.fromkeys(constraints))

    @staticmethod
    def _observed_positions(graph: ParseGraph, points: tuple[ParseObject, ...]) -> dict[str, tuple[float, float]]:
        image_size = graph.metadata.get("image_size")
        height = float(image_size[1]) if isinstance(image_size, (list, tuple)) and len(image_size) == 2 else 0.0
        positions: dict[str, tuple[float, float]] = {}
        for point in points:
            pixel = point.geometry.get("pixel")
            if isinstance(pixel, (list, tuple)) and len(pixel) == 2:
                try:
                    x, y = float(pixel[0]), float(pixel[1])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(x) and math.isfinite(y):
                    positions[point.id] = (x, height - y if height else -y)
        return positions

    @staticmethod
    def _semantic_scaffold_positions(points: tuple[ParseObject, ...]) -> dict[str, tuple[float, float]]:
        positions: dict[str, tuple[float, float]] = {}
        for point in points:
            value = point.metadata.get("seed_normalized")
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                continue
            try:
                x, y = float(value[0]), float(value[1])
            except (TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y) and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                positions[point.id] = (x, 1.0 - y)
        return positions

    @staticmethod
    def _support_edges_by_point(
        constraints: tuple[Any, ...],
        edges: Mapping[str, tuple[str, str]],
    ) -> dict[str, list[str]]:
        output: dict[str, list[str]] = {}
        for relation in constraints:
            if relation.predicate in _POINT_ON_SUPPORT_LAYOUT and len(relation.arguments) >= 2:
                refs = _relation_refs(relation)
                output.setdefault(refs[0], []).append(refs[1])
                continue
            if relation.predicate == "ordered_points":
                support = next(
                    (argument.ref for argument in relation.arguments if argument.role == "support"),
                    None,
                )
                if support in edges:
                    for argument in relation.arguments:
                        if argument.role != "support" and argument.ref not in edges[support]:
                            output.setdefault(argument.ref, []).append(support)
                continue
            if relation.predicate != "collinear" or len(relation.arguments) < 3:
                continue
            point_ids = [argument.ref for argument in relation.arguments]
            point_set = set(point_ids)
            for support_id, (start, end) in edges.items():
                if start not in point_set or end not in point_set:
                    continue
                for point_id in point_ids:
                    if point_id not in {start, end}:
                        output.setdefault(point_id, []).append(support_id)
        return output

    @staticmethod
    def _numeric_length_targets(
        graph: ParseGraph,
        edges: Mapping[str, tuple[str, str]],
    ) -> tuple[tuple[str, str, float], ...]:
        targets: list[tuple[str, str, float]] = []
        for quantity in graph.quantities:
            if quantity.kind not in {"length", "unknown"} or len(quantity.arguments) != 1:
                continue
            edge_id = quantity.arguments[0].ref
            if edge_id not in edges:
                continue
            expression = quantity.expression.to_sympy()
            if expression.free_symbols or expression.is_real is False or expression.is_finite is False:
                continue
            value = float(expression)
            if math.isfinite(value) and value > 0.0:
                targets.append((quantity.id, edge_id, value))
        return tuple(targets)

    @staticmethod
    def _refine_numeric_witness(
        positions: Mapping[str, tuple[float, float]],
        edges: Mapping[str, tuple[str, str]],
        domains: Mapping[str, str],
        constraints: tuple[Any, ...],
        numeric_targets: tuple[tuple[str, str, float], ...],
        radial_constraints: tuple[tuple[str, str, float], ...] = (),
        *,
        anchor_weight: float = 1e-6,
    ) -> dict[str, tuple[float, float]]:
        point_ids = tuple(sorted(positions))
        index = {point_id: offset for offset, point_id in enumerate(point_ids)}
        values = np.asarray([positions[point_id] for point_id in point_ids], dtype=float)
        anchor = values.copy()
        reference_scale = max(
            (target for _, _, target in numeric_targets),
            default=max(float(np.linalg.norm(np.ptp(values, axis=0))), 1e-6),
        )

        def vector(flat: np.ndarray, edge_id: str) -> np.ndarray | None:
            edge = edges.get(edge_id)
            if edge is None or edge[0] not in index or edge[1] not in index:
                return None
            coords = flat.reshape((-1, 2))
            return coords[index[edge[1]]] - coords[index[edge[0]]]

        def residuals(flat: np.ndarray) -> np.ndarray:
            coords = flat.reshape((-1, 2))
            residual = list(((coords - anchor) / reference_scale * anchor_weight).ravel())
            for start, end in sorted(set(edges.values())):
                distance = float(np.linalg.norm(coords[index[end]] - coords[index[start]]))
                residual.append(min(distance / reference_scale - 1e-4, 0.0) * 8.0)
            for _, edge_id, target in numeric_targets:
                direction = vector(flat, edge_id)
                if direction is not None:
                    residual.append((float(np.linalg.norm(direction)) - target) / target * 8.0)
            for center_id, point_id, radius in radial_constraints:
                if center_id in index and point_id in index:
                    distance = float(np.linalg.norm(coords[index[point_id]] - coords[index[center_id]]))
                    residual.append((distance - radius) / radius * 8.0)
            for relation in constraints:
                refs = _relation_refs(relation)
                if relation.predicate in {"parallel", "perpendicular"} and len(refs) >= 2:
                    left, right = vector(flat, refs[0]), vector(flat, refs[1])
                    if left is None or right is None:
                        continue
                    scale = max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-9)
                    value = (
                        left[0] * right[1] - left[1] * right[0]
                        if relation.predicate == "parallel"
                        else float(np.dot(left, right))
                    )
                    residual.append(float(value) / scale * 8.0)
                elif relation.predicate == "collinear" and len(refs) >= 3 and all(ref in index for ref in refs):
                    first, second = (coords[index[ref]] for ref in refs[:2])
                    direction = second - first
                    scale = max(float(np.linalg.norm(direction)), 1e-9)
                    for ref in refs[2:]:
                        offset = coords[index[ref]] - first
                        residual.append(float(direction[0] * offset[1] - direction[1] * offset[0]) / scale / reference_scale * 8.0)
                elif relation.predicate == "equal_length" and len(refs) >= 2:
                    left, right = vector(flat, refs[0]), vector(flat, refs[1])
                    if left is not None and right is not None:
                        residual.append(float(np.linalg.norm(left) - np.linalg.norm(right)) / reference_scale * 8.0)
                elif relation.predicate == "midpoint":
                    midpoint_refs = _midpoint_refs(relation, edges)
                    if midpoint_refs and all(ref in index for ref in midpoint_refs):
                        point_id, start_id, end_id = midpoint_refs
                        midpoint = (coords[index[start_id]] + coords[index[end_id]]) / 2
                        residual.extend((coords[index[point_id]] - midpoint) / reference_scale * 8.0)
                elif relation.predicate in _POINT_ON_SUPPORT and len(refs) >= 2 and refs[0] in index and refs[1] in edges:
                    start_id, end_id = edges[refs[1]]
                    if start_id not in index or end_id not in index:
                        continue
                    point = coords[index[refs[0]]]
                    start, end = coords[index[start_id]], coords[index[end_id]]
                    direction = end - start
                    scale = max(float(np.dot(direction, direction)), 1e-9)
                    fraction = float(np.dot(point - start, direction) / scale)
                    cross = float((point[0] - start[0]) * direction[1] - (point[1] - start[1]) * direction[0])
                    residual.append(cross / math.sqrt(scale) / reference_scale * 8.0)
                    domain = domains.get(refs[1], "segment") if relation.predicate == "incidence" else {
                        "point_on_line": "line",
                        "point_on_ray": "ray",
                        "point_on_segment": "segment",
                    }.get(relation.predicate, "segment")
                    if domain == "segment":
                        residual.append((min(fraction, 0.0) + max(fraction - 1.0, 0.0)) * 8.0)
                    elif domain == "ray":
                        residual.append(min(fraction, 0.0) * 8.0)
            return np.asarray(residual, dtype=float)

        flat = values.ravel()
        damping = 1e-3
        for _ in range(48):
            current = residuals(flat)
            loss = float(np.dot(current, current))
            epsilon = max(reference_scale, 1.0) * 1e-6
            jacobian = np.empty((len(current), len(flat)), dtype=float)
            for column in range(len(flat)):
                shifted = flat.copy()
                shifted[column] += epsilon
                jacobian[:, column] = (residuals(shifted) - current) / epsilon
            system = jacobian.T @ jacobian + damping * np.eye(len(flat))
            gradient = jacobian.T @ current
            try:
                delta = np.linalg.solve(system, -gradient)
            except np.linalg.LinAlgError:
                delta, *_ = np.linalg.lstsq(system, -gradient, rcond=None)
            candidate = flat + delta
            candidate_residual = residuals(candidate)
            if float(np.dot(candidate_residual, candidate_residual)) < loss:
                flat = candidate
                damping = max(damping * 0.5, 1e-8)
                if float(np.linalg.norm(delta)) <= 1e-8:
                    break
            else:
                damping = min(damping * 10.0, 1e8)
        solved = flat.reshape((-1, 2))
        return {point_id: tuple(float(value) for value in solved[index[point_id]]) for point_id in point_ids}

    def _seed_positions(
        self,
        graph: ParseGraph,
        points: tuple[ParseObject, ...],
        edges: Mapping[str, tuple[str, str]],
        constraints: tuple[Any, ...],
    ) -> dict[str, tuple[float, float]]:
        positions: dict[str, tuple[float, float]] = {}
        topology_positions = False
        parallel_pairs = [relation for relation in constraints if relation.predicate == "parallel" and len(relation.arguments) >= 2]
        support_by_point = self._support_edges_by_point(constraints, edges)

        for relation in parallel_pairs:
            inner_id, base_id = (argument.ref for argument in relation.arguments[:2])
            inner = edges.get(inner_id)
            base = edges.get(base_id)
            if inner is None or base is None:
                continue
            base_start, base_end = base
            base_points = {base_start, base_end}
            support_edges = [edges.get(support) for point in inner for support in support_by_point.get(point, ())]
            support_edges = [edge for edge in support_edges if edge is not None]
            apex_candidates = [
                set(edge) - base_points for edge in support_edges
                if len(set(edge) & base_points) == 1 and len(set(edge) - base_points) == 1
            ]
            apex_values = {next(iter(value)) for value in apex_candidates}
            if len(apex_values) != 1:
                continue
            apex = next(iter(apex_values))
            base_for_inner: dict[str, str] = {}
            for point in inner:
                for support_id in support_by_point.get(point, ()):
                    edge = edges.get(support_id)
                    if edge is None:
                        continue
                    other = set(edge) - {apex, point}
                    candidates = other & base_points
                    if len(candidates) == 1:
                        base_for_inner[point] = next(iter(candidates))
            if set(base_for_inner) != set(inner):
                continue
            positions[base_start] = (0.0, 0.0)
            positions[base_end] = (7.0, 0.0)
            positions[apex] = (3.5, 5.0)
            for point, base_point in base_for_inner.items():
                apex_xy = positions[apex]
                base_xy = positions[base_point]
                positions[point] = self._lerp(apex_xy, base_xy, self.point_fraction)
            break

        if not positions:
            positions = self._topology_seed_positions(graph, points, edges)
            topology_positions = bool(positions) and graph.metadata.get("extraction_contract") == "geometry_semantics_v1"

        if not positions:
            positions = self._direction_seed_positions(points, edges, constraints)

        if not positions:
            names = sorted(self._point_name(point, graph) for point in points)
            radius = max(2.0, len(names) / 2.0)
            for index, point in enumerate(sorted(points, key=lambda item: self._point_name(item, graph))):
                angle = math.pi / 2.0 + 2.0 * math.pi * index / max(1, len(points))
                positions[point.id] = (radius * math.cos(angle), radius * math.sin(angle))

        point_ids = {point.id for point in points}
        for index, point in enumerate(sorted(points, key=lambda item: self._point_name(item, graph))):
            if point.id not in positions:
                positions[point.id] = (float(index), 1.0 if index % 2 else -1.0)
        positions = self._place_incident_points(positions, support_by_point, edges)
        positions = self._place_ordered_points(
            positions,
            constraints,
            edges,
            preserve_existing=topology_positions,
        )
        return {key: value for key, value in positions.items() if key in point_ids}

    @classmethod
    def _topology_seed_positions(
        cls,
        graph: ParseGraph,
        points: tuple[ParseObject, ...],
        edges: Mapping[str, tuple[str, str]],
    ) -> dict[str, tuple[float, float]]:
        """Choose a readable deterministic layout from topology alone."""
        point_ids = {point.id for point in points}
        adjacency: dict[str, set[str]] = {point_id: set() for point_id in point_ids}
        for start, end in edges.values():
            if start in point_ids and end in point_ids:
                adjacency[start].add(end)
                adjacency[end].add(start)

        cycles: list[tuple[str, ...]] = []
        for item in graph.objects:
            if item.kind is ParseObjectKind.POLYGON:
                vertices = tuple(
                    reference.ref
                    for reference in item.references
                    if reference.role.startswith("vertex_") and reference.ref in point_ids
                )
                if len(vertices) >= 3:
                    cycles.append(vertices)
        for region in graph.regions:
            boundary_edges = [boundary.ref for boundary in region.boundary]
            boundary_vertices = cls._ordered_boundary_vertices(boundary_edges, edges)
            if len(boundary_vertices) >= 3:
                cycles.append(boundary_vertices)
        if not cycles:
            inferred = cls._infer_cycle(adjacency)
            if inferred:
                cycles.append(inferred)

        if cycles:
            cycle = max(cycles, key=len)
            positions: dict[str, tuple[float, float]] = {}
            horizontal_radius = max(3.5, len(cycle) * 0.9)
            vertical_radius = max(2.8, horizontal_radius * 0.68)
            for index, point_id in enumerate(cycle):
                angle = math.pi / 2.0 - 2.0 * math.pi * index / len(cycle)
                positions[point_id] = (
                    horizontal_radius * math.cos(angle),
                    vertical_radius * math.sin(angle),
                )
            cls._place_remaining_topology_points(positions, adjacency, point_ids)
            return positions

        if not any(adjacency.values()):
            return {}
        point_order = sorted(point_ids)
        positions = {}
        radius = max(3.0, len(point_order) * 0.55)
        for index, point_id in enumerate(point_order):
            angle = math.pi / 2.0 - 2.0 * math.pi * index / max(1, len(point_order))
            positions[point_id] = (radius * math.cos(angle), radius * math.sin(angle))
        for _ in range(48):
            forces = {point_id: np.zeros(2, dtype=float) for point_id in point_order}
            for left_index, left_id in enumerate(point_order):
                left = np.asarray(positions[left_id], dtype=float)
                for right_id in point_order[left_index + 1:]:
                    right = np.asarray(positions[right_id], dtype=float)
                    delta = left - right
                    distance = max(float(np.linalg.norm(delta)), 1e-6)
                    repulsion = delta / distance * (1.4 / distance)
                    forces[left_id] += repulsion
                    forces[right_id] -= repulsion
            for start, end in edges.values():
                if start not in positions or end not in positions:
                    continue
                delta = np.asarray(positions[end]) - np.asarray(positions[start])
                distance = max(float(np.linalg.norm(delta)), 1e-6)
                correction = delta / distance * (distance - 4.0) * 0.12
                forces[start] += correction
                forces[end] -= correction
            for point_id in point_order:
                positions[point_id] = tuple(
                    np.asarray(positions[point_id]) + np.clip(forces[point_id], -0.45, 0.45)
                )
        return positions

    @staticmethod
    def _ordered_boundary_vertices(
        boundary_edges: list[str],
        edges: Mapping[str, tuple[str, str]],
    ) -> tuple[str, ...]:
        resolved = [edges[edge_id] for edge_id in boundary_edges if edge_id in edges]
        if len(resolved) < 3:
            return ()
        vertices = [resolved[0][0], resolved[0][1]]
        for start, end in resolved[1:]:
            if start == vertices[-1]:
                vertices.append(end)
            elif end == vertices[-1]:
                vertices.append(start)
            else:
                return ()
        if vertices[-1] == vertices[0]:
            vertices.pop()
        return tuple(dict.fromkeys(vertices)) if len(vertices) >= 3 else ()

    @staticmethod
    def _infer_cycle(adjacency: Mapping[str, set[str]]) -> tuple[str, ...]:
        for start in sorted(adjacency):
            path = [start]
            previous = None
            current = start
            for _ in range(len(adjacency)):
                choices = sorted(neighbor for neighbor in adjacency[current] if neighbor != previous)
                next_point = next((candidate for candidate in choices if candidate == start or candidate not in path), None)
                if next_point is None:
                    break
                if next_point == start:
                    return tuple(path) if len(path) >= 3 else ()
                path.append(next_point)
                previous, current = current, next_point
        return ()

    @staticmethod
    def _place_remaining_topology_points(
        positions: dict[str, tuple[float, float]],
        adjacency: Mapping[str, set[str]],
        point_ids: set[str],
    ) -> None:
        for _ in range(len(point_ids)):
            changed = False
            for point_id in sorted(point_ids - set(positions)):
                neighbors = [positions[item] for item in adjacency.get(point_id, ()) if item in positions]
                if not neighbors:
                    continue
                center = np.mean(np.asarray(neighbors, dtype=float), axis=0)
                positions[point_id] = (float(center[0]), float(center[1] + 0.8))
                changed = True
            if not changed:
                break

    @classmethod
    def _place_incident_points(
        cls,
        positions: dict[str, tuple[float, float]],
        support_by_point: Mapping[str, list[str]],
        edges: Mapping[str, tuple[str, str]],
    ) -> dict[str, tuple[float, float]]:
        output = dict(positions)
        for point_id, support_ids in sorted(support_by_point.items()):
            supports = [edges[support_id] for support_id in support_ids if support_id in edges]
            supported_lines = [
                (output[start], output[end])
                for start, end in supports
                if start in output and end in output and start != point_id and end != point_id
            ]
            intersection = next(
                (
                    candidate
                    for index, left in enumerate(supported_lines)
                    for right in supported_lines[index + 1 :]
                    if (candidate := cls._line_intersection(left, right)) is not None
                    and cls._point_within_segment(candidate, left)
                    and cls._point_within_segment(candidate, right)
                ),
                None,
            )
            if point_id in output:
                if intersection is not None and len(supported_lines) >= 2:
                    output[point_id] = intersection
                continue
            if intersection is not None:
                output[point_id] = intersection
            elif supported_lines:
                start, end = supported_lines[0]
                output[point_id] = cls._lerp(start, end, 0.5)
        return output

    @classmethod
    def _place_ordered_points(
        cls,
        positions: dict[str, tuple[float, float]],
        constraints: tuple[Any, ...],
        edges: Mapping[str, tuple[str, str]],
        *,
        preserve_existing: bool = False,
    ) -> dict[str, tuple[float, float]]:
        output = dict(positions)
        for relation in constraints:
            if relation.predicate != "ordered_points":
                continue
            support_id = next(
                (argument.ref for argument in relation.arguments if argument.role == "support"),
                None,
            )
            point_ids = [argument.ref for argument in relation.arguments if argument.role != "support"]
            support = edges.get(support_id or "")
            if support is None or len(point_ids) < 2 or any(point_id not in output for point_id in support):
                continue
            start, end = output[support[0]], output[support[1]]
            includes_endpoints = point_ids[0] == support[0] and point_ids[-1] == support[1]
            reverse_endpoints = point_ids[0] == support[1] and point_ids[-1] == support[0]
            ordered = list(reversed(point_ids)) if reverse_endpoints else point_ids
            if includes_endpoints or reverse_endpoints:
                fractions = [index / (len(ordered) - 1) for index in range(len(ordered))]
            else:
                fractions = [(index + 1) / (len(ordered) + 1) for index in range(len(ordered))]
            for point_id, fraction in zip(ordered, fractions):
                if not preserve_existing or point_id not in output:
                    output[point_id] = cls._lerp(start, end, fraction)
        return output

    @staticmethod
    def _line_intersection(
        left: tuple[tuple[float, float], tuple[float, float]],
        right: tuple[tuple[float, float], tuple[float, float]],
    ) -> tuple[float, float] | None:
        (ax, ay), (bx, by) = left
        (cx, cy), (dx, dy) = right
        denominator = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx)
        if abs(denominator) <= 1e-9:
            return None
        fraction = ((cx - ax) * (dy - cy) - (cy - ay) * (dx - cx)) / denominator
        return ax + fraction * (bx - ax), ay + fraction * (by - ay)

    @staticmethod
    def _point_within_segment(
        point: tuple[float, float],
        segment: tuple[tuple[float, float], tuple[float, float]],
    ) -> bool:
        start, end = segment
        return (
            min(start[0], end[0]) - 1e-6 <= point[0] <= max(start[0], end[0]) + 1e-6
            and min(start[1], end[1]) - 1e-6 <= point[1] <= max(start[1], end[1]) + 1e-6
        )

    @staticmethod
    def _direction_seed_positions(
        points: tuple[ParseObject, ...],
        edges: Mapping[str, tuple[str, str]],
        constraints: tuple[Any, ...],
    ) -> dict[str, tuple[float, float]]:
        directional = [
            relation for relation in constraints
            if relation.predicate in {"parallel", "perpendicular"}
        ]
        if not directional:
            return {}
        for relation in directional:
            refs = [argument.ref for argument in relation.arguments[:2]]
            if (
                relation.predicate == "parallel"
                and len(refs) == 2
                and refs[0] in edges
                and refs[1] in edges
                and set(edges[refs[0]]) & set(edges[refs[1]])
            ):
                return {}

        edge_angles: dict[str, float] = {}
        first_edge = next(
            (
                argument.ref
                for relation in directional
                for argument in relation.arguments
                if argument.ref in edges
            ),
            None,
        )
        if first_edge is None:
            return {}
        edge_angles[first_edge] = 0.0
        changed = True
        while changed:
            changed = False
            for relation in directional:
                refs = [argument.ref for argument in relation.arguments[:2]]
                if len(refs) < 2 or any(ref not in edges for ref in refs):
                    continue
                left, right = refs
                offset = math.pi / 2.0 if relation.predicate == "perpendicular" else 0.0
                if left in edge_angles and right not in edge_angles:
                    edge_angles[right] = edge_angles[left] + offset
                    changed = True
                elif right in edge_angles and left not in edge_angles:
                    edge_angles[left] = edge_angles[right] - offset
                    changed = True

        for edge_id, edge in edges.items():
            if edge_id in edge_angles:
                continue
            neighbors = [
                known_id for known_id, known_edge in edges.items()
                if known_id in edge_angles and set(edge) & set(known_edge)
            ]
            edge_angles[edge_id] = (
                edge_angles[neighbors[0]] + math.pi / 3.0
                if neighbors else math.pi / 3.0 * (len(edge_angles) % 3)
            )

        positions: dict[str, tuple[float, float]] = {}
        ordered_edges = sorted(edges, key=lambda edge_id: (edge_id != first_edge, edge_id))
        for _ in range(max(1, len(ordered_edges))):
            changed = False
            for edge_id in ordered_edges:
                start, end = edges[edge_id]
                angle = edge_angles[edge_id]
                vector = 4.0 * math.cos(angle), 4.0 * math.sin(angle)
                if start not in positions and end not in positions:
                    if positions:
                        continue
                    positions[start] = (0.0, 0.0)
                    positions[end] = vector
                    changed = True
                elif start in positions and end not in positions:
                    positions[end] = positions[start][0] + vector[0], positions[start][1] + vector[1]
                    changed = True
                elif end in positions and start not in positions:
                    positions[start] = positions[end][0] - vector[0], positions[end][1] - vector[1]
                    changed = True
            if not changed:
                break

        point_ids = {point.id for point in points}
        return {point_id: position for point_id, position in positions.items() if point_id in point_ids}

    @staticmethod
    def _lerp(start: tuple[float, float], end: tuple[float, float], fraction: float) -> tuple[float, float]:
        return (
            start[0] + fraction * (end[0] - start[0]),
            start[1] + fraction * (end[1] - start[1]),
        )

    def _constraints_hold(
        self,
        positions: Mapping[str, tuple[float, float]],
        constraints: tuple[Any, ...],
        edges: Mapping[str, tuple[str, str]],
        domains: Mapping[str, str],
    ) -> bool:
        return all(self._relation_holds(relation, positions, edges, domains) for relation in constraints)

    @staticmethod
    def _relation_holds(
        relation: Any,
        positions: Mapping[str, tuple[float, float]],
        edges: Mapping[str, tuple[str, str]],
        domains: Mapping[str, str],
    ) -> bool:
        args = _relation_refs(relation)
        if relation.predicate in _POINT_ON_SUPPORT and len(args) >= 2:
            edge = edges.get(args[1])
            if edge is None or args[0] not in positions or any(point not in positions for point in edge):
                return False
            point = positions[args[0]]
            start, end = positions[edge[0]], positions[edge[1]]
            dx, dy = end[0] - start[0], end[1] - start[1]
            length_sq = dx * dx + dy * dy
            if length_sq <= 1e-12:
                return False
            fraction = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq
            cross = (point[0] - start[0]) * dy - (point[1] - start[1]) * dx
            if abs(cross) / math.sqrt(length_sq) > 1e-5:
                return False
            domain = domains.get(args[1], "segment") if relation.predicate == "incidence" else {
                "point_on_line": "line",
                "point_on_ray": "ray",
                "point_on_segment": "segment",
            }[relation.predicate]
            if domain == "line":
                return True
            if domain == "ray":
                return fraction >= -1e-6
            return -1e-6 <= fraction <= 1.0 + 1e-6
        if relation.predicate == "ordered_points":
            support_id = next(
                (argument.ref for argument in relation.arguments if argument.role == "support"),
                None,
            )
            point_ids = [argument.ref for argument in relation.arguments if argument.role != "support"]
            edge = edges.get(support_id or "")
            if edge is None or len(point_ids) < 2 or any(point_id not in positions for point_id in (*edge, *point_ids)):
                return False
            start, end = positions[edge[0]], positions[edge[1]]
            dx, dy = end[0] - start[0], end[1] - start[1]
            length_sq = dx * dx + dy * dy
            if length_sq <= 1e-12:
                return False
            fractions = []
            for point_id in point_ids:
                point = positions[point_id]
                cross = (point[0] - start[0]) * dy - (point[1] - start[1]) * dx
                if abs(cross) / math.sqrt(length_sq) > 1e-5:
                    return False
                fractions.append(((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq)
            increasing = all(right - left > 1e-6 for left, right in zip(fractions, fractions[1:]))
            decreasing = all(left - right > 1e-6 for left, right in zip(fractions, fractions[1:]))
            return increasing or decreasing
        if relation.predicate in {"parallel", "perpendicular"} and len(args) >= 2:
            left, right = edges.get(args[0]), edges.get(args[1])
            if left is None or right is None or any(point not in positions for edge in (left, right) for point in edge):
                return False
            if relation.predicate == "parallel" and set(left) & set(right):
                return False
            lv = (positions[left[1]][0] - positions[left[0]][0], positions[left[1]][1] - positions[left[0]][1])
            rv = (positions[right[1]][0] - positions[right[0]][0], positions[right[1]][1] - positions[right[0]][1])
            dot = lv[0] * rv[0] + lv[1] * rv[1]
            cross = lv[0] * rv[1] - lv[1] * rv[0]
            scale = math.hypot(*lv) * math.hypot(*rv)
            if scale <= 1e-12:
                return False
            return abs(cross if relation.predicate == "parallel" else dot) / scale <= 1e-5
        if relation.predicate == "equal_length" and len(args) >= 2:
            left, right = edges.get(args[0]), edges.get(args[1])
            if left is None or right is None or any(point not in positions for edge in (left, right) for point in edge):
                return False
            left_length = math.dist(positions[left[0]], positions[left[1]])
            right_length = math.dist(positions[right[0]], positions[right[1]])
            return min(left_length, right_length) > 1e-12 and abs(left_length - right_length) <= 1e-5 * max(left_length, right_length, 1.0)
        if relation.predicate == "midpoint":
            midpoint_refs = _midpoint_refs(relation, edges)
            if not midpoint_refs or any(item not in positions for item in midpoint_refs):
                return False
            point_id, *endpoints = midpoint_refs
            if math.dist(positions[endpoints[0]], positions[endpoints[1]]) <= 1e-12:
                return False
            midpoint = (
                (positions[endpoints[0]][0] + positions[endpoints[1]][0]) / 2,
                (positions[endpoints[0]][1] + positions[endpoints[1]][1]) / 2,
            )
            return math.dist(positions[point_id], midpoint) <= 1e-5
        if relation.predicate == "collinear" and len(args) >= 3:
            if not all(arg in positions for arg in args):
                return False
            first, second = (positions[arg] for arg in args[:2])
            direction = (second[0] - first[0], second[1] - first[1])
            length = math.hypot(*direction)
            if length <= 1e-12:
                return False
            return all(
                abs(direction[0] * (positions[arg][1] - first[1]) - direction[1] * (positions[arg][0] - first[0])) / length <= 1e-5
                for arg in args[2:]
            )
        return False
