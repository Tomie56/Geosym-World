"""Auditable source redraw, semantic redraw, and exact resynthesis."""

from __future__ import annotations

import copy
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import sympy as sp

from geosym_world.construction import ConstructionBuilder, ConstructionOp
from geosym_world.errors import InvalidGeometryInput
from geosym_world.rendering import (
    AnnotationPolicy,
    DeterministicStylePolicy,
    RenderScene,
    source_graph_to_render_scene,
    state_to_render_scene,
)
from geosym_world.schema import FormalGeometryState
from geosym_world.schema.canonical import stable_checksum
from geosym_world.verifier import ParseGraphAdmissionReport, TraceVerifier

from .reconstruction import ReconstructionBranchSet, ReconstructionReport, ReconstructionStatus


REDRAW_SCHEMA_VERSION = "1.1"
BRANCH_REDRAW_SCHEMA_VERSION = "1.0"


class RedrawMode(str, Enum):
    SOURCE = "source_redraw"
    SEMANTIC = "semantic_redraw"
    RESYNTHESIS = "resynthesis"


class ResynthesisTransform(str, Enum):
    RIGID = "rigid"
    SIMILARITY = "similarity"


@dataclass(frozen=True)
class RedrawResult:
    mode: RedrawMode
    source_graph_checksum: str
    admitted_graph_checksum: str
    scene: RenderScene
    trace: tuple[ConstructionOp, ...]
    verification: Mapping[str, Any]
    lineage: Mapping[str, Any]
    invariants: Mapping[str, Any]
    schema_version: str = REDRAW_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "mode": self.mode.value,
            "source_graph_checksum": self.source_graph_checksum,
            "admitted_graph_checksum": self.admitted_graph_checksum,
            "scene": self.scene.to_dict(),
            "trace": [_json_value(operation.to_dict()) for operation in self.trace],
            "verification": dict(self.verification),
            "lineage": dict(self.lineage),
            "invariants": dict(self.invariants),
        }
        payload["checksum"] = stable_checksum(payload)
        return payload


@dataclass(frozen=True)
class BranchRedrawResult:
    branch_checksum: str
    branch_choices: Mapping[str, str]
    redraw: RedrawResult
    schema_version: str = BRANCH_REDRAW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.redraw.mode is not RedrawMode.SEMANTIC:
            raise InvalidGeometryInput("Branch redraws must use semantic redraw mode")
        if self.redraw.lineage.get("reconstruction_checksum") != self.branch_checksum:
            raise InvalidGeometryInput("Branch redraw lineage does not match its reconstruction checksum")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "branch_checksum": self.branch_checksum,
            "branch_choices": dict(sorted(self.branch_choices.items())),
            "redraw": self.redraw.to_dict(),
        }
        payload["checksum"] = stable_checksum(payload)
        return payload


class ParseRedrawPipeline:
    """Render one admitted parse through an explicitly selected output mode."""

    def render(
        self,
        admission: ParseGraphAdmissionReport,
        *,
        mode: RedrawMode,
        reconstruction: ReconstructionReport | None = None,
        seed: int = 0,
        resynthesis_transform: ResynthesisTransform = ResynthesisTransform.RIGID,
    ) -> RedrawResult:
        if not isinstance(admission, ParseGraphAdmissionReport):
            raise TypeError("ParseRedrawPipeline requires a ParseGraphAdmissionReport")
        if not isinstance(mode, RedrawMode):
            raise InvalidGeometryInput(f"Unknown redraw mode: {mode}")
        if not isinstance(resynthesis_transform, ResynthesisTransform):
            raise InvalidGeometryInput(
                f"Unknown resynthesis transform: {resynthesis_transform}"
            )
        if mode is RedrawMode.SOURCE:
            return self._source(admission)
        checked = _require_reconstruction(admission, reconstruction)
        if mode is RedrawMode.SEMANTIC:
            return self._semantic(admission, checked)
        return self._resynthesis(
            admission,
            checked,
            seed=seed,
            transform_class=resynthesis_transform,
        )

    def render_branches(
        self,
        admission: ParseGraphAdmissionReport,
        branch_set: ReconstructionBranchSet,
    ) -> tuple[BranchRedrawResult, ...]:
        if not isinstance(branch_set, ReconstructionBranchSet):
            raise TypeError("render_branches requires a ReconstructionBranchSet")
        if branch_set.source_graph_checksum != admission.source_graph_checksum:
            raise InvalidGeometryInput("Branch set and admission source checksums differ")
        rendered = []
        for branch in branch_set.branches:
            if branch.status is not ReconstructionStatus.EXACT:
                raise InvalidGeometryInput("Branch set contains a non-exact admitted branch")
            branch_payload = branch.to_dict()
            redraw = self.render(
                admission,
                mode=RedrawMode.SEMANTIC,
                reconstruction=branch,
            )
            rendered.append(
                BranchRedrawResult(
                    branch_checksum=branch_payload["checksum"],
                    branch_choices=branch.branch_choices,
                    redraw=redraw,
                )
            )
        return tuple(rendered)

    def _source(self, admission: ParseGraphAdmissionReport) -> RedrawResult:
        graph = admission.admitted_graph
        scene = source_graph_to_render_scene(graph)
        quality = scene.metadata["render_quality"]
        return RedrawResult(
            mode=RedrawMode.SOURCE,
            source_graph_checksum=admission.source_graph_checksum,
            admitted_graph_checksum=graph.checksum(),
            scene=scene,
            trace=(),
            verification={"valid": bool(quality["valid"]), "render_quality": quality},
            lineage=_lineage(admission, RedrawMode.SOURCE),
            invariants={
                "layout": "observed_pixel_witness",
                "annotations": "explicit_source_only",
                "mathematical_realization": False,
            },
        )

    def _semantic(
        self,
        admission: ParseGraphAdmissionReport,
        reconstruction: ReconstructionReport,
    ) -> RedrawResult:
        state = _required_state(reconstruction)
        state_verification = TraceVerifier().verify_state(state)
        if not state_verification.valid:
            raise InvalidGeometryInput("Semantic redraw requires a valid reconstructed state")
        reconstruction_checksum = reconstruction.to_dict()["checksum"]
        scene = state_to_render_scene(
            state,
            scene_id=(
                f"semantic_redraw:{admission.admitted_graph.id}:"
                f"{reconstruction_checksum[:12]}"
            ),
            annotation_policy=AnnotationPolicy(
                level="givens",
                relation_markers=True,
                inferred_markers=False,
            ),
        )
        return RedrawResult(
            mode=RedrawMode.SEMANTIC,
            source_graph_checksum=admission.source_graph_checksum,
            admitted_graph_checksum=admission.admitted_graph.checksum(),
            scene=scene,
            trace=reconstruction.trace,
            verification={
                "valid": state_verification.valid and bool(scene.metadata["render_quality"]["valid"]),
                "state": state_verification.to_dict(),
                "render_quality": scene.metadata["render_quality"],
                "reconstruction_status": reconstruction.status.value,
            },
            lineage={
                **_lineage(admission, RedrawMode.SEMANTIC),
                "reconstruction_checksum": reconstruction_checksum,
                "branch_choices": dict(sorted(reconstruction.branch_choices.items())),
            },
            invariants={
                "topology_signature": _state_topology_signature(state),
                "annotation_projection": "admitted_source_only",
                "inferred_markers": False,
            },
        )

    def _resynthesis(
        self,
        admission: ParseGraphAdmissionReport,
        reconstruction: ReconstructionReport,
        *,
        seed: int,
        transform_class: ResynthesisTransform,
    ) -> RedrawResult:
        if reconstruction.status is not ReconstructionStatus.EXACT:
            raise InvalidGeometryInput("Resynthesis requires an exact reconstruction")
        parent = _required_state(reconstruction)
        if transform_class is ResynthesisTransform.RIGID:
            trace, transform = _rigid_transform_trace(reconstruction.trace, seed=seed)
            preserved = ["distance", "angle", "incidence", "orientation", "region_topology", "target"]
            invariant_class = "exact_rigid_2d"
        else:
            _validate_similarity_resynthesis(reconstruction)
            trace, transform = _similarity_transform_trace(reconstruction.trace, seed=seed)
            preserved = ["angle", "ratio", "incidence", "orientation", "region_topology", "target_type"]
            invariant_class = "exact_similarity_2d"
        verification = TraceVerifier().verify_trace(trace)
        if not verification.valid:
            raise InvalidGeometryInput(
                f"{transform_class.value} resynthesis produced an invalid construction trace"
            )
        state = ConstructionBuilder().execute(trace)
        parent_topology = _state_topology_signature(parent)
        child_topology = _state_topology_signature(state)
        if parent_topology != child_topology:
            raise InvalidGeometryInput("Rigid resynthesis changed topology")
        scene = state_to_render_scene(
            state,
            scene_id=f"resynthesis:{admission.admitted_graph.id}:{seed}",
            style_policy=DeterministicStylePolicy(seed=seed),
            annotation_policy=AnnotationPolicy(
                level="givens",
                relation_markers=True,
                inferred_markers=False,
            ),
        )
        source_facts_checksum = stable_checksum(reconstruction.source_facts.to_list())
        return RedrawResult(
            mode=RedrawMode.RESYNTHESIS,
            source_graph_checksum=admission.source_graph_checksum,
            admitted_graph_checksum=admission.admitted_graph.checksum(),
            scene=scene,
            trace=trace,
            verification={
                "valid": verification.valid and bool(scene.metadata["render_quality"]["valid"]),
                "trace": verification.to_dict(),
                "render_quality": scene.metadata["render_quality"],
            },
            lineage={
                **_lineage(admission, RedrawMode.RESYNTHESIS),
                "seed": seed,
                "parent_state_checksum": parent.checksum(),
                "transform": transform,
            },
            invariants={
                "transform_class": invariant_class,
                "topology_signature": child_topology,
                "source_facts_checksum": source_facts_checksum,
                "preserves": preserved,
            },
        )


def _require_reconstruction(
    admission: ParseGraphAdmissionReport,
    reconstruction: ReconstructionReport | None,
) -> ReconstructionReport:
    if reconstruction is None:
        raise InvalidGeometryInput("This redraw mode requires a reconstruction report")
    if not isinstance(reconstruction, ReconstructionReport):
        raise TypeError("reconstruction must be a ReconstructionReport")
    if reconstruction.source_graph_checksum != admission.source_graph_checksum:
        raise InvalidGeometryInput("Reconstruction and admission source checksums differ")
    if reconstruction.admitted_graph_checksum != admission.admitted_graph.checksum():
        raise InvalidGeometryInput("Reconstruction and admitted graph checksums differ")
    return reconstruction


def _required_state(reconstruction: ReconstructionReport) -> FormalGeometryState:
    if reconstruction.state is None:
        raise InvalidGeometryInput("Redraw requires a materialized reconstruction state")
    return reconstruction.state


def _rigid_transform_trace(
    trace: Sequence[ConstructionOp],
    *,
    seed: int,
) -> tuple[tuple[ConstructionOp, ...], dict[str, Any]]:
    rng = random.Random(seed)
    quarter_turn = rng.randrange(4)
    dx = rng.randint(-3, 3)
    dy = rng.randint(-3, 3)
    transformed = []
    for operation in trace:
        params = copy.deepcopy(operation.params)
        if operation.op == "add_point":
            coords = params.get("coord", params.get("coords"))
            if not isinstance(coords, Sequence) or isinstance(coords, (str, bytes)) or len(coords) != 2:
                raise InvalidGeometryInput("Rigid 2D resynthesis requires two-dimensional add_point coordinates")
            x, y = (sp.sympify(value) for value in coords)
            rotated = ((x, y), (-y, x), (-x, -y), (y, -x))[quarter_turn]
            params.pop("coord", None)
            params["coords"] = [sp.simplify(rotated[0] + dx), sp.simplify(rotated[1] + dy)]
        elif operation.op == "add_point3d":
            raise InvalidGeometryInput("Rigid 2D resynthesis does not accept 3D points")
        transformed.append(ConstructionOp(operation.op, operation.id, params))
    return tuple(transformed), {
        "quarter_turns_ccw": quarter_turn,
        "translation": [dx, dy],
    }


def _similarity_transform_trace(
    trace: Sequence[ConstructionOp],
    *,
    seed: int,
) -> tuple[tuple[ConstructionOp, ...], dict[str, Any]]:
    rng = random.Random(seed)
    scale = rng.choice((sp.Rational(1, 2), sp.Rational(3, 2), sp.Integer(2)))
    quarter_turn = rng.randrange(4)
    dx = rng.randint(-3, 3)
    dy = rng.randint(-3, 3)
    transformed = []
    for operation in trace:
        params = copy.deepcopy(operation.params)
        if operation.op == "add_point":
            coords = params.get("coord", params.get("coords"))
            if not isinstance(coords, Sequence) or isinstance(coords, (str, bytes)) or len(coords) != 2:
                raise InvalidGeometryInput("Similarity 2D resynthesis requires two-dimensional add_point coordinates")
            x, y = (sp.sympify(value) for value in coords)
            rotated = ((x, y), (-y, x), (-x, -y), (y, -x))[quarter_turn]
            params.pop("coord", None)
            params["coords"] = [
                sp.simplify(scale * rotated[0] + dx),
                sp.simplify(scale * rotated[1] + dy),
            ]
        elif operation.op == "add_circle":
            params["radius"] = sp.simplify(scale * sp.sympify(params["radius"]))
        elif operation.op == "point_on_ray_distance":
            params["distance"] = sp.simplify(scale * sp.sympify(params["distance"]))
        elif operation.op == "add_point3d":
            raise InvalidGeometryInput("Similarity 2D resynthesis does not accept 3D points")
        transformed.append(ConstructionOp(operation.op, operation.id, params))
    return tuple(transformed), {
        "scale": str(scale),
        "quarter_turns_ccw": quarter_turn,
        "translation": [dx, dy],
    }


def _validate_similarity_resynthesis(reconstruction: ReconstructionReport) -> None:
    dimensional_fact_types = {
        "area",
        "arc_length",
        "coordinate",
        "diameter",
        "distance",
        "length",
        "perimeter",
        "point_on_ray_distance",
        "radius",
        "volume",
    }
    blocked_facts = sorted(
        {
            fact.type
            for fact in reconstruction.source_facts
            if fact.type in dimensional_fact_types
        }
    )
    if blocked_facts:
        raise InvalidGeometryInput(
            "Similarity resynthesis cannot change dimensional source facts: "
            + ", ".join(blocked_facts)
        )
    state = _required_state(reconstruction)
    if (
        state.target is not None
        and state.target.mode == "constraint"
        and state.target.type in dimensional_fact_types
    ):
        raise InvalidGeometryInput(
            f"Similarity resynthesis cannot change dimensional constraint target: {state.target.type}"
        )


def _state_topology_signature(state: FormalGeometryState) -> str:
    payload = {
        "points": sorted(state.points),
        "lines": sorted((item.id, list(item.source_points)) for item in state.lines.values()),
        "segments": sorted((item.id, item.start, item.end) for item in state.segments.values()),
        "rays": sorted((item.id, item.start, item.through) for item in state.rays.values()),
        "circles": sorted((item.id, item.center) for item in state.circles.values()),
        "arcs": sorted((item.id, item.circle, item.start, item.end, item.orientation) for item in state.arcs.values()),
        "polygons": sorted((item.id, list(item.vertices)) for item in state.polygons.values()),
        "regions": sorted((item.id, list(item.boundary), list(item.holes), item.kind) for item in state.regions.values()),
        "composite_regions": sorted(
            (item.id, item.operation, list(item.components)) for item in state.composite_regions.values()
        ),
        "relations": sorted(
            stable_checksum(
                {
                    key: value
                    for key, value in relation.items()
                    if key not in {"metadata", "result"}
                }
            )
            for relation in state.relations
        ),
        "target": state.target.to_dict() if state.target is not None else None,
    }
    return stable_checksum(payload)


def _lineage(admission: ParseGraphAdmissionReport, mode: RedrawMode) -> dict[str, Any]:
    return {
        "root": admission.source_graph_checksum,
        "group_key": admission.source_graph_checksum,
        "role": mode.value,
        "admission_policy": admission.policy.policy_id,
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, sp.Basic):
        return str(sp.simplify(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value
