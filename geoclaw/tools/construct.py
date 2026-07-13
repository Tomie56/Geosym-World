"""Transactional auxiliary-construction tool."""

from __future__ import annotations

from typing import Any, Dict

from geosym_world_geometry.builder import ConstructionBuilder
from geosym_world_geometry.errors import GeometryWorldError, UnsupportedConstructionError
from geosym_world_geometry.solver import FactClosureSolver
from geosym_world_geometry.verifier import TraceVerifier

from ..feedback import GEOMETRICALLY_INVALID, INVALID_ARGUMENT, UNKNOWN_OBJECT, UNSUPPORTED_OPERATION, VALID
from .base import ToolResult


ALLOWED_CONSTRUCTIONS = {
    "add_segment",
    "add_line",
    "midpoint",
    "parallel_through",
    "perpendicular_through",
    "foot_of_perpendicular",
    "angle_bisector",
    "intersection_line_line",
    "intersection_line_circle",
    "intersection_circle_circle",
    "add_circle",
    "add_chord",
    "point_on_segment_ratio",
}


class ConstructTool:
    name = "construct"

    def execute(self, environment: Any, arguments: Dict[str, Any]) -> ToolResult:
        operation = arguments.get("operation")
        if operation not in ALLOWED_CONSTRUCTIONS:
            return ToolResult(UNSUPPORTED_OPERATION, errors=[{"code": "construct.unsupported", "operation": operation}])
        params = arguments.get("params", {})
        if not isinstance(params, dict):
            return ToolResult(INVALID_ARGUMENT, errors=[{"code": "construct.invalid_params"}])
        try:
            params = _resolve_params(environment, operation, params)
        except ValueError as exc:
            return ToolResult(UNKNOWN_OBJECT, errors=[{"code": "construct.reference_resolution_failed", "message": str(exc)}])
        op = {"op": operation, **params}
        if arguments.get("id") is not None:
            op["id"] = str(arguments["id"])
        candidate_trace = [*environment.base_trace, *environment.agent_trace, op]
        try:
            report = TraceVerifier().verify_trace(candidate_trace)
            if not report.valid:
                errors = [issue.to_dict() for issue in report.issues if issue.severity == "error"]
                return ToolResult(GEOMETRICALLY_INVALID, errors=errors)
            candidate_state = ConstructionBuilder().execute(candidate_trace)
            candidate_facts = FactClosureSolver().solve(candidate_state)
        except UnsupportedConstructionError as exc:
            return ToolResult(UNSUPPORTED_OPERATION, errors=[{"code": "construct.unsupported", "message": str(exc)}])
        except (GeometryWorldError, KeyError, TypeError, ValueError) as exc:
            return ToolResult(GEOMETRICALLY_INVALID, errors=[{"code": "construct.failed", "message": str(exc)}])
        before_ids = set(environment.resolver.visible_object_ids(environment.state))
        after_ids = set(environment.resolver.visible_object_ids(candidate_state))
        created = sorted(after_ids - before_ids)
        environment.agent_trace.append(op)
        environment.state = candidate_state
        environment.facts = candidate_facts
        evidence = [fact.to_dict() for fact in candidate_facts if fact.object in created][:20]
        return ToolResult(
            VALID,
            result={"operation": operation, "created_objects": created, "trace_op": op},
            evidence=evidence,
            state_changed=True,
            image_changed=True,
        )


def _resolve_params(environment: Any, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
    resolved = dict(params)
    fields = {
        "add_segment": {"points": "point", "start": "point", "end": "point"},
        "add_line": {"points": "point", "point": "point"},
        "midpoint": {"of": "point", "points": "point"},
        "parallel_through": {"point": "point", "line": "line_like"},
        "perpendicular_through": {"point": "point", "line": "line_like"},
        "foot_of_perpendicular": {"point": "point", "line": "line_like"},
        "angle_bisector": {"vertex": "point", "ray1": "point", "ray2": "point"},
        "intersection_line_line": {"line1": "line_like", "line2": "line_like"},
        "intersection_line_circle": {"line": "line_like", "circle": "circle"},
        "intersection_circle_circle": {"circle1": "circle", "circle2": "circle"},
        "add_circle": {"center": "point"},
        "add_chord": {"circle": "circle", "points": "point", "start": "point", "end": "point"},
        "point_on_segment_ratio": {"segment": "segment", "of": "point", "points": "point"},
    }.get(operation, {})
    for field, expected in fields.items():
        if field not in resolved:
            continue
        value = resolved[field]
        if isinstance(value, list):
            resolved[field] = [_resolve_reference(environment, item, expected) for item in value]
        elif isinstance(value, str):
            resolved[field] = _resolve_reference(environment, value, expected)
    return resolved


def _resolve_reference(environment: Any, reference: Any, expected: str) -> Any:
    if not isinstance(reference, str):
        return reference
    expected_types = ("line", "segment", "ray") if expected == "line_like" else (expected,)
    resolution = environment.resolver.resolve(reference, environment.state, expected_types)
    if resolution.status != "resolved":
        raise ValueError(f"Cannot resolve {reference!r} as {expected}: {resolution.status} {resolution.candidates}")
    return resolution.object_id
