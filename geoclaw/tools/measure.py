"""Exact-or-image-proxy geometry measurement tool."""

from __future__ import annotations

from typing import Any, Dict

import math

import sympy as sp
from geosym_world_geometry.rendering import state_to_render_scene
from geosym_world_geometry.rendering.layout import build_view_transform

from ..feedback import INVALID_ARGUMENT, NOT_VERIFIABLE, VALID
from .base import ToolResult


SUPPORTED = {"length", "distance", "angle", "area", "coordinate", "perimeter", "volume"}


class MeasureTool:
    name = "measure"

    def execute(self, environment: Any, arguments: Dict[str, Any]) -> ToolResult:
        measurement = arguments.get("measurement")
        mode = arguments.get("mode", "image")
        if measurement not in SUPPORTED or mode not in {"image", "oracle"}:
            return ToolResult(INVALID_ARGUMENT, errors=[{"code": "measure.invalid_request"}])
        object_id = arguments.get("object")
        objects = arguments.get("objects")
        if object_id is None and isinstance(objects, list):
            object_id = "|".join(str(item) for item in objects)
        if measurement == "coordinate":
            point_id = str(object_id or (objects[0] if isinstance(objects, list) and objects else ""))
            if point_id not in environment.state.points:
                return ToolResult(NOT_VERIFIABLE, errors=[{"code": "measure.unknown_point"}])
            exact = [str(value) for value in environment.state.points[point_id].coords]
            numeric = [float(sp.N(value)) for value in environment.state.points[point_id].coords]
            return ToolResult(VALID, result={"measurement": measurement, "object": point_id, "mode": mode, "exact_value": exact if mode == "oracle" else None, "numeric_value": numeric})
        if object_id is not None and "|" not in str(object_id):
            resolution = environment.resolver.resolve(str(object_id), environment.state)
            if resolution.status == "resolved":
                object_id = resolution.object_id
        matches = environment.facts.find(measurement, str(object_id))
        if not matches and measurement == "distance" and isinstance(objects, list) and len(objects) == 2:
            reverse_id = "|".join(str(item) for item in reversed(objects))
            matches = environment.facts.find(measurement, reverse_id)
        valued = [fact for fact in matches if fact.value is not None]
        if not valued:
            return ToolResult(NOT_VERIFIABLE, result={"measurement": measurement, "object": object_id})
        fact = valued[0]
        numeric = float(sp.N(sp.sympify(fact.value)))
        pixel_value = None
        backend = "oracle_fact"
        if mode == "image":
            measured = _pixel_measurement(environment, measurement, str(object_id))
            if measured is not None:
                numeric, pixel_value = measured
                backend = "render_scene_pixels"
            else:
                backend = "geometry_proxy"
            numeric = round(numeric, int(arguments.get("precision", 2)))
        audit = {"oracle_numeric_value": float(sp.N(sp.sympify(fact.value)))}
        if mode == "image":
            audit["visual_solver_residual"] = numeric - audit["oracle_numeric_value"]
        return ToolResult(
            VALID,
            result={"measurement": measurement, "object": object_id, "mode": mode, "backend": backend, "exact_value": fact.value if mode == "oracle" else None, "numeric_value": numeric, "pixel_value": pixel_value},
            evidence=[fact.to_dict()],
            audit=audit,
        )


def _pixel_measurement(environment: Any, measurement: str, object_id: str) -> tuple[float, float] | None:
    if measurement not in {"length", "distance"}:
        return None
    scene = state_to_render_scene(environment.state)
    transform = build_view_transform(scene)
    primitive = next((item for item in scene.primitives if item.id == object_id and item.kind == "segment"), None)
    if primitive is None and measurement == "distance" and "|" in object_id:
        left, right = object_id.split("|", 1)
        if left in environment.state.points and right in environment.state.points:
            points = [_project(environment.state.points[left].coords), _project(environment.state.points[right].coords)]
        else:
            return None
    elif primitive is not None:
        points = [tuple(point) for point in primitive.geometry["points"]]
    else:
        return None
    pixel_points = [transform.point(point) for point in points]
    pixel_distance = math.dist(pixel_points[0], pixel_points[1])
    return pixel_distance / transform.scale, pixel_distance


def _project(coords) -> tuple[float, float]:
    values = [float(sp.N(value)) for value in coords]
    if len(values) == 2:
        return values[0], values[1]
    return values[0] - 0.55 * values[1], values[2] + 0.35 * values[1]
