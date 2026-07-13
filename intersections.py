"""Structured intersection helpers for synthesis and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import sympy as sp

from .geometry import Coord, add, circle_circle_intersections_2d, cross2, dot, line_circle_intersections_2d, mul, norm_sq, sub


@dataclass(frozen=True)
class IntersectionResult:
    kind: str
    status: str
    objects: Tuple[str, ...] = ()
    points: Tuple[Coord, ...] = ()
    parameters: Tuple[Dict[str, Any], ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def multiplicity(self) -> int | str:
        if self.status in {"coincident", "overlap"}:
            return "infinite"
        return len(self.points)

    @property
    def is_constructible(self) -> bool:
        return bool(self.points) and self.status in {"unique", "two_points", "tangent"}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "status": self.status,
            "objects": list(self.objects),
            "multiplicity": self.multiplicity,
            "points": [[str(value) for value in point] for point in self.points],
            "parameters": [dict(item) for item in self.parameters],
            "metadata": self.metadata,
        }


def line_line_intersection_result_2d(
    p: Coord,
    r: Coord,
    q: Coord,
    s: Coord,
    *,
    objects: Tuple[str, str] = (),
) -> IntersectionResult:
    denom = sp.simplify(cross2(r, s))
    if sp.simplify(denom) == 0:
        offset = sub(q, p)
        status = "coincident" if sp.simplify(cross2(offset, r)) == 0 else "parallel"
        return IntersectionResult("line_line", status, objects=objects)
    t = sp.simplify(cross2(sub(q, p), s) / denom)
    u = sp.simplify(cross2(sub(q, p), r) / denom)
    return IntersectionResult(
        "line_line",
        "unique",
        objects=objects,
        points=(add(p, mul(r, t)),),
        parameters=({"t": str(t), "u": str(u)},),
    )


def segment_segment_intersection_result_2d(
    a: Coord,
    b: Coord,
    c: Coord,
    d: Coord,
    *,
    objects: Tuple[str, str] = (),
) -> IntersectionResult:
    line_result = line_line_intersection_result_2d(a, sub(b, a), c, sub(d, c), objects=objects)
    if line_result.status != "unique":
        return IntersectionResult("segment_segment", line_result.status, objects=objects, metadata={"line_status": line_result.status})
    params = line_result.parameters[0]
    t = sp.sympify(params["t"])
    u = sp.sympify(params["u"])
    if _in_unit_interval(t) and _in_unit_interval(u):
        return IntersectionResult("segment_segment", "unique", objects=objects, points=line_result.points, parameters=line_result.parameters)
    return IntersectionResult("segment_segment", "disjoint", objects=objects, metadata={"line_intersection": line_result.to_dict()})


def line_circle_intersection_result_2d(
    line_point: Coord,
    line_dir: Coord,
    center: Coord,
    radius,
    *,
    objects: Tuple[str, str] = (),
) -> IntersectionResult:
    radius = sp.sympify(radius)
    a = norm_sq(line_dir)
    if sp.simplify(a) == 0:
        return IntersectionResult("line_circle", "degenerate", objects=objects, metadata={"reason": "zero_direction"})
    f = sub(line_point, center)
    b = 2 * dot(f, line_dir)
    c = norm_sq(f) - radius ** 2
    disc = sp.simplify(b ** 2 - 4 * a * c)
    if _is_negative(disc):
        return IntersectionResult("line_circle", "disjoint", objects=objects, metadata={"discriminant": str(disc)})
    points = line_circle_intersections_2d(line_point, line_dir, center, radius)
    parameters = tuple({"t": str(_line_parameter(line_point, line_dir, point))} for point in points)
    status = "tangent" if len(points) == 1 else "two_points"
    ordered = _sort_by_parameter(points, parameters, key="t")
    return IntersectionResult("line_circle", status, objects=objects, points=ordered[0], parameters=ordered[1], metadata={"discriminant": str(disc)})


def circle_circle_intersection_result_2d(
    c1: Coord,
    r1,
    c2: Coord,
    r2,
    *,
    objects: Tuple[str, str] = (),
) -> IntersectionResult:
    r1 = sp.sympify(r1)
    r2 = sp.sympify(r2)
    delta = sub(c2, c1)
    d_sq = norm_sq(delta)
    if sp.simplify(d_sq) == 0:
        status = "coincident" if sp.simplify(r1 - r2) == 0 else "concentric"
        return IntersectionResult("circle_circle", status, objects=objects)
    d = sp.sqrt(d_sq)
    if _is_negative((r1 + r2) ** 2 - d_sq) or _is_negative(d_sq - (r1 - r2) ** 2):
        return IntersectionResult("circle_circle", "disjoint", objects=objects, metadata={"center_distance": str(d)})
    points = circle_circle_intersections_2d(c1, r1, c2, r2)
    status = "tangent" if len(points) == 1 else "two_points"
    ordered = tuple(sorted(points, key=lambda point: (float(sp.N(point[1])), float(sp.N(point[0])))))
    return IntersectionResult("circle_circle", status, objects=objects, points=ordered, metadata={"center_distance": str(d)})


def select_intersection_points(result: IntersectionResult, select: str = "all") -> Tuple[Coord, ...]:
    if select == "all":
        return result.points
    if not result.points:
        return ()
    if select == "first":
        return (result.points[0],)
    if select == "second":
        return result.points[1:2]
    if select == "upper":
        return (max(result.points, key=lambda point: float(sp.N(point[1]))),)
    if select == "lower":
        return (min(result.points, key=lambda point: float(sp.N(point[1]))),)
    if select == "left":
        return (min(result.points, key=lambda point: float(sp.N(point[0]))),)
    if select == "right":
        return (max(result.points, key=lambda point: float(sp.N(point[0]))),)
    raise ValueError(f"Unsupported intersection selection: {select}")


def _line_parameter(line_point: Coord, line_dir: Coord, point: Coord) -> sp.Expr:
    denom = norm_sq(line_dir)
    return sp.simplify(dot(sub(point, line_point), line_dir) / denom)


def _sort_by_parameter(points: Tuple[Coord, ...], parameters: Tuple[Dict[str, str], ...], *, key: str) -> tuple[Tuple[Coord, ...], Tuple[Dict[str, str], ...]]:
    pairs = sorted(zip(points, parameters), key=lambda item: float(sp.N(sp.sympify(item[1][key]))))
    return tuple(point for point, _ in pairs), tuple(param for _, param in pairs)


def _in_unit_interval(value: sp.Expr) -> bool:
    numeric = float(sp.N(value))
    return -1e-12 <= numeric <= 1 + 1e-12


def _is_negative(value: sp.Expr) -> bool:
    simplified = sp.simplify(value)
    if simplified.is_number:
        return bool(sp.N(simplified) < 0)
    return False
