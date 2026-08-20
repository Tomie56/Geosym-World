"""Structured intersection helpers for synthesis and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cmp_to_key
from typing import Any, Dict, Tuple

import sympy as sp

from geosym_world.kernel.geometry2d.analytic import Coord, add, circle_circle_intersections_2d, cross2, dot, line_circle_intersections_2d, mul, norm_sq, sub


INTERSECTION_STATUSES = frozenset({
    "unique",
    "two_points",
    "tangent",
    "parallel",
    "coincident",
    "concentric",
    "overlap",
    "disjoint",
    "degenerate",
})


@dataclass(frozen=True)
class PredicateDiagnostic:
    predicate: str
    classification: str
    exact: bool
    residual: str
    normalized_residual: float | None
    tolerance: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicate": self.predicate,
            "classification": self.classification,
            "exact": self.exact,
            "residual": self.residual,
            "normalized_residual": self.normalized_residual,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True)
class IntersectionResult:
    kind: str
    status: str
    objects: Tuple[str, ...] = ()
    points: Tuple[Coord, ...] = ()
    parameters: Tuple[Dict[str, Any], ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in INTERSECTION_STATUSES:
            raise ValueError(f"Unsupported intersection status: {self.status}")

    @property
    def multiplicity(self) -> int | str:
        if self.status in {"coincident", "overlap"}:
            return "infinite"
        return len(self.points)

    @property
    def is_constructible(self) -> bool:
        return bool(self.points) and self.status in {"unique", "two_points", "tangent"}

    @property
    def contact_dimension(self) -> int | None:
        if self.status in {"unique", "two_points", "tangent"}:
            return 0
        if self.status in {"coincident", "overlap"}:
            return 1
        return None

    @property
    def contact_kind(self) -> str:
        if self.contact_dimension == 0:
            return "point"
        if self.contact_dimension == 1:
            return "curve"
        if self.status == "degenerate":
            return "invalid"
        return "none"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "status": self.status,
            "objects": list(self.objects),
            "multiplicity": self.multiplicity,
            "points": [[str(value) for value in point] for point in self.points],
            "parameters": [dict(item) for item in self.parameters],
            "topology": {
                "contact_dimension": self.contact_dimension,
                "contact_kind": self.contact_kind,
                "exact": True,
            },
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
    if sp.simplify(norm_sq(r)) == 0 or sp.simplify(norm_sq(s)) == 0:
        return IntersectionResult("line_line", "degenerate", objects=objects, metadata={"reason": "zero_direction"})
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


def linear_linear_intersection_result_2d(
    p: Coord,
    r: Coord,
    q: Coord,
    s: Coord,
    *,
    first_domain: str = "line",
    second_domain: str = "line",
    objects: Tuple[str, str] = (),
) -> IntersectionResult:
    """Intersect line-like objects while respecting line/ray/segment domains."""

    _validate_linear_domain(first_domain)
    _validate_linear_domain(second_domain)
    raw = line_line_intersection_result_2d(p, r, q, s, objects=objects)
    kind = f"{first_domain}_{second_domain}"
    if raw.status == "unique":
        t = sp.sympify(raw.parameters[0]["t"])
        u = sp.sympify(raw.parameters[0]["u"])
        if _parameter_in_domain(t, first_domain) and _parameter_in_domain(u, second_domain):
            return IntersectionResult(kind, "unique", objects=objects, points=raw.points, parameters=raw.parameters)
        return IntersectionResult(kind, "disjoint", objects=objects, metadata={"line_intersection": raw.to_dict()})
    if raw.status != "coincident":
        return IntersectionResult(kind, raw.status, objects=objects, metadata={"line_status": raw.status})
    return _coincident_linear_intersection(
        p,
        r,
        q,
        s,
        first_domain=first_domain,
        second_domain=second_domain,
        objects=objects,
    )


def segment_segment_intersection_result_2d(
    a: Coord,
    b: Coord,
    c: Coord,
    d: Coord,
    *,
    objects: Tuple[str, str] = (),
) -> IntersectionResult:
    result = linear_linear_intersection_result_2d(
        a,
        sub(b, a),
        c,
        sub(d, c),
        first_domain="segment",
        second_domain="segment",
        objects=objects,
    )
    if result.status == "parallel":
        return IntersectionResult("segment_segment", "disjoint", objects=objects, metadata=result.metadata)
    return IntersectionResult(
        "segment_segment",
        result.status,
        objects=result.objects,
        points=result.points,
        parameters=result.parameters,
        metadata=result.metadata,
    )


def line_circle_intersection_result_2d(
    line_point: Coord,
    line_dir: Coord,
    center: Coord,
    radius,
    *,
    domain: str = "line",
    objects: Tuple[str, str] = (),
) -> IntersectionResult:
    _validate_linear_domain(domain)
    radius = sp.sympify(radius)
    if _compare_real(radius, sp.Integer(0)) <= 0:
        return IntersectionResult("line_circle", "degenerate", objects=objects, metadata={"reason": "nonpositive_radius"})
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
    ordered = _sort_by_parameter(points, parameters, key="t")
    retained = tuple(
        (point, parameter)
        for point, parameter in zip(*ordered)
        if _parameter_in_domain(sp.sympify(parameter["t"]), domain)
    )
    if not retained:
        return IntersectionResult(f"{domain}_circle", "disjoint", objects=objects, metadata={"discriminant": str(disc), "line_status": "intersects_extension"})
    retained_points = tuple(point for point, _ in retained)
    retained_parameters = tuple(parameter for _, parameter in retained)
    status = "tangent" if len(points) == 1 else "two_points" if len(retained_points) == 2 else "unique"
    return IntersectionResult(f"{domain}_circle", status, objects=objects, points=retained_points, parameters=retained_parameters, metadata={"discriminant": str(disc)})


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
    if _compare_real(r1, sp.Integer(0)) <= 0 or _compare_real(r2, sp.Integer(0)) <= 0:
        return IntersectionResult("circle_circle", "degenerate", objects=objects, metadata={"reason": "nonpositive_radius"})
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
    ordered = tuple(sorted(points, key=cmp_to_key(_compare_points_yx)))
    return IntersectionResult("circle_circle", status, objects=objects, points=ordered, metadata={"center_distance": str(d)})


def directed_arc_angle_2d(
    center: Coord,
    start: Coord,
    end: Coord,
    *,
    orientation: str = "ccw",
) -> sp.Expr:
    """Return the directed sweep from start to end in [0, 2*pi)."""

    if orientation not in {"ccw", "cw"}:
        raise ValueError(f"Unsupported arc orientation: {orientation}")
    start_vector = sub(start, center)
    end_vector = sub(end, center)
    if sp.simplify(norm_sq(start_vector)) == 0 or sp.simplify(norm_sq(end_vector)) == 0:
        raise ValueError("Arc endpoints must differ from the center")
    cross = cross2(start_vector, end_vector)
    if orientation == "cw":
        cross = -cross
    angle = sp.simplify(sp.atan2(cross, dot(start_vector, end_vector)))
    if _compare_real(angle, sp.Integer(0)) < 0:
        angle = sp.simplify(angle + 2 * sp.pi)
    return angle


def point_on_arc_domain_2d(
    point: Coord,
    center: Coord,
    radius,
    start: Coord,
    end: Coord,
    *,
    orientation: str = "ccw",
    central_angle=None,
    allow_full_circle: bool = False,
) -> bool:
    """Check circle membership and directed arc-domain membership exactly."""

    radius = sp.sympify(radius)
    if sp.simplify(norm_sq(sub(point, center)) - radius ** 2) != 0:
        return False
    if allow_full_circle and _coords_equal(start, end):
        return True
    sweep = directed_arc_angle_2d(center, start, end, orientation=orientation)
    if central_angle is not None and sp.simplify(sweep - sp.sympify(central_angle)) != 0:
        return False
    candidate = directed_arc_angle_2d(center, start, point, orientation=orientation)
    return _compare_real(candidate, sweep) <= 0


def linear_arc_intersection_result_2d(
    line_point: Coord,
    line_dir: Coord,
    center: Coord,
    radius,
    start: Coord,
    end: Coord,
    *,
    domain: str = "line",
    orientation: str = "ccw",
    central_angle=None,
    allow_full_circle: bool = False,
    objects: Tuple[str, str] = (),
) -> IntersectionResult:
    raw = line_circle_intersection_result_2d(
        line_point,
        line_dir,
        center,
        radius,
        domain=domain,
        objects=objects,
    )
    return _filter_result_by_arcs(
        raw,
        kind=f"{domain}_arc",
        arc_domains=((center, radius, start, end, orientation, central_angle, allow_full_circle),),
    )


def circle_arc_intersection_result_2d(
    circle_center: Coord,
    circle_radius,
    arc_center: Coord,
    arc_radius,
    arc_start: Coord,
    arc_end: Coord,
    *,
    orientation: str = "ccw",
    central_angle=None,
    allow_full_circle: bool = False,
    objects: Tuple[str, str] = (),
) -> IntersectionResult:
    raw = circle_circle_intersection_result_2d(
        circle_center,
        circle_radius,
        arc_center,
        arc_radius,
        objects=objects,
    )
    if raw.status == "coincident":
        return IntersectionResult("circle_arc", "overlap", objects=objects, metadata={"circle_status": raw.status})
    return _filter_result_by_arcs(
        raw,
        kind="circle_arc",
        arc_domains=((arc_center, arc_radius, arc_start, arc_end, orientation, central_angle, allow_full_circle),),
    )


def arc_arc_intersection_result_2d(
    first_center: Coord,
    first_radius,
    first_start: Coord,
    first_end: Coord,
    second_center: Coord,
    second_radius,
    second_start: Coord,
    second_end: Coord,
    *,
    first_orientation: str = "ccw",
    second_orientation: str = "ccw",
    first_central_angle=None,
    second_central_angle=None,
    first_allow_full_circle: bool = False,
    second_allow_full_circle: bool = False,
    objects: Tuple[str, str] = (),
) -> IntersectionResult:
    raw = circle_circle_intersection_result_2d(
        first_center,
        first_radius,
        second_center,
        second_radius,
        objects=objects,
    )
    first_domain = (
        first_center,
        first_radius,
        first_start,
        first_end,
        first_orientation,
        first_central_angle,
        first_allow_full_circle,
    )
    second_domain = (
        second_center,
        second_radius,
        second_start,
        second_end,
        second_orientation,
        second_central_angle,
        second_allow_full_circle,
    )
    if raw.status != "coincident":
        return _filter_result_by_arcs(raw, kind="arc_arc", arc_domains=(first_domain, second_domain))
    shared = _unique_points(
        point
        for point in (first_start, first_end, second_start, second_end)
        if _point_on_arc_domain(point, first_domain) and _point_on_arc_domain(point, second_domain)
    )
    for point in shared:
        if not _is_arc_endpoint(point, first_start, first_end) or not _is_arc_endpoint(point, second_start, second_end):
            return IntersectionResult("arc_arc", "overlap", objects=objects, points=shared, metadata={"circle_status": "coincident"})
    same_directed_arc = (
        _coords_equal(first_start, second_start)
        and _coords_equal(first_end, second_end)
        and first_orientation == second_orientation
    ) or (
        _coords_equal(first_start, second_end)
        and _coords_equal(first_end, second_start)
        and first_orientation != second_orientation
    )
    if len(shared) == 2 and same_directed_arc:
        return IntersectionResult("arc_arc", "overlap", objects=objects, points=shared, metadata={"circle_status": "coincident"})
    status = "disjoint" if not shared else "unique" if len(shared) == 1 else "two_points"
    return IntersectionResult("arc_arc", status, objects=objects, points=shared, metadata={"circle_status": "coincident"})


def _filter_result_by_arcs(
    raw: IntersectionResult,
    *,
    kind: str,
    arc_domains: tuple[tuple[Any, ...], ...],
) -> IntersectionResult:
    if not raw.is_constructible:
        return IntersectionResult(kind, raw.status, objects=raw.objects, metadata={"source_result": raw.to_dict()})
    retained = tuple(point for point in raw.points if all(_point_on_arc_domain(point, domain) for domain in arc_domains))
    if not retained:
        status = "disjoint"
    elif len(retained) == 1:
        status = "tangent" if raw.status == "tangent" else "unique"
    else:
        status = "two_points"
    return IntersectionResult(
        kind,
        status,
        objects=raw.objects,
        points=retained,
        metadata={"source_result": raw.to_dict(), "arc_filtered": len(raw.points) - len(retained)},
    )


def _point_on_arc_domain(point: Coord, domain: tuple[Any, ...]) -> bool:
    center, radius, start, end, orientation, central_angle, allow_full_circle = domain
    return point_on_arc_domain_2d(
        point,
        center,
        radius,
        start,
        end,
        orientation=orientation,
        central_angle=central_angle,
        allow_full_circle=allow_full_circle,
    )


def _is_arc_endpoint(point: Coord, start: Coord, end: Coord) -> bool:
    return _coords_equal(point, start) or _coords_equal(point, end)


def _unique_points(points) -> Tuple[Coord, ...]:
    unique: list[Coord] = []
    for point in points:
        if not any(_coords_equal(point, existing) for existing in unique):
            unique.append(point)
    return tuple(sorted(unique, key=cmp_to_key(_compare_points_yx)))


def _coords_equal(left: Coord, right: Coord) -> bool:
    return len(left) == len(right) and all(sp.simplify(a - b) == 0 for a, b in zip(left, right))


def select_intersection_points(
    result: IntersectionResult,
    select: str = "all",
    *,
    reference: Coord | None = None,
) -> Tuple[Coord, ...]:
    if select == "all":
        return result.points
    if not result.points:
        return ()
    if select == "first":
        return (result.points[0],)
    if select == "second":
        return result.points[1:2]
    if select == "upper":
        return (max(result.points, key=cmp_to_key(_compare_points_yx)),)
    if select == "lower":
        return (min(result.points, key=cmp_to_key(_compare_points_yx)),)
    if select == "left":
        return (min(result.points, key=cmp_to_key(_compare_points_xy)),)
    if select == "right":
        return (max(result.points, key=cmp_to_key(_compare_points_xy)),)
    if select in {"nearest", "farthest"}:
        if reference is None:
            raise ValueError(f"Intersection selection {select!r} requires a reference point")
        ordered = sorted(
            result.points,
            key=cmp_to_key(lambda left, right: _compare_real(norm_sq(sub(left, reference)), norm_sq(sub(right, reference)))),
        )
        return (ordered[0] if select == "nearest" else ordered[-1],)
    raise ValueError(f"Unsupported intersection selection: {select}")


def diagnose_collinearity_2d(
    first: Coord,
    second: Coord,
    third: Coord,
    *,
    tolerance: float = 1e-9,
) -> PredicateDiagnostic:
    """Diagnose near-collinearity without promoting it to an exact fact."""

    _validate_tolerance(tolerance)
    left = sub(second, first)
    right = sub(third, first)
    residual = sp.simplify(cross2(left, right))
    scale = sp.sqrt(norm_sq(left) * norm_sq(right))
    normalized = _normalized_numeric(residual, scale)
    exact = residual == 0
    classification = "exact" if exact else "approximately_degenerate" if normalized is not None and normalized <= tolerance else "separated"
    return PredicateDiagnostic("collinear", classification, exact, str(residual), normalized, tolerance)


def diagnose_line_circle_tangency_2d(
    line_point: Coord,
    line_dir: Coord,
    center: Coord,
    radius,
    *,
    tolerance: float = 1e-9,
) -> PredicateDiagnostic:
    """Diagnose a near-tangent discriminant without changing exact topology."""

    _validate_tolerance(tolerance)
    radius = sp.sympify(radius)
    vector = sub(line_point, center)
    a = sp.simplify(norm_sq(line_dir))
    b = sp.simplify(2 * dot(vector, line_dir))
    c = sp.simplify(norm_sq(vector) - radius ** 2)
    residual = sp.simplify(b ** 2 - 4 * a * c)
    scale = sp.Abs(b ** 2) + sp.Abs(4 * a * c)
    normalized = _normalized_numeric(residual, scale)
    exact = a != 0 and residual == 0
    classification = "exact" if exact else "approximately_degenerate" if a != 0 and normalized is not None and normalized <= tolerance else "separated"
    return PredicateDiagnostic("line_circle_tangent", classification, exact, str(residual), normalized, tolerance)


def _validate_tolerance(tolerance: float) -> None:
    if not isinstance(tolerance, (int, float)) or tolerance <= 0:
        raise ValueError("Diagnostic tolerance must be positive")


def _normalized_numeric(residual: sp.Expr, scale: sp.Expr) -> float | None:
    try:
        residual_value = abs(float(sp.N(residual, 30)))
        scale_value = abs(float(sp.N(scale, 30)))
    except (TypeError, ValueError):
        return None
    return residual_value / max(scale_value, 1.0)


def _line_parameter(line_point: Coord, line_dir: Coord, point: Coord) -> sp.Expr:
    denom = norm_sq(line_dir)
    return sp.simplify(dot(sub(point, line_point), line_dir) / denom)


def _sort_by_parameter(points: Tuple[Coord, ...], parameters: Tuple[Dict[str, str], ...], *, key: str) -> tuple[Tuple[Coord, ...], Tuple[Dict[str, str], ...]]:
    pairs = sorted(zip(points, parameters), key=lambda item: float(sp.N(sp.sympify(item[1][key]))))
    return tuple(point for point, _ in pairs), tuple(param for _, param in pairs)


def _coincident_linear_intersection(
    p: Coord,
    r: Coord,
    q: Coord,
    s: Coord,
    *,
    first_domain: str,
    second_domain: str,
    objects: Tuple[str, str],
) -> IntersectionResult:
    denominator = norm_sq(r)
    if sp.simplify(denominator) == 0 or sp.simplify(norm_sq(s)) == 0:
        return IntersectionResult(f"{first_domain}_{second_domain}", "degenerate", objects=objects, metadata={"reason": "zero_direction"})
    origin = sp.simplify(dot(sub(q, p), r) / denominator)
    scale = sp.simplify(dot(s, r) / denominator)
    first_interval = _domain_interval(first_domain, sp.Integer(0), sp.Integer(1))
    second_interval = _domain_interval(second_domain, origin, scale)
    low, high = _intersect_intervals(first_interval, second_interval)
    metadata = {"line_status": "coincident", "overlap_interval": [_bound_text(low, lower=True), _bound_text(high, lower=False)]}
    if low is None and high is None:
        return IntersectionResult(f"{first_domain}_{second_domain}", "overlap", objects=objects, metadata=metadata)
    if low is not None and high is not None:
        comparison = _compare_real(high, low)
    else:
        comparison = 1
    if comparison < 0:
        return IntersectionResult(f"{first_domain}_{second_domain}", "disjoint", objects=objects, metadata=metadata)
    if comparison == 0:
        point = add(p, mul(r, low))
        return IntersectionResult(
            f"{first_domain}_{second_domain}",
            "unique",
            objects=objects,
            points=(point,),
            parameters=({"t": str(low)},),
            metadata=metadata,
        )
    finite_bounds = tuple(bound for bound in (low, high) if bound is not None)
    points = tuple(add(p, mul(r, bound)) for bound in finite_bounds)
    return IntersectionResult(f"{first_domain}_{second_domain}", "overlap", objects=objects, points=points, metadata=metadata)


def _validate_linear_domain(domain: str) -> None:
    if domain not in {"line", "ray", "segment"}:
        raise ValueError(f"Unsupported linear domain: {domain}")


def _parameter_in_domain(value: sp.Expr, domain: str) -> bool:
    if domain == "line":
        return True
    if domain == "ray":
        return _compare_real(value, sp.Integer(0)) >= 0
    return _compare_real(value, sp.Integer(0)) >= 0 and _compare_real(value, sp.Integer(1)) <= 0


def _domain_interval(domain: str, origin: sp.Expr, scale: sp.Expr) -> tuple[sp.Expr | None, sp.Expr | None]:
    if domain == "line":
        return None, None
    if domain == "ray":
        return (origin, None) if _compare_real(scale, sp.Integer(0)) > 0 else (None, origin)
    return _ordered_pair(origin, sp.simplify(origin + scale))


def _intersect_intervals(
    first: tuple[sp.Expr | None, sp.Expr | None],
    second: tuple[sp.Expr | None, sp.Expr | None],
) -> tuple[sp.Expr | None, sp.Expr | None]:
    first_low, first_high = first
    second_low, second_high = second
    low = second_low if first_low is None else first_low if second_low is None else _symbolic_max(first_low, second_low)
    high = second_high if first_high is None else first_high if second_high is None else _symbolic_min(first_high, second_high)
    return low, high


def _bound_text(value: sp.Expr | None, *, lower: bool) -> str:
    if value is None:
        return "-inf" if lower else "inf"
    return str(value)


def _ordered_pair(left: sp.Expr, right: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    return (left, right) if _compare_real(left, right) <= 0 else (right, left)


def _symbolic_min(left: sp.Expr, right: sp.Expr) -> sp.Expr:
    return left if _compare_real(left, right) <= 0 else right


def _symbolic_max(left: sp.Expr, right: sp.Expr) -> sp.Expr:
    return left if _compare_real(left, right) >= 0 else right


def _compare_real(left: sp.Expr, right: sp.Expr) -> int:
    difference = sp.simplify(left - right)
    if difference == 0 or difference.is_zero is True:
        return 0
    if difference.is_negative is True:
        return -1
    if difference.is_positive is True:
        return 1
    numeric = float(sp.N(difference))
    if abs(numeric) <= 1e-12:
        return 0
    return -1 if numeric < 0 else 1


def _compare_points_yx(left: Coord, right: Coord) -> int:
    y_comparison = _compare_real(left[1], right[1])
    return y_comparison if y_comparison else _compare_real(left[0], right[0])


def _compare_points_xy(left: Coord, right: Coord) -> int:
    x_comparison = _compare_real(left[0], right[0])
    return x_comparison if x_comparison else _compare_real(left[1], right[1])


def _is_negative(value: sp.Expr) -> bool:
    simplified = sp.simplify(value)
    if simplified.is_number:
        return bool(sp.N(simplified) < 0)
    return False
