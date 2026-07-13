"""Small exact-geometry helpers backed by SymPy."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import sympy as sp

from .errors import DegenerateGeometryError

Coord = Tuple[sp.Expr, ...]


def expr(value) -> sp.Expr:
    return sp.sympify(value)


def coords(values: Iterable) -> Coord:
    return tuple(sp.sympify(v) for v in values)


def simplify_coord(value: sp.Expr) -> sp.Expr:
    return sp.simplify(value)


def simplify_coords(values: Sequence[sp.Expr]) -> Coord:
    return tuple(sp.simplify(v) for v in values)


def add(a: Coord, b: Coord) -> Coord:
    _same_dim(a, b)
    return simplify_coords(x + y for x, y in zip(a, b))


def sub(a: Coord, b: Coord) -> Coord:
    _same_dim(a, b)
    return simplify_coords(x - y for x, y in zip(a, b))


def mul(a: Coord, k) -> Coord:
    k = sp.sympify(k)
    return simplify_coords(x * k for x in a)


def dot(a: Coord, b: Coord) -> sp.Expr:
    _same_dim(a, b)
    return sp.simplify(sum(x * y for x, y in zip(a, b)))


def cross2(a: Coord, b: Coord) -> sp.Expr:
    if len(a) != 2 or len(b) != 2:
        raise ValueError("cross2 expects 2D vectors")
    return sp.simplify(a[0] * b[1] - a[1] * b[0])


def cross3(a: Coord, b: Coord) -> Coord:
    if len(a) != 3 or len(b) != 3:
        raise ValueError("cross3 expects 3D vectors")
    return simplify_coords(
        (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )
    )


def norm_sq(a: Coord) -> sp.Expr:
    return sp.simplify(dot(a, a))


def norm(a: Coord) -> sp.Expr:
    return sp.sqrt(norm_sq(a))


def unit(a: Coord) -> Coord:
    n = norm(a)
    if is_zero(n):
        raise DegenerateGeometryError("Cannot normalize a zero vector")
    return simplify_coords(x / n for x in a)


def distance(a: Coord, b: Coord) -> sp.Expr:
    return sp.simplify(norm(sub(a, b)))


def midpoint(a: Coord, b: Coord) -> Coord:
    _same_dim(a, b)
    return simplify_coords((x + y) / 2 for x, y in zip(a, b))


def rotate2(vec: Coord, angle) -> Coord:
    if len(vec) != 2:
        raise ValueError("rotate2 expects a 2D vector")
    t = sp.sympify(angle)
    c, s = sp.cos(t), sp.sin(t)
    return simplify_coords((vec[0] * c - vec[1] * s, vec[0] * s + vec[1] * c))


def line_intersection_2d(p: Coord, r: Coord, q: Coord, s: Coord) -> Coord:
    if len(p) != 2 or len(q) != 2 or len(r) != 2 or len(s) != 2:
        raise ValueError("line_intersection_2d expects 2D inputs")
    denom = cross2(r, s)
    if is_zero(denom):
        raise DegenerateGeometryError("Parallel or coincident lines do not have a unique intersection")
    t = sp.simplify(cross2(sub(q, p), s) / denom)
    return add(p, mul(r, t))


def line_circle_intersections_2d(line_point: Coord, line_dir: Coord, center: Coord, radius) -> Tuple[Coord, ...]:
    if len(line_point) != 2 or len(line_dir) != 2 or len(center) != 2:
        raise ValueError("line_circle_intersections_2d expects 2D inputs")
    radius = sp.sympify(radius)
    a = norm_sq(line_dir)
    if is_zero(a):
        raise DegenerateGeometryError("Cannot intersect a circle with a zero-direction line")
    f = sub(line_point, center)
    b = 2 * dot(f, line_dir)
    c = norm_sq(f) - radius ** 2
    disc = sp.simplify(b ** 2 - 4 * a * c)
    if _is_negative(disc):
        raise DegenerateGeometryError("Line and circle do not intersect over the real plane")
    sqrt_disc = sp.sqrt(disc)
    t1 = sp.simplify((-b - sqrt_disc) / (2 * a))
    t2 = sp.simplify((-b + sqrt_disc) / (2 * a))
    p1 = add(line_point, mul(line_dir, t1))
    if is_zero(disc):
        return (p1,)
    return (p1, add(line_point, mul(line_dir, t2)))


def circle_circle_intersections_2d(c1: Coord, r1, c2: Coord, r2) -> Tuple[Coord, ...]:
    if len(c1) != 2 or len(c2) != 2:
        raise ValueError("circle_circle_intersections_2d expects 2D centers")
    r1, r2 = sp.sympify(r1), sp.sympify(r2)
    delta = sub(c2, c1)
    d = norm(delta)
    if is_zero(d):
        raise DegenerateGeometryError("Concentric circles do not have finite unique intersections")
    a = sp.simplify((r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d))
    h_sq = sp.simplify(r1 ** 2 - a ** 2)
    if _is_negative(h_sq):
        raise DegenerateGeometryError("Circles do not intersect over the real plane")
    base = add(c1, mul(delta, a / d))
    perp = simplify_coords((-delta[1] / d, delta[0] / d))
    h = sp.sqrt(h_sq)
    p1 = add(base, mul(perp, h))
    if is_zero(h_sq):
        return (p1,)
    return (p1, add(base, mul(perp, -h)))


def tangent_points_from_point_to_circle_2d(point: Coord, center: Coord, radius) -> Tuple[Coord, ...]:
    if len(point) != 2 or len(center) != 2:
        raise ValueError("tangent_points_from_point_to_circle_2d expects 2D inputs")
    radius = sp.sympify(radius)
    d_sq = norm_sq(sub(point, center))
    if is_zero(d_sq - radius ** 2):
        return (point,)
    if _is_negative(d_sq - radius ** 2):
        raise DegenerateGeometryError("Point lies inside the circle and has no real tangent points")
    helper_radius = sp.sqrt(d_sq - radius ** 2)
    return circle_circle_intersections_2d(center, radius, point, helper_radius)


def foot_point_2d(point: Coord, line_point: Coord, line_dir: Coord) -> Coord:
    if len(point) != 2:
        raise ValueError("foot_point_2d expects 2D inputs")
    denom = norm_sq(line_dir)
    if is_zero(denom):
        raise DegenerateGeometryError("Cannot project onto a zero-direction line")
    t = sp.simplify(dot(sub(point, line_point), line_dir) / denom)
    return add(line_point, mul(line_dir, t))


def project_point_to_plane(point: Coord, plane_point: Coord, plane_normal: Coord) -> Coord:
    if len(point) != 3 or len(plane_point) != 3 or len(plane_normal) != 3:
        raise ValueError("project_point_to_plane expects 3D inputs")
    denom = norm_sq(plane_normal)
    if is_zero(denom):
        raise DegenerateGeometryError("Cannot project onto a plane with zero normal")
    signed_scale = sp.simplify(dot(sub(point, plane_point), plane_normal) / denom)
    return sub(point, mul(plane_normal, signed_scale))


def project_vector_to_plane(vector: Coord, plane_normal: Coord) -> Coord:
    if len(vector) != 3 or len(plane_normal) != 3:
        raise ValueError("project_vector_to_plane expects 3D inputs")
    denom = norm_sq(plane_normal)
    if is_zero(denom):
        raise DegenerateGeometryError("Cannot project onto a plane with zero normal")
    normal_component = mul(plane_normal, sp.simplify(dot(vector, plane_normal) / denom))
    return sub(vector, normal_component)


def angle_between(a: Coord, b: Coord) -> sp.Expr:
    if is_zero(norm_sq(a)) or is_zero(norm_sq(b)):
        raise DegenerateGeometryError("Cannot compute an angle with a zero vector")
    if len(a) == 2:
        return sp.simplify(sp.atan2(sp.Abs(cross2(a, b)), dot(a, b)))
    return sp.simplify(sp.acos(dot(a, b) / (norm(a) * norm(b))))


def undirected_angle_between(a: Coord, b: Coord) -> sp.Expr:
    if is_zero(norm_sq(a)) or is_zero(norm_sq(b)):
        raise DegenerateGeometryError("Cannot compute an angle with a zero vector")
    if len(a) == 2:
        return sp.simplify(sp.atan2(sp.Abs(cross2(a, b)), sp.Abs(dot(a, b))))
    return sp.simplify(sp.acos(sp.Abs(dot(a, b)) / (norm(a) * norm(b))))


def polygon_area_2d(points: Sequence[Coord]) -> sp.Expr:
    if len(points) < 3:
        raise DegenerateGeometryError("A polygon needs at least three points")
    area2 = sp.Integer(0)
    for i, p in enumerate(points):
        q = points[(i + 1) % len(points)]
        if len(p) != 2 or len(q) != 2:
            raise ValueError("polygon_area_2d expects 2D points")
        area2 += p[0] * q[1] - p[1] * q[0]
    return sp.simplify(sp.Abs(area2) / 2)


def polygon_area_3d(points: Sequence[Coord]) -> sp.Expr:
    if len(points) < 3:
        raise DegenerateGeometryError("A 3D polygon needs at least three points")
    p0 = points[0]
    if len(p0) != 3:
        raise ValueError("polygon_area_3d expects 3D points")
    area = sp.Integer(0)
    for idx in range(1, len(points) - 1):
        v1 = sub(points[idx], p0)
        v2 = sub(points[idx + 1], p0)
        area += norm(cross3(v1, v2)) / 2
    return sp.simplify(area)


def tetrahedron_volume(a: Coord, b: Coord, c: Coord, d: Coord) -> sp.Expr:
    if any(len(p) != 3 for p in (a, b, c, d)):
        raise ValueError("tetrahedron_volume expects 3D points")
    triple = dot(sub(b, a), cross3(sub(c, a), sub(d, a)))
    return sp.simplify(sp.Abs(triple) / 6)


def polyhedron_volume_by_origin(vertices: dict[str, Coord], faces: Sequence[Sequence[str]]) -> sp.Expr:
    volume6 = sp.Integer(0)
    origin = (sp.Integer(0), sp.Integer(0), sp.Integer(0))
    for face in faces:
        if len(face) < 3:
            continue
        p0 = vertices[face[0]]
        for idx in range(1, len(face) - 1):
            p1 = vertices[face[idx]]
            p2 = vertices[face[idx + 1]]
            volume6 += dot(p0, cross3(p1, p2))
    _ = origin
    return sp.simplify(sp.Abs(volume6) / 6)


def is_zero(value) -> bool:
    return sp.simplify(value) == 0


def is_parallel(a: Coord, b: Coord) -> bool:
    _same_dim(a, b)
    if len(a) == 2:
        return is_zero(cross2(a, b))
    return all(is_zero(v) for v in cross3(a, b))


def is_perpendicular(a: Coord, b: Coord) -> bool:
    _same_dim(a, b)
    return is_zero(dot(a, b))


def point_on_line(point: Coord, anchor: Coord, direction: Coord) -> bool:
    _same_dim(point, anchor)
    _same_dim(point, direction)
    if len(point) == 2:
        return is_zero(cross2(sub(point, anchor), direction))
    return is_parallel(sub(point, anchor), direction)


def point_on_plane(point: Coord, plane_point: Coord, plane_normal: Coord) -> bool:
    if len(point) != 3 or len(plane_point) != 3 or len(plane_normal) != 3:
        raise ValueError("point_on_plane expects 3D inputs")
    return is_zero(dot(sub(point, plane_point), plane_normal))


def line_parallel_to_plane(line_direction: Coord, plane_normal: Coord) -> bool:
    if len(line_direction) != 3 or len(plane_normal) != 3:
        raise ValueError("line_parallel_to_plane expects 3D inputs")
    return is_zero(dot(line_direction, plane_normal))


def segment_plane_intersection_3d(a: Coord, b: Coord, plane_point: Coord, plane_normal: Coord) -> Coord | None:
    if len(a) != 3 or len(b) != 3 or len(plane_point) != 3 or len(plane_normal) != 3:
        raise ValueError("segment_plane_intersection_3d expects 3D inputs")
    da = sp.simplify(dot(sub(a, plane_point), plane_normal))
    db = sp.simplify(dot(sub(b, plane_point), plane_normal))
    if is_zero(da) and is_zero(db):
        return None
    if is_zero(da):
        return a
    if is_zero(db):
        return b
    try:
        if float(sp.N(da)) * float(sp.N(db)) > 0:
            return None
    except Exception:
        if sp.simplify(da * db).is_positive is True:
            return None
    t = sp.simplify(da / (da - db))
    return add(a, mul(sub(b, a), t))


def point_on_segment(point: Coord, a: Coord, b: Coord) -> bool:
    if not point_on_line(point, a, sub(b, a)):
        return False
    for x, lo, hi in zip(point, a, b):
        lo2, hi2 = (lo, hi) if float(sp.N(lo)) <= float(sp.N(hi)) else (hi, lo)
        if float(sp.N(x)) < float(sp.N(lo2)) - 1e-9:
            return False
        if float(sp.N(x)) > float(sp.N(hi2)) + 1e-9:
            return False
    return True


def _same_dim(a: Coord, b: Coord) -> None:
    if len(a) != len(b):
        raise ValueError(f"Dimension mismatch: {len(a)} vs {len(b)}")


def _is_negative(value) -> bool:
    simplified = sp.simplify(value)
    if simplified.is_negative is True:
        return True
    try:
        return float(sp.N(simplified)) < -1e-12
    except Exception:
        return False
