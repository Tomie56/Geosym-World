"""Model-free schema, geometry, and quantity verification."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import sympy as sp

from .problem_normalize import expressions_equivalent, get_answer_expr, parse_expr
from .relation_normalization import normalize_relation_type, normalize_target_type, RELATION_TYPES, TARGET_TYPES
from .problem import GeoSymProblemState, VerificationReport


PointMap = Dict[str, Tuple[sp.Expr, sp.Expr]]


class GeoSymVerifier:
    """Verifier for GeoSymProblemState.

    The verifier deliberately stays model-free. It checks references and simple
    coordinate-backed relations. It can be extended with theorem templates later.
    """

    def verify(self, state: GeoSymProblemState) -> VerificationReport:
        report = VerificationReport()
        build_error = (state.geometry.raw or {}).get("build_error")
        if build_error:
            report.add_issue(
                "geometry.build_failed",
                f"Canonical geometry build failed: {build_error}",
                location="geometry.raw.build_error",
            )
        point_map = self._point_map(state, report)
        line_map = self._line_map(state, report, point_map)
        self._verify_arcs(state, report, point_map)
        self._verify_entities(state, report, point_map, line_map)
        self._verify_relations(state, report, point_map, line_map)
        self._verify_target(state, report, point_map, line_map)
        self._verify_derivation(state, report)
        return report

    def _point_map(self, state: GeoSymProblemState, report: VerificationReport) -> PointMap:
        out: PointMap = {}
        for idx, point in enumerate(state.geometry.points):
            pid = point.get("id")
            if not pid:
                report.add_issue("schema.point_missing_id", "Point is missing id", location=f"points[{idx}]")
                continue
            if pid in out:
                report.add_issue("schema.duplicate_point", f"Duplicate point id: {pid}", location=f"points[{idx}]")
                continue
            x = parse_expr((point.get("x") or {}).get("expr"))
            y = parse_expr((point.get("y") or {}).get("expr"))
            if x is None or y is None:
                report.add_issue("schema.point_bad_coord", f"Point {pid} has invalid coordinates", location=pid)
                continue
            out[pid] = (x, y)
        return out

    def _line_map(
        self,
        state: GeoSymProblemState,
        report: VerificationReport,
        point_map: PointMap,
    ) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for idx, line in enumerate(state.geometry.lines):
            lid = line.get("id")
            if not lid:
                report.add_issue("schema.line_missing_id", "Line is missing id", location=f"lines[{idx}]")
                continue
            if lid in out:
                report.add_issue("schema.duplicate_line", f"Duplicate line id: {lid}", location=f"lines[{idx}]")
                continue
            start = line.get("start_point_id")
            end = line.get("end_point_id")
            if start not in point_map:
                report.add_issue("schema.line_bad_start", f"Line {lid} start point does not exist: {start}", location=lid)
            if end not in point_map:
                report.add_issue("schema.line_bad_end", f"Line {lid} end point does not exist: {end}", location=lid)
            if start in point_map and end in point_map:
                if self._same_point(point_map[start], point_map[end]):
                    report.add_issue("geometry.degenerate_line", f"Line {lid} is degenerate", location=lid)
            out[lid] = line
        return out

    def _verify_arcs(self, state: GeoSymProblemState, report: VerificationReport, point_map: PointMap) -> None:
        for idx, arc in enumerate(state.geometry.arcs):
            aid = arc.get("id", f"arcs[{idx}]")
            for key in ["center_point_id", "start_point_id", "end_point_id"]:
                value = arc.get(key)
                if value and value not in point_map:
                    report.add_issue("schema.arc_bad_point", f"Arc {aid} references missing point {value}", location=aid)

    def _verify_entities(
        self,
        state: GeoSymProblemState,
        report: VerificationReport,
        point_map: PointMap,
        line_map: Dict[str, Dict[str, Any]],
    ) -> None:
        for idx, entity in enumerate(state.geometry.entities):
            eid = entity.get("id", f"entities[{idx}]")
            for point_id in entity.get("vertices") or []:
                if isinstance(point_id, str) and point_id not in point_map:
                    report.add_issue("schema.entity_bad_vertex", f"Entity {eid} references missing point {point_id}", location=eid)
            for line_id in entity.get("lines") or []:
                if isinstance(line_id, str) and line_id not in line_map:
                    report.add_issue("schema.entity_bad_line", f"Entity {eid} references missing line {line_id}", location=eid)

    def _verify_relations(
        self,
        state: GeoSymProblemState,
        report: VerificationReport,
        point_map: PointMap,
        line_map: Dict[str, Dict[str, Any]],
    ) -> None:
        for idx, relation in enumerate(state.geometry.relations):
            rtype = normalize_relation_type(str(relation.get("type", "")))
            location = f"relations[{idx}]"
            if rtype not in RELATION_TYPES:
                report.add_issue("schema.unknown_relation", f"Unknown relation type: {rtype}", location=location)
                continue
            if rtype == "perpendicular":
                lines = relation.get("lines") or relation.get("args") or []
                if len(lines) >= 2 and self._lines_exist(lines[:2], line_map, report, location):
                    if not self._is_perpendicular(line_map[lines[0]], line_map[lines[1]], point_map):
                        report.add_issue("geometry.relation_false", f"Lines {lines[0]} and {lines[1]} are not perpendicular", location=location)
            elif rtype == "parallel":
                lines = relation.get("lines") or relation.get("args") or []
                if len(lines) >= 2 and self._lines_exist(lines[:2], line_map, report, location):
                    if not self._is_parallel(line_map[lines[0]], line_map[lines[1]], point_map):
                        report.add_issue("geometry.relation_false", f"Lines {lines[0]} and {lines[1]} are not parallel", location=location)
            elif rtype == "collinear":
                points = relation.get("points") or relation.get("args") or []
                if len(points) >= 3 and self._points_exist(points, point_map, report, location):
                    if not self._is_collinear(points, point_map):
                        report.add_issue("geometry.relation_false", f"Points are not collinear: {points}", location=location)
            elif rtype == "length":
                obj = relation.get("object")
                value = relation.get("value")
                if obj in line_map and value is not None:
                    actual = self._line_length(line_map[obj], point_map)
                    if actual is not None and not expressions_equivalent(actual, value):
                        report.add_issue("quantity.length_mismatch", f"Length relation mismatch for {obj}", location=location, actual=str(actual), expected=str(value))

    def _verify_target(
        self,
        state: GeoSymProblemState,
        report: VerificationReport,
        point_map: PointMap,
        line_map: Dict[str, Dict[str, Any]],
    ) -> None:
        if state.target is None:
            return
        target_type = normalize_target_type(state.target.type)
        if target_type not in TARGET_TYPES:
            report.add_issue(
                "schema.unknown_target",
                f"Unknown target type: {target_type}",
                severity="warning",
                location="target",
            )
        obj = state.target.object
        if obj and target_type == "length" and obj not in line_map:
            report.add_issue("schema.target_missing_object", f"Target line does not exist: {obj}", location="target.object")
        if target_type == "length" and obj in line_map and state.target.answer:
            actual = self._line_length(line_map[obj], point_map)
            expected = get_answer_expr(state.target.answer)
            if actual is not None and expected is not None:
                answer_valid = expressions_equivalent(actual, expected)
                report.answer_valid = answer_valid
                if not answer_valid:
                    report.add_issue("answer.mismatch", f"Target answer mismatch for {obj}", location="target.answer", actual=str(actual), expected=expected)

    def _verify_derivation(self, state: GeoSymProblemState, report: VerificationReport) -> None:
        known_ids = set()
        for item in state.proof.known_conditions:
            if item.get("id"):
                known_ids.add(item["id"])
        for item in state.proof.derivation_trace:
            if item.get("step_id"):
                known_ids.add(item["step_id"])
        for idx, step in enumerate(state.proof.derivation_trace):
            for dep in step.get("depends_on") or []:
                if dep not in known_ids:
                    report.add_issue("derivation.missing_dependency", f"Missing derivation dependency: {dep}", location=f"derivation_trace[{idx}]")

    def _lines_exist(
        self,
        line_ids: Iterable[str],
        line_map: Dict[str, Dict[str, Any]],
        report: VerificationReport,
        location: str,
    ) -> bool:
        ok = True
        for line_id in line_ids:
            if line_id not in line_map:
                ok = False
                report.add_issue("schema.relation_bad_line", f"Relation references missing line {line_id}", location=location)
        return ok

    def _points_exist(self, point_ids: Iterable[str], point_map: PointMap, report: VerificationReport, location: str) -> bool:
        ok = True
        for point_id in point_ids:
            if point_id not in point_map:
                ok = False
                report.add_issue("schema.relation_bad_point", f"Relation references missing point {point_id}", location=location)
        return ok

    def _same_point(self, left: Tuple[sp.Expr, sp.Expr], right: Tuple[sp.Expr, sp.Expr]) -> bool:
        return sp.simplify(left[0] - right[0]) == 0 and sp.simplify(left[1] - right[1]) == 0

    def _line_vector(self, line: Dict[str, Any], point_map: PointMap) -> Optional[Tuple[sp.Expr, sp.Expr]]:
        start = line.get("start_point_id")
        end = line.get("end_point_id")
        if start not in point_map or end not in point_map:
            return None
        sx, sy = point_map[start]
        ex, ey = point_map[end]
        return sp.simplify(ex - sx), sp.simplify(ey - sy)

    def _line_length(self, line: Dict[str, Any], point_map: PointMap) -> Optional[sp.Expr]:
        vec = self._line_vector(line, point_map)
        if vec is None:
            return None
        return sp.simplify(sp.sqrt(vec[0] ** 2 + vec[1] ** 2))

    def _is_perpendicular(self, l1: Dict[str, Any], l2: Dict[str, Any], point_map: PointMap) -> bool:
        v1 = self._line_vector(l1, point_map)
        v2 = self._line_vector(l2, point_map)
        if v1 is None or v2 is None:
            return False
        return sp.simplify(v1[0] * v2[0] + v1[1] * v2[1]) == 0

    def _is_parallel(self, l1: Dict[str, Any], l2: Dict[str, Any], point_map: PointMap) -> bool:
        v1 = self._line_vector(l1, point_map)
        v2 = self._line_vector(l2, point_map)
        if v1 is None or v2 is None:
            return False
        return sp.simplify(v1[0] * v2[1] - v1[1] * v2[0]) == 0

    def _is_collinear(self, points: List[str], point_map: PointMap) -> bool:
        p0 = point_map[points[0]]
        p1 = point_map[points[1]]
        base = (p1[0] - p0[0], p1[1] - p0[1])
        for pid in points[2:]:
            p = point_map[pid]
            vec = (p[0] - p0[0], p[1] - p0[1])
            if sp.simplify(base[0] * vec[1] - base[1] * vec[0]) != 0:
                return False
        return True
