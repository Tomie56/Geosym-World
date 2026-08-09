"""Canonical SymPy coordinate reconstruction from normalized constraints."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple
import math

import sympy as sp

from geosym_world.schema.problem_normalize import expr_payload, line_id, make_line, make_point, normalize_id, parse_expr


class CanonicalBuildError(ValueError):
    """Raised when a constraint graph cannot be canonically built."""


class CanonicalBuilder:
    """Build a canonical GeoSym geometry from a normalized constraint graph.

    This is deliberately template-first. General symbolic reconstruction is a
    later extension; v0 supports common triangle/segment cases needed by smoke
    tests and early real-data adapter work.
    """

    def build(self, constraint_graph: Dict[str, Any]) -> Dict[str, Any]:
        objects = constraint_graph.get("objects") or {}
        constraints = list(constraint_graph.get("constraints") or [])
        points = [normalize_id(p) for p in objects.get("points") or []]
        raw_lines = objects.get("lines") or []
        lines = self._normalize_lines(raw_lines)
        if not points:
            points = sorted({p for line in lines for p in line})
        if not points:
            raise CanonicalBuildError("No points found in constraint graph")
        self._assert_buildable(points, lines, constraints, constraint_graph)

        coords = self._assign_coordinates(points, lines, constraints)
        point_payloads = [make_point(pid, coords[pid][0], coords[pid][1], role="vertex") for pid in sorted(coords)]
        line_payloads = [
            make_line(line_id(start, end), start, end, role="edge")
            for start, end in lines
            if start in coords and end in coords
        ]
        relations = self._relations_from_constraints(constraints)
        return {
            "points": point_payloads,
            "lines": line_payloads,
            "arcs": [],
            "entities": self._entities_from_lines(lines),
            "relations": relations,
            "quantities": [],
            "raw": {"constraint_graph": constraint_graph},
        }

    def _assert_buildable(
        self,
        points: List[str],
        lines: List[Tuple[str, str]],
        constraints: List[Dict[str, Any]],
        constraint_graph: Dict[str, Any],
    ) -> None:
        if len(lines) == 0:
            raise CanonicalBuildError("No line objects found; semantic-only extraction is not buildable")
        supported = {"length", "perpendicular", "parallel"}
        unsupported = sorted({str(c.get("type")) for c in constraints if c.get("type") not in supported})
        if unsupported:
            raise CanonicalBuildError(f"Unsupported constraint types for canonical build: {unsupported}")
        extraction = constraint_graph.get("extraction") or {}
        relation_text = self._semantic_relation_text(extraction)
        object_text = self._semantic_object_text(extraction)
        semantic_markers = [
            "arc",
            "chord",
            "circle",
            "tangent",
            "inscribed",
            "similar",
            "congruent",
            "bisect",
            "intersecting chords",
            "theorem",
        ]
        if any(marker in relation_text or marker in object_text for marker in semantic_markers):
            # The current builder has no circle/arc/theorem construction kernel.
            # Building from only surviving segment lengths would fabricate a
            # misleading sketch.
            raise CanonicalBuildError("Extraction contains circle/arc/theorem semantics unsupported by canonical builder")
        if len(points) > 4 and len(lines) < len(points) - 1:
            raise CanonicalBuildError("Too few line constraints to place all points reliably")

    def _semantic_relation_text(self, extraction: Dict[str, Any]) -> str:
        chunks: List[str] = []
        chunks.extend(str(item) for item in extraction.get("relations_text") or [])
        for relation in extraction.get("relations") or []:
            if isinstance(relation, dict):
                chunks.append(str(relation.get("type", "")))
                chunks.append(str(relation.get("raw", "")))
            else:
                chunks.append(str(relation))
        return " ".join(chunks).lower()

    def _semantic_object_text(self, extraction: Dict[str, Any]) -> str:
        objects = extraction.get("objects") or []
        chunks: List[str] = []
        if isinstance(objects, dict):
            for value in objects.values():
                if isinstance(value, list):
                    chunks.extend(str(item) for item in value if item)
                elif value:
                    chunks.append(str(value))
        elif isinstance(objects, list):
            chunks.extend(str(item) for item in objects if item)
        elif objects:
            chunks.append(str(objects))
        return " ".join(chunks).lower()

    def _normalize_lines(self, raw_lines: List[Any]) -> List[Tuple[str, str]]:
        lines: List[Tuple[str, str]] = []
        for item in raw_lines:
            if isinstance(item, str):
                clean = normalize_id(item)
                if len(clean) >= 2:
                    lines.append((clean[0], clean[1]))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                lines.append((normalize_id(item[0]), normalize_id(item[1])))
        out: List[Tuple[str, str]] = []
        seen = set()
        for start, end in lines:
            if not start or not end or start == end:
                continue
            key = tuple(sorted((start, end)))
            if key not in seen:
                seen.add(key)
                out.append((start, end))
        return out

    def _assign_coordinates(
        self,
        points: List[str],
        lines: List[Tuple[str, str]],
        constraints: List[Dict[str, Any]],
    ) -> Dict[str, Tuple[sp.Expr, sp.Expr]]:
        # Common right triangle: AB length, BC length, AB perpendicular BC.
        right = self._find_right_triangle(points, constraints)
        if right:
            a, b, c, ab_len, bc_len = right
            return {
                a: (sp.Integer(0), sp.Integer(0)),
                b: (ab_len, sp.Integer(0)),
                c: (ab_len, bc_len),
            }

        return self._layout_graph(points, lines, constraints)

    def _layout_graph(
        self,
        points: List[str],
        lines: List[Tuple[str, str]],
        constraints: List[Dict[str, Any]],
    ) -> Dict[str, Tuple[sp.Expr, sp.Expr]]:
        """Heuristic 2D canonical layout for partially parsed real problems.

        This is not a theorem solver. It avoids the old degenerate fallback that
        placed every point on the x-axis, while honoring simple length and
        perpendicular constraints when possible.
        """
        adjacency: Dict[str, List[str]] = {pid: [] for pid in points}
        for start, end in lines:
            adjacency.setdefault(start, []).append(end)
            adjacency.setdefault(end, []).append(start)

        perp_dirs = self._perpendicular_direction_hints(constraints)
        coords: Dict[str, Tuple[sp.Expr, sp.Expr]] = {}
        component_index = 0
        for root in points:
            if root in coords:
                continue
            offset_x = sp.Integer(component_index * 8)
            coords[root] = (offset_x, sp.Integer(0))
            direction: Dict[Tuple[str, str], Tuple[sp.Expr, sp.Expr]] = {}
            queue = deque([root])
            used_dirs: Dict[str, int] = {root: 0}
            while queue:
                current = queue.popleft()
                neighbors = adjacency.get(current, [])
                for neighbor_idx, neighbor in enumerate(neighbors):
                    if neighbor in coords:
                        continue
                    vec = self._direction_for_edge(
                        current,
                        neighbor,
                        direction,
                        perp_dirs,
                        used_dirs.get(current, neighbor_idx),
                        adjacency,
                    )
                    length = self._length_constraint(current, neighbor, constraints) or sp.Integer(1)
                    cx, cy = coords[current]
                    coords[neighbor] = (sp.simplify(cx + vec[0] * length), sp.simplify(cy + vec[1] * length))
                    edge_key = tuple(sorted((current, neighbor)))
                    direction[edge_key] = vec
                    used_dirs[neighbor] = used_dirs.get(current, neighbor_idx) + 1
                    queue.append(neighbor)

            # Isolated points: put them on a small circle in this component.
            isolated = [pid for pid in points if pid not in coords and not adjacency.get(pid)]
            for idx, pid in enumerate(isolated):
                angle = 2 * math.pi * idx / max(len(isolated), 1)
                coords[pid] = (
                    sp.N(offset_x + 2 * math.cos(angle), 8),
                    sp.N(2 * math.sin(angle), 8),
                )
            component_index += 1

        return coords

    def _perpendicular_direction_hints(self, constraints: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        hints: Dict[str, List[str]] = {}
        for constraint in constraints:
            if constraint.get("type") != "perpendicular":
                continue
            lines = [normalize_id(item) for item in constraint.get("lines") or []]
            if len(lines) < 2:
                continue
            hints.setdefault(lines[0], []).append(lines[1])
            hints.setdefault(lines[1], []).append(lines[0])
        return hints

    def _direction_for_edge(
        self,
        start: str,
        end: str,
        known_directions: Dict[Tuple[str, str], Tuple[sp.Expr, sp.Expr]],
        perp_dirs: Dict[str, List[str]],
        fallback_idx: int,
        adjacency: Dict[str, List[str]],
    ) -> Tuple[sp.Expr, sp.Expr]:
        current_line = line_id(start, end)
        reverse_line = line_id(end, start)
        for other_line in perp_dirs.get(current_line, []) + perp_dirs.get(reverse_line, []):
            other_key = tuple(sorted((other_line[0], other_line[1]))) if len(other_line) >= 2 else None
            if other_key and other_key in known_directions:
                ox, oy = known_directions[other_key]
                return (-oy, ox)
        # If this is a simple chain and the current vertex already has one
        # placed incident edge, continue in the same direction. Branches still
        # spread out below to avoid collapse.
        if len(adjacency.get(start, [])) <= 2:
            for key, vec in known_directions.items():
                if start in key:
                    return vec
        # Spread unknown branches around instead of putting everything on x-axis.
        base_dirs = [
            (sp.Integer(1), sp.Integer(0)),
            (sp.Integer(0), sp.Integer(1)),
            (sp.Integer(-1), sp.Integer(0)),
            (sp.Integer(0), sp.Integer(-1)),
            (sp.sqrt(2) / 2, sp.sqrt(2) / 2),
            (-sp.sqrt(2) / 2, sp.sqrt(2) / 2),
            (-sp.sqrt(2) / 2, -sp.sqrt(2) / 2),
            (sp.sqrt(2) / 2, -sp.sqrt(2) / 2),
        ]
        return base_dirs[fallback_idx % len(base_dirs)]

    def _find_right_triangle(
        self,
        points: List[str],
        constraints: List[Dict[str, Any]],
    ) -> Optional[Tuple[str, str, str, sp.Expr, sp.Expr]]:
        for constraint in constraints:
            if constraint.get("type") != "perpendicular":
                continue
            lines = constraint.get("lines") or []
            if len(lines) < 2:
                continue
            l1 = normalize_id(lines[0])
            l2 = normalize_id(lines[1])
            if len(l1) < 2 or len(l2) < 2:
                continue
            shared = set(l1[:2]) & set(l2[:2])
            if len(shared) != 1:
                continue
            b = sorted(shared)[0]
            a = l1[0] if l1[1] == b else l1[1]
            c = l2[0] if l2[1] == b else l2[1]
            len_ab = self._length_constraint(a, b, constraints)
            len_bc = self._length_constraint(b, c, constraints)
            if len_ab is not None and len_bc is not None:
                return a, b, c, len_ab, len_bc
        return None

    def _length_constraint(self, p1: str, p2: str, constraints: List[Dict[str, Any]]) -> Optional[sp.Expr]:
        wanted = {line_id(p1, p2), line_id(p2, p1)}
        for constraint in constraints:
            if constraint.get("type") != "length":
                continue
            obj = normalize_id(constraint.get("object"))
            if obj in wanted:
                parsed = parse_expr(constraint.get("value"))
                if parsed is not None:
                    return parsed
        return None

    def _relations_from_constraints(self, constraints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        relations: List[Dict[str, Any]] = []
        for constraint in constraints:
            ctype = constraint.get("type")
            if ctype in {"perpendicular", "parallel"}:
                relations.append({
                    "type": ctype,
                    "lines": [normalize_id(item) for item in constraint.get("lines", [])],
                    "source": constraint.get("source", "constraint"),
                    "extracted_by": constraint.get("extracted_by"),
                    "confidence": constraint.get("confidence"),
                })
            elif ctype == "length":
                relations.append({
                    "type": "length",
                    "object": normalize_id(constraint.get("object")),
                    "value": str(constraint.get("value")),
                    "source": constraint.get("source", "constraint"),
                    "extracted_by": constraint.get("extracted_by"),
                    "confidence": constraint.get("confidence"),
                })
        return relations

    def _entities_from_lines(self, lines: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        vertices = sorted({p for line in lines for p in line})
        if len(vertices) >= 3:
            return [{
                "id": "entity_0",
                "type": "polygon",
                "vertices": vertices,
                "lines": [line_id(start, end) for start, end in lines],
            }]
        return []
