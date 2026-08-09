"""Check whether buildable semantic relations are covered by instructions."""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

from .relation_object_normalizer import RelationObjectNormalizer


class InstructionCoverageAnalyzer:
    """Measure conversion from relations to builder constraints."""

    BUILDABLE_RELATION_TYPES = {"length", "parallel", "perpendicular", "right_angle"}

    def __init__(self) -> None:
        self.object_normalizer = RelationObjectNormalizer()

    def analyze(self, formal: Dict[str, Any], instruction_plan: Dict[str, Any]) -> Dict[str, Any]:
        required = self._required_constraints(formal.get("relations") or [])
        actual = self._actual_constraints((instruction_plan.get("constraint_graph") or {}).get("constraints") or [])
        covered = sorted(req for req in required if req in actual)
        missing = sorted(req for req in required if req not in actual)
        return {
            "num_required_constraints": len(required),
            "num_covered_constraints": len(covered),
            "num_missing_constraints": len(missing),
            "coverage": (len(covered) / len(required)) if required else None,
            "covered_constraints": [self._constraint_payload(item) for item in covered],
            "missing_constraints": [self._constraint_payload(item) for item in missing],
        }

    def _required_constraints(self, relations: List[Dict[str, Any]]) -> Set[Tuple[Any, ...]]:
        out: Set[Tuple[Any, ...]] = set()
        for rel in relations:
            rtype = rel.get("type")
            if rtype == "length":
                line = self._line_from_relation(rel)
                if line:
                    value = str(rel.get("value") or "").strip().rstrip(".")
                    out.add(("length", line, value))
            elif rtype in {"parallel", "perpendicular"}:
                lines = self._lines_from_relation(rel)
                if len(lines) >= 2:
                    out.add((rtype, tuple(sorted(lines[:2]))))
            elif rtype == "right_angle":
                lines = self._lines_from_relation(rel)
                if len(lines) >= 2:
                    out.add(("perpendicular", tuple(sorted(lines[:2]))))
        return out

    def _actual_constraints(self, constraints: List[Dict[str, Any]]) -> Set[Tuple[Any, ...]]:
        out: Set[Tuple[Any, ...]] = set()
        for constraint in constraints:
            ctype = constraint.get("type")
            if ctype == "length":
                line = self._line_id(constraint.get("object"))
                if line:
                    value = str(constraint.get("value") or "").strip().rstrip(".")
                    out.add(("length", line, value))
            elif ctype in {"parallel", "perpendicular"}:
                lines = [self._line_id(item) for item in constraint.get("lines") or []]
                lines = [line for line in lines if line]
                if len(lines) >= 2:
                    out.add((ctype, tuple(sorted(lines[:2]))))
        return out

    def _line_from_relation(self, rel: Dict[str, Any]) -> str:
        return self.object_normalizer.line_id_from_relation(rel)

    def _lines_from_relation(self, rel: Dict[str, Any]) -> List[str]:
        return self.object_normalizer.lines_from_relation(rel)

    def _line_id(self, value: Any) -> str:
        return self.object_normalizer.line_id(value)

    def _constraint_payload(self, item: Tuple[Any, ...]) -> Dict[str, Any]:
        if item[0] == "length":
            return {"type": item[0], "object": item[1], "value": item[2]}
        return {"type": item[0], "lines": list(item[1])}
