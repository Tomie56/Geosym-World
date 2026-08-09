"""Build formal graph and buildability classification from grounded relations."""

from __future__ import annotations

from typing import Any, Dict, List


class FormalGraphBuilder:
    COORDINATE_TYPES = {"length", "perpendicular", "parallel", "right_angle"}
    THEOREM_TYPES = {
        "angle",
        "arc_measure",
        "area",
        "chord",
        "circle",
        "congruent",
        "incidence",
        "intersecting_chords",
        "perimeter",
        "ratio",
        "right_triangle",
        "similar",
        "triangle",
    }

    def build(self, grounded: Dict[str, Any]) -> Dict[str, Any]:
        accepted = list(grounded.get("accepted_relations") or [])
        uncertain = list(grounded.get("uncertain_relations") or [])
        all_relations = accepted + uncertain
        types = {
            str(rel.get("type", "unknown"))
            for rel in all_relations
            if str(rel.get("type", "unknown")) != "target"
        }
        labels = self._labels_from_bindings(grounded.get("bindings") or [])
        objects = self._objects_from_relations(all_relations, labels)

        coordinate_types = types & self.COORDINATE_TYPES
        theorem_types = types & self.THEOREM_TYPES
        unsupported_types = types - self.COORDINATE_TYPES - self.THEOREM_TYPES

        if not all_relations:
            mode = "failed"
            reason = "no grounded or semantic relations"
        elif theorem_types and not coordinate_types:
            mode = "theorem_only"
            reason = f"contains only theorem/circle/arc relation types: {sorted(theorem_types)}"
        elif coordinate_types and objects.get("lines") and not unsupported_types:
            mode = "coordinate_buildable"
            if theorem_types:
                reason = f"coordinate scaffold is buildable; theorem relations remain unsolved: {sorted(theorem_types)}"
            else:
                reason = "relations are coordinate-buildable"
        else:
            mode = "unsupported"
            reason = f"unsupported relation types: {sorted(unsupported_types or types)}"

        return {
            "mode": mode,
            "reason": reason,
            "coordinate_relation_types": sorted(coordinate_types),
            "theorem_relation_types": sorted(theorem_types),
            "unsupported_relation_types": sorted(unsupported_types),
            "requires_theorem_solver": bool(theorem_types),
            "solver_coverage": self._solver_coverage(
                mode=mode,
                relations=all_relations,
                target=grounded.get("target") or {},
                coordinate_types=coordinate_types,
                theorem_types=theorem_types,
                unsupported_types=unsupported_types,
            ),
            "objects": objects,
            "relations": all_relations,
            "target": grounded.get("target") or {},
            "grounding_summary": {
                "accepted": len(accepted),
                "uncertain": len(uncertain),
                "rejected": len(grounded.get("rejected_relations") or []),
            },
        }

    def _solver_coverage(
        self,
        *,
        mode: str,
        relations: List[Dict[str, Any]],
        target: Dict[str, Any],
        coordinate_types: set[str],
        theorem_types: set[str],
        unsupported_types: set[str],
    ) -> Dict[str, Any]:
        symbolic_variables = sorted(self._symbolic_variables(relations, target))
        theorem_counts = {
            rtype: sum(1 for rel in relations if str(rel.get("type") or "") == rtype)
            for rtype in sorted(theorem_types)
        }
        coordinate_counts = {
            rtype: sum(1 for rel in relations if str(rel.get("type") or "") == rtype)
            for rtype in sorted(coordinate_types)
        }
        if not relations:
            status = "empty"
            next_solver = "relation_extraction"
        elif unsupported_types:
            status = "unsupported_relation_gap"
            next_solver = "relation_schema_or_parser"
        elif theorem_types and coordinate_types:
            status = "coordinate_scaffold_with_theorem_gap"
            next_solver = "theorem_equation_solver" if symbolic_variables else "theorem_solver"
        elif theorem_types:
            status = "theorem_only_gap"
            next_solver = "theorem_equation_solver" if symbolic_variables else "theorem_solver"
        elif mode == "coordinate_buildable":
            status = "coordinate_only"
            next_solver = "coordinate_builder"
        else:
            status = "not_buildable"
            next_solver = "builder_or_relation_parser"
        return {
            "status": status,
            "next_solver": next_solver,
            "coordinate_builder_covered": status == "coordinate_only",
            "has_coordinate_scaffold": bool(coordinate_types),
            "requires_theorem_solver": bool(theorem_types),
            "requires_equation_solver": bool(symbolic_variables and theorem_types),
            "requires_relation_schema_extension": bool(unsupported_types),
            "symbolic_variables": symbolic_variables,
            "theorem_relation_counts": theorem_counts,
            "coordinate_relation_counts": coordinate_counts,
            "unsupported_relation_types": sorted(unsupported_types),
        }

    def _symbolic_variables(self, relations: List[Dict[str, Any]], target: Dict[str, Any]) -> set[str]:
        variables: set[str] = set()
        for item in [target, *relations]:
            if not isinstance(item, dict):
                continue
            for field in ["value_symbol", "value", "object", "raw"]:
                variables.update(self._symbols_in_text(item.get(field)))
            evidence = item.get("evidence") or {}
            if isinstance(evidence, dict):
                for field in ["supporting_text", "text_refs"]:
                    value = evidence.get(field)
                    if isinstance(value, list):
                        for part in value:
                            variables.update(self._symbols_in_text(part))
                    else:
                        variables.update(self._symbols_in_text(value))
        return variables

    def _symbols_in_text(self, value: Any) -> set[str]:
        text = str(value or "")
        variables: set[str] = set()
        for idx, ch in enumerate(text):
            if ch.lower() not in {"x", "y", "z", "t", "r"}:
                continue
            prev_ok = idx == 0 or not text[idx - 1].isalpha()
            next_ok = idx == len(text) - 1 or not text[idx + 1].isalpha()
            if prev_ok and next_ok:
                variables.add(ch.lower())
        return variables

    def _labels_from_bindings(self, bindings: List[Dict[str, Any]]) -> List[str]:
        labels = []
        for binding in bindings:
            if binding.get("grounding_status") == "accepted" and binding.get("label"):
                label = str(binding["label"]).strip()
                if label and label not in labels:
                    labels.append(label)
        return labels

    def _objects_from_relations(self, relations: List[Dict[str, Any]], labels: List[str]) -> Dict[str, Any]:
        points = set(label.upper() for label in labels if self._looks_like_point_label(label))
        lines = set()
        circles = set()
        arcs = set()
        for rel in relations:
            rtype = rel.get("type")
            objs = [str(o) for o in rel.get("objects") or []]
            if rtype in {"length", "chord"}:
                if len(objs) >= 2 and all(len(obj) == 1 and obj.isalnum() for obj in objs[:2]):
                    lines.add((objs[0], objs[1]))
                    points.update(objs[:2])
                for obj in objs:
                    clean = "".join(ch for ch in obj if ch.isalnum())
                    if len(clean) == 2 and clean.isupper():
                        lines.add((clean[0], clean[1]))
                        points.update(clean[:2])
            if rtype in {"parallel", "perpendicular"}:
                for obj in objs:
                    clean = "".join(ch for ch in obj if ch.isalnum())
                    if len(clean) == 2 and clean.isupper():
                        lines.add((clean[0], clean[1]))
                        points.update(clean[:2])
            if rtype == "right_angle":
                for obj in objs[:2]:
                    clean = "".join(ch for ch in obj if ch.isalnum())
                    if len(clean) >= 2:
                        lines.add((clean[0], clean[1]))
                        points.update(clean[:2])
            if rtype == "circle":
                circles.update(objs)
            if rtype == "arc_measure":
                arcs.update(objs)
        return {
            "points": sorted(points),
            "lines": [list(item) for item in sorted(lines)],
            "circles": sorted(circles),
            "arcs": sorted(arcs),
        }

    def _looks_like_point_label(self, label: Any) -> bool:
        clean = "".join(ch for ch in str(label or "").upper() if ch.isalnum())
        if not clean or clean in {"X", "Y", "Z", "R", "L", "W", "H"}:
            return False
        if clean.isdigit():
            return False
        return 1 <= len(clean) <= 3 and clean[0].isalpha()
