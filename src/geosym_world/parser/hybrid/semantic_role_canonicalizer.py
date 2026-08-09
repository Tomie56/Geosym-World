"""Canonicalize VLM geometry role names into buildable symbolic objects."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List
import re


class SemanticRoleCanonicalizer:
    """Repair common VLM role objects before builder conversion.

    Real geometry images often do not label points. VLMs therefore describe
    objects as roles such as ``vertical_leg``, ``horizontal_leg`` or
    ``hypotenuse``. Those are useful semantics, but the GeoSym builder needs
    symbolic segments with shared endpoints. This pass maps a recognized
    right-triangle role frame to a small canonical symbolic frame:

    - first/right/vertical leg -> AB
    - second/base/horizontal leg -> BC
    - hypotenuse -> AC

    It does not infer metric values from pixels and it only fires when the
    semantic relations already indicate a right-triangle-like structure.
    """

    LEG_A = "leg_a"
    LEG_B = "leg_b"
    HYPOTENUSE = "hypotenuse"

    ROLE_SEGMENTS = {
        LEG_A: "AB",
        LEG_B: "BC",
        HYPOTENUSE: "AC",
    }

    def canonicalize(self, semantic: Dict[str, Any]) -> Dict[str, Any]:
        out = deepcopy(semantic or {})
        role_map = self._role_map(out)
        if not role_map:
            return out

        changes: List[Dict[str, Any]] = []
        for rel in out.get("relations") or []:
            if not isinstance(rel, dict):
                continue
            before = self._relation_object_snapshot(rel)
            self._canonicalize_relation(rel, role_map)
            after = self._relation_object_snapshot(rel)
            if before != after:
                changes.append({
                    "kind": "relation",
                    "candidate_id": rel.get("candidate_id") or rel.get("from_candidate"),
                    "type": rel.get("type"),
                    "before": before,
                    "after": after,
                })

        for key in ["target"]:
            if isinstance(out.get(key), dict):
                before = dict(out[key])
                self._canonicalize_target(out[key], role_map)
                if before != out[key]:
                    changes.append({"kind": key, "before": before, "after": dict(out[key])})
        targets = out.get("targets")
        if isinstance(targets, list):
            for target in targets:
                if not isinstance(target, dict):
                    continue
                before = dict(target)
                self._canonicalize_target(target, role_map)
                if before != target:
                    changes.append({"kind": "targets[]", "before": before, "after": dict(target)})

        out["build_instructions"] = self._canonical_build_instructions(out, role_map)

        if changes:
            metadata = dict(out.get("metadata") or {})
            metadata["semantic_role_canonicalizer"] = {
                "used": True,
                "role_map": dict(role_map),
                "num_changes": len(changes),
                "changes": changes[:100],
            }
            out["metadata"] = metadata
        return out

    def _role_map(self, semantic: Dict[str, Any]) -> Dict[str, str]:
        roles = set()
        for rel in semantic.get("relations") or []:
            if not isinstance(rel, dict):
                continue
            for value in self._relation_object_values(rel):
                role = self._classify_role(value)
                if role:
                    roles.add(role)
        for target in [semantic.get("target")] + list(semantic.get("targets") or []):
            if isinstance(target, dict):
                role = self._classify_role(target.get("object"))
                if role:
                    roles.add(role)
        has_right_angle = any(
            isinstance(rel, dict)
            and str(rel.get("type") or "") in {"right_angle", "perpendicular"}
            and any(self._classify_role(value) in {self.LEG_A, self.LEG_B} for value in self._relation_object_values(rel))
            for rel in semantic.get("relations") or []
        )
        if self.HYPOTENUSE in roles and {self.LEG_A, self.LEG_B} & roles:
            pass
        elif has_right_angle and len({self.LEG_A, self.LEG_B} & roles) >= 1:
            pass
        else:
            return {}
        return {role: self.ROLE_SEGMENTS[role] for role in roles if role in self.ROLE_SEGMENTS}

    def _canonicalize_relation(self, rel: Dict[str, Any], role_map: Dict[str, str]) -> None:
        rtype = str(rel.get("type") or "")
        if rtype == "length":
            segment = self._segment_for_value(rel.get("object"), role_map) or self._segment_for_values(rel.get("objects"), role_map)
            if segment:
                rel["object"] = segment
                rel["objects"] = [segment[0], segment[1]]
                self._mark_relation(rel, role_map)
        elif rtype in {"right_angle", "perpendicular", "parallel"}:
            lines = self._lines_for_relation(rel, role_map)
            if len(lines) >= 2:
                rel["lines"] = lines[:2]
                rel["objects"] = lines[:2]
                self._mark_relation(rel, role_map)
        elif rtype == "target":
            segment = self._segment_for_value(rel.get("object"), role_map) or self._segment_for_values(rel.get("objects"), role_map)
            if segment:
                rel["object"] = segment
                rel["objects"] = [segment]
                self._mark_relation(rel, role_map)

    def _canonicalize_target(self, target: Dict[str, Any], role_map: Dict[str, str]) -> None:
        segment = self._segment_for_value(target.get("object"), role_map)
        if segment:
            original = target.get("object")
            target["object"] = segment
            target.setdefault("type", "length")
            metadata = dict(target.get("metadata") or {})
            metadata["original_role_object"] = original
            metadata["semantic_role_canonicalized"] = True
            target["metadata"] = metadata

    def _canonicalize_instruction(self, instruction: Dict[str, Any], role_map: Dict[str, str]) -> Dict[str, Any]:
        inst = deepcopy(instruction)
        action = str(inst.get("action") or "")
        if action in {"add_segment", "set_length"}:
            segment = self._segment_for_values(inst.get("points"), role_map)
            if segment:
                inst["points"] = [segment[0], segment[1]]
        elif action in {"set_parallel", "set_perpendicular"}:
            lines = []
            for line in inst.get("lines") or []:
                lines.append(self._segment_for_value(line, role_map) or str(line))
            inst["lines"] = lines
        elif action == "set_target" and isinstance(inst.get("target"), dict):
            self._canonicalize_target(inst["target"], role_map)
        return inst

    def _canonical_build_instructions(self, semantic: Dict[str, Any], role_map: Dict[str, str]) -> List[Dict[str, Any]]:
        kept: List[Dict[str, Any]] = []
        for inst in semantic.get("build_instructions") or []:
            if not isinstance(inst, dict):
                continue
            if self._instruction_looks_role_derived(inst):
                continue
            kept.append(self._canonicalize_instruction(inst, role_map))

        generated: List[Dict[str, Any]] = []
        for idx, rel in enumerate(semantic.get("relations") or []):
            if not isinstance(rel, dict):
                continue
            if not (rel.get("metadata") or {}).get("semantic_role_canonicalized"):
                continue
            cid = str(rel.get("candidate_id") or rel.get("from_candidate") or f"role_rel_{idx}")
            visual_ids = [str(item) for item in rel.get("visual_ids") or [] if item]
            rtype = str(rel.get("type") or "")
            if rtype == "length":
                segment = self._segment_for_value(rel.get("object"), role_map) or self._segment_for_values(rel.get("objects"), role_map)
                if segment:
                    generated.extend(self._segment_instructions(segment, cid, visual_ids))
                    if rel.get("value") not in {None, ""}:
                        generated.append(self._with_visual_ids({"action": "set_length", "points": [segment[0], segment[1]], "value": str(rel.get("value")), "from_candidate": cid}, visual_ids))
            elif rtype in {"right_angle", "perpendicular", "parallel"}:
                lines = [str(item) for item in rel.get("lines") or [] if len(str(item)) >= 2]
                for line in lines:
                    generated.extend(self._segment_instructions(line, cid, visual_ids))
                if len(lines) >= 2:
                    action = "set_perpendicular" if rtype == "right_angle" else f"set_{rtype}"
                    generated.append(self._with_visual_ids({"action": action, "lines": lines[:2], "from_candidate": cid}, visual_ids))
            elif rtype == "target":
                segment = self._segment_for_value(rel.get("object"), role_map) or self._segment_for_values(rel.get("objects"), role_map)
                target = {"type": "length", "object": segment or rel.get("object"), "value_symbol": str(rel.get("value") or "x"), "from_candidate": cid}
                if segment:
                    generated.extend(self._segment_instructions(segment, cid, visual_ids))
                generated.append(self._with_visual_ids({"action": "set_target", "target": target, "from_candidate": cid}, visual_ids))

        target = semantic.get("target")
        if isinstance(target, dict):
            segment = self._segment_for_value(target.get("object"), role_map)
            if segment:
                cid = str(target.get("candidate_id") or target.get("from_candidate") or "role_target")
                generated.extend(self._segment_instructions(segment, cid, []))
                generated.append({"action": "set_target", "target": dict(target), "from_candidate": cid})

        return self._dedupe_instructions(kept + generated)

    def _segment_instructions(self, segment: str, candidate_id: str, visual_ids: List[str]) -> List[Dict[str, Any]]:
        if len(segment) < 2:
            return []
        return [
            {"action": "add_point", "id": segment[0], "from_candidate": candidate_id},
            {"action": "add_point", "id": segment[1], "from_candidate": candidate_id},
            self._with_visual_ids({"action": "add_segment", "points": [segment[0], segment[1]], "from_candidate": candidate_id}, visual_ids),
        ]

    def _with_visual_ids(self, instruction: Dict[str, Any], visual_ids: List[str]) -> Dict[str, Any]:
        out = dict(instruction)
        if visual_ids:
            out["visual_ids"] = list(visual_ids)
        return out

    def _dedupe_instructions(self, instructions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen = set()
        for inst in instructions:
            key = repr(sorted(inst.items()))
            if key in seen:
                continue
            seen.add(key)
            out.append(inst)
        return out

    def _instruction_looks_role_derived(self, inst: Dict[str, Any]) -> bool:
        for key in ["from_candidate", "candidate_id", "id"]:
            if self._classify_role(inst.get(key)):
                return True
        for key in ["points", "lines"]:
            for value in inst.get(key) or []:
                if self._classify_role(value):
                    return True
        target = inst.get("target")
        if isinstance(target, dict) and self._classify_role(target.get("object")):
            return True
        return False

    def _lines_for_relation(self, rel: Dict[str, Any], role_map: Dict[str, str]) -> List[str]:
        out: List[str] = []
        raw = rel.get("lines") or rel.get("segments") or rel.get("objects") or []
        if isinstance(raw, (str, dict)):
            raw = [raw]
        for value in raw:
            segment = self._segment_for_value(value, role_map)
            if segment and segment not in out:
                out.append(segment)
        if len(out) < 2:
            for key in ["line1", "line_1", "first_line", "lhs", "left_line", "line2", "line_2", "second_line", "rhs", "right_line"]:
                segment = self._segment_for_value(rel.get(key), role_map)
                if segment and segment not in out:
                    out.append(segment)
        return out

    def _segment_for_values(self, values: Any, role_map: Dict[str, str]) -> str:
        if isinstance(values, (list, tuple)):
            if len(values) == 1:
                return self._segment_for_value(values[0], role_map)
            segments = [self._segment_for_value(value, role_map) for value in values]
            segments = [segment for segment in segments if segment]
            if len(segments) == 1:
                return segments[0]
            if len(values) >= 2 and all(self._looks_like_point(value) for value in values[:2]):
                return "".join(str(value).strip().upper()[:1] for value in values[:2])
        return self._segment_for_value(values, role_map)

    def _segment_for_value(self, value: Any, role_map: Dict[str, str]) -> str:
        role = self._classify_role(value)
        if role and role in role_map:
            return role_map[role]
        text = str(value or "").strip().upper()
        clean = "".join(ch for ch in text if ch.isalnum())
        return clean[:2] if len(clean) == 2 and clean.isalpha() else ""

    def _classify_role(self, value: Any) -> str:
        if isinstance(value, dict):
            for key in ["role", "object", "name", "label", "id", "segment", "line"]:
                role = self._classify_role(value.get(key))
                if role:
                    return role
            return ""
        text = str(value or "").lower()
        token = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
        compact = token.replace("_", "")
        if not token:
            return ""
        if any(key in token for key in ["hypotenuse", "hypoten", "斜边"]) or compact in {"xside", "unknownside"}:
            return self.HYPOTENUSE
        if any(key in token for key in ["vertical_leg", "left_leg", "height", "altitude", "upright_leg", "leg_1", "first_leg"]):
            return self.LEG_A
        if any(key in token for key in ["horizontal_leg", "base", "bottom_leg", "leg_2", "second_leg", "horiz"]):
            return self.LEG_B
        if compact in {"verticalleg", "leftleg", "heightleg"} or any(key in compact for key in ["verticalleg", "leftleg", "heightleg"]):
            return self.LEG_A
        if compact in {"horizontalleg", "baseleg", "bottomleg"} or any(key in compact for key in ["horizontalleg", "baseleg", "bottomleg"]):
            return self.LEG_B
        return ""

    def _relation_object_values(self, rel: Dict[str, Any]) -> List[Any]:
        values: List[Any] = []
        for key in ["object", "objects", "lines", "segments", "line1", "line2", "first_line", "second_line"]:
            value = rel.get(key)
            if isinstance(value, list):
                values.extend(value)
            elif value is not None:
                values.append(value)
        return values

    def _relation_object_snapshot(self, rel: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: deepcopy(rel.get(key))
            for key in ["object", "objects", "lines", "segments", "line1", "line2"]
            if key in rel
        }

    def _mark_relation(self, rel: Dict[str, Any], role_map: Dict[str, str]) -> None:
        metadata = dict(rel.get("metadata") or {})
        metadata["semantic_role_canonicalized"] = True
        metadata["role_map"] = dict(role_map)
        rel["metadata"] = metadata

    def _looks_like_point(self, value: Any) -> bool:
        clean = "".join(ch for ch in str(value or "").upper() if ch.isalnum())
        return len(clean) == 1 and clean.isalpha()
