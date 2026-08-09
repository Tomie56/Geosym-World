"""Normalize named geometry objects from VLM/script relation drafts."""

from __future__ import annotations

from typing import Any, Dict, List
import re


STOP_TOKENS = {
    "LINE",
    "LINES",
    "SEGMENT",
    "SEGMENTS",
    "SIDE",
    "SIDES",
    "ANGLE",
    "POINT",
    "POINTS",
    "PARALLEL",
    "PERPENDICULAR",
    "FIRST",
    "SECOND",
    "THIRD",
    "FOURTH",
    "LEG",
    "LEGS",
    "VERTICAL",
    "HORIZONTAL",
    "HYPOTENUSE",
    "HYPOTEN",
    "BASE",
    "HEIGHT",
    "ALTITUDE",
    "HORIZ",
    "UPRIGHT",
    "LEFT",
    "RIGHT",
    "BOTTOM",
    "TOP",
    "UNKNOWN_SIDE",
    "UNKNOWNSIDE",
    "VISIBLE",
    "GIVEN",
    "UNKNOWN",
    "OTHER",
    "ANOTHER",
}


class RelationObjectNormalizer:
    """Convert common relation object variants into named point/line ids.

    This normalizer intentionally works only on symbolic labels such as A, B,
    AB, or CD. It does not interpret pixel coordinates as mathematical
    coordinates.
    """

    def line_from_relation(self, rel: Dict[str, Any]) -> List[str]:
        for key in ["points", "endpoints"]:
            line = self.line_from_value(rel.get(key))
            if line:
                return line
        for key in ["object", "segment", "line", "target"]:
            line = self.line_from_value(rel.get(key))
            if line:
                return line
        objects = rel.get("objects") or []
        if isinstance(objects, list):
            if len(objects) >= 2 and all(self.looks_like_point(item) for item in objects[:2]):
                return [self.point_id(objects[0]), self.point_id(objects[1])]
            for item in objects:
                line = self.line_from_value(item)
                if line:
                    return line
        return []

    def line_id_from_relation(self, rel: Dict[str, Any]) -> str:
        line = self.line_from_relation(rel)
        return "".join(line[:2]) if len(line) >= 2 else ""

    def lines_from_relation(self, rel: Dict[str, Any]) -> List[str]:
        raw = rel.get("lines") or rel.get("segments") or rel.get("objects") or []
        if isinstance(raw, (str, dict)):
            raw = [raw]
        out: List[str] = []
        for item in raw:
            out.extend(self.line_ids_from_value(item))
        if len(out) < 2:
            for key in ["line1", "line_1", "first_line", "lhs", "left_line"]:
                line = self.line_id(rel.get(key))
                if line:
                    out.append(line)
                    break
        if len(out) < 2:
            for key in ["line2", "line_2", "second_line", "rhs", "right_line"]:
                line = self.line_id(rel.get(key))
                if line:
                    out.append(line)
                    break
        if len(out) < 2:
            object_line = self.line_from_value(rel.get("object"))
            if object_line:
                out.append("".join(object_line[:2]))
        deduped: List[str] = []
        seen = set()
        for line in out:
            normalized = self.line_id(line)
            if normalized and normalized not in seen:
                seen.add(normalized)
                deduped.append(normalized)
        return deduped

    def line_from_value(self, value: Any) -> List[str]:
        line = self.line_id(value)
        if len(line) >= 2:
            return [line[0], line[1]]
        return []

    def line_ids_from_value(self, value: Any) -> List[str]:
        if isinstance(value, (list, tuple)):
            if len(value) >= 2 and all(self.looks_like_point(item) for item in value[:2]):
                return [self.line_id(value[:2])]
            out: List[str] = []
            for item in value:
                out.extend(self.line_ids_from_value(item))
            return out
        if isinstance(value, dict):
            for key in ["points", "endpoints"]:
                if value.get(key):
                    return self.line_ids_from_value(value.get(key))
            for key in ["id", "name", "label", "object", "segment", "line"]:
                if value.get(key):
                    return self.line_ids_from_value(value.get(key))
            return []
        text = str(value or "").upper()
        tokens = re.findall(r"[A-Z][A-Z0-9]*", text)
        candidates: List[str] = []
        for token in tokens:
            clean = "".join(ch for ch in token if ch.isalnum())
            if clean in STOP_TOKENS:
                continue
            if len(clean) >= 2:
                candidates.append(clean[:2])
        if candidates:
            return candidates
        line = self.line_id(value)
        return [line] if line else []

    def point_id(self, value: Any) -> str:
        if isinstance(value, dict):
            for key in ["id", "label", "name", "point"]:
                if value.get(key):
                    return self.point_id(value.get(key))
        clean = "".join(ch for ch in str(value or "").upper() if ch.isalnum())
        if len(clean) > 1 and not self.strip_stop_prefixes(clean):
            return ""
        return clean[:4]

    def line_id(self, value: Any) -> str:
        if isinstance(value, (list, tuple)):
            if len(value) >= 2:
                return "".join(self.point_id(part)[:1] for part in value[:2])
            return self.line_id(value[0]) if value else ""
        if isinstance(value, dict):
            for key in ["points", "endpoints"]:
                if value.get(key):
                    return self.line_id(value.get(key))
            for key in ["id", "name", "label", "object", "segment", "line"]:
                if value.get(key):
                    return self.line_id(value.get(key))
            return ""
        text = str(value or "").upper()
        tokens = re.findall(r"[A-Z][A-Z0-9]*", text)
        filtered = [item for item in tokens if item not in STOP_TOKENS]
        if len(filtered) >= 2 and all(len(item) == 1 for item in filtered[:2]):
            return "".join(filtered[:2])
        for token in filtered:
            clean = "".join(ch for ch in token if ch.isalnum())
            if len(clean) >= 2:
                return clean[:2]
        clean = self.strip_stop_prefixes("".join(ch for ch in text if ch.isalnum()))
        return clean[:2] if len(clean) >= 2 else ""

    def looks_like_point(self, value: Any) -> bool:
        if isinstance(value, dict):
            return any(value.get(key) for key in ["id", "label", "name", "point"])
        clean = "".join(ch for ch in str(value or "").upper() if ch.isalnum())
        return 1 <= len(clean) <= 4

    def strip_stop_prefixes(self, value: str) -> str:
        clean = str(value or "").upper()
        changed = True
        prefixes = sorted(STOP_TOKENS, key=len, reverse=True)
        while changed and clean:
            changed = False
            for prefix in prefixes:
                if clean == prefix:
                    return ""
                if clean.startswith(prefix):
                    clean = clean[len(prefix):]
                    changed = True
                    break
        return clean
