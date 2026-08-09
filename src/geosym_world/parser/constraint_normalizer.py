"""Normalize VLM/OCR relation candidates into GeoSym constraints."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import re

from geosym_world.schema.problem_normalize import line_id, normalize_id
from geosym_world.schema.relation_normalization import normalize_relation_type


class ConstraintNormalizer:
    def normalize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "constraints" in data and "objects" in data:
            return self._normalize_graph(data)
        return self._from_relation_text(data)

    def _normalize_graph(self, data: Dict[str, Any]) -> Dict[str, Any]:
        objects = data.get("objects") or {}
        points = [normalize_id(p) for p in objects.get("points") or []]
        lines = []
        for item in objects.get("lines") or []:
            if isinstance(item, str):
                clean = normalize_id(item)
                if len(clean) >= 2:
                    lines.append([clean[0], clean[1]])
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                lines.append([normalize_id(item[0]), normalize_id(item[1])])
        constraints = [self._normalize_constraint(c) for c in data.get("constraints") or []]
        return {
            **data,
            "objects": {
                **objects,
                "points": points,
                "lines": lines,
                "circles": objects.get("circles") or [],
            },
            "constraints": constraints,
        }

    def _from_relation_text(self, data: Dict[str, Any]) -> Dict[str, Any]:
        relation_texts = list(data.get("relations_text") or [])
        points = set()
        lines = set()
        constraints: List[Dict[str, Any]] = []
        for text in relation_texts:
            parsed = self._parse_relation_sentence(text)
            if parsed:
                constraints.append(parsed)
                for line in parsed.get("lines") or []:
                    clean = normalize_id(line)
                    if len(clean) >= 2:
                        points.update(clean[:2])
                        lines.add((clean[0], clean[1]))
                obj = parsed.get("object")
                if obj:
                    clean = normalize_id(obj)
                    if len(clean) >= 2:
                        points.update(clean[:2])
                        lines.add((clean[0], clean[1]))
        for obj in data.get("objects") or []:
            kind, ids = self._parse_object_text(obj)
            if kind == "point":
                points.update(ids)
            elif kind == "line":
                if len(ids) >= 2:
                    lines.add((ids[0], ids[1]))
                    points.update(ids[:2])
        return {
            "objects": {
                "points": sorted(points),
                "lines": [list(item) for item in sorted(lines)],
                "circles": [],
            },
            "constraints": constraints,
            "target": self._parse_target(relation_texts) or data.get("target", {}),
            "extraction": data,
        }

    def _normalize_constraint(self, c: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(c)
        out["type"] = normalize_relation_type(str(c.get("type", "")))
        if "object" in out:
            out["object"] = normalize_id(out["object"])
        if "lines" in out:
            out["lines"] = [normalize_id(item) for item in out["lines"]]
        return out

    def _parse_relation_sentence(self, text: str) -> Dict[str, Any] | None:
        clean = text.strip()
        line_tokens = re.findall(r"\b[A-Z]{2}\b", clean)
        lower = clean.lower()
        if any(word in lower for word in ["perpendicular", "orthogonal", "right angle"]) and len(line_tokens) >= 2:
            return {"type": "perpendicular", "lines": [line_tokens[0], line_tokens[1]], "source": "vlm_text", "raw": text}
        if line_tokens and "find" not in lower:
            length = re.search(r"(?:length|has length|=|is)\s*([0-9A-Za-z_+*/^().\\{}]+)", clean, re.I)
            if length:
                return {"type": "length", "object": line_tokens[0], "value": length.group(1), "source": "vlm_text", "raw": text}
        return None

    def _parse_object_text(self, obj: Any) -> Tuple[str | None, List[str]]:
        text = str(obj or "").strip()
        clean = normalize_id(text)
        if not clean:
            return None, []
        lower = text.lower()
        if re.search(r"\b(point|vertex)\b", lower):
            match = re.search(r"\b(?:point|vertex)\s+([A-Z][A-Za-z0-9_]*)\b", text, re.I)
            if match:
                return "point", [normalize_id(match.group(1))]
            if len(clean) == 1:
                return "point", [clean]
        if re.search(r"\b(segment|line|ray|side|chord|diameter|radius)\b", lower):
            match = re.search(r"\b(?:segment|line|ray|side|chord|diameter|radius)\s+([A-Z]{2})\b", text, re.I)
            if match:
                item = normalize_id(match.group(1))
                return "line", [item[0], item[1]]
            if len(clean) == 2 and clean.isupper():
                return "line", [clean[0], clean[1]]
        # Bare two-letter uppercase tokens are usually line ids.
        if re.fullmatch(r"[A-Z]{2}", clean):
            return "line", [clean[0], clean[1]]
        # Do not infer points/lines from words like Circle, Arc, Angle, or
        # Quadrilateral; VLM object labels are too noisy for that.
        return None, []

    def _parse_target(self, texts: List[str]) -> Dict[str, Any] | None:
        for text in texts:
            match = re.search(r"(?:find|what is|determine).*?\b([A-Z]{2})\b", text, re.I)
            if match:
                return {"type": "length", "object": normalize_id(match.group(1))}
        return None
