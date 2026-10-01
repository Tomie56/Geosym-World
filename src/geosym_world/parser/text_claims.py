"""Deterministic extraction of explicit geometry claims from problem text."""

from __future__ import annotations

import re
from typing import Any


class TextClaimExtractor:
    _TRIANGLE_RELATION = re.compile(
        r"(?:triangle\s*|\\triangle\s*)?([A-Z]\s*[A-Z]\s*[A-Z])\s*"
        r"(?:is\s+)?(congruent|similar|\\cong|\\sim|≅|∼)(?:\s+to)?\s*"
        r"(?:triangle\s*|\\triangle\s*)?([A-Z]\s*[A-Z]\s*[A-Z])",
        re.IGNORECASE,
    )
    _SEGMENT_RELATION = re.compile(
        r"\b(?!is\b)([A-Z]{2})\b\s*(?:is\s+)?(parallel to|perpendicular to|[⊥∥])\s*\b([A-Z]{2})\b",
        re.IGNORECASE,
    )
    _LENGTH_RELATION = re.compile(r"\b([A-Z]{2})\b\s*(?:=|is)\s*(\d+(?:\.\d+)?)\b")
    _ANGLE_RELATION = re.compile(
        r"(?:m\s*)?(?:∠|angle\s+)([A-Z]{3})\s*(?:=|is)\s*(\d+(?:\.\d+)?)\s*(?:degrees?|°)?",
        re.IGNORECASE,
    )
    _QUADRILATERAL_SHAPE = re.compile(
        r"\$?\s*([A-Z]\s*[A-Z]\s*[A-Z]\s*[A-Z])\s*\$?\s+is\s+(?:a\s+)?"
        r"(rectangle|parallelogram|trapezoid|rhombus|kite)\b",
        re.IGNORECASE,
    )

    def extract(self, text: str) -> list[dict[str, Any]]:
        claims: list[dict[str, Any]] = []
        for match in self._TRIANGLE_RELATION.finditer(text or ""):
            first = "".join(match.group(1).upper().split())
            second = "".join(match.group(3).upper().split())
            relation = "congruent" if match.group(2).lower() in {"congruent", "\\cong", "≅"} else "similar"
            claims.append(self._claim(relation, [first, second], match.group(0), len(claims)))
        for match in self._SEGMENT_RELATION.finditer(text or ""):
            marker = match.group(2).lower()
            relation = "parallel" if marker in {"parallel to", "∥"} else "perpendicular"
            claims.append(self._claim(relation, [match.group(1).upper(), match.group(3).upper()], match.group(0), len(claims)))
        for match in self._LENGTH_RELATION.finditer(text or ""):
            claims.append(self._claim("length", [match.group(1).upper()], match.group(0), len(claims), match.group(2)))
        for match in self._ANGLE_RELATION.finditer(text or ""):
            claims.append(self._claim("angle", [match.group(1).upper()], match.group(0), len(claims), match.group(2)))
        for match in self._QUADRILATERAL_SHAPE.finditer(text or ""):
            claims.append(self._claim(
                match.group(2).lower(),
                ["".join(match.group(1).upper().split())],
                match.group(0),
                len(claims),
            ))
        return claims

    @staticmethod
    def _claim(
        relation: str,
        objects: list[str],
        raw: str,
        index: int,
        value: str | None = None,
    ) -> dict[str, Any]:
        claim: dict[str, Any] = {
            "type": relation,
            "objects": objects,
            "confidence": 0.98,
            "source": "text",
            "candidate_id": f"text_claim_{index}",
            "raw": raw,
        }
        if value is not None:
            claim["value"] = value
        return claim
