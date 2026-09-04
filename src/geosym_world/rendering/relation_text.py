"""Formatting for public relation-value annotations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .math_text import format_math_expression


RelationFormatter = Callable[[Mapping[str, Any]], str | None]


def format_relation_value(relation: Mapping[str, Any]) -> str | None:
    metadata = relation.get("metadata") or {}
    if metadata.get("label_text") is not None:
        return str(metadata["label_text"])
    formatter = RELATION_VALUE_FORMATTERS.get(str(relation.get("type", "")))
    return formatter(relation) if formatter is not None else None


def _proportional_lengths(relation: Mapping[str, Any]) -> str | None:
    ratios = tuple(tuple(str(item) for item in pair) for pair in relation.get("ratios", ()))
    if not ratios or any(len(pair) != 2 for pair in ratios) or relation.get("value") is None:
        return None
    ratio_text = " = ".join(f"{numerator}/{denominator}" for numerator, denominator in ratios)
    return f"{ratio_text} = {format_math_expression(relation['value'])}"


def _equal_length(relation: Mapping[str, Any]) -> str | None:
    objects = tuple(str(item) for item in relation.get("segments") or relation.get("objects") or ())
    return " = ".join(objects) if len(objects) >= 2 else None


def _equal_angle(relation: Mapping[str, Any]) -> str | None:
    angles = tuple(tuple(str(item) for item in angle) for angle in relation.get("angles", ()))
    if len(angles) < 2 or any(len(angle) != 3 for angle in angles):
        return None
    return " = ".join(f"∠{''.join(angle)}" for angle in angles)


RELATION_VALUE_FORMATTERS: dict[str, RelationFormatter] = {
    "equal_angle": _equal_angle,
    "equal_length": _equal_length,
    "proportional_lengths": _proportional_lengths,
}
