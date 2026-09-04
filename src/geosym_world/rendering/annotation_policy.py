"""Parameterized annotation visibility for geometry rendering."""

from __future__ import annotations

from dataclasses import dataclass


ANNOTATION_LEVELS = ("points", "givens")


@dataclass(frozen=True)
class AnnotationPolicy:
    level: str = "givens"
    relation_markers: bool = True

    def __post_init__(self) -> None:
        if self.level not in ANNOTATION_LEVELS:
            raise ValueError(f"Unsupported annotation level: {self.level}")

    @property
    def show_public_values(self) -> bool:
        return self.level == "givens"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "level": self.level,
            "relation_markers": self.relation_markers,
        }
