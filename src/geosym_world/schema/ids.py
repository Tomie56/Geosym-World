"""Stable identifiers used by geometry records."""

from __future__ import annotations

GeometryId = str
EventId = str
RecordId = str


def validate_geometry_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Geometry IDs must be non-empty strings")
    if any(character.isspace() for character in value):
        raise ValueError(f"Geometry IDs cannot contain whitespace: {value!r}")
    return value
