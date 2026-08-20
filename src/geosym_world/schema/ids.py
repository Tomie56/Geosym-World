"""Stable identifiers used by geometry records."""

from __future__ import annotations

from geosym_world.errors import InvalidGeometryInput

GeometryId = str
EventId = str
RecordId = str


def validate_geometry_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidGeometryInput("Geometry IDs must be non-empty strings")
    if any(character.isspace() for character in value):
        raise InvalidGeometryInput(f"Geometry IDs cannot contain whitespace: {value!r}")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise InvalidGeometryInput(f"Geometry IDs cannot contain control characters: {value!r}")
    return value
