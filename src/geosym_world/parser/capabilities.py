"""Declared support levels for the single ParseGraph parser pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from geosym_world.schema import ParseObjectKind
from geosym_world.schema.parse_graph import PARSE_RELATION_TYPES


class CapabilityLevel(str, Enum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    DEFERRED = "deferred"


@dataclass(frozen=True)
class ParserCapability:
    representation: CapabilityLevel
    extraction: CapabilityLevel
    realization: CapabilityLevel
    verification: CapabilityLevel
    note: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "representation": self.representation.value,
            "extraction": self.extraction.value,
            "realization": self.realization.value,
            "verification": self.verification.value,
            "note": self.note,
        }


_SUPPORTED = CapabilityLevel.SUPPORTED
_PARTIAL = CapabilityLevel.PARTIAL
_DEFERRED = CapabilityLevel.DEFERRED

_OBJECT_CAPABILITIES: Mapping[ParseObjectKind, ParserCapability] = {
    ParseObjectKind.POINT: ParserCapability(_SUPPORTED, _PARTIAL, _SUPPORTED, _SUPPORTED),
    ParseObjectKind.LINE: ParserCapability(_SUPPORTED, _PARTIAL, _SUPPORTED, _SUPPORTED),
    ParseObjectKind.SEGMENT: ParserCapability(_SUPPORTED, _PARTIAL, _SUPPORTED, _SUPPORTED),
    ParseObjectKind.RAY: ParserCapability(_SUPPORTED, _PARTIAL, _SUPPORTED, _SUPPORTED),
    ParseObjectKind.POLYGON: ParserCapability(_SUPPORTED, _PARTIAL, _PARTIAL, _PARTIAL),
    ParseObjectKind.CIRCLE: ParserCapability(_SUPPORTED, _PARTIAL, _PARTIAL, _PARTIAL, "Explicit center/radius and point-on-circle constraints; no generic tangency solver"),
    ParseObjectKind.ARC: ParserCapability(_SUPPORTED, _PARTIAL, _PARTIAL, _PARTIAL, "Center/endpoints or circle reference; numeric endpoint verification"),
    ParseObjectKind.TEXT: ParserCapability(_SUPPORTED, _PARTIAL, _DEFERRED, _SUPPORTED),
    ParseObjectKind.MARKER: ParserCapability(_SUPPORTED, _PARTIAL, _DEFERRED, _PARTIAL),
    ParseObjectKind.PLANE: ParserCapability(_SUPPORTED, _PARTIAL, _DEFERRED, _PARTIAL, "3D is outside the current parser closure"),
    ParseObjectKind.POLYHEDRON: ParserCapability(_SUPPORTED, _PARTIAL, _DEFERRED, _PARTIAL, "3D is outside the current parser closure"),
    ParseObjectKind.UNKNOWN: ParserCapability(_SUPPORTED, _PARTIAL, _DEFERRED, _DEFERRED),
}

_REALIZED_RELATIONS = frozenset({
    "collinear",
    "incidence",
    "ordered_points",
    "parallel",
    "perpendicular",
    "point_on_line",
    "point_on_ray",
    "point_on_segment",
})
_COMPLETION_RELATIONS = frozenset({
    "point_on_circle",
    "foot_of_perpendicular",
    "midpoint",
    "point_on_segment_ratio",
    "rectangle",
})


def object_capability(kind: ParseObjectKind) -> ParserCapability:
    return _OBJECT_CAPABILITIES[kind]


def relation_capability(predicate: str) -> ParserCapability:
    if predicate not in PARSE_RELATION_TYPES:
        raise ValueError(f"Unknown ParseGraph relation: {predicate}")
    if predicate in _REALIZED_RELATIONS:
        return ParserCapability(_SUPPORTED, _PARTIAL, _SUPPORTED, _SUPPORTED)
    if predicate in _COMPLETION_RELATIONS:
        return ParserCapability(
            _SUPPORTED,
            _PARTIAL,
            _PARTIAL,
            _SUPPORTED,
            "Supported when an explicit completion strategy or required scaffold is available",
        )
    return ParserCapability(
        _SUPPORTED,
        _PARTIAL,
        _DEFERRED,
        _PARTIAL,
        "Preserved in ParseGraph; generic realization is not implemented yet",
    )


def parser_capability_report() -> dict[str, dict[str, dict[str, str]]]:
    return {
        "objects": {
            kind.value: object_capability(kind).to_dict()
            for kind in sorted(ParseObjectKind, key=lambda item: item.value)
        },
        "relations": {
            predicate: relation_capability(predicate).to_dict()
            for predicate in sorted(PARSE_RELATION_TYPES)
        },
    }
