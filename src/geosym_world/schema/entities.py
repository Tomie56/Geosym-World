"""Geometry object schemas for 2D and reserved 3D worlds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import sympy as sp

from geosym_world.errors import InvalidGeometryInput, NumericalInstabilityError
from geosym_world.schema.ids import validate_geometry_id

Coord = Tuple[sp.Expr, ...]


def payload_expr(value: sp.Expr) -> Dict[str, Any]:
    value = sp.simplify(value)
    return {"expr": str(value), "value": float(sp.N(value))}


def payload_coords(values: Coord) -> List[Dict[str, Any]]:
    return [payload_expr(v) for v in values]


@dataclass(frozen=True)
class Point:
    id: str
    coords: Coord
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return len(self.coords)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "point",
            "dim": self.dim,
            "coords": payload_coords(self.coords),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class Line:
    id: str
    point: Coord
    direction: Coord
    source_points: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return len(self.point)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "line",
            "dim": self.dim,
            "point": payload_coords(self.point),
            "direction": payload_coords(self.direction),
            "source_points": list(self.source_points),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class Segment:
    id: str
    start: str
    end: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "type": "segment", "start": self.start, "end": self.end, "metadata": self.metadata}


@dataclass(frozen=True)
class Ray:
    id: str
    start: str
    direction: Coord
    through: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "ray",
            "start": self.start,
            "direction": payload_coords(self.direction),
            "through": self.through,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class Circle:
    id: str
    center: str
    radius: sp.Expr
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "circle",
            "center": self.center,
            "radius": payload_expr(self.radius),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class Arc:
    id: str
    circle: str
    start: str
    end: str
    central_angle: sp.Expr | None = None
    orientation: str = "ccw"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "arc",
            "circle": self.circle,
            "start": self.start,
            "end": self.end,
            "central_angle": payload_expr(self.central_angle) if self.central_angle is not None else None,
            "orientation": self.orientation,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class Polygon:
    id: str
    vertices: Tuple[str, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "type": "polygon", "vertices": list(self.vertices), "metadata": self.metadata}


@dataclass(frozen=True)
class Region:
    id: str
    boundary: Tuple[str, ...]
    kind: str = "polygon_region"
    metadata: Dict[str, Any] = field(default_factory=dict)
    holes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "region",
            "kind": self.kind,
            "boundary": list(self.boundary),
            "metadata": self.metadata,
            "holes": list(self.holes),
        }


@dataclass(frozen=True)
class CompositeRegion:
    """A region whose area is derived from named region components."""

    id: str
    operation: str
    components: Tuple[str, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "composite_region",
            "operation": self.operation,
            "components": list(self.components),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class Plane:
    id: str
    point: Coord
    normal: Coord
    source_points: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "plane",
            "dim": 3,
            "point": payload_coords(self.point),
            "normal": payload_coords(self.normal),
            "source_points": list(self.source_points),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class Polyhedron:
    id: str
    vertices: Tuple[str, ...]
    faces: Tuple[Tuple[str, ...], ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "polyhedron",
            "vertices": list(self.vertices),
            "faces": [list(face) for face in self.faces],
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class Target:
    type: str
    object: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    mode: str = "value"
    expectation: Dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.type, str) or not self.type.strip():
            raise InvalidGeometryInput("Target type must be non-empty")
        validate_geometry_id(self.object)
        if self.mode not in {"value", "proof", "constraint"}:
            raise InvalidGeometryInput(f"Unknown target mode: {self.mode}")
        if self.mode == "constraint":
            if not isinstance(self.expectation, dict):
                raise InvalidGeometryInput("Constraint target requires an expectation mapping")
            if self.expectation.get("operator") not in {"eq", "ne", "lt", "le", "gt", "ge"}:
                raise InvalidGeometryInput("Constraint target requires operator eq/ne/lt/le/gt/ge")
            if "value" not in self.expectation:
                raise InvalidGeometryInput("Constraint target requires an expected value")
        elif self.expectation is not None:
            raise InvalidGeometryInput(f"{self.mode} target cannot define a constraint expectation")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "object": self.object,
            "mode": self.mode,
            "expectation": dict(self.expectation) if self.expectation is not None else None,
            "metadata": self.metadata,
        }
