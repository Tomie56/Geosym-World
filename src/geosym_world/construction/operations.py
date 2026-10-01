"""Construction trace schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from geosym_world.errors import InvalidGeometryInput
from geosym_world.schema.ids import validate_geometry_id


@dataclass(frozen=True)
class ConstructionOp:
    """One replayable construction operation."""

    op: str
    id: str | None = None
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.op, str) or not self.op.strip():
            raise InvalidGeometryInput("Construction operation name must be non-empty")
        if self.id is not None:
            validate_geometry_id(self.id)
        if not isinstance(self.params, dict):
            raise InvalidGeometryInput("Construction operation params must be a mapping")

    @classmethod
    def from_dict(cls, item: Dict[str, Any]) -> "ConstructionOp":
        params = dict(item)
        op = str(params.pop("op"))
        object_id = params.pop("id", None)
        return cls(op=op, id=object_id, params=params)

    def to_dict(self) -> Dict[str, Any]:
        data = {"op": self.op}
        if self.id is not None:
            data["id"] = self.id
        data.update(self.params)
        return data


def normalize_trace(trace: Iterable[ConstructionOp | Dict[str, Any]]) -> List[ConstructionOp]:
    ops: List[ConstructionOp] = []
    for item in trace:
        if isinstance(item, ConstructionOp):
            ops.append(item)
        else:
            ops.append(ConstructionOp.from_dict(item))
    return ops


def operation_references(operation: ConstructionOp) -> tuple[tuple[str, str], ...]:
    """Return typed input references used by one construction operation."""

    params = operation.params
    references: list[tuple[str, str]] = []
    if operation.op == "add_segment":
        references.extend(("point", item) for item in params.get("points", [params.get("start"), params.get("end")]) if item)
    elif operation.op == "add_relation":
        relation_type = str(params.get("type", ""))
        if relation_type in {"parallel", "perpendicular", "equal_length"}:
            references.extend(("line", item) for item in (params.get("objects") or params.get("segments") or ()))
        elif relation_type == "tangent_to_circle":
            references.extend(("circle", item) for item in params.get("circles", ()))
        elif relation_type == "proportional_lengths":
            references.extend(("segment", item) for pair in params.get("ratios", ()) for item in pair)
        elif relation_type in {"point_on_line", "point_on_ray"}:
            references.append(("point", params["point"]))
            references.append(("line", params["line"]))
        elif relation_type == "point_on_segment":
            references.append(("point", params["point"]))
            references.append(("segment", params["segment"]))
        elif relation_type == "ordered_points":
            references.append(("line", params["support"]))
            references.extend(("point", item) for item in params.get("points", ()))
        elif relation_type == "collinear":
            references.extend(("point", item) for item in params.get("points", ()))
    elif operation.op == "add_line":
        references.extend(("point", item) for item in params.get("points", []))
        if "point" in params:
            references.append(("point", params["point"]))
    elif operation.op == "add_ray":
        references.append(("point", params["start"]))
        if "through" in params:
            references.append(("point", params["through"]))
    elif operation.op == "add_circle":
        references.append(("point", params["center"]))
    elif operation.op == "add_arc":
        references.extend((("circle", params["circle"]), ("point", params["start"]), ("point", params["end"])))
    elif operation.op == "add_polygon":
        references.extend(("point", item) for item in params.get("vertices", []))
        area_formula = (params.get("metadata") or {}).get("area_formula") or {}
        references.extend(("segment", area_formula[item]) for item in ("base",) if isinstance(area_formula.get(item), str))
        references.extend(("segment", item) for item in area_formula.get("bases", ()) if isinstance(item, str))
        references.extend(("segment", item) for item in area_formula.get("diagonals", ()) if isinstance(item, str))
        height = area_formula.get("height")
        if isinstance(height, str):
            references.append(("segment", height))
        elif isinstance(height, (list, tuple)):
            references.extend(("point", item) for item in height)
        if area_formula.get("sector_region"):
            references.append(("region", area_formula["sector_region"]))
        perimeter_formula = (params.get("metadata") or {}).get("perimeter_formula") or {}
        references.extend(("segment", perimeter_formula[item]) for item in ("width", "height") if perimeter_formula.get(item))
    elif operation.op == "mark_polygon_region":
        if "polygon" in params:
            references.append(("polygon", params["polygon"]))
        references.extend(("point", item) for item in params.get("vertices", []))
    elif operation.op == "add_region":
        references.extend(("object", item) for item in params.get("boundary", []))
        references.extend(("object", item) for item in params.get("holes", []))
        metadata = params.get("metadata") or {}
        references.extend(("circle", metadata[item]) for item in ("inner_circle", "outer_circle") if metadata.get(item))
        references.extend(("circle", item) for item in metadata.get("circles", ()) if item)
        references.extend(("arc", item) for item in metadata.get("arcs", ()) if item)
        references.extend(("point", item) for item in metadata.get("intersections", ()) if item)
        if metadata.get("sector_region"):
            references.append(("region", metadata["sector_region"]))
        if metadata.get("triangle_polygon"):
            references.append(("polygon", metadata["triangle_polygon"]))
        if metadata.get("arc"):
            references.append(("arc", metadata["arc"]))
        if metadata.get("chord"):
            references.append(("segment", metadata["chord"]))
        area_formula = metadata.get("area_formula") or {}
        if area_formula.get("outer_region"):
            references.append(("region", area_formula["outer_region"]))
        references.extend(("segment", area_formula[item]) for item in ("outer_radius_segment", "inner_radius_segment") if area_formula.get(item))
        references.extend(("segment", area_formula[item]) for item in ("radius_segment", "center_segment") if area_formula.get(item))
        references.extend(("segment", item) for item in area_formula.get("radius_segments", ()) if item)
        perimeter_formula = metadata.get("perimeter_formula") or {}
        if perimeter_formula.get("outer_region"):
            references.append(("region", perimeter_formula["outer_region"]))
        references.extend(("segment", perimeter_formula[item]) for item in ("outer_radius_segment", "inner_radius_segment") if perimeter_formula.get(item))
        references.extend(("segment", perimeter_formula[item]) for item in ("radius_segment", "center_segment") if perimeter_formula.get(item))
        references.extend(("segment", item) for item in perimeter_formula.get("radius_segments", ()) if item)
    elif operation.op == "add_composite_region":
        references.extend(("region", item) for item in params.get("components", []))
        metadata = params.get("metadata") or {}
        if metadata.get("parent_region"):
            references.append(("region", metadata["parent_region"]))
    elif operation.op == "mark_sector_region":
        references.extend((("circle", params["circle"]), ("point", params["start"]), ("point", params["end"])))
        radius_segment = (params.get("metadata") or {}).get("radius_segment")
        if radius_segment:
            references.append(("segment", radius_segment))
    elif operation.op == "midpoint":
        references.extend(("point", item) for item in (params.get("of") or params.get("points") or []))
    elif operation.op == "intersection_line_line":
        references.extend((("line", params["line1"]), ("line", params["line2"])))
    elif operation.op == "intersection_line_circle":
        references.extend((("line", params["line"]), ("circle", params["circle"])))
    elif operation.op == "intersection_circle_circle":
        references.extend((("circle", params["circle1"]), ("circle", params["circle2"])))
    elif operation.op == "intersection_line_arc":
        references.extend((("line", params["line"]), ("arc", params["arc"])))
    elif operation.op == "intersection_circle_arc":
        references.extend((("circle", params["circle"]), ("arc", params["arc"])))
    elif operation.op == "intersection_arc_arc":
        references.extend((("arc", params["arc1"]), ("arc", params["arc2"])))
    elif operation.op == "tangent_points_from_point":
        references.extend((("point", params["point"]), ("circle", params["circle"])))
    elif operation.op == "add_chord":
        references.append(("circle", params["circle"]))
        references.extend(("point", item) for item in params.get("points", []))
    elif operation.op in {"parallel_through", "perpendicular_through"}:
        references.extend((("point", params["point"]), ("line", params["line"])))
    elif operation.op == "line_through_angle":
        references.append(("point", params["point"]))
        if "base_line" in params:
            references.append(("line", params["base_line"]))
    elif operation.op == "rotate_ray":
        references.append(("ray", params["ray"]))
    elif operation.op == "angle_bisector":
        references.append(("point", params["vertex"]))
        references.extend(("point", item) for item in params.get("ray1", []))
        references.extend(("point", item) for item in params.get("ray2", []))
    elif operation.op == "foot_of_perpendicular":
        references.extend((("point", params["point"]), ("line", params["line"])))
    elif operation.op == "point_on_ray_distance":
        if "ray" in params:
            references.append(("ray", params["ray"]))
        if "start" in params:
            references.append(("point", params["start"]))
    elif operation.op == "point_on_segment_ratio":
        if "segment" in params:
            references.append(("segment", params["segment"]))
        references.extend(("point", item) for item in params.get("points", []))
    elif operation.op == "project_point_to_plane":
        references.extend((("point", params["point"]), ("plane", params["plane"])))
    elif operation.op == "project_line_to_plane":
        references.extend((("line", params["line"]), ("plane", params["plane"])))
    elif operation.op == "add_plane":
        references.extend(("point", item) for item in params.get("points", []))
        if "point" in params:
            references.append(("point", params["point"]))
    elif operation.op == "add_polyhedron":
        references.extend(("point", item) for item in params.get("vertices", []))
    elif operation.op == "dihedral_angle":
        references.extend(("plane", item) for item in params.get("planes", []))
        if "polyhedron" in params:
            references.append(("polyhedron", params["polyhedron"]))
    elif operation.op == "section_polyhedron_with_plane":
        references.extend((("polyhedron", params["polyhedron"]), ("plane", params["plane"])))
        references.extend(("point", item) for item in params.get("boundary", []))
    if operation.op.startswith("intersection_") and params.get("reference"):
        references.append(("point", params["reference"]))
    return tuple(dict.fromkeys((kind, str(reference)) for kind, reference in references if reference is not None))


def operation_outputs(operation: ConstructionOp) -> tuple[str, ...]:
    """Return stable object IDs created by one construction operation."""

    if operation.op in {"set_target", "add_relation"}:
        return ()
    outputs = []
    if operation.id is not None:
        outputs.append(operation.id)
    outputs.extend(str(item) for item in operation.params.get("ids", ()) if item is not None)
    return tuple(dict.fromkeys(outputs))
