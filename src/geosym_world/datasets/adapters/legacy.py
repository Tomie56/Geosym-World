"""Lightweight adapter from legacy/synthetic geometry dictionaries to traces.

This module intentionally avoids importing the old scripts package. It accepts
plain dictionaries and emits construction traces that the new world can verify
and execute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List


@dataclass
class LegacyAdapterReport:
    """Structured report for one legacy record adaptation."""

    input_keys: List[str]
    op_type_counts: Dict[str, int] = field(default_factory=dict)
    used_ids: Dict[str, List[str]] = field(default_factory=dict)
    preserved_metadata: Dict[str, Any] = field(default_factory=dict)
    skipped: List[Dict[str, Any]] = field(default_factory=list)
    unhandled_top_level_keys: List[str] = field(default_factory=list)

    def add_ops(self, category: str, ops: List[Dict[str, Any]]) -> None:
        ids = self.used_ids.setdefault(category, [])
        for op in ops:
            op_type = str(op.get("op"))
            self.op_type_counts[op_type] = self.op_type_counts.get(op_type, 0) + 1
            if op.get("id"):
                ids.append(str(op["id"]))
            for point_id in op.get("ids") or []:
                ids.append(str(point_id))

    def skip(self, category: str, item_id: str | None, reason: str, **metadata: Any) -> None:
        self.skipped.append(
            {
                "category": category,
                "id": item_id,
                "reason": reason,
                "metadata": metadata,
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        coverage = _coverage_metrics(
            input_keys=self.input_keys,
            unhandled_top_level_keys=self.unhandled_top_level_keys,
            op_type_counts=self.op_type_counts,
            num_skipped=len(self.skipped),
        )
        return {
            "input_keys": self.input_keys,
            "op_type_counts": dict(sorted(self.op_type_counts.items())),
            "used_ids": {key: sorted(set(value)) for key, value in sorted(self.used_ids.items())},
            "preserved_metadata": self.preserved_metadata,
            "coverage": coverage,
            "coverage_score": coverage["coverage_score"],
            "num_skipped": len(self.skipped),
            "skipped": self.skipped,
            "unhandled_top_level_keys": self.unhandled_top_level_keys,
        }


class LegacyGeometryAdapter:
    """Converts tolerant legacy-like records into construction traces."""

    def to_trace(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self.adapt(record)["trace"]

    def adapt(self, record: Dict[str, Any]) -> Dict[str, Any]:
        report = LegacyAdapterReport(input_keys=sorted(str(key) for key in record.keys()))
        report.preserved_metadata = _record_metadata(record)
        trace: List[Dict[str, Any]] = []
        for category, builder in [
            ("points", self._point_ops),
            ("lines", self._line_ops),
            ("segments", self._segment_ops),
            ("circles", self._circle_ops),
            ("chords", self._chord_ops),
            ("arcs", self._arc_ops),
            ("polygons", self._polygon_ops),
            ("faces", self._face_ops),
            ("polyhedra", self._polyhedron_ops),
            ("regions", self._region_ops),
        ]:
            ops = builder(record, report)
            trace.extend(ops)
            report.add_ops(category, ops)
        target = self._target(record)
        if target:
            trace.append(target)
            report.add_ops("target", [target])
        report.unhandled_top_level_keys = _unhandled_keys(record)
        return {"trace": trace, "report": report.to_dict()}

    def _point_ops(self, record: Dict[str, Any], report: LegacyAdapterReport) -> List[Dict[str, Any]]:
        points = record.get("points") or record.get("vertices") or {}
        ops = []
        if isinstance(points, dict):
            items = points.items()
        else:
            items = ((item.get("id") or item.get("name"), item) for item in points)
        for point_id, payload in items:
            if not point_id:
                report.skip("points", None, "missing_point_id")
                continue
            coord = _coord(payload)
            if coord is None:
                report.skip("points", str(point_id), "missing_or_unreadable_coordinates")
                continue
            ops.append({"op": "add_point3d" if len(coord) == 3 else "add_point", "id": str(point_id), "coord": coord})
        return ops

    def _line_ops(self, record: Dict[str, Any], report: LegacyAdapterReport) -> List[Dict[str, Any]]:
        ops = []
        for item in _iter_items(record.get("lines") or []):
            line_id = item.get("id") or item.get("name")
            if not line_id:
                report.skip("lines", None, "missing_line_id")
                continue
            points = item.get("points") or item.get("endpoints")
            if not points and item.get("start_point_id") and item.get("end_point_id"):
                points = [item["start_point_id"], item["end_point_id"]]
            if points and len(points) == 2:
                if item.get("type", "segment") == "infinite_line":
                    ops.append({"op": "add_line", "id": str(line_id), "points": [str(points[0]), str(points[1])]})
                else:
                    ops.append({"op": "add_segment", "id": str(line_id), "points": [str(points[0]), str(points[1])]})
            elif "point" in item and "direction" in item:
                ops.append({"op": "add_line", "id": str(line_id), "point": str(item["point"]), "direction": item["direction"]})
            else:
                report.skip("lines", str(line_id), "missing_endpoints_or_direction")
        return ops

    def _segment_ops(self, record: Dict[str, Any], report: LegacyAdapterReport) -> List[Dict[str, Any]]:
        ops = []
        raw_segments = record.get("segments") or record.get("edges") or []
        for item in _iter_items(raw_segments):
            seg_id = item.get("id") or item.get("name")
            points = item.get("points") or item.get("endpoints")
            if not points and "start" in item and "end" in item:
                points = [item["start"], item["end"]]
            if not points and item.get("start_point_id") and item.get("end_point_id"):
                points = [item["start_point_id"], item["end_point_id"]]
            if not points or len(points) != 2:
                report.skip("segments", str(seg_id) if seg_id else None, "missing_segment_endpoints")
                continue
            op = {"op": "add_segment", "points": [str(points[0]), str(points[1])]}
            if seg_id:
                op["id"] = str(seg_id)
            else:
                op["id"] = f"{points[0]}{points[1]}"
            ops.append(op)
        return ops

    def _circle_ops(self, record: Dict[str, Any], report: LegacyAdapterReport) -> List[Dict[str, Any]]:
        ops = []
        for item in _iter_items(record.get("circles") or []):
            circle_id = item.get("id") or item.get("name")
            center = item.get("center")
            radius = item.get("radius")
            if circle_id and center and radius is not None:
                ops.append({"op": "add_circle", "id": str(circle_id), "center": str(center), "radius": str(radius)})
            else:
                report.skip("circles", str(circle_id) if circle_id else None, "missing_circle_id_center_or_radius")
        return ops

    def _chord_ops(self, record: Dict[str, Any], report: LegacyAdapterReport) -> List[Dict[str, Any]]:
        ops = []
        for item in _iter_items(record.get("chords") or []):
            chord_id = item.get("id") or item.get("name")
            circle = item.get("circle")
            points = item.get("points") or item.get("endpoints")
            if chord_id and circle and points and len(points) == 2:
                ops.append({"op": "add_chord", "id": str(chord_id), "circle": str(circle), "points": [str(points[0]), str(points[1])]})
            else:
                report.skip("chords", str(chord_id) if chord_id else None, "missing_chord_id_circle_or_points")
        return ops

    def _arc_ops(self, record: Dict[str, Any], report: LegacyAdapterReport) -> List[Dict[str, Any]]:
        ops = []
        for item in _iter_items(record.get("arcs") or []):
            arc_id = item.get("id") or item.get("name")
            circle = item.get("circle")
            start = item.get("start")
            end = item.get("end")
            if not (arc_id and circle and start and end):
                report.skip("arcs", str(arc_id) if arc_id else None, "missing_arc_id_circle_or_endpoints")
                continue
            op = {"op": "add_arc", "id": str(arc_id), "circle": str(circle), "start": str(start), "end": str(end)}
            if "central_angle" in item:
                op["central_angle"] = str(item["central_angle"])
            if "orientation" in item:
                op["orientation"] = str(item["orientation"])
            ops.append(op)
        return ops

    def _polygon_ops(self, record: Dict[str, Any], report: LegacyAdapterReport) -> List[Dict[str, Any]]:
        ops = []
        raw_polygons = []
        raw_polygons.extend(_iter_items(record.get("polygons") or []))
        raw_polygons.extend(_iter_items(record.get("shapes") or []))
        for entity in _iter_items(record.get("entities") or []):
            if entity.get("vertices") and not entity.get("faces"):
                raw_polygons.append(entity)
        for item in raw_polygons:
            polygon_id = item.get("id") or item.get("name")
            vertices = item.get("vertices")
            if polygon_id and vertices and len(vertices) >= 3:
                ops.append({"op": "add_polygon", "id": str(polygon_id), "vertices": [str(v) for v in vertices]})
            else:
                report.skip("polygons", str(polygon_id) if polygon_id else None, "missing_polygon_id_or_vertices")
        return ops

    def _face_ops(self, record: Dict[str, Any], report: LegacyAdapterReport) -> List[Dict[str, Any]]:
        ops = []
        for item in _iter_items(record.get("faces") or []):
            face_id = item.get("id") or item.get("name")
            vertices = item.get("vertices")
            if face_id and vertices and len(vertices) >= 3:
                ops.append({"op": "add_polygon", "id": str(face_id), "vertices": [str(v) for v in vertices]})
            else:
                report.skip("faces", str(face_id) if face_id else None, "missing_face_id_or_vertices")
        return ops

    def _polyhedron_ops(self, record: Dict[str, Any], report: LegacyAdapterReport) -> List[Dict[str, Any]]:
        ops = []
        faces_by_id = {
            str(item.get("id") or item.get("name")): item
            for item in _iter_items(record.get("faces") or [])
            if item.get("id") or item.get("name")
        }
        for item in _iter_items(record.get("polyhedra") or []):
            op = _polyhedron_op(item, faces_by_id)
            if op:
                ops.append(op)
            else:
                report.skip("polyhedra", str(item.get("id") or item.get("name")) if item else None, "missing_polyhedron_vertices_or_faces")
        for item in _iter_items(record.get("entities") or []):
            if item.get("faces") and item.get("vertices"):
                op = _polyhedron_op(item, faces_by_id)
                if op:
                    ops.append(op)
                else:
                    report.skip("polyhedra", str(item.get("id") or item.get("name")) if item else None, "unusable_entity_polyhedron")
        return ops

    def _region_ops(self, record: Dict[str, Any], report: LegacyAdapterReport) -> List[Dict[str, Any]]:
        ops = []
        for item in _iter_items(record.get("regions") or []):
            region_id = item.get("id") or item.get("name")
            if not region_id:
                report.skip("regions", None, "missing_region_id")
                continue
            if item.get("kind") == "sector_region":
                ops.append(
                    {
                        "op": "mark_sector_region",
                        "id": str(region_id),
                        "circle": str(item["circle"]),
                        "start": str(item["start"]),
                        "end": str(item["end"]),
                        "central_angle": str(item["central_angle"]),
                    }
                )
            elif "polygon" in item:
                ops.append({"op": "mark_polygon_region", "id": str(region_id), "polygon": str(item["polygon"])})
            elif "boundary" in item:
                ops.append({"op": "mark_polygon_region", "id": str(region_id), "vertices": [str(v) for v in item["boundary"]]})
            else:
                report.skip("regions", str(region_id), "unsupported_region_shape")
        ops.extend(_shadow_region_ops(record, report))
        return ops

    def _target(self, record: Dict[str, Any]) -> Dict[str, Any] | None:
        target = record.get("target")
        if isinstance(target, dict):
            target_type = target.get("type") or target.get("target_type")
            target_object = target.get("object") or target.get("id")
            if target_type and target_object:
                return {"op": "set_target", "type": str(target_type), "object": str(target_object)}
        if record.get("target_type") and record.get("target_object"):
            return {"op": "set_target", "type": str(record["target_type"]), "object": str(record["target_object"])}
        return None


def _iter_items(value: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(value, dict):
        for item_id, item in value.items():
            if isinstance(item, dict):
                payload = dict(item)
                payload.setdefault("id", item_id)
                yield payload
        return
    for item in value:
        if isinstance(item, dict):
            yield item


def _coord(payload: Any) -> List[str] | None:
    if isinstance(payload, dict):
        if "coord" in payload:
            return [str(v) for v in payload["coord"]]
        if "coords" in payload:
            return [str(v) for v in payload["coords"]]
        if "x" in payload and "y" in payload:
            coord = [_scalar(payload["x"]), _scalar(payload["y"])]
            if "z" in payload:
                coord.append(_scalar(payload["z"]))
            return [str(v) for v in coord]
    if isinstance(payload, (list, tuple)) and len(payload) in {2, 3}:
        return [str(v) for v in payload]
    return None


def _scalar(value: Any) -> Any:
    if isinstance(value, dict):
        if "expr" in value:
            return value["expr"]
        if "value" in value:
            return value["value"]
    return value


def _polyhedron_op(item: Dict[str, Any], faces_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any] | None:
    poly_id = item.get("id") or item.get("name")
    vertices = item.get("vertices")
    raw_faces = item.get("face_vertices") or item.get("faces")
    if not poly_id or not vertices or not raw_faces:
        return None
    faces: List[List[str]] = []
    for face in raw_faces:
        if isinstance(face, str) and face in faces_by_id:
            face_vertices = faces_by_id[face].get("vertices")
            if face_vertices:
                faces.append([str(v) for v in face_vertices])
        elif isinstance(face, (list, tuple)):
            faces.append([str(v) for v in face])
    if not faces:
        return None
    return {"op": "add_polyhedron", "id": str(poly_id), "vertices": [str(v) for v in vertices], "faces": faces}


def _shadow_region_ops(record: Dict[str, Any], report: LegacyAdapterReport) -> List[Dict[str, Any]]:
    shadow_regions = record.get("shadow_regions")
    if shadow_regions is None:
        return []
    if isinstance(shadow_regions, int):
        if shadow_regions > 0:
            report.skip("shadow_regions", None, "shadow_region_count_without_geometry", count=shadow_regions)
        return []
    if not isinstance(shadow_regions, list):
        report.skip("shadow_regions", None, "unsupported_shadow_regions_payload", payload_type=type(shadow_regions).__name__)
        return []

    ops = []
    for index, item in enumerate(shadow_regions):
        if not isinstance(item, dict):
            report.skip("shadow_regions", f"shadow_region_{index}", "unsupported_shadow_region_item", payload_type=type(item).__name__)
            continue
        region_id = str(item.get("id") or item.get("name") or f"shadow_region_{index}")
        polygon = item.get("polygon") or item.get("polygon_id")
        if polygon:
            ops.append(
                {
                    "op": "mark_polygon_region",
                    "id": region_id,
                    "polygon": str(polygon),
                    "metadata": {"source": "legacy_shadow_regions", "legacy_index": index},
                }
            )
            continue
        boundary = item.get("boundary") or item.get("vertices") or item.get("points")
        if boundary and len(boundary) >= 3:
            ops.append(
                {
                    "op": "mark_polygon_region",
                    "id": region_id,
                    "vertices": [str(v) for v in boundary],
                    "metadata": {"source": "legacy_shadow_regions", "legacy_index": index},
                }
            )
            continue
        report.skip("shadow_regions", region_id, "missing_shadow_region_polygon_or_boundary")
    return ops


def _record_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    image_paths = {
        key: record[key]
        for key in ("raw_path", "annotated_raw_path", "shaded_path", "annotated_shaded_path")
        if record.get(key)
    }
    if image_paths:
        metadata["image_paths"] = image_paths
        metadata["has_raw_image"] = any(key in image_paths for key in ("raw_path", "annotated_raw_path"))
        metadata["has_shaded_image"] = any(key in image_paths for key in ("shaded_path", "annotated_shaded_path"))
    for key in ("description", "shader_enabled", "shadow_type", "base_id", "enhance_id", "is_base", "completed_rounds", "timeout_occurred"):
        if key in record:
            metadata[key] = record[key]
    if isinstance(record.get("metadata"), dict):
        metadata["legacy_metadata"] = dict(record["metadata"])
    if "shadow_regions" in record:
        shadow_regions = record["shadow_regions"]
        if isinstance(shadow_regions, int):
            metadata["shadow_region_count"] = shadow_regions
            metadata["shadow_region_geometry_available"] = False
        elif isinstance(shadow_regions, list):
            metadata["shadow_region_count"] = len(shadow_regions)
            metadata["shadow_region_geometry_available"] = any(
                isinstance(item, dict)
                and (item.get("polygon") or item.get("polygon_id") or item.get("boundary") or item.get("vertices") or item.get("points"))
                for item in shadow_regions
            )
        else:
            metadata["shadow_region_payload_type"] = type(shadow_regions).__name__
            metadata["shadow_region_geometry_available"] = False
    return metadata


def _coverage_metrics(*, input_keys: List[str], unhandled_top_level_keys: List[str], op_type_counts: Dict[str, int], num_skipped: int) -> Dict[str, Any]:
    total_keys = len(input_keys)
    unhandled_keys = len(unhandled_top_level_keys)
    handled_keys = max(total_keys - unhandled_keys, 0)
    key_coverage = 1.0 if total_keys == 0 else handled_keys / total_keys
    generated_ops = sum(op_type_counts.values())
    attempted_items = generated_ops + num_skipped
    item_success_rate = 1.0 if attempted_items == 0 else generated_ops / attempted_items
    has_geometry_ops = any(op != "set_target" and count > 0 for op, count in op_type_counts.items())
    geometry_presence = 1.0 if has_geometry_ops else 0.0
    coverage_score = round(key_coverage * item_success_rate * geometry_presence, 4)
    return {
        "coverage_score": coverage_score,
        "key_coverage": round(key_coverage, 4),
        "item_success_rate": round(item_success_rate, 4),
        "geometry_presence": geometry_presence,
        "total_top_level_keys": total_keys,
        "handled_top_level_keys": handled_keys,
        "unhandled_top_level_keys": unhandled_keys,
        "generated_ops": generated_ops,
        "num_skipped": num_skipped,
    }


def _unhandled_keys(record: Dict[str, Any]) -> List[str]:
    handled = {
        "points",
        "vertices",
        "lines",
        "segments",
        "edges",
        "circles",
        "chords",
        "arcs",
        "polygons",
        "shapes",
        "entities",
        "faces",
        "polyhedra",
        "regions",
        "target",
        "target_type",
        "target_object",
        "description",
        "raw_path",
        "annotated_raw_path",
        "shaded_path",
        "annotated_shaded_path",
        "shader_enabled",
        "shadow_type",
        "shadow_regions",
        "base_id",
        "enhance_id",
        "is_base",
        "completed_rounds",
        "timeout_occurred",
        "metadata",
    }
    return sorted(str(key) for key in record.keys() if key not in handled)
