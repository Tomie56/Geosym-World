"""Compile coordinate-free model statements into the existing ParseGraph input."""

from __future__ import annotations

from typing import Any

from geosym_world.errors import InvalidGeometryInput
from geosym_world.schema.canonical import stable_checksum
from geosym_world.schema.parse_graph import (
    BoundaryReference, EvidencePrecision, ParseEvidence, ParseEvidenceKind, ParseGraph,
    ParseObject, ParseObjectKind, ParseOrigin, ParseQuantity, ParseReference,
    ParseRegion, ParseRelation, ParseTarget, math_expression_from_text,
)


def compile_semantic_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return _SemanticCompiler(payload).compile()


def compile_semantic_graph(payload: dict[str, Any], *, source_id: str, question: str = "") -> ParseGraph:
    normalized_payload, normalization_audit = _normalize_semantic_payload(payload)
    compiled = compile_semantic_payload(normalized_payload)
    compiled["metadata"]["semantic_payload"] = payload
    compiled["metadata"]["normalized_semantic_payload"] = normalized_payload
    evidence = ParseEvidence(
        id="semantic_source", source_id=source_id, kind=ParseEvidenceKind.SOURCE_DOCUMENT,
        precision=EvidencePrecision.IMAGE_ONLY, raw_content=question,
        metadata={"provenance": "vlm_geometry_semantics", "semantic_payload_checksum": stable_checksum(payload)},
    )
    common = {"confidence": 0.5, "score_kind": "uncalibrated_model", "evidence_refs": (evidence.id,)}
    objects = tuple(ParseObject(
        id=item["id"], kind=ParseObjectKind(item["type"]), origin=ParseOrigin.IMAGE_TEXT,
        references=tuple(ParseReference.from_dict(ref) for ref in item.get("references", [])),
        visible_name=item.get("label"), metadata=item.get("metadata", {}), **common,
    ) for item in compiled["objects"])
    relations = tuple(ParseRelation(
        id=item["id"], predicate=item["type"], origin=ParseOrigin(item["origin"]),
        arguments=tuple(ParseReference.from_dict(ref) for ref in item["arguments"]),
        metadata=item["metadata"], **common,
    ) for item in compiled["relations"])
    quantities = tuple(ParseQuantity(
        id=item["id"], kind=item["type"], origin=ParseOrigin.IMAGE if item["metadata"]["show_label"] else ParseOrigin.TEXT,
        arguments=tuple(ParseReference.from_dict(ref) for ref in item["arguments"]),
        expression=math_expression_from_text(str(item["value"])), raw_text=str(item["value"]),
        unit=str(item["unit"]).strip() or None if item.get("unit") is not None else None,
        metadata=item["metadata"], **common,
    ) for item in {item["id"]: item for item in compiled["quantities"]}.values())
    regions = tuple(ParseRegion(
        id=item["id"], boundary=tuple(BoundaryReference(ref) for ref in item["boundary"]),
        holes=tuple(item["holes"]), origin=ParseOrigin.IMAGE, metadata=item["metadata"], **common,
    ) for item in compiled["regions"])
    targets = tuple(ParseTarget(
        id=item["id"], kind=item["type"], mode=item["mode"], origin=ParseOrigin.IMAGE_TEXT,
        arguments=tuple(ParseReference.from_dict(ref) for ref in item["arguments"]),
        metadata=item["metadata"], **common,
    ) for item in compiled["targets"])
    return ParseGraph(
        id="semantic_graph_" + stable_checksum(source_id)[:16], source_id=source_id,
        evidence=(evidence,), objects=tuple(sorted(objects, key=lambda item: item.id)),
        relations=tuple(sorted(relations, key=lambda item: item.id)),
        quantities=tuple(sorted(quantities, key=lambda item: item.id)),
        regions=tuple(sorted(regions, key=lambda item: item.id)), targets=targets,
        metadata={
            **compiled["metadata"],
            "compilation": {"compiler": "geometry_semantics_v1", "dropped_candidates": 0},
            "normalization_audit": normalization_audit,
        },
    )


_SEMANTIC_FIELDS = {
    "points", "strokes", "circles", "arcs", "relations", "measurements", "regions", "target", "uncertain",
}
_REFERENCE_NAME_FIELDS = {
    "point", "center", "through", "ends", "segment", "line", "ray", "angle", "polygon",
}
_NON_POINT_REFERENCE_FIELDS = {"circle", "arc", "region"}
_RELATION_NOISE = {"basis", "mark", "diagram", "text", "image"}


def _normalize_semantic_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize model variation before the strict coordinate-free compiler.

    The semantic contract remains strict when called directly. External model output
    gets this one compatibility pass so that harmless schema variation does not erase
    the rest of an otherwise useful diagram. Every repair is recorded in the graph
    metadata and unresolved candidates remain hypotheses or are dropped explicitly.
    """

    if not isinstance(payload, dict):
        raise InvalidGeometryInput("Semantic response must be an object")
    value = {key: item for key, item in payload.items() if key in _SEMANTIC_FIELDS or key == "point_names"}
    audit: dict[str, Any] = {
        "normalizer": "semantic_payload_v1",
        "renamed": [],
        "inferred_points": [],
        "dropped_candidates": [],
        "field_aliases": [],
    }
    if "points" not in value or value.get("points") is None:
        value["points"] = value.pop("point_names", []) or []
        if "point_names" in payload:
            audit["field_aliases"].append({"from": "point_names", "to": "points"})
    elif "point_names" in value:
        audit["field_aliases"].append({"from": "point_names", "to": "points_ignored"})
        value.pop("point_names", None)
    for field in _SEMANTIC_FIELDS - {"target"}:
        if field == "uncertain":
            value[field] = list(value.get(field) or []) if isinstance(value.get(field) or [], list) else [str(value[field])]
        elif not isinstance(value.get(field), list):
            value[field] = []
            if field in payload:
                audit["field_aliases"].append({"from": field, "to": f"{field}=[]"})
    if value.get("target") is not None and not isinstance(value["target"], dict):
        audit["dropped_candidates"].append({"kind": "target", "reason": "target_not_mapping"})
        value["target"] = None

    declared_points = _clean_names(value["points"])
    value["points"] = list(dict.fromkeys(declared_points))
    name_maps = {kind: {} for kind in ("point", "circle", "arc", "region")}
    for name in value["points"]:
        name_maps["point"].setdefault(name, name)

    for collection, kind in (("circles", "circle"), ("arcs", "arc"), ("regions", "region")):
        normalized_items = []
        for index, raw in enumerate(value[collection]):
            if not isinstance(raw, dict):
                audit["dropped_candidates"].append({"kind": kind, "index": index, "reason": "not_mapping"})
                continue
            item = dict(raw)
            original = str(item.get("name", f"_{kind}_{index}")).strip()
            if not original:
                original = f"_{kind}_{index}"
            renamed = original
            if renamed in name_maps["point"] or any(renamed in mapping.values() for mapping in name_maps.values()):
                suffix = f"_{kind}"
                renamed = original + suffix
                counter = 2
                while any(renamed in mapping.values() for mapping in name_maps.values()):
                    renamed = f"{original}{suffix}_{counter}"
                    counter += 1
                audit["renamed"].append({"kind": kind, "from": original, "to": renamed})
            name_maps[kind].setdefault(original, renamed)
            item["name"] = renamed
            normalized_items.append(item)
        value[collection] = normalized_items

    point_aliases: dict[str, str] = {}
    inferred_candidates = _collect_semantic_point_candidates(value)
    for candidate in inferred_candidates:
        if candidate in name_maps["point"] or any(candidate in name_maps[kind] for kind in ("circle", "arc", "region")) or not candidate:
            continue
        if "_" in candidate or candidate.lower().startswith(("mid", "left", "right", "top", "bottom")):
            generated = f"_inferred_{len(point_aliases) + 1}"
            point_aliases[candidate] = generated
            name_maps["point"][candidate] = generated
            value["points"].append(generated)
            audit["inferred_points"].append({"from": candidate, "to": generated, "visible": False})
        else:
            name_maps["point"][candidate] = candidate
            value["points"].append(candidate)
            audit["inferred_points"].append({"from": candidate, "to": candidate, "visible": True})

    def rename_typed(value_to_rewrite: Any, field: str | None = None) -> Any:
        if isinstance(value_to_rewrite, list):
            return [rename_typed(item, field) for item in value_to_rewrite]
        if not isinstance(value_to_rewrite, dict):
            if isinstance(value_to_rewrite, str):
                if field in _NON_POINT_REFERENCE_FIELDS:
                    return name_maps[field][value_to_rewrite] if value_to_rewrite in name_maps[field] else value_to_rewrite
                if field in _REFERENCE_NAME_FIELDS or field in {"points", "on"}:
                    return name_maps["point"].get(value_to_rewrite, value_to_rewrite)
            return value_to_rewrite
        result = {}
        for key, item in value_to_rewrite.items():
            rewritten = rename_typed(item, key)
            if key in {"segment", "line", "ray"} and isinstance(rewritten, list) and len(rewritten) > 2:
                rewritten = [rewritten[0], rewritten[-1]]
            result[key] = rewritten
        return result

    for item in value["circles"]:
        for field in ("center", "through"):
            if field in item:
                item[field] = rename_typed(item[field], field)
    for item in value["arcs"]:
        for field in ("circle", "ends", "through"):
            if field in item:
                item[field] = rename_typed(item[field], field)
    normalized_regions = []
    generated_holes = []
    for region_index, raw_region in enumerate(value["regions"]):
        if not isinstance(raw_region, dict):
            continue
        region = dict(raw_region)
        normalized_holes = []
        for hole_index, hole in enumerate(region.get("holes", []) or []):
            if isinstance(hole, str):
                normalized_holes.append(hole)
                continue
            if isinstance(hole, dict) and isinstance(hole.get("polygon"), list):
                hole_name = f"_hole_{region_index + 1}_{hole_index + 1}"
                while any(item.get("name") == hole_name for item in value["regions"] + generated_holes):
                    hole_name += "_"
                generated_holes.append({
                    "name": hole_name,
                    "boundary": [{"polygon": rename_typed(hole["polygon"], "polygon")}],
                    "holes": [],
                    "shaded": False,
                })
                normalized_holes.append(hole_name)
                audit["field_aliases"].append({
                    "region": region_index,
                    "from": "polygon_hole",
                    "to": hole_name,
                })
                continue
            audit["dropped_candidates"].append({
                "kind": "region_hole",
                "index": region_index,
                "reason": "unsupported_hole_reference",
            })
        region["holes"] = normalized_holes
        normalized_regions.append(region)
    value["regions"] = normalized_regions + generated_holes
    for item in value["regions"]:
        for field in ("boundary", "holes"):
            if field in item:
                item[field] = rename_typed(item[field], field)
    for item in value["strokes"]:
        if isinstance(item, dict) and isinstance(item.get("points"), list):
            item["points"] = [name_maps["point"].get(str(point), str(point)) for point in item["points"]]

    normalized_relations = []
    for index, item in enumerate(value["relations"]):
        if not isinstance(item, dict):
            audit["dropped_candidates"].append({"kind": "relation", "index": index, "reason": "not_mapping"})
            continue
        current = dict(item)
        on = current.get("on")
        if isinstance(on, list):
            cleaned_on = [entry for entry in on if not (isinstance(entry, str) and entry.strip().lower() in _RELATION_NOISE)]
            if len(cleaned_on) != len(on):
                audit["field_aliases"].append({"relation": index, "from": "on_noise", "removed": len(on) - len(cleaned_on)})
            current["on"] = rename_typed(cleaned_on, "on")
        elif on is not None:
            current["on"] = rename_typed(on, "on")
        if not isinstance(current.get("on"), list):
            audit["dropped_candidates"].append({"kind": "relation", "index": index, "reason": "on_not_array"})
            continue
        relation_type = str(current.get("type", "")).strip().lower()
        if relation_type in {"diameter", "arc_measure", "value"}:
            audit["dropped_candidates"].append({"kind": "relation", "index": index, "reason": f"unsupported_relation_type:{relation_type}"})
            continue
        required = {"point_on_circle": 2, "point_on_arc": 2, "point_on_segment": 2, "point_on_line": 2, "point_on_ray": 2}
        if relation_type in required and len(current["on"]) < required[relation_type]:
            audit["dropped_candidates"].append({"kind": "relation", "index": index, "reason": "insufficient_arguments"})
            continue
        required_count = {
            "parallel": 2, "perpendicular": 2, "equal_length": 2, "equal_angle": 2,
            "similar": 2, "congruent": 2, "tangent": 2, "midpoint": 2,
            "angle_bisector": 2, "rectangle": 1, "rhombus": 1,
            "parallelogram": 1, "triangle": 1,
        }.get(relation_type)
        if required_count is not None and len(current["on"]) != required_count:
            audit["dropped_candidates"].append({"kind": "relation", "index": index, "reason": "invalid_argument_count"})
            continue
        if relation_type == "collinear" and len(current["on"]) < 3:
            audit["dropped_candidates"].append({"kind": "relation", "index": index, "reason": "invalid_argument_count"})
            continue
        if relation_type == "angle_bisector" and (
            not isinstance(current["on"][1], dict) or "ray" not in current["on"][1]
        ):
            audit["dropped_candidates"].append({"kind": "relation", "index": index, "reason": "bisector_without_ray"})
            continue
        normalized_relations.append(current)
    value["relations"] = normalized_relations

    measurement_aliases = {"arc_measure": "arc_length", "same_length": "length", "segment_length": "length"}
    normalized_measurements = []
    for index, item in enumerate(value["measurements"]):
        if not isinstance(item, dict):
            audit["dropped_candidates"].append({"kind": "measurement", "index": index, "reason": "not_mapping"})
            continue
        current = dict(item)
        kind = str(current.get("type", "")).strip().lower()
        if kind in measurement_aliases:
            current["type"] = measurement_aliases[kind]
            audit["field_aliases"].append({"measurement": index, "from": kind, "to": current["type"]})
        current["on"] = rename_typed(current.get("on"), "on")
        if kind == "angle" and not (
            isinstance(current["on"], dict) and set(current["on"]) == {"angle"}
        ):
            audit["dropped_candidates"].append({
                "kind": "measurement",
                "index": index,
                "reason": "angle_requires_three_point_reference",
            })
            continue
        normalized_measurements.append(current)
    value["measurements"] = normalized_measurements
    value["target"] = _normalize_semantic_target(value.get("target"), value["measurements"], name_maps, audit)
    value["uncertain"] = list(value.get("uncertain", [])) + [
        f"semantic normalizer: {entry}" for entry in audit["dropped_candidates"]
    ]
    return value, audit


def _clean_names(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if isinstance(value, str) and value.strip()]


def _collect_semantic_point_candidates(payload: dict[str, Any]) -> set[str]:
    candidates: set[str] = set()

    def collect(value: Any, field: str | None = None) -> None:
        if isinstance(value, str):
            if field in _REFERENCE_NAME_FIELDS or field in {"points", "on"}:
                candidates.add(value.strip())
            return
        if isinstance(value, list):
            for item in value:
                collect(item, field)
            return
        if not isinstance(value, dict):
            return
        for key, item in value.items():
            if key in _NON_POINT_REFERENCE_FIELDS:
                continue
            collect(item, key)

    for stroke in payload.get("strokes", []):
        if isinstance(stroke, dict):
            collect(stroke.get("points"), "points")
    for circle in payload.get("circles", []):
        if isinstance(circle, dict):
            collect(circle.get("center"), "center")
            collect(circle.get("through"), "through")
    for arc in payload.get("arcs", []):
        if isinstance(arc, dict):
            collect(arc.get("ends"), "ends")
            collect(arc.get("through"), "through")
    for collection in ("relations", "measurements"):
        for item in payload.get(collection, []):
            if isinstance(item, dict):
                collect(item.get("on"), "on")
    if isinstance(payload.get("target"), dict):
        collect(payload["target"].get("on"), "on")
    return {candidate for candidate in candidates if candidate and candidate.lower() not in _RELATION_NOISE}


def _normalize_semantic_target(
    target: Any,
    measurements: list[dict[str, Any]],
    name_maps: dict[str, dict[str, str]],
    audit: dict[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(target, dict):
        return None
    normalized = dict(target)
    kind = str(normalized.get("type", "")).strip().lower()
    on = normalized.get("on")
    if kind == "value":
        symbol = on.get("symbol") if isinstance(on, dict) else None
        match = next(
            (item for item in measurements if str(item.get("value", "")).strip() == str(symbol).strip()),
            None,
        ) if symbol else None
        if match is not None:
            normalized["type"] = str(match.get("type", "length"))
            normalized["on"] = match.get("on")
            audit["field_aliases"].append({"target": "value", "to": normalized["type"], "source": "matching_measurement"})
        else:
            inferred = _infer_measurement_type(on)
            if inferred is not None:
                normalized["type"] = inferred
                audit["field_aliases"].append({"target": "value", "to": inferred})
            else:
                normalized["type"] = "unknown"
                normalized["mode"] = "proof"
                audit["dropped_candidates"].append({"kind": "target", "reason": "unbound_value_symbol"})
                return None
    elif kind == "angle" and not (
        isinstance(on, dict) and set(on) == {"angle"}
    ):
        audit["dropped_candidates"].append({
            "kind": "target",
            "reason": "angle_requires_three_point_reference",
        })
        return None
    elif kind == "angle_sum":
        normalized["on"] = _rename_reference_names(on, name_maps)
    else:
        normalized["on"] = _rename_reference_names(on, name_maps)
    if kind == "length" and isinstance(normalized.get("on"), dict) and "circle" in normalized["on"]:
        normalized["type"] = "radius"
        audit["field_aliases"].append({"target": "length", "to": "radius", "reason": "circle_support"})
    elif kind == "length" and isinstance(normalized.get("on"), dict) and "arc" in normalized["on"]:
        normalized["type"] = "arc_length"
        audit["field_aliases"].append({"target": "length", "to": "arc_length", "reason": "arc_support"})
    if isinstance(normalized.get("on"), dict) and "region" in normalized["on"]:
        region_name = str(normalized["on"]["region"])
        if region_name not in name_maps["region"].values():
            audit["dropped_candidates"].append({"kind": "target", "reason": "unknown_region", "ref": region_name})
            return None
    return normalized


def _infer_measurement_type(reference: Any) -> str | None:
    if not isinstance(reference, dict) or len(reference) != 1:
        return None
    key = next(iter(reference))
    return {
        "segment": "length", "line": "length", "ray": "length", "angle": "angle",
        "circle": "radius", "arc": "arc_length", "polygon": "area", "region": "area",
    }.get(key)


def _rename_reference_names(value: Any, name_maps: dict[str, dict[str, str]]) -> Any:
    if isinstance(value, list):
        return [_rename_reference_names(item, name_maps) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in _NON_POINT_REFERENCE_FIELDS and isinstance(item, str):
                result[key] = name_maps[key].get(item, item)
            elif key in _REFERENCE_NAME_FIELDS:
                rewritten = _rename_reference_names(item, name_maps)
                if key in {"segment", "line", "ray"} and isinstance(rewritten, list) and len(rewritten) > 2:
                    rewritten = [rewritten[0], rewritten[-1]]
                result[key] = rewritten
            else:
                result[key] = _rename_reference_names(item, name_maps)
        return result
    if isinstance(value, str):
        return name_maps["point"].get(value, value)
    return value


class _SemanticCompiler:
    def __init__(self, payload: dict[str, Any]) -> None:
        allowed = {"points", "strokes", "circles", "arcs", "relations", "measurements", "regions", "target", "uncertain"}
        self._fields(payload, allowed, "response")
        for section in allowed - {"target"}:
            if not isinstance(payload.get(section, []), list):
                raise InvalidGeometryInput(f"Semantic {section} must be an array")
        self.payload = payload
        self.objects: dict[str, dict[str, Any]] = {}
        self.names: dict[str, tuple[str, str]] = {}
        self.relations: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _fields(value: Any, allowed: set[str], path: str) -> None:
        if not isinstance(value, dict) or set(value) - allowed:
            raise InvalidGeometryInput(f"Invalid semantic fields at {path}; expected {sorted(allowed)}")

    @staticmethod
    def _id(kind: str, definition: Any) -> str:
        return f"semantic_{kind}_{stable_checksum(definition)[:16]}"

    def _declare(self, name: str, kind: str) -> str:
        if not isinstance(name, str) or not name.strip() or name in self.names:
            raise InvalidGeometryInput(f"Invalid or duplicate semantic name: {name!r}")
        object_id = self._id(kind, name)
        self.names[name] = (kind, object_id)
        return object_id

    def _named(self, name: str, kind: str) -> str:
        if not isinstance(name, str) or self.names.get(name, (None,))[0] != kind:
            raise InvalidGeometryInput(f"Unknown semantic {kind}: {name!r}")
        return self.names[name][1]

    def _points(self, names: Any, minimum: int, maximum: int | None = None) -> list[str]:
        if not isinstance(names, list) or len(names) < minimum or (maximum is not None and len(names) > maximum):
            raise InvalidGeometryInput(f"Expected {minimum}..{maximum or 'many'} semantic points")
        points = [self._named(name, "point") for name in names]
        if len(set(points)) != len(points):
            raise InvalidGeometryInput("Repeated point in semantic support")
        return points

    def _object(self, kind: str, points: list[str]) -> str:
        canonical = sorted(points) if kind in {"segment", "line"} else points
        if kind == "polygon":
            canonical = min(sequence[offset:] + sequence[:offset] for sequence in (points, points[::-1]) for offset in range(len(points)))
        object_id = self._id(kind, canonical)
        roles = [f"vertex_{index}" for index in range(len(points))] if kind == "polygon" else ["start", "through" if kind == "ray" else "end"]
        self.objects.setdefault(object_id, {
            "id": object_id, "type": kind,
            "references": [{"role": role, "ref": point} for role, point in zip(roles, points)],
        })
        return object_id

    def _ref(self, value: Any) -> str:
        if isinstance(value, str):
            return self._named(value, "point")
        if not isinstance(value, dict) or len(value) != 1:
            raise InvalidGeometryInput(f"Invalid semantic reference: {value!r}")
        kind, definition = next(iter(value.items()))
        if kind in {"circle", "arc", "region"}:
            return self._named(definition, kind)
        if kind in {"segment", "line", "ray", "polygon"}:
            return self._object(kind, self._points(definition, 3 if kind == "polygon" else 2, None if kind == "polygon" else 2))
        raise InvalidGeometryInput(f"Unsupported semantic reference kind: {kind}")

    def _typed_ref(self, value: Any, kind: str) -> str:
        if isinstance(value, str):
            return self._named(value, kind)
        return self._ref(value)

    def _angle(self, value: Any, prefix: str = "") -> list[dict[str, str]]:
        self._fields(value, {"angle"}, "angle reference")
        points = self._points(value.get("angle"), 3, 3)
        return [{"role": prefix + role, "ref": point} for role, point in zip(("start", "vertex", "end"), points)]

    @staticmethod
    def _basis(statement: dict[str, Any]) -> str:
        basis = statement.get("basis", "diagram")
        if basis not in {"diagram", "mark", "text"}:
            raise InvalidGeometryInput(f"Invalid semantic evidence basis: {basis!r}")
        return basis

    def _relation_arguments(self, kind: str, on: Any) -> list[dict[str, str]]:
        pairs = {"parallel", "perpendicular", "equal_length", "equal_angle", "similar", "congruent", "tangent"}
        shapes = {"rectangle", "rhombus", "parallelogram", "triangle"}
        supports = {"point_on_segment": "segment", "point_on_line": "line", "point_on_ray": "ray", "point_on_circle": "circle", "point_on_arc": "arc"}
        if not isinstance(on, list):
            raise InvalidGeometryInput(f"Semantic {kind}.on must be an array")
        count = 1 if kind in shapes else 2
        if (kind == "collinear" and len(on) < 3) or (kind != "collinear" and len(on) != count):
            raise InvalidGeometryInput(f"Invalid semantic {kind} argument count")
        if kind == "equal_angle":
            return self._angle(on[0], "left_") + self._angle(on[1], "right_")
        if kind == "angle_bisector":
            angle = self._angle(on[0])
            self._fields(on[1], {"ray"}, "bisector ray")
            start, through = self._points(on[1].get("ray"), 2, 2)
            if start != angle[1]["ref"]:
                raise InvalidGeometryInput("Bisector ray must start at the angle vertex")
            self._ref(on[1])
            return angle + [{"role": "through", "ref": through}]
        if kind in supports and len(on) == 2:
            refs = [self._ref(on[0]), self._typed_ref(on[1], supports[kind])]
        elif kind == "tangent" and len(on) == 2 and isinstance(on[1], str):
            refs = [self._ref(on[0]), self._shape_ref(on[1])]
        else:
            refs = [self._ref(value) for value in on]
        kinds = [self.objects[ref]["type"] if ref in self.objects else "region" for ref in refs]
        if kind in pairs:
            expected = {"segment", "line", "ray"} if kind in {"parallel", "perpendicular"} else {"segment"} if kind == "equal_length" else {"polygon"} if kind in {"similar", "congruent"} else {"segment", "line", "ray", "circle", "arc"}
            if not set(kinds) <= expected or (kind == "tangent" and not {"circle", "arc"}.intersection(kinds)):
                raise InvalidGeometryInput(f"Invalid supports for semantic {kind}: {kinds}")
            roles = ["left", "right"]
        elif kind in supports or kind == "midpoint":
            support_kind = supports.get(kind, "segment")
            if kinds != ["point", support_kind]:
                raise InvalidGeometryInput(f"Invalid supports for semantic {kind}: {kinds}")
            roles = ["point", support_kind if support_kind in {"circle", "arc"} else "support"]
        elif kind == "collinear" and set(kinds) == {"point"}:
            roles = [f"point_{index}" for index in range(len(refs))]
        elif kind in shapes and kinds == ["polygon"]:
            roles = ["object"]
        else:
            raise InvalidGeometryInput(f"Unsupported semantic relation: {kind}")
        return [{"role": role, "ref": ref} for role, ref in zip(roles, refs)]

    def _relation(self, kind: str, arguments: list[dict[str, str]], source: str, basis: str = "diagram") -> None:
        relation_id = self._id("relation", [kind, arguments])
        relation = self.relations.setdefault(relation_id, {
            "id": relation_id, "type": kind, "arguments": arguments,
            "origin": "text" if basis == "text" else "image", "confidence": 0.5,
            "metadata": {"semantic_sources": [], "show_marker": False},
        })
        relation["metadata"]["semantic_sources"].append(source)
        relation["metadata"]["show_marker"] |= basis == "mark"

    def _value_arguments(self, kind: str, on: Any) -> list[dict[str, str]]:
        if kind == "angle":
            return self._angle(on)
        if kind == "ratio":
            if not isinstance(on, list) or len(on) != 2:
                raise InvalidGeometryInput("Semantic ratio requires numerator and denominator")
            return [{"role": role, "ref": self._ref(value)} for role, value in zip(("numerator", "denominator"), on)]
        if kind not in {"length", "radius", "diameter", "arc_length", "area", "perimeter"}:
            raise InvalidGeometryInput(f"Unsupported semantic measurement: {kind}")
        reference = self._ref(on)
        support_kind = self.objects[reference]["type"] if reference in self.objects else "region"
        expected = {"length": {"segment"}, "radius": {"circle"}, "diameter": {"circle"}, "arc_length": {"arc"}, "area": {"polygon", "region"}, "perimeter": {"polygon", "region", "circle"}}
        if support_kind not in expected[kind]:
            raise InvalidGeometryInput(f"Invalid {kind} measurement support: {support_kind}")
        return [{"role": "object", "ref": reference}]

    def _target_arguments(self, kind: str, on: Any, mode: str) -> list[dict[str, str]]:
        if mode == "value" and kind == "angle_sum":
            return [{"role": "object", "ref": self._ref(on)}]
        if mode == "value":
            return self._value_arguments(kind, on)
        return self._relation_arguments(kind, on)

    def _shape_ref(self, value: str) -> str:
        for kind in ("circle", "arc", "region", "point"):
            if value in self.names and self.names[value][0] == kind:
                return self.names[value][1]
        return self._ref(value)

    def compile(self) -> dict[str, Any]:
        for name in self.payload.get("points", []):
            point_id = self._declare(name, "point")
            self.objects[point_id] = {"id": point_id, "type": "point", "label": None if name.startswith("_") else name, "metadata": {"anonymous": name.startswith("_")}}
        for kind, fields in (("circle", {"name", "center", "through"}), ("arc", {"name", "circle", "ends", "through"}), ("region", {"name", "boundary", "holes", "shaded"})):
            for item in self.payload.get(kind + "s", []):
                self._fields(item, fields, kind)
                self._declare(item.get("name"), kind)
        for index, stroke in enumerate(self.payload.get("strokes", [])):
            self._fields(stroke, {"kind", "points"}, f"strokes[{index}]")
            kind = stroke.get("kind")
            if kind not in {"segment", "line", "ray"}:
                raise InvalidGeometryInput(f"Invalid semantic stroke kind: {kind}")
            points = self._points(stroke.get("points"), 2)
            support = self._object(kind, [points[0], points[-1]])
            if len(points) > 2:
                self._relation("ordered_points", [{"role": "support", "ref": support}] + [{"role": f"point_{offset}", "ref": point} for offset, point in enumerate(points)], f"strokes[{index}]")
                for point in points[1:-1]:
                    self._relation(f"point_on_{kind}", [{"role": "point", "ref": point}, {"role": "support", "ref": support}], f"strokes[{index}]")
        for kind in ("circle", "arc"):
            for item in self.payload.get(kind + "s", []):
                object_id = self._named(item["name"], kind)
                refs = []
                if kind == "circle" and item.get("center") is not None:
                    refs.append({"role": "center", "ref": self._named(item["center"], "point")})
                if kind == "arc":
                    refs.append({"role": "circle", "ref": self._named(item.get("circle"), "circle")})
                    endpoints = self._points(item.get("ends"), 2, 2)
                    refs.extend({"role": role, "ref": point} for role, point in zip(("start", "end"), endpoints))
                    for point in endpoints:
                        self._relation("point_on_circle", [{"role": "point", "ref": point}, refs[0]], item["name"])
                self.objects[object_id] = {"id": object_id, "type": kind, "references": refs}
                for point in self._points(item.get("through", []), 0):
                    self._relation(f"point_on_{kind}", [{"role": "point", "ref": point}, {"role": kind, "ref": object_id}], item["name"])
        regions = []
        for item in self.payload.get("regions", []):
            if not isinstance(item.get("boundary"), list) or not item["boundary"]:
                raise InvalidGeometryInput("Semantic region requires an explicit boundary")
            regions.append({"id": self._named(item["name"], "region"), "boundary": [self._ref(value) for value in item["boundary"]], "holes": [self._named(name, "region") for name in item.get("holes", [])], "metadata": {"shaded": item.get("shaded", False)}})
        for index, statement in enumerate(self.payload.get("relations", [])):
            self._fields(statement, {"type", "on", "basis"}, f"relations[{index}]")
            self._relation(statement.get("type"), self._relation_arguments(statement.get("type"), statement.get("on")), f"relations[{index}]", self._basis(statement))
        quantities = []
        for index, item in enumerate(self.payload.get("measurements", [])):
            self._fields(item, {"type", "on", "value", "unit", "basis"}, f"measurements[{index}]")
            if "value" not in item:
                raise InvalidGeometryInput(f"Missing value at measurements[{index}]")
            quantities.append({"id": self._id("quantity", item), "type": item.get("type"), "arguments": self._value_arguments(item.get("type"), item.get("on")), "value": item["value"], "unit": item.get("unit"), "metadata": {"show_label": self._basis(item) != "text", "semantic_source": f"measurements[{index}]"}})
        targets = []
        target = self.payload.get("target")
        if target is not None:
            self._fields(target, {"type", "on", "mode", "symbol"}, "target")
            mode = target.get("mode", "value")
            if mode not in {"value", "proof", "constraint"}:
                raise InvalidGeometryInput(f"Invalid semantic target mode: {mode}")
            arguments = self._target_arguments(target.get("type"), target.get("on"), mode)
            targets.append({"id": self._id("target", target), "type": target.get("type"), "arguments": arguments, "mode": mode, "metadata": {"value_symbol": target.get("symbol")}})
        return {
            "objects": list(self.objects.values()), "relations": list(self.relations.values()),
            "quantities": quantities, "regions": regions, "targets": targets,
            "metadata": {"extraction_contract": "geometry_semantics_v1", "semantic_payload": self.payload, "ambiguities": self.payload.get("uncertain", []), "coordinate_source": "none", "confidence_policy": "uncalibrated_default_0.5"},
        }
