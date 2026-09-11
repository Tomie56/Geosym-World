"""Public/private dataset records for image-to-geometry parser evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from geosym_world.errors import InvalidGeometryInput
from geosym_world.parser.contracts import ParserInput
from geosym_world.rendering.scene import RenderScene
from geosym_world.schema.canonical import stable_checksum
from geosym_world.schema.state import FormalGeometryState


PARSER_REFERENCE_SCHEMA_VERSION = "1.0"
_STATE_BUCKETS = (
    "points",
    "lines",
    "segments",
    "rays",
    "circles",
    "arcs",
    "polygons",
    "regions",
    "composite_regions",
    "planes",
    "polyhedra",
)


def build_parser_benchmark_pair(
    record: Mapping[str, Any],
    *,
    state: FormalGeometryState,
    scene: RenderScene,
    image_format: str | None = None,
    parser_input: ParserInput | Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a leak-free parser input and its evaluator-only reference."""

    record_id = str(record.get("id", ""))
    question = str(record.get("question", ""))
    if parser_input is None:
        public_input = ParserInput(id=record_id, image=_select_image(record, image_format), question=question)
    else:
        public_input = parser_input if isinstance(parser_input, ParserInput) else ParserInput.from_dict(parser_input)
        if public_input.question != question:
            raise InvalidGeometryInput("Provided ParserInput question does not match source record")
    parser_input_payload = public_input.to_dict()
    scene_payload = scene.to_dict()
    formal_state = state.to_dict()
    observable_state = _observable_state(formal_state, scene_payload, record.get("public_givens") or ())
    reference = {
        "schema_version": PARSER_REFERENCE_SCHEMA_VERSION,
        "id": public_input.id,
        "parser_input_checksum": stable_checksum(parser_input_payload),
        "observable_state": observable_state,
        "private_supervision": {
            "formal_state": formal_state,
            "source_record_id": record_id,
            "formal_state_checksum": state.checksum(),
            "scene": scene_payload,
            "public_givens": list(record.get("public_givens") or ()),
            "facts": list(record.get("facts") or ()),
            "derivation": dict(record.get("derivation") or {}),
            "construction_trace": list(record.get("construction_trace") or ()),
            "answer": record.get("answer"),
        },
    }
    reference["checksum"] = stable_checksum(reference)
    return parser_input_payload, reference


def validate_parser_reference(reference: Mapping[str, Any], parser_input: Mapping[str, Any]) -> None:
    if reference.get("schema_version") != PARSER_REFERENCE_SCHEMA_VERSION:
        raise InvalidGeometryInput("Unsupported parser reference schema version")
    if reference.get("id") != parser_input.get("id"):
        raise InvalidGeometryInput("Parser input and private reference IDs do not match")
    if reference.get("parser_input_checksum") != stable_checksum(parser_input):
        raise InvalidGeometryInput("Parser input checksum does not match private reference")
    checksum_payload = {key: value for key, value in reference.items() if key != "checksum"}
    if reference.get("checksum") != stable_checksum(checksum_payload):
        raise InvalidGeometryInput("Parser reference checksum does not match payload")


def _select_image(record: Mapping[str, Any], requested: str | None) -> str:
    formats = dict((record.get("image") or {}).get("formats") or {})
    if requested is not None:
        if requested not in formats:
            raise InvalidGeometryInput(f"Parser benchmark image format is unavailable: {requested}")
        return str(formats[requested])
    for candidate in ("png", "svg"):
        if candidate in formats:
            return str(formats[candidate])
    raise InvalidGeometryInput("Parser benchmark requires a rendered PNG or SVG image")


def _observable_state(
    formal_state: Mapping[str, Any],
    scene: Mapping[str, Any],
    public_givens: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    object_index = {
        object_id: (bucket, payload)
        for bucket in _STATE_BUCKETS
        for object_id, payload in (formal_state.get(bucket) or {}).items()
    }
    visible_ids = {
        str(primitive.get("id"))
        for primitive in scene.get("primitives") or ()
        if str(primitive.get("id")) in object_index
    }
    pending = list(visible_ids)
    while pending:
        object_id = pending.pop()
        _, payload = object_index[object_id]
        for dependency in _object_dependencies(payload):
            if dependency in object_index and dependency not in visible_ids:
                visible_ids.add(dependency)
                pending.append(dependency)

    objects = {bucket: {} for bucket in _STATE_BUCKETS}
    for object_id in sorted(visible_ids):
        bucket, payload = object_index[object_id]
        objects[bucket][object_id] = payload
    relations = [dict(given) for given in public_givens if given.get("kind") == "relation"]
    quantities = [dict(given) for given in public_givens if given.get("kind") == "quantity"]
    visual_evidence = [
        {
            "id": primitive.get("id"),
            "kind": primitive.get("kind"),
            "role": primitive.get("role"),
            "metadata": dict(primitive.get("metadata") or {}),
        }
        for primitive in scene.get("primitives") or ()
        if primitive.get("role") in {"marker", "auxiliary"} or str(primitive.get("id", "")).startswith("intersection:")
    ]
    rendered_label_ids = {
        str(label.get("id", "")).split(":", 1)[1]
        for label in scene.get("labels") or ()
        if str(label.get("id", "")).startswith("label:")
    }
    point_label_map = dict((scene.get("metadata") or {}).get("point_labels") or {})
    payload = {
        "schema_version": "1.0",
        "objects": objects,
        "relations": relations,
        "quantities": quantities,
        "point_labels": {point_id: point_label_map[point_id] for point_id in sorted(rendered_label_ids) if point_id in point_label_map},
        "visual_evidence": visual_evidence,
        "target": formal_state.get("target"),
    }
    payload["checksum"] = stable_checksum(payload)
    return payload


def _object_dependencies(payload: Mapping[str, Any]) -> set[str]:
    dependencies: set[str] = set()
    for key in ("start", "end", "through", "center", "circle"):
        value = payload.get(key)
        if isinstance(value, str):
            dependencies.add(value)
    for key in ("source_points", "vertices", "boundary", "holes", "components"):
        values = payload.get(key) or ()
        dependencies.update(str(value) for value in values if isinstance(value, str))
    for face in payload.get("faces") or ():
        dependencies.update(str(value) for value in face)
    return dependencies
