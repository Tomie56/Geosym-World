"""Minimal OpenRouter adapter for strict, proposal-only ParseGraph extraction."""

from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.error
from importlib.resources import files
from pathlib import Path
from typing import Any

from PIL import Image

from geosym_world.errors import InvalidGeometryInput
from geosym_world.schema import ParseGraph
from geosym_world.schema.state import PROOF_TARGET_TYPES
from geosym_world.schema.parse_graph import PARSE_RELATION_TYPES, QUANTITY_KINDS
from geosym_world.schema.parse_graph import (
    EvidencePrecision, ParseConflictGroup, ParseEvidence, ParseEvidenceKind, ParseObject,
    ParseQuantity, ParseRegion, ParseRelation, ParseTarget,
    math_expression_from_text,
)
from geosym_world.schema.relation_normalization import normalize_relation_type, normalize_target_type
from geosym_world.schema.canonical import stable_checksum

from .contracts import ParserInput, validate_parser_input
from .prediction import ExternalParseGraphPrediction
from .provider_config import provider_base_url
from .semantic_compiler import compile_semantic_graph
from .text_claims import TextClaimExtractor


PROMPT_VERSION = "geometry-semantics-zh-v12"
EXTRACTION_POLICY_VERSION = "structural-seed-v3"


def _json_content(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    payload = json.loads(content)
    if isinstance(payload, list):
        return {"objects": payload}
    if not isinstance(payload, dict):
        preview = content[:160].replace("\n", " ")
        raise ValueError(
            f"OpenRouter response must be a JSON object, got {type(payload).__name__}: {preview!r}"
        )
    return payload


def _model_references(
    item: dict[str, Any],
    fields: tuple[str, ...],
) -> list[dict[str, str]]:
    """Turn common VLM reference shorthands into ParseReference payloads."""
    raw = item.get("arguments")
    field_name = "arguments"
    if raw is None:
        for first, second in (("source", "target"), ("subject", "object")):
            if first in item and second in item:
                raw = [item[first], item[second]]
                field_name = "relation"
                break
    if raw is None and "left" in item and "right" in item:
        raw = [item["left"], item["right"]]
        field_name = "relation"
    if raw is None:
        for field in fields:
            if field in item:
                raw = item[field]
                field_name = field
                break
    if raw is None:
        indexed: list[tuple[int, str, Any]] = []
        for key, value in item.items():
            match = re.fullmatch(r"(?:point|vertex|object|segment|line|support)_(\d+)", str(key))
            if match:
                indexed.append((int(match.group(1)), str(key).rsplit("_", 1)[0], value))
        if indexed:
            raw = [value for _, _, value in sorted(indexed)]
            field_name = "indexed"
    if isinstance(raw, dict):
        indexed_values = []
        for key, value in raw.items():
            match = re.fullmatch(r"(?:point|vertex|object|segment|line|support)_(\d+)", str(key))
            if match:
                indexed_values.append((int(match.group(1)), value))
        if indexed_values:
            raw = [value for _, value in sorted(indexed_values)]
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    references: list[dict[str, str]] = []
    for index, value in enumerate(raw):
        role = f"{field_name}_{index}"
        reference = value
        if isinstance(value, dict):
            role = str(value.get("role", role))
            reference = value.get("ref", value.get("object", value.get("id")))
        if reference is not None and str(reference).strip():
            references.append({"role": role, "ref": str(reference).strip()})
    return references


def _canonical_relation_arguments(
    predicate: str,
    arguments: list[dict[str, str]],
) -> list[dict[str, str]]:
    if predicate in {"parallel", "perpendicular", "equal_length", "equal_angle", "similar", "congruent"} and len(arguments) == 2:
        return [
            {"role": role, "ref": argument["ref"]}
            for role, argument in zip(("left", "right"), arguments)
        ]
    if predicate in {"incidence", "point_on_line", "point_on_segment", "point_on_ray", "midpoint"} and len(arguments) == 2:
        point = next((item for item in arguments if item.get("role") == "point"), arguments[0])
        support = next((item for item in arguments if item is not point), arguments[1])
        return [
            {"role": "point", "ref": point["ref"]},
            {"role": "support", "ref": support["ref"]},
        ]
    if predicate == "collinear" and len(arguments) >= 3:
        return [
            {"role": f"point_{index}", "ref": argument["ref"]}
            for index, argument in enumerate(arguments)
        ]
    if predicate == "midpoint" and len(arguments) == 3:
        point = next((item for item in arguments if item.get("role") == "point"), arguments[0])
        endpoints = [item for item in arguments if item is not point]
        return [{"role": "point", "ref": point["ref"]}] + [
            {"role": role, "ref": item["ref"]}
            for role, item in zip(("start", "end"), endpoints)
        ]
    return arguments


def _model_evidence_refs(item: dict[str, Any], evidence_ids: set[str]) -> list[str]:
    raw = item.get("evidence_refs", item.get("evidence_ids", []))
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    refs = [str(ref) for ref in raw if str(ref) in evidence_ids]
    return refs or [next(iter(evidence_ids))]


def _model_confidence(item: dict[str, Any]) -> float:
    try:
        return max(0.0, min(1.0, float(item.get("confidence", 0.5))))
    except (TypeError, ValueError):
        return 0.5


def _model_expression(item: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    expression = item.get("expression")
    if isinstance(expression, dict):
        return expression, str(item.get("raw_text", item.get("value", ""))) or None
    raw = expression if isinstance(expression, str) else None
    if raw is None:
        for key in ("value", "value_symbol", "raw_value", "answer"):
            if item.get(key) is not None:
                raw = str(item[key])
                break
    if raw is None or not raw.strip():
        return None, None
    try:
        return math_expression_from_text(raw).to_dict(), raw
    except (InvalidGeometryInput, TypeError, ValueError):
        return None, raw


def _is_pixel_point(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(coordinate, (int, float)) for coordinate in value)
    )


def _normalize_linear_endpoints(item: dict[str, Any]) -> None:
    start = item.pop("start", None)
    end = item.pop("end", item.pop("through", None))
    if start is None and end is None:
        return

    if item.get("kind") == "arc":
        references = list(item.get("references", [])) if isinstance(item.get("references"), list) else []
        existing_roles = {
            str(reference.get("role"))
            for reference in references
            if isinstance(reference, dict)
        }
        for role, value in (("start", start), ("end", end)):
            if isinstance(value, str) and value.strip() and role not in existing_roles:
                references.append({"role": role, "ref": value.strip()})
        item["references"] = references
        return

    if item.get("kind") not in {"segment", "line", "ray"}:
        metadata = dict(item.get("metadata", {})) if isinstance(item.get("metadata"), dict) else {}
        metadata["ignored_linear_endpoints"] = True
        item["metadata"] = metadata
        return

    geometry = dict(item.get("geometry", {})) if isinstance(item.get("geometry"), dict) else {}
    references = list(item.get("references", [])) if isinstance(item.get("references"), list) else []
    if _is_pixel_point(start) and _is_pixel_point(end):
        geometry.setdefault("p1", list(start))
        geometry.setdefault("p2", list(end))
    elif isinstance(start, str) and start.strip() and isinstance(end, str) and end.strip():
        second_role = "through" if item["kind"] == "ray" else "end"
        existing_roles = {
            str(reference.get("role"))
            for reference in references
            if isinstance(reference, dict)
        }
        if "start" not in existing_roles:
            references.append({"role": "start", "ref": start.strip()})
        if second_role not in existing_roles:
            references.append({"role": second_role, "ref": end.strip()})
    else:
        metadata = dict(item.get("metadata", {})) if isinstance(item.get("metadata"), dict) else {}
        metadata["endpoint_decode_error"] = "mixed_or_invalid_top_level_endpoints"
        item["metadata"] = metadata
    item["geometry"] = geometry
    item["references"] = references


def _normalize_object_kind(value: Any) -> str:
    normalized = str(value or "unknown").strip().lower().replace("-", "_").replace(" ", "_")
    return {
        "vertex": "point",
        "edge": "segment",
        "side": "segment",
        "line_segment": "segment",
        "linesegment": "segment",
        "triangle": "polygon",
        "quadrilateral": "polygon",
        "label": "text",
        "text_label": "text",
        "right_angle_marker": "marker",
        "parallel_marker": "marker",
        "equal_length_marker": "marker",
        "equal_angle_marker": "marker",
    }.get(normalized, normalized)


def _normalize_compact_object_geometry(item: dict[str, Any]) -> None:
    geometry = dict(item.get("geometry", {})) if isinstance(item.get("geometry"), dict) else {}
    references = list(item.get("references", [])) if isinstance(item.get("references"), list) else []
    x = item.pop("x", None)
    y = item.pop("y", None)
    bbox = item.pop("bbox", None)
    width = item.pop("width", None)
    height = item.pop("height", None)
    center = item.pop("center", None)
    radius = item.pop("radius_px", item.pop("radius", None))

    pixel = [x, y] if isinstance(x, (int, float)) and isinstance(y, (int, float)) else None
    kind = item.get("kind")
    if kind == "point" and pixel is not None:
        geometry.setdefault("pixel", pixel)
    elif kind == "circle":
        if _is_pixel_point(center):
            geometry.setdefault("center", list(center))
        elif pixel is not None:
            geometry.setdefault("center", pixel)
        elif isinstance(center, str) and center.strip():
            references.append({"role": "center", "ref": center.strip()})
        if isinstance(radius, (int, float)) and radius > 0:
            geometry.setdefault("radius_px", radius)
    elif kind == "arc":
        if isinstance(radius, (int, float)) and radius > 0:
            geometry.setdefault("radius_px", radius)
        if item.get("orientation") in {"cw", "ccw"}:
            geometry.setdefault("orientation", item["orientation"])
        if _is_pixel_point(center):
            geometry.setdefault("center", list(center))
        elif isinstance(center, str) and center.strip():
            references.append({"role": "center", "ref": center.strip()})
        for role, value in (("circle", item.get("circle")), ("start", item.get("start")), ("end", item.get("end"))):
            if isinstance(value, str) and value.strip():
                references.append({"role": role, "ref": value.strip()})
    elif kind in {"text", "marker"}:
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            geometry.setdefault("bbox", list(bbox))
        elif pixel is not None:
            half_width = float(width) / 2 if isinstance(width, (int, float)) and width > 0 else 6.0
            half_height = float(height) / 2 if isinstance(height, (int, float)) and height > 0 else 6.0
            geometry.setdefault(
                "bbox",
                [pixel[0] - half_width, pixel[1] - half_height, pixel[0] + half_width, pixel[1] + half_height],
            )
    elif pixel is not None or center is not None or radius is not None or bbox is not None:
        metadata = dict(item.get("metadata", {})) if isinstance(item.get("metadata"), dict) else {}
        metadata["ignored_compact_geometry"] = True
        item["metadata"] = metadata
    item["geometry"] = geometry
    item["references"] = references


def _normalize_object_references(item: dict[str, Any]) -> None:
    raw = item.get("references")
    if not isinstance(raw, list):
        return
    if all(isinstance(reference, dict) for reference in raw):
        return
    values = [str(reference).strip() for reference in raw if str(reference).strip()]
    if item.get("kind") == "polygon":
        item["references"] = [
            {"role": f"vertex_{index}", "ref": reference}
            for index, reference in enumerate(values)
        ]
    elif item.get("kind") in {"segment", "line", "ray"} and len(values) == 2:
        second_role = "through" if item["kind"] == "ray" else "end"
        item["references"] = [
            {"role": "start", "ref": values[0]},
            {"role": second_role, "ref": values[1]},
        ]
    else:
        item["references"] = []


def _structural_extraction_policy() -> dict[str, Any]:
    return {
        "mode": "constraint_only",
        "numeric_completeness": "not_required",
        "coordinate_recovery": "not_attempted",
        "reason": "structure_is_extracted_before_exact_coordinate_recovery",
        "policy_version": EXTRACTION_POLICY_VERSION,
    }


def parse_one(
    parser_input: ParserInput | dict[str, Any],
    *,
    model: str,
    api_key: str,
    base_url: str | None = None,
    timeout: float = 120.0,
    raw_output: str | Path | None = None,
) -> ExternalParseGraphPrediction:
    public_input = validate_parser_input(parser_input)
    base_url = base_url or provider_base_url()
    image = Path(public_input.image).read_bytes()
    import base64

    with Image.open(public_input.image) as source_image:
        image_size = [int(source_image.width), int(source_image.height)]
        mime_type = Image.MIME[source_image.format]
    image_url = f"data:{mime_type};base64," + base64.b64encode(image).decode("ascii")
    extraction_policy = _structural_extraction_policy()
    result, graph_payload = _request_json_completion(
        model=model,
        api_key=api_key,
        base_url=base_url,
        image_url=image_url,
        instruction=_parsegraph_instruction(public_input),
        timeout=timeout,
    )
    debug_path = raw_output or os.environ.get("GEOSYM_OPENROUTER_RAW_RESPONSE")
    if debug_path:
        Path(debug_path).parent.mkdir(parents=True, exist_ok=True)
        Path(debug_path).write_text(json.dumps(graph_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if "parse_graph" in graph_payload and isinstance(graph_payload["parse_graph"], dict):
        graph_payload = graph_payload["parse_graph"]
    if "strokes" in graph_payload:
        graph = compile_semantic_graph(graph_payload, source_id=public_input.id, question=public_input.question)
        return ExternalParseGraphPrediction.create(
            public_input, backend="openrouter", model=model, model_revision=model,
            prompt_version=PROMPT_VERSION, parse_graph=graph, request_id=result.get("id"),
            decoding={"temperature": 0, "extraction_policy": extraction_policy},
            usage={"parse": result.get("usage", {})},
        )
    graph_payload.setdefault("schema_version", "1.0")
    graph_payload.setdefault("id", "graph_" + public_input.id)
    graph_payload["source_id"] = public_input.id
    for collection in ("evidence", "objects", "relations", "quantities", "regions", "targets", "conflicts"):
        graph_payload.setdefault(collection, [])
    raw_collection_shapes = {
        collection: type(graph_payload.get(collection)).__name__
        for collection in ("objects", "relations", "quantities", "regions", "targets", "conflicts")
    }
    raw_collection_counts = {
        collection: len(graph_payload.get(collection, []))
        if isinstance(graph_payload.get(collection), (list, tuple, dict)) else 0
        for collection in raw_collection_shapes
    }
    dropped_relations: list[dict[str, str]] = []
    dropped_quantities: list[dict[str, str]] = []
    dropped_regions: list[dict[str, str]] = []
    dropped_targets: list[dict[str, str]] = []
    unresolved_references: list[dict[str, Any]] = []
    if isinstance(graph_payload.get("objects"), dict):
        raw_objects = graph_payload["objects"]
        object_groups = {
            "points": "point",
            "segments": "segment",
            "lines": "line",
            "rays": "ray",
            "circles": "circle",
            "arcs": "arc",
            "polygons": "polygon",
            "text": "text",
            "labels": "text",
            "markers": "marker",
        }
        flattened_objects: list[dict[str, Any]] = []
        for key, value in raw_objects.items():
            if key in object_groups and isinstance(value, list):
                for index, member in enumerate(value):
                    if not isinstance(member, dict):
                        continue
                    item = dict(member)
                    item.setdefault("id", f"{key}_{index}")
                    item.setdefault("type", object_groups[key])
                    flattened_objects.append(item)
                continue
            if isinstance(value, dict):
                flattened_objects.append({**value, "id": str(value.get("id", key))})
        graph_payload["objects"] = flattened_objects
    elif not isinstance(graph_payload.get("objects"), list):
        graph_payload["objects"] = []
    for collection in ("relations", "quantities", "regions", "targets", "conflicts"):
        if isinstance(graph_payload.get(collection), dict):
            graph_payload[collection] = [
                {**value, "id": str(value.get("id", key))}
                if isinstance(value, dict)
                else {"id": str(key)}
                for key, value in graph_payload[collection].items()
            ]
        elif not isinstance(graph_payload.get(collection), list):
            graph_payload[collection] = []
    if isinstance(graph_payload.get("evidence"), str):
        graph_payload["evidence"] = [{"id": "model_evidence", "source_id": public_input.id, "kind": "source_document", "precision": "image_only", "raw_content": graph_payload["evidence"], "metadata": {}}]
    normalized_evidence = []
    for index, item in enumerate(graph_payload.get("evidence", [])):
        if not isinstance(item, dict) or not {"id", "kind", "precision", "source_id"} <= set(item):
            normalized_evidence.append({"id": "model_evidence_" + str(index), "source_id": public_input.id, "kind": "source_document", "precision": "image_only", "raw_content": str(item), "metadata": {}})
        else:
            normalized_evidence.append(item)
    graph_payload["evidence"] = normalized_evidence
    evidence_ids = {item.get("id") for item in graph_payload.get("evidence", []) if isinstance(item, dict)}
    if not evidence_ids:
        graph_payload["evidence"] = [{"id": "model_evidence", "source_id": public_input.id, "kind": "source_document", "precision": "image_only", "raw_content": "model observation", "metadata": {}}]
        evidence_ids = {"model_evidence"}
    metadata = graph_payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    for key in ("diagram_summary", "topology", "layout_hints", "display_annotations", "ambiguities"):
        if key in graph_payload:
            metadata[key] = graph_payload.pop(key)
    reported_image_size = metadata.get("image_size")
    metadata["image_size"] = image_size
    if reported_image_size is not None and reported_image_size != image_size:
        metadata["model_image_size"] = reported_image_size
    metadata["extraction_policy"] = extraction_policy
    metadata["visual_policy"] = "pixel_observation_only"
    stripped_exact_coordinate_ids: list[str] = []
    graph_payload["metadata"] = metadata
    for item in graph_payload.get("objects", []):
        if isinstance(item, dict):
            object_metadata = dict(item.get("metadata", {})) if isinstance(item.get("metadata"), dict) else {}
            for key in ("layout", "layout_hint", "display", "display_annotations", "annotations", "semantic_role", "subgraph"):
                if key in item:
                    object_metadata[key] = item.pop(key)
            item["metadata"] = object_metadata
            for key in set(item) - {"id", "kind", "origin", "confidence", "score_kind", "references", "evidence_refs", "visible_name", "aliases", "visible", "geometry", "metadata", "type", "label", "provenance", "x", "y", "start", "end", "through", "circle", "bbox", "width", "height", "center", "radius", "radius_px"}:
                item.pop(key, None)
            item.setdefault("id", "model_object_" + str(len([x for x in graph_payload.get("objects", []) if isinstance(x, dict)])))
            if item.get("visible_name") is None and item.get("label") is not None:
                item["visible_name"] = str(item["label"])
        if isinstance(item, dict) and "type" in item and "kind" not in item:
            item["kind"] = _normalize_object_kind(item.pop("type"))
            item["origin"] = "image_text"
            item["score_kind"] = "model"
            item["evidence_refs"] = ["model_evidence"]
            item["visible_name"] = item.pop("label", None)
            item["geometry"] = item.get("geometry", {})
            item["metadata"] = {
                **dict(item.get("metadata", {})),
                "provenance": item.pop("provenance", "model"),
            }
        if isinstance(item, dict):
            item["kind"] = _normalize_object_kind(item.get("kind"))
            item.pop("type", None)
            item.setdefault("origin", "image_text")
            item.setdefault("score_kind", "model")
            item.setdefault("evidence_refs", ["model_evidence"])
            item["evidence_refs"] = [ref for ref in item.get("evidence_refs", []) if ref in evidence_ids] or [next(iter(evidence_ids))]
            item.setdefault("confidence", 0.5)
            item.setdefault("metadata", {"provenance": "model"})
            if isinstance(item.get("metadata"), dict):
                item["metadata"].setdefault("provenance", "model")
            _normalize_linear_endpoints(item)
            _normalize_object_references(item)
            item.pop("label", None)
            _normalize_compact_object_geometry(item)
            geometry = item.get("geometry", {})
            if not geometry and {"x", "y"} <= set(item):
                geometry = {"x": item.pop("x"), "y": item.pop("y")}
                item["geometry"] = geometry
            if isinstance(geometry, dict) and "exact_coords" in geometry:
                stripped_exact_coordinate_ids.append(str(item.get("id", "")))
                geometry = dict(geometry)
                geometry.pop("exact_coords", None)
                item["geometry"] = geometry
            if item.get("kind") == "point" and "pixel" not in geometry and {"x", "y"} <= set(geometry):
                item["geometry"] = {"pixel": [float(geometry["x"]), float(geometry["y"])]}
            elif item.get("kind") in {"segment", "line", "ray"}:
                if "p1" not in geometry and "start" in geometry:
                    geometry["p1"] = geometry.pop("start")
                if "p2" not in geometry and "end" in geometry:
                    geometry["p2"] = geometry.pop("end")
                item["geometry"] = geometry
            if item.get("kind") in {"segment", "line", "ray"}:
                geometry = item.get("geometry", {})
                if any(key in geometry and (not isinstance(geometry[key], (list, tuple)) or len(geometry[key]) != 2 or any(not isinstance(value, (int, float)) for value in geometry[key])) for key in ("p1", "p2")):
                    item["kind"] = "unknown"
                    item["geometry"] = {}
    if stripped_exact_coordinate_ids:
        metadata["model_geometry_policy_audit"] = {
            "stripped_exact_coordinate_ids": sorted(set(stripped_exact_coordinate_ids)),
            "reason": "external_model_coordinates_are_observations_until_constraint_completion_and_admission",
        }
        graph_payload["metadata"] = metadata
    localized = []
    for item in graph_payload.get("objects", []):
        geometry = item.get("geometry", {})
        points = []
        if isinstance(geometry.get("pixel"), (list, tuple)):
            points = [geometry["pixel"]]
        elif isinstance(geometry.get("p1"), (list, tuple)) and isinstance(geometry.get("p2"), (list, tuple)):
            points = [geometry["p1"], geometry["p2"]]
        elif isinstance(geometry.get("center"), (list, tuple)):
            points = [geometry["center"]]
        elif isinstance(geometry.get("bbox"), (list, tuple)) and len(geometry["bbox"]) == 4:
            points = [geometry["bbox"][:2], geometry["bbox"][2:]]
        if points and all(len(point) == 2 and all(isinstance(value, (int, float)) for value in point) for point in points):
            xs, ys = zip(*points)
            evidence_id = "pixel_evidence_" + str(item["id"])
            localized.append({"id": evidence_id, "source_id": public_input.id, "kind": "image_region", "precision": "localized", "bbox": [min(xs) - 6, min(ys) - 6, max(xs) + 6, max(ys) + 6], "metadata": {"provenance": "model_pixel_geometry"}})
            item["evidence_refs"] = [evidence_id]
    graph_payload["evidence"].extend(localized)
    evidence_by_id = {}
    for item in graph_payload["evidence"]:
        if isinstance(item, dict):
            evidence_by_id.setdefault(str(item.get("id", "model_evidence")), item)
    graph_payload["evidence"] = list(evidence_by_id.values())
    relation_fields = (
        "source", "target", "subject", "object", "objects", "points",
        "lines", "segments", "vertices", "supports", "left", "right", "point", "support",
    )
    relation_aliases = {
        "is_perpendicular_to": "perpendicular",
        "on_segment": "point_on_segment_ratio",
        "on_line": "point_on_line",
        "on_circle": "point_on_circle",
        "is_part_of": "contains",
        "intersects": "intersection",
        "intersection_of": "intersection",
        "intersections": "intersection",
        "between": "ordered_points",
        "is_similar_to": "similar",
        "is_congruent_to": "congruent",
        "tangent_internal": "tangent",
        "quarter_circle": "arc",
        "explicit_marker": "__display_annotation__",
        "explicit_markers": "__display_annotation__",
        "question_conditions": "__display_annotation__",
    }
    # Models frequently classify visible equal-length ticks as a quantity even
    # though the formal graph represents them as a relation. Preserve the
    # candidate and compile it into the canonical relation collection.
    compiled_quantity_relations = []
    retained_quantities = []
    for item in graph_payload.get("quantities", []):
        if isinstance(item, dict) and str(item.get("type", item.get("kind", ""))).strip().lower() in {
            "equal_length", "same_length", "equal_segments",
        }:
            relation_item = dict(item)
            relation_item["type"] = "equal_length"
            compiled_quantity_relations.append(relation_item)
        else:
            retained_quantities.append(item)
    graph_payload["quantities"] = retained_quantities
    graph_payload["relations"].extend(compiled_quantity_relations)
    for index, item in enumerate(graph_payload.get("relations", [])):
        if not isinstance(item, dict):
            continue
        allowed_relation_fields = {
            "id", "predicate", "arguments", "origin", "confidence", "score_kind",
            "evidence_refs", "evidence_ids", "dependencies", "metadata", "source",
            "target", "subject", "object", "objects", "points", "lines", "segments",
            "vertices", "supports", "left", "right", "point", "support", "type", "provenance",
        }
        for key in list(item):
            if key not in allowed_relation_fields and not re.fullmatch(
                r"(?:point|vertex|object|segment|line|support)_\d+", str(key)
            ):
                item.pop(key, None)
        predicate = item.get("predicate", item.get("type", ""))
        predicate = relation_aliases.get(str(predicate).strip().lower(), str(predicate))
        item["predicate"] = normalize_relation_type(predicate)
        item.setdefault("id", f"model_relation_{index}")
        item["arguments"] = _canonical_relation_arguments(
            item["predicate"],
            _model_references(item, relation_fields),
        )
        provenance = item.get("provenance")
        for key in relation_fields + ("type", "provenance"):
            item.pop(key, None)
        for key in list(item):
            if re.fullmatch(r"(?:point|vertex|object|segment|line|support)_\d+", str(key)):
                item.pop(key, None)
        if item["predicate"] not in PARSE_RELATION_TYPES or not item["arguments"]:
            item["_skip"] = True
            continue
        item["origin"] = "image_text"
        item["confidence"] = _model_confidence(item)
        item["score_kind"] = "model"
        item["evidence_refs"] = _model_evidence_refs(item, evidence_ids)
        relation_metadata = dict(item.get("metadata", {})) if isinstance(item.get("metadata"), dict) else {}
        relation_metadata["provenance"] = str(provenance or relation_metadata.get("provenance", "model"))
        item["metadata"] = relation_metadata
    graph_payload["relations"] = [item for item in graph_payload.get("relations", []) if isinstance(item, dict) and not item.get("_skip")]
    graph_payload["regions"] = [item for item in graph_payload.get("regions", []) if isinstance(item, dict) and item.get("boundary")]
    seen_ids: set[str] = set()
    id_remap: dict[str, str] = {}
    for collection in ("objects", "relations", "quantities", "regions", "targets"):
        for index, item in enumerate(graph_payload.get(collection, [])):
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id", f"model_{collection}_{index}"))
            if item_id in seen_ids:
                item_id = f"{item_id}_{index}"
            original_id = str(item.get("id", item_id))
            item["id"] = item_id
            id_remap.setdefault(original_id, item_id)
            seen_ids.add(item_id)
    object_ids = {item["id"] for item in graph_payload.get("objects", []) if isinstance(item, dict)}
    visible_name_ids: dict[str, str] = {}
    suffix_ids: dict[str, str | None] = {}
    for item in graph_payload.get("objects", []):
        if not isinstance(item, dict) or not item.get("id"):
            continue
        if item.get("visible_name"):
            name = str(item["visible_name"]).strip().upper()
            if name and name not in visible_name_ids:
                visible_name_ids[name] = str(item["id"])
            elif name:
                visible_name_ids.pop(name, None)
        object_id = str(item["id"])
        if "_" in object_id:
            suffix = object_id.rsplit("_", 1)[-1].upper()
            suffix_ids[suffix] = object_id if suffix not in suffix_ids else None

    def resolve_ref(value: str) -> str:
        candidate = str(value).strip()
        if candidate in object_ids:
            return candidate
        if candidate in id_remap:
            return id_remap[candidate]
        visible_id = visible_name_ids.get(candidate.upper())
        if visible_id is not None:
            return visible_id
        for prefix in ("point_", "line_", "segment_"):
            if candidate.startswith(prefix):
                candidate = candidate[len(prefix):]
                if candidate in object_ids:
                    return candidate
                if candidate in id_remap:
                    return id_remap[candidate]
                visible_id = visible_name_ids.get(candidate.upper())
                if visible_id is not None:
                    return visible_id
        suffix_id = suffix_ids.get(candidate.upper())
        if suffix_id is not None:
            return suffix_id
        return id_remap.get(candidate, candidate)

    def ensure_explicit_named_segment(value: str) -> str:
        """Compile a short named segment only when both endpoint points exist."""

        resolved = resolve_ref(value)
        if resolved in object_ids:
            return resolved
        token = str(value).strip()
        if token.lower().startswith(("segment_", "line_", "seg_")):
            token = token.split("_", 1)[1]
        if len(token) != 2 or not token.isalpha():
            return resolved
        start_id = visible_name_ids.get(token[0].upper())
        end_id = visible_name_ids.get(token[1].upper())
        object_by_id = {
            str(item.get("id")): item
            for item in graph_payload.get("objects", [])
            if isinstance(item, dict)
        }
        if start_id is None or end_id is None:
            return resolved
        if object_by_id.get(start_id, {}).get("kind") != "point" or object_by_id.get(end_id, {}).get("kind") != "point":
            return resolved
        segment_id = f"segment_{token[0].upper()}_{token[1].upper()}"
        if segment_id not in object_ids:
            start_geometry = object_by_id.get(start_id, {}).get("geometry", {})
            end_geometry = object_by_id.get(end_id, {}).get("geometry", {})
            segment_geometry = {}
            if _is_pixel_point(start_geometry.get("pixel")) and _is_pixel_point(end_geometry.get("pixel")):
                segment_geometry = {
                    "p1": list(start_geometry["pixel"]),
                    "p2": list(end_geometry["pixel"]),
                }
            graph_payload["objects"].append({
                "id": segment_id,
                "kind": "segment",
                "origin": "image_text",
                "confidence": 0.9,
                "score_kind": "explicit_pair_reference",
                "references": [
                    {"role": "start", "ref": start_id},
                    {"role": "end", "ref": end_id},
                ],
                "evidence_refs": ["model_evidence"],
                "visible_name": token.upper(),
                "aliases": [],
                "visible": True,
                "geometry": segment_geometry,
                "metadata": {
                    "provenance": "compiled_explicit_pair_reference",
                    "semantic_role": "named_support",
                },
            })
            object_ids.add(segment_id)
            visible_name_ids[token.upper()] = segment_id
        return segment_id

    def ensure_segment_for_endpoints(start_id: str, end_id: str) -> str | None:
        """Return an existing support or compile a semantic support segment."""

        endpoint_set = {start_id, end_id}
        for candidate in graph_payload.get("objects", []):
            if not isinstance(candidate, dict) or candidate.get("kind") not in {"segment", "line", "ray"}:
                continue
            refs = candidate.get("references", [])
            endpoints = {
                str(reference.get("ref"))
                for reference in refs
                if isinstance(reference, dict) and reference.get("role") in {"start", "end", "through"}
            }
            if endpoints == endpoint_set:
                return str(candidate["id"])
        segment_id = f"segment_{start_id}_{end_id}"
        if segment_id not in object_ids:
            start_geometry = object_by_id.get(start_id, {}).get("geometry", {})
            end_geometry = object_by_id.get(end_id, {}).get("geometry", {})
            segment_geometry = {}
            if _is_pixel_point(start_geometry.get("pixel")) and _is_pixel_point(end_geometry.get("pixel")):
                segment_geometry = {
                    "p1": list(start_geometry["pixel"]),
                    "p2": list(end_geometry["pixel"]),
                }
            graph_payload["objects"].append({
                "id": segment_id,
                "kind": "segment",
                "origin": "image_text",
                "confidence": 0.8,
                "score_kind": "compiled_support",
                "references": [
                    {"role": "start", "ref": start_id},
                    {"role": "end", "ref": end_id},
                ],
                "evidence_refs": ["model_evidence"],
                "visible": True,
                "geometry": segment_geometry,
                "metadata": {
                    "provenance": "compiled_ordered_points_support",
                    "requires_seed_verification": True,
                },
            })
            object_ids.add(segment_id)
            object_by_id[segment_id] = graph_payload["objects"][-1]
        return segment_id

    def ensure_placeholder_reference(value: str) -> str:
        """Keep an unresolved model reference without inventing its geometry."""

        resolved = resolve_ref(value)
        if resolved in object_ids:
            return resolved
        if not resolved or any(
            resolved == str(item.get("id"))
            for collection in ("regions", "quantities", "relations", "targets")
            for item in graph_payload.get(collection, [])
            if isinstance(item, dict)
        ):
            return resolved
        placeholder = {
            "id": resolved,
            "kind": "unknown",
            "origin": "image_text",
            "confidence": 0.2,
            "score_kind": "unresolved_reference",
            "references": [],
            "evidence_refs": ["model_evidence"],
            "visible": False,
            "geometry": {},
            "metadata": {
                "provenance": "unresolved_model_reference",
                "requires_grounding": True,
            },
        }
        graph_payload["objects"].append(placeholder)
        object_ids.add(resolved)
        object_by_id[resolved] = placeholder
        return resolved
    for item in graph_payload.get("objects", []):
        if not isinstance(item, dict):
            continue
        references = item.get("references", [])
        if not isinstance(references, list):
            continue
        for reference in references:
            if isinstance(reference, dict) and "ref" in reference:
                reference["ref"] = resolve_ref(reference["ref"])
    point_pixels = {
        item["id"]: item.get("geometry", {}).get("pixel")
        for item in graph_payload.get("objects", [])
        if isinstance(item, dict)
        and item.get("kind") == "point"
        and _is_pixel_point(item.get("geometry", {}).get("pixel"))
    }
    for item in graph_payload.get("objects", []):
        if not isinstance(item, dict) or item.get("kind") not in {"segment", "line", "ray", "polygon"}:
            continue
        referenced_pixels = [
            point_pixels[reference["ref"]]
            for reference in item.get("references", [])
            if isinstance(reference, dict) and reference.get("ref") in point_pixels
        ]
        if item.get("kind") in {"segment", "line", "ray"} and len(referenced_pixels) >= 2:
            geometry = dict(item.get("geometry", {}))
            geometry.setdefault("p1", list(referenced_pixels[0]))
            geometry.setdefault("p2", list(referenced_pixels[1]))
            item["geometry"] = geometry
        if len(referenced_pixels) >= 2:
            xs = [float(point[0]) for point in referenced_pixels]
            ys = [float(point[1]) for point in referenced_pixels]
            evidence_id = "pixel_evidence_" + str(item["id"])
            graph_payload["evidence"].append({
                "id": evidence_id,
                "source_id": public_input.id,
                "kind": "image_region",
                "precision": "localized",
                "bbox": [min(xs) - 6, min(ys) - 6, max(xs) + 6, max(ys) + 6],
                "metadata": {"provenance": "compiled_from_referenced_point_pixels"},
            })
            item["evidence_refs"] = [evidence_id]
    evidence_by_id = {}
    for item in graph_payload["evidence"]:
        if isinstance(item, dict):
            evidence_by_id.setdefault(str(item.get("id", "model_evidence")), item)
    graph_payload["evidence"] = list(evidence_by_id.values())
    object_by_id = {
        str(item.get("id")): item
        for item in graph_payload.get("objects", [])
        if isinstance(item, dict)
    }
    for item in list(graph_payload.get("objects", [])):
        if not isinstance(item, dict) or not isinstance(item.get("references"), list):
            continue
        for reference in item["references"]:
            if not isinstance(reference, dict) or "ref" not in reference:
                continue
            reference["ref"] = ensure_placeholder_reference(
                ensure_explicit_named_segment(str(reference["ref"]))
            )

    def point_reference(ref: str) -> bool:
        return object_by_id.get(ref, {}).get("kind") == "point"

    def expand_relation(item: dict[str, Any]) -> list[dict[str, Any]]:
        predicate = item.get("predicate")
        arguments = item.get("arguments", [])
        resolved = [
            {**argument, "ref": resolve_ref(argument["ref"])}
            for argument in arguments
            if isinstance(argument, dict) and argument.get("ref")
        ]
        if predicate == "__display_annotation__":
            return []
        if predicate in {"incidence", "point_on_line", "point_on_segment", "point_on_ray"}:
            points = [argument for argument in resolved if point_reference(argument["ref"])]
            supports = [argument for argument in resolved if not point_reference(argument["ref"])]
            if not points or not supports:
                return []
            return [
                {
                    **item,
                    "id": item["id"] if len(supports) == 1 else f"{item['id']}_{index}",
                    "arguments": [
                        {"role": "point", "ref": points[0]["ref"]},
                        {"role": "support", "ref": support["ref"]},
                    ],
                }
                for index, support in enumerate(supports)
            ]
        if predicate == "ordered_points":
            support = next((argument["ref"] for argument in resolved if not point_reference(argument["ref"])), None)
            points = [argument["ref"] for argument in resolved if point_reference(argument["ref"])]
            if len(points) < 2:
                return []
            if support is None:
                support = ensure_segment_for_endpoints(points[0], points[-1])
            if support is None:
                return []
            return [{
                **item,
                "arguments": [
                    {"role": "support", "ref": support},
                    *({"role": f"point_{index}", "ref": ref} for index, ref in enumerate(points)),
                ],
            }]
        if predicate == "collinear":
            points = [argument["ref"] for argument in resolved if point_reference(argument["ref"])]
            if len(points) >= 3:
                return [{**item, "arguments": [{"role": f"point_{index}", "ref": ref} for index, ref in enumerate(points)]}]
            return []
        return [{**item, "arguments": resolved}]

    valid_relations = []
    for item in graph_payload.get("relations", []):
        arguments = item.get("arguments", [])
        for argument in arguments:
            if isinstance(argument, dict) and "ref" in argument:
                argument["ref"] = ensure_placeholder_reference(
                    ensure_explicit_named_segment(argument["ref"])
                )
        expanded_items = expand_relation(item)
        if not expanded_items and item.get("predicate") != "__display_annotation__":
            dropped_relations.append({
                "id": str(item.get("id", "")),
                "reason": "unsupported_or_unbound_relation",
            })
        accepted = False
        for expanded in expanded_items:
            expanded_arguments = expanded.get("arguments", [])
            if expanded_arguments:
                valid_relations.append(expanded)
                accepted = True
                missing = [
                    str(argument.get("ref"))
                    for argument in expanded_arguments
                    if argument.get("ref") not in object_ids
                ]
                if missing:
                    unresolved_references.append({"kind": "relation", "id": str(expanded.get("id", "")), "refs": missing})
        if expanded_items and not accepted and item.get("predicate") != "__display_annotation__":
            dropped_relations.append({
                "id": str(item.get("id", "")),
                "reason": "relation_references_unknown_candidates",
            })
    graph_payload["relations"] = valid_relations
    normalized_quantities = []
    region_ids = {str(region.get("id")) for region in graph_payload.get("regions", [])}
    for index, item in enumerate(graph_payload.get("quantities", [])):
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", item.get("type", "unknown"))).strip().lower()
        if kind not in QUANTITY_KINDS:
            dropped_quantities.append({
                "id": str(item.get("id", f"model_quantity_{index}")),
                "reason": "unknown_quantity_kind",
            })
            continue
        references = _model_references(
            item,
            ("associated_with", "object", "segment", "line", "circle", "arc", "region", "target", "objects"),
        )
        for argument in references:
            if argument["ref"] not in region_ids:
                argument["ref"] = ensure_placeholder_reference(
                    ensure_explicit_named_segment(argument["ref"])
                )
        expression, raw_text = _model_expression(item)
        if not references or expression is None:
            dropped_quantities.append({
                "id": str(item.get("id", f"model_quantity_{index}")),
                "reason": "quantity_unbound_or_expression_invalid",
            })
            continue
        missing = [argument["ref"] for argument in references if argument["ref"] not in object_ids | region_ids]
        if missing:
            unresolved_references.append({"kind": "quantity", "id": str(item.get("id", f"model_quantity_{index}")), "refs": missing})
        quantity_metadata = dict(item.get("metadata", {})) if isinstance(item.get("metadata"), dict) else {}
        quantity_metadata["provenance"] = str(item.get("provenance", quantity_metadata.get("provenance", "model")))
        normalized_quantities.append({
            "id": item.get("id", f"model_quantity_{index}"), "kind": kind,
            "arguments": references, "expression": expression, "origin": "image_text",
            "confidence": _model_confidence(item), "score_kind": "model",
            "unit": str(item.get("unit")).strip() if item.get("unit") is not None and str(item.get("unit")).strip() else None,
            "raw_text": raw_text, "evidence_refs": _model_evidence_refs(item, evidence_ids),
            "dependencies": item.get("dependencies", []), "metadata": quantity_metadata,
        })
    graph_payload["quantities"] = normalized_quantities
    normalized_targets = []
    for index, item in enumerate(graph_payload.get("targets", [])):
        if not isinstance(item, dict):
            continue
        kind = normalize_target_type(str(item.get("kind", item.get("type", item.get("target_type", "unknown")))))
        if not kind or kind == "unknown":
            kind = str(item.get("kind", item.get("type", "unknown"))).strip().lower()
        arguments = _model_references(item, ("object", "objects", "target", "entity", "region"))
        for argument in arguments:
            argument["ref"] = ensure_placeholder_reference(
                ensure_explicit_named_segment(argument["ref"])
            )
        region_ids = {str(region.get("id")) for region in graph_payload.get("regions", []) if isinstance(region, dict)}
        missing = [argument["ref"] for argument in arguments if argument["ref"] not in object_ids | region_ids]
        mode = str(item.get("mode", "proof" if kind in PROOF_TARGET_TYPES else "value")).strip().lower()
        mode = {
            "numeric": "value", "find": "value", "solve": "value", "compute": "value", "query": "value",
            "calculation": "value", "calculate": "value", "radius": "value",
            "theorem": "proof", "prove": "proof", "condition": "constraint",
        }.get(mode, mode)
        target_metadata = dict(item.get("metadata", {})) if isinstance(item.get("metadata"), dict) else {}
        if missing:
            target_metadata["unresolved_references"] = missing
            unresolved_references.append({"kind": "target", "id": str(item.get("id", f"model_target_{index}")), "refs": missing})
        if mode not in {"value", "proof", "constraint"}:
            raise InvalidGeometryInput(f"Target {item.get('id', index)} has unknown mode: {mode}")
        expression, raw_text = _model_expression(item)
        target_metadata["provenance"] = str(item.get("provenance", target_metadata.get("provenance", "model")))
        normalized_targets.append({
            "id": item.get("id", f"model_target_{index}"), "kind": kind, "mode": mode,
            "arguments": arguments, "origin": "image_text", "confidence": _model_confidence(item),
            "score_kind": "model", "expression": expression, "evidence_refs": _model_evidence_refs(item, evidence_ids),
            "dependencies": item.get("dependencies", []), "metadata": target_metadata,
        })
    graph_payload["targets"] = normalized_targets
    normalized_regions = []
    for index, item in enumerate(graph_payload.get("regions", [])):
        if not isinstance(item, dict):
            dropped_regions.append({"id": f"model_region_{index}", "reason": "region_not_mapping"})
            continue
        boundary = item.get("boundary", item.get("references", item.get("objects", [])))
        if not isinstance(boundary, (list, tuple)):
            boundary = [boundary]
        boundary_refs = []
        unresolved_boundary_refs = []
        for value in boundary:
            if isinstance(value, dict):
                ref = value.get("ref", value.get("object", value.get("id")))
                orientation = str(value.get("orientation", "unknown"))
            else:
                ref = value
                orientation = "unknown"
            if ref is not None and str(ref).strip():
                resolved_ref = resolve_ref(str(ref))
                if resolved_ref not in object_ids:
                    resolved_ref = ensure_explicit_named_segment(str(ref))
                if resolved_ref not in object_ids:
                    resolved_ref = ensure_placeholder_reference(resolved_ref)
                if resolved_ref in object_ids:
                    boundary_refs.append({"ref": resolved_ref, "orientation": orientation})
                else:
                    boundary_refs.append({"ref": resolved_ref, "orientation": orientation})
                    unresolved_boundary_refs.append(str(ref))
        expression = item.get("expression")
        if not boundary_refs and expression is None:
            dropped_regions.append({
                "id": str(item.get("id", f"model_region_{index}")),
                "reason": (
                    "region_boundary_references_unknown"
                    if unresolved_boundary_refs else "region_has_no_boundary_or_expression"
                ),
            })
            continue
        normalized_regions.append({
            "id": str(item.get("id", f"model_region_{index}")),
            "boundary": boundary_refs,
            "origin": (
                str(item.get("origin", "image_text"))
                if str(item.get("origin", "image_text")) in {"image", "text", "image_text", "native", "compiled"}
                else "image_text"
            ),
            "confidence": _model_confidence(item),
            "score_kind": str(item.get("score_kind", "model")),
            "holes": [resolve_ref(str(ref)) for ref in item.get("holes", []) if str(ref).strip()],
            "components": [resolve_ref(str(ref)) for ref in item.get("components", []) if str(ref).strip()],
            "expression": expression,
            "evidence_refs": _model_evidence_refs(item, evidence_ids),
            "metadata": {
                **(dict(item.get("metadata", {})) if isinstance(item.get("metadata"), dict) else {}),
                "label": item.get("label"),
                "source_type": item.get("type"),
                "provenance": str(item.get("provenance", "model")),
                "unresolved_boundary_refs": unresolved_boundary_refs,
            },
        })
        if unresolved_boundary_refs:
            unresolved_references.append({
                "kind": "region",
                "id": str(item.get("id", f"model_region_{index}")),
                "refs": unresolved_boundary_refs,
            })
    graph_payload["regions"] = normalized_regions
    normalized_conflicts = []
    uncompiled_conflicts = []
    for index, item in enumerate(graph_payload.get("conflicts", [])):
        if not isinstance(item, dict):
            continue
        members = item.get("members", item.get("objects", []))
        if not isinstance(members, (list, tuple)):
            members = [members]
        members = [str(member) for member in members if str(member).strip()]
        if len(members) < 2 or len(set(members)) != len(members):
            uncompiled_conflicts.append({
                "id": str(item.get("id", f"model_conflict_{index}")),
                "raw": dict(item),
                "reason": "conflict_requires_two_distinct_candidates",
            })
            continue
        normalized_conflicts.append({
            "id": str(item.get("id", f"model_conflict_{index}")),
            "members": members,
            "status": str(item.get("status", "unresolved")),
            "selected": item.get("selected"),
            "reason": str(item.get("reason", item.get("alternative", ""))),
        })
    graph_payload["conflicts"] = normalized_conflicts
    if uncompiled_conflicts:
        metadata.setdefault("uncompiled_conflicts", []).extend(uncompiled_conflicts)
    metadata["normalization_audit"] = {
        "input_shapes": raw_collection_shapes,
        "input_counts": raw_collection_counts,
        "output_counts": {
            collection: len(graph_payload.get(collection, []))
            for collection in ("objects", "relations", "quantities", "regions", "targets", "conflicts")
        },
        "dropped": {
            "relations": dropped_relations,
            "quantities": dropped_quantities,
            "regions": dropped_regions,
            "targets": dropped_targets,
        },
        "uncompiled_conflicts": uncompiled_conflicts,
        "unresolved_references": unresolved_references,
        "compiled_object_count": max(
            0,
            len(graph_payload.get("objects", [])) - raw_collection_counts.get("objects", 0),
        ),
    }
    text_claim_audit = _augment_explicit_text_claims(
        graph_payload,
        question=public_input.question,
        source_id=public_input.id,
    )
    metadata["text_claim_extraction"] = text_claim_audit
    graph_payload["metadata"] = metadata
    for collection in ("evidence", "objects", "relations", "quantities", "regions", "targets", "conflicts"):
        graph_payload[collection] = sorted(graph_payload.get(collection, []), key=lambda item: str(item.get("id", "")))
    graph_payload["evidence"] = [ParseEvidence.from_dict(item).to_dict() for item in graph_payload["evidence"]]
    graph_payload["objects"] = [ParseObject.from_dict(item).to_dict() for item in graph_payload["objects"]]
    graph_payload["relations"] = [ParseRelation.from_dict(item).to_dict() for item in graph_payload["relations"]]
    graph_payload["quantities"] = [ParseQuantity.from_dict(item).to_dict() for item in graph_payload["quantities"]]
    graph_payload["regions"] = [ParseRegion.from_dict(item).to_dict() for item in graph_payload["regions"]]
    graph_payload["targets"] = [ParseTarget.from_dict(item).to_dict() for item in graph_payload["targets"]]
    graph_payload["conflicts"] = [ParseConflictGroup.from_dict(item).to_dict() for item in graph_payload["conflicts"]]
    required = {"schema_version", "id", "source_id", "evidence", "objects", "relations", "quantities", "regions", "targets", "conflicts", "metadata"}
    if required.issubset(graph_payload):
        graph_payload.pop("checksum", None)
        graph_payload["checksum"] = stable_checksum(graph_payload)
    if not required.union({"checksum"}).issubset(graph_payload):
        raise ValueError(
            "Model did not return strict ParseGraph; keys=" + ",".join(sorted(graph_payload))
        )
    graph = ParseGraph.from_dict(graph_payload)
    return ExternalParseGraphPrediction.create(
        public_input, backend="openrouter", model=model,
        model_revision=model, prompt_version=PROMPT_VERSION, parse_graph=graph,
        request_id=result.get("id"),
        decoding={"temperature": 0, "extraction_policy": extraction_policy},
        usage={"parse": result.get("usage", {})},
    )


def _augment_explicit_text_claims(
    graph_payload: dict[str, Any],
    *,
    question: str,
    source_id: str,
) -> dict[str, Any]:
    """Preserve explicit public-text claims when the VLM omits them.

    This is a deterministic evidence pass, not a solver. It creates only
    text-backed hypotheses and leaves coordinates absent so later grounding
    and admission still control whether a claim can affect state.
    """
    claims = TextClaimExtractor().extract(question)
    if not claims:
        return {"extractor": "text_theorem_v0", "claims_seen": 0, "claims_added": 0}

    evidence = graph_payload.setdefault("evidence", [])
    objects = graph_payload.setdefault("objects", [])
    relations = graph_payload.setdefault("relations", [])
    quantities = graph_payload.setdefault("quantities", [])
    used_ids = {
        str(item.get("id"))
        for collection in (evidence, objects, relations, quantities,
                           graph_payload.setdefault("regions", []),
                           graph_payload.setdefault("targets", []),
                           graph_payload.setdefault("conflicts", []))
        for item in collection
        if isinstance(item, dict) and item.get("id")
    }
    name_to_id: dict[str, str] = {}

    def token(value: Any) -> str:
        text = str(value or "").strip()
        text = re.sub(r"^(?:triangle|polygon)\s+", "", text, flags=re.IGNORECASE)
        text = text.replace("\\triangle", "").replace("△", "").replace("∠", "")
        return re.sub(r"[^A-Za-z0-9]", "", text).upper()

    def register(item: dict[str, Any]) -> None:
        candidates = [item.get("visible_name"), item.get("id")]
        candidates.extend(item.get("aliases") or [])
        item_id = str(item.get("id", ""))
        candidates.extend((item_id.rsplit("_", 1)[-1], item_id.rsplit(":", 1)[-1]))
        for candidate in candidates:
            normalized = token(candidate)
            if normalized and normalized not in name_to_id:
                name_to_id[normalized] = item_id

    for item in objects:
        if isinstance(item, dict) and item.get("id"):
            register(item)

    def unique_id(prefix: str, normalized: str) -> str:
        base = f"text:{prefix}:{normalized}"
        candidate = base
        suffix = 1
        while candidate in used_ids:
            suffix += 1
            candidate = f"{base}:{suffix}"
        used_ids.add(candidate)
        return candidate

    def ensure_object(raw_name: str, kind: str, evidence_id: str) -> str | None:
        normalized = token(raw_name)
        if not normalized:
            return None
        existing_id = name_to_id.get(normalized)
        if existing_id:
            existing = next((item for item in objects if item.get("id") == existing_id), None)
            if existing is not None and kind == "polygon" and not existing.get("references") and len(normalized) >= 3:
                references = []
                for index, point_name in enumerate(normalized):
                    point_id = ensure_object(point_name, "point", evidence_id)
                    if point_id:
                        references.append({"role": f"vertex_{index}", "ref": point_id})
                existing["references"] = references
            return existing_id

        object_id = unique_id(kind, normalized)
        references = []
        if kind == "segment" and len(normalized) == 2:
            for index, point_name in enumerate(normalized):
                point_id = ensure_object(point_name, "point", evidence_id)
                if point_id:
                    references.append({"role": "start" if index == 0 else "end", "ref": point_id})
        elif kind == "polygon" and len(normalized) >= 3:
            for index, point_name in enumerate(normalized):
                point_id = ensure_object(point_name, "point", evidence_id)
                if point_id:
                    references.append({"role": f"vertex_{index}", "ref": point_id})
        item = {
            "id": object_id,
            "kind": kind,
            "origin": "text",
            "confidence": 0.98,
            "score_kind": "deterministic_text_claim",
            "references": references,
            "evidence_refs": [evidence_id],
            "visible_name": normalized,
            "aliases": [],
            "visible": None,
            "geometry": {},
            "metadata": {
                "provenance": "text_theorem_v0",
                "binding_status": "unresolved",
                "source_token": normalized,
            },
        }
        objects.append(item)
        register(item)
        return object_id

    def claim_span(raw: str, search_start: int) -> tuple[int, int] | None:
        start = question.find(raw, search_start)
        if start < 0:
            start = question.find(raw)
        return (start, start + len(raw)) if start >= 0 else None

    existing_relation_signatures = {
        (
            str(item.get("predicate")),
            tuple(str(argument.get("ref")) for argument in item.get("arguments", []) if isinstance(argument, dict)),
        )
        for item in relations
        if isinstance(item, dict)
    }
    existing_quantity_signatures = {
        (str(item.get("kind")), tuple(str(argument.get("ref")) for argument in item.get("arguments", []) if isinstance(argument, dict)))
        for item in quantities
        if isinstance(item, dict)
    }
    added = 0
    skipped = []
    search_start = 0
    for index, claim in enumerate(claims):
        raw = str(claim.get("raw") or "").strip()
        span = claim_span(raw, search_start)
        if span is None:
            skipped.append({"index": index, "reason": "claim_text_not_located"})
            continue
        search_start = span[1]
        evidence_id = unique_id("evidence", str(index))
        evidence.append({
            "id": evidence_id,
            "source_id": source_id,
            "kind": "text_span",
            "precision": "bound",
            "span": list(span),
            "raw_content": raw,
            "metadata": {"provenance": "text_theorem_v0", "claim_index": index},
        })
        relation_type = str(claim.get("type") or "")
        claim_objects = [str(item) for item in claim.get("objects") or []]
        if relation_type in {"congruent", "similar"}:
            object_refs = [ensure_object(item, "polygon", evidence_id) for item in claim_objects]
        elif relation_type in {"rectangle", "parallelogram", "trapezoid", "rhombus", "kite"}:
            object_refs = [ensure_object(item, "polygon", evidence_id) for item in claim_objects]
        elif relation_type in {"parallel", "perpendicular"}:
            object_refs = [ensure_object(item, "segment", evidence_id) for item in claim_objects]
        elif relation_type == "length":
            object_refs = [ensure_object(item, "segment", evidence_id) for item in claim_objects]
        elif relation_type == "angle":
            object_refs = [ensure_object(item, "marker", evidence_id) for item in claim_objects]
        else:
            skipped.append({"index": index, "reason": "unsupported_text_claim_type", "type": relation_type})
            continue
        object_refs = [item for item in object_refs if item]
        if not object_refs:
            skipped.append({"index": index, "reason": "claim_objects_unresolved"})
            continue
        if relation_type in {"length", "angle"}:
            signature = (relation_type, tuple(object_refs))
            if signature in existing_quantity_signatures:
                continue
            try:
                expression = math_expression_from_text(str(claim.get("value") or ""))
            except Exception:
                skipped.append({"index": index, "reason": "claim_value_not_supported", "type": relation_type})
                continue
            quantities.append({
                "id": unique_id("quantity", str(index)),
                "kind": relation_type,
                "arguments": [{"role": "object", "ref": item} for item in object_refs],
                "expression": expression.to_dict(),
                "origin": "text",
                "confidence": float(claim.get("confidence", 0.96)),
                "score_kind": "deterministic_text_claim",
                "unit": "degree" if relation_type == "angle" else None,
                "raw_text": raw,
                "evidence_refs": [evidence_id],
                "dependencies": [],
                "metadata": {"provenance": "text_theorem_v0", "source_claim_id": claim.get("candidate_id")},
            })
            existing_quantity_signatures.add(signature)
            added += 1
            continue
        signature = (relation_type, tuple(object_refs))
        if signature in existing_relation_signatures:
            continue
        relation_roles = (
            ("left", "right")
            if relation_type in {"parallel", "perpendicular"}
            else tuple("object" for _ in object_refs)
        )
        relations.append({
            "id": unique_id("relation", str(index)),
            "predicate": relation_type,
            "arguments": [
                {"role": role, "ref": item}
                for role, item in zip(relation_roles, object_refs)
            ],
            "origin": "text",
            "confidence": float(claim.get("confidence", 0.96)),
            "score_kind": "deterministic_text_claim",
            "evidence_refs": [evidence_id],
            "dependencies": [],
            "metadata": {
                "provenance": "text_theorem_v0",
                "source_claim_id": claim.get("candidate_id"),
                "supporting_text": raw,
            },
        })
        existing_relation_signatures.add(signature)
        added += 1
    return {
        "extractor": "text_theorem_v0",
        "claims_seen": len(claims),
        "claims_added": added,
        "claims_skipped": skipped,
    }


def _request_json_completion(
    *,
    model: str,
    api_key: str,
    base_url: str,
    image_url: str,
    instruction: str,
    timeout: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    body = json.dumps({
        "model": model,
        "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": instruction},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]}],
        "response_format": {"type": "json_object"},
    }).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", data=body,
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
    try:
        payload = _json_content(result["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("OpenRouter response did not contain a JSON message") from exc
    return result, payload


def _parsegraph_instruction(parser_input: ParserInput) -> str:
    prompt = files("geosym_world.parser").joinpath("prompts", "geometry_extraction_zh.txt").read_text(encoding="utf-8")
    context = {"question": parser_input.question}
    return prompt.rstrip() + "\n\nINPUT\n" + json.dumps(context, ensure_ascii=False)


def api_key_from_env() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    return key
