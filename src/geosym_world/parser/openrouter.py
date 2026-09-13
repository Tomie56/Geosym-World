"""Minimal OpenRouter adapter for strict, proposal-only ParseGraph extraction."""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

from geosym_world.schema import ParseGraph
from geosym_world.schema.parse_graph import (
    ParseConflictGroup, ParseEvidence, ParseObject,
    ParseQuantity, ParseRegion, ParseRelation, ParseTarget,
)
from geosym_world.schema.canonical import stable_checksum

from .contracts import ParserInput, validate_parser_input
from .external_v0 import ExternalParseGraphPrediction


PROMPT_VERSION = "openrouter-parsegraph-v1"


def _json_content(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("OpenRouter response must be a JSON object")
    return payload


def _compact_to_graph(payload: dict[str, Any], parser_input: ParserInput) -> dict[str, Any]:
    """Compile common VLM shorthand into the strict ParseGraph wire format."""
    if "candidates" not in payload:
        return payload
    graph = {key: payload.get(key, []) for key in ("evidence", "objects", "relations", "quantities", "regions", "targets", "conflicts")}
    graph.update({"schema_version": "1.0", "id": "graph_" + parser_input.id, "source_id": parser_input.id, "metadata": {}})
    evidence_id = "model_evidence"
    evidence_text = " ".join(str(item.get("evidence", "")) for item in payload["candidates"] if isinstance(item, dict)).strip()
    graph["evidence"] = [{"id": evidence_id, "source_id": parser_input.id, "kind": "source_document", "precision": "image_only", "raw_content": evidence_text, "metadata": {}}]
    for index, candidate in enumerate(payload["candidates"]):
        if not isinstance(candidate, dict):
            continue
        source = payload if payload.get("objects") else candidate
        for item in source.get("objects", []):
            if not isinstance(item, dict):
                continue
            object_id = str(item.get("id", f"model_object_{index}"))
            kind = str(item.get("type", "unknown")).replace("line_segment", "segment").replace("triangle", "polygon")
            if kind not in {"point", "line", "segment", "ray", "circle", "arc", "polygon", "plane", "polyhedron", "text", "marker", "unknown"}:
                kind = "unknown"
            geometry = dict(item.get("geometry", {}))
            if kind == "segment" and not geometry:
                endpoints = object_id.rsplit("_", 1)[-1]
                if len(endpoints) == 2:
                    geometry = {"p1": endpoints[0], "p2": endpoints[1]}
            graph["objects"].append({"id": object_id, "kind": kind, "origin": "image_text", "confidence": float(item.get("confidence", candidate.get("confidence", 0.5))), "score_kind": "model", "evidence_refs": [evidence_id], "visible_name": item.get("label"), "geometry": geometry, "metadata": {"provenance": item.get("provenance", "model")}})
        for item in source.get("relations", []):
            if not isinstance(item, dict) or not item.get("type"):
                continue
            predicate = str(item["type"])
            if predicate == "partitioned_by":
                predicate = "contains"
            if predicate not in {"contains", "perpendicular", "on_line", "parallel", "intersection_of", "shaded_region", "inside", "disjoint", "touches", "overlaps", "supports"}:
                continue
            source_ref = str(item.get("source", "")).replace("point_", "", 1)
            target_ref = str(item.get("target", "")).replace("point_", "", 1)
            arguments = [{"role": "source", "ref": source_ref}, {"role": "target", "ref": target_ref}]
            graph["relations"].append({"id": str(item.get("id", f"model_relation_{index}")), "predicate": predicate, "arguments": arguments, "origin": "image_text", "confidence": float(item.get("confidence", candidate.get("confidence", 0.5))), "score_kind": "model", "evidence_refs": [evidence_id], "metadata": {"provenance": item.get("provenance", "model")}})
    return graph


def parse_one(
    parser_input: ParserInput | dict[str, Any],
    *,
    model: str,
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    timeout: float = 120.0,
) -> ExternalParseGraphPrediction:
    public_input = validate_parser_input(parser_input)
    image = Path(public_input.image).read_bytes()
    import base64

    image_url = "data:image/png;base64," + base64.b64encode(image).decode("ascii")
    instruction = (
        "You are a geometry diagram parser, not a solver. Extract only visible geometric structure "
        "from the image and the question. Never calculate or return an answer. Return ONLY a complete "
        "Geosym ParseGraph JSON object with exactly these top-level fields: schema_version, id, source_id, "
        "evidence, objects, relations, quantities, regions, targets, conflicts, metadata, checksum. "
        "Use schema_version '1.0', id 'graph_" + public_input.id + "', source_id '" + public_input.id + "'. "
        "Use empty arrays when a category is not visible. Every object/relation/quantity/region/target "
        "must use the exact Geosym schema and remain a proposed hypothesis with confidence and provenance. "
        "For every visible point or line, include pixel coordinates in geometry and a localized image-region "
        "evidence item with bbox [x0,y0,x1,y1] in the 768x768 image; if uncertain, omit that candidate. "
        "Do not include proposed_answer, solution, derivation, formal_state, or scene metadata. "
        "The client will validate the checksum. Question text: " + public_input.question
    )
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
    content = result["choices"][0]["message"]["content"]
    graph_payload = _json_content(content)
    debug_path = os.environ.get("GEOSYM_OPENROUTER_RAW_RESPONSE")
    if debug_path:
        Path(debug_path).parent.mkdir(parents=True, exist_ok=True)
        Path(debug_path).write_text(json.dumps(graph_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if "parse_graph" in graph_payload and isinstance(graph_payload["parse_graph"], dict):
        graph_payload = graph_payload["parse_graph"]
    graph_payload = _compact_to_graph(graph_payload, public_input)
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
    for item in graph_payload.get("objects", []):
        if isinstance(item, dict):
            for key in set(item) - {"id", "kind", "origin", "confidence", "score_kind", "references", "evidence_refs", "visible_name", "aliases", "visible", "geometry", "metadata", "type", "label", "provenance", "x", "y", "start", "end"}:
                item.pop(key, None)
            item.setdefault("id", "model_object_" + str(len([x for x in graph_payload.get("objects", []) if isinstance(x, dict)])))
        if isinstance(item, dict) and "type" in item and "kind" not in item:
            item["kind"] = str(item.pop("type")).replace("line_segment", "segment").replace("triangle", "polygon").replace("region", "polygon")
            item["origin"] = "image_text"
            item["score_kind"] = "model"
            item["evidence_refs"] = ["model_evidence"]
            item["visible_name"] = item.pop("label", None)
            item["geometry"] = item.get("geometry", {})
            item["metadata"] = {"provenance": item.pop("provenance", "model")}
        if isinstance(item, dict):
            item.setdefault("origin", "image_text")
            item.setdefault("score_kind", "model")
            item.setdefault("evidence_refs", ["model_evidence"])
            item["evidence_refs"] = [ref for ref in item.get("evidence_refs", []) if ref in evidence_ids] or [next(iter(evidence_ids))]
            item.setdefault("confidence", 0.5)
            item.setdefault("metadata", {"provenance": "model"})
            geometry = item.get("geometry", {})
            if not geometry and {"x", "y"} <= set(item):
                geometry = {"x": item.pop("x"), "y": item.pop("y")}
                item["geometry"] = geometry
            if item.get("kind") == "point" and "pixel" not in geometry and {"x", "y"} <= set(geometry):
                item["geometry"] = {"pixel": [float(geometry["x"]), float(geometry["y"])]}
            elif item.get("kind") in {"segment", "line", "ray"}:
                if "p1" not in geometry and "start" in geometry:
                    geometry["p1"] = geometry.pop("start")
                if "p2" not in geometry and "end" in geometry:
                    geometry["p2"] = geometry.pop("end")
                item["geometry"] = geometry
            if item.get("kind") == "line":
                item["kind"] = "segment"
            if item.get("kind") == "segment" and not item.get("geometry"):
                endpoints = str(item.get("id", "")).rsplit("_", 1)[-1]
                if len(endpoints) == 2:
                    item["geometry"] = {"p1": endpoints[0], "p2": endpoints[1]}
            if item.get("kind") in {"segment", "line", "ray"}:
                geometry = item.get("geometry", {})
                if any(key in geometry and (not isinstance(geometry[key], (list, tuple)) or len(geometry[key]) != 2 or any(not isinstance(value, (int, float)) for value in geometry[key])) for key in ("p1", "p2")):
                    endpoints = str(item.get("id", "")).rsplit("_", 1)[-1]
                    if len(endpoints) == 2 and all(character.isalnum() for character in endpoints):
                        item["references"] = [{"role": "start", "ref": endpoints[0]}, {"role": "end", "ref": endpoints[1]}]
                    else:
                        item["kind"] = "unknown"
                    item["geometry"] = {}
            if item.get("kind") == "polygon" and not item.get("references"):
                vertex_text = str(item.get("id", ""))
                if "_" in vertex_text:
                    vertex_text = vertex_text.rsplit("_", 1)[-1]
                if len(vertex_text) >= 3 and all(character.isalnum() for character in vertex_text):
                    item["references"] = [{"role": "vertex", "ref": character} for character in vertex_text]
    localized = []
    for item in graph_payload.get("objects", []):
        geometry = item.get("geometry", {})
        points = []
        if isinstance(geometry.get("pixel"), (list, tuple)):
            points = [geometry["pixel"]]
        elif isinstance(geometry.get("p1"), (list, tuple)) and isinstance(geometry.get("p2"), (list, tuple)):
            points = [geometry["p1"], geometry["p2"]]
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
    for item in graph_payload.get("relations", []):
        if isinstance(item, dict):
            for key in set(item) - {"id", "predicate", "arguments", "origin", "confidence", "score_kind", "evidence_refs", "dependencies", "metadata", "source", "target", "subject", "object", "type", "provenance"}:
                item.pop(key, None)
        if isinstance(item, dict) and "arguments" not in item and not ({"source", "target"} <= set(item)) and not ({"subject", "object"} <= set(item)):
            item["_skip"] = True
        if isinstance(item, dict) and "type" in item and "predicate" not in item:
            item["predicate"] = item.pop("type")
        if isinstance(item, dict) and "predicate" in item:
            item.setdefault("id", "model_relation_" + str(len(item)))
            if "arguments" not in item and "subject" in item and "object" in item:
                item["arguments"] = [{"role": "source", "ref": str(item.pop("subject")).replace("line_", "").replace("point_", "", 1)}, {"role": "target", "ref": str(item.pop("object")).replace("line_", "").replace("point_", "", 1)}]
            aliases = {"is_perpendicular_to": "perpendicular", "on_segment": "inside", "on_line": "inside", "is_part_of": "contains", "intersects": "intersection_of", "between": "ordered_points"}
            item["predicate"] = aliases.get(item["predicate"], item["predicate"])
            if item["predicate"] not in {"contains", "perpendicular", "parallel", "intersection_of", "shaded_region", "inside", "disjoint", "touches", "overlaps", "support_of", "boundary_of", "ordered_points"}:
                item["_skip"] = True
            if "arguments" not in item and "source" in item and "target" in item:
                item["arguments"] = [{"role": "source", "ref": str(item.pop("source")).replace("point_", "", 1)}, {"role": "target", "ref": str(item.pop("target")).replace("point_", "", 1)}]
            item["origin"] = "image_text"
            item["score_kind"] = "model"
            item["evidence_refs"] = ["model_evidence"]
            item["metadata"] = {"provenance": item.pop("provenance", "model")}
    graph_payload["relations"] = [item for item in graph_payload.get("relations", []) if isinstance(item, dict) and not item.get("_skip")]
    graph_payload["regions"] = [item for item in graph_payload.get("regions", []) if isinstance(item, dict) and item.get("boundary")]
    seen_ids: set[str] = set()
    for collection in ("objects", "relations", "quantities", "regions", "targets"):
        for index, item in enumerate(graph_payload.get(collection, [])):
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id", f"model_{collection}_{index}"))
            if item_id in seen_ids:
                item_id = f"{item_id}_{index}"
            item["id"] = item_id
            seen_ids.add(item_id)
    object_ids = {item["id"] for item in graph_payload.get("objects", []) if isinstance(item, dict)}
    def resolve_ref(value: str) -> str:
        candidate = str(value)
        for prefix in ("point_", "line_", "segment_"):
            if candidate.startswith(prefix):
                candidate = candidate[len(prefix):]
        return candidate
    valid_relations = []
    for item in graph_payload.get("relations", []):
        arguments = item.get("arguments", [])
        for argument in arguments:
            if isinstance(argument, dict) and "ref" in argument:
                argument["ref"] = resolve_ref(argument["ref"])
        if arguments and all(argument.get("ref") in object_ids for argument in arguments):
            valid_relations.append(item)
    graph_payload["relations"] = valid_relations
    normalized_quantities = []
    for index, item in enumerate(graph_payload.get("quantities", [])):
        if not isinstance(item, dict) or "kind" in item:
            normalized_quantities.append(item)
            continue
        if not item.get("value") or not item.get("associated_with"):
            continue
        reference = resolve_ref(item["associated_with"])
        if reference not in object_ids:
            continue
        normalized_quantities.append({"id": item.get("id", f"model_quantity_{index}"), "kind": item.get("type", "unknown"), "arguments": [{"role": "object", "ref": reference}], "expression": {"operator": "literal", "value": str(item["value"]), "operands": []}, "origin": "image_text", "confidence": float(item.get("confidence", 0.5)), "score_kind": "model", "unit": item.get("unit"), "raw_text": str(item["value"]), "evidence_refs": ["model_evidence"], "dependencies": [], "metadata": {"provenance": item.get("provenance", "model")}})
    graph_payload["quantities"] = normalized_quantities
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
        request_id=result.get("id"), decoding={"temperature": 0},
        usage=result.get("usage", {}),
    )


def api_key_from_env() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    return key
