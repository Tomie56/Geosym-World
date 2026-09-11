"""Typed ParseGraph references projected from verified synthetic states."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import sympy as sp

from geosym_world.errors import InvalidGeometryInput
from geosym_world.parser.contracts import ParserInput
from geosym_world.rendering.label_layout import position_labels
from geosym_world.rendering.layout import build_view_transform
from geosym_world.rendering.marker_geometry import MARKER_KINDS, marker_pixel_bbox
from geosym_world.rendering.scene import RenderPrimitive, RenderScene
from geosym_world.schema import (
    BoundaryReference,
    EvidencePrecision,
    FormalGeometryState,
    MathExpression,
    ParseEvidence,
    ParseEvidenceKind,
    ParseGraph,
    ParseObject,
    ParseObjectKind,
    ParseOrigin,
    ParseQuantity,
    ParseReference,
    ParseRegion,
    ParseRelation,
    ParseTarget,
    RegionExpression,
    math_expression_from_sympy,
    math_expression_from_text,
)
from geosym_world.schema.parse_graph import PARSE_RELATION_TYPES

from .parser_sources import (
    ParserProtocol,
    ParserSourceManifest,
    build_typed_parser_reference,
    hash_asset,
)


def build_synthetic_typed_parser_reference(
    record: Mapping[str, Any],
    parser_input: ParserInput | Mapping[str, Any],
    *,
    state: FormalGeometryState,
    scene: RenderScene,
    width: int = 768,
    height: int = 768,
    padding: int = 48,
    manifest: ParserSourceManifest | None = None,
) -> dict[str, Any]:
    """Build evaluator-only typed truth from the exact state and rendered layout."""

    public_input = parser_input if isinstance(parser_input, ParserInput) else ParserInput.from_dict(parser_input)
    if width <= 0 or height <= 0 or padding < 0 or padding * 2 >= min(width, height):
        raise InvalidGeometryInput("Typed parser reference requires a valid render viewport")
    synthetic = record.get("synthetic") or {}
    lineage = synthetic.get("lineage") or {}
    task = record.get("task") or {}
    source_manifest = manifest or ParserSourceManifest(
        source_id=f"synthetic:{public_input.id}",
        protocol=ParserProtocol.IMAGE_QUESTION if public_input.question else ParserProtocol.IMAGE_ONLY,
        source_version=str(task.get("checksum") or state.checksum()),
        license_id="geosym-world-synthetic",
        asset_sha256=hash_asset(public_input.image),
        split_group=str(lineage.get("root") or public_input.id),
        public_fields=("image", "question") if public_input.question else ("image",),
        metadata={
            "generator": "geosym-world",
            "formal_state_checksum": state.checksum(),
            "scene_id": scene.id,
        },
    )
    if source_manifest.asset_sha256 != hash_asset(public_input.image):
        raise InvalidGeometryInput("Synthetic typed reference manifest does not match the public image")
    expected_protocol = ParserProtocol.IMAGE_QUESTION if public_input.question else ParserProtocol.IMAGE_ONLY
    if source_manifest.protocol is not expected_protocol:
        raise InvalidGeometryInput("Synthetic typed reference manifest protocol does not match ParserInput")
    source_id = source_manifest.source_id
    graph = synthetic_state_to_parse_graph(
        record,
        state=state,
        scene=scene,
        source_id=source_id,
        width=width,
        height=height,
        padding=padding,
    )
    return build_typed_parser_reference(public_input, graph=graph, manifest=source_manifest)


def synthetic_state_to_parse_graph(
    record: Mapping[str, Any],
    *,
    state: FormalGeometryState,
    scene: RenderScene,
    source_id: str,
    width: int = 768,
    height: int = 768,
    padding: int = 48,
) -> ParseGraph:
    render_dim = scene.metadata.get("render_dim")
    if render_dim != 2:
        raise InvalidGeometryInput(
            f"Synthetic typed Parser v0 references require a 2D render scene, got render_dim={render_dim!r}"
        )
    transform = build_view_transform(scene, width=width, height=height, padding=padding)
    state_objects = _state_objects(state)
    visible_ids = _visible_state_ids(scene, state_objects)
    evidence: list[ParseEvidence] = []
    objects: list[ParseObject] = []
    regions: list[ParseRegion] = []
    relations: list[ParseRelation] = []
    quantities: list[ParseQuantity] = []
    unsupported: list[str] = []

    point_pixels = {
        point_id: transform.point(tuple(float(sp.N(value)) for value in state.points[point_id].coords))
        for point_id in visible_ids & set(state.points)
        if len(state.points[point_id].coords) == 2
    }
    rendered_object_ids = {primitive.id for primitive in scene.primitives}
    rendered_point_names = _rendered_point_names(scene)

    for object_id in sorted(visible_ids):
        kind, candidate = state_objects[object_id]
        if kind in {ParseObjectKind.PLANE, ParseObjectKind.POLYHEDRON}:
            unsupported.append(object_id)
            continue
        if kind is None:
            continue
        native_evidence = _native_evidence(source_id, object_id)
        parse_object = _parse_object(
            candidate,
            kind,
            native_evidence.id,
            point_pixels,
            transform,
            rendered_object_ids,
            rendered_point_names,
        )
        if parse_object is None:
            unsupported.append(object_id)
            continue
        evidence.append(native_evidence)
        objects.append(parse_object)

    object_ids = {item.id for item in objects}
    for region_id in sorted(visible_ids & set(state.regions)):
        region = state.regions[region_id]
        if any(reference not in object_ids for reference in region.boundary):
            unsupported.append(region_id)
            continue
        native_evidence = _native_evidence(source_id, region_id)
        evidence.append(native_evidence)
        regions.append(
            ParseRegion(
                id=region.id,
                boundary=tuple(BoundaryReference(reference) for reference in region.boundary),
                holes=tuple(region.holes),
                origin=ParseOrigin.NATIVE,
                confidence=1.0,
                score_kind="verified_synthetic_state",
                evidence_refs=(native_evidence.id,),
                metadata={"region_kind": region.kind},
            )
        )

    region_ids = {item.id for item in regions}
    for region_id in sorted(visible_ids & set(state.composite_regions)):
        region = state.composite_regions[region_id]
        expression = _composite_expression(region.operation, region.components)
        if expression is None or any(reference not in region_ids for reference in region.components):
            unsupported.append(region_id)
            continue
        native_evidence = _native_evidence(source_id, region_id)
        evidence.append(native_evidence)
        regions.append(
            ParseRegion(
                id=region.id,
                boundary=(),
                components=tuple(region.components),
                expression=expression,
                origin=ParseOrigin.NATIVE,
                confidence=1.0,
                score_kind="verified_synthetic_state",
                evidence_refs=(native_evidence.id,),
            )
        )
        region_ids.add(region.id)

    text_objects, text_evidence, bindings = _point_label_truth(scene, source_id, object_ids, transform)
    objects.extend(text_objects)
    evidence.extend(text_evidence)
    relations.extend(bindings)
    marker_objects, marker_evidence = _marker_truth(scene, source_id, object_ids, transform)
    objects.extend(marker_objects)
    evidence.extend(marker_evidence)
    referable_ids = {item.id for item in objects} | region_ids
    parse_objects_by_id = {item.id: item for item in objects}

    public_givens = tuple(record.get("public_givens") or ())
    for index, given in enumerate(public_givens):
        if given.get("kind") == "quantity":
            quantity = _parse_quantity_given(given, index, source_id, referable_ids)
            if quantity is None:
                unsupported.append(f"public_quantity:{index}")
                continue
            quantities.append(quantity[0])
            evidence.append(quantity[1])
        elif given.get("kind") == "relation":
            relation = _parse_relation_given(
                given,
                index,
                source_id,
                referable_ids,
                parse_objects_by_id,
            )
            if relation is None:
                unsupported.append(f"public_relation:{index}")
                continue
            relations.append(relation[0])
            evidence.append(relation[1])

    target = _parse_target(state, source_id, referable_ids)
    targets = ()
    if target is not None:
        targets = (target[0],)
        evidence.append(target[1])
    elif state.target is not None:
        unsupported.append(f"target:{state.target.object}")

    return ParseGraph(
        id=f"reference:{source_id}",
        source_id=source_id,
        evidence=tuple(evidence),
        objects=tuple(objects),
        relations=tuple(relations),
        quantities=tuple(quantities),
        regions=tuple(regions),
        targets=targets,
        metadata={
            "protocol": "image_question",
            "reference_scope": "verified_visible_2d",
            "image_size": [width, height],
            "padding": padding,
            "state_checksum": state.checksum(),
            "num_visible_markers": len(marker_objects),
            "unsupported_reference_objects": sorted(set(unsupported)),
        },
    )


def _state_objects(state: FormalGeometryState) -> dict[str, tuple[ParseObjectKind | None, Any]]:
    groups = (
        (ParseObjectKind.POINT, state.points),
        (ParseObjectKind.LINE, state.lines),
        (ParseObjectKind.SEGMENT, state.segments),
        (ParseObjectKind.RAY, state.rays),
        (ParseObjectKind.CIRCLE, state.circles),
        (ParseObjectKind.ARC, state.arcs),
        (ParseObjectKind.POLYGON, state.polygons),
        (None, state.regions),
        (None, state.composite_regions),
        (ParseObjectKind.PLANE, state.planes),
        (ParseObjectKind.POLYHEDRON, state.polyhedra),
    )
    return {object_id: (kind, value) for kind, values in groups for object_id, value in values.items()}


def _visible_state_ids(scene: RenderScene, state_objects: Mapping[str, tuple[ParseObjectKind | None, Any]]) -> set[str]:
    visible = {primitive.id for primitive in scene.primitives if primitive.id in state_objects}
    pending = list(visible)
    while pending:
        object_id = pending.pop()
        for dependency in _state_dependencies(state_objects[object_id][1]):
            if dependency in state_objects and dependency not in visible:
                visible.add(dependency)
                pending.append(dependency)
    return visible


def _state_dependencies(candidate: Any) -> tuple[str, ...]:
    dependencies = []
    for key in ("start", "end", "through", "center", "circle"):
        value = getattr(candidate, key, None)
        if isinstance(value, str):
            dependencies.append(value)
    for key in ("source_points", "vertices", "boundary", "holes", "components"):
        dependencies.extend(getattr(candidate, key, ()) or ())
    return tuple(dict.fromkeys(dependencies))


def _parse_object(candidate, kind, evidence_id, point_pixels, transform, rendered_object_ids, point_names) -> ParseObject | None:
    geometry: dict[str, Any] = {}
    references: tuple[ParseReference, ...] = ()
    visible_name = point_names.get(candidate.id) if kind is ParseObjectKind.POINT else None
    if kind is ParseObjectKind.POINT:
        if candidate.id not in point_pixels:
            return None
        geometry = {
            "pixel": list(point_pixels[candidate.id]),
            "exact_coords": [math_expression_from_sympy(value).to_dict() for value in candidate.coords],
        }
    elif kind is ParseObjectKind.SEGMENT:
        references = (ParseReference("start", candidate.start), ParseReference("end", candidate.end))
        geometry = _linear_geometry(point_pixels.get(candidate.start), point_pixels.get(candidate.end))
    elif kind is ParseObjectKind.LINE:
        if len(candidate.point) != 2 or len(candidate.direction) != 2:
            return None
        references = tuple(ParseReference(f"point_{index}", value) for index, value in enumerate(candidate.source_points[:2]))
        start = transform.point(tuple(float(sp.N(value)) for value in candidate.point))
        end = transform.point(tuple(float(sp.N(candidate.point[index] + candidate.direction[index])) for index in range(2)))
        geometry = _linear_geometry(start, end)
    elif kind is ParseObjectKind.RAY:
        if len(candidate.direction) != 2:
            return None
        references = (ParseReference("start", candidate.start),)
        if candidate.through is not None:
            references += (ParseReference("through", candidate.through),)
        start = point_pixels.get(candidate.start)
        end = point_pixels.get(candidate.through) if candidate.through is not None else None
        if start is not None and end is None:
            direction = tuple(float(sp.N(value)) for value in candidate.direction)
            end = start[0] + direction[0] * transform.scale, start[1] - direction[1] * transform.scale
        geometry = _linear_geometry(start, end)
    elif kind is ParseObjectKind.CIRCLE:
        center = point_pixels.get(candidate.center)
        if center is None:
            return None
        references = (ParseReference("center", candidate.center),)
        geometry = {
            "center": list(center),
            "radius_px": transform.length(float(sp.N(candidate.radius))),
            "exact_radius": math_expression_from_sympy(candidate.radius).to_dict(),
        }
    elif kind is ParseObjectKind.ARC:
        references = (
            ParseReference("circle", candidate.circle),
            ParseReference("start", candidate.start),
            ParseReference("end", candidate.end),
        )
        geometry = {"orientation": candidate.orientation}
        if candidate.central_angle is not None:
            geometry["exact_central_angle"] = math_expression_from_sympy(candidate.central_angle).to_dict()
    elif kind is ParseObjectKind.POLYGON:
        references = tuple(ParseReference(f"vertex_{index}", value) for index, value in enumerate(candidate.vertices))
    else:
        return None
    return ParseObject(
        id=candidate.id,
        kind=kind,
        origin=ParseOrigin.NATIVE,
        confidence=1.0,
        score_kind="verified_synthetic_state",
        references=references,
        evidence_refs=(evidence_id,),
        visible_name=visible_name,
        visible=candidate.id in rendered_object_ids or (kind is ParseObjectKind.POINT and candidate.id in point_names),
        geometry=geometry,
    )


def _linear_geometry(start, end) -> dict[str, Any]:
    if start is None or end is None:
        return {}
    dx, dy = end[0] - start[0], end[1] - start[1]
    return {
        "p1": list(start),
        "p2": list(end),
        "angle_deg": math.degrees(math.atan2(dy, dx)) % 180.0,
    }


def _rendered_point_names(scene: RenderScene) -> dict[str, str]:
    point_labels = dict(scene.metadata.get("point_labels") or {})
    rendered = {
        label.id.split(":", 1)[1]
        for label in scene.labels
        if label.id.startswith("label:")
    }
    return {point_id: str(point_labels[point_id]) for point_id in rendered if point_id in point_labels}


def _point_label_truth(scene, source_id, object_ids, transform):
    text_objects = []
    evidence = []
    bindings = []
    for positioned in position_labels(scene, transform):
        label = positioned.label
        if not label.id.startswith("label:"):
            continue
        point_id = label.id.split(":", 1)[1]
        if point_id not in object_ids:
            continue
        text_id = f"reference:text:{point_id}"
        evidence_id = f"reference:evidence:text:{point_id}"
        evidence.append(
            ParseEvidence(
                id=evidence_id,
                source_id=source_id,
                kind=ParseEvidenceKind.IMAGE_REGION,
                precision=EvidencePrecision.BOUND,
                bbox=positioned.bbox,
                raw_content=label.text,
            )
        )
        text_objects.append(
            ParseObject(
                id=text_id,
                kind=ParseObjectKind.TEXT,
                origin=ParseOrigin.IMAGE_TEXT,
                confidence=1.0,
                score_kind="verified_render_label",
                evidence_refs=(evidence_id,),
                visible_name=label.text,
                visible=True,
                geometry={"bbox": list(positioned.bbox)},
            )
        )
        bindings.append(
            ParseRelation(
                id=f"reference:binding:{point_id}",
                predicate="binds",
                arguments=(ParseReference("label", text_id), ParseReference("object", point_id)),
                origin=ParseOrigin.IMAGE_TEXT,
                confidence=1.0,
                score_kind="verified_render_binding",
                evidence_refs=(evidence_id,),
                metadata={"binding_kind": "point_name", "label": label.text},
            )
        )
    return tuple(text_objects), tuple(evidence), tuple(bindings)


def _marker_truth(scene, source_id, object_ids, transform):
    objects = []
    evidence = []
    markers = [primitive for primitive in scene.primitives if primitive.role == "marker"]
    for index, primitive in enumerate(markers):
        if primitive.kind not in MARKER_KINDS:
            raise InvalidGeometryInput(f"Unsupported rendered marker kind: {primitive.kind!r}")
        marker_kind = _marker_kind(primitive.kind)
        marker_id = f"reference:marker:{index}"
        evidence_id = f"reference:evidence:marker:{index}"
        bbox = marker_pixel_bbox(
            primitive,
            transform,
            padding=max(1.0, float(primitive.style.stroke_width) / 2.0),
        )
        references = _marker_references(primitive, object_ids)
        metadata = {
            "marker_kind": marker_kind,
            "render_primitive_id": primitive.id,
            "relation": str(primitive.metadata.get("relation") or ""),
        }
        evidence.append(
            ParseEvidence(
                id=evidence_id,
                source_id=source_id,
                kind=ParseEvidenceKind.IMAGE_REGION,
                precision=EvidencePrecision.BOUND,
                bbox=bbox,
                raw_content=marker_kind,
                metadata={"render_primitive_id": primitive.id, "render_kind": primitive.kind},
            )
        )
        objects.append(
            ParseObject(
                id=marker_id,
                kind=ParseObjectKind.MARKER,
                origin=ParseOrigin.IMAGE,
                confidence=1.0,
                score_kind="verified_render_marker",
                references=references,
                evidence_refs=(evidence_id,),
                visible=True,
                geometry={"bbox": list(bbox)},
                metadata=metadata,
            )
        )
    return tuple(objects), tuple(evidence)


def _marker_kind(render_kind: str) -> str:
    return {
        "angle_marker": "angle_arc",
        "right_angle_marker": "right_angle",
        "tick_marker": "equal_length_tick",
        "parallel_marker": "parallel",
    }[render_kind]


def _marker_references(primitive: RenderPrimitive, object_ids: set[str]) -> tuple[ParseReference, ...]:
    metadata = primitive.metadata
    support = metadata.get("support")
    if isinstance(support, str) and support in object_ids:
        return (ParseReference("support", support),)

    values: list[tuple[str, str]] = []
    for key in ("objects", "points"):
        candidates = metadata.get(key) or ()
        if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
            values.extend(
                (f"{key[:-1]}_{index}", value)
                for index, value in enumerate(candidates)
                if isinstance(value, str) and value in object_ids
            )
    for role in ("vertex", "from", "line", "polygon", "region"):
        value = metadata.get(role)
        if isinstance(value, str) and value in object_ids:
            values.append((role, value))
    return tuple(ParseReference(role, reference) for role, reference in dict.fromkeys(values))


def _parse_quantity_given(given, index, source_id, referable_ids):
    object_id = str(given.get("object") or given.get("display_object") or "")
    if object_id not in referable_ids or given.get("value") is None:
        return None
    kind = {"distance": "length", "central_angle": "angle"}.get(str(given.get("type")), str(given.get("type")))
    evidence = _native_evidence(source_id, f"public_quantity:{index}")
    return (
        ParseQuantity(
            id=f"reference:quantity:{index}",
            kind=kind,
            arguments=(ParseReference("object", object_id),),
            expression=(
                math_expression_from_text(given["value"])
                if isinstance(given["value"], str)
                else math_expression_from_sympy(given["value"])
            ),
            origin=ParseOrigin.NATIVE,
            confidence=1.0,
            score_kind="verified_public_given",
            raw_text=str(given["value"]),
            evidence_refs=(evidence.id,),
        ),
        evidence,
    )


def _parse_relation_given(given, index, source_id, referable_ids, objects_by_id):
    predicate = str(given.get("type", ""))
    if predicate not in PARSE_RELATION_TYPES:
        return None
    projection = _relation_projection(
        predicate,
        given.get("payload") or {},
        referable_ids,
        objects_by_id,
    )
    if projection is None:
        return None
    arguments, metadata = projection
    if not arguments:
        return None
    evidence = _native_evidence(source_id, f"public_relation:{index}")
    source_relation_id = given.get("relation_id")
    if source_relation_id is not None:
        metadata = {"source_relation_id": str(source_relation_id), **metadata}
    return (
        ParseRelation(
            id=f"reference:relation:{index}",
            predicate=predicate,
            arguments=arguments,
            origin=ParseOrigin.NATIVE,
            confidence=1.0,
            score_kind="verified_public_given",
            evidence_refs=(evidence.id,),
            metadata=metadata,
        ),
        evidence,
    )


def _relation_projection(
    predicate: str,
    payload: Mapping[str, Any],
    referable_ids: set[str],
    objects_by_id: Mapping[str, ParseObject],
) -> tuple[tuple[ParseReference, ...], dict[str, Any]] | None:
    if not isinstance(payload, Mapping):
        return None

    if predicate in {"parallel", "perpendicular", "equal_length"}:
        values = payload.get("objects") or payload.get("segments")
        arguments = _indexed_relation_refs("object", values, referable_ids, exact=2)
        return (arguments, {}) if arguments else None

    if predicate == "collinear":
        arguments = _indexed_relation_refs("point", payload.get("points"), referable_ids, minimum=2)
        return (arguments, {}) if arguments else None

    if predicate == "intersection":
        supports = _repeated_relation_refs("support", payload.get("objects"), referable_ids, exact=2)
        point_values = payload.get("points")
        if point_values is None and payload.get("point") is not None:
            point_values = (payload["point"],)
        if _reference_sequence(point_values, referable_ids, exact=1):
            points = (ParseReference("point", str(point_values[0])),)
        else:
            points = _indexed_relation_refs("point", point_values, referable_ids, minimum=1)
        if not supports or not points:
            return None
        metadata = _relation_scalar_metadata(payload, {"objects", "point", "points", "dependencies"})
        return (*supports, *points), metadata

    if predicate == "midpoint":
        point = _single_relation_ref("point", payload.get("point"), referable_ids)
        endpoints = _indexed_relation_refs("of", payload.get("of"), referable_ids, exact=2)
        if point is None or not endpoints:
            return None
        return (point, *endpoints), {}

    if predicate == "foot_of_perpendicular":
        arguments = _required_relation_refs(
            (("point", payload.get("point")), ("from", payload.get("from")), ("line", payload.get("line"))),
            referable_ids,
        )
        return (arguments, {}) if arguments else None

    if predicate == "point_on_segment_ratio":
        point = _single_relation_ref("point", payload.get("point"), referable_ids)
        resolved = _segment_reference(payload.get("segment"), objects_by_id)
        ratio = _typed_relation_expression(payload.get("ratio"))
        if point is None or resolved is None or ratio is None:
            return None
        segment_id, reversed_orientation = resolved
        if reversed_orientation:
            ratio_value = MathExpression.from_dict(ratio).to_sympy()
            ratio = math_expression_from_sympy(sp.Integer(1) - ratio_value).to_dict()
        return (point, ParseReference("segment", segment_id)), {"ratio": ratio}

    if predicate == "point_on_ray_distance":
        arguments = _required_relation_refs(
            (("point", payload.get("point")), ("ray", payload.get("ray"))),
            referable_ids,
        )
        distance = _typed_relation_expression(payload.get("distance"))
        if not arguments or distance is None:
            return None
        return arguments, {"distance": distance}

    if predicate == "angle_bisector":
        line = _single_relation_ref("line", payload.get("line"), referable_ids)
        angle = payload.get("angle")
        if line is None or not _reference_sequence(angle, referable_ids, exact=3):
            return None
        mode = str(payload.get("mode", "internal"))
        if mode not in {"internal", "external"}:
            return None
        return (
            line,
            ParseReference("arm_0", str(angle[0])),
            ParseReference("vertex", str(angle[1])),
            ParseReference("arm_1", str(angle[2])),
        ), {"mode": mode}

    if predicate == "angle_from_line":
        line_id = payload.get("line")
        line = _single_relation_ref("line", line_id, referable_ids)
        base_line = _single_relation_ref("base_line", payload.get("base_line"), referable_ids)
        point_id = payload.get("point") or _single_source_point(line_id, objects_by_id)
        point = _single_relation_ref("point", point_id, referable_ids)
        angle = _typed_relation_expression(payload.get("angle"))
        if line is None or point is None or base_line is None or angle is None:
            return None
        return (line, point, base_line), {"angle": angle}

    if predicate in {"point_on_line", "point_on_ray"}:
        support = payload.get("line") or payload.get("ray") or payload.get("segment")
        arguments = _required_relation_refs(
            (("point", payload.get("point")), ("line", support)),
            referable_ids,
        )
        return (arguments, {}) if arguments else None

    if predicate == "point_on_circle":
        arguments = _required_relation_refs(
            (("point", payload.get("point")), ("circle", payload.get("circle"))),
            referable_ids,
        )
        return (arguments, {}) if arguments else None

    if predicate == "point_on_arc":
        arguments = _required_relation_refs(
            (("point", payload.get("point")), ("arc", payload.get("arc"))),
            referable_ids,
        )
        return (arguments, {}) if arguments else None

    if predicate in {"similar", "congruent", "triangle"}:
        polygons = payload.get("polygons") or payload.get("polygon")
        arguments = _indexed_relation_refs("polygon", polygons, referable_ids, minimum=1)
        return (arguments, {}) if arguments else None

    return _generic_relation_projection(payload, referable_ids)


def _single_relation_ref(role: str, value: Any, referable_ids: set[str]) -> ParseReference | None:
    if isinstance(value, str) and value in referable_ids:
        return ParseReference(role, value)
    return None


def _required_relation_refs(
    values: Sequence[tuple[str, Any]],
    referable_ids: set[str],
) -> tuple[ParseReference, ...]:
    references = tuple(_single_relation_ref(role, value, referable_ids) for role, value in values)
    return tuple(reference for reference in references if reference is not None) if all(references) else ()


def _indexed_relation_refs(
    role: str,
    values: Any,
    referable_ids: set[str],
    *,
    exact: int | None = None,
    minimum: int = 0,
) -> tuple[ParseReference, ...]:
    if not _reference_sequence(values, referable_ids, exact=exact, minimum=minimum):
        return ()
    return tuple(ParseReference(f"{role}_{index}", str(value)) for index, value in enumerate(values))


def _repeated_relation_refs(
    role: str,
    values: Any,
    referable_ids: set[str],
    *,
    exact: int | None = None,
    minimum: int = 0,
) -> tuple[ParseReference, ...]:
    if not _reference_sequence(values, referable_ids, exact=exact, minimum=minimum):
        return ()
    return tuple(ParseReference(role, str(value)) for value in values)


def _reference_sequence(
    values: Any,
    referable_ids: set[str],
    *,
    exact: int | None = None,
    minimum: int = 0,
) -> bool:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        return False
    if exact is not None and len(values) != exact:
        return False
    if len(values) < minimum:
        return False
    return all(isinstance(value, str) and value in referable_ids for value in values)


def _segment_reference(
    value: Any,
    objects_by_id: Mapping[str, ParseObject],
) -> tuple[str, bool] | None:
    if isinstance(value, str):
        candidate = objects_by_id.get(value)
        return (value, False) if candidate is not None and candidate.kind is ParseObjectKind.SEGMENT else None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        return None
    start, end = (str(item) for item in value)
    matches: list[tuple[str, bool]] = []
    for candidate in objects_by_id.values():
        if candidate.kind is not ParseObjectKind.SEGMENT:
            continue
        references = {reference.role: reference.ref for reference in candidate.references}
        endpoints = references.get("start"), references.get("end")
        if endpoints == (start, end):
            matches.append((candidate.id, False))
        elif endpoints == (end, start):
            matches.append((candidate.id, True))
    return matches[0] if len(matches) == 1 else None


def _single_source_point(line_id: Any, objects_by_id: Mapping[str, ParseObject]) -> str | None:
    if not isinstance(line_id, str):
        return None
    candidate = objects_by_id.get(line_id)
    if candidate is None or candidate.kind is not ParseObjectKind.LINE:
        return None
    points = [reference.ref for reference in candidate.references if reference.role.startswith("point_")]
    return points[0] if len(points) == 1 else None


def _typed_relation_expression(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        if isinstance(value, Mapping):
            return MathExpression.from_dict(value).to_dict()
        if isinstance(value, str):
            return math_expression_from_text(value).to_dict()
        return math_expression_from_sympy(value).to_dict()
    except (InvalidGeometryInput, TypeError, ValueError):
        return None


def _relation_scalar_metadata(payload: Mapping[str, Any], excluded: set[str]) -> dict[str, Any]:
    metadata = {}
    for key in sorted(payload):
        if key in excluded:
            continue
        value = payload[key]
        if key in {"angle", "central_angle", "distance", "ratio"}:
            expression = _typed_relation_expression(value)
            if expression is not None:
                metadata[key] = expression
        elif value is None or isinstance(value, (str, int, float, bool)):
            metadata[key] = value
    return metadata


def _generic_relation_projection(
    payload: Mapping[str, Any],
    referable_ids: set[str],
) -> tuple[tuple[ParseReference, ...], dict[str, Any]] | None:
    arguments = []
    consumed = set()
    for key in sorted(payload):
        value = payload[key]
        if isinstance(value, str) and value in referable_ids:
            arguments.append(ParseReference(key, value))
            consumed.add(key)
        elif _reference_sequence(value, referable_ids, minimum=1):
            arguments.extend(ParseReference(f"{key}_{index}", item) for index, item in enumerate(value))
            consumed.add(key)
    if not arguments:
        return None
    metadata = _relation_scalar_metadata(payload, consumed | {"dependencies"})
    return tuple(arguments), metadata


def _parse_target(state, source_id, referable_ids):
    if state.target is None or state.target.object not in referable_ids:
        return None
    evidence = _native_evidence(source_id, "target")
    return (
        ParseTarget(
            id="reference:target",
            kind=state.target.type,
            mode=state.target.mode,
            arguments=(ParseReference("object", state.target.object),),
            origin=ParseOrigin.NATIVE,
            confidence=1.0,
            score_kind="verified_synthetic_state",
            evidence_refs=(evidence.id,),
        ),
        evidence,
    )


def _composite_expression(operation: str, components: Sequence[str]) -> RegionExpression | None:
    if operation in {"union", "intersection"} and len(components) >= 2:
        return RegionExpression(operation, operands=tuple(RegionExpression("reference", ref=item) for item in components))
    if operation == "difference" and len(components) == 2:
        return RegionExpression("difference", operands=tuple(RegionExpression("reference", ref=item) for item in components))
    return None


def _native_evidence(source_id: str, object_id: str) -> ParseEvidence:
    return ParseEvidence(
        id=f"reference:evidence:{object_id}",
        source_id=source_id,
        kind=ParseEvidenceKind.NATIVE_ENTITY,
        precision=EvidencePrecision.BOUND,
    )
