"""Typed image-only ParseGraph baseline built on deterministic visual candidates."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from geosym_world.errors import InvalidGeometryInput
from geosym_world.parser.adapters.hypothesis_record import HypothesisRecordAdapter
from geosym_world.parser.clean_v0 import LabelRecognizer, UnresolvedLabelRecognizer
from geosym_world.parser.contracts import ParserInput, validate_parser_input
from geosym_world.parser.grounding import ground_line_labels, ground_point_labels
from geosym_world.parser.question import (
    RelationReferenceMatch,
    infer_quantity_assertions,
    infer_relation_assertions,
    infer_target_reference,
    infer_target_type,
)
from geosym_world.parser.quantity_grounding import ground_segment_length_labels
from geosym_world.parser.vision import VisualPrimitiveDetector
from geosym_world.schema import (
    BoundaryReference,
    EvidencePrecision,
    GeometryRecord,
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
    math_expression_from_text,
)


TYPED_PARSER_VERSION = "0.13"


@dataclass(frozen=True)
class TypedParseResult:
    parser_input: ParserInput
    graph: ParseGraph
    record: GeometryRecord
    stages: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "parser_input": self.parser_input.to_dict(),
            "parse_graph": self.graph.to_dict(),
            "record": self.record.to_dict(include_state=False),
            "stages": [dict(stage) for stage in self.stages],
        }


class TypedDiagramParserV0:
    """Convert image evidence to a proposed ParseGraph without formal-state admission."""

    parser_name = "typed_diagram_parser_v0"

    def __init__(
        self,
        *,
        detector: VisualPrimitiveDetector | None = None,
        label_recognizer: LabelRecognizer | None = None,
        point_tolerance: float = 8.0,
        point_label_max_distance: float = 28.0,
        point_label_ambiguity_margin: float = 8.0,
        line_label_max_distance: float = 28.0,
        line_label_ambiguity_margin: float = 8.0,
        infer_planar_faces: bool = True,
        minimum_face_area_px2: float = 64.0,
        maximum_face_vertices: int = 12,
        image_quantity_max_distance: float = 32.0,
        image_quantity_ambiguity_margin: float = 8.0,
    ):
        if point_tolerance <= 0.0:
            raise ValueError("point_tolerance must be positive")
        if point_label_max_distance <= 0.0 or point_label_ambiguity_margin < 0.0:
            raise ValueError("Point-label grounding distances must be non-negative with a positive maximum")
        if line_label_max_distance <= 0.0 or line_label_ambiguity_margin < 0.0:
            raise ValueError("Line-label grounding distances must be non-negative with a positive maximum")
        if not isinstance(infer_planar_faces, bool):
            raise ValueError("infer_planar_faces must be boolean")
        if minimum_face_area_px2 <= 0.0:
            raise ValueError("minimum_face_area_px2 must be positive")
        if isinstance(maximum_face_vertices, bool) or not isinstance(maximum_face_vertices, int) or maximum_face_vertices < 3:
            raise ValueError("maximum_face_vertices must be an integer of at least three")
        if image_quantity_max_distance <= 0.0 or image_quantity_ambiguity_margin < 0.0:
            raise ValueError("Image-quantity grounding requires a positive maximum distance and non-negative margin")
        self.detector = detector or VisualPrimitiveDetector()
        self.label_recognizer = label_recognizer or UnresolvedLabelRecognizer()
        self.point_tolerance = point_tolerance
        self.point_label_max_distance = point_label_max_distance
        self.point_label_ambiguity_margin = point_label_ambiguity_margin
        self.line_label_max_distance = line_label_max_distance
        self.line_label_ambiguity_margin = line_label_ambiguity_margin
        self.infer_planar_faces = infer_planar_faces
        self.minimum_face_area_px2 = minimum_face_area_px2
        self.maximum_face_vertices = maximum_face_vertices
        self.image_quantity_max_distance = image_quantity_max_distance
        self.image_quantity_ambiguity_margin = image_quantity_ambiguity_margin
        self.record_adapter = HypothesisRecordAdapter()

    def parse(self, data: ParserInput | Mapping[str, Any]) -> TypedParseResult:
        parser_input = validate_parser_input(data)
        image_path = Path(parser_input.image)
        if not image_path.is_file():
            raise InvalidGeometryInput(f"Parser input image does not exist: {image_path}")
        visual = self.detector.detect(str(image_path))
        if visual.get("error"):
            raise InvalidGeometryInput(f"Parser could not read image {image_path}: {visual['error']}")
        image_size = _image_size(visual.get("image_size"))
        labels = self.label_recognizer.recognize(str(image_path), visual.get("text_regions") or ())
        graph = _build_graph(
            parser_input,
            visual,
            labels,
            image_size=image_size,
            detector_configuration=_detector_configuration(self.detector),
            point_tolerance=self.point_tolerance,
            point_label_max_distance=self.point_label_max_distance,
            point_label_ambiguity_margin=self.point_label_ambiguity_margin,
            line_label_max_distance=self.line_label_max_distance,
            line_label_ambiguity_margin=self.line_label_ambiguity_margin,
            infer_planar_faces=self.infer_planar_faces,
            minimum_face_area_px2=self.minimum_face_area_px2,
            maximum_face_vertices=self.maximum_face_vertices,
            image_quantity_max_distance=self.image_quantity_max_distance,
            image_quantity_ambiguity_margin=self.image_quantity_ambiguity_margin,
        )
        target_count = len(graph.targets)
        stages = (
            _stage(
                "visual_candidates",
                "complete",
                len(visual.get("lines") or ())
                + len(visual.get("circles") or ())
                + len(visual.get("arcs") or ())
                + len((visual.get("topology") or {}).get("point_candidates") or ())
                + len(visual.get("fill_regions") or ())
                + len(visual.get("markers") or ()),
            ),
            _stage("text_candidates", "complete" if any(item.get("text") for item in labels) else "partial", len(labels)),
            _stage("topology_graph", "complete", len(graph.objects) + len(graph.relations)),
            _stage(
                "point_label_grounding",
                "partial" if graph.metadata["point_label_grounding"]["num_conflicts"] else "complete",
                sum(
                    item.predicate == "binds"
                    and item.metadata.get("binding_kind") == "point_name"
                    for item in graph.relations
                ),
            ),
            _stage(
                "line_label_grounding",
                "partial" if graph.metadata["line_label_grounding"]["num_conflicts"] else "complete",
                graph.metadata["line_label_grounding"]["num_bindings"],
            ),
            _stage(
                "image_quantity_grounding",
                "partial" if graph.metadata.get("image_quantity_issues") else "complete",
                sum(item.origin is ParseOrigin.IMAGE for item in graph.quantities),
            ),
            _stage(
                "marker_grounding",
                "complete",
                sum(item.kind is ParseObjectKind.MARKER for item in graph.objects),
            ),
            _stage(
                "fill_grounding",
                "partial"
                if graph.metadata["face_extraction"]["num_fill_candidates"]
                != graph.metadata["face_extraction"]["num_fill_bindings"]
                or graph.metadata["face_extraction"]["num_interior_gap_candidates"]
                != graph.metadata["face_extraction"]["num_hole_region_links"]
                else "complete",
                graph.metadata["face_extraction"]["num_fill_bindings"],
            ),
            _stage(
                "question_relations",
                "partial" if graph.metadata.get("question_relation_issues") else "complete",
                sum(item.score_kind == "explicit_question_relation" for item in graph.relations),
            ),
            _stage(
                "question_quantities",
                "partial" if graph.metadata.get("question_quantity_issues") else "complete",
                sum(item.origin is ParseOrigin.TEXT for item in graph.quantities),
            ),
            _stage("target_type", "complete" if target_count else "partial", target_count),
            _stage("admission", "pending", 0),
        )
        confidence = _graph_confidence(graph)
        record = self.record_adapter.convert_parse_graph(
            sample_id=parser_input.id,
            dataset_name="parser_benchmark",
            parser_name=self.parser_name,
            backend="opencv_typed_v0",
            graph=graph,
            confidence=confidence,
            verification={"status": "not_admitted", "reason": "typed_parse_graph_requires_evidence_and_geometry_review"},
            image_path=parser_input.image,
            metadata={"typed_parser_version": TYPED_PARSER_VERSION, "stage_count": len(stages)},
        )
        return TypedParseResult(parser_input=parser_input, graph=graph, record=record, stages=stages)


def _build_graph(
    parser_input: ParserInput,
    visual: Mapping[str, Any],
    labels: Sequence[Mapping[str, Any]],
    *,
    image_size: tuple[int, int],
    detector_configuration: Mapping[str, Any],
    point_tolerance: float,
    point_label_max_distance: float,
    point_label_ambiguity_margin: float,
    line_label_max_distance: float,
    line_label_ambiguity_margin: float,
    infer_planar_faces: bool,
    minimum_face_area_px2: float,
    maximum_face_vertices: int,
    image_quantity_max_distance: float,
    image_quantity_ambiguity_margin: float,
) -> ParseGraph:
    lines = tuple(visual.get("lines") or ())
    circles = tuple(visual.get("circles") or ())
    arcs = tuple(visual.get("arcs") or ())
    fill_regions = _validate_fill_region_candidates(tuple(visual.get("fill_regions") or ()))
    markers = tuple(visual.get("markers") or ())
    topology = dict(visual.get("topology") or {})
    topology.setdefault("arc_endpoints", _arc_endpoints(arcs))
    evidence: list[ParseEvidence] = []
    objects: list[ParseObject] = []
    relations: list[ParseRelation] = []
    regions: list[ParseRegion] = []

    line_evidence: dict[str, str] = {}
    for line in lines:
        line_id = str(line["id"])
        evidence_id = f"evidence:{line_id}"
        line_evidence[line_id] = evidence_id
        evidence.append(
            ParseEvidence(
                id=evidence_id,
                source_id=parser_input.id,
                kind=ParseEvidenceKind.IMAGE_REGION,
                precision=EvidencePrecision.LOCALIZED,
                bbox=_line_bbox(line),
                metadata={"source_ids": list(line.get("source_ids") or ())},
            )
        )

    point_observations = _point_observations(topology, parser_input.id)
    evidence.extend(observation["evidence"] for observation in point_observations)
    points, observation_to_point = _cluster_points(point_observations, point_tolerance)
    objects.extend(points)

    for line in lines:
        line_id = str(line["id"])
        segment_id = f"segment:{line_id}"
        start_id = observation_to_point[f"{line_id}:endpoint:0"]
        end_id = observation_to_point[f"{line_id}:endpoint:1"]
        objects.append(
            ParseObject(
                id=segment_id,
                kind=ParseObjectKind.SEGMENT,
                origin=ParseOrigin.IMAGE,
                confidence=0.75,
                score_kind="hough_segment_score",
                references=(ParseReference("start", start_id), ParseReference("end", end_id)),
                evidence_refs=(line_evidence[line_id],),
                visible=True,
                geometry={"p1": list(line["p1"]), "p2": list(line["p2"]), "angle_deg": line.get("angle_deg")},
            )
        )
        ordered_points = _ordered_line_points(line, topology, observation_to_point)
        relations.append(
            ParseRelation(
                id=f"relation:ordered:{line_id}",
                predicate="ordered_points",
                arguments=(
                    ParseReference("support", segment_id),
                    *(ParseReference(f"point_{index}", point_id) for index, point_id in enumerate(ordered_points)),
                ),
                origin=ParseOrigin.IMAGE,
                confidence=0.72,
                score_kind="pixel_projection_order",
                evidence_refs=(line_evidence[line_id],),
            )
        )

    point_on_line_count = 0
    for candidate in topology.get("point_candidates") or ():
        point_id = observation_to_point.get(f"{candidate['id']}:point_candidate")
        if point_id is None:
            continue
        for support in candidate.get("line_support") or ():
            line_id = str(support.get("line_id", ""))
            if line_id not in line_evidence:
                continue
            relation_id = f"relation:point_on_line:{candidate['id']}:{line_id}"
            relations.append(
                ParseRelation(
                    id=relation_id,
                    predicate="point_on_line",
                    arguments=(
                        ParseReference("point", point_id),
                        ParseReference("support", f"segment:{line_id}"),
                    ),
                    origin=ParseOrigin.IMAGE,
                    confidence=float(candidate.get("confidence", 0.6)),
                    score_kind="filled_point_line_residual",
                    evidence_refs=(
                        f"evidence:{candidate['id']}:point_candidate",
                        line_evidence[line_id],
                    ),
                    metadata={
                        "projection": support.get("projection"),
                        "residual_px": support.get("residual_px"),
                        "bounded": True,
                    },
                )
            )
            point_on_line_count += 1

    if infer_planar_faces:
        face_evidence, face_objects, face_regions = _planar_face_candidates(
            lines,
            topology,
            observation_to_point,
            points,
            line_evidence,
            fill_regions,
            parser_input.id,
            minimum_area=minimum_face_area_px2,
            maximum_vertices=maximum_face_vertices,
        )
        evidence.extend(face_evidence)
        objects.extend(face_objects)
        regions.extend(face_regions)
    else:
        evidence.extend(_fill_region_evidence(fill_regions, {}, parser_input.id))

    circle_evidence: dict[str, str] = {}
    for circle in circles:
        circle_id = str(circle["id"])
        evidence_id = f"evidence:{circle_id}"
        circle_evidence[circle_id] = evidence_id
        center_evidence_id = f"evidence:{circle_id}:center"
        center_id = f"point:{circle_id}:center"
        cx, cy = (float(value) for value in circle["center"])
        radius = float(circle["radius_px"])
        evidence.append(
            ParseEvidence(
                id=evidence_id,
                source_id=parser_input.id,
                kind=ParseEvidenceKind.IMAGE_REGION,
                precision=EvidencePrecision.LOCALIZED,
                bbox=(cx - radius, cy - radius, cx + radius, cy + radius),
                metadata={"edge_support": circle.get("edge_support")},
            )
        )
        evidence.append(
            ParseEvidence(
                id=center_evidence_id,
                source_id=parser_input.id,
                kind=ParseEvidenceKind.PIXEL_POINT,
                precision=EvidencePrecision.LOCALIZED,
                point=(cx, cy),
                metadata={"derived_from": evidence_id},
            )
        )
        objects.append(
            ParseObject(
                id=center_id,
                kind=ParseObjectKind.POINT,
                origin=ParseOrigin.IMAGE,
                confidence=float(circle.get("edge_support", 0.6)),
                score_kind="circle_center_score",
                evidence_refs=(center_evidence_id,),
                visible_name=None,
                visible=False,
                geometry={"pixel": [cx, cy]},
                metadata={"anonymous": True, "role": "circle_center"},
            )
        )
        objects.append(
            ParseObject(
                id=f"circle:{circle_id}",
                kind=ParseObjectKind.CIRCLE,
                origin=ParseOrigin.IMAGE,
                confidence=float(circle.get("edge_support", 0.6)),
                score_kind="circle_edge_support",
                references=(ParseReference("center", center_id),),
                evidence_refs=(evidence_id,),
                visible=True,
                geometry={"center": [cx, cy], "radius_px": radius},
            )
        )

    arc_evidence: dict[str, str] = {}
    arc_object_ids: dict[str, str] = {}
    arc_object_count = 0
    for arc in arcs:
        arc_id = str(arc["id"])
        evidence_id = f"evidence:{arc_id}"
        arc_evidence[arc_id] = evidence_id
        start_id = observation_to_point.get(f"{arc_id}:arc_endpoint:start")
        end_id = observation_to_point.get(f"{arc_id}:arc_endpoint:end")
        support_circle = arc.get("circle_id")
        support_circle_id = (
            f"circle:{support_circle}"
            if support_circle is not None and str(support_circle) in circle_evidence
            else None
        )
        evidence.append(
            ParseEvidence(
                id=evidence_id,
                source_id=parser_input.id,
                kind=ParseEvidenceKind.IMAGE_REGION,
                precision=EvidencePrecision.LOCALIZED,
                bbox=_arc_bbox(arc),
                metadata={
                    "method": arc.get("score_kind", "visual_arc_candidate"),
                    "center": list(arc.get("center") or ()),
                    "radius_px": arc.get("radius_px"),
                    "sweep_deg": arc.get("sweep_deg"),
                    "residual_px": arc.get("residual_px"),
                    "support_circle_id": support_circle_id,
                    "support_circle_status": "resolved" if support_circle_id else "unresolved",
                },
            )
        )
        if support_circle_id is None or start_id is None or end_id is None:
            continue
        objects.append(
            ParseObject(
                id=f"arc:{arc_id}",
                kind=ParseObjectKind.ARC,
                origin=ParseOrigin.IMAGE,
                confidence=float(arc.get("edge_support", 0.6)),
                score_kind=str(arc.get("score_kind") or "visual_arc_score"),
                references=(
                    ParseReference("circle", support_circle_id),
                    ParseReference("start", start_id),
                    ParseReference("end", end_id),
                ),
                evidence_refs=(evidence_id,),
                visible=True,
                geometry={"orientation": _cartesian_arc_orientation(arc)},
                metadata={
                    "center": list(arc.get("center") or ()),
                    "radius_px": arc.get("radius_px"),
                    "sweep_deg": arc.get("sweep_deg"),
                    "residual_px": arc.get("residual_px"),
                    "image_orientation": str(arc.get("orientation") or "ccw"),
                    "orientation_coordinate_frame": "cartesian",
                    "support_circle_status": "resolved",
                },
            )
        )
        arc_object_ids[arc_id] = f"arc:{arc_id}"
        arc_object_count += 1

    bound_fill_ids = {
        str(region.metadata["fill_candidate_id"])
        for region in regions
        if region.metadata.get("fill_candidate_id") is not None
    }
    curved_objects, curved_regions, curved_fill_bindings = _filled_arc_chord_region_candidates(
        arcs,
        lines,
        fill_regions,
        bound_fill_ids,
        topology,
        observation_to_point,
        line_evidence,
        arc_evidence,
        arc_object_ids,
    )
    sector_objects, sector_regions, sector_fill_bindings = _filled_sector_region_candidates(
        arcs,
        lines,
        fill_regions,
        {*bound_fill_ids, *curved_fill_bindings},
        topology,
        observation_to_point,
        points,
        line_evidence,
        arc_evidence,
        arc_object_ids,
    )
    curved_objects.extend(sector_objects)
    curved_regions.extend(sector_regions)
    curved_fill_bindings.update(sector_fill_bindings)
    objects.extend(curved_objects)
    regions.extend(curved_regions)
    _promote_curved_fill_evidence(evidence, curved_fill_bindings)

    for label in labels:
        label_id = str(label["id"])
        evidence_id = f"evidence:{label_id}"
        bbox = _component_bbox(label["bbox"])
        evidence.append(
            ParseEvidence(
                id=evidence_id,
                source_id=parser_input.id,
                kind=ParseEvidenceKind.IMAGE_REGION,
                precision=EvidencePrecision.BOUND if label.get("text") else EvidencePrecision.LOCALIZED,
                bbox=bbox,
                raw_content=str(label["text"]) if label.get("text") is not None else None,
                metadata={
                    "recognizer_status": label.get("status"),
                    "recognizer_provenance": dict(label.get("provenance") or {}),
                },
            )
        )
        objects.append(
            ParseObject(
                id=f"text:{label_id}",
                kind=ParseObjectKind.TEXT,
                origin=ParseOrigin.IMAGE,
                confidence=float(label.get("confidence", 0.0)),
                score_kind="ocr_score" if label.get("text") else "unresolved_region_score",
                evidence_refs=(evidence_id,),
                visible_name=str(label["text"]) if label.get("text") is not None else None,
                visible=True,
                geometry={"bbox": list(bbox)},
                metadata={
                    "recognizer_status": label.get("status"),
                    "recognizer_provenance": dict(label.get("provenance") or {}),
                },
            )
        )

    marker_relations = _marker_candidates(
        markers,
        parser_input.id,
        lines,
        topology,
        observation_to_point,
        line_evidence,
        evidence,
        objects,
    )

    for index, intersection in enumerate(topology.get("line_intersections") or ()):
        point_id = observation_to_point.get(f"{intersection['id']}:intersection")
        line_ids = [str(item) for item in intersection.get("line_ids") or ()]
        if point_id is None or any(line_id not in line_evidence for line_id in line_ids):
            continue
        relations.append(
            ParseRelation(
                id=f"relation:intersection:{index}",
                predicate="intersection",
                arguments=(
                    *(ParseReference("support", f"segment:{line_id}") for line_id in line_ids),
                    ParseReference("point", point_id),
                ),
                origin=ParseOrigin.IMAGE,
                confidence=0.8,
                score_kind="bounded_segment_intersection",
                evidence_refs=tuple(line_evidence[line_id] for line_id in line_ids),
            )
        )

    for index, overlap in enumerate(topology.get("line_overlaps") or ()):
        line_ids = tuple(str(item) for item in overlap.get("line_ids") or ())
        if len(line_ids) != 2 or any(line_id not in line_evidence for line_id in line_ids):
            continue
        relations.append(
            ParseRelation(
                id=f"relation:linear_overlap:{index}",
                predicate="linear_overlap",
                arguments=tuple(
                    ParseReference("support", f"segment:{line_id}") for line_id in line_ids
                ),
                origin=ParseOrigin.IMAGE,
                confidence=0.68,
                score_kind="near_collinear_segment_overlap",
                evidence_refs=tuple(line_evidence[line_id] for line_id in line_ids),
                metadata={
                    "overlap_length_px": overlap.get("overlap_length_px"),
                    "residual_px": overlap.get("residual_px"),
                    "exact_relation_not_admitted": True,
                },
            )
        )

    for index, intersection in enumerate(topology.get("circle_line_intersections") or ()):
        point_id = observation_to_point.get(f"{intersection['id']}:intersection")
        line_id = str(intersection.get("line_id", ""))
        circle_id = str(intersection.get("circle_id", ""))
        if point_id is None or line_id not in line_evidence or circle_id not in circle_evidence:
            continue
        contact = str(intersection.get("contact", "unknown"))
        relations.append(
            ParseRelation(
                id=f"relation:circle_line_intersection:{index}",
                predicate="intersection",
                arguments=(
                    ParseReference("support", f"segment:{line_id}"),
                    ParseReference("support", f"circle:{circle_id}"),
                    ParseReference("point", point_id),
                ),
                origin=ParseOrigin.IMAGE,
                confidence=0.78,
                score_kind="bounded_segment_circle_intersection",
                evidence_refs=(line_evidence[line_id], circle_evidence[circle_id]),
                metadata={"contact": contact},
            )
        )
        if contact == "tangent":
            relations.append(
                ParseRelation(
                    id=f"relation:tangent:{index}",
                    predicate="tangent",
                    arguments=(
                        ParseReference("point", point_id),
                        ParseReference("line", f"segment:{line_id}"),
                        ParseReference("circle", f"circle:{circle_id}"),
                    ),
                    origin=ParseOrigin.IMAGE,
                    confidence=0.72,
                    score_kind="visual_line_circle_tangent_candidate",
                    evidence_refs=(line_evidence[line_id], circle_evidence[circle_id]),
                    dependencies=(f"relation:circle_line_intersection:{index}",),
                    metadata={
                        "contact": "tangent",
                        "derived_from_intersection": f"relation:circle_line_intersection:{index}",
                        "exact_relation_not_admitted": True,
                    },
                )
            )

    for index, intersection in enumerate(topology.get("circle_circle_intersections") or ()):
        point_id = observation_to_point.get(f"{intersection['id']}:intersection")
        circle_ids = tuple(str(item) for item in intersection.get("circle_ids") or ())
        if point_id is None or len(circle_ids) != 2 or any(circle_id not in circle_evidence for circle_id in circle_ids):
            continue
        relations.append(
            ParseRelation(
                id=f"relation:circle_circle_intersection:{index}",
                predicate="intersection",
                arguments=(
                    ParseReference("support", f"circle:{circle_ids[0]}"),
                    ParseReference("support", f"circle:{circle_ids[1]}"),
                    ParseReference("point", point_id),
                ),
                origin=ParseOrigin.IMAGE,
                confidence=0.78,
                score_kind="circle_edge_intersection",
                evidence_refs=tuple(circle_evidence[circle_id] for circle_id in circle_ids),
                metadata={"contact": intersection.get("contact", "unknown")},
            )
        )

    for index, intersection in enumerate(topology.get("arc_line_intersections") or ()):
        point_id = observation_to_point.get(f"{intersection['id']}:intersection")
        arc_id = str(intersection.get("arc_id", ""))
        line_id = str(intersection.get("line_id", ""))
        if point_id is None or arc_id not in arc_object_ids or line_id not in line_evidence:
            continue
        relations.append(
            ParseRelation(
                id=f"relation:arc_line_intersection:{index}",
                predicate="intersection",
                arguments=(
                    ParseReference("support", f"segment:{line_id}"),
                    ParseReference("support", arc_object_ids[arc_id]),
                    ParseReference("point", point_id),
                ),
                origin=ParseOrigin.IMAGE,
                confidence=0.74,
                score_kind="bounded_segment_arc_intersection",
                evidence_refs=(line_evidence[line_id], arc_evidence[arc_id]),
                metadata={"contact": intersection.get("contact", "unknown")},
            )
        )

    for index, intersection in enumerate(topology.get("arc_circle_intersections") or ()):
        point_id = observation_to_point.get(f"{intersection['id']}:intersection")
        arc_id = str(intersection.get("arc_id", ""))
        circle_id = str(intersection.get("circle_id", ""))
        if point_id is None or arc_id not in arc_object_ids or circle_id not in circle_evidence:
            continue
        relations.append(
            ParseRelation(
                id=f"relation:arc_circle_intersection:{index}",
                predicate="intersection",
                arguments=(
                    ParseReference("support", arc_object_ids[arc_id]),
                    ParseReference("support", f"circle:{circle_id}"),
                    ParseReference("point", point_id),
                ),
                origin=ParseOrigin.IMAGE,
                confidence=0.74,
                score_kind="arc_circle_intersection",
                evidence_refs=(arc_evidence[arc_id], circle_evidence[circle_id]),
                metadata={"contact": intersection.get("contact", "unknown")},
            )
        )

    for index, intersection in enumerate(topology.get("arc_arc_intersections") or ()):
        point_id = observation_to_point.get(f"{intersection['id']}:intersection")
        arc_ids = tuple(str(item) for item in intersection.get("arc_ids") or ())
        if point_id is None or len(arc_ids) != 2 or any(arc_id not in arc_object_ids for arc_id in arc_ids):
            continue
        relations.append(
            ParseRelation(
                id=f"relation:arc_arc_intersection:{index}",
                predicate="intersection",
                arguments=(
                    ParseReference("support", arc_object_ids[arc_ids[0]]),
                    ParseReference("support", arc_object_ids[arc_ids[1]]),
                    ParseReference("point", point_id),
                ),
                origin=ParseOrigin.IMAGE,
                confidence=0.72,
                score_kind="arc_arc_intersection",
                evidence_refs=tuple(arc_evidence[arc_id] for arc_id in arc_ids),
                metadata={"contact": intersection.get("contact", "unknown")},
            )
        )

    relations.extend(_angle_relations(lines, line_evidence))
    suppressed_marker_metrics = _suppress_redundant_visual_relations(relations, marker_relations)
    relations.extend(marker_relations)
    bindings, point_label_conflicts = ground_point_labels(
        objects,
        {item.id: item for item in evidence},
        max_distance=point_label_max_distance,
        ambiguity_margin=point_label_ambiguity_margin,
    )
    relations.extend(bindings)
    line_bindings, line_label_conflicts = ground_line_labels(
        objects,
        {item.id: item for item in evidence},
        max_distance=line_label_max_distance,
        ambiguity_margin=line_label_ambiguity_margin,
    )
    relations.extend(line_bindings)
    image_quantities, image_quantity_conflicts, image_quantity_issues = ground_segment_length_labels(
        objects,
        {item.id: item for item in evidence},
        max_distance=image_quantity_max_distance,
        ambiguity_margin=image_quantity_ambiguity_margin,
    )
    question_relations, relation_evidence, relation_issues = _question_relations(parser_input, objects, bindings)
    suppressed_question_metrics = _suppress_redundant_visual_relations(relations, question_relations)
    relations.extend(question_relations)
    evidence.extend(relation_evidence)
    question_quantities, quantity_evidence, quantity_issues = _question_quantities(parser_input, objects, bindings)
    evidence.extend(quantity_evidence)
    target, question_evidence, target_relations = _target(
        parser_input,
        objects,
        regions,
        bindings,
        question_relations,
    )
    relations.extend(target_relations)
    if question_evidence is not None:
        evidence.append(question_evidence)
    return ParseGraph(
        id=f"parse:{parser_input.id}",
        source_id=parser_input.id,
        evidence=tuple(evidence),
        objects=tuple(objects),
        relations=tuple(relations),
        quantities=(*image_quantities, *question_quantities),
        regions=tuple(regions),
        targets=(target,) if target is not None else (),
        conflicts=(*point_label_conflicts, *line_label_conflicts, *image_quantity_conflicts),
        metadata={
            "parser": TypedDiagramParserV0.parser_name,
            "parser_version": TYPED_PARSER_VERSION,
            "detector": dict(detector_configuration),
            "protocol": "image_question" if parser_input.question else "image_only",
            "image_size": list(image_size),
            "question_quantity_issues": list(quantity_issues),
            "question_relation_issues": list(relation_issues),
            "suppressed_visual_relation_ids": [*suppressed_marker_metrics, *suppressed_question_metrics],
            "marker_grounding": {
                "num_candidates": len(markers),
                "num_relations": len(marker_relations),
                "supported_kinds": ["right_angle", "parallel", "equal_length_tick", "angle_arc"],
            },
            "arc_grounding": {
                "num_candidates": len(arcs),
                "num_bound_arcs": arc_object_count,
                "num_unresolved_support_circles": len(arcs) - arc_object_count,
                "policy": "arc_requires_existing_unique_support_circle",
            },
            "point_detection": {
                "num_candidates": len((topology.get("point_candidates") or ())),
                "num_line_bindings": point_on_line_count,
                "policy": "filled_blob_only; endpoints_and_intersections_are_deduplicated",
            },
            "point_label_grounding": {
                "num_bindings": sum(item.predicate == "binds" for item in bindings),
                "num_conflicts": len(point_label_conflicts),
            },
            "line_label_grounding": {
                "num_bindings": len(line_bindings),
                "num_conflicts": len(line_label_conflicts),
                "max_distance_px": line_label_max_distance,
                "ambiguity_margin_px": line_label_ambiguity_margin,
            },
            "image_quantity_issues": list(image_quantity_issues),
            "image_quantity_grounding": {
                "max_distance_px": image_quantity_max_distance,
                "ambiguity_margin_px": image_quantity_ambiguity_margin,
                "supported_kinds": ["segment_length"],
            },
            "face_extraction": {
                "enabled": infer_planar_faces,
                "minimum_area_px2": minimum_face_area_px2,
                "maximum_vertices": maximum_face_vertices,
                "num_faces": len(regions),
                "num_straight_line_faces": len(regions) - len(curved_regions),
                "num_curved_faces": len(curved_regions),
                "num_arc_chord_faces": sum(
                    region.metadata.get("boundary_kind") == "arc_chord" for region in curved_regions
                ),
                "num_arc_radii_faces": sum(
                    region.metadata.get("boundary_kind") == "arc_radii" for region in curved_regions
                ),
                "num_fill_candidates": len(fill_regions),
                "num_fill_bindings": sum(region.metadata.get("fill_status") == "observed" for region in regions),
                "num_interior_gap_candidates": sum(
                    len(candidate.get("interior_gaps") or ()) for candidate in fill_regions
                ),
                "num_regions_with_unresolved_gaps": sum(
                    region.metadata.get("interior_gap_status")
                    in {"unresolved_boundary_candidate", "partially_grounded"}
                    for region in regions
                ),
                "num_hole_region_links": sum(
                    len(region.holes) for region in regions
                ),
                "semantic_shading_policy": (
                    "explicit_shaded_target_plus_unique_fill_creates_unadmitted_semantic_hypothesis"
                ),
                "hole_policy": "requires_closed_parse_region_boundary_support",
            },
        },
    )


def _point_observations(topology: Mapping[str, Any], source_id: str) -> list[dict[str, Any]]:
    observations = []
    for endpoint in topology.get("endpoints") or ():
        line_id = str(endpoint["line_id"])
        endpoint_index = 0 if str(endpoint["id"]).endswith("_p1") else 1
        key = f"{line_id}:endpoint:{endpoint_index}"
        observations.append(_point_observation(key, endpoint["xy"], source_id, "line_endpoint"))
    for intersection in topology.get("line_intersections") or ():
        key = f"{intersection['id']}:intersection"
        observations.append(_point_observation(key, intersection["xy"], source_id, "line_intersection"))
    for intersection in topology.get("circle_line_intersections") or ():
        key = f"{intersection['id']}:intersection"
        observations.append(_point_observation(key, intersection["xy"], source_id, "circle_line_intersection"))
    for intersection in topology.get("circle_circle_intersections") or ():
        key = f"{intersection['id']}:intersection"
        observations.append(_point_observation(key, intersection["xy"], source_id, "circle_circle_intersection"))
    for intersection in topology.get("arc_line_intersections") or ():
        key = f"{intersection['id']}:intersection"
        observations.append(_point_observation(key, intersection["xy"], source_id, "arc_line_intersection"))
    for intersection in topology.get("arc_circle_intersections") or ():
        key = f"{intersection['id']}:intersection"
        observations.append(_point_observation(key, intersection["xy"], source_id, "arc_circle_intersection"))
    for intersection in topology.get("arc_arc_intersections") or ():
        key = f"{intersection['id']}:intersection"
        observations.append(_point_observation(key, intersection["xy"], source_id, "arc_arc_intersection"))
    for endpoint in topology.get("arc_endpoints") or ():
        key = f"{endpoint['arc_id']}:arc_endpoint:{endpoint['endpoint']}"
        observations.append(_point_observation(key, endpoint["xy"], source_id, "arc_endpoint"))
    for point in topology.get("point_candidates") or ():
        key = f"{point['id']}:point_candidate"
        observations.append(_point_observation(key, point["xy"], source_id, "explicit_point_blob"))
    return observations


def _point_observation(key: str, xy: Sequence[Any], source_id: str, kind: str) -> dict[str, Any]:
    point = float(xy[0]), float(xy[1])
    evidence = ParseEvidence(
        id=f"evidence:{key}",
        source_id=source_id,
        kind=ParseEvidenceKind.PIXEL_POINT,
        precision=EvidencePrecision.LOCALIZED,
        point=point,
        metadata={"observation_kind": kind},
    )
    return {"key": key, "point": point, "evidence": evidence}


def _cluster_points(
    observations: Sequence[Mapping[str, Any]],
    tolerance: float,
) -> tuple[list[ParseObject], dict[str, str]]:
    clusters: list[dict[str, Any]] = []
    for observation in sorted(observations, key=lambda item: (item["point"][1], item["point"][0], item["key"])):
        x, y = observation["point"]
        cluster = next(
            (candidate for candidate in clusters if math.dist((x, y), candidate["center"]) <= tolerance),
            None,
        )
        if cluster is None:
            clusters.append({"center": (x, y), "items": [observation]})
            continue
        cluster["items"].append(observation)
        count = len(cluster["items"])
        cluster["center"] = (
            (cluster["center"][0] * (count - 1) + x) / count,
            (cluster["center"][1] * (count - 1) + y) / count,
        )
    points = []
    observation_to_point = {}
    for index, cluster in enumerate(clusters):
        point_id = f"visual_point_{index:03d}"
        evidence_refs = tuple(sorted(str(item["evidence"].id) for item in cluster["items"]))
        points.append(
            ParseObject(
                id=point_id,
                kind=ParseObjectKind.POINT,
                origin=ParseOrigin.IMAGE,
                confidence=min(0.95, 0.55 + 0.08 * len(cluster["items"])),
                score_kind="cluster_support_score",
                evidence_refs=evidence_refs,
                visible_name=None,
                visible=True,
                geometry={"pixel": [round(value, 3) for value in cluster["center"]]},
                metadata={"anonymous": True},
            )
        )
        for item in cluster["items"]:
            observation_to_point[str(item["key"])] = point_id
    return points, observation_to_point


def _ordered_line_points(
    line: Mapping[str, Any],
    topology: Mapping[str, Any],
    observation_to_point: Mapping[str, str],
) -> tuple[str, ...]:
    line_id = str(line["id"])
    start = tuple(float(value) for value in line["p1"])
    end = tuple(float(value) for value in line["p2"])
    direction = end[0] - start[0], end[1] - start[1]
    point_positions = [
        (observation_to_point[f"{line_id}:endpoint:0"], start),
        (observation_to_point[f"{line_id}:endpoint:1"], end),
    ]
    for intersection in topology.get("line_intersections") or ():
        if line_id not in intersection.get("line_ids", ()):
            continue
        point_id = observation_to_point.get(f"{intersection['id']}:intersection")
        if point_id is not None:
            point_positions.append((point_id, tuple(float(value) for value in intersection["xy"])))
    for intersection in topology.get("circle_line_intersections") or ():
        if line_id != str(intersection.get("line_id", "")):
            continue
        point_id = observation_to_point.get(f"{intersection['id']}:intersection")
        if point_id is not None:
            point_positions.append((point_id, tuple(float(value) for value in intersection["xy"])))
    for intersection in topology.get("arc_line_intersections") or ():
        if line_id != str(intersection.get("line_id", "")):
            continue
        point_id = observation_to_point.get(f"{intersection['id']}:intersection")
        if point_id is not None:
            point_positions.append((point_id, tuple(float(value) for value in intersection["xy"])))
    for candidate in topology.get("point_candidates") or ():
        if not any(str(item.get("line_id", "")) == line_id for item in candidate.get("line_support") or ()):
            continue
        point_id = observation_to_point.get(f"{candidate['id']}:point_candidate")
        if point_id is not None:
            point_positions.append((point_id, tuple(float(value) for value in candidate["xy"])))
    ordered = sorted(
        point_positions,
        key=lambda item: (item[1][0] - start[0]) * direction[0] + (item[1][1] - start[1]) * direction[1],
    )
    return tuple(dict.fromkeys(point_id for point_id, _ in ordered))


def _filled_arc_chord_region_candidates(
    arcs: Sequence[Mapping[str, Any]],
    lines: Sequence[Mapping[str, Any]],
    fill_regions: Sequence[Mapping[str, Any]],
    excluded_fill_ids: set[str],
    topology: Mapping[str, Any],
    observation_to_point: Mapping[str, str],
    line_evidence: Mapping[str, str],
    arc_evidence: Mapping[str, str],
    arc_object_ids: Mapping[str, str],
) -> tuple[list[ParseObject], list[ParseRegion], dict[str, dict[str, Any]]]:
    boundary_candidates = []
    for arc in arcs:
        arc_id = str(arc["id"])
        if arc_id not in arc_object_ids:
            continue
        start_id = observation_to_point.get(f"{arc_id}:arc_endpoint:start")
        end_id = observation_to_point.get(f"{arc_id}:arc_endpoint:end")
        if start_id is None or end_id is None or start_id == end_id:
            continue
        arc_points = _visual_arc_points(arc)
        for line in lines:
            line_id = str(line["id"])
            if line_id not in line_evidence or not _line_supports_arc_chord(line, arc):
                continue
            ordered = _ordered_line_points(line, topology, observation_to_point)
            if start_id not in ordered or end_id not in ordered:
                continue
            reuse_support = len(ordered) == 2 and set(ordered) == {start_id, end_id}
            boundary_orientation = (
                "forward"
                if reuse_support and ordered == (end_id, start_id)
                else "reverse"
            )
            boundary_candidates.append(
                {
                    "key": (arc_id, line_id),
                    "arc_id": arc_id,
                    "line_id": line_id,
                    "start_id": start_id,
                    "end_id": end_id,
                    "polygon": arc_points,
                    "arc": arc,
                    "reuse_support": reuse_support,
                    "boundary_orientation": boundary_orientation,
                }
            )

    selected = _select_unique_fill_boundaries(fill_regions, excluded_fill_ids, boundary_candidates)

    objects = []
    regions = []
    bindings: dict[str, dict[str, Any]] = {}
    for score, fill_id, fill, boundary in sorted(selected, key=lambda item: item[1]):
        arc_id = boundary["arc_id"]
        line_id = boundary["line_id"]
        chord_id = (
            f"segment:{line_id}"
            if boundary["reuse_support"]
            else f"segment:curved-boundary:{arc_id}:{line_id}"
        )
        region_id = f"region:curved:{arc_id}:{line_id}"
        confidence = min(
            float(fill.get("confidence", 0.6)),
            float(boundary["arc"].get("edge_support", 0.6)),
            0.9,
        )
        if not boundary["reuse_support"]:
            objects.append(
                ParseObject(
                    id=chord_id,
                    kind=ParseObjectKind.SEGMENT,
                    origin=ParseOrigin.COMPILED,
                    confidence=confidence,
                    score_kind="arc_endpoint_chord_support",
                    references=(
                        ParseReference("start", boundary["start_id"]),
                        ParseReference("end", boundary["end_id"]),
                    ),
                    evidence_refs=(line_evidence[line_id], arc_evidence[arc_id]),
                    visible=True,
                    geometry={
                        "p1": list(boundary["arc"]["start"]),
                        "p2": list(boundary["arc"]["end"]),
                        "angle_deg": _line_angle(boundary["arc"]["start"], boundary["arc"]["end"]),
                    },
                    metadata={
                        "source_support_id": f"segment:{line_id}",
                        "boundary_role": "arc_chord",
                    },
                )
            )
        region_metadata = {
            "kind": "observed_curved_region",
            "boundary_kind": "arc_chord",
            "fill_status": "observed",
            "shading_status": "unknown",
            "fill_candidate_id": fill_id,
            "fill_match_score": round(score, 3),
            **_interior_gap_status(fill),
        }
        regions.append(
            ParseRegion(
                id=region_id,
                boundary=(
                    BoundaryReference(arc_object_ids[arc_id]),
                    BoundaryReference(chord_id, orientation=boundary["boundary_orientation"]),
                ),
                origin=ParseOrigin.IMAGE,
                confidence=confidence,
                score_kind="filled_arc_chord_region_match",
                evidence_refs=(
                    f"evidence:{fill_id}",
                    arc_evidence[arc_id],
                    line_evidence[line_id],
                ),
                metadata=region_metadata,
            )
        )
        bindings[fill_id] = {
            "bound_curved_region_id": region_id,
            "bound_curve_id": arc_object_ids[arc_id],
            "bound_chord_id": chord_id,
            "curved_boundary_match_score": round(score, 3),
        }
    return objects, regions, bindings


def _filled_sector_region_candidates(
    arcs: Sequence[Mapping[str, Any]],
    lines: Sequence[Mapping[str, Any]],
    fill_regions: Sequence[Mapping[str, Any]],
    excluded_fill_ids: set[str],
    topology: Mapping[str, Any],
    observation_to_point: Mapping[str, str],
    points: Sequence[ParseObject],
    line_evidence: Mapping[str, str],
    arc_evidence: Mapping[str, str],
    arc_object_ids: Mapping[str, str],
) -> tuple[list[ParseObject], list[ParseRegion], dict[str, dict[str, Any]]]:
    boundary_candidates = []
    for arc in arcs:
        arc_id = str(arc["id"])
        if arc_id not in arc_object_ids:
            continue
        start_id = observation_to_point.get(f"{arc_id}:arc_endpoint:start")
        end_id = observation_to_point.get(f"{arc_id}:arc_endpoint:end")
        center = float(arc["center"][0]), float(arc["center"][1])
        center_id = _unique_point_near(points, center, max(4.0, 0.04 * float(arc["radius_px"])))
        if start_id is None or end_id is None or center_id is None or len({start_id, end_id, center_id}) != 3:
            continue
        end_supports = _radial_supports(
            lines,
            topology,
            observation_to_point,
            end_id,
            center_id,
            arc["end"],
            center,
        )
        start_supports = _radial_supports(
            lines,
            topology,
            observation_to_point,
            center_id,
            start_id,
            center,
            arc["start"],
        )
        for end_support in end_supports:
            for start_support in start_supports:
                if end_support["line_id"] == start_support["line_id"]:
                    continue
                boundary_candidates.append(
                    {
                        "key": (arc_id, end_support["line_id"], start_support["line_id"]),
                        "arc_id": arc_id,
                        "arc": arc,
                        "center_id": center_id,
                        "polygon": (*_visual_arc_points(arc), center),
                        "supports": (end_support, start_support),
                    }
                )

    selected = _select_unique_fill_boundaries(fill_regions, excluded_fill_ids, boundary_candidates)

    objects = []
    regions = []
    bindings: dict[str, dict[str, Any]] = {}
    for score, fill_id, fill, boundary in sorted(selected, key=lambda item: item[1]):
        arc_id = boundary["arc_id"]
        confidence = min(
            float(fill.get("confidence", 0.6)),
            float(boundary["arc"].get("edge_support", 0.6)),
            0.9,
        )
        radius_ids = []
        radius_boundaries = []
        for role, support in zip(("end_radius", "start_radius"), boundary["supports"]):
            segment_id = (
                f"segment:{support['line_id']}"
                if support["reuse_support"]
                else f"segment:curved-boundary:{arc_id}:{support['line_id']}:{role}"
            )
            radius_ids.append(segment_id)
            radius_boundaries.append(BoundaryReference(segment_id, orientation=support["orientation"]))
            if support["reuse_support"]:
                continue
            objects.append(
                ParseObject(
                    id=segment_id,
                    kind=ParseObjectKind.SEGMENT,
                    origin=ParseOrigin.COMPILED,
                    confidence=confidence,
                    score_kind="arc_center_radius_support",
                    references=(
                        ParseReference("start", support["start_id"]),
                        ParseReference("end", support["end_id"]),
                    ),
                    evidence_refs=(line_evidence[support["line_id"]], arc_evidence[arc_id]),
                    visible=True,
                    geometry={
                        "p1": list(support["p1"]),
                        "p2": list(support["p2"]),
                        "angle_deg": _line_angle(support["p1"], support["p2"]),
                    },
                    metadata={
                        "source_support_id": f"segment:{support['line_id']}",
                        "boundary_role": role,
                    },
                )
            )
        region_id = f"region:curved-sector:{arc_id}:{boundary['supports'][0]['line_id']}:{boundary['supports'][1]['line_id']}"
        regions.append(
            ParseRegion(
                id=region_id,
                boundary=(BoundaryReference(arc_object_ids[arc_id]), *radius_boundaries),
                origin=ParseOrigin.IMAGE,
                confidence=confidence,
                score_kind="filled_arc_radii_region_match",
                evidence_refs=(
                    f"evidence:{fill_id}",
                    arc_evidence[arc_id],
                    *(line_evidence[support["line_id"]] for support in boundary["supports"]),
                ),
                metadata={
                    "kind": "observed_curved_region",
                    "boundary_kind": "arc_radii",
                    "center_point_id": boundary["center_id"],
                    "radius_segment_ids": radius_ids,
                    "fill_status": "observed",
                    "shading_status": "unknown",
                    "fill_candidate_id": fill_id,
                    "fill_match_score": round(score, 3),
                    **_interior_gap_status(fill),
                },
            )
        )
        bindings[fill_id] = {
            "bound_curved_region_id": region_id,
            "bound_curve_id": arc_object_ids[arc_id],
            "bound_radius_segment_ids": radius_ids,
            "curved_boundary_match_score": round(score, 3),
        }
    return objects, regions, bindings


def _select_unique_fill_boundaries(
    fill_regions: Sequence[Mapping[str, Any]],
    excluded_fill_ids: set[str],
    boundary_candidates: Sequence[Mapping[str, Any]],
) -> list[tuple[float, str, Mapping[str, Any], Mapping[str, Any]]]:
    proposals = []
    for fill in fill_regions:
        fill_id = str(fill["id"])
        if fill_id in excluded_fill_ids:
            continue
        contour = _pixel_polygon(fill["outer_contour"], f"fill-region {fill_id} outer contour")
        scored = sorted(
            (
                (score, candidate)
                for candidate in boundary_candidates
                if (score := _fill_face_match_score(candidate["polygon"], contour, fill)) is not None
            ),
            key=lambda item: (item[0], item[1]["key"]),
            reverse=True,
        )
        if not scored or scored[0][0] < 0.65:
            continue
        if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.08:
            continue
        proposals.append((scored[0][0], fill_id, fill, scored[0][1]))

    selected = []
    used_boundaries = set()
    for proposal in sorted(proposals, key=lambda item: (item[0], item[1], item[3]["key"]), reverse=True):
        boundary_key = proposal[3]["key"]
        if boundary_key in used_boundaries:
            continue
        used_boundaries.add(boundary_key)
        selected.append(proposal)
    return selected


def _radial_supports(
    lines: Sequence[Mapping[str, Any]],
    topology: Mapping[str, Any],
    observation_to_point: Mapping[str, str],
    start_id: str,
    end_id: str,
    start: Sequence[Any],
    end: Sequence[Any],
) -> list[dict[str, Any]]:
    tolerance = max(4.0, 0.04 * math.dist(start, end))
    candidates = []
    for line in lines:
        if not _line_supports_points(line, start, end, tolerance):
            continue
        ordered = _ordered_line_points(line, topology, observation_to_point)
        if start_id not in ordered or end_id not in ordered:
            continue
        reuse_support = len(ordered) == 2 and set(ordered) == {start_id, end_id}
        orientation = (
            "forward"
            if reuse_support and ordered == (start_id, end_id)
            else "reverse"
            if reuse_support
            else "forward"
        )
        candidates.append(
            {
                "line_id": str(line["id"]),
                "start_id": start_id,
                "end_id": end_id,
                "p1": tuple(float(value) for value in start),
                "p2": tuple(float(value) for value in end),
                "reuse_support": reuse_support,
                "orientation": orientation,
            }
        )
    return candidates


def _line_supports_points(
    line: Mapping[str, Any],
    first: Sequence[Any],
    second: Sequence[Any],
    tolerance: float,
) -> bool:
    line_start = float(line["p1"][0]), float(line["p1"][1])
    line_end = float(line["p2"][0]), float(line["p2"][1])
    return all(
        _point_segment_distance((float(point[0]), float(point[1])), line_start, line_end) <= tolerance
        for point in (first, second)
    )


def _unique_point_near(
    points: Sequence[ParseObject],
    target: tuple[float, float],
    tolerance: float,
) -> str | None:
    ranked = sorted(
        (
            (math.dist(target, (float(point.geometry["pixel"][0]), float(point.geometry["pixel"][1]))), point.id)
            for point in points
            if point.kind is ParseObjectKind.POINT and point.geometry.get("pixel") is not None
        ),
        key=lambda item: (item[0], item[1]),
    )
    if not ranked or ranked[0][0] > tolerance:
        return None
    if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < max(1.5, 0.25 * tolerance):
        return None
    return ranked[0][1]


def _promote_curved_fill_evidence(
    evidence: list[ParseEvidence],
    bindings: Mapping[str, Mapping[str, Any]],
) -> None:
    if not bindings:
        return
    for index, candidate in enumerate(evidence):
        fill_id = candidate.id.removeprefix("evidence:")
        binding = bindings.get(fill_id)
        if binding is None:
            continue
        evidence[index] = replace(
            candidate,
            precision=EvidencePrecision.BOUND,
            metadata={**dict(candidate.metadata), **dict(binding)},
        )


def _line_supports_arc_chord(
    line: Mapping[str, Any],
    arc: Mapping[str, Any],
) -> bool:
    radius = float(arc["radius_px"])
    tolerance = max(4.0, 0.04 * radius)
    return all(
        _point_segment_distance(
            (float(point[0]), float(point[1])),
            (float(line["p1"][0]), float(line["p1"][1])),
            (float(line["p2"][0]), float(line["p2"][1])),
        )
        <= tolerance
        for point in (arc["start"], arc["end"])
    )


def _visual_arc_points(arc: Mapping[str, Any]) -> tuple[tuple[float, float], ...]:
    center = float(arc["center"][0]), float(arc["center"][1])
    start = float(arc["start"][0]), float(arc["start"][1])
    radius = float(arc["radius_px"])
    sweep = math.radians(float(arc["sweep_deg"]))
    direction = -1.0 if str(arc.get("orientation") or "ccw") == "cw" else 1.0
    start_angle = math.atan2(start[1] - center[1], start[0] - center[0])
    steps = min(128, max(12, int(math.ceil(radius * sweep / 2.0))))
    return tuple(
        (
            center[0] + radius * math.cos(start_angle + direction * sweep * index / steps),
            center[1] + radius * math.sin(start_angle + direction * sweep * index / steps),
        )
        for index in range(steps + 1)
    )


def _line_angle(start: Sequence[Any], end: Sequence[Any]) -> float:
    return round(
        math.degrees(math.atan2(float(end[1]) - float(start[1]), float(end[0]) - float(start[0]))),
        3,
    )


def _cartesian_arc_orientation(arc: Mapping[str, Any]) -> str:
    orientation = str(arc.get("orientation") or "ccw")
    if orientation not in {"ccw", "cw"}:
        raise InvalidGeometryInput(f"Visual arc {arc.get('id')} has unsupported orientation: {orientation!r}")
    return "cw" if orientation == "ccw" else "ccw"


def _planar_face_candidates(
    lines: Sequence[Mapping[str, Any]],
    topology: Mapping[str, Any],
    observation_to_point: Mapping[str, str],
    points: Sequence[ParseObject],
    line_evidence: Mapping[str, str],
    fill_regions: Sequence[Mapping[str, Any]],
    source_id: str,
    *,
    minimum_area: float,
    maximum_vertices: int,
) -> tuple[list[ParseEvidence], list[ParseObject], list[ParseRegion]]:
    positions = {
        point.id: (float(point.geometry["pixel"][0]), float(point.geometry["pixel"][1]))
        for point in points
    }
    adjacency: dict[str, set[str]] = {}
    edge_evidence: dict[frozenset[str], set[str]] = {}
    for line in lines:
        line_id = str(line["id"])
        ordered = _ordered_line_points(line, topology, observation_to_point)
        for start, end in zip(ordered, ordered[1:]):
            if start == end:
                continue
            adjacency.setdefault(start, set()).add(end)
            adjacency.setdefault(end, set()).add(start)
            edge_evidence.setdefault(frozenset((start, end)), set()).add(line_evidence[line_id])

    cycles = _bounded_face_cycles(adjacency, positions, minimum_area, maximum_vertices)
    fill_assignments = _match_fill_regions_to_faces(cycles, positions, fill_regions)
    fill_to_face = {
        str(candidate["id"]): face_index
        for face_index, (candidate, _) in fill_assignments.items()
    }
    hole_assignments = _match_fill_gaps_to_faces(cycles, positions, fill_assignments)
    fill_hole_regions = {
        str(candidate["id"]): tuple(
            f"region:face:{face_index:03d}" for face_index in hole_faces
        )
        for outer_index, (candidate, _) in fill_assignments.items()
        if (hole_faces := hole_assignments.get(outer_index))
    }
    evidence = _fill_region_evidence(fill_regions, fill_to_face, source_id, fill_hole_regions)
    polygons = []
    regions = []
    for index, cycle in enumerate(cycles):
        polygon_id = f"polygon:face:{index:03d}"
        region_id = f"region:face:{index:03d}"
        evidence_id = f"evidence:face:{index:03d}"
        face_points = [positions[point_id] for point_id in cycle]
        edge_supporting_evidence = sorted(
            {
                edge_evidence_id
                for start, end in zip(cycle, (*cycle[1:], cycle[0]))
                for edge_evidence_id in edge_evidence.get(frozenset((start, end)), ())
            }
        )
        fill_match = fill_assignments.get(index)
        fill_evidence_id = f"evidence:{fill_match[0]['id']}" if fill_match is not None else None
        hole_region_ids = tuple(
            f"region:face:{face_index:03d}" for face_index in hole_assignments.get(index, ())
        )
        supporting_evidence = [
            *edge_supporting_evidence,
            *((fill_evidence_id,) if fill_evidence_id is not None else ()),
        ]
        xs = [point[0] for point in face_points]
        ys = [point[1] for point in face_points]
        evidence.append(
            ParseEvidence(
                id=evidence_id,
                source_id=source_id,
                kind=ParseEvidenceKind.IMAGE_REGION,
                precision=EvidencePrecision.BOUND,
                bbox=(min(xs), min(ys), max(xs), max(ys)),
                metadata={
                    "method": "planar_half_edge_walk",
                    "supporting_evidence": supporting_evidence,
                },
            )
        )
        polygons.append(
            ParseObject(
                id=polygon_id,
                kind=ParseObjectKind.POLYGON,
                origin=ParseOrigin.COMPILED,
                confidence=0.7,
                score_kind="planar_face_cycle",
                references=tuple(
                    ParseReference(f"vertex_{vertex_index}", point_id)
                    for vertex_index, point_id in enumerate(cycle)
                ),
                evidence_refs=(evidence_id,),
                visible=True,
                metadata={"topology_only": True, "supporting_evidence": edge_supporting_evidence},
            )
        )
        fill_metadata = (
            {
                "fill_status": "observed",
                "fill_candidate_id": str(fill_match[0]["id"]),
                "fill_match_score": round(fill_match[1], 3),
                **_interior_gap_status(fill_match[0], hole_region_ids=hole_region_ids),
            }
            if fill_match is not None
            else {"fill_status": "unknown", "interior_gap_status": "not_bound", "hole_region_ids": []}
        )
        regions.append(
            ParseRegion(
                id=region_id,
                boundary=(BoundaryReference(polygon_id),),
                origin=ParseOrigin.COMPILED,
                confidence=0.7,
                score_kind="planar_face_cycle_with_observed_fill" if fill_match is not None else "planar_face_cycle",
                holes=hole_region_ids,
                evidence_refs=(evidence_id, *((fill_evidence_id,) if fill_evidence_id is not None else ())),
                metadata={
                    "kind": "polygon_region",
                    "shading_status": "unknown",
                    "region_role": "hole_boundary" if any(
                        index in hole_faces for hole_faces in hole_assignments.values()
                    ) else "face",
                    **fill_metadata,
                },
            )
        )
    return evidence, polygons, regions


def _fill_region_evidence(
    fill_regions: Sequence[Mapping[str, Any]],
    fill_to_face: Mapping[str, int],
    source_id: str,
    fill_hole_regions: Mapping[str, Sequence[str]] | None = None,
) -> list[ParseEvidence]:
    evidence = []
    for candidate in fill_regions:
        candidate_id = str(candidate["id"])
        bbox = tuple(candidate.get("bbox") or ())
        face_index = fill_to_face.get(candidate_id)
        hole_region_ids = tuple((fill_hole_regions or {}).get(candidate_id, ()))
        interior_gaps = tuple(candidate.get("interior_gaps") or ())
        evidence.append(
            ParseEvidence(
                id=f"evidence:{candidate_id}",
                source_id=source_id,
                kind=ParseEvidenceKind.IMAGE_REGION,
                precision=EvidencePrecision.BOUND if face_index is not None else EvidencePrecision.LOCALIZED,
                bbox=_component_bbox(bbox),
                metadata={
                    "method": candidate.get("score_kind", "visual_fill_component"),
                    "appearance_semantics": "observed_fill_not_shading_label",
                    "outer_contour": [list(point) for point in candidate.get("outer_contour") or ()],
                    "interior_gaps": [
                        [list(point) for point in gap]
                        for gap in interior_gaps
                    ],
                    **_interior_gap_status(candidate, hole_region_ids=hole_region_ids),
                    "area_px": candidate.get("area_px"),
                    "mean_rgb": list(candidate.get("mean_rgb") or ()),
                    "bound_face_id": f"region:face:{face_index:03d}" if face_index is not None else None,
                },
            )
        )
    return evidence


def _interior_gap_status(
    candidate: Mapping[str, Any],
    *,
    hole_region_ids: Sequence[str] = (),
) -> dict[str, Any]:
    interior_gaps = tuple(candidate.get("interior_gaps") or ())
    hole_region_ids = tuple(hole_region_ids)
    if not interior_gaps:
        status = "none_observed"
        semantics = "not_applicable"
    elif len(hole_region_ids) == len(interior_gaps):
        status = "grounded_closed_boundary"
        semantics = "grounded_closed_parse_region_boundary"
    elif hole_region_ids:
        status = "partially_grounded"
        semantics = "partially_grounded_closed_parse_region_boundary"
    else:
        status = "unresolved_boundary_candidate"
        semantics = "not_admitted_without_closed_parse_region_boundary"
    return {
        "interior_gap_count": len(interior_gaps),
        "interior_gap_status": status,
        "hole_region_ids": list(hole_region_ids),
        "hole_semantics": semantics,
    }


def _match_fill_gaps_to_faces(
    cycles: Sequence[Sequence[str]],
    positions: Mapping[str, tuple[float, float]],
    fill_assignments: Mapping[int, tuple[Mapping[str, Any], float]],
) -> dict[int, tuple[int, ...]]:
    proposals = []
    for outer_index, (candidate, _) in fill_assignments.items():
        outer_face = tuple(positions[point_id] for point_id in cycles[outer_index])
        for gap_index, gap in enumerate(candidate.get("interior_gaps") or ()):
            contour = _pixel_polygon(gap, f"fill-region {candidate['id']} interior gap {gap_index}")
            scored_faces = []
            for face_index, cycle in enumerate(cycles):
                if face_index == outer_index:
                    continue
                face = tuple(positions[point_id] for point_id in cycle)
                if not _polygon_contains(face, outer_face):
                    continue
                score = _fill_face_match_score(
                    face,
                    contour,
                    {"area_px": abs(_signed_polygon_area(contour))},
                )
                if score is not None:
                    scored_faces.append((score, face_index))
            scored_faces.sort(reverse=True)
            if not scored_faces or scored_faces[0][0] < 0.72:
                continue
            if len(scored_faces) > 1 and scored_faces[0][0] - scored_faces[1][0] < 0.08:
                continue
            proposals.append((scored_faces[0][0], outer_index, gap_index, scored_faces[0][1]))

    grouped = {}
    for proposal in proposals:
        grouped.setdefault((proposal[1], proposal[2]), []).append(proposal)
    unique_proposals = [items[0] for items in grouped.values() if len(items) == 1]
    target_groups = {}
    for proposal in unique_proposals:
        target_groups.setdefault(proposal[3], []).append(proposal)
    unique_targets = {
        proposal[3]
        for proposals_for_target in target_groups.values()
        if len(proposals_for_target) == 1
        for proposal in proposals_for_target
    }
    assignments: dict[int, list[int]] = {}
    for _, outer_index, _, face_index in sorted(unique_proposals, reverse=True):
        if face_index not in unique_targets:
            continue
        assignments.setdefault(outer_index, []).append(face_index)
    return {
        outer_index: tuple(sorted(face_indices))
        for outer_index, face_indices in assignments.items()
    }


def _polygon_contains(
    inner: Sequence[tuple[float, float]],
    outer: Sequence[tuple[float, float]],
) -> bool:
    return all(_point_in_polygon(point, outer) for point in _evenly_spaced_points(inner, 32))


def _validate_fill_region_candidates(
    fill_regions: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    seen_ids = set()
    for candidate in fill_regions:
        if not isinstance(candidate, Mapping):
            raise InvalidGeometryInput("Visual fill-region candidates must be mappings")
        candidate_id = str(candidate.get("id") or "")
        if not candidate_id:
            raise InvalidGeometryInput("Visual fill-region candidate is missing an id")
        if candidate_id in seen_ids:
            raise InvalidGeometryInput(f"Duplicate visual fill-region candidate id: {candidate_id}")
        seen_ids.add(candidate_id)
        bbox = tuple(candidate.get("bbox") or ())
        if len(bbox) != 4:
            raise InvalidGeometryInput(f"Visual fill-region candidate {candidate_id} requires a four-value bbox")
        try:
            x, y, width, height = (float(value) for value in bbox)
        except (TypeError, ValueError) as exc:
            raise InvalidGeometryInput(f"Visual fill-region candidate {candidate_id} has a non-numeric bbox") from exc
        if not all(math.isfinite(value) for value in (x, y, width, height)) or width <= 0.0 or height <= 0.0:
            raise InvalidGeometryInput(f"Visual fill-region candidate {candidate_id} has an invalid bbox")
        _pixel_polygon(candidate.get("outer_contour") or (), f"fill-region {candidate_id} outer contour")
        try:
            area = float(candidate.get("area_px"))
        except (TypeError, ValueError) as exc:
            raise InvalidGeometryInput(f"Visual fill-region candidate {candidate_id} requires numeric area_px") from exc
        if not math.isfinite(area) or area <= 0.0:
            raise InvalidGeometryInput(f"Visual fill-region candidate {candidate_id} requires positive area_px")
        for gap_index, gap in enumerate(candidate.get("interior_gaps") or ()):
            _pixel_polygon(gap, f"fill-region {candidate_id} interior gap {gap_index}")
    return tuple(fill_regions)


def _match_fill_regions_to_faces(
    cycles: Sequence[Sequence[str]],
    positions: Mapping[str, tuple[float, float]],
    fill_regions: Sequence[Mapping[str, Any]],
) -> dict[int, tuple[Mapping[str, Any], float]]:
    proposals = []
    for candidate in fill_regions:
        contour = _pixel_polygon(candidate["outer_contour"], f"fill-region {candidate['id']} outer contour")
        scored_faces = sorted(
            (
                (score, face_index)
                for face_index, cycle in enumerate(cycles)
                if (score := _fill_face_match_score([positions[point_id] for point_id in cycle], contour, candidate))
                is not None
            ),
            reverse=True,
        )
        if not scored_faces or scored_faces[0][0] < 0.65:
            continue
        if len(scored_faces) > 1 and scored_faces[0][0] - scored_faces[1][0] < 0.08:
            continue
        proposals.append((scored_faces[0][0], str(candidate.get("id") or ""), scored_faces[0][1], candidate))
    assignments = {}
    for score, _, face_index, candidate in sorted(
        proposals,
        key=lambda item: (item[0], item[1], item[2]),
        reverse=True,
    ):
        if face_index not in assignments:
            assignments[face_index] = (candidate, score)
    return assignments


def _fill_face_match_score(
    face: Sequence[tuple[float, float]],
    contour: Sequence[tuple[float, float]],
    candidate: Mapping[str, Any],
) -> float | None:
    face_area = abs(_signed_polygon_area(face))
    fill_area = float(candidate.get("area_px") or abs(_signed_polygon_area(contour)))
    if face_area <= 1e-9 or fill_area <= 1e-9:
        return None
    area_ratio = fill_area / face_area
    if not 0.35 <= area_ratio <= 1.15:
        return None
    contour_samples = _evenly_spaced_points(contour, 32)
    contour_inside = sum(_point_in_polygon(point, face) for point in contour_samples) / len(contour_samples)
    centroid = _polygon_centroid(face)
    face_samples = [centroid] + [
        ((centroid[0] + point[0]) / 2.0, (centroid[1] + point[1]) / 2.0)
        for point in face
    ]
    face_inside = sum(_point_in_polygon(point, contour) for point in face_samples) / len(face_samples)
    if contour_inside < 0.75 or face_inside < 0.5:
        return None
    area_similarity = min(area_ratio, 1.0 / area_ratio)
    return 0.35 * contour_inside + 0.35 * face_inside + 0.3 * area_similarity


def _pixel_polygon(
    points: Sequence[Sequence[Any]],
    label: str,
) -> tuple[tuple[float, float], ...]:
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)) or len(points) < 3:
        raise InvalidGeometryInput(f"{label} requires at least three points")
    polygon = []
    for index, point in enumerate(points):
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) < 2:
            raise InvalidGeometryInput(f"{label} point {index} must contain two coordinates")
        try:
            pixel = float(point[0]), float(point[1])
        except (TypeError, ValueError) as exc:
            raise InvalidGeometryInput(f"{label} point {index} has non-numeric coordinates") from exc
        if not all(math.isfinite(value) for value in pixel):
            raise InvalidGeometryInput(f"{label} point {index} has non-finite coordinates")
        polygon.append(pixel)
    return tuple(polygon)


def _evenly_spaced_points(
    points: Sequence[tuple[float, float]],
    limit: int,
) -> tuple[tuple[float, float], ...]:
    if len(points) <= limit:
        return tuple(points)
    return tuple(points[index * len(points) // limit] for index in range(limit))


def _signed_polygon_area(points: Sequence[tuple[float, float]]) -> float:
    return 0.5 * sum(
        left[0] * right[1] - right[0] * left[1]
        for left, right in zip(points, (*points[1:], points[0]))
    )


def _polygon_centroid(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    cross_products = [
        left[0] * right[1] - right[0] * left[1]
        for left, right in zip(points, (*points[1:], points[0]))
    ]
    cross_sum = math.fsum(cross_products)
    if abs(cross_sum) <= 1e-9:
        return (
            math.fsum(point[0] for point in points) / len(points),
            math.fsum(point[1] for point in points) / len(points),
        )
    return (
        math.fsum(
            (left[0] + right[0]) * cross
            for left, right, cross in zip(points, (*points[1:], points[0]), cross_products)
        )
        / (3.0 * cross_sum),
        math.fsum(
            (left[1] + right[1]) * cross
            for left, right, cross in zip(points, (*points[1:], points[0]), cross_products)
        )
        / (3.0 * cross_sum),
    )


def _point_in_polygon(
    point: tuple[float, float],
    polygon: Sequence[tuple[float, float]],
    tolerance: float = 1.5,
) -> bool:
    edges = zip(polygon, (*polygon[1:], polygon[0]))
    if any(_point_segment_distance(point, start, end) <= tolerance for start, end in edges):
        return True
    x, y = point
    inside = False
    for start, end in zip(polygon, (*polygon[1:], polygon[0])):
        if (start[1] > y) == (end[1] > y):
            continue
        crossing_x = start[0] + (y - start[1]) * (end[0] - start[0]) / (end[1] - start[1])
        if crossing_x >= x:
            inside = not inside
    return inside


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-12:
        return math.dist(point, start)
    parameter = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared
    parameter = max(0.0, min(1.0, parameter))
    projection = start[0] + parameter * dx, start[1] + parameter * dy
    return math.dist(point, projection)


def _bounded_face_cycles(
    adjacency: Mapping[str, set[str]],
    positions: Mapping[str, tuple[float, float]],
    minimum_area: float,
    maximum_vertices: int,
) -> tuple[tuple[str, ...], ...]:
    neighbor_order = {
        point_id: tuple(
            sorted(
                neighbors,
                key=lambda neighbor: math.atan2(
                    positions[neighbor][1] - positions[point_id][1],
                    positions[neighbor][0] - positions[point_id][0],
                ),
            )
        )
        for point_id, neighbors in adjacency.items()
    }
    visited: set[tuple[str, str]] = set()
    cycles = set()
    max_steps = max(1, sum(len(neighbors) for neighbors in adjacency.values()))
    for start in sorted((left, right) for left, neighbors in adjacency.items() for right in neighbors):
        if start in visited:
            continue
        directed = start
        path = []
        local = set()
        closed = False
        for _ in range(max_steps + 1):
            if directed in local:
                closed = directed == start
                break
            if directed in visited:
                break
            local.add(directed)
            visited.add(directed)
            left, right = directed
            path.append(left)
            neighbors = neighbor_order.get(right, ())
            if left not in neighbors or len(neighbors) < 2:
                break
            incoming_index = neighbors.index(left)
            directed = right, neighbors[(incoming_index - 1) % len(neighbors)]
        if not closed or len(path) < 3 or len(path) > maximum_vertices or len(set(path)) != len(path):
            continue
        area = _signed_pixel_area(path, positions)
        if area < minimum_area:
            continue
        cycles.add(_canonical_cycle(path))
    return tuple(sorted(cycles))


def _signed_pixel_area(cycle: Sequence[str], positions: Mapping[str, tuple[float, float]]) -> float:
    return 0.5 * sum(
        positions[left][0] * positions[right][1] - positions[right][0] * positions[left][1]
        for left, right in zip(cycle, (*cycle[1:], cycle[0]))
    )


def _canonical_cycle(cycle: Sequence[str]) -> tuple[str, ...]:
    values = tuple(cycle)
    rotations = tuple(values[index:] + values[:index] for index in range(len(values)))
    return min(rotations)


def _marker_candidates(
    markers: Sequence[Mapping[str, Any]],
    source_id: str,
    lines: Sequence[Mapping[str, Any]],
    topology: Mapping[str, Any],
    observation_to_point: Mapping[str, str],
    line_evidence: Mapping[str, str],
    evidence: list[ParseEvidence],
    objects: list[ParseObject],
) -> tuple[ParseRelation, ...]:
    relations = []
    marker_objects: dict[str, ParseObject] = {}
    marker_evidence: dict[str, str] = {}
    marker_subsegments: dict[str, ParseObject] = {}
    grouped_markers: dict[str, list[Mapping[str, Any]]] = {}
    for marker in markers:
        marker_id = str(marker.get("id", ""))
        kind = str(marker.get("kind", ""))
        line_ids = tuple(str(value) for value in marker.get("line_ids") or ())
        if not marker_id:
            raise InvalidGeometryInput("Visual marker candidate is missing an id")
        if kind not in {"right_angle", "parallel", "equal_length_tick", "angle_arc"}:
            raise InvalidGeometryInput(f"Unsupported visual marker kind: {kind!r}")
        if kind == "right_angle" and (len(line_ids) != 2 or len(set(line_ids)) != 2):
            raise InvalidGeometryInput(f"Right-angle marker {marker_id} requires two distinct support lines")
        if kind in {"parallel", "equal_length_tick"} and len(line_ids) != 1:
            raise InvalidGeometryInput(f"Visual marker {marker_id} of kind {kind!r} requires one support line")
        if kind == "angle_arc" and (len(line_ids) != 2 or len(set(line_ids)) != 2):
            raise InvalidGeometryInput(f"Angle-arc marker {marker_id} requires two distinct support lines")
        missing = tuple(line_id for line_id in line_ids if line_id not in line_evidence)
        if missing:
            raise InvalidGeometryInput(f"Visual marker {marker_id} references unknown support lines: {missing}")
        bbox = tuple(float(value) for value in marker.get("bbox") or ())
        if len(bbox) != 4:
            raise InvalidGeometryInput(f"Visual marker {marker_id} requires a four-value bbox")
        confidence = float(marker.get("confidence", 0.0))
        evidence_id = f"evidence:{marker_id}"
        object_id = f"marker:{marker_id}"
        ordered_angle = (
            _ground_marker_angle(marker, lines, topology, observation_to_point, objects)
            if kind == "angle_arc"
            else None
        )
        marked_segment_id, marked_subsegment = (
            _ground_equal_length_marker(marker, lines, topology, observation_to_point, objects, line_evidence)
            if kind == "equal_length_tick"
            else (None, None)
        )
        if marked_subsegment is not None and marked_subsegment.id not in marker_subsegments:
            marker_subsegments[marked_subsegment.id] = marked_subsegment
            objects.append(marked_subsegment)
        evidence.append(
            ParseEvidence(
                id=evidence_id,
                source_id=source_id,
                kind=ParseEvidenceKind.IMAGE_REGION,
                precision=EvidencePrecision.BOUND,
                bbox=bbox,
                raw_content=kind,
                metadata={
                    "marker_kind": kind,
                    "support_line_ids": list(line_ids),
                    "stroke_support": list(marker.get("stroke_support") or ()),
                    "relation_group_id": marker.get("relation_group_id"),
                },
            )
        )
        support_references = tuple(ParseReference("support", f"segment:{line_id}") for line_id in line_ids)
        angle_references = (
            (
                ParseReference("arm_0", ordered_angle[0]),
                ParseReference("vertex", ordered_angle[1]),
                ParseReference("arm_1", ordered_angle[2]),
            )
            if ordered_angle is not None
            else ()
        )
        marked_segment_references = (
            (ParseReference("marked_segment", marked_segment_id),)
            if marked_segment_id is not None
            else ()
        )
        marker_object = ParseObject(
            id=object_id,
            kind=ParseObjectKind.MARKER,
            origin=ParseOrigin.IMAGE,
            confidence=confidence,
            score_kind=str(marker.get("score_kind") or "visual_marker_score"),
            references=(*support_references, *angle_references, *marked_segment_references),
            evidence_refs=(evidence_id,),
            visible=True,
            geometry={"bbox": list(bbox)},
            metadata={
                "marker_kind": kind,
                "vertex": list(marker.get("vertex") or ()),
                "center": list(marker.get("center") or ()),
                "path": [list(point) for point in marker.get("path") or ()],
                "size_px": marker.get("size_px"),
                "relation_group_id": marker.get("relation_group_id"),
                "angle_span_deg": marker.get("angle_span_deg"),
                "support_line_ids": list(line_ids),
                "ray_signs": list(marker.get("ray_signs") or ()),
                "ordered_angle": list(ordered_angle) if ordered_angle is not None else None,
                "angle_grounding_status": "resolved" if ordered_angle is not None else "unresolved",
                "marked_segment_id": marked_segment_id,
                "segment_grounding_status": (
                    "resolved" if marked_segment_id is not None else "unresolved"
                ),
            },
        )
        objects.append(marker_object)
        marker_objects[marker_id] = marker_object
        marker_evidence[marker_id] = evidence_id
        group_id = marker.get("relation_group_id")
        if isinstance(group_id, str) and group_id:
            grouped_markers.setdefault(group_id, []).append(marker)
        if kind == "right_angle":
            relations.append(
                ParseRelation(
                    id=f"relation:marker:{marker_id}",
                    predicate="perpendicular",
                    arguments=tuple(
                        ParseReference(role, f"segment:{line_id}")
                        for role, line_id in zip(("left", "right"), line_ids)
                    ),
                    origin=ParseOrigin.IMAGE,
                    confidence=confidence,
                    score_kind="explicit_right_angle_marker",
                    evidence_refs=(evidence_id,),
                    dependencies=(object_id,),
                    metadata={
                        "marker_id": object_id,
                        "angle_residual_deg": marker.get("angle_residual_deg"),
                        "semantic_source": "explicit_visual_glyph",
                    },
                )
            )
    for group_id, group in sorted(grouped_markers.items()):
        relation = _grouped_marker_relation(group_id, group, marker_objects, marker_evidence)
        if relation is not None:
            relations.append(relation)
            angle_bisector = _shared_ray_angle_bisector_relation(
                group_id,
                group,
                marker_objects,
                relation,
                lines,
            )
            if angle_bisector is not None:
                relations.append(angle_bisector)
    return tuple(relations)


def _grouped_marker_relation(
    group_id: str,
    markers: Sequence[Mapping[str, Any]],
    marker_objects: Mapping[str, ParseObject],
    marker_evidence: Mapping[str, str],
) -> ParseRelation | None:
    if len(markers) != 2:
        return None
    semantic_relations = {str(marker.get("semantic_relation") or "") for marker in markers}
    if len(semantic_relations) != 1:
        raise InvalidGeometryInput(f"Visual marker group {group_id} has inconsistent semantic relations")
    predicate = next(iter(semantic_relations))
    marker_kinds = {str(marker.get("kind") or "") for marker in markers}
    expected_kind = {
        "parallel": "parallel",
        "equal_length": "equal_length_tick",
        "equal_angle": "angle_arc",
    }.get(predicate)
    if expected_kind is None:
        return None
    if marker_kinds != {expected_kind}:
        raise InvalidGeometryInput(
            f"Visual marker group {group_id} cannot express {predicate!r} with marker kinds {sorted(marker_kinds)}"
        )
    marker_ids = tuple(str(marker["id"]) for marker in markers)
    if any(marker_id not in marker_objects or marker_id not in marker_evidence for marker_id in marker_ids):
        raise InvalidGeometryInput(f"Visual marker group {group_id} references an unknown marker")
    confidence = min(marker_objects[marker_id].confidence for marker_id in marker_ids)
    evidence_refs = tuple(marker_evidence[marker_id] for marker_id in marker_ids)
    object_ids = tuple(marker_objects[marker_id].id for marker_id in marker_ids)
    if predicate == "equal_angle":
        angles = tuple(marker_objects[marker_id].metadata.get("ordered_angle") for marker_id in marker_ids)
        if any(not isinstance(angle, list) or len(angle) != 3 for angle in angles):
            return None
        arguments = tuple(
            ParseReference(role, point_id)
            for angle_index, angle in enumerate(angles)
            for role, point_id in zip(
                (f"angle_{angle_index}_arm_0", f"angle_{angle_index}_vertex", f"angle_{angle_index}_arm_1"),
                angle,
            )
        )
    elif predicate == "equal_length":
        marked_segment_ids = tuple(
            marker_objects[marker_id].metadata.get("marked_segment_id")
            for marker_id in marker_ids
        )
        if any(not isinstance(segment_id, str) or not segment_id for segment_id in marked_segment_ids):
            return None
        if len(set(marked_segment_ids)) != 2:
            return None
        arguments = tuple(
            ParseReference(role, segment_id)
            for role, segment_id in zip(("left", "right"), marked_segment_ids)
        )
    else:
        line_ids = tuple(
            dict.fromkeys(
                str(line_id)
                for marker in markers
                for line_id in marker.get("line_ids") or ()
            )
        )
        if len(line_ids) != 2:
            return None
        arguments = tuple(
            ParseReference(role, f"segment:{line_id}")
            for role, line_id in zip(("left", "right"), line_ids)
        )
    return ParseRelation(
        id=f"relation:marker-group:{group_id}",
        predicate=predicate,
        arguments=arguments,
        origin=ParseOrigin.IMAGE,
        confidence=confidence,
        score_kind=f"explicit_{predicate}_marker_group",
        evidence_refs=evidence_refs,
        dependencies=object_ids,
        metadata={
            "marker_ids": list(object_ids),
            "relation_group_id": group_id,
            "semantic_source": "explicit_visual_glyph_group",
        },
    )


def _shared_ray_angle_bisector_relation(
    group_id: str,
    markers: Sequence[Mapping[str, Any]],
    marker_objects: Mapping[str, ParseObject],
    equal_angle: ParseRelation,
    lines: Sequence[Mapping[str, Any]],
) -> ParseRelation | None:
    if equal_angle.predicate != "equal_angle" or len(markers) != 2:
        return None
    marker_ids = tuple(str(marker["id"]) for marker in markers)
    angle_rays = []
    vertices = []
    for marker_id in marker_ids:
        marker_object = marker_objects[marker_id]
        line_ids = tuple(marker_object.metadata.get("support_line_ids") or ())
        ray_signs = tuple(marker_object.metadata.get("ray_signs") or ())
        angle = tuple(marker_object.metadata.get("ordered_angle") or ())
        if len(line_ids) != 2 or len(ray_signs) != 2 or len(angle) != 3:
            return None
        rays = {
            (str(line_id), int(ray_sign)): str(arm_id)
            for line_id, ray_sign, arm_id in zip(line_ids, ray_signs, (angle[0], angle[2]))
        }
        if len(rays) != 2:
            return None
        angle_rays.append(rays)
        vertices.append(str(angle[1]))
    if len(set(vertices)) != 1:
        return None
    shared_rays = set(angle_rays[0]) & set(angle_rays[1])
    if len(shared_rays) != 1:
        return None
    shared_ray = next(iter(shared_rays))
    if angle_rays[0][shared_ray] != angle_rays[1][shared_ray]:
        return None
    outer_rays = [tuple(ray for ray in rays if ray != shared_ray) for rays in angle_rays]
    if any(len(rays) != 1 for rays in outer_rays):
        return None
    first_outer, second_outer = outer_rays[0][0], outer_rays[1][0]
    first_arm = angle_rays[0][first_outer]
    second_arm = angle_rays[1][second_outer]
    if first_arm == second_arm:
        return None
    line_index = {str(line["id"]): line for line in lines}
    mode = _shared_ray_bisector_mode(shared_ray, first_outer, second_outer, line_index)
    if mode is None:
        return None
    return ParseRelation(
        id=f"relation:marker-group:{group_id}:angle-bisector",
        predicate="angle_bisector",
        arguments=(
            ParseReference("line", f"segment:{shared_ray[0]}"),
            ParseReference("arm_0", first_arm),
            ParseReference("vertex", vertices[0]),
            ParseReference("arm_1", second_arm),
        ),
        origin=ParseOrigin.COMPILED,
        confidence=equal_angle.confidence,
        score_kind="shared_ray_equal_angle_definition",
        evidence_refs=equal_angle.evidence_refs,
        dependencies=(equal_angle.id,),
        metadata={
            "mode": mode,
            "derived_from_equal_angle": equal_angle.id,
            "shared_ray": {"support_line_id": shared_ray[0], "ray_sign": shared_ray[1]},
            "semantic_source": "equal_angle_glyphs_with_shared_grounded_ray",
        },
    )


def _shared_ray_bisector_mode(
    shared_ray: tuple[str, int],
    first_outer: tuple[str, int],
    second_outer: tuple[str, int],
    lines: Mapping[str, Mapping[str, Any]],
) -> str | None:
    try:
        shared = _visual_ray_direction(lines[shared_ray[0]], shared_ray[1])
        first = _visual_ray_direction(lines[first_outer[0]], first_outer[1])
        second = _visual_ray_direction(lines[second_outer[0]], second_outer[1])
    except KeyError:
        return None
    expected = {
        "internal": (first[0] + second[0], first[1] + second[1]),
        "external": (first[0] - second[0], first[1] - second[1]),
    }
    residuals = {
        mode: abs(shared[0] * vector[1] - shared[1] * vector[0]) / math.hypot(*vector)
        for mode, vector in expected.items()
        if math.hypot(*vector) > 1e-9
    }
    if not residuals:
        return None
    mode, residual = min(residuals.items(), key=lambda item: (item[1], item[0]))
    return mode if residual <= math.sin(math.radians(5.0)) else None


def _visual_ray_direction(line: Mapping[str, Any], ray_sign: int) -> tuple[float, float]:
    if ray_sign not in {-1, 1}:
        raise InvalidGeometryInput(f"Invalid visual ray sign: {ray_sign}")
    start = tuple(float(value) for value in line["p1"])
    end = tuple(float(value) for value in line["p2"])
    length = math.dist(start, end)
    if length <= 1e-9:
        raise InvalidGeometryInput(f"Visual line {line.get('id')} has zero length")
    return (
        ray_sign * (end[0] - start[0]) / length,
        ray_sign * (end[1] - start[1]) / length,
    )


def _ground_equal_length_marker(
    marker: Mapping[str, Any],
    lines: Sequence[Mapping[str, Any]],
    topology: Mapping[str, Any],
    observation_to_point: Mapping[str, str],
    objects: Sequence[ParseObject],
    line_evidence: Mapping[str, str],
) -> tuple[str | None, ParseObject | None]:
    line_ids = tuple(str(value) for value in marker.get("line_ids") or ())
    center_values = tuple(marker.get("center") or ())
    if len(line_ids) != 1 or len(center_values) != 2:
        return None, None
    line_id = line_ids[0]
    line = next((candidate for candidate in lines if str(candidate["id"]) == line_id), None)
    if line is None:
        return None, None
    point_positions = {
        item.id: tuple(float(value) for value in item.geometry["pixel"])
        for item in objects
        if item.kind is ParseObjectKind.POINT and "pixel" in item.geometry
    }
    ordered_point_ids = _ordered_line_points(line, topology, observation_to_point)
    if any(point_id not in point_positions for point_id in ordered_point_ids):
        return None, None
    start = tuple(float(value) for value in line["p1"])
    end = tuple(float(value) for value in line["p2"])
    length = math.dist(start, end)
    if length <= 1e-9:
        return None, None
    direction = (end[0] - start[0]) / length, (end[1] - start[1]) / length
    center = float(center_values[0]), float(center_values[1])
    center_projection = _line_projection(center, start, direction)
    ordered = tuple(
        sorted(
            (
                (_line_projection(point_positions[point_id], start, direction), point_id)
                for point_id in ordered_point_ids
            ),
            key=lambda item: (item[0], item[1]),
        )
    )
    if len(ordered) == 2:
        lower, upper = ordered[0][0], ordered[1][0]
        return (f"segment:{line_id}", None) if lower < center_projection < upper else (None, None)

    matching_intervals = []
    for interval_index, (left, right) in enumerate(zip(ordered, ordered[1:])):
        interval_length = right[0] - left[0]
        endpoint_margin = min(4.0, interval_length * 0.2)
        if interval_length > 2.0 * endpoint_margin and (
            left[0] + endpoint_margin < center_projection < right[0] - endpoint_margin
        ):
            matching_intervals.append((interval_index, left, right))
    if len(matching_intervals) != 1:
        return None, None
    interval_index, left, right = matching_intervals[0]
    segment_id = f"segment:{line_id}:part:{interval_index:03d}"
    evidence_refs = tuple(
        dict.fromkeys(
            (
                line_evidence[line_id],
                f"evidence:{marker['id']}",
            )
        )
    )
    return segment_id, ParseObject(
        id=segment_id,
        kind=ParseObjectKind.SEGMENT,
        origin=ParseOrigin.COMPILED,
        confidence=min(0.95, float(marker.get("confidence", 0.0))),
        score_kind="topology_partition_interval",
        references=(ParseReference("start", left[1]), ParseReference("end", right[1])),
        evidence_refs=evidence_refs,
        visible=True,
        geometry={
            "p1": list(point_positions[left[1]]),
            "p2": list(point_positions[right[1]]),
            "angle_deg": line.get("angle_deg"),
        },
        metadata={
            "parent_support_id": f"segment:{line_id}",
            "ordered_interval_index": interval_index,
            "semantic_source": "explicit_visual_glyph_on_known_topology_interval",
        },
    )


def _line_projection(
    point: tuple[float, float],
    origin: tuple[float, float],
    direction: tuple[float, float],
) -> float:
    return (point[0] - origin[0]) * direction[0] + (point[1] - origin[1]) * direction[1]


def _ground_marker_angle(
    marker: Mapping[str, Any],
    lines: Sequence[Mapping[str, Any]],
    topology: Mapping[str, Any],
    observation_to_point: Mapping[str, str],
    objects: Sequence[ParseObject],
) -> tuple[str, str, str] | None:
    line_ids = tuple(str(value) for value in marker.get("line_ids") or ())
    ray_signs = tuple(marker.get("ray_signs") or ())
    vertex_values = tuple(marker.get("vertex") or ())
    if len(line_ids) != 2 or len(ray_signs) != 2 or len(vertex_values) != 2:
        return None
    if any(sign not in {-1, 1} for sign in ray_signs):
        raise InvalidGeometryInput(f"Angle marker {marker.get('id')} has invalid ray signs")
    line_index = {str(line["id"]): line for line in lines}
    if any(line_id not in line_index for line_id in line_ids):
        return None
    vertex = float(vertex_values[0]), float(vertex_values[1])
    intersection_points = {
        observation_to_point[key]
        for intersection in topology.get("line_intersections") or ()
        if set(str(value) for value in intersection.get("line_ids") or ()) == set(line_ids)
        if math.dist(vertex, tuple(float(value) for value in intersection.get("xy") or ())) <= 4.0
        if (key := f"{intersection['id']}:intersection") in observation_to_point
    }
    if len(intersection_points) != 1:
        return None
    vertex_id = next(iter(intersection_points))
    point_positions = {
        item.id: tuple(float(value) for value in item.geometry["pixel"])
        for item in objects
        if item.kind is ParseObjectKind.POINT and "pixel" in item.geometry
    }
    arm_ids = []
    for line_id, ray_sign in zip(line_ids, ray_signs):
        line = line_index[line_id]
        start = tuple(float(value) for value in line["p1"])
        end = tuple(float(value) for value in line["p2"])
        length = math.dist(start, end)
        if length <= 1e-9:
            return None
        direction = (end[0] - start[0]) / length, (end[1] - start[1]) / length
        candidates = []
        for point_id in _ordered_line_points(line, topology, observation_to_point):
            if point_id == vertex_id or point_id not in point_positions:
                continue
            point = point_positions[point_id]
            projection = ((point[0] - vertex[0]) * direction[0] + (point[1] - vertex[1]) * direction[1]) * ray_sign
            if projection > 2.0:
                candidates.append((projection, point_id))
        if not candidates:
            return None
        arm_ids.append(min(candidates)[1])
    return arm_ids[0], vertex_id, arm_ids[1]


def _angle_relations(
    lines: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, str],
) -> list[ParseRelation]:
    relations = []
    for index, left in enumerate(lines):
        for right in lines[index + 1 :]:
            delta = _angle_distance(float(left["angle_deg"]), float(right["angle_deg"]))
            predicate = "parallel" if delta <= 2.5 else "perpendicular" if abs(delta - 90.0) <= 3.0 else None
            if predicate is None:
                continue
            residual = delta if predicate == "parallel" else abs(delta - 90.0)
            left_id = str(left["id"])
            right_id = str(right["id"])
            relations.append(
                ParseRelation(
                    id=f"relation:{predicate}:{len(relations)}",
                    predicate=predicate,
                    arguments=(ParseReference("left", f"segment:{left_id}"), ParseReference("right", f"segment:{right_id}")),
                    origin=ParseOrigin.IMAGE,
                    confidence=round(max(0.5, 0.9 - residual / 10.0), 3),
                    score_kind="pixel_angle_residual",
                    evidence_refs=(evidence[left_id], evidence[right_id]),
                    metadata={"residual_deg": round(residual, 3), "exact_relation_not_admitted": True},
                )
            )
    return relations


def _target(
    parser_input: ParserInput,
    objects: Sequence[ParseObject],
    regions: Sequence[ParseRegion],
    bindings: Sequence[ParseRelation],
    question_relations: Sequence[ParseRelation] = (),
) -> tuple[ParseTarget | None, ParseEvidence | None, tuple[ParseRelation, ...]]:
    match = infer_target_type(parser_input.question)
    if match is None:
        return None, None, ()
    reference_match = infer_target_reference(parser_input.question, match.target_type)
    target_object, dependencies = _resolve_target_object(reference_match, objects, bindings)
    proof_relation = (
        question_relations[0]
        if match.target_type == "proof" and len(question_relations) == 1
        else None
    )
    if proof_relation is not None:
        target_object = "|".join(argument.ref for argument in proof_relation.arguments)
        dependencies = tuple(dict.fromkeys((proof_relation.id, *proof_relation.dependencies)))
        evidence_span = tuple(proof_relation.metadata.get("text_span") or match.span)
    else:
        evidence_span = reference_match.span if reference_match is not None else match.span
    observed_fill_region = None
    if target_object is None and match.target_type == "shadow_area":
        observed_fill_regions = tuple(
            region for region in regions if region.metadata.get("fill_status") == "observed"
        )
        if len(observed_fill_regions) == 1:
            observed_fill_region = observed_fill_regions[0]
            target_object = observed_fill_region.id
    appearance_hypothesis = observed_fill_region is not None
    evidence = ParseEvidence(
        id="evidence:question_target",
        source_id=f"{parser_input.id}:question",
        kind=ParseEvidenceKind.TEXT_SPAN,
        precision=(
            EvidencePrecision.BOUND
            if target_object is not None and not appearance_hypothesis
            else EvidencePrecision.LOCALIZED
        ),
        span=evidence_span,
        raw_content=parser_input.question[evidence_span[0] : evidence_span[1]],
        metadata={
            "type_pattern": match.pattern,
            "reference_pattern": reference_match.pattern if reference_match is not None else None,
            "semantic_shading_unverified": appearance_hypothesis,
            "proof_relation_id": proof_relation.id if proof_relation is not None else None,
        },
    )
    target_evidence_refs = tuple(
        dict.fromkeys(
            (
                evidence.id,
                *(observed_fill_region.evidence_refs if observed_fill_region is not None else ()),
            )
        )
    )
    semantic_relations = ()
    if observed_fill_region is not None:
        shaded_relation = ParseRelation(
            id="relation:shaded_region:question",
            predicate="shaded_region",
            arguments=(ParseReference("region", observed_fill_region.id),),
            origin=ParseOrigin.IMAGE_TEXT,
            confidence=min(0.7, observed_fill_region.confidence),
            score_kind="question_unique_observed_fill_hypothesis",
            evidence_refs=target_evidence_refs,
            metadata={
                "semantic_status": "appearance_hypothesis",
                "semantic_shading_unverified": True,
                "target_grounding": "unique_observed_fill",
                "text_span": list(match.span),
            },
        )
        semantic_relations = (shaded_relation,)
        dependencies = tuple(dict.fromkeys((*dependencies, shaded_relation.id)))
    target = ParseTarget(
        id="target:question",
        kind=("tangent_to_circle" if proof_relation.predicate == "tangent" else proof_relation.predicate)
        if proof_relation is not None
        else match.target_type,
        mode="proof" if match.target_type == "proof" else "value",
        arguments=(
            proof_relation.arguments
            if proof_relation is not None
            else (
                ParseReference("region" if appearance_hypothesis else "object", target_object),
            )
            if target_object is not None
            else ()
        ),
        origin=ParseOrigin.IMAGE_TEXT if appearance_hypothesis else ParseOrigin.TEXT,
        confidence=(proof_relation.confidence if proof_relation is not None else min(0.7, observed_fill_region.confidence) if observed_fill_region is not None else 0.8),
        score_kind=(
            "question_unique_observed_fill_hypothesis"
            if appearance_hypothesis
            else "question_proof_relation"
            if proof_relation is not None
            else "question_pattern_score"
        ),
        evidence_refs=tuple(
            dict.fromkeys((*target_evidence_refs, *(proof_relation.evidence_refs if proof_relation is not None else ())))
        ),
        dependencies=dependencies,
        metadata={
            "object_status": (
                "appearance_hypothesis"
                if appearance_hypothesis
                else "resolved" if target_object is not None else "unresolved"
            ),
            "textual_reference": list(reference_match.point_names) if reference_match is not None else None,
            "target_grounding": "unique_observed_fill" if appearance_hypothesis else None,
            "semantic_shading_unverified": appearance_hypothesis,
            "proof_relation_id": proof_relation.id if proof_relation is not None else None,
        },
    )
    return target, evidence, semantic_relations


def _question_relations(
    parser_input: ParserInput,
    objects: Sequence[ParseObject],
    bindings: Sequence[ParseRelation],
) -> tuple[tuple[ParseRelation, ...], tuple[ParseEvidence, ...], tuple[dict[str, Any], ...]]:
    relations = []
    evidence = []
    issues = []
    for index, match in enumerate(infer_relation_assertions(parser_input.question)):
        evidence_id = f"evidence:question_relation:{index}"
        arguments = []
        dependencies = []
        unresolved = []
        for reference in match.references:
            object_id, reference_dependencies = _resolve_relation_reference(reference, objects, bindings)
            if object_id is None:
                unresolved.append(
                    {
                        "role": reference.role,
                        "reference_kind": reference.reference_kind,
                        "point_names": list(reference.point_names),
                    }
                )
                continue
            arguments.append(ParseReference(reference.role, object_id))
            dependencies.extend(reference_dependencies)
        resolved = not unresolved
        evidence.append(
            ParseEvidence(
                id=evidence_id,
                source_id=f"{parser_input.id}:question",
                kind=ParseEvidenceKind.TEXT_SPAN,
                precision=EvidencePrecision.BOUND if resolved else EvidencePrecision.LOCALIZED,
                span=match.span,
                raw_content=parser_input.question[match.span[0] : match.span[1]],
                metadata={
                    "assertion_pattern": match.pattern,
                    "predicate": match.predicate,
                    "binding_status": "resolved" if resolved else "unresolved",
                },
            )
        )
        if not resolved:
            issues.append(
                {
                    "span": list(match.span),
                    "status": "reference_unresolved",
                    "predicate": match.predicate,
                    "references": unresolved,
                }
            )
            continue
        relations.append(
            ParseRelation(
                id=f"relation:question:{index}:{match.predicate}",
                predicate=match.predicate,
                arguments=tuple(arguments),
                origin=ParseOrigin.TEXT,
                confidence=0.98,
                score_kind="explicit_question_relation",
                evidence_refs=(evidence_id,),
                dependencies=tuple(dict.fromkeys(dependencies)),
                metadata={
                    "textual_references": [
                        {
                            "role": reference.role,
                            "reference_kind": reference.reference_kind,
                            "point_names": list(reference.point_names),
                        }
                        for reference in match.references
                    ],
                    "text_span": list(match.span),
                },
            )
        )
    return tuple(relations), tuple(evidence), tuple(issues)


def _suppress_redundant_visual_relations(
    relations: list[ParseRelation],
    question_relations: Sequence[ParseRelation],
) -> tuple[str, ...]:
    explicit_keys = {
        _symmetric_relation_key(relation)
        for relation in question_relations
        if _symmetric_relation_key(relation) is not None
    }
    suppressed = [
        relation.id
        for relation in relations
        if relation.score_kind == "pixel_angle_residual"
        and _symmetric_relation_key(relation) in explicit_keys
    ]
    if suppressed:
        suppressed_set = set(suppressed)
        relations[:] = [relation for relation in relations if relation.id not in suppressed_set]
    return tuple(suppressed)


def _symmetric_relation_key(relation: ParseRelation) -> tuple[str, frozenset[str]] | None:
    if relation.predicate not in {"parallel", "perpendicular", "equal_length"} or len(relation.arguments) != 2:
        return None
    object_ids = frozenset(argument.ref for argument in relation.arguments)
    return (relation.predicate, object_ids) if len(object_ids) == 2 else None


def _question_quantities(
    parser_input: ParserInput,
    objects: Sequence[ParseObject],
    bindings: Sequence[ParseRelation],
) -> tuple[tuple[ParseQuantity, ...], tuple[ParseEvidence, ...], tuple[dict[str, Any], ...]]:
    quantities = []
    evidence = []
    issues = []
    for index, match in enumerate(infer_quantity_assertions(parser_input.question)):
        evidence_id = f"evidence:question_quantity:{index}"
        object_id, dependencies = _resolve_segment_object(match.point_names, objects, bindings)
        try:
            expression = math_expression_from_text(match.expression_text)
        except InvalidGeometryInput as exc:
            expression = None
            issues.append({"span": list(match.expression_span), "status": "invalid_expression", "message": str(exc)})
        if object_id is None:
            issues.append({"span": list(match.span), "status": "object_unresolved", "point_names": list(match.point_names)})
        resolved = object_id is not None and expression is not None
        evidence.append(
            ParseEvidence(
                id=evidence_id,
                source_id=f"{parser_input.id}:question",
                kind=ParseEvidenceKind.TEXT_SPAN,
                precision=EvidencePrecision.BOUND if resolved else EvidencePrecision.LOCALIZED,
                span=match.span,
                raw_content=parser_input.question[match.span[0] : match.span[1]],
                metadata={
                    "assertion_pattern": match.pattern,
                    "binding_status": "resolved" if object_id is not None else "unresolved",
                    "expression_status": "resolved" if expression is not None else "unsupported",
                },
            )
        )
        if not resolved:
            continue
        quantities.append(
            ParseQuantity(
                id=f"quantity:question:{index}",
                kind=match.quantity_kind,
                arguments=(ParseReference("object", object_id),),
                expression=expression,
                origin=ParseOrigin.TEXT,
                confidence=0.95,
                score_kind="explicit_question_assertion",
                raw_text=match.expression_text,
                evidence_refs=(evidence_id,),
                dependencies=dependencies,
                metadata={"textual_reference": list(match.point_names)},
            )
        )
    return tuple(quantities), tuple(evidence), tuple(issues)


def _resolve_segment_object(
    point_names: Sequence[str],
    objects: Sequence[ParseObject],
    bindings: Sequence[ParseRelation],
) -> tuple[str | None, tuple[str, ...]]:
    if len(point_names) != 2:
        return None, ()
    binding_index = _point_binding_index(bindings)
    resolved = [binding_index.get(name, ()) for name in point_names]
    if any(len(items) != 1 for items in resolved):
        return None, ()
    point_ids = tuple(items[0][0] for items in resolved)
    dependencies = tuple(items[0][1] for items in resolved)
    segments = [
        item
        for item in objects
        if item.kind is ParseObjectKind.SEGMENT
        and {reference.ref for reference in item.references if reference.role in {"start", "end"}} == set(point_ids)
    ]
    if len(segments) != 1:
        return None, ()
    return segments[0].id, dependencies


def _resolve_target_object(reference_match, objects, bindings) -> tuple[str | None, tuple[str, ...]]:
    if reference_match is None:
        return None, ()
    if reference_match.reference_kind == "segment":
        return _resolve_segment_object(reference_match.point_names, objects, bindings)
    if reference_match.reference_kind == "polygon":
        return _resolve_polygon_object(reference_match.point_names, objects, bindings)
    return None, ()


def _resolve_relation_reference(
    reference: RelationReferenceMatch,
    objects: Sequence[ParseObject],
    bindings: Sequence[ParseRelation],
) -> tuple[str | None, tuple[str, ...]]:
    if reference.reference_kind == "point" and len(reference.point_names) == 1:
        return _resolve_named_point(reference.point_names[0], bindings)
    if reference.reference_kind == "segment":
        return _resolve_segment_object(reference.point_names, objects, bindings)
    return None, ()


def _resolve_named_point(
    point_name: str,
    bindings: Sequence[ParseRelation],
) -> tuple[str | None, tuple[str, ...]]:
    candidates = _point_binding_index(bindings).get(point_name.upper(), ())
    if len(candidates) != 1:
        return None, ()
    point_id, binding_id = candidates[0]
    return point_id, (binding_id,)


def _resolve_polygon_object(
    point_names: Sequence[str],
    objects: Sequence[ParseObject],
    bindings: Sequence[ParseRelation],
) -> tuple[str | None, tuple[str, ...]]:
    if len(point_names) < 3:
        return None, ()
    binding_index = _point_binding_index(bindings)
    resolved = [binding_index.get(name, ()) for name in point_names]
    if any(len(items) != 1 for items in resolved):
        return None, ()
    point_ids = tuple(items[0][0] for items in resolved)
    dependencies = tuple(items[0][1] for items in resolved)
    polygons = [
        candidate
        for candidate in objects
        if candidate.kind is ParseObjectKind.POLYGON
        and _same_polygon_cycle(
            tuple(reference.ref for reference in candidate.references if reference.role.startswith("vertex_")),
            point_ids,
        )
    ]
    if len(polygons) != 1:
        return None, ()
    return polygons[0].id, dependencies


def _point_binding_index(bindings: Sequence[ParseRelation]) -> dict[str, list[tuple[str, str]]]:
    index: dict[str, list[tuple[str, str]]] = {}
    for relation in bindings:
        label = relation.metadata.get("label")
        point_id = next((item.ref for item in relation.arguments if item.role == "object"), None)
        if isinstance(label, str) and point_id is not None:
            index.setdefault(label.upper(), []).append((point_id, relation.id))
    return index


def _same_polygon_cycle(left: Sequence[str], right: Sequence[str]) -> bool:
    if len(left) != len(right) or len(left) < 3:
        return False
    left_tuple = tuple(left)
    right_tuple = tuple(right)
    variants = []
    for values in (right_tuple, tuple(reversed(right_tuple))):
        variants.extend(values[index:] + values[:index] for index in range(len(values)))
    return left_tuple in variants


def _line_bbox(line: Mapping[str, Any]) -> tuple[float, float, float, float]:
    x1, y1 = (float(value) for value in line["p1"])
    x2, y2 = (float(value) for value in line["p2"])
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def _component_bbox(bbox: Sequence[Any]) -> tuple[float, float, float, float]:
    x, y, width, height = (float(value) for value in bbox)
    return x, y, x + width, y + height


def _arc_bbox(arc: Mapping[str, Any]) -> tuple[float, float, float, float]:
    bbox = tuple(float(value) for value in arc.get("bbox") or ())
    if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
        raise InvalidGeometryInput(f"Visual arc {arc.get('id', '<unknown>')} requires a finite bbox")
    left, top, right, bottom = bbox
    if left > right or top > bottom:
        raise InvalidGeometryInput(f"Visual arc {arc.get('id', '<unknown>')} has an inverted bbox")
    return bbox


def _arc_endpoints(arcs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"{arc['id']}_arc_endpoint_{endpoint}",
            "arc_id": str(arc["id"]),
            "endpoint": endpoint,
            "xy": list(arc[endpoint]),
        }
        for arc in arcs
        for endpoint in ("start", "end")
    ]


def _image_size(value: Any) -> tuple[int, int]:
    if not isinstance(value, Mapping):
        raise InvalidGeometryInput("Visual detector must report image_size")
    width = int(value.get("width", 0))
    height = int(value.get("height", 0))
    if width <= 0 or height <= 0:
        raise InvalidGeometryInput("Visual detector returned an invalid image_size")
    return width, height


def _detector_configuration(detector: Any) -> dict[str, Any]:
    configuration = getattr(detector, "configuration", None)
    if callable(configuration):
        return dict(configuration())
    return {"backend": type(detector).__name__}


def _angle_distance(left: float, right: float) -> float:
    delta = abs(left - right) % 180.0
    return min(delta, 180.0 - delta)


def _graph_confidence(graph: ParseGraph) -> float:
    candidates = (*graph.objects, *graph.relations, *graph.quantities, *graph.regions, *graph.targets)
    return sum(item.confidence for item in candidates) / len(candidates) if candidates else 0.0


def _stage(name: str, status: str, output_count: int) -> dict[str, Any]:
    return {"name": name, "status": status, "output_count": output_count}
