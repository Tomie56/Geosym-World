from dataclasses import replace

import cv2
import numpy as np
import pytest

from geosym_world.errors import InvalidGeometryInput
from geosym_world.schema import (
    AdmissionStatus,
    EvidencePrecision,
    ParseEvidence,
    ParseEvidenceKind,
    ParseGraph,
    ParseObject,
    ParseObjectKind,
    ParseOrigin,
    ParseReference,
    ParseRelation,
)
from geosym_world.verifier import (
    ParseAdmissionPolicy,
    ParseGraphAdmission,
    PixelTopologyVerifier,
)


def test_pixel_topology_verifies_visible_paths_intersection_and_point_order(tmp_path):
    image_path = tmp_path / "cross.png"
    image = np.full((120, 120), 255, dtype=np.uint8)
    cv2.line(image, (10, 60), (110, 60), 0, 2, cv2.LINE_AA)
    cv2.line(image, (60, 10), (60, 110), 0, 2, cv2.LINE_AA)
    assert cv2.imwrite(str(image_path), image)
    graph = _cross_graph()

    report = PixelTopologyVerifier().verify(graph, image_path)

    assert report.status == "passed"
    assert {item.candidate_id for item in report.checks} == {
        "segment_h",
        "segment_v",
        "relation_intersection",
        "relation_order_h",
        "relation_order_v",
    }
    assert report.to_dict()["asset_sha256"]
    admission = ParseGraphAdmission(
        ParseAdmissionPolicy(require_pixel_topology=True, policy_id="test_pixels")
    ).admit(graph, pixel_topology=report)
    assert "relation_intersection" in admission.admitted_ids


def test_pixel_topology_verifies_arc_path_and_arc_intersection(tmp_path):
    image_path = tmp_path / "arc_intersection.png"
    image = np.full((160, 160), 255, dtype=np.uint8)
    cv2.ellipse(image, (80, 80), (50, 50), 0, 270, 360, 0, 2, cv2.LINE_AA)
    cv2.line(image, (30, 45), (130, 45), 0, 2, cv2.LINE_AA)
    assert cv2.imwrite(str(image_path), image)

    evidence = (
        _evidence("evidence_line", bbox=(30.0, 43.0, 130.0, 47.0)),
        _evidence("evidence_arc", bbox=(78.0, 28.0, 132.0, 82.0)),
        _evidence("evidence_start", point=(130.0, 80.0)),
        _evidence("evidence_end", point=(80.0, 30.0)),
        _evidence("evidence_cross", point=(115.7, 45.0)),
        _evidence("evidence_left", point=(30.0, 45.0)),
        _evidence("evidence_right", point=(130.0, 45.0)),
    )
    points = (
        _point("point_start", (130.0, 80.0), "evidence_start"),
        _point("point_end", (80.0, 30.0), "evidence_end"),
        _point("point_cross", (115.7, 45.0), "evidence_cross"),
        _point("point_left", (30.0, 45.0), "evidence_left"),
        _point("point_right", (130.0, 45.0), "evidence_right"),
    )
    line = _segment("segment_cut", "point_left", "point_right", (30.0, 45.0), (130.0, 45.0), "evidence_line")
    circle = ParseObject(
        id="circle_support",
        kind=ParseObjectKind.CIRCLE,
        origin=ParseOrigin.COMPILED,
        confidence=1.0,
        score_kind="fixture_support",
        visible=False,
        geometry={"center": [80.0, 80.0], "radius_px": 50.0},
    )
    arc = ParseObject(
        id="arc_quarter",
        kind=ParseObjectKind.ARC,
        origin=ParseOrigin.IMAGE,
        confidence=0.9,
        score_kind="fixture_arc",
        references=(
            ParseReference("circle", circle.id),
            ParseReference("start", "point_start"),
            ParseReference("end", "point_end"),
        ),
        evidence_refs=("evidence_arc",),
        visible=True,
        geometry={"orientation": "ccw"},
        metadata={
            "center": [80.0, 80.0],
            "radius_px": 50.0,
            "sweep_deg": 90.0,
            "image_orientation": "cw",
            "orientation_coordinate_frame": "cartesian",
        },
    )
    intersection = ParseRelation(
        id="relation_arc_line",
        predicate="intersection",
        arguments=(
            ParseReference("support", line.id),
            ParseReference("support", arc.id),
            ParseReference("point", "point_cross"),
        ),
        origin=ParseOrigin.IMAGE,
        confidence=0.9,
        score_kind="fixture_arc_line_intersection",
        evidence_refs=("evidence_line", "evidence_arc", "evidence_cross"),
    )
    incidence = ParseRelation(
        id="relation_point_on_line",
        predicate="point_on_line",
        arguments=(
            ParseReference("point", "point_cross"),
            ParseReference("support", line.id),
        ),
        origin=ParseOrigin.IMAGE,
        confidence=0.9,
        score_kind="fixture_point_on_line",
        evidence_refs=("evidence_line", "evidence_cross"),
    )
    ordered = _ordered("relation_order_arc_line", line.id, ("point_left", "point_cross", "point_right"), "evidence_line")
    graph = ParseGraph(
        id="parse_arc_intersection",
        source_id="source_arc_intersection",
        evidence=evidence,
        objects=(*points, line, circle, arc),
        relations=(intersection, incidence, ordered),
        metadata={"image_size": [160, 160]},
    )

    report = PixelTopologyVerifier().verify(graph, image_path)

    assert report.status == "passed"
    assert report.check_for("arc_quarter").code == "stroke_path_supported"
    assert report.check_for("relation_arc_line").code == "intersection_supported"
    assert report.check_for("relation_point_on_line").code == "point_on_support"


def test_pixel_topology_rejects_a_formal_intersection_without_local_strokes(tmp_path):
    image_path = tmp_path / "broken_cross.png"
    image = np.full((120, 120), 255, dtype=np.uint8)
    cv2.line(image, (10, 60), (110, 60), 0, 2, cv2.LINE_AA)
    cv2.line(image, (60, 10), (60, 43), 0, 2, cv2.LINE_AA)
    cv2.line(image, (60, 77), (60, 110), 0, 2, cv2.LINE_AA)
    assert cv2.imwrite(str(image_path), image)
    graph = _cross_graph()

    report = PixelTopologyVerifier().verify(graph, image_path)
    failed = {item.candidate_id: item for item in report.checks if item.status.value == "failed"}

    assert report.status == "failed"
    assert failed["relation_intersection"].code == "intersection_not_supported"
    admission = ParseGraphAdmission(
        ParseAdmissionPolicy(require_pixel_topology=True, policy_id="test_pixels")
    ).admit(graph, pixel_topology=report)
    relation_decision = next(item for item in admission.decisions if item.candidate_id == "relation_intersection")
    assert relation_decision.decision is AdmissionStatus.REJECTED
    assert relation_decision.reason == "pixel_topology_verification_failed"


def test_required_pixel_topology_abstains_without_a_bound_report():
    graph = _cross_graph()

    report = ParseGraphAdmission(
        ParseAdmissionPolicy(require_pixel_topology=True, policy_id="test_pixels")
    ).admit(graph)

    decision = next(item for item in report.decisions if item.candidate_id == "segment_h")
    assert decision.decision is AdmissionStatus.ABSTAINED
    assert decision.reason == "pixel_topology_verification_required"


def test_admission_rejects_a_pixel_report_for_another_graph(tmp_path):
    image_path = tmp_path / "cross.png"
    image = np.full((120, 120), 255, dtype=np.uint8)
    cv2.line(image, (10, 60), (110, 60), 0, 2)
    cv2.line(image, (60, 10), (60, 110), 0, 2)
    assert cv2.imwrite(str(image_path), image)
    graph = _cross_graph()
    report = PixelTopologyVerifier().verify(graph, image_path)
    changed = replace(graph, metadata={**graph.metadata, "revision": 2})

    with pytest.raises(InvalidGeometryInput, match="does not match"):
        ParseGraphAdmission().admit(changed, pixel_topology=report)


def test_pixel_topology_rejects_malformed_image_size_metadata(tmp_path):
    image_path = tmp_path / "cross.png"
    image = np.full((120, 120), 255, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)
    graph = replace(_cross_graph(), metadata={"image_size": {"width": 120, "height": 120}})

    with pytest.raises(InvalidGeometryInput, match=r"\[width, height\]"):
        PixelTopologyVerifier().verify(graph, image_path)


def _cross_graph() -> ParseGraph:
    evidence = (
        _evidence("evidence_h", bbox=(10.0, 58.0, 110.0, 62.0)),
        _evidence("evidence_v", bbox=(58.0, 10.0, 62.0, 110.0)),
        _evidence("evidence_p0", point=(10.0, 60.0)),
        _evidence("evidence_p1", point=(60.0, 60.0)),
        _evidence("evidence_p2", point=(110.0, 60.0)),
        _evidence("evidence_p3", point=(60.0, 10.0)),
        _evidence("evidence_p4", point=(60.0, 110.0)),
    )
    points = (
        _point("point_left", (10.0, 60.0), "evidence_p0"),
        _point("point_cross", (60.0, 60.0), "evidence_p1"),
        _point("point_right", (110.0, 60.0), "evidence_p2"),
        _point("point_top", (60.0, 10.0), "evidence_p3"),
        _point("point_bottom", (60.0, 110.0), "evidence_p4"),
    )
    horizontal = _segment(
        "segment_h",
        "point_left",
        "point_right",
        (10.0, 60.0),
        (110.0, 60.0),
        "evidence_h",
    )
    vertical = _segment(
        "segment_v",
        "point_top",
        "point_bottom",
        (60.0, 10.0),
        (60.0, 110.0),
        "evidence_v",
    )
    relations = (
        ParseRelation(
            id="relation_intersection",
            predicate="intersection",
            arguments=(
                ParseReference("support", horizontal.id),
                ParseReference("support", vertical.id),
                ParseReference("point", "point_cross"),
            ),
            origin=ParseOrigin.IMAGE,
            confidence=0.9,
            score_kind="fixture",
            evidence_refs=("evidence_h", "evidence_v", "evidence_p1"),
        ),
        _ordered("relation_order_h", horizontal.id, ("point_left", "point_cross", "point_right"), "evidence_h"),
        _ordered("relation_order_v", vertical.id, ("point_top", "point_cross", "point_bottom"), "evidence_v"),
    )
    return ParseGraph(
        id="parse_cross",
        source_id="source_cross",
        evidence=evidence,
        objects=(*points, horizontal, vertical),
        relations=relations,
        metadata={"image_size": [120, 120]},
    )


def _evidence(evidence_id, *, bbox=None, point=None):
    return ParseEvidence(
        id=evidence_id,
        source_id="source_cross",
        kind=ParseEvidenceKind.IMAGE_REGION if bbox is not None else ParseEvidenceKind.PIXEL_POINT,
        precision=EvidencePrecision.LOCALIZED,
        bbox=bbox,
        point=point,
    )


def _point(point_id, pixel, evidence_id):
    return ParseObject(
        id=point_id,
        kind=ParseObjectKind.POINT,
        origin=ParseOrigin.IMAGE,
        confidence=0.9,
        score_kind="fixture",
        evidence_refs=(evidence_id,),
        visible=True,
        geometry={"pixel": list(pixel)},
    )


def _segment(segment_id, start, end, p1, p2, evidence_id):
    return ParseObject(
        id=segment_id,
        kind=ParseObjectKind.SEGMENT,
        origin=ParseOrigin.IMAGE,
        confidence=0.9,
        score_kind="fixture",
        references=(ParseReference("start", start), ParseReference("end", end)),
        evidence_refs=(evidence_id,),
        visible=True,
        geometry={"p1": list(p1), "p2": list(p2), "angle_deg": 0.0},
    )


def _ordered(relation_id, segment_id, point_ids, evidence_id):
    return ParseRelation(
        id=relation_id,
        predicate="ordered_points",
        arguments=(
            ParseReference("support", segment_id),
            *(ParseReference(f"point_{index}", point_id) for index, point_id in enumerate(point_ids)),
        ),
        origin=ParseOrigin.IMAGE,
        confidence=0.9,
        score_kind="fixture",
        evidence_refs=(evidence_id,),
    )
