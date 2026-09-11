from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from geosym_world.errors import InvalidGeometryInput
from geosym_world.parser import (
    ParseGraphReconstructor,
    ParserInput,
    ReconstructionMode,
    TypedDiagramParserV0,
    ground_point_labels,
    validate_parse_graph_hypothesis,
)
from geosym_world.rendering import source_graph_to_render_scene
from geosym_world.parser.vision import VisualMarkerDetector, VisualPrimitiveDetector
from geosym_world.parser.vision.marker_detector import _topology_points
from geosym_world.parser.vision.primitive_detector import _fit_arc_contour
from geosym_world.schema import (
    AdmissionStatus,
    EvidencePrecision,
    ParseEvidence,
    ParseEvidenceKind,
    ParseObject,
    ParseObjectKind,
    ParseOrigin,
)
from geosym_world.verifier import ParseAdmissionPolicy, ParseGraphAdmission


class StubDetector:
    def detect(self, image_path: str):
        return {
            "image_size": {"width": 200, "height": 200},
            "lines": [
                {"id": "vl_0", "p1": [20, 100], "p2": [180, 100], "angle_deg": 0.0, "source_ids": ["raw_0"]},
                {"id": "vl_1", "p1": [100, 20], "p2": [100, 180], "angle_deg": 90.0, "source_ids": ["raw_1"]},
            ],
            "circles": [],
            "text_regions": [{"id": "vt_0", "bbox": [8, 92, 8, 12]}],
            "topology": {
                "endpoints": [
                    {"id": "vl_0_p1", "line_id": "vl_0", "xy": [20, 100]},
                    {"id": "vl_0_p2", "line_id": "vl_0", "xy": [180, 100]},
                    {"id": "vl_1_p1", "line_id": "vl_1", "xy": [100, 20]},
                    {"id": "vl_1_p2", "line_id": "vl_1", "xy": [100, 180]},
                ],
                "line_intersections": [{"id": "vx_0", "line_ids": ["vl_0", "vl_1"], "xy": [100, 100]}],
                "circle_line_hits": [],
            },
        }


class StubRecognizer:
    def recognize(self, image_path, text_regions):
        return [{"id": "vt_0", "bbox": [8, 92, 8, 12], "text": "A", "confidence": 0.95, "status": "resolved"}]


class NamedSegmentDetector:
    def detect(self, image_path: str):
        return {
            "image_size": {"width": 200, "height": 200},
            "lines": [
                {"id": "vl_ab", "p1": [20, 100], "p2": [180, 100], "angle_deg": 0.0, "source_ids": ["raw_ab"]}
            ],
            "circles": [],
            "text_regions": [
                {"id": "vt_a", "bbox": [8, 92, 8, 12]},
                {"id": "vt_b", "bbox": [184, 92, 8, 12]},
            ],
            "topology": {
                "endpoints": [
                    {"id": "vl_ab_p1", "line_id": "vl_ab", "xy": [20, 100]},
                    {"id": "vl_ab_p2", "line_id": "vl_ab", "xy": [180, 100]},
                ],
                "line_intersections": [],
                "circle_line_hits": [],
            },
        }


class NamedSegmentRecognizer:
    def recognize(self, image_path, text_regions):
        return [
            {"id": "vt_a", "bbox": [8, 92, 8, 12], "text": "A", "confidence": 0.95, "status": "resolved"},
            {"id": "vt_b", "bbox": [184, 92, 8, 12], "text": "B", "confidence": 0.96, "status": "resolved"},
        ]


class NamedLineDetector(NamedSegmentDetector):
    def detect(self, image_path: str):
        result = super().detect(image_path)
        result["text_regions"].append({"id": "vt_line", "bbox": [96, 88, 8, 8]})
        return result


class NamedLineRecognizer(NamedSegmentRecognizer):
    def recognize(self, image_path, text_regions):
        return [
            *super().recognize(image_path, text_regions),
            {"id": "vt_line", "bbox": [96, 88, 8, 8], "text": "k", "confidence": 0.93, "status": "resolved"},
        ]


class CurvedIntersectionDetector:
    def detect(self, image_path: str):
        lines = [
            {"id": "vl_0", "p1": [20, 100], "p2": [180, 100], "angle_deg": 0.0, "source_ids": ["raw_0"]}
        ]
        circles = [
            {"id": "vc_0", "center": [80, 100], "radius_px": 40, "edge_support": 0.95},
            {"id": "vc_1", "center": [120, 100], "radius_px": 40, "edge_support": 0.95},
        ]
        topology = VisualPrimitiveDetector()._derive_topology(lines, circles)
        return {
            "image_size": {"width": 200, "height": 200},
            "lines": lines,
            "circles": circles,
            "text_regions": [],
            "topology": topology,
        }


class RightAngleMarkerDetector:
    def detect(self, image_path: str):
        return {
            "image_size": {"width": 200, "height": 200},
            "lines": [
                {"id": "vl_h", "p1": [20, 160], "p2": [180, 160], "length_px": 160.0, "angle_deg": 0.0},
                {"id": "vl_v", "p1": [20, 20], "p2": [20, 160], "length_px": 140.0, "angle_deg": 90.0},
            ],
            "circles": [],
            "markers": [
                {
                    "id": "vm_0",
                    "kind": "right_angle",
                    "semantic_relation": "perpendicular",
                    "line_ids": ["vl_h", "vl_v"],
                    "vertex": [20.0, 160.0],
                    "path": [[36.0, 160.0], [36.0, 144.0], [20.0, 144.0]],
                    "bbox": [18.0, 142.0, 38.0, 162.0],
                    "size_px": 16,
                    "confidence": 0.98,
                    "score_kind": "right_angle_l_stroke_support",
                    "stroke_support": [1.0, 1.0],
                    "angle_residual_deg": 0.0,
                }
            ],
            "text_regions": [],
            "topology": {
                "endpoints": [
                    {"id": "vl_h_p1", "line_id": "vl_h", "xy": [20, 160]},
                    {"id": "vl_h_p2", "line_id": "vl_h", "xy": [180, 160]},
                    {"id": "vl_v_p1", "line_id": "vl_v", "xy": [20, 20]},
                    {"id": "vl_v_p2", "line_id": "vl_v", "xy": [20, 160]},
                ],
                "line_intersections": [{"id": "vx_0", "line_ids": ["vl_h", "vl_v"], "xy": [20, 160]}],
                "circle_line_hits": [],
            },
        }


class GroupedMarkerDetector:
    def detect(self, image_path: str):
        lines = [
            {"id": "vl_top", "p1": [20, 60], "p2": [180, 60], "length_px": 160.0, "angle_deg": 0.0},
            {"id": "vl_bottom", "p1": [20, 140], "p2": [180, 140], "length_px": 160.0, "angle_deg": 0.0},
            {"id": "vl_vertical", "p1": [20, 20], "p2": [20, 140], "length_px": 120.0, "angle_deg": 90.0},
        ]
        markers = [
            _visual_marker("vm_parallel_top", "parallel", ["vl_top"], [92.0, 60.0], "vg_parallel_0", "parallel"),
            _visual_marker("vm_parallel_bottom", "parallel", ["vl_bottom"], [92.0, 140.0], "vg_parallel_0", "parallel"),
            _visual_marker(
                "vm_tick_top", "equal_length_tick", ["vl_top"], [55.0, 60.0], "vg_equal_length_0", "equal_length"
            ),
            _visual_marker(
                "vm_tick_bottom", "equal_length_tick", ["vl_bottom"], [55.0, 140.0], "vg_equal_length_0", "equal_length"
            ),
            _visual_marker("vm_angle", "angle_arc", ["vl_top", "vl_vertical"], [28.0, 68.0]),
        ]
        return {
            "image_size": {"width": 200, "height": 200},
            "lines": lines,
            "circles": [],
            "markers": markers,
            "text_regions": [],
            "topology": {
                "endpoints": [
                    {"id": f"{line['id']}_p{index}", "line_id": line["id"], "xy": line[key]}
                    for line in lines
                    for index, key in enumerate(("p1", "p2"), start=1)
                ],
                "line_intersections": [
                    {"id": "vx_0", "line_ids": ["vl_top", "vl_vertical"], "xy": [20, 60]},
                    {"id": "vx_1", "line_ids": ["vl_bottom", "vl_vertical"], "xy": [20, 140]},
                ],
                "circle_line_hits": [],
            },
        }


class SameLineEqualLengthDetector:
    def __init__(self, *, include_split: bool = True):
        self.include_split = include_split

    def detect(self, image_path: str):
        lines = [
            {"id": "vl_main", "p1": [20, 100], "p2": [180, 100], "length_px": 160.0, "angle_deg": 0.0},
        ]
        intersections = []
        if self.include_split:
            lines.append(
                {"id": "vl_cross", "p1": [100, 40], "p2": [100, 160], "length_px": 120.0, "angle_deg": 90.0}
            )
            intersections.append(
                {"id": "vx_split", "line_ids": ["vl_main", "vl_cross"], "xy": [100.0, 100.0]}
            )
        return {
            "image_size": {"width": 200, "height": 200},
            "lines": lines,
            "circles": [],
            "fill_regions": [],
            "markers": [
                _visual_marker(
                    "vm_tick_left", "equal_length_tick", ["vl_main"], [60.0, 100.0], "vg_equal_parts", "equal_length"
                ),
                _visual_marker(
                    "vm_tick_right", "equal_length_tick", ["vl_main"], [140.0, 100.0], "vg_equal_parts", "equal_length"
                ),
            ],
            "text_regions": [],
            "topology": {
                "endpoints": [
                    {"id": f"{line['id']}_p{index}", "line_id": line["id"], "xy": line[key]}
                    for line in lines
                    for index, key in enumerate(("p1", "p2"), start=1)
                ],
                "line_intersections": intersections,
                "circle_line_hits": [],
            },
        }


class TangentIntersectionDetector:
    def detect(self, image_path: str):
        lines = [
            {"id": "tangent_line", "p1": [20, 100], "p2": [180, 100], "length_px": 160.0, "angle_deg": 0.0}
        ]
        circles = [{"id": "tangent_circle", "center": [100, 140], "radius_px": 40, "edge_support": 0.95}]
        topology = VisualPrimitiveDetector()._derive_topology(lines, circles)
        return {
            "image_size": {"width": 200, "height": 200},
            "lines": lines,
            "circles": circles,
            "text_regions": [],
            "topology": topology,
        }


class ArcCandidateDetector:
    def __init__(self, *, support_circle: bool):
        self.support_circle = support_circle

    def detect(self, image_path: str):
        arc = {
            "id": "arc_candidate",
            "center": [100.0, 100.0],
            "radius_px": 60.0,
            "start": [160.0, 100.0],
            "end": [100.0, 40.0],
            "orientation": "ccw",
            "sweep_deg": 90.0,
            "bbox": [98.0, 38.0, 162.0, 102.0],
            "residual_px": 0.8,
            "edge_support": 0.9,
            "score_kind": "fixture_arc_fit",
            "circle_id": "arc_circle" if self.support_circle else None,
        }
        circles = (
            {
                "id": "arc_circle",
                "center": [100.0, 100.0],
                "radius_px": 60.0,
                "edge_support": 0.92,
            },
        ) if self.support_circle else []
        return {
            "image_size": {"width": 200, "height": 200},
            "lines": [],
            "circles": circles,
            "arcs": [arc],
            "fill_regions": [],
            "markers": [],
            "text_regions": [],
            "topology": {
                "endpoints": [],
                "line_intersections": [],
                "circle_line_hits": [],
                "arc_endpoints": [
                    {
                        "id": f"arc_candidate_{endpoint}",
                        "arc_id": "arc_candidate",
                        "endpoint": endpoint,
                        "xy": arc[endpoint],
                    }
                    for endpoint in ("start", "end")
                ],
            },
        }


class FilledArcChordDetector:
    def detect(self, image_path: str):
        arc = {
            "id": "arc_candidate",
            "center": [100.0, 100.0],
            "radius_px": 60.0,
            "start": [160.0, 100.0],
            "end": [40.0, 100.0],
            "orientation": "cw",
            "sweep_deg": 180.0,
            "bbox": [38.0, 38.0, 162.0, 102.0],
            "residual_px": 0.8,
            "edge_support": 0.9,
            "score_kind": "fixture_arc_fit",
            "circle_id": "arc_circle",
        }
        line = {
            "id": "chord",
            "p1": [40.0, 100.0],
            "p2": [160.0, 100.0],
            "length_px": 120.0,
            "angle_deg": 0.0,
        }
        circle = {
            "id": "arc_circle",
            "center": [100.0, 100.0],
            "radius_px": 60.0,
            "edge_support": 0.92,
        }
        return {
            "image_size": {"width": 200, "height": 200},
            "lines": [line],
            "circles": [circle],
            "arcs": [arc],
            "fill_regions": [
                {
                    "id": "fill_segment",
                    "bbox": [40.0, 40.0, 120.0, 60.0],
                    "outer_contour": [
                        [160.0, 100.0],
                        [142.0, 58.0],
                        [100.0, 40.0],
                        [58.0, 58.0],
                        [40.0, 100.0],
                    ],
                    "interior_gaps": [],
                    "area_px": 5400.0,
                    "confidence": 0.9,
                    "score_kind": "fixture_fill",
                    "mean_rgb": [220, 240, 245],
                }
            ],
            "markers": [],
            "text_regions": [],
            "topology": VisualPrimitiveDetector()._derive_topology([line], [circle], [arc]),
        }


class FilledSectorDetector:
    def detect(self, image_path: str):
        arc = {
            "id": "sector_arc",
            "center": [100.0, 100.0],
            "radius_px": 60.0,
            "start": [160.0, 100.0],
            "end": [100.0, 40.0],
            "orientation": "cw",
            "sweep_deg": 90.0,
            "bbox": [98.0, 38.0, 162.0, 102.0],
            "residual_px": 0.8,
            "edge_support": 0.9,
            "score_kind": "fixture_arc_fit",
            "circle_id": "sector_circle",
        }
        end_radius = {
            "id": "end_radius",
            "p1": [100.0, 40.0],
            "p2": [100.0, 100.0],
            "length_px": 60.0,
            "angle_deg": 90.0,
        }
        start_radius = {
            "id": "start_radius",
            "p1": [100.0, 100.0],
            "p2": [160.0, 100.0],
            "length_px": 60.0,
            "angle_deg": 0.0,
        }
        circle = {
            "id": "sector_circle",
            "center": [100.0, 100.0],
            "radius_px": 60.0,
            "edge_support": 0.92,
        }
        lines = [end_radius, start_radius]
        return {
            "image_size": {"width": 200, "height": 200},
            "lines": lines,
            "circles": [circle],
            "arcs": [arc],
            "fill_regions": [
                {
                    "id": "fill_sector",
                    "bbox": [100.0, 40.0, 60.0, 60.0],
                    "outer_contour": [
                        [160.0, 100.0],
                        [142.0, 58.0],
                        [100.0, 40.0],
                        [100.0, 100.0],
                    ],
                    "interior_gaps": [],
                    "area_px": 2700.0,
                    "confidence": 0.9,
                    "score_kind": "fixture_fill",
                    "mean_rgb": [220, 240, 245],
                }
            ],
            "markers": [],
            "text_regions": [],
            "topology": VisualPrimitiveDetector()._derive_topology(lines, [circle], [arc]),
        }


class AmbiguousFilledSectorDetector(FilledSectorDetector):
    def detect(self, image_path: str):
        result = super().detect(image_path)
        duplicate = {**result["lines"][0], "id": "end_radius_duplicate"}
        result["lines"] = [*result["lines"], duplicate]
        result["topology"] = VisualPrimitiveDetector()._derive_topology(
            result["lines"],
            result["circles"],
            result["arcs"],
        )
        return result


class OverlappingLineDetector:
    def detect(self, image_path: str):
        lines = [
            {"id": "overlap_left", "p1": [20, 100], "p2": [120, 100], "length_px": 100.0, "angle_deg": 0.0},
            {"id": "overlap_right", "p1": [80, 100], "p2": [180, 100], "length_px": 100.0, "angle_deg": 0.0},
        ]
        topology = VisualPrimitiveDetector()._derive_topology(lines, [])
        return {
            "image_size": {"width": 200, "height": 200},
            "lines": lines,
            "circles": [],
            "text_regions": [],
            "topology": topology,
        }


class FilledFaceDetector:
    def detect(self, image_path: str):
        lines = [
            {"id": "top", "p1": [20, 20], "p2": [180, 20], "length_px": 160.0, "angle_deg": 0.0},
            {"id": "right", "p1": [180, 20], "p2": [180, 180], "length_px": 160.0, "angle_deg": 90.0},
            {"id": "bottom", "p1": [180, 180], "p2": [20, 180], "length_px": 160.0, "angle_deg": 180.0},
            {"id": "left", "p1": [20, 180], "p2": [20, 20], "length_px": 160.0, "angle_deg": -90.0},
        ]
        return {
            "image_size": {"width": 200, "height": 200},
            "lines": lines,
            "circles": [],
            "fill_regions": [
                {
                    "id": "vf_face",
                    "bbox": [23, 23, 155, 155],
                    "outer_contour": [[23, 23], [177, 23], [177, 177], [23, 177]],
                    "interior_gaps": [],
                    "area_px": 23716,
                    "mean_rgb": [219, 234, 254],
                    "confidence": 0.94,
                    "score_kind": "fixture_fill_component",
                },
                {
                    "id": "vf_unmatched",
                    "bbox": [2, 2, 10, 10],
                    "outer_contour": [[2, 2], [12, 2], [12, 12], [2, 12]],
                    "interior_gaps": [],
                    "area_px": 100,
                    "mean_rgb": [254, 202, 202],
                    "confidence": 0.8,
                    "score_kind": "fixture_fill_component",
                },
            ],
            "markers": [],
            "text_regions": [],
            "topology": {
                "endpoints": [
                    {"id": f"{line['id']}_p{index}", "line_id": line["id"], "xy": line[key]}
                    for line in lines
                    for index, key in enumerate(("p1", "p2"), start=1)
                ],
                "line_intersections": [],
                "circle_line_hits": [],
            },
        }


class MultipleFilledFaceDetector(FilledFaceDetector):
    def detect(self, image_path: str):
        result = super().detect(image_path)
        translated_lines = [
            {
                **line,
                "id": f"second_{line['id']}",
                "p1": [line["p1"][0] + 200, line["p1"][1]],
                "p2": [line["p2"][0] + 200, line["p2"][1]],
            }
            for line in result["lines"]
        ]
        result["image_size"] = {"width": 400, "height": 200}
        result["lines"].extend(translated_lines)
        result["fill_regions"] = [
            result["fill_regions"][0],
            {
                **result["fill_regions"][0],
                "id": "vf_second_face",
                "bbox": [223, 23, 155, 155],
                "outer_contour": [[223, 23], [377, 23], [377, 177], [223, 177]],
            },
        ]
        result["topology"] = VisualPrimitiveDetector()._derive_topology(result["lines"], [])
        return result


class FilledFaceWithGapDetector(FilledFaceDetector):
    def detect(self, image_path: str):
        result = super().detect(image_path)
        result["fill_regions"][0]["interior_gaps"] = [[[60, 60], [80, 60], [80, 80], [60, 80]]]
        return result


class FilledFaceWithClosedGapDetector(FilledFaceWithGapDetector):
    def detect(self, image_path: str):
        result = super().detect(image_path)
        inner_lines = [
            {"id": "hole_top", "p1": [60, 60], "p2": [80, 60], "angle_deg": 0.0},
            {"id": "hole_right", "p1": [80, 60], "p2": [80, 80], "angle_deg": 90.0},
            {"id": "hole_bottom", "p1": [80, 80], "p2": [60, 80], "angle_deg": 180.0},
            {"id": "hole_left", "p1": [60, 80], "p2": [60, 60], "angle_deg": -90.0},
        ]
        result["lines"].extend(inner_lines)
        result["topology"]["endpoints"].extend(
            {
                "id": f"{line['id']}_p{index}",
                "line_id": line["id"],
                "xy": line[key],
            }
            for line in inner_lines
            for index, key in enumerate(("p1", "p2"), start=1)
        )
        return result


class EqualAngleMarkerDetector:
    def detect(self, image_path: str):
        lines = [
            {"id": "first_x", "p1": [20, 40], "p2": [80, 40], "length_px": 60.0, "angle_deg": 0.0},
            {"id": "first_y", "p1": [20, 40], "p2": [20, 100], "length_px": 60.0, "angle_deg": 90.0},
            {"id": "second_x", "p1": [120, 40], "p2": [180, 40], "length_px": 60.0, "angle_deg": 0.0},
            {"id": "second_y", "p1": [120, 40], "p2": [120, 100], "length_px": 60.0, "angle_deg": 90.0},
        ]
        markers = [
            _angle_marker("vm_angle_first", ["first_x", "first_y"], [20.0, 40.0]),
            _angle_marker("vm_angle_second", ["second_x", "second_y"], [120.0, 40.0]),
        ]
        return {
            "image_size": {"width": 200, "height": 140},
            "lines": lines,
            "circles": [],
            "fill_regions": [],
            "markers": markers,
            "text_regions": [],
            "topology": {
                "endpoints": [
                    {"id": f"{line['id']}_p{index}", "line_id": line["id"], "xy": line[key]}
                    for line in lines
                    for index, key in enumerate(("p1", "p2"), start=1)
                ],
                "line_intersections": [
                    {"id": "vx_first", "line_ids": ["first_x", "first_y"], "xy": [20.0, 40.0]},
                    {"id": "vx_second", "line_ids": ["second_x", "second_y"], "xy": [120.0, 40.0]},
                ],
                "circle_line_hits": [],
            },
        }


class AngleBisectorMarkerDetector:
    def detect(self, image_path: str):
        vertex = [20.0, 100.0]
        lines = [
            {"id": "outer_0", "p1": vertex, "p2": [180.0, 100.0], "length_px": 160.0, "angle_deg": 0.0},
            {"id": "bisector", "p1": vertex, "p2": [100.0, 20.0], "length_px": 113.1, "angle_deg": -45.0},
            {"id": "outer_1", "p1": vertex, "p2": [20.0, 20.0], "length_px": 80.0, "angle_deg": -90.0},
        ]
        first = _angle_marker("vm_angle_first", ["outer_0", "bisector"], vertex)
        second = _angle_marker("vm_angle_second", ["bisector", "outer_1"], vertex)
        first["angle_span_deg"] = 45.0
        second["angle_span_deg"] = 45.0
        return {
            "image_size": {"width": 200, "height": 140},
            "lines": lines,
            "circles": [],
            "fill_regions": [],
            "markers": [first, second],
            "text_regions": [],
            "topology": {
                "endpoints": [
                    {"id": f"{line['id']}_p{index}", "line_id": line["id"], "xy": line[key]}
                    for line in lines
                    for index, key in enumerate(("p1", "p2"), start=1)
                ],
                "line_intersections": [
                    {"id": "vx_first", "line_ids": ["outer_0", "bisector"], "xy": vertex},
                    {"id": "vx_second", "line_ids": ["bisector", "outer_1"], "xy": vertex},
                    {"id": "vx_outer", "line_ids": ["outer_0", "outer_1"], "xy": vertex},
                ],
                "circle_line_hits": [],
            },
        }


def _visual_marker(marker_id, kind, line_ids, center, group_id=None, semantic_relation=None):
    payload = {
        "id": marker_id,
        "kind": kind,
        "line_ids": line_ids,
        "center": center,
        "path": [[center[0] - 4.0, center[1] - 4.0], [center[0] + 4.0, center[1] + 4.0]],
        "bbox": [center[0] - 6.0, center[1] - 6.0, center[0] + 6.0, center[1] + 6.0],
        "size_px": 6,
        "confidence": 0.94,
        "score_kind": f"fixture_{kind}",
        "stroke_support": [1.0],
    }
    if group_id is not None:
        payload["relation_group_id"] = group_id
        payload["semantic_relation"] = semantic_relation
    return payload


def _angle_marker(marker_id, line_ids, vertex):
    marker = _visual_marker(
        marker_id,
        "angle_arc",
        line_ids,
        [vertex[0] + 8.0, vertex[1] + 8.0],
        "vg_equal_angle_0",
        "equal_angle",
    )
    marker.update({"vertex": vertex, "ray_signs": [1, 1], "angle_span_deg": 90.0})
    return marker


def test_typed_parser_materializes_segments_anonymous_intersection_and_point_order(tmp_path):
    image_path = tmp_path / "opaque.png"
    image_path.write_bytes(b"fixture")
    result = TypedDiagramParserV0(detector=StubDetector(), label_recognizer=StubRecognizer()).parse(
        ParserInput(id="opaque_sample", image=str(image_path), question="Find the area.")
    )

    graph = validate_parse_graph_hypothesis(result.record.hypotheses[0])
    points = [item for item in graph.objects if item.kind is ParseObjectKind.POINT]
    segments = [item for item in graph.objects if item.kind is ParseObjectKind.SEGMENT]
    intersection = next(item for item in graph.relations if item.predicate == "intersection")
    ordered = [item for item in graph.relations if item.predicate == "ordered_points"]

    assert result.record.state is None
    assert len(points) == 5
    assert len(segments) == 2
    assert len(intersection.arguments) == 3
    assert all(len(item.arguments) == 4 for item in ordered)
    assert all(point.visible_name is None for point in points)
    binding = next(item for item in graph.relations if item.predicate == "binds")
    assert [(item.role, item.ref) for item in binding.arguments] == [
        ("label", "text:vt_0"),
        ("object", next(point.id for point in points if point.geometry["pixel"] == [20.0, 100.0])),
    ]
    admission = ParseGraphAdmission().admit(graph)
    assert next(item for item in admission.decisions if item.candidate_id == binding.id).reason == "bound_label_grounding"
    assert admission.admitted_graph.point_names() == {binding.arguments[1].ref: "A"}
    assert any(label.text == "A" for label in source_graph_to_render_scene(admission.admitted_graph).labels)
    assert graph.targets[0].kind == "area"
    assert graph.targets[0].metadata["object_status"] == "unresolved"


def test_pixel_angle_relation_remains_an_unadmitted_visual_hypothesis(tmp_path):
    image_path = Path(tmp_path) / "opaque.png"
    image_path.write_bytes(b"fixture")
    result = TypedDiagramParserV0(detector=StubDetector(), label_recognizer=StubRecognizer()).parse(
        ParserInput(id="opaque_sample", image=str(image_path), question="")
    )

    perpendicular = next(item for item in result.graph.relations if item.predicate == "perpendicular")

    assert perpendicular.score_kind == "pixel_angle_residual"
    assert perpendicular.metadata["exact_relation_not_admitted"] is True
    assert result.record.hypotheses[0].status.value == "proposed"
    assert result.stages[-1] == {"name": "admission", "status": "pending", "output_count": 0}


def test_curved_intersections_become_anonymous_points_relations_and_line_order(tmp_path):
    image_path = Path(tmp_path) / "opaque.png"
    image_path.write_bytes(b"fixture")
    result = TypedDiagramParserV0(detector=CurvedIntersectionDetector()).parse(
        ParserInput(id="curved_intersections", image=str(image_path), question="")
    )

    intersections = [item for item in result.graph.relations if item.predicate == "intersection"]
    ordered = next(item for item in result.graph.relations if item.predicate == "ordered_points")
    assert len(intersections) == 6
    assert len(ordered.arguments) == 7
    assert all(item.metadata["contact"] == "crossing" for item in intersections)
    report = ParseGraphAdmission().admit(result.graph)
    assert all(item.id in report.admitted_ids for item in intersections)


def test_tangent_contact_becomes_an_unadmitted_tangent_hypothesis(tmp_path):
    image_path = Path(tmp_path) / "opaque.png"
    image_path.write_bytes(b"fixture")
    result = TypedDiagramParserV0(detector=TangentIntersectionDetector()).parse(
        ParserInput(id="tangent_intersection", image=str(image_path), question="")
    )

    tangent = next(item for item in result.graph.relations if item.predicate == "tangent")
    assert [argument.role for argument in tangent.arguments] == ["point", "line", "circle"]
    assert tangent.dependencies == ("relation:circle_line_intersection:0",)
    assert tangent.metadata["exact_relation_not_admitted"] is True
    admission = ParseGraphAdmission().admit(result.graph)
    decision = next(item for item in admission.decisions if item.candidate_id == tangent.id)
    assert decision.decision is not AdmissionStatus.ADMITTED


def test_collinear_segment_overlap_becomes_an_unadmitted_topology_hypothesis(tmp_path):
    image_path = Path(tmp_path) / "opaque.png"
    image_path.write_bytes(b"fixture")
    result = TypedDiagramParserV0(detector=OverlappingLineDetector()).parse(
        ParserInput(id="overlapping_segments", image=str(image_path), question="")
    )

    overlap = next(item for item in result.graph.relations if item.predicate == "linear_overlap")
    assert overlap.metadata["overlap_length_px"] == 40.0
    assert overlap.metadata["residual_px"] == 0.0
    assert overlap.metadata["exact_relation_not_admitted"] is True
    admission = ParseGraphAdmission().admit(result.graph)
    decision = next(item for item in admission.decisions if item.candidate_id == overlap.id)
    assert decision.decision is not AdmissionStatus.ADMITTED


def test_arc_detector_fits_open_circular_contour_but_rejects_full_circle_and_line():
    image = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(image, (128, 128), (70, 70), 0, 20, 160, 255, 2)
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    candidate = next(_fit_arc_contour(contour, image.shape) for contour in contours)

    assert candidate["sweep_deg"] == pytest.approx(141.9, abs=1.0)
    assert candidate["radius_px"] == pytest.approx(70.0, abs=2.0)

    full_circle = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(full_circle, (128, 128), 70, 255, 2)
    full_contours, _ = cv2.findContours(full_circle, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    assert all(_fit_arc_contour(contour, full_circle.shape) is None for contour in full_contours)


def test_arc_detector_suppresses_major_arc_candidates_from_a_supported_full_circle():
    image = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(image, (128, 128), (70, 70), 0, 0, 300, 255, 2)
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    circle = {"id": "vc_0", "center": [128, 128], "radius_px": 70, "edge_support": 0.9}
    detector = VisualPrimitiveDetector()

    supported = detector._detect_arcs(edges, [circle])
    unsupported = detector._detect_arcs(edges, [{**circle, "edge_support": 0.5}])

    assert not any(float(candidate["sweep_deg"]) > 240.0 for candidate in supported)
    assert any(float(candidate["sweep_deg"]) > 240.0 for candidate in unsupported)


def test_visual_detector_recovers_an_interior_filled_point_without_text_false_positive(tmp_path):
    image_path = Path(tmp_path) / "interior_point.png"
    image = np.full((240, 240), 255, dtype=np.uint8)
    cv2.line(image, (20, 120), (220, 120), 0, 2, cv2.LINE_AA)
    cv2.circle(image, (120, 120), 5, 0, -1, cv2.LINE_AA)
    cv2.putText(image, "A", (110, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, 0, 2, cv2.LINE_AA)
    assert cv2.imwrite(str(image_path), image)

    result = VisualPrimitiveDetector().detect(str(image_path))
    point_candidates = result["topology"]["point_candidates"]

    assert len(point_candidates) == 1
    assert point_candidates[0]["xy"] == pytest.approx([120.5, 119.5], abs=1.5)
    assert point_candidates[0]["score_kind"] == "filled_point_blob"

    parsed = TypedDiagramParserV0(detector=VisualPrimitiveDetector()).parse(
        ParserInput(id="interior_point", image=str(image_path), question="")
    )
    assert any(
        point.kind is ParseObjectKind.POINT
        and point.geometry.get("pixel") == pytest.approx([120.5, 119.5], abs=1.5)
        for point in parsed.graph.objects
    )
    assert parsed.graph.metadata["point_detection"]["num_candidates"] == 1
    point_on_line = next(item for item in parsed.graph.relations if item.predicate == "point_on_line")
    assert point_on_line.metadata["bounded"] is True
    ordered = next(item for item in parsed.graph.relations if item.predicate == "ordered_points")
    assert point_on_line.arguments[0].ref in {argument.ref for argument in ordered.arguments[1:]}


def test_visual_detector_does_not_turn_a_plain_line_into_a_point(tmp_path):
    image_path = Path(tmp_path) / "plain_line.png"
    image = np.full((240, 240), 255, dtype=np.uint8)
    cv2.line(image, (20, 120), (220, 120), 0, 2, cv2.LINE_AA)
    assert cv2.imwrite(str(image_path), image)

    result = VisualPrimitiveDetector().detect(str(image_path))

    assert result["topology"]["point_candidates"] == []


def test_typed_parser_binds_arc_only_with_unique_support_circle(tmp_path):
    image_path = Path(tmp_path) / "opaque.png"
    image_path.write_bytes(b"fixture")
    supported = TypedDiagramParserV0(detector=ArcCandidateDetector(support_circle=True)).parse(
        ParserInput(id="arc_supported", image=str(image_path), question="")
    )
    arc = next(item for item in supported.graph.objects if item.kind is ParseObjectKind.ARC)
    assert [reference.role for reference in arc.references] == ["circle", "start", "end"]
    assert arc.geometry["orientation"] == "cw"
    assert arc.metadata["image_orientation"] == "ccw"
    assert supported.graph.metadata["arc_grounding"]["num_bound_arcs"] == 1

    unsupported = TypedDiagramParserV0(detector=ArcCandidateDetector(support_circle=False)).parse(
        ParserInput(id="arc_unsupported", image=str(image_path), question="")
    )
    assert not any(item.kind is ParseObjectKind.ARC for item in unsupported.graph.objects)
    evidence = next(item for item in unsupported.graph.evidence if item.id == "evidence:arc_candidate")
    assert evidence.metadata["support_circle_status"] == "unresolved"
    assert unsupported.graph.metadata["arc_grounding"]["num_unresolved_support_circles"] == 1


def test_typed_parser_binds_unique_filled_arc_chord_region(tmp_path):
    image_path = Path(tmp_path) / "opaque.png"
    image_path.write_bytes(b"fixture")

    result = TypedDiagramParserV0(detector=FilledArcChordDetector()).parse(
        ParserInput(id="filled_arc_chord", image=str(image_path), question="")
    )

    region = next(item for item in result.graph.regions if item.metadata.get("boundary_kind") == "arc_chord")
    assert [reference.orientation for reference in region.boundary] == ["unknown", "forward"]
    assert region.boundary[1].ref == "segment:chord"
    assert not any(item.id.startswith("segment:curved-boundary:") for item in result.graph.objects)
    assert region.metadata["shading_status"] == "unknown"
    assert result.graph.metadata["face_extraction"]["num_curved_faces"] == 1
    fill_evidence = next(item for item in result.graph.evidence if item.id == "evidence:fill_segment")
    assert fill_evidence.precision is EvidencePrecision.BOUND
    assert fill_evidence.metadata["bound_curved_region_id"] == region.id
    scene = source_graph_to_render_scene(result.graph)
    region_primitive = next(item for item in scene.primitives if item.id == region.id)
    assert len(region_primitive.geometry["points"]) > 12
    assert region.id not in scene.metadata["unsupported_candidates"]


def test_typed_parser_binds_unique_filled_sector_region(tmp_path):
    image_path = Path(tmp_path) / "opaque.png"
    image_path.write_bytes(b"fixture")

    result = TypedDiagramParserV0(detector=FilledSectorDetector()).parse(
        ParserInput(id="filled_sector", image=str(image_path), question="")
    )

    region = next(item for item in result.graph.regions if item.metadata.get("boundary_kind") == "arc_radii")
    assert len(region.boundary) == 3
    assert [reference.ref for reference in region.boundary[1:]] == [
        "segment:end_radius",
        "segment:start_radius",
    ]
    assert region.metadata["shading_status"] == "unknown"
    assert region.metadata["center_point_id"]
    assert result.graph.metadata["face_extraction"]["num_curved_faces"] == 1
    assert result.graph.metadata["face_extraction"]["num_arc_radii_faces"] == 1
    assert not any(item.id.startswith("segment:curved-boundary:") for item in result.graph.objects)
    scene = source_graph_to_render_scene(result.graph)
    region_primitive = next(item for item in scene.primitives if item.id == region.id)
    assert len(region_primitive.geometry["points"]) > 12


def test_typed_parser_abstains_on_ambiguous_sector_supports(tmp_path):
    image_path = Path(tmp_path) / "opaque.png"
    image_path.write_bytes(b"fixture")

    result = TypedDiagramParserV0(detector=AmbiguousFilledSectorDetector()).parse(
        ParserInput(id="ambiguous_sector", image=str(image_path), question="")
    )

    assert not any(item.metadata.get("boundary_kind") == "arc_radii" for item in result.graph.regions)
    fill_evidence = next(item for item in result.graph.evidence if item.id == "evidence:fill_sector")
    assert fill_evidence.precision is EvidencePrecision.LOCALIZED


def test_arc_intersections_are_materialized_as_points_and_relations(tmp_path):
    image_path = Path(tmp_path) / "opaque.png"
    image_path.write_bytes(b"fixture")
    arc = {
        "id": "arc_candidate",
        "center": [100.0, 100.0],
        "radius_px": 60.0,
        "start": [160.0, 100.0],
        "end": [100.0, 40.0],
        "orientation": "cw",
        "sweep_deg": 90.0,
        "bbox": [38.0, 38.0, 162.0, 162.0],
        "residual_px": 0.8,
        "edge_support": 0.9,
        "circle_id": "arc_circle",
    }
    line = {"id": "cut", "p1": [40, 60], "p2": [160, 60], "angle_deg": 0.0}
    circles = [
        {"id": "arc_circle", "center": [100.0, 100.0], "radius_px": 60.0, "edge_support": 0.92},
        {"id": "other_circle", "center": [150.0, 100.0], "radius_px": 40.0, "edge_support": 0.92},
    ]
    topology = VisualPrimitiveDetector()._derive_topology([line], circles, [arc])

    assert len(topology["arc_line_intersections"]) == 1
    assert len(topology["arc_circle_intersections"]) == 1
    assert len(topology["arc_arc_intersections"]) == 0

    class Detector:
        def detect(self, image_path):
            return {
                "image_size": {"width": 200, "height": 200},
                "lines": [line],
                "circles": circles,
                "arcs": [arc],
                "fill_regions": [],
                "markers": [],
                "text_regions": [],
                "topology": topology,
            }

    result = TypedDiagramParserV0(detector=Detector()).parse(
        ParserInput(id="arc_intersections", image=str(image_path), question="")
    )
    arc_line = next(item for item in result.graph.relations if item.id == "relation:arc_line_intersection:0")
    arc_circle = next(item for item in result.graph.relations if item.id == "relation:arc_circle_intersection:0")
    ordered = next(item for item in result.graph.relations if item.id == "relation:ordered:cut")
    assert all(argument.role == "support" for argument in arc_line.arguments[:2])
    assert all(argument.role == "support" for argument in arc_circle.arguments[:2])
    assert len(ordered.arguments) == 5
    assert arc_line.arguments[2].ref in {argument.ref for argument in ordered.arguments[1:]}


def test_visual_curve_intersection_solver_distinguishes_tangency_and_finite_segments():
    detector = VisualPrimitiveDetector()

    assert detector._segment_circle_intersections([0, 3], [10, 3], [5, 0], 3) == ((5.0, 3.0),)
    assert detector._segment_circle_intersections([0, 0], [1, 0], [5, 0], 2) == ()
    assert detector._circle_circle_intersections([0, 0], 5, [10, 0], 5) == ((5.0, 0.0),)
    assert len(detector._circle_circle_intersections([0, 0], 5, [6, 0], 5)) == 2
    topology = detector._derive_topology(
        [
            {"id": "left", "p1": [0, 0], "p2": [10, 0]},
            {"id": "right", "p1": [5, 0], "p2": [15, 0]},
        ],
        [],
    )
    assert topology["line_overlaps"][0]["overlap_length_px"] == 5.0


def test_marker_detector_uses_curve_topology_points_for_exclusion():
    points = _topology_points(
        {
            "arc_endpoints": [{"xy": [10.0, 20.0]}],
            "arc_line_intersections": [{"xy": [30.0, 40.0]}],
            "arc_circle_intersections": [{"xy": [50.0, 60.0]}],
            "arc_arc_intersections": [{"xy": [70.0, 80.0]}],
        }
    )

    assert points == ((10.0, 20.0), (30.0, 40.0), (50.0, 60.0), (70.0, 80.0))


def test_visual_detector_merges_collinear_segments_with_opposite_directions():
    detector = VisualPrimitiveDetector()
    merged = detector._merge_collinear_lines(
        [
            {"id": "raw_right", "p1": [90, 20], "p2": [55, 20], "length_px": 35.0, "angle_deg": 180.0},
            {"id": "raw_left", "p1": [10, 20], "p2": [45, 20], "length_px": 35.0, "angle_deg": 0.0},
            {"id": "raw_bridge", "p1": [40, 20], "p2": [60, 20], "length_px": 20.0, "angle_deg": 0.0},
        ]
    )

    assert len(merged) == 1
    assert merged[0]["p1"] == [10, 20]
    assert merged[0]["p2"] == [90, 20]
    assert set(merged[0]["source_ids"]) == {"raw_left", "raw_bridge", "raw_right"}
    assert detector.configuration()["minimum_line_length_ratio"] == 0.1


def test_visual_detector_extracts_colored_fill_without_inventing_fill_from_black_lines():
    detector = VisualPrimitiveDetector()
    image = np.full((240, 240, 3), 255, dtype=np.uint8)
    polygon = np.array([[30, 200], [120, 30], [210, 200]], dtype=np.int32)
    cv2.fillPoly(image, [polygon], (254, 234, 219), cv2.LINE_AA)
    cv2.polylines(image, [polygon], True, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(image, "A", (108, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)

    fill_regions = detector._detect_fill_regions(image)

    assert len(fill_regions) == 1
    assert fill_regions[0]["appearance_semantics"] == "observed_fill_not_shading_label"
    assert fill_regions[0]["area_px"] > 10000
    assert len(fill_regions[0]["outer_contour"]) >= 3

    outline_only = np.full((240, 240, 3), 255, dtype=np.uint8)
    cv2.polylines(outline_only, [polygon], True, (20, 20, 20), 3, cv2.LINE_AA)
    assert detector._detect_fill_regions(outline_only) == []


def test_typed_parser_binds_only_unambiguous_fill_appearance_to_planar_face(tmp_path):
    image_path = tmp_path / "opaque.png"
    image_path.write_bytes(b"fixture")

    result = TypedDiagramParserV0(detector=FilledFaceDetector()).parse(
        ParserInput(id="filled_face", image=str(image_path), question="")
    )

    assert len(result.graph.regions) == 1
    region = result.graph.regions[0]
    assert region.metadata["fill_status"] == "observed"
    assert region.metadata["shading_status"] == "unknown"
    assert region.metadata["fill_candidate_id"] == "vf_face"
    bound_fill = next(item for item in result.graph.evidence if item.id == "evidence:vf_face")
    unmatched_fill = next(item for item in result.graph.evidence if item.id == "evidence:vf_unmatched")
    assert bound_fill.precision is EvidencePrecision.BOUND
    assert bound_fill.metadata["bound_face_id"] == region.id
    assert bound_fill.id in region.evidence_refs
    assert unmatched_fill.precision is EvidencePrecision.LOCALIZED
    assert unmatched_fill.metadata["bound_face_id"] is None
    assert result.graph.metadata["face_extraction"]["num_fill_bindings"] == 1


def test_typed_parser_keeps_interior_gap_as_unresolved_evidence(tmp_path):
    image_path = tmp_path / "opaque.png"
    image_path.write_bytes(b"fixture")

    result = TypedDiagramParserV0(detector=FilledFaceWithGapDetector()).parse(
        ParserInput(id="filled_face_with_gap", image=str(image_path), question="")
    )

    region = result.graph.regions[0]
    fill_evidence = next(item for item in result.graph.evidence if item.id == "evidence:vf_face")
    assert region.holes == ()
    assert region.metadata["interior_gap_count"] == 1
    assert region.metadata["interior_gap_status"] == "unresolved_boundary_candidate"
    assert fill_evidence.metadata["interior_gap_count"] == 1
    assert fill_evidence.metadata["interior_gap_status"] == "unresolved_boundary_candidate"
    assert fill_evidence.metadata["hole_semantics"] == "not_admitted_without_closed_parse_region_boundary"
    assert result.graph.metadata["face_extraction"]["num_interior_gap_candidates"] == 1
    assert result.graph.metadata["face_extraction"]["num_regions_with_unresolved_gaps"] == 1
    assert next(stage for stage in result.stages if stage["name"] == "fill_grounding")["status"] == "partial"


def test_typed_parser_links_gap_to_unique_closed_face(tmp_path):
    image_path = tmp_path / "opaque.png"
    image_path.write_bytes(b"fixture")

    result = TypedDiagramParserV0(detector=FilledFaceWithClosedGapDetector()).parse(
        ParserInput(id="filled_face_with_closed_gap", image=str(image_path), question="")
    )

    outer = next(region for region in result.graph.regions if region.metadata.get("fill_status") == "observed")
    assert len(result.graph.regions) == 2
    assert len(outer.holes) == 1
    hole = next(region for region in result.graph.regions if region.id == outer.holes[0])
    assert hole.metadata["region_role"] == "hole_boundary"
    assert outer.metadata["interior_gap_status"] == "grounded_closed_boundary"
    assert outer.metadata["hole_semantics"] == "grounded_closed_parse_region_boundary"
    fill_evidence = next(item for item in result.graph.evidence if item.id == "evidence:vf_face")
    assert fill_evidence.metadata["hole_region_ids"] == list(outer.holes)
    assert result.graph.metadata["face_extraction"]["num_hole_region_links"] == 1
    assert next(stage for stage in result.stages if stage["name"] == "fill_grounding")["status"] == "partial"

    admission = ParseGraphAdmission().admit(result.graph)
    assert outer.id in admission.admitted_ids
    assert hole.id in admission.admitted_ids


def test_shaded_area_target_uses_unique_fill_only_as_unadmitted_appearance_hypothesis(tmp_path):
    image_path = tmp_path / "opaque.png"
    image_path.write_bytes(b"fixture")
    result = TypedDiagramParserV0(detector=FilledFaceDetector()).parse(
        ParserInput(
            id="shaded_face_target",
            image=str(image_path),
            question="Find the area of the shaded region.",
        )
    )

    target = result.graph.targets[0]
    region = result.graph.regions[0]
    shaded_relation = next(item for item in result.graph.relations if item.predicate == "shaded_region")
    target_evidence = next(item for item in result.graph.evidence if item.id == "evidence:question_target")
    assert target.kind == "shadow_area"
    assert target.arguments[0].ref == region.id
    assert target.arguments[0].role == "region"
    assert target.origin is ParseOrigin.IMAGE_TEXT
    assert target.metadata["object_status"] == "appearance_hypothesis"
    assert target.metadata["semantic_shading_unverified"] is True
    assert target_evidence.precision is EvidencePrecision.LOCALIZED
    assert "evidence:vf_face" in target.evidence_refs
    assert [(item.role, item.ref) for item in shaded_relation.arguments] == [("region", region.id)]
    assert shaded_relation.origin is ParseOrigin.IMAGE_TEXT
    assert shaded_relation.metadata["semantic_status"] == "appearance_hypothesis"
    assert shaded_relation.id in target.dependencies

    admission = ParseGraphAdmission().admit(result.graph)
    decisions = {item.candidate_id: item for item in admission.decisions}
    assert decisions[shaded_relation.id].decision is AdmissionStatus.ABSTAINED
    assert decisions[shaded_relation.id].reason == "semantic_shading_unverified"
    assert decisions[target.id].decision is AdmissionStatus.ABSTAINED
    assert decisions[target.id].reason == "dependency_not_admitted"

    permitted = ParseGraphAdmission(
        ParseAdmissionPolicy(
            permit_unique_fill_shading=True,
            policy_id="test_unique_fill_shading",
        )
    ).admit(result.graph)
    permitted_decisions = {item.candidate_id: item for item in permitted.decisions}
    assert permitted_decisions[shaded_relation.id].decision is AdmissionStatus.ADMITTED
    assert permitted_decisions[shaded_relation.id].reason == "unique_fill_shading_verified"
    assert permitted_decisions[shaded_relation.id].metadata["passed"] is True
    assert permitted_decisions[target.id].decision is AdmissionStatus.ADMITTED
    assert permitted_decisions[target.id].reason == "shaded_target_grounded_by_verified_relation"

    decoy = replace(region, id="region:decoy")
    invalid_graph = replace(result.graph, regions=(*result.graph.regions, decoy))
    rejected = ParseGraphAdmission(
        ParseAdmissionPolicy(
            permit_unique_fill_shading=True,
            policy_id="test_non_unique_fill_shading",
        )
    ).admit(invalid_graph)
    rejected_decisions = {item.candidate_id: item for item in rejected.decisions}
    assert rejected_decisions[shaded_relation.id].decision is AdmissionStatus.REJECTED
    assert rejected_decisions[shaded_relation.id].reason == "unique_fill_shading_verification_failed"
    assert rejected_decisions[shaded_relation.id].metadata["checks"]["one_observed_fill_region"] is False
    assert rejected_decisions[target.id].reason == "dependency_not_admitted"


def test_shaded_area_target_abstains_when_multiple_filled_regions_are_visible(tmp_path):
    image_path = tmp_path / "opaque.png"
    image_path.write_bytes(b"fixture")
    result = TypedDiagramParserV0(detector=MultipleFilledFaceDetector()).parse(
        ParserInput(
            id="ambiguous_shaded_face_target",
            image=str(image_path),
            question="Find the area of the shaded region.",
        )
    )

    target = result.graph.targets[0]
    assert len([region for region in result.graph.regions if region.metadata.get("fill_status") == "observed"]) == 2
    assert not any(item.predicate == "shaded_region" for item in result.graph.relations)
    assert target.arguments == ()
    assert target.metadata["object_status"] == "unresolved"

    admission = ParseGraphAdmission().admit(result.graph)
    decision = next(item for item in admission.decisions if item.candidate_id == target.id)
    assert decision.decision is AdmissionStatus.ABSTAINED
    assert decision.reason == "target_object_unresolved"


def test_typed_parser_preserves_fill_evidence_when_face_inference_is_disabled(tmp_path):
    image_path = tmp_path / "opaque.png"
    image_path.write_bytes(b"fixture")

    result = TypedDiagramParserV0(
        detector=FilledFaceDetector(),
        infer_planar_faces=False,
    ).parse(ParserInput(id="fill_evidence_only", image=str(image_path), question=""))

    assert result.graph.regions == ()
    fill_evidence = [item for item in result.graph.evidence if item.id.startswith("evidence:vf_")]
    assert len(fill_evidence) == 2
    assert all(item.precision is EvidencePrecision.LOCALIZED for item in fill_evidence)
    assert result.graph.metadata["face_extraction"]["num_fill_bindings"] == 0
    assert next(stage for stage in result.stages if stage["name"] == "fill_grounding")["status"] == "partial"


def test_typed_parser_rejects_malformed_fill_candidate(tmp_path):
    class MalformedFillDetector(FilledFaceDetector):
        def detect(self, image_path: str):
            result = super().detect(image_path)
            result["fill_regions"][0]["outer_contour"] = [[23, 23], [177, 23]]
            return result

    image_path = tmp_path / "opaque.png"
    image_path.write_bytes(b"fixture")

    with pytest.raises(InvalidGeometryInput, match="outer contour requires at least three points"):
        TypedDiagramParserV0(detector=MalformedFillDetector()).parse(
            ParserInput(id="malformed_fill", image=str(image_path), question="")
        )


def test_visual_marker_detector_requires_an_explicit_right_angle_glyph():
    gray = np.full((200, 200), 255, dtype=np.uint8)
    gray[159:162, 20:181] = 20
    gray[20:161, 19:22] = 20
    lines = [
        {"id": "vl_h", "p1": [20, 160], "p2": [180, 160], "length_px": 160.0, "angle_deg": 0.0},
        {"id": "vl_v", "p1": [20, 20], "p2": [20, 160], "length_px": 140.0, "angle_deg": 90.0},
    ]
    topology = {"line_intersections": [{"line_ids": ["vl_h", "vl_v"], "xy": [20, 160]}]}
    detector = VisualMarkerDetector(minimum_size_px=8)

    assert detector.detect(gray, lines, topology) == []

    gray[149:152, 20:31] = 20
    gray[150:161, 29:32] = 20
    markers = detector.detect(gray, lines, topology)

    assert len(markers) == 1
    assert markers[0]["kind"] == "right_angle"
    assert markers[0]["semantic_relation"] == "perpendicular"
    assert markers[0]["line_ids"] == ["vl_h", "vl_v"]
    assert min(markers[0]["stroke_support"]) >= 0.78


def test_visual_marker_detector_groups_parallel_chevrons_and_equal_length_ticks():
    gray = np.full((240, 240), 255, dtype=np.uint8)
    for y in (60, 180):
        cv2.line(gray, (20, y), (220, y), 20, 3, cv2.LINE_AA)
        cv2.line(gray, (70, y - 6), (70, y + 6), 20, 2, cv2.LINE_AA)
        cv2.line(gray, (135, y - 4), (145, y), 20, 2, cv2.LINE_AA)
        cv2.line(gray, (145, y), (135, y + 4), 20, 2, cv2.LINE_AA)
    lines = [
        {"id": "top", "p1": [20, 60], "p2": [220, 60], "length_px": 200.0, "angle_deg": 0.0},
        {"id": "bottom", "p1": [20, 180], "p2": [220, 180], "length_px": 200.0, "angle_deg": 0.0},
    ]
    topology = {
        "endpoints": [{"xy": point} for point in ([20, 60], [220, 60], [20, 180], [220, 180])],
        "line_intersections": [],
    }

    markers = VisualMarkerDetector().detect(gray, lines, topology)

    assert [(marker["kind"], marker["line_ids"]) for marker in markers] == [
        ("equal_length_tick", ["top"]),
        ("parallel", ["top"]),
        ("equal_length_tick", ["bottom"]),
        ("parallel", ["bottom"]),
    ]
    assert {marker["relation_group_id"] for marker in markers if marker["kind"] == "parallel"} == {"vg_parallel_0"}
    assert {marker["relation_group_id"] for marker in markers if marker["kind"] == "equal_length_tick"} == {
        "vg_equal_length_0"
    }


def test_visual_marker_detector_recognizes_angle_arc_without_inventing_equal_angle():
    gray = np.full((200, 200), 255, dtype=np.uint8)
    cv2.line(gray, (20, 100), (180, 100), 20, 3, cv2.LINE_AA)
    cv2.line(gray, (100, 100), (160, 48), 20, 3, cv2.LINE_AA)
    cv2.ellipse(gray, (100, 100), (16, 16), 0, -40, 0, 20, 2, cv2.LINE_AA)
    lines = [
        {"id": "horizontal", "p1": [20, 100], "p2": [180, 100], "length_px": 160.0, "angle_deg": 0.0},
        {"id": "diagonal", "p1": [100, 100], "p2": [160, 48], "length_px": 79.4, "angle_deg": -40.9},
    ]
    topology = {"line_intersections": [{"line_ids": ["horizontal", "diagonal"], "xy": [100, 100]}]}

    markers = VisualMarkerDetector(minimum_size_px=8).detect(gray, lines, topology)

    assert len(markers) == 1
    assert markers[0]["kind"] == "angle_arc"
    assert markers[0]["line_ids"] == ["horizontal", "diagonal"]
    assert "relation_group_id" not in markers[0]
    assert "semantic_relation" not in markers[0]


def test_equal_angle_marker_grouping_permits_one_shared_support_ray():
    detector = VisualMarkerDetector()
    markers = [
        {"kind": "angle_arc", "line_ids": ["outer_0", "bisector"], "angle_span_deg": 45.0, "size_px": 12},
        {"kind": "angle_arc", "line_ids": ["bisector", "outer_1"], "angle_span_deg": 45.0, "size_px": 12},
    ]

    detector._assign_relation_groups(markers, {})

    assert {marker["relation_group_id"] for marker in markers} == {"vg_equal_angle_0"}
    assert {marker["semantic_relation"] for marker in markers} == {"equal_angle"}


def test_explicit_right_angle_marker_is_bound_and_admitted(tmp_path):
    image_path = tmp_path / "opaque.png"
    image_path.write_bytes(b"fixture")
    result = TypedDiagramParserV0(detector=RightAngleMarkerDetector()).parse(
        ParserInput(id="right_angle_marker", image=str(image_path), question="")
    )

    marker = next(item for item in result.graph.objects if item.kind is ParseObjectKind.MARKER)
    perpendicular = next(item for item in result.graph.relations if item.predicate == "perpendicular")
    marker_evidence = next(item for item in result.graph.evidence if item.id == marker.evidence_refs[0])

    assert marker_evidence.precision is EvidencePrecision.BOUND
    assert marker.metadata["marker_kind"] == "right_angle"
    assert perpendicular.score_kind == "explicit_right_angle_marker"
    assert perpendicular.dependencies == (marker.id,)
    assert result.graph.metadata["suppressed_visual_relation_ids"]
    report = ParseGraphAdmission().admit(result.graph)
    assert marker.id in report.admitted_ids
    assert perpendicular.id in report.admitted_ids


def test_grouped_visual_markers_create_relations_with_glyph_dependencies(tmp_path):
    image_path = tmp_path / "opaque.png"
    image_path.write_bytes(b"fixture")
    result = TypedDiagramParserV0(detector=GroupedMarkerDetector()).parse(
        ParserInput(id="grouped_markers", image=str(image_path), question="")
    )

    markers = [item for item in result.graph.objects if item.kind is ParseObjectKind.MARKER]
    explicit = [item for item in result.graph.relations if item.score_kind.startswith("explicit_")]

    assert len(markers) == 5
    assert {item.metadata["marker_kind"] for item in markers} == {
        "angle_arc",
        "equal_length_tick",
        "parallel",
    }
    assert {item.predicate for item in explicit} == {"parallel", "equal_length"}
    assert all(len(item.dependencies) == 2 for item in explicit)
    assert all(set(item.dependencies) <= {marker.id for marker in markers} for item in explicit)
    angle_marker = next(item for item in markers if item.metadata["marker_kind"] == "angle_arc")
    assert not any(angle_marker.id in relation.dependencies for relation in explicit)
    report = ParseGraphAdmission().admit(result.graph)
    assert {item.id for item in markers} <= set(report.admitted_ids)
    assert {item.id for item in explicit} <= set(report.admitted_ids)


def test_same_line_ticks_bind_known_topology_subsegments_and_reconstruct_equal_length(tmp_path):
    image_path = tmp_path / "opaque.png"
    image_path.write_bytes(b"fixture")
    result = TypedDiagramParserV0(detector=SameLineEqualLengthDetector()).parse(
        ParserInput(id="same_line_equal_parts", image=str(image_path), question="")
    )

    subsegments = [
        item
        for item in result.graph.objects
        if item.kind is ParseObjectKind.SEGMENT and item.score_kind == "topology_partition_interval"
    ]
    markers = [item for item in result.graph.objects if item.kind is ParseObjectKind.MARKER]
    equal_length = next(item for item in result.graph.relations if item.predicate == "equal_length")

    assert len(subsegments) == 2
    assert {item.metadata["marked_segment_id"] for item in markers} == {item.id for item in subsegments}
    assert {argument.ref for argument in equal_length.arguments} == {item.id for item in subsegments}
    admission = ParseGraphAdmission().admit(result.graph)
    assert equal_length.id in admission.admitted_ids
    reconstruction = ParseGraphReconstructor().reconstruct(
        admission,
        mode=ReconstructionMode.PIXEL_WITNESS,
    )
    assert reconstruction.state is not None
    assert reconstruction.candidate_outcomes[equal_length.id] == "materialized"


def test_same_line_ticks_without_known_split_do_not_invent_equal_subsegments(tmp_path):
    image_path = tmp_path / "opaque.png"
    image_path.write_bytes(b"fixture")
    result = TypedDiagramParserV0(detector=SameLineEqualLengthDetector(include_split=False)).parse(
        ParserInput(id="same_line_unsplit_ticks", image=str(image_path), question="")
    )

    markers = [item for item in result.graph.objects if item.kind is ParseObjectKind.MARKER]
    assert {item.metadata["marked_segment_id"] for item in markers} == {"segment:vl_main"}
    assert not any(item.score_kind == "topology_partition_interval" for item in result.graph.objects)
    assert not any(item.predicate == "equal_length" for item in result.graph.relations)


def test_grouped_angle_arcs_ground_ordered_angles_and_reconstruct_equal_angle(tmp_path):
    image_path = tmp_path / "opaque.png"
    image_path.write_bytes(b"fixture")
    result = TypedDiagramParserV0(detector=EqualAngleMarkerDetector()).parse(
        ParserInput(id="equal_angle_markers", image=str(image_path), question="")
    )

    angle_markers = [
        item
        for item in result.graph.objects
        if item.kind is ParseObjectKind.MARKER and item.metadata["marker_kind"] == "angle_arc"
    ]
    equal_angle = next(item for item in result.graph.relations if item.predicate == "equal_angle")

    assert len(angle_markers) == 2
    assert all(marker.metadata["angle_grounding_status"] == "resolved" for marker in angle_markers)
    assert [argument.role for argument in equal_angle.arguments] == [
        "angle_0_arm_0",
        "angle_0_vertex",
        "angle_0_arm_1",
        "angle_1_arm_0",
        "angle_1_vertex",
        "angle_1_arm_1",
    ]
    admission = ParseGraphAdmission().admit(result.graph)
    assert equal_angle.id in admission.admitted_ids
    reconstruction = ParseGraphReconstructor().reconstruct(
        admission,
        mode=ReconstructionMode.PIXEL_WITNESS,
    )
    assert reconstruction.state is not None
    assert reconstruction.candidate_outcomes[equal_angle.id] == "materialized"
    state_relation = next(item for item in reconstruction.state.relations if item["id"] == equal_angle.id)
    assert state_relation["type"] == "equal_angle"
    assert len(state_relation["angles"]) == 2


def test_grouped_angle_arcs_without_ordered_arms_remain_marker_evidence(tmp_path):
    class UngroundedEqualAngleDetector(EqualAngleMarkerDetector):
        def detect(self, image_path: str):
            result = super().detect(image_path)
            for marker in result["markers"]:
                marker.pop("ray_signs")
            return result

    image_path = tmp_path / "opaque.png"
    image_path.write_bytes(b"fixture")
    result = TypedDiagramParserV0(detector=UngroundedEqualAngleDetector()).parse(
        ParserInput(id="ungrounded_equal_angle_markers", image=str(image_path), question="")
    )

    angle_markers = [
        item
        for item in result.graph.objects
        if item.kind is ParseObjectKind.MARKER and item.metadata["marker_kind"] == "angle_arc"
    ]
    assert len(angle_markers) == 2
    assert all(marker.metadata["angle_grounding_status"] == "unresolved" for marker in angle_markers)
    assert not any(item.predicate == "equal_angle" for item in result.graph.relations)


def test_shared_ray_equal_angle_markers_compile_verified_angle_bisector(tmp_path):
    image_path = tmp_path / "opaque.png"
    image_path.write_bytes(b"fixture")
    result = TypedDiagramParserV0(detector=AngleBisectorMarkerDetector()).parse(
        ParserInput(id="angle_bisector_markers", image=str(image_path), question="")
    )

    relations = {item.predicate: item for item in result.graph.relations if item.score_kind != "pixel_angle_residual"}
    equal_angle = relations["equal_angle"]
    angle_bisector = relations["angle_bisector"]
    assert angle_bisector.origin is ParseOrigin.COMPILED
    assert angle_bisector.dependencies == (equal_angle.id,)
    assert angle_bisector.metadata["mode"] == "internal"
    assert next(argument.ref for argument in angle_bisector.arguments if argument.role == "line") == "segment:bisector"

    admission = ParseGraphAdmission().admit(result.graph)
    assert angle_bisector.id in admission.admitted_ids
    reconstruction = ParseGraphReconstructor().reconstruct(
        admission,
        mode=ReconstructionMode.PIXEL_WITNESS,
    )
    assert reconstruction.state is not None
    assert reconstruction.candidate_outcomes[angle_bisector.id] == "materialized"
    assert reconstruction.verification["valid"] is True
    state_relation = next(item for item in reconstruction.state.relations if item["id"] == angle_bisector.id)
    assert state_relation["type"] == "angle_bisector"
    assert state_relation["line"] == "segment:bisector"


def test_explicit_segment_target_depends_on_two_admitted_point_bindings(tmp_path):
    image_path = Path(tmp_path) / "opaque.png"
    image_path.write_bytes(b"fixture")
    result = TypedDiagramParserV0(
        detector=NamedSegmentDetector(),
        label_recognizer=NamedSegmentRecognizer(),
    ).parse(ParserInput(id="named_segment", image=str(image_path), question="Find the length of AB."))

    target = result.graph.targets[0]
    bindings = [item for item in result.graph.relations if item.predicate == "binds"]

    assert target.arguments[0].ref == "segment:vl_ab"
    assert set(target.dependencies) == {item.id for item in bindings}
    assert target.metadata["object_status"] == "resolved"
    report = ParseGraphAdmission().admit(result.graph)
    decision = next(item for item in report.decisions if item.candidate_id == target.id)
    assert decision.reason == "evidence_policy_satisfied"
    assert target.id in report.admitted_ids


def test_explicit_question_quantity_uses_safe_ast_and_binding_dependencies(tmp_path):
    image_path = Path(tmp_path) / "opaque.png"
    image_path.write_bytes(b"fixture")
    result = TypedDiagramParserV0(
        detector=NamedSegmentDetector(),
        label_recognizer=NamedSegmentRecognizer(),
    ).parse(
        ParserInput(
            id="named_segment_quantity",
            image=str(image_path),
            question=r"Given AB = \frac{3}{2}. Find the length of AB.",
        )
    )

    quantity = result.graph.quantities[0]
    bindings = [item for item in result.graph.relations if item.predicate == "binds"]
    assert quantity.arguments[0].ref == "segment:vl_ab"
    assert str(quantity.expression.to_sympy()) == "3/2"
    assert set(quantity.dependencies) == {item.id for item in bindings}
    report = ParseGraphAdmission().admit(result.graph)
    decision = next(item for item in report.decisions if item.candidate_id == quantity.id)
    assert decision.reason == "evidence_policy_satisfied"
    assert quantity.id in report.admitted_ids


def test_unsupported_question_expression_is_preserved_as_unresolved_evidence(tmp_path):
    image_path = Path(tmp_path) / "opaque.png"
    image_path.write_bytes(b"fixture")
    result = TypedDiagramParserV0(
        detector=NamedSegmentDetector(),
        label_recognizer=NamedSegmentRecognizer(),
    ).parse(
        ParserInput(
            id="unsafe_quantity",
            image=str(image_path),
            question="Given AB = sqrt(1).__class__. Find the length of AB.",
        )
    )

    assert result.graph.quantities == ()
    assert result.graph.metadata["question_quantity_issues"][0]["status"] == "invalid_expression"
    evidence = next(item for item in result.graph.evidence if item.id == "evidence:question_quantity:0")
    assert evidence.precision is EvidencePrecision.LOCALIZED


def test_point_label_grounding_preserves_equal_distance_ambiguity():
    label_evidence = ParseEvidence(
        id="evidence_label_a",
        source_id="source_grounding",
        kind=ParseEvidenceKind.IMAGE_REGION,
        precision=EvidencePrecision.BOUND,
        bbox=(9.0, 9.0, 11.0, 11.0),
        raw_content="A",
    )
    left_evidence = ParseEvidence(
        id="evidence_left",
        source_id="source_grounding",
        kind=ParseEvidenceKind.PIXEL_POINT,
        precision=EvidencePrecision.LOCALIZED,
        point=(0.0, 10.0),
    )
    right_evidence = ParseEvidence(
        id="evidence_right",
        source_id="source_grounding",
        kind=ParseEvidenceKind.PIXEL_POINT,
        precision=EvidencePrecision.LOCALIZED,
        point=(20.0, 10.0),
    )
    objects = (
        _point("point_left", left_evidence),
        _point("point_right", right_evidence),
        ParseObject(
            id="text_a",
            kind=ParseObjectKind.TEXT,
            origin=ParseOrigin.IMAGE,
            confidence=0.9,
            score_kind="ocr_score",
            evidence_refs=(label_evidence.id,),
            visible_name="A",
            geometry={"bbox": list(label_evidence.bbox)},
        ),
    )

    bindings, conflicts = ground_point_labels(
        objects,
        {item.id: item for item in (label_evidence, left_evidence, right_evidence)},
        max_distance=12.0,
        ambiguity_margin=0.0,
    )

    assert len(bindings) == 2
    assert len(conflicts) == 1
    assert set(conflicts[0].members) == {item.id for item in bindings}


def test_point_label_grounding_marks_duplicate_clear_names_as_a_conflict():
    point_evidence = ParseEvidence(
        id="evidence_duplicate_point",
        source_id="source_grounding",
        kind=ParseEvidenceKind.PIXEL_POINT,
        precision=EvidencePrecision.LOCALIZED,
        point=(20.0, 20.0),
    )
    point = _point("duplicate_point", point_evidence)
    labels = []
    evidence = {point_evidence.id: point_evidence}
    for index, bbox in enumerate(((10.0, 10.0, 14.0, 14.0), (26.0, 10.0, 30.0, 14.0))):
        label_evidence = ParseEvidence(
            id=f"evidence_duplicate_label_{index}",
            source_id="source_grounding",
            kind=ParseEvidenceKind.IMAGE_REGION,
            precision=EvidencePrecision.BOUND,
            bbox=bbox,
            raw_content="A",
        )
        labels.append(
            ParseObject(
                id=f"duplicate_label_{index}",
                kind=ParseObjectKind.TEXT,
                origin=ParseOrigin.IMAGE_TEXT,
                confidence=0.95,
                score_kind="fixture",
                evidence_refs=(label_evidence.id,),
                visible_name="A",
                geometry={"bbox": list(bbox)},
            )
        )
        evidence[label_evidence.id] = label_evidence

    bindings, conflicts = ground_point_labels((point, *labels), evidence, max_distance=30.0, ambiguity_margin=1.0)

    assert len(bindings) == 2
    assert len(conflicts) == 1
    assert conflicts[0].reason == "point_name_bindings_must_be_one_to_one"
    assert set(conflicts[0].members) == {item.id for item in bindings}


def test_typed_parser_ground_line_name_to_support_interior(tmp_path):
    image_path = tmp_path / "opaque.png"
    image_path.write_bytes(b"fixture")
    result = TypedDiagramParserV0(
        detector=NamedLineDetector(),
        label_recognizer=NamedLineRecognizer(),
    ).parse(ParserInput(id="named_line", image=str(image_path), question=""))

    line_bindings = [
        item
        for item in result.graph.relations
        if item.predicate == "binds" and item.metadata.get("binding_kind") == "line_name"
    ]

    assert len(line_bindings) == 1
    assert [(item.role, item.ref) for item in line_bindings[0].arguments] == [
        ("label", "text:vt_line"),
        ("object", "segment:vl_ab"),
    ]
    assert line_bindings[0].metadata["label"] == "k"
    assert result.graph.metadata["line_label_grounding"]["num_bindings"] == 1
    assert next(stage for stage in result.stages if stage["name"] == "line_label_grounding")["output_count"] == 1
    admission = ParseGraphAdmission().admit(result.graph)
    assert line_bindings[0].id in admission.admitted_ids


def _point(point_id, evidence):
    return ParseObject(
        id=point_id,
        kind=ParseObjectKind.POINT,
        origin=ParseOrigin.IMAGE,
        confidence=0.9,
        score_kind="fixture",
        evidence_refs=(evidence.id,),
        geometry={"pixel": list(evidence.point)},
    )
