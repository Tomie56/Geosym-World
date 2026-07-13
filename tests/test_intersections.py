"""Unit tests for structured intersection helpers."""

from geosym_world_geometry.geometry import coords
from geosym_world_geometry.intersections import (
    circle_circle_intersection_result_2d,
    line_line_intersection_result_2d,
    segment_segment_intersection_result_2d,
    select_intersection_points,
)


def test_line_line_unique_intersection():
    result = line_line_intersection_result_2d(coords((0, 0)), coords((1, 1)), coords((0, 1)), coords((1, -1)))
    assert result.status == "unique"
    assert [[str(value) for value in point] for point in result.points] == [["1/2", "1/2"]]


def test_segment_segment_rejects_extension_only_intersection():
    result = segment_segment_intersection_result_2d(coords((0, 0)), coords((1, 0)), coords((2, -1)), coords((2, 1)))
    assert result.status == "disjoint"


def test_circle_circle_selection_upper_lower():
    result = circle_circle_intersection_result_2d(coords((0, 0)), 5, coords((6, 0)), 5)
    assert result.status == "two_points"
    upper = select_intersection_points(result, "upper")[0]
    lower = select_intersection_points(result, "lower")[0]
    assert str(upper[1]) == "4"
    assert str(lower[1]) == "-4"
