"""Unit tests for structured intersection helpers."""

import pytest

from geosym_world.construction import ConstructionBuilder
from geosym_world.errors import DegenerateGeometryError, InvalidGeometryInput
from geosym_world.kernel.geometry2d.analytic import coords
from geosym_world.kernel.geometry2d.intersections import (
    arc_arc_intersection_result_2d,
    circle_circle_intersection_result_2d,
    circle_arc_intersection_result_2d,
    directed_arc_angle_2d,
    diagnose_collinearity_2d,
    diagnose_line_circle_tangency_2d,
    linear_arc_intersection_result_2d,
    linear_linear_intersection_result_2d,
    line_line_intersection_result_2d,
    point_on_arc_domain_2d,
    segment_segment_intersection_result_2d,
    select_intersection_points,
)
from geosym_world.solver import FactClosureSolver
from geosym_world.verifier import TraceVerifier


def test_line_line_unique_intersection():
    result = line_line_intersection_result_2d(coords((0, 0)), coords((1, 1)), coords((0, 1)), coords((1, -1)))
    assert result.status == "unique"
    assert [[str(value) for value in point] for point in result.points] == [["1/2", "1/2"]]
    assert result.contact_dimension == 0
    assert result.to_dict()["topology"] == {"contact_dimension": 0, "contact_kind": "point", "exact": True}


def test_zero_direction_line_is_explicitly_degenerate():
    result = line_line_intersection_result_2d(coords((0, 0)), coords((0, 0)), coords((0, 1)), coords((1, 0)))
    assert result.status == "degenerate"
    assert result.contact_kind == "invalid"


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


def test_intersection_selection_uses_explicit_reference_point():
    result = circle_circle_intersection_result_2d(coords((0, 0)), 5, coords((6, 0)), 5)
    assert str(select_intersection_points(result, "nearest", reference=coords((3, 10)))[0][1]) == "4"
    assert str(select_intersection_points(result, "farthest", reference=coords((3, 10)))[0][1]) == "-4"
    with pytest.raises(ValueError, match="requires a reference point"):
        select_intersection_points(result, "nearest")


def test_circle_intersection_construction_selects_nearest_reference():
    trace = [
        {"op": "add_point", "id": "O1", "coord": [0, 0]},
        {"op": "add_point", "id": "O2", "coord": [6, 0]},
        {"op": "add_point", "id": "R", "coord": [3, 10]},
        {"op": "add_circle", "id": "c1", "center": "O1", "radius": 5},
        {"op": "add_circle", "id": "c2", "center": "O2", "radius": 5},
        {"op": "intersection_circle_circle", "id": "X", "circle1": "c1", "circle2": "c2", "select": "nearest", "reference": "R"},
    ]
    assert TraceVerifier().verify_trace(trace).valid
    state = ConstructionBuilder().execute(trace)
    assert tuple(str(value) for value in state.points["X"].coords) == ("3", "4")

    invalid = [*trace[:-1], {**trace[-1], "reference": "missing"}]
    report = TraceVerifier().verify_trace(invalid)
    assert not report.valid
    assert "trace.unknown_reference" in {issue.code for issue in report.issues}


def test_near_degeneracy_is_diagnostic_only():
    collinear = diagnose_collinearity_2d(coords((0, 0)), coords((10, 0)), coords((5, "1e-12")), tolerance=1e-9)
    assert not collinear.exact
    assert collinear.classification == "approximately_degenerate"
    tangent = diagnose_line_circle_tangency_2d(coords((-2, "1.000000000001")), coords((1, 0)), coords((0, 0)), 1, tolerance=1e-9)
    assert not tangent.exact
    assert tangent.classification == "approximately_degenerate"
    exact_result = line_line_intersection_result_2d(coords((0, 0)), coords((10, 0)), coords((5, "1e-12")), coords((0, 1)))
    assert exact_result.status == "unique"


def test_directed_arc_domain_distinguishes_minor_and_major_sweeps():
    center = coords((0, 0))
    start = coords((2, 0))
    end = coords((0, 2))

    assert str(directed_arc_angle_2d(center, start, end, orientation="ccw")) == "pi/2"
    assert str(directed_arc_angle_2d(center, start, end, orientation="cw")) == "3*pi/2"
    assert point_on_arc_domain_2d(coords((0, 2)), center, 2, start, end, orientation="ccw", central_angle="pi/2")
    assert not point_on_arc_domain_2d(coords((-2, 0)), center, 2, start, end, orientation="ccw", central_angle="pi/2")
    assert point_on_arc_domain_2d(coords((-2, 0)), center, 2, start, end, orientation="cw", central_angle="3*pi/2")


def test_arc_state_rejects_off_circle_endpoint_and_false_sweep():
    common = [
        {"op": "add_point", "id": "O", "coord": [0, 0]},
        {"op": "add_point", "id": "A", "coord": [2, 0]},
        {"op": "add_point", "id": "B", "coord": [0, 2]},
        {"op": "add_point", "id": "C", "coord": [1, 1]},
        {"op": "add_circle", "id": "circle_O", "center": "O", "radius": 2},
    ]
    with pytest.raises(InvalidGeometryInput, match="endpoints must lie on its circle"):
        ConstructionBuilder().execute([*common, {"op": "add_arc", "id": "bad", "circle": "circle_O", "start": "A", "end": "C", "central_angle": "pi/2"}])
    with pytest.raises(InvalidGeometryInput, match="central angle does not match"):
        ConstructionBuilder().execute([*common, {"op": "add_arc", "id": "bad", "circle": "circle_O", "start": "A", "end": "B", "central_angle": "pi"}])


def test_linear_arc_filters_circle_solution_outside_arc_domain():
    result = linear_arc_intersection_result_2d(
        coords((0, 0)),
        coords((1, 0)),
        coords((0, 0)),
        2,
        coords((2, 0)),
        coords((0, 2)),
        orientation="ccw",
        central_angle="pi/2",
    )
    assert result.status == "unique"
    assert [[str(value) for value in point] for point in result.points] == [["2", "0"]]
    assert result.metadata["arc_filtered"] == 1


def test_circle_arc_filters_one_of_two_circle_intersections():
    result = circle_arc_intersection_result_2d(
        coords((3, 0)),
        5,
        coords((0, 0)),
        5,
        coords((5, 0)),
        coords((0, 5)),
        orientation="ccw",
        central_angle="pi/2",
    )
    assert result.status == "unique"
    assert str(result.points[0][1]) == "sqrt(91)/2"


def test_arc_arc_filters_both_arc_domains_and_reports_overlap():
    crossing = arc_arc_intersection_result_2d(
        coords((0, 0)),
        5,
        coords((5, 0)),
        coords((0, 5)),
        coords((3, 0)),
        5,
        coords((8, 0)),
        coords((-2, 0)),
        first_central_angle="pi/2",
        second_central_angle="pi",
    )
    assert crossing.status == "unique"

    overlap = arc_arc_intersection_result_2d(
        coords((0, 0)),
        2,
        coords((2, 0)),
        coords((0, 2)),
        coords((0, 0)),
        2,
        coords((2, 0)),
        coords((0, 2)),
        first_central_angle="pi/2",
        second_central_angle="pi/2",
    )
    assert overlap.status == "overlap"
    assert overlap.multiplicity == "infinite"


def test_collinear_segment_overlap_has_explicit_interval_endpoints():
    result = segment_segment_intersection_result_2d(coords((0, 0)), coords((3, 0)), coords((1, 0)), coords((4, 0)))
    assert result.status == "overlap"
    assert result.multiplicity == "infinite"
    assert [[str(value) for value in point] for point in result.points] == [["1", "0"], ["3", "0"]]


def test_collinear_segments_touch_at_one_endpoint():
    result = segment_segment_intersection_result_2d(coords((0, 0)), coords((2, 0)), coords((2, 0)), coords((4, 0)))
    assert result.status == "unique"
    assert [[str(value) for value in point] for point in result.points] == [["2", "0"]]


def test_line_circle_construction_selects_named_upper_solution():
    state = ConstructionBuilder().execute(
        [
            {"op": "add_point", "id": "O", "coord": [0, 0]},
            {"op": "add_point", "id": "A", "coord": [0, -2]},
            {"op": "add_point", "id": "B", "coord": [0, 2]},
            {"op": "add_line", "id": "vertical", "points": ["A", "B"]},
            {"op": "add_circle", "id": "circle_O", "center": "O", "radius": 2},
            {"op": "intersection_line_circle", "id": "U", "line": "vertical", "circle": "circle_O", "select": "upper"},
        ]
    )
    assert tuple(str(value) for value in state.points["U"].coords) == ("0", "2")
    relation = state.relations[-1]
    assert relation["selection"] == "upper"
    assert relation["result"]["status"] == "two_points"


def test_parallel_line_intersection_fails_without_mutating_state():
    builder = ConstructionBuilder()
    builder.execute(
        [
            {"op": "add_point", "id": "A", "coord": [0, 0]},
            {"op": "add_point", "id": "B", "coord": [1, 0]},
            {"op": "add_point", "id": "C", "coord": [0, 1]},
            {"op": "add_point", "id": "D", "coord": [1, 1]},
            {"op": "add_line", "id": "AB", "points": ["A", "B"]},
            {"op": "add_line", "id": "CD", "points": ["C", "D"]},
        ]
    )
    checksum = builder.state.checksum()
    with pytest.raises(DegenerateGeometryError, match="status=parallel"):
        builder.execute([{"op": "intersection_line_line", "id": "X", "line1": "AB", "line2": "CD"}])
    assert builder.state.checksum() == checksum
    assert "X" not in builder.state.points


def test_segment_domain_rejects_infinite_line_intersection():
    result = linear_linear_intersection_result_2d(
        coords((0, 0)),
        coords((1, 0)),
        coords((2, -1)),
        coords((0, 1)),
        first_domain="segment",
        second_domain="line",
    )
    assert result.status == "disjoint"


def test_opposing_collinear_rays_have_bounded_overlap():
    result = linear_linear_intersection_result_2d(
        coords((0, 0)),
        coords((1, 0)),
        coords((3, 0)),
        coords((-1, 0)),
        first_domain="ray",
        second_domain="ray",
    )
    assert result.status == "overlap"
    assert result.metadata["overlap_interval"] == ["0", "3"]


def test_segment_circle_rejects_extension_only_points():
    builder = ConstructionBuilder()
    builder.execute(
        [
            {"op": "add_point", "id": "O", "coord": [0, 0]},
            {"op": "add_point", "id": "A", "coord": [3, 0]},
            {"op": "add_point", "id": "B", "coord": [4, 0]},
            {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
            {"op": "add_circle", "id": "circle_O", "center": "O", "radius": 2},
        ]
    )
    checksum = builder.state.checksum()
    with pytest.raises(DegenerateGeometryError, match="status=disjoint"):
        builder.execute([{"op": "intersection_line_circle", "ids": ["P", "Q"], "line": "AB", "circle": "circle_O"}])
    assert builder.state.checksum() == checksum


def test_ray_circle_keeps_only_forward_intersection():
    state = ConstructionBuilder().execute(
        [
            {"op": "add_point", "id": "O", "coord": [0, 0]},
            {"op": "add_point", "id": "A", "coord": [0, 0]},
            {"op": "add_point", "id": "B", "coord": [1, 0]},
            {"op": "add_ray", "id": "ray_AB", "start": "A", "through": "B"},
            {"op": "add_circle", "id": "circle_O", "center": "O", "radius": 2},
            {"op": "intersection_line_circle", "id": "P", "line": "ray_AB", "circle": "circle_O"},
        ]
    )
    assert tuple(str(value) for value in state.points["P"].coords) == ("2", "0")


def test_solver_does_not_claim_segment_extension_intersection():
    state = ConstructionBuilder().execute(
        [
            {"op": "add_point", "id": "A", "coord": [0, 0]},
            {"op": "add_point", "id": "B", "coord": [1, 0]},
            {"op": "add_point", "id": "C", "coord": [2, -1]},
            {"op": "add_point", "id": "D", "coord": [2, 1]},
            {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
            {"op": "add_segment", "id": "CD", "points": ["C", "D"]},
        ]
    )
    facts = FactClosureSolver(profile="full").solve(state)
    assert not facts.find("line_intersection", "AB|CD")


def test_solver_point_on_ray_respects_forward_direction():
    state = ConstructionBuilder().execute(
        [
            {"op": "add_point", "id": "A", "coord": [0, 0]},
            {"op": "add_point", "id": "B", "coord": [1, 0]},
            {"op": "add_point", "id": "C", "coord": [-1, 0]},
            {"op": "add_ray", "id": "ray_AB", "start": "A", "through": "B"},
        ]
    )
    facts = FactClosureSolver(profile="full").solve(state)
    assert facts.find("point_on_ray", "B|ray_AB")
    assert not facts.find("point_on_ray", "C|ray_AB")


def test_verifier_accepts_ray_as_line_like_intersection_input():
    report = TraceVerifier().verify_trace(
        [
            {"op": "add_point", "id": "A", "coord": [0, 0]},
            {"op": "add_point", "id": "B", "coord": [1, 0]},
            {"op": "add_point", "id": "C", "coord": [2, -1]},
            {"op": "add_point", "id": "D", "coord": [2, 1]},
            {"op": "add_ray", "id": "ray_AB", "start": "A", "through": "B"},
            {"op": "add_line", "id": "CD", "points": ["C", "D"]},
            {"op": "intersection_line_line", "id": "X", "line1": "ray_AB", "line2": "CD"},
        ]
    )
    assert report.valid


def test_line_arc_construction_names_only_arc_domain_solution():
    state = ConstructionBuilder().execute(
        [
            {"op": "add_point", "id": "O", "coord": [0, 0]},
            {"op": "add_point", "id": "A", "coord": [2, 0]},
            {"op": "add_point", "id": "B", "coord": [0, 2]},
            {"op": "add_point", "id": "L", "coord": [-3, 0]},
            {"op": "add_point", "id": "R", "coord": [3, 0]},
            {"op": "add_circle", "id": "circle_O", "center": "O", "radius": 2},
            {"op": "add_arc", "id": "arc_AB", "circle": "circle_O", "start": "A", "end": "B", "central_angle": "pi/2"},
            {"op": "add_line", "id": "axis", "points": ["L", "R"]},
            {"op": "intersection_line_arc", "id": "X", "line": "axis", "arc": "arc_AB"},
        ]
    )
    assert tuple(str(value) for value in state.points["X"].coords) == ("2", "0")
    assert state.relations[-1]["result"]["metadata"]["arc_filtered"] == 1
    assert TraceVerifier().verify_state(state).valid
    facts = FactClosureSolver(profile="full").solve(state)
    assert facts.find("point_on_arc", "X|arc_AB")
    assert facts.find("curve_intersection", "axis|arc_AB")


def test_circle_arc_construction_selects_valid_upper_intersection():
    state = ConstructionBuilder().execute(
        [
            {"op": "add_point", "id": "O", "coord": [0, 0]},
            {"op": "add_point", "id": "C", "coord": [3, 0]},
            {"op": "add_point", "id": "A", "coord": [5, 0]},
            {"op": "add_point", "id": "B", "coord": [0, 5]},
            {"op": "add_circle", "id": "base", "center": "O", "radius": 5},
            {"op": "add_circle", "id": "other", "center": "C", "radius": 5},
            {"op": "add_arc", "id": "arc_AB", "circle": "base", "start": "A", "end": "B", "central_angle": "pi/2"},
            {"op": "intersection_circle_arc", "id": "X", "circle": "other", "arc": "arc_AB"},
        ]
    )
    assert str(state.points["X"].coords[1]) == "sqrt(91)/2"
    assert TraceVerifier().verify_state(state).valid


def test_arc_arc_construction_and_static_reference_validation():
    trace = [
        {"op": "add_point", "id": "O1", "coord": [0, 0]},
        {"op": "add_point", "id": "O2", "coord": [3, 0]},
        {"op": "add_point", "id": "A1", "coord": [5, 0]},
        {"op": "add_point", "id": "B1", "coord": [-5, 0]},
        {"op": "add_point", "id": "A2", "coord": [8, 0]},
        {"op": "add_point", "id": "B2", "coord": [-2, 0]},
        {"op": "add_circle", "id": "c1", "center": "O1", "radius": 5},
        {"op": "add_circle", "id": "c2", "center": "O2", "radius": 5},
        {"op": "add_arc", "id": "a1", "circle": "c1", "start": "A1", "end": "B1", "central_angle": "pi"},
        {"op": "add_arc", "id": "a2", "circle": "c2", "start": "A2", "end": "B2", "central_angle": "pi"},
        {"op": "intersection_arc_arc", "id": "X", "arc1": "a1", "arc2": "a2"},
    ]
    report = TraceVerifier().verify_trace(trace)
    assert report.valid
    state = ConstructionBuilder().execute(trace)
    assert str(state.points["X"].coords[1]) == "sqrt(91)/2"
    assert TraceVerifier().verify_state(state).valid

    invalid = [*trace[:-1], {"op": "intersection_arc_arc", "id": "X", "arc1": "a1", "arc2": "missing"}]
    invalid_report = TraceVerifier().verify_trace(invalid)
    assert not invalid_report.valid
    assert "trace.unknown_reference" in {issue.code for issue in invalid_report.issues}


def test_arc_intersection_rejection_rolls_back_state():
    builder = ConstructionBuilder()
    builder.execute(
        [
            {"op": "add_point", "id": "O", "coord": [0, 0]},
            {"op": "add_point", "id": "A", "coord": [2, 0]},
            {"op": "add_point", "id": "B", "coord": [0, 2]},
            {"op": "add_point", "id": "L", "coord": [-3, -2]},
            {"op": "add_point", "id": "R", "coord": [3, -2]},
            {"op": "add_circle", "id": "circle_O", "center": "O", "radius": 2},
            {"op": "add_arc", "id": "arc_AB", "circle": "circle_O", "start": "A", "end": "B", "central_angle": "pi/2"},
            {"op": "add_segment", "id": "below", "points": ["L", "R"]},
        ]
    )
    checksum = builder.state.checksum()
    with pytest.raises(DegenerateGeometryError, match="status=disjoint"):
        builder.execute([{"op": "intersection_line_arc", "id": "X", "line": "below", "arc": "arc_AB"}])
    assert builder.state.checksum() == checksum
    assert "X" not in builder.state.points
