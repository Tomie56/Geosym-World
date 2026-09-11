import sympy as sp
import pytest

from geosym_world.errors import InvalidGeometryInput
from geosym_world.parser import Clause, ClauseGraph, materialize_exact_point_state


def test_materializer_builds_formal_points_and_reports_unsupported_clauses():
    graph = ClauseGraph(
        "clauses_1", "problem_1",
        (
            Clause("point_a", "object.point", ("A",), 1.0,
                   value={"exact_coords": [0, {"operator": "sqrt", "operands": [{"operator": "literal", "value": "2"}]}]},
                   status="admitted"),
            Clause("point_b", "object.point", ("B",), 1.0,
                   value={"exact_coords": [1, 2]}, status="admitted"),
            Clause("point_c", "object.point", ("C",), 1.0,
                   value={"exact_coords": [0, 3]}, status="admitted"),
            Clause("line_1", "object.segment", ("AB",), 1.0,
                   value={"p1": "A", "p2": "B"}, status="admitted"),
            Clause("line_2", "object.line", ("AB",), 1.0,
                   value={"p1": "A", "p2": "B"}, status="admitted"),
            Clause("circle_1", "object.circle", ("A",), 1.0,
                   value={"center": "A", "exact_radius": {"operator": "literal", "value": "3"}},
                   status="admitted"),
            Clause("polygon_1", "object.polygon", ("A", "B"), 1.0,
                   value={"vertices": ["A", "B", "C"]}, status="admitted"),
            Clause("region_1", "region", ("A", "B", "C"), 1.0,
                   metadata={"boundary": ["A", "B", "C"], "holes": [], "components": []},
                   status="admitted"),
            Clause("region_union", "region", ("region_1",), 1.0,
                   metadata={"boundary": [], "holes": [], "components": ["region_1"], "operation": "union"},
                   status="admitted"),
            Clause("region_with_hole", "region", ("A", "B", "C", "region_1"), 1.0,
                   metadata={"boundary": ["A", "B", "C"], "holes": ["region_1"], "components": []},
                   status="admitted"),
            Clause("relation_1", "parallel", ("AB", "CD"), 1.0, status="admitted"),
            Clause("relation_2", "parallel", ("line_1", "line_2"), 1.0, status="admitted"),
            Clause("relation_equal_length", "equal_length", ("line_1", "line_2"), 1.0, status="admitted"),
            Clause("relation_3", "point_on_line", ("A", "line_1"), 1.0, status="admitted"),
            Clause("relation_4", "point_on_circle", ("B", "circle_1"), 1.0, status="admitted"),
            Clause("relation_5", "midpoint", ("A", "line_1"), 1.0, status="admitted"),
            Clause("relation_tangent", "tangent", ("A", "line_2", "circle_1"), 1.0,
                   provenance=("visual_1",), metadata={"parser_clause_id": "spoofed"}, status="admitted"),
        ),
    )

    result = materialize_exact_point_state(graph)

    assert result.state.points["A"].coords == (sp.Integer(0), sp.sqrt(2))
    assert result.state.segments["line_1"].start == "A"
    assert result.state.lines["line_2"].source_points == ("A", "B")
    assert result.state.circles["circle_1"].radius == sp.Integer(3)
    assert result.state.polygons["polygon_1"].vertices == ("A", "B", "C")
    assert result.state.regions["region_1"].boundary == ("A", "B", "C")
    assert result.state.composite_regions["region_union"].components == ("region_1",)
    assert result.state.regions["region_with_hole"].holes == ("region_1",)
    assert result.state.relations[0]["type"] == "parallel"
    assert result.state.relations[1]["type"] == "equal_length"
    assert result.state.relations[2]["type"] == "point_on_line"
    assert result.state.relations[3]["type"] == "point_on_circle"
    assert result.state.relations[4]["type"] == "midpoint"
    assert result.state.relations[5]["type"] == "tangent"
    assert result.state.relations[5]["metadata"]["parser_clause_id"] == "relation_tangent"
    assert result.state.relations[5]["metadata"]["provenance"] == ["visual_1"]
    assert result.unsupported_clause_ids == ("relation_1",)


def test_materializer_builds_ray_arc_and_target():
    graph = ClauseGraph(
        "clauses_2", "problem_2",
        (
            Clause("point_a", "object.point", ("A",), 1.0, value={"exact_coords": [0, 0]}, status="admitted"),
            Clause("point_b", "object.point", ("B",), 1.0, value={"exact_coords": [3, 0]}, status="admitted"),
            Clause("point_c", "object.point", ("C",), 1.0, value={"exact_coords": [0, 3]}, status="admitted"),
            Clause("circle_1", "object.circle", ("O",), 1.0,
                   value={"center": "A", "exact_radius": 3}, status="admitted"),
            Clause("ray_1", "object.ray", ("A",), 1.0,
                   value={"start": "A", "through": "B"}, status="admitted"),
            Clause("arc_1", "object.arc", ("A", "B"), 1.0,
                   value={"circle": "circle_1", "start": "B", "end": "C"}, status="admitted"),
            Clause("target_1", "target", ("circle_1",), 1.0,
                   value={"type": "radius", "object": "circle_1",
                          "expression": {"operator": "literal", "value": "3"}}, status="admitted"),
        ),
    )

    result = materialize_exact_point_state(graph)

    assert result.unsupported_clause_ids == ()
    assert result.state.rays["ray_1"].through == "B"
    assert result.state.arcs["arc_1"].circle == "circle_1"
    assert result.state.target.type == "radius"
    assert result.state.target.metadata["expression"]["value"] == "3"


def test_materializer_is_independent_of_clause_order():
    graph = ClauseGraph(
        "clauses_3", "problem_3",
        (
            Clause("target_1", "target", ("circle_1",), 1.0,
                   value={"type": "radius", "object": "circle_1"}, status="admitted"),
            Clause("segment_1", "object.segment", ("AB",), 1.0,
                   value={"p1": "A", "p2": "B"}, status="admitted"),
            Clause("circle_1", "object.circle", ("O",), 1.0,
                   value={"center": "A", "exact_radius": 3}, status="admitted"),
            Clause("point_b", "object.point", ("B",), 1.0, value={"exact_coords": [3, 0]}, status="admitted"),
            Clause("point_a", "object.point", ("A",), 1.0, value={"exact_coords": [0, 0]}, status="admitted"),
        ),
    )

    result = materialize_exact_point_state(graph)

    assert result.unsupported_clause_ids == ()
    assert result.state.segments["segment_1"].start == "A"
    assert result.state.target.object == "circle_1"


def test_materializer_preserves_foot_of_perpendicular_relation():
    graph = ClauseGraph(
        "clauses_4", "problem_4",
        (
            Clause("point_a", "object.point", ("A",), 1.0, value={"exact_coords": [0, 0]}, status="admitted"),
            Clause("point_b", "object.point", ("B",), 1.0, value={"exact_coords": [4, 0]}, status="admitted"),
            Clause("point_c", "object.point", ("C",), 1.0, value={"exact_coords": [1, 2]}, status="admitted"),
            Clause("line_1", "object.segment", ("AB",), 1.0,
                   value={"p1": "A", "p2": "B"}, status="admitted"),
            Clause("foot_1", "foot_of_perpendicular", ("A", "C", "line_1"), 1.0, status="admitted"),
        ),
    )

    result = materialize_exact_point_state(graph)

    assert result.unsupported_clause_ids == ()
    assert result.state.relations[-1]["from"] == "C"


def test_materializer_preserves_similarity_relation_between_polygons():
    graph = ClauseGraph(
        "clauses_5", "problem_5",
        (
            Clause("a", "object.point", ("A",), 1.0, value={"exact_coords": [0, 0]}, status="admitted"),
            Clause("b", "object.point", ("B",), 1.0, value={"exact_coords": [1, 0]}, status="admitted"),
            Clause("c", "object.point", ("C",), 1.0, value={"exact_coords": [0, 1]}, status="admitted"),
            Clause("d", "object.point", ("D",), 1.0, value={"exact_coords": [2, 0]}, status="admitted"),
            Clause("e", "object.point", ("E",), 1.0, value={"exact_coords": [2, 2]}, status="admitted"),
            Clause("f", "object.point", ("F",), 1.0, value={"exact_coords": [0, 2]}, status="admitted"),
            Clause("triangle_1", "object.polygon", ("A",), 1.0, value={"vertices": ["A", "B", "C"]}, status="admitted"),
            Clause("triangle_2", "object.polygon", ("D",), 1.0, value={"vertices": ["D", "E", "F"]}, status="admitted"),
            Clause("similarity_1", "similar", ("triangle_1", "triangle_2"), 1.0, status="admitted"),
        ),
    )

    result = materialize_exact_point_state(graph)

    assert result.unsupported_clause_ids == ()
    assert result.state.relations[-1]["polygons"] == ["triangle_1", "triangle_2"]


def test_materializer_preserves_explicit_intersection_point():
    graph = ClauseGraph(
        "clauses_6", "problem_6",
        (
            Clause("point_a", "object.point", ("A",), 1.0, value={"exact_coords": [0, 0]}, status="admitted"),
            Clause("point_b", "object.point", ("B",), 1.0, value={"exact_coords": [2, 2]}, status="admitted"),
            Clause("point_c", "object.point", ("C",), 1.0, value={"exact_coords": [0, 2]}, status="admitted"),
            Clause("point_d", "object.point", ("D",), 1.0, value={"exact_coords": [2, 0]}, status="admitted"),
            Clause("point_x", "object.point", ("X",), 1.0, value={"exact_coords": [1, 1]}, status="admitted"),
            Clause("line_ab", "object.segment", ("AB",), 1.0, value={"p1": "A", "p2": "B"}, status="admitted"),
            Clause("line_cd", "object.segment", ("CD",), 1.0, value={"p1": "C", "p2": "D"}, status="admitted"),
            Clause("intersection_1", "intersection", ("line_ab", "line_cd", "X"), 1.0, status="admitted"),
        ),
    )

    result = materialize_exact_point_state(graph)

    assert result.unsupported_clause_ids == ()
    assert result.state.relations[-1]["objects"] == ["line_ab", "line_cd"]
    assert result.state.relations[-1]["points"] == ["X"]


def test_materializer_rejects_cyclic_clause_dependencies():
    graph = ClauseGraph(
        "clauses_7", "problem_7",
        (
            Clause("first", "object.point", ("A",), 1.0,
                   value={"exact_coords": [0, 0]}, dependencies=("second",), status="admitted"),
            Clause("second", "object.point", ("B",), 1.0,
                   value={"exact_coords": [1, 0]}, dependencies=("first",), status="admitted"),
        ),
    )

    with pytest.raises(InvalidGeometryInput, match="cycle"):
        materialize_exact_point_state(graph)
