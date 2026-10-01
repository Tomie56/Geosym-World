import json
import sys
from dataclasses import replace

import pytest
import sympy as sp

from geosym_world.construction.builder import ConstructionBuilder
from geosym_world.parser.constraint_completion import CompletionStatus, build_constraint_completion_plan
from geosym_world.tools.seed_synthesis_from_parser_v0 import main as seed_main
from geosym_world.solver.closure import FactClosureSolver
from geosym_world.schema import (
    MathExpression,
    ParseGraph,
    ParseObject,
    ParseObjectKind,
    ParseOrigin,
    ParseQuantity,
    ParseReference,
    ParseRelation,
    ParseTarget,
)
from geosym_world.verifier.trace import TraceVerifier


def _point(point_id: str) -> ParseObject:
    return ParseObject(
        id=point_id,
        kind=ParseObjectKind.POINT,
        origin=ParseOrigin.IMAGE_TEXT,
        confidence=0.95,
        score_kind="test_point",
        visible_name=point_id,
    )


def _segment(segment_id: str, start: str, end: str) -> ParseObject:
    return ParseObject(
        id=segment_id,
        kind=ParseObjectKind.SEGMENT,
        origin=ParseOrigin.IMAGE_TEXT,
        confidence=0.9,
        score_kind="test_segment",
        visible_name=segment_id,
        references=(ParseReference("start", start), ParseReference("end", end)),
    )


def _exact_point(point_id: str, x: str, y: str) -> ParseObject:
    return ParseObject(
        id=point_id,
        kind=ParseObjectKind.POINT,
        origin=ParseOrigin.IMAGE_TEXT,
        confidence=0.95,
        score_kind="test_point",
        visible_name=point_id,
        geometry={
            "exact_coords": [
                {"operator": "literal", "value": x, "operands": []},
                {"operator": "literal", "value": y, "operands": []},
            ]
        },
    )


def _length(quantity_id: str, segment_id: str, value: str) -> ParseQuantity:
    return ParseQuantity(
        id=quantity_id,
        kind="length",
        arguments=(ParseReference("object", segment_id),),
        expression=MathExpression("literal", value=value),
        origin=ParseOrigin.TEXT,
        confidence=0.95,
        score_kind="test_length",
        raw_text=value,
    )


def _right_triangle_graph(*, include_lengths: bool = True) -> ParseGraph:
    objects = (
        _point("A"), _point("B"), _point("C"),
        _segment("AB", "A", "B"), _segment("BC", "B", "C"),
        ParseObject(
            id="tri_ABC",
            kind=ParseObjectKind.POLYGON,
            origin=ParseOrigin.IMAGE_TEXT,
            confidence=0.85,
            score_kind="test_polygon",
            visible_name="tri_ABC",
            references=(
                ParseReference("vertex_0", "A"),
                ParseReference("vertex_1", "B"),
                ParseReference("vertex_2", "C"),
            ),
        ),
    )
    quantities = (_length("q_AB", "AB", "3"), _length("q_BC", "BC", "4")) if include_lengths else ()
    relations = (
        ParseRelation(
            id="perp_1",
            predicate="perpendicular",
            arguments=(ParseReference("left", "AB"), ParseReference("right", "BC")),
            origin=ParseOrigin.TEXT,
            confidence=0.95,
            score_kind="test_perpendicular",
        ),
    )
    targets = (
        ParseTarget(
            id="target_1",
            kind="area",
            mode="value",
            arguments=(ParseReference("object", "tri_ABC"),),
            origin=ParseOrigin.TEXT,
            confidence=0.9,
            score_kind="test_target",
        ),
    )
    return ParseGraph(
        id="graph_1",
        source_id="source_1",
        objects=objects,
        relations=relations,
        quantities=quantities,
        targets=targets,
    )


def _sss_triangle_graph() -> ParseGraph:
    base = _right_triangle_graph()
    objects = list(base.objects)
    objects.append(_segment("CA", "C", "A"))
    quantities = list(base.quantities)
    quantities.append(_length("q_CA", "CA", "5"))
    return ParseGraph(
        id="graph_sss",
        source_id="source_sss",
        objects=tuple(objects),
        quantities=tuple(quantities),
        targets=base.targets,
    )


def _derived_point_graph(kind: str) -> ParseGraph:
    points = [_exact_point("A", "0", "0"), _exact_point("B", "4", "0")]
    objects: list[ParseObject] = [*points, _segment("AB", "A", "B")]
    if kind == "midpoint":
        objects.append(_point("M"))
        relation = ParseRelation(
            id="rel_midpoint",
            predicate="midpoint",
            arguments=(
                ParseReference("point", "M"),
                ParseReference("of_0", "A"),
                ParseReference("of_1", "B"),
            ),
            origin=ParseOrigin.TEXT,
            confidence=0.9,
            score_kind="test_relation",
        )
    elif kind == "ratio":
        objects.append(_point("R"))
        relation = ParseRelation(
            id="rel_ratio",
            predicate="point_on_segment_ratio",
            arguments=(ParseReference("point", "R"), ParseReference("segment", "AB")),
            origin=ParseOrigin.TEXT,
            confidence=0.9,
            score_kind="test_relation",
            metadata={"ratio": {"operator": "div", "value": None, "operands": [
                {"operator": "literal", "value": "1", "operands": []},
                {"operator": "literal", "value": "3", "operands": []},
            ]}},
        )
    elif kind == "foot":
        objects.extend([_exact_point("C", "2", "3"), ParseObject(
            id="line_AB",
            kind=ParseObjectKind.LINE,
            origin=ParseOrigin.IMAGE_TEXT,
            confidence=0.9,
            score_kind="test_line",
            visible_name="base",
            references=(ParseReference("start", "A"), ParseReference("end", "B")),
        ), _point("H")])
        relation = ParseRelation(
            id="rel_foot",
            predicate="foot_of_perpendicular",
            arguments=(
                ParseReference("point", "H"),
                ParseReference("from", "C"),
                ParseReference("line", "line_AB"),
            ),
            origin=ParseOrigin.TEXT,
            confidence=0.9,
            score_kind="test_relation",
        )
    else:
        raise AssertionError(kind)
    return ParseGraph(
        id=f"graph_{kind}",
        source_id=f"source_{kind}",
        objects=tuple(objects),
        relations=(relation,),
    )


def _rectangle_extension_graph() -> ParseGraph:
    points = tuple(_point(name) for name in "ABCDEF")
    segments = tuple(
        _segment(name, name[0], name[1])
        for name in ("AB", "BC", "CD", "DA", "AC", "CE", "AE")
    )
    polygon = ParseObject(
        id="rect_ABCD",
        kind=ParseObjectKind.POLYGON,
        origin=ParseOrigin.TEXT,
        confidence=1.0,
        score_kind="test_rectangle",
        visible_name="ABCD",
        references=tuple(ParseReference(f"vertex_{index}", name) for index, name in enumerate("ABCD")),
    )
    relations = (
        ParseRelation(
            id="rel_rectangle",
            predicate="rectangle",
            arguments=(ParseReference("object", "rect_ABCD"),),
            origin=ParseOrigin.TEXT,
            confidence=1.0,
            score_kind="test_rectangle",
        ),
        ParseRelation(
            id="rel_perpendicular",
            predicate="perpendicular",
            arguments=(ParseReference("left", "AC"), ParseReference("right", "CE")),
            origin=ParseOrigin.TEXT,
            confidence=1.0,
            score_kind="test_perpendicular",
        ),
        ParseRelation(
            id="rel_AFE",
            predicate="collinear",
            arguments=tuple(ParseReference("point", name) for name in "AFE"),
            origin=ParseOrigin.TEXT,
            confidence=1.0,
            score_kind="test_collinear",
        ),
        ParseRelation(
            id="rel_CFD",
            predicate="collinear",
            arguments=tuple(ParseReference("point", name) for name in "CFD"),
            origin=ParseOrigin.TEXT,
            confidence=1.0,
            score_kind="test_collinear",
        ),
    )
    return ParseGraph(
        id="graph_rectangle_extension",
        source_id="source_rectangle_extension",
        objects=points + segments + (polygon,),
        relations=relations,
        quantities=(
            _length("q_AB", "AB", "16"),
            _length("q_BC", "BC", "12"),
            _length("q_CE", "CE", "15"),
        ),
    )


@pytest.mark.parametrize("predicate", ["point_on_segment", "point_on_line", "ordered_points"])
def test_intersection_completion_accepts_equivalent_semantic_supports(predicate):
    graph = _rectangle_extension_graph()
    relations = []
    for relation in graph.relations:
        if relation.predicate != "collinear":
            relations.append(relation)
            continue
        start, point, end = [argument.ref for argument in relation.arguments]
        if predicate == "ordered_points":
            arguments = (ParseReference("support", start + end), *(ParseReference(f"point_{index}", name) for index, name in enumerate((start, point, end))))
        else:
            arguments = (ParseReference("point", point), ParseReference("support", start + end))
        relations.append(replace(relation, predicate=predicate, arguments=arguments))
    expected = build_constraint_completion_plan(graph)
    actual = build_constraint_completion_plan(replace(graph, relations=tuple(relations)))
    assert actual.status is CompletionStatus.DETERMINISTIC_COMPLETION
    actual_points = {step["id"]: step["coord"] for step in actual.trace if step["op"] == "add_point"}
    expected_points = {step["id"]: step["coord"] for step in expected.trace if step["op"] == "add_point"}
    assert actual_points == expected_points


def test_numeric_constraints_produce_replayable_non_degenerate_trace():
    plan = build_constraint_completion_plan(_right_triangle_graph())

    assert plan.status is CompletionStatus.DETERMINISTIC_COMPLETION
    points = {step["id"]: step["coord"] for step in plan.trace if step["op"] == "add_point"}
    assert points == {"A": ["0", "0"], "B": ["3", "0"], "C": ["3", "4"]}
    assert any(step["op"] == "set_target" and step["object"] == "tri_ABC" for step in plan.trace)

    verification = TraceVerifier().verify_trace(plan.trace, execute=False)
    assert verification.valid, verification.to_dict()
    state = ConstructionBuilder().execute(plan.trace)
    assert sp.simplify(state.polygons["tri_ABC"].vertices == ("A", "B", "C"))


def test_rectangle_constraints_produce_exact_state_with_derived_intersection():
    plan = build_constraint_completion_plan(_rectangle_extension_graph())

    assert plan.status is CompletionStatus.DETERMINISTIC_COMPLETION
    assert plan.metadata["method"] == "exact_rectangle_constraints_v1"
    state = ConstructionBuilder().execute(plan.trace)
    assert state.points["A"].coords == (0, 0)
    assert state.points["B"].coords == (0, 16)
    assert state.points["C"].coords == (12, 16)
    assert state.points["D"].coords == (12, 0)
    assert state.points["E"].coords == (24, 7)
    assert state.points["F"].coords == (12, sp.Rational(7, 2))
    assert state.polygons["ABCD"].vertices == ("A", "B", "C", "D")


def test_theorem_only_graph_is_not_given_fabricated_coordinates():
    plan = build_constraint_completion_plan(_right_triangle_graph(include_lengths=False))

    assert plan.status is CompletionStatus.UNSUPPORTED
    assert plan.trace == ()
    assert plan.diagnostics == ("no_explicit_numeric_length_constraints",)


def test_sss_constraints_emit_two_solver_compatible_mirror_branches():
    plan = build_constraint_completion_plan(_sss_triangle_graph())

    assert plan.status is CompletionStatus.BRANCHING_COMPLETION
    assert len(plan.branches) == 2
    assert plan.to_dict()["branch_count"] == 2
    apex_coordinates = []
    for branch in plan.branches:
        verification = TraceVerifier().verify_trace(branch, execute=False)
        assert verification.valid, verification.to_dict()
        state = ConstructionBuilder().execute(branch)
        facts = FactClosureSolver().solve(state)
        area_facts = facts.find("area", "tri_ABC")
        assert area_facts and area_facts[0].value == "6"
        apex_coordinates.append(state.points["C"].coords)
    assert apex_coordinates[0][1] == -apex_coordinates[1][1]


def test_plan_checksum_changes_when_trace_changes():
    plan = build_constraint_completion_plan(_right_triangle_graph())
    assert plan.to_dict()["checksum"] == plan.checksum()


def test_exact_scaffold_completes_midpoint_with_formal_operation():
    plan = build_constraint_completion_plan(_derived_point_graph("midpoint"))

    assert plan.status is CompletionStatus.DETERMINISTIC_COMPLETION
    assert plan.metadata["method"] == "derived_construction_v1"
    assert any(step["op"] == "midpoint" for step in plan.trace)
    state = ConstructionBuilder().execute(plan.trace)
    assert state.points["M"].coords == (sp.Integer(2), sp.Integer(0))
    assert TraceVerifier().verify_trace(plan.trace).valid


def test_exact_scaffold_completes_ratio_point_with_numeric_expression():
    plan = build_constraint_completion_plan(_derived_point_graph("ratio"))

    assert plan.status is CompletionStatus.DETERMINISTIC_COMPLETION
    state = ConstructionBuilder().execute(plan.trace)
    assert state.points["R"].coords == (sp.Rational(4, 3), sp.Integer(0))
    assert TraceVerifier().verify_trace(plan.trace).valid


def test_exact_scaffold_completes_perpendicular_foot_on_line():
    plan = build_constraint_completion_plan(_derived_point_graph("foot"))

    assert plan.status is CompletionStatus.DETERMINISTIC_COMPLETION
    assert any(step["op"] == "add_line" and step["id"] == "base" for step in plan.trace)
    state = ConstructionBuilder().execute(plan.trace)
    assert state.points["H"].coords == (sp.Integer(2), sp.Integer(0))
    assert TraceVerifier().verify_trace(plan.trace).valid


def test_seed_tool_emits_direct_completion_for_replayable_parse_graph(tmp_path, monkeypatch):
    source = tmp_path / "parse_graph.json"
    source.write_text(json.dumps({"parse_graph": _right_triangle_graph().to_dict()}), encoding="utf-8")
    output = tmp_path / "seeds.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "seed_synthesis_from_parser_v0",
        "--semantic", str(source),
        "--output", str(output),
    ])

    seed_main()

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["family"] == "parser_constraint_completion"
    assert rows[0]["metadata"]["constraint_completion_plan"]["status"] == "deterministic_completion"
    assert any(step["op"] == "set_target" for step in rows[0]["trace"])


def test_seed_tool_exports_each_sss_completion_branch(tmp_path, monkeypatch):
    source = tmp_path / "sss_parse_graph.json"
    source.write_text(json.dumps({"parse_graph": _sss_triangle_graph().to_dict()}), encoding="utf-8")
    output = tmp_path / "sss_seeds.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "seed_synthesis_from_parser_v0",
        "--semantic", str(source),
        "--output", str(output),
    ])

    seed_main()

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == ["parser_completion_source_sss_b0", "parser_completion_source_sss_b1"]
    assert all(row["metadata"]["constraint_completion_plan"]["status"] == "branching_completion" for row in rows)
