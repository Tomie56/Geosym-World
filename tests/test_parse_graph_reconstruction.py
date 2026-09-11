import json
from dataclasses import replace

import pytest
import sympy as sp

from geosym_world.errors import InvalidGeometryInput
from geosym_world.parser import (
    ParserInput,
    ParseGraphReconstructor,
    ParseRedrawPipeline,
    ReconstructionMode,
    ReconstructionStatus,
    RedrawMode,
    ResynthesisTransform,
    TypedParseResult,
    VerifiedParseWorkflow,
    summarize_workflow_results,
)
from geosym_world.parser.adapters.hypothesis_record import HypothesisRecordAdapter
from geosym_world.rendering import source_graph_to_render_scene
from geosym_world.schema import (
    BoundaryReference,
    EvidencePrecision,
    FactScope,
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
)
from geosym_world.verifier import ParseGraphAdmission


def test_exact_reconstruction_uses_typed_expressions_and_keeps_source_facts_separate():
    graph = _graph(exact=True)
    admission = ParseGraphAdmission().admit(graph)

    report = ParseGraphReconstructor().reconstruct(admission)

    assert report.status is ReconstructionStatus.EXACT
    assert report.state is not None
    assert report.state.points["point_b"].coords == (sp.sqrt(2), sp.Rational(1, 2))
    assert report.state.segments["segment_ab"].start == "point_a"
    assert report.state.target.object == "segment_ab"
    assert report.source_facts.find("length", "segment_ab")[0].scope is FactScope.SOURCE_ASSERTION
    assert all(fact.scope is FactScope.REALIZATION_FACT for fact in report.realization_facts)
    assert report.to_dict()["checksum"]


def test_pixel_reconstruction_is_explicitly_a_numeric_witness():
    admission = ParseGraphAdmission().admit(_graph(exact=False))

    report = ParseGraphReconstructor().reconstruct(admission, mode=ReconstructionMode.PIXEL_WITNESS)

    assert report.status is ReconstructionStatus.NUMERIC_WITNESS
    assert report.state.points["point_a"].coords == (sp.Float(10.0), sp.Float(80.0))
    assert report.state.points["point_a"].metadata["fact_scope"] == FactScope.REALIZATION_FACT.value


def test_exact_mode_never_falls_back_to_pixel_coordinates():
    admission = ParseGraphAdmission().admit(_graph(exact=False))

    report = ParseGraphReconstructor().reconstruct(admission)

    assert report.status is ReconstructionStatus.UNSUPPORTED
    assert report.state is None
    assert report.candidate_outcomes["point_a"] == "unsupported"
    assert any(issue.code == "reconstruction.underconstrained" for issue in report.issues)
    assert report.analysis.free_parameter_status == "underconstrained"
    assert report.analysis.unplaced_points == ("point_a", "point_b")


def test_exact_reconstruction_constructs_unique_line_intersection():
    graph = _exact_intersection_graph("line_line")

    report = ParseGraphReconstructor().reconstruct(ParseGraphAdmission().admit(graph))

    assert report.status is ReconstructionStatus.EXACT
    assert report.state is not None
    assert report.state.points["point_x"].coords == (sp.Integer(1), sp.Integer(1))
    assert report.candidate_outcomes["relation_x"] == "materialized"
    assert any(operation.op == "intersection_line_line" for operation in report.trace)
    relation = next(item for item in report.state.relations if item["id"] == "relation_x")
    assert set(relation["dependencies"]) == {"point_x", "segment_ab", "segment_cd"}
    assert relation["metadata"]["parse_candidate_id"] == "relation_x"
    assert relation["metadata"]["evidence_refs"] == ["native_source"]


@pytest.mark.parametrize(
    ("quantity_kind", "quantity_value"),
    (("radius", 3), ("diameter", 6)),
)
def test_exact_reconstruction_constructs_circle_from_admitted_size_quantity(
    quantity_kind,
    quantity_value,
):
    graph = _circle_from_quantity_graph(((quantity_kind, quantity_value),))

    report = ParseGraphReconstructor().reconstruct(ParseGraphAdmission().admit(graph))

    assert report.status is ReconstructionStatus.EXACT
    assert report.state is not None
    assert report.state.circles["circle_o"].radius == sp.Integer(3)
    assert report.state.circles["circle_o"].metadata["parse_quantity_ids"] == [f"quantity_{quantity_kind}"]
    assert report.source_facts.find(quantity_kind, "circle_o")


def test_exact_reconstruction_reports_conflicting_circle_size_quantities():
    graph = _circle_from_quantity_graph((("radius", 3), ("diameter", 8)))

    report = ParseGraphReconstructor().reconstruct(ParseGraphAdmission().admit(graph))

    assert report.status is ReconstructionStatus.PARTIAL
    assert report.state is not None
    assert "circle_o" not in report.state.circles
    assert report.candidate_outcomes["circle_o"] == "unsupported"
    assert any(issue.code == "reconstruction.condition_conflict" for issue in report.issues)


@pytest.mark.parametrize(
    ("kind", "point_id", "expected", "operation", "dependency_ids"),
    (
        ("midpoint", "point_m", (sp.Integer(2), sp.Integer(0)), "midpoint", {"point_a", "point_b"}),
        ("foot", "point_h", (sp.Integer(1), sp.Integer(0)), "foot_of_perpendicular", {"point_c", "segment_ab"}),
        ("ratio", "point_r", (sp.Rational(4, 3), sp.Integer(0)), "point_on_segment_ratio", {"segment_ab"}),
        ("ray_distance", "point_d", (sp.Integer(3), sp.Integer(0)), "point_on_ray_distance", {"ray_ab"}),
    ),
)
def test_exact_reconstruction_compiles_derived_points_as_dependency_ordered_constructions(
    kind,
    point_id,
    expected,
    operation,
    dependency_ids,
):
    graph = _exact_derived_point_graph(kind)

    report = ParseGraphReconstructor().reconstruct(ParseGraphAdmission().admit(graph))

    assert report.status is ReconstructionStatus.EXACT
    assert report.state is not None
    assert report.state.points[point_id].coords == expected
    assert report.candidate_outcomes[f"relation_{kind}"] == "materialized"
    relation = next(
        item
        for item in report.state.relations
        if item["id"] == f"relation_{kind}"
    )
    assert point_id in relation["dependencies"]
    assert relation["metadata"]["parse_candidate_id"] == f"relation_{kind}"
    assert relation["metadata"]["evidence_refs"] == ["native_source"]
    assert relation["metadata"]["score_kind"] == "fixture"
    operation_index = next(index for index, item in enumerate(report.trace) if item.op == operation)
    dependency_indices = [
        index
        for index, item in enumerate(report.trace)
        if item.id in dependency_ids
    ]
    assert operation_index > max(dependency_indices)


@pytest.mark.parametrize(
    ("kind", "point_id", "exact_coords"),
    (
        ("foot", "point_h", [{"operator": "literal", "value": "1"}, {"operator": "literal", "value": "0"}]),
        (
            "ratio",
            "point_r",
            [
                {
                    "operator": "div",
                    "operands": [
                        {"operator": "literal", "value": "4"},
                        {"operator": "literal", "value": "3"},
                    ],
                },
                {"operator": "literal", "value": "0"},
            ],
        ),
        ("ray_distance", "point_d", [{"operator": "literal", "value": "3"}, {"operator": "literal", "value": "0"}]),
    ),
)
def test_exact_reconstruction_verifies_declared_derived_point_relation(
    kind,
    point_id,
    exact_coords,
):
    graph = _exact_derived_point_graph(kind)
    output = next(candidate for candidate in graph.objects if candidate.id == point_id)
    graph = replace(
        graph,
        objects=tuple(
            replace(candidate, geometry={**candidate.geometry, "exact_coords": exact_coords})
            if candidate.id == output.id
            else candidate
            for candidate in graph.objects
        ),
    )

    report = ParseGraphReconstructor().reconstruct(ParseGraphAdmission().admit(graph))

    assert report.status is ReconstructionStatus.EXACT
    assert report.state is not None
    assert any(
        operation.op == "add_relation" and operation.id == f"relation_{kind}"
        for operation in report.trace
    )
    assert any(relation["id"] == f"relation_{kind}" for relation in report.state.relations)


def test_exact_reconstruction_rejects_false_declared_segment_ratio():
    graph = _exact_derived_point_graph("ratio")
    graph = replace(
        graph,
        objects=tuple(
            replace(
                candidate,
                geometry={
                    **candidate.geometry,
                    "exact_coords": [_literal(2), _literal(0)],
                },
            )
            if candidate.id == "point_r"
            else candidate
            for candidate in graph.objects
        ),
    )

    report = ParseGraphReconstructor().reconstruct(ParseGraphAdmission().admit(graph))

    assert report.status is ReconstructionStatus.PARTIAL
    assert report.state is None
    assert not report.verification["valid"]
    assert any(
        issue["code"] == "state.false_point_on_segment_ratio"
        for issue in report.verification["issues"]
    )


def test_exact_reconstruction_rejects_external_segment_ratio():
    graph = _exact_derived_point_graph("ratio", expression=_literal(2))

    report = ParseGraphReconstructor().reconstruct(ParseGraphAdmission().admit(graph))

    assert report.status is ReconstructionStatus.PARTIAL
    assert report.state is not None
    assert "point_r" not in report.state.points
    assert any("strictly between 0 and 1" in issue.message for issue in report.issues)


def test_exact_reconstruction_rejects_negative_ray_distance():
    graph = _exact_derived_point_graph("ray_distance", expression=_literal(-1))

    report = ParseGraphReconstructor().reconstruct(ParseGraphAdmission().admit(graph))

    assert report.status is ReconstructionStatus.PARTIAL
    assert report.state is not None
    assert "point_d" not in report.state.points
    assert any("distance must be positive" in issue.message for issue in report.issues)


def test_exact_reconstruction_reports_competing_point_constructions():
    graph = _exact_derived_point_graph("ratio")
    competing = ParseRelation(
        id="relation_midpoint_competing",
        predicate="midpoint",
        arguments=(
            ParseReference("point", "point_r"),
            ParseReference("of_0", "point_a"),
            ParseReference("of_1", "point_b"),
        ),
        origin=ParseOrigin.NATIVE,
        confidence=1.0,
        score_kind="fixture",
        evidence_refs=("native_source",),
    )
    graph = replace(graph, relations=(*graph.relations, competing))

    report = ParseGraphReconstructor().reconstruct(ParseGraphAdmission().admit(graph))

    assert report.status is ReconstructionStatus.PARTIAL
    assert report.candidate_outcomes["point_r"] == "ambiguous"
    assert any(issue.code == "reconstruction.construction_ambiguous" for issue in report.issues)


@pytest.mark.parametrize(
    ("predicate", "operation", "expected_direction"),
    (
        ("parallel", "parallel_through", (sp.Integer(4), sp.Integer(0))),
        ("perpendicular", "perpendicular_through", (sp.Integer(0), sp.Integer(4))),
    ),
)
def test_exact_reconstruction_constructs_line_from_point_and_direction_relation(
    predicate,
    operation,
    expected_direction,
):
    graph = _exact_derived_line_graph(predicate)

    report = ParseGraphReconstructor().reconstruct(ParseGraphAdmission().admit(graph))

    assert report.status is ReconstructionStatus.EXACT
    assert report.state is not None
    assert report.state.lines["line_l"].point == (sp.Integer(0), sp.Integer(2))
    assert report.state.lines["line_l"].direction == expected_direction
    trace_operation = next(item for item in report.trace if item.id == "line_l")
    assert trace_operation.op == operation
    relation = next(item for item in report.state.relations if item["id"] == f"relation_{predicate}")
    assert set(relation["dependencies"]) == {"line_l", "segment_ab"}
    assert relation["metadata"]["parse_candidate_id"] == f"relation_{predicate}"


def test_exact_reconstruction_reports_competing_line_direction_constructions():
    graph = _exact_derived_line_graph("parallel")
    competing = replace(
        graph.relations[0],
        id="relation_perpendicular",
        predicate="perpendicular",
    )
    graph = replace(graph, relations=(*graph.relations, competing))

    report = ParseGraphReconstructor().reconstruct(ParseGraphAdmission().admit(graph))

    assert report.status is ReconstructionStatus.PARTIAL
    assert report.state is not None
    assert "line_l" not in report.state.lines
    assert report.candidate_outcomes["line_l"] == "ambiguous"
    assert any(issue.code == "reconstruction.construction_ambiguous" for issue in report.issues)


def test_exact_reconstruction_rejects_cyclic_line_direction_definitions():
    graph = _exact_derived_line_graph("parallel")
    point_c = next(candidate for candidate in graph.objects if candidate.id == "point_c")
    line_m = ParseObject(
        id="line_m",
        kind=ParseObjectKind.LINE,
        origin=ParseOrigin.NATIVE,
        confidence=1.0,
        score_kind="fixture",
        references=(ParseReference("point", point_c.id),),
        evidence_refs=("native_source",),
    )
    line_l = replace(
        next(candidate for candidate in graph.objects if candidate.id == "line_l"),
        references=(ParseReference("point", "point_a"),),
    )
    relation = replace(
        graph.relations[0],
        arguments=(
            ParseReference("object_0", line_l.id),
            ParseReference("object_1", line_m.id),
        ),
    )
    graph = replace(
        graph,
        objects=(line_l, line_m, *[candidate for candidate in graph.objects if candidate.kind is ParseObjectKind.POINT]),
        relations=(relation,),
    )

    report = ParseGraphReconstructor().reconstruct(ParseGraphAdmission().admit(graph))

    assert report.status is ReconstructionStatus.PARTIAL
    assert report.state is not None
    assert not report.state.lines
    unresolved = {
        issue.candidate_id
        for issue in report.issues
        if issue.code == "reconstruction.reference_unresolved"
    }
    assert unresolved == {"line_l", "line_m"}


@pytest.mark.parametrize(
    ("mode", "expected_direction"),
    (
        ("internal", (sp.Integer(1), sp.Integer(1))),
        ("external", (sp.Integer(1), sp.Integer(-1))),
    ),
)
def test_exact_reconstruction_constructs_angle_bisector_line(mode, expected_direction):
    graph = _exact_angle_bisector_graph(mode=mode)

    report = ParseGraphReconstructor().reconstruct(ParseGraphAdmission().admit(graph))

    assert report.status is ReconstructionStatus.EXACT
    assert report.state is not None
    assert report.state.lines["line_bisector"].direction == expected_direction
    relation = next(item for item in report.state.relations if item["id"] == "relation_bisector")
    assert relation["mode"] == mode
    assert set(relation["dependencies"]) == {"line_bisector", "point_a", "point_v", "point_c"}
    assert relation["metadata"]["parse_candidate_id"] == "relation_bisector"


def test_exact_reconstruction_rejects_angle_bisector_not_declared_through_vertex():
    graph = _exact_angle_bisector_graph(through_vertex=False)

    report = ParseGraphReconstructor().reconstruct(ParseGraphAdmission().admit(graph))

    assert report.status is ReconstructionStatus.PARTIAL
    assert report.state is not None
    assert "line_bisector" not in report.state.lines
    assert any("must construct a line through its vertex" in issue.message for issue in report.issues)


def test_exact_reconstruction_verifies_declared_angle_bisector_relation():
    graph = _exact_angle_bisector_graph()
    graph = replace(
        graph,
        objects=tuple(
            replace(
                candidate,
                references=(
                    ParseReference("point_0", "point_v"),
                    ParseReference("point_1", "point_w"),
                ),
            )
            if candidate.id == "line_bisector"
            else candidate
            for candidate in graph.objects
        ),
    )

    report = ParseGraphReconstructor().reconstruct(ParseGraphAdmission().admit(graph))

    assert report.status is ReconstructionStatus.EXACT
    assert report.state is not None
    assert any(
        operation.op == "add_relation" and operation.id == "relation_bisector"
        for operation in report.trace
    )
    assert any(relation["id"] == "relation_bisector" for relation in report.state.relations)


def test_exact_reconstruction_constructs_line_from_exact_relative_angle():
    graph = _exact_angle_from_line_graph()

    report = ParseGraphReconstructor().reconstruct(ParseGraphAdmission().admit(graph))

    assert report.status is ReconstructionStatus.EXACT
    assert report.state is not None
    assert report.state.lines["line_rotated"].direction == (sp.Integer(0), sp.Integer(4))
    relation = next(item for item in report.state.relations if item["id"] == "relation_angle_from_line")
    assert sp.simplify(sp.sympify(relation["angle"]) - sp.pi / 2) == 0
    assert relation["metadata"]["parse_candidate_id"] == "relation_angle_from_line"


def test_exact_reconstruction_rejects_symbolic_relative_angle():
    graph = _exact_angle_from_line_graph(
        expression={"operator": "symbol", "value": "theta"},
    )

    report = ParseGraphReconstructor().reconstruct(ParseGraphAdmission().admit(graph))

    assert report.status is ReconstructionStatus.PARTIAL
    assert report.state is not None
    assert "line_rotated" not in report.state.lines
    assert any("requires a real numeric angle" in issue.message for issue in report.issues)


def test_exact_reconstruction_requires_explicit_circle_intersection_branch():
    graph = _exact_intersection_graph("circle_circle")

    report = ParseGraphReconstructor().reconstruct(ParseGraphAdmission().admit(graph))

    assert report.status is ReconstructionStatus.PARTIAL
    assert report.state is not None
    assert "point_x" not in report.state.points
    assert report.analysis.branch_status == "ambiguous"
    assert report.analysis.ambiguous_candidates == ("point_x",)
    assert any(issue.code == "reconstruction.branch_ambiguous" for issue in report.issues)


def test_exact_reconstruction_honors_explicit_circle_intersection_branch():
    graph = _exact_intersection_graph("circle_circle", selection="upper")

    report = ParseGraphReconstructor().reconstruct(ParseGraphAdmission().admit(graph))

    assert report.status is ReconstructionStatus.EXACT
    assert report.state is not None
    assert report.state.points["point_x"].coords == (sp.Integer(2), sp.sqrt(5))
    operation = next(operation for operation in report.trace if operation.id == "point_x")
    assert operation.op == "intersection_circle_circle"
    assert operation.params["select"] == "upper"


def test_exact_reconstruction_constructs_a_tangent_point_without_branch_selection():
    graph = _exact_intersection_graph("line_circle_tangent")
    tangent = ParseRelation(
        id="relation_tangent",
        predicate="tangent",
        arguments=(
            ParseReference("point", "point_x"),
            ParseReference("line", "segment_ab"),
            ParseReference("circle", "circle_1"),
        ),
        origin=ParseOrigin.NATIVE,
        confidence=1.0,
        score_kind="fixture",
        evidence_refs=("native_source",),
    )
    graph = replace(graph, relations=(*graph.relations, tangent))

    report = ParseGraphReconstructor().reconstruct(ParseGraphAdmission().admit(graph))

    assert report.status is ReconstructionStatus.EXACT
    assert report.state is not None
    assert report.state.points["point_x"].coords == (sp.Integer(5), sp.Integer(3))
    operation = next(operation for operation in report.trace if operation.id == "point_x")
    assert operation.op == "intersection_line_circle"
    assert operation.params["select"] == "all"
    state_tangent = next(item for item in report.state.relations if item["id"] == tangent.id)
    assert state_tangent["type"] == "tangent"
    assert state_tangent["line"] == ["point_a", "point_b"]


@pytest.mark.parametrize(
    ("kind", "expected_operation"),
    (
        ("line_arc", "intersection_line_arc"),
        ("circle_arc", "intersection_circle_arc"),
        ("arc_arc", "intersection_arc_arc"),
    ),
)
def test_exact_reconstruction_routes_arc_intersections_through_existing_kernel(kind, expected_operation):
    graph = _exact_arc_intersection_graph(kind)

    report = ParseGraphReconstructor().reconstruct(ParseGraphAdmission().admit(graph))

    assert report.status is ReconstructionStatus.EXACT
    assert report.state is not None
    operation = next(item for item in report.trace if item.id == "point_x")
    assert operation.op == expected_operation
    assert operation.params["select"] == "all"
    assert report.state.relations[-1]["id"] == "relation_x"


def test_source_redraw_preserves_arc_candidates():
    graph = _exact_arc_intersection_graph("line_arc")

    scene = source_graph_to_render_scene(graph)

    arc_primitives = [primitive for primitive in scene.primitives if primitive.kind == "arc"]
    assert len(arc_primitives) == 1
    assert len(arc_primitives[0].geometry["start"]) == 2
    assert "arc" not in scene.metadata["unsupported_candidates"]


def test_source_redraw_materializes_mixed_arc_segment_region():
    evidence = ParseEvidence(
        id="native_source",
        source_id="source_curved_region",
        kind=ParseEvidenceKind.NATIVE_ENTITY,
        precision=EvidencePrecision.BOUND,
    )
    center = _point("center", [0, 0], [50, 50])
    start = _point("start", [2, 0], [70, 50])
    end = _point("end", [0, 2], [50, 30])
    circle = _circle("circle", center.id, 2)
    arc = _arc("arc", circle.id, start.id, end.id, _pi_div(2))
    end_radius = _segment("end_radius", end.id, center.id)
    start_radius = _segment("start_radius", center.id, start.id)
    region = ParseRegion(
        id="sector_candidate",
        boundary=(
            BoundaryReference(arc.id),
            BoundaryReference(end_radius.id),
            BoundaryReference(start_radius.id),
        ),
        origin=ParseOrigin.IMAGE,
        confidence=0.9,
        score_kind="fixture_curved_boundary",
        evidence_refs=(evidence.id,),
    )
    graph = ParseGraph(
        id="graph_curved_region",
        source_id=evidence.source_id,
        evidence=(evidence,),
        objects=(center, start, end, circle, arc, end_radius, start_radius),
        regions=(region,),
        metadata={"image_size": [100, 100]},
    )

    scene = source_graph_to_render_scene(graph)

    region_primitive = next(primitive for primitive in scene.primitives if primitive.id == region.id)
    assert region_primitive.kind == "polygon"
    assert len(region_primitive.geometry["points"]) > 12
    assert region.id not in scene.metadata["unsupported_candidates"]


def test_exact_branch_enumeration_preserves_both_circle_solutions():
    graph = _exact_intersection_graph("circle_circle")

    result = ParseGraphReconstructor().enumerate_branches(ParseGraphAdmission().admit(graph))

    assert len(result.branches) == 2
    assert result.ambiguous_candidates == ("point_x",)
    assert {branch.branch_choices["point_x"] for branch in result.branches} == {"first", "second"}
    assert all(branch.status is ReconstructionStatus.EXACT for branch in result.branches)
    assert all(branch.analysis.branch_status == "selected" for branch in result.branches)
    assert len({branch.state.points["point_x"].coords for branch in result.branches}) == 2
    assert result.to_dict()["checksum"]


def test_exact_branch_enumeration_pairs_two_points_on_the_same_circles():
    graph = _two_point_circle_intersection_graph()

    result = ParseGraphReconstructor().enumerate_branches(ParseGraphAdmission().admit(graph))

    assert len(result.branches) == 2
    assert result.ambiguous_candidates == ("point_x", "point_y")
    for branch in result.branches:
        assert branch.status is ReconstructionStatus.EXACT
        assert set(branch.branch_choices.values()) == {"first", "second"}
        assert branch.state.points["point_x"].coords != branch.state.points["point_y"].coords


def test_exact_branch_enumeration_pairs_two_points_on_a_full_arc():
    graph = _two_point_full_arc_intersection_graph()

    result = ParseGraphReconstructor().enumerate_branches(ParseGraphAdmission().admit(graph))

    assert len(result.branches) == 2
    assert result.ambiguous_candidates == ("point_x", "point_y")
    for branch in result.branches:
        assert branch.status is ReconstructionStatus.EXACT
        assert set(branch.branch_choices.values()) == {"first", "second"}
        assert branch.state.points["point_x"].coords != branch.state.points["point_y"].coords


def test_exact_branch_enumeration_enforces_the_branch_limit():
    graph = _exact_intersection_graph("circle_circle")

    with pytest.raises(InvalidGeometryInput, match="2 branches"):
        ParseGraphReconstructor().enumerate_branches(
            ParseGraphAdmission().admit(graph),
            max_branches=1,
        )


def test_exact_sss_gauge_requires_an_explicit_mirror_branch():
    graph = _sss_triangle_graph()

    report = ParseGraphReconstructor().reconstruct(ParseGraphAdmission().admit(graph))

    assert report.status is ReconstructionStatus.PARTIAL
    assert report.state is not None
    assert report.state.points["point_a"].coords == (sp.Integer(0), sp.Integer(0))
    assert report.state.points["point_b"].coords == (sp.Integer(5), sp.Integer(0))
    assert "point_c" not in report.state.points
    assert report.analysis.branch_status == "ambiguous"
    assert report.analysis.ambiguous_candidates == ("point_c",)
    assert any(issue.code == "reconstruction.branch_ambiguous" for issue in report.issues)


def test_exact_sss_gauge_enumerates_both_realization_branches():
    graph = _sss_triangle_graph()
    admission = ParseGraphAdmission().admit(graph)

    result = ParseGraphReconstructor().enumerate_branches(admission)

    assert len(result.branches) == 2
    assert result.ambiguous_candidates == ("point_c",)
    assert {branch.branch_choices["point_c"] for branch in result.branches} == {"first", "second"}
    assert all(branch.status is ReconstructionStatus.EXACT for branch in result.branches)
    assert all(branch.analysis.coordinate_basis == "constructed_exact_coordinates" for branch in result.branches)
    apex_coordinates = {branch.state.points["point_c"].coords for branch in result.branches}
    assert apex_coordinates == {
        (sp.Rational(16, 5), sp.Rational(12, 5)),
        (sp.Rational(16, 5), sp.Rational(-12, 5)),
    }
    for branch in result.branches:
        assert branch.state.points["point_c"].metadata["fact_scope"] == FactScope.REALIZATION_FACT.value
        assert branch.state.points["point_c"].metadata["reconstruction_gauge"] == "sss_triangle_v1"
        assert {fact.scope for fact in branch.source_facts if fact.type == "length"} == {
            FactScope.SOURCE_ASSERTION
        }


def test_exact_sss_gauge_reports_impossible_side_lengths():
    graph = _sss_triangle_graph(ab=3, ac=1, bc=1)

    report = ParseGraphReconstructor().reconstruct(ParseGraphAdmission().admit(graph))

    assert report.status is ReconstructionStatus.UNSUPPORTED
    assert report.state is None
    assert any(issue.code == "reconstruction.condition_conflict" for issue in report.issues)
    assert report.candidate_outcomes["point_c"] == "unsupported"


def test_exact_triangulated_distance_graph_keeps_shared_edge_branches_explicit():
    graph = _shared_edge_distance_graph()

    report = ParseGraphReconstructor().reconstruct(ParseGraphAdmission().admit(graph))

    assert report.status is ReconstructionStatus.PARTIAL
    assert report.state is not None
    assert set(report.state.points) == {"point_b", "point_c"}
    assert report.analysis.branch_status == "ambiguous"
    assert report.analysis.ambiguous_candidates == ("point_a", "point_d")
    assert set(report.candidate_outcomes) >= {"point_a", "point_b", "point_c", "point_d"}
    assert report.candidate_outcomes["point_a"] == "ambiguous"
    assert report.candidate_outcomes["point_d"] == "ambiguous"


def test_exact_triangulated_distance_graph_enumerates_every_shared_edge_realization():
    graph = _shared_edge_distance_graph()
    admission = ParseGraphAdmission().admit(graph)

    result = ParseGraphReconstructor().enumerate_branches(admission)

    assert len(result.branches) == 4
    assert result.ambiguous_candidates == ("point_a", "point_d")
    assert {
        tuple(sorted(branch.branch_choices.items()))
        for branch in result.branches
    } == {
        (("point_a", "first"), ("point_d", "first")),
        (("point_a", "first"), ("point_d", "second")),
        (("point_a", "second"), ("point_d", "first")),
        (("point_a", "second"), ("point_d", "second")),
    }
    assert all(branch.status is ReconstructionStatus.EXACT for branch in result.branches)
    assert all(branch.verification["valid"] is True for branch in result.branches)
    assert {
        (
            branch.state.points["point_a"].coords,
            branch.state.points["point_d"].coords,
        )
        for branch in result.branches
    } == {
        ((sp.Integer(3), sp.Integer(4)), (sp.Integer(0), sp.Integer(4))),
        ((sp.Integer(3), sp.Integer(4)), (sp.Integer(0), sp.Integer(-4))),
        ((sp.Integer(3), sp.Integer(-4)), (sp.Integer(0), sp.Integer(4))),
        ((sp.Integer(3), sp.Integer(-4)), (sp.Integer(0), sp.Integer(-4))),
    }
    for branch in result.branches:
        assert set(branch.state.points) == {"point_a", "point_b", "point_c", "point_d"}
        assert {
            point.metadata["reconstruction_gauge"]
            for point in branch.state.points.values()
        } == {"triangulated_distance_graph_v1"}
        assert len(branch.state.points["point_a"].metadata["support_quantity_ids"]) == 3
        assert len(branch.state.points["point_d"].metadata["support_quantity_ids"]) == 3
        assert len([fact for fact in branch.source_facts if fact.type == "length"]) == 5


def test_exact_triangulated_distance_graph_enforces_combined_branch_limit():
    admission = ParseGraphAdmission().admit(_shared_edge_distance_graph())

    with pytest.raises(InvalidGeometryInput, match="4 branches"):
        ParseGraphReconstructor().enumerate_branches(admission, max_branches=3)


def test_branch_semantic_redraws_bind_each_exact_realization():
    admission = ParseGraphAdmission().admit(_shared_edge_distance_graph())
    branch_set = ParseGraphReconstructor().enumerate_branches(admission)

    redraws = ParseRedrawPipeline().render_branches(admission, branch_set)

    assert len(redraws) == 4
    assert {tuple(sorted(item.branch_choices.items())) for item in redraws} == {
        tuple(sorted(branch.branch_choices.items())) for branch in branch_set.branches
    }
    assert all(item.redraw.mode is RedrawMode.SEMANTIC for item in redraws)
    assert all(item.redraw.verification["valid"] is True for item in redraws)
    assert all(
        item.redraw.lineage["reconstruction_checksum"] == item.branch_checksum
        for item in redraws
    )
    assert len({item.redraw.scene.id for item in redraws}) == 4
    assert len({item.to_dict()["checksum"] for item in redraws}) == 4


def test_redundant_distance_constraint_prunes_inconsistent_mirror_branches():
    graph = _redundant_shared_edge_distance_graph()
    admission = ParseGraphAdmission().admit(graph)

    result = ParseGraphReconstructor().enumerate_branches(admission)
    payload = result.to_dict()

    assert result.ambiguous_candidates == ("point_c", "point_d")
    assert len(result.branches) == 2
    assert len(result.rejected_branches) == 2
    assert all(branch.status is ReconstructionStatus.EXACT for branch in result.branches)
    assert all(branch.status is ReconstructionStatus.PARTIAL for branch in result.rejected_branches)
    assert all(
        sp.simplify(
            branch.state.points["point_c"].coords[1]
            * branch.state.points["point_d"].coords[1]
        ).is_negative
        for branch in result.branches
    )
    assert all(
        any(
            issue.code == "reconstruction.condition_conflict"
            and issue.candidate_id == "segment_cd"
            for issue in branch.issues
        )
        for branch in result.rejected_branches
    )
    assert {
        point.metadata["reconstruction_gauge"]
        for branch in result.branches
        for point in branch.state.points.values()
    } == {"redundant_triangulated_distance_graph_v1"}
    assert payload["num_candidate_branches"] == 4
    assert payload["num_branches"] == 2
    assert payload["num_rejected_branches"] == 2
    assert payload["status_counts"] == {"exact": 2}
    assert payload["candidate_status_counts"] == {"exact": 2, "partial": 2}
    assert len(payload["rejected_branches"]) == 2

    redraws = ParseRedrawPipeline().render_branches(admission, result)
    assert len(redraws) == 2
    assert {
        item.branch_checksum for item in redraws
    } == {
        branch.to_dict()["checksum"] for branch in result.branches
    }


def test_redundant_distance_branch_limit_applies_before_constraint_pruning():
    admission = ParseGraphAdmission().admit(_redundant_shared_edge_distance_graph())

    with pytest.raises(InvalidGeometryInput, match="4 branches"):
        ParseGraphReconstructor().enumerate_branches(admission, max_branches=3)


def test_runtime_branch_choices_reject_noncanonical_or_unknown_selections():
    admission = ParseGraphAdmission().admit(_exact_intersection_graph("circle_circle"))
    reconstructor = ParseGraphReconstructor()

    with pytest.raises(InvalidGeometryInput, match="Invalid reconstruction branch"):
        reconstructor.reconstruct(admission, branch_choices={"point_x": "upper"})
    with pytest.raises(InvalidGeometryInput, match="Unknown or non-branching"):
        reconstructor.reconstruct(admission, branch_choices={"point_missing": "first"})


def test_reconstruction_reports_degenerate_trace_without_persisting_state():
    graph = _graph(exact=True)
    duplicate_b = replace(graph.objects[1], geometry=graph.objects[0].geometry)
    graph = replace(graph, objects=(graph.objects[0], duplicate_b, graph.objects[2]))
    admission = ParseGraphAdmission().admit(graph)

    report = ParseGraphReconstructor().reconstruct(admission)

    assert report.status is ReconstructionStatus.PARTIAL
    assert report.state is None
    assert report.verification["valid"] is False


def test_object_compilation_is_independent_of_parse_graph_order():
    graph = _graph(exact=True)
    graph = replace(graph, objects=(graph.objects[2], graph.objects[1], graph.objects[0]))
    admission = ParseGraphAdmission().admit(graph)

    report = ParseGraphReconstructor().reconstruct(admission)

    assert report.status is ReconstructionStatus.EXACT
    assert report.state is not None
    assert set(report.state.points) == {"point_a", "point_b"}
    assert set(report.state.segments) == {"segment_ab"}


def test_reconstructor_refuses_unadmitted_graphs():
    with pytest.raises(TypeError, match="AdmissionReport"):
        ParseGraphReconstructor().reconstruct(_graph(exact=True))


def test_source_redraw_projects_only_explicit_names_and_source_values():
    graph = _graph(exact=False)
    graph = replace(
        graph,
        objects=(
            replace(graph.objects[0], visible_name="A", visible=True),
            replace(graph.objects[1], visible_name=None, visible=True),
            graph.objects[2],
        ),
    )
    admission = ParseGraphAdmission().admit(graph)

    result = ParseRedrawPipeline().render(admission, mode=RedrawMode.SOURCE)

    assert {label.text for label in result.scene.labels} == {"A", "4"}
    assert result.invariants["mathematical_realization"] is False
    assert result.scene.metadata["annotation_projection"] == "explicit_source_only"


def test_semantic_redraw_uses_source_quantity_not_witness_length():
    graph = _graph(exact=True)
    graph = replace(
        graph,
        objects=(
            replace(graph.objects[0], visible_name="A", visible=True),
            replace(graph.objects[1], visible_name="B", visible=True),
            graph.objects[2],
        ),
    )
    admission = ParseGraphAdmission().admit(graph)
    reconstruction = ParseGraphReconstructor().reconstruct(admission)

    result = ParseRedrawPipeline().render(
        admission,
        mode=RedrawMode.SEMANTIC,
        reconstruction=reconstruction,
    )

    labels = {label.id: label.text for label in result.scene.labels}
    assert labels["label:point_a"] == "A"
    assert labels["label:point_b"] == "B"
    assert labels["measure:segment_ab"] == "4"
    assert result.scene.metadata["annotation_policy"]["inferred_markers"] is False


def test_rigid_resynthesis_is_deterministic_and_lineage_grouped():
    graph = _graph(exact=True)
    admission = ParseGraphAdmission().admit(graph)
    reconstruction = ParseGraphReconstructor().reconstruct(admission)
    pipeline = ParseRedrawPipeline()

    first = pipeline.render(admission, mode=RedrawMode.RESYNTHESIS, reconstruction=reconstruction, seed=19)
    repeated = pipeline.render(admission, mode=RedrawMode.RESYNTHESIS, reconstruction=reconstruction, seed=19)
    other = pipeline.render(admission, mode=RedrawMode.RESYNTHESIS, reconstruction=reconstruction, seed=20)

    assert first.to_dict() == repeated.to_dict()
    assert first.lineage["group_key"] == other.lineage["group_key"]
    assert first.invariants["topology_signature"] == other.invariants["topology_signature"]
    assert first.invariants["transform_class"] == "exact_rigid_2d"
    assert first.trace != reconstruction.trace


def test_similarity_resynthesis_scales_dimensionless_exact_construction():
    graph = _exact_derived_point_graph("midpoint")
    admission = ParseGraphAdmission().admit(graph)
    reconstruction = ParseGraphReconstructor().reconstruct(admission)
    pipeline = ParseRedrawPipeline()
    semantic = pipeline.render(
        admission,
        mode=RedrawMode.SEMANTIC,
        reconstruction=reconstruction,
    )

    first = pipeline.render(
        admission,
        mode=RedrawMode.RESYNTHESIS,
        reconstruction=reconstruction,
        seed=19,
        resynthesis_transform=ResynthesisTransform.SIMILARITY,
    )
    repeated = pipeline.render(
        admission,
        mode=RedrawMode.RESYNTHESIS,
        reconstruction=reconstruction,
        seed=19,
        resynthesis_transform=ResynthesisTransform.SIMILARITY,
    )

    assert first.to_dict() == repeated.to_dict()
    assert first.invariants["transform_class"] == "exact_similarity_2d"
    assert first.lineage["transform"]["scale"] in {"1/2", "3/2", "2"}
    assert first.invariants["topology_signature"] == semantic.invariants["topology_signature"]
    assert first.trace != reconstruction.trace


def test_similarity_resynthesis_rejects_dimensional_source_facts():
    graph = _graph(exact=True)
    admission = ParseGraphAdmission().admit(graph)
    reconstruction = ParseGraphReconstructor().reconstruct(admission)

    with pytest.raises(InvalidGeometryInput, match="dimensional source facts: length"):
        ParseRedrawPipeline().render(
            admission,
            mode=RedrawMode.RESYNTHESIS,
            reconstruction=reconstruction,
            seed=19,
            resynthesis_transform=ResynthesisTransform.SIMILARITY,
        )


def test_similarity_resynthesis_preserves_intersection_topology_not_coordinates():
    graph = _exact_intersection_graph("line_line")
    admission = ParseGraphAdmission().admit(graph)
    reconstruction = ParseGraphReconstructor().reconstruct(admission)
    pipeline = ParseRedrawPipeline()
    semantic = pipeline.render(
        admission,
        mode=RedrawMode.SEMANTIC,
        reconstruction=reconstruction,
    )

    result = pipeline.render(
        admission,
        mode=RedrawMode.RESYNTHESIS,
        reconstruction=reconstruction,
        seed=7,
        resynthesis_transform=ResynthesisTransform.SIMILARITY,
    )

    assert result.invariants["topology_signature"] == semantic.invariants["topology_signature"]
    parent_relation = next(item for item in reconstruction.state.relations if item["id"] == "relation_x")
    child_relation = next(
        operation
        for operation in result.trace
        if operation.id == "point_x"
    )
    assert parent_relation["result"]
    assert child_relation.op == "intersection_line_line"


def test_polygon_hole_projection_and_reconstruction_preserve_the_hole():
    graph = _region_graph()
    admission = ParseGraphAdmission().admit(graph)

    source_scene = source_graph_to_render_scene(admission.admitted_graph)
    reconstruction = ParseGraphReconstructor().reconstruct(admission)
    semantic = ParseRedrawPipeline().render(
        admission,
        mode=RedrawMode.SEMANTIC,
        reconstruction=reconstruction,
    )

    source_region = next(item for item in source_scene.primitives if item.id == "shaded_region")
    semantic_region = next(item for item in semantic.scene.primitives if item.id == "shaded_region")
    assert len(source_region.geometry["holes"]) == 1
    assert len(semantic_region.geometry["holes"]) == 1
    assert reconstruction.state.composite_regions["shaded_region"].operation == "difference"


def test_verified_workflow_serializes_and_summarizes_every_stage():
    graph = _graph(exact=True)
    parser_input = ParserInput(id="source_1", image="opaque.png", question="Find AB.")
    record = HypothesisRecordAdapter().convert_parse_graph(
        sample_id=parser_input.id,
        dataset_name="fixture",
        parser_name="fixture_parser",
        backend="fixture",
        graph=graph,
        confidence=1.0,
        verification={"status": "not_admitted"},
    )
    parser_result = TypedParseResult(parser_input, graph, record, ({"name": "fixture", "status": "complete"},))
    workflow = VerifiedParseWorkflow(parser=_FixtureParser(parser_result))

    result = workflow.run(parser_input, reconstruction_mode=ReconstructionMode.EXACT, resynthesis_seeds=(3,))
    payload = result.to_dict()
    summary = summarize_workflow_results([result])

    json.dumps(payload)
    assert payload["checksum"]
    assert [item["mode"] for item in payload["redraws"]] == ["source_redraw", "semantic_redraw", "resynthesis"]
    assert summary["reconstruction_status_counts"] == {"exact": 1}
    assert summary["num_resyntheses"] == 1


def _graph(*, exact: bool) -> ParseGraph:
    evidence = ParseEvidence(
        id="native_source",
        source_id="source_1",
        kind=ParseEvidenceKind.NATIVE_ENTITY,
        precision=EvidencePrecision.BOUND,
    )
    point_a = _point("point_a", [0, 0] if exact else None, [10, 20])
    point_b = _point("point_b", [_sqrt(2), _div(1, 2)] if exact else None, [50, 20])
    segment = ParseObject(
        id="segment_ab",
        kind=ParseObjectKind.SEGMENT,
        origin=ParseOrigin.NATIVE,
        confidence=1.0,
        score_kind="fixture",
        references=(ParseReference("start", point_a.id), ParseReference("end", point_b.id)),
        evidence_refs=(evidence.id,),
    )
    quantity = ParseQuantity(
        id="quantity_ab",
        kind="length",
        arguments=(ParseReference("object", segment.id),),
        expression=MathExpression("literal", value="4"),
        origin=ParseOrigin.NATIVE,
        confidence=1.0,
        score_kind="fixture",
        evidence_refs=(evidence.id,),
    )
    target = ParseTarget(
        id="target_ab",
        kind="length",
        mode="value",
        arguments=(ParseReference("object", segment.id),),
        origin=ParseOrigin.NATIVE,
        confidence=1.0,
        score_kind="fixture",
        evidence_refs=(evidence.id,),
    )
    return ParseGraph(
        id="graph_1",
        source_id="source_1",
        evidence=(evidence,),
        objects=(point_a, point_b, segment),
        quantities=(quantity,),
        targets=(target,),
        metadata={"image_size": [100, 100]},
    )


def _exact_intersection_graph(kind: str, *, selection: str | None = None) -> ParseGraph:
    evidence = ParseEvidence(
        id="native_source",
        source_id="source_intersection",
        kind=ParseEvidenceKind.NATIVE_ENTITY,
        precision=EvidencePrecision.BOUND,
    )
    point_x = _point("point_x", None, [50, 50])
    if kind == "line_line":
        points = (
            _point("point_a", [0, 0], [0, 100]),
            _point("point_b", [2, 2], [100, 0]),
            _point("point_c", [0, 2], [0, 0]),
            _point("point_d", [2, 0], [100, 100]),
        )
        supports = (
            _segment("segment_ab", "point_a", "point_b"),
            _segment("segment_cd", "point_c", "point_d"),
        )
    elif kind == "circle_circle":
        points = (
            _point("center_1", [0, 0], [20, 50]),
            _point("center_2", [4, 0], [80, 50]),
        )
        supports = (
            _circle("circle_1", "center_1", 3),
            _circle("circle_2", "center_2", 3),
        )
    elif kind == "line_circle_tangent":
        points = (
            _point("point_a", [0, 3], [0, 20]),
            _point("point_b", [10, 3], [100, 20]),
            _point("center_1", [5, 0], [50, 50]),
        )
        supports = (
            _segment("segment_ab", "point_a", "point_b"),
            _circle("circle_1", "center_1", 3),
        )
    else:
        raise AssertionError(f"Unsupported fixture intersection kind: {kind}")
    metadata = {"contact": "tangent" if kind == "line_circle_tangent" else "crossing"}
    if selection is not None:
        metadata["selection"] = selection
    relation = ParseRelation(
        id="relation_x",
        predicate="intersection",
        arguments=(
            ParseReference("support", supports[0].id),
            ParseReference("support", supports[1].id),
            ParseReference("point", point_x.id),
        ),
        origin=ParseOrigin.NATIVE,
        confidence=1.0,
        score_kind="fixture",
        evidence_refs=(evidence.id,),
        metadata=metadata,
    )
    return ParseGraph(
        id=f"graph_{kind}",
        source_id=evidence.source_id,
        evidence=(evidence,),
        objects=(point_x, *supports, *points),
        relations=(relation,),
        metadata={"image_size": [100, 100]},
    )


def _sss_triangle_graph(*, ab: int = 5, ac: int = 4, bc: int = 3) -> ParseGraph:
    evidence = ParseEvidence(
        id="native_source",
        source_id="source_sss_triangle",
        kind=ParseEvidenceKind.NATIVE_ENTITY,
        precision=EvidencePrecision.BOUND,
    )
    points = (
        _point("point_a", None, [10, 80]),
        _point("point_b", None, [90, 80]),
        _point("point_c", None, [60, 20]),
    )
    segments = (
        _segment("segment_ab", "point_a", "point_b"),
        _segment("segment_ac", "point_a", "point_c"),
        _segment("segment_bc", "point_b", "point_c"),
    )
    quantities = tuple(
        ParseQuantity(
            id=f"quantity_{segment_id}",
            kind="length",
            arguments=(ParseReference("object", segment_id),),
            expression=MathExpression("literal", value=str(value)),
            origin=ParseOrigin.NATIVE,
            confidence=1.0,
            score_kind="fixture",
            evidence_refs=(evidence.id,),
        )
        for segment_id, value in (
            ("segment_ab", ab),
            ("segment_ac", ac),
            ("segment_bc", bc),
        )
    )
    return ParseGraph(
        id="graph_sss_triangle",
        source_id=evidence.source_id,
        evidence=(evidence,),
        objects=(*points, *segments),
        quantities=quantities,
        metadata={"image_size": [100, 100]},
    )


def _shared_edge_distance_graph() -> ParseGraph:
    evidence = ParseEvidence(
        id="native_source",
        source_id="source_shared_edge_distance_graph",
        kind=ParseEvidenceKind.NATIVE_ENTITY,
        precision=EvidencePrecision.BOUND,
    )
    points = (
        _point("point_a", None, [60, 20]),
        _point("point_b", None, [20, 60]),
        _point("point_c", None, [50, 60]),
        _point("point_d", None, [20, 20]),
    )
    segment_specs = (
        ("segment_ab", "point_a", "point_b", 5),
        ("segment_ac", "point_a", "point_c", 4),
        ("segment_bc", "point_b", "point_c", 3),
        ("segment_bd", "point_b", "point_d", 4),
        ("segment_cd", "point_c", "point_d", 5),
    )
    segments = tuple(
        _segment(segment_id, start_id, end_id)
        for segment_id, start_id, end_id, _ in segment_specs
    )
    quantities = tuple(
        ParseQuantity(
            id=f"quantity_{segment_id}",
            kind="length",
            arguments=(ParseReference("object", segment_id),),
            expression=MathExpression("literal", value=str(length)),
            origin=ParseOrigin.NATIVE,
            confidence=1.0,
            score_kind="fixture",
            evidence_refs=(evidence.id,),
        )
        for segment_id, _, _, length in segment_specs
    )
    return ParseGraph(
        id="graph_shared_edge_distance_graph",
        source_id=evidence.source_id,
        evidence=(evidence,),
        objects=(*points, *segments),
        quantities=quantities,
        metadata={"image_size": [100, 100]},
    )


def _redundant_shared_edge_distance_graph() -> ParseGraph:
    graph = _shared_edge_distance_graph()
    segment = _segment("segment_ad", "point_a", "point_d")
    quantity = ParseQuantity(
        id="quantity_segment_ad",
        kind="length",
        arguments=(ParseReference("object", segment.id),),
        expression=MathExpression("literal", value="3"),
        origin=ParseOrigin.NATIVE,
        confidence=1.0,
        score_kind="fixture",
        evidence_refs=(graph.evidence[0].id,),
    )
    return replace(
        graph,
        id="graph_redundant_shared_edge_distance_graph",
        objects=(*graph.objects, segment),
        quantities=(*graph.quantities, quantity),
    )


def _two_point_circle_intersection_graph() -> ParseGraph:
    graph = _exact_intersection_graph("circle_circle")
    point_y = _point("point_y", None, [50, 70])
    relation_x = graph.relations[0]
    relation_y = replace(
        relation_x,
        id="relation_y",
        arguments=(
            relation_x.arguments[0],
            relation_x.arguments[1],
            ParseReference("point", point_y.id),
        ),
    )
    return replace(
        graph,
        objects=(point_y, *graph.objects),
        relations=(relation_x, relation_y),
    )


def _exact_arc_intersection_graph(kind: str) -> ParseGraph:
    evidence = ParseEvidence(
        id="native_source",
        source_id="source_arc_intersection",
        kind=ParseEvidenceKind.NATIVE_ENTITY,
        precision=EvidencePrecision.BOUND,
    )
    point_x = _point("point_x", None, [50, 50])
    if kind == "line_arc":
        points = (
            _point("center", [0, 0], [50, 50]),
            _point("arc_start", [2, 0], [70, 50]),
            _point("arc_end", [0, 2], [50, 30]),
            _point("line_start", [-3, 0], [20, 50]),
            _point("line_end", [3, 0], [80, 50]),
        )
        circle = _circle("circle", "center", 2)
        arc = _arc("arc", "circle", "arc_start", "arc_end", _pi_div(2))
        supports = (_segment("axis", "line_start", "line_end"), arc)
        objects = (*supports, circle, *points)
    elif kind == "circle_arc":
        points = (
            _point("center", [0, 0], [35, 50]),
            _point("other_center", [3, 0], [65, 50]),
            _point("arc_start", [5, 0], [85, 50]),
            _point("arc_end", [0, 5], [35, 10]),
        )
        base = _circle("base", "center", 5)
        other = _circle("other", "other_center", 5)
        arc = _arc("arc", "base", "arc_start", "arc_end", _pi_div(2))
        supports = (other, arc)
        objects = (*supports, base, *points)
    elif kind == "arc_arc":
        points = (
            _point("center_1", [0, 0], [35, 50]),
            _point("center_2", [3, 0], [65, 50]),
            _point("start_1", [5, 0], [85, 50]),
            _point("end_1", [-5, 0], [5, 50]),
            _point("start_2", [8, 0], [95, 50]),
            _point("end_2", [-2, 0], [15, 50]),
        )
        first_circle = _circle("circle_1", "center_1", 5)
        second_circle = _circle("circle_2", "center_2", 5)
        first_arc = _arc("arc_1", "circle_1", "start_1", "end_1", _pi_div(1))
        second_arc = _arc("arc_2", "circle_2", "start_2", "end_2", _pi_div(1))
        supports = (first_arc, second_arc)
        objects = (*supports, first_circle, second_circle, *points)
    else:
        raise AssertionError(f"Unsupported arc fixture intersection kind: {kind}")
    relation = ParseRelation(
        id="relation_x",
        predicate="intersection",
        arguments=(
            ParseReference("support", supports[0].id),
            ParseReference("support", supports[1].id),
            ParseReference("point", point_x.id),
        ),
        origin=ParseOrigin.NATIVE,
        confidence=1.0,
        score_kind="fixture",
        evidence_refs=(evidence.id,),
        metadata={"contact": "crossing", "selection": "all"},
    )
    return ParseGraph(
        id=f"graph_{kind}",
        source_id=evidence.source_id,
        evidence=(evidence,),
        objects=(point_x, *objects),
        relations=(relation,),
        metadata={"image_size": [100, 100]},
    )


def _two_point_full_arc_intersection_graph() -> ParseGraph:
    evidence = ParseEvidence(
        id="native_source",
        source_id="source_full_arc_intersection",
        kind=ParseEvidenceKind.NATIVE_ENTITY,
        precision=EvidencePrecision.BOUND,
    )
    points = (
        _point("center", [0, 0], [50, 50]),
        _point("arc_start", [2, 0], [70, 50]),
        _point("line_start", [-3, 0], [20, 50]),
        _point("line_end", [3, 0], [80, 50]),
    )
    circle = _circle("circle", "center", 2)
    arc = _arc(
        "full_arc",
        "circle",
        "arc_start",
        "arc_start",
        _two_pi(),
        allow_full_circle=True,
    )
    axis = _segment("axis", "line_start", "line_end")
    point_x = _point("point_x", None, [30, 50])
    point_y = _point("point_y", None, [70, 50])

    def relation(relation_id: str, point_id: str) -> ParseRelation:
        return ParseRelation(
            id=relation_id,
            predicate="intersection",
            arguments=(
                ParseReference("support", axis.id),
                ParseReference("support", arc.id),
                ParseReference("point", point_id),
            ),
            origin=ParseOrigin.NATIVE,
            confidence=1.0,
            score_kind="fixture",
            evidence_refs=(evidence.id,),
            metadata={"contact": "crossing"},
        )

    return ParseGraph(
        id="graph_full_arc_intersection",
        source_id=evidence.source_id,
        evidence=(evidence,),
        objects=(point_x, point_y, axis, arc, circle, *points),
        relations=(relation("relation_x", point_x.id), relation("relation_y", point_y.id)),
        metadata={"image_size": [100, 100]},
    )


def _exact_derived_point_graph(kind: str, *, expression=None) -> ParseGraph:
    evidence = ParseEvidence(
        id="native_source",
        source_id="source_derived_point",
        kind=ParseEvidenceKind.NATIVE_ENTITY,
        precision=EvidencePrecision.BOUND,
    )
    point_a = _point("point_a", [0, 0], [0, 50])
    point_b = _point("point_b", [4, 0], [80, 50])
    segment = _segment("segment_ab", point_a.id, point_b.id)
    extra_objects = ()
    if kind == "midpoint":
        output = _point("point_m", None, [40, 50])
        relation = ParseRelation(
            id="relation_midpoint",
            predicate="midpoint",
            arguments=(
                ParseReference("point", output.id),
                ParseReference("of_0", point_a.id),
                ParseReference("of_1", point_b.id),
            ),
            origin=ParseOrigin.NATIVE,
            confidence=1.0,
            score_kind="fixture",
            evidence_refs=(evidence.id,),
        )
        extra_points = ()
    elif kind == "foot":
        output = _point("point_h", None, [20, 50])
        point_c = _point("point_c", [1, 3], [20, 10])
        relation = ParseRelation(
            id="relation_foot",
            predicate="foot_of_perpendicular",
            arguments=(
                ParseReference("point", output.id),
                ParseReference("from", point_c.id),
                ParseReference("line", segment.id),
            ),
            origin=ParseOrigin.NATIVE,
            confidence=1.0,
            score_kind="fixture",
            evidence_refs=(evidence.id,),
        )
        extra_points = (point_c,)
    elif kind == "ratio":
        output = _point("point_r", None, [27, 50])
        relation = ParseRelation(
            id="relation_ratio",
            predicate="point_on_segment_ratio",
            arguments=(
                ParseReference("point", output.id),
                ParseReference("segment", segment.id),
            ),
            origin=ParseOrigin.NATIVE,
            confidence=1.0,
            score_kind="fixture",
            evidence_refs=(evidence.id,),
            metadata={"ratio": expression if expression is not None else _div(1, 3)},
        )
        extra_points = ()
    elif kind == "ray_distance":
        output = _point("point_d", None, [60, 50])
        ray = _ray("ray_ab", point_a.id, point_b.id)
        relation = ParseRelation(
            id="relation_ray_distance",
            predicate="point_on_ray_distance",
            arguments=(
                ParseReference("point", output.id),
                ParseReference("ray", ray.id),
            ),
            origin=ParseOrigin.NATIVE,
            confidence=1.0,
            score_kind="fixture",
            evidence_refs=(evidence.id,),
            metadata={"distance": expression if expression is not None else _literal(3)},
        )
        extra_points = ()
        extra_objects = (ray,)
    else:
        raise AssertionError(f"Unsupported derived point fixture: {kind}")
    return ParseGraph(
        id=f"graph_{kind}",
        source_id=evidence.source_id,
        evidence=(evidence,),
        objects=(output, *extra_objects, segment, point_a, point_b, *extra_points),
        relations=(relation,),
        metadata={"image_size": [100, 100]},
    )


def _exact_derived_line_graph(predicate: str) -> ParseGraph:
    evidence = ParseEvidence(
        id="native_source",
        source_id="source_derived_line",
        kind=ParseEvidenceKind.NATIVE_ENTITY,
        precision=EvidencePrecision.BOUND,
    )
    point_a = _point("point_a", [0, 0], [0, 50])
    point_b = _point("point_b", [4, 0], [80, 50])
    point_c = _point("point_c", [0, 2], [0, 10])
    segment = _segment("segment_ab", point_a.id, point_b.id)
    line = ParseObject(
        id="line_l",
        kind=ParseObjectKind.LINE,
        origin=ParseOrigin.NATIVE,
        confidence=1.0,
        score_kind="fixture",
        references=(ParseReference("point", point_c.id),),
        evidence_refs=(evidence.id,),
    )
    relation = ParseRelation(
        id=f"relation_{predicate}",
        predicate=predicate,
        arguments=(
            ParseReference("object_0", line.id),
            ParseReference("object_1", segment.id),
        ),
        origin=ParseOrigin.NATIVE,
        confidence=1.0,
        score_kind="fixture",
        evidence_refs=(evidence.id,),
    )
    return ParseGraph(
        id=f"graph_line_{predicate}",
        source_id=evidence.source_id,
        evidence=(evidence,),
        objects=(line, segment, point_a, point_b, point_c),
        relations=(relation,),
        metadata={"image_size": [100, 100]},
    )


def _exact_angle_bisector_graph(*, mode: str = "internal", through_vertex: bool = True) -> ParseGraph:
    evidence = ParseEvidence(
        id="native_source",
        source_id="source_angle_bisector",
        kind=ParseEvidenceKind.NATIVE_ENTITY,
        precision=EvidencePrecision.BOUND,
    )
    point_a = _point("point_a", [4, 0], [80, 50])
    point_v = _point("point_v", [0, 0], [20, 50])
    point_c = _point("point_c", [0, 4], [20, 10])
    point_w = _point("point_w", [2, 2], [50, 30])
    through_point = point_v if through_vertex else point_w
    line = ParseObject(
        id="line_bisector",
        kind=ParseObjectKind.LINE,
        origin=ParseOrigin.NATIVE,
        confidence=1.0,
        score_kind="fixture",
        references=(ParseReference("point", through_point.id),),
        evidence_refs=(evidence.id,),
    )
    relation = ParseRelation(
        id="relation_bisector",
        predicate="angle_bisector",
        arguments=(
            ParseReference("line", line.id),
            ParseReference("arm_0", point_a.id),
            ParseReference("vertex", point_v.id),
            ParseReference("arm_1", point_c.id),
        ),
        origin=ParseOrigin.NATIVE,
        confidence=1.0,
        score_kind="fixture",
        evidence_refs=(evidence.id,),
        metadata={"mode": mode},
    )
    return ParseGraph(
        id=f"graph_angle_bisector_{mode}",
        source_id=evidence.source_id,
        evidence=(evidence,),
        objects=(line, point_a, point_v, point_c, point_w),
        relations=(relation,),
        metadata={"image_size": [100, 100]},
    )


def _circle_from_quantity_graph(
    quantities: tuple[tuple[str, int], ...],
) -> ParseGraph:
    evidence = ParseEvidence(
        id="native_source",
        source_id="source_circle_quantity",
        kind=ParseEvidenceKind.NATIVE_ENTITY,
        precision=EvidencePrecision.BOUND,
    )
    center = _point("point_o", [0, 0], [50, 50])
    circle = ParseObject(
        id="circle_o",
        kind=ParseObjectKind.CIRCLE,
        origin=ParseOrigin.NATIVE,
        confidence=1.0,
        score_kind="fixture",
        references=(ParseReference("center", center.id),),
        evidence_refs=(evidence.id,),
    )
    parse_quantities = tuple(
        ParseQuantity(
            id=f"quantity_{kind}",
            kind=kind,
            arguments=(ParseReference("object", circle.id),),
            expression=MathExpression.from_dict(_literal(value)),
            origin=ParseOrigin.NATIVE,
            confidence=1.0,
            score_kind="fixture",
            evidence_refs=(evidence.id,),
        )
        for kind, value in quantities
    )
    return ParseGraph(
        id="graph_circle_quantity",
        source_id=evidence.source_id,
        evidence=(evidence,),
        objects=(circle, center),
        quantities=parse_quantities,
        metadata={"image_size": [100, 100]},
    )


def _exact_angle_from_line_graph(*, expression=None) -> ParseGraph:
    evidence = ParseEvidence(
        id="native_source",
        source_id="source_angle_from_line",
        kind=ParseEvidenceKind.NATIVE_ENTITY,
        precision=EvidencePrecision.BOUND,
    )
    point_a = _point("point_a", [0, 0], [10, 50])
    point_b = _point("point_b", [4, 0], [80, 50])
    point_c = _point("point_c", [0, 2], [10, 20])
    base = _segment("segment_ab", point_a.id, point_b.id)
    line = ParseObject(
        id="line_rotated",
        kind=ParseObjectKind.LINE,
        origin=ParseOrigin.NATIVE,
        confidence=1.0,
        score_kind="fixture",
        references=(ParseReference("point", point_c.id),),
        evidence_refs=(evidence.id,),
    )
    angle = expression or {
        "operator": "div",
        "operands": [
            {"operator": "symbol", "value": "pi"},
            {"operator": "literal", "value": "2"},
        ],
    }
    relation = ParseRelation(
        id="relation_angle_from_line",
        predicate="angle_from_line",
        arguments=(
            ParseReference("line", line.id),
            ParseReference("point", point_c.id),
            ParseReference("base_line", base.id),
        ),
        origin=ParseOrigin.NATIVE,
        confidence=1.0,
        score_kind="fixture",
        evidence_refs=(evidence.id,),
        metadata={"angle": angle},
    )
    return ParseGraph(
        id="graph_angle_from_line",
        source_id=evidence.source_id,
        evidence=(evidence,),
        objects=(line, base, point_a, point_b, point_c),
        relations=(relation,),
        metadata={"image_size": [100, 100]},
    )


def _segment(segment_id: str, start: str, end: str) -> ParseObject:
    return ParseObject(
        id=segment_id,
        kind=ParseObjectKind.SEGMENT,
        origin=ParseOrigin.NATIVE,
        confidence=1.0,
        score_kind="fixture",
        references=(ParseReference("start", start), ParseReference("end", end)),
        evidence_refs=("native_source",),
    )


def _ray(ray_id: str, start: str, through: str) -> ParseObject:
    return ParseObject(
        id=ray_id,
        kind=ParseObjectKind.RAY,
        origin=ParseOrigin.NATIVE,
        confidence=1.0,
        score_kind="fixture",
        references=(ParseReference("start", start), ParseReference("through", through)),
        evidence_refs=("native_source",),
    )


def _circle(circle_id: str, center: str, radius: int) -> ParseObject:
    return ParseObject(
        id=circle_id,
        kind=ParseObjectKind.CIRCLE,
        origin=ParseOrigin.NATIVE,
        confidence=1.0,
        score_kind="fixture",
        references=(ParseReference("center", center),),
        evidence_refs=("native_source",),
        geometry={"exact_radius": _literal(radius)},
    )


def _arc(
    arc_id: str,
    circle: str,
    start: str,
    end: str,
    central_angle,
    *,
    allow_full_circle: bool = False,
) -> ParseObject:
    return ParseObject(
        id=arc_id,
        kind=ParseObjectKind.ARC,
        origin=ParseOrigin.NATIVE,
        confidence=1.0,
        score_kind="fixture",
        references=(
            ParseReference("circle", circle),
            ParseReference("start", start),
            ParseReference("end", end),
        ),
        evidence_refs=("native_source",),
        geometry={"orientation": "ccw", "exact_central_angle": central_angle},
        metadata={"allow_full_circle": allow_full_circle},
    )


def _region_graph() -> ParseGraph:
    evidence = ParseEvidence(
        id="native_source",
        source_id="source_regions",
        kind=ParseEvidenceKind.NATIVE_ENTITY,
        precision=EvidencePrecision.BOUND,
    )
    coordinates = {
        "A": (0, 0),
        "B": (6, 0),
        "C": (6, 5),
        "D": (0, 5),
        "E": (2, 2),
        "F": (4, 2),
        "G": (4, 3),
        "H": (2, 3),
    }
    points = tuple(
        ParseObject(
            id=point_id,
            kind=ParseObjectKind.POINT,
            origin=ParseOrigin.NATIVE,
            confidence=1.0,
            score_kind="fixture",
            evidence_refs=(evidence.id,),
            geometry={
                "pixel": [x * 10, y * 10],
                "exact_coords": [_literal(x), _literal(y)],
            },
        )
        for point_id, (x, y) in coordinates.items()
    )
    outer = ParseRegion(
        id="outer_region",
        boundary=tuple(BoundaryReference(point_id) for point_id in "ABCD"),
        origin=ParseOrigin.NATIVE,
        confidence=1.0,
        score_kind="fixture",
        evidence_refs=(evidence.id,),
    )
    hole = ParseRegion(
        id="hole_region",
        boundary=tuple(BoundaryReference(point_id) for point_id in "EFGH"),
        origin=ParseOrigin.NATIVE,
        confidence=1.0,
        score_kind="fixture",
        evidence_refs=(evidence.id,),
    )
    shaded = ParseRegion(
        id="shaded_region",
        boundary=(),
        components=(outer.id, hole.id),
        holes=(hole.id,),
        expression=RegionExpression(
            "difference",
            operands=(RegionExpression("reference", ref=outer.id), RegionExpression("reference", ref=hole.id)),
        ),
        origin=ParseOrigin.COMPILED,
        confidence=1.0,
        score_kind="deterministic_rule",
    )
    target = ParseTarget(
        id="target_region",
        kind="area",
        mode="value",
        arguments=(ParseReference("object", shaded.id),),
        origin=ParseOrigin.NATIVE,
        confidence=1.0,
        score_kind="fixture",
        evidence_refs=(evidence.id,),
    )
    return ParseGraph(
        id="graph_regions",
        source_id="source_regions",
        evidence=(evidence,),
        objects=points,
        regions=(outer, hole, shaded),
        targets=(target,),
        metadata={"image_size": [100, 100]},
    )


class _FixtureParser:
    def __init__(self, result: TypedParseResult):
        self.result = result

    def parse(self, data):
        assert data == self.result.parser_input
        return self.result


def _point(point_id: str, exact_coords, pixel) -> ParseObject:
    geometry = {"pixel": pixel}
    if exact_coords is not None:
        geometry["exact_coords"] = [_literal(value) if isinstance(value, int) else value for value in exact_coords]
    return ParseObject(
        id=point_id,
        kind=ParseObjectKind.POINT,
        origin=ParseOrigin.NATIVE,
        confidence=1.0,
        score_kind="fixture",
        evidence_refs=("native_source",),
        geometry=geometry,
    )


def _literal(value: int) -> dict:
    return MathExpression("literal", value=str(value)).to_dict()


def _sqrt(value: int) -> dict:
    return MathExpression("sqrt", operands=(MathExpression("literal", value=str(value)),)).to_dict()


def _div(left: int, right: int) -> dict:
    return MathExpression(
        "div",
        operands=(MathExpression("literal", value=str(left)), MathExpression("literal", value=str(right))),
    ).to_dict()


def _pi_div(denominator: int) -> dict:
    if denominator == 1:
        return MathExpression("symbol", value="pi").to_dict()
    return MathExpression(
        "div",
        operands=(
            MathExpression("symbol", value="pi"),
            MathExpression("literal", value=str(denominator)),
        ),
    ).to_dict()


def _two_pi() -> dict:
    return MathExpression(
        "mul",
        operands=(
            MathExpression("literal", value="2"),
            MathExpression("symbol", value="pi"),
        ),
    ).to_dict()
