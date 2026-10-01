import math
from dataclasses import replace

import pytest

from geosym_world.parser import ConstraintApplier, SeedRealizationStatus, SeedRealizer
from geosym_world.errors import InvalidGeometryInput
from geosym_world.schema import (
    ConflictStatus,
    EvidencePrecision,
    MathExpression,
    ParseConflictGroup,
    ParseEvidence,
    ParseEvidenceKind,
    ParseGraph,
    ParseObject,
    ParseObjectKind,
    ParseOrigin,
    ParseQuantity,
    ParseReference,
    ParseRelation,
    ParseTarget,
)
from geosym_world.verifier import ParseAdmissionPolicy, ParseGraphAdmissionReport
from geosym_world.construction import ConstructionBuilder
from geosym_world.verifier import TraceVerifier


def _native_object(object_id, kind, name=None, references=()):
    return ParseObject(
        id=object_id,
        kind=kind,
        origin=ParseOrigin.NATIVE,
        confidence=1.0,
        score_kind="seed_fixture",
        visible_name=name,
        references=tuple(ParseReference(role, ref) for role, ref in references),
    )


@pytest.mark.parametrize("paired", [True, False])
@pytest.mark.parametrize("exact", [True, False])
def test_midpoint_is_checked_materialized_and_independently_verified(paired, exact):
    points = tuple(
        replace(_native_object(name, ParseObjectKind.POINT, name), geometry=(
            {"exact_coords": [MathExpression("literal", value=str(value)).to_dict() for value in coords]}
            if exact else {"pixel": list(coords)}
        ))
        for name, coords in [("U", (0, 0)), ("V", (4, 0)), ("M", (0, 2))]
    )
    segment = _native_object("edge", ParseObjectKind.SEGMENT, references=(("start", "U"), ("end", "V")))
    arguments = (("support", "edge"), ("point", "M")) if paired else (("end", "V"), ("point", "M"), ("start", "U"))
    relation = ParseRelation(
        id="middle", predicate="midpoint", arguments=tuple(ParseReference(*item) for item in arguments),
        origin=ParseOrigin.NATIVE, confidence=1.0, score_kind="test",
    )
    graph = ParseGraph(id="midpoint", source_id="midpoint", objects=points + (segment,), relations=(relation,))

    result = SeedRealizer().realize_proposal(graph)

    if exact:
        assert result.status is SeedRealizationStatus.PARTIAL
        assert "middle" not in result.satisfied_constraints
        assert result.constraint_outcomes[0].status.value == "violated"
    else:
        assert result.status is SeedRealizationStatus.REALIZED
        positions = result.point_positions
        expected = tuple((left + right) / 2 for left, right in zip(positions["U"], positions["V"]))
        assert math.dist(positions["M"], expected) < 1e-5
        assert result.geometry_state.relations[0]["type"] == "midpoint"
        assert TraceVerifier().verify_state(result.geometry_state).valid
        replay = ConstructionBuilder().execute(result.geometry_state.construction_trace)
        assert replay.checksum() == result.geometry_state.checksum()


def test_unknown_relation_check_and_degenerate_edges_never_pass():
    relation = ParseRelation(
        id="test", predicate="parallel", arguments=(ParseReference("left", "first"), ParseReference("right", "second")),
        origin=ParseOrigin.NATIVE, confidence=1.0, score_kind="test",
    )
    positions = {"A": (0, 0), "B": (0, 0), "C": (1, 0), "D": (2, 0)}
    edges = {"first": ("A", "B"), "second": ("C", "D")}
    assert not SeedRealizer._relation_holds(relation, positions, edges, {})
    assert not SeedRealizer._relation_holds(replace(relation, predicate="equal_angle"), positions, edges, {})
    with pytest.raises(InvalidGeometryInput, match="midpoint"):
        replace(relation, predicate="midpoint")


def test_center_defined_arc_is_materialized_with_hidden_support_circle():
    points = tuple(
        replace(_native_object(name, ParseObjectKind.POINT, name), geometry={"pixel": list(coords)})
        for name, coords in [("O", (0, 0)), ("U", (4, 0)), ("V", (0, 4))]
    )
    segment = _native_object("OU", ParseObjectKind.SEGMENT, references=(("start", "O"), ("end", "U")))
    arc = _native_object("curve", ParseObjectKind.ARC, references=(("center", "O"), ("start", "U"), ("end", "V")))
    graph = ParseGraph(id="arc", source_id="arc", objects=points + (segment, arc))

    result = SeedRealizer().realize_proposal(graph)

    assert result.geometry_state is not None
    assert result.verification_report["valid"]
    assert result.status is SeedRealizationStatus.REALIZED
    assert result.metadata["materialization"]["missing_object_ids"] == []
    assert "curve" in result.geometry_state.arcs
    support = result.geometry_state.arcs["curve"].circle
    assert result.geometry_state.circles[support].metadata["render"] is False
    assert not any(primitive.kind == "circle" for primitive in result.scene.primitives)
    replayed = ConstructionBuilder().execute(result.geometry_state.construction_trace)
    assert replayed.checksum() == result.geometry_state.checksum()


@pytest.mark.parametrize("exact", [True, False])
def test_value_target_is_preserved_without_becoming_a_constraint(exact):
    points = tuple(
        replace(_native_object(f"point_{name}", ParseObjectKind.POINT, name), geometry=(
            {"exact_coords": [MathExpression("literal", value=str(value)).to_dict() for value in coords]}
            if exact else {"pixel": list(coords)}
        ))
        for name, coords in [("A", (0, 0)), ("B", (4, 0))]
    )
    edge = _native_object("edge", ParseObjectKind.SEGMENT, references=(("start", "point_A"), ("end", "point_B")))
    target = ParseTarget(
        id="ask", kind="length", mode="value", arguments=(ParseReference("object", "edge"),),
        origin=ParseOrigin.TEXT, confidence=0.8, score_kind="test",
    )
    graph = ParseGraph(id="task", source_id="task", objects=points + (edge,), targets=(target,))

    result = SeedRealizer().realize_proposal(graph)

    assert result.status is SeedRealizationStatus.REALIZED
    assert result.geometry_state.target.type == "length"
    assert result.geometry_state.target.object in result.geometry_state.segments
    assert result.geometry_state.target.metadata["parse_target"] == target.to_dict()
    assert not result.constraint_outcomes
    assert not result.geometry_state.relations
    replay = ConstructionBuilder().execute(result.geometry_state.construction_trace)
    assert replay.checksum() == result.geometry_state.checksum()
    multiple = SeedRealizer().realize_proposal(replace(graph, targets=(target, replace(target, id="second"))))
    assert multiple.geometry_state.target is None
    assert multiple.metadata["materialization"]["missing_target_ids"] == ["ask", "second"]
    assert multiple.metadata["task_targets"] == [target.to_dict(), replace(target, id="second").to_dict()]


@pytest.mark.parametrize("measurement_kind,value", [("length", "10"), ("radius", "10"), ("diameter", "20")])
def test_curve_scale_and_explicit_circle_incidence_share_mathematical_units(measurement_kind, value):
    points = tuple(
        replace(_native_object(name, ParseObjectKind.POINT, name), geometry={"pixel": list(coords)})
        for name, coords in [("O", (0, 0)), ("P", (92, 8)), ("Q", (0, 100))]
    )
    circle = replace(_native_object("ring", ParseObjectKind.CIRCLE, references=(("center", "O"),)), geometry={"radius_px": 100})
    segment = _native_object("OQ", ParseObjectKind.SEGMENT, references=(("start", "O"), ("end", "Q")))
    quantity = ParseQuantity(
        id="measure", kind=measurement_kind, expression=MathExpression("literal", value=value),
        arguments=(ParseReference("object", "OQ" if measurement_kind == "length" else "ring"),),
        origin=ParseOrigin.NATIVE, confidence=1.0, score_kind="test",
    )
    incidence = ParseRelation(
        id="on_ring", predicate="point_on_circle",
        arguments=(ParseReference("circle", "ring"), ParseReference("point", "P")),
        origin=ParseOrigin.NATIVE, confidence=1.0, score_kind="test",
    )
    graph = ParseGraph(id="scaled", source_id="scaled", objects=points + (circle, segment), quantities=(quantity,), relations=(incidence,))
    result = SeedRealizer().realize_proposal(graph)
    assert result.status is SeedRealizationStatus.REALIZED
    assert result.metadata["layout_scale"] == pytest.approx(0.1)
    assert float(result.geometry_state.circles["ring"].radius) == pytest.approx(10)
    assert math.dist(result.point_positions["O"], result.point_positions["P"]) == pytest.approx(10, abs=1e-5)
    assert result.verification_report["valid"]
    assert set(result.satisfied_constraints) == {"on_ring", "measure"}


def test_structural_conflict_stops_realization_without_deforming_the_picture():
    points = tuple(
        replace(_native_object(name, ParseObjectKind.POINT, name), geometry={"pixel": list(coords)})
        for name, coords in [("U", (0, 0)), ("V", (5, 0)), ("X", (2, 0))]
    )
    segments = tuple(_native_object(name, ParseObjectKind.SEGMENT, references=(("start", start), ("end", end))) for name, start, end in [("base", "U", "V"), ("height", "U", "X")])
    relations = (
        ParseRelation(id="on", predicate="point_on_segment", arguments=(ParseReference("point", "X"), ParseReference("support", "base")), origin=ParseOrigin.NATIVE, confidence=1.0, score_kind="test"),
        ParseRelation(id="right", predicate="perpendicular", arguments=(ParseReference("left", "height"), ParseReference("right", "base")), origin=ParseOrigin.NATIVE, confidence=1.0, score_kind="test"),
    )
    graph = ParseGraph(id="conflict", source_id="conflict", objects=points + segments, relations=relations)
    result = SeedRealizer().realize_proposal(graph)
    assert result.status is SeedRealizationStatus.UNSUPPORTED
    assert result.scene is None and result.geometry_state is None
    assert result.metadata["constraint_conflicts"][0]["candidate_ids"] == ["on", "right"]
    assert result.diagnostics == ("inconsistent_structural_constraints",)


def test_numeric_arc_tolerance_does_not_relax_exact_state_validation():
    trace = [
        {"op": "add_point", "id": "O", "coord": [0, 0]},
        {"op": "add_point", "id": "A", "coord": [1, 0]},
        {"op": "add_point", "id": "B", "coord": [0, 1.00000001]},
        {"op": "add_circle", "id": "circle", "center": "O", "radius": 1},
    ]
    arc = {"op": "add_arc", "id": "arc", "circle": "circle", "start": "A", "end": "B"}
    with pytest.raises(InvalidGeometryInput, match="lie on its circle"):
        ConstructionBuilder().execute([*trace, arc])
    state = ConstructionBuilder().execute([*trace, {**arc, "metadata": {"verification_mode": "numeric_witness", "numeric_tolerance": 1e-5}}])
    assert TraceVerifier().verify_state(state).valid


@pytest.mark.parametrize("angle", [0.0, math.pi / 3])
def test_seed_preserves_observed_layout_while_solving_equal_perpendicular_edges(angle):
    original = {"u": (20.0, 30.0), "v": (24.0, 30.1), "w": (20.1, 34.2)}
    positions = {
        name: (x * math.cos(angle) - y * math.sin(angle), x * math.sin(angle) + y * math.cos(angle))
        for name, (x, y) in original.items()
    }
    points = tuple(
        replace(_native_object(name, ParseObjectKind.POINT), geometry={"pixel": [x, -y]})
        for name, (x, y) in positions.items()
    )
    edges = tuple(
        _native_object(edge, ParseObjectKind.SEGMENT, references=(("start", "u"), ("end", end)))
        for edge, end in [("first", "v"), ("second", "w")]
    )
    relations = tuple(
        ParseRelation(
            id=predicate, predicate=predicate,
            arguments=(ParseReference("left", "first"), ParseReference("right", "second")),
            origin=ParseOrigin.NATIVE, confidence=1.0, score_kind="test",
        )
        for predicate in ["equal_length", "perpendicular"]
    )
    graph = ParseGraph(id="layout", source_id="layout", objects=points + edges, relations=relations)

    result = SeedRealizer().realize_proposal(graph)

    assert result.status is SeedRealizationStatus.REALIZED
    assert result.metadata["coordinate_source"] == "refined_observed_pixels"
    assert result.verification_report["valid"]
    assert set(result.satisfied_constraints) == {"equal_length", "perpendicular"}
    assert all(math.dist(result.point_positions[name], position) < 0.3 for name, position in positions.items())


def test_exact_coordinates_do_not_automatically_satisfy_all_proposed_relations():
    points = tuple(
        replace(_native_object(name, ParseObjectKind.POINT, name), geometry={
            "exact_coords": [MathExpression("literal", value=str(value)).to_dict() for value in coords],
        })
        for name, coords in [("U", [0, 0]), ("V", [1, 0]), ("W", [0, 2])]
    )
    edges = tuple(
        _native_object(edge, ParseObjectKind.SEGMENT, references=(("start", "U"), ("end", end)))
        for edge, end in [("UV", "V"), ("UW", "W")]
    )
    relation = ParseRelation(
        id="unequal", predicate="equal_length",
        arguments=(ParseReference("left", "UV"), ParseReference("right", "UW")),
        origin=ParseOrigin.NATIVE, confidence=1.0, score_kind="test",
    )
    graph = ParseGraph(id="exact_conflict", source_id="exact_conflict", objects=points + edges, relations=(relation,))

    result = SeedRealizer().realize_proposal(graph)

    assert result.status is SeedRealizationStatus.PARTIAL
    assert "unequal" in result.unresolved_constraints
    assert "unequal" not in result.satisfied_constraints


@pytest.mark.parametrize("error, numeric, expected", [(1e-8, True, True), (0.1, True, False), (1e-8, False, False)])
def test_equal_length_numeric_witness_tolerance_does_not_relax_exact_verification(error, numeric, expected):
    metadata = {"verification_mode": "numeric_witness", "numeric_tolerance": 1e-5} if numeric else {}
    state = ConstructionBuilder().execute([
        {"op": "add_point", "id": "A", "coord": [0, 0]},
        {"op": "add_point", "id": "B", "coord": [1, 0]},
        {"op": "add_point", "id": "C", "coord": [0, 1 + error]},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
        {"op": "add_segment", "id": "AC", "points": ["A", "C"]},
        {"op": "add_relation", "id": "equal", "type": "equal_length", "objects": ["AB", "AC"], "metadata": metadata},
    ])

    assert TraceVerifier().verify_state(state).valid is expected


def _seed_point(point_id, position, name=None):
    return ParseObject(
        id=point_id,
        kind=ParseObjectKind.POINT,
        origin=ParseOrigin.NATIVE,
        confidence=1.0,
        score_kind="seed_fixture",
        visible_name=name,
        metadata={"seed_normalized": position},
    )


def _parallel_triangle_admission():
    points = tuple(_native_object(f"point:{name}", ParseObjectKind.POINT, name) for name in "PRQST")
    segments = (
        _native_object("segment:PR", ParseObjectKind.SEGMENT, "PR", (("start", "point:P"), ("end", "point:R"))),
        _native_object("segment:PQ", ParseObjectKind.SEGMENT, "PQ", (("start", "point:P"), ("end", "point:Q"))),
        _native_object("segment:RQ", ParseObjectKind.SEGMENT, "RQ", (("start", "point:R"), ("end", "point:Q"))),
        _native_object("segment:ST", ParseObjectKind.SEGMENT, "ST", (("start", "point:S"), ("end", "point:T"))),
    )
    relations = (
        ParseRelation(
            id="relation:S_on_PR",
            predicate="incidence",
            arguments=(ParseReference("point", "point:S"), ParseReference("support", "segment:PR")),
            origin=ParseOrigin.NATIVE,
            confidence=1.0,
            score_kind="seed_fixture",
        ),
        ParseRelation(
            id="relation:T_on_PQ",
            predicate="incidence",
            arguments=(ParseReference("point", "point:T"), ParseReference("support", "segment:PQ")),
            origin=ParseOrigin.NATIVE,
            confidence=1.0,
            score_kind="seed_fixture",
        ),
        ParseRelation(
            id="relation:ST_parallel_RQ",
            predicate="parallel",
            arguments=(ParseReference("left", "segment:ST"), ParseReference("right", "segment:RQ")),
            origin=ParseOrigin.NATIVE,
            confidence=1.0,
            score_kind="seed_fixture",
        ),
    )
    graph = ParseGraph(
        id="parse_graph:seed_fixture",
        source_id="seed_fixture",
        objects=points + segments,
        relations=relations,
    )
    return ParseGraphAdmissionReport(graph.checksum(), ParseAdmissionPolicy(), (), graph)


def test_seed_realizer_handles_underdetermined_parallel_triangle():
    result = SeedRealizer().realize(_parallel_triangle_admission())

    assert result.status is SeedRealizationStatus.REALIZED
    assert result.metadata["realization_mode"] == "seed"
    assert result.metadata["exact"] is False
    assert set(result.satisfied_constraints) == {
        "relation:S_on_PR",
        "relation:T_on_PQ",
        "relation:ST_parallel_RQ",
    }
    assert result.unresolved_constraints == ()
    assert result.scene is not None
    assert {label.text for label in result.scene.labels} == set("PRQST")
    assert len([item for item in result.scene.primitives if item.kind == "segment"]) == 4
    assert "point_fraction" in result.free_parameters


def test_seed_realizer_serialization_is_stable():
    result = SeedRealizer(point_fraction=0.4).realize(_parallel_triangle_admission())
    restored = result.to_dict()

    assert restored["checksum"] == result.checksum()
    assert restored["status"] == "realized"
    assert restored["metadata"]["point_fraction"] == 0.4


def test_seed_realizer_materializes_admitted_anonymous_structure_without_labels():
    points = tuple(_native_object(f"visual:vp_{index}", ParseObjectKind.POINT) for index in range(2))
    segment = _native_object(
        "visual:vs_0",
        ParseObjectKind.SEGMENT,
        references=(("start", "visual:vp_0"), ("end", "visual:vp_1")),
    )
    graph = ParseGraph(
        id="parse_graph:visual_only",
        source_id="visual_only",
        objects=points + (segment,),
    )
    admission = ParseGraphAdmissionReport(graph.checksum(), ParseAdmissionPolicy(), (), graph)

    result = SeedRealizer().realize(admission)

    assert result.status is SeedRealizationStatus.REALIZED
    assert result.geometry_state is not None
    assert set(result.geometry_state.points) == {"visual:vp_0", "visual:vp_1"}
    assert set(result.geometry_state.segments) == {"visual:vs_0"}
    assert result.scene is not None
    assert result.scene.labels == ()


def test_seed_realizer_materializes_circle_without_linear_support():
    center = _seed_point("point:o", (0.5, 0.5), "O")
    circle = ParseObject(
        id="circle_o",
        kind=ParseObjectKind.CIRCLE,
        origin=ParseOrigin.NATIVE,
        confidence=1.0,
        score_kind="seed_fixture",
        references=(ParseReference("center", "point:o"),),
        geometry={"radius_px": 2.0},
    )
    graph = ParseGraph(
        id="parse_graph:circle_only",
        source_id="circle_only",
        objects=(center, circle),
    )

    result = SeedRealizer().realize_proposal(graph)

    assert result.status is SeedRealizationStatus.REALIZED
    assert result.geometry_state is not None
    assert set(result.geometry_state.circles) == {"circle_o"}
    assert result.scene is not None
    assert any(item.id == "circle_o" for item in result.scene.primitives)


def test_seed_realizer_completes_circle_tangent_to_perpendicular_carriers():
    points = tuple(_native_object(name, ParseObjectKind.POINT, name) for name in ("O", "A", "B", "K"))
    segments = (
        _native_object("OA", ParseObjectKind.SEGMENT, references=(("start", "O"), ("end", "A"))),
        _native_object("OB", ParseObjectKind.SEGMENT, references=(("start", "O"), ("end", "B"))),
    )
    outer = _native_object(
        "outer",
        ParseObjectKind.CIRCLE,
        references=(("center", "O"),),
    )
    inner = _native_object(
        "inner",
        ParseObjectKind.CIRCLE,
        references=(("center", "K"),),
    )
    arc = _native_object(
        "quarter",
        ParseObjectKind.ARC,
        references=(("circle", "outer"), ("start", "A"), ("end", "B")),
    )
    perpendicular = ParseRelation(
        id="right_angle",
        predicate="perpendicular",
        arguments=(ParseReference("left", "OA"), ParseReference("right", "OB")),
        origin=ParseOrigin.NATIVE,
        confidence=1.0,
        score_kind="test",
    )
    tangencies = tuple(
        ParseRelation(
            id=relation_id,
            predicate="tangent",
            arguments=(ParseReference("left", "inner"), ParseReference(role, reference)),
            origin=ParseOrigin.NATIVE,
            confidence=1.0,
            score_kind="test",
        )
        for relation_id, role, reference in (
            ("inner_outer", "right", "outer"),
            ("inner_oa", "right", "OA"),
            ("inner_ob", "right", "OB"),
        )
    )
    quantity = ParseQuantity(
        id="outer_radius",
        kind="radius",
        expression=MathExpression("literal", value="6"),
        arguments=(ParseReference("object", "outer"),),
        origin=ParseOrigin.TEXT,
        confidence=1.0,
        score_kind="test",
    )
    target = ParseTarget(
        id="radius_target",
        kind="radius",
        mode="value",
        arguments=(ParseReference("object", "inner"),),
        origin=ParseOrigin.TEXT,
        confidence=1.0,
        score_kind="test",
    )
    graph = ParseGraph(
        id="quarter_tangent_circle",
        source_id="quarter_tangent_circle",
        objects=points + segments + (outer, inner, arc),
        relations=(perpendicular, *tangencies),
        quantities=(quantity,),
        targets=(target,),
    )

    result = SeedRealizer().realize_proposal(graph)

    assert result.status is SeedRealizationStatus.REALIZED
    assert result.geometry_state is not None
    assert result.geometry_state.target.object == "inner"
    assert float(result.geometry_state.circles["inner"].radius) == pytest.approx(6 * (math.sqrt(2) - 1), abs=1e-6)
    assert {
        item.get("metadata", {}).get("source_parse_relation_id")
        for item in result.geometry_state.relations
        if item.get("metadata", {}).get("source_parse_relation_id")
    } == {
        "right_angle", "inner_outer", "inner_oa", "inner_ob",
    }
    assert result.verification_report["valid"]


@pytest.mark.parametrize(
    ("kind", "predicate", "state_bucket"),
    (
        (ParseObjectKind.LINE, "point_on_line", "lines"),
        (ParseObjectKind.RAY, "point_on_ray", "rays"),
    ),
)
def test_seed_realizer_preserves_linear_object_domain(kind, predicate, state_bucket):
    points = (
        _seed_point("point:a", (0.1, 0.5)),
        _seed_point("point:b", (0.4, 0.5)),
        _seed_point("point:x", (0.8, 0.5)),
    )
    support = _native_object(
        "support:ab",
        kind,
        references=(("start", "point:a"), ("through", "point:b")),
    )
    relation = ParseRelation(
        id="relation:x_on_support",
        predicate=predicate,
        arguments=(ParseReference("point", "point:x"), ParseReference("support", "support:ab")),
        origin=ParseOrigin.NATIVE,
        confidence=1.0,
        score_kind="seed_fixture",
    )
    graph = ParseGraph(
        id=f"parse_graph:{kind.value}_domain",
        source_id=f"{kind.value}_domain",
        objects=points + (support,),
        relations=(relation,),
    )

    result = SeedRealizer().realize_proposal(graph)

    assert result.status is SeedRealizationStatus.REALIZED
    assert result.geometry_state is not None
    assert set(getattr(result.geometry_state, state_bucket)) == {"support:ab"}
    assert result.geometry_state.relations[0]["type"] == predicate


def test_seed_realizer_materializes_strict_order_on_segment():
    points = tuple(
        _native_object(f"point:{name}", ParseObjectKind.POINT, name)
        for name in "ABCD"
    )
    segment = _native_object(
        "segment:AD",
        ParseObjectKind.SEGMENT,
        "AD",
        (("start", "point:A"), ("end", "point:D")),
    )
    relation = ParseRelation(
        id="relation:ABCD_order",
        predicate="ordered_points",
        arguments=(
            ParseReference("support", "segment:AD"),
            *(ParseReference(f"point_{index}", f"point:{name}") for index, name in enumerate("ABCD")),
        ),
        origin=ParseOrigin.NATIVE,
        confidence=1.0,
        score_kind="seed_fixture",
    )
    graph = ParseGraph(
        id="parse_graph:ordered_points",
        source_id="ordered_points",
        objects=points + (segment,),
        relations=(relation,),
    )

    result = SeedRealizer().realize_proposal(graph)

    assert result.status is SeedRealizationStatus.REALIZED
    coordinates = [result.point_positions[f"point:{name}"] for name in "ABCD"]
    distances = [math.dist(coordinates[index], coordinates[index + 1]) for index in range(3)]
    assert min(distances) > 0.0
    assert max(distances) == pytest.approx(min(distances))
    assert result.geometry_state is not None
    assert result.geometry_state.relations[0]["type"] == "ordered_points"


def test_seed_realizer_accepts_named_proposal_without_promoting_facts():
    admission = _parallel_triangle_admission()

    result = SeedRealizer().realize_proposal(admission.admitted_graph)

    assert result.status is SeedRealizationStatus.REALIZED
    assert result.metadata["input_status"] == "proposal"
    assert result.metadata["exact"] is False
    assert set(result.satisfied_constraints) == {
        "relation:S_on_PR",
        "relation:T_on_PQ",
        "relation:ST_parallel_RQ",
    }


def test_seed_realizer_places_point_at_two_segment_intersection():
    points = tuple(_native_object(f"point:{name}", ParseObjectKind.POINT, name) for name in "ABCDX")
    segments = (
        _native_object("segment:AC", ParseObjectKind.SEGMENT, "AC", (("start", "point:A"), ("end", "point:C"))),
        _native_object("segment:BD", ParseObjectKind.SEGMENT, "BD", (("start", "point:B"), ("end", "point:D"))),
    )
    relations = tuple(
        ParseRelation(
            id=f"relation:X_on_{segment}",
            predicate="point_on_segment",
            arguments=(ParseReference("point", "point:X"), ParseReference("support", f"segment:{segment}")),
            origin=ParseOrigin.NATIVE,
            confidence=1.0,
            score_kind="seed_fixture",
        )
        for segment in ("AC", "BD")
    )
    graph = ParseGraph(
        id="parse_graph:segment_intersection_seed",
        source_id="segment_intersection_seed",
        objects=points + segments,
        relations=relations,
    )
    admission = ParseGraphAdmissionReport(graph.checksum(), ParseAdmissionPolicy(), (), graph)

    result = SeedRealizer().realize(admission)

    assert result.status is SeedRealizationStatus.REALIZED
    assert set(result.satisfied_constraints) == {"relation:X_on_AC", "relation:X_on_BD"}
    assert result.unresolved_constraints == ()


def test_seed_realizer_uses_visual_scaffold_and_snaps_named_intersection():
    positions = {
        "A": [0.1, 0.9],
        "B": [0.1, 0.1],
        "C": [0.6, 0.1],
        "D": [0.6, 0.9],
        "E": [0.9, 0.6],
        "F": [0.6, 0.7],
    }
    points = tuple(
        ParseObject(
            id=f"point:{name}",
            kind=ParseObjectKind.POINT,
            origin=ParseOrigin.NATIVE,
            confidence=1.0,
            score_kind="seed_fixture",
            visible_name=name,
            metadata={"seed_normalized": position, "coordinate_policy": "visual_scaffold_only"},
        )
        for name, position in positions.items()
    )
    segments = (
        _native_object("segment:AE", ParseObjectKind.SEGMENT, "AE", (("start", "point:A"), ("end", "point:E"))),
        _native_object("segment:CD", ParseObjectKind.SEGMENT, "CD", (("start", "point:C"), ("end", "point:D"))),
    )
    relations = tuple(
        ParseRelation(
            id=f"relation:F_on_{segment}",
            predicate="point_on_segment",
            arguments=(ParseReference("point", "point:F"), ParseReference("support", f"segment:{segment}")),
            origin=ParseOrigin.NATIVE,
            confidence=1.0,
            score_kind="seed_fixture",
        )
        for segment in ("AE", "CD")
    )
    graph = ParseGraph(
        id="parse_graph:visual_scaffold",
        source_id="visual_scaffold",
        objects=points + segments,
        relations=relations,
    )

    result = SeedRealizer().realize_proposal(graph)

    assert result.status is SeedRealizationStatus.REALIZED
    assert result.metadata["coordinate_source"] == "semantic_scaffold"
    assert result.point_positions["point:F"] == pytest.approx((0.6, 0.2875))
    assert set(result.satisfied_constraints) == {"relation:F_on_AE", "relation:F_on_CD"}


def test_seed_realizer_projects_collinear_middle_point_onto_declared_supports():
    positions = {
        "A": [0.1, 0.9],
        "C": [0.6, 0.1],
        "D": [0.6, 0.9],
        "E": [0.9, 0.6],
        "F": [0.6, 0.7],
    }
    points = tuple(
        ParseObject(
            id=f"point:{name}",
            kind=ParseObjectKind.POINT,
            origin=ParseOrigin.NATIVE,
            confidence=1.0,
            score_kind="seed_fixture",
            visible_name=name,
            metadata={"seed_normalized": position},
        )
        for name, position in positions.items()
    )
    segments = (
        _native_object("segment:AE", ParseObjectKind.SEGMENT, "AE", (("start", "point:A"), ("end", "point:E"))),
        _native_object("segment:CD", ParseObjectKind.SEGMENT, "CD", (("start", "point:C"), ("end", "point:D"))),
    )
    relations = (
        ParseRelation(
            id="relation:AFE",
            predicate="collinear",
            arguments=tuple(ParseReference("point", f"point:{name}") for name in "AFE"),
            origin=ParseOrigin.NATIVE,
            confidence=1.0,
            score_kind="seed_fixture",
        ),
        ParseRelation(
            id="relation:CFD",
            predicate="collinear",
            arguments=tuple(ParseReference("point", f"point:{name}") for name in "CFD"),
            origin=ParseOrigin.NATIVE,
            confidence=1.0,
            score_kind="seed_fixture",
        ),
    )
    graph = ParseGraph(
        id="parse_graph:collinear_scaffold",
        source_id="collinear_scaffold",
        objects=points + segments,
        relations=relations,
    )

    result = SeedRealizer().realize_proposal(graph)

    assert result.status is SeedRealizationStatus.REALIZED
    assert set(result.satisfied_constraints) == {"relation:AFE", "relation:CFD"}


def test_seed_realizer_rejects_proposal_with_unresolved_conflicts():
    graph = _parallel_triangle_admission().admitted_graph
    conflicted = ParseGraph(
        id="parse_graph:conflicted_seed",
        source_id=graph.source_id,
        objects=graph.objects,
        relations=graph.relations,
        conflicts=(
            ParseConflictGroup(
                id="conflict:parallel",
                members=("relation:S_on_PR", "relation:T_on_PQ"),
                status=ConflictStatus.UNRESOLVED,
                reason="fixture_conflict",
            ),
        ),
    )

    result = SeedRealizer().realize_proposal(conflicted)

    assert result.status is SeedRealizationStatus.UNSUPPORTED
    assert result.diagnostics == ("unresolved_conflicts",)


def test_seed_realizer_rejects_points_inferred_only_from_relation_tokens():
    points = tuple(
        ParseObject(
            id=f"semantic:{name}",
            kind=ParseObjectKind.POINT,
            origin=ParseOrigin.IMAGE_TEXT,
            confidence=0.25,
            score_kind="semantic_unbound_candidate",
            visible_name=name,
            metadata={"binding_status": "unresolved", "declared_in_inventory": False},
        )
        for name in "THE"
    )
    polygon = ParseObject(
        id="semantic:THE",
        kind=ParseObjectKind.POLYGON,
        origin=ParseOrigin.IMAGE_TEXT,
        confidence=0.25,
        score_kind="semantic_unbound_candidate",
        visible_name="THE",
        references=tuple(ParseReference(f"vertex_{index}", f"semantic:{name}") for index, name in enumerate("THE")),
        metadata={"binding_status": "unresolved", "declared_in_inventory": False},
    )
    graph = ParseGraph(
        id="parse_graph:token_inference",
        source_id="token_inference",
        objects=points + (polygon,),
    )

    result = SeedRealizer().realize_proposal(graph)

    assert result.status is SeedRealizationStatus.UNSUPPORTED
    assert result.diagnostics == ("no_supported_points",)


def test_seed_realizer_rejects_unbound_image_inventory_names():
    points = tuple(
        ParseObject(
            id=f"semantic:{name}",
            kind=ParseObjectKind.POINT,
            origin=ParseOrigin.IMAGE_TEXT,
            confidence=0.8,
            score_kind="model",
            visible_name=name,
            metadata={"binding_status": "unresolved", "declared_in_inventory": True},
        )
        for name in "ABC"
    )
    polygon = ParseObject(
        id="semantic:ABC",
        kind=ParseObjectKind.POLYGON,
        origin=ParseOrigin.IMAGE_TEXT,
        confidence=0.8,
        score_kind="model",
        visible_name="ABC",
        references=tuple(ParseReference(f"vertex_{index}", f"semantic:{name}") for index, name in enumerate("ABC")),
        metadata={"binding_status": "unresolved", "declared_in_inventory": True},
    )
    graph = ParseGraph(
        id="parse_graph:unbound_image_inventory",
        source_id="unbound_image_inventory",
        objects=points + (polygon,),
    )

    result = SeedRealizer().realize_proposal(graph)

    assert result.status is SeedRealizationStatus.UNSUPPORTED
    assert result.diagnostics == ("no_supported_points",)


def test_seed_realizer_accepts_points_supported_by_text_relation():
    points = tuple(
        ParseObject(
            id=f"semantic:{name}",
            kind=ParseObjectKind.POINT,
            origin=ParseOrigin.IMAGE_TEXT,
            confidence=0.8,
            score_kind="model",
            visible_name=name,
            metadata={"binding_status": "unresolved", "declared_in_inventory": True},
        )
        for name in "ABC"
    )
    segments = (
        ParseObject(
            id="semantic:AB",
            kind=ParseObjectKind.SEGMENT,
            origin=ParseOrigin.IMAGE_TEXT,
            confidence=0.8,
            score_kind="model",
            references=(ParseReference("start", "semantic:A"), ParseReference("end", "semantic:B")),
        ),
        ParseObject(
            id="semantic:AC",
            kind=ParseObjectKind.SEGMENT,
            origin=ParseOrigin.IMAGE_TEXT,
            confidence=0.8,
            score_kind="model",
            references=(ParseReference("start", "semantic:A"), ParseReference("end", "semantic:C")),
        ),
    )
    relation = ParseRelation(
        id="relation:text_perpendicular",
        predicate="perpendicular",
        arguments=(ParseReference("left", "semantic:AB"), ParseReference("right", "semantic:AC")),
        origin=ParseOrigin.IMAGE_TEXT,
        confidence=0.9,
        score_kind="model",
        metadata={"source_type": "text"},
    )
    graph = ParseGraph(
        id="parse_graph:text_supported",
        source_id="text_supported",
        objects=points + segments,
        relations=(relation,),
    )

    result = SeedRealizer().realize_proposal(graph)

    assert result.status is not SeedRealizationStatus.UNSUPPORTED
    assert result.metadata["input_status"] == "proposal"


def test_seed_realizer_accepts_point_names_from_original_problem_text():
    points = tuple(
        ParseObject(
            id=f"semantic:{name}",
            kind=ParseObjectKind.POINT,
            origin=ParseOrigin.IMAGE_TEXT,
            confidence=0.8,
            score_kind="model",
            visible_name=name,
            metadata={"binding_status": "unresolved", "declared_in_inventory": True},
        )
        for name in "ABC"
    )
    polygon = ParseObject(
        id="semantic:ABC",
        kind=ParseObjectKind.POLYGON,
        origin=ParseOrigin.IMAGE_TEXT,
        confidence=0.8,
        score_kind="model",
        visible_name="ABC",
        references=tuple(ParseReference(f"vertex_{index}", f"semantic:{name}") for index, name in enumerate("ABC")),
    )
    graph = ParseGraph(
        id="parse_graph:problem_text_supported",
        source_id="problem_text_supported",
        evidence=(ParseEvidence(
            id="evidence:problem_text",
            source_id="problem_text_supported",
            kind=ParseEvidenceKind.TEXT_SPAN,
            precision=EvidencePrecision.BOUND,
            span=(0, 31),
            raw_content="Triangle ABC has AB perpendicular AC.",
        ),),
        objects=points + (polygon,),
    )

    result = SeedRealizer().realize_proposal(graph)

    assert result.status is SeedRealizationStatus.REALIZED
    assert result.scene is not None


def test_seed_realizer_keeps_partial_topology_for_review():
    points = tuple(_native_object(f"point:{name}", ParseObjectKind.POINT, name) for name in "ABC")
    segments = (
        _native_object("segment:AB", ParseObjectKind.SEGMENT, "AB", (("start", "point:A"), ("end", "point:B"))),
        _native_object("segment:AC", ParseObjectKind.SEGMENT, "AC", (("start", "point:A"), ("end", "point:C"))),
    )
    relation = ParseRelation(
        id="relation:impossible_parallel",
        predicate="parallel",
        arguments=(ParseReference("left", "segment:AB"), ParseReference("right", "segment:AC")),
        origin=ParseOrigin.NATIVE,
        confidence=1.0,
        score_kind="seed_fixture",
    )
    graph = ParseGraph(
        id="parse_graph:partial_seed",
        source_id="partial_seed",
        objects=points + segments,
        relations=(relation,),
    )

    result = SeedRealizer().realize_proposal(graph)

    assert result.status is SeedRealizationStatus.PARTIAL
    assert result.scene is not None
    assert result.unresolved_constraints == ("relation:impossible_parallel",)
    assert result.metadata["input_status"] == "proposal"


def test_constraint_applier_commits_valid_relation_and_rolls_back_invalid_one():
    state = ConstructionBuilder().execute([
        {"op": "add_point", "id": "A", "coord": [0, 0]},
        {"op": "add_point", "id": "B", "coord": [2, 0]},
        {"op": "add_point", "id": "C", "coord": [0, 1]},
        {"op": "add_point", "id": "D", "coord": [2, 1]},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
        {"op": "add_segment", "id": "CD", "points": ["C", "D"]},
    ])

    report = ConstraintApplier().apply(
        state,
        [
            {"op": "add_relation", "id": "parallel", "type": "parallel", "objects": ["AB", "CD"]},
            {"op": "add_relation", "id": "bad", "type": "perpendicular", "objects": ["AB", "missing"]},
        ],
        allow_partial=True,
    )

    assert report.applied_ids == ("parallel",)
    assert len(report.skipped_operations) == 1
    assert [item["id"] for item in report.state.relations] == ["parallel"]


def test_seed_realizer_preserves_unsupported_arc_constraints_without_fake_redraw():
    points = (
        _native_object("point:O", ParseObjectKind.POINT, "O"),
        _native_object("point:A", ParseObjectKind.POINT, "A"),
        _native_object("point:B", ParseObjectKind.POINT, "B"),
    )
    objects = points + (
        _native_object("segment:AB", ParseObjectKind.SEGMENT, "AB", (("start", "point:A"), ("end", "point:B"))),
        _native_object("arc:upper", ParseObjectKind.ARC, "upper"),
    )
    relations = (
        ParseRelation(
            id="relation:center",
            predicate="center_of",
            arguments=(ParseReference("center", "point:O"), ParseReference("arc", "arc:upper")),
            origin=ParseOrigin.NATIVE,
            confidence=1.0,
            score_kind="seed_fixture",
        ),
        ParseRelation(
            id="relation:on_arc",
            predicate="point_on_arc",
            arguments=(ParseReference("point", "point:A"), ParseReference("arc", "arc:upper")),
            origin=ParseOrigin.NATIVE,
            confidence=1.0,
            score_kind="seed_fixture",
        ),
    )
    graph = ParseGraph(
        id="parse_graph:arc_seed",
        source_id="arc_seed",
        objects=objects,
        relations=relations,
    )

    result = SeedRealizer().realize_proposal(graph)

    assert result.status is SeedRealizationStatus.PARTIAL
    assert result.scene is not None
    assert not [item for item in result.scene.primitives if item.kind == "arc"]
    assert result.geometry_state is not None
    assert result.geometry_state.arcs == {}
    assert {item.candidate_id: item.status.value for item in result.constraint_outcomes} == {
        "relation:center": "unsupported",
        "relation:on_arc": "unsupported",
    }
    assert set(result.unresolved_constraints) == {"relation:center", "relation:on_arc"}


def test_seed_realizer_uses_lengths_and_perpendicular_relation_for_numeric_witness():
    points = tuple(_native_object(f"point:{name}", ParseObjectKind.POINT, name) for name in "ACE")
    segments = (
        _native_object("segment:AC", ParseObjectKind.SEGMENT, "AC", (("start", "point:A"), ("end", "point:C"))),
        _native_object("segment:CE", ParseObjectKind.SEGMENT, "CE", (("start", "point:C"), ("end", "point:E"))),
    )
    relation = ParseRelation(
        id="relation:AC_perpendicular_CE",
        predicate="perpendicular",
        arguments=(ParseReference("left", "segment:AC"), ParseReference("right", "segment:CE")),
        origin=ParseOrigin.NATIVE,
        confidence=1.0,
        score_kind="seed_fixture",
    )
    quantities = tuple(
        ParseQuantity(
            id=f"quantity:{segment}",
            kind="length",
            arguments=(ParseReference("object", f"segment:{segment}"),),
            expression=MathExpression("literal", value=value),
            origin=ParseOrigin.NATIVE,
            confidence=1.0,
            score_kind="seed_fixture",
        )
        for segment, value in (("AC", "20"), ("CE", "15"))
    )
    graph = ParseGraph(
        id="parse_graph:numeric_right_angle",
        source_id="numeric_right_angle",
        objects=points + segments,
        relations=(relation,),
        quantities=quantities,
    )

    result = SeedRealizer().realize_proposal(graph)

    assert result.status is SeedRealizationStatus.REALIZED
    ac = tuple(result.point_positions[name] for name in ("point:A", "point:C"))
    ce = tuple(result.point_positions[name] for name in ("point:C", "point:E"))
    assert math.dist(*ac) == pytest.approx(20.0, rel=1e-4)
    assert math.dist(*ce) == pytest.approx(15.0, rel=1e-4)
    ac_vector = (ac[1][0] - ac[0][0], ac[1][1] - ac[0][1])
    ce_vector = (ce[1][0] - ce[0][0], ce[1][1] - ce[0][1])
    assert sum(left * right for left, right in zip(ac_vector, ce_vector)) == pytest.approx(0.0, abs=1e-4)
    assert set(result.satisfied_constraints) == {
        "relation:AC_perpendicular_CE",
        "quantity:AC",
        "quantity:CE",
    }
    assert result.scene is not None
    assert len([item for item in result.scene.primitives if item.kind == "right_angle_marker"]) == 1
    assert {label.text for label in result.scene.labels} >= {"15", "20"}
