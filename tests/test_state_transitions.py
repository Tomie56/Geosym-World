import sympy as sp
import pytest

from geosym_world.construction.builder import ConstructionBuilder
from geosym_world.construction.operations import ConstructionOp
from geosym_world.construction.trace import record_construction_steps
from geosym_world.errors import DegenerateGeometryError, DuplicateObjectError, InvalidGeometryInput, NumericalInstabilityError, UnknownObjectError
from geosym_world.schema import (
    AdmissionStatus,
    Arc,
    Circle,
    FormalGeometryState,
    GeometryHypothesis,
    Line,
    Point,
    Polygon,
    Region,
    Ray,
    CompositeRegion,
    Segment,
    STATE_SCHEMA_VERSION,
    Target,
    canonical_json,
    record_checksum,
    stable_checksum,
    state_checksum,
    StateTransition,
)
from geosym_world.schema.facts import FactStore
from geosym_world.schema.derivations import DerivationTracer, validate_derivation_payload
from geosym_world.schema.records import GeometryRecord
from geosym_world.solver import FactClosureSolver, IncrementalFactClosure
from geosym_world.synthesis import select_template_families
from geosym_world.kernel.geometry2d.regions import boolean_polygon_boundaries_2d, region_geometry_area_2d, resolve_region_geometry_2d
from geosym_world.schema.provenance import EvidenceKind, Provenance
from geosym_world.schema.quantities import QuantityValue
from geosym_world.verifier import HypothesisAdmission, TraceVerifier


def _base_state() -> FormalGeometryState:
    state = FormalGeometryState()
    state.add_point(Point("A", (sp.Integer(0), sp.Integer(0))))
    state.add_point(Point("B", (sp.Integer(4), sp.Integer(0))))
    return state


def test_canonical_checksums_ignore_mapping_insertion_order():
    assert canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2})
    left = _base_state()
    right = FormalGeometryState()
    right.add_point(Point("B", (sp.Integer(4), sp.Integer(0))))
    right.add_point(Point("A", (sp.Integer(0), sp.Integer(0))))
    assert state_checksum(left) == state_checksum(right)


def test_record_checksum_is_stable_after_round_trip():
    record = GeometryRecord("record_1", {"type": "fixture"}, state=_base_state())
    restored = GeometryRecord.from_dict(record.to_dict())
    assert record_checksum(record) == record.checksum() == record_checksum(restored)


def test_state_schema_version_is_serialized_and_checked():
    state = _base_state()
    payload = state.to_dict()
    assert payload["schema_version"] == STATE_SCHEMA_VERSION
    restored = FormalGeometryState.from_dict(payload)
    assert restored.checksum() == state.checksum()
    with pytest.raises(ValueError, match="Unsupported geometry state schema version"):
        FormalGeometryState.from_dict({"schema_version": "9.9"})


def test_fact_ids_are_stable_when_facts_are_added_in_a_different_order():
    left = FactStore()
    left_first = left.add("length", "AB", "4", rule="given")
    left.add("area", "tri_ABC", "6", rule="target")

    right = FactStore()
    right.add("area", "tri_ABC", "6", rule="target")
    right_second = right.add("length", "AB", "4", rule="given")

    assert left_first.id == right_second.id


def test_fact_store_keeps_same_claims_with_distinct_dependencies_separate():
    facts = FactStore()
    first = facts.add("length", "AB", "4", rule="distance_formula", dependencies=("point:A", "point:B"))
    second = facts.add("length", "AB", "4", rule="distance_formula", dependencies=("point:C", "point:D"))
    assert first.id != second.id
    assert len(facts.find("length", "AB")) == 2


def test_derivation_tracer_expands_object_dependency_tokens():
    facts = FactStore()
    facts.add("exists", "A", source="constructed", rule="state_registry")
    length = facts.add("length", "AB", "4", rule="distance_formula", dependencies=("point:A",))
    facts.add("target", "AB", "4", rule="target_lookup", dependencies=(length.id,))
    derivation = DerivationTracer().trace_target(facts)
    assert {fact["type"] for fact in derivation["facts"]} == {"target", "length", "exists"}
    assert derivation["unresolved_dependencies"] == []
    assert any(edge["reference"] == "point:A" for edge in derivation["edges"])
    assert derivation["schema_version"] == "1.0"
    assert derivation["integrity"]["valid"]
    assert derivation["integrity"]["num_steps"] == len(derivation["facts"])
    assert derivation["steps"][-1]["fact_id"] == derivation["target_fact"]["id"]
    step_indices = {step["fact_id"]: step["index"] for step in derivation["steps"]}
    assert all(step_indices[edge["from"]] < step_indices[edge["to"]] for edge in derivation["edges"])
    assert derivation == DerivationTracer().trace_target(facts)
    assert validate_derivation_payload(derivation).valid

    tampered = {**derivation, "checksum": "0" * 64}
    validation = validate_derivation_payload(tampered)
    assert not validation.valid
    assert validation.errors[0]["code"] == "derivation.checksum"

    tampered_depth = {
        **derivation,
        "steps": [*derivation["steps"][:-1], {**derivation["steps"][-1], "depth": 99}],
    }
    tampered_depth["checksum"] = stable_checksum({key: value for key, value in tampered_depth.items() if key != "checksum"})
    assert any(error["code"] == "derivation.depth" for error in validate_derivation_payload(tampered_depth).errors)


def test_derivation_tracer_reports_missing_target_and_unresolved_inputs():
    without_target = DerivationTracer().trace_target(FactStore())
    assert not without_target["integrity"]["valid"]
    assert without_target["integrity"]["status"] == "missing_target"

    unresolved_facts = FactStore()
    unresolved_facts.add("target", "AB", "4", rule="target_lookup", dependencies=("missing_fact",))
    unresolved = DerivationTracer().trace_target(unresolved_facts)
    assert not unresolved["integrity"]["valid"]
    assert unresolved["integrity"]["status"] == "incomplete"
    assert unresolved["integrity"]["unresolved_count"] == 1
    assert unresolved["steps"][0]["unresolved_references"] == ["missing_fact"]
    assert not validate_derivation_payload(unresolved).valid

    ambiguous_facts = FactStore()
    ambiguous_facts.add("target", "AB", "4", rule="first_target")
    ambiguous_facts.add("target", "CD", "5", rule="second_target")
    ambiguous = DerivationTracer().trace_target(ambiguous_facts)
    assert not ambiguous["integrity"]["valid"]
    assert ambiguous["integrity"]["status"] == "ambiguous_target"
    assert ambiguous["integrity"]["target_fact_count"] == 2
    assert not validate_derivation_payload(ambiguous).valid


def test_admission_commits_verified_trace_and_reports_transition():
    base = _base_state()
    hypothesis = GeometryHypothesis(
        id="hyp_midpoint",
        kind="construction_trace",
        payload={"construction_trace": [{"op": "midpoint", "id": "M", "of": ["A", "B"]}]},
        confidence=0.9,
    )
    report = HypothesisAdmission().admit(hypothesis, initial_state=base)
    assert report.decision == AdmissionStatus.ADMITTED.value
    assert report.grounding.valid
    assert report.state is not None and "M" in report.state.points
    assert report.transition is not None and report.transition.changed
    assert report.transition.trace_length_after == report.transition.trace_length_before + 1
    assert report.transition.added_relation_ids
    assert report.hypothesis.status is AdmissionStatus.ADMITTED


def test_construction_steps_capture_replay_state_and_input_references():
    trace = [
        {"op": "add_point", "id": "A", "coord": [0, 0]},
        {"op": "add_point", "id": "B", "coord": [4, 0]},
        {"op": "midpoint", "id": "M", "of": ["A", "B"]},
    ]
    steps = record_construction_steps(trace)
    assert len(steps) == 3
    assert steps[-1]["operation"]["op"] == "midpoint"
    assert steps[-1]["input_references"] == [{"kind": "point", "id": "A"}, {"kind": "point", "id": "B"}]
    transition = steps[-1]["transition"]
    assert transition["trace_length_after"] == 3
    assert transition["changed"]
    assert "M" in transition["added_objects"]
    assert len(transition["before_checksum"]) == 64
    assert len(transition["after_checksum"]) == 64


def test_admission_rejects_unresolved_reference_without_mutating_state():
    base = _base_state()
    before = base.checksum()
    hypothesis = GeometryHypothesis(
        id="hyp_bad",
        kind="construction_trace",
        payload={"construction_trace": [{"op": "midpoint", "id": "M", "of": ["A", "Z"]}]},
        confidence=0.9,
    )
    report = HypothesisAdmission().admit(hypothesis, initial_state=base)
    assert report.decision == AdmissionStatus.REJECTED.value
    assert report.reason == "grounding_failed"
    assert report.state is None
    assert base.checksum() == before
    assert report.transition.before_checksum == report.transition.after_checksum


def test_admission_abstains_without_executable_trace():
    hypothesis = GeometryHypothesis(id="hyp_visual", kind="point_detection", payload={"id": "C"}, confidence=0.4)
    report = HypothesisAdmission().admit(hypothesis, confidence_threshold=0.5)
    assert report.decision == AdmissionStatus.ABSTAINED.value
    assert report.reason == "below_confidence_threshold"
    assert report.hypothesis.status is AdmissionStatus.ABSTAINED


def test_admission_abstains_on_empty_trace():
    hypothesis = GeometryHypothesis(id="hyp_empty", kind="construction_trace", payload={"trace": []}, confidence=0.9)
    report = HypothesisAdmission().admit(hypothesis)
    assert report.decision == AdmissionStatus.ABSTAINED.value
    assert report.reason == "missing_executable_trace"


def test_invalid_region_reference_is_rejected_at_state_boundary():
    state = FormalGeometryState()
    with pytest.raises(UnknownObjectError, match="Unknown geometry object"):
        state.add_region(Region("region_missing", ("segment_missing",)))


def test_zero_direction_is_rejected_at_state_boundary():
    state = FormalGeometryState()
    state.add_point(Point("A", (sp.Integer(0), sp.Integer(0))))
    with pytest.raises(DegenerateGeometryError, match="Zero line"):
        state.add_line(Line("line_zero", (sp.Integer(0), sp.Integer(0)), (sp.Integer(0), sp.Integer(0)), ("A",)))


def test_add_point3d_does_not_mutate_on_invalid_dimension():
    state = FormalGeometryState()
    with pytest.raises(DegenerateGeometryError, match="requires three coordinates"):
        ConstructionBuilder(state).execute([ConstructionOp(op="add_point3d", id="P", params={"coord": [0, 1]})])
    assert state.object_ids() == set()


def test_failed_construction_step_is_atomic_after_partial_outputs():
    state = _base_state()
    before = state.checksum()
    with pytest.raises(DuplicateObjectError):
        ConstructionBuilder(state).execute(
            [
                ConstructionOp(op="add_point", id="C", params={"coord": [0, 1]}),
                ConstructionOp(op="add_point", id="A", params={"coord": [1, 1]}),
            ]
        )
    assert state.checksum() != before
    assert "C" in state.points
    assert state.points["A"].coords == (sp.Integer(0), sp.Integer(0))


def test_failed_multi_output_step_restores_all_outputs():
    state = _base_state()
    state.add_point(Point("C", (sp.Integer(0), sp.Integer(2))))
    state.add_circle(Circle("circle_a", "A", sp.Integer(2)))
    state.add_line(Line("AB", (sp.Integer(0), sp.Integer(0)), (sp.Integer(1), sp.Integer(0)), ("A", "B")))
    before = state.checksum()
    with pytest.raises(DuplicateObjectError):
        ConstructionBuilder(state).execute(
            [
                ConstructionOp(op="intersection_line_circle", id=None, params={"line": "AB", "circle": "circle_a", "ids": ["P", "P"]}),
            ]
        )
    assert state.checksum() == before
    assert "P" not in state.points


def test_verifier_exposes_single_operation_gate_against_persistent_state():
    state = _base_state()
    report = TraceVerifier().verify_operation(
        {"op": "midpoint", "id": "M", "of": ["A", "B"]},
        initial_state=state,
        execute=False,
    )
    assert report.static_valid
    assert report.valid
    assert state.object_ids() == {"A", "B"}


def test_verifier_rejects_false_deterministic_relations():
    state = ConstructionBuilder().execute(
        [
            {"op": "add_point", "id": "A", "coord": [0, 0]},
            {"op": "add_point", "id": "B", "coord": [2, 0]},
            {"op": "add_point", "id": "C", "coord": [0, 1]},
            {"op": "add_point", "id": "D", "coord": [1, 2]},
            {"op": "add_point", "id": "M", "coord": [0, 0]},
            {"op": "add_line", "id": "AB", "points": ["A", "B"]},
            {"op": "add_line", "id": "CD", "points": ["C", "D"]},
        ]
    )
    state.add_relation({"type": "parallel", "objects": ["AB", "CD"]})
    state.add_relation({"type": "midpoint", "point": "M", "of": ["A", "B"]})
    state.add_relation({"type": "intersection", "point": "M", "objects": ["AB", "CD"]})

    report = TraceVerifier().verify_state(state)

    codes = {issue.code for issue in report.issues}
    assert "state.false_parallel" in codes
    assert "state.false_midpoint" in codes
    assert "state.false_intersection" in codes
    assert not report.valid


def test_verifier_accepts_constructed_intersection_relation_metadata():
    state = ConstructionBuilder().execute(
        [
            {"op": "add_point", "id": "A", "coord": [0, 0]},
            {"op": "add_point", "id": "B", "coord": [2, 2]},
            {"op": "add_point", "id": "C", "coord": [0, 2]},
            {"op": "add_point", "id": "D", "coord": [2, 0]},
            {"op": "add_line", "id": "AB", "points": ["A", "B"]},
            {"op": "add_line", "id": "CD", "points": ["C", "D"]},
            {"op": "intersection_line_line", "id": "X", "line1": "AB", "line2": "CD"},
        ]
    )
    assert TraceVerifier().verify_state(state).valid


def test_verifier_and_solver_check_explicit_equal_length_relation():
    state = _base_state()
    state.add_point(Point("C", (sp.Integer(0), sp.Integer(3))))
    state.add_point(Point("D", (sp.Integer(4), sp.Integer(3))))
    state.add_segment(Segment("AC", "A", "C"))
    state.add_segment(Segment("BD", "B", "D"))
    state.add_relation({"type": "same_length", "segments": ["AC", "BD"]})

    report = TraceVerifier().verify_state(state)
    facts = FactClosureSolver(profile="minimal").solve(state)

    assert report.valid
    assert facts.find("equal_length", "AC|BD")[0].source == "asserted"
    assert facts.find("equal_length_residual", "AC|BD")[0].value == "0"
    assert facts.find("equal_length", "AC|BD")[0].provenance[0].kind is EvidenceKind.VERIFIER
    assert facts.find("equal_length", "AC|BD")[0].provenance[0].dependencies == ("AC", "BD")


def test_verifier_rejects_false_explicit_equal_angle_relation():
    state = _base_state()
    state.add_point(Point("C", (sp.Integer(0), sp.Integer(3))))
    state.add_point(Point("D", (sp.Integer(4), sp.Integer(1))))
    state.add_relation({"type": "equal_angle", "angles": [["A", "B", "C"], ["A", "B", "D"]]})

    report = TraceVerifier().verify_state(state)

    assert not report.valid
    assert any(issue.code == "state.false_equal_angle" for issue in report.issues)


def test_explicit_similar_and_congruent_polygon_relations_are_verified():
    state = _base_state()
    for point in (
        Point("C", (sp.Integer(0), sp.Integer(3))),
        Point("D", (sp.Integer(8), sp.Integer(0))),
        Point("E", (sp.Integer(16), sp.Integer(0))),
        Point("F", (sp.Integer(8), sp.Integer(6))),
        Point("G", (sp.Integer(0), sp.Integer(3))),
    ):
        state.add_point(point)
    state.add_polygon(Polygon("tri_ABC", ("A", "B", "C")))
    state.add_polygon(Polygon("tri_DEF", ("D", "E", "F")))
    state.add_polygon(Polygon("tri_GAB", ("G", "A", "B")))
    state.add_relation({
        "type": "similar",
        "polygons": ["tri_ABC", "tri_DEF"],
        "mapping": {"A": "D", "B": "E", "C": "F"},
    })
    state.add_relation({
        "type": "congruent",
        "polygons": ["tri_ABC", "tri_GAB"],
        "mapping": {"A": "A", "B": "B", "C": "G"},
    })

    report = TraceVerifier().verify_state(state)
    facts = FactClosureSolver(profile="minimal").solve(state)

    assert report.valid
    assert facts.find("similar", "tri_ABC|tri_DEF")[0].source == "asserted"
    assert facts.find("similar_residual", "tri_ABC|tri_DEF")[0].value == "0"
    assert facts.find("congruent", "tri_ABC|tri_GAB")[0].source == "asserted"
    assert facts.find("congruent_residual", "tri_ABC|tri_GAB")[0].value == "0"


def test_verifier_rejects_false_similar_polygon_relation():
    state = _base_state()
    state.add_point(Point("C", (sp.Integer(0), sp.Integer(3))))
    state.add_point(Point("D", (sp.Integer(8), sp.Integer(0))))
    state.add_point(Point("E", (sp.Integer(16), sp.Integer(0))))
    state.add_point(Point("F", (sp.Integer(8), sp.Integer(5))))
    state.add_polygon(Polygon("tri_ABC", ("A", "B", "C")))
    state.add_polygon(Polygon("tri_DEF", ("D", "E", "F")))
    state.add_relation({
        "type": "similar",
        "polygons": ["tri_ABC", "tri_DEF"],
        "mapping": {"A": "D", "B": "E", "C": "F"},
    })

    report = TraceVerifier().verify_state(state)

    assert not report.valid
    assert any(issue.code == "state.false_similar" for issue in report.issues)


def test_incremental_closure_recomputes_only_new_object_dependencies():
    previous = _base_state()
    current = FormalGeometryState.from_dict(previous.to_dict())
    current.add_point(Point("C", (sp.Integer(4), sp.Integer(3))))
    current.add_segment(Segment("BC", "B", "C"))
    solver = FactClosureSolver(profile="minimal")
    incremental = IncrementalFactClosure(solver)
    initial_report = incremental.initialize(previous)
    report = incremental.update(previous, initial_report.facts, current, changed_object_ids=("C", "BC"))
    expected = solver.solve(current)
    assert report.mode == "local_recompute"
    assert "BC" in report.changed_object_ids
    assert report.facts.to_list() == expected.to_list()
    assert report.invalidated_fact_ids == ()


def test_incremental_closure_recomputes_locally_when_existing_object_changes():
    previous = _base_state()
    current = FormalGeometryState.from_dict(previous.to_dict())
    current.points["B"] = Point("B", (sp.Integer(5), sp.Integer(0)))
    solver = FactClosureSolver(profile="minimal")
    incremental = IncrementalFactClosure(solver)
    initial_report = incremental.initialize(previous)
    report = incremental.update(previous, initial_report.facts, current, changed_object_ids=("B",))
    assert report.mode == "local_recompute"
    assert report.current_checksum == current.checksum()
    assert sorted(report.facts.to_list(), key=lambda item: item["id"]) == sorted(
        solver.solve(current).to_list(), key=lambda item: item["id"]
    )


def test_incremental_relation_update_invalidates_only_relation_facts():
    previous = _base_state()
    previous.add_segment(Segment("AB", "A", "B"))
    current = FormalGeometryState.from_dict(previous.to_dict())
    current.add_point(Point("C", (sp.Integer(0), sp.Integer(3))))
    current.add_segment(Segment("AC", "A", "C"))
    current.add_relation({"type": "equal_length", "segments": ["AB", "AC"]})
    solver = FactClosureSolver(profile="minimal")
    incremental = IncrementalFactClosure(solver)
    initial = incremental.initialize(previous)
    report = incremental.update(previous, initial.facts, current)
    assert report.mode == "local_recompute"
    assert report.changed_relation_ids
    assert sorted(report.facts.to_list(), key=lambda item: item["id"]) == sorted(
        solver.solve(current).to_list(), key=lambda item: item["id"]
    )


def test_state_validate_rejects_invalid_relation_references_but_allows_fact_targets():
    state = _base_state()
    state.set_target(Target(type="length", object="missing"))
    state.validate()
    state.target = None
    state.relations.append({"type": "parallel", "objects": ["AB", "missing"]})
    with pytest.raises(UnknownObjectError, match="Unknown line-like relation reference"):
        state.validate()


def test_relation_reference_is_rejected_at_mutation_boundary():
    state = _base_state()
    with pytest.raises(UnknownObjectError, match="Unknown line-like relation reference"):
        state.add_relation({"type": "parallel", "objects": ["AB", "missing"]})


def test_tangent_relation_requires_exact_circle_contact_and_perpendicular_radius():
    state = FormalGeometryState()
    for point_id, coords in {"O": (0, 0), "T": (3, 0), "P": (3, 4), "Q": (4, 4)}.items():
        state.add_point(Point(point_id, tuple(sp.Integer(value) for value in coords)))
    state.add_circle(Circle("circle_O", "O", sp.Integer(3)))
    state.add_relation({"type": "tangent", "point": "T", "line": ["T", "P"], "circle": "circle_O"})
    assert TraceVerifier().verify_state(state).valid
    assert FactClosureSolver(profile="minimal").solve(state).find("tangent_to_circle", "T|circle_O")

    false_state = FormalGeometryState.from_dict(state.to_dict())
    false_state.relations = []
    false_state.add_relation({"type": "tangent", "point": "T", "line": ["T", "Q"], "circle": "circle_O"})
    report = TraceVerifier().verify_state(false_state)
    assert not report.valid
    assert any(issue.code == "state.false_tangent" for issue in report.issues)
    assert FactClosureSolver(profile="minimal").solve(false_state).find("tangent_to_circle", "T|circle_O") == []


@pytest.mark.parametrize(
    ("relation", "issue_code"),
    [
        ({"type": "point_on_line", "point": "C", "line": "AB"}, "state.false_point_on_line"),
        ({"type": "point_on_ray", "point": "C", "line": "ray_AB"}, "state.false_point_on_ray"),
        ({"type": "point_on_circle", "point": "C", "circle": "circle_A"}, "state.false_point_on_circle"),
        ({"type": "point_on_arc", "point": "C", "arc": "arc_BD"}, "state.false_point_on_arc"),
        ({"type": "collinear", "points": ["A", "B", "C"]}, "state.false_collinear"),
        ({"type": "foot_of_perpendicular", "point": "C", "from": "D", "line": "AB"}, "state.false_foot_of_perpendicular"),
        ({"type": "point_on_segment_ratio", "point": "C", "segment": ["A", "B"], "ratio": "1/2"}, "state.false_point_on_segment_ratio"),
    ],
)
def test_coordinate_backed_incidence_relations_reject_false_claims(relation, issue_code):
    state = FormalGeometryState()
    for point_id, coords in {"A": (0, 0), "B": (4, 0), "C": (1, 1), "D": (0, 4)}.items():
        state.add_point(Point(point_id, tuple(sp.Integer(value) for value in coords)))
    state.add_segment(Segment("AB", "A", "B"))
    state.add_ray(Ray("ray_AB", "A", (sp.Integer(1), sp.Integer(0)), through="B"))
    state.add_circle(Circle("circle_A", "A", sp.Integer(4)))
    state.add_arc(Arc("arc_BD", "circle_A", "B", "D", central_angle=sp.pi / 2))
    state.add_relation(relation)
    report = TraceVerifier().verify_state(state)
    assert not report.valid
    assert any(issue.code == issue_code for issue in report.issues)


def test_constructed_foot_and_segment_ratio_relations_are_admitted():
    state = ConstructionBuilder().execute([
        {"op": "add_point", "id": "A", "coord": [0, 0]},
        {"op": "add_point", "id": "B", "coord": [4, 0]},
        {"op": "add_point", "id": "C", "coord": [1, 3]},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
        {"op": "foot_of_perpendicular", "id": "H", "point": "C", "line": "AB"},
        {"op": "point_on_segment_ratio", "id": "M", "segment": "AB", "ratio": "1/2"},
    ])
    assert TraceVerifier().verify_state(state).valid


def test_state_transition_rejects_invalid_status_or_checksum():
    with pytest.raises(ValueError, match="Unknown state transition status"):
        StateTransition("event_1", "invalid", "0" * 64, "0" * 64, False)
    with pytest.raises(ValueError, match="SHA-256 checksum"):
        StateTransition("event_1", "committed", "bad", "0" * 64, False)


def test_incremental_closure_noop_preserves_fact_store_identity_values():
    state = _base_state()
    solver = FactClosureSolver(profile="minimal")
    incremental = IncrementalFactClosure(solver)
    initial_report = incremental.initialize(state)
    report = incremental.update(state, initial_report.facts, FormalGeometryState.from_dict(state.to_dict()))
    assert report.mode == "no_op"
    assert report.facts.to_list() == initial_report.facts.to_list()


def test_incremental_closure_adds_partition_with_parent_dependency():
    previous = FormalGeometryState()
    for point in (
        Point("A", (sp.Integer(0), sp.Integer(0))),
        Point("B", (sp.Integer(5), sp.Integer(0))),
        Point("C", (sp.Integer(2), sp.Integer(3))),
        Point("H", (sp.Integer(2), sp.Integer(0))),
    ):
        previous.add_point(point)
    for polygon in (
        Polygon("parent_poly", ("A", "B", "C")),
        Polygon("left_poly", ("A", "H", "C")),
        Polygon("right_poly", ("H", "B", "C")),
    ):
        previous.add_polygon(polygon)
    previous.add_region(Region("parent", ("parent_poly",)))
    previous.add_region(Region("left", ("left_poly",)))
    previous.add_region(Region("right", ("right_poly",)))
    current = FormalGeometryState.from_dict(previous.to_dict())
    current.add_composite_region(CompositeRegion("partition", "partition", ("left", "right"), {"parent_region": "parent"}))

    solver = FactClosureSolver(profile="minimal")
    incremental = IncrementalFactClosure(solver)
    initial = incremental.initialize(previous)
    report = incremental.update(previous, initial.facts, current)

    assert report.mode == "local_recompute"
    assert sorted(report.facts.to_list(), key=lambda item: item["id"]) == sorted(
        solver.solve(current).to_list(), key=lambda item: item["id"]
    )
    assert report.facts.find("area", "partition")[0].value == "15/2"


def test_incremental_closure_adds_perimeter_only_lens():
    spec = next(iter(select_template_families(["circle_lens_perimeter"])[0].variants()))
    region_index = next(index for index, step in enumerate(spec.trace) if step.get("id") == "lens_12")
    previous = ConstructionBuilder().execute(spec.trace[:region_index])
    previous.set_target(Target("perimeter", "lens_12", {"preferred_rule": "lens_two_radius_perimeter"}))
    current = ConstructionBuilder().execute(spec.trace)
    solver = FactClosureSolver(profile="minimal")
    incremental = IncrementalFactClosure(solver)
    initial = incremental.initialize(previous)
    changed_ids = tuple(sorted(set(current.object_ids()) - set(previous.object_ids())))
    report = incremental.update(previous, initial.facts, current, changed_object_ids=changed_ids)

    assert report.mode == "local_recompute"
    assert report.facts.find("perimeter", "lens_12")
    assert report.facts.find("target", "lens_12")
    assert sorted(report.facts.to_list(), key=lambda item: item["id"]) == sorted(
        solver.solve(current).to_list(), key=lambda item: item["id"]
    )


def test_composite_region_round_trip_and_area_closure():
    state = _base_state()
    state.add_point(Point("C", (sp.Integer(0), sp.Integer(3))))
    state.add_point(Point("D", (sp.Integer(4), sp.Integer(3))))
    state.add_polygon(Polygon("left", ("A", "B", "D", "C")))
    state.add_region(Region("outer", ("left",)))
    state.add_region(Region("hole", ("A", "B", "D")))
    state.add_composite_region(CompositeRegion("shaded", "difference", ("outer", "hole")))
    restored = FormalGeometryState.from_dict(state.to_dict())
    assert restored.composite_regions["shaded"].components == ("outer", "hole")
    facts = FactClosureSolver(profile="minimal").solve(restored)
    assert facts.find("area", "shaded")[0].value == "6"


def test_partition_region_requires_verified_parent_coverage():
    state = FormalGeometryState()
    for point in (
        Point("A", (sp.Integer(0), sp.Integer(0))),
        Point("B", (sp.Integer(5), sp.Integer(0))),
        Point("C", (sp.Integer(2), sp.Integer(3))),
        Point("H", (sp.Integer(2), sp.Integer(0))),
    ):
        state.add_point(point)
    state.add_polygon(Polygon("parent_poly", ("A", "B", "C")))
    state.add_polygon(Polygon("left_poly", ("A", "H", "C")))
    state.add_polygon(Polygon("right_poly", ("H", "B", "C")))
    state.add_region(Region("parent", ("parent_poly",)))
    state.add_region(Region("left", ("left_poly",)))
    state.add_region(Region("right", ("right_poly",)))
    state.add_composite_region(CompositeRegion("partition", "partition", ("left", "right"), {"parent_region": "parent"}))
    state.set_target(Target("area", "partition", {"preferred_rule": "region_partition_area_sum"}))

    facts = FactClosureSolver(profile="minimal").solve(state)
    area = facts.find("area", "partition")[0]
    residual = facts.find("area_partition_residual", "partition|parent")[0]
    derivation = DerivationTracer().trace_target(facts)

    assert resolve_region_geometry_2d(state, "partition").status == "polygon"
    assert area.value == "15/2"
    assert residual.value == "0"
    assert area.metadata["admission_fact_id"] == residual.id
    assert residual.id not in area.dependencies
    assert "area_partition_residual" not in {fact["type"] for fact in derivation["facts"]}
    assert derivation["integrity"]["valid"]


def test_partition_region_rejects_incomplete_component_coverage():
    state = FormalGeometryState()
    for point in (
        Point("A", (sp.Integer(0), sp.Integer(0))),
        Point("B", (sp.Integer(5), sp.Integer(0))),
        Point("C", (sp.Integer(2), sp.Integer(3))),
        Point("H", (sp.Integer(2), sp.Integer(0))),
    ):
        state.add_point(point)
    state.add_polygon(Polygon("parent_poly", ("A", "B", "C")))
    state.add_polygon(Polygon("left_poly", ("A", "H", "C")))
    state.add_region(Region("parent", ("parent_poly",)))
    state.add_region(Region("left", ("left_poly",)))
    state.add_region(Region("copy", ("left_poly",)))
    state.add_composite_region(CompositeRegion("invalid_partition", "partition", ("left", "copy"), {"parent_region": "parent"}))

    resolved = resolve_region_geometry_2d(state, "invalid_partition")
    facts = FactClosureSolver(profile="minimal").solve(state)

    assert not resolved.resolved
    assert resolved.reason == "partition_components_overlap"
    assert facts.find("area", "invalid_partition") == []
    assert facts.find("area_unresolved", "invalid_partition")[0].metadata["reason"] == "partition_components_overlap"
    report = TraceVerifier().verify_state(state)
    assert not report.valid
    assert any(issue.code == "state.invalid_region_partition" for issue in report.issues)


def test_partition_region_requires_parent_metadata():
    state = _base_state()
    state.add_point(Point("C", (sp.Integer(0), sp.Integer(3))))
    state.add_region(Region("left", ("A", "B", "C")))
    state.add_region(Region("right", ("A", "C", "B")))
    with pytest.raises(InvalidGeometryInput, match="parent_region"):
        state.add_composite_region(CompositeRegion("partition", "partition", ("left", "right")))


def test_composite_difference_area_formula_requires_ordered_components():
    state = _base_state()
    state.add_point(Point("C", (sp.Integer(4), sp.Integer(3))))
    state.add_point(Point("D", (sp.Integer(0), sp.Integer(3))))
    state.add_region(Region("outer", ("A", "B", "C", "D")))
    state.add_region(Region("inner", ("A", "B", "C")))
    with pytest.raises(InvalidGeometryInput, match="ordered components"):
        state.add_composite_region(
            CompositeRegion(
                "frame",
                "difference",
                ("outer", "inner"),
                {"area_formula": {"rule": "region_difference", "minuend": "inner", "subtrahend": "outer"}},
            )
        )


def test_area_ratio_target_requires_distinct_existing_regions():
    state = _base_state()
    state.add_point(Point("C", (sp.Integer(0), sp.Integer(3))))
    state.add_region(Region("triangle", ("A", "B", "C")))
    with pytest.raises(InvalidGeometryInput, match="distinct"):
        state.set_target(Target("ratio", "same_ratio", {"ratio_kind": "area", "numerator": "triangle", "denominator": "triangle"}))
    with pytest.raises(UnknownObjectError, match="missing"):
        state.set_target(Target("ratio", "missing_ratio", {"ratio_kind": "area", "numerator": "triangle", "denominator": "missing"}))


def test_target_type_is_canonicalized_at_state_boundary():
    state = _base_state()
    state.add_point(Point("C", (sp.Integer(0), sp.Integer(3))))
    state.add_region(Region("left", ("A", "B", "C")))
    state.add_region(Region("right", ("A", "C", "B")))
    state.set_target(Target("Ratio", "area_ratio", {"ratio_kind": "area", "numerator": "left", "denominator": "right"}))
    checksum = state.checksum()

    assert state.target.type == "ratio"
    restored = FormalGeometryState.from_dict(state.to_dict())
    assert restored.target.type == "ratio"
    assert restored.checksum() == checksum


def test_verifier_rejects_semantic_difference_without_contained_hole():
    state = FormalGeometryState()
    for point_id, coords in {
        "A": (0, 0), "B": (4, 0), "C": (4, 4), "D": (0, 4),
        "E": (3, 1), "F": (5, 1), "G": (5, 3), "H": (3, 3),
    }.items():
        state.add_point(Point(point_id, tuple(sp.Integer(value) for value in coords)))
    state.add_region(Region("outer", ("A", "B", "C", "D")))
    state.add_region(Region("overlap", ("E", "F", "G", "H")))
    state.add_composite_region(
        CompositeRegion(
            "invalid_frame",
            "difference",
            ("outer", "overlap"),
            {"area_formula": {"rule": "region_difference", "minuend": "outer", "subtrahend": "overlap"}},
        )
    )

    report = TraceVerifier().verify_state(state)

    assert not report.valid
    assert any(issue.code == "state.invalid_region_difference" for issue in report.issues)


def test_boundary_notch_difference_materializes_and_derives_exact_area():
    state = FormalGeometryState()
    for point_id, coords in {
        "A": (0, 0), "B": (8, 0), "C": (8, 7), "D": (0, 7),
        "E": (5, 4), "F": (8, 4), "G": (8, 7), "H": (5, 7),
    }.items():
        state.add_point(Point(point_id, tuple(sp.Integer(value) for value in coords)))
    state.add_region(Region("outer", ("A", "B", "C", "D")))
    state.add_region(Region("notch", ("E", "F", "G", "H")))
    state.add_composite_region(
        CompositeRegion(
            "l_shape",
            "difference",
            ("outer", "notch"),
            {
                "area_formula": {
                    "rule": "region_difference",
                    "mode": "boundary_notch",
                    "minuend": "outer",
                    "subtrahend": "notch",
                }
            },
        )
    )
    state.set_target(Target("area", "l_shape", {"preferred_rule": "region_difference_area"}))

    geometry = resolve_region_geometry_2d(state, "l_shape")
    facts = FactClosureSolver(profile="minimal").solve(state)
    report = TraceVerifier().verify_state(state)

    assert geometry.status == "polygon"
    assert len(geometry.boundary) == 6
    assert region_geometry_area_2d(geometry) == 47
    semantic_area = next(fact for fact in facts.find("area", "l_shape") if fact.rule == "region_difference_area")
    assert semantic_area.value == "47"
    assert facts.find("area_difference_residual", "l_shape")[0].value == "0"
    assert report.valid


def test_boundary_notch_difference_rejects_non_corner_cutout():
    state = FormalGeometryState()
    for point_id, coords in {
        "A": (0, 0), "B": (8, 0), "C": (8, 7), "D": (0, 7),
        "E": (2, 2), "F": (4, 2), "G": (4, 4), "H": (2, 4),
    }.items():
        state.add_point(Point(point_id, tuple(sp.Integer(value) for value in coords)))
    state.add_region(Region("outer", ("A", "B", "C", "D")))
    state.add_region(Region("notch", ("E", "F", "G", "H")))
    state.add_composite_region(
        CompositeRegion(
            "invalid_l_shape",
            "difference",
            ("outer", "notch"),
            {
                "area_formula": {
                    "rule": "region_difference",
                    "mode": "boundary_notch",
                    "minuend": "outer",
                    "subtrahend": "notch",
                }
            },
        )
    )

    geometry = resolve_region_geometry_2d(state, "invalid_l_shape")
    report = TraceVerifier().verify_state(state)

    assert geometry.status == "unsupported"
    assert geometry.reason == "boundary_notch_must_remove_one_outer_corner"
    assert not report.valid
    assert any(issue.code == "state.invalid_region_difference" for issue in report.issues)


def test_region_boundary_fact_is_in_target_derivation_subgraph():
    state = _base_state()
    state.add_point(Point("C", (sp.Integer(0), sp.Integer(3))))
    state.add_point(Point("D", (sp.Integer(4), sp.Integer(3))))
    state.add_region(Region("quad", ("A", "B", "D", "C")))
    state.set_target(Target(type="area", object="quad"))
    facts = FactClosureSolver(profile="minimal").solve(state)
    derivation = DerivationTracer().trace_target(facts)
    fact_types = {fact["type"] for fact in derivation["facts"]}
    assert {"target", "area", "region_boundary"}.issubset(fact_types)
    assert derivation["unresolved_dependencies"] == []
    assert derivation["integrity"]["valid"]
    assert derivation["integrity"]["max_depth"] >= 2


def test_convex_composite_region_boolean_area_closure():
    state = _base_state()
    for point in (
        Point("C", (sp.Integer(0), sp.Integer(3))),
        Point("D", (sp.Integer(4), sp.Integer(3))),
        Point("E", (sp.Integer(2), sp.Integer(1))),
        Point("F", (sp.Integer(6), sp.Integer(1))),
        Point("G", (sp.Integer(6), sp.Integer(4))),
        Point("H", (sp.Integer(2), sp.Integer(4))),
    ):
        state.add_point(point)
    state.add_polygon(Polygon("left_poly", ("A", "B", "D", "C")))
    state.add_polygon(Polygon("right_poly", ("E", "F", "G", "H")))
    state.add_region(Region("left_region", ("left_poly",)))
    state.add_region(Region("right_region", ("right_poly",)))
    state.add_composite_region(CompositeRegion("intersection_region", "intersection", ("left_region", "right_region")))
    state.add_composite_region(CompositeRegion("union_region", "union", ("left_region", "right_region")))
    state.add_composite_region(CompositeRegion("difference_region", "difference", ("left_region", "right_region")))

    facts = FactClosureSolver(profile="minimal").solve(state)

    assert facts.find("area", "intersection_region")[0].value == "4"
    assert facts.find("area", "union_region")[0].value == "20"
    assert facts.find("area", "difference_region")[0].value == "8"
    resolved = resolve_region_geometry_2d(state, "intersection_region")
    assert resolved.status == "polygon"
    assert resolved.source_ids[:3] == ("intersection_region", "left_region", "left_poly")
    assert "right_poly" in resolved.source_ids
    assert facts.find("region_boundary", "intersection_region")[0].id in facts.find("area", "intersection_region")[0].dependencies


def test_polygon_region_boundary_resolves_segment_loop_and_holes():
    state = _base_state()
    for point in (
        Point("C", (sp.Integer(0), sp.Integer(3))),
        Point("D", (sp.Integer(4), sp.Integer(3))),
        Point("E", (sp.Integer(1), sp.Integer(1))),
        Point("F", (sp.Integer(2), sp.Integer(1))),
        Point("G", (sp.Integer(2), sp.Integer(2))),
        Point("H", (sp.Integer(1), sp.Integer(2))),
    ):
        state.add_point(point)
    for segment_id, endpoints in (("AB", ("A", "B")), ("BD", ("B", "D")), ("DC", ("D", "C")), ("CA", ("C", "A"))):
        state.add_segment(Segment(segment_id, *endpoints))
    state.add_region(Region("hole", ("E", "F", "G", "H")))
    state.add_region(Region("outer", ("DC", "AB", "CA", "BD"), holes=("hole",)))
    resolved = resolve_region_geometry_2d(state, "outer")
    assert resolved.status == "polygon_with_holes"
    facts = FactClosureSolver(profile="minimal").solve(state)
    assert facts.find("area", "outer")[0].value == "11"
    assert facts.find("region_boundary", "outer")[0].metadata["status"] == "polygon_with_holes"
    assert facts.find("region_boundary", "outer")[0].id in facts.find("area", "outer")[0].dependencies


def test_convex_boolean_boundaries_cover_containment_and_disjoint_union():
    state = FormalGeometryState()
    for point_id, coords in {
        "A": (0, 0), "B": (4, 0), "C": (4, 4), "D": (0, 4),
        "E": (1, 1), "F": (3, 1), "G": (3, 3), "H": (1, 3),
        "I": (6, 0), "J": (8, 0), "K": (8, 2), "L": (6, 2),
    }.items():
        state.add_point(Point(point_id, tuple(sp.Integer(value) for value in coords)))
    state.add_region(Region("outer", ("A", "B", "C", "D")))
    state.add_region(Region("inner", ("E", "F", "G", "H")))
    state.add_region(Region("far", ("I", "J", "K", "L")))
    for operation, components in (
        ("union", ("outer", "inner")),
        ("intersection", ("outer", "inner")),
        ("difference", ("outer", "inner")),
        ("union", ("outer", "far")),
    ):
        result_id = f"result_{operation}_{components[1]}"
        state.add_composite_region(CompositeRegion(result_id, operation, components))
        geometry = resolve_region_geometry_2d(state, result_id)
        assert geometry.resolved
        if result_id == "result_difference_inner":
            assert geometry.status == "polygon_with_holes"
            assert region_geometry_area_2d(geometry) == 12
        elif result_id == "result_union_far":
            assert geometry.status == "multi_polygon"
            assert region_geometry_area_2d(geometry) == 20


def test_non_convex_composite_boolean_materializes_exact_boundaries():
    state = FormalGeometryState()
    coordinates = {
        "A": (0, 0), "B": (4, 0), "C": (4, 1), "D": (1, 1), "E": (1, 4), "F": (0, 4),
        "G": (sp.Rational(1, 2), sp.Rational(1, 2)), "H": (2, sp.Rational(1, 2)),
        "I": (2, 2), "J": (sp.Rational(1, 2), 2),
    }
    for point_id, coords in coordinates.items():
        state.add_point(Point(point_id, tuple(sp.sympify(value) for value in coords)))
    state.add_region(Region("concave", ("A", "B", "C", "D", "E", "F")))
    state.add_region(Region("clip", ("G", "H", "I", "J")))
    expected = {"intersection": sp.Rational(5, 4), "union": sp.Integer(8), "difference": sp.Rational(23, 4)}
    for operation, area in expected.items():
        result_id = f"concave_{operation}"
        state.add_composite_region(CompositeRegion(result_id, operation, ("concave", "clip")))
        geometry = resolve_region_geometry_2d(state, result_id)
        assert geometry.resolved
        assert region_geometry_area_2d(geometry) == area

    facts = FactClosureSolver(profile="minimal").solve(state)
    for operation, area in expected.items():
        result_id = f"concave_{operation}"
        assert sp.sympify(facts.find("area", result_id)[0].value) == area
        assert facts.find("region_boundary", result_id)


def test_non_convex_boolean_identical_boundary_is_idempotent():
    points = tuple((sp.Integer(x), sp.Integer(y)) for x, y in ((0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4)))
    union = boolean_polygon_boundaries_2d(points, points, "union")
    intersection = boolean_polygon_boundaries_2d(points, points, "intersection")
    difference = boolean_polygon_boundaries_2d(points, points, "difference")
    assert union == intersection == (points,)
    assert difference == ()


def test_verifier_rejects_self_intersecting_region_boundary():
    state = FormalGeometryState()
    for point_id, coords in {"A": (0, 0), "B": (2, 2), "C": (0, 2), "D": (2, 0)}.items():
        state.add_point(Point(point_id, tuple(sp.Integer(value) for value in coords)))
    state.add_region(Region("bowtie", ("A", "B", "C", "D")))
    report = TraceVerifier().verify_state(state)
    assert not report.valid
    assert any(issue.code == "state.invalid_region_boundary" for issue in report.issues)


def test_verifier_rejects_hole_outside_region_boundary():
    state = FormalGeometryState()
    for point_id, coords in {
        "A": (0, 0), "B": (4, 0), "C": (4, 4), "D": (0, 4),
        "E": (5, 5), "F": (6, 5), "G": (6, 6), "H": (5, 6),
    }.items():
        state.add_point(Point(point_id, tuple(sp.Integer(value) for value in coords)))
    state.add_region(Region("hole", ("E", "F", "G", "H")))
    state.add_region(Region("outer", ("A", "B", "C", "D"), holes=("hole",)))
    report = TraceVerifier().verify_state(state)
    assert not report.valid
    assert any(issue.metadata.get("reason") == "hole_outside_boundary" for issue in report.issues)


def test_multi_hole_region_derives_semantic_area_from_distinct_components():
    state = FormalGeometryState()
    for point_id, coords in {
        "A": (0, 0), "B": (10, 0), "C": (10, 6), "D": (0, 6),
        "E": (1, 1), "F": (3, 1), "G": (3, 3), "H": (1, 3),
        "I": (6, 2), "J": (8, 2), "K": (8, 4), "L": (6, 4),
    }.items():
        state.add_point(Point(point_id, tuple(sp.Integer(value) for value in coords)))
    state.add_region(Region("outer", ("A", "B", "C", "D")))
    state.add_region(Region("left_hole", ("E", "F", "G", "H")))
    state.add_region(Region("right_hole", ("I", "J", "K", "L")))
    state.add_region(
        Region(
            "plate",
            ("A", "B", "C", "D"),
            holes=("left_hole", "right_hole"),
            metadata={
                "area_formula": {
                    "rule": "region_hole_difference",
                    "outer_region": "outer",
                    "holes": ["left_hole", "right_hole"],
                }
            },
        )
    )
    state.set_target(Target("area", "plate", {"preferred_rule": "region_hole_difference_area"}))

    geometry = resolve_region_geometry_2d(state, "plate")
    facts = FactClosureSolver(profile="minimal").solve(state)
    report = TraceVerifier().verify_state(state)
    semantic_area = next(fact for fact in facts.find("area", "plate") if fact.rule == "region_hole_difference_area")
    residual = facts.find("area_hole_difference_residual", "plate")[0]

    assert geometry.status == "polygon_with_holes"
    assert len(geometry.holes) == 2
    assert region_geometry_area_2d(geometry) == 52
    assert semantic_area.value == "52"
    assert len(semantic_area.dependencies) == 3
    assert residual.value == "0"
    assert report.valid


def test_multi_hole_region_rejects_touching_holes():
    state = FormalGeometryState()
    for point_id, coords in {
        "A": (0, 0), "B": (10, 0), "C": (10, 6), "D": (0, 6),
        "E": (1, 1), "F": (3, 1), "G": (3, 3), "H": (1, 3),
        "I": (3, 1), "J": (5, 1), "K": (5, 3), "L": (3, 3),
    }.items():
        state.add_point(Point(point_id, tuple(sp.Integer(value) for value in coords)))
    state.add_region(Region("left_hole", ("E", "F", "G", "H")))
    state.add_region(Region("right_hole", ("I", "J", "K", "L")))
    state.add_region(Region("plate", ("A", "B", "C", "D"), holes=("left_hole", "right_hole")))

    geometry = resolve_region_geometry_2d(state, "plate")
    report = TraceVerifier().verify_state(state)

    assert geometry.status == "unsupported"
    assert geometry.reason == "holes_touch_or_overlap"
    assert not report.valid
    assert any(issue.metadata.get("reason") == "holes_touch_or_overlap" for issue in report.issues)


def test_region_hole_difference_requires_formula_hole_order_to_match():
    state = FormalGeometryState()
    for point_id, coords in {
        "A": (0, 0), "B": (8, 0), "C": (8, 6), "D": (0, 6),
        "E": (1, 1), "F": (2, 1), "G": (2, 2), "H": (1, 2),
        "I": (5, 3), "J": (6, 3), "K": (6, 4), "L": (5, 4),
    }.items():
        state.add_point(Point(point_id, tuple(sp.Integer(value) for value in coords)))
    state.add_region(Region("outer", ("A", "B", "C", "D")))
    state.add_region(Region("left_hole", ("E", "F", "G", "H")))
    state.add_region(Region("right_hole", ("I", "J", "K", "L")))

    with pytest.raises(InvalidGeometryInput, match="holes must match"):
        state.add_region(
            Region(
                "plate",
                ("A", "B", "C", "D"),
                holes=("left_hole", "right_hole"),
                metadata={
                    "area_formula": {
                        "rule": "region_hole_difference",
                        "outer_region": "outer",
                        "holes": ["right_hole", "left_hole"],
                    }
                },
            )
        )


def test_multi_hole_region_round_trip_does_not_depend_on_mapping_order():
    state = FormalGeometryState()
    for point_id, coords in {
        "A": (0, 0), "B": (8, 0), "C": (8, 6), "D": (0, 6),
        "E": (1, 1), "F": (2, 1), "G": (2, 2), "H": (1, 2),
        "I": (5, 3), "J": (6, 3), "K": (6, 4), "L": (5, 4),
    }.items():
        state.add_point(Point(point_id, tuple(sp.Integer(value) for value in coords)))
    state.add_region(Region("outer", ("A", "B", "C", "D")))
    state.add_region(Region("left_hole", ("E", "F", "G", "H")))
    state.add_region(Region("right_hole", ("I", "J", "K", "L")))
    state.add_region(
        Region(
            "plate",
            ("A", "B", "C", "D"),
            holes=("left_hole", "right_hole"),
            metadata={
                "area_formula": {
                    "rule": "region_hole_difference",
                    "outer_region": "outer",
                    "holes": ["left_hole", "right_hole"],
                }
            },
        )
    )
    payload = state.to_dict()
    payload["regions"] = {
        region_id: payload["regions"][region_id]
        for region_id in ("plate", "right_hole", "outer", "left_hole")
    }

    restored = FormalGeometryState.from_dict(payload)

    assert restored.regions["plate"].holes == ("left_hole", "right_hole")
    assert restored.checksum() == state.checksum()


def test_rectangle_and_multi_hole_perimeters_derive_from_semantic_components():
    state = ConstructionBuilder().execute(
        next(iter(select_template_families(["double_rectangular_hole_perimeter"])[0].variants())).trace
    )
    facts = FactClosureSolver(profile="minimal").solve(state)
    report = TraceVerifier().verify_state(state)
    semantic = next(fact for fact in facts.find("perimeter", "perforated_plate") if fact.rule == "region_hole_boundary_sum")
    residual = facts.find("perimeter_hole_sum_residual", "perforated_plate")[0]
    rectangle_facts = [fact for fact in facts if fact.rule == "rectangle_side_perimeter"]

    assert semantic.value == "54"
    assert len(semantic.dependencies) == 3
    assert len(rectangle_facts) == 3
    assert residual.value == "0"
    assert report.valid


def test_rectangle_perimeter_formula_rejects_diagonal_reference():
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": ["5", "0"]},
        {"op": "add_point", "id": "C", "coord": ["5", "3"]},
        {"op": "add_point", "id": "D", "coord": ["0", "3"]},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
        {"op": "add_segment", "id": "AC", "points": ["A", "C"]},
        {
            "op": "add_polygon",
            "id": "rect",
            "vertices": ["A", "B", "C", "D"],
            "metadata": {"perimeter_formula": {"rule": "rectangle_side_perimeter", "width": "AB", "height": "AC"}},
        },
        {"op": "set_target", "type": "perimeter", "object": "rect", "metadata": {"preferred_rule": "rectangle_side_perimeter"}},
    ]
    state = ConstructionBuilder().execute(trace)
    facts = FactClosureSolver(profile="minimal").solve(state)
    report = TraceVerifier().verify_state(state)

    assert not any(fact.rule == "rectangle_side_perimeter" for fact in facts.find("perimeter", "rect"))
    assert not report.valid
    assert any(issue.code == "state.invalid_rectangle_perimeter_formula" for issue in report.issues)


def test_verifier_rejects_mutated_lens_perimeter_formula():
    state = ConstructionBuilder().execute(
        next(iter(select_template_families(["circle_lens_perimeter"])[0].variants())).trace
    )
    state.regions["lens_12"].metadata["perimeter_formula"]["radius_segments"] = ["O1O2", "O2R2"]
    report = TraceVerifier().verify_state(state)
    assert not report.valid
    assert any(issue.code == "state.invalid_lens_perimeter_formula" for issue in report.issues)


def test_strict_quantity_provenance_and_target_inputs():
    with pytest.raises(NumericalInstabilityError):
        QuantityValue(numeric=float("inf"))
    with pytest.raises(ValueError, match="Target type"):
        Target(type="", object="AB")
    provenance = Provenance(source_id="image_1", kind=EvidenceKind.PARSER, event_id="event_1", dependencies=("point:A",))
    assert provenance.to_dict()["kind"] == "parser"
