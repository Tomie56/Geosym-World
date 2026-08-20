import json

import pytest
import sympy as sp

from geosym_world.schema import (
    AdmissionStatus,
    Arc,
    Circle,
    EvidenceKind,
    FormalGeometryState,
    GeometryHypothesis,
    GeometryRecord,
    Line,
    Plane,
    Point,
    Polygon,
    Polyhedron,
    Provenance,
    Ray,
    Region,
    SCHEMA_VERSION,
    Segment,
    Target,
)


def test_geometry_record_round_trips_all_declared_primitives():
    state = _complete_state()
    hypothesis = GeometryHypothesis(
        id="hyp_angle_1",
        kind="angle_relation",
        payload={"points": ["A", "B", "C"], "value": "pi/2"},
        confidence=0.82,
        provenance=(
            Provenance(
                source_id="diagram_1",
                kind=EvidenceKind.PARSER,
                event_id="detect_angle_1",
                dependencies=("pixel_region_4",),
            ),
        ),
    )
    record = GeometryRecord(
        id="record_1",
        source={"type": "synthetic", "dataset": "contract_fixture"},
        state=state,
        hypotheses=[hypothesis],
        metadata={"split": "test"},
    )

    payload = json.loads(json.dumps(record.to_dict()))
    restored = GeometryRecord.from_dict(payload)

    assert restored.to_dict() == record.to_dict()
    assert restored.hypotheses[0].status is AdmissionStatus.PROPOSED
    assert restored.state is not None
    assert restored.state.target == Target(type="volume", object="tetra")
    assert restored.state.points["C"].coords[0] == sp.sqrt(2)


@pytest.mark.parametrize("version", [None, "0.9", "2.0"])
def test_geometry_record_rejects_missing_or_unknown_schema_versions(version):
    payload = {"id": "record_1", "source": {}, "state": None, "hypotheses": []}
    if version is not None:
        payload["schema_version"] = version
    with pytest.raises(ValueError, match="Unsupported geometry record schema version"):
        GeometryRecord.from_dict(payload)


def test_current_schema_version_is_explicit():
    assert SCHEMA_VERSION == "1.0"


def test_relations_get_stable_ids_and_symmetric_normalization():
    state = FormalGeometryState()
    state.add_point(Point("A", (sp.Integer(0), sp.Integer(0))))
    state.add_point(Point("B", (sp.Integer(1), sp.Integer(0))))
    state.add_point(Point("C", (sp.Integer(0), sp.Integer(1))))
    state.add_line(Line("line_a", (sp.Integer(0), sp.Integer(0)), (sp.Integer(1), sp.Integer(0)), ("A", "B")))
    state.add_line(Line("line_b", (sp.Integer(0), sp.Integer(0)), (sp.Integer(0), sp.Integer(1)), ("A", "C")))
    state.add_relation({"type": "orthogonal", "objects": ["line_b", "line_a"]})
    assert state.relations[0]["type"] == "perpendicular"
    assert state.relations[0]["objects"] == ["line_a", "line_b"]
    relation_id = state.relations[0]["id"]
    restored = FormalGeometryState.from_dict(state.to_dict())
    assert restored.relations[0]["id"] == relation_id
    assert restored.checksum() == state.checksum()


def test_unknown_state_relation_type_fails_at_boundary():
    state = FormalGeometryState()
    with pytest.raises(ValueError, match="Unknown geometry relation type"):
        state.add_relation({"type": "not_a_geometry_relation", "objects": ["A", "B"]})


def _complete_state() -> FormalGeometryState:
    state = FormalGeometryState()
    for point in (
        Point("A", (sp.Integer(0), sp.Integer(0))),
        Point("B", (sp.Integer(2), sp.Integer(0))),
        Point("C", (sp.sqrt(2), sp.sqrt(2))),
        Point("P0", (sp.Integer(0), sp.Integer(0), sp.Integer(0))),
        Point("P1", (sp.Integer(1), sp.Integer(0), sp.Integer(0))),
        Point("P2", (sp.Integer(0), sp.Integer(1), sp.Integer(0))),
        Point("P3", (sp.Integer(0), sp.Integer(0), sp.Integer(1))),
    ):
        state.add_point(point)
    state.add_line(Line("line_ab", (sp.Integer(0), sp.Integer(0)), (sp.Integer(1), sp.Integer(0)), ("A", "B")))
    state.add_segment(Segment("segment_ab", "A", "B"))
    state.add_ray(Ray("ray_ac", "A", (sp.sqrt(2), sp.Integer(2)), through="C"))
    state.add_circle(Circle("circle_a", "A", sp.Integer(2)))
    state.add_arc(Arc("arc_bc", "circle_a", "B", "C", central_angle=sp.pi / 4))
    state.add_polygon(Polygon("triangle_abc", ("A", "B", "C")))
    state.add_region(Region("region_abc", ("segment_ab", "arc_bc", "ray_ac"), kind="mixed_region"))
    state.add_plane(Plane("plane_xy", (sp.Integer(0), sp.Integer(0), sp.Integer(0)), (sp.Integer(0), sp.Integer(0), sp.Integer(1))))
    state.add_polyhedron(
        Polyhedron(
            "tetra",
            ("P0", "P1", "P2", "P3"),
            (("P0", "P1", "P2"), ("P0", "P1", "P3"), ("P0", "P2", "P3"), ("P1", "P2", "P3")),
        )
    )
    state.add_relation({"type": "perpendicular", "objects": ["line_ab", "ray_ac"]})
    state.construction_trace.append({"op": "fixture", "id": "event_1"})
    state.set_target(Target(type="volume", object="tetra"))
    return state
