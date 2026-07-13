"""Run one sampled construction trace and print a compact JSON summary."""

from __future__ import annotations

import json

from geosym_world_geometry import ConstructionBuilder, FactClosureSolver
from geosym_world_geometry.export import export_record


def build_sample_record():
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": ["4", "0"]},
        {"op": "line_through_angle", "id": "l_AC", "point": "A", "angle": "pi/6"},
        {"op": "line_through_angle", "id": "l_BC", "point": "B", "angle": "2*pi/3"},
        {"op": "intersection_line_line", "id": "C", "line1": "l_AC", "line2": "l_BC"},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
        {"op": "add_segment", "id": "AC", "points": ["A", "C"]},
        {"op": "add_segment", "id": "BC", "points": ["B", "C"]},
        {"op": "add_polygon", "id": "tri_ABC", "vertices": ["A", "B", "C"]},
        {"op": "midpoint", "id": "M", "of": ["A", "B"]},
        {"op": "angle_bisector", "id": "bis_ABC", "vertex": "B", "ray1": ["B", "A"], "ray2": ["B", "C"]},
        {"op": "foot_of_perpendicular", "id": "H", "point": "C", "line": "AB"},
        {"op": "add_point3d", "id": "P", "coord": ["0", "0", "0"]},
        {"op": "add_point3d", "id": "Q", "coord": ["1", "0", "0"]},
        {"op": "add_point3d", "id": "R", "coord": ["0", "1", "0"]},
        {"op": "add_point3d", "id": "S", "coord": ["0", "0", "1"]},
        {"op": "add_plane", "id": "plane_PQR", "points": ["P", "Q", "R"]},
        {"op": "add_polyhedron", "id": "tetra_PQRS", "vertices": ["P", "Q", "R", "S"], "faces": [["P", "Q", "R"], ["P", "Q", "S"], ["P", "R", "S"], ["Q", "R", "S"]]},
        {"op": "set_target", "type": "area", "object": "tri_ABC"},
    ]
    state = ConstructionBuilder().execute(trace)
    facts = FactClosureSolver().solve(state)
    return export_record(
        state,
        facts,
        record_id="sample_angle_triangle_with_3d_interface",
        question="Find the area of triangle ABC.",
    )


def main() -> None:
    record = build_sample_record()
    summary = {
        "id": record["id"],
        "num_points": len(record["formal_state"]["points"]),
        "num_lines": len(record["formal_state"]["lines"]),
        "num_segments": len(record["formal_state"]["segments"]),
        "num_planes": len(record["formal_state"]["planes"]),
        "num_polyhedra": len(record["formal_state"]["polyhedra"]),
        "num_facts": len(record["facts"]),
        "target": record["derivation"]["target_fact"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
