"""Run one sampled circle/region/3D construction trace."""

from __future__ import annotations

import json

from geosym_world_geometry import ConstructionBuilder, FactClosureSolver
from geosym_world_geometry.export import export_record


def build_sample_record():
    trace = [
        {"op": "add_point", "id": "O", "coord": ["0", "0"]},
        {"op": "add_circle", "id": "circle_O", "center": "O", "radius": "5"},
        {"op": "add_line", "id": "diameter_line", "point": "O", "direction": ["1", "0"]},
        {"op": "intersection_line_circle", "ids": ["A", "B"], "line": "diameter_line", "circle": "circle_O"},
        {"op": "add_point", "id": "P", "coord": ["13", "0"]},
        {"op": "tangent_points_from_point", "ids": ["T1", "T2"], "point": "P", "circle": "circle_O"},
        {"op": "add_chord", "id": "T1T2", "circle": "circle_O", "points": ["T1", "T2"]},
        {"op": "add_arc", "id": "arc_AB", "circle": "circle_O", "start": "A", "end": "B", "central_angle": "pi"},
        {"op": "mark_sector_region", "id": "sector_AOB", "circle": "circle_O", "start": "A", "end": "B", "central_angle": "pi"},
        {"op": "add_point3d", "id": "U", "coord": ["0", "0", "0"]},
        {"op": "add_point3d", "id": "V", "coord": ["1", "0", "0"]},
        {"op": "add_point3d", "id": "W", "coord": ["0", "1", "0"]},
        {"op": "add_point3d", "id": "Z", "coord": ["0", "0", "1"]},
        {"op": "add_polyhedron", "id": "tetra_UVWZ", "vertices": ["U", "V", "W", "Z"], "faces": [["U", "W", "V"], ["U", "V", "Z"], ["V", "W", "Z"], ["W", "U", "Z"]]},
        {"op": "set_target", "type": "area", "object": "sector_AOB"},
    ]
    state = ConstructionBuilder().execute(trace)
    facts = FactClosureSolver().solve(state)
    return export_record(
        state,
        facts,
        record_id="sample_circle_sector_and_3d_interface",
        question="Find the area of sector AOB.",
    )


def main() -> None:
    record = build_sample_record()
    facts_by_type = {}
    for fact in record["facts"]:
        facts_by_type[fact["type"]] = facts_by_type.get(fact["type"], 0) + 1
    summary = {
        "id": record["id"],
        "num_points": len(record["formal_state"]["points"]),
        "num_circles": len(record["formal_state"]["circles"]),
        "num_arcs": len(record["formal_state"]["arcs"]),
        "num_regions": len(record["formal_state"]["regions"]),
        "num_polyhedra": len(record["formal_state"]["polyhedra"]),
        "num_facts": len(record["facts"]),
        "selected_fact_counts": {
            key: facts_by_type.get(key, 0)
            for key in ["point_on_circle", "tangent_to_circle", "chord_of_circle", "arc_length", "area", "volume", "target"]
        },
        "target": record["derivation"]["target_fact"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
