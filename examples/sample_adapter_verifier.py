"""Validate the trace verifier and legacy dictionary adapter on one sample."""

from __future__ import annotations

import json

from geosym_world_geometry import ConstructionBuilder, FactClosureSolver, LegacyGeometryAdapter, TraceVerifier
from geosym_world_geometry.export import export_record


def build_sample_record():
    legacy_like = {
        "points": {
            "A": {"x": "0", "y": "0"},
            "B": {"x": "4", "y": "0"},
            "C": {"x": "4", "y": "3"},
            "O": {"x": "0", "y": "0"},
        },
        "segments": [
            {"id": "AB", "points": ["A", "B"]},
            {"id": "BC", "points": ["B", "C"]},
            {"id": "AC", "points": ["A", "C"]},
        ],
        "circles": [
            {"id": "circle_O", "center": "O", "radius": "5"},
        ],
        "polygons": [
            {"id": "tri_ABC", "vertices": ["A", "B", "C"]},
        ],
        "regions": [
            {"id": "region_ABC", "polygon": "tri_ABC"},
        ],
        "target": {"type": "length", "object": "AC"},
    }
    trace = LegacyGeometryAdapter().to_trace(legacy_like)
    verification = TraceVerifier().verify_trace(trace)
    state = ConstructionBuilder().execute(trace)
    facts = FactClosureSolver().solve(state)
    return export_record(
        state,
        facts,
        record_id="sample_legacy_adapter_verifier",
        question="Find the length of AC.",
        verification=verification,
    )


def main() -> None:
    record = build_sample_record()
    summary = {
        "id": record["id"],
        "verification_valid": record["verification"]["valid"],
        "verification_metrics": record["verification"]["metrics"],
        "num_trace_ops": len(record["formal_state"]["construction_trace"]),
        "num_facts": len(record["facts"]),
        "target": record["derivation"]["target_fact"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
