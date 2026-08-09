"""In-memory smoke checks for the GeoSym formal geometry world."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable

from geosym_world import PilotBenchmarkExporter
from geosym_world.datasets.pilot_exporter import _theorem_family


EXPECTED_TARGETS = {
    "pilot_3d_auto_section_area": ("section_area", "section_mid_area", "1/8"),
    "pilot_3d_auto_section_perimeter": ("section_perimeter", "section_mid_perim", "sqrt(2)/2 + 1"),
    "pilot_3d_space_angle": ("space_angle", "diag_OD|base_plane", "pi/4"),
    "pilot_parallel_cut_similarity": (
        "parallel_cut_ratio_residual",
        "tri_ADE|tri_ABC|apex:A|cut:D-E|base:B-C",
        "0",
    ),
    "pilot_tangent_chord_angle": ("tangent_chord_angle_residual", "AB|tan:P-A|obs:C|circle_O", "0"),
}

REQUIRED_THEOREM_FAMILIES = {
    "angle_bisector",
    "circle_chord",
    "circle_tangent_angle",
    "dihedral_angle_3d",
    "inscribed_angle",
    "isosceles_triangle",
    "line_plane_3d",
    "projection_3d",
    "right_triangle",
    "similar_triangle",
    "solid_section_3d",
    "space_angle_3d",
}


def main() -> None:
    records = PilotBenchmarkExporter().build_records()
    by_id = {record["id"]: record for record in records}
    _assert_all_valid(records)
    _assert_targets(by_id)
    family_counts, family_records = _family_coverage(records)
    missing_families = REQUIRED_THEOREM_FAMILIES.difference(family_counts)
    if missing_families:
        raise AssertionError(f"Missing theorem families: {sorted(missing_families)}")
    summary = {
        "num_records": len(records),
        "num_valid": sum(1 for record in records if record["verification"]["valid"]),
        "checked_targets": sorted(EXPECTED_TARGETS),
        "theorem_family_counts": dict(sorted(family_counts.items())),
        "theorem_family_records": {key: sorted(value) for key, value in sorted(family_records.items())},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _assert_all_valid(records: Iterable[Dict[str, Any]]) -> None:
    invalid = [record["id"] for record in records if not record["verification"]["valid"]]
    if invalid:
        raise AssertionError(f"Invalid pilot records: {invalid}")


def _assert_targets(by_id: Dict[str, Dict[str, Any]]) -> None:
    missing = set(EXPECTED_TARGETS).difference(by_id)
    if missing:
        raise AssertionError(f"Missing target records: {sorted(missing)}")
    for record_id, (fact_type, object_id, value) in EXPECTED_TARGETS.items():
        target = by_id[record_id]["derivation"]["target_fact"]
        if target is None:
            raise AssertionError(f"{record_id} has no target fact")
        actual = (target["dependencies"], target["object"], target["value"])
        if target["object"] != object_id or target["value"] != value:
            raise AssertionError(
                f"{record_id} target mismatch: expected ({fact_type}, {object_id}, {value}), "
                f"got ({target['type']}, {target['object']}, {target['value']}); dependencies={actual[0]}"
            )
        facts = by_id[record_id]["derivation"]["facts"]
        if not any(fact["type"] == fact_type and fact["object"] == object_id and fact["value"] == value for fact in facts):
            raise AssertionError(f"{record_id} derivation does not contain expected source fact {fact_type}:{object_id}={value}")


def _family_coverage(records: Iterable[Dict[str, Any]]):
    family_counts: Counter[str] = Counter()
    family_records: dict[str, set[str]] = defaultdict(set)
    for record in records:
        for fact in record.get("facts") or []:
            family = _theorem_family(str(fact.get("type")))
            if family:
                family_counts[family] += 1
                family_records[family].add(str(record["id"]))
    return family_counts, family_records


if __name__ == "__main__":
    main()
