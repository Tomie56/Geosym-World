"""Inspect theorem-lite facts on the built-in pilot samples."""

from __future__ import annotations

import json

from geosym_world import PilotBenchmarkExporter
from geosym_world.datasets.samples import built_in_pilot_samples


def main() -> None:
    records = PilotBenchmarkExporter().build_records(built_in_pilot_samples())
    wanted = {
        "right_triangle",
        "pythagorean_relation",
        "central_angle",
        "chord_length_formula_check",
        "equal_chord_equal_central_angle",
    }
    summary = {}
    for record in records:
        counts = {}
        examples = {}
        for fact in record["facts"]:
            fact_type = fact["type"]
            if fact_type in wanted:
                counts[fact_type] = counts.get(fact_type, 0) + 1
                examples.setdefault(fact_type, fact)
        summary[record["id"]] = {"counts": counts, "examples": examples}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
