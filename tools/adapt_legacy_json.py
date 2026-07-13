"""Convert one legacy-like geometry JSON file into a formal-world record."""

from __future__ import annotations

from pathlib import Path
import argparse
import json

from geosym_world_geometry import ConstructionBuilder, FactClosureSolver, LegacyGeometryAdapter, TraceVerifier
from geosym_world_geometry.export import export_record
from geosym_world_geometry.expected import compare_expected_facts, legacy_expected_facts


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt one legacy-like GeoSym JSON record into the new formal world format.")
    parser.add_argument("--input", required=True, help="Input legacy-like JSON or JSONL path.")
    parser.add_argument("--input-jsonl", action="store_true", help="Read one line from a JSONL input file.")
    parser.add_argument("--line-index", type=int, default=0, help="0-based JSONL line index when --input-jsonl is set.")
    parser.add_argument(
        "--output",
        default=str(Path("geosym_world_geometry") / "outputs" / "adapted_legacy_record.json"),
        help="Output JSON path. Must be under geosym_world_geometry/.",
    )
    parser.add_argument("--record-id", default="adapted_legacy_record")
    parser.add_argument("--question", default="")
    parser.add_argument("--answer", default="")
    parser.add_argument("--target-type", help="Override or provide target type before adaptation.")
    parser.add_argument("--target-object", help="Override or provide target object before adaptation.")
    args = parser.parse_args()

    output = Path(args.output)
    if "geosym_world_geometry" not in output.parts:
        raise SystemExit("Refusing to write outside geosym_world_geometry/")
    record = _read_input(Path(args.input), input_jsonl=args.input_jsonl, line_index=args.line_index)
    if args.target_type and args.target_object:
        record["target"] = {"type": args.target_type, "object": args.target_object}
    adapted = LegacyGeometryAdapter().adapt(record)
    trace = adapted["trace"]
    adapter_report = adapted["report"]
    verification = TraceVerifier().verify_trace(trace)
    state = ConstructionBuilder().execute(trace)
    facts = FactClosureSolver().solve(state)
    formal_record = export_record(
        state,
        facts,
        record_id=args.record_id,
        question=args.question,
        answer=args.answer,
        verification=verification,
    )
    expected_facts = legacy_expected_facts(record)
    expected_fact_comparison = compare_expected_facts(expected_facts, formal_record["facts"])
    formal_record["adapter"] = {
        "source": str(args.input),
        "num_trace_ops": len(trace),
        "verification_valid": verification.valid,
        "preserved_metadata": adapter_report.get("preserved_metadata", {}),
        "expected_facts": expected_facts,
        "expected_fact_comparison": expected_fact_comparison,
        "report": adapter_report,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(formal_record, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "output": str(output),
        "input": str(args.input),
        "num_trace_ops": len(trace),
        "num_facts": len(formal_record["facts"]),
        "verification_valid": verification.valid,
        "num_skipped": adapter_report["num_skipped"],
        "coverage_score": adapter_report.get("coverage_score"),
        "coverage": adapter_report.get("coverage", {}),
        "expected_fact_summary": {
            key: expected_fact_comparison[key]
            for key in ("num_expected", "num_matched", "num_missing_formal_fact", "num_value_mismatch", "num_unchecked", "quality_label", "method_counts")
        },
        "preserved_metadata": adapter_report.get("preserved_metadata", {}),
        "unhandled_top_level_keys": adapter_report["unhandled_top_level_keys"],
        "target": formal_record["derivation"]["target_fact"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _read_input(path: Path, *, input_jsonl: bool, line_index: int):
    if not input_jsonl:
        return json.loads(path.read_text(encoding="utf-8"))
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if line_index < 0 or line_index >= len(lines):
        raise SystemExit(f"JSONL line index out of range: {line_index}; num_lines={len(lines)}")
    return json.loads(lines[line_index])


if __name__ == "__main__":
    main()
