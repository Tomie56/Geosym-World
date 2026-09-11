"""Evaluate typed ParseGraph predictions against private, checksum-bound references."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from geosym_world.benchmark import ParseGraphEvaluator, validate_typed_parser_reference
from geosym_world.parser import ParserInput
from geosym_world.schema import ParseGraph
from geosym_world.schema.canonical import stable_checksum


DEFAULT_OUTPUT = "artifacts/benchmark/parse_graph_v0/metrics.jsonl"
COMPONENTS = ("objects", "relations", "labels", "markers", "bindings", "topology", "quantities", "regions", "targets")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Typed parser or workflow result JSONL.")
    parser.add_argument("--references", required=True, help="Private typed parser reference JSONL.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Per-record metric JSONL.")
    parser.add_argument("--point-tolerance", type=float, default=0.035)
    args = parser.parse_args()
    try:
        manifest = run(
            predictions_path=Path(args.predictions),
            references_path=Path(args.references),
            output_path=Path(args.output),
            point_tolerance=args.point_tolerance,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def run(
    *,
    predictions_path: Path,
    references_path: Path,
    output_path: Path,
    point_tolerance: float = 0.035,
) -> dict[str, Any]:
    prediction_rows = _read_jsonl(predictions_path)
    reference_rows = _read_jsonl(references_path)
    references = _index_rows(reference_rows, references_path)
    predictions = _index_predictions(prediction_rows, predictions_path)
    if set(predictions) != set(references):
        missing = sorted(set(references) - set(predictions))
        unexpected = sorted(set(predictions) - set(references))
        raise ValueError(f"Prediction/reference IDs differ; missing={missing}, unexpected={unexpected}")

    evaluator = ParseGraphEvaluator(point_tolerance=point_tolerance)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    diagnostic_counts: Counter[str] = Counter()
    component_totals: Counter[str] = Counter()
    component_exact_counts: Counter[str] = Counter()
    perfect_parsing_count = 0
    core_perfect_parsing_count = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for sample_id in sorted(predictions):
            parser_input, prediction = predictions[sample_id]
            _, reference = validate_typed_parser_reference(references[sample_id], parser_input)
            metrics = evaluator.evaluate(prediction, reference)
            diagnostic_counts.update(metrics["diagnostic_axes"])
            for component in COMPONENTS:
                component_totals[component] += metrics[component]["f1"]
                component_exact_counts[component] += int(metrics["component_exact_match"][component])
            perfect_parsing_count += int(metrics["perfect_parsing"])
            core_perfect_parsing_count += int(metrics["core_perfect_parsing"])
            row = {"schema_version": "geosym.parse_graph_metric.v2", "id": sample_id, "metrics": metrics}
            row["checksum"] = stable_checksum(row)
            output_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            rows.append(row)

    count = len(rows)
    manifest = {
        "schema_version": "geosym.parse_graph_eval_manifest.v2",
        "predictions": str(predictions_path),
        "predictions_sha256": _file_hash(predictions_path),
        "references": str(references_path),
        "references_sha256": _file_hash(references_path),
        "output": str(output_path),
        "output_sha256": _file_hash(output_path),
        "configuration": {"point_tolerance": point_tolerance},
        "num_records": count,
        "component_macro_f1": {
            component: component_totals[component] / count if count else 0.0
            for component in COMPONENTS
        },
        "component_sample_accuracy": {
            component: component_exact_counts[component] / count if count else 0.0
            for component in COMPONENTS
        },
        "perfect_parsing_rate": perfect_parsing_count / count if count else 0.0,
        "core_perfect_parsing_rate": core_perfect_parsing_count / count if count else 0.0,
        "diagnostic_counts": dict(sorted(diagnostic_counts.items())),
    }
    manifest["checksum"] = stable_checksum(manifest)
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**manifest, "manifest": str(manifest_path)}


def _index_predictions(rows: list[dict[str, Any]], path: Path) -> dict[str, tuple[ParserInput, ParseGraph]]:
    indexed = {}
    for line_number, row in enumerate(rows, start=1):
        payload = row.get("result", row)
        parser_payload = payload.get("parser", payload)
        if not isinstance(parser_payload, dict):
            raise ValueError(f"Missing parser payload at {path}:{line_number}")
        try:
            parser_input = ParserInput.from_dict(parser_payload["parser_input"])
            graph = ParseGraph.from_dict(parser_payload["parse_graph"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid typed prediction at {path}:{line_number}: {exc}") from exc
        if parser_input.id in indexed:
            raise ValueError(f"Duplicate prediction id {parser_input.id!r} at {path}:{line_number}")
        indexed[parser_input.id] = parser_input, graph
    return indexed


def _index_rows(rows: list[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    indexed = {}
    for line_number, row in enumerate(rows, start=1):
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"Reference is missing id at {path}:{line_number}")
        if sample_id in indexed:
            raise ValueError(f"Duplicate reference id {sample_id!r} at {path}:{line_number}")
        indexed[sample_id] = row
    return indexed


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"JSONL input does not exist: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
