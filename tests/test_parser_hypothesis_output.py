import json
from pathlib import Path
from dataclasses import replace

import pytest

from geosym_world.benchmark import ParserBenchmarkSlicer, ParserStateEvaluator
from geosym_world.parser import parser_record_audit, validate_proposed_parser_record
from geosym_world.parser.pipeline import RealParsePipeline
from geosym_world.schema import AdmissionStatus, FormalGeometryState, GeometryRecord


FIXTURE = Path(__file__).parent / "fixtures" / "parser_v0_right_triangle.json"


def test_real_parser_persists_proposed_hypothesis_without_state(tmp_path):
    parsed_root = tmp_path / "parsed"
    pipeline = RealParsePipeline(
        "fixture",
        raw_root=tmp_path / "raw",
        parsed_root=parsed_root,
        extractor_backend="mock",
    )

    report = pipeline.process_record({"id": "sample_1", "question": "Find AC."})

    record_path = parsed_root / "fixture" / "records" / "sample_1.json"
    record = GeometryRecord.from_dict(json.loads(record_path.read_text(encoding="utf-8")))
    assert report["paths"]["record"] == str(record_path)
    assert record.state is None
    assert len(record.hypotheses) == 1
    assert record.hypotheses[0].status is AdmissionStatus.PROPOSED
    assert record.hypotheses[0].metadata["admission_required"] is True
    assert record.hypotheses[0].payload["problem_id"] == "sample_1"
    assert not (parsed_root / "fixture" / "problem_states").exists()


def test_parser_v0_fixture_reports_component_metrics(tmp_path):
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    parsed_root = tmp_path / "parsed"
    pipeline = RealParsePipeline(
        "fixture",
        raw_root=tmp_path / "raw",
        parsed_root=parsed_root,
        extractor_backend="manual_json",
    )

    pipeline.process_record(fixture["record"])

    record_path = parsed_root / "fixture" / "records" / "parser_v0_right_triangle.json"
    record = GeometryRecord.from_dict(json.loads(record_path.read_text(encoding="utf-8")))
    metrics = ParserStateEvaluator().evaluate(
        record.hypotheses[0].payload,
        fixture["reference"],
    )

    assert record.state is None
    assert metrics["objects"]["f1"] == 1.0
    assert metrics["relations"]["f1"] == 1.0
    assert metrics["quantities"]["f1"] == 1.0
    assert metrics["target"]["accuracy"] == 1.0
    assert metrics["grounding"] == {
        "accuracy": 1.0,
        "resolved": 5,
        "total": 5,
        "unresolved_ids": [],
    }
    assert metrics["macro_score"] == 1.0


def test_parser_record_contract_audit_requires_proposed_hypothesis(tmp_path):
    pipeline = RealParsePipeline(
        "fixture",
        raw_root=tmp_path / "raw",
        parsed_root=tmp_path / "parsed",
        extractor_backend="mock",
    )
    pipeline.process_record({"id": "sample_contract", "question": "Find AC."})
    record_path = tmp_path / "parsed" / "fixture" / "records" / "sample_contract.json"
    record = GeometryRecord.from_dict(json.loads(record_path.read_text(encoding="utf-8")))

    audit = parser_record_audit(record)

    assert audit["valid"] is True
    assert audit["state_present"] is False
    assert audit["admission_required"] is True
    with pytest.raises(ValueError, match="must not contain admitted state"):
        validate_proposed_parser_record(replace(record, state=FormalGeometryState()))


def test_parser_benchmark_slicer_aggregates_diagnostic_axes():
    rows = [
        {"parser": "real", "task_type": "area", "difficulty": "medium", "diagnostic_axes": ["region", "shadow"], "metrics": {"macro_score": 0.5}, "admission": {"decision": "admitted"}},
        {"parser": "real", "task_type": "area", "difficulty": "medium", "diagnostic_axes": ["region"], "metrics": {"macro_score": 1.0}, "admission": {"decision": "rejected"}},
    ]

    summary = ParserBenchmarkSlicer().summarize(rows)

    assert summary["num_rows"] == 2
    assert summary["slices"]["all"]["macro_score_mean"] == 0.75
    assert summary["slices"]["diagnostic_axis=region"]["num_rows"] == 2
    assert summary["slices"]["diagnostic_axis=shadow"]["num_admitted"] == 1
