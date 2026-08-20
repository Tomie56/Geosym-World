import json

import sympy as sp

from geosym_world.construction.builder import ConstructionBuilder
from geosym_world.parser import SyntheticParserDebugger, SyntheticParserV0
from geosym_world.tools.parse_synthetic_v0 import main as parse_synthetic_v0_main
from geosym_world.tools.diagnose_parser_v0 import main as diagnose_parser_v0_main


def _record():
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": ["4", "0"]},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
        {"op": "set_target", "type": "length", "object": "AB"},
    ]
    state = ConstructionBuilder().execute(trace)
    return {
        "id": "parser_fixture_1",
        "formal_state": state.to_dict(),
        "construction_trace": trace,
        "answer": "4",
        "facts": [{"type": "length", "object": "AB", "value": "4"}],
    }


def test_synthetic_parser_v0_keeps_candidate_proposed_until_admission():
    result = SyntheticParserV0().parse(_record())
    assert result.hypothesis.status.value == "proposed"
    assert result.record.state is None
    assert result.observation.metadata["facts_excluded"] is True
    assert result.hypothesis.metadata["admission_required"] is True


def test_synthetic_parser_v0_admission_replays_trace_into_debug_state():
    record = _record()
    result = SyntheticParserV0().parse_and_admit(record)
    assert result.admitted
    assert result.admitted_state is not None
    assert result.admitted_state.checksum() == result.record.state.checksum()
    assert result.admission.transition is not None
    assert result.admission.transition.provenance["hypothesis_id"] == result.hypothesis.id
    assert result.admitted_state.points["B"].coords == (sp.Integer(4), sp.Integer(0))


def test_synthetic_parser_v0_rejection_preserves_debug_diagnostics():
    record = _record()
    record["construction_trace"] = [
        {"op": "add_point", "id": "C", "coord": ["0", "1"]},
        {"op": "midpoint", "id": "M", "of": ["C", "MISSING"]},
    ]
    result = SyntheticParserV0().parse_and_admit(record)
    assert not result.admitted
    assert result.admission.reason == "grounding_failed"
    assert result.admission.grounding.unresolved_objects == ("MISSING",)
    assert result.admitted_state is None


def test_synthetic_parser_debugger_reports_components_and_admission():
    record = _record()
    result = SyntheticParserV0().parse_and_admit(record)
    reference = {
        "geometry": {
            "points": [{"id": "A"}, {"id": "B"}],
            "lines": [],
            "segments": [{"id": "AB"}],
            "relations": [],
            "quantities": [],
            "arcs": [],
            "entities": [],
        },
        "target": {"type": "length", "object": "AB"},
    }
    diagnosis = SyntheticParserDebugger().diagnose(result, reference)
    assert diagnosis["metrics"]["macro_score"] == 1.0
    assert diagnosis["admission"]["decision"] == "admitted"
    assert diagnosis["admitted_state_checksum"] == diagnosis["reference_state_checksum"]


def test_synthetic_parser_v0_cli_writes_results_and_progress(tmp_path, monkeypatch, capsys):
    input_path = tmp_path / "records.jsonl"
    output_path = tmp_path / "parsed.jsonl"
    input_path.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "parse_synthetic_v0",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--admit",
            "--progress-every",
            "1",
        ],
    )

    parse_synthetic_v0_main()

    result = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert result["admission"]["decision"] == "admitted"
    assert result["record"]["state"] is None
    assert "processed=1 admitted=1" in capsys.readouterr().err


def test_synthetic_parser_diagnosis_cli_writes_slice_summary(tmp_path, monkeypatch, capsys):
    input_path = tmp_path / "records.jsonl"
    output_path = tmp_path / "diagnosis.jsonl"
    input_path.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "diagnose_parser_v0",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--admit",
            "--progress-every",
            "1",
        ],
    )

    diagnose_parser_v0_main()

    row = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert row["metrics"]["macro_score"] == 1.0
    assert summary["slices"]["all"]["num_admitted"] == 1
    assert "processed=1" in captured.err
