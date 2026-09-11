import json

import pytest

from geosym_world.benchmark import (
    GeoparsingCorpusRecord,
    GeoparsingGrammarAuditor,
    load_geoparsing_corpus,
)
from geosym_world.parser import GeoparsingFormalDescription
from geosym_world.tools.audit_geoparsing_grammar_v0 import run


def test_geoparsing_grammar_audit_counts_supported_and_unsupported_syntax(tmp_path):
    input_path = tmp_path / "gdp.jsonl"
    output_path = tmp_path / "audit.json"
    rows = [
        {
            "id": "supported",
            "diagram": "not-read-by-the-auditor.png",
            "points": ["A", "B"],
            "lines": ["line A B"],
            "circles": [],
            "semantics": ["AB = 5"],
        },
        {
            "id": "unsupported",
            "diagram": "also-not-read.png",
            "points": ["A", "B"],
            "lines": ["line A B"],
            "circles": [],
            "semantics": ["A is somewhere near B"],
        },
    ]
    input_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    manifest = run(
        input_path=input_path,
        output_path=output_path,
        id_field="id",
        formal_field=None,
        max_examples=5,
        progress_every=0,
    )

    audit = manifest["audit"]
    assert audit["num_records"] == 2
    assert audit["num_complete"] == 1
    assert audit["complete_rate"] == 0.5
    assert audit["diagnostic_counts"] == {"unsupported_semantic": 1}
    assert audit["examples"][0]["record_id"] == "unsupported"
    assert audit["interpretation"] == "syntax coverage only; this is not image parsing accuracy"
    assert json.loads(output_path.read_text(encoding="utf-8")) == manifest


def test_geoparsing_corpus_loader_supports_an_explicit_nested_formal_field(tmp_path):
    path = tmp_path / "nested.jsonl"
    path.write_text(
        json.dumps(
            {
                "sample_id": "nested",
                "formal": {
                    "points": ["A", "B"],
                    "lines": ["line A B"],
                    "circles": [],
                    "semantics": [],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_geoparsing_corpus(path, id_field="sample_id", formal_field="formal")

    assert records[0].id == "nested"
    assert records[0].formal_description.lines == ("line A B",)


def test_geoparsing_grammar_auditor_rejects_duplicate_record_ids():
    description = GeoparsingFormalDescription.from_dict(
        {"points": ["A", "B"], "lines": ["line A B"], "circles": [], "semantics": []}
    )
    records = (
        GeoparsingCorpusRecord("duplicate", description),
        GeoparsingCorpusRecord("duplicate", description),
    )

    with pytest.raises(ValueError, match="unique"):
        GeoparsingGrammarAuditor().audit(records)


def test_geoparsing_grammar_audit_refuses_to_overwrite_input(tmp_path):
    path = tmp_path / "gdp.jsonl"
    path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="overwrite"):
        run(
            input_path=path,
            output_path=path,
            id_field="id",
            formal_field=None,
            max_examples=5,
            progress_every=0,
        )
