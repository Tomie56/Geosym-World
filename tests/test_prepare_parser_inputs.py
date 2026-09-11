import json
from pathlib import Path

import pytest
from PIL import Image

from geosym_world.tools.prepare_parser_inputs_v0 import run


def test_prepare_parser_inputs_materializes_only_protocol_public_fields(tmp_path):
    first_image = tmp_path / "source_a.png"
    second_image = tmp_path / "source_b.png"
    Image.new("RGB", (24, 24), "white").save(first_image)
    Image.new("RGB", (24, 24), "black").save(second_image)
    catalog_path = tmp_path / "catalog.jsonl"
    rows = [
        _catalog_row("source_a", "image_only", first_image, question=None),
        _catalog_row("source_b", "image_question", second_image, question="Find the length of AB."),
    ]
    catalog_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    manifest = run(catalog_path=catalog_path, output_dir=tmp_path / "prepared", progress_every=0)

    parser_rows = _read_jsonl(tmp_path / "prepared" / "parser-input.jsonl")
    audit_rows = _read_jsonl(tmp_path / "prepared" / "source-audit.jsonl")
    assert manifest["num_records"] == 2
    assert manifest["protocol_counts"] == {"image_only": 1, "image_question": 1}
    assert all(set(row) == {"schema_version", "id", "image", "question"} for row in parser_rows)
    assert all("source_" not in row["id"] for row in parser_rows)
    assert all((tmp_path / "prepared" / "public_inputs") in Path(row["image"]).parents for row in parser_rows)
    assert {row["source_manifest"]["source_id"] for row in audit_rows} == {"source_a", "source_b"}
    assert all(row["checksum"] for row in audit_rows)


def test_prepare_parser_inputs_rejects_duplicates_before_materialization(tmp_path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (16, 16), "white").save(image_path)
    row = _catalog_row("duplicate", "image_only", image_path, question=None)
    catalog_path = tmp_path / "catalog.jsonl"
    catalog_path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate source IDs"):
        run(catalog_path=catalog_path, output_dir=tmp_path / "prepared", progress_every=0)

    assert not (tmp_path / "prepared").exists()


def _catalog_row(source_id, protocol, image, *, question):
    row = {
        "schema_version": "1.0",
        "source_id": source_id,
        "protocol": protocol,
        "source_version": "fixture-v1",
        "license_id": "test-only",
        "split_group": source_id,
        "image": str(image),
    }
    if question is not None:
        row["question"] = question
    return row


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
