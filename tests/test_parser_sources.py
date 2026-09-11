from dataclasses import replace
from pathlib import Path

import pytest

from geosym_world.benchmark import (
    ParserProtocol,
    ParserSourceManifest,
    build_typed_parser_reference,
    hash_asset,
    materialize_parser_input,
    validate_typed_parser_reference,
)
from geosym_world.schema import ParseGraph


def test_materialized_image_input_has_opaque_identity_and_no_sidecar_fields(tmp_path):
    source = tmp_path / "right_triangle_b3_h4_answer5.png"
    source.write_bytes(b"not-a-real-image-but-stable-fixture")
    manifest = _manifest(source, ParserProtocol.IMAGE_QUESTION)

    parser_input, audit = materialize_parser_input(
        source,
        manifest=manifest,
        output_dir=tmp_path / "run",
        question="Find the hypotenuse.",
    )

    assert "right_triangle" not in parser_input.id
    assert "answer5" not in parser_input.image
    assert set(parser_input.to_dict()) == {"schema_version", "id", "image", "question"}
    assert hash_asset(parser_input.image) == manifest.asset_sha256
    assert audit["source_manifest_checksum"] == manifest.checksum()


def test_source_manifest_rejects_protocol_leak_and_asset_tampering(tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(b"fixture")
    with pytest.raises(ValueError, match="public_fields"):
        replace(_manifest(source, ParserProtocol.IMAGE_ONLY), public_fields=("image", "question"))

    manifest = _manifest(source, ParserProtocol.IMAGE_ONLY)
    with pytest.raises(ValueError, match="cannot expose a question"):
        materialize_parser_input(source, manifest=manifest, output_dir=tmp_path / "run", question="secret")

    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash does not match"):
        materialize_parser_input(source, manifest=manifest, output_dir=tmp_path / "run")
    with pytest.raises(ValueError, match="metadata must be a mapping"):
        replace(manifest, metadata=[])


def test_typed_reference_binds_public_input_manifest_and_parse_graph(tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(b"fixture")
    manifest = _manifest(source, ParserProtocol.IMAGE_ONLY)
    parser_input, _ = materialize_parser_input(source, manifest=manifest, output_dir=tmp_path / "run")
    graph = ParseGraph(id="graph_1", source_id=manifest.source_id)
    reference = build_typed_parser_reference(parser_input, graph=graph, manifest=manifest)

    restored_manifest, restored_graph = validate_typed_parser_reference(reference, parser_input)

    assert restored_manifest == manifest
    assert restored_graph == graph
    assert "parse_graph" not in parser_input.to_dict()
    with pytest.raises(ValueError, match="checksum"):
        validate_typed_parser_reference({**reference, "parse_graph": {}}, parser_input)

    Path(parser_input.image).write_bytes(b"tampered-public-image")
    with pytest.raises(ValueError, match="image hash"):
        validate_typed_parser_reference(reference, parser_input)


def _manifest(path, protocol: ParserProtocol) -> ParserSourceManifest:
    fields = ("image", "question") if protocol is ParserProtocol.IMAGE_QUESTION else ("image",)
    return ParserSourceManifest(
        source_id="source_fixture_1",
        protocol=protocol,
        source_version="fixture-v1",
        license_id="test-only",
        asset_sha256=hash_asset(path),
        split_group="fixture-family-1",
        public_fields=fields,
    )
