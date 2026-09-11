"""Traceable parser sources and leak-resistant benchmark materialization."""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from geosym_world.errors import InvalidGeometryInput
from geosym_world.parser.contracts import ParserInput
from geosym_world.schema import ParseGraph
from geosym_world.schema.canonical import stable_checksum
from geosym_world.schema.ids import validate_geometry_id


PARSER_SOURCE_SCHEMA_VERSION = "1.0"
PARSER_SOURCE_CATALOG_SCHEMA_VERSION = "1.0"
TYPED_PARSER_REFERENCE_SCHEMA_VERSION = "1.0"
_SHA256_LENGTH = 64


class ParserProtocol(str, Enum):
    IMAGE_ONLY = "image_only"
    IMAGE_QUESTION = "image_question"
    NATIVE_IMPORT = "native_import"


@dataclass(frozen=True)
class ParserSourceCatalogEntry:
    source_id: str
    protocol: ParserProtocol
    source_version: str
    license_id: str
    split_group: str
    image: str
    question: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = PARSER_SOURCE_CATALOG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_geometry_id(self.source_id)
        if self.schema_version != PARSER_SOURCE_CATALOG_SCHEMA_VERSION:
            raise InvalidGeometryInput(f"Unsupported parser source catalog schema version: {self.schema_version}")
        if self.protocol not in {ParserProtocol.IMAGE_ONLY, ParserProtocol.IMAGE_QUESTION}:
            raise InvalidGeometryInput("Parser image source catalog only supports image_only and image_question")
        for name, value in (
            ("source_version", self.source_version),
            ("license_id", self.license_id),
            ("split_group", self.split_group),
            ("image", self.image),
        ):
            if not isinstance(value, str) or not value.strip():
                raise InvalidGeometryInput(f"Parser source catalog {name} must be non-empty")
        if not isinstance(self.question, str):
            raise InvalidGeometryInput("Parser source catalog question must be a string")
        if self.protocol is ParserProtocol.IMAGE_ONLY and self.question:
            raise InvalidGeometryInput("image_only source catalog entries cannot expose a question")
        if self.protocol is ParserProtocol.IMAGE_QUESTION and not self.question.strip():
            raise InvalidGeometryInput("image_question source catalog entries require a question")
        if not isinstance(self.metadata, Mapping):
            raise InvalidGeometryInput("Parser source catalog metadata must be a mapping")

    def build_manifest(self) -> "ParserSourceManifest":
        return ParserSourceManifest(
            source_id=self.source_id,
            protocol=self.protocol,
            source_version=self.source_version,
            license_id=self.license_id,
            asset_sha256=hash_asset(self.image),
            split_group=self.split_group,
            public_fields=("image",) if self.protocol is ParserProtocol.IMAGE_ONLY else ("image", "question"),
            metadata=dict(self.metadata),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParserSourceCatalogEntry":
        if not isinstance(data, Mapping):
            raise InvalidGeometryInput("Parser source catalog entry must be a mapping")
        required = {"schema_version", "source_id", "protocol", "source_version", "license_id", "split_group", "image"}
        optional = {"question", "metadata"}
        missing = sorted(required - set(data))
        unknown = sorted(set(data) - required - optional)
        if missing:
            raise InvalidGeometryInput(f"Parser source catalog entry is missing fields: {missing}")
        if unknown:
            raise InvalidGeometryInput(f"Parser source catalog entry has unknown fields: {unknown}")
        metadata = data.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise InvalidGeometryInput("Parser source catalog metadata must be a mapping")
        return cls(
            source_id=str(data["source_id"]),
            protocol=ParserProtocol(data["protocol"]),
            source_version=str(data["source_version"]),
            license_id=str(data["license_id"]),
            split_group=str(data["split_group"]),
            image=str(data["image"]),
            question=str(data.get("question", "")),
            metadata=dict(metadata),
            schema_version=str(data["schema_version"]),
        )


@dataclass(frozen=True)
class ParserSourceManifest:
    source_id: str
    protocol: ParserProtocol
    source_version: str
    license_id: str
    asset_sha256: str
    split_group: str
    public_fields: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = PARSER_SOURCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_geometry_id(self.source_id)
        if not isinstance(self.protocol, ParserProtocol):
            raise InvalidGeometryInput(f"Unknown parser protocol: {self.protocol}")
        for name, value in (
            ("source_version", self.source_version),
            ("license_id", self.license_id),
            ("split_group", self.split_group),
        ):
            if not isinstance(value, str) or not value.strip():
                raise InvalidGeometryInput(f"Parser source {name} must be non-empty")
        if len(self.asset_sha256) != _SHA256_LENGTH or any(character not in "0123456789abcdef" for character in self.asset_sha256):
            raise InvalidGeometryInput("Parser source asset_sha256 must be a lowercase SHA-256 digest")
        if len(set(self.public_fields)) != len(self.public_fields):
            raise InvalidGeometryInput("Parser source public_fields must be unique")
        expected_fields = {
            ParserProtocol.IMAGE_ONLY: ("image",),
            ParserProtocol.IMAGE_QUESTION: ("image", "question"),
            ParserProtocol.NATIVE_IMPORT: ("native_payload",),
        }[self.protocol]
        if tuple(self.public_fields) != expected_fields:
            raise InvalidGeometryInput(
                f"Parser source protocol {self.protocol.value} requires public_fields={expected_fields}"
            )
        if not isinstance(self.metadata, Mapping):
            raise InvalidGeometryInput("Parser source metadata must be a mapping")
        if self.schema_version != PARSER_SOURCE_SCHEMA_VERSION:
            raise InvalidGeometryInput(f"Unsupported parser source schema version: {self.schema_version}")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "protocol": self.protocol.value,
            "source_version": self.source_version,
            "license_id": self.license_id,
            "asset_sha256": self.asset_sha256,
            "split_group": self.split_group,
            "public_fields": list(self.public_fields),
            "metadata": dict(self.metadata),
        }

    def checksum(self) -> str:
        return stable_checksum(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["checksum"] = stable_checksum(payload)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParserSourceManifest":
        allowed = {
            "schema_version",
            "source_id",
            "protocol",
            "source_version",
            "license_id",
            "asset_sha256",
            "split_group",
            "public_fields",
            "metadata",
            "checksum",
        }
        unknown = sorted(set(data) - allowed)
        missing = sorted(allowed - set(data))
        if unknown:
            raise InvalidGeometryInput(f"Parser source manifest has unknown fields: {unknown}")
        if missing:
            raise InvalidGeometryInput(f"Parser source manifest is missing fields: {missing}")
        checksum_payload = {key: value for key, value in data.items() if key != "checksum"}
        if data["checksum"] != stable_checksum(checksum_payload):
            raise InvalidGeometryInput("Parser source manifest checksum does not match payload")
        manifest = cls(
            source_id=str(data["source_id"]),
            protocol=ParserProtocol(data["protocol"]),
            source_version=str(data["source_version"]),
            license_id=str(data["license_id"]),
            asset_sha256=str(data["asset_sha256"]),
            split_group=str(data["split_group"]),
            public_fields=tuple(str(item) for item in data["public_fields"]),
            metadata=dict(data["metadata"]),
            schema_version=str(data["schema_version"]),
        )
        if manifest.checksum() != data["checksum"]:
            raise InvalidGeometryInput("Parser source manifest canonical checksum does not match payload")
        return manifest


def hash_asset(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise InvalidGeometryInput(f"Parser source asset does not exist: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_parser_input(
    source_image: str | Path,
    *,
    manifest: ParserSourceManifest,
    output_dir: str | Path,
    question: str = "",
) -> tuple[ParserInput, dict[str, Any]]:
    """Copy one image to an opaque public path and return a traceable audit record."""

    if manifest.protocol is ParserProtocol.NATIVE_IMPORT:
        raise InvalidGeometryInput("Native import sources cannot be materialized as image ParserInput")
    if manifest.protocol is ParserProtocol.IMAGE_ONLY and question:
        raise InvalidGeometryInput("image_only protocol cannot expose a question")
    source = Path(source_image)
    actual_sha256 = hash_asset(source)
    if actual_sha256 != manifest.asset_sha256:
        raise InvalidGeometryInput("Parser source asset hash does not match manifest")
    suffix = source.suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        raise InvalidGeometryInput(f"Unsupported parser image extension: {suffix or '<missing>'}")
    public_root = Path(output_dir) / "public_inputs"
    public_root.mkdir(parents=True, exist_ok=True)
    public_path = public_root / f"asset_{actual_sha256[:24]}{suffix}"
    if public_path.exists() and hash_asset(public_path) != actual_sha256:
        raise InvalidGeometryInput(f"Opaque parser asset path contains different data: {public_path}")
    if not public_path.exists():
        temporary = public_path.with_name(f".{public_path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            shutil.copyfile(source, temporary)
            if hash_asset(temporary) != actual_sha256:
                raise InvalidGeometryInput("Copied parser source asset hash does not match manifest")
            temporary.replace(public_path)
        finally:
            temporary.unlink(missing_ok=True)
    public_id = f"sample_{stable_checksum({'source_id': manifest.source_id, 'asset_sha256': actual_sha256})[:16]}"
    parser_input = ParserInput(id=public_id, image=str(public_path), question=question)
    audit = {
        "schema_version": "1.0",
        "public_id": public_id,
        "public_asset_sha256": actual_sha256,
        "parser_input_checksum": stable_checksum(parser_input.to_dict()),
        "source_manifest_checksum": manifest.checksum(),
    }
    audit["checksum"] = stable_checksum(audit)
    return parser_input, audit


def build_typed_parser_reference(
    parser_input: ParserInput | Mapping[str, Any],
    *,
    graph: ParseGraph,
    manifest: ParserSourceManifest,
) -> dict[str, Any]:
    """Build evaluator-only ParseGraph truth bound to one public input."""

    public_input = parser_input if isinstance(parser_input, ParserInput) else ParserInput.from_dict(parser_input)
    if graph.source_id != manifest.source_id:
        raise InvalidGeometryInput("ParseGraph source_id does not match parser source manifest")
    _validate_public_input_manifest(public_input, manifest)
    reference = {
        "schema_version": TYPED_PARSER_REFERENCE_SCHEMA_VERSION,
        "id": public_input.id,
        "parser_input_checksum": stable_checksum(public_input.to_dict()),
        "source_manifest": manifest.to_dict(),
        "parse_graph": graph.to_dict(),
    }
    reference["checksum"] = stable_checksum(reference)
    return reference


def validate_typed_parser_reference(
    reference: Mapping[str, Any],
    parser_input: ParserInput | Mapping[str, Any],
) -> tuple[ParserSourceManifest, ParseGraph]:
    public_input = parser_input if isinstance(parser_input, ParserInput) else ParserInput.from_dict(parser_input)
    if reference.get("schema_version") != TYPED_PARSER_REFERENCE_SCHEMA_VERSION:
        raise InvalidGeometryInput("Unsupported typed parser reference schema version")
    if reference.get("id") != public_input.id:
        raise InvalidGeometryInput("Typed parser reference and public input IDs do not match")
    if reference.get("parser_input_checksum") != stable_checksum(public_input.to_dict()):
        raise InvalidGeometryInput("Typed parser input checksum does not match reference")
    checksum_payload = {key: value for key, value in reference.items() if key != "checksum"}
    if reference.get("checksum") != stable_checksum(checksum_payload):
        raise InvalidGeometryInput("Typed parser reference checksum does not match payload")
    manifest = ParserSourceManifest.from_dict(reference["source_manifest"])
    _validate_public_input_manifest(public_input, manifest)
    graph = ParseGraph.from_dict(reference["parse_graph"])
    if graph.source_id != manifest.source_id:
        raise InvalidGeometryInput("Typed parser reference source IDs do not match")
    return manifest, graph


def _validate_public_input_manifest(parser_input: ParserInput, manifest: ParserSourceManifest) -> None:
    if manifest.protocol is ParserProtocol.NATIVE_IMPORT:
        raise InvalidGeometryInput("Native import manifests cannot be bound to image ParserInput")
    if manifest.protocol is ParserProtocol.IMAGE_ONLY and parser_input.question:
        raise InvalidGeometryInput("image_only manifest cannot expose a ParserInput question")
    if manifest.protocol is ParserProtocol.IMAGE_QUESTION and not parser_input.question.strip():
        raise InvalidGeometryInput("image_question manifest requires a ParserInput question")
    if hash_asset(parser_input.image) != manifest.asset_sha256:
        raise InvalidGeometryInput("ParserInput image hash does not match source manifest")
