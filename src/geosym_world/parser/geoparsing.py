"""Strict import of Geoparsing/GDP-29K formal descriptions."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from geosym_world.errors import InvalidGeometryInput
from geosym_world.schema import (
    EvidencePrecision,
    ParseEvidence,
    ParseEvidenceKind,
    ParseGraph,
    ParseObject,
    ParseObjectKind,
    ParseOrigin,
    ParseQuantity,
    ParseReference,
    ParseRelation,
    math_expression_from_text,
)
from geosym_world.schema.canonical import stable_checksum

from .contracts import ParserInput, validate_parser_input
from .external_v0 import ExternalParseGraphPrediction
from .geoparsing_contracts import (
    GEOPARSING_ADAPTER_VERSION,
    GEOPARSING_PREDICTION_SCHEMA_VERSION,
    GeoparsingFormalDescription,
    GeoparsingPrediction,
    load_geoparsing_predictions,
)
from .ocr import image_sha256


_SOLID_KINDS = frozenset(
    {"pyramid", "prism", "cube", "frustum", "cylinder", "cone", "frustumcone", "spheroid"}
)


@dataclass(frozen=True)
class GeoparsingImportDiagnostic:
    code: str
    field: str
    index: int
    raw: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "field": self.field,
            "index": self.index,
            "raw": self.raw,
            "message": self.message,
        }


@dataclass(frozen=True)
class GeoparsingImportResult:
    prediction: ExternalParseGraphPrediction
    diagnostics: tuple[GeoparsingImportDiagnostic, ...]

    @property
    def complete(self) -> bool:
        return not self.diagnostics


@dataclass(frozen=True)
class GeoparsingCompilationResult:
    graph: ParseGraph
    diagnostics: tuple[GeoparsingImportDiagnostic, ...]

    @property
    def complete(self) -> bool:
        return not self.diagnostics


class GeoparsingAdapter:
    """Translate the published Geoparsing language without inventing visual grounding."""

    def __init__(self, *, strict: bool = True):
        self.strict = strict

    def convert(
        self,
        parser_input: ParserInput | Mapping[str, Any],
        prediction: GeoparsingPrediction,
    ) -> GeoparsingImportResult:
        public_input = validate_parser_input(parser_input)
        if prediction.id != public_input.id:
            raise InvalidGeometryInput("Geoparsing prediction id does not match ParserInput")
        if prediction.parser_input_checksum != stable_checksum(public_input.to_dict()):
            raise InvalidGeometryInput("Geoparsing prediction ParserInput checksum does not match")
        if prediction.image_sha256 != image_sha256(public_input.image):
            raise InvalidGeometryInput("Geoparsing prediction image SHA-256 does not match")
        compiled = compile_geoparsing_formal_description(
            prediction.formal_description,
            source_id=public_input.id,
            confidence=prediction.confidence,
            source_checksum=prediction.checksum(),
        )
        graph = compiled.graph
        diagnostics = compiled.diagnostics
        if self.strict and diagnostics:
            first = diagnostics[0]
            raise InvalidGeometryInput(
                f"Geoparsing import produced {len(diagnostics)} diagnostic(s); "
                f"first={first.field}[{first.index}] {first.code}: {first.message}"
            )
        external = ExternalParseGraphPrediction.create(
            public_input,
            backend=prediction.backend,
            model=prediction.model,
            model_revision=prediction.model_revision,
            prompt_version=prediction.prompt_version,
            parse_graph=graph,
            request_id=prediction.request_id,
            decoding=prediction.decoding,
            usage=prediction.usage,
        )
        return GeoparsingImportResult(external, diagnostics)


def compile_geoparsing_formal_description(
    description: GeoparsingFormalDescription | Mapping[str, Any],
    *,
    source_id: str,
    confidence: float | None = None,
    source_checksum: str | None = None,
) -> GeoparsingCompilationResult:
    """Compile the public formal language without reading image bytes or private data."""

    formal = (
        description
        if isinstance(description, GeoparsingFormalDescription)
        else GeoparsingFormalDescription.from_dict(description)
    )
    checksum = source_checksum or stable_checksum(
        {"source_id": source_id, "formal_description": formal.to_dict()}
    )
    builder = _GraphBuilder(
        source_id=source_id,
        description=formal,
        confidence=confidence,
        source_checksum=checksum,
        source_kind="prediction" if source_checksum is not None else "formal_description",
    )
    graph, diagnostics = builder.build()
    return GeoparsingCompilationResult(graph, diagnostics)


class _GraphBuilder:
    def __init__(
        self,
        *,
        source_id: str,
        description: GeoparsingFormalDescription,
        confidence: float | None,
        source_checksum: str,
        source_kind: str,
    ):
        self.source_id = source_id
        self.description = description
        self.source_checksum = source_checksum
        self.source_kind = source_kind
        self.confidence = float(confidence or 0.0)
        self.score_kind = (
            "external_sequence_confidence"
            if confidence is not None
            else "unscored_external_sequence"
        )
        self.evidence: list[ParseEvidence] = []
        self.objects: list[ParseObject] = []
        self.relations: list[ParseRelation] = []
        self.quantities: list[ParseQuantity] = []
        self.diagnostics: list[GeoparsingImportDiagnostic] = []
        self.point_ids: dict[str, str] = {}
        self.named_line_ids: dict[str, str] = {}
        self.segment_ids: dict[tuple[str, str], str] = {}
        self.circle_points: dict[str, frozenset[str]] = {}

    def build(self) -> tuple[ParseGraph, tuple[GeoparsingImportDiagnostic, ...]]:
        self._add_points()
        self._add_lines()
        self._add_circles()
        self._add_planes()
        self._add_structures()
        self._add_semantics()
        graph = ParseGraph(
            id=f"geoparsing:{_digest(self.source_checksum)}",
            source_id=self.source_id,
            evidence=tuple(self.evidence),
            objects=tuple(self.objects),
            relations=tuple(self.relations),
            quantities=tuple(self.quantities),
            metadata={
                "producer": GEOPARSING_ADAPTER_VERSION,
                "upstream_schema": "GDP-29K unified formal language",
                "upstream_source_checksum": self.source_checksum,
                "upstream_source_kind": self.source_kind,
                "confidence_available": self.score_kind == "external_sequence_confidence",
                "import_complete": not self.diagnostics,
                "import_diagnostics": [item.to_dict() for item in self.diagnostics],
                "grounding_scope": "global_image_only",
            },
        )
        return graph, tuple(self.diagnostics)

    def _add_points(self) -> None:
        for index, raw in enumerate(self.description.points):
            name = raw.strip()
            evidence_id = self._add_evidence("points", index, raw)
            point_id = _item_id("point", index, raw)
            self.point_ids[name] = point_id
            self.objects.append(
                ParseObject(
                    id=point_id,
                    kind=ParseObjectKind.POINT,
                    origin=ParseOrigin.IMAGE,
                    confidence=self.confidence,
                    score_kind=self.score_kind,
                    evidence_refs=(evidence_id,),
                    visible_name=name,
                    visible=True,
                    metadata={"upstream_field": "points", "upstream_index": index},
                )
            )

    def _add_lines(self) -> None:
        pattern = re.compile(r"^line\s+(?:(\S+)\s+lieson\s+)?(.+)$", re.IGNORECASE)
        for index, raw in enumerate(self.description.lines):
            match = pattern.fullmatch(_normalize_space(raw))
            if match is None:
                self._diagnose("invalid_line_syntax", "lines", index, raw, "Expected 'line A B' or 'line k lieson A B'")
                continue
            name, point_text = match.groups()
            point_names = tuple(point_text.split())
            point_ids = self._known_points(point_names, "lines", index, raw, minimum=2)
            if point_ids is None:
                continue
            evidence_id = self._add_evidence("lines", index, raw)
            line_id = _item_id("line", index, raw)
            if name is not None:
                if name in self.named_line_ids:
                    self._diagnose("duplicate_named_line", "lines", index, raw, f"Named line {name!r} is repeated")
                    continue
                self.named_line_ids[name] = line_id
            self.objects.append(
                ParseObject(
                    id=line_id,
                    kind=ParseObjectKind.LINE,
                    origin=ParseOrigin.IMAGE,
                    confidence=self.confidence,
                    score_kind=self.score_kind,
                    references=tuple(ParseReference(f"point_{offset}", point_id) for offset, point_id in enumerate(point_ids)),
                    evidence_refs=(evidence_id,),
                    visible_name=name,
                    visible=True,
                    metadata={"upstream_field": "lines", "upstream_index": index},
                )
            )
            self.relations.append(
                ParseRelation(
                    id=_item_id("ordered", index, raw),
                    predicate="ordered_points",
                    arguments=(
                        ParseReference("support", line_id),
                        *(ParseReference(f"point_{offset}", point_id) for offset, point_id in enumerate(point_ids)),
                    ),
                    origin=ParseOrigin.IMAGE,
                    confidence=self.confidence,
                    score_kind=self.score_kind,
                    evidence_refs=(evidence_id,),
                    metadata={"upstream_field": "lines", "upstream_index": index},
                )
            )

    def _add_circles(self) -> None:
        pattern = re.compile(r"^(?:\\odot|odot)\s+(\S+)\s+lieson\s+(.+)$", re.IGNORECASE)
        for index, raw in enumerate(self.description.circles):
            match = pattern.fullmatch(_normalize_space(raw))
            if match is None:
                self._diagnose("invalid_circle_syntax", "circles", index, raw, "Expected '\\odot O lieson A B C'")
                continue
            center_name, point_text = match.groups()
            center_id = self.point_ids.get(center_name)
            point_names = tuple(point_text.split())
            point_ids = self._known_points(point_names, "circles", index, raw, minimum=1)
            if center_id is None or point_ids is None:
                if center_id is None:
                    self._diagnose("unknown_point", "circles", index, raw, f"Unknown circle center {center_name!r}")
                continue
            evidence_id = self._add_evidence("circles", index, raw)
            circle_id = _item_id("circle", index, raw)
            self.circle_points[circle_id] = frozenset(point_names)
            self.objects.append(
                ParseObject(
                    id=circle_id,
                    kind=ParseObjectKind.CIRCLE,
                    origin=ParseOrigin.IMAGE,
                    confidence=self.confidence,
                    score_kind=self.score_kind,
                    references=(
                        ParseReference("center", center_id),
                        *(ParseReference(f"point_{offset}", point_id) for offset, point_id in enumerate(point_ids)),
                    ),
                    evidence_refs=(evidence_id,),
                    visible_name=center_name,
                    visible=True,
                    metadata={"upstream_field": "circles", "upstream_index": index},
                )
            )
            for offset, point_id in enumerate(point_ids):
                self.relations.append(
                    ParseRelation(
                        id=_item_id(f"on_circle_{offset}", index, raw),
                        predicate="point_on_circle",
                        arguments=(ParseReference("point", point_id), ParseReference("circle", circle_id)),
                        origin=ParseOrigin.IMAGE,
                        confidence=self.confidence,
                        score_kind=self.score_kind,
                        evidence_refs=(evidence_id,),
                        metadata={"upstream_field": "circles", "upstream_index": index},
                    )
                )

    def _add_planes(self) -> None:
        pattern = re.compile(r"^plane\s+(.+)$", re.IGNORECASE)
        for index, raw in enumerate(self.description.planes):
            match = pattern.fullmatch(_normalize_space(raw))
            if match is None:
                self._diagnose("invalid_plane_syntax", "planes", index, raw, "Expected 'plane A B C'")
                continue
            point_names = tuple(match.group(1).split())
            point_ids = self._known_points(point_names, "planes", index, raw, minimum=3)
            if point_ids is None:
                continue
            evidence_id = self._add_evidence("planes", index, raw)
            self.objects.append(
                ParseObject(
                    id=_item_id("plane", index, raw),
                    kind=ParseObjectKind.PLANE,
                    origin=ParseOrigin.IMAGE,
                    confidence=self.confidence,
                    score_kind=self.score_kind,
                    references=tuple(ParseReference(f"point_{offset}", point_id) for offset, point_id in enumerate(point_ids)),
                    evidence_refs=(evidence_id,),
                    visible=True,
                    metadata={"upstream_field": "planes", "upstream_index": index},
                )
            )

    def _add_structures(self) -> None:
        pattern = re.compile(r"^(\S+)\s+(.+)$")
        for index, raw in enumerate(self.description.structure):
            match = pattern.fullmatch(_normalize_space(raw))
            if match is None or match.group(1).lower() not in _SOLID_KINDS:
                self._diagnose("unsupported_solid_structure", "structure", index, raw, "Unknown or malformed solid structure")
                continue
            kind, notation = match.groups()
            point_names = _split_structure_notation(notation, tuple(self.point_ids))
            if point_names is None:
                self._diagnose("ambiguous_structure_vertices", "structure", index, raw, "Could not uniquely resolve structure vertices")
                continue
            evidence_id = self._add_evidence("structure", index, raw)
            self.objects.append(
                ParseObject(
                    id=_item_id("solid", index, raw),
                    kind=ParseObjectKind.POLYHEDRON,
                    origin=ParseOrigin.IMAGE,
                    confidence=self.confidence,
                    score_kind=self.score_kind,
                    references=tuple(
                        ParseReference(f"vertex_{offset}", self.point_ids[name])
                        for offset, name in enumerate(point_names)
                    ),
                    evidence_refs=(evidence_id,),
                    visible=True,
                    metadata={
                        "upstream_field": "structure",
                        "upstream_index": index,
                        "solid_kind": kind.lower(),
                        "vertex_notation": notation,
                    },
                )
            )

    def _add_semantics(self) -> None:
        for index, raw in enumerate(self.description.semantics):
            normalized = _normalize_space(raw)
            if "\\perp" in normalized:
                self._add_perpendicular(index, raw, normalized)
            elif "\\parallel" in normalized:
                self._add_parallel(index, raw, normalized)
            elif "=" in normalized:
                self._add_equality(index, raw, normalized)
            else:
                self._diagnose("unsupported_semantic", "semantics", index, raw, "No supported relation operator found")

    def _add_perpendicular(self, index: int, raw: str, normalized: str) -> None:
        match = re.fullmatch(r"(.+?)\s+\\perp\s+(.+?)\s+on\s+(\S+)", normalized)
        if match is None:
            self._diagnose("invalid_perpendicular_syntax", "semantics", index, raw, "Expected 'AB \\perp CD on E'")
            return
        evidence_id = self._add_evidence("semantics", index, raw)
        left = self._geometric_reference(match.group(1), evidence_id, index, raw)
        right = self._geometric_reference(match.group(2), evidence_id, index, raw)
        intersection = self.point_ids.get(match.group(3))
        if left is None or right is None or intersection is None:
            if intersection is None:
                self._diagnose("unknown_point", "semantics", index, raw, f"Unknown intersection point {match.group(3)!r}")
            return
        self.relations.extend(
            (
                ParseRelation(
                    id=_item_id("perpendicular", index, raw),
                    predicate="perpendicular",
                    arguments=(ParseReference("left", left), ParseReference("right", right)),
                    origin=ParseOrigin.IMAGE,
                    confidence=self.confidence,
                    score_kind=self.score_kind,
                    evidence_refs=(evidence_id,),
                    metadata={"upstream_field": "semantics", "upstream_index": index},
                ),
                ParseRelation(
                    id=_item_id("intersection", index, raw),
                    predicate="intersection",
                    arguments=(
                        ParseReference("support", left),
                        ParseReference("support", right),
                        ParseReference("point_0", intersection),
                    ),
                    origin=ParseOrigin.IMAGE,
                    confidence=self.confidence,
                    score_kind=self.score_kind,
                    evidence_refs=(evidence_id,),
                    metadata={"upstream_field": "semantics", "upstream_index": index},
                ),
            )
        )

    def _add_parallel(self, index: int, raw: str, normalized: str) -> None:
        evidence_id = self._add_evidence("semantics", index, raw)
        terms = tuple(item.strip() for item in normalized.split("\\parallel"))
        refs = tuple(self._geometric_reference(term, evidence_id, index, raw) for term in terms)
        if len(refs) < 2 or any(ref is None for ref in refs):
            if len(refs) < 2:
                self._diagnose("invalid_parallel_syntax", "semantics", index, raw, "Parallel relation requires at least two terms")
            return
        for offset, (left, right) in enumerate(zip(refs, refs[1:])):
            self.relations.append(
                ParseRelation(
                    id=_item_id(f"parallel_{offset}", index, raw),
                    predicate="parallel",
                    arguments=(ParseReference("left", left), ParseReference("right", right)),
                    origin=ParseOrigin.IMAGE,
                    confidence=self.confidence,
                    score_kind=self.score_kind,
                    evidence_refs=(evidence_id,),
                    metadata={"upstream_field": "semantics", "upstream_index": index},
                )
            )

    def _add_equality(self, index: int, raw: str, normalized: str) -> None:
        evidence_id = self._add_evidence("semantics", index, raw)
        terms = tuple(item.strip() for item in normalized.split("="))
        if len(terms) < 2 or any(not term for term in terms):
            self._diagnose("invalid_equality_syntax", "semantics", index, raw, "Equality requires at least two non-empty terms")
            return
        if all(_is_angle_term(term) for term in terms[:-1]):
            self._add_angle_equality(index, raw, terms, evidence_id)
            return
        if all(_is_arc_term(term) for term in terms[:-1]):
            self._add_arc_equality(index, raw, terms, evidence_id)
            return
        self._add_segment_equality(index, raw, terms, evidence_id)

    def _add_segment_equality(self, index: int, raw: str, terms: tuple[str, ...], evidence_id: str) -> None:
        segment_refs: list[str] = []
        final_expression = None
        for offset, term in enumerate(terms):
            ref = self._segment_reference(term, evidence_id, index, raw, diagnose=False)
            if ref is not None:
                segment_refs.append(ref)
                continue
            if offset != len(terms) - 1:
                self._diagnose("unsupported_equality_term", "semantics", index, raw, f"Could not resolve segment term {term!r}")
                return
            final_expression = self._expression(term, index, raw)
            if final_expression is None:
                return
        if not segment_refs:
            self._diagnose("unsupported_equality", "semantics", index, raw, "No segment term was resolved")
            return
        for offset, (left, right) in enumerate(zip(segment_refs, segment_refs[1:])):
            self.relations.append(
                ParseRelation(
                    id=_item_id(f"equal_length_{offset}", index, raw),
                    predicate="equal_length",
                    arguments=(ParseReference("left", left), ParseReference("right", right)),
                    origin=ParseOrigin.IMAGE,
                    confidence=self.confidence,
                    score_kind=self.score_kind,
                    evidence_refs=(evidence_id,),
                    metadata={"upstream_field": "semantics", "upstream_index": index},
                )
            )
        if final_expression is not None:
            for offset, segment_id in enumerate(segment_refs):
                self.quantities.append(
                    ParseQuantity(
                        id=_item_id(f"length_{offset}", index, raw),
                        kind="length",
                        arguments=(ParseReference("object", segment_id),),
                        expression=final_expression,
                        origin=ParseOrigin.IMAGE,
                        confidence=self.confidence,
                        score_kind=self.score_kind,
                        raw_text=terms[-1],
                        evidence_refs=(evidence_id,),
                        metadata={"upstream_field": "semantics", "upstream_index": index},
                    )
                )

    def _add_angle_equality(self, index: int, raw: str, terms: tuple[str, ...], evidence_id: str) -> None:
        angles = [self._angle_references(term, evidence_id, index, raw) for term in terms[:-1]]
        if any(angle is None for angle in angles):
            return
        final_angle = self._angle_references(terms[-1], evidence_id, index, raw) if _is_angle_term(terms[-1]) else None
        if final_angle is not None:
            angles.append(final_angle)
            expression = None
        else:
            expression = self._expression(terms[-1], index, raw)
            if expression is None:
                return
        for offset, (left, right) in enumerate(zip(angles, angles[1:])):
            arguments = tuple(
                ParseReference(f"angle_0_{reference.role}", reference.ref) for reference in left
            ) + tuple(ParseReference(f"angle_1_{reference.role}", reference.ref) for reference in right)
            self.relations.append(
                ParseRelation(
                    id=_item_id(f"equal_angle_{offset}", index, raw),
                    predicate="equal_angle",
                    arguments=arguments,
                    origin=ParseOrigin.IMAGE,
                    confidence=self.confidence,
                    score_kind=self.score_kind,
                    evidence_refs=(evidence_id,),
                    metadata={"upstream_field": "semantics", "upstream_index": index},
                )
            )
        if expression is not None:
            for offset, angle in enumerate(angles):
                self.quantities.append(
                    ParseQuantity(
                        id=_item_id(f"angle_{offset}", index, raw),
                        kind="angle",
                        arguments=angle,
                        expression=expression,
                        origin=ParseOrigin.IMAGE,
                        confidence=self.confidence,
                        score_kind=self.score_kind,
                        unit="degree",
                        raw_text=terms[-1],
                        evidence_refs=(evidence_id,),
                        metadata={"upstream_field": "semantics", "upstream_index": index},
                    )
                )

    def _add_arc_equality(self, index: int, raw: str, terms: tuple[str, ...], evidence_id: str) -> None:
        arcs = [self._arc_references(term, index, raw) for term in terms[:-1]]
        if any(arc is None for arc in arcs):
            return
        final_arc = self._arc_references(terms[-1], index, raw) if _is_arc_term(terms[-1]) else None
        if final_arc is not None:
            arcs.append(final_arc)
            expression = None
        else:
            expression = self._expression(terms[-1], index, raw)
            if expression is None:
                return
        if expression is None:
            self._diagnose("unsupported_arc_equality", "semantics", index, raw, "Equal arc measures without a value are not representable in ParseGraph v1")
            return
        for offset, (kind, references, arc_metadata) in enumerate(arcs):
            self.quantities.append(
                ParseQuantity(
                    id=_item_id(f"arc_{offset}", index, raw),
                    kind="arc_length" if kind == "l" else "angle",
                    arguments=references,
                    expression=expression,
                    origin=ParseOrigin.IMAGE,
                    confidence=self.confidence,
                    score_kind=self.score_kind,
                    unit=None if kind == "l" else "degree",
                    raw_text=terms[-1],
                    evidence_refs=(evidence_id,),
                    metadata={
                        "upstream_field": "semantics",
                        "upstream_index": index,
                        "measure_of": "arc",
                        **arc_metadata,
                    },
                )
            )

    def _angle_references(
        self,
        term: str,
        evidence_id: str,
        index: int,
        raw: str,
    ) -> tuple[ParseReference, ...] | None:
        match = re.fullmatch(r"m\s+\\angle\s+(.+)", term)
        if match is None:
            self._diagnose("invalid_angle_syntax", "semantics", index, raw, f"Invalid angle term {term!r}")
            return None
        token = match.group(1).strip()
        names = _split_compact_points(token, tuple(self.point_ids), expected_count=3)
        if names is not None:
            return (
                ParseReference("arm_0", self.point_ids[names[0]]),
                ParseReference("vertex", self.point_ids[names[1]]),
                ParseReference("arm_1", self.point_ids[names[2]]),
            )
        if re.fullmatch(r"[A-Za-z0-9_]+", token):
            object_id = _item_id("symbolic_angle", index, token)
            if not any(item.id == object_id for item in self.objects):
                self.objects.append(
                    ParseObject(
                        id=object_id,
                        kind=ParseObjectKind.TEXT,
                        origin=ParseOrigin.IMAGE,
                        confidence=self.confidence,
                        score_kind=self.score_kind,
                        evidence_refs=(evidence_id,),
                        visible_name=token,
                        visible=True,
                        metadata={"semantic_role": "symbolic_angle"},
                    )
                )
            return (ParseReference("symbol", object_id),)
        self._diagnose("ambiguous_angle_points", "semantics", index, raw, f"Could not uniquely split angle token {token!r}")
        return None

    def _arc_references(
        self,
        term: str,
        index: int,
        raw: str,
    ) -> tuple[str, tuple[ParseReference, ...], dict[str, Any]] | None:
        match = re.fullmatch(r"([ml])\s+(?:(major|minor)\s+)?\\widehat\s*\{?([^{}\s]+)\}?", term)
        if match is None:
            self._diagnose("invalid_arc_syntax", "semantics", index, raw, f"Invalid arc term {term!r}")
            return None
        kind, extent, token = match.groups()
        names = _split_compact_points(token, tuple(self.point_ids), expected_count=2)
        if names is None:
            self._diagnose("ambiguous_arc_points", "semantics", index, raw, f"Could not uniquely split arc token {token!r}")
            return None
        circles = tuple(
            circle_id
            for circle_id, circle_points in self.circle_points.items()
            if set(names).issubset(circle_points)
        )
        if len(circles) != 1:
            code = "arc_circle_unresolved" if not circles else "arc_circle_ambiguous"
            self._diagnose(
                code,
                "semantics",
                index,
                raw,
                f"Arc {token!r} must resolve to exactly one declared circle; candidates={list(circles)}",
            )
            return None
        return (
            kind,
            (
                ParseReference("circle", circles[0]),
                ParseReference("start", self.point_ids[names[0]]),
                ParseReference("end", self.point_ids[names[1]]),
            ),
            {"extent": extent or "unspecified"},
        )

    def _geometric_reference(self, term: str, evidence_id: str, index: int, raw: str) -> str | None:
        normalized = _normalize_space(term)
        if normalized.lower().startswith("line "):
            name = normalized[5:].strip()
            line_id = self.named_line_ids.get(name)
            if line_id is None:
                self._diagnose("unknown_named_line", "semantics", index, raw, f"Unknown named line {name!r}")
            return line_id
        return self._segment_reference(normalized, evidence_id, index, raw, diagnose=True)

    def _segment_reference(
        self,
        token: str,
        evidence_id: str,
        index: int,
        raw: str,
        *,
        diagnose: bool,
    ) -> str | None:
        names = _split_compact_points(token.strip(), tuple(self.point_ids), expected_count=2)
        if names is None:
            if diagnose:
                self._diagnose("ambiguous_segment_points", "semantics", index, raw, f"Could not uniquely split segment token {token!r}")
            return None
        key = tuple(sorted(names))
        existing = self.segment_ids.get(key)
        if existing is not None:
            return existing
        segment_id = f"segment:{_digest('|'.join(key))}"
        self.segment_ids[key] = segment_id
        self.objects.append(
            ParseObject(
                id=segment_id,
                kind=ParseObjectKind.SEGMENT,
                origin=ParseOrigin.IMAGE,
                confidence=self.confidence,
                score_kind=self.score_kind,
                references=(
                    ParseReference("start", self.point_ids[names[0]]),
                    ParseReference("end", self.point_ids[names[1]]),
                ),
                evidence_refs=(evidence_id,),
                visible=True,
                metadata={"endpoint_names": list(names), "created_from_semantic": index},
            )
        )
        return segment_id

    def _expression(self, text: str, index: int, raw: str):
        try:
            return math_expression_from_text(_normalize_mixed_number(text))
        except InvalidGeometryInput as exc:
            self._diagnose("invalid_math_expression", "semantics", index, raw, str(exc))
            return None

    def _known_points(
        self,
        names: tuple[str, ...],
        field_name: str,
        index: int,
        raw: str,
        *,
        minimum: int,
    ) -> tuple[str, ...] | None:
        if len(names) < minimum:
            self._diagnose("too_few_points", field_name, index, raw, f"Expected at least {minimum} points")
            return None
        missing = tuple(name for name in names if name not in self.point_ids)
        if missing:
            self._diagnose("unknown_point", field_name, index, raw, f"Unknown points: {list(missing)}")
            return None
        if len(set(names)) != len(names):
            self._diagnose("duplicate_point_reference", field_name, index, raw, "Point sequence contains duplicates")
            return None
        return tuple(self.point_ids[name] for name in names)

    def _add_evidence(self, field_name: str, index: int, raw: str) -> str:
        evidence_id = _item_id(f"evidence_{field_name}", index, raw)
        self.evidence.append(
            ParseEvidence(
                id=evidence_id,
                source_id=self.source_id,
                kind=ParseEvidenceKind.SOURCE_DOCUMENT,
                precision=EvidencePrecision.IMAGE_ONLY,
                raw_content=raw,
                metadata={
                    "producer": GEOPARSING_ADAPTER_VERSION,
                    "upstream_field": field_name,
                    "upstream_index": index,
                    "grounding_scope": "global_image_only",
                },
            )
        )
        return evidence_id

    def _diagnose(self, code: str, field_name: str, index: int, raw: str, message: str) -> None:
        self.diagnostics.append(GeoparsingImportDiagnostic(code, field_name, index, raw, message))


def _split_compact_points(
    value: str,
    point_names: tuple[str, ...],
    *,
    expected_count: int | None = None,
) -> tuple[str, ...] | None:
    candidates: list[tuple[str, ...]] = []

    def visit(offset: int, parts: tuple[str, ...]) -> None:
        if offset == len(value):
            if expected_count is None or len(parts) == expected_count:
                candidates.append(parts)
            return
        if expected_count is not None and len(parts) >= expected_count:
            return
        for name in sorted(point_names, key=lambda item: (-len(item), item)):
            if value.startswith(name, offset):
                visit(offset + len(name), (*parts, name))

    visit(0, ())
    unique = tuple(dict.fromkeys(candidates))
    return unique[0] if len(unique) == 1 else None


def _split_structure_notation(value: str, point_names: tuple[str, ...]) -> tuple[str, ...] | None:
    parts = []
    for token in value.split("-"):
        resolved = _split_compact_points(token, point_names)
        if resolved is None:
            return None
        parts.extend(resolved)
    return tuple(parts) if parts else None


def _is_angle_term(value: str) -> bool:
    return re.fullmatch(r"m\s+\\angle\s+.+", value) is not None


def _is_arc_term(value: str) -> bool:
    return re.fullmatch(r"[ml]\s+(?:(?:major|minor)\s+)?\\widehat\s*\{?[^{}\s]+\}?", value) is not None


def _normalize_mixed_number(value: str) -> str:
    return re.sub(r"^(\d+)\s*\\frac\{([^{}]+)\}\{([^{}]+)\}$", r"\1+(\2)/(\3)", value.strip())


def _normalize_space(value: str) -> str:
    return " ".join(value.strip().split())


def _item_id(kind: str, index: int, raw: str) -> str:
    return f"gdp:{kind}:{index}:{_digest(raw)}"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
