"""Parser wrapper for deterministic visual and recorded semantic fusion."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from geosym_world.parser.adapters.hypothesis_record import HypothesisRecordAdapter
from geosym_world.schema import ParseGraph

from .contracts import ParserInput, validate_parser_input
from .evidence_fusion import (
    PARSE_GRAPH_COMPATIBILITY_VERSION,
    PARSE_GRAPH_FUSION_VERSION,
    ParseGraphEvidenceFusion,
)
from .typed_v0 import TypedParseResult


class TypedParser(Protocol):
    def parse(self, data: ParserInput | Mapping[str, Any]) -> TypedParseResult: ...


class FusedParseGraphParser:
    """Run visual and semantic parsers, then fuse their proposed ParseGraphs."""

    parser_name = "fused_parse_graph_v0"

    def __init__(
        self,
        *,
        visual_parser: TypedParser,
        semantic_parser: TypedParser,
        fusion: ParseGraphEvidenceFusion | None = None,
    ):
        self.visual_parser = visual_parser
        self.semantic_parser = semantic_parser
        self.fusion = fusion or ParseGraphEvidenceFusion()
        self.record_adapter = HypothesisRecordAdapter()

    def configuration(self) -> dict[str, Any]:
        return {
            "kind": self.parser_name,
            "fusion_version": PARSE_GRAPH_FUSION_VERSION,
            "compatibility_version": PARSE_GRAPH_COMPATIBILITY_VERSION,
            "visual_parser": _parser_configuration(self.visual_parser),
            "semantic_parser": _parser_configuration(self.semantic_parser),
        }

    def parse(self, data: ParserInput | Mapping[str, Any]) -> TypedParseResult:
        parser_input = validate_parser_input(data)
        visual = self.visual_parser.parse(parser_input)
        semantic = self.semantic_parser.parse(parser_input)
        result = self.fusion.fuse(visual.graph, semantic.graph)
        graph = result.graph
        stages = (
            {
                "name": "visual_parse_graph",
                "status": "complete",
                "output_count": _candidate_count(visual.graph),
                "metadata": {"graph_checksum": visual.graph.checksum()},
            },
            {
                "name": "semantic_parse_graph",
                "status": "complete",
                "output_count": _candidate_count(semantic.graph),
                "metadata": {"graph_checksum": semantic.graph.checksum()},
            },
            {
                "name": "evidence_fusion",
                "status": "complete",
                "output_count": _candidate_count(graph),
                "metadata": {
                    "status_counts": result.status_counts(),
                    "compatibility_status_counts": result.compatibility.status_counts(),
                },
            },
            {"name": "admission", "status": "pending", "output_count": 0, "metadata": {}},
        )
        record = self.record_adapter.convert_parse_graph(
            sample_id=parser_input.id,
            dataset_name="parser_benchmark",
            parser_name=self.parser_name,
            backend="deterministic_visual_plus_recorded_semantic",
            graph=graph,
            confidence=_graph_confidence(graph),
            verification={
                "status": "not_admitted",
                "reason": "fused_parse_graph_requires_independent_admission",
            },
            image_path=parser_input.image,
            metadata={
                "fusion_version": PARSE_GRAPH_FUSION_VERSION,
                "compatibility_version": result.compatibility.version,
                "stage_count": len(stages),
            },
        )
        return TypedParseResult(parser_input=parser_input, graph=graph, record=record, stages=stages)


def _candidate_count(graph: ParseGraph) -> int:
    return sum(
        len(items)
        for items in (graph.objects, graph.relations, graph.quantities, graph.regions, graph.targets)
    )


def _parser_configuration(parser: TypedParser) -> dict[str, Any]:
    configuration = getattr(parser, "configuration", None)
    if callable(configuration):
        value = configuration()
        return dict(value) if isinstance(value, Mapping) else {"kind": str(value)}
    payload: dict[str, Any] = {
        "kind": getattr(parser, "parser_name", type(parser).__name__),
    }
    label_recognizer = getattr(parser, "label_recognizer", None)
    label_configuration = getattr(label_recognizer, "configuration", None)
    if callable(label_configuration):
        payload["ocr"] = dict(label_configuration())
    return payload


def _graph_confidence(graph: ParseGraph) -> float:
    candidates = (*graph.objects, *graph.relations, *graph.quantities, *graph.regions, *graph.targets)
    if not candidates:
        return 0.0
    return sum(float(candidate.confidence) for candidate in candidates) / len(candidates)
