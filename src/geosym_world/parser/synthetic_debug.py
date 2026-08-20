"""Debug and evaluation helpers for the deterministic synthetic Parser v0."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from geosym_world.benchmark.parser_evaluator import ParserStateEvaluator
from geosym_world.parser.synthetic_v0 import SyntheticParseResult


class SyntheticParserDebugger:
    """Produce component metrics without using answer/fact leakage."""

    def __init__(self, evaluator: ParserStateEvaluator | None = None):
        self.evaluator = evaluator or ParserStateEvaluator()

    def diagnose(
        self,
        result: SyntheticParseResult,
        reference: Mapping[str, Any],
    ) -> dict[str, Any]:
        predicted = _observation_state(result)
        metrics = self.evaluator.evaluate(predicted, reference)
        admission = result.admission.to_dict() if result.admission is not None else None
        return {
            "sample_id": result.sample_id,
            "parser": "synthetic_parser_v0",
            "metrics": metrics,
            "admission": admission,
            "hypothesis_status": result.hypothesis.status.value,
            "reference_state_checksum": result.hypothesis.metadata.get("reference_state_checksum"),
            "admitted_state_checksum": result.admitted_state.checksum() if result.admitted_state is not None else None,
            "debug_axes": {
                "grounding": admission["grounding"] if admission else None,
                "verification": admission["verification"] if admission else None,
                "transition": admission["transition"] if admission else None,
            },
        }


def _observation_state(result: SyntheticParseResult) -> dict[str, Any]:
    observation = result.observation.to_dict()
    return {
        "geometry": {
            **observation["objects"],
            "relations": observation["relations"],
            "quantities": [],
        },
        "target": observation["target"],
    }
