"""Deterministic Parser v0 closed loop for canonical synthetic records.

This adapter is intentionally synthetic-only. It exercises the observation,
hypothesis, grounding, admission, and debug-state contracts without claiming
to solve open-world image parsing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Mapping
from typing import Any

from geosym_world.schema import EvidenceKind, FormalGeometryState, GeometryHypothesis, GeometryRecord, Provenance
from geosym_world.verifier import AdmissionReport, HypothesisAdmission


@dataclass(frozen=True)
class SyntheticObservation:
    """Typed observation emitted by the deterministic synthetic adapter."""

    sample_id: str
    objects: Mapping[str, tuple[dict[str, Any], ...]]
    relations: tuple[dict[str, Any], ...]
    target: dict[str, Any] | None
    construction_trace: tuple[dict[str, Any], ...]
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "objects": {key: [dict(item) for item in values] for key, values in self.objects.items()},
            "relations": [dict(item) for item in self.relations],
            "target": dict(self.target) if self.target is not None else None,
            "construction_trace": [dict(item) for item in self.construction_trace],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SyntheticParseResult:
    sample_id: str
    observation: SyntheticObservation
    hypothesis: GeometryHypothesis
    record: GeometryRecord
    admission: AdmissionReport | None = None

    @property
    def admitted(self) -> bool:
        return self.admission is not None and self.admission.admitted

    @property
    def admitted_state(self) -> FormalGeometryState | None:
        return self.admission.state if self.admission is not None and self.admission.admitted else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "observation": self.observation.to_dict(),
            "hypothesis": self.hypothesis.to_dict(),
            "record": self.record.to_dict(include_state=False),
            "admission": self.admission.to_dict() if self.admission is not None else None,
            "admitted_state": self.admitted_state.to_dict() if self.admitted_state is not None else None,
        }


class SyntheticParserV0:
    """Emit a proposed hypothesis and optionally run the admission gate."""

    parser_name = "synthetic_parser_v0"

    def __init__(self, *, admission: HypothesisAdmission | None = None):
        self.admission = admission or HypothesisAdmission()

    def parse(self, record: Mapping[str, Any]) -> SyntheticParseResult:
        sample_id = _sample_id(record)
        state = _reference_state(record)
        trace = _trace(record, state)
        observation = _observation(sample_id, state, trace, record)
        payload = {
            "observation": observation.to_dict(),
            "construction_trace": [dict(item) for item in trace],
            "target": dict(observation.target) if observation.target is not None else None,
        }
        hypothesis_id = f"hyp_{sample_id}_synthetic_v0"
        hypothesis = GeometryHypothesis(
            id=hypothesis_id,
            kind="synthetic_geometry_state",
            payload=payload,
            confidence=1.0,
            provenance=(
                Provenance(
                    source_id=sample_id,
                    kind=EvidenceKind.PARSER,
                    event_id=f"{self.parser_name}:observation",
                    metadata={"synthetic_oracle": True, "parser_version": "v0"},
                ),
            ),
            metadata={
                "parser": self.parser_name,
                "synthetic_oracle": True,
                "admission_required": True,
                "reference_state_checksum": state.checksum(),
            },
        )
        parser_record = GeometryRecord(
            id=f"record_{sample_id}_synthetic_v0",
            source={"type": "synthetic_parser", "original_id": sample_id, "parser": self.parser_name},
            state=None,
            hypotheses=[hypothesis],
            metadata={"parser_status": "proposed", "synthetic_oracle": True},
        )
        return SyntheticParseResult(sample_id, observation, hypothesis, parser_record)

    def parse_and_admit(
        self,
        record: Mapping[str, Any],
        *,
        initial_state: FormalGeometryState | None = None,
    ) -> SyntheticParseResult:
        result = self.parse(record)
        admission = self.admission.admit(
            result.hypothesis,
            initial_state=initial_state,
        )
        admitted_record = result.record
        if admission.admitted:
            admitted_record = replace(
                admitted_record,
                state=admission.state,
                hypotheses=[admission.hypothesis],
                metadata={"parser_status": "admitted", "synthetic_oracle": True},
            )
        return replace(result, record=admitted_record, admission=admission)


def _sample_id(record: Mapping[str, Any]) -> str:
    value = record.get("id") or record.get("record_id")
    if not value:
        raise ValueError("Synthetic parser record requires id")
    return str(value)


def _reference_state(record: Mapping[str, Any]) -> FormalGeometryState:
    payload = record.get("formal_state") or record.get("state")
    if not isinstance(payload, Mapping):
        raise ValueError("Synthetic parser record requires formal_state or state")
    return FormalGeometryState.from_dict(payload)


def _trace(record: Mapping[str, Any], state: FormalGeometryState) -> tuple[dict[str, Any], ...]:
    payload = record.get("construction_trace") or state.construction_trace
    if not isinstance(payload, list) or not payload:
        raise ValueError("Synthetic parser record requires non-empty construction_trace")
    return tuple(dict(item) for item in payload)


def _observation(sample_id: str, state: FormalGeometryState, trace: tuple[dict[str, Any], ...], record: Mapping[str, Any]) -> SyntheticObservation:
    buckets = {
        "points": tuple(item.to_dict() for item in state.points.values()),
        "lines": tuple(item.to_dict() for item in state.lines.values()),
        "segments": tuple(item.to_dict() for item in state.segments.values()),
        "rays": tuple(item.to_dict() for item in state.rays.values()),
        "circles": tuple(item.to_dict() for item in state.circles.values()),
        "arcs": tuple(item.to_dict() for item in state.arcs.values()),
        "polygons": tuple(item.to_dict() for item in state.polygons.values()),
        "regions": tuple(item.to_dict() for item in state.regions.values()),
        "composite_regions": tuple(item.to_dict() for item in state.composite_regions.values()),
        "planes": tuple(item.to_dict() for item in state.planes.values()),
        "polyhedra": tuple(item.to_dict() for item in state.polyhedra.values()),
    }
    return SyntheticObservation(
        sample_id=sample_id,
        objects=buckets,
        relations=tuple(dict(item) for item in state.relations),
        target=state.target.to_dict() if state.target else None,
        construction_trace=trace,
        metadata={
            "synthetic_oracle": True,
            "parser_version": "v0",
            "source_record_id": str(record.get("id") or sample_id),
            "answer_excluded": True,
            "facts_excluded": True,
        },
    )
