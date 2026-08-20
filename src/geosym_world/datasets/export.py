"""Export helpers for pilot benchmark records."""

from __future__ import annotations

from typing import Any, Dict

from geosym_world.schema.derivations import DerivationTracer, validate_derivation_payload
from geosym_world.construction.trace import record_construction_steps
from geosym_world.schema.facts import FactStore
from geosym_world.schema.state import FormalGeometryState
from geosym_world.schema.canonical import state_checksum
from geosym_world.verifier.trace import VerificationReport


DATASET_RECORD_SCHEMA_VERSION = "1.0"


def export_record(state: FormalGeometryState, facts: FactStore, *, record_id: str, question: str = "", answer: str = "",
                  verification: VerificationReport | None = None) -> Dict[str, Any]:
    derivation = DerivationTracer().trace_target(facts)
    derivation["validation"] = validate_derivation_payload(derivation).to_dict()
    return {
        "schema_version": DATASET_RECORD_SCHEMA_VERSION,
        "id": record_id,
        "question": question,
        "answer": answer,
        "formal_state": state.to_dict(),
        "formal_state_checksum": state_checksum(state),
        "construction_steps": record_construction_steps(state.construction_trace),
        "facts": facts.to_list(),
        "derivation": derivation,
        "verification": verification.to_dict() if verification else None,
    }
