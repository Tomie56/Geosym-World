"""Export helpers for pilot benchmark records."""

from __future__ import annotations

from typing import Any, Dict

from geosym_world.schema.derivations import DerivationTracer
from geosym_world.schema.facts import FactStore
from geosym_world.schema.state import FormalGeometryState
from geosym_world.verifier.trace import VerificationReport


def export_record(state: FormalGeometryState, facts: FactStore, *, record_id: str, question: str = "", answer: str = "",
                  verification: VerificationReport | None = None) -> Dict[str, Any]:
    return {
        "id": record_id,
        "question": question,
        "answer": answer,
        "formal_state": state.to_dict(),
        "facts": facts.to_list(),
        "derivation": DerivationTracer().trace_target(facts),
        "verification": verification.to_dict() if verification else None,
    }
