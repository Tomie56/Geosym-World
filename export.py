"""Export helpers for pilot benchmark records."""

from __future__ import annotations

from typing import Any, Dict

from .derivation import DerivationTracer
from .fact import FactStore
from .state import FormalGeometryState
from .verifier import VerificationReport


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
