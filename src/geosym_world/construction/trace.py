"""Step-level construction supervision for replayable geometry traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from geosym_world.schema.state import FormalGeometryState
from geosym_world.schema.transitions import StateTransition

from .builder import ConstructionBuilder
from .operations import ConstructionOp, normalize_trace, operation_references


@dataclass(frozen=True)
class ConstructionTraceStep:
    index: int
    operation: dict[str, Any]
    input_references: tuple[dict[str, str], ...]
    transition: StateTransition

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "operation": dict(self.operation),
            "input_references": [dict(item) for item in self.input_references],
            "transition": self.transition.to_dict(),
        }


def record_construction_steps(
    trace: Iterable[ConstructionOp | Mapping[str, Any]],
    *,
    initial_state: FormalGeometryState | None = None,
) -> list[dict[str, Any]]:
    """Replay a trace once and emit auditable state transitions per operation."""

    seed = FormalGeometryState.from_dict(initial_state.to_dict()) if initial_state is not None else FormalGeometryState()
    builder = ConstructionBuilder(seed)
    steps: list[dict[str, Any]] = []
    for index, operation in enumerate(normalize_trace(trace)):
        before = FormalGeometryState.from_dict(builder.state.to_dict())
        builder.execute_op(operation)
        after = builder.state
        transition = StateTransition.from_states(
            before,
            after,
            event_id=f"construction_step:{index}",
            provenance={"operation": operation.op, "step_index": index},
        )
        step = ConstructionTraceStep(
            index=index,
            operation=operation.to_dict(),
            input_references=tuple({"kind": kind, "id": reference} for kind, reference in operation_references(operation)),
            transition=transition,
        )
        steps.append(step.to_dict())
    return steps
