"""Construction trace schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class ConstructionOp:
    """One replayable construction operation."""

    op: str
    id: str | None = None
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, item: Dict[str, Any]) -> "ConstructionOp":
        params = dict(item)
        op = str(params.pop("op"))
        object_id = params.pop("id", None)
        return cls(op=op, id=object_id, params=params)

    def to_dict(self) -> Dict[str, Any]:
        data = {"op": self.op}
        if self.id is not None:
            data["id"] = self.id
        data.update(self.params)
        return data


def normalize_trace(trace: Iterable[ConstructionOp | Dict[str, Any]]) -> List[ConstructionOp]:
    ops: List[ConstructionOp] = []
    for item in trace:
        if isinstance(item, ConstructionOp):
            ops.append(item)
        else:
            ops.append(ConstructionOp.from_dict(item))
    return ops
