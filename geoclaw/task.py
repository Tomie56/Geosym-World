"""Task schema for solver-grounded geometry tool use."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable

import sympy as sp
from geosym_world_geometry.builder import ConstructionBuilder
from geosym_world_geometry.solver import FactClosureSolver

from .schema import ACTION_NAMES, DifficultyProfile, VisualRegion


@dataclass(frozen=True)
class GeoClawTask:
    id: str
    problem_text: str
    trace: tuple[Dict[str, Any], ...]
    answer: str
    answer_type: str = "symbolic"
    original_image: str | None = None
    required_tool_sets: tuple[tuple[str, ...], ...] = ()
    acceptable_intermediate_facts: tuple[Dict[str, Any], ...] = ()
    reference_trace: tuple[Dict[str, Any], ...] = ()
    reference_cost: float | None = None
    max_steps: int = 12
    numeric_tolerance: float = 1e-6
    visual_regions: tuple[VisualRegion, ...] = ()
    difficulty: DifficultyProfile = field(default_factory=DifficultyProfile)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GeoClawTask":
        return cls(
            id=str(payload["id"]),
            problem_text=str(payload["problem_text"]),
            trace=tuple(dict(item) for item in payload["trace"]),
            answer=str(payload["answer"]),
            answer_type=str(payload.get("answer_type", "symbolic")),
            original_image=payload.get("original_image"),
            required_tool_sets=tuple(tuple(items) for items in payload.get("required_tool_sets", ())),
            acceptable_intermediate_facts=tuple(dict(item) for item in payload.get("acceptable_intermediate_facts", ())),
            reference_trace=tuple(dict(item) for item in payload.get("reference_trace", ())),
            reference_cost=payload.get("reference_cost"),
            max_steps=int(payload.get("max_steps", 12)),
            numeric_tolerance=float(payload.get("numeric_tolerance", 1e-6)),
            visual_regions=tuple(VisualRegion.from_dict(item) for item in payload.get("visual_regions", ())),
            difficulty=DifficultyProfile.from_dict(payload.get("difficulty")),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "problem_text": self.problem_text,
            "trace": list(self.trace),
            "answer": self.answer,
            "answer_type": self.answer_type,
            "original_image": self.original_image,
            "required_tool_sets": [list(items) for items in self.required_tool_sets],
            "acceptable_intermediate_facts": list(self.acceptable_intermediate_facts),
            "reference_trace": list(self.reference_trace),
            "reference_cost": self.reference_cost,
            "max_steps": self.max_steps,
            "numeric_tolerance": self.numeric_tolerance,
            "visual_regions": [item.to_dict() for item in self.visual_regions],
            "difficulty": self.difficulty.to_dict(),
            "metadata": self.metadata,
        }

    def validate(self, *, execute: bool = True) -> list[Dict[str, Any]]:
        issues: list[Dict[str, Any]] = []
        if not self.id.strip():
            issues.append({"code": "task.empty_id"})
        if not self.problem_text.strip():
            issues.append({"code": "task.empty_problem"})
        if not self.trace:
            issues.append({"code": "task.empty_trace"})
        if self.max_steps <= 0:
            issues.append({"code": "task.invalid_max_steps", "value": self.max_steps})
        if self.numeric_tolerance < 0:
            issues.append({"code": "task.invalid_tolerance", "value": self.numeric_tolerance})
        for index, tool_set in enumerate(self.required_tool_sets):
            unknown = sorted(set(tool_set) - set(ACTION_NAMES))
            if unknown:
                issues.append({"code": "task.unknown_required_tool", "index": index, "tools": unknown})
        for field_name in ("visual_complexity", "construction_depth", "theorem_depth", "symbolic_growth"):
            value = getattr(self.difficulty, field_name)
            if value < 0:
                issues.append({"code": "task.invalid_difficulty", "field": field_name, "value": value})
        if execute and self.trace:
            try:
                state = ConstructionBuilder().execute(self.trace)
                facts = FactClosureSolver().solve(state)
                if state.target is None:
                    issues.append({"code": "task.missing_target"})
                else:
                    target_facts = facts.find("target", state.target.object)
                    if not target_facts:
                        issues.append({"code": "task.unresolved_target", "object": state.target.object})
                    elif self.answer_type not in {"choice", "text"} and target_facts[0].value is not None:
                        try:
                            residual = sp.simplify(sp.sympify(target_facts[0].value) - sp.sympify(self.answer))
                            if residual != 0 and abs(float(sp.N(residual))) > self.numeric_tolerance:
                                issues.append({"code": "task.answer_target_mismatch", "target_value": target_facts[0].value, "answer": self.answer})
                        except Exception:
                            issues.append({"code": "task.answer_parse_failed", "target_value": target_facts[0].value, "answer": self.answer})
            except Exception as exc:
                issues.append({"code": "task.trace_execution_failed", "message": str(exc)})
        return issues

    def require_valid(self, *, execute: bool = True) -> None:
        issues = self.validate(execute=execute)
        if issues:
            raise ValueError(f"Invalid GeoClaw task {self.id}: {issues}")


def task_collection(items: Iterable[Dict[str, Any]]) -> list[GeoClawTask]:
    return [GeoClawTask.from_dict(item) for item in items]
