"""Final-answer submission tool."""

from __future__ import annotations

from typing import Any, Dict

import sympy as sp

from ..feedback import INVALID_ARGUMENT, VALID
from .base import ToolResult


class FinalTool:
    name = "final"

    def execute(self, environment: Any, arguments: Dict[str, Any]) -> ToolResult:
        answer = arguments.get("answer")
        if answer is None:
            return ToolResult(INVALID_ARGUMENT, errors=[{"code": "final.missing_answer"}])
        correct = _answer_equal(str(answer), environment.task.answer, environment.task.answer_type, environment.task.numeric_tolerance)
        environment.terminated = True
        environment.final_answer = str(answer)
        environment.final_correct = correct
        return ToolResult(VALID, result={"answer": str(answer), "correct": correct, "expected": environment.task.answer})


def _answer_equal(candidate: str, expected: str, answer_type: str, tolerance: float) -> bool:
    if answer_type in {"choice", "text"}:
        return candidate.strip().lower() == expected.strip().lower()
    try:
        residual = sp.simplify(sp.sympify(candidate) - sp.sympify(expected))
        return residual == 0 or abs(float(sp.N(residual))) <= tolerance
    except Exception:
        return candidate.strip() == expected.strip()
