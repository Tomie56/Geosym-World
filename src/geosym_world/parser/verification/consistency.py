"""Answer consistency verification."""

from __future__ import annotations

from typing import Any, Dict

from geosym_world.benchmark.answer_grader import AnswerGrader


class ConsistencyVerifier:
    def __init__(self):
        self.grader = AnswerGrader()

    def verify_answer(self, solver_answer: Any, original_answer: Any) -> Dict[str, Any]:
        result = self.grader.grade(solver_answer, original_answer)
        return {
            "valid": result["correct"],
            "solver_answer": result["prediction"],
            "original_answer": result["ground_truth"],
            "grade": result,
        }
