"""Final-answer grading with exact, SymPy, and numeric modes."""

from __future__ import annotations

from typing import Any, Dict

from geosym_world.schema.problem_normalize import expressions_equivalent, get_answer_expr


class AnswerGrader:
    def grade(self, prediction: Any, ground_truth: Any) -> Dict[str, Any]:
        pred_expr = get_answer_expr(prediction)
        gt_expr = get_answer_expr(ground_truth)
        if pred_expr is None or gt_expr is None:
            return {
                "correct": False,
                "mode": "missing",
                "prediction": pred_expr,
                "ground_truth": gt_expr,
            }
        exact = str(pred_expr).strip() == str(gt_expr).strip()
        equivalent = expressions_equivalent(pred_expr, gt_expr)
        return {
            "correct": bool(exact or equivalent),
            "exact": exact,
            "equivalent": equivalent,
            "prediction": pred_expr,
            "ground_truth": gt_expr,
            "mode": "exact" if exact else "sympy_or_numeric",
        }
