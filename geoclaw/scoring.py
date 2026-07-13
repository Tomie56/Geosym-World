"""Behavioral metrics for solver-grounded geometry tool trajectories."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable


DEFAULT_TOOL_COSTS = {"zoom": 1.0, "calculate": 1.0, "measure": 1.0, "construct": 1.5, "verify": 1.0, "final": 0.0}


def score_trajectory(task: Any, steps: Iterable[Any], *, tool_costs: Dict[str, float] | None = None, recovery_window: int = 3) -> Dict[str, Any]:
    records = [_step_dict(step) for step in steps]
    tool_steps = [item for item in records if item["action"] != "final"]
    valid_steps = [item for item in tool_steps if item["status"] == "valid"]
    used_tools = {item["action"] for item in tool_steps}
    selection = _tool_selection(task.required_tool_sets, used_tools)
    schema_valid = [item for item in tool_steps if item["status"] != "invalid_schema"]
    argument_valid = [item for item in tool_steps if item["status"] not in {"invalid_schema", "invalid_argument", "unknown_object"}]
    verified = [item for item in tool_steps if item["action"] == "verify" and item["result"].get("truth") in {"true", "false", "unknown"}]
    verified_true = [item for item in verified if item["result"].get("truth") == "true"]
    useful = [item for item in valid_steps if item["utility"] == "progress"]
    final = next((item for item in reversed(records) if item["action"] == "final"), None)
    costs = tool_costs or DEFAULT_TOOL_COSTS
    actual_cost = sum(costs.get(item["action"], 1.0) for item in records)
    reference_cost = task.reference_cost
    efficiency = None if not reference_cost or actual_cost <= 0 else min(1.0, float(reference_cost) / actual_cost)
    recovery = _recovery_metrics(tool_steps, recovery_window)
    calculator_steps = [item for item in tool_steps if item["action"] == "calculate"]
    parsed_calculations = [item for item in calculator_steps if item["status"] == "valid"]
    supported_calculations = [item for item in parsed_calculations if item["result"].get("provenance", {}).get("geometry_supported")]
    zoom_steps = [item for item in valid_steps if item["action"] == "zoom"]
    zoom_ious = [item["result"].get("best_region_iou") for item in zoom_steps if item["result"].get("best_region_iou") is not None]
    required_zoom = [item for item in zoom_steps if item["result"].get("required_region_covered") is not None]
    covered_zoom = [item for item in required_zoom if item["result"].get("required_region_covered")]
    measure_steps = [item for item in valid_steps if item["action"] == "measure"]
    visual_measure_steps = [item for item in measure_steps if item["result"].get("mode") == "image"]
    construct_steps = [item for item in tool_steps if item["action"] == "construct"]
    valid_construct_steps = [item for item in construct_steps if item["status"] == "valid"]
    useful_construct_steps = [item for item in valid_construct_steps if item["utility"] == "progress"]
    evidence_counts = Counter(item.get("evidence_strength", "unknown") for item in records if item["action"] == "verify")
    measurement_residuals = [abs(float(item["audit"]["visual_solver_residual"])) for item in measure_steps if item.get("audit", {}).get("visual_solver_residual") is not None]
    unsupported_calculations = [item for item in parsed_calculations if item.get("audit", {}).get("unsupported_geometry_expression")]
    final_correct = bool(final and final["result"].get("correct"))
    return {
        "final_accuracy": float(bool(final and final["result"].get("correct"))),
        "tool_selection_precision": selection["precision"],
        "tool_selection_recall": selection["recall"],
        "tool_selection_f1": selection["f1"],
        "matched_required_tool_set": selection["matched_set"],
        "schema_validity": _ratio(len(schema_valid), len(tool_steps)),
        "argument_validity": _ratio(len(argument_valid), len(tool_steps)),
        "invalid_action_rate": _ratio(len(tool_steps) - len(valid_steps), len(tool_steps)),
        "verified_step_ratio": _ratio(len(verified_true), len(verified)),
        "useful_action_rate": _ratio(len(useful), len(valid_steps)),
        "tool_efficiency": efficiency,
        "actual_tool_cost": actual_cost,
        "recovery_rate": recovery["rate"],
        "num_failure_events": recovery["num_failures"],
        "num_recovered_events": recovery["num_recovered"],
        "tool_counts": dict(Counter(item["action"] for item in records)),
        "expression_validity": _optional_ratio(len(parsed_calculations), len(calculator_steps)),
        "geometry_expression_support_rate": _optional_ratio(len(supported_calculations), len(parsed_calculations)),
        "zoom_grounding_iou": _mean(zoom_ious),
        "required_region_coverage_rate": _optional_ratio(len(covered_zoom), len(required_zoom)),
        "visual_measurement_rate": _optional_ratio(len(visual_measure_steps), len(measure_steps)),
        "construction_validity": _optional_ratio(len(valid_construct_steps), len(construct_steps)),
        "construction_utility": _optional_ratio(len(useful_construct_steps), len(valid_construct_steps)),
        "evidence_strength_counts": dict(evidence_counts),
        "visual_solver_residual_mean": _mean(measurement_residuals),
        "measurement_reliance_error": None if not visual_measure_steps else float(not final_correct),
        "calculation_masking_rate": None if not parsed_calculations else (_ratio(len(unsupported_calculations), len(parsed_calculations)) if final_correct else 0.0),
    }


def aggregate_scores(scores: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    items = list(scores)
    keys = ["final_accuracy", "tool_selection_f1", "argument_validity", "invalid_action_rate", "verified_step_ratio", "useful_action_rate", "tool_efficiency", "recovery_rate", "expression_validity", "geometry_expression_support_rate", "zoom_grounding_iou", "required_region_coverage_rate", "visual_measurement_rate", "construction_validity", "construction_utility", "visual_solver_residual_mean", "measurement_reliance_error", "calculation_masking_rate"]
    return {key: _mean([item[key] for item in items if item.get(key) is not None]) for key in keys} | {"num_episodes": len(items)}


def feedback_gain(visible_scores: Iterable[Dict[str, Any]], hidden_scores: Iterable[Dict[str, Any]]) -> Dict[str, float | None]:
    visible = aggregate_scores(visible_scores)
    hidden = aggregate_scores(hidden_scores)
    return {
        "final_accuracy_gain": _difference(visible.get("final_accuracy"), hidden.get("final_accuracy")),
        "recovery_rate_gain": _difference(visible.get("recovery_rate"), hidden.get("recovery_rate")),
        "invalid_action_rate_change": _difference(visible.get("invalid_action_rate"), hidden.get("invalid_action_rate")),
        "useful_action_rate_gain": _difference(visible.get("useful_action_rate"), hidden.get("useful_action_rate")),
    }


def _tool_selection(required_sets: Iterable[Iterable[str]], used: set[str]) -> Dict[str, Any]:
    candidates = [set(items) for items in required_sets] or [set()]
    scored = []
    for required in candidates:
        overlap = len(required & used)
        precision = _ratio(overlap, len(used))
        recall = _ratio(overlap, len(required))
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        scored.append((f1, precision, recall, required))
    f1, precision, recall, required = max(scored, key=lambda item: item[0])
    return {"f1": f1, "precision": precision, "recall": recall, "matched_set": sorted(required)}


def _recovery_metrics(steps: list[Dict[str, Any]], window: int) -> Dict[str, Any]:
    failure_indices = []
    previous_failed = False
    for index, step in enumerate(steps):
        failed = step["status"] != "valid"
        if failed and not previous_failed:
            failure_indices.append(index)
        previous_failed = failed
    recovered = 0
    for index in failure_indices:
        subsequent = steps[index + 1:index + 1 + window]
        if any(item["status"] == "valid" and item["utility"] == "progress" for item in subsequent):
            recovered += 1
    return {"num_failures": len(failure_indices), "num_recovered": recovered, "rate": _ratio(recovered, len(failure_indices))}


def _step_dict(step: Any) -> Dict[str, Any]:
    payload = step.to_dict() if hasattr(step, "to_dict") else dict(step)
    feedback = payload.get("feedback", payload)
    parsed = payload.get("parsed_action") or payload.get("action") or {}
    action = parsed.get("action") if isinstance(parsed, dict) else parsed
    return {"action": action, "status": feedback.get("status"), "utility": feedback.get("utility"), "result": feedback.get("result", {}), "evidence_strength": feedback.get("evidence_strength", feedback.get("result", {}).get("evidence_strength", "unknown")), "audit": payload.get("audit", {})}


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def _optional_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else float(numerator) / float(denominator)


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _difference(left: Any, right: Any) -> float | None:
    return None if left is None or right is None else float(left) - float(right)
