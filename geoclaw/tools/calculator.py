"""Restricted symbolic calculator tool."""

from __future__ import annotations

from typing import Any, Dict

import sympy as sp

from ..feedback import INVALID_ARGUMENT, VALID
from .base import ToolResult


_ALLOWED_LOCALS = {
    "pi": sp.pi,
    "E": sp.E,
    "sqrt": sp.sqrt,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "Abs": sp.Abs,
    "Rational": sp.Rational,
}


class CalculatorTool:
    name = "calculate"

    def execute(self, environment: Any, arguments: Dict[str, Any]) -> ToolResult:
        expression = arguments.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            return ToolResult(INVALID_ARGUMENT, errors=[{"code": "calculator.missing_expression"}])
        if len(expression) > 256 or "__" in expression or any(token in expression for token in ("import", "lambda", "exec", "eval", "open(")):
            return ToolResult(INVALID_ARGUMENT, errors=[{"code": "calculator.unsafe_expression"}])
        try:
            value = sp.simplify(sp.sympify(expression, locals=_ALLOWED_LOCALS, evaluate=True))
            if value.free_symbols:
                return ToolResult(INVALID_ARGUMENT, errors=[{"code": "calculator.unbound_symbols", "symbols": sorted(str(item) for item in value.free_symbols)}])
            numeric = float(sp.N(value))
        except Exception as exc:
            return ToolResult(INVALID_ARGUMENT, errors=[{"code": "calculator.parse_failed", "message": str(exc)}])
        provenance = _calculation_provenance(environment, arguments)
        return ToolResult(VALID, result={"expression": expression, "exact_value": str(value), "numeric_value": numeric, "provenance": provenance}, audit={"unsupported_geometry_expression": not provenance["geometry_supported"]})


def _calculation_provenance(environment: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
    fact_ids = arguments.get("fact_ids", [])
    if not isinstance(fact_ids, list):
        fact_ids = []
    known = {fact.id: fact for fact in environment.facts}
    valid_ids = [str(item) for item in fact_ids if str(item) in known]
    missing_ids = [str(item) for item in fact_ids if str(item) not in known]
    return {
        "fact_ids": valid_ids,
        "missing_fact_ids": missing_ids,
        "geometry_supported": bool(valid_ids) and not missing_ids,
        "supporting_facts": [known[fact_id].to_dict() for fact_id in valid_ids],
    }
