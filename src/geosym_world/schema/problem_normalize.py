"""Normalization utilities for expressions, ids, and answers."""

from __future__ import annotations

from typing import Any, Dict, Optional
import math
import re

import sympy as sp


MATH_LOCALS = {
    "sqrt": sp.sqrt,
    "pi": sp.pi,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "Rational": sp.Rational,
}


def normalize_id(value: Any) -> str:
    text = str(value or "").strip()
    text = text.replace("Segment ", "").replace("Line ", "").replace("Point ", "")
    text = re.sub(r"[^A-Za-z0-9_]", "", text)
    return text


def line_id(p1: str, p2: str) -> str:
    return f"{normalize_id(p1)}{normalize_id(p2)}"


def preprocess_expr(expr: Any) -> str:
    text = str(expr if expr is not None else "").strip()
    text = text.replace("×", "*").replace("÷", "/").replace("π", "pi")
    text = text.replace("^", "**")
    text = re.sub(r"\\sqrt\s*\{\s*([^{}]+?)\s*\}", r"sqrt(\1)", text)
    text = re.sub(r"(?<=\d)\s*(?=sqrt\s*\()", "*", text)
    text = re.sub(r"(?<=\))\s*(?=sqrt\s*\()", "*", text)
    text = re.sub(r"(?<=\d)\s+(?=[A-Za-z])", "*", text)
    text = text.replace("\\pi", "pi")
    return text


def parse_expr(expr: Any) -> Optional[sp.Expr]:
    text = preprocess_expr(expr)
    if not text:
        return None
    try:
        return sp.sympify(text, locals=MATH_LOCALS)
    except Exception:
        return None


def expr_payload(expr: Any) -> Dict[str, Any]:
    parsed = parse_expr(expr)
    if parsed is None:
        return {"expr": str(expr), "latex": str(expr), "value": None}
    simplified = sp.simplify(parsed)
    try:
        value = float(simplified.evalf())
        if math.isfinite(value):
            value = round(value, 12)
    except Exception:
        value = None
    return {
        "expr": str(simplified),
        "latex": sp.latex(simplified),
        "value": value,
    }


def expressions_equivalent(left: Any, right: Any, tol: float = 1e-6) -> bool:
    l_expr = parse_expr(left)
    r_expr = parse_expr(right)
    if l_expr is not None and r_expr is not None:
        try:
            if sp.simplify(l_expr - r_expr) == 0:
                return True
        except Exception:
            pass
        try:
            return abs(float(l_expr.evalf()) - float(r_expr.evalf())) <= tol
        except Exception:
            return False
    try:
        return abs(float(left) - float(right)) <= tol
    except Exception:
        return str(left).strip() == str(right).strip()


def get_answer_expr(answer: Any) -> Optional[str]:
    if answer is None:
        return None
    if isinstance(answer, dict):
        if "expr" in answer:
            return str(answer["expr"])
        if "value" in answer:
            return str(answer["value"])
    return str(answer)


def make_point(pid: str, x: Any, y: Any, **extra: Any) -> Dict[str, Any]:
    payload = {
        "id": normalize_id(pid),
        "x": expr_payload(x),
        "y": expr_payload(y),
    }
    payload.update(extra)
    return payload


def make_line(line: str, start: str, end: str, **extra: Any) -> Dict[str, Any]:
    payload = {
        "id": normalize_id(line),
        "start_point_id": normalize_id(start),
        "end_point_id": normalize_id(end),
    }
    payload.update(extra)
    return payload
