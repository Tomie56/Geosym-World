"""
Exact symbolic quantity schema.

SymPy is the source of truth. CAD/OpenCascade values are validators only.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import sympy as sp


def expr_payload(expr: sp.Expr) -> Dict[str, Any]:
    simplified = sp.simplify(expr)
    return {
        "expr": str(simplified),
        "value": float(simplified.evalf()),
    }


def quantity_payload(expr: sp.Expr, unit: str = "", role: str = "answer") -> Dict[str, Any]:
    payload = expr_payload(expr)
    payload.update({
        "source": "sympy",
        "validator": None,
        "cad_value": None,
        "abs_error": None,
        "unit": unit,
        "role": role,
    })
    return payload


def attach_cad_validation(quantity: Dict[str, Any], cad_value: Optional[float],
                          validator: str = "opencascade",
                          tolerance: float = 1e-6) -> Dict[str, Any]:
    if cad_value is None:
        return quantity
    quantity["validator"] = validator
    quantity["cad_value"] = float(cad_value)
    quantity["abs_error"] = abs(float(quantity["value"]) - float(cad_value))
    quantity["validated"] = quantity["abs_error"] <= tolerance
    return quantity
