"""Compact Unicode formatting for exact geometry expressions."""

from __future__ import annotations

import sympy as sp


_SUPERSCRIPTS = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
_SUBSCRIPTS = str.maketrans("0123456789-", "₀₁₂₃₄₅₆₇₈₉₋")


def format_math_expression(value: sp.Expr | str | int) -> str:
    """Format an exact SymPy expression for SVG and Unicode-capable PNG text."""

    return _format(sp.sympify(value))


def _format(expr: sp.Expr) -> str:
    expr = sp.simplify(expr)
    if expr == sp.pi:
        return "π"
    if expr.is_Integer:
        return str(expr).replace("-", "−")
    if expr.is_Rational:
        numerator, denominator = int(expr.p), int(expr.q)
        sign = "−" if numerator < 0 else ""
        return f"{sign}{str(abs(numerator)).translate(_SUPERSCRIPTS)}⁄{str(denominator).translate(_SUBSCRIPTS)}"
    if expr.is_Add:
        parts: list[str] = []
        for index, term in enumerate(expr.as_ordered_terms()):
            negative = term.could_extract_minus_sign()
            text = _format(-term if negative else term)
            if index == 0:
                parts.append(f"−{text}" if negative else text)
            else:
                parts.append(f" {'−' if negative else '+'} {text}")
        return "".join(parts)
    if expr.is_Mul:
        numerator, denominator = sp.fraction(expr)
        if denominator != 1:
            numerator_text = _format(numerator)
            if numerator.is_Add:
                numerator_text = f"({numerator_text})"
            denominator_text = _format(denominator)
            if denominator.is_Integer and int(denominator) >= 0:
                denominator_text = str(denominator).translate(_SUBSCRIPTS)
            return f"{numerator_text}⁄{denominator_text}"
        factors = list(expr.as_ordered_factors())
        negative = factors and factors[0] == -1
        if negative:
            factors = factors[1:]
        rendered = [_format(factor) for factor in factors]
        text = "".join(rendered) if all(_can_juxtapose(factor) for factor in factors) else "·".join(rendered)
        return f"−{text}" if negative else text
    if expr.is_Pow:
        base, exponent = expr.as_base_exp()
        if exponent == sp.Rational(1, 2):
            radicand = _format(base)
            return f"√{radicand}" if base.is_Atom else f"√({radicand})"
        if exponent.is_Integer:
            base_text = _format(base)
            if not base.is_Atom:
                base_text = f"({base_text})"
            return f"{base_text}{str(exponent).translate(_SUPERSCRIPTS)}"
    return str(expr).replace("sqrt", "√").replace("pi", "π").replace("-", "−")


def _can_juxtapose(expr: sp.Expr) -> bool:
    return bool(expr.is_Number or expr.is_Symbol or (expr.is_Pow and expr.exp == sp.Rational(1, 2)))
