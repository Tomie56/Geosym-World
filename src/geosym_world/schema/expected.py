"""Expected-fact extraction and comparison helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import sympy as sp


def legacy_expected_facts(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract lightweight expected facts from legacy-like records."""

    expected: List[Dict[str, Any]] = []
    for index, item in enumerate(_iter_items(record.get("entities") or [])):
        object_id = item.get("id") or item.get("name")
        if not object_id and item.get("type") == "shadow":
            object_id = f"legacy_shadow_{index}"
        if not object_id:
            continue
        for fact_type in ("length", "area", "perimeter", "surface_area", "volume", "radius"):
            if fact_type not in item:
                continue
            value = _metric_expr(item[fact_type])
            if value is None:
                continue
            expected.append(_expected_fact(fact_type, str(object_id), value, item, source_path=f"entities[{index}].{fact_type}"))

    for category in ("lines", "segments", "edges", "polygons", "faces", "regions", "polyhedra", "circles"):
        for index, item in enumerate(_iter_items(record.get(category) or [])):
            object_id = item.get("id") or item.get("name")
            if not object_id:
                continue
            for fact_type in ("length", "area", "perimeter", "surface_area", "volume", "radius"):
                if fact_type not in item:
                    continue
                value = _metric_expr(item[fact_type])
                if value is None:
                    continue
                expected.append(_expected_fact(fact_type, str(object_id), value, item, source_path=f"{category}[{index}].{fact_type}"))
    return _dedupe_expected(expected)


def compare_expected_facts(expected_facts: Iterable[Dict[str, Any]], formal_facts: Iterable[Dict[str, Any]], *, numeric_tol: float = 1e-8) -> Dict[str, Any]:
    """Compare expected facts against exported formal facts."""

    formal_by_key: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for fact in formal_facts:
        formal_by_key.setdefault((str(fact.get("type")), str(fact.get("object"))), []).append(fact)

    comparisons = []
    for expected in expected_facts:
        key = (str(expected["type"]), str(expected["object"]))
        candidates = formal_by_key.get(key, [])
        if not candidates:
            comparisons.append({"status": "missing_formal_fact", "method": "key_lookup", "expected": expected, "formal": None})
            continue
        best = candidates[0]
        status, residual, method = _compare_values(expected.get("value"), best.get("value"), expected.get("numeric_value"), best.get("numeric_value"), numeric_tol=numeric_tol)
        comparisons.append(
            {
                "status": status,
                "method": method,
                "expected": expected,
                "formal": best,
                "residual": residual,
            }
        )

    num_expected = len(comparisons)
    num_matched = sum(1 for item in comparisons if item["status"] == "matched")
    num_missing = sum(1 for item in comparisons if item["status"] == "missing_formal_fact")
    num_mismatch = sum(1 for item in comparisons if item["status"] == "value_mismatch")
    num_unchecked = sum(1 for item in comparisons if item["status"] == "unchecked")
    return {
        "num_expected": len(comparisons),
        "num_matched": num_matched,
        "num_missing_formal_fact": num_missing,
        "num_value_mismatch": num_mismatch,
        "num_unchecked": num_unchecked,
        "quality_label": _quality_label(num_expected, num_matched, num_missing, num_mismatch, num_unchecked),
        "method_counts": _method_counts(comparisons),
        "numeric_tolerance": numeric_tol,
        "comparisons": comparisons,
    }


def _expected_fact(fact_type: str, object_id: str, value: str, item: Dict[str, Any], *, source_path: str) -> Dict[str, Any]:
    return {
        "type": fact_type,
        "object": object_id,
        "value": str(sp.simplify(sp.sympify(value))),
        "numeric_value": _numeric(value),
        "source": "legacy_expected",
        "metadata": {
            "source_path": source_path,
            "legacy_type": item.get("type"),
        },
    }


def _compare_values(expected_value: Any, formal_value: Any, expected_numeric: Any = None, formal_numeric: Any = None, *, numeric_tol: float = 1e-8) -> tuple[str, str | None, str]:
    if expected_value is None or formal_value is None:
        return "unchecked", None, "missing_value"
    if str(expected_value) == str(formal_value):
        return "matched", "0", "string_equal"
    if expected_numeric is not None and formal_numeric is not None:
        try:
            residual_float = float(formal_numeric) - float(expected_numeric)
            if abs(residual_float) <= numeric_tol:
                return "matched", str(residual_float), "numeric_tolerance"
        except Exception:
            pass
    try:
        residual_expr = sp.simplify(sp.sympify(formal_value) - sp.sympify(expected_value))
    except Exception:
        return "unchecked", None, "symbolic_parse_failed"
    residual = str(residual_expr)
    if residual_expr == 0:
        return "matched", residual, "symbolic_simplify"
    return "value_mismatch", residual, "symbolic_simplify"


def _quality_label(num_expected: int, num_matched: int, num_missing: int, num_mismatch: int, num_unchecked: int) -> str:
    if num_expected == 0:
        return "no_expected_facts"
    if num_mismatch > 0:
        return "has_value_mismatch"
    if num_missing == 0 and num_unchecked == 0 and num_matched == num_expected:
        return "all_matched"
    if num_missing > 0 and num_matched > 0 and num_mismatch == 0:
        return "partial_missing"
    if num_unchecked > 0 and num_mismatch == 0:
        return "partially_unchecked"
    return "unresolved"


def _method_counts(comparisons: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in comparisons:
        method = str(item.get("method", "unknown"))
        counts[method] = counts.get(method, 0) + 1
    return dict(sorted(counts.items()))


def _metric_expr(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in ("expr", "value", "latex"):
            if key in value and value[key] is not None:
                if key == "latex":
                    continue
                return str(value[key])
        return None
    if value is None:
        return None
    return str(value)


def _numeric(value: Any):
    try:
        return float(sp.N(sp.sympify(value)))
    except Exception:
        return None


def _iter_items(value: Any):
    if isinstance(value, dict):
        for item_id, item in value.items():
            if isinstance(item, dict):
                payload = dict(item)
                payload.setdefault("id", item_id)
                yield payload
        return
    for item in value:
        if isinstance(item, dict):
            yield item


def _dedupe_expected(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for item in items:
        key = (item["type"], item["object"], item["value"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped
