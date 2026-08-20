"""Deterministic Parser benchmark slice aggregation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


class ParserBenchmarkSlicer:
    """Aggregate parser diagnostics by reproducible metadata axes."""

    def summarize(self, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        total = 0
        for row in rows:
            if not isinstance(row, Mapping):
                raise TypeError("Parser benchmark rows must be mappings")
            total += 1
            for key in self._slice_keys(row):
                buckets[key].append(row)
        slices = {key: _summarize_rows(value) for key, value in sorted(buckets.items())}
        return {"num_rows": total, "num_slices": len(slices), "slices": slices}

    def _slice_keys(self, row: Mapping[str, Any]) -> tuple[str, ...]:
        keys = ["all"]
        for field in ("parser", "task_type", "difficulty"):
            value = row.get(field)
            if value not in {None, ""}:
                keys.append(f"{field}={value}")
        axes = row.get("diagnostic_axes") or row.get("debug_axes") or ()
        if isinstance(axes, Mapping):
            axes = axes.keys()
        for axis in sorted({str(value) for value in axes if value not in {None, ""}}):
            keys.append(f"diagnostic_axis={axis}")
        return tuple(keys)


def _summarize_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    scores = [_macro_score(row) for row in rows if _macro_score(row) is not None]
    admitted = [row for row in rows if _admission_decision(row) == "admitted"]
    return {
        "num_rows": len(rows),
        "macro_score_mean": sum(scores) / len(scores) if scores else None,
        "macro_score_min": min(scores) if scores else None,
        "macro_score_max": max(scores) if scores else None,
        "num_scored": len(scores),
        "num_admitted": len(admitted),
        "admission_rate": len(admitted) / len(rows) if rows else 0.0,
    }


def _macro_score(row: Mapping[str, Any]) -> float | None:
    metrics = row.get("metrics")
    value = metrics.get("macro_score") if isinstance(metrics, Mapping) else row.get("macro_score")
    return float(value) if value is not None else None


def _admission_decision(row: Mapping[str, Any]) -> str | None:
    admission = row.get("admission")
    if isinstance(admission, Mapping):
        return str(admission.get("decision")) if admission.get("decision") is not None else None
    value = row.get("admission_status")
    return str(value) if value is not None else None
