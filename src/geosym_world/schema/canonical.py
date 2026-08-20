"""Canonical serialization and stable checksums for geometry contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import sympy as sp


def canonicalize(value: Any) -> Any:
    if isinstance(value, sp.Basic):
        return {"__sympy__": sp.srepr(sp.simplify(value))}
    if isinstance(value, Mapping):
        return {str(key): canonicalize(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [canonicalize(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("Canonical JSON does not support non-finite floats")
        return value
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        canonicalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_checksum(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def state_checksum(state: Any) -> str:
    return stable_checksum(state.to_dict())


def record_checksum(record: Any) -> str:
    return stable_checksum(record.to_dict())
