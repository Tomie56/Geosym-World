"""Deterministic trace mutators for controlled synthetic variation."""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List

import sympy as sp

Trace = List[Dict[str, Any]]


def mutate_trace(trace: Trace, *, seed: int, scale_choices: tuple[int, ...] = (1, 2), translate_choices: tuple[int, ...] = (-1, 0, 1)) -> Trace:
    rng = random.Random(seed)
    scale = rng.choice(scale_choices)
    dx = rng.choice(translate_choices)
    dy = rng.choice(translate_choices)
    dz = rng.choice(translate_choices)
    mutated = copy.deepcopy(trace)
    if scale == 1 and dx == 0 and dy == 0 and dz == 0:
        return mutated
    for op in mutated:
        if op.get("op") in {"add_point", "add_point3d"}:
            coord = op.get("coord") or op.get("coords")
            if coord:
                offsets = [dx, dy, dz][: len(coord)]
                op["coord"] = [str(sp.simplify(sp.sympify(value) * scale + offsets[idx])) for idx, value in enumerate(coord)]
        elif op.get("op") == "add_circle" and "radius" in op:
            op["radius"] = str(sp.simplify(sp.sympify(op["radius"]) * scale))
    return mutated
