"""Geometry fact closure solver."""

from .closure import FactClosureSolver
from .incremental import IncrementalFactClosure, IncrementalSolveReport

__all__ = ["FactClosureSolver", "IncrementalFactClosure", "IncrementalSolveReport"]
