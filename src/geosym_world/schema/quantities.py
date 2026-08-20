"""Exact and approximate quantity values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from geosym_world.errors import InvalidGeometryInput, NumericalInstabilityError


@dataclass(frozen=True)
class QuantityValue:
    exact: str | None = None
    numeric: float | None = None
    unit: str | None = None
    tolerance: float | None = None

    def __post_init__(self) -> None:
        if self.exact is None and self.numeric is None:
            raise InvalidGeometryInput("A quantity needs an exact or numeric value")
        if self.exact is not None and not isinstance(self.exact, str):
            raise InvalidGeometryInput("Quantity exact value must be a string")
        if self.numeric is not None:
            if not isinstance(self.numeric, (int, float)):
                raise InvalidGeometryInput("Quantity numeric value must be a number")
            if self.numeric != self.numeric or self.numeric in {float("inf"), float("-inf")}:
                raise NumericalInstabilityError("Quantity numeric value must be finite")
        if self.tolerance is not None and self.tolerance < 0:
            raise InvalidGeometryInput("Quantity tolerance cannot be negative")
        if self.tolerance is not None and self.tolerance != self.tolerance:
            raise NumericalInstabilityError("Quantity tolerance must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "exact": self.exact,
            "numeric": self.numeric,
            "unit": self.unit,
            "tolerance": self.tolerance,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QuantityValue":
        return cls(
            exact=str(data["exact"]) if data.get("exact") is not None else None,
            numeric=float(data["numeric"]) if data.get("numeric") is not None else None,
            unit=str(data["unit"]) if data.get("unit") is not None else None,
            tolerance=float(data["tolerance"]) if data.get("tolerance") is not None else None,
        )
