"""
OpenCascade/CadQuery numeric validation.

This module deliberately does not produce exact expressions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional
import importlib.util


@dataclass
class CadValidation:
    backend: str
    available: bool
    volume: Optional[float] = None
    surface_area: Optional[float] = None
    step_path: Optional[str] = None
    stl_path: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "role": "numeric_validator_and_renderer_backend",
            "available": self.available,
            "volume": self.volume,
            "surface_area": self.surface_area,
            "step_path": self.step_path,
            "stl_path": self.stl_path,
            "error": self.error,
        }


class CadValidator:
    """CadQuery/OpenCascade mirror backend."""

    def __init__(self, cad_dir: str):
        self.cad_dir = Path(cad_dir)
        self.cad_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("cadquery") is not None

    def build_export_validate(self, scene_id: str, solid_builder: Callable[[Any], Any]) -> CadValidation:
        if not self.available():
            return CadValidation("opencascade/cadquery", False, error="cadquery is not installed")
        try:
            import cadquery as cq

            solid = solid_builder(cq)
            shape = solid.val() if hasattr(solid, "val") else solid
            step_path = self.cad_dir / f"{scene_id}.step"
            stl_path = self.cad_dir / f"{scene_id}.stl"
            cq.exporters.export(solid, str(step_path))
            cq.exporters.export(solid, str(stl_path))
            return CadValidation(
                backend="opencascade/cadquery",
                available=True,
                volume=float(shape.Volume()),
                surface_area=float(shape.Area()),
                step_path=str(step_path),
                stl_path=str(stl_path),
            )
        except Exception as exc:
            return CadValidation("opencascade/cadquery", False, error=repr(exc))
