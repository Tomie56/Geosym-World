"""
OpenCascade HLR hidden-line extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import importlib.util


Point2 = Tuple[float, float]
Segment2 = Tuple[Point2, Point2]


@dataclass
class HLRResult:
    backend: str
    available: bool
    visible: List[Segment2]
    hidden: List[Segment2]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "available": self.available,
            "visible_segments": [
                {"start": list(start), "end": list(end)}
                for start, end in self.visible
            ],
            "hidden_segments": [
                {"start": list(start), "end": list(end)}
                for start, end in self.hidden
            ],
            "error": self.error,
        }


class OpenCascadeHLR:
    """Run OpenCascade HLR for orthographic views."""

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("OCP") is not None and importlib.util.find_spec("cadquery") is not None

    def _shape_from_builder(self, solid_builder: Callable[[Any], Any]):
        import cadquery as cq

        solid = solid_builder(cq)
        shape = solid.val() if hasattr(solid, "val") else solid
        return shape.wrapped if hasattr(shape, "wrapped") else shape

    def _project_point(self, p, view: str) -> Point2:
        x, y, z = p.X(), p.Y(), p.Z()
        if view == "front":
            return (x, z)
        if view == "side":
            return (y, z)
        if view == "top":
            return (x, y)
        return (x - 0.54 * y, z + 0.34 * y)

    def _projector_axis(self, view: str):
        from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

        if view == "front":
            return gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0))
        if view == "side":
            return gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0))
        if view == "top":
            return gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
        return gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(1, 1, 1))

    def _segments_from_compound(self, compound, view: str) -> List[Segment2]:
        from OCP.BRepAdaptor import BRepAdaptor_Curve
        from OCP.TopAbs import TopAbs_EDGE
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS
        import math

        segments: List[Segment2] = []
        if compound.IsNull():
            return segments
        exp = TopExp_Explorer(compound, TopAbs_EDGE)
        while exp.More():
            edge = TopoDS.Edge_s(exp.Current())
            curve = BRepAdaptor_Curve(edge)
            first = curve.FirstParameter()
            last = curve.LastParameter()
            if not math.isfinite(first) or not math.isfinite(last):
                exp.Next()
                continue
            samples = 8 if curve.IsClosed() else 2
            pts = []
            for idx in range(samples):
                t = first + (last - first) * idx / max(samples - 1, 1)
                pts.append(self._project_point(curve.Value(t), view))
            for idx in range(len(pts) - 1):
                if pts[idx] != pts[idx + 1]:
                    segments.append((pts[idx], pts[idx + 1]))
            exp.Next()
        return segments

    def extract(self, solid_builder: Callable[[Any], Any], view: str) -> HLRResult:
        if not self.available():
            return HLRResult("opencascade_hlr", False, [], [], "OCP/cadquery is not available")
        try:
            from OCP.HLRAlgo import HLRAlgo_Projector
            from OCP.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape

            shape = self._shape_from_builder(solid_builder)
            algo = HLRBRep_Algo()
            algo.Add(shape)
            algo.Projector(HLRAlgo_Projector(self._projector_axis(view)))
            algo.Update()
            algo.Hide()
            hlr = HLRBRep_HLRToShape(algo)
            visible = self._segments_from_compound(hlr.VCompound(), view)
            hidden = self._segments_from_compound(hlr.HCompound(), view)
            return HLRResult("opencascade_hlr", True, visible, hidden)
        except Exception as exc:
            return HLRResult("opencascade_hlr", False, [], [], repr(exc))
