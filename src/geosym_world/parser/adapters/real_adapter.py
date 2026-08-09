"""Adapter from real-problem constraints into GeoSymProblemState."""

from __future__ import annotations

from typing import Any, Dict, Optional
import hashlib

from geosym_world.parser.adapters.canonical_builder import CanonicalBuildError, CanonicalBuilder
from geosym_world.schema.problem_normalize import expr_payload
from geosym_world.schema.problem import (
    GeoSymDiagnostic,
    GeoSymGeometry,
    GeoSymInput,
    GeoSymProblemState,
    GeoSymProof,
    GeoSymSource,
    GeoSymTarget,
)


class RealDataAdapter:
    """Build a problem state from normalized real-data constraints."""

    def __init__(self, builder: Optional[CanonicalBuilder] = None):
        self.builder = builder or CanonicalBuilder()

    def convert(
        self,
        constraint_graph: Dict[str, Any],
        image_path: Optional[str] = None,
        problem_text: str = "",
        solution_text: Optional[str] = None,
        answer: Optional[Any] = None,
        problem_id: Optional[str] = None,
    ) -> GeoSymProblemState:
        target_data = constraint_graph.get("target") or {}
        answer_value = answer if answer is not None else constraint_graph.get("answer")
        target = None
        if self._has_usable_target(target_data):
            target = GeoSymTarget(
                type=target_data.get("type", "unknown"),
                object=target_data.get("object"),
                answer=expr_payload(answer_value) if answer_value is not None else None,
            )
        problem_id = problem_id or self._stable_id(constraint_graph, problem_text)
        try:
            geometry_dict = self.builder.build(constraint_graph)
            build_error = None
        except CanonicalBuildError as exc:
            geometry_dict = {
                "points": [],
                "lines": [],
                "arcs": [],
                "entities": [],
                "relations": [],
                "quantities": [],
                "raw": {
                    "constraint_graph": constraint_graph,
                    "build_error": str(exc),
                    "unsupported_for_canonical_build": True,
                },
            }
            build_error = str(exc)
        return GeoSymProblemState(
            problem_id=problem_id,
            source=GeoSymSource(
                type="real",
                dataset=constraint_graph.get("dataset"),
                original_id=constraint_graph.get("original_id"),
                metadata={
                    "extraction": constraint_graph.get("extraction"),
                    "raw_constraint_graph": constraint_graph,
                },
            ),
            input=GeoSymInput(
                image_path=image_path,
                problem_text=problem_text,
                solution_text=solution_text,
                metadata={"raw_answer": answer_value},
            ),
            geometry=GeoSymGeometry.from_dict(geometry_dict),
            target=target,
            proof=GeoSymProof(
                known_conditions=list(constraint_graph.get("constraints") or []),
                construction_trace=list(constraint_graph.get("construction_trace") or []),
                derivation_trace=list(constraint_graph.get("solution_trace") or []),
            ),
            diagnostic=GeoSymDiagnostic(
                task_type=target.type if target else "unknown",
                requires=list(constraint_graph.get("requires") or []),
                metadata={"build_error": build_error} if build_error else {},
            ),
        )

    def _has_usable_target(self, target_data: Dict[str, Any]) -> bool:
        if not isinstance(target_data, dict) or not target_data:
            return False
        target_type = str(target_data.get("type") or "").strip().lower()
        obj = target_data.get("object")
        if not target_type or target_type == "unknown":
            return False
        if obj in {None, ""}:
            return False
        return True

    def _stable_id(self, constraint_graph: Dict[str, Any], problem_text: str) -> str:
        seed = repr((constraint_graph, problem_text))
        return "geosym_real_" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
