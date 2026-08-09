"""Adapter from legacy GeoSym synthetic JSON/QA into GeoSymProblemState."""

from __future__ import annotations

from typing import Any, Dict, Optional
import hashlib

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


class SyntheticAdapter:
    """Wrap old 2D GeoSym generated data without changing the old pipeline."""

    def convert(
        self,
        geo_json: Dict[str, Any],
        qa_item: Optional[Dict[str, Any]] = None,
        image_path: Optional[str] = None,
        problem_id: Optional[str] = None,
    ) -> GeoSymProblemState:
        qa_item = qa_item or {}
        problem_id = problem_id or qa_item.get("id") or self._stable_id(geo_json, qa_item)
        question_type = qa_item.get("question_type") or qa_item.get("task_type") or "unknown"
        target = self._target_from_qa(qa_item)
        geometry = GeoSymGeometry(
            points=list(geo_json.get("points") or []),
            lines=list(geo_json.get("lines") or []),
            arcs=list(geo_json.get("arcs") or []),
            entities=list(geo_json.get("entities") or []),
            relations=list(geo_json.get("relations") or []),
            quantities=list(geo_json.get("quantities") or []),
            raw={
                "description": geo_json.get("description", ""),
                "legacy": geo_json,
            },
        )
        proof = GeoSymProof(
            known_conditions=list(geo_json.get("known_conditions") or []),
            construction_trace=list(
                geo_json.get("construction_trace")
                or geo_json.get("enhancement_history")
                or []
            ),
            derivation_trace=list(geo_json.get("derivation_trace") or []),
            required_steps=list(geo_json.get("required_steps") or []),
        )
        diagnostic = GeoSymDiagnostic(
            task_type=question_type,
            difficulty=self._difficulty(geo_json, qa_item),
            requires=list(qa_item.get("requires") or []),
        )
        return GeoSymProblemState(
            problem_id=problem_id,
            source=GeoSymSource(
                type="synthetic",
                dataset="geosym",
                generator="legacy_template_builder",
                original_id=geo_json.get("id") or qa_item.get("geometry_id"),
            ),
            input=GeoSymInput(
                image_path=image_path,
                problem_text=qa_item.get("question", ""),
                choices=qa_item.get("choices"),
                solution_text=qa_item.get("solution"),
                metadata={"description": geo_json.get("description", "")},
            ),
            geometry=geometry,
            target=target,
            proof=proof,
            diagnostic=diagnostic,
            metadata={"legacy_qa": qa_item},
        )

    def _target_from_qa(self, qa_item: Dict[str, Any]) -> Optional[GeoSymTarget]:
        if not qa_item:
            return None
        qtype = qa_item.get("question_type") or qa_item.get("task_type") or "unknown"
        target_object = qa_item.get("target_id") or qa_item.get("target_quantity")
        answer = qa_item.get("gt") or qa_item.get("answer")
        if isinstance(answer, (str, int, float)):
            answer = expr_payload(answer)
        return GeoSymTarget(
            type=qtype,
            object=target_object,
            answer=answer,
            metadata={"qt_description": qa_item.get("qt_description")},
        )

    def _difficulty(self, geo_json: Dict[str, Any], qa_item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "num_points": len(geo_json.get("points") or []),
            "num_lines": len(geo_json.get("lines") or []),
            "num_arcs": len(geo_json.get("arcs") or []),
            "num_entities": len(geo_json.get("entities") or []),
            "graph_complexity": qa_item.get("graph_complexity"),
            "question_complexity": qa_item.get("question_complexity"),
            "answer_complexity": qa_item.get("answer_complexity"),
            "overall_difficulty": qa_item.get("difficulty"),
        }

    def _stable_id(self, geo_json: Dict[str, Any], qa_item: Dict[str, Any]) -> str:
        seed = repr((geo_json.get("description", ""), qa_item.get("question", ""), qa_item.get("answer", "")))
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
        return f"geosym_synth_{digest}"
