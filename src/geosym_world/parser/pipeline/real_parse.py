"""Real-data parse, build, verify, and export pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import json
import traceback

from geosym_world.parser.adapters import HypothesisRecordAdapter, RealDataAdapter
from geosym_world.schema.problem_verifier import GeoSymVerifier
from geosym_world.datasets.layout import DataLayout, default_parsed_root, default_raw_root
from geosym_world.parser import ConstraintNormalizer, RelationExtractor
from geosym_world.parser.visualization import ReviewImageComposer, SketchRenderer
from geosym_world.parser.verification import ConsistencyVerifier, ImageAlignmentVerifier, LLMJudgeSignal, merge_hard_and_llm_signals
from geosym_world.parser.verification.llm_judge import OpenRouterLLMJudge


class RealParsePipeline:
    def __init__(
        self,
        dataset_name: str,
        raw_root: str | Path | None = None,
        parsed_root: str | Path | None = None,
        extractor_backend: str = "mock",
        model: Optional[str] = None,
        judge_backend: str = "none",
        judge_model: Optional[str] = None,
    ):
        self.layout = DataLayout(
            dataset_name,
            Path(raw_root) if raw_root is not None else default_raw_root(),
            Path(parsed_root) if parsed_root is not None else default_parsed_root(),
        ).ensure()
        self.extractor = RelationExtractor(backend=extractor_backend, model=model)
        self.normalizer = ConstraintNormalizer()
        self.adapter = RealDataAdapter()
        self.record_adapter = HypothesisRecordAdapter()
        self.verifier = GeoSymVerifier()
        self.renderer = SketchRenderer()
        self.review_composer = ReviewImageComposer()
        self.alignment = ImageAlignmentVerifier()
        self.consistency = ConsistencyVerifier()
        self.judge_backend = judge_backend
        self.judge = OpenRouterLLMJudge(model=judge_model) if judge_backend == "openrouter" else None

    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        sample_id = str(record.get("id") or record.get("problem_id") or "sample")
        image_path = record.get("image_path") or record.get("image")
        problem_text = record.get("problem_text") or record.get("question") or record.get("problem") or ""
        solution_text = record.get("solution")
        answer = record.get("answer")
        manual_json = record.get("relations") or record.get("extraction")

        extraction = self.extractor.extract(
            image_path=image_path,
            problem_text=problem_text,
            solution_text=solution_text,
            manual_json=manual_json,
        )
        constraint_graph = self.normalizer.normalize(extraction)
        if answer is not None and "answer" not in constraint_graph:
            constraint_graph["answer"] = answer
        state = self.adapter.convert(
            constraint_graph,
            image_path=image_path,
            problem_text=problem_text,
            solution_text=solution_text,
            answer=answer,
            problem_id=sample_id,
        )
        report = self.verifier.verify(state)
        state.verification = report

        extracted_path = self.layout.extracted_dir / f"{sample_id}.json"
        constraints_path = self.layout.constraints_dir / f"{sample_id}.json"
        record_path = self.layout.records_dir / f"{sample_id}.json"
        sketch_path = self.layout.sketches_dir / f"{sample_id}.png"
        review_path = self.layout.review_images_dir / f"{sample_id}.png"
        report_path = self.layout.reports_dir / f"{sample_id}.json"

        self._write_json(extracted_path, extraction)
        self._write_json(constraints_path, constraint_graph)

        image_report = None
        sketch_rendered = False
        if (state.geometry.raw or {}).get("unsupported_for_canonical_build"):
            report.add_issue(
                "image.sketch_not_rendered",
                "Canonical sketch not rendered because the relation graph is unsupported for canonical build",
                severity="warning",
            )
            if image_path and Path(str(image_path)).exists():
                extraction_lines = []
                extraction = constraint_graph.get("extraction") or {}
                extraction_lines.extend(str(item) for item in extraction.get("relations_text") or [])
                extraction_lines.extend(str(item) for item in extraction.get("uncertain_relations_text") or [])
                if not extraction_lines:
                    extraction_lines = [str((state.geometry.raw or {}).get("build_error", "unsupported"))]
                try:
                    self.review_composer.compose_text_panel(str(image_path), extraction_lines, str(review_path), title=sample_id)
                except Exception as exc:
                    report.add_issue(
                        "image.review_render_failed",
                        f"Review image render failed: {exc}",
                        severity="warning",
                    )
        else:
            try:
                self.renderer.render(state, str(sketch_path))
                sketch_rendered = True
            except Exception as exc:
                report.add_issue(
                    "image.sketch_render_failed",
                    f"Canonical sketch render failed: {exc}",
                    severity="warning",
                )
        if sketch_rendered and image_path and Path(str(image_path)).exists():
            try:
                image_report = self.alignment.verify_paths(str(image_path), str(sketch_path))
            except Exception as exc:
                image_report = {
                    "valid": False,
                    "error": f"image alignment failed: {exc}",
                }
            try:
                self.review_composer.compose(str(image_path), str(sketch_path), str(review_path), title=sample_id)
            except Exception as exc:
                report.add_issue(
                    "image.review_render_failed",
                    f"Review image render failed: {exc}",
                    severity="warning",
                )
        consistency_report = None
        canonical_build_ok = not (state.geometry.raw or {}).get("unsupported_for_canonical_build")
        if canonical_build_ok and answer is not None and state.target and state.target.answer:
            consistency_report = self.consistency.verify_answer(answer, state.target.answer)
        judge = LLMJudgeSignal(used=False)
        if self.judge is not None:
            judge = self.judge.judge(
                image_path=str(image_path) if image_path else None,
                problem_text=problem_text,
                original_answer=answer,
                canonical_state=state.to_dict(),
                verification_report=report.to_dict(),
            )
        hard_valid = report.valid
        if image_report is not None:
            hard_valid = hard_valid and image_report.get("valid", False)
        if consistency_report is not None:
            hard_valid = hard_valid and consistency_report.get("valid", False)
        merged = merge_hard_and_llm_signals(hard_valid, judge)
        final_report = {
            "verification": report.to_dict(),
            "image_alignment": image_report,
            "consistency": consistency_report,
            "llm_judge": judge.to_dict(),
            "merged_decision": merged,
            "paths": {
                "extracted": str(extracted_path),
                "constraints": str(constraints_path),
                "record": str(record_path),
                "sketch": str(sketch_path),
                "review_image": str(review_path),
                "report": str(report_path),
            },
        }
        parser_record = self.record_adapter.convert(
            sample_id=sample_id,
            dataset_name=self.layout.dataset_name,
            parser_name="real_parse",
            backend=self.extractor.backend,
            candidate_kind="canonical_problem_state_candidate",
            candidate_payload=state.to_dict(),
            confidence=self._candidate_confidence(manual_json),
            verification=final_report,
            image_path=str(image_path) if image_path else None,
            metadata={"recommended_decision": merged},
        )
        self._write_json(record_path, parser_record.to_dict())
        self._write_json(report_path, final_report)
        return final_report

    def write_error_report(self, record: Dict[str, Any], exc: Exception) -> Dict[str, Any]:
        sample_id = str(record.get("id") or record.get("problem_id") or "sample")
        report_path = self.layout.reports_dir / f"{sample_id}.json"
        extracted_path = self.layout.extracted_dir / f"{sample_id}.json"
        constraints_path = self.layout.constraints_dir / f"{sample_id}.json"
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }
        self._write_json(extracted_path, {"error": error, "record": record})
        self._write_json(constraints_path, {"error": error})
        final_report = {
            "verification": {
                "schema_valid": False,
                "geometry_valid": False,
                "quantity_valid": False,
                "derivation_valid": False,
                "image_alignment_valid": None,
                "answer_valid": None,
                "llm_judge_valid": None,
                "valid": False,
                "issues": [
                    {
                        "severity": "error",
                        "code": "pipeline.record_failed",
                        "message": str(exc),
                        "location": sample_id,
                        "metadata": {"exception_type": type(exc).__name__},
                    }
                ],
                "metrics": {},
            },
            "image_alignment": None,
            "consistency": None,
            "llm_judge": LLMJudgeSignal(used=False).to_dict(),
            "merged_decision": "reject_or_repair",
            "error": error,
            "paths": {
                "extracted": str(extracted_path),
                "constraints": str(constraints_path),
                "state": None,
                "sketch": None,
                "review_image": None,
                "report": str(report_path),
            },
        }
        self._write_json(report_path, final_report)
        return final_report

    def _write_json(self, path: Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _candidate_confidence(self, manual_json: Any) -> float:
        if manual_json is not None:
            return 0.95
        if self.extractor.backend == "openrouter":
            return 0.7
        return 0.5
