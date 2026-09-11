"""Hybrid real-data parser: script vision + VLM semantics + grounding."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import traceback
from collections.abc import Mapping

from geosym_world.parser.adapters import HypothesisRecordAdapter, RealDataAdapter
from geosym_world.schema.problem_verifier import GeoSymVerifier
from geosym_world.datasets.layout import DataLayout, default_parsed_root, default_raw_root
from geosym_world.parser.hybrid import BoardAdjudicationAnalyzer, BuildInstructionPlanner, BuildPlanCompiler, BuildPlanSeedCoverageAnalyzer, CandidateUsageAnalyzer, EngineAssessmentValidator, EvidenceGraphBuilder, ExtractionContractAuditor, FormalGraphBuilder, GEOSYM_2D_BUILDER_CAPABILITIES, GeometryEvidenceBoardBuilder, GroundingDecisionRepairer, HybridSemanticExtractor, HybridStageClassifier, InstructionCoverageAnalyzer, InstructionRepairer, InstructionTaskPacketBuilder, InstructionTaskPacketCoverageAnalyzer, RelationBuilderDecisionAnalyzer, RelationDraftRepairer, RelationEngineMetadataAnnotator, RelationGrounder, RelationTemplateAdherenceAnalyzer, ResponseTemplateAdherenceAnalyzer, ScriptObservationExtractor, SemanticOutputValidator, SemanticRoleCanonicalizer, StageContractAnalyzer, StageContractRepairer, TextClaimsBuilder
from geosym_world.parser.vision import VisualPrimitiveDetector
from geosym_world.parser.visualization import ReviewImageComposer, SketchRenderer, VisualOverlayRenderer
from geosym_world.parser.verification import ConsistencyVerifier, ImageAlignmentVerifier, LLMJudgeSignal, merge_hard_and_llm_signals
from geosym_world.parser.verification.llm_judge import OpenRouterLLMJudge
from geosym_world.parser.contracts import ParserInput, validate_parser_input


class HybridParsePipeline:
    def __init__(
        self,
        dataset_name: str,
        raw_root: str | Path | None = None,
        parsed_root: str | Path | None = None,
        semantic_backend: str = "openrouter",
        model: Optional[str] = None,
        relation_model: Optional[str] = None,
        instruction_model: Optional[str] = None,
        extraction_mode: str = "single",
        judge_backend: str = "none",
        judge_model: Optional[str] = None,
    ):
        self.layout = DataLayout(
            dataset_name,
            Path(raw_root) if raw_root is not None else default_raw_root(),
            Path(parsed_root) if parsed_root is not None else default_parsed_root(),
        ).ensure()
        self.visual = VisualPrimitiveDetector()
        self.script_observations = ScriptObservationExtractor()
        self.evidence_graph = EvidenceGraphBuilder()
        self.text_claims_builder = TextClaimsBuilder()
        self.evidence_board_builder = GeometryEvidenceBoardBuilder()
        self.board_adjudication = BoardAdjudicationAnalyzer()
        self.relation_builder_decision = RelationBuilderDecisionAnalyzer()
        self.semantic = HybridSemanticExtractor(
            backend=semantic_backend,
            model=model,
            relation_model=relation_model,
            instruction_model=instruction_model,
            extraction_mode=extraction_mode,
        )
        self.semantic_backend = semantic_backend
        self.semantic_model = self.semantic.model
        self.relation_model = self.semantic.relation_model
        self.instruction_model = self.semantic.instruction_model
        self.extraction_mode = extraction_mode
        self.semantic_validator = SemanticOutputValidator()
        self.relation_engine_metadata = RelationEngineMetadataAnnotator()
        self.candidate_usage = CandidateUsageAnalyzer()
        self.instruction_coverage = InstructionCoverageAnalyzer()
        self.instruction_repairer = InstructionRepairer()
        self.stage_contract_repairer = StageContractRepairer()
        self.relation_draft_repairer = RelationDraftRepairer()
        self.grounding_decision_repairer = GroundingDecisionRepairer()
        self.role_canonicalizer = SemanticRoleCanonicalizer()
        self.instruction_task_packet_builder = InstructionTaskPacketBuilder()
        self.instruction_task_packet_coverage = InstructionTaskPacketCoverageAnalyzer()
        self.response_template_adherence = ResponseTemplateAdherenceAnalyzer()
        self.relation_template_adherence = RelationTemplateAdherenceAnalyzer()
        self.stage_contract = StageContractAnalyzer()
        self.contract_auditor = ExtractionContractAuditor()
        self.grounder = RelationGrounder()
        self.formalizer = FormalGraphBuilder()
        self.instruction_planner = BuildInstructionPlanner()
        self.build_plan_compiler = BuildPlanCompiler()
        self.build_plan_seed_coverage = BuildPlanSeedCoverageAnalyzer()
        self.engine_assessment = EngineAssessmentValidator()
        self.stage_classifier = HybridStageClassifier()
        self.adapter = RealDataAdapter()
        self.record_adapter = HypothesisRecordAdapter()
        self.verifier = GeoSymVerifier()
        self.renderer = SketchRenderer()
        self.visual_overlay = VisualOverlayRenderer()
        self.review = ReviewImageComposer()
        self.alignment = ImageAlignmentVerifier()
        self.consistency = ConsistencyVerifier()
        self.judge_backend = judge_backend
        self.judge = OpenRouterLLMJudge(model=judge_model) if judge_backend == "openrouter" else None

    @property
    def visual_dir(self) -> Path:
        return self.layout.visual_state_dir

    @property
    def visual_overlay_dir(self) -> Path:
        return self.layout.visual_overlays_dir

    @property
    def script_observations_dir(self) -> Path:
        return self.layout.script_observations_dir

    @property
    def evidence_graphs_dir(self) -> Path:
        return self.layout.evidence_graphs_dir

    @property
    def text_claims_dir(self) -> Path:
        return self.layout.text_claims_dir

    @property
    def evidence_boards_dir(self) -> Path:
        return self.layout.evidence_boards_dir

    @property
    def semantic_dir(self) -> Path:
        return self.layout.semantic_relations_dir

    @property
    def semantic_raw_dir(self) -> Path:
        return self.layout.semantic_raw_dir

    @property
    def semantic_repaired_dir(self) -> Path:
        return self.layout.semantic_repaired_dir

    @property
    def vlm_passes_dir(self) -> Path:
        return self.layout.vlm_passes_dir

    @property
    def vlm_transcripts_dir(self) -> Path:
        return self.layout.vlm_transcripts_dir

    @property
    def instruction_task_packets_dir(self) -> Path:
        return self.layout.instruction_task_packets_dir

    @property
    def grounded_dir(self) -> Path:
        return self.layout.grounded_relations_dir

    @property
    def formal_dir(self) -> Path:
        return self.layout.formal_graphs_dir

    @property
    def build_plans_dir(self) -> Path:
        return self.layout.build_plans_dir

    def ensure_dirs(self) -> None:
        for path in [self.visual_dir, self.visual_overlay_dir, self.script_observations_dir, self.evidence_graphs_dir, self.text_claims_dir, self.evidence_boards_dir, self.semantic_dir, self.semantic_raw_dir, self.semantic_repaired_dir, self.vlm_passes_dir, self.vlm_transcripts_dir, self.instruction_task_packets_dir, self.grounded_dir, self.formal_dir, self.build_plans_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def process_record(self, record: Mapping[str, Any] | ParserInput) -> Dict[str, Any]:
        """Process a dataset row after projecting it to the public input contract."""

        return self._process_input(_public_parser_input(record))

    def process_input(self, data: ParserInput | Mapping[str, Any]) -> Dict[str, Any]:
        """Process only a strict four-field ParserInput payload."""

        return self._process_input(validate_parser_input(data))

    def _process_input(self, parser_input: ParserInput) -> Dict[str, Any]:
        self.ensure_dirs()
        sample_id = parser_input.id
        image_path = parser_input.image
        problem_text = parser_input.question
        solution_text = None
        raw_record: Dict[str, Any] = {}
        answer = None

        paths = self._paths(sample_id)
        visual_state = self.visual.detect(str(image_path)) if image_path else {"error": "missing_image_path"}
        if visual_state.get("error"):
            raise ValueError(f"Hybrid parser could not read image {image_path}: {visual_state['error']}")
        self._write_json(paths["visual"], visual_state)
        script_observations = self.script_observations.extract(
            visual_state=visual_state,
            problem_text=problem_text,
            solution_text=solution_text,
            raw_record=raw_record,
            answer=answer,
        )
        visual_state["visual_structure_candidates"] = script_observations.get("visual_structure_candidates") or {}
        self._write_json(paths["visual"], visual_state)
        if image_path and Path(str(image_path)).exists():
            self.visual_overlay.render(str(image_path), visual_state, str(paths["visual_overlay"]))
        self._write_json(paths["script_observations"], script_observations)
        evidence_graph = self.evidence_graph.build(
            visual_state=visual_state,
            script_observations=script_observations,
            problem_text=problem_text,
            answer=answer,
        )
        self._write_json(paths["evidence_graph"], evidence_graph)
        text_claims = self.text_claims_builder.build(script_observations=script_observations)
        self._write_json(paths["text_claims"], text_claims)
        evidence_board = self.evidence_board_builder.build(
            visual_state=visual_state,
            script_observations=script_observations,
            text_claims=text_claims,
            evidence_graph=evidence_graph,
        )
        self._write_json(paths["evidence_board"], evidence_board)

        raw_semantic = self.semantic.extract(
            image_path=str(image_path),
            problem_text=problem_text,
            answer=answer,
            visual_state=visual_state,
            overlay_path=str(paths["visual_overlay"]) if paths["visual_overlay"].exists() else None,
            script_observations=script_observations,
            evidence_graph=evidence_graph,
            evidence_board=evidence_board,
        )
        raw_semantic = self.relation_engine_metadata.annotate(semantic=raw_semantic)
        semantic_validation = self.semantic_validator.validate(raw_semantic)
        semantic_before_repair = semantic_validation["semantic"]
        self._write_json(paths["semantic_raw"], semantic_before_repair)
        self._write_staged_outputs(paths, raw_semantic)
        semantic_for_builder = self.role_canonicalizer.canonicalize(semantic_before_repair)
        if semantic_for_builder != semantic_before_repair:
            semantic_validation = self.semantic_validator.validate(semantic_for_builder)
            semantic_for_builder = semantic_validation["semantic"]
        instruction_task_packet = self._instruction_task_packet(raw_semantic, semantic_for_builder, evidence_graph)
        self._write_json(paths["instruction_task_packet"], instruction_task_packet)
        raw_build_plan_seed = self.build_plan_compiler.compile(
            semantic=semantic_for_builder,
            evidence_board=evidence_board,
            instruction_task_packet=instruction_task_packet,
        )
        raw_build_plan_seed_coverage = self.build_plan_seed_coverage.analyze(
            build_plan_seed=raw_build_plan_seed,
            semantic=semantic_for_builder,
        )
        raw_task_packet_coverage = self.instruction_task_packet_coverage.analyze(
            packet=instruction_task_packet,
            semantic=semantic_for_builder,
        )
        raw_template_adherence = self.response_template_adherence.analyze(
            packet=instruction_task_packet,
            semantic=semantic_for_builder,
        )
        raw_relation_template_adherence = self.relation_template_adherence.analyze(
            evidence_graph=evidence_graph,
            semantic=semantic_for_builder,
        )
        raw_relation_builder_decisions = self.relation_builder_decision.analyze(
            semantic=semantic_for_builder,
        )
        raw_stage_contract = self.stage_contract.analyze(
            evidence_graph=evidence_graph,
            instruction_task_packet=instruction_task_packet,
            semantic=semantic_for_builder,
        )
        stage_contract_repair = self.stage_contract_repairer.repair(
            semantic=semantic_for_builder,
            evidence_graph=evidence_graph,
        )
        semantic_for_instruction_repair = stage_contract_repair["semantic"]
        if stage_contract_repair.get("used"):
            semantic_validation = self.semantic_validator.validate(semantic_for_instruction_repair)
            semantic_for_instruction_repair = semantic_validation["semantic"]
        relation_draft_repair = self.relation_draft_repairer.repair(
            semantic=semantic_for_instruction_repair,
            evidence_graph=evidence_graph,
        )
        semantic_for_instruction_repair = relation_draft_repair["semantic"]
        if relation_draft_repair.get("used"):
            semantic_validation = self.semantic_validator.validate(semantic_for_instruction_repair)
            semantic_for_instruction_repair = semantic_validation["semantic"]
        grounding_decision_repair = self.grounding_decision_repairer.repair(
            semantic=semantic_for_instruction_repair,
        )
        semantic_for_instruction_repair = grounding_decision_repair["semantic"]
        if grounding_decision_repair.get("used"):
            semantic_validation = self.semantic_validator.validate(semantic_for_instruction_repair)
            semantic_for_instruction_repair = semantic_validation["semantic"]
        candidate_usage_before_repair = self.candidate_usage.analyze(script_observations, semantic_for_instruction_repair)
        repair = self.instruction_repairer.repair(
            semantic=semantic_for_instruction_repair,
            evidence_graph=evidence_graph,
            candidate_usage=candidate_usage_before_repair,
            instruction_task_packet=instruction_task_packet,
        )
        if repair.get("used"):
            semantic_validation = self.semantic_validator.validate(repair["semantic"])
        semantic = semantic_validation["semantic"]
        final_stage_contract_repair = self.stage_contract_repairer.repair(
            semantic=semantic,
            evidence_graph=evidence_graph,
            instruction_task_packet=instruction_task_packet,
        )
        if final_stage_contract_repair.get("used"):
            semantic_validation = self.semantic_validator.validate(final_stage_contract_repair["semantic"])
            semantic = semantic_validation["semantic"]
        semantic = self.role_canonicalizer.canonicalize(semantic)
        semantic_validation = self.semantic_validator.validate(semantic)
        self._write_json(paths["semantic_repaired"], semantic)
        final_task_packet_coverage = self.instruction_task_packet_coverage.analyze(
            packet=instruction_task_packet,
            semantic=semantic,
        )
        final_template_adherence = self.response_template_adherence.analyze(
            packet=instruction_task_packet,
            semantic=semantic,
        )
        final_relation_template_adherence = self.relation_template_adherence.analyze(
            evidence_graph=evidence_graph,
            semantic=semantic,
        )
        final_relation_builder_decisions = self.relation_builder_decision.analyze(
            semantic=semantic,
        )
        final_stage_contract = self.stage_contract.analyze(
            evidence_graph=evidence_graph,
            instruction_task_packet=instruction_task_packet,
            semantic=semantic,
        )
        board_adjudication = self.board_adjudication.analyze(
            semantic=semantic,
            evidence_board=evidence_board,
        )
        build_plan = self.build_plan_compiler.compile(
            semantic=semantic,
            evidence_board=evidence_board,
            instruction_task_packet=instruction_task_packet,
        )
        final_build_plan_seed_coverage = self.build_plan_seed_coverage.analyze(
            build_plan_seed=build_plan,
            semantic=semantic,
        )
        self._write_json(paths["build_plan"], build_plan)
        candidate_usage = self.candidate_usage.analyze(script_observations, semantic)
        self._write_json(paths["semantic"], semantic)

        grounded = self.grounder.ground(semantic, visual_state)
        self._write_json(paths["grounded"], grounded)

        formal = self.formalizer.build(grounded)
        formal["answer"] = answer
        formal["problem_text"] = problem_text
        formal["image_path"] = image_path
        instruction_plan = self.instruction_planner.plan(semantic, formal, visual_state)
        instruction_coverage = self.instruction_coverage.analyze(formal, instruction_plan)
        engine_assessment = self.engine_assessment.validate(semantic, formal, instruction_plan)
        contract_audit = self.contract_auditor.audit(
            evidence_graph=evidence_graph,
            semantic_validation=semantic_validation,
            candidate_usage=candidate_usage,
            instruction_coverage=instruction_coverage,
            instruction_plan=instruction_plan,
            engine_assessment=engine_assessment,
        )
        formal["instruction_plan"] = {
            "buildable": instruction_plan["buildable"],
            "reason": instruction_plan["reason"],
            "num_instructions": instruction_plan["num_instructions"],
            "num_accepted": instruction_plan["num_accepted"],
            "num_rejected": instruction_plan["num_rejected"],
            "num_synthesized": instruction_plan["num_synthesized"],
            "accepted_instructions": instruction_plan["accepted_instructions"],
            "rejected_instructions": instruction_plan["rejected_instructions"],
            "synthesized_instructions": instruction_plan["synthesized_instructions"],
        }
        formal["engine_assessment"] = engine_assessment
        formal["extraction_contract"] = contract_audit
        self._write_json(paths["formal"], formal)

        report = self._build_report(sample_id, visual_state, script_observations, evidence_graph, text_claims, evidence_board, board_adjudication, build_plan, raw_build_plan_seed_coverage, final_build_plan_seed_coverage, raw_semantic, instruction_task_packet, raw_task_packet_coverage, final_task_packet_coverage, raw_template_adherence, final_template_adherence, raw_relation_template_adherence, final_relation_template_adherence, raw_relation_builder_decisions, final_relation_builder_decisions, raw_stage_contract, final_stage_contract, semantic_before_repair, semantic, semantic_validation, candidate_usage, instruction_coverage, contract_audit, grounded, formal, instruction_plan, engine_assessment, paths, repair, self._merge_stage_contract_repairs(stage_contract_repair, final_stage_contract_repair), relation_draft_repair, grounding_decision_repair)
        self._write_json(paths["report"], report)
        return report

    def _build_report(
        self,
        sample_id: str,
        visual_state: Dict[str, Any],
        script_observations: Dict[str, Any],
        evidence_graph: Dict[str, Any],
        text_claims: Dict[str, Any],
        evidence_board: Dict[str, Any],
        board_adjudication: Dict[str, Any],
        build_plan: Dict[str, Any],
        raw_build_plan_seed_coverage: Dict[str, Any],
        final_build_plan_seed_coverage: Dict[str, Any],
        raw_semantic: Dict[str, Any],
        instruction_task_packet: Dict[str, Any],
        raw_task_packet_coverage: Dict[str, Any],
        final_task_packet_coverage: Dict[str, Any],
        raw_template_adherence: Dict[str, Any],
        final_template_adherence: Dict[str, Any],
        raw_relation_template_adherence: Dict[str, Any],
        final_relation_template_adherence: Dict[str, Any],
        raw_relation_builder_decisions: Dict[str, Any],
        final_relation_builder_decisions: Dict[str, Any],
        raw_stage_contract: Dict[str, Any],
        final_stage_contract: Dict[str, Any],
        semantic_before_repair: Dict[str, Any],
        semantic: Dict[str, Any],
        semantic_validation: Dict[str, Any],
        candidate_usage: Dict[str, Any],
        instruction_coverage: Dict[str, Any],
        contract_audit: Dict[str, Any],
        grounded: Dict[str, Any],
        formal: Dict[str, Any],
        instruction_plan: Dict[str, Any],
        engine_assessment: Dict[str, Any],
        paths: Dict[str, Path],
        instruction_repair: Dict[str, Any] | None = None,
        stage_contract_repair: Dict[str, Any] | None = None,
        relation_draft_repair: Dict[str, Any] | None = None,
        grounding_decision_repair: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        image_path = formal.get("image_path")
        mode = formal.get("mode")
        report_mode = "partial_coordinate_scaffold" if (
            mode == "coordinate_buildable" and bool(formal.get("requires_theorem_solver"))
        ) else mode
        buildable = bool(instruction_plan.get("buildable"))
        target_resolution_metrics = self._target_resolution_metrics(semantic, instruction_task_packet, instruction_plan)
        image_report = None
        consistency_report = None
        verification = {
            "schema_valid": buildable,
            "geometry_valid": buildable,
            "quantity_valid": True,
            "derivation_valid": True,
            "image_alignment_valid": None,
            "answer_valid": None,
            "llm_judge_valid": None,
            "valid": False,
            "issues": [],
            "metrics": {
                **self._visual_metrics(visual_state),
                "visual_lines": len(visual_state.get("lines") or []),
                "visual_circles": len(visual_state.get("circles") or []),
                "visual_text_regions": len(visual_state.get("text_regions") or []),
                **self._script_observation_metrics(script_observations),
                **self._evidence_graph_metrics(evidence_graph),
                **self._board_adjudication_metrics(board_adjudication),
                **self._build_plan_seed_coverage_metrics(raw_build_plan_seed_coverage, prefix="raw_build_plan_seed"),
                **self._build_plan_seed_coverage_metrics(final_build_plan_seed_coverage, prefix="final_build_plan_seed"),
                **self._staged_output_metrics(raw_semantic),
                **self._instruction_task_packet_metrics(instruction_task_packet),
                **self._task_packet_coverage_metrics(raw_task_packet_coverage, prefix="raw_task_packet"),
                **self._task_packet_coverage_metrics(final_task_packet_coverage, prefix="final_task_packet"),
                **self._template_adherence_metrics(raw_template_adherence, prefix="raw_template"),
                **self._template_adherence_metrics(final_template_adherence, prefix="final_template"),
                **self._relation_template_adherence_metrics(raw_relation_template_adherence, prefix="raw_relation_template"),
                **self._relation_template_adherence_metrics(final_relation_template_adherence, prefix="final_relation_template"),
                **self._relation_builder_decision_metrics(raw_relation_builder_decisions, prefix="raw_relation_builder"),
                **self._relation_builder_decision_metrics(final_relation_builder_decisions, prefix="final_relation_builder"),
                **self._stage_contract_metrics(raw_stage_contract, prefix="raw_stage_contract"),
                **self._stage_contract_metrics(final_stage_contract, prefix="final_stage_contract"),
                **self._candidate_usage_metrics(candidate_usage),
                **self._instruction_coverage_metrics(instruction_coverage),
                **self._instruction_repair_metrics(instruction_repair or {}),
                **target_resolution_metrics,
                **self._semantic_role_metrics(semantic),
                **self._stage_contract_repair_metrics(stage_contract_repair or {}),
                **self._relation_draft_repair_metrics(relation_draft_repair or {}),
                **self._grounding_decision_repair_metrics(grounding_decision_repair or {}),
                **self._contract_audit_metrics(contract_audit),
                **self._formal_metrics(formal),
                "semantic_schema_warnings": semantic_validation.get("metrics", {}).get("semantic_schema_warnings", 0),
                "semantic_relations": semantic_validation.get("metrics", {}).get("semantic_relations", 0),
                "semantic_named_object_bindings": semantic_validation.get("metrics", {}).get("semantic_named_object_bindings", len(semantic.get("named_object_bindings") or [])),
                "semantic_build_instructions": semantic_validation.get("metrics", {}).get("semantic_build_instructions", 0),
                "semantic_candidate_decisions": semantic_validation.get("metrics", {}).get("semantic_candidate_decisions", 0),
                "semantic_grounding_decisions": semantic_validation.get("metrics", {}).get("semantic_grounding_decisions", len(semantic.get("grounding_decisions") or [])),
                "semantic_relation_builder_decisions": semantic_validation.get("metrics", {}).get("semantic_relation_builder_decisions", len(semantic.get("relation_builder_decisions") or [])),
                "semantic_instruction_task_decisions": semantic_validation.get("metrics", {}).get("semantic_instruction_task_decisions", len(semantic.get("instruction_task_decisions") or [])),
                "semantic_targets": semantic_validation.get("metrics", {}).get("semantic_targets", len(semantic.get("targets") or [])),
                "raw_semantic_grounding_decisions": len(semantic_before_repair.get("grounding_decisions") or []),
                "final_semantic_grounding_decisions": len(semantic.get("grounding_decisions") or []),
                "raw_semantic_relation_builder_decisions": len(semantic_before_repair.get("relation_builder_decisions") or []),
                "final_semantic_relation_builder_decisions": len(semantic.get("relation_builder_decisions") or []),
                "raw_semantic_named_object_bindings": len(semantic_before_repair.get("named_object_bindings") or []),
                "final_semantic_named_object_bindings": len(semantic.get("named_object_bindings") or []),
                "raw_semantic_instruction_task_decisions": len(semantic_before_repair.get("instruction_task_decisions") or []),
                "final_semantic_instruction_task_decisions": len(semantic.get("instruction_task_decisions") or []),
                "raw_semantic_targets": len(semantic_before_repair.get("targets") or []),
                "final_semantic_targets": len(semantic.get("targets") or []),
                "grounded_accepted": (formal.get("grounding_summary") or {}).get("accepted", 0),
                "grounded_uncertain": (formal.get("grounding_summary") or {}).get("uncertain", 0),
                "grounded_rejected": (formal.get("grounding_summary") or {}).get("rejected", 0),
                "build_instructions": instruction_plan.get("num_instructions", 0),
                "build_instructions_accepted": instruction_plan.get("num_accepted", 0),
                "build_instructions_rejected": instruction_plan.get("num_rejected", 0),
                "build_instructions_synthesized": instruction_plan.get("num_synthesized", 0),
            },
        }
        instruction_issues = [
            self._issue(
                "instruction.rejected",
                inst.get("instruction_reason", "Build instruction rejected"),
                severity="warning",
                action=inst.get("action"),
                instruction_index=inst.get("instruction_index"),
                visual_ids=inst.get("visual_ids"),
            )
            for inst in instruction_plan.get("rejected_instructions") or []
        ]
        semantic_issues = [
            self._issue(
                warning.get("code", "semantic.schema_warning"),
                warning.get("message", "Semantic output schema warning"),
                severity="warning",
                location=warning.get("location"),
            )
            for warning in semantic_validation.get("warnings") or []
        ]
        candidate_issues = [
            self._issue(
                warning.get("code", "candidate.warning"),
                warning.get("message", "Candidate usage warning"),
                severity="warning",
                location=warning.get("location"),
            )
            for warning in candidate_usage.get("warnings") or []
        ]
        staged_metrics = self._staged_output_metrics(raw_semantic)
        staged_issues = [
            self._issue(
                "staged.relation_pass_leaked_instructions",
                "Stage-1 relation pass output build_instructions; they are ignored by the staged merger",
                severity="warning",
                count=staged_metrics.get("staged_relation_pass_leaked_instructions", 0),
            )
        ] if staged_metrics.get("staged_relation_pass_leaked_instructions", 0) > 0 else []
        staged_issues += [
            self._issue(
                "staged.instruction_pass_extra_relations",
                "Stage-2 instruction pass output extra relations; relation pass remains authoritative",
                severity="warning",
                count=staged_metrics.get("staged_instruction_pass_extra_relations", 0),
            )
        ] if staged_metrics.get("staged_instruction_pass_extra_relations", 0) > 0 else []
        staged_issues += [
            self._issue(
                "staged.relation_pass_schema_warning",
                "Stage-1 relation pass had schema warnings before staged merge",
                severity="warning",
                count=staged_metrics.get("staged_relation_pass_schema_warnings", 0),
            )
        ] if staged_metrics.get("staged_relation_pass_schema_warnings", 0) > 0 else []
        staged_issues += [
            self._issue(
                "staged.instruction_pass_schema_warning",
                "Stage-2 instruction pass had schema warnings before staged merge",
                severity="warning",
                count=staged_metrics.get("staged_instruction_pass_schema_warnings", 0),
            )
        ] if staged_metrics.get("staged_instruction_pass_schema_warnings", 0) > 0 else []
        contract_issues = [
            self._issue(
                item.get("code", "contract.warning"),
                item.get("message", "Extraction contract warning"),
                severity="warning",
                **(item.get("metadata") or {}),
            )
            for item in (contract_audit.get("warnings") or [])
        ] + [
            self._issue(
                item.get("code", "contract.error"),
                item.get("message", "Extraction contract error"),
                **(item.get("metadata") or {}),
            )
            for item in (contract_audit.get("errors") or [])
        ]
        repair_issues = [
            self._issue(
                "instruction.script_repair_applied",
                "GeoSym added builder instructions from script suggestions for candidates already confirmed or used by semantic extraction",
                severity="warning",
                num_added=(instruction_repair or {}).get("num_added_instructions", 0),
            )
        ] if (instruction_repair or {}).get("used") else []
        relation_repair_issues = [
            self._issue(
                "relation_draft.template_repair_applied",
                "GeoSym added relation/target drafts from relation_pass_response_template for candidates already confirmed or used by semantic extraction",
                severity="warning",
                num_added_relations=(relation_draft_repair or {}).get("num_added_relations", 0),
                target_added=bool((relation_draft_repair or {}).get("target_added")),
            )
        ] if (relation_draft_repair or {}).get("used") else []
        task_packet_issues = [
            self._issue(
                "task_packet.raw_missing_instruction",
                "Raw semantic output did not cover all required instruction task packet actions",
                severity="warning",
                candidate_id=missing.get("candidate_id"),
                missing_actions=missing.get("missing_actions") or [],
                covered_actions=missing.get("covered_actions") or [],
                missing_instruction_requirements=missing.get("num_missing_instruction_requirements", 0),
            )
            for missing in raw_task_packet_coverage.get("missing_tasks") or []
        ]
        vlm_evidence_issues = self._vlm_derived_evidence_issues(instruction_task_packet)
        raw_stage1 = raw_stage_contract.get("stage1") or {}
        raw_stage2 = raw_stage_contract.get("stage2") or {}
        stage_contract_issues = []
        if raw_stage1.get("num_missing_candidate_decisions", 0) > 0:
            stage_contract_issues.append(
                self._issue(
                    "stage_contract.raw_missing_candidate_decisions",
                    "Raw VLM relation stage did not review all script candidates",
                    severity="warning",
                    missing_candidate_decision_ids=raw_stage1.get("missing_candidate_decision_ids") or [],
                )
            )
        if raw_stage1.get("num_missing_relation_refs", 0) > 0:
            stage_contract_issues.append(
                self._issue(
                    "stage_contract.raw_missing_relation_refs",
                    "Raw VLM relation stage did not output relation drafts for all script relation candidates",
                    severity="warning",
                    missing_relation_ref_ids=raw_stage1.get("missing_relation_ref_ids") or [],
                )
            )
        if raw_stage1.get("num_missing_relation_buildability_refs", 0) > 0:
            stage_contract_issues.append(
                self._issue(
                    "stage_contract.raw_missing_relation_buildability",
                    "Raw VLM relation stage did not preserve buildability metadata for all relation candidates",
                    severity="warning",
                    missing_relation_buildability_ids=raw_stage1.get("missing_relation_buildability_ids") or [],
                )
            )
        if raw_stage1.get("num_missing_relation_builder_mapping_refs", 0) > 0:
            stage_contract_issues.append(
                self._issue(
                    "stage_contract.raw_missing_relation_builder_mapping",
                    "Raw VLM relation stage did not preserve relation-to-builder mapping for all relation candidates",
                    severity="warning",
                    missing_relation_builder_mapping_ids=raw_stage1.get("missing_relation_builder_mapping_ids") or [],
                )
            )
        if raw_stage1.get("num_inconsistent_relation_buildability", 0) > 0:
            stage_contract_issues.append(
                self._issue(
                    "stage_contract.raw_inconsistent_relation_buildability",
                    "Raw VLM relation stage assigned buildability or builder_mapping that conflicts with current GeoSym builder capability",
                    severity="warning",
                    inconsistent_relation_buildability_reports=raw_stage1.get("inconsistent_relation_buildability_reports") or [],
                )
            )
        if raw_stage1.get("num_missing_target_refs", 0) > 0:
            stage_contract_issues.append(
                self._issue(
                    "stage_contract.raw_missing_target_refs",
                    "Raw VLM relation stage did not output target draft for all script target candidates",
                    severity="warning",
                    missing_target_ref_ids=raw_stage1.get("missing_target_ref_ids") or [],
                )
            )
        if raw_stage2.get("num_missing_required_tasks", 0) > 0:
            stage_contract_issues.append(
                self._issue(
                    "stage_contract.raw_missing_required_tasks",
                    "Raw VLM instruction stage did not cover all required GeoSym build tasks",
                    severity="warning",
                    missing_task_reports=raw_stage2.get("missing_task_reports") or [],
                )
            )
        if raw_stage2.get("num_inconsistent_instruction_task_decisions", 0) > 0:
            stage_contract_issues.append(
                self._issue(
                    "stage_contract.raw_inconsistent_instruction_task_decisions",
                    "Raw VLM instruction task decisions conflict with emitted GeoSym build instructions",
                    severity="warning",
                    inconsistent_instruction_task_decision_reports=raw_stage2.get("inconsistent_instruction_task_decision_reports") or [],
                )
            )
        relation_builder_issues = []
        if raw_relation_builder_decisions.get("num_missing_relation_builder_decisions", 0) > 0:
            relation_builder_issues.append(
                self._issue(
                    "relation_builder.raw_missing_decisions",
                    "Raw VLM relation stage did not explicitly map all extracted relations to GeoSym builder capability decisions",
                    severity="warning",
                    missing_relation_builder_decision_ids=raw_relation_builder_decisions.get("missing_relation_builder_decision_ids") or [],
                )
            )
        if raw_relation_builder_decisions.get("num_inconsistent_relation_builder_decisions", 0) > 0:
            relation_builder_issues.append(
                self._issue(
                    "relation_builder.raw_inconsistent_decisions",
                    "Raw VLM relation stage mapped relations to builder decisions that conflict with current GeoSym capabilities",
                    severity="warning",
                    inconsistent_relation_builder_decision_reports=raw_relation_builder_decisions.get("inconsistent_relation_builder_decision_reports") or [],
                )
            )
        target_issues = [
            self._issue(
                "target.unresolved",
                "VLM/text target was present but not bound to a GeoSym-usable target object",
                severity="warning",
                semantic_targets=target_resolution_metrics.get("target_semantic_targets", 0),
                usable_targets=target_resolution_metrics.get("target_usable_targets", 0),
                deferred_target_tasks=target_resolution_metrics.get("target_deferred_tasks", 0),
            )
        ] if target_resolution_metrics.get("target_unresolved", 0) else []
        if target_resolution_metrics.get("target_symbolic_targets", 0) > 0 and not target_resolution_metrics.get("target_usable_targets", 0):
            target_issues.append(
                self._issue(
                    "target.symbolic_variable",
                    "Target is a symbolic variable from algebraic geometry relations, not a directly renderable GeoSym object",
                    severity="warning",
                    semantic_targets=target_resolution_metrics.get("target_semantic_targets", 0),
                    symbolic_targets=target_resolution_metrics.get("target_symbolic_targets", 0),
                )
            )
        pre_verification_issues = list(semantic_issues) + list(candidate_issues) + list(staged_issues) + list(contract_issues) + list(instruction_issues) + list(relation_repair_issues) + list(repair_issues) + list(task_packet_issues) + list(vlm_evidence_issues) + list(stage_contract_issues) + list(relation_builder_issues) + list(target_issues)
        if engine_assessment.get("consistent") is False:
            pre_verification_issues.append(
                self._issue(
                    "engine_assessment.mismatch",
                    engine_assessment.get("reason", "VLM engine assessment differs from GeoSym planner"),
                    severity="warning",
                    expected_mode=engine_assessment.get("expected_mode"),
                    actual_mode=engine_assessment.get("actual_mode"),
                )
            )
        verification["issues"].extend(pre_verification_issues)
        judge = LLMJudgeSignal(used=False)
        candidate_payload = {
            "formal_graph": formal,
            "canonical_problem_state": None,
        }

        if buildable:
            constraint_graph = instruction_plan.get("constraint_graph") or self._formal_to_constraint_graph(formal)
            try:
                state = self.adapter.convert(
                    constraint_graph,
                    image_path=image_path,
                    problem_text=formal.get("problem_text") or "",
                    answer=formal.get("answer"),
                    problem_id=sample_id,
                )
                hard_report = self.verifier.verify(state)
                state.verification = hard_report
                candidate_payload["canonical_problem_state"] = state.to_dict()
                build_error = ((state.diagnostic.metadata or {}).get("build_error") if state.diagnostic else None)
                has_renderable_geometry = bool(state.geometry.points and state.geometry.lines)
                render_full_canonical = (
                    not build_error
                    and has_renderable_geometry
                    and not bool(formal.get("requires_theorem_solver"))
                )
                if render_full_canonical:
                    self.renderer.render(state, str(paths["sketch"]))
                    if image_path and Path(str(image_path)).exists():
                        image_report = self.alignment.verify_paths(str(image_path), str(paths["sketch"]))
                        self.review.compose(str(image_path), str(paths["sketch"]), str(paths["review"]), title=sample_id)
                elif image_path and Path(str(image_path)).exists():
                    if paths["sketch"].exists():
                        paths["sketch"].unlink()
                    if has_renderable_geometry:
                        self.renderer.render(state, str(paths["scaffold_sketch"]))
                        self.review.compose(
                            str(image_path),
                            str(paths["scaffold_sketch"]),
                            str(paths["scaffold_review"]),
                            title=f"{sample_id} scaffold",
                        )
                    reason = build_error or "partial coordinate scaffold requires theorem solver"
                    self._text_review(
                        image_path,
                        paths["review"],
                        sample_id,
                        formal,
                        extra=[
                            "No full canonical sketch rendered.",
                            f"Build status: {reason}",
                            "Reason: current GeoSym builder only renders fully supported coordinate constraints.",
                            "The semantic graph is preserved for diagnosis/review.",
                        ],
                        overlay_path=paths["visual_overlay"],
                    )
                if formal.get("answer") is not None and state.target and state.target.answer:
                    consistency_report = self.consistency.verify_answer(state.target.answer, formal.get("answer"))
                verification = hard_report.to_dict()
                verification.setdefault("issues", []).extend(pre_verification_issues)
                verification.setdefault("metrics", {}).update({
                    **self._visual_metrics(visual_state),
                    "visual_lines": len(visual_state.get("lines") or []),
                    "visual_circles": len(visual_state.get("circles") or []),
                    "visual_text_regions": len(visual_state.get("text_regions") or []),
                    **self._script_observation_metrics(script_observations),
                    **self._evidence_graph_metrics(evidence_graph),
                    **self._board_adjudication_metrics(board_adjudication),
                    **self._build_plan_seed_coverage_metrics(raw_build_plan_seed_coverage, prefix="raw_build_plan_seed"),
                    **self._build_plan_seed_coverage_metrics(final_build_plan_seed_coverage, prefix="final_build_plan_seed"),
                    **self._staged_output_metrics(raw_semantic),
                    **self._instruction_task_packet_metrics(instruction_task_packet),
                    **self._task_packet_coverage_metrics(raw_task_packet_coverage, prefix="raw_task_packet"),
                    **self._task_packet_coverage_metrics(final_task_packet_coverage, prefix="final_task_packet"),
                    **self._template_adherence_metrics(raw_template_adherence, prefix="raw_template"),
                    **self._template_adherence_metrics(final_template_adherence, prefix="final_template"),
                    **self._relation_template_adherence_metrics(raw_relation_template_adherence, prefix="raw_relation_template"),
                    **self._relation_template_adherence_metrics(final_relation_template_adherence, prefix="final_relation_template"),
                    **self._relation_builder_decision_metrics(raw_relation_builder_decisions, prefix="raw_relation_builder"),
                    **self._relation_builder_decision_metrics(final_relation_builder_decisions, prefix="final_relation_builder"),
                    **self._stage_contract_metrics(raw_stage_contract, prefix="raw_stage_contract"),
                    **self._stage_contract_metrics(final_stage_contract, prefix="final_stage_contract"),
                    **self._candidate_usage_metrics(candidate_usage),
                    **self._instruction_coverage_metrics(instruction_coverage),
                    **self._instruction_repair_metrics(instruction_repair or {}),
                    **target_resolution_metrics,
                    **self._semantic_role_metrics(semantic),
                    **self._stage_contract_repair_metrics(stage_contract_repair or {}),
                    **self._relation_draft_repair_metrics(relation_draft_repair or {}),
                    **self._grounding_decision_repair_metrics(grounding_decision_repair or {}),
                    **self._contract_audit_metrics(contract_audit),
                    **self._formal_metrics(formal),
                    "semantic_schema_warnings": semantic_validation.get("metrics", {}).get("semantic_schema_warnings", 0),
                    "semantic_relations": semantic_validation.get("metrics", {}).get("semantic_relations", 0),
                    "semantic_build_instructions": semantic_validation.get("metrics", {}).get("semantic_build_instructions", 0),
                    "semantic_candidate_decisions": semantic_validation.get("metrics", {}).get("semantic_candidate_decisions", 0),
                    "semantic_grounding_decisions": semantic_validation.get("metrics", {}).get("semantic_grounding_decisions", len(semantic.get("grounding_decisions") or [])),
                    "semantic_relation_builder_decisions": semantic_validation.get("metrics", {}).get("semantic_relation_builder_decisions", len(semantic.get("relation_builder_decisions") or [])),
                    "semantic_instruction_task_decisions": semantic_validation.get("metrics", {}).get("semantic_instruction_task_decisions", len(semantic.get("instruction_task_decisions") or [])),
                    "semantic_targets": semantic_validation.get("metrics", {}).get("semantic_targets", len(semantic.get("targets") or [])),
                    "raw_semantic_grounding_decisions": len(semantic_before_repair.get("grounding_decisions") or []),
                    "final_semantic_grounding_decisions": len(semantic.get("grounding_decisions") or []),
                    "raw_semantic_relation_builder_decisions": len(semantic_before_repair.get("relation_builder_decisions") or []),
                    "final_semantic_relation_builder_decisions": len(semantic.get("relation_builder_decisions") or []),
                    "raw_semantic_instruction_task_decisions": len(semantic_before_repair.get("instruction_task_decisions") or []),
                    "final_semantic_instruction_task_decisions": len(semantic.get("instruction_task_decisions") or []),
                    "raw_semantic_targets": len(semantic_before_repair.get("targets") or []),
                    "final_semantic_targets": len(semantic.get("targets") or []),
                    "grounded_accepted": (formal.get("grounding_summary") or {}).get("accepted", 0),
                    "grounded_uncertain": (formal.get("grounding_summary") or {}).get("uncertain", 0),
                    "grounded_rejected": (formal.get("grounding_summary") or {}).get("rejected", 0),
                    "build_instructions": instruction_plan.get("num_instructions", 0),
                    "build_instructions_accepted": instruction_plan.get("num_accepted", 0),
                    "build_instructions_rejected": instruction_plan.get("num_rejected", 0),
                    "build_instructions_synthesized": instruction_plan.get("num_synthesized", 0),
                })
                judge = self._judge_or_review(image_path=image_path, formal=formal, state_dict=state.to_dict(), verification=verification)
            except Exception as exc:
                verification["issues"].append(self._issue("pipeline.coordinate_build_failed", str(exc)))
                candidate_payload["build_error"] = str(exc)
                self._text_review(image_path, paths["review"], sample_id, formal, extra=[f"coordinate build failed: {exc}"], overlay_path=paths["visual_overlay"])
        else:
            code = "formal.theorem_only" if mode == "theorem_only" else "formal.not_buildable"
            verification["issues"].append(self._issue(code, instruction_plan.get("reason") or formal.get("reason", "not coordinate buildable")))
            if paths["sketch"].exists():
                paths["sketch"].unlink()
            self._text_review(image_path, paths["review"], sample_id, formal, overlay_path=paths["visual_overlay"])
            judge = self._judge_or_review(image_path=image_path, formal=formal, state_dict={"formal": formal, "state": None}, verification=verification)

        hard_valid = bool(verification.get("valid"))
        if image_report is not None:
            hard_valid = hard_valid and image_report.get("valid", False)
        if consistency_report is not None:
            hard_valid = hard_valid and consistency_report.get("valid", False)
        acceptance_policy = self._acceptance_policy(
            hard_valid=hard_valid,
            judge=judge,
            instruction_task_packet=instruction_task_packet,
            verification=verification,
        )
        stage_diagnosis = self.stage_classifier.classify(
            visual_state=visual_state,
            semantic=semantic,
            grounded=grounded,
            formal=formal,
            instruction_plan=instruction_plan,
            verification=verification,
            image_alignment=image_report,
            consistency=consistency_report,
            contract_audit=contract_audit,
        )
        parser_record = self.record_adapter.convert(
            sample_id=sample_id,
            dataset_name=self.layout.dataset_name,
            parser_name="hybrid_parse",
            backend=self.semantic_backend,
            candidate_kind="hybrid_geometry_candidate",
            candidate_payload=candidate_payload,
            confidence=self._semantic_confidence(semantic),
            verification=verification,
            image_path=str(image_path) if image_path else None,
            metadata={"recommended_decision": acceptance_policy["decision"]},
        )
        self._write_json(paths["record"], parser_record.to_dict())
        return {
            "mode": report_mode,
            "reason": formal.get("reason"),
            "formal_summary": {
                "mode": formal.get("mode"),
                "reason": formal.get("reason"),
                "requires_theorem_solver": bool(formal.get("requires_theorem_solver")),
                "coordinate_relation_types": formal.get("coordinate_relation_types") or [],
                "theorem_relation_types": formal.get("theorem_relation_types") or [],
                "unsupported_relation_types": formal.get("unsupported_relation_types") or [],
                "num_relations": len(formal.get("relations") or []),
            },
            "stage_diagnosis": stage_diagnosis,
            "script_observations_summary": {
                "counts": script_observations.get("counts") or {},
                "text_hints": script_observations.get("text_hints") or {},
                "text_sources": [
                    {
                        "name": item.get("name"),
                        "confidence": item.get("confidence"),
                        "chars": len(item.get("text") or ""),
                    }
                    for item in (script_observations.get("text_sources") or [])[:20]
                ],
                "semantic_candidates": {
                    "num_relation_candidates": len((script_observations.get("semantic_candidates") or {}).get("relation_candidates") or []),
                    "num_target_candidates": len((script_observations.get("semantic_candidates") or {}).get("target_candidates") or []),
                },
                "engine_build_suggestions": {
                    "expected_mode_hint": (script_observations.get("engine_build_suggestions") or {}).get("expected_mode_hint"),
                    "num_suggested_build_instructions": len((script_observations.get("engine_build_suggestions") or {}).get("suggested_build_instructions") or []),
                    "unsupported_candidate_types": (script_observations.get("engine_build_suggestions") or {}).get("unsupported_candidate_types") or [],
                },
            },
            "evidence_graph_summary": self.evidence_graph.summary(evidence_graph),
            "text_claims_summary": {
                "path": str(paths.get("text_claims")) if paths.get("text_claims") else None,
                **(text_claims.get("metrics") or {}),
            },
            "evidence_board_summary": {
                "path": str(paths.get("evidence_board")) if paths.get("evidence_board") else None,
                **(evidence_board.get("metrics") or {}),
                "visual_candidate_summary": evidence_board.get("visual_candidate_summary") or {},
            },
            "board_adjudication": board_adjudication,
            "build_plan_summary": {
                "path": str(paths.get("build_plan")) if paths.get("build_plan") else None,
                **(build_plan.get("metrics") or {}),
                "source_metrics": build_plan.get("source_metrics") or {},
                "raw_seed_coverage": raw_build_plan_seed_coverage,
                "final_seed_coverage": final_build_plan_seed_coverage,
            },
            "builder_capability": {
                "version": GEOSYM_2D_BUILDER_CAPABILITIES.version,
                "supported_actions": sorted(GEOSYM_2D_BUILDER_CAPABILITIES.supported_actions),
                "supported_constraint_types": sorted(GEOSYM_2D_BUILDER_CAPABILITIES.supported_constraint_types),
                "theorem_relation_types": sorted(GEOSYM_2D_BUILDER_CAPABILITIES.theorem_relation_types),
            },
            "extraction_mode": self.extraction_mode,
            "semantic_backend": self.semantic_backend,
            "semantic_model": self.semantic_model,
            "relation_model": self.relation_model,
            "instruction_model": self.instruction_model,
            "staged_outputs_summary": self._staged_outputs_summary(raw_semantic),
            "vlm_transcript_summary": self._vlm_transcript_summary(raw_semantic, paths),
            "instruction_task_packet_summary": {
                "path": str(paths.get("instruction_task_packet")) if paths.get("instruction_task_packet") else None,
                **self._instruction_task_packet_metrics(instruction_task_packet),
                "engine_translation_summary": instruction_task_packet.get("engine_translation_summary") or {},
            },
            "instruction_task_packet_coverage": {
                "raw": raw_task_packet_coverage,
                "final": final_task_packet_coverage,
            },
            "response_template_adherence": {
                "raw": raw_template_adherence,
                "final": final_template_adherence,
            },
            "relation_template_adherence": {
                "raw": raw_relation_template_adherence,
                "final": final_relation_template_adherence,
            },
            "stage_contract": {
                "raw": raw_stage_contract,
                "final": final_stage_contract,
            },
            "semantic_versions": {
                "raw": {
                    "path": str(paths.get("semantic_raw")) if paths.get("semantic_raw") else None,
                    "num_relations": len(semantic_before_repair.get("relations") or []),
                    "num_build_instructions": len(semantic_before_repair.get("build_instructions") or []),
                    "num_candidate_decisions": len(semantic_before_repair.get("candidate_decisions") or []),
                    "num_grounding_decisions": len(semantic_before_repair.get("grounding_decisions") or []),
                    "num_instruction_task_decisions": len(semantic_before_repair.get("instruction_task_decisions") or []),
                    "num_targets": len(semantic_before_repair.get("targets") or []),
                },
                "final": {
                    "path": str(paths.get("semantic")) if paths.get("semantic") else None,
                    "num_relations": len(semantic.get("relations") or []),
                    "num_build_instructions": len(semantic.get("build_instructions") or []),
                    "num_candidate_decisions": len(semantic.get("candidate_decisions") or []),
                    "num_grounding_decisions": len(semantic.get("grounding_decisions") or []),
                    "num_instruction_task_decisions": len(semantic.get("instruction_task_decisions") or []),
                    "num_targets": len(semantic.get("targets") or []),
                },
                "repaired": {
                    "path": str(paths.get("semantic_repaired")) if paths.get("semantic_repaired") else None,
                    "used": bool((instruction_repair or {}).get("used") or (stage_contract_repair or {}).get("used") or (relation_draft_repair or {}).get("used") or ((semantic.get("metadata") or {}).get("semantic_role_canonicalizer") or {}).get("used")),
                },
            },
            "semantic_validation": {
                "valid": semantic_validation.get("valid"),
                "num_warnings": semantic_validation.get("metrics", {}).get("semantic_schema_warnings", 0),
                "warnings": semantic_validation.get("warnings") or [],
            },
            "candidate_usage": {
                "num_script_candidates": candidate_usage.get("num_script_candidates"),
                "num_confirmed_candidates": candidate_usage.get("num_confirmed_candidates"),
                "num_unconfirmed_candidates": candidate_usage.get("num_unconfirmed_candidates"),
                "num_unknown_candidate_refs": candidate_usage.get("num_unknown_candidate_refs"),
                "num_vlm_derived_candidate_refs": candidate_usage.get("num_vlm_derived_candidate_refs"),
                "num_reviewed_candidates": candidate_usage.get("num_reviewed_candidates"),
                "num_unreviewed_candidates": candidate_usage.get("num_unreviewed_candidates"),
                "num_rejected_candidates": candidate_usage.get("num_rejected_candidates"),
                "num_corrected_candidates": candidate_usage.get("num_corrected_candidates"),
                "num_uncertain_candidates": candidate_usage.get("num_uncertain_candidates"),
                "num_decision_conflicts": candidate_usage.get("num_decision_conflicts"),
                "confirmed_candidate_ids": candidate_usage.get("confirmed_candidate_ids") or [],
                "unconfirmed_candidate_ids": candidate_usage.get("unconfirmed_candidate_ids") or [],
                "rejected_candidate_ids": candidate_usage.get("rejected_candidate_ids") or [],
                "corrected_candidate_ids": candidate_usage.get("corrected_candidate_ids") or [],
                "uncertain_candidate_ids": candidate_usage.get("uncertain_candidate_ids") or [],
                "unreviewed_candidate_ids": candidate_usage.get("unreviewed_candidate_ids") or [],
                "decision_conflict_candidate_ids": candidate_usage.get("decision_conflict_candidate_ids") or [],
                "unknown_candidate_refs": candidate_usage.get("unknown_candidate_refs") or [],
                "vlm_derived_candidate_refs": candidate_usage.get("vlm_derived_candidate_refs") or [],
                "candidate_decisions": candidate_usage.get("candidate_decisions") or [],
            },
            "instruction_coverage": instruction_coverage,
            "instruction_repair": {
                "used": bool((instruction_repair or {}).get("used")),
                "num_original_instructions": int((instruction_repair or {}).get("num_original_instructions") or 0),
                "num_added_instructions": int((instruction_repair or {}).get("num_added_instructions") or 0),
                "num_final_instructions": int((instruction_repair or {}).get("num_final_instructions") or 0),
                "reasons": (instruction_repair or {}).get("reasons") or [],
                "blocked": (instruction_repair or {}).get("blocked") or [],
            },
            "stage_contract_repair": {
                "used": bool((stage_contract_repair or {}).get("used")),
                "num_original_decisions": int((stage_contract_repair or {}).get("num_original_decisions") or 0),
                "num_added_decisions": int((stage_contract_repair or {}).get("num_added_decisions") or 0),
                "num_final_decisions": int((stage_contract_repair or {}).get("num_final_decisions") or 0),
                "num_original_instruction_task_decisions": int((stage_contract_repair or {}).get("num_original_instruction_task_decisions") or 0),
                "num_added_instruction_task_decisions": int((stage_contract_repair or {}).get("num_added_instruction_task_decisions") or 0),
                "num_final_instruction_task_decisions": int((stage_contract_repair or {}).get("num_final_instruction_task_decisions") or 0),
                "additions": (stage_contract_repair or {}).get("additions") or [],
                "instruction_task_decision_additions": (stage_contract_repair or {}).get("instruction_task_decision_additions") or [],
            },
            "relation_draft_repair": {
                "used": bool((relation_draft_repair or {}).get("used")),
                "num_original_relations": int((relation_draft_repair or {}).get("num_original_relations") or 0),
                "num_added_relations": int((relation_draft_repair or {}).get("num_added_relations") or 0),
                "num_metadata_patched_relations": int((relation_draft_repair or {}).get("num_metadata_patched_relations") or 0),
                "num_added_buildability": int((relation_draft_repair or {}).get("num_added_buildability") or 0),
                "num_added_builder_mapping": int((relation_draft_repair or {}).get("num_added_builder_mapping") or 0),
                "target_added": bool((relation_draft_repair or {}).get("target_added")),
                "repairable_candidate_ids": (relation_draft_repair or {}).get("repairable_candidate_ids") or [],
                "additions": (relation_draft_repair or {}).get("additions") or [],
                "metadata_patches": (relation_draft_repair or {}).get("metadata_patches") or [],
                "target_addition": (relation_draft_repair or {}).get("target_addition") or {},
            },
            "grounding_decision_repair": {
                "used": bool((grounding_decision_repair or {}).get("used")),
                "num_decision_candidates": int((grounding_decision_repair or {}).get("num_decision_candidates") or 0),
                "num_relation_patches": int((grounding_decision_repair or {}).get("num_relation_patches") or 0),
                "num_visual_id_patches": int((grounding_decision_repair or {}).get("num_visual_id_patches") or 0),
                "num_grounded_patches": int((grounding_decision_repair or {}).get("num_grounded_patches") or 0),
                "patches": (grounding_decision_repair or {}).get("patches") or [],
            },
            "extraction_contract": contract_audit,
            "engine_assessment": engine_assessment,
            "instruction_plan": {
                "buildable": instruction_plan.get("buildable"),
                "reason": instruction_plan.get("reason"),
                "num_instructions": instruction_plan.get("num_instructions"),
                "num_accepted": instruction_plan.get("num_accepted"),
                "num_rejected": instruction_plan.get("num_rejected"),
                "num_synthesized": instruction_plan.get("num_synthesized"),
            },
            "verification": verification,
            "image_alignment": image_report,
            "consistency": consistency_report,
            "llm_judge": judge.to_dict(),
            "acceptance_policy": acceptance_policy,
            "merged_decision": acceptance_policy["decision"],
            "paths": self._report_paths(paths),
        }

    def _acceptance_policy(
        self,
        *,
        hard_valid: bool,
        judge: LLMJudgeSignal,
        instruction_task_packet: Dict[str, Any],
        verification: Dict[str, Any],
    ) -> Dict[str, Any]:
        base_decision = merge_hard_and_llm_signals(hard_valid, judge)
        metrics = instruction_task_packet.get("metrics") or {}
        vlm_required = int(metrics.get("num_vlm_derived_required_tasks") or 0)
        vlm_requirements = int(metrics.get("num_vlm_derived_instruction_requirements") or 0)
        issues = []
        if hard_valid and vlm_required > 0 and base_decision == "auto_accept":
            issues.append(self._issue(
                "acceptance.vlm_derived_constraints_require_review",
                "Canonical reconstruction depends on VLM-derived build constraints; require review before auto-accept",
                severity="warning",
                vlm_derived_required_tasks=vlm_required,
                vlm_derived_instruction_requirements=vlm_requirements,
            ))
            verification.setdefault("issues", []).extend(issues)
            verification.setdefault("metrics", {}).update({
                "acceptance_vlm_derived_requires_review": 1,
                "acceptance_vlm_derived_required_tasks": vlm_required,
                "acceptance_vlm_derived_instruction_requirements": vlm_requirements,
            })
            return {
                "decision": "human_review",
                "base_decision": base_decision,
                "hard_valid": bool(hard_valid),
                "requires_review": True,
                "review_reasons": ["vlm_derived_constraints"],
                "vlm_derived_required_tasks": vlm_required,
                "vlm_derived_instruction_requirements": vlm_requirements,
            }
        verification.setdefault("metrics", {}).update({
            "acceptance_vlm_derived_requires_review": 0,
            "acceptance_vlm_derived_required_tasks": vlm_required,
            "acceptance_vlm_derived_instruction_requirements": vlm_requirements,
        })
        return {
            "decision": base_decision,
            "base_decision": base_decision,
            "hard_valid": bool(hard_valid),
            "requires_review": False,
            "review_reasons": [],
            "vlm_derived_required_tasks": vlm_required,
            "vlm_derived_instruction_requirements": vlm_requirements,
        }

    def _vlm_derived_evidence_issues(self, instruction_task_packet: Dict[str, Any]) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        for item in instruction_task_packet.get("vlm_derived_evidence") or []:
            if not isinstance(item, dict):
                continue
            candidate_id = item.get("candidate_id")
            evidence = item.get("evidence") or {}
            invalid_visual_ids = item.get("invalid_visual_ids") or evidence.get("invalid_visual_ids") or []
            invalid_text_refs = item.get("invalid_text_refs") or evidence.get("invalid_text_refs") or []
            if invalid_visual_ids:
                issues.append(self._issue(
                    "vlm_evidence.invalid_visual_ids",
                    "VLM-derived relation cited visual evidence ids that are not in the script/visual candidate pool",
                    severity="warning",
                    candidate_id=candidate_id,
                    invalid_visual_ids=invalid_visual_ids,
                    relation_type=item.get("relation_type"),
                    task_status=item.get("task_status"),
                ))
            if invalid_text_refs:
                issues.append(self._issue(
                    "vlm_evidence.invalid_text_refs",
                    "VLM-derived relation cited text evidence that is not found in problem/OCR/script text evidence",
                    severity="warning",
                    candidate_id=candidate_id,
                    invalid_text_refs=invalid_text_refs,
                    relation_type=item.get("relation_type"),
                    task_status=item.get("task_status"),
                ))
            if (
                item.get("can_generate_build_instruction")
                and not item.get("passes_evidence_gate")
                and item.get("task_status") == "deferred"
            ):
                issues.append(self._issue(
                    "vlm_evidence.gate_failed",
                    "VLM-derived relation is buildable by GeoSym but lacks valid visual/text evidence, so it was deferred",
                    severity="warning",
                    candidate_id=candidate_id,
                    relation_type=item.get("relation_type"),
                    candidate_object=item.get("candidate_object"),
                    invalid_visual_ids=invalid_visual_ids,
                    invalid_text_refs=invalid_text_refs,
                ))
            if item.get("missing_relation_builder_decision"):
                issues.append(self._issue(
                    "vlm_evidence.missing_relation_builder_decision",
                    "VLM-derived relation has valid semantic/evidence fields but no relation_builder_decisions entry mapping it to current GeoSym builder capability",
                    severity="warning",
                    candidate_id=candidate_id,
                    relation_type=item.get("relation_type"),
                    candidate_object=item.get("candidate_object"),
                    task_status=item.get("task_status"),
                ))
            if item.get("object_normalization_failed"):
                issues.append(self._issue(
                    "vlm_evidence.object_normalization_failed",
                    "VLM-derived relation is supported by the GeoSym builder, but its named objects could not be normalized into builder instructions",
                    severity="warning",
                    candidate_id=candidate_id,
                    relation_type=item.get("relation_type"),
                    candidate_object=item.get("candidate_object"),
                    task_status=item.get("task_status"),
                    object_normalization=item.get("object_normalization") or {},
                ))
        return issues

    def _visual_metrics(self, visual_state: Dict[str, Any]) -> Dict[str, int]:
        topology = visual_state.get("topology") or {}
        return {
            "visual_intersections": len(topology.get("line_intersections") or []),
            "visual_circle_line_hits": len(topology.get("circle_line_hits") or []),
            "visual_endpoints": len(topology.get("endpoints") or []),
        }

    def _script_observation_metrics(self, observations: Dict[str, Any]) -> Dict[str, int]:
        relation_hints = observations.get("relation_hints") or {}
        text_hints = observations.get("text_hints") or {}
        semantic_candidates = observations.get("semantic_candidates") or {}
        engine_suggestions = observations.get("engine_build_suggestions") or {}
        visual_structure = observations.get("visual_structure_candidates") or {}
        visual_summary = visual_structure.get("summary") or {}
        named_proposals = observations.get("named_object_proposals") or {}
        named_summary = named_proposals.get("summary") or {}
        return {
            "script_parallel_hints": len(relation_hints.get("parallel_line_candidates") or []),
            "script_perpendicular_hints": len(relation_hints.get("perpendicular_line_candidates") or []),
            "script_similar_length_hints": len(relation_hints.get("similar_length_line_candidates") or []),
            "script_visual_point_candidates": int(visual_summary.get("num_point_candidates") or len(visual_structure.get("point_candidates") or [])),
            "script_visual_segment_candidates": int(visual_summary.get("num_segment_candidates") or len(visual_structure.get("segment_candidates") or [])),
            "script_visual_angle_candidates": int(visual_summary.get("num_angle_candidates") or len(visual_structure.get("angle_candidates") or [])),
            "script_visual_label_region_candidates": int(visual_summary.get("num_label_region_candidates") or len(visual_structure.get("label_region_candidates") or [])),
            "script_named_object_proposals": int(named_summary.get("num_items") or len(named_proposals.get("items") or [])),
            "script_named_point_label_proposals": int(named_summary.get("num_point_labels") or 0),
            "script_named_segment_proposals": int(named_summary.get("num_segments") or 0),
            "script_labels_mentioned": len(text_hints.get("labels_mentioned") or []),
            "script_segments_mentioned": len(text_hints.get("segments_mentioned") or []),
            "script_numbers_mentioned": len(text_hints.get("numbers_mentioned") or []),
            "script_length_claims": len(text_hints.get("length_claims") or []),
            "script_angle_claims": len(text_hints.get("angle_claims") or []),
            "script_area_claims": len(text_hints.get("area_claims") or []),
            "script_perimeter_claims": len(text_hints.get("perimeter_claims") or []),
            "script_ratio_claims": len(text_hints.get("ratio_claims") or []),
            "script_triangle_claims": len(text_hints.get("triangle_claims") or []),
            "script_target_angle_hints": int(bool(text_hints.get("target_angle_hint"))),
            "script_target_area_hints": int(bool(text_hints.get("target_area_hint"))),
            "script_target_perimeter_hints": int(bool(text_hints.get("target_perimeter_hint"))),
            "script_target_ratio_hints": int(bool(text_hints.get("target_ratio_hint"))),
            "script_text_sources": len(observations.get("text_sources") or []),
            "script_relation_candidates": len(semantic_candidates.get("relation_candidates") or []),
            "script_target_candidates": len(semantic_candidates.get("target_candidates") or []),
            "script_grounding_tasks": len(observations.get("grounding_tasks") or []),
            "script_build_suggestions": len(engine_suggestions.get("suggested_build_instructions") or []),
        }

    def _evidence_graph_metrics(self, evidence_graph: Dict[str, Any]) -> Dict[str, int]:
        summary = self.evidence_graph.summary(evidence_graph)
        return {
            "evidence_visual_lines": int(summary.get("num_visual_lines") or 0),
            "evidence_visual_circles": int(summary.get("num_visual_circles") or 0),
            "evidence_visual_text_regions": int(summary.get("num_visual_text_regions") or 0),
            "evidence_visual_point_candidates": int(summary.get("num_visual_point_candidates") or 0),
            "evidence_visual_segment_candidates": int(summary.get("num_visual_segment_candidates") or 0),
            "evidence_visual_angle_candidates": int(summary.get("num_visual_angle_candidates") or 0),
            "evidence_relation_candidates": int(summary.get("num_relation_candidates") or 0),
            "evidence_target_candidates": int(summary.get("num_target_candidates") or 0),
            "evidence_grounding_tasks": int(summary.get("num_grounding_tasks") or 0),
            "evidence_candidate_review_items": int(summary.get("num_candidate_review_items") or 0),
            "evidence_build_suggestions": int(summary.get("num_build_suggestions") or 0),
            "evidence_build_tasks": int(summary.get("num_build_tasks") or 0),
            "evidence_named_object_proposals": int(summary.get("num_named_object_proposals") or 0),
            "evidence_named_point_label_proposals": int(summary.get("num_named_point_label_proposals") or 0),
            "evidence_named_segment_proposals": int(summary.get("num_named_segment_proposals") or 0),
            "evidence_relation_mapping_contract_entries": len(summary.get("relation_mapping_contract") or []),
        }

    def _board_adjudication_metrics(self, adjudication: Dict[str, Any]) -> Dict[str, int | float]:
        metrics = adjudication.get("metrics") or {}
        return {
            "board_named_object_cards": int(metrics.get("num_named_object_cards") or 0),
            "board_reviewed_named_object_cards": int(metrics.get("num_reviewed_named_object_cards") or 0),
            "board_missing_named_object_cards": int(metrics.get("num_missing_named_object_cards") or 0),
            "board_named_object_card_coverage": float(metrics.get("named_object_card_coverage", 1.0)),
            "board_relation_cards": int(metrics.get("num_relation_cards") or 0),
            "board_reviewed_relation_cards": int(metrics.get("num_reviewed_relation_cards") or 0),
            "board_missing_relation_cards": int(metrics.get("num_missing_relation_cards") or 0),
            "board_relation_card_coverage": float(metrics.get("relation_card_coverage", 1.0)),
            "board_target_cards": int(metrics.get("num_target_cards") or 0),
            "board_reviewed_target_cards": int(metrics.get("num_reviewed_target_cards") or 0),
            "board_missing_target_cards": int(metrics.get("num_missing_target_cards") or 0),
            "board_target_card_coverage": float(metrics.get("target_card_coverage", 1.0)),
        }

    def _build_plan_seed_coverage_metrics(self, coverage: Dict[str, Any], *, prefix: str) -> Dict[str, int | float]:
        return {
            f"{prefix}_expected_instructions": int(coverage.get("num_expected_instructions") or 0),
            f"{prefix}_covered_instructions": int(coverage.get("num_covered_instructions") or 0),
            f"{prefix}_missing_instructions": int(coverage.get("num_missing_instructions") or 0),
            f"{prefix}_coverage": float(coverage.get("coverage", 1.0)),
        }

    def _relation_builder_decision_metrics(self, analysis: Dict[str, Any], *, prefix: str) -> Dict[str, int | float]:
        return {
            f"{prefix}_expected_decisions": int(analysis.get("num_expected_relation_builder_decisions") or 0),
            f"{prefix}_decisions": int(analysis.get("num_relation_builder_decisions") or 0),
            f"{prefix}_missing_decisions": int(analysis.get("num_missing_relation_builder_decisions") or 0),
            f"{prefix}_unknown_decisions": int(analysis.get("num_unknown_relation_builder_decisions") or 0),
            f"{prefix}_inconsistent_decisions": int(analysis.get("num_inconsistent_relation_builder_decisions") or 0),
            f"{prefix}_coverage": float(analysis.get("coverage", 1.0)),
        }

    def _candidate_usage_metrics(self, usage: Dict[str, Any]) -> Dict[str, int]:
        return {
            "candidate_script_total": int(usage.get("num_script_candidates") or 0),
            "candidate_confirmed": int(usage.get("num_confirmed_candidates") or 0),
            "candidate_unconfirmed": int(usage.get("num_unconfirmed_candidates") or 0),
            "candidate_reviewed": int(usage.get("num_reviewed_candidates") or 0),
            "candidate_unreviewed": int(usage.get("num_unreviewed_candidates") or 0),
            "candidate_rejected": int(usage.get("num_rejected_candidates") or 0),
            "candidate_corrected": int(usage.get("num_corrected_candidates") or 0),
            "candidate_uncertain": int(usage.get("num_uncertain_candidates") or 0),
            "candidate_decision_conflicts": int(usage.get("num_decision_conflicts") or 0),
            "candidate_unknown_refs": int(usage.get("num_unknown_candidate_refs") or 0),
            "candidate_vlm_derived_refs": int(usage.get("num_vlm_derived_candidate_refs") or 0),
        }

    def _instruction_coverage_metrics(self, coverage: Dict[str, Any]) -> Dict[str, int]:
        return {
            "instruction_required_constraints": int(coverage.get("num_required_constraints") or 0),
            "instruction_covered_constraints": int(coverage.get("num_covered_constraints") or 0),
            "instruction_missing_constraints": int(coverage.get("num_missing_constraints") or 0),
        }

    def _target_resolution_metrics(
        self,
        semantic: Dict[str, Any],
        instruction_task_packet: Dict[str, Any],
        instruction_plan: Dict[str, Any],
    ) -> Dict[str, int]:
        semantic_targets = self._target_items(semantic)
        usable_targets = [target for target in semantic_targets if self._is_usable_target(target)]
        symbolic_targets = [target for target in semantic_targets if self._is_symbolic_target(target, semantic)]
        required_target_tasks = [
            task for task in instruction_task_packet.get("required_tasks") or []
            if task.get("candidate_type") == "target"
        ]
        deferred_target_tasks = [
            task for task in instruction_task_packet.get("deferred_tasks") or []
            if task.get("candidate_type") == "target"
        ]
        template_target = (instruction_task_packet.get("response_template") or {}).get("target") or {}
        plan_target = ((instruction_plan.get("constraint_graph") or {}).get("target") or {})
        unresolved = bool(
            semantic_targets
            and not usable_targets
            and not symbolic_targets
            and not required_target_tasks
            and not self._is_usable_target(template_target)
            and not self._is_usable_target(plan_target)
        )
        return {
            "target_semantic_targets": len(semantic_targets),
            "target_usable_targets": len(usable_targets),
            "target_symbolic_targets": len(symbolic_targets),
            "target_symbolic_resolved": int(bool(symbolic_targets)),
            "target_required_tasks": len(required_target_tasks),
            "target_deferred_tasks": len(deferred_target_tasks),
            "target_response_template_usable": int(self._is_usable_target(template_target)),
            "target_instruction_plan_usable": int(self._is_usable_target(plan_target)),
            "target_unresolved": int(unresolved),
        }

    def _target_items(self, semantic: Dict[str, Any]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        target = semantic.get("target")
        if isinstance(target, dict) and target:
            items.append(target)
        for item in semantic.get("targets") or []:
            if isinstance(item, dict) and item:
                items.append(item)
        out: List[Dict[str, Any]] = []
        seen = set()
        for item in items:
            key = (
                str(item.get("candidate_id") or item.get("from_candidate") or ""),
                str(item.get("type") or ""),
                str(item.get("object") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    def _is_usable_target(self, target: Dict[str, Any]) -> bool:
        if not isinstance(target, dict) or not target:
            return False
        target_type = str(target.get("type") or "").strip().lower()
        obj = str(target.get("object") or "").strip()
        if not target_type or target_type == "unknown" or not obj or obj.upper() == "X":
            return False
        if target_type == "length":
            clean = "".join(ch for ch in obj.upper() if ch.isalnum())
            return len(clean) >= 2 and clean[:2].isalpha()
        return target_type in {"angle", "arc_measure", "area", "entity_area", "shadow_area", "perimeter", "ratio"}

    def _is_symbolic_target(self, target: Dict[str, Any], semantic: Dict[str, Any]) -> bool:
        symbol = self._target_symbol(target)
        if not symbol:
            return False
        return self._semantic_mentions_symbol(semantic, symbol)

    def _target_symbol(self, target: Dict[str, Any]) -> str:
        if not isinstance(target, dict):
            return ""
        for field in ["value_symbol", "object", "value"]:
            raw = str(target.get(field) or "").strip()
            if self._is_symbolic_placeholder(raw):
                return raw.lower()
        return ""

    def _is_symbolic_placeholder(self, raw: str) -> bool:
        clean = str(raw or "").strip().strip("$").strip()
        if len(clean) != 1 or not clean.isalpha():
            return False
        return clean.lower() in {"x", "y", "z", "t", "r"}

    def _semantic_mentions_symbol(self, semantic: Dict[str, Any], symbol: str) -> bool:
        needle = str(symbol or "").lower()
        if not needle:
            return False
        containers: List[Any] = []
        containers.extend(semantic.get("relations") or [])
        containers.extend(semantic.get("targets") or [])
        target = semantic.get("target")
        if isinstance(target, dict):
            containers.append(target)
        for item in containers:
            if not isinstance(item, dict):
                continue
            for field in ["value", "expression", "raw", "text", "supporting_text"]:
                if self._text_has_symbol(item.get(field), needle):
                    return True
            evidence = item.get("evidence") or {}
            if isinstance(evidence, dict):
                for field in ["supporting_text", "text_refs"]:
                    if self._text_has_symbol(evidence.get(field), needle):
                        return True
        return False

    def _text_has_symbol(self, value: Any, symbol: str) -> bool:
        if isinstance(value, list):
            return any(self._text_has_symbol(item, symbol) for item in value)
        text = str(value or "").lower()
        if not text:
            return False
        for idx, ch in enumerate(text):
            if ch != symbol:
                continue
            prev_ok = idx == 0 or not text[idx - 1].isalpha()
            next_ok = idx == len(text) - 1 or not text[idx + 1].isalpha()
            if prev_ok and next_ok:
                return True
        return False

    def _formal_metrics(self, formal: Dict[str, Any]) -> Dict[str, int]:
        solver_coverage = formal.get("solver_coverage") or {}
        return {
            "formal_relations": len(formal.get("relations") or []),
            "formal_semantic_graph_available": int(bool(formal.get("relations"))),
            "formal_semantic_graph_theorem_only": int(formal.get("mode") == "theorem_only"),
            "formal_requires_theorem_solver": int(bool(formal.get("requires_theorem_solver"))),
            "formal_coordinate_relation_types": len(formal.get("coordinate_relation_types") or []),
            "formal_theorem_relation_types": len(formal.get("theorem_relation_types") or []),
            "formal_unsupported_relation_types": len(formal.get("unsupported_relation_types") or []),
            "formal_solver_coverage_available": int(bool(solver_coverage)),
            "formal_solver_coordinate_covered": int(bool(solver_coverage.get("coordinate_builder_covered"))),
            "formal_solver_has_coordinate_scaffold": int(bool(solver_coverage.get("has_coordinate_scaffold"))),
            "formal_solver_requires_equation_solver": int(bool(solver_coverage.get("requires_equation_solver"))),
            "formal_solver_requires_relation_schema_extension": int(bool(solver_coverage.get("requires_relation_schema_extension"))),
            "formal_solver_symbolic_variables": len(solver_coverage.get("symbolic_variables") or []),
            "formal_partial_coordinate_scaffold": int(
                formal.get("mode") == "coordinate_buildable"
                and bool(formal.get("requires_theorem_solver"))
            ),
        }

    def _instruction_repair_metrics(self, repair: Dict[str, Any]) -> Dict[str, int]:
        return {
            "instruction_repair_used": int(bool(repair.get("used"))),
            "instruction_repair_added": int(repair.get("num_added_instructions") or 0),
            "instruction_repair_original": int(repair.get("num_original_instructions") or 0),
            "instruction_repair_final": int(repair.get("num_final_instructions") or 0),
            "instruction_repair_blocked_by_task_decision": int(len(repair.get("blocked") or [])),
            "instruction_repair_blocked_by_relation_builder_decision": int(
                sum(1 for item in repair.get("blocked") or [] if item.get("blocked_by_relation_builder_decision"))
            ),
            "instruction_repair_vlm_derived_candidates": int(len(repair.get("vlm_derived_candidate_ids") or [])),
        }

    def _semantic_role_metrics(self, semantic: Dict[str, Any]) -> Dict[str, int]:
        role = (semantic.get("metadata") or {}).get("semantic_role_canonicalizer") or {}
        return {
            "semantic_role_canonicalizer_used": int(bool(role.get("used"))),
            "semantic_role_canonicalizer_changes": int(role.get("num_changes") or 0),
            "semantic_role_canonicalizer_roles": int(len(role.get("role_map") or {})),
        }

    def _merge_stage_contract_repairs(self, first: Dict[str, Any], second: Dict[str, Any]) -> Dict[str, Any]:
        first = first or {}
        second = second or {}
        semantic = second.get("semantic") or first.get("semantic") or {}
        return {
            "used": bool(first.get("used") or second.get("used")),
            "num_original_decisions": int(first.get("num_original_decisions") or second.get("num_original_decisions") or 0),
            "num_added_decisions": int(first.get("num_added_decisions") or 0) + int(second.get("num_added_decisions") or 0),
            "num_final_decisions": int(second.get("num_final_decisions") or first.get("num_final_decisions") or 0),
            "num_original_instruction_task_decisions": int(first.get("num_original_instruction_task_decisions") or second.get("num_original_instruction_task_decisions") or 0),
            "num_added_instruction_task_decisions": int(first.get("num_added_instruction_task_decisions") or 0) + int(second.get("num_added_instruction_task_decisions") or 0),
            "num_final_instruction_task_decisions": int(second.get("num_final_instruction_task_decisions") or first.get("num_final_instruction_task_decisions") or 0),
            "additions": (first.get("additions") or []) + (second.get("additions") or []),
            "instruction_task_decision_additions": (first.get("instruction_task_decision_additions") or []) + (second.get("instruction_task_decision_additions") or []),
            "semantic": semantic,
        }

    def _stage_contract_repair_metrics(self, repair: Dict[str, Any]) -> Dict[str, int]:
        return {
            "stage_contract_repair_used": int(bool(repair.get("used"))),
            "stage_contract_repair_added_decisions": int(repair.get("num_added_decisions") or 0),
            "stage_contract_repair_original_decisions": int(repair.get("num_original_decisions") or 0),
            "stage_contract_repair_final_decisions": int(repair.get("num_final_decisions") or 0),
            "stage_contract_repair_added_instruction_task_decisions": int(repair.get("num_added_instruction_task_decisions") or 0),
            "stage_contract_repair_final_instruction_task_decisions": int(repair.get("num_final_instruction_task_decisions") or 0),
        }

    def _relation_draft_repair_metrics(self, repair: Dict[str, Any]) -> Dict[str, int]:
        return {
            "relation_draft_repair_used": int(bool(repair.get("used"))),
            "relation_draft_repair_added_relations": int(repair.get("num_added_relations") or 0),
            "relation_draft_repair_metadata_patched_relations": int(repair.get("num_metadata_patched_relations") or 0),
            "relation_draft_repair_added_buildability": int(repair.get("num_added_buildability") or 0),
            "relation_draft_repair_added_builder_mapping": int(repair.get("num_added_builder_mapping") or 0),
            "relation_draft_repair_original_relations": int(repair.get("num_original_relations") or 0),
            "relation_draft_repair_target_added": int(bool(repair.get("target_added"))),
        }

    def _grounding_decision_repair_metrics(self, repair: Dict[str, Any]) -> Dict[str, int]:
        return {
            "grounding_decision_repair_used": int(bool(repair.get("used"))),
            "grounding_decision_repair_decision_candidates": int(repair.get("num_decision_candidates") or 0),
            "grounding_decision_repair_relation_patches": int(repair.get("num_relation_patches") or 0),
            "grounding_decision_repair_visual_id_patches": int(repair.get("num_visual_id_patches") or 0),
            "grounding_decision_repair_grounded_patches": int(repair.get("num_grounded_patches") or 0),
        }

    def _contract_audit_metrics(self, contract: Dict[str, Any]) -> Dict[str, int]:
        metrics = contract.get("metrics") or {}
        return {
            "contract_valid": int(bool(contract.get("valid"))),
            "contract_errors": int(contract.get("num_errors") or 0),
            "contract_warnings": int(contract.get("num_warnings") or 0),
            "contract_buildable_script_candidates": int(metrics.get("buildable_script_candidates") or 0),
            "contract_missing_build_constraints": int(metrics.get("missing_build_constraints") or 0),
            "contract_rejected_build_instructions": int(metrics.get("rejected_build_instructions") or 0),
            "contract_synthesized_build_instructions": int(metrics.get("synthesized_build_instructions") or 0),
            "contract_unreviewed_script_candidates": int(metrics.get("unreviewed_script_candidates") or 0),
            "contract_candidate_decision_conflicts": int(metrics.get("candidate_decision_conflicts") or 0),
            "contract_required_build_tasks": int(metrics.get("required_build_tasks") or 0),
            "contract_covered_build_tasks": int(metrics.get("covered_build_tasks") or 0),
            "contract_missing_build_tasks": int(metrics.get("missing_build_tasks") or 0),
            "contract_required_instruction_requirements": int(metrics.get("required_instruction_requirements") or 0),
            "contract_covered_instruction_requirements": int(metrics.get("covered_instruction_requirements") or 0),
            "contract_missing_instruction_requirements": int(metrics.get("missing_instruction_requirements") or 0),
        }

    def _staged_output_metrics(self, semantic: Dict[str, Any]) -> Dict[str, int]:
        staged = semantic.get("staged_outputs") or {}
        relation_pass = staged.get("relation_pass") or {}
        instruction_pass = staged.get("instruction_pass") or {}
        instruction_task_packet = staged.get("instruction_task_packet") or {}
        build_plan_seed = staged.get("build_plan_seed") or {}
        relation_validation = staged.get("relation_pass_validation") or {}
        instruction_validation = staged.get("instruction_pass_validation") or {}
        instruction_error = staged.get("instruction_pass_error") or ((instruction_pass.get("metadata") or {}).get("instruction_pass_error") or {})
        relation_validation_metrics = relation_validation.get("metrics") or {}
        instruction_validation_metrics = instruction_validation.get("metrics") or {}
        adapter_metrics = ((relation_pass.get("metadata") or {}).get("relation_pass_adapter") or {})
        instruction_adapter_metrics = ((instruction_pass.get("metadata") or {}).get("instruction_pass_adapter") or {})
        packet_metrics = self._instruction_task_packet_metrics(instruction_task_packet, prefix="staged_instruction_task_packet")
        return {
            "staged_relation_pass_available": int(bool(relation_pass)),
            "staged_instruction_pass_available": int(bool(instruction_pass)),
            "staged_instruction_pass_error": int(bool(instruction_error)),
            "staged_instruction_task_packet_available": int(bool(instruction_task_packet)),
            "staged_build_plan_seed_available": int(bool(build_plan_seed)),
            "staged_build_plan_seed_tasks": len(build_plan_seed.get("tasks") or []),
            "staged_build_plan_seed_expected_instructions": int((build_plan_seed.get("metrics") or {}).get("num_expected_instructions") or 0),
            "staged_relation_pass_schema_warnings": int(relation_validation_metrics.get("semantic_schema_warnings") or 0),
            "staged_instruction_pass_schema_warnings": int(instruction_validation_metrics.get("semantic_schema_warnings") or 0),
            "staged_relation_pass_relations": len(relation_pass.get("relations") or []),
            "staged_relation_pass_bindings": len(relation_pass.get("bindings") or []),
            "staged_relation_pass_named_object_bindings": len(relation_pass.get("named_object_bindings") or []),
            "staged_relation_pass_targets": len(relation_pass.get("targets") or []),
            "staged_relation_pass_candidate_decisions": len(relation_pass.get("candidate_decisions") or []),
            "staged_relation_pass_relation_builder_decisions": len(relation_pass.get("relation_builder_decisions") or []),
            "staged_relation_pass_leaked_instructions": len(relation_pass.get("build_instructions") or []),
            "staged_relation_pass_adapter_used": int(bool(adapter_metrics.get("used"))),
            "staged_relation_pass_adapter_relation_patches": int(adapter_metrics.get("num_relation_patches") or 0),
            "staged_relation_pass_adapter_candidate_id_patches": int(adapter_metrics.get("num_candidate_id_patches") or 0),
            "staged_relation_pass_adapter_buildability_patches": int(adapter_metrics.get("num_buildability_patches") or 0),
            "staged_relation_pass_adapter_builder_mapping_patches": int(adapter_metrics.get("num_builder_mapping_patches") or 0),
            "staged_relation_pass_adapter_engine_metadata_patches": int(adapter_metrics.get("engine_metadata_patches") or 0),
            "staged_relation_pass_adapter_engine_metadata_buildability_patches": int(adapter_metrics.get("engine_metadata_buildability_patches") or 0),
            "staged_relation_pass_adapter_engine_metadata_builder_mapping_patches": int(adapter_metrics.get("engine_metadata_builder_mapping_patches") or 0),
            "staged_relation_pass_adapter_capability_decision_patches": int(adapter_metrics.get("capability_decision_patches") or 0),
            "staged_relation_pass_adapter_capability_decisions_added": int(adapter_metrics.get("capability_decisions_added") or 0),
            "staged_relation_pass_adapter_capability_decisions_corrected": int(adapter_metrics.get("capability_decisions_corrected") or 0),
            "staged_instruction_pass_instructions": len(instruction_pass.get("build_instructions") or []),
            "staged_instruction_pass_targets": len(instruction_pass.get("targets") or []),
            "staged_instruction_pass_adapter_used": int(bool(instruction_adapter_metrics.get("used"))),
            "staged_instruction_pass_adapter_instruction_patches": int(instruction_adapter_metrics.get("num_instruction_patches") or 0),
            "staged_instruction_pass_adapter_from_candidate_patches": int(instruction_adapter_metrics.get("num_from_candidate_patches") or 0),
            "staged_instruction_pass_adapter_field_patches": int(instruction_adapter_metrics.get("num_field_patches") or 0),
            "staged_instruction_pass_extra_relations": len(instruction_pass.get("relations") or []),
            "staged_instruction_pass_extra_candidate_decisions": len(instruction_pass.get("candidate_decisions") or []),
            **packet_metrics,
        }

    def _instruction_task_packet_metrics(self, packet: Dict[str, Any], prefix: str = "instruction_task_packet") -> Dict[str, int]:
        metrics = packet.get("metrics") or {}
        return {
            f"{prefix}_required_tasks": int(metrics.get("num_required_tasks") or len(packet.get("required_tasks") or [])),
            f"{prefix}_task_decision_template": int(
                metrics.get("num_instruction_task_decision_template")
                or len(packet.get("instruction_task_decision_template") or [])
                or len((packet.get("response_template") or {}).get("instruction_task_decisions") or [])
            ),
            f"{prefix}_deferred_tasks": int(metrics.get("num_deferred_tasks") or len(packet.get("deferred_tasks") or [])),
            f"{prefix}_required_actions": int(
                metrics.get("num_required_actions")
                or sum(len(item.get("required_actions") or []) for item in packet.get("required_tasks") or [])
            ),
            f"{prefix}_instruction_requirements": int(
                metrics.get("num_instruction_requirements")
                or sum(len(item.get("instruction_requirements") or []) for item in packet.get("required_tasks") or [])
            ),
            f"{prefix}_template_build_instructions": int(
                metrics.get("num_template_build_instructions")
                or len((packet.get("response_template") or {}).get("build_instructions") or [])
            ),
            f"{prefix}_corrected_relation_tasks": int(metrics.get("num_corrected_relation_tasks") or 0),
            f"{prefix}_builder_mapped_tasks": int(metrics.get("num_builder_mapped_tasks") or 0),
            f"{prefix}_relation_builder_decision_blocked_tasks": int(metrics.get("num_relation_builder_decision_blocked_tasks") or 0),
            f"{prefix}_relation_builder_decision_theorem_only_blocked_tasks": int(metrics.get("num_relation_builder_decision_theorem_only_blocked_tasks") or 0),
            f"{prefix}_relation_builder_decision_unexpected_blocked_tasks": int(metrics.get("num_relation_builder_decision_unexpected_blocked_tasks") or 0),
            f"{prefix}_vlm_derived_required_tasks": int(metrics.get("num_vlm_derived_required_tasks") or 0),
            f"{prefix}_vlm_derived_deferred_tasks": int(metrics.get("num_vlm_derived_deferred_tasks") or 0),
            f"{prefix}_vlm_derived_instruction_requirements": int(metrics.get("num_vlm_derived_instruction_requirements") or 0),
            f"{prefix}_vlm_derived_evidence_items": int(
                metrics.get("num_vlm_derived_evidence_items")
                or len(packet.get("vlm_derived_evidence") or [])
            ),
            f"{prefix}_vlm_derived_evidence_with_visual_ids": int(metrics.get("num_vlm_derived_evidence_with_visual_ids") or 0),
            f"{prefix}_vlm_derived_evidence_with_named_object_binding_visual_ids": int(metrics.get("num_vlm_derived_evidence_with_named_object_binding_visual_ids") or 0),
            f"{prefix}_vlm_derived_evidence_named_object_binding_visual_ids": int(metrics.get("num_vlm_derived_evidence_named_object_binding_visual_ids") or 0),
            f"{prefix}_vlm_derived_evidence_invalid_visual_ids": int(metrics.get("num_vlm_derived_evidence_invalid_visual_ids") or 0),
            f"{prefix}_vlm_derived_evidence_invalid_visual_id_tasks": int(metrics.get("num_vlm_derived_evidence_invalid_visual_id_tasks") or 0),
            f"{prefix}_vlm_derived_evidence_invalid_text_refs": int(metrics.get("num_vlm_derived_evidence_invalid_text_refs") or 0),
            f"{prefix}_vlm_derived_evidence_invalid_text_ref_tasks": int(metrics.get("num_vlm_derived_evidence_invalid_text_ref_tasks") or 0),
            f"{prefix}_vlm_derived_evidence_invalid_text_ref_blocking_tasks": int(metrics.get("num_vlm_derived_evidence_invalid_text_ref_blocking_tasks") or 0),
            f"{prefix}_vlm_derived_evidence_buildable": int(
                metrics.get("num_vlm_derived_evidence_required_after_gate")
                or metrics.get("num_vlm_derived_evidence_buildable")
                or 0
            ),
            f"{prefix}_vlm_derived_evidence_can_generate_build_instruction": int(metrics.get("num_vlm_derived_evidence_can_generate_build_instruction") or 0),
            f"{prefix}_vlm_derived_evidence_engine_can_encode_relation": int(metrics.get("num_vlm_derived_evidence_engine_can_encode_relation") or 0),
            f"{prefix}_vlm_derived_object_normalization_failed": int(metrics.get("num_vlm_derived_object_normalization_failed") or 0),
            f"{prefix}_vlm_derived_missing_relation_builder_decisions": int(metrics.get("num_vlm_derived_missing_relation_builder_decisions") or 0),
            f"{prefix}_vlm_derived_blocked_by_relation_builder_decision": int(metrics.get("num_vlm_derived_blocked_by_relation_builder_decision") or 0),
            f"{prefix}_vlm_derived_theorem_only_deferred": int(metrics.get("num_vlm_derived_theorem_only_deferred") or 0),
            f"{prefix}_vlm_derived_unexpected_deferred": int(metrics.get("num_vlm_derived_unexpected_deferred") or 0),
            f"{prefix}_vlm_derived_evidence_gate_passed": int(metrics.get("num_vlm_derived_evidence_gate_passed") or 0),
            f"{prefix}_vlm_derived_evidence_required_after_gate": int(metrics.get("num_vlm_derived_evidence_required_after_gate") or 0),
            f"{prefix}_capability_guide_relation_rules": int(
                metrics.get("num_capability_guide_relation_rules")
                or len((packet.get("capability_guide") or {}).get("relation_to_instruction_rules") or [])
            ),
            f"{prefix}_capability_guide_task_decisions": int(
                metrics.get("num_capability_guide_task_decisions")
                or len((packet.get("capability_guide") or {}).get("task_decision_guide") or [])
            ),
            f"{prefix}_capability_guide_deferred_tasks": int(
                metrics.get("num_capability_guide_deferred_tasks")
                or len((packet.get("capability_guide") or {}).get("deferred_task_guide") or [])
            ),
            f"{prefix}_engine_translation_contract_items": int(
                metrics.get("num_engine_translation_contract_items")
                or len((packet.get("engine_translation_contract") or {}).get("items") or [])
            ),
            f"{prefix}_engine_translation_contract_emit_items": int(metrics.get("num_engine_translation_contract_emit_items") or 0),
            f"{prefix}_engine_translation_contract_blocked_items": int(metrics.get("num_engine_translation_contract_blocked_items") or 0),
            f"{prefix}_engine_translation_missing_decision_items": int(metrics.get("num_engine_translation_summary_missing_decision_items") or 0),
            f"{prefix}_engine_translation_theorem_only_items": int(metrics.get("num_engine_translation_summary_theorem_only_items") or 0),
            f"{prefix}_engine_translation_unsupported_items": int(metrics.get("num_engine_translation_summary_unsupported_items") or 0),
            f"{prefix}_engine_translation_evidence_gate_failed_items": int(metrics.get("num_engine_translation_summary_evidence_gate_failed_items") or 0),
            f"{prefix}_engine_translation_object_normalization_failed_items": int(metrics.get("num_engine_translation_summary_object_normalization_failed_items") or 0),
        }

    def _build_plan_seed_from_staged(self, semantic: Dict[str, Any]) -> Dict[str, Any]:
        staged = semantic.get("staged_outputs") or {}
        seed = staged.get("build_plan_seed")
        return seed if isinstance(seed, dict) else {}

    def _task_packet_coverage_metrics(self, coverage: Dict[str, Any], prefix: str) -> Dict[str, int]:
        return {
            f"{prefix}_required_tasks": int(coverage.get("num_required_tasks") or 0),
            f"{prefix}_covered_tasks": int(coverage.get("num_covered_tasks") or 0),
            f"{prefix}_missing_tasks": int(coverage.get("num_missing_tasks") or 0),
            f"{prefix}_required_actions": int(coverage.get("num_required_actions") or 0),
            f"{prefix}_covered_actions": int(coverage.get("num_covered_actions") or 0),
            f"{prefix}_missing_actions": int(coverage.get("num_missing_actions") or 0),
            f"{prefix}_instruction_requirements": int(coverage.get("num_instruction_requirements") or 0),
            f"{prefix}_covered_instruction_requirements": int(coverage.get("num_covered_instruction_requirements") or 0),
            f"{prefix}_missing_instruction_requirements": int(coverage.get("num_missing_instruction_requirements") or 0),
        }

    def _template_adherence_metrics(self, adherence: Dict[str, Any], prefix: str) -> Dict[str, int]:
        return {
            f"{prefix}_instructions": int(adherence.get("num_template_instructions") or 0),
            f"{prefix}_actual_instructions": int(adherence.get("num_actual_instructions") or 0),
            f"{prefix}_matched_instructions": int(adherence.get("num_matched_template_instructions") or 0),
            f"{prefix}_missing_instructions": int(adherence.get("num_missing_template_instructions") or 0),
            f"{prefix}_extra_instructions": int(adherence.get("num_extra_instructions") or 0),
            f"{prefix}_instruction_task_decisions": int(adherence.get("num_template_instruction_task_decisions") or 0),
            f"{prefix}_actual_instruction_task_decisions": int(adherence.get("num_actual_instruction_task_decisions") or 0),
            f"{prefix}_matched_instruction_task_decisions": int(adherence.get("num_matched_template_instruction_task_decisions") or 0),
            f"{prefix}_missing_instruction_task_decisions": int(adherence.get("num_missing_template_instruction_task_decisions") or 0),
            f"{prefix}_extra_instruction_task_decisions": int(adherence.get("num_extra_instruction_task_decisions") or 0),
        }

    def _relation_template_adherence_metrics(self, adherence: Dict[str, Any], prefix: str) -> Dict[str, int]:
        return {
            f"{prefix}_candidate_decisions": int(adherence.get("num_template_candidate_decisions") or 0),
            f"{prefix}_actual_candidate_decisions": int(adherence.get("num_actual_candidate_decisions") or 0),
            f"{prefix}_matched_candidate_decisions": int(adherence.get("num_matched_template_candidate_decisions") or 0),
            f"{prefix}_missing_candidate_decisions": int(adherence.get("num_missing_template_candidate_decisions") or 0),
            f"{prefix}_extra_candidate_decisions": int(adherence.get("num_extra_candidate_decisions") or 0),
            f"{prefix}_relations": int(adherence.get("num_template_relations") or 0),
            f"{prefix}_actual_relations": int(adherence.get("num_actual_relations") or 0),
            f"{prefix}_matched_relations": int(adherence.get("num_matched_template_relations") or 0),
            f"{prefix}_missing_relations": int(adherence.get("num_missing_template_relations") or 0),
            f"{prefix}_target_expected": int(bool(adherence.get("template_target_expected"))),
            f"{prefix}_target_matched": int(bool(adherence.get("template_target_matched"))),
            f"{prefix}_missing_target_refs": int(adherence.get("num_missing_template_target_refs") or 0),
            f"{prefix}_grounding_decisions": int(adherence.get("num_template_grounding_decisions") or 0),
            f"{prefix}_actual_grounding_decisions": int(adherence.get("num_actual_grounding_decisions") or 0),
            f"{prefix}_matched_grounding_decisions": int(adherence.get("num_matched_template_grounding_decisions") or 0),
            f"{prefix}_missing_grounding_decisions": int(adherence.get("num_missing_template_grounding_decisions") or 0),
            f"{prefix}_extra_grounding_decisions": int(adherence.get("num_extra_grounding_decisions") or 0),
            f"{prefix}_builder_decisions": int(adherence.get("num_template_builder_decisions") or 0),
            f"{prefix}_actual_builder_decisions": int(adherence.get("num_actual_builder_decisions") or 0),
            f"{prefix}_matched_builder_decisions": int(adherence.get("num_matched_template_builder_decisions") or 0),
            f"{prefix}_missing_builder_decisions": int(adherence.get("num_missing_template_builder_decisions") or 0),
            f"{prefix}_extra_builder_decisions": int(adherence.get("num_extra_builder_decisions") or 0),
        }

    def _stage_contract_metrics(self, contract: Dict[str, Any], prefix: str) -> Dict[str, int]:
        stage1 = contract.get("stage1") or {}
        stage2 = contract.get("stage2") or {}
        return {
            f"{prefix}_valid": int(bool(contract.get("valid"))),
            f"{prefix}_candidates": int(stage1.get("num_candidates") or 0),
            f"{prefix}_reviewed_candidates": int(stage1.get("num_reviewed_candidates") or 0),
            f"{prefix}_missing_candidate_decisions": int(stage1.get("num_missing_candidate_decisions") or 0),
            f"{prefix}_unknown_candidate_decisions": int(stage1.get("num_unknown_candidate_decisions") or 0),
            f"{prefix}_grounding_tasks": int(stage1.get("num_grounding_tasks") or 0),
            f"{prefix}_reviewed_grounding_tasks": int(stage1.get("num_reviewed_grounding_tasks") or 0),
            f"{prefix}_missing_grounding_decisions": int(stage1.get("num_missing_grounding_decisions") or 0),
            f"{prefix}_unknown_grounding_decisions": int(stage1.get("num_unknown_grounding_decisions") or 0),
            f"{prefix}_invalid_grounding_visual_ids": int(stage1.get("num_invalid_grounding_visual_ids") or 0),
            f"{prefix}_invalid_grounding_decisions": int(stage1.get("num_invalid_grounding_decisions") or 0),
            f"{prefix}_named_object_proposals": int(stage1.get("num_named_object_proposals") or 0),
            f"{prefix}_reviewed_named_object_bindings": int(stage1.get("num_reviewed_named_object_bindings") or 0),
            f"{prefix}_missing_named_object_bindings": int(stage1.get("num_missing_named_object_bindings") or 0),
            f"{prefix}_unknown_named_object_bindings": int(stage1.get("num_unknown_named_object_bindings") or 0),
            f"{prefix}_invalid_named_object_visual_ids": int(stage1.get("num_invalid_named_object_visual_ids") or 0),
            f"{prefix}_invalid_named_object_bindings": int(stage1.get("num_invalid_named_object_bindings") or 0),
            f"{prefix}_expected_relation_refs": int(stage1.get("num_expected_relation_refs") or 0),
            f"{prefix}_covered_relation_refs": int(stage1.get("num_covered_relation_refs") or 0),
            f"{prefix}_missing_relation_refs": int(stage1.get("num_missing_relation_refs") or 0),
            f"{prefix}_expected_relation_buildability_refs": int(stage1.get("num_expected_relation_buildability_refs") or 0),
            f"{prefix}_missing_relation_buildability_refs": int(stage1.get("num_missing_relation_buildability_refs") or 0),
            f"{prefix}_missing_relation_builder_mapping_refs": int(stage1.get("num_missing_relation_builder_mapping_refs") or 0),
            f"{prefix}_inconsistent_relation_buildability": int(stage1.get("num_inconsistent_relation_buildability") or 0),
            f"{prefix}_expected_target_refs": int(stage1.get("num_expected_target_refs") or 0),
            f"{prefix}_covered_target_refs": int(stage1.get("num_covered_target_refs") or 0),
            f"{prefix}_missing_target_refs": int(stage1.get("num_missing_target_refs") or 0),
            f"{prefix}_required_tasks": int(stage2.get("num_required_tasks") or 0),
            f"{prefix}_covered_required_tasks": int(stage2.get("num_covered_required_tasks") or 0),
            f"{prefix}_missing_required_tasks": int(stage2.get("num_missing_required_tasks") or 0),
            f"{prefix}_instruction_task_decisions": int(stage2.get("num_instruction_task_decisions") or 0),
            f"{prefix}_reviewed_instruction_task_decisions": int(stage2.get("num_reviewed_instruction_task_decisions") or 0),
            f"{prefix}_missing_instruction_task_decisions": int(stage2.get("num_missing_instruction_task_decisions") or 0),
            f"{prefix}_unknown_instruction_task_decisions": int(stage2.get("num_unknown_instruction_task_decisions") or 0),
            f"{prefix}_inconsistent_instruction_task_decisions": int(stage2.get("num_inconsistent_instruction_task_decisions") or 0),
        }

    def _staged_outputs_summary(self, semantic: Dict[str, Any]) -> Dict[str, Any]:
        metrics = self._staged_output_metrics(semantic)
        return {
            "available": bool(metrics["staged_relation_pass_available"] or metrics["staged_instruction_pass_available"]),
            **metrics,
        }

    def _vlm_transcript_summary(self, semantic: Dict[str, Any], paths: Dict[str, Path]) -> Dict[str, Any]:
        transcript = semantic.get("staged_transcript") or {}
        stages = transcript.get("stages") or []
        return {
            "available": bool(transcript),
            "path": str(paths.get("vlm_transcript")) if paths.get("vlm_transcript") else None,
            "backend": transcript.get("backend"),
            "model": transcript.get("model"),
            "relation_model": transcript.get("relation_model"),
            "instruction_model": transcript.get("instruction_model"),
            "num_stages": len(stages),
            "stages": [
                {
                    "name": stage.get("name"),
                    "model": (stage.get("request_audit") or {}).get("model"),
                    "prompt_chars": len(stage.get("prompt") or ""),
                    "request_prompt_chars": (stage.get("request_audit") or {}).get("prompt_chars"),
                    "num_image_parts": (stage.get("request_audit") or {}).get("num_image_parts"),
                    "has_error": bool(stage.get("error") or ((stage.get("request_audit") or {}).get("error"))),
                    "error": stage.get("error") or ((stage.get("request_audit") or {}).get("error") or {}),
                    "openrouter_num_attempts": (stage.get("request_audit") or {}).get("openrouter_num_attempts"),
                    "openrouter_retry_used": (stage.get("request_audit") or {}).get("openrouter_retry_used"),
                    "input_image_roles": [
                        item.get("role")
                        for item in ((stage.get("request_audit") or {}).get("input_images") or [])
                    ],
                    "has_response": isinstance(stage.get("response"), dict) and bool(stage.get("response")),
                    "has_instruction_task_packet": bool(stage.get("instruction_task_packet")),
                }
                for stage in stages
            ],
        }

    def _judge_or_review(
        self,
        *,
        image_path: str | None,
        formal: Dict[str, Any],
        state_dict: Dict[str, Any],
        verification: Dict[str, Any],
    ) -> LLMJudgeSignal:
        if self.judge is None:
            return LLMJudgeSignal(used=False)
        try:
            return self.judge.judge(
                image_path=str(image_path) if image_path else None,
                problem_text=formal.get("problem_text") or "",
                original_answer=formal.get("answer"),
                canonical_state=state_dict,
                verification_report=verification,
            )
        except Exception as exc:
            return LLMJudgeSignal(
                used=True,
                pass_signal="needs_review",
                confidence=0.0,
                blocking_recommendation=True,
                checks={"judge_call": "fail"},
                reason=f"LLM judge failed: {exc}",
            )

    def write_error_report(self, record: Dict[str, Any], exc: Exception) -> Dict[str, Any]:
        sample_id = str(record.get("id") or record.get("problem_id") or "sample")
        paths = self._paths(sample_id)
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }
        report = {
            "mode": "failed",
            "reason": str(exc),
            "stage_diagnosis": {
                "primary_stage": "pipeline_error",
                "stage_flags": {"pipeline_error": True},
            },
            "verification": {
                "valid": False,
                "issues": [self._issue("pipeline.record_failed", str(exc), exception_type=type(exc).__name__)],
            },
            "image_alignment": None,
            "consistency": None,
            "llm_judge": LLMJudgeSignal(used=False).to_dict(),
            "merged_decision": "reject_or_repair",
            "error": error,
            "paths": self._report_paths(paths),
        }
        image_path = record.get("image_path") or record.get("image")
        if image_path:
            self.review.compose_text_panel(
                str(image_path),
                [
                    "Pipeline failed before canonical reconstruction.",
                    f"Error type: {type(exc).__name__}",
                    f"Error: {exc}",
                    "This sample should be retried; no semantic graph was completed.",
                ],
                str(paths["review"]),
                title=sample_id,
            )
        self._write_json(paths["report"], report)
        return report
    def _formal_to_constraint_graph(self, formal: Dict[str, Any]) -> Dict[str, Any]:
        constraints = []
        for rel in formal.get("relations") or []:
            rtype = rel.get("type")
            if rtype == "length":
                objects = rel.get("objects") or []
                constraints.append({"type": "length", "object": objects[0] if objects else rel.get("object"), "value": rel.get("value"), "source": "hybrid"})
            elif rtype in {"perpendicular", "parallel"}:
                constraints.append({"type": rtype, "lines": rel.get("objects") or rel.get("lines") or [], "source": "hybrid"})
        return {
            "objects": formal.get("objects") or {},
            "constraints": constraints,
            "target": formal.get("target") or {},
            "answer": formal.get("answer"),
            "extraction": formal,
        }

    def _text_review(
        self,
        image_path: str | None,
        output_path: Path,
        sample_id: str,
        formal: Dict[str, Any],
        extra: list[str] | None = None,
        overlay_path: Path | None = None,
    ) -> None:
        objects = formal.get("objects") or {}
        relations = list(formal.get("relations") or [])
        coordinate_types = set(formal.get("coordinate_relation_types") or [])
        theorem_types = set(formal.get("theorem_relation_types") or [])
        unsupported_types = set(formal.get("unsupported_relation_types") or [])
        buildable_relations = [rel for rel in relations if rel.get("type") in coordinate_types]
        theorem_relations = [rel for rel in relations if rel.get("type") in theorem_types]
        unsupported_relations = [rel for rel in relations if rel.get("type") in unsupported_types]
        lines = [
            f"mode: {formal.get('mode')}",
            f"reason: {formal.get('reason')}",
            "semantic graph: "
            f"relations={len(relations)} "
            f"buildable={len(buildable_relations)} "
            f"theorem={len(theorem_relations)} "
            f"unsupported={len(unsupported_relations)}",
            "objects: "
            f"points={len(objects.get('points') or [])} "
            f"lines={len(objects.get('lines') or [])} "
            f"circles={len(objects.get('circles') or [])} "
            f"arcs={len(objects.get('arcs') or [])}",
        ]
        solver_coverage = formal.get("solver_coverage") or {}
        if solver_coverage:
            lines.append(
                "solver coverage: "
                f"status={solver_coverage.get('status')} "
                f"next={solver_coverage.get('next_solver')} "
                f"symbols={','.join(solver_coverage.get('symbolic_variables') or []) or '-'}"
            )
        summary = formal.get("grounding_summary") or {}
        if summary:
            lines.append(
                "grounding: "
                f"accepted={summary.get('accepted', 0)} "
                f"uncertain={summary.get('uncertain', 0)} "
                f"rejected={summary.get('rejected', 0)}"
            )
        target = formal.get("target") or {}
        if target:
            lines.append(f"target: {target}")
        plan = formal.get("instruction_plan") or {}
        if plan:
            lines.append(
                "instructions: "
                f"buildable={plan.get('buildable')} "
                f"accepted={plan.get('num_accepted', 0)} "
                f"rejected={plan.get('num_rejected', 0)} "
                f"synthesized={plan.get('num_synthesized', 0)}"
            )
            lines.append(f"instruction reason: {plan.get('reason')}")
        lines.extend(extra or [])
        if buildable_relations:
            lines.append("buildable relations:")
            lines.extend(self._relation_review_lines(buildable_relations, limit=5))
        if theorem_relations:
            lines.append("theorem-only relations:")
            lines.extend(self._relation_review_lines(theorem_relations, limit=7))
        if unsupported_relations:
            lines.append("unsupported relations:")
            lines.extend(self._relation_review_lines(unsupported_relations, limit=4))
        if overlay_path and overlay_path.exists():
            self.review.compose_evidence_panel(str(image_path or ""), str(overlay_path), lines, str(output_path), title=sample_id)
        else:
            self.review.compose_text_panel(str(image_path or ""), lines, str(output_path), title=sample_id)

    def _relation_review_lines(self, relations: list[Dict[str, Any]], limit: int) -> list[str]:
        lines: list[str] = []
        for rel in relations[:limit]:
            status = rel.get("grounding_status", "")
            objects = rel.get("objects") or rel.get("lines") or rel.get("object") or ""
            value = rel.get("value")
            confidence = rel.get("confidence")
            suffix = []
            if value not in {None, ""}:
                suffix.append(f"value={value}")
            if confidence not in {None, ""}:
                suffix.append(f"conf={confidence}")
            suffix_text = f" ({', '.join(str(item) for item in suffix)})" if suffix else ""
            lines.append(f"- {rel.get('type')}[{status}]: {objects}{suffix_text}")
        remaining = len(relations) - limit
        if remaining > 0:
            lines.append(f"- ... {remaining} more")
        return lines

    def _paths(self, sample_id: str) -> Dict[str, Path]:
        return {
            "visual": self.visual_dir / f"{sample_id}.json",
            "visual_overlay": self.visual_overlay_dir / f"{sample_id}.png",
            "script_observations": self.script_observations_dir / f"{sample_id}.json",
            "evidence_graph": self.evidence_graphs_dir / f"{sample_id}.json",
            "text_claims": self.text_claims_dir / f"{sample_id}.json",
            "evidence_board": self.evidence_boards_dir / f"{sample_id}.json",
            "semantic": self.semantic_dir / f"{sample_id}.json",
            "semantic_raw": self.semantic_raw_dir / f"{sample_id}.json",
            "semantic_repaired": self.semantic_repaired_dir / f"{sample_id}.json",
            "vlm_relation_pass": self.vlm_passes_dir / f"{sample_id}.relation_pass.json",
            "vlm_relation_pass_raw": self.vlm_passes_dir / f"{sample_id}.relation_pass.raw.json",
            "vlm_instruction_pass": self.vlm_passes_dir / f"{sample_id}.instruction_pass.json",
            "vlm_instruction_pass_raw": self.vlm_passes_dir / f"{sample_id}.instruction_pass.raw.json",
            "vlm_transcript": self.vlm_transcripts_dir / f"{sample_id}.json",
            "instruction_task_packet": self.instruction_task_packets_dir / f"{sample_id}.json",
            "grounded": self.grounded_dir / f"{sample_id}.json",
            "formal": self.formal_dir / f"{sample_id}.json",
            "build_plan": self.build_plans_dir / f"{sample_id}.json",
            "record": self.layout.records_dir / f"{sample_id}.json",
            "sketch": self.layout.sketches_dir / f"{sample_id}.png",
            "scaffold_sketch": self.layout.scaffold_sketches_dir / f"{sample_id}.png",
            "scaffold_review": self.layout.scaffold_review_images_dir / f"{sample_id}.png",
            "review": self.layout.review_images_dir / f"{sample_id}.png",
            "report": self.layout.reports_dir / f"{sample_id}.json",
        }

    def _report_paths(self, paths: Dict[str, Path]) -> Dict[str, str | None]:
        out = {key: str(value) if value is not None else None for key, value in paths.items()}
        out["review_image"] = out.get("review")
        out["canonical_sketch"] = out.get("sketch")
        out["scaffold_sketch"] = out.get("scaffold_sketch")
        out["scaffold_review"] = out.get("scaffold_review")
        out["visual_overlay"] = out.get("visual_overlay")
        return out

    def _semantic_confidence(self, semantic: Dict[str, Any]) -> float:
        values = []
        for relation in semantic.get("relations") or []:
            try:
                values.append(float(relation["confidence"]))
            except (KeyError, TypeError, ValueError):
                continue
        return sum(values) / len(values) if values else 0.5

    def _write_staged_outputs(self, paths: Dict[str, Path], semantic: Dict[str, Any]) -> None:
        staged = semantic.get("staged_outputs") or {}
        relation_pass = staged.get("relation_pass")
        raw_relation_pass = staged.get("raw_relation_pass")
        instruction_pass = staged.get("instruction_pass")
        raw_instruction_pass = staged.get("raw_instruction_pass")
        instruction_task_packet = staged.get("instruction_task_packet")
        transcript = semantic.get("staged_transcript")
        if relation_pass is not None:
            self._write_json(paths["vlm_relation_pass"], relation_pass)
        if raw_relation_pass is not None:
            self._write_json(paths["vlm_relation_pass_raw"], raw_relation_pass)
        if instruction_pass is not None:
            self._write_json(paths["vlm_instruction_pass"], instruction_pass)
        if raw_instruction_pass is not None:
            self._write_json(paths["vlm_instruction_pass_raw"], raw_instruction_pass)
        if instruction_task_packet is not None:
            self._write_json(paths["instruction_task_packet"], instruction_task_packet)
        if transcript is not None:
            self._write_json(paths["vlm_transcript"], transcript)

    def _instruction_task_packet(
        self,
        raw_semantic: Dict[str, Any],
        semantic_before_repair: Dict[str, Any],
        evidence_graph: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self.instruction_task_packet_builder.build(
            relation_pass=semantic_before_repair,
            evidence_graph=evidence_graph,
        )

    def _issue(self, code: str, message: str, **metadata: Any) -> Dict[str, Any]:
        severity = metadata.pop("severity", "error")
        location = metadata.pop("location", None)
        return {
            "severity": severity,
            "code": code,
            "message": message,
            "location": location,
            "metadata": metadata,
        }

    def _write_json(self, path: Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _public_parser_input(record: Mapping[str, Any] | ParserInput) -> ParserInput:
    """Project dataset rows to the four-field runtime ParserInput contract."""

    if isinstance(record, ParserInput):
        return record
    if not isinstance(record, Mapping):
        raise TypeError("Hybrid parser input must be a mapping or ParserInput")
    image = record["image"] if "image" in record else record.get("image_path")
    if "question" in record:
        question = record["question"]
    elif "problem_text" in record:
        question = record["problem_text"]
    else:
        question = record.get("problem", "")
    sample_id = record["id"] if "id" in record else record.get("problem_id")
    if sample_id is None:
        raise ValueError("Hybrid parser input requires id or problem_id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise TypeError("Hybrid parser input id must be a non-empty string")
    if image is not None and not isinstance(image, (str, Path)):
        raise TypeError("Hybrid parser input image must be a path string")
    if not isinstance(question, str):
        raise TypeError("Hybrid parser input question must be a string")
    return ParserInput(
        id=sample_id,
        image=str(image) if image is not None else "",
        question=question,
    )
