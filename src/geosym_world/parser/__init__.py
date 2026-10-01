"""Public contracts for image-to-geometry parsing."""

from .capabilities import (
    CapabilityLevel,
    ParserCapability,
    object_capability,
    parser_capability_report,
    relation_capability,
)
from .constraint_completion import (
    COMPLETION_PLAN_SCHEMA_VERSION,
    CompletionStatus,
    ConstraintCompletionPlan,
    build_constraint_completion_plan,
)
from .contracts import (
    PARSER_INPUT_SCHEMA_VERSION,
    ParserInput,
    parser_record_audit,
    validate_parse_graph_hypothesis,
    validate_parser_input,
    validate_proposed_parser_record,
)
from .ocr import (
    OCR_PREDICTION_SCHEMA_VERSION,
    OCRImagePrediction,
    OCRRegionPrediction,
    RecordedLabelRecognizer,
    TesseractLabelRecognizer,
    image_sha256,
    load_ocr_predictions,
)
from .openrouter import PROMPT_VERSION, api_key_from_env, parse_one as parse_one_openrouter
from .pipeline import (
    PARSER_PIPELINE_RESULT_SCHEMA_VERSION,
    PARSER_QUALITY_REPORT_SCHEMA_VERSION,
    CandidateLifecycle,
    ParserPipeline,
    ParserPipelineResult,
    ParserQualityReport,
)
from .prediction import EXTERNAL_PARSE_GRAPH_SCHEMA_VERSION, ExternalParseGraphPrediction
from .seed_realizer import (
    ConstraintApplicationReport,
    ConstraintApplier,
    ConstraintOutcome,
    ConstraintStatus,
    SEED_REALIZATION_SCHEMA_VERSION,
    SeedRealization,
    SeedRealizationStatus,
    SkeletonBuildReport,
    SkeletonSeedBuilder,
    SeedRealizer,
)
from .text_claims import TextClaimExtractor

__all__ = [
    "COMPLETION_PLAN_SCHEMA_VERSION",
    "ConstraintApplicationReport",
    "ConstraintApplier",
    "CapabilityLevel",
    "CandidateLifecycle",
    "CompletionStatus",
    "ConstraintCompletionPlan",
    "ConstraintOutcome",
    "ConstraintStatus",
    "EXTERNAL_PARSE_GRAPH_SCHEMA_VERSION",
    "ExternalParseGraphPrediction",
    "OCR_PREDICTION_SCHEMA_VERSION",
    "OCRImagePrediction",
    "OCRRegionPrediction",
    "PARSER_INPUT_SCHEMA_VERSION",
    "PARSER_PIPELINE_RESULT_SCHEMA_VERSION",
    "PARSER_QUALITY_REPORT_SCHEMA_VERSION",
    "PROMPT_VERSION",
    "ParserInput",
    "ParserCapability",
    "ParserPipeline",
    "ParserPipelineResult",
    "ParserQualityReport",
    "RecordedLabelRecognizer",
    "SEED_REALIZATION_SCHEMA_VERSION",
    "SeedRealization",
    "SeedRealizationStatus",
    "SkeletonBuildReport",
    "SkeletonSeedBuilder",
    "SeedRealizer",
    "TesseractLabelRecognizer",
    "TextClaimExtractor",
    "api_key_from_env",
    "build_constraint_completion_plan",
    "image_sha256",
    "load_ocr_predictions",
    "object_capability",
    "parse_one_openrouter",
    "parser_record_audit",
    "parser_capability_report",
    "relation_capability",
    "validate_parse_graph_hypothesis",
    "validate_parser_input",
    "validate_proposed_parser_record",
]
