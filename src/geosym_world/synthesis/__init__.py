"""Synthetic trace generation and image+QA export."""

from .exporter import SyntheticRecordExporter
from .diagnostics import CandidateDiagnostics
from .difficulty import DIFFICULTY_SCHEMA_VERSION, difficulty_in_range, score_task, score_trace
from .pipeline import GeometryGenerationPipeline, PipelineRunConfig
from .preflight import EnhancementPreflight, PREFLIGHT_SCHEMA_VERSION, PreflightDiagnostics, PreflightResult
from .sampler import SyntheticSampler
from .splits import SPLIT_GROUP_MODES, assign_split_labels, assign_splits, lineage_root, split_group_key, topology_key, validate_split_disjoint
from .task_contract import TASK_SCHEMA_VERSION, TaskContractValidation, build_task_contract, validate_task_contract
from .task_graph import TASK_GRAPH_SCHEMA_VERSION, build_task_dependency_graph, validate_task_dependency_graph
from .templates import SyntheticSpec, TemplateFamily, built_in_template_families, load_grid_overrides, select_template_families

__all__ = [
    "GeometryGenerationPipeline",
    "CandidateDiagnostics",
    "DIFFICULTY_SCHEMA_VERSION",
    "PipelineRunConfig",
    "PREFLIGHT_SCHEMA_VERSION",
    "PreflightDiagnostics",
    "PreflightResult",
    "EnhancementPreflight",
    "SyntheticRecordExporter",
    "SyntheticSampler",
    "TASK_SCHEMA_VERSION",
    "TASK_GRAPH_SCHEMA_VERSION",
    "TaskContractValidation",
    "SPLIT_GROUP_MODES",
    "assign_split_labels",
    "assign_splits",
    "difficulty_in_range",
    "lineage_root",
    "split_group_key",
    "topology_key",
    "validate_split_disjoint",
    "validate_task_contract",
    "validate_task_dependency_graph",
    "SyntheticSpec",
    "TemplateFamily",
    "built_in_template_families",
    "build_task_contract",
    "build_task_dependency_graph",
    "load_grid_overrides",
    "select_template_families",
    "score_task",
    "score_trace",
]
