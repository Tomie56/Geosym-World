"""Synthetic trace generation and image+QA export."""

from .exporter import SyntheticRecordExporter
from .diagnostics import CandidateDiagnostics
from .pipeline import GeometryGenerationPipeline, PipelineRunConfig
from .sampler import SyntheticSampler
from .splits import SPLIT_GROUP_MODES, assign_split_labels, assign_splits, lineage_root, split_group_key, topology_key, validate_split_disjoint
from .task_contract import TASK_SCHEMA_VERSION, TaskContractValidation, build_task_contract, validate_task_contract
from .templates import SyntheticSpec, TemplateFamily, built_in_template_families, load_grid_overrides, select_template_families

__all__ = [
    "GeometryGenerationPipeline",
    "CandidateDiagnostics",
    "PipelineRunConfig",
    "SyntheticRecordExporter",
    "SyntheticSampler",
    "TASK_SCHEMA_VERSION",
    "TaskContractValidation",
    "SPLIT_GROUP_MODES",
    "assign_split_labels",
    "assign_splits",
    "lineage_root",
    "split_group_key",
    "topology_key",
    "validate_split_disjoint",
    "validate_task_contract",
    "SyntheticSpec",
    "TemplateFamily",
    "built_in_template_families",
    "build_task_contract",
    "load_grid_overrides",
    "select_template_families",
]
