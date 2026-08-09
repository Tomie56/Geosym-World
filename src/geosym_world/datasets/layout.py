"""Filesystem layout for GeoSym benchmark data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict
import json
import os


def artifact_root() -> Path:
    configured = os.environ.get("GEOSYM_WORLD_ARTIFACTS")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "artifacts"


def default_raw_root() -> Path:
    return artifact_root() / "datasets" / "raw"


def default_parsed_root() -> Path:
    return artifact_root() / "datasets" / "parsed"


@dataclass
class DataLayout:
    dataset_name: str
    raw_root: Path = field(default_factory=default_raw_root)
    parsed_root: Path = field(default_factory=default_parsed_root)

    @property
    def raw_dataset_dir(self) -> Path:
        return self.raw_root / self.dataset_name

    @property
    def parsed_dataset_dir(self) -> Path:
        return self.parsed_root / self.dataset_name

    @property
    def raw_images_dir(self) -> Path:
        return self.raw_dataset_dir / "images"

    @property
    def raw_metadata_dir(self) -> Path:
        return self.raw_dataset_dir / "metadata"

    @property
    def extracted_dir(self) -> Path:
        return self.parsed_dataset_dir / "extracted_relations"

    @property
    def constraints_dir(self) -> Path:
        return self.parsed_dataset_dir / "constraints"

    @property
    def visual_state_dir(self) -> Path:
        return self.parsed_dataset_dir / "visual_state"

    @property
    def visual_overlays_dir(self) -> Path:
        return self.parsed_dataset_dir / "visual_overlays"

    @property
    def script_observations_dir(self) -> Path:
        return self.parsed_dataset_dir / "script_observations"

    @property
    def evidence_graphs_dir(self) -> Path:
        return self.parsed_dataset_dir / "evidence_graphs"

    @property
    def text_claims_dir(self) -> Path:
        return self.parsed_dataset_dir / "text_claims"

    @property
    def evidence_boards_dir(self) -> Path:
        return self.parsed_dataset_dir / "evidence_boards"

    @property
    def build_plans_dir(self) -> Path:
        return self.parsed_dataset_dir / "build_plans"

    @property
    def semantic_relations_dir(self) -> Path:
        return self.parsed_dataset_dir / "semantic_relations"

    @property
    def semantic_raw_dir(self) -> Path:
        return self.parsed_dataset_dir / "semantic_raw"

    @property
    def semantic_repaired_dir(self) -> Path:
        return self.parsed_dataset_dir / "semantic_repaired"

    @property
    def vlm_passes_dir(self) -> Path:
        return self.parsed_dataset_dir / "vlm_passes"

    @property
    def vlm_transcripts_dir(self) -> Path:
        return self.parsed_dataset_dir / "vlm_transcripts"

    @property
    def instruction_task_packets_dir(self) -> Path:
        return self.parsed_dataset_dir / "instruction_task_packets"

    @property
    def grounded_relations_dir(self) -> Path:
        return self.parsed_dataset_dir / "grounded_relations"

    @property
    def formal_graphs_dir(self) -> Path:
        return self.parsed_dataset_dir / "formal_graphs"

    @property
    def records_dir(self) -> Path:
        return self.parsed_dataset_dir / "records"

    @property
    def sketches_dir(self) -> Path:
        return self.parsed_dataset_dir / "sketches"

    @property
    def scaffold_sketches_dir(self) -> Path:
        return self.parsed_dataset_dir / "scaffold_sketches"

    @property
    def scaffold_review_images_dir(self) -> Path:
        return self.parsed_dataset_dir / "scaffold_review_images"

    @property
    def reports_dir(self) -> Path:
        return self.parsed_dataset_dir / "reports"

    @property
    def review_images_dir(self) -> Path:
        return self.parsed_dataset_dir / "review_images"

    @property
    def exports_dir(self) -> Path:
        return self.parsed_dataset_dir / "exports"

    def ensure(self) -> "DataLayout":
        for path in [
            self.raw_dataset_dir,
            self.raw_images_dir,
            self.raw_metadata_dir,
            self.parsed_dataset_dir,
            self.extracted_dir,
            self.constraints_dir,
            self.visual_state_dir,
            self.visual_overlays_dir,
            self.script_observations_dir,
            self.evidence_graphs_dir,
            self.text_claims_dir,
            self.evidence_boards_dir,
            self.semantic_relations_dir,
            self.semantic_raw_dir,
            self.semantic_repaired_dir,
            self.vlm_passes_dir,
            self.vlm_transcripts_dir,
            self.instruction_task_packets_dir,
            self.grounded_relations_dir,
            self.formal_graphs_dir,
            self.build_plans_dir,
            self.records_dir,
            self.sketches_dir,
            self.scaffold_sketches_dir,
            self.scaffold_review_images_dir,
            self.reports_dir,
            self.review_images_dir,
            self.exports_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
        self.write_manifest()
        return self

    def manifest(self) -> Dict[str, str]:
        return {
            "dataset_name": self.dataset_name,
            "raw_dataset_dir": str(self.raw_dataset_dir),
            "raw_images_dir": str(self.raw_images_dir),
            "raw_metadata_dir": str(self.raw_metadata_dir),
            "parsed_dataset_dir": str(self.parsed_dataset_dir),
            "extracted_dir": str(self.extracted_dir),
            "constraints_dir": str(self.constraints_dir),
            "visual_state_dir": str(self.visual_state_dir),
            "visual_overlays_dir": str(self.visual_overlays_dir),
            "script_observations_dir": str(self.script_observations_dir),
            "evidence_graphs_dir": str(self.evidence_graphs_dir),
            "text_claims_dir": str(self.text_claims_dir),
            "evidence_boards_dir": str(self.evidence_boards_dir),
            "semantic_relations_dir": str(self.semantic_relations_dir),
            "semantic_raw_dir": str(self.semantic_raw_dir),
            "semantic_repaired_dir": str(self.semantic_repaired_dir),
            "vlm_passes_dir": str(self.vlm_passes_dir),
            "vlm_transcripts_dir": str(self.vlm_transcripts_dir),
            "instruction_task_packets_dir": str(self.instruction_task_packets_dir),
            "grounded_relations_dir": str(self.grounded_relations_dir),
            "formal_graphs_dir": str(self.formal_graphs_dir),
            "build_plans_dir": str(self.build_plans_dir),
            "records_dir": str(self.records_dir),
            "sketches_dir": str(self.sketches_dir),
            "scaffold_sketches_dir": str(self.scaffold_sketches_dir),
            "scaffold_review_images_dir": str(self.scaffold_review_images_dir),
            "reports_dir": str(self.reports_dir),
            "review_images_dir": str(self.review_images_dir),
            "exports_dir": str(self.exports_dir),
        }

    def write_manifest(self) -> Path:
        self.parsed_root.mkdir(parents=True, exist_ok=True)
        path = self.parsed_dataset_dir / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.manifest(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path
