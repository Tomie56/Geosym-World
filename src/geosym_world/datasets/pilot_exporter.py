"""Pilot JSONL exporter for GeoSym-Hard / GeoSym-Debug records."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List
import json

from geosym_world.construction.builder import ConstructionBuilder
from geosym_world.datasets.export import export_record
from geosym_world.datasets.samples import PilotSample, built_in_pilot_samples
from geosym_world.solver.closure import FactClosureSolver
from geosym_world.verifier.trace import TraceVerifier


class PilotBenchmarkExporter:
    """Builds benchmark records from pilot samples."""

    def __init__(self):
        self.verifier = TraceVerifier()
        self.solver = FactClosureSolver()

    def build_record(self, sample: PilotSample) -> Dict[str, Any]:
        verification = self.verifier.verify_trace(sample.trace)
        state = ConstructionBuilder().execute(sample.trace)
        facts = self.solver.solve(state)
        record = export_record(
            state,
            facts,
            record_id=sample.id,
            question=sample.question,
            answer=sample.answer,
            verification=verification,
        )
        target = record["derivation"]["target_fact"]
        record["benchmark"] = {
            "version": "geosym-world-pilot-v0",
            "task_type": sample.task_type,
            "difficulty": sample.difficulty,
            "diagnostic_axes": list(sample.diagnostic_axes),
            "target_value": target["value"] if target else None,
            "verification_valid": verification.valid,
            "metadata": sample.metadata,
        }
        return record

    def build_records(self, samples: Iterable[PilotSample] | None = None) -> List[Dict[str, Any]]:
        return [self.build_record(sample) for sample in (samples or built_in_pilot_samples())]

    def write_jsonl(self, path: str | Path, samples: Iterable[PilotSample] | None = None) -> Dict[str, Any]:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        records = self.build_records(samples)
        with output_path.open("w", encoding="utf-8") as f:
            for record in records:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")
        manifest = self._manifest(output_path, records)
        manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "path": str(output_path),
            "manifest_path": str(manifest_path),
            "num_records": len(records),
            "num_valid": sum(1 for record in records if (record.get("verification") or {}).get("valid")),
            "task_types": sorted({record["benchmark"]["task_type"] for record in records}),
        }

    def _manifest(self, output_path: Path, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        fact_type_counts: Dict[str, int] = {}
        diagnostic_axes: Dict[str, int] = {}
        theorem_family_counts: Dict[str, int] = {}
        theorem_family_records: Dict[str, set[str]] = {}
        for record in records:
            for fact in record.get("facts") or []:
                fact_type = str(fact.get("type"))
                fact_type_counts[fact_type] = fact_type_counts.get(fact_type, 0) + 1
                family = _theorem_family(fact_type)
                if family:
                    theorem_family_counts[family] = theorem_family_counts.get(family, 0) + 1
                    theorem_family_records.setdefault(family, set()).add(str(record["id"]))
            for axis in record.get("benchmark", {}).get("diagnostic_axes") or []:
                diagnostic_axes[str(axis)] = diagnostic_axes.get(str(axis), 0) + 1
        return {
            "version": "geosym-world-pilot-v0",
            "path": str(output_path),
            "num_records": len(records),
            "num_valid": sum(1 for record in records if (record.get("verification") or {}).get("valid")),
            "record_ids": [record["id"] for record in records],
            "task_types": sorted({record["benchmark"]["task_type"] for record in records}),
            "difficulty_counts": _count(record["benchmark"]["difficulty"] for record in records),
            "diagnostic_axes": dict(sorted(diagnostic_axes.items())),
            "theorem_family_counts": dict(sorted(theorem_family_counts.items())),
            "theorem_family_records": {key: sorted(value) for key, value in sorted(theorem_family_records.items())},
            "fact_type_counts": dict(sorted(fact_type_counts.items())),
        }


def _count(values: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _theorem_family(fact_type: str) -> str | None:
    mapping = {
        "right_triangle": "right_triangle",
        "pythagorean_relation": "right_triangle",
        "isosceles_triangle": "isosceles_triangle",
        "equal_base_angles": "isosceles_triangle",
        "equal_base_angle_residual": "isosceles_triangle",
        "angle_bisector": "angle_bisector",
        "equal_angle": "angle_bisector",
        "angle_bisector_residual": "angle_bisector",
        "central_angle": "circle_chord",
        "chord_length_formula_check": "circle_chord",
        "equal_chord_equal_central_angle": "circle_chord",
        "chord_of_circle": "circle_chord",
        "similar_triangle": "similar_triangle",
        "aa_similarity": "similar_triangle",
        "aa_similarity_residual": "similar_triangle",
        "parallel_cut_similarity": "similar_triangle",
        "parallel_cut_ratio_residual": "similar_triangle",
        "similarity_ratio_residual": "similar_triangle",
        "similarity_scale_squared": "similar_triangle",
        "equal_corresponding_angles": "similar_triangle",
        "similar_angle_residual": "similar_triangle",
        "inscribed_angle": "inscribed_angle",
        "inscribed_angle_central_residual": "inscribed_angle",
        "inscribed_angle_arc_residual": "inscribed_angle",
        "arc_subtends_inscribed_angle": "inscribed_angle",
        "equal_inscribed_angles": "inscribed_angle",
        "tangent_chord_angle": "circle_tangent_angle",
        "alternate_segment_angle": "circle_tangent_angle",
        "tangent_chord_angle_residual": "circle_tangent_angle",
        "line_on_plane": "line_plane_3d",
        "line_parallel_plane": "line_plane_3d",
        "line_perpendicular_plane": "line_plane_3d",
        "line_plane_angle": "line_plane_3d",
        "line_plane_distance": "line_plane_3d",
        "space_angle": "space_angle_3d",
        "plane_parallel": "line_plane_3d",
        "plane_perpendicular": "line_plane_3d",
        "point_on_plane": "line_plane_3d",
        "point_projection_plane": "projection_3d",
        "projection_distance": "projection_3d",
        "projection_plane_residual": "projection_3d",
        "line_projection_plane": "projection_3d",
        "line_projection_plane_residual": "projection_3d",
        "dihedral_angle": "dihedral_angle_3d",
        "dihedral_shared_edge": "dihedral_angle_3d",
        "section_interface": "solid_section_3d",
        "section_polygon": "solid_section_3d",
        "section_area": "solid_section_3d",
        "section_perimeter": "solid_section_3d",
        "section_degeneracy_measure": "solid_section_3d",
        "section_boundary_size": "solid_section_3d",
        "section_boundary_plane_residual": "solid_section_3d",
    }
    return mapping.get(fact_type)
