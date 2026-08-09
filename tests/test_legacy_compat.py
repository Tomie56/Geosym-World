"""Coverage and smoke tests for the complete scripts/tools compatibility path."""

import json
import time
from pathlib import Path

import cv2
import numpy as np

from geosym_world.legacy import (
    LEGACY_CAPABILITIES,
    LegacyCompatiblePipeline,
    LegacyPipelineOptions,
    QAGenerator,
    RandomGeometryBuilder,
    RegionExtractConfig,
    RegionExtractor,
    SHADERS,
    TemplateGenerator,
    compatibility_report,
    grade_answer_sympy,
)
from geosym_world.legacy_tools.builder import RandomGeometryBuilder as SourceBuilder
from geosym_world.legacy_tools.qa_template import QAGenerator as SourceQA
from geosym_world.legacy_tools.template import TemplateGenerator as SourceTemplate


def _slow_test_worker(payload):
    delay, index = payload
    time.sleep(delay)
    return {"ok": True, "index": index, "value": index}


def test_legacy_classes_are_exact_source_aliases():
    assert TemplateGenerator is SourceTemplate
    assert RandomGeometryBuilder is SourceBuilder
    assert QAGenerator is SourceQA
    report = compatibility_report()
    assert report["available"] is True
    assert set(report["capabilities"]["question_types"]) == set(LEGACY_CAPABILITIES["question_types"])
    assert "generate_derivations" in report["classes"]["TemplateGenerator"]["methods"]
    assert "detect_new_intersections" in report["classes"]["RandomGeometryBuilder"]["methods"]
    assert "_generate_all_shadow_area_questions" in report["classes"]["QAGenerator"]["methods"]


def test_all_legacy_shaders_apply_to_mask():
    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    cv2.rectangle(mask, (12, 12), (52, 52), 255, -1)
    params = {
        "hatch": {"spacing": 5, "intensity": 0.6},
        "crosshatch": {"spacing": 5, "angle1": 45, "angle2": 135, "intensity": 0.6},
        "solid": {"color": [20, 80, 120], "intensity": 0.6},
        "gradient": {"start_color": [10, 20, 30], "end_color": [100, 120, 140], "angle_deg": 30, "intensity": 0.6},
    }
    assert set(SHADERS) == {"hatch", "crosshatch", "solid", "gradient"}
    for name, shader in SHADERS.items():
        output = shader.apply(image.copy(), mask, **params[name])
        assert output.shape == image.shape
        assert np.any(output[mask > 0] != 255)


def test_legacy_answer_grading_works_without_optional_latex_dependency():
    assert grade_answer_sympy(r"\frac{1}{2}", "1/2") is True


def test_legacy_region_extractor_finds_closed_region():
    image = np.full((128, 128, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (24, 24), (104, 104), (0, 0, 0), 3)
    extractor = RegionExtractor(RegionExtractConfig(min_area=100, arc_detect=True))
    masks, pixel_contours, math_contours, stats, polygons = extractor.extract(image)
    assert masks
    assert pixel_contours and math_contours and stats and polygons
    assert max(item["pixel"]["area_px_est"] for item in stats) > 5_000


def test_polygon_side_polygon_derivation_is_dispatched():
    project_root = Path(__file__).resolve().parents[1]
    config = _fast_legacy_config(project_root)["template"]
    config["seed"] = 1
    config["base"]["types"] = ["polygon"]
    config["base"]["type_weights"] = {"polygon": 1.0}
    config["base"]["polygon"]["n_choices"] = [4]
    config["derivation"]["round_range"] = [1, 1]
    config["derivation"]["rules"]["polygon"]["supported_rules"] = [
        {"name": "side_polygon", "prob": 1.0, "n_choices": [3]}
    ]
    generator = TemplateGenerator(config, seed=9)
    generator.generate_base_shape()
    generator.generate_derivations()
    entities = generator.export_json()["entities"]
    assert len(entities) == 2
    assert entities[-1]["id"].startswith("side_polygon_n3_")


def test_minimal_legacy_pipeline_without_shader(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    config = _fast_legacy_config(project_root)
    options = LegacyPipelineOptions(
        output_root=tmp_path / "legacy",
        records=1,
        seed=123,
        workers=1,
        stages=("template", "builder", "shader", "gt", "qa"),
        rounds_distribution={0: 1},
        shader_enabled=False,
        questions_per_record=1,
        question_types=("length",),
        fail_fast=True,
    )
    manifest = LegacyCompatiblePipeline(config, options).run()
    assert manifest["num_final_records"] == 1
    assert manifest["num_qa"] == 1
    assert not manifest["failures"]
    assert Path(manifest["artifacts"]["base"]).exists()
    assert Path(manifest["artifacts"]["enhanced"]).exists()
    assert Path(manifest["artifacts"]["shaded"]).exists()
    assert Path(manifest["artifacts"]["gt"]).exists()
    assert Path(manifest["artifacts"]["qa"]).exists()
    shaded = json.loads(Path(manifest["artifacts"]["shaded"]).read_text(encoding="utf-8"))
    assert Path(shaded["raw_path"]).exists()
    assert Path(shaded["annotated_raw_path"]).exists()


def test_fused_generation_is_deterministic_across_workers(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    config = _fast_legacy_config(project_root)
    outputs = []
    for workers in (1, 2):
        root = tmp_path / f"workers_{workers}"
        options = LegacyPipelineOptions(
            output_root=root,
            records=2,
            seed=777,
            workers=workers,
            stages=("template", "builder"),
            rounds_distribution={0: 1},
            task_timeout_seconds=None,
            fail_fast=True,
        )
        manifest = LegacyCompatiblePipeline(config, options).run()
        outputs.append(Path(manifest["artifacts"]["enhanced"]).read_text(encoding="utf-8"))
        assert manifest["stage_metrics"]["template"]["fused_with_builder"] is True
    assert outputs[0] == outputs[1]


def test_isolated_tasks_are_killed_and_reported(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    pipeline = LegacyCompatiblePipeline(
        _fast_legacy_config(project_root),
        LegacyPipelineOptions(
            output_root=tmp_path / "timeout",
            task_timeout_seconds=1.0,
            workers=2,
        ),
    )
    assert pipeline._collect("test", _slow_test_worker, [(0.01, 0), (3.0, 1)]) == [0]
    assert pipeline.stage_metrics["test"] == {
        "requested": 2,
        "accepted": 1,
        "rejected": 1,
        "timed_out": 1,
    }
    assert pipeline.failures[0]["exception_type"] == "TaskTimeout"


def _fast_legacy_config(project_root: Path):
    config = json.loads((project_root / "configs" / "legacy" / "config_easy.json").read_text(encoding="utf-8"))
    config["template"]["base"]["types"] = ["special_rectangle"]
    config["template"]["base"]["type_weights"] = {"special_rectangle": 1.0}
    config["template"]["derivation"]["round_range"] = [0, 0]
    return config
