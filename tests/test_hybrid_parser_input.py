import json

from PIL import Image
import pytest

from geosym_world.parser.hybrid.semantic_extractor import HybridSemanticExtractor
from geosym_world.parser.pipeline import HybridParsePipeline
from geosym_world.tools.parse_real_data_hybrid_v0 import run


def test_hybrid_pipeline_projects_only_public_parser_fields(tmp_path):
    image_path = tmp_path / "diagram.png"
    Image.new("RGB", (96, 96), "white").save(image_path)
    pipeline = HybridParsePipeline(
        "public-input-fixture",
        raw_root=tmp_path / "raw",
        parsed_root=tmp_path / "parsed",
        semantic_backend="mock",
        judge_backend="none",
    )

    pipeline.process_record(
        {
            "id": "sample_public",
            "image_path": str(image_path),
            "problem_text": "Find AB.",
            "answer": "999",
            "solution": "AB = 999",
            "raw": {
                "diagram_logic_form": "AB = 999",
                "formal_state": {"secret": True},
            },
        }
    )

    dataset_dir = tmp_path / "parsed" / "public-input-fixture"
    script = json.loads((dataset_dir / "script_observations" / "sample_public.json").read_text())
    semantic = json.loads((dataset_dir / "semantic_raw" / "sample_public.json").read_text())
    formal = json.loads((dataset_dir / "formal_graphs" / "sample_public.json").read_text())

    assert [item["name"] for item in script["text_sources"]] == ["problem_text"]
    assert "999" not in json.dumps(script, ensure_ascii=False)
    assert semantic["metadata"]["answer"] is None
    assert formal["answer"] is None


def test_hybrid_batch_cli_records_progress_configuration_and_failures(tmp_path, capsys):
    image_path = tmp_path / "diagram.png"
    Image.new("RGB", (64, 64), "white").save(image_path)
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        json.dumps(
            {
                "id": "sample_batch",
                "image": str(image_path),
                "question": "Find AB.",
                "answer": "secret",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = run(
        dataset_name="batch-fixture",
        records_path=records_path,
        raw_root=tmp_path / "raw",
        parsed_root=tmp_path / "parsed",
        backend="mock",
        model=None,
        relation_model=None,
        instruction_model=None,
        extraction_mode="single",
        judge_backend="none",
        judge_model=None,
        limit=None,
        progress_every=1,
        continue_on_error=False,
    )

    assert manifest["counts"] == {
        "records_failed": 0,
        "records_read": 1,
        "records_selected": 1,
        "records_succeeded": 1,
    }
    assert manifest["configuration"]["runtime_input_fields"] == [
        "schema_version",
        "id",
        "image",
        "question",
    ]
    assert "processed=1" in capsys.readouterr().err


def test_hybrid_batch_cli_requires_explicit_continue_for_failed_input(tmp_path):
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        json.dumps({"id": "broken", "image": str(tmp_path / "missing.png"), "question": "Find AB."}) + "\n",
        encoding="utf-8",
    )

    manifest = run(
        dataset_name="failure-fixture",
        records_path=records_path,
        raw_root=tmp_path / "raw",
        parsed_root=tmp_path / "parsed",
        backend="mock",
        model=None,
        relation_model=None,
        instruction_model=None,
        extraction_mode="single",
        judge_backend="none",
        judge_model=None,
        limit=None,
        progress_every=0,
        continue_on_error=True,
    )

    assert manifest["counts"]["records_failed"] == 1
    error_report = tmp_path / "parsed" / "failure-fixture" / "reports" / "broken.json"
    assert json.loads(error_report.read_text())["stage_diagnosis"]["primary_stage"] == "pipeline_error"


def test_hybrid_semantic_prompts_make_answer_boundary_explicit():
    extractor = HybridSemanticExtractor(backend="mock")
    common = {
        "problem_text": "Find AB.",
        "answer": None,
        "visual_state": {},
        "script_observations": {},
        "evidence_graph": {},
        "evidence_board": {},
    }

    prompts = [
        extractor._prompt(**common),
        extractor._relation_prompt(**common),
        extractor._instruction_prompt(
            **common,
            relation_pass={},
            instruction_task_packet={},
            build_plan_seed={},
        ),
    ]

    for prompt in prompts:
        assert "No answer is available to the parser runtime." in prompt
        assert "Known dataset answer" not in prompt


def test_hybrid_semantic_extractor_rejects_answer_at_runtime_boundary():
    extractor = HybridSemanticExtractor(backend="mock")

    with pytest.raises(ValueError, match="answer is evaluator-only"):
        extractor.extract(
            image_path="diagram.png",
            problem_text="Find AB.",
            answer="secret",
            visual_state={},
        )


def test_hybrid_pipeline_rejects_invalid_public_field_types(tmp_path):
    pipeline = HybridParsePipeline(
        "strict-input-fixture",
        raw_root=tmp_path / "raw",
        parsed_root=tmp_path / "parsed",
        semantic_backend="mock",
        judge_backend="none",
    )

    with pytest.raises(TypeError, match="question must be a string"):
        pipeline.process_record({"id": "sample", "image": "diagram.png", "question": {}})

    with pytest.raises(ValueError, match="unsupported fields: answer"):
        pipeline.process_input(
            {
                "schema_version": "1.0",
                "id": "sample",
                "image": "diagram.png",
                "question": "Find AB.",
                "answer": "secret",
            }
        )
