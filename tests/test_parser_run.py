import json
import sys

import pytest
from PIL import Image, ImageDraw

from geosym_world.parser import openrouter
from geosym_world.parser.contracts import ParserInput
from geosym_world.tools import parse_openrouter_v0 as runner


@pytest.mark.parametrize("mode", ["run-dir", "output"])
def test_parser_runner_limits_attempts_and_preserves_failed_images(tmp_path, monkeypatch, mode):
    image = tmp_path / "input.png"
    original = Image.new("RGB", (80, 60), "white")
    ImageDraw.Draw(original).line([(10, 30), (70, 30)], fill="black", width=2)
    original.save(image)
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text("\n".join(json.dumps(ParserInput(id=name, image=str(image), question="").to_dict()) for name in ["good", "bad", "not_attempted"]))
    requests = []

    def request(**kwargs):
        requests.append(kwargs)
        if len(requests) == 2:
            return {}, {"points": ["A"], "strokes": [{"kind": "segment", "points": ["A"]}]}
        return {}, {"points": ["A", "B"], "strokes": [{"kind": "segment", "points": ["A", "B"]}]}

    monkeypatch.setattr(openrouter, "_request_json_completion", request)
    monkeypatch.setattr(runner, "load_project_env", lambda: None)
    monkeypatch.setattr(runner, "api_key_from_env", lambda: "fixture")
    destination = tmp_path / ("run" if mode == "run-dir" else "predictions.jsonl")
    monkeypatch.setattr(sys, "argv", ["parse_openrouter_v0", "--input", str(inputs), "--" + mode, str(destination), "--max-records", "2", "--model", "fixture"])
    runner.main()
    assert len(requests) == 2
    if mode == "output":
        assert len(destination.read_text().splitlines()) == 1
        failures = destination.with_name(destination.name + ".failed.jsonl")
        assert json.loads(failures.read_text())["id"] == "bad"
        return
    summary = json.loads((destination / "summary.json").read_text())
    assert summary["attempted"] == 2 and summary["completed"] == 1
    assert (destination / "images" / "good" / "redraw.png").is_file()
    assert (destination / "images" / "bad" / "input.png").is_file()
    assert not (destination / "images" / "bad" / "redraw.png").exists()
    assert json.loads((destination / "json" / "raw" / "bad.json").read_text())["points"] == ["A"]
    assert (destination / "contact.png").is_file()
    record = json.loads((destination / "json" / "good.json").read_text())
    assert record["quality"]["constraint_satisfied"]
    assert record["quality"]["pixel_topology_passed"] is None
    assert record["realization"]["geometry_state"]
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["max_records"] == 2 and manifest["retries"] == 0
    assert "parser/prompts/geometry_extraction_zh.txt" in manifest["code_sha256"]
