import json

from PIL import Image

from geosym_world.tools.build_parser_review_sheet import build_contact_sheet


def test_build_parser_review_sheet_includes_missing_redraw_and_status(tmp_path):
    run_dir = tmp_path / "run"
    sample = run_dir / "images" / "sample_b"
    sample.mkdir(parents=True)
    Image.new("RGB", (20, 10), "white").save(sample / "input.png")
    (run_dir / "summary.json").write_text(json.dumps({"samples": [{"id": "sample_b", "status": "partial", "quality": {"seed_realized": True, "constraint_satisfied": False}}]}))

    output = build_contact_sheet(run_dir)

    assert output == run_dir / "contact.png"
    with Image.open(output) as image:
        assert image.size == (1520, 390)
