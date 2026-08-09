from pathlib import Path

from geosym_world.datasets.layout import DataLayout, artifact_root


def test_artifact_root_defaults_to_repository_artifacts(monkeypatch):
    monkeypatch.delenv("GEOSYM_WORLD_ARTIFACTS", raising=False)
    root = artifact_root()
    assert root.name == "artifacts"
    assert root.parent.name == "Geosym-World"


def test_data_layout_honors_artifact_override(monkeypatch, tmp_path):
    configured = tmp_path / "custom-artifacts"
    monkeypatch.setenv("GEOSYM_WORLD_ARTIFACTS", str(configured))
    layout = DataLayout("fixture")

    assert layout.raw_root == configured.resolve() / "datasets" / "raw"
    assert layout.parsed_root == configured.resolve() / "datasets" / "parsed"
    assert layout.raw_images_dir == configured.resolve() / "datasets" / "raw" / "fixture" / "images"
    assert layout.records_dir == configured.resolve() / "datasets" / "parsed" / "fixture" / "records"


def test_data_layout_manifest_is_written_inside_artifacts(tmp_path):
    layout = DataLayout("fixture", raw_root=tmp_path / "raw", parsed_root=tmp_path / "parsed").ensure()
    manifest = layout.parsed_dataset_dir / "manifest.json"

    assert manifest.is_file()
    assert Path(layout.manifest()["records_dir"]).is_relative_to(tmp_path)
