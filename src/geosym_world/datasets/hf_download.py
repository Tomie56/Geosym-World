"""HuggingFace dataset download helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import json
import shutil

from .layout import DataLayout, default_parsed_root, default_raw_root


def download_hf_dataset(
    repo_id: str,
    dataset_name: Optional[str] = None,
    split: Optional[str] = None,
    max_samples: Optional[int] = None,
    raw_root: str | Path | None = None,
    parsed_root: str | Path | None = None,
) -> DataLayout:
    """Download or materialize a HF dataset into the GeoSym raw layout.

    The helper first tries `datasets.load_dataset`; if unavailable, it falls
    back to `huggingface_hub.snapshot_download`.
    """
    name = dataset_name or repo_id.replace("/", "__")
    layout = DataLayout(
        name,
        Path(raw_root) if raw_root is not None else default_raw_root(),
        Path(parsed_root) if parsed_root is not None else default_parsed_root(),
    ).ensure()
    try:
        return _download_with_datasets(repo_id, layout, split=split, max_samples=max_samples)
    except ImportError:
        return _download_with_snapshot(repo_id, layout)


def _download_with_datasets(repo_id: str, layout: DataLayout, split: Optional[str], max_samples: Optional[int]) -> DataLayout:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("datasets is not installed") from exc
    ds = load_dataset(repo_id, split=split or "train")
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    records_path = layout.raw_metadata_dir / "records.jsonl"
    with open(records_path, "w", encoding="utf-8") as f:
        for idx, item in enumerate(ds):
            record = _materialize_record(item, idx, layout)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return layout


def _download_with_snapshot(repo_id: str, layout: DataLayout) -> DataLayout:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install `datasets` or `huggingface_hub` to download HuggingFace data") from exc
    snapshot_dir = snapshot_download(repo_id=repo_id, local_dir=str(layout.raw_dataset_dir / "snapshot"))
    (layout.raw_metadata_dir / "snapshot_path.txt").write_text(snapshot_dir, encoding="utf-8")
    return layout


def _materialize_record(item: dict, idx: int, layout: DataLayout) -> dict:
    record = {"id": item.get("id") or f"sample_{idx:06d}"}
    image_value = item.get("image") or item.get("diagram") or item.get("img")
    if image_value is not None:
        image_path = layout.raw_images_dir / f"{record['id']}.png"
        try:
            image_value.save(image_path)
            record["image_path"] = str(image_path)
        except AttributeError:
            src = Path(str(image_value))
            if src.exists():
                shutil.copy(src, image_path)
                record["image_path"] = str(image_path)
            else:
                record["image"] = str(image_value)
    for key in ["question", "problem", "problem_text", "answer", "solution", "choices"]:
        if key in item:
            record[key] = item[key]
    return record
