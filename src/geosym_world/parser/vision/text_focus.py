"""Deterministic local views for reading small diagram labels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import hashlib
import math

from PIL import Image


TEXT_FOCUS_VERSION = "geosym.text_focus.v1"


def build_text_focus_image(
    image_path: str,
    text_regions: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    padding_px: int = 18,
    scale: int = 3,
    max_regions: int = 80,
    columns: int = 4,
    tile_width: int = 320,
    tile_height: int = 192,
) -> dict[str, Any]:
    if padding_px < 0 or scale <= 0 or max_regions <= 0 or columns <= 0:
        raise ValueError("Text focus configuration must be positive, with non-negative padding")
    source = Path(image_path)
    if not source.is_file():
        raise ValueError(f"Text focus source image does not exist: {source}")
    destination = Path(output_path)
    with Image.open(source) as opened:
        image = opened.convert("RGB")
        width, height = image.size
        normalized_regions = _normalize_regions(text_regions, width, height)
        regions = []
        excluded_region_ids = []
        max_region_width = max(160.0, 0.25 * width)
        max_region_height = max(80.0, 0.20 * height)
        for region in normalized_regions:
            region_width = float(region["bbox"][2])
            region_height = float(region["bbox"][3])
            if region_width > max_region_width or region_height > max_region_height:
                excluded_region_ids.append(region["id"])
                continue
            regions.append(region)
        regions = regions[:max_regions]
        groups = _group_regions(regions)
        if not groups:
            return {
                "version": TEXT_FOCUS_VERSION,
                "status": "no_regions",
                "path": None,
                "source_image": str(source),
                "source_sha256": _sha256(source),
                "excluded_region_ids": excluded_region_ids,
                "tiles": [],
            }
        rows = math.ceil(len(groups) / columns)
        sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
        tiles = []
        for index, group in enumerate(groups):
            crop_box = _expanded_box(group, width, height, padding_px)
            crop = image.crop(crop_box)
            resized = _fit_crop(crop, tile_width - 16, tile_height - 16, scale)
            left = (tile_width - resized.width) // 2
            top = (tile_height - resized.height) // 2
            tile_x = (index % columns) * tile_width
            tile_y = (index // columns) * tile_height
            sheet.paste(resized, (tile_x + left, tile_y + top))
            tiles.append({
                "index": index,
                "source_region_ids": [item["id"] for item in group],
                "source_bboxes": [item["bbox"] for item in group],
                "crop_bbox": list(crop_box),
                "tile_bbox": [tile_x, tile_y, tile_width, tile_height],
            })
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="PNG")
    return {
        "version": TEXT_FOCUS_VERSION,
        "status": "created",
        "path": str(destination),
        "source_image": str(source),
        "source_sha256": _sha256(source),
        "excluded_region_ids": excluded_region_ids,
        "excluded_region_policy": {
            "max_region_width_px": max_region_width,
            "max_region_height_px": max_region_height,
            "reason": "large_connected_component_is_not_a_local_label_view",
        },
        "configuration": {
            "padding_px": padding_px,
            "scale": scale,
            "max_regions": max_regions,
            "columns": columns,
            "tile_width": tile_width,
            "tile_height": tile_height,
        },
        "tiles": tiles,
    }


def _normalize_regions(
    regions: Sequence[Mapping[str, Any]],
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    normalized = []
    seen = set()
    for region in regions:
        if not isinstance(region, Mapping):
            continue
        region_id = str(region.get("id") or "").strip()
        bbox = region.get("bbox")
        if not region_id or region_id in seen or not isinstance(bbox, Sequence) or len(bbox) != 4:
            continue
        try:
            x, y, width, height = (float(value) for value in bbox)
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            continue
        if width <= 0 or height <= 0:
            continue
        left = max(0.0, min(float(image_width), x))
        top = max(0.0, min(float(image_height), y))
        right = max(left, min(float(image_width), x + width))
        bottom = max(top, min(float(image_height), y + height))
        if right <= left or bottom <= top:
            continue
        seen.add(region_id)
        normalized.append({
            "id": region_id,
            "bbox": [round(left, 3), round(top, 3), round(right - left, 3), round(bottom - top, 3)],
        })
    normalized.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], item["id"]))
    return normalized


def _group_regions(regions: Sequence[Mapping[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for region in regions:
        current = dict(region)
        matching = []
        for index, group in enumerate(groups):
            if any(_same_label_line(current["bbox"], item["bbox"]) for item in group):
                matching.append(index)
        if not matching:
            groups.append([current])
            continue
        first = matching[0]
        groups[first].append(current)
        for index in reversed(matching[1:]):
            groups[first].extend(groups.pop(index))
    for group in groups:
        group.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], item["id"]))
    groups.sort(key=lambda group: (
        min(item["bbox"][1] for item in group),
        min(item["bbox"][0] for item in group),
    ))
    return groups


def _same_label_line(left: Sequence[float], right: Sequence[float]) -> bool:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    left_bottom = ly + lh
    right_bottom = ry + rh
    vertical_gap = max(0.0, max(ly, ry) - min(left_bottom, right_bottom))
    center_delta = abs((ly + left_bottom) / 2.0 - (ry + right_bottom) / 2.0)
    horizontal_gap = max(0.0, max(lx, rx) - min(lx + lw, rx + rw))
    max_height = max(lh, rh)
    return (
        vertical_gap <= max(4.0, 0.35 * min(lh, rh))
        and center_delta <= max(8.0, 0.9 * max_height)
        and horizontal_gap <= max(18.0, 2.5 * max_height)
    )


def _expanded_box(group: Sequence[Mapping[str, Any]], image_width: int, image_height: int, padding: int) -> tuple[int, int, int, int]:
    left = min(float(item["bbox"][0]) for item in group)
    top = min(float(item["bbox"][1]) for item in group)
    right = max(float(item["bbox"][0]) + float(item["bbox"][2]) for item in group)
    bottom = max(float(item["bbox"][1]) + float(item["bbox"][3]) for item in group)
    return (
        max(0, int(math.floor(left - padding))),
        max(0, int(math.floor(top - padding))),
        min(image_width, int(math.ceil(right + padding))),
        min(image_height, int(math.ceil(bottom + padding))),
    )


def _fit_crop(image: Image.Image, max_width: int, max_height: int, scale: int) -> Image.Image:
    target_width = max(1, image.width * scale)
    target_height = max(1, image.height * scale)
    factor = min(1.0, max_width / target_width, max_height / target_height)
    size = (
        max(1, int(round(target_width * factor))),
        max(1, int(round(target_height * factor))),
    )
    return image.resize(size, Image.Resampling.LANCZOS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
