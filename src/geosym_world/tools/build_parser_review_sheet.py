"""Build a compact original-versus-redraw contact sheet for a Parser run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_COLUMNS = 2
DEFAULT_CELL_WIDTH = 760
DEFAULT_CELL_HEIGHT = 390
BACKGROUND = (255, 255, 255)
TEXT = (25, 25, 25)
MUTED = (90, 90, 90)


def _font(size: int):
    return ImageFont.load_default(size=size)


def _summary_by_id(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "summary.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("samples", ()) if isinstance(payload, dict) else ()
    return {
        str(row["id"]): row
        for row in rows
        if isinstance(row, dict) and row.get("id")
    }


def _record_summary(run_dir: Path, sample_id: str) -> dict[str, Any]:
    path = run_dir / "json" / f"{sample_id}.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    pipeline = payload.get("pipeline") if isinstance(payload.get("pipeline"), dict) else payload
    if isinstance(pipeline, dict) and isinstance(pipeline.get("prediction"), dict):
        quality = pipeline.get("quality") or {}
        realization = pipeline.get("realization") or {}
        prediction = pipeline.get("prediction") or {}
        graph = prediction.get("parse_graph") or {}
        return {
            "status": realization.get("status", payload.get("status", "unknown")),
            "objects": len(graph.get("objects", ())),
            "relations": len(graph.get("relations", ())),
            "quantities": len(graph.get("quantities", ())),
            "targets": len(graph.get("targets", ())),
            "quality": quality,
            "unresolved": realization.get("unresolved_constraints", ()),
            "diagnostics": realization.get("diagnostics", ()),
        }
    return {}


def _as_text(value: Any, default: str = "?") -> str:
    if value is None:
        return default
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _panel(image_path: Path | None, size: tuple[int, int], title: str) -> Image.Image:
    panel = Image.new("RGB", size, BACKGROUND)
    draw = ImageDraw.Draw(panel)
    draw.text((10, 5), title, fill=TEXT, font=_font(16))
    if image_path is None or not image_path.is_file():
        draw.text((size[0] // 2, size[1] // 2), "NO REDRAW", fill=MUTED, font=_font(20), anchor="mm")
        return panel
    with Image.open(image_path) as opened:
        fitted = ImageOps.contain(opened.convert("RGB"), (size[0] - 20, size[1] - 38))
    panel.paste(fitted, ((size[0] - fitted.width) // 2, 31 + (size[1] - 38 - fitted.height) // 2))
    return panel


def _cell(sample_dir: Path, row: dict[str, Any], width: int, height: int) -> Image.Image:
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    sample_id = sample_dir.name
    draw.text((10, 4), sample_id, fill=TEXT, font=_font(13))
    panel_width = width // 2 - 15
    panel_height = height - 83
    original = _panel(sample_dir / "input.png", (panel_width, panel_height), "original")
    redraw_path = next(
        (sample_dir / name for name in ("redraw.png", "seed.png") if (sample_dir / name).is_file()),
        None,
    )
    redraw = _panel(redraw_path, (panel_width, panel_height), "candidate state")
    canvas.paste(original, (5, 25))
    canvas.paste(redraw, (width // 2 + 5, 25))
    quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
    if not quality:
        quality = row
    status = _as_text(row.get("status"), "unknown")
    line = (
        f"status={status}  objects={_as_text(row.get('objects'))} "
        f"relations={_as_text(row.get('relations'))} "
        f"quantities={_as_text(row.get('quantities'))}"
    )
    checks = (
        f"seed={_as_text(quality.get('seed_realized'))} "
        f"constraints={_as_text(quality.get('constraint_satisfied'))} "
        f"source={_as_text(quality.get('source_reconstruction_passed'), 'unassessed')}"
    )
    diagnostics = row.get("diagnostics", ())
    if not redraw_path and diagnostics:
        checks = "reason=" + str(diagnostics[0])
    elif not redraw_path and row.get("error"):
        checks = f"{row.get('error_type', 'error')}: {row['error']}"[:110]
    draw.text((10, height - 48), line, fill=MUTED, font=_font(12))
    draw.text((10, height - 27), checks, fill=MUTED, font=_font(12))
    return canvas


def build_contact_sheet(
    run_dir: str | Path,
    output: str | Path | None = None,
    *,
    columns: int = DEFAULT_COLUMNS,
    cell_width: int = DEFAULT_CELL_WIDTH,
    cell_height: int = DEFAULT_CELL_HEIGHT,
) -> Path:
    run_path = Path(run_dir)
    image_root = run_path / "images"
    if not image_root.is_dir():
        raise FileNotFoundError(f"Parser run image directory does not exist: {image_root}")
    if columns <= 0 or cell_width < 300 or cell_height < 150:
        raise ValueError("columns must be positive; cells must be at least 300 by 150 pixels")
    sample_dirs = sorted(path for path in image_root.iterdir() if path.is_dir() and (path / "input.png").is_file())
    if not sample_dirs:
        raise ValueError(f"No Parser sample images found under {image_root}")
    summaries = _summary_by_id(run_path)
    rows = {sample_dir.name: {**_record_summary(run_path, sample_dir.name), **summaries.get(sample_dir.name, {})} for sample_dir in sample_dirs}
    row_count = (len(sample_dirs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, row_count * cell_height), BACKGROUND)
    for index, sample_dir in enumerate(sample_dirs):
        x = index % columns * cell_width
        y = index // columns * cell_height
        sheet.paste(_cell(sample_dir, rows[sample_dir.name], cell_width, cell_height), (x, y))
    destination = Path(output) if output is not None else run_path / "contact.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--columns", type=int, default=DEFAULT_COLUMNS)
    parser.add_argument("--cell-width", type=int, default=DEFAULT_CELL_WIDTH)
    parser.add_argument("--cell-height", type=int, default=DEFAULT_CELL_HEIGHT)
    args = parser.parse_args()
    destination = build_contact_sheet(
        args.run_dir,
        args.output,
        columns=args.columns,
        cell_width=args.cell_width,
        cell_height=args.cell_height,
    )
    print(f"[parser-review] wrote={destination}", flush=True)


if __name__ == "__main__":
    main()
