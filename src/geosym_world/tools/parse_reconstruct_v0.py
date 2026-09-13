"""Run typed image parsing, admission, reconstruction, and redraw over public JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from geosym_world.parser import (
    BranchRedrawResult,
    ExternalParseGraphPrediction,
    FusedParseGraphParser,
    GeoparsingAdapter,
    RecordedParseGraphParser,
    ReconstructionMode,
    ResynthesisTransform,
    TesseractLabelRecognizer,
    TypedDiagramParserV0,
    VerifiedParseWorkflow,
    image_sha256,
    load_geoparsing_predictions,
    load_ocr_predictions,
    summarize_workflow_results,
)
from geosym_world.parser.contracts import validate_parser_input
from geosym_world.rendering import write_png, write_svg
from geosym_world.schema.canonical import stable_checksum
from geosym_world.verifier import ParseAdmissionPolicy, ParseGraphAdmission


DEFAULT_OUTPUT = "artifacts/parser/verified_v0/results.jsonl"
OUTPUT_FORMATS = ("none", "svg", "png", "both")
PARSER_RUN_MANIFEST_SCHEMA_VERSION = "geosym.parser_run_manifest.v8"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Public ParserInput JSONL path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Verified workflow result JSONL path.")
    prediction_group = parser.add_mutually_exclusive_group()
    prediction_group.add_argument(
        "--external-predictions",
        default=None,
        help="Optional checksum-bound ExternalParseGraphPrediction JSONL; bypasses the built-in detector.",
    )
    parser.add_argument(
        "--ocr-predictions",
        default=None,
        help="Optional checksum-bound OCRImagePrediction JSONL for the built-in detector.",
    )
    parser.add_argument(
        "--tesseract-ocr",
        action="store_true",
        help="Run the optional Tesseract OCR provider on detected text regions.",
    )
    prediction_group.add_argument(
        "--geoparsing-predictions",
        default=None,
        help="Optional checksum-bound GeoparsingPrediction JSONL; imports GDP formal language as hypotheses.",
    )
    parser.add_argument(
        "--fuse-geoparsing-visual",
        action="store_true",
        help="Ground Geoparsing references against the built-in visual ParseGraph before admission.",
    )
    parser.add_argument(
        "--fuse-external-visual",
        action="store_true",
        help="Fuse recorded VLM ParseGraphs with the built-in OpenCV visual ParseGraph.",
    )
    parser.add_argument(
        "--allow-partial-geoparsing",
        action="store_true",
        help="Keep supported candidates and record diagnostics instead of failing on unsupported GDP syntax.",
    )
    parser.add_argument(
        "--reconstruction-mode",
        choices=tuple(mode.value for mode in ReconstructionMode),
        default=ReconstructionMode.PIXEL_WITNESS.value,
    )
    parser.add_argument("--format", choices=OUTPUT_FORMATS, default="svg", help="Redraw artifact format.")
    parser.add_argument("--render-dir", default=None, help="Defaults to OUTPUT parent/renders.")
    parser.add_argument(
        "--enumerate-branches",
        action="store_true",
        help="Export every finite exact reconstruction branch without selecting one.",
    )
    parser.add_argument(
        "--max-branches",
        type=int,
        default=16,
        help="Maximum exact reconstruction branches exported per record.",
    )
    parser.add_argument("--resynthesis-seeds", default="", help="Comma-separated seeds; requires exact reconstruction.")
    parser.add_argument(
        "--resynthesis-transform",
        choices=tuple(item.value for item in ResynthesisTransform),
        default=ResynthesisTransform.RIGID.value,
        help="Exact transform family used for resynthesis seeds.",
    )
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--permit-unique-fill-shading",
        action="store_true",
        help=(
            "Admit a shaded-region relation only when an explicit shaded-area question, "
            "one observed fill region, and bound fill evidence agree."
        ),
    )
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every must be >= 0")
    if args.max_branches <= 0:
        parser.error("--max-branches must be positive")
    try:
        seeds = _parse_seeds(args.resynthesis_seeds)
    except ValueError as exc:
        parser.error(str(exc))
    if args.allow_partial_geoparsing and not args.geoparsing_predictions:
        parser.error("--allow-partial-geoparsing requires --geoparsing-predictions")
    if args.fuse_geoparsing_visual and not args.geoparsing_predictions:
        parser.error("--fuse-geoparsing-visual requires --geoparsing-predictions")
    if args.fuse_external_visual and not args.external_predictions:
        parser.error("--fuse-external-visual requires --external-predictions")
    if args.external_predictions and args.ocr_predictions:
        parser.error("--ocr-predictions cannot be combined with --external-predictions")
    if args.tesseract_ocr and (args.external_predictions or args.ocr_predictions):
        parser.error("--tesseract-ocr cannot be combined with external or recorded OCR predictions")
    if args.tesseract_ocr and args.geoparsing_predictions and not args.fuse_geoparsing_visual:
        parser.error("--tesseract-ocr with Geoparsing predictions requires --fuse-geoparsing-visual")
    if args.geoparsing_predictions and args.ocr_predictions and not args.fuse_geoparsing_visual:
        parser.error("Combining OCR and Geoparsing predictions requires --fuse-geoparsing-visual")

    manifest = run(
        input_path=Path(args.input),
        output_path=Path(args.output),
        render_dir=Path(args.render_dir) if args.render_dir else Path(args.output).parent / "renders",
        reconstruction_mode=ReconstructionMode(args.reconstruction_mode),
        enumerate_branches=args.enumerate_branches,
        max_branches=args.max_branches,
        output_format=args.format,
        resynthesis_seeds=seeds,
        resynthesis_transform=ResynthesisTransform(args.resynthesis_transform),
        width=args.width,
        height=args.height,
        progress_every=args.progress_every,
        external_predictions_path=Path(args.external_predictions) if args.external_predictions else None,
        ocr_predictions_path=Path(args.ocr_predictions) if args.ocr_predictions else None,
        tesseract_ocr=args.tesseract_ocr,
        geoparsing_predictions_path=(
            Path(args.geoparsing_predictions) if args.geoparsing_predictions else None
        ),
        geoparsing_strict=not args.allow_partial_geoparsing,
        fuse_geoparsing_visual=args.fuse_geoparsing_visual,
        fuse_external_visual=args.fuse_external_visual,
        permit_unique_fill_shading=args.permit_unique_fill_shading,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def run(
    *,
    input_path: Path,
    output_path: Path,
    render_dir: Path,
    reconstruction_mode: ReconstructionMode,
    enumerate_branches: bool = False,
    max_branches: int = 16,
    output_format: str,
    resynthesis_seeds: tuple[int, ...],
    width: int,
    height: int,
    progress_every: int,
    external_predictions_path: Path | None = None,
    ocr_predictions_path: Path | None = None,
    tesseract_ocr: bool = False,
    geoparsing_predictions_path: Path | None = None,
    geoparsing_strict: bool = True,
    fuse_geoparsing_visual: bool = False,
    fuse_external_visual: bool = False,
    permit_unique_fill_shading: bool = False,
    resynthesis_transform: ResynthesisTransform = ResynthesisTransform.RIGID,
) -> dict[str, Any]:
    if not input_path.is_file():
        raise ValueError(f"Parser input JSONL does not exist: {input_path}")
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"Unknown output format: {output_format}")
    if not isinstance(enumerate_branches, bool):
        raise ValueError("enumerate_branches must be boolean")
    if isinstance(max_branches, bool) or not isinstance(max_branches, int) or max_branches <= 0:
        raise ValueError("max_branches must be a positive integer")
    if enumerate_branches and reconstruction_mode is not ReconstructionMode.EXACT:
        raise ValueError("Branch enumeration requires exact reconstruction mode")
    if enumerate_branches and resynthesis_seeds:
        raise ValueError("Branch enumeration cannot use resynthesis seeds because no branch is selected")
    if not isinstance(resynthesis_transform, ResynthesisTransform):
        raise ValueError(f"Unknown resynthesis transform: {resynthesis_transform}")
    if external_predictions_path is not None and (
        ocr_predictions_path is not None or tesseract_ocr or geoparsing_predictions_path is not None
    ):
        raise ValueError("External ParseGraph predictions cannot be combined with OCR or Geoparsing predictions")
    if not isinstance(tesseract_ocr, bool):
        raise ValueError("tesseract_ocr must be boolean")
    if not isinstance(permit_unique_fill_shading, bool):
        raise ValueError("permit_unique_fill_shading must be boolean")
    if tesseract_ocr and ocr_predictions_path is not None:
        raise ValueError("Tesseract OCR cannot be combined with recorded OCR predictions")
    if tesseract_ocr and geoparsing_predictions_path is not None and not fuse_geoparsing_visual:
        raise ValueError("Tesseract OCR with Geoparsing predictions requires visual fusion")
    if fuse_geoparsing_visual and geoparsing_predictions_path is None:
        raise ValueError("Geoparsing visual fusion requires Geoparsing predictions")
    if (
        geoparsing_predictions_path is not None
        and ocr_predictions_path is not None
        and not fuse_geoparsing_visual
    ):
        raise ValueError("Combining OCR and Geoparsing predictions requires visual fusion")
    protected_inputs = {input_path.resolve()}
    if external_predictions_path is not None:
        protected_inputs.add(external_predictions_path.resolve())
    if ocr_predictions_path is not None:
        protected_inputs.add(ocr_predictions_path.resolve())
    if geoparsing_predictions_path is not None:
        protected_inputs.add(geoparsing_predictions_path.resolve())
    if output_path.resolve() in protected_inputs:
        raise ValueError("Verified Parser output must not overwrite an input file")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format != "none":
        render_dir.mkdir(parents=True, exist_ok=True)

    geoparsing_import = None
    if external_predictions_path is not None:
        recorded_parser = _load_recorded_parser(external_predictions_path)
    elif geoparsing_predictions_path is not None:
        recorded_parser, geoparsing_import = _load_geoparsing_parser(
            geoparsing_predictions_path,
            input_path,
            strict=geoparsing_strict,
        )
    else:
        recorded_parser = None
    recorded_ocr = load_ocr_predictions(ocr_predictions_path) if ocr_predictions_path else None
    active_ocr = TesseractLabelRecognizer() if tesseract_ocr else recorded_ocr
    if recorded_parser is not None:
        input_ids = _parser_input_ids(input_path)
        if input_ids != recorded_parser.prediction_ids:
            raise ValueError(
                "External prediction IDs differ from ParserInput IDs; "
                f"missing={sorted(input_ids - recorded_parser.prediction_ids)}, "
                f"unexpected={sorted(recorded_parser.prediction_ids - input_ids)}"
            )
    if recorded_ocr is not None:
        image_hashes = _parser_input_image_hashes(input_path)
        if image_hashes != recorded_ocr.image_hashes:
            raise ValueError(
                "OCR prediction image hashes differ from ParserInput images; "
                f"missing={sorted(image_hashes - recorded_ocr.image_hashes)}, "
                f"unexpected={sorted(recorded_ocr.image_hashes - image_hashes)}"
            )
    typed_parser = TypedDiagramParserV0(label_recognizer=active_ocr) if active_ocr is not None else None
    if fuse_external_visual:
        selected_parser = FusedParseGraphParser(
            visual_parser=typed_parser or TypedDiagramParserV0(),
            semantic_parser=recorded_parser,
        )
    elif fuse_geoparsing_visual:
        selected_parser = FusedParseGraphParser(
            visual_parser=typed_parser or TypedDiagramParserV0(),
            semantic_parser=recorded_parser,
        )
    else:
        selected_parser = recorded_parser or typed_parser
    workflow = VerifiedParseWorkflow(
        parser=selected_parser,
        admission=ParseGraphAdmission(
            ParseAdmissionPolicy(
                require_pixel_topology=True,
                permit_unique_fill_shading=permit_unique_fill_shading,
                policy_id=(
                    "verified_pixels_unique_fill_shading_v1"
                    if permit_unique_fill_shading
                    else "verified_pixels_v1"
                ),
            )
        ),
    )
    results = []
    seen_ids: set[str] = set()
    artifact_counts: Counter[str] = Counter()
    branch_status_counts: Counter[str] = Counter()
    candidate_branch_status_counts: Counter[str] = Counter()
    branch_rejection_code_counts: Counter[str] = Counter()
    branch_redraw_status_counts: Counter[str] = Counter()
    num_branch_sets = 0
    num_candidate_branches = 0
    num_enumerated_branches = 0
    num_rejected_branches = 0
    num_records_with_multiple_branches = 0
    num_branch_semantic_redraws = 0
    with input_path.open("r", encoding="utf-8") as input_file, output_path.open("w", encoding="utf-8") as output_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            payload = _json_object(line, input_path, line_number)
            record_id = str(payload.get("id", f"line-{line_number}"))
            if record_id in seen_ids:
                raise ValueError(f"Duplicate ParserInput id {record_id!r} at {input_path}:{line_number}")
            seen_ids.add(record_id)
            try:
                result = workflow.run(
                    payload,
                    reconstruction_mode=reconstruction_mode,
                    resynthesis_seeds=resynthesis_seeds,
                    resynthesis_transform=resynthesis_transform,
                )
                branch_set = (
                    workflow.reconstructor.enumerate_branches(
                        result.admission,
                        max_branches=max_branches,
                    )
                    if enumerate_branches
                    else None
                )
                branch_redraws = (
                    workflow.redraw.render_branches(result.admission, branch_set)
                    if branch_set is not None
                    else ()
                )
            except Exception as exc:
                raise ValueError(
                    f"Verified Parser workflow failed for record {record_id!r} at {input_path}:{line_number}: {exc}"
                ) from exc
            artifacts = _write_redraws(
                result,
                render_dir=render_dir,
                output_format=output_format,
                width=width,
                height=height,
                index=len(results),
            )
            artifact_counts.update(path.rsplit(".", 1)[-1] for path in artifacts.values())
            branch_redraw_rows = _write_branch_redraws(
                result.id,
                branch_redraws,
                render_dir=render_dir,
                output_format=output_format,
                width=width,
                height=height,
                index=len(results),
            )
            artifact_counts.update(
                path.rsplit(".", 1)[-1]
                for branch_row in branch_redraw_rows
                for path in branch_row["artifacts"].values()
            )
            result_payload = result.to_dict()
            branch_payload = branch_set.to_dict() if branch_set is not None else None
            row = {
                "result": result_payload,
                "branch_set": branch_payload,
                "branch_redraws": branch_redraw_rows,
                "artifacts": artifacts,
            }
            row["checksum"] = stable_checksum(row)
            output_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            results.append(result)
            if branch_set is not None:
                num_branch_sets += 1
                candidate_branches = (*branch_set.branches, *branch_set.rejected_branches)
                num_candidate_branches += len(candidate_branches)
                num_enumerated_branches += len(branch_set.branches)
                num_rejected_branches += len(branch_set.rejected_branches)
                num_records_with_multiple_branches += len(branch_set.branches) > 1
                branch_status_counts.update(branch.status.value for branch in branch_set.branches)
                candidate_branch_status_counts.update(
                    branch.status.value for branch in candidate_branches
                )
                branch_rejection_code_counts.update(
                    issue.code
                    for branch in branch_set.rejected_branches
                    for issue in branch.issues
                )
                num_branch_semantic_redraws += len(branch_redraws)
                branch_redraw_status_counts.update(
                    "valid" if item.redraw.verification.get("valid") is True else "invalid"
                    for item in branch_redraws
                )
            if progress_every and len(results) % progress_every == 0:
                print(
                    f"[verified-parser-v0] processed={len(results)} "
                    f"status={result.reconstruction.status.value} diagnostics={len(result.diagnostics)}",
                    file=sys.stderr,
                    flush=True,
                )

    summary = summarize_workflow_results(results)
    summary.update(
        {
            "num_branch_sets": num_branch_sets,
            "num_candidate_branches": num_candidate_branches,
            "num_enumerated_branches": num_enumerated_branches,
            "num_rejected_branches": num_rejected_branches,
            "num_records_with_multiple_branches": num_records_with_multiple_branches,
            "branch_reconstruction_status_counts": dict(sorted(branch_status_counts.items())),
            "candidate_branch_status_counts": dict(sorted(candidate_branch_status_counts.items())),
            "branch_rejection_code_counts": dict(sorted(branch_rejection_code_counts.items())),
            "num_branch_semantic_redraws": num_branch_semantic_redraws,
            "branch_redraw_status_counts": dict(sorted(branch_redraw_status_counts.items())),
        }
    )
    manifest = {
        "schema_version": PARSER_RUN_MANIFEST_SCHEMA_VERSION,
        "input": str(input_path),
        "input_sha256": _file_hash(input_path),
        "output": str(output_path),
        "output_sha256": _file_hash(output_path),
        "render_dir": str(render_dir) if output_format != "none" else None,
        "external_predictions": str(external_predictions_path) if external_predictions_path else None,
        "external_predictions_sha256": _file_hash(external_predictions_path) if external_predictions_path else None,
        "ocr_predictions": str(ocr_predictions_path) if ocr_predictions_path else None,
        "ocr_predictions_sha256": _file_hash(ocr_predictions_path) if ocr_predictions_path else None,
        "geoparsing_predictions": str(geoparsing_predictions_path) if geoparsing_predictions_path else None,
        "geoparsing_predictions_sha256": (
            _file_hash(geoparsing_predictions_path) if geoparsing_predictions_path else None
        ),
        "geoparsing_import": geoparsing_import,
        "configuration": {
            "parser": (
                selected_parser.configuration()
                if selected_parser is not None and hasattr(selected_parser, "configuration")
                else {
                    "kind": "opencv_typed_v0",
                    "ocr": active_ocr.configuration() if active_ocr is not None else None,
                }
            ),
            "reconstruction_mode": reconstruction_mode.value,
            "enumerate_branches": enumerate_branches,
            "max_branches": max_branches,
            "output_format": output_format,
            "resynthesis_seeds": list(resynthesis_seeds),
            "resynthesis_transform": resynthesis_transform.value,
            "width": width,
            "height": height,
            "progress_every": progress_every,
            "geoparsing_strict": geoparsing_strict if geoparsing_predictions_path else None,
            "fuse_geoparsing_visual": fuse_geoparsing_visual,
            "fuse_external_visual": fuse_external_visual,
            "admission_policy": workflow.admission.policy.to_dict(),
            "pixel_topology": workflow.pixel_topology.config.to_dict(),
        },
        "summary": summary,
        "artifact_counts": dict(sorted(artifact_counts.items())),
    }
    manifest["checksum"] = stable_checksum(manifest)
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**manifest, "manifest": str(manifest_path)}


def _write_redraws(result, *, render_dir: Path, output_format: str, width: int, height: int, index: int):
    if output_format == "none":
        return {}
    artifacts = {}
    stem = f"{index:06d}_{hashlib.sha256(result.id.encode('utf-8')).hexdigest()[:12]}"
    for redraw_index, redraw in enumerate(result.redraws):
        mode_dir = render_dir / redraw.mode.value
        mode_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{redraw_index}" if redraw.mode.value == "resynthesis" else ""
        base = mode_dir / f"{stem}{suffix}"
        if output_format in {"svg", "both"}:
            path = base.with_suffix(".svg")
            write_svg(redraw.scene, path, width=width, height=height)
            artifacts[f"{redraw.mode.value}{suffix}_svg"] = str(path)
        if output_format in {"png", "both"}:
            path = base.with_suffix(".png")
            write_png(redraw.scene, path, width=width, height=height)
            artifacts[f"{redraw.mode.value}{suffix}_png"] = str(path)
    return artifacts


def _write_branch_redraws(
    record_id: str,
    branch_redraws: Sequence[BranchRedrawResult],
    *,
    render_dir: Path,
    output_format: str,
    width: int,
    height: int,
    index: int,
) -> list[dict[str, Any]]:
    stem = f"{index:06d}_{hashlib.sha256(record_id.encode('utf-8')).hexdigest()[:12]}"
    rows = []
    for branch_index, branch_redraw in enumerate(branch_redraws):
        artifacts = {}
        base = (
            render_dir
            / "semantic_redraw_branches"
            / f"{stem}_{branch_index:03d}_{branch_redraw.branch_checksum[:12]}"
        )
        if output_format != "none":
            base.parent.mkdir(parents=True, exist_ok=True)
        if output_format in {"svg", "both"}:
            path = base.with_suffix(".svg")
            write_svg(branch_redraw.redraw.scene, path, width=width, height=height)
            artifacts["semantic_redraw_svg"] = str(path)
        if output_format in {"png", "both"}:
            path = base.with_suffix(".png")
            write_png(branch_redraw.redraw.scene, path, width=width, height=height)
            artifacts["semantic_redraw_png"] = str(path)
        payload = branch_redraw.to_dict()
        row = {"result": payload, "artifacts": artifacts}
        row["checksum"] = stable_checksum(row)
        rows.append(row)
    return rows


def _json_object(line: str, input_path: Path, line_number: int) -> dict[str, Any]:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {input_path}:{line_number}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object at {input_path}:{line_number}")
    return payload


def _load_recorded_parser(path: Path) -> RecordedParseGraphParser:
    if not path.is_file():
        raise ValueError(f"External prediction JSONL does not exist: {path}")
    predictions = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = _json_object(line, path, line_number)
            try:
                predictions.append(ExternalParseGraphPrediction.from_dict(payload))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid external prediction at {path}:{line_number}: {exc}") from exc
    return RecordedParseGraphParser(predictions)


def _load_geoparsing_parser(
    path: Path,
    input_path: Path,
    *,
    strict: bool,
) -> tuple[RecordedParseGraphParser, dict[str, Any]]:
    predictions = load_geoparsing_predictions(path)
    parser_inputs = _parser_inputs_by_id(input_path)
    prediction_ids = {prediction.id for prediction in predictions}
    input_ids = set(parser_inputs)
    if prediction_ids != input_ids:
        raise ValueError(
            "Geoparsing prediction IDs differ from ParserInput IDs; "
            f"missing={sorted(input_ids - prediction_ids)}, "
            f"unexpected={sorted(prediction_ids - input_ids)}"
        )
    adapter = GeoparsingAdapter(strict=strict)
    imported = [adapter.convert(parser_inputs[prediction.id], prediction) for prediction in predictions]
    diagnostic_counts = Counter(
        diagnostic.code
        for result in imported
        for diagnostic in result.diagnostics
    )
    return (
        RecordedParseGraphParser(tuple(result.prediction for result in imported)),
        {
            "strict": strict,
            "num_predictions": len(imported),
            "num_complete": sum(result.complete for result in imported),
            "num_with_diagnostics": sum(not result.complete for result in imported),
            "diagnostic_counts": dict(sorted(diagnostic_counts.items())),
        },
    )


def _parser_inputs_by_id(path: Path) -> dict[str, Any]:
    parser_inputs = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            parser_input = validate_parser_input(_json_object(line, path, line_number))
            if parser_input.id in parser_inputs:
                raise ValueError(f"Duplicate ParserInput id {parser_input.id!r} at {path}:{line_number}")
            parser_inputs[parser_input.id] = parser_input
    if not parser_inputs:
        raise ValueError(f"Parser input JSONL is empty: {path}")
    return parser_inputs


def _parser_input_ids(path: Path) -> frozenset[str]:
    ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = _json_object(line, path, line_number)
            record_id = payload.get("id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"ParserInput is missing id at {path}:{line_number}")
            if record_id in ids:
                raise ValueError(f"Duplicate ParserInput id {record_id!r} at {path}:{line_number}")
            ids.add(record_id)
    if not ids:
        raise ValueError(f"Parser input JSONL is empty: {path}")
    return frozenset(ids)


def _parser_input_image_hashes(path: Path) -> frozenset[str]:
    hashes = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            parser_input = validate_parser_input(_json_object(line, path, line_number))
            hashes.add(image_sha256(parser_input.image))
    if not hashes:
        raise ValueError(f"Parser input JSONL is empty: {path}")
    return frozenset(hashes)


def _parse_seeds(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("--resynthesis-seeds must be comma-separated integers") from exc
    if len(set(seeds)) != len(seeds):
        raise ValueError("--resynthesis-seeds cannot contain duplicates")
    return seeds


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
