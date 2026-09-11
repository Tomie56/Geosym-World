"""Deterministic perturbation benchmark for image-only parser inputs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from geosym_world.errors import InvalidGeometryInput
from geosym_world.parser.clean_v0 import CleanDiagramParserV0
from geosym_world.parser.contracts import ParserInput, validate_parser_input
from geosym_world.schema.canonical import stable_checksum

from .parser_dataset import validate_parser_reference
from .parser_v0_evaluator import CleanParserEvaluatorV0


class ParserPerturbationBenchmark:
    def __init__(self, *, parser: CleanDiagramParserV0 | None = None, evaluator: CleanParserEvaluatorV0 | None = None):
        self.parser = parser or CleanDiagramParserV0()
        self.evaluator = evaluator or CleanParserEvaluatorV0()

    def run(
        self,
        parser_input: ParserInput | Mapping[str, Any],
        reference: Mapping[str, Any],
        *,
        output_dir: str | Path,
        seed: int = 0,
    ) -> dict[str, Any]:
        source_input = validate_parser_input(parser_input)
        validate_parser_reference(reference, source_input.to_dict())
        image = cv2.imread(source_input.image, cv2.IMREAD_COLOR)
        if image is None:
            raise InvalidGeometryInput(f"Perturbation benchmark could not read image: {source_input.image}")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        variants = _variants(image, seed)
        rows = []
        for name, variant in variants:
            path = output / f"{source_input.id}.{name}.png"
            if not cv2.imwrite(str(path), variant):
                raise InvalidGeometryInput(f"Failed to write perturbed parser image: {path}")
            variant_input = ParserInput(
                id=f"{source_input.id}_{name}",
                image=str(path),
                question=source_input.question,
            )
            result = self.parser.parse(variant_input)
            metrics = self.evaluator.evaluate(result.record.hypotheses[0].payload, reference)
            rows.append({"variant": name, "input": variant_input.to_dict(), "metrics": metrics, "stage_trace": list(result.stages)})
        clean_score = next(row["metrics"]["macro_score"] for row in rows if row["variant"] == "clean")
        for row in rows:
            row["score_drop_from_clean"] = clean_score - row["metrics"]["macro_score"]
        report = {
            "schema_version": "1.0",
            "source_id": source_input.id,
            "seed": seed,
            "variants": rows,
            "clean_score": clean_score,
            "worst_score": min(row["metrics"]["macro_score"] for row in rows),
        }
        report["checksum"] = stable_checksum(report)
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report


def _variants(image: np.ndarray, seed: int) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 10.0, image.shape)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(image, (5, 5), 1.1)
    low_contrast = cv2.convertScaleAbs(image, alpha=0.65, beta=55)
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), 2.0, 1.0)
    rotated = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    return [("clean", image.copy()), ("noise", noisy), ("blur", blurred), ("low_contrast", low_contrast), ("rotate_2deg", rotated)]
