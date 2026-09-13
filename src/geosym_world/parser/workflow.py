"""End-to-end typed parsing workflow with admission and graded reconstruction."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from geosym_world.schema import AdmissionStatus
from geosym_world.schema.canonical import stable_checksum
from geosym_world.verifier import (
    ParseAdmissionPolicy,
    ParseGraphAdmission,
    ParseGraphAdmissionReport,
    PixelTopologyReport,
    PixelTopologyVerifier,
)

from .contracts import ParserInput, validate_parser_input
from .reconstruction import ParseGraphReconstructor, ReconstructionMode, ReconstructionReport
from .redraw import ParseRedrawPipeline, RedrawMode, RedrawResult, ResynthesisTransform
from .typed_v0 import TypedDiagramParserV0, TypedParseResult


PARSER_WORKFLOW_SCHEMA_VERSION = "1.1"


class TypedParser(Protocol):
    def parse(self, data: ParserInput | Mapping[str, Any]) -> TypedParseResult: ...


@dataclass(frozen=True)
class VerifiedParseResult:
    parser_result: TypedParseResult
    pixel_topology: PixelTopologyReport
    admission: ParseGraphAdmissionReport
    reconstruction: ReconstructionReport
    redraws: tuple[RedrawResult, ...]
    diagnostics: tuple[dict[str, Any], ...]
    schema_version: str = PARSER_WORKFLOW_SCHEMA_VERSION

    @property
    def id(self) -> str:
        return self.parser_result.parser_input.id

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "id": self.id,
            "parser": self.parser_result.to_dict(),
            "pixel_topology": self.pixel_topology.to_dict(),
            "admission": self.admission.to_dict(),
            "reconstruction": self.reconstruction.to_dict(),
            "redraws": [redraw.to_dict() for redraw in self.redraws],
            "diagnostics": [dict(item) for item in self.diagnostics],
        }
        payload["checksum"] = stable_checksum(payload)
        return payload


class VerifiedParseWorkflow:
    """Run public input through proposal, admission, reconstruction, and redraw."""

    def __init__(
        self,
        *,
        parser: TypedParser | None = None,
        admission: ParseGraphAdmission | None = None,
        pixel_topology: PixelTopologyVerifier | None = None,
        reconstructor: ParseGraphReconstructor | None = None,
        redraw: ParseRedrawPipeline | None = None,
    ):
        self.parser = parser or TypedDiagramParserV0()
        self.admission = admission or ParseGraphAdmission(
            ParseAdmissionPolicy(require_pixel_topology=True, policy_id="verified_pixels_v1")
        )
        self.pixel_topology = pixel_topology or PixelTopologyVerifier()
        self.reconstructor = reconstructor or ParseGraphReconstructor()
        self.redraw = redraw or ParseRedrawPipeline()

    def run(
        self,
        data: ParserInput | Mapping[str, Any],
        *,
        reconstruction_mode: ReconstructionMode = ReconstructionMode.PIXEL_WITNESS,
        resynthesis_seeds: tuple[int, ...] = (),
        resynthesis_transform: ResynthesisTransform = ResynthesisTransform.RIGID,
    ) -> VerifiedParseResult:
        parser_input = validate_parser_input(data)
        if not isinstance(resynthesis_transform, ResynthesisTransform):
            raise ValueError(f"Unknown resynthesis transform: {resynthesis_transform}")
        parsed = self.parser.parse(parser_input)
        pixel_topology = self.pixel_topology.verify(parsed.graph, parser_input.image)
        admitted = self.admission.admit(parsed.graph, pixel_topology=pixel_topology)
        reconstructed = self.reconstructor.reconstruct(admitted, mode=reconstruction_mode)
        if resynthesis_seeds and reconstructed.status.value != "exact":
            raise ValueError("Resynthesis seeds require an exact reconstruction")
        source_redraw = self.redraw.render(admitted, mode=RedrawMode.SOURCE)
        redraws = [] if _is_empty_render_scene(source_redraw.scene) else [source_redraw]
        if reconstructed.state is not None:
            redraws.append(
                self.redraw.render(
                    admitted,
                    mode=RedrawMode.SEMANTIC,
                    reconstruction=reconstructed,
                )
            )
        if reconstructed.status.value == "exact":
            redraws.extend(
                self.redraw.render(
                    admitted,
                    mode=RedrawMode.RESYNTHESIS,
                    reconstruction=reconstructed,
                    seed=seed,
                    resynthesis_transform=resynthesis_transform,
                )
                for seed in resynthesis_seeds
            )
        diagnostics = _diagnostics(pixel_topology, admitted, reconstructed, redraws)
        if _is_empty_render_scene(source_redraw.scene):
            diagnostics = (
                *diagnostics,
                {
                    "stage": RedrawMode.SOURCE.value,
                    "code": "render.empty_scene",
                    "candidate_id": None,
                    "message": "Admitted parse contains no renderable geometry; source redraw omitted",
                },
            )
        return VerifiedParseResult(parsed, pixel_topology, admitted, reconstructed, tuple(redraws), diagnostics)


def _is_empty_render_scene(scene) -> bool:
    return not scene.primitives and not scene.labels


def summarize_workflow_results(results: list[VerifiedParseResult]) -> dict[str, Any]:
    reconstruction_counts = Counter(result.reconstruction.status.value for result in results)
    admission_counts = Counter(
        decision.decision.value
        for result in results
        for decision in result.admission.decisions
    )
    diagnostic_counts = Counter(
        str(diagnostic["code"])
        for result in results
        for diagnostic in result.diagnostics
    )
    return {
        "num_records": len(results),
        "reconstruction_status_counts": dict(sorted(reconstruction_counts.items())),
        "candidate_admission_counts": dict(sorted(admission_counts.items())),
        "pixel_topology_status_counts": dict(
            sorted(Counter(result.pixel_topology.status for result in results).items())
        ),
        "diagnostic_counts": dict(sorted(diagnostic_counts.items())),
        "num_semantic_redraws": sum(
            redraw.mode is RedrawMode.SEMANTIC
            for result in results
            for redraw in result.redraws
        ),
        "num_resyntheses": sum(
            redraw.mode is RedrawMode.RESYNTHESIS
            for result in results
            for redraw in result.redraws
        ),
    }


def _diagnostics(
    pixel_topology: PixelTopologyReport,
    admission: ParseGraphAdmissionReport,
    reconstruction: ReconstructionReport,
    redraws: list[RedrawResult],
) -> tuple[dict[str, Any], ...]:
    diagnostics = []
    for check in pixel_topology.checks:
        if check.status.value == "passed":
            continue
        diagnostics.append(
            {
                "stage": "pixel_topology",
                "code": f"pixel_topology.{check.code}",
                "candidate_id": check.candidate_id,
                "status": check.status.value,
            }
        )
    for decision in admission.decisions:
        if decision.decision is AdmissionStatus.ADMITTED:
            continue
        diagnostics.append(
            {
                "stage": "admission",
                "code": f"admission.{decision.reason}",
                "candidate_id": decision.candidate_id,
                "decision": decision.decision.value,
            }
        )
    diagnostics.extend(
        {
            "stage": "reconstruction",
            "code": issue.code,
            "candidate_id": issue.candidate_id,
            "message": issue.message,
        }
        for issue in reconstruction.issues
    )
    diagnostics.extend(
        {
            "stage": "reconstruction_verifier",
            "code": f"reconstruction.{issue['code']}",
            "candidate_id": None,
            "message": issue["message"],
        }
        for issue in reconstruction.verification.get("issues", ())
    )
    for redraw in redraws:
        quality = redraw.scene.metadata.get("render_quality") or {}
        for issue in quality.get("issues") or ():
            diagnostics.append(
                {
                    "stage": redraw.mode.value,
                    "code": f"render.{issue['code']}",
                    "candidate_id": None,
                    "message": issue["message"],
                }
            )
    return tuple(diagnostics)
