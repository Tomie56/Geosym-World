#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from geosym_world.construction import ConstructionBuilder
from geosym_world.schema.derivations import DerivationTracer
from geosym_world.schema.solutions import build_solution_paths
from geosym_world.solver import FactClosureSolver
from geosym_world.synthesis import SyntheticRecordExporter, SyntheticSpec
from geosym_world.verifier import TraceVerifier


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "generated" / "review_diverse_2d3d_20260910"
COMPLEX_2D_GROUPS = (
    (
        "triangle_similarity",
        (
            "triangle_partition_area",
            "triangle_partition_area_ratio",
            "parallel_cut_similarity",
            "similarity_dual_proof",
        ),
    ),
    (
        "curved_regions",
        (
            "circular_segment_area",
            "circular_segment_perimeter",
            "circular_segment_area_ratio",
            "equal_circle_lens_area",
            "equal_circle_lens_perimeter",
            "circle_lens_area",
            "circle_lens_perimeter",
        ),
    ),
    (
        "composite_regions",
        (
            "double_rectangular_hole_area",
            "double_rectangular_hole_perimeter",
            "concave_composite_area",
            "annulus_area",
            "annulus_perimeter",
        ),
    ),
    (
        "quadrilateral_relations",
        (
            "rhombus_diagonal_theorem",
            "cyclic_quadrilateral",
            "tangential_quadrilateral",
        ),
    ),
)


@dataclass(frozen=True)
class PolyhedronCase:
    shape: str
    token: str
    description: str
    points: tuple[tuple[str, tuple[str, str, str]], ...]
    vertices: tuple[str, ...]
    faces: tuple[tuple[str, ...], ...]
    dimensions: tuple[tuple[str, str, str], ...]
    section_height: str


def _cases() -> tuple[PolyhedronCase, ...]:
    return (
        _tetrahedron_case(4, 3, 5),
        _tetrahedron_case(6, 4, 3),
        _cuboid_case(4, 3, 2),
        _cuboid_case(6, 4, 3),
        _triangular_prism_case(3, 4, 5),
        _triangular_prism_case(5, 2, 4),
        _square_pyramid_case(4, 3),
        _square_pyramid_case(6, 4),
    )


def _tetrahedron_case(x: int, y: int, z: int) -> PolyhedronCase:
    return PolyhedronCase(
        shape="tetrahedron",
        token=f"x{x}_y{y}_z{z}",
        description=f"right tetrahedron with mutually perpendicular edges OX={x}, OY={y}, and OZ={z}",
        points=(
            ("O", ("0", "0", "0")),
            ("X", (str(x), "0", "0")),
            ("Y", ("0", str(y), "0")),
            ("Z", ("0", "0", str(z))),
        ),
        vertices=("O", "X", "Y", "Z"),
        faces=(("O", "Y", "X"), ("O", "X", "Z"), ("O", "Z", "Y"), ("X", "Y", "Z")),
        dimensions=(("OX", "O", "X"), ("OY", "O", "Y"), ("OZ", "O", "Z")),
        section_height=f"{z}/2",
    )


def _cuboid_case(width: int, depth: int, height: int) -> PolyhedronCase:
    return PolyhedronCase(
        shape="cuboid",
        token=f"w{width}_d{depth}_h{height}",
        description=f"cuboid with width {width}, depth {depth}, and height {height}",
        points=(
            ("A", ("0", "0", "0")),
            ("B", (str(width), "0", "0")),
            ("C", (str(width), str(depth), "0")),
            ("D", ("0", str(depth), "0")),
            ("E", ("0", "0", str(height))),
            ("F", (str(width), "0", str(height))),
            ("G", (str(width), str(depth), str(height))),
            ("H", ("0", str(depth), str(height))),
        ),
        vertices=("A", "B", "C", "D", "E", "F", "G", "H"),
        faces=(
            ("A", "D", "C", "B"),
            ("E", "F", "G", "H"),
            ("A", "B", "F", "E"),
            ("B", "C", "G", "F"),
            ("C", "D", "H", "G"),
            ("D", "A", "E", "H"),
        ),
        dimensions=(("AB", "A", "B"), ("AD", "A", "D"), ("AE", "A", "E")),
        section_height=f"{height}/2",
    )


def _triangular_prism_case(base: int, side: int, height: int) -> PolyhedronCase:
    return PolyhedronCase(
        shape="triangular_prism",
        token=f"b{base}_s{side}_h{height}",
        description=f"right triangular prism with base legs {base} and {side}, and prism height {height}",
        points=(
            ("A", ("0", "0", "0")),
            ("B", (str(base), "0", "0")),
            ("C", ("0", str(side), "0")),
            ("D", ("0", "0", str(height))),
            ("E", (str(base), "0", str(height))),
            ("F", ("0", str(side), str(height))),
        ),
        vertices=("A", "B", "C", "D", "E", "F"),
        faces=(
            ("A", "C", "B"),
            ("D", "E", "F"),
            ("A", "B", "E", "D"),
            ("B", "C", "F", "E"),
            ("C", "A", "D", "F"),
        ),
        dimensions=(("AB", "A", "B"), ("AC", "A", "C"), ("AD", "A", "D")),
        section_height=f"{height}/2",
    )


def _square_pyramid_case(side: int, height: int) -> PolyhedronCase:
    half = side // 2
    return PolyhedronCase(
        shape="square_pyramid",
        token=f"s{side}_h{height}",
        description=f"right square pyramid with base side {side} and perpendicular height {height}",
        points=(
            ("A", ("0", "0", "0")),
            ("B", (str(side), "0", "0")),
            ("C", (str(side), str(side), "0")),
            ("D", ("0", str(side), "0")),
            ("O", (str(half), str(half), "0")),
            ("E", (str(half), str(half), str(height))),
        ),
        vertices=("A", "B", "C", "D", "E"),
        faces=(
            ("A", "D", "C", "B"),
            ("A", "B", "E"),
            ("B", "C", "E"),
            ("C", "D", "E"),
            ("D", "A", "E"),
        ),
        dimensions=(("AB", "A", "B"), ("OE", "O", "E")),
        section_height=f"{height}/2",
    )


def _base_trace(case: PolyhedronCase, polyhedron_id: str) -> list[dict[str, Any]]:
    trace = [
        {"op": "add_point3d", "id": point_id, "coord": list(coords)}
        for point_id, coords in case.points
    ]
    trace.extend(
        {
            "op": "add_segment",
            "id": segment_id,
            "points": [start, end],
            "metadata": {"show_length": True, "public_given": True},
        }
        for segment_id, start, end in case.dimensions
    )
    trace.append(
        {
            "op": "add_polyhedron",
            "id": polyhedron_id,
            "vertices": list(case.vertices),
            "faces": [list(face) for face in case.faces],
            "metadata": {"shape": case.shape},
        }
    )
    return trace


def _measurement_specs(case: PolyhedronCase) -> list[SyntheticSpec]:
    polyhedron_id = f"solid_{case.shape}"
    specs = []
    for target_type, target_text in (("volume", "volume"), ("surface_area", "total surface area")):
        trace = _base_trace(case, polyhedron_id)
        trace.append({"op": "set_target", "type": target_type, "object": polyhedron_id})
        specs.append(
            SyntheticSpec(
                id=f"{case.shape}_{case.token}_{target_type}",
                family=f"{case.shape}_measure",
                trace=trace,
                question=f"A {case.description}. Find its exact {target_text}.",
                difficulty="hard",
                diagnostic_axes=("3d_interface", case.shape, target_type),
                metadata={"shape": case.shape, "variant": case.token},
            )
        )
    return specs


def _section_specs(case: PolyhedronCase) -> list[SyntheticSpec]:
    polyhedron_id = f"solid_{case.shape}"
    specs = []
    for target_type, target_text in (("section_area", "area"), ("section_perimeter", "perimeter")):
        trace = _base_trace(case, polyhedron_id)
        trace.extend(
            (
                {
                    "op": "add_point3d",
                    "id": "M",
                    "coord": ["0", "0", case.section_height],
                    "metadata": {"render": False},
                },
                {"op": "add_plane", "id": "section_plane", "point": "M", "normal": ["0", "0", "1"]},
                {
                    "op": "section_polyhedron_with_plane",
                    "id": "section_horizontal",
                    "polyhedron": polyhedron_id,
                    "plane": "section_plane",
                },
                {"op": "set_target", "type": target_type, "object": "section_horizontal"},
            )
        )
        specs.append(
            SyntheticSpec(
                id=f"{case.shape}_{case.token}_horizontal_{target_type}",
                family=f"{case.shape}_section",
                trace=trace,
                question=(
                    f"A {case.description}. A horizontal plane at half the height cuts the solid. "
                    f"Find the exact {target_text} of the section."
                ),
                difficulty="hard",
                diagnostic_axes=("3d_interface", case.shape, "solid_section", target_type),
                metadata={"shape": case.shape, "variant": case.token, "section": "horizontal_half_height"},
            )
        )
    return specs


def _build_3d_specs() -> list[SyntheticSpec]:
    specs = []
    for case in _cases():
        specs.extend(_measurement_specs(case))
        specs.extend(_section_specs(case))
    return specs


def _validate_3d_specs(specs: list[SyntheticSpec]) -> dict[str, Any]:
    verifier = TraceVerifier()
    solver = FactClosureSolver(profile="full")
    family_counts = Counter()
    target_counts = Counter()
    for spec in specs:
        verification = verifier.verify_trace(spec.trace, execute=False)
        if not verification.valid:
            raise ValueError(f"Invalid trace for {spec.id}: {verification.to_dict()}")
        state = ConstructionBuilder().execute(spec.trace)
        facts = solver.solve(state)
        if state.target is None:
            raise ValueError(f"Missing target for {spec.id}")
        target_facts = facts.find(state.target.type, state.target.object)
        if not target_facts or target_facts[-1].value is None:
            raise ValueError(f"Unresolved target for {spec.id}: {state.target.to_dict()}")
        derivation = DerivationTracer().trace_target(facts)
        if not derivation["integrity"]["valid"]:
            raise ValueError(f"Invalid derivation for {spec.id}: {derivation['integrity']}")
        solutions = build_solution_paths(facts, state.target)
        if not solutions["validation"]["valid"]:
            raise ValueError(f"Invalid solution path for {spec.id}: {solutions['validation']}")
        family_counts[spec.family] += 1
        target_counts[state.target.type] += 1
    return {
        "num_specs": len(specs),
        "family_counts": dict(sorted(family_counts.items())),
        "target_counts": dict(sorted(target_counts.items())),
    }


def _export_3d(output_root: Path, *, seed: int, workers: int) -> dict[str, Any]:
    output = output_root / "geometry_3d" / "records.jsonl"
    specs = _build_3d_specs()
    audit = _validate_3d_specs(specs)
    run_config = {
        "pipeline": "diverse_deterministic_3d_spec_export",
        "seed": seed,
        "target_records": len(specs),
        "families": sorted(audit["family_counts"]),
        "solver_profile": "full",
        "format": "both",
        "workers": workers,
        "scene_payload": "metadata",
        "annotation_level": "givens",
        "relation_markers": True,
        "split_group_mode": "topology",
        "pre_export_audit": audit,
    }
    return SyntheticRecordExporter(
        solver_profile="full",
        annotation_level="givens",
        relation_markers=True,
    ).write_jsonl(
        specs,
        output,
        image_format="both",
        style_seed=seed,
        require_count=len(specs),
        run_config=run_config,
        workers=workers,
        scene_payload="metadata",
        progress_every=4,
    )


def _run_complex_2d_group(
    output_root: Path,
    *,
    group_name: str,
    families: tuple[str, ...],
    records: int,
    seed: int,
    workers: int,
) -> None:
    output = output_root / "complex_2d" / group_name / "records.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "geosym_world.tools.synthesize_records",
        "--output",
        str(output),
        "--max-records",
        str(records),
        "--families",
        ",".join(families),
        "--per-family",
        "12",
        "--pipeline",
        "--base-records",
        str(records * 3),
        "--base-reserve-records",
        str(records * 3),
        "--base-mutations-per-variant",
        "1",
        "--enhancement-rounds-json",
        '{"5":1,"6":1}',
        "--enhancement-operations",
        "connect_points,foot_of_perpendicular,angle_bisector,parallel_through,perpendicular_through,point_on_segment_ratio",
        "--max-enhancements-per-base",
        "1",
        "--solver-profile",
        "full",
        "--format",
        "both",
        "--workers",
        str(workers),
        "--scene-payload",
        "metadata",
        "--annotation-level",
        "givens",
        "--relation-markers",
        "--split-group-mode",
        "topology",
        "--progress-every",
        "1",
        "--require-exact-count",
        "--seed",
        str(seed),
    ]
    log_path = output.parent / "generation.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a diverse 3D and complex 2D GeoSym review set.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--complex-2d-records", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_root = args.output.resolve()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.complex_2d_records <= 0:
        raise ValueError("--complex-2d-records must be positive")
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")
    (output_root / "geometry_3d").mkdir(parents=True)
    (output_root / "complex_2d").mkdir(parents=True)
    launch = {
        "script": str(Path(__file__).resolve()),
        "python": sys.executable,
        "output": str(output_root),
        "seed": args.seed,
        "workers": args.workers,
        "complex_2d_records": args.complex_2d_records,
    }
    (output_root / "launch.json").write_text(
        json.dumps(launch, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[geosym-launch] output={output_root}", flush=True)
    print("[geosym-launch] validating and exporting 32 diverse 3D records", flush=True)
    manifest_3d = _export_3d(output_root, seed=args.seed, workers=args.workers)
    print(json.dumps(manifest_3d, ensure_ascii=False, indent=2), flush=True)
    group_count = len(COMPLEX_2D_GROUPS)
    records_per_group = args.complex_2d_records // group_count
    if records_per_group <= 0:
        raise ValueError("--complex-2d-records must be at least the number of complex 2D groups")
    print(
        f"[geosym-launch] generating {records_per_group} complex 2D records per "
        f"group across {group_count} groups",
        flush=True,
    )
    group_manifest = []
    for index, (group_name, families) in enumerate(COMPLEX_2D_GROUPS):
        print(f"[geosym-launch] 2D group={group_name} families={','.join(families)}", flush=True)
        _run_complex_2d_group(
            output_root,
            group_name=group_name,
            families=families,
            records=records_per_group,
            seed=args.seed + 1 + index,
            workers=args.workers,
        )
        group_manifest.append({"name": group_name, "families": families, "records": records_per_group})
    (output_root / "complex_2d" / "groups.json").write_text(
        json.dumps(group_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[geosym-launch] complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()
