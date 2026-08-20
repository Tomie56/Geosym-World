"""Generate synthetic image+QA+GT records from procedural template families."""

from __future__ import annotations

import argparse
import json
import sys
from geosym_world.datasets.layout import artifact_root
from geosym_world.synthesis import GeometryGenerationPipeline, PipelineRunConfig, SyntheticRecordExporter, SyntheticSampler, assign_split_labels, load_grid_overrides, select_template_families
from geosym_world.synthesis.pipeline import parse_operation_list, parse_rounds_distribution


DEFAULT_OUTPUT = str(artifact_root() / "generated" / "synthetic_v0" / "records.jsonl")


LARGE_500_GRID = {
    "triangle_area_from_height": {"base": list(range(3, 13)), "height": list(range(2, 9)), "skew": list(range(0, 6))},
    "triangle_partition_area": {"left_base": list(range(1, 8)), "right_base": list(range(1, 8)), "height": list(range(2, 10))},
    "triangle_partition_area_ratio": {"left_base": list(range(1, 8)), "right_base": list(range(1, 8)), "height": list(range(2, 10))},
    "angled_line_intersection": {
        "base": list(range(4, 14)),
        "left_angle": ["pi/8", "pi/6", "pi/5", "pi/4", "pi/3"],
        "right_angle": ["2*pi/3", "3*pi/4", "4*pi/5", "5*pi/6", "7*pi/8"],
    },
    "circle_sector": {"radius": list(range(2, 15)), "central_angle": ["pi/6", "pi/4", "pi/3", "pi/2", "2*pi/3", "3*pi/4", "pi"]},
    "sector_arc_length": {"radius": list(range(2, 15)), "central_angle": ["pi/6", "pi/4", "pi/3", "pi/2", "2*pi/3", "3*pi/4", "pi"]},
    "sector_perimeter": {"radius": list(range(2, 15)), "central_angle": ["pi/6", "pi/4", "pi/3", "pi/2", "2*pi/3", "3*pi/4", "pi"]},
    "circular_segment_area": {"radius": list(range(2, 15)), "central_angle": ["pi/6", "pi/4", "pi/3", "pi/2", "2*pi/3", "3*pi/4"]},
    "circular_segment_perimeter": {"radius": list(range(2, 15)), "central_angle": ["pi/6", "pi/4", "pi/3", "pi/2", "2*pi/3", "3*pi/4"]},
    "circular_segment_area_ratio": {"radius": list(range(2, 15)), "central_angle": ["pi/6", "pi/4", "pi/3", "pi/2", "2*pi/3", "3*pi/4"]},
    "equal_circle_lens_area": {"radius": list(range(3, 15)), "center_distance": list(range(1, 6))},
    "equal_circle_lens_perimeter": {"radius": list(range(3, 15)), "center_distance": list(range(1, 6))},
    "circle_lens_area": {
        "case": [[radius, radius - 1, radius + 1] for radius in range(4, 15)]
        + [[radius, radius, radius + 1] for radius in range(4, 15)]
    },
    "circle_lens_perimeter": {
        "case": [[radius, radius - 1, radius + 1] for radius in range(4, 15)]
        + [[radius, radius, radius + 1] for radius in range(4, 15)]
    },
    "parallel_cut_similarity": {"base": list(range(6, 18, 2)), "height": list(range(3, 11)), "ratio": ["1/4", "1/3", "1/2", "2/3", "3/4"]},
    "rectangle_area_diagonal": {"width": list(range(3, 15)), "height": list(range(2, 10))},
    "rectangular_frame_area": {"outer_width": list(range(8, 20)), "outer_height": list(range(6, 16)), "border": [1, 2]},
    "double_rectangular_hole_area": {"outer_width": list(range(12, 21)), "outer_height": list(range(7, 15)), "left_width": [2, 3, 4], "left_height": [2, 3, 4], "right_width": [2, 3, 4], "right_height": [2, 3, 4]},
    "double_rectangular_hole_perimeter": {"outer_width": list(range(12, 21)), "outer_height": list(range(7, 15)), "left_width": [2, 3, 4], "left_height": [2, 3, 4], "right_width": [2, 3, 4], "right_height": [2, 3, 4]},
    "parallelogram_area": {"base": list(range(4, 16)), "height": list(range(2, 10)), "skew": list(range(1, 7))},
    "trapezoid_area": {"bottom": list(range(6, 18)), "top": list(range(2, 8)), "height": list(range(2, 10)), "offset": list(range(1, 5))},
    "kite_area_diagonals": {"horizontal": list(range(6, 22, 2)), "upper": list(range(3, 10)), "lower": list(range(2, 9))},
    "rhombus_diagonal_theorem": {"horizontal": list(range(4, 20, 2)), "vertical": list(range(4, 20, 2))},
    "cyclic_quadrilateral": {"radius": list(range(2, 15)), "lower_angle": ["7*pi/6", "4*pi/3", "3*pi/2", "5*pi/3", "11*pi/6"]},
    "tangential_quadrilateral": {"bottom": list(range(7, 19)), "top": list(range(2, 7))},
    "concave_composite_area": {"width": list(range(6, 16)), "height": list(range(5, 13)), "notch_width": [2, 3, 4], "notch_height": [2, 3, 4]},
    "annulus_area": {"outer_radius": list(range(5, 20)), "inner_radius": list(range(1, 5))},
    "annulus_perimeter": {"outer_radius": list(range(5, 20)), "inner_radius": list(range(1, 5))},
    "tetrahedron_section": {"scale": list(range(1, 11)), "height": ["1/5", "1/4", "1/3", "1/2", "2/3"], "target": ["section_area", "section_perimeter"]},
}


LARGE_500_PRESET = {
    "max_records": 500,
    "families": "triangle_area_from_height,triangle_partition_area,triangle_partition_area_ratio,angled_line_intersection,circle_sector,sector_arc_length,sector_perimeter,circular_segment_area,circular_segment_perimeter,circular_segment_area_ratio,equal_circle_lens_area,equal_circle_lens_perimeter,circle_lens_area,circle_lens_perimeter,parallel_cut_similarity,similarity_dual_proof,rectangle_area_diagonal,rectangular_frame_area,double_rectangular_hole_area,double_rectangular_hole_perimeter,parallelogram_area,trapezoid_area,kite_area_diagonals,rhombus_diagonal_theorem,cyclic_quadrilateral,tangential_quadrilateral,concave_composite_area,annulus_area,annulus_perimeter",
    "per_family": 50,
    "format": "png",
    "output": str(artifact_root() / "generated" / "synthetic_500" / "records.jsonl"),
    "require_exact_count": True,
    "pipeline": True,
    "base_records": 500,
    "base_reserve_records": 200,
    "base_mutations_per_variant": 0,
    "enhancement_rounds_json": '{"1": 1, "2": 1, "3": 1}',
    "enhancement_operations": "add_midpoint,connect_points,foot_of_perpendicular,angle_bisector,parallel_through,perpendicular_through,point_on_segment_ratio",
    "max_enhancements_per_base": 3,
    "grid_json": json.dumps(LARGE_500_GRID, ensure_ascii=False),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-records", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--format", choices=["none", "svg", "png", "both"], default="both")
    parser.add_argument("--preset", choices=["large-500"], default=None, help="Apply a maintained generation preset before explicit command-line overrides.")
    parser.add_argument("--no-mutation", action="store_true")
    parser.add_argument("--families", default="", help="Comma-separated template family names. Empty means all built-in families.")
    parser.add_argument("--per-family", type=int, default=None, help="Maximum accepted records per selected family.")
    parser.add_argument("--mutations-per-variant", type=int, default=1)
    parser.add_argument("--grid-json", default=None, help="Inline JSON object overriding parameter grids.")
    parser.add_argument("--grid-json-file", default=None, help="Path to a JSON object overriding parameter grids.")
    parser.add_argument("--qa-output", default=None, help="Optional QA JSONL path. Defaults to OUTPUT.qa.jsonl.")
    parser.add_argument("--gt-output", default=None, help="Optional ground-truth JSONL path. Defaults to OUTPUT.gt.jsonl.")
    parser.add_argument("--require-exact-count", action="store_true", help="Fail if sampling accepts fewer than --max-records records.")
    parser.add_argument("--pipeline", action="store_true", help="Run staged base -> enhanced -> records/QA/GT generation.")
    parser.add_argument("--base-records", type=int, default=None, help="Number of base specs to generate before enhancement.")
    parser.add_argument("--base-reserve-records", type=int, default=None, help="Extra base specs used to backfill rejected enhanced candidates.")
    parser.add_argument("--base-mutations-per-variant", type=int, default=1, help="Coordinate mutations used while generating base specs in staged mode.")
    parser.add_argument("--enhancement-rounds-json", default=None, help='Rounds distribution for staged enhancement, e.g. \'{"0":1,"1":3,"2":1}\'.')
    parser.add_argument("--enhancement-operations", default=None, help="Comma-separated staged enhancement operations.")
    parser.add_argument("--max-enhancements-per-base", type=int, default=None, help="Cap accepted enhanced candidates from each base spec.")
    parser.add_argument("--base-output", default=None, help="Optional staged base JSONL path.")
    parser.add_argument("--enhanced-output", default=None, help="Optional staged enhanced JSONL path.")
    parser.add_argument("--input-enhanced", default=None, help="Resume export from an accepted enhanced-spec JSONL, skipping sampling and enhancement.")
    parser.add_argument("--min-difficulty", choices=["easy", "medium", "hard", "expert"], default=None)
    parser.add_argument("--max-difficulty", choices=["easy", "medium", "hard", "expert"], default=None)
    parser.add_argument("--solver-profile", choices=["minimal", "auto", "full"], default="auto")
    parser.add_argument("--workers", type=int, default=1, help="Parallel record/render workers; output order remains deterministic.")
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N accepted/exported records; 0 disables progress output.")
    parser.add_argument("--scene-payload", choices=["full", "metadata", "none"], default="metadata", help="Amount of renderer scene data stored in each JSON record.")
    parser.add_argument("--split-ratios-json", default=None, help='Train/dev/test ratios, e.g. \'{"train":0.8,"dev":0.1,"test":0.1}\'.')
    parser.add_argument("--split-group-mode", choices=["lineage", "topology"], default="topology", help="Keep either one lineage or all parameter variants of one task topology in the same split.")
    parser.add_argument("--list-families", action="store_true", help="Print built-in family names and exit without generating data.")
    provided_options = _provided_options(sys.argv[1:])
    args = parser.parse_args()
    args = _apply_preset(args, provided_options)

    if args.list_families:
        families = select_template_families()
        print(json.dumps([{"name": family.name, "description": family.description, "parameter_grid": family.parameter_grid} for family in families], ensure_ascii=False, indent=2))
        return

    if args.pipeline or args.input_enhanced:
        manifest = _run_pipeline(args)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    family_names = [item.strip() for item in args.families.split(",") if item.strip()]
    overrides = load_grid_overrides(args.grid_json_file, args.grid_json)
    families = select_template_families(family_names, grid_overrides=overrides)
    sampler = SyntheticSampler(
        families=families,
        seed=args.seed,
        enable_mutation=not args.no_mutation,
        mutations_per_variant=args.mutations_per_variant,
        per_family_limit=args.per_family,
        minimum_difficulty=args.min_difficulty,
        maximum_difficulty=args.max_difficulty,
        solver_profile=args.solver_profile,
    )
    specs = sampler.sample(max_records=args.max_records)
    split_ratios = _parse_split_ratios(args.split_ratios_json)
    specs = assign_split_labels(specs, seed=args.seed, ratios=split_ratios, group_mode=args.split_group_mode)
    require_count = args.max_records if args.require_exact_count else None
    run_config = {
        "max_records": args.max_records,
        "seed": args.seed,
        "format": args.format,
        "families": family_names or "all",
        "per_family": args.per_family,
        "mutation_enabled": not args.no_mutation,
        "mutations_per_variant": args.mutations_per_variant,
        "grid_json": args.grid_json,
        "grid_json_file": args.grid_json_file,
        "require_exact_count": args.require_exact_count,
        "preset": args.preset,
        "minimum_difficulty": args.min_difficulty,
        "maximum_difficulty": args.max_difficulty,
        "solver_profile": args.solver_profile,
        "workers": args.workers,
        "scene_payload": args.scene_payload,
        "split_ratios": split_ratios or {"train": 0.8, "dev": 0.1, "test": 0.1},
        "split_group_mode": args.split_group_mode,
    }
    manifest = SyntheticRecordExporter(solver_profile=args.solver_profile).write_jsonl(
        specs,
        args.output,
        image_format=args.format,
        style_seed=args.seed,
        qa_path=args.qa_output,
        gt_path=args.gt_output,
        require_count=require_count,
        run_config=run_config,
        workers=args.workers,
        scene_payload=args.scene_payload,
        progress_every=args.progress_every,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _run_pipeline(args: argparse.Namespace) -> dict:
    family_names = tuple(item.strip() for item in args.families.split(",") if item.strip())
    config = PipelineRunConfig(
        target_records=args.max_records,
        base_records=args.base_records,
        base_reserve_records=args.base_reserve_records,
        seed=args.seed,
        image_format=args.format,
        require_exact_count=args.require_exact_count,
        families=family_names,
        per_family_limit=args.per_family,
        base_mutations_per_variant=args.base_mutations_per_variant,
        enhancement_rounds=parse_rounds_distribution(args.enhancement_rounds_json),
        enhancement_operations=parse_operation_list(args.enhancement_operations),
        max_enhancements_per_base=args.max_enhancements_per_base,
        grid_json=args.grid_json,
        grid_json_file=args.grid_json_file,
        minimum_difficulty=args.min_difficulty,
        maximum_difficulty=args.max_difficulty,
        solver_profile=args.solver_profile,
        workers=args.workers,
        scene_payload=args.scene_payload,
        progress_every=args.progress_every,
        split_ratios=_parse_split_ratios(args.split_ratios_json),
        split_group_mode=args.split_group_mode,
    )
    return GeometryGenerationPipeline(config).run(
        args.output,
        input_enhanced_path=args.input_enhanced,
        base_path=args.base_output,
        enhanced_path=args.enhanced_output,
        qa_path=args.qa_output,
        gt_path=args.gt_output,
    )


def _provided_options(argv: list[str]) -> set[str]:
    option_to_dest = {
        "--output": "output",
        "--max-records": "max_records",
        "--format": "format",
        "--families": "families",
        "--per-family": "per_family",
        "--mutations-per-variant": "mutations_per_variant",
        "--require-exact-count": "require_exact_count",
        "--pipeline": "pipeline",
        "--base-records": "base_records",
        "--base-reserve-records": "base_reserve_records",
        "--base-mutations-per-variant": "base_mutations_per_variant",
        "--enhancement-rounds-json": "enhancement_rounds_json",
        "--enhancement-operations": "enhancement_operations",
        "--max-enhancements-per-base": "max_enhancements_per_base",
        "--input-enhanced": "input_enhanced",
        "--grid-json": "grid_json",
        "--grid-json-file": "grid_json_file",
        "--min-difficulty": "min_difficulty",
        "--max-difficulty": "max_difficulty",
        "--solver-profile": "solver_profile",
        "--workers": "workers",
        "--progress-every": "progress_every",
        "--scene-payload": "scene_payload",
        "--split-ratios-json": "split_ratios_json",
        "--split-group-mode": "split_group_mode",
    }
    provided: set[str] = set()
    for token in argv:
        option = token.split("=", 1)[0]
        if option in option_to_dest:
            provided.add(option_to_dest[option])
    return provided


def _apply_preset(args: argparse.Namespace, provided_options: set[str]) -> argparse.Namespace:
    if args.preset != "large-500":
        return args
    defaults = {
        "output": DEFAULT_OUTPUT,
        "max_records": 20,
        "format": "both",
        "families": "",
        "per_family": None,
        "mutations_per_variant": 1,
        "require_exact_count": False,
        "pipeline": False,
        "base_records": None,
        "base_reserve_records": None,
        "base_mutations_per_variant": 1,
        "enhancement_rounds_json": None,
        "enhancement_operations": None,
        "max_enhancements_per_base": None,
        "grid_json": None,
    }
    for key, value in LARGE_500_PRESET.items():
        if key not in provided_options and getattr(args, key) == defaults[key]:
            setattr(args, key, value)
    return args


def _parse_split_ratios(value: str | None) -> dict[str, float] | None:
    if value is None:
        return None
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("Split ratios must be a JSON object")
    return {str(key): float(weight) for key, weight in payload.items()}


if __name__ == "__main__":
    main()
