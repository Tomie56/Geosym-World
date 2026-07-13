"""Generate synthetic image+QA+GT records from procedural template families."""

from __future__ import annotations

import argparse
import json
import sys

from geosym_world_geometry.synthesis import GeometryGenerationPipeline, PipelineRunConfig, SyntheticRecordExporter, SyntheticSampler, load_grid_overrides, select_template_families
from geosym_world_geometry.synthesis.pipeline import parse_operation_list, parse_rounds_distribution


LARGE_500_GRID = {
    "triangle_area_from_height": {"base": list(range(3, 13)), "height": list(range(2, 9)), "skew": list(range(0, 6))},
    "angled_line_intersection": {
        "base": list(range(4, 14)),
        "left_angle": ["pi/8", "pi/6", "pi/5", "pi/4", "pi/3"],
        "right_angle": ["2*pi/3", "3*pi/4", "4*pi/5", "5*pi/6", "7*pi/8"],
    },
    "circle_sector": {"radius": list(range(2, 15)), "external_x": list(range(8, 28, 2))},
    "parallel_cut_similarity": {"base": list(range(6, 18, 2)), "height": list(range(3, 11)), "ratio": ["1/4", "1/3", "1/2", "2/3", "3/4"]},
    "rectangle_area_diagonal": {"width": list(range(3, 15)), "height": list(range(2, 10))},
    "parallelogram_area": {"base": list(range(4, 16)), "height": list(range(2, 10)), "skew": list(range(1, 7))},
    "trapezoid_area": {"bottom": list(range(6, 18)), "top": list(range(2, 8)), "height": list(range(2, 10)), "offset": list(range(1, 5))},
    "tetrahedron_section": {"scale": list(range(1, 11)), "height": ["1/5", "1/4", "1/3", "1/2", "2/3"], "target": ["section_area", "section_perimeter"]},
}


LARGE_500_PRESET = {
    "max_records": 500,
    "families": "triangle_area_from_height,angled_line_intersection,circle_sector,parallel_cut_similarity,rectangle_area_diagonal,parallelogram_area,trapezoid_area",
    "per_family": 72,
    "format": "png",
    "output": "geosym_world_geometry/outputs/synthetic_500/synthetic_500.jsonl",
    "require_exact_count": True,
    "pipeline": True,
    "base_records": 500,
    "base_mutations_per_variant": 0,
    "enhancement_rounds_json": '{"1": 1}',
    "enhancement_operations": "add_midpoint,connect_points,foot_of_perpendicular,angle_bisector",
    "max_enhancements_per_base": 1,
    "grid_json": json.dumps(LARGE_500_GRID, ensure_ascii=False),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="geosym_world_geometry/outputs/synthetic_v0/synthetic_v0.jsonl")
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
    parser.add_argument("--base-mutations-per-variant", type=int, default=1, help="Coordinate mutations used while generating base specs in staged mode.")
    parser.add_argument("--enhancement-rounds-json", default=None, help='Rounds distribution for staged enhancement, e.g. \'{"0":1,"1":3,"2":1}\'.')
    parser.add_argument("--enhancement-operations", default=None, help="Comma-separated staged enhancement operations.")
    parser.add_argument("--max-enhancements-per-base", type=int, default=None, help="Cap accepted enhanced candidates from each base spec.")
    parser.add_argument("--base-output", default=None, help="Optional staged base JSONL path.")
    parser.add_argument("--enhanced-output", default=None, help="Optional staged enhanced JSONL path.")
    provided_options = _provided_options(sys.argv[1:])
    args = parser.parse_args()
    args = _apply_preset(args, provided_options)

    if args.pipeline:
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
    )
    specs = sampler.sample(max_records=args.max_records)
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
    }
    manifest = SyntheticRecordExporter().write_jsonl(
        specs,
        args.output,
        image_format=args.format,
        style_seed=args.seed,
        qa_path=args.qa_output,
        gt_path=args.gt_output,
        require_count=require_count,
        run_config=run_config,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _run_pipeline(args: argparse.Namespace) -> dict:
    family_names = tuple(item.strip() for item in args.families.split(",") if item.strip())
    config = PipelineRunConfig(
        target_records=args.max_records,
        base_records=args.base_records,
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
    )
    return GeometryGenerationPipeline(config).run(
        args.output,
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
        "--base-mutations-per-variant": "base_mutations_per_variant",
        "--enhancement-rounds-json": "enhancement_rounds_json",
        "--enhancement-operations": "enhancement_operations",
        "--max-enhancements-per-base": "max_enhancements_per_base",
        "--grid-json": "grid_json",
        "--grid-json-file": "grid_json_file",
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
        "output": "geosym_world_geometry/outputs/synthetic_v0/synthetic_v0.jsonl",
        "max_records": 20,
        "format": "both",
        "families": "",
        "per_family": None,
        "mutations_per_variant": 1,
        "require_exact_count": False,
        "pipeline": False,
        "base_records": None,
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


if __name__ == "__main__":
    main()
