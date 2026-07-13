# GeoSym Synthesis Engine v0

This document describes the current synthesis pipeline under `geosym_world_geometry/synthesis`.

## Design Goal

The synthesis engine is intentionally procedural and auditable. It is not a large black-box generator. The pipeline is:

```text
TemplateFamily + parameter grid
  -> SyntheticSpec
  -> deterministic trace mutator
  -> TraceVerifier
  -> FactClosureSolver target check
  -> DuplicateFilter
  -> RenderScene with deterministic style policy
  -> SVG / OpenCV PNG
  -> image + QA JSONL record
```

## Template Construction Method

The engine does not rely on manually writing a large number of templates. Instead, it defines compact `TemplateFamily` objects:

- a family name;
- a parameter grid;
- a builder that maps parameters to a formal trace;
- diagnostic axes and question metadata.

The current built-in families are:

| family | expansion idea |
|---|---|
| `triangle_area_from_height` | vary base, height, skew, altitude, midpoint |
| `angled_line_intersection` | vary base and line angles, construct intersection point |
| `circle_sector` | vary radius and external tangent point |
| `parallel_cut_similarity` | vary triangle shape and cut ratio |
| `tetrahedron_section` | vary tetrahedron scale, section height, and target measure |

This gives a controlled route to many templates without hand-authoring many one-off traces.

## Mutator

`mutators.py` provides deterministic coordinate-level mutation:

- uniform scaling;
- translation in 2D/3D;
- radius scaling for circles.

Mutation is deliberately conservative so that formal relationships are preserved.

`SyntheticSampler` samples families in round-robin order. Small smoke exports therefore cover multiple construction families instead of exhausting the first family before moving to the next one.

## Difficulty Policy

`difficulty.py` assigns difficulty from:

- operation count;
- intersection operations;
- higher-level construction operations;
- 3D axes;
- similarity axes.

The output includes both a label and policy metadata.

## Degeneracy Rejection

`DegeneracyRejector` rejects traces that fail:

- static/executable trace verification;
- state construction;
- fact closure;
- target presence;
- target value availability.

This keeps generated records auditable before rendering.

## Duplicate Filtering

`DuplicateFilter` computes a stable hash over family, trace, and question. This is a first-pass deduplication layer. Future versions should add image hash and semantic target-value deduplication.

## Staged Generation Pipeline

The maintained large-run path mirrors the original GeoSym script flow:

| original stage | formal pipeline stage |
|---|---|
| `TemplateGenerator` writes base JSONL | `SyntheticSampler` writes `json/base/*.base.jsonl` |
| `RandomGeometryBuilder` expands each base geometry | `FormalTraceEnhancer` writes `json/enhanced/*.enhanced.jsonl` |
| drawer/shader produce images | `SyntheticRecordExporter` renders image paths from `RenderScene` |
| `GeometryCalculator` writes GT | exporter writes `*.gt.jsonl` from target facts and derivation |
| `QAGenerator` writes QA | exporter writes `*.qa.jsonl` with question, answer, image, family, difficulty |

The enhancement stage is config-driven. It expands each base spec with a rounds distribution such as `{"0": 1, "1": 3, "2": 1}` and uses verifier-filtered formal operations:

```text
coordinate_mutation
add_midpoint
connect_points
foot_of_perpendicular
angle_bisector
```

This keeps the original "base × enhancements -> final examples" generation shape while using the formal trace verifier and exact solver as the GT source.

For high-diversity runs, prefer more base specs and fewer enhanced variants per base. The `large-500` preset follows that policy: it expands the parameter grids, uses seven 2D families, caps each family at 72 records, and keeps only one enhanced image per base spec. This avoids generating dozens of near-identical images from one geometry skeleton. The 3D `tetrahedron_section` family remains available when requested explicitly, but it is excluded from the default large preset because it is slower and is less aligned with the original 2D GeoSym generation path.

## Image + QA + GT Record Export

`SyntheticRecordExporter` writes JSONL records with:

- formal state;
- facts;
- derivation;
- verification;
- synthetic metadata;
- render scene metadata;
- SVG and/or PNG paths;
- compact QA sidecar records;
- compact ground-truth sidecar records.

CLI:

```bash
python -m geosym_world_geometry.tools.synthesize_records \
  --output geosym_world_geometry/outputs/synthetic_v1/synthetic_v1.jsonl \
  --max-records 100 \
  --per-family 20 \
  --mutations-per-variant 2 \
  --format both \
  --seed 23
```

Large controlled run for 500 PNG images plus QA and GT:

```bash
PYTHONPATH=/data/Geosym127K python -m geosym_world_geometry.tools.synthesize_records \
  --preset large-500 \
  --seed 23
```

The `large-500` preset expands to:

```text
--output geosym_world_geometry/outputs/synthetic_500/synthetic_500.jsonl
--max-records 500
--families triangle_area_from_height,angled_line_intersection,circle_sector,parallel_cut_similarity,rectangle_area_diagonal,parallelogram_area,trapezoid_area
--per-family 72
--pipeline
--base-records 500
--base-mutations-per-variant 0
--enhancement-rounds-json '{"1": 1}'
--enhancement-operations add_midpoint,connect_points,foot_of_perpendicular,angle_bisector
--max-enhancements-per-base 1
--grid-json '<expanded built-in family grids>'
--format png
--require-exact-count
```

It writes:

| artifact | default path |
|---|---|
| full records | `geosym_world_geometry/outputs/synthetic_500/synthetic_500.jsonl` |
| base specs | `geosym_world_geometry/outputs/synthetic_500/json/base/synthetic_500.base.jsonl` |
| enhanced specs | `geosym_world_geometry/outputs/synthetic_500/json/enhanced/synthetic_500.enhanced.jsonl` |
| QA sidecar | `geosym_world_geometry/outputs/synthetic_500/synthetic_500.jsonl.qa.jsonl` |
| GT sidecar | `geosym_world_geometry/outputs/synthetic_500/synthetic_500.jsonl.gt.jsonl` |
| manifest | `geosym_world_geometry/outputs/synthetic_500/synthetic_500.jsonl.manifest.json` |
| images | `geosym_world_geometry/outputs/synthetic_500/images/*.png` |

You can restrict families:

```bash
python -m geosym_world_geometry.tools.synthesize_records \
  --families triangle_area_from_height,tetrahedron_section \
  --per-family 30 \
  --max-records 60
```

You can override parameter grids inline:

```bash
python -m geosym_world_geometry.tools.synthesize_records \
  --families triangle_area_from_height \
  --grid-json '{"triangle_area_from_height": {"base": [3, 4, 5], "height": [2, 3], "skew": [1, 2]}}'
```

For cheap configuration checks without writing images, use `--format none` with a small `--max-records`. Do not use that mode as evidence that a full image build succeeded.

## Intersection Schema

`intersections.py` introduces `IntersectionResult`:

- `kind`;
- `status`;
- `points`;
- `parameters`;
- `multiplicity`;
- `metadata`.

It covers:

- line-line;
- segment-segment;
- line-circle;
- circle-circle;
- stable point selection such as `upper`, `lower`, `left`, `right`, `first`, `second`, and `all`.

The current builder remains backward-compatible. Future work should route construction ops through this schema.

## Test Matrix

The current pytest matrix covers:

- structured intersections;
- synthetic sampler;
- image+QA exporter;
- deterministic render style policy.

Run:

```bash
python -m pytest geosym_world_geometry/tests -q
```

## Boundaries

This is not yet a large production generator. Missing pieces include:

- full global degeneracy policies;
- image-level deduplication;
- richer template-family grammar;
- LLM-assisted trace proposal and rejection loop;
- theorem-fact-driven render markers;
- large-scale distributed generation.
