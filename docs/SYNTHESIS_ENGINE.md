# GeoSym Synthesis Engine

This document describes the current synthesis pipeline under `geosym_world/synthesis`.

## Design Goal

The synthesis engine is intentionally procedural and auditable. It is not a large black-box generator. The pipeline is:

```text
TemplateFamily + parameter grid
  -> SyntheticSpec
  -> deterministic trace mutator
  -> static TraceVerifier
  -> one state construction
  -> target-directed FactClosureSolver check
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
| `rectangle_area_diagonal` | vary rectangle dimensions and diagonal |
| `parallelogram_area` | vary base, height, and skew |
| `trapezoid_area` | vary both bases, height, and offset |
| `kite_area_diagonals` | perpendicular diagonals with an explicit intersection |
| `rhombus_diagonal_theorem` | four equal sides with a diagonal-area residual |
| `cyclic_quadrilateral` | concyclic determinant and cyclicity residual |
| `tangential_quadrilateral` | incircle tangency and Pitot residual |
| `concave_composite_area` | subtract a rectangular notch from a concave polygon |
| `annulus_area` | subtract concentric-circle areas and shade the annulus |
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
- similarity axes;
- composite and multi-step area axes;
- circle-difference axes.

The output includes `easy`, `medium`, `hard`, or `expert` plus policy metadata. Use `--min-difficulty` and `--max-difficulty` to constrain sampling.

## Degeneracy Rejection

`DegeneracyRejector` rejects traces that fail:

- static/executable trace verification;
- state construction;
- fact closure;
- target presence;
- target value availability.

This keeps generated records auditable before rendering.

Difficulty scoring combines trace length, construction complexity, and theorem axes. In particular, rhombus diagonal, cyclic-quadrilateral, and tangential-quadrilateral records receive theorem-specific weight instead of being classified from operation count alone.

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
parallel_through
perpendicular_through
point_on_segment_ratio
```

This keeps the original "base × enhancements -> final examples" generation shape while using the formal trace verifier and exact solver as the GT source.

For high-diversity runs, prefer more base specs and a bounded enhancement depth per base. The `large-500` preset expands the parameter grids, uses thirteen 2D families, caps the final output at 50 records per family, and rotates one-, two-, and three-round candidates across distinct base specs. It includes kite, concave composite, sector, and annulus area records, all of which render the target region as a shaded area. It also samples 200 additional, family-balanced reserve bases so verifier- or degeneracy-rejected enhancements can be backfilled without lowering the exact final count. Coincident point IDs are rejected even when the duplicate comes from a base template rather than a later enhancement. The 3D `tetrahedron_section` family remains available when requested explicitly, but it is excluded from the default large preset because it is slower and is less aligned with the original 2D GeoSym generation path.

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

Records and sidecars are written incrementally instead of retained in memory. `--workers` enables ordered parallel build/render work. `--progress-every 25` prints flushed sampling, enhancement, and export progress to stderr; set it to `0` to disable progress. The manifest reports enhancement-round and operation counts so complex-iteration coverage is auditable. `--solver-profile auto` avoids combinatorial topology closure when the requested target is already resolved; `full` retains exhaustive audit behavior. `--scene-payload metadata` is the recommended large-run setting, while `full` stores every render primitive and `none` stores no scene payload.

SVG and PNG are independent backends over the same `RenderScene`; PNG geometry is drawn directly with OpenCV and its Unicode labels are drawn with Pillow rather than converted from SVG. Both backends use the same deterministic pixel-space label placement, so point names and measurements share collision decisions. Generated internal point IDs are mapped to unique sequential display labels without changing formal IDs, and exact measurements use Unicode radicals and compact fractions. Construction extensions render behind finite solid edges. The adapter also emits altitude segments, oriented perpendicular-foot squares, angle arcs for explicit angle relations, and exact-coordinate right-angle markers for polygon vertices.

CLI:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.synthesize_records \
  --output artifacts/generated/review_hard/records.jsonl \
  --max-records 8 \
  --min-difficulty hard \
  --per-family 2 \
  --no-mutation \
  --format png \
  --workers 4 \
  --scene-payload metadata \
  --seed 23
```

List families and parameter names without generating records:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.synthesize_records --list-families
```

Large controlled run for 500 PNG images plus QA and GT:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.synthesize_records \
  --preset large-500 \
  --seed 23
```

The `large-500` preset expands to:

```text
--output artifacts/generated/synthetic_500/records.jsonl
--max-records 500
--families triangle_area_from_height,angled_line_intersection,circle_sector,parallel_cut_similarity,rectangle_area_diagonal,parallelogram_area,trapezoid_area,kite_area_diagonals,rhombus_diagonal_theorem,cyclic_quadrilateral,tangential_quadrilateral,concave_composite_area,annulus_area
--per-family 50
--pipeline
--base-records 500
--base-reserve-records 200
--base-mutations-per-variant 0
--enhancement-rounds-json '{"1": 1, "2": 1, "3": 1}'
--enhancement-operations add_midpoint,connect_points,foot_of_perpendicular,angle_bisector,parallel_through,perpendicular_through,point_on_segment_ratio
--max-enhancements-per-base 3
--grid-json '<expanded built-in family grids>'
--format png
--progress-every 25
--require-exact-count
```

It writes:

| artifact | default path |
|---|---|
| full records | `artifacts/generated/synthetic_500/records.jsonl` |
| base specs | `artifacts/generated/synthetic_500/json/base/records.base.jsonl` |
| enhanced specs | `artifacts/generated/synthetic_500/json/enhanced/records.enhanced.jsonl` |
| QA sidecar | `artifacts/generated/synthetic_500/records.jsonl.qa.jsonl` |
| GT sidecar | `artifacts/generated/synthetic_500/records.jsonl.gt.jsonl` |
| manifest | `artifacts/generated/synthetic_500/records.jsonl.manifest.json` |
| images | `artifacts/generated/synthetic_500/images/*.png` |

To re-export an accepted enhanced-spec file with a different image format, resume directly from that JSONL instead of repeating sampling and enhancement:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.synthesize_records \
  --input-enhanced artifacts/generated/synthetic_500/json/enhanced/records.enhanced.jsonl \
  --output artifacts/generated/synthetic_500_both/records.jsonl \
  --max-records 500 --format both --workers 4 \
  --seed 23 --progress-every 25 --require-exact-count
```

Resume mode still executes trace/state verification, solving, rendering, QA/GT export, and manifest generation. It only skips base sampling and enhancement selection.

You can restrict families:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.synthesize_records \
  --families triangle_area_from_height,tetrahedron_section \
  --per-family 30 \
  --max-records 60
```

You can override parameter grids inline:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.synthesize_records \
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
python -m pytest geosym_world/tests -q
```

## Boundaries

Current boundaries include:

- full global degeneracy policies;
- image-level deduplication;
- learned or grammar-driven template proposals beyond the controlled built-ins;
- LLM-assisted trace proposal and rejection loop;
- theorem-fact-driven render markers;
- distributed generation beyond one process with ordered worker threads.
