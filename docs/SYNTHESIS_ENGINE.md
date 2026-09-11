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
| `triangle_partition_area` | altitude partitions a triangle into two verified subregions and sums their areas |
| `triangle_partition_area_ratio` | target the exact ratio of two admitted altitude-partition subregion areas |
| `angled_line_intersection` | vary base and line angles, construct intersection point |
| `circle_sector` | vary public radius and exact central angle for sector area |
| `sector_arc_length` | derive arc length from the same public radius-angle contract |
| `sector_perimeter` | derive arc plus two radii as total sector boundary length |
| `circular_segment_area` | subtract the chord triangle from a sector and shade the arc-chord segment |
| `circular_segment_perimeter` | derive the hidden chord from radius-angle facts and add it to the admitted arc length |
| `circular_segment_area_ratio` | divide the admitted shaded segment area by its containing sector area |
| `equal_circle_lens_area` | construct two equal-circle intersections and derive the common lens from two sectors minus two triangles |
| `equal_circle_lens_perimeter` | derive the equal-circle lens boundary as two admitted minor arc lengths |
| `circle_lens_area` | derive an unequal-radius two-circle lens from two public radii and center distance |
| `circle_lens_perimeter` | derive an unequal-radius lens boundary from two public radii and center distance |
| `parallel_cut_similarity` | vary triangle shape and cut ratio |
| `similarity_dual_proof` | prove the same triangle similarity through independently validated SSS and AA paths |
| `rectangle_area_diagonal` | vary rectangle dimensions and diagonal |
| `rectangular_frame_area` | outer rectangle minus a contained inner rectangle using four rendered dimensions |
| `double_rectangular_hole_area` | subtract two disjoint rectangular holes using six rendered dimensions |
| `double_rectangular_hole_perimeter` | sum the outer and two inner boundary lengths using six rendered dimensions |
| `parallelogram_area` | vary base, height, and skew |
| `trapezoid_area` | vary both bases, height, and offset |
| `kite_area_diagonals` | perpendicular diagonals with an explicit intersection |
| `rhombus_diagonal_theorem` | four equal sides with a diagonal-area residual |
| `cyclic_quadrilateral` | concyclic determinant and cyclicity residual |
| `tangential_quadrilateral` | incircle tangency and Pitot residual |
| `concave_composite_area` | derive an L-shaped region as an outer rectangle minus one corner notch |
| `annulus_area` | derive `pi(R^2-r^2)` from two rendered radius segments and shade the annulus |
| `annulus_perimeter` | derive total inner-plus-outer boundary length `2*pi(R+r)` from the same public radii |
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

Difficulty schema `2.0` assigns an auditable feature vector from:

- constructive operation counts and weighted operation types;
- target mode and target type;
- semantic axes such as proof, multi-path, region boolean, and multi-step area;
- derivation rule count and maximum depth after fact closure;
- number and depth of independently validated solution paths.

Trace-only scoring is a cheap sampling estimate. Candidate admission and final export recompute the authoritative score from the derivation and solution paths. Every record stores the versioned feature breakdown, and the manifest reports score range/mean and schema versions. Use `--min-difficulty` and `--max-difficulty` to constrain admitted candidates.

## Degeneracy Rejection

`DegeneracyRejector` rejects traces that fail:

- static/executable trace verification;
- state construction;
- fact closure;
- target presence;
- target value availability;
- derivation integrity;
- requested solution-path completeness.

Each rejection keeps a compatible reason plus a stable diagnostic code and stage/component classification. Manifests aggregate these classifications without retaining unbounded failed payloads.

The semantic-axis feature group gives theorem, region, proof, and multi-path tasks explicit weight instead of classifying them from operation count alone.

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

For high-diversity runs, prefer more base specs and a bounded enhancement depth per base. The `large-500` preset expands the parameter grids, uses twenty-nine 2D families, caps the final output at 50 records per family, and rotates one-, two-, and three-round candidates across distinct base specs. It includes numeric, area-ratio, theorem-residual, and dual-path proof structures, including circular-segment/sector area ratio and SSS/AA similarity proof. It samples 200 additional family-balanced reserve bases so rejected enhancements can be backfilled without lowering the exact final count. The 3D `tetrahedron_section` family remains opt-in.

## Image + QA + GT Record Export

`SyntheticRecordExporter` writes JSONL records with:

- record schema version, public givens, and a task-complexity contract;
- formal state;
- facts;
- versioned derivation steps with integrity, checksum, and independent validation;
- versioned alternative solution paths with per-path provenance, checksum, and integrity;
- verification;
- synthetic metadata;
- render scene metadata;
- SVG and/or PNG paths;
- compact QA sidecar records;
- compact ground-truth sidecar records.
- strict image/question-only Parser input sidecar records;
- evaluator-only Parser references with observable GT and private supervision.

Records and sidecars are written incrementally instead of retained in memory. `--workers` enables ordered parallel build/render work. `--progress-every 25` prints flushed sampling, enhancement, and export progress to stderr; set it to `0` to disable progress. The manifest reports enhancement coverage, exact preflight skip reasons, task-graph dead-operation rejection, difficulty score range/mean and schema version, and classified candidate failures. `--solver-profile auto` avoids combinatorial topology closure when the requested target is already resolved; `full` retains exhaustive audit behavior. `--scene-payload metadata` is the recommended large-run setting, while `full` stores every render primitive and `none` stores no scene payload.

The CLI assigns deterministic topology-aware train/dev/test labels while preserving record order. Its topology key uses the template family, construction operation sequence, target type, and diagnostic axes, but excludes coordinates, values, object IDs, and answers. Parameter variants of one topology therefore remain in one split. Use `--split-group-mode lineage` for lineage-only grouping and `--split-ratios-json '{"train":0.8,"dev":0.1,"test":0.1}'` to override the default ratios. Resumed exports preserve existing labels and reject conflicts within the selected group. The manifest reports group mode/count and independently rejects any group that crosses splits. Public givens contain only rendered measurement values and relations with explicit marker semantics; formal coordinates and derived target facts stay outside the public view. The task block has its own checksum and validation report. The manifest additionally reports split counts, target types, construction/derivation depths, derivation rules, invalid derivations, and distinct structure signatures.

The triangle, rectangle, parallelogram, trapezoid, and kite area families bind their targets to explicit semantic rules over rendered measurements. Their target derivations consume public base/height or diagonal fact IDs, while the exact shoelace area is retained only as an admission check. The solver emits an `area_formula_residual`; a nonzero residual prevents the semantic area fact from being admitted and the candidate/export boundary rejects the sample. This prevents benchmark answers from being justified only by hidden coordinates.

`triangle_partition_area` adds a formal multi-step region task: altitude `CH` splits the parent triangle into `ACH` and `BCH`; each child area is derived from a rendered base and height; the strict `partition` verifier proves non-overlap and complete parent coverage; and the target area is the sum of the two admitted child-area facts. The complete record keeps the partition residual and verifier report, while the derivation supervision exposes only the public-measurement calculation chain.

`triangle_partition_area_ratio` reuses the same verified partition but targets `[ACH]/[BCH]`. The ratio fact depends on the two semantic region-area facts and is marked dimensionless. The two compared regions remain separately visible with different fills, while the composite parent rendering is suppressed.

`circular_segment_area_ratio` combines the public radius-angle sector chain, the admitted sector-minus-triangle circular-segment chain, and the canonical region-area ratio target. `similarity_dual_proof` exposes a proof target with two distinct root facts: SSS depends on six public side-length facts, while AA depends on independently derived angle facts. Export fails unless both requested paths are complete and independently valid.

`rectangular_frame_area` covers a first-class shaded difference. The outer and inner rectangle areas each derive from rendered width/height facts, the verifier requires the inner rectangle to materialize as one contained hole, and the target subtracts the two semantic region-area facts. `area_difference_residual` independently checks the result against exact boolean geometry before admission.

`double_rectangular_hole_area` extends this supervision to a first-class multi-hole `Region`. Six rendered lengths derive the outer rectangle and both hole rectangle areas. Auxiliary hole vertices remain formal objects but suppress point names and markers because the question references dimensions rather than vertex IDs; vertical dimensions are placed on the hole sides away from the outer-height label. The target depends on the three admitted region-area facts through `region_hole_difference_area`; `area_hole_difference_residual` checks the result against one materialized outer ring with exactly two disjoint holes. SVG and PNG consume the same rings, and PNG regression verifies both holes remain transparent.

`double_rectangular_hole_perimeter` reuses the same formal construction with a perimeter target. Each rectangle perimeter derives as `2(width + height)` from two rendered lengths and is admitted against its exact closed boundary. The final `region_hole_boundary_sum` adds the outer and both inner boundary perimeters; `perimeter_hole_sum_residual` independently verifies the result against the materialized two-hole region. The area and perimeter families therefore share geometry while exposing distinct target semantics and derivations.

`equal_circle_lens_area` is a first-class curved intersection task. Two exact circle-circle intersection points define one minor boundary arc on each equal-radius circle. The public radius and center distance derive the half chord and central angle, then two sector areas and two isosceles-triangle areas; `lens_equal_radius_area` subtracts the triangle sum from the sector sum. A coordinate-backed closed form independently emits `lens_coordinate_area`, and `lens_area_residual` must be zero before admission. SVG and PNG close the same two-arc path, while the construction arcs may stay hidden as standalone primitives.

`circle_lens_area` generalizes that contract to unequal radii while retaining the explicit two-intersection, two-minor-arc subset. Its built-in grid stores validated `(r1, r2, d)` cases rather than an unconstrained Cartesian product: in addition to two-circle intersection, both center projections must be nonnegative so each boundary arc is at most a semicircle. The public inputs are `r1`, `r2`, and center distance `d`; the solver derives the two center projections, common half-chord, two central angles, two sector areas, and two triangle areas before admitting `lens_two_radius_area`. The independent coordinate expression and `lens_two_radius_area_check` must agree. Tangency, containment, and major-arc boundaries remain rejected rather than silently converted into a different topology.

The lens perimeter families reuse the same boundary contract. `lens_equal_radius_perimeter` and `lens_two_radius_perimeter` derive the sum of the two minor arc lengths from the same public radius and center-distance facts, then compare it against the coordinate `lens_coordinate_perimeter`. Equivalent `atan2` and `acos` forms are canonicalized only when SymPy certifies the residual as zero at high precision; arbitrary numeric tolerance is not used for admission.

`concave_composite_area` uses the restricted `boundary_notch` difference mode. Four rendered side lengths derive the outer rectangle and removed corner rectangle areas; the target then subtracts those admitted semantic areas. The hidden completion point belongs to formal construction state but is excluded from point markers, labels, visual bounds, and implicit-intersection suppression. The verifier requires one exact corner cut, the renderer consumes the resulting six-vertex L boundary, and PNG regression checks that the removed corner remains unfilled.

SVG and PNG are independent backends over the same `RenderScene`; PNG geometry is drawn directly with OpenCV and its Unicode labels are drawn with Pillow rather than converted from SVG. Both backends use the same deterministic pixel-space label placement, so point names and measurements share collision decisions. Generated internal point IDs are mapped to unique sequential display labels without changing formal IDs, and exact measurements use Unicode radicals and compact fractions. Construction extensions render behind finite solid edges. The adapter also emits altitude segments, oriented perpendicular-foot squares, angle arcs for explicit angle relations, and exact-coordinate right-angle markers for polygon vertices.

The compact visual-regression fixtures store no generated image. Their versioned signatures cover the formal scene fingerprint, exact SVG output, direct PNG RGB pixels, PNG foreground occupancy, and scene counts. A small representative matrix fixes canonical sector, circular-segment, equal-circle lens area, unequal-radius lens area/perimeter, two-hole, and annulus scenes, catching curved-boundary, two-arc closure, and transparent-hole regressions while keeping generated SVG/PNG files out of Git.

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
--families triangle_area_from_height,triangle_partition_area,triangle_partition_area_ratio,angled_line_intersection,circle_sector,sector_arc_length,sector_perimeter,circular_segment_area,circular_segment_perimeter,circular_segment_area_ratio,equal_circle_lens_area,equal_circle_lens_perimeter,circle_lens_area,circle_lens_perimeter,parallel_cut_similarity,similarity_dual_proof,rectangle_area_diagonal,rectangular_frame_area,double_rectangular_hole_area,double_rectangular_hole_perimeter,parallelogram_area,trapezoid_area,kite_area_diagonals,rhombus_diagonal_theorem,cyclic_quadrilateral,tangential_quadrilateral,concave_composite_area,annulus_area,annulus_perimeter
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
| public Parser input | `artifacts/generated/synthetic_500/records.jsonl.parser-input.jsonl` |
| private Parser reference | `artifacts/generated/synthetic_500/records.jsonl.parser-reference.jsonl` |
| typed ParseGraph reference | `artifacts/generated/synthetic_500/records.jsonl.typed-parser-reference.jsonl` |
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
- line/ray/segment-arc;
- circle-arc;
- arc-arc, including explicit same-circle overlap;
- stable point selection such as `upper`, `lower`, `left`, `right`, `first`, `second`, `nearest`, `farthest`, and `all`.

Arc endpoints and optional central angle are validated against the supporting circle before construction. Curve intersections filter supporting-circle solutions by the exact clockwise/counterclockwise sweep. The verifier recomputes those domains, while solver closure emits `point_on_arc` and independently checked `curve_intersection` evidence only for admitted points.

Builder operations, solver facts, and verifier audits share this topology contract. Approximate collinearity/tangency diagnostics are reported separately and do not alter exact statuses.

## Test Matrix

The current pytest matrix covers:

- structured intersections;
- synthetic sampler;
- image+QA exporter;
- deterministic render style policy.

Run:

```bash
/data/miniconda3/envs/geosym/bin/python -m pytest -q
```

## Boundaries

Current boundaries include:

- approximate diagnostics beyond collinearity and line-circle tangency;
- generic curved-region boolean operations;
- image-level deduplication;
- learned or grammar-driven template proposals beyond the controlled built-ins;
- LLM-assisted trace proposal and rejection loop;
- theorem-fact-driven render markers;
- distributed generation beyond one process with ordered worker threads.
