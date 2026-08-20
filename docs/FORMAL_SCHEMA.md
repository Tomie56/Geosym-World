# GeoSym Formal Geometry Schema

This document is the working schema for `geosym_world`. It describes the executable construction trace, formal state objects, derived fact families, and the intended boundaries of the current prototype. Contract payloads use canonical JSON serialization for stable checksums. Parser candidates remain hypotheses until the admission verifier accepts an executable transition.

## Canonical Checksums and Admission

`canonical_json`, `stable_checksum`, `state_checksum`, and `record_checksum` are the stable identity helpers for serialized contract values. Mapping insertion order does not affect their result, and SymPy expressions are normalized before hashing. `StateTransition` records before/after checksums, changed objects, event ID, status, and provenance.

The supported state flow is:

```text
observation -> proposed hypothesis -> grounding -> trace/state verification
            -> admitted fact/state transition, or rejected/abstained
```

`HypothesisAdmission` never mutates the supplied initial state. An admitted report contains a candidate state and transition; rejected or abstained reports contain an unchanged transition and no candidate state. Low-confidence hypotheses can be explicitly abstained with `confidence_threshold`.

## Construction Trace

Every construction operation is a JSON object with:

```json
{"op": "operation_name", "id": "optional_output_id", "...": "operation parameters"}
```

Exported records preserve the original replayable `construction_trace` and additionally expose `construction_steps`. Each step contains the normalized operation, typed input references, a `StateTransition`, before/after state checksums, added/removed objects and relations, target changes, trace lengths, and provenance. The final step checksum must equal `formal_state_checksum`; this is supervision metadata and does not alter the canonical state contract.

The trace is replayed by `ConstructionBuilder` into a `FormalGeometryState`. The verifier performs lightweight static reference checks, then executes the trace.

## Core 2D/Generic Operations

| op | output | purpose |
|---|---|---|
| `add_point` | point | Add a 2D or 3D point from coordinates. |
| `add_segment` | segment | Add a segment from two points. |
| `add_line` | line | Add a line from two points or point + direction. |
| `add_ray` | ray | Add a ray from start + through point or start + direction. |
| `add_circle` | circle | Add a circle from center + radius. |
| `add_arc` | arc | Add a circle arc and an `arc` construction relation. |
| `add_polygon` | polygon | Add a polygon from ordered vertices. |
| `add_region` | region | Add a generic region, optionally with `holes`. |
| `add_composite_region` | composite region | Combine region components with `sum`, `difference`, `union`, or `intersection` semantics. |
| `mark_polygon_region` | region | Mark a polygon boundary as a region. |
| `mark_sector_region` | region | Mark a circular sector region. |
| `set_target` | target | Select the formal target fact for export. |

## Derived 2D Construction Operations

| op | output | purpose |
|---|---|---|
| `midpoint` | point | Construct midpoint of two points. |
| `intersection_line_line` | point | Construct 2D line-line intersection. |
| `intersection_line_circle` | point(s) | Construct 2D line-circle intersections. |
| `intersection_circle_circle` | point(s) | Construct 2D circle-circle intersections. |
| `intersection_line_arc` | point(s) | Intersect a line/ray/segment with a directed 2D arc domain. |
| `intersection_circle_arc` | point(s) | Intersect a circle with a directed 2D arc domain. |
| `intersection_arc_arc` | point(s) | Intersect two directed arc domains, preserving same-circle overlap. |
| `tangent_points_from_point` | point(s) | Construct tangent points from an external point to a circle. |
| `add_chord` | segment | Add a segment and a `chord` relation. |
| `parallel_through` | line | Construct line through point parallel to a line-like object. |
| `perpendicular_through` | line | Construct 2D perpendicular line through a point. |
| `line_through_angle` | line | Construct a line by rotating a base direction. |
| `rotate_ray` | ray | Rotate an existing ray. |
| `angle_bisector` | line | Construct internal or external angle bisector. |
| `foot_of_perpendicular` | point | Project a 2D point to a 2D line. |
| `point_on_ray_distance` | point | Place a point along a ray or direction at a given distance. |
| `point_on_segment_ratio` | point | Place a point at a segment ratio. |

## 3D Interface Operations

| op | output | purpose |
|---|---|---|
| `add_point3d` | point | Add a 3D point. |
| `add_plane` | plane | Add a plane from three points or point + normal. |
| `add_polyhedron` | polyhedron | Add a polyhedron from vertices and faces. |
| `project_point_to_plane` | point | Orthogonally project a 3D point to a plane. |
| `project_line_to_plane` | line | Orthogonally project a 3D line-like object to a plane. |
| `dihedral_angle` | relation | Declare a dihedral angle from planes or polyhedron face indices. |
| `section_polyhedron_with_plane` | relation, optional polygon | Declare or auto-generate a plane section of a polyhedron. |

`section_polyhedron_with_plane` has two modes:

- Explicit boundary mode: pass `boundary`.
- Automatic mode: omit `boundary`; the builder intersects polyhedron edges with the plane, creates section points, sorts them on the section plane, and materializes `{section_id}_polygon` when at least three points are found.

## Formal State Objects

| object | key fields |
|---|---|
| point | `id`, `coords`, `dim`, `metadata` |
| line | `id`, `point`, `direction`, `source_points`, `metadata` |
| segment | `id`, `start`, `end`, `metadata` |
| ray | `id`, `start`, `direction`, `through`, `metadata` |
| circle | `id`, `center`, `radius`, `metadata` |
| arc | `id`, `circle`, `start`, `end`, `central_angle`, `orientation`, `metadata` |
| polygon | `id`, `vertices`, `metadata` |
| region | `id`, `boundary`, `kind`, `metadata` |
| composite region | `id`, `operation`, `components`, `metadata` |
| plane | `id`, `point`, `normal`, `source_points`, `metadata` |
| polyhedron | `id`, `vertices`, `faces`, `metadata` |
| target | `mode`, `type`, fact-reference `object`, optional `expectation`, `metadata` |

## Fact Schema

Each fact has:

| field | meaning |
|---|---|
| `id` | Stable content-derived fact id. |
| `type` | Fact type, such as `length`, `area`, `space_angle`. |
| `object` | Object or relation identifier. |
| `value` | Exact symbolic value when available. |
| `numeric_value` | Float approximation when parseable. |
| `source` | Evidence origin such as `constructed`, `admitted`, or `derived`. |
| `rule` | Rule or formula that produced the fact. |
| `dependencies` | Input object/fact references. |
| `metadata` | Structured auxiliary details. |

## Core Metric Facts

| family | fact types |
|---|---|
| Object existence | `exists` |
| Point/segment metrics | `distance`, `length` |
| Circle metrics | `radius`, `circumference`, `area` |
| Arc/sector metrics | `arc_length`, `sector_area` |
| Polygon/region metrics | `perimeter`, `area` |
| Polyhedron metrics | `surface_area`, `volume` |
| Target lookup | `target`, `target_unresolved` |

`Region` boundaries may reference points, polygons, or an unordered set of segments forming a closed loop. `holes` records explicit inner regions. Every materialized hole must be a simple polygon strictly inside the outer boundary; holes may not cross, touch, overlap, or contain one another. The shared `resolve_region_geometry_2d` contract returns `polygon`, `polygon_with_holes`, `parametric`, `area_only`, `empty`, or `unsupported`, and its result is recorded as a `region_boundary` fact with source IDs and provenance. Unsupported boundary materialization is reported explicitly rather than guessed.

A semantic multi-hole area uses `Region.metadata.area_formula.rule="region_hole_difference"`, an existing `outer_region`, and an ordered `holes` list exactly matching `Region.holes`. State mutation rejects missing, repeated, reordered, or self-referential components. Deserialization loads regions in dependency order, so canonical mapping order does not weaken this reference contract. Solver closure subtracts every hole's preferred admitted area from the outer region's preferred admitted area, records `area_hole_difference_residual` against the exact materialized geometry, and admits `region_hole_difference_area` only when the residual is zero. Incremental closure indexes both the outer region and every hole.

Semantic perimeter uses a separate `perimeter_formula` contract. `Polygon.metadata.perimeter_formula.rule="rectangle_side_perimeter"` references two distinct adjacent polygon-edge segments and is accepted only when the exact four-point geometry is a rectangle; it derives `2(width + height)` and records `perimeter_formula_residual` against `closed_polyline_perimeter`. A holed region uses `Region.metadata.perimeter_formula.rule="region_hole_boundary_sum"` with the same ordered `outer_region`/`holes` reference discipline as semantic area. Its perimeter is the outer boundary perimeter plus every inner boundary perimeter, not an area-style subtraction. `perimeter_hole_sum_residual` checks the public-measurement result against the materialized rings before `region_hole_boundary_sum` is admitted.

An `annulus_region` is a typed curved region backed by distinct concentric `inner_circle` and `outer_circle` objects, with the common center as its boundary reference and a strictly larger outer radius. Public-measurement derivation uses `area_formula.rule="annulus_radius_area"` and `perimeter_formula.rule="annulus_radius_perimeter"`; both formulas reference distinct `outer_radius_segment` and `inner_radius_segment` IDs. The verifier requires each segment to run from the common center to a point exactly on its corresponding circle. Solver closure derives `pi(R^2-r^2)` and the total two-boundary length `2*pi(R+r)`, records `annulus_area_residual` and `annulus_perimeter_residual` against the circle-backed coordinate facts, and admits the semantic facts only at exact zero residual. Formula segments and circles are included in construction and incremental dependency indexes.

A `sector_region` currently represents a deterministic minor sector or semicircle with `0 < theta <= pi`. Its boundary must be the circle center followed by two distinct endpoints on the circle, and it must reference one existing radius segment. Reserved circle, endpoint, angle, orientation, and formula fields cannot be overridden through free metadata. `show_central_angle=true` produces a visible angle marker, an exact math label, and a public `central_angle` quantity in radians. Semantic `area`, `arc_length`, and total `perimeter` facts use `sector_radius_angle_area`, `sector_radius_angle_arc_length`, and `sector_radius_angle_perimeter`; each depends only on the public radius length and angle fact. Coordinate-backed sector facts independently recompute the endpoint angle, while `central_angle_residual`, `sector_area_residual`, `sector_arc_length_residual`, and `sector_perimeter_residual` gate admission. Major sectors and full-circle sectors remain outside this strict contract.

A `circular_segment_region` is bounded by one typed arc and its matching chord, and references the corresponding sector plus triangle formed by the center and chord endpoints. Polygon `area_formula` metadata is validated at the state boundary and included in construction/incremental dependencies; `chord_triangle_radius_angle` derives the triangle area `r^2*sin(theta)/2` from the same public sector radius and angle. `circular_segment_difference` then subtracts that admitted triangle area from the admitted sector area. Its independent `perimeter_formula.rule="circular_segment_arc_chord_perimeter"` references the same sector and chord. The chord is not exposed as a public measurement: `chord_radius_angle_length` derives `2*r*sin(theta/2)`, verifies it against the exact endpoint distance, and the final perimeter adds that fact to `sector_radius_angle_arc_length`. Coordinate-backed area/perimeter facts plus `circular_segment_area_residual`, `chord_length_residual`, and `circular_segment_perimeter_residual` gate both targets. SVG and PNG render the real arc-plus-chord boundary, while the sector and triangle supervision objects may remain hidden.

A `lens_region` represents the deterministic two-intersection subset of circle overlap. It must reference two distinct circles with distinct centers, their two exact intersection points, and exactly one matching minor arc from each circle. Its ordered boundary must equal those two arc references. `area_formula.rule="lens_equal_radius_area"` requires a public segment equal to the common radius and a public segment joining the two centers. `area_formula.rule="lens_two_radius_area"` instead requires two ordered public radius segments, one from each center, plus the center-distance segment. The semantic chain for the latter derives the two center projections, common half chord, two central angles, two sectors, and two isosceles triangles from those measurements; the final area is admitted only when it matches the independently evaluated coordinate closed form through `lens_two_radius_area_check`. Tangency, containment, coincident circles, major-arc boundaries, and generic curved boolean regions remain outside this contract and fail explicitly.

`perimeter_formula.rule` may be `lens_equal_radius_perimeter` or `lens_two_radius_perimeter`, with the same radius/center segment references as the matching area formula. The solver derives the two minor arc lengths and admits their sum only after comparison with `lens_coordinate_perimeter`; the residual types are `lens_equal_radius_perimeter_check` and `lens_two_radius_perimeter_check`.

`CompositeRegion(operation="partition")` is the strict region-decomposition contract. It requires at least two component regions plus `metadata.parent_region`. Every component must be a materializable convex polygon contained in the simple parent polygon, component interiors must not overlap, and their exact area sum must cover the parent. The state verifier rejects an invalid partition before admission. Solver closure derives component areas, records an `area_partition_residual` as verifier evidence, and admits `region_partition_area_sum` only when that residual is zero. The admitted area depends on component area fact IDs, while the residual remains in the complete fact store as admission evidence rather than becoming a calculation step in the public target derivation.

A semantic shaded difference uses `CompositeRegion(operation="difference")` with `metadata.area_formula.rule="region_difference"` and ordered `minuend`/`subtrahend` references matching the components. The default `mode="contained_hole"` requires exactly one fully contained hole. The restricted `mode="boundary_notch"` requires two axis-aligned rectangles where the subtrahend removes exactly one outer corner and materializes a deterministic six-vertex concave boundary. The verifier rejects partial overlap, external subtrahends, internal cutouts mislabeled as boundary notches, and unsupported modes. Solver closure subtracts the components' preferred semantic area facts, compares the result with the exact materialized area through `area_difference_residual`, and admits `region_difference_area` only at zero residual. As with partition admission, the residual stays in complete facts while the benchmark derivation exposes the public calculation chain.

Area ratios reuse the canonical `ratio` target with `metadata.ratio_kind="area"`, plus distinct existing `numerator` and `denominator` region IDs. The state contract rejects missing, repeated, or non-region references. Solver closure reads each region's preferred admitted area fact, rejects a zero denominator through `ratio_unresolved`, and emits a dimensionless `ratio` fact under `region_area_ratio`. This keeps ratio supervision connected to the same public-measurement area derivations rather than recomputing from coordinates.

Target types are normalized at the `FormalGeometryState` mutation/deserialization boundary. State schema `1.1` supports `value`, `proof`, and `constraint` modes. A proof target must use an admitted proof-capable relation/fact type; a constraint target additionally carries an exact operator and expected value. Solver output distinguishes proved/unproved claims and satisfied/violated/undetermined constraints instead of coercing every target into a numeric lookup. Stored state, checksum, rendering, solver lookup, task signatures, and round trips use one canonical target spelling.

Rendered scenes carry a non-blocking `metadata.render_quality` report. It records label bounds/overlap, duplicate display names, non-finite primitive coordinates, and counts. The same deterministic label placement is used by SVG and PNG output; the report is intended for benchmark filtering and visual regression diagnostics, not as a substitute for raster pixel comparison.

Point rendering separates geometry participation from visual annotation. `Point.metadata.show_label=false` suppresses a point name while retaining its marker, and `show_marker=false` suppresses the marker while preserving the point for segments, regions, solving, and provenance. Templates use these controls for auxiliary boundary vertices that are not referenced by the public question, preventing formal construction IDs from becoming visual clutter.

Deterministic visual regression uses `render_regression_signature(scene, width, height, padding)`. The versioned `geosym.render_regression.v1` payload binds the renderer-neutral scene fingerprint to the exact SVG bytes, direct PNG RGB pixels, PNG foreground mask, foreground-pixel count, and primitive/label counts. `render_png_rgb` produces the same RGB array consumed by `write_png` without creating an artifact, so tests keep only a compact JSON signature instead of committing generated images. A backend-only rendering change is therefore visible even when the formal scene is unchanged.

The maintained regression matrix fixes canonical variants for a sector, circular segment, two-hole polygon region, and annulus at one explicit viewport. Every case must pass renderer quality inspection and match all signature layers. The matrix is intentionally small and capability-oriented; it is not a generated-dataset snapshot.

Synthetic records additionally preserve the executable `construction_trace`, a `synthetic.lineage` block, the assigned `synthetic.split`, and the exported `render_quality` summary. Split assignment is topology-aware by default and keeps parameter/mutation variants with the same answer-free task structure in one partition; lineage-only grouping remains available explicitly.

Generation manifests expose bounded `candidate_diagnostics`. `checked` counts candidates that reached acceptance checks and always equals `accepted + rejected`; `skipped` records policy decisions such as a reached family quota; `considered` equals `checked + skipped`. Legacy-compatible reasons are supplemented by versioned diagnostic codes and aggregate counts by code, stage, component, and family. Domain exceptions distinguish invalid input, unknown references, duplicate objects, degeneracy, numerical instability, verification failure, unsupported construction, and unexpected execution failure. Each reason retains only a bounded number of examples. Accepted records carry `synthetic.metadata.candidate_validation` with verifier, derivation, solution-path, and difficulty summaries.

Every exported synthetic record receives a deterministic train/dev/test split at the exporter boundary. The default topology key excludes values, coordinates, IDs, and answers while including family, construction operations, target semantics, and diagnostic axes. The default ratio is 0.8/0.1/0.1 and can be changed with `--split-ratios-json`; input order remains unchanged. Existing labels persist during resumed export, and conflicting labels inside one selected split group fail immediately. The manifest reports split counts and group audits.

Synthetic records expose a versioned `task` block and top-level `public_givens`. Public givens contain only quantities rendered as measurement labels and relations represented by explicit construction or marker semantics; point coordinates, target facts, and other hidden closure facts are not copied into this public view. Task schema `1.1` records target mode, ordered construction operations, derivation summary, validated solution-path count/rules/checksum, object kinds, dimensions, diagnostic axes, and a deterministic structure signature. `solution_paths` schema `1.0` stores one or more topologically ordered fact subgraphs rooted at distinct target facts; requested alternative rules must each resolve exactly once and every path has an independent checksum/integrity report. The task checksum and validation audit these summaries before export.

Polygon measurement closure separates coordinate truth from benchmark-facing derivation. `shoelace_formula` and `closed_polyline_perimeter` remain internal exact checks, while templates may declare typed area or perimeter formulas. The solver first validates geometric preconditions and resolves referenced public measurement facts, then records a zero-expected residual against coordinate truth. Semantic facts are admitted only when that residual is exactly zero. A target may set `preferred_rule` so its exported derivation depends on rendered public measurements instead of silently selecting the coordinate formula. Nonzero residuals and unresolved preferred targets are rejected before record export.

`IncrementalFactClosure` maintains an explicit token-to-fact and fact-to-dependent index. Additions, modifications, deletions, relation changes, and target changes invalidate the transitive dependent subgraph and run an explicit rule agenda over the affected object/region dependency scope. Reports include changed tokens, invalidated/recomputed fact IDs, semantic scope, rule agenda, and before/after checksums. Optional equivalence verification compares the incremental result with a full solver run and fails on checksum divergence; global theorem discovery outside the affected dependency scope remains a full-closure concern.

`DerivationTracer` resolves both canonical fact IDs and typed object references such as `point:A` or `segment:AB`. Edges include the original `reference`; unresolved references are retained in `unresolved_dependencies` instead of being silently dropped.

Target derivations use `schema_version=1.0` and are exported as a topologically ordered supervision program. Every `steps` item names the produced fact, rule, exact/numeric value, input fact IDs, original object/fact references, depth, metadata, and provenance. `integrity` reports target uniqueness/reachability, acyclicity, unresolved dependencies, counts, and maximum depth. `checksum` covers the derivation payload excluding its independently computed `validation` result. `validate_derivation_payload` can audit an exported derivation without the original `FactStore`. Candidate admission rejects `invalid_derivation`, and the exporter refuses records whose derivation integrity or serialized validation fails.

Region area and perimeter facts reference the materialized `region_boundary` fact ID when a 2D boundary is resolved. This keeps target derivations connected to the exact outer ring, holes, and multi-component rings rather than only to an opaque region object token. The state verifier rejects self-intersecting 2D region boundaries, degenerate boundaries, holes outside their outer boundary, and holes that touch or overlap one another.

`SyntheticParserV0` is the first parser closed loop for canonical synthetic records. It emits typed `SyntheticObservation` data and a proposed `GeometryHypothesis`, explicitly excludes answer/facts from the observation, and uses `HypothesisAdmission` to produce an admitted debug state. `SyntheticParserDebugger` reports parser component metrics, grounding, verification, transition, and checksum axes. This is an oracle contract/debug harness, not evidence that open-world image parsing is solved.

The batch entry point is `python -m geosym_world.tools.parse_synthetic_v0`. It fails with the input path and JSONL line when a record is malformed, writes one structured result per input record, and prints periodic progress to stderr. `--admit` enables the grounding and admission pass; without it, every result remains a proposed hypothesis.

## 2D Linear Domains and Intersections

Line-like objects preserve their actual domain during deterministic intersection and fact closure: a line has an unbounded parameter, a ray requires `t >= 0`, and a segment requires `0 <= t <= 1`. Structured intersection results distinguish `unique`, `two_points`, `tangent`, `parallel`, `coincident`, `overlap`, `disjoint`, and `degenerate`, and serialize contact dimension, contact kind, and exactness. Collinear segment/ray overlap carries an explicit parameter interval, while extension-only intersections are not admitted as outputs or facts. Multi-solution intersections use deterministic ordering and may request positional selectors or `nearest`/`farthest` relative to an explicit reference point. Near-collinearity and near-tangency are diagnostic measurements only and never replace exact admission predicates.

An `Arc` is a strict directed domain rather than an unchecked circle annotation. Both endpoints must lie exactly on its circle, orientation is `ccw` or `cw`, and any supplied positive `central_angle <= 2*pi` must equal the endpoint sweep in that direction. Equal endpoints require explicit full-circle metadata. Line-arc, circle-arc, and arc-arc intersections solve the supporting equations and then filter every candidate through the exact directed sweep. A same-circle shared interval is `overlap`, not a finite point set. Builder operations, relation verification, and solver closure consume the same kernel semantics; constructed points receive `point_on_arc` facts and explicit curve relations receive independently checked `curve_intersection` facts. Extension-only or off-sweep candidates never enter persistent state.

State relations are normalized at the state boundary. Relation aliases such as `orthogonal`, `same_length`, `same_angle`, and `line_intersection` map to canonical types; symmetric `objects` lists are sorted; and every relation receives a deterministic `rel_*` ID. Explicit `equal_length` relations use `segments` or two `objects`, while explicit `equal_angle` relations use two three-point `angles`. The verifier checks these claims against exact coordinates and the solver emits asserted relation facts plus zero-expected residual facts with provenance.

Explicit `similar` and `congruent` polygon relations use `polygons: [left_id, right_id]` plus a total vertex `mapping`. Similarity is checked by corresponding side-square ratios; congruence is checked by corresponding side-square equality. Both produce asserted facts and exact residual facts, so a theorem claim cannot enter the admitted state without geometric support.

All real Parser pipelines use `HypothesisRecordAdapter` and `validate_proposed_parser_record`: parser records must contain no persistent `state`, at least one proposed hypothesis, provenance, and `admission_required=true`. `parser_record_audit` checks this contract without changing state. `ParserBenchmarkSlicer` aggregates component metrics and admission decisions by parser, task type, difficulty, and diagnostic axis; it is a diagnosis view, not an admission mechanism.

`python -m geosym_world.tools.diagnose_parser_v0` is the batch diagnosis entry point for synthetic Parser v0. It writes one diagnosis row per canonical input record, prints progress to stderr, and emits aggregate slices to stdout. Its reference is derived from formal state only for component debugging; it must not be interpreted as open-world parser evidence.

Composite region boolean operations use an exact polygon ring-set kernel. Inputs may contain multiple outer components and holes; rings are normalized by removing duplicate/collinear intermediate vertices and canonicalizing orientation/start. Collinear overlapping edges are split before exact ownership resolution. `intersection`, `union`, and `difference` can produce empty results, one ring, holes, or multiple components, and serialize component count, hole count, Euler characteristic, and vertex counts. Identical polygons retain explicit idempotent semantics. Curved boundaries, self-intersection, open/cyclic stitched boundaries, and other unresolved topology remain explicit `unsupported`/`area_only`; no boundary is guessed. SVG and PNG consume the same normalized rings with even-odd material semantics.

## Topology and Relation Facts

| family | fact types |
|---|---|
| Line/curve topology | `angle`, `parallel`, `perpendicular`, `line_intersection`, `curve_intersection` |
| Point incidence | `point_on_line`, `point_on_segment`, `point_on_circle`, `point_on_arc`, `collinear` |
| Plane topology | `point_on_plane`, `line_parallel_plane`, `line_on_plane`, `line_perpendicular_plane`, `line_plane_normal_dot`, `plane_parallel`, `plane_perpendicular` |
| 3D line-plane measures | `line_plane_angle`, `line_plane_distance` |
| Unified 3D angles | `space_angle` with metadata `kind` |

## Theorem-Lite Fact Families

These facts are coordinate-backed theorem checks. They are not a full synthetic proof system.

| family | fact types |
|---|---|
| Right triangle | `right_triangle`, `pythagorean_relation` |
| Isosceles triangle | `isosceles_triangle`, `equal_base_angles`, `equal_base_angle_residual` |
| Similar triangle | `similar_triangle`, `similarity_ratio_residual`, `similarity_scale_squared`, `equal_corresponding_angles`, `similar_angle_residual`, `aa_similarity`, `aa_similarity_residual`, `parallel_cut_similarity`, `parallel_cut_ratio_residual` |
| Angle bisector | `angle_bisector`, `equal_angle`, `angle_bisector_residual` |
| Circle chord | `chord_of_circle`, `central_angle`, `chord_length_formula_check`, `equal_chord_equal_central_angle` |
| Inscribed angle | `inscribed_angle`, `inscribed_angle_central_residual`, `arc_subtends_inscribed_angle`, `inscribed_angle_arc_residual`, `equal_inscribed_angles` |
| Tangent-chord | `tangent_to_circle`, `tangent_chord_angle`, `alternate_segment_angle`, `tangent_chord_angle_residual` |

## 3D Fact Families

| family | fact types |
|---|---|
| Projection | `point_projection_plane`, `projection_distance`, `projection_plane_residual`, `line_projection_plane`, `line_projection_plane_residual` |
| Dihedral angle | `dihedral_angle`, `dihedral_shared_edge` |
| Solid section | `section_interface`, `section_polygon`, `section_area`, `section_perimeter`, `section_degeneracy_measure`, `section_boundary_size`, `section_boundary_plane_residual` |
| Space angle | `space_angle` with `kind=line_line`, `line_plane`, or `plane_plane` |

## Current Boundaries

- The system is exact-coordinate-backed, using SymPy expressions.
- The theorem-lite layer can export validated alternative fact paths, but it is not a complete theorem prover and does not generate natural-language proofs.
- Generic boolean operations over curved regions remain outside the exact polygon ring-set contract.
- Automatic polyhedron-plane section currently intersects edges and sorts section points on the plane; it does not handle every degenerate coplanar-edge case.
- 3D operations are intentionally interface-first, with small pilot coverage rather than large data generation.
- Pilot export is the main integration smoke path; `tools/smoke_check.py` provides an in-memory assertion layer for selected invariants.
