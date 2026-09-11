# Verified Parser Pipeline

The Parser pipeline recovers evidence-bearing geometry candidates from public inputs, admits only supported candidates, compiles admitted structure into a graded geometric realization, and produces auditable redraws. It does not treat successful rendering or solvability as proof that the source image was parsed correctly.

## Runtime Boundary

`ParserInput` exposes exactly four fields:

```json
{
  "schema_version": "1.0",
  "id": "sample_id",
  "image": "path/to/public/image.png",
  "question": "Find the length of AB."
}
```

Formal state, scene metadata, answer, derivation, construction trace, and evaluator truth are forbidden parser inputs. `ParserSourceManifest` records source version, license, asset hash, split group, protocol, and public fields. A private typed reference is checksum-bound to both the public input and source manifest. Synthetic exports also replace parameter-bearing record IDs and image names with opaque sample IDs and content-addressed PNG paths; the mapping is retained only in the private source audit sidecar.

Typed synthetic Parser v0 references currently cover only scenes with `render_dim=2`. Image export still keeps 3D records and renderings in the main dataset, but excludes them from Parser input/reference sidecars instead of emitting incomplete projected truth. The synthesis manifest reports `num_parser_benchmark_records` and classified `parser_benchmark_skip_reasons`, so Parser benchmark coverage is explicit.

Supported protocols are `image_only`, `image_question`, and `native_import`. The current runnable visual baseline implements the first two. `native_import` has a source contract but no general importer yet.

Real image sources first enter a private catalog. Each JSONL row has strict fields:

```json
{
  "schema_version": "1.0",
  "source_id": "book_a_problem_17",
  "protocol": "image_question",
  "source_version": "scan-v1",
  "license_id": "internal-review-only",
  "split_group": "book_a_problem_17",
  "image": "artifacts/parser/sources/book_a_problem_17.png",
  "question": "Find the length of AB."
}
```

`prepare_parser_inputs_v0` validates all rows and duplicate source IDs before materialization, hashes every source image, copies it to a content-addressed opaque public name, and emits separate public input, private source audit, and dataset manifest files. The Parser receives only the public JSONL.

## Trusted Human References

Real-image evaluator truth uses a separate private annotation chain. It is not produced by Parser admission and cannot be inferred from a successful reconstruction:

The normative inclusion, evidence, role-signature, ambiguity, review, and adjudication rules are frozen in [`PARSER_GT_V1.md`](PARSER_GT_V1.md). The code accepts only guideline version `parser-gt-v1`; a new semantic policy requires an explicit version and migration.

```text
public ParserInput + source manifest
  -> draft primary ParseGraph annotation
  -> independent ParseGraph review + explicit disagreements
  -> adjudication + one final ParseGraph
  -> evaluator reference + private lineage
```

`ParserAnnotationRecord`, `ParserAnnotationReview`, and `ParserAnnotationAdjudication` are strict, checksummed records. Every stage binds the same public-input checksum, source-manifest checksum, source ID, and annotation-guideline version. The review must be authored by a different person. A stricter policy can also require a third adjudicator. Disagreeing reviews enumerate typed disagreement records; every disagreement must have exactly one adjudication resolution before the final graph can be trusted. Unresolved adjudications and unresolved final-graph conflicts are rejected.

This chain deliberately accepts only declared unassisted human annotation: `model_assistance` must be false, human-observed candidates use `confidence=1.0` and `score_kind=human_annotation`, and deterministic candidates use `score_kind=deterministic_rule`. Parser/model output may be shown to annotators in a separate experimental protocol, but it cannot be relabeled as independent human GT through this contract.

Materialization produces two different private assets. `references.jsonl` contains evaluator input without annotator identities. `references.lineage.jsonl` retains annotator, reviewer, adjudicator, policy, and every upstream checksum for audit. Source audits, annotations, reviews, adjudications, lineage, and references all remain outside Parser runtime input and should be distributed only according to source licenses.

## ParseGraph

`ParseGraph` is the parser intermediate representation. It contains:

- localized image/text/native evidence;
- typed objects and stable references;
- proposed relations and explicit candidate dependencies;
- quantities represented by a safe expression AST;
- regions, holes, components, and region set expressions;
- value, proof, or constraint targets;
- unresolved or resolved conflict groups.

Every candidate has `origin`, `confidence`, `score_kind`, and evidence references. Anonymous intersections use stable IDs without fabricated visible names. Geometry fields are kind-specific:

| kind | accepted geometry fields |
| --- | --- |
| `point` | `pixel`, `exact_coords` |
| `line`, `segment`, `ray` | `p1`, `p2`, `angle_deg` |
| `circle` | `center`, `radius_px`, `exact_radius` |
| `arc` | `orientation`, `exact_central_angle` |
| `text`, `marker` | `bbox` |

References carry semantic roles such as segment `start`/`end`, ray `start`/`through`, circle `center`, arc `circle`/`start`/`end`, and numbered polygon vertices. Unsupported extensions belong in `metadata`; changing executable geometry requires a schema version change.

Visible relation glyphs are first-class marker objects rather than generic line fragments. Synthetic private references project each renderer marker to a tight pixel bbox using the same renderer-neutral marker geometry consumed by SVG and PNG. They retain a canonical marker subtype and references to observable support points or linear objects. This scene-derived information is evaluator-only truth and never enters `ParserInput`.

## Stages

The current `TypedDiagramParserV0` runs:

```text
ParserInput
  -> OpenCV primitive detection
  -> endpoint/intersection clustering
  -> object and topology proposals
  -> observed-fill component grounding
  -> explicit visual-marker grounding
  -> optional OCR label proposals
  -> one-to-one point-name grounding
  -> conservative image-measurement grounding
  -> explicit question-relation grounding
  -> safe explicit question quantity parsing
  -> conservative question target proposal
  -> ParseGraph
```

The bundled implementation detects line segments, circles, open circular-arc contours, filled point blobs, light chromatic fill components, text regions, endpoints, bounded line/curve intersections, collinear segment-overlap candidates, line-circle tangent candidates, and four explicit marker classes: right-angle boxes, parallel chevrons, equal-length ticks, and angle arcs. Hough line fragments use canonical undirected angles, transitive collinear merging, and a configurable image-relative minimum length so local circle arcs and glyph strokes do not dominate the candidate graph. Filled point blobs are only visual point evidence: small circular candidates require a dark disk, avoid text boxes and already-known topology points/circle centers, and are never named from OCR without a separate binding. Arc fitting requires a stable open contour with bounded radial residual and rejects full circular contours; a typed ARC object is created only when the fitted arc has one unique detected support circle, while an unsupported arc remains localized evidence. When a high-support full circle is already present, near-major arc candidates from that same raster contour are suppressed to avoid duplicating the circle as an arc; genuine major arcs without full-circle support remain candidates. Finite line-arc, circle-arc, and arc-arc intersections are filtered by the fitted arc span, become shared anonymous point observations, and are compiled into typed intersection relations only when the involved arc objects are grounded. Detector and marker-detector versions and thresholds are serialized in ParseGraph metadata. Apparent parallel/perpendicular, tangent, and linear-overlap relations remain proposals marked unsuitable for exact automatic admission unless independent evidence supports them. OCR is injectable and unresolved by default. Recognized point names are grounded spatially; lowercase line-name labels are grounded only near the interior of a visual line or segment. Ambiguous or duplicate bindings remain conflicts.

`OCRImagePrediction` is the provider-neutral OCR replay contract. Each JSONL row binds one image SHA-256, the complete detected text-region ID/bbox set checksum, backend, model revision, per-region text/status/confidence, and its own checksum. `RecordedLabelRecognizer` refuses changed image bytes, missing or additional images, detector-region drift, duplicate regions, malformed confidence, and tampered payloads. OCR provenance is copied into text-object and evidence metadata. OCR remains an uncertain observation source: point-name and quantity grounding, conflicts, admission, and geometric verification still run afterward.

`TesseractLabelRecognizer` is an optional live provider. It crops each detector text region independently, runs Tesseract with the configured language and page-segmentation mode, and returns resolved text only when Tesseract supplies a positive confidence; empty or non-positive results remain unresolved. Enable it in the verified workflow with `--tesseract-ocr`. The `pytesseract` package and a system Tesseract binary are required; no automatic installation, retry, or recorded-prediction fallback is performed. Recorded OCR remains the reproducible option for benchmark runs.

Right-angle detection requires a supported L-shaped glyph near a finite intersection of two detected lines. It creates a `marker` object bound to both support segments and a `perpendicular` relation that depends on that marker; the weaker angle-residual proposal for the same pair is suppressed. Bound marker evidence can pass strict admission, while a visually near-90-degree intersection without the glyph cannot.

Linear marker detection estimates each support's raster stroke width before matching transverse ticks or chevrons, excludes known endpoints and intersections, and suppresses duplicate templates around the same glyph. Every equal-length tick is grounded to one segment interval: the whole detected support when it has only two known endpoints, or one unique adjacent subsegment when admitted endpoints/intersections already partition a merged support. The Parser never creates a midpoint from the tick position. Exactly two compatible ticks on two distinct grounded intervals may create one `equal_length` relation, including two intervals on the same merged support; two ticks that resolve to the same unsplit segment remain marker evidence only.

Angle-arc detection requires a curve core outside both measured support bands and stable support across at least three neighboring radii, preventing an ordinary thick vertex from masquerading as an arc. Detector ray signs, a unique support intersection, and the nearest known topology point on each indicated ray jointly define an ordered `(arm, vertex, arm)` angle. Exactly two compatible, fully grounded arc markers may create one `equal_angle` relation. When those angles share exactly one grounded ray and the shared direction matches the internal or external bisector within tolerance, a compiled `angle_bisector` relation depends on the admitted equal-angle relation. Missing ray direction, vertex, or arm points leaves the arc as evidence only. A generic arc does not imply either semantic relation.

Fill detection is an appearance-only baseline for clean colored diagrams. It extracts sufficiently large light chromatic components, their outer pixel contours, coarse interior gaps, color, bbox, and score. A component is bound to a straight-line planar face only when contour containment, interior coverage, and area agreement select one face with a clear margin; unmatched components remain localized evidence, including when planar-face inference is disabled. A bound face records `fill_status=observed`, while `shading_status` remains `unknown`. Interior gaps are retained in fill evidence and bound-face metadata with `interior_gap_count` and `interior_gap_status=unresolved_boundary_candidate`; they do not populate `ParseRegion.holes` until a separate closed `ParseRegion` boundary is grounded. The face-extraction metadata records the gap count and `hole_policy=requires_closed_parse_region_boundary_support`, and the fill stage reports `partial` while such candidates remain unresolved. Ordinary polygon styling, target highlighting, and mathematical shading can look similar, and text occlusion can create white gaps, so fill color or an `interior_gap` is never promoted directly to a shaded-set or hole fact. When the public question explicitly asks for the shaded area and exactly one parsed region has observed fill, the Parser emits a separate `shaded_region(region)` appearance hypothesis and makes the target depend on it. The default gate abstains. The opt-in `permit_unique_fill_shading` policy independently checks an image-text origin, one region argument, exactly one observed-fill region, argument identity, an explicit English/Chinese shaded-area text span, and bound fill evidence before admitting the relation and dependent target. Multiple fills or a tampered graph fail the certificate instead of selecting by detector order. Monochrome hatching, gradient fills, general curved boundaries, and open-world shaded-set admission are not covered by this clean-subset policy.

The clean curved-boundary subset additionally recognizes two uniquely matched forms. A filled arc-chord region requires one grounded visible arc, one finite visible chord support, and one matching fill contour. A filled arc-radii region requires the arc, two distinct finite radial supports, a unique visible support intersection near the fitted circle center, and a matching fill contour. The Parser reuses an original support segment when it exactly equals a boundary edge and creates a compiled bounded subsegment only when that support extends beyond the required endpoints. Resulting `ParseRegion` candidates record `boundary_kind=arc_chord` or `arc_radii`, keep `shading_status=unknown`, and can be source-redrawn with sampled arc boundaries. Ambiguous contour matches, ungrounded arcs, missing supports, lenses, hatching, and general curve Boolean arrangements remain unresolved.

Recognized image expressions can produce segment-length quantity candidates when their label center lies near the interior of a finite segment. Roots, fractions, powers, and `pi` use the same restricted AST as question quantities. Competing nearby segments create an unresolved conflict; invalid expressions and labels without a supported object produce stage issues. Spatial proximity supplies localized evidence, not a trusted semantic binding, so these image quantities abstain under the default bound-evidence admission policy until stronger grounding is available. Point labels such as `A` and `A1` are excluded from measurement classification.

Public question text currently recognizes only explicit segment-length equalities such as `AB = 4`, `AB = 2√13/3`, or `AB = \frac{3}{2}`. Values compile through a restricted AST parser; unsupported syntax remains unresolved evidence.

The question relation stage recognizes explicit parallel, perpendicular, equal-length, midpoint, and three-point collinear assertions in a bounded English/Chinese grammar. Segment and point references resolve only through unique point-name bindings; every relation depends on those bindings and retains its exact text span. Missing or ambiguous references leave localized evidence plus a structured `reference_unresolved` issue and do not create a partially grounded relation. `AB = CD` is treated as equal segment lengths rather than a symbolic numeric value. When an explicit text relation agrees with a pixel-angle proposal for the same supports, the weaker visual duplicate is suppressed and its ID is recorded in graph metadata; visual/text disagreement remains visible.

When the question uses an explicit proof cue such as `Prove that AB is parallel to CD`, the target stage reuses the unique parsed question relation. The target kind becomes the proved predicate, its arguments retain the grounded objects, and its dependencies include the relation candidate. Reconstruction serializes proof targets with the stable ordered key `object_0|object_1`; an unresolved or ambiguous proof assertion remains a target hypothesis instead of being compiled as a generic `proof` object.

`ExternalParseGraphPrediction` is the provider-neutral comparison path for direct VLM or specialized-model output. It stores only a public-input checksum, one strict ParseGraph, backend/model/revision/prompt identifiers, decoding settings, token usage, and an optional request ID. External candidates may use image/text origins, but cannot claim trusted `native` or `compiled` provenance; the model must not self-authorize evaluator truth or derived-fact admission. It rejects human/verified provenance claims, private source IDs, evaluator reference markers, formal state, answer, solution, scene, derivation, source-manifest metadata, and secret-bearing audit fields. `RecordedParseGraphParser` replays these fixed predictions through the same pixel verifier, admission, reconstruction, and redraw workflow as the OpenCV baseline. It does not call a model or repair malformed model output.

`GeoparsingPrediction` is the stricter adapter input for the GDP-29K/Geoparsing formal language. It binds the public `ParserInput` checksum and image-byte SHA-256 to a fixed backend, model revision, prompt version, and the six formal fields `points`, `lines`, `circles`, `planes`, `structure`, and `semantics`. The adapter supports ordered and named lines, circles, planes, the published solid categories, perpendicular relations with named intersections, parallel chains, segment equalities and lengths, angle equalities and measures, and numeric arc measures. Compact references such as `A1B1` are accepted only when they have one unique segmentation under the declared point set. Unsupported or ambiguous items become typed import diagnostics; strict mode fails the record instead of dropping them.

The public Geoparsing format has no point coordinates, text boxes, marker boxes, stroke support, or other localized evidence. Every imported item therefore carries `IMAGE_ONLY` evidence. Missing model confidence is represented as `confidence=0` with `score_kind=unscored_external_sequence`, not an invented probability. Even syntactically complete output remains a `PROPOSED` hypothesis and abstains under the default admission policy until an independent localization or geometric verification stage supplies stronger evidence. Dataset metadata such as source diagram paths, split, style, and category is not accepted inside the runtime formal-description payload.

Before admission, `PixelTopologyVerifier` independently checks structural image candidates against the source raster. It verifies direction-aware stroke support for lines, rays, segments, circles, and grounded arcs; local support and geometric residuals at intersections; and declared point order along finite supports. Arc checks sample only the declared start, orientation, and sweep interval, so the supporting full circle is not required to be visibly complete. Each candidate receives `passed`, `failed`, or `unknown`, together with metrics, source graph checksum, source image hash, verifier version, and configuration. A nearby crossing stroke cannot by itself certify a missing support because line checks sample across the candidate normal and circle checks sample radially.

`ParseGraphAdmission` then applies confidence, evidence precision, pixel-topology results, conflicts, and dependency closure. The verified workflow requires a checksum-bound topology result for image-origin structural candidates. Failed candidates are rejected, unavailable or unknown checks abstain, and no result is promoted merely because reconstruction later succeeds. Admission returns a separate admitted graph and per-candidate decisions; it never mutates persistent geometry state.

`ParseGraphReconstructor` accepts only an admission report. It topologically compiles admitted points, segments, lines, rays, circles, arcs, polygons, supported regions, relations, quantities, and targets into a construction trace. The trace is verified before state materialization.

In exact mode, a point without explicit coordinates can be recovered from one admitted midpoint, perpendicular-foot, internal segment-ratio, positive ray-distance, or intersection construction when its dependencies are already exact. A line with one exact point can be recovered when exactly one admitted parallel-through, perpendicular-through, or internal/external angle-bisector definition supplies its direction. A circle can use one consistent admitted radius or diameter quantity when `exact_radius` is absent. Competing exact constructions produce `reconstruction.construction_ambiguous`; inconsistent circle size constraints produce `reconstruction.condition_conflict`.

Linear-linear intersections must be unique. A line-circle or circle-circle tangent is single-valued; a two-point intersection requires an explicit `first`, `second`, `upper`, `lower`, `left`, or `right` selection in relation metadata. Missing or competing branch specifications produce `reconstruction.branch_ambiguous`; observed pixels are not used to silently choose an exact branch. Every relation-created object keeps the source ParseRelation ID, dependencies, evidence references, score provenance, and candidate metadata in the replayed State. The same midpoint, foot, ratio, ray-distance, and angle-bisector relations are independently checked when exact coordinates were supplied instead of reconstructed.

`ParseGraphReconstructor.enumerate_branches()` handles the finite two-point subset without changing source assertions. It emits checksum-bearing `ReconstructionBranchSet` output, records runtime choices separately from source metadata, independently replays and verifies every candidate branch, and enforces an explicit `max_branches` search limit. Schema `1.1` places only exact realizations in `branches`; non-exact candidates remain fully auditable in `rejected_branches` with their choices, issues, trace, and checksum. Two anonymous points on the same support pair are assigned by the two valid bijections, rather than generating duplicate-point combinations.

Exact reconstruction also recognizes an isolated complete SSS triangle component when its three point candidates have no exact coordinates or competing construction, its three distinct segment supports form exactly one closed three-point component, and every side has one consistent finite numeric length assertion. A deterministic realization gauge places the lexicographically first support on the x-axis and derives the third point by the exact SSS equations. The two mirror realizations remain explicit `first`/`second` runtime branches and are exposed by `enumerate_branches()`; pixels never select the branch. Generated point coordinates carry `fact_scope=realization_fact`, `reconstruction_gauge=sss_triangle_v1`, source segment IDs, and source quantity IDs, while the side lengths remain separate source assertions. Non-positive lengths, degenerate/impossible triangles, symbolic side values, open components, duplicate supports, and points with another exact construction do not enter this planner.

## Reconstruction Grades

| status | meaning |
| --- | --- |
| `exact` | Every materialized coordinate or parameter came from a validated exact expression tree. |
| `numeric_witness` | Observed pixels produced one explicit, non-source geometric realization. |
| `partial` | Some admitted candidates were underconstrained, ambiguous, unsupported, or failed verification. |
| `unsupported` | No executable geometry could be materialized. |

Reconstruction analysis reports coordinate basis, free symbols, unplaced points, branch ambiguity, and unsupported candidates. Exact mode never falls back to pixels. Pixel mode does not promote measured lengths, incidental equalities, or coordinate-induced properties to source facts.

Facts use four scopes:

- `source_assertion`: admitted source condition;
- `entailed_fact`: derived only from source reasoning;
- `realization_fact`: true only of a selected coordinate witness;
- `generation_parameter`: synthesis-only value.

An entailed derivation cannot depend on realization or generation facts.

## Redraw Modes

| mode | contract |
| --- | --- |
| `source_redraw` | Uses admitted observed pixels; preserves explicit labels/markers only; makes no mathematical-realization claim. |
| `semantic_redraw` | Requires a verified state; renders admitted source annotations and suppresses inferred relation markers. |
| `resynthesis` | Requires exact reconstruction; applies an explicitly selected seeded rigid or eligible similarity transform, then verifies trace and topology invariants. |

All modes use the shared `RenderScene` and SVG/PNG backends. Each result stores source/admitted checksums, verification, lineage, invariants, scene metadata, and its own checksum.

## Commands

Prepare public inputs from a private source catalog:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.prepare_parser_inputs_v0 \
  --catalog artifacts/parser/sources/catalog.jsonl \
  --output-dir artifacts/parser/datasets/prepared_v0 \
  --progress-every 25
```

After two independent human annotations and adjudication, materialize trusted evaluator references:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.materialize_parser_references_v0 \
  --inputs artifacts/parser/datasets/prepared_v0/parser-input.jsonl \
  --source-audit artifacts/parser/datasets/prepared_v0/source-audit.jsonl \
  --annotations artifacts/parser/annotations/primary.jsonl \
  --reviews artifacts/parser/annotations/reviews.jsonl \
  --adjudications artifacts/parser/annotations/adjudications.jsonl \
  --output artifacts/parser/datasets/adjudicated_v0/references.jsonl \
  --progress-every 25
```

Add `--require-distinct-adjudicator` when policy requires three different people. The command checks every record and ID set before writing references, writes atomically, hashes all input and output files, and emits a manifest beside the reference JSONL.

Run the verified workflow over public inputs:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.parse_reconstruct_v0 \
  --input artifacts/generated/review/records.jsonl.parser-input.jsonl \
  --output artifacts/parser/verified_v0/results.jsonl \
  --reconstruction-mode pixel_witness \
  --format both \
  --progress-every 25
```

Replay provider-generated typed predictions through the identical verified workflow:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.parse_reconstruct_v0 \
  --input artifacts/parser/datasets/prepared_v0/parser-input.jsonl \
  --external-predictions artifacts/parser/model_runs/run_001/external-predictions.jsonl \
  --output artifacts/parser/model_runs/run_001/verified.jsonl \
  --reconstruction-mode pixel_witness \
  --format both \
  --progress-every 25
```

The external JSONL must contain one checksummed `ExternalParseGraphPrediction` per public input and no extra IDs. The command validates the complete ID set before opening the output and records model configurations plus input/output hashes in its manifest.

Replay externally produced OCR regions while retaining the built-in visual detector and grounding stages:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.parse_reconstruct_v0 \
  --input artifacts/parser/datasets/prepared_v0/parser-input.jsonl \
  --ocr-predictions artifacts/parser/ocr_runs/run_001/ocr-predictions.jsonl \
  --output artifacts/parser/ocr_runs/run_001/verified.jsonl \
  --reconstruction-mode pixel_witness \
  --format both \
  --progress-every 25
```

Replay checksum-bound GDP-29K/Geoparsing formal descriptions without granting them trusted evidence:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.parse_reconstruct_v0 \
  --input artifacts/parser/datasets/prepared_v0/parser-input.jsonl \
  --geoparsing-predictions artifacts/parser/geoparsing/run_001/predictions.jsonl \
  --output artifacts/parser/geoparsing/run_001/verified.jsonl \
  --reconstruction-mode pixel_witness \
  --format none \
  --progress-every 25
```

The Geoparsing JSONL contains one checksummed `GeoparsingPrediction` per public input. Strict import is the default. `--allow-partial-geoparsing` keeps only representable candidates and records diagnostic counts in the run manifest; it does not strengthen evidence or admission status.

Ground Geoparsing object references against the deterministic visual graph with:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.parse_reconstruct_v0 \
  --input artifacts/parser/datasets/prepared_v0/parser-input.jsonl \
  --geoparsing-predictions artifacts/parser/geoparsing/run_001/predictions.jsonl \
  --ocr-predictions artifacts/parser/ocr_runs/run_001/ocr-predictions.jsonl \
  --fuse-geoparsing-visual \
  --output artifacts/parser/geoparsing/run_001/fused-verified.jsonl \
  --reconstruction-mode pixel_witness \
  --format both
```

Fusion accepts only unique, non-conflicting point-name bindings and unique endpoint/ordered-support matches. It rewrites semantic references to localized visual objects and removes exact typed duplicates, while preserving every unmatched semantic candidate's original `IMAGE_ONLY` evidence. Fusion never upgrades evidence precision or creates a fact; admission remains a separate stage. Ambiguous labels and non-unique local matches remain unresolved.

Fusion also emits a checksummed `compatibility_certificate` with one
`compatible`/`incompatible`/`unknown` check per semantic object, relation, quantity,
region, and target. It can compare mapped linear directions, point-circle pixel
residuals, exact local typed candidates, and grounded target references. The
certificate is an audit signal only: an incompatible check is not a parser crash,
and a compatible check never upgrades `IMAGE_ONLY` evidence or bypasses admission.
For marker-backed relations, the certificate also records the canonical glyph
subtype, dependency completeness, and whether support lines, marked segments,
or ordered angle arms cover the grounded relation references.
For a locally grounded quantity with a recognized label object, it also compares
the label's restricted mathematical AST with the semantic expression; missing
or invalid label text remains `unknown`.

Audit the adapter's syntax coverage on GDP-style JSONL without reading images or calling a model:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.audit_geoparsing_grammar_v0 \
  --input artifacts/parser/geoparsing/gdp-formal.jsonl \
  --output artifacts/parser/geoparsing/grammar-audit.json \
  --id-field id \
  --progress-every 1000
```

By default, the tool selects the six formal fields from each top-level row and ignores unrelated dataset metadata. Use `--formal-field formal_description` only when those fields are nested. The checksummed report counts complete records, emitted typed candidates, diagnostic codes/fields, and bounded examples. Its `complete_rate` measures adapter grammar coverage only; it is not image parsing accuracy, grounding accuracy, or model quality.

`--external-predictions` cannot be combined with OCR or Geoparsing input. OCR and Geoparsing predictions can be combined only with `--fuse-geoparsing-visual`. The OCR JSONL contains one checksummed `OCRImagePrediction` per unique input-image hash. Its region IDs and `[x, y, width, height]` boxes must exactly match the current detector output. The run validates complete image-hash and sample-ID coverage before opening the output and records both sidecar hashes plus backend/model/revision sets in the manifest. Parser run manifests use schema `geosym.parser_run_manifest.v8` and record the selected external source, fusion configuration, import diagnostics, branch-enumeration configuration, accepted/rejected branch statistics, branch redraw status, and the complete admission policy.

Use `--permit-unique-fill-shading` only for the documented clean colored-diagram subset. The flag selects policy `verified_pixels_unique_fill_shading_v1`; without it the same appearance hypothesis remains unadmitted. The run manifest records the flag through `configuration.admission_policy.permit_unique_fill_shading`, so permissive and strict results cannot be silently mixed.

Use `--reconstruction-mode exact --resynthesis-seeds 11,23` only when inputs provide exact expression trees. Outputs are grouped under `renders/source_redraw`, `renders/semantic_redraw`, and `renders/resynthesis`. The run manifest records input/output hashes, admission and pixel-topology configuration, status counts, admission counts, diagnostics, and artifact counts.

Export every supported finite exact realization without silently selecting one:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.parse_reconstruct_v0 \
  --input artifacts/parser/datasets/prepared_v0/parser-input.jsonl \
  --external-predictions artifacts/parser/model_runs/run_001/external-predictions.jsonl \
  --output artifacts/parser/model_runs/run_001/verified-branches.jsonl \
  --reconstruction-mode exact \
  --enumerate-branches \
  --max-branches 16 \
  --format none
```

Each output row keeps the ordinary unresolved `result.reconstruction` and adds a checksummed `branch_set`. Every accepted or rejected candidate contains its own branch choices, trace, state, fact scopes, verification, analysis, issues, and checksum. Every accepted exact branch also produces a checksummed `BranchRedrawResult` in `branch_redraws`; its semantic scene ID and lineage bind the reconstruction checksum and branch choices. With `--format svg`, `png`, or `both`, files are written under `renders/semantic_redraw_branches`; rejected branches are never rendered. The manifest separately aggregates searched candidates, exact branches, rejected branches, statuses, rejection codes, and branch redraw validity. Enumeration fails when the pre-pruning candidate count exceeds `--max-branches`; it remains intentionally incompatible with `--resynthesis-seeds` because no preferred branch has been established.

Exact distance reconstruction currently supports isolated SSS triangles, finite 2-tree meshes, and overconstrained graphs that contain a deterministic spanning triangular scaffold. The planner fixes only translation/rotation gauge, exports every local mirror choice explicitly, and then evaluates every admitted redundant length as an exact squared-distance residual. Conflicting redundant edges reject only the incompatible candidate branches. Distance graphs without a spanning triangular scaffold, mixed nonlinear constraints, and continuous solution families remain unsupported rather than receiving guessed coordinates.

The default resynthesis transform is `rigid`. `--resynthesis-transform similarity` enables exact positive uniform scaling plus rotation and translation. Similarity mode is rejected when admitted source facts contain absolute length, radius, diameter, distance, coordinate, perimeter, area, arc-length, or volume values, or when a dimensional constraint target would be changed. It preserves dimensionless geometry and target type, not the original numeric answer. No automatic fallback to rigid mode occurs.

Evaluate against private references:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.evaluate_parse_graph_v0 \
  --predictions artifacts/parser/verified_v0/results.jsonl \
  --references artifacts/generated/review/records.jsonl.typed-parser-reference.jsonl \
  --output artifacts/benchmark/parse_graph_v0/metrics.jsonl
```

Evaluation matches objects before relations and keeps the original five-component macro score over objects, relations, quantities, regions, and targets. It additionally reports label-text, marker, label-binding, and topology metrics so failures are not hidden inside aggregate object or relation scores. Marker matching requires both spatial agreement and the same canonical subtype. Each component has a strict per-sample exact-match flag; the dataset manifest aggregates component Sample Accuracy, five-core-component perfect rate, and all-nine-component Perfect Parsing Rate. Evaluator result schema `1.3` and metric/manifest envelopes `v2` add the marker component and symmetric `linear_overlap` topology accounting without changing the five-core macro score. Equivalent safe expression trees are compared by canonical mathematical value, including degree/radian normalization for angles. Diagnostics include wrong point position, wrong label, wrong marker, wrong binding, topology mismatch, unmatched IDs, and missing holes. Ground-truth relations are not used to choose object matches.

## Current Limits

- No bundled OCR model or general text-to-object binding model. Optional Tesseract execution and checksum-bound external OCR replay are supported, but no OCR backend accuracy has been measured.
- GDP-29K/Geoparsing import covers its published formal language as global image hypotheses. Conservative fusion can ground uniquely named points and supported objects to an independently detected visual graph, but does not upgrade semantic evidence, resolve ambiguous labels, or recover regions, shading, targets, construction traces, or exact coordinates.
- GDP grammar coverage has an offline audit path, but no GDP corpus revision has yet been pinned or audited in this repository.
- Fill grounding covers clean light-chromatic components, straight-line planar faces, and uniquely matched filled arc-chord or arc-radii regions. An explicit shaded-area question plus one unique observed fill produces an auditable `shaded_region` hypothesis. It remains unadmitted by default; the opt-in clean-subset policy can admit it only after its composite certificate passes. General semantic shading, general holes, hatching, lenses, and arbitrary curved region Boolean structure remain unsupported.
- Same-support equal-length ticks require an existing topology split and two unambiguous bounded intervals; tick positions never create missing split points. Equal-angle requires two grouped arc glyphs with fully grounded ordered angles. Angle-bisector inference is limited to two such angles sharing one grounded ray with a verified internal/external direction; dedicated or ambiguous bisector glyphs remain unsupported.
- No model provider is bundled into the verified path; external typed predictions can be replayed, but no model accuracy has been measured.
- No claim of open-world real-diagram accuracy.
- No general nonlinear constraint solver for recovering missing exact coordinates. Exact gauge recovery currently adds only isolated complete SSS triangle components to the explicit construction subset.
- No general branch search for multi-solution inverse constructions; the exact subset requires an explicit branch token for two-point line-circle and circle-circle intersections.
- No native CAD/vector importer implementation.
- Curved ParseRegion source redraw supports explicit arc/segment boundaries; automatic parsing is limited to unique filled arc-chord and arc-radii subsets and does not yet provide general curved-region Boolean reconstruction.
- Non-rigid resynthesis currently covers only exact 2D similarities on tasks without dimensional source assertions; general constraint-preserving parameter edits are not implemented.
- The annotation/review/adjudication contract is implemented, but a licensed, manually annotated real-image benchmark is still required.

These limits produce explicit abstentions, partial status, or unsupported diagnostics. They must not be hidden by pixel defaults, solver success, or a visually plausible redraw.

## Hybrid Batch Entry

For a staged script-plus-VLM experiment over real images, use the formal package
entry point. The input JSONL may be a private dataset row, but the pipeline
projects each row to `ParserInput` before any detector, script extractor, VLM, or
builder sees it. `answer`, `solution`, formal state, scene metadata, and other
supervision are never forwarded to Parser runtime.

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.parse_real_data_hybrid_v0 \
  --dataset-name geometry3k \
  --records-jsonl artifacts/datasets/raw/geometry3k/metadata/records.jsonl \
  --backend openrouter \
  --model qwen/qwen3.7-plus \
  --extraction-mode staged \
  --judge-backend none \
  --parsed-root artifacts/datasets/parsed \
  --progress-every 25
```

The command writes per-record artifacts under
`artifacts/datasets/parsed/<dataset>/` and a checksummed `run_manifest.json`.
It stops at the first record error by default. Add `--continue-on-error` only
when an explicit failed-record report is desired; failed records are counted
separately and are never reported as successful parses. `--backend mock` is the
offline contract test, while `--backend scripted` is a model-free lower bound,
not a parser quality result.
