# Parser Core

The Parser uses one typed semantic graph and one orchestration path. It does not claim production-grade open-world reconstruction.

## Runtime Boundary

`ParserInput` contains exactly `schema_version`, `id`, `image`, and `question`. Formal state, answers, construction traces, scene metadata, and benchmark references are not Parser inputs.

## Main Flow

```text
image + optional question
        -> VLM geometry semantics (no coordinates)
        -> semantic_compiler (IDs, supports, incidence, references)
        -> ParseGraph proposal
        -> optional PixelTopologyVerifier
        -> independent ParseGraphAdmission report
        -> SkeletonSeedBuilder (drawable state)
        -> incremental constraint application
        -> candidate Geometry State + independent verification
        -> state_to_render_scene
```

`ParserPipeline` is the only public orchestration API for this sequence. The extractor is injected, so provider transport remains separate from geometry logic. It validates that every prediction is checksum-bound to the exact public input and preserves the prediction, optional pixel report, admission decisions, realization, verification report, and per-candidate lifecycle in one result.

The pipeline realizes the full interpretation as a conditional candidate, independently of admission. Pixel rejection or insufficient bound evidence remains visible in the admission report but does not delete the interpretation before candidate construction. Unresolved conflict groups still prevent proposal realization. This is one explicit execution path, not an automatic fallback from a rejected graph.

The OpenRouter adapter performs one structural extraction request. Prompt version `geometry-semantics-zh-v12` lives in `src/geosym_world/parser/prompts/geometry_extraction_zh.txt`, distributed as package data and loaded directly by `_parsegraph_instruction()`. Edit that file to change extraction instructions; there is no duplicate inline prompt. Only the optional question is appended as a JSON input block. Image dimensions remain local diagnostic metadata, not a request for coordinates.

The model describes `points`, `strokes`, `circles`, `arcs`, `relations`, `measurements`, `regions`, `target`, and `uncertain`. It supplies geometric meaning rather than ParseGraph IDs, argument roles, confidence bookkeeping, coordinates, bounding boxes, pixel radii or layout hints. An ordered stroke such as `[A,H,B]` states that H lies between A and B; a reference such as `{"segment":["A","B"]}` names a support without requiring the model to declare a second object record. Anonymous junction names such as `_1` are not printed point labels. `basis` distinguishes diagram, mark and text evidence. Missing measurements do not block structural extraction.

`semantic_compiler.py` deterministically assigns IDs, deduplicates supports, expands ordered strokes into incidence/order constraints, binds measurements and queries, and translates circle/arc membership. It rejects undeclared names, invalid supports and coordinate fields with explicit errors. The original semantic payload and statement locations remain in ParseGraph metadata for diagnosis. Display switches are computed from evidence basis. All outputs remain hypotheses; the default confidence of 0.5 is explicitly uncalibrated, not a model confidence estimate or an admission decision.

`SkeletonSeedBuilder` first materializes drawable points, supports, curves, polygons and regions. It applies each optional construction operation independently; an invalid arc, region or relation is recorded as a skipped operation and cannot remove an otherwise drawable skeleton. `SeedRealizer` then evaluates constraints and reports satisfied, violated or unsupported hypotheses. When the input is a semantic-only proposal, deterministic topology layout uses declared polygon/region cycles or the support graph before applying incidence and ordered-point completion; existing layout coordinates are preserved instead of repeatedly flattening them onto every support. Numeric witness refinement is reserved for observed or exact paths, so a non-exact seed remains readable even when measurements are incomplete. These coordinates are not recovered source coordinates. Anonymous junctions do not receive invented labels. Extracted but unsupported constraints and incomplete circle definitions remain explicit limitations. In particular, removing VLM coordinates does not implement a general tangent/curved-region solver, recover an unknown center automatically, or reproduce the original layout.

New semantic responses compile directly into typed `ParseGraph` objects. They bypass the legacy shape-normalization, pixel-localization and text-claim supplementation path. The compiler preserves the raw semantic payload, statement sources, evidence basis and uncalibrated confidence; it does not silently add omitted text conditions. Recorded legacy model shapes remain supported by the existing decoding path with `metadata.normalization_audit`. Incomplete measurements preserve useful topology, relations, and labels for a non-exact seed. Exact-coordinate recovery remains a later, explicitly registered completion capability.

The proposal contract is deliberately structure-first:

- declare visible points and support objects before referring to them;
- use explicit endpoint references for lines, segments, rays, circles, arcs, polygons, and regions;
- compile a shorthand such as `PQ` into a named segment only when `P` and `Q` are already declared points;
- keep layout, marker, label, and display hints as metadata rather than silently turning them into mathematical facts;
- preserve unresolved or unsupported semantic claims as candidate outcomes instead of dropping the whole seed.

For model-originated `image_text` structural candidates, admission does not require pixel localization when the proposal has source-document evidence. Such candidates remain hypotheses and are checked by `SeedRealizer` and state verification. OpenCV/OCR evidence is optional corroboration and local measurement; it is not the semantic authority and cannot replace the VLM's object and relation graph.

The modules have one responsibility each:

- `contracts.py`: public input and proposal-record boundaries.
- `prediction.py`: provider-neutral, checksum-bound external prediction.
- `openrouter.py`: one VLM transport and strict ParseGraph decoding path.
- `semantic_compiler.py`: coordinate-free geometric statements to the existing graph contract.
- `pipeline.py`: extraction-to-realization orchestration without repair or retry policy.
- `vision/`: OpenCV primitive and marker observations.
- `ocr.py`: checksum-bound OCR evidence and optional Tesseract execution.
- `constraint_completion.py`: explicit registered exact-completion strategies.
- `seed_realizer.py`: observation, semantic-seed, and exact realization results.
- `geosym_world.verifier.parse_graph`: candidate admission without persistent-state mutation.

## Evidence Levels

The pipeline keeps observation, proposal, admitted structure, realization, and exact completion distinct. A successful render is not evidence that the source diagram was reconstructed correctly. A non-exact Geometry State is useful for controlled synthesis but is not source-coordinate truth.

Candidate realizations carry `state_scope=candidate_instance`, `source_facts_admitted=false`, and `selected_assumption_ids`. Construction operations and their materialized objects carry `parser_state_scope`, `parser_input_status`, and the source ParseGraph checksum. These relations describe a conditional instance, not newly admitted facts about the source. The original ParseGraph and admission report remain unchanged. Targets are queries, not constraints supplied to the numeric solver.

Every proposal candidate receives an admission decision. Relations and quantities additionally receive a realization outcome: `satisfied`, `violated`, or `unsupported`. These are exposed through `ParserPipelineResult.candidate_lifecycle`; absence from the state is never treated as implicit success.

Midpoint relations accept a point and segment support, or a point and two explicit endpoints. Solving, materialization and verification use the same endpoints. Unsupported relation checks return failure, and a relation cannot receive a satisfied outcome without a corresponding State relation for independent verification. Numeric midpoint verification uses the declared witness tolerance; exact verification remains symbolic.

All targets are preserved in realization metadata as `task_targets`, with separate `target_outcomes`. A single supported value query (segment length, arc length, area or perimeter on the corresponding materialized object) is written using `set_target` into the State and its replayable trace. `query` means `value`, not `proof`; an unknown mode is a compilation error. Declared region IDs are valid target references. Unsupported targets, multiple targets, unresolved area-ratio notation and unsupported measurement roles remain preserved with an explicit reason; no target is added to the optimizer's assumptions. Binding a target does not assert that the source problem was reconstructed or solved.

All successful and partial seeds are materialized through `ConstructionBuilder` as `FormalGeometryState`. Rendering consumes that state only. Numeric seed witnesses carry an explicit tolerance and residual report; exact states continue to use strict symbolic verification.

Complete observed layouts remain the initialization even when relations require adjustment. The existing numeric witness solver jointly adjusts incidence, collinearity, direction, midpoint, equal-length and numeric length conditions rather than replacing the entire layout. Unsupported conditions remain explicit outcomes; convergence is not a proof of source reconstruction or uniqueness. Numeric equal-length verification uses the declared witness tolerance and records residuals; exact verification remains symbolic.

Observed points and pixel circle radii share one `layout_scale`, estimated from explicit supported measurements. Typed radius and diameter quantities supply mathematical circle sizes. Explicit `point_on_circle` relations and arc endpoints enter the radial constraint solve. A center/start/end arc is materialized on a hidden support circle; visible bounds follow its directed sweep. Numerical arcs require explicitly declared witness tolerance (at most `1e-5`), while exact arc validation remains strict. A legacy `length(circle)` is not silently reinterpreted as radius or circumference. Generic tangency solving is still unsupported.

Before numerical optimization, structural checks detect perpendicular nonzero segments forced onto one carrier and conflicting same-object length/radius/diameter measurements. These cases return `inconsistent_structural_constraints`, include candidate IDs in `constraint_conflicts`, and produce no seed. This is a limited contradiction check, not a complete satisfiability procedure.

Parser redraws disable inferred markers. Materialized equal-length groups receive distinct tick groups; right-angle symbols face the defining segment arms. Parallel arrows require explicit `show_marker=true`. Polygon and region fills require `shaded=true`; line-style hints pass through State metadata. Image-origin quantities, explicit `show_label`, and matching image display annotations control numeric labels; a text-only given remains a constraint without automatically being printed on the diagram. These display policies do not admit source facts or establish source-image fidelity.

## Quality Gates

Pipeline result and quality report schema version `v3` distinguish drawable structure from semantic completeness and unassessed evidence. `ParserPipelineResult.quality` reports these stages:

- `payload_compiled`: the external response compiled into a checksum-valid `ParseGraph`;
- `skeleton_built`: a Geometry State containing the drawable structural skeleton was materialized;
- `skeleton_redraw`: a candidate scene was produced from that state;
- `semantic_graph_complete`: `false` when graph integrity fails, otherwise `null` until source coverage is independently assessed; `metrics.graph_integrity_passed` separately records internal normalization/reference integrity;
- `seed_realized`: a `FormalGeometryState` and render scene were materialized;
- `constraint_satisfied`: a valid candidate exists and every input relation and quantity has a satisfied outcome; missing outcomes block success, and target queries are excluded from assumptions;
- `target_bound`: every declared target is bound to a materialized supported object, or the graph declares no target;
- `pixel_topology_passed`: the optional input-proposal pixel verifier passed/failed, or `null` when not assessed; this does not check the final redraw;
- `source_reconstruction_passed`: `null` in the automatic pipeline because no independent source-reference or human reconstruction assessment is supplied. It is never inferred from the other gates.

Aggregators must count `null` as unassessed, separately from both success and failure. The denominator for candidate constraint coverage is the full input graph, not the admitted subset. A valid candidate may still omit unsupported objects, regions, or targets; it must not be presented as full source recovery.

`realization.metadata.materialization` and quality metrics report missing object, region and target IDs separately. Text and marker observations are excluded from mathematical-object coverage. If a State exists but this coverage is incomplete, the result is `partial`, even when every implemented constraint passes. `constraint_satisfied` stays independent of whether the query is supported. Neither materialization completeness nor `realized` measures fidelity to the original image.

When a model names a segment, region, or target that it did not declare, the adapter creates an `unknown` placeholder with `requires_grounding=true` and records it in `metadata.normalization_audit.unresolved_references`. This keeps the ParseGraph serializable and reviewable without silently fabricating geometry; admission and realization may still abstain.

`state_valid` is intentionally not one of these gates. It only means that the generated state obeys the internal Geometry State contract. A valid state can still be an incomplete or semantically wrong subgraph of the source image. When parsing fails before a `ParserPipelineResult` exists, the caller must record `payload_compiled=false` and the exception separately.

## Capability Reporting

`parser_capability_report()` reports representation, extraction, realization, and verification independently. Extraction remains `partial` until measured on a held-out image set. Explicit circles and arcs have partial realization support; generic tangency and full curved-region relation solving remain deferred.

## Current Exact Strategies

- complete explicit exact 2D point coordinates;
- midpoint, segment-ratio point, and perpendicular-foot derivation from exact scaffolds;
- a two-length perpendicular chain;
- a named rectangle with adjacent side lengths, perpendicular extension, and collinear intersections;
- SSS triangle completion with both mirror branches.

Unsupported constraint families return an explicit diagnostic. The deleted canonical fallback no longer assigns arbitrary coordinates to otherwise unsupported graphs.

Intersection completion accepts collinearity, explicit point-on-line/segment/ray incidence, and ordered points as equivalent ways to identify the carrier lines. The resulting candidate still undergoes independent domain checks; computing the carrier-line intersection does not establish that it lies inside a requested segment.

## Python API

```python
from functools import partial

from geosym_world.parser import ParserPipeline, parse_one_openrouter

extractor = partial(parse_one_openrouter, model=model, api_key=api_key)
result = ParserPipeline(extractor).run(parser_input)
payload = result.to_dict()
```

Pass `pixel_topology_verifier=PixelTopologyVerifier()` for recorded predictions that contain image geometry. For `geometry_semantics_v1`, the pipeline does not invoke that verifier even when supplied: there are no pixel observations to verify. Quality reports `pixel_topology_not_applicable`, a null pixel gate, and `metrics.pixel_topology_applicable=false`; this is not a failed reconstruction. The report is independent of candidate realization. The pipeline does not silently enable OCR, OpenCV or retries.

## Commands

Run the semantic pipeline through extraction, typed compilation, candidate realization, verification and rendering:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.parse_openrouter_v0 \
  --input artifacts/parser/datasets/prepared_v0/parser-input.jsonl \
  --run-dir artifacts/parser/evaluations/<new_run_id> \
  --model qwen/qwen3.7-flash --max-records 8 --progress-every 1
```

`--run-dir` must be new. It contains `json/<id>.json` (all pipeline stages, semantic payload, State and trace), `json/raw/<id>.json` (decoded VLM response saved before compilation), `images/<id>/input.png`, an optional `redraw.png`, `summary.json`, `manifest.json` and an automatically generated `contact.png`. Failed samples retain their input image, available raw response and typed error; missing redraws remain visible in the sheet. Empty optional unit strings compile as an unspecified unit. `--max-records` limits attempted inputs including failures, not the number of successes. There are no retries. Manifest records model, prompt version/hash, relevant code hashes, input hash, command, environment executable and deterministic run settings. Summary records image hashes. `--output` remains a separate proposal-only mode and cannot be combined with `--run-dir`.

Build a compact original-versus-candidate review sheet after a Parser run:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.build_parser_review_sheet \
  --run-dir artifacts/parser/evaluations/<run_id>
```

The command writes `contact.png` at the run root. Each cell contains the original image, the candidate State redraw when available, status, object/relation/quantity counts, and candidate constraint gates. An absent redraw is shown explicitly. Batch Parser runners should call the same function after writing `summary.json`; the sheet is an audit view and does not change evaluation metrics.

Prepare public inputs:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.prepare_parser_inputs_v0 \
  --catalog artifacts/parser/sources/catalog.jsonl \
  --output-dir artifacts/parser/datasets/prepared_v0
```

Extract proposal-only ParseGraphs through the configured VLM provider:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.parse_openrouter_v0 \
  --input artifacts/parser/datasets/prepared_v0/parser-input.jsonl \
  --output artifacts/parser/proposals/predictions.jsonl \
  --model qwen/qwen3.7-flash
```

Evaluate predictions against separate references:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.evaluate_parse_graph_v0 \
  --predictions artifacts/parser/proposals/predictions.jsonl \
  --references artifacts/parser/references/parse-graph.jsonl \
  --output artifacts/parser/evaluations/metrics.jsonl
```

## Removed Legacy Paths

The cleanup removed the parallel synthetic/clean/typed/fused Parser implementations, Hybrid instruction-repair pipeline, arbitrary canonical builder, duplicate reconstruction/redraw workflow, Clause v1 IR, Geoparsing compatibility adapter, and their dedicated CLIs. They are not compatibility-supported APIs.

Current implementation work continues with linear-geometry relation closure and generic semantic realization. Circle, arc, and region objects can now survive into non-exact seeds, while their full metric relation closure remains an explicit later capability.
