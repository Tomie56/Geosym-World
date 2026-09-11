# Geosym-World

Geosym-World is the geometry data and computation foundation of the GeoSym workspace. It owns the versioned geometry-state contract, deterministic 2D/3D geometry, construction, solving, verification, rendering, synthesis, parsing, dataset adapters, and component benchmarks.

GeoClaw is a separate repository and depends on this package. Geosym-World must not import Agent runtime, model-provider, policy, or trajectory code from GeoClaw.

## Current Scope

- exact SymPy-backed 2D geometry state and construction traces;
- a minimal 3D kernel for points, planes, polyhedra, sections, and measurements;
- fact closure and trace verification;
- SVG and PNG rendering from one shared `RenderScene`;
- angle, right-angle, perpendicular-foot, extension-line, exact-expression, and collision-aware label annotations;
- parameterized synthetic generation with exact geometric preflight, target-dependency auditing, deduplication, exact-count enforcement, QA/GT and Parser sidecars, manifests, and progress reporting;
- value, proof, and constraint targets with validated derivations and alternative solution paths;
- versioned difficulty features and classified failed-candidate diagnostics;
- frozen compatibility support for the former `scripts/tools` synthesis surface;
- a staged OpenCV clean-diagram Parser v0 that writes confidence- and provenance-bearing visual hypotheses without reading formal state;
- a shared parser-record contract that keeps real and synthetic candidates proposed until explicit admission;
- deterministic Parser benchmark slicing by parser, task type, difficulty, and diagnostic axis;
- dataset adapters plus benchmark and diagnosis foundations.

Real-world diagram parsing, broad 3D parsing, complete theorem proving, and CAD-grade rendering remain progressive research goals. Parser output is not admitted as persistent fact without explicit verification.

## Layout

```text
src/geosym_world/
  schema/          Geometry State Contract v1
  kernel/          2D and 3D geometry kernels
  construction/    Typed construction execution
  solver/          Fact closure
  verifier/        Trace and state verification
  rendering/       Shared SVG/PNG scene pipeline
  synthesis/       Sampling, enhancement, export
  parser/          Structured and visual parser prototypes
  datasets/        Layout, adapters, records, manifests
  benchmark/       Component evaluation and diagnosis
  legacy_tools/    Frozen former scripts/tools implementation
  legacy/          Compatibility facade and runner
  tools/           Supported command-line entry points
artifacts/          Local data and generated outputs, ignored by Git
configs/legacy/     Maintained legacy synthesis configurations
tests/              Focused regression tests
```

## Environment

All development and tests use the `geosym` Conda environment:

```bash
cd /data/Geosym127K/Geosym-World
/data/miniconda3/envs/geosym/bin/python -m pip install -e '.[test]'
/data/miniconda3/envs/geosym/bin/python -m pytest -q
```

Set `GEOSYM_WORLD_ARTIFACTS` to override the default `artifacts/` root.

## Core API

```python
from geosym_world import ConstructionBuilder, FactClosureSolver, TraceVerifier

verification = TraceVerifier().verify_trace(trace)
state = ConstructionBuilder().execute(trace)
facts = FactClosureSolver().solve(state)
```

Legacy synthesis imports remain available under an explicit compatibility namespace:

```python
from geosym_world.legacy import EnhancedDrawer, GeometryCalculator
from geosym_world.legacy import QAGenerator, RandomGeometryBuilder, TemplateGenerator
```

## Commands

List available synthesis families:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.synthesize_records --list-families
```

Generate a small review set:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.synthesize_records \
  --output artifacts/generated/review/records.jsonl \
  --max-records 6 --per-family 1 --no-mutation \
  --format png --workers 4 --scene-payload metadata --seed 23
```

Annotation output is parameterized independently from geometry generation:

```bash
# Point names only, with all relation symbols suppressed.
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.synthesize_records \
  --annotation-level points --no-relation-markers

# Point names plus public numeric/relation-value labels and relation symbols.
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.synthesize_records \
  --annotation-level givens --relation-markers
```

`--relation-markers` is the single switch for right-angle, parallel, equal-length, and angle-bisector symbols. It does not remove the underlying auxiliary geometry.

Run the maintained staged 500-record preset:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.synthesize_records \
  --preset large-500 --format png --workers 4 --require-exact-count
```

The preset prints progress every 25 accepted/exported records and defaults to `artifacts/generated/synthetic_500/records.jsonl`.
Every CLI export receives a deterministic topology-aware train/dev/test label, so parameter variants of one task topology cannot cross splits. Use `--split-group-mode lineage` for lineage-only grouping, and override the default 0.8/0.1/0.1 ratios with `--split-ratios-json`.

Reuse accepted enhanced specs without rerunning sampling and enhancement:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.synthesize_records \
  --input-enhanced artifacts/generated/synthetic_500/json/enhanced/records.enhanced.jsonl \
  --output artifacts/generated/synthetic_500_both/records.jsonl \
  --max-records 500 --format both --workers 4 --require-exact-count
```

Run the deterministic synthetic Parser v0 closed loop:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.parse_synthetic_v0 \
  --input artifacts/generated/review/records.jsonl \
  --output artifacts/parser/synthetic_v0/review.jsonl \
  --admit --progress-every 25
```

The parser output keeps observations and proposed hypotheses separate from admitted state. It is a synthetic contract/debug harness; it does not represent open-world image parsing.

Image-only Parser inputs use the strict `ParserInput` contract: `schema_version`, `id`, `image`, and `question` are the only allowed runtime fields. Image exports additionally write `*.parser-input.jsonl`, legacy clean-parser `*.parser-reference.jsonl`, typed evaluator-only `*.typed-parser-reference.jsonl`, and private `*.parser-source-audit.jsonl`. Synthetic Parser inputs use opaque sample IDs and content-addressed PNG names even when the requested dataset image format is SVG; template family, parameters, formal state, scene metadata, facts, derivation, construction trace, and answer never enter the Parser input. `TypedDiagramParserV0` covers OpenCV line/circle/fill candidates, visual topology, explicit relation glyphs, topology-bounded equal subsegments and ordered angles, conservative one-to-one point-name grounding, safe explicit segment-length assertions, relation hypotheses, and explicit segment targets. Synthetic references and evaluation expose marker detection as a separate component while preserving the original five-core macro score. Typed Parser v0 sidecars include only 2D render scenes; 3D records remain in the main export and are counted under classified Parser skip reasons in the manifest. No OCR dependency is bundled; checksum-bound external OCR observations can be replayed with `--ocr-predictions` and still pass through grounding and admission.

Prepare opaque public inputs from a private real-image source catalog:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.prepare_parser_inputs_v0 \
  --catalog artifacts/parser/sources/catalog.jsonl \
  --output-dir artifacts/parser/datasets/prepared_v0 \
  --progress-every 25
```

The prepared directory separates `parser-input.jsonl` from private `source-audit.jsonl`, copies images to content-addressed opaque names, and records source, protocol, license, split group, and file hashes in `manifest.json`.

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.parse_images_v0 \
  --input artifacts/generated/review/records.jsonl.parser-input.jsonl \
  --output artifacts/parser/clean_v0/results.jsonl \
  --progress-every 25
```

Run the typed proposal, evidence-gated admission, graded reconstruction, and redraw workflow:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.parse_reconstruct_v0 \
  --input artifacts/generated/review/records.jsonl.parser-input.jsonl \
  --output artifacts/parser/verified_v0/results.jsonl \
  --reconstruction-mode pixel_witness --format both --progress-every 25
```

Replay a fixed provider-generated ParseGraph JSONL through the same workflow by adding:

```bash
  --external-predictions artifacts/parser/model_runs/run_001/external-predictions.jsonl
```

External predictions remain proposals and are rejected if they claim native/human/verified provenance or contain private benchmark fields. The run manifest records model, revision, prompt, decoding, usage, request IDs, and file hashes; Geosym-World does not call a model in this mode.

Replay checksum-bound region OCR while retaining the built-in visual detector by adding:

```bash
  --ocr-predictions artifacts/parser/ocr_runs/run_001/ocr-predictions.jsonl
```

Import checksum-bound GDP-29K/Geoparsing formal descriptions as global hypotheses by adding:

```bash
  --geoparsing-predictions artifacts/parser/geoparsing/run_001/predictions.jsonl
```

Geoparsing import supports its published 2D/3D primitives and core semantic expressions, but never invents coordinates, localized evidence, or confidence. Unsupported syntax fails strictly unless `--allow-partial-geoparsing` is explicit. Add `--fuse-geoparsing-visual` to map uniquely named points and endpoint/topology-supported objects onto the deterministic visual ParseGraph; this may be combined with `--ocr-predictions`. Fusion preserves the semantic candidates' original evidence precision and cannot bypass admission. Full external ParseGraph predictions remain mutually exclusive with OCR and Geoparsing input.

Use `python -m geosym_world.tools.audit_geoparsing_grammar_v0` for an image-free, model-free audit of GDP-style formal-description JSONL. The report measures adapter syntax coverage and diagnostics only, not Parser accuracy.

`pixel_witness` records an observed-layout realization and never promotes pixel measurements to source assertions. `exact` requires explicit safe expression trees for coordinates and refuses missing exact geometry. Constraint-preserving resynthesis is available only for exact reconstructions.

Evaluate typed predictions against the separate private reference:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.evaluate_parse_graph_v0 \
  --predictions artifacts/parser/verified_v0/results.jsonl \
  --references artifacts/generated/review/records.jsonl.typed-parser-reference.jsonl \
  --output artifacts/benchmark/parse_graph_v0/metrics.jsonl
```

Diagnose Parser v0 and aggregate component slices:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.diagnose_parser_v0 \
  --input artifacts/generated/review/records.jsonl \
  --output artifacts/parser/synthetic_v0/diagnosis.jsonl \
  --admit --progress-every 25
```

Render pilot scenes and run in-memory checks:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.smoke_check
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.render_smoke_check
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.render_pilot --format both --max-samples 5
```

Run the frozen legacy pipeline:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.run_legacy_synthesis \
  --config configs/legacy/config_easy.json --records 1 --disable-shader
```

See `docs/PARSER_PIPELINE.md`, `docs/FORMAL_SCHEMA.md`, `docs/SYNTHESIS_ENGINE.md`, and `docs/LEGACY_TOOL_COVERAGE.md` for detailed contracts.
