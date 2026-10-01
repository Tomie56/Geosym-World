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
- a contract-first Parser core with strict VLM extraction, optional pixel-topology evidence, candidate admission, state-backed seed realization, and per-candidate lifecycle diagnostics;
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

The Parser currently exposes a small contract-first core. It accepts only public image/question inputs and uses `ParserPipeline` to preserve proposal-only `ParseGraph` predictions, admission decisions, state-backed realization, and verification as separate stages. It does not claim production-grade open-world reconstruction. See `docs/PARSER_PIPELINE.md` for the current capability boundary.

Prepare opaque public inputs from a private real-image source catalog:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.prepare_parser_inputs_v0 \
  --catalog artifacts/parser/sources/catalog.jsonl \
  --output-dir artifacts/parser/datasets/prepared_v0 \
  --progress-every 25
```

The prepared directory separates `parser-input.jsonl` from private `source-audit.jsonl`, copies images to content-addressed opaque names, and records source, protocol, license, split group, and file hashes in `manifest.json`.

Extract strict proposal-only ParseGraphs through the configured VLM provider:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.parse_openrouter_v0 \
  --input artifacts/parser/datasets/prepared_v0/parser-input.jsonl \
  --output artifacts/parser/proposals/predictions.jsonl \
  --model qwen/qwen3.7-flash
```

Evaluate typed predictions against the separate private reference:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.evaluate_parse_graph_v0 \
  --predictions artifacts/parser/proposals/predictions.jsonl \
  --references artifacts/generated/review/records.jsonl.typed-parser-reference.jsonl \
  --output artifacts/benchmark/parse_graph_v0/metrics.jsonl
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
