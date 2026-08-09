# Geosym-World

Geosym-World is the geometry data and computation foundation of the GeoSym workspace. It owns the versioned geometry-state contract, deterministic 2D/3D geometry, construction, solving, verification, rendering, synthesis, parsing, dataset adapters, and component benchmarks.

GeoClaw is a separate repository and depends on this package. Geosym-World must not import Agent runtime, model-provider, policy, or trajectory code from GeoClaw.

## Current Scope

- exact SymPy-backed 2D geometry state and construction traces;
- a minimal 3D kernel for points, planes, polyhedra, sections, and measurements;
- fact closure and trace verification;
- SVG and PNG rendering from one shared `RenderScene`;
- angle, right-angle, perpendicular-foot, extension-line, exact-expression, and collision-aware label annotations;
- parameterized synthetic generation with mutation, staged enhancement, deduplication, exact-count enforcement, QA/GT sidecars, manifests, and progress reporting;
- frozen compatibility support for the former `scripts/tools` synthesis surface;
- parser prototypes that write confidence- and provenance-bearing hypotheses;
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

Run the maintained staged 500-record preset:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.synthesize_records \
  --preset large-500 --format png --workers 4 --require-exact-count
```

The preset prints progress every 25 accepted/exported records and defaults to `artifacts/generated/synthetic_500/records.jsonl`.

Reuse accepted enhanced specs without rerunning sampling and enhancement:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.synthesize_records \
  --input-enhanced artifacts/generated/synthetic_500/json/enhanced/records.enhanced.jsonl \
  --output artifacts/generated/synthetic_500_both/records.jsonl \
  --max-records 500 --format both --workers 4 --require-exact-count
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

See `docs/FORMAL_SCHEMA.md`, `docs/SYNTHESIS_ENGINE.md`, and `docs/LEGACY_TOOL_COVERAGE.md` for detailed contracts.
