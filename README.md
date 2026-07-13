# GeoSym World Geometry

Chinese documentation: `README_ZH.md`.

This directory is an isolated prototype for the next GeoSym-Hard / GeoSym-Debug geometry world.

GeoClaw is not positioned as a collection of calculator, zoom, measurement, and auxiliary-line features. It models multimodal geometry solving as an observable and interactive tool-use trajectory in which geometry-bound actions and intermediate claims receive solver-grounded feedback. The primary research objects are tool routing, argument grounding, action utility, evidence strength, and recovery from invalid steps.

The current implementation focuses on:

- replayable construction traces;
- exact SymPy-backed 2D and small 3D geometry states;
- first-class interfaces for 3D points, planes, and polyhedra;
- circle, arc, chord, tangent, polygon region, and sector region primitives;
- coordinate-backed fact closure for lengths, angles, polygon areas, circle metrics, arc lengths, sector areas, incidence, parallelism, perpendicularity, midpoint facts, angle-bisector facts, and target lookup;
- theorem-lite checks for right triangles, isosceles triangles, similar triangles, angle bisectors, circle chords, inscribed angles, and tangent-chord relations;
- 3D facts for segment lengths, plane/polyhedron existence, point-on-plane, line-plane relations, plane-plane relations, projection residuals, dihedral angles, polyhedron surface area, polyhedron volume, and polyhedron-plane sections;
- compact export records for future benchmark and diagnosis pipelines.

Schema reference:

```text
geosym_world_geometry/docs/FORMAL_SCHEMA.md
```

Current package entry points:

```python
from geosym_world_geometry import ConstructionBuilder, FactClosureSolver
from geosym_world_geometry import TraceVerifier

report = TraceVerifier().verify_trace(trace)
state = ConstructionBuilder().execute(trace)
facts = FactClosureSolver().solve(state)
```

GeoClaw agent environment:

```python
from geosym_world_geometry.geoclaw import GeoClawEnvironment
from geosym_world_geometry.geoclaw.datasets import pilot_task

environment = GeoClawEnvironment(pilot_task("construct_midpoint"))
observation = environment.step({
    "action": "construct",
    "arguments": {"operation": "midpoint", "id": "M", "params": {"of": ["A", "B"]}},
})
```

See `docs/GEOCLAW_QUICK_START.md` for the repository guide, action protocol, metrics, and commands.

GeoClaw task and trajectory schemas additionally support:

- annotated visual regions for model-selected versus oracle zoom evaluation;
- structured difficulty profiles covering visual complexity, construction depth, theorem depth, and symbolic growth;
- calculation provenance linking expressions to solver facts;
- evidence-strength labels separating exact, solver-derived, numeric, and counterexample evidence;
- private audit fields for visual-versus-solver measurement residuals without leaking oracle data to the Agent;
- trace metrics for expression support, zoom grounding, measurement reliance, and construction utility.

Rendering entry points:

```python
from geosym_world_geometry.rendering import state_to_render_scene, write_png, write_svg

scene = state_to_render_scene(state)
write_svg(scene, "geosym_world_geometry/outputs/example.svg")
write_png(scene, "geosym_world_geometry/outputs/example.png")
```

Sample validation:

```bash
python -m geosym_world_geometry.examples.sample_trace
python -m geosym_world_geometry.examples.sample_circle_region
python -m geosym_world_geometry.examples.sample_adapter_verifier
python -m geosym_world_geometry.examples.sample_theorem_lite
python -m geosym_world_geometry.tools.smoke_check
python -m geosym_world_geometry.tools.render_smoke_check
python -m geosym_world_geometry.tools.render_pilot --output-dir geosym_world_geometry/outputs/rendered_pilot --format both --max-samples 5
python -m geosym_world_geometry.tools.synthesize_records --output geosym_world_geometry/outputs/synthetic_v1/synthetic_v1.jsonl --max-records 100 --per-family 20 --mutations-per-variant 2 --format both --seed 23
python -m geosym_world_geometry.tools.synthesize_records --preset large-500 --seed 23
python -m geosym_world_geometry.tools.export_pilot --output geosym_world_geometry/outputs/pilot_v0.jsonl
python -m geosym_world_geometry.tools.adapt_legacy_json --input path/to/legacy.json --output geosym_world_geometry/outputs/adapted_legacy_record.json
python -m geosym_world_geometry.tools.adapt_legacy_json --input path/to/legacy.jsonl --input-jsonl --line-index 0 --target-type length --target-object L0 --output geosym_world_geometry/outputs/adapted_legacy_record.json
python -m geosym_world_geometry.tools.run_geoclaw_pilot --list-tasks
python -m geosym_world_geometry.tools.run_geoclaw_pilot --task construct_midpoint --output-dir geosym_world_geometry/outputs/geoclaw_demo
python -m geosym_world_geometry.tools.run_geoclaw_batch --baseline oracle-final --output-dir geosym_world_geometry/outputs/geoclaw_batch
```

Pilot status:

- `tools/smoke_check.py` runs in memory and checks all built-in pilot records, selected exact target values, and required theorem-family coverage.
- `tools/render_smoke_check.py` renders five pilot samples to SVG and OpenCV PNG, then checks that the files are valid and nonblank.
- `outputs/pilot_v0.jsonl` is the current pilot export.
- `outputs/pilot_v0.jsonl.manifest.json` records aggregate coverage.
- `outputs/render_smoke/` contains the small renderer smoke outputs.
- `tools/synthesize_records.py` runs the procedural template-family sampler, rejection filters, deduplication, renderer, and image+QA JSONL exporter.
- `tools/synthesize_records.py --preset large-500` is the maintained staged 500-PNG generation preset. It prioritizes distinct 2D base specs across seven families, then writes base specs, enhanced specs, full records, QA JSONL, GT JSONL, images, and a manifest under `outputs/synthetic_500/`.

The rendering surface is intentionally conservative in this phase. SVG is the primary audit/debug format, while PNG is produced directly with OpenCV from the same `RenderScene`. The 3D renderer currently uses a fixed orthographic projection. CAD validation, hidden-line rendering, analytic curved solids, and complete degenerate section handling are reserved for later phases.

Renderer v0 currently includes viewport clipping for infinite lines/rays, target-aware 2D/3D filtering, dashed auxiliary lines in both SVG and PNG, text halos, compact display labels for generated objects, and lightweight geometry markers for midpoint ticks, angle arcs, and right angles.

Perpendicular-foot relations now render the auxiliary perpendicular segment in addition to the right-angle marker. Polyhedron edges use an approximate fixed-view hidden-line pass; back-facing edges are rendered dashed in SVG and OpenCV PNG.

Synthesis v0 uses template families rather than hand-written one-off templates. Each family exposes a parameter grid and optional deterministic mutator, then runs trace verification, target-fact rejection, duplicate filtering, rendering, and JSONL export. See `docs/SYNTHESIS_ENGINE.md`.
