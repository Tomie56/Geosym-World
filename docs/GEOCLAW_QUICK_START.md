# GeoClaw Repository Guide and Quick Start

## What GeoClaw Is

GeoClaw is the agent-facing layer of `geosym_world_geometry`. It models geometry solving as an observable, interactive, and step-verifiable tool-use trajectory.

The contribution is not the presence of individual tools. Calculator use, visual cropping, sketchpads, auxiliary constructions, and symbolic solvers all have strong precedents. GeoClaw instead defines geometry-bound action semantics, solver-verified feedback, replayable traces, and metrics for whether tool actions are correctly selected, grounded, useful, and recoverable.

```text
geometry trace -> exact state -> fact closure
               -> GeoClaw environment
               -> tool action + verifier feedback + sketch image
               -> trajectory metrics
```

It reuses the existing SymPy-backed builder, solver, verifier, and renderer. GeoClaw adds a JSON action protocol, private/public state isolation, six tools, transactional construction, trace logging, and behavioral scoring.

## Repository Map

```text
geoclaw/
├── schema.py          # actions, observations, feedback
├── task.py            # benchmark task schema
├── environment.py     # interactive episode loop
├── resolver.py        # object reference resolution
├── utility.py         # progress/redundant/harmful
├── trajectory.py      # replayable logs
├── scoring.py         # tool-use metrics
├── tools/             # zoom/calculate/measure/construct/verify/final
├── datasets/pilot.py  # ten deterministic pilot tasks
└── runners/           # replay and scoring helpers
```

CLI and examples:

```text
tools/run_geoclaw_pilot.py
tools/score_geoclaw_trajectory.py
examples/sample_geoclaw_episode.py
tests/test_geoclaw_*.py
```

## State Isolation

Private environment state contains the complete geometry state, exact fact closure, target, answer, and oracle annotations. The model-facing observation contains only the problem, current image, crops, visible object IDs, action history, optional verifier feedback, and remaining budget.

Trajectory steps may contain a private `audit` payload for oracle residuals and diagnostic labels. `trajectory.to_dict(include_private=False)` removes private state summaries and audit fields.

## Research Schemas

- `VisualRegion` describes oracle regions for small labels, dense intersections, occluded angle marks, or ambiguous shaded boundaries. Zoom results report IoU and coverage against these annotations.
- `DifficultyProfile` separates education/competition level from visual complexity, construction depth, theorem depth, and symbolic expression growth.
- `EvidenceRecord` uses explicit evidence levels such as `symbolic_proved`, `exact_coordinate`, `solver_derived`, `numeric_verified`, and `counterexample_found`.

## Action Protocol

```json
{
  "thought": "Construct the midpoint of AB.",
  "action": "construct",
  "arguments": {
    "operation": "midpoint",
    "id": "M",
    "params": {"of": ["A", "B"]}
  }
}
```

| Action | Behavior |
|---|---|
| `zoom` | Crop and enlarge a normalized image region. |
| `calculate` | Evaluate a restricted symbolic SymPy expression. |
| `measure` | Return image-style rounded or oracle exact measurements. |
| `construct` | Add a verified auxiliary construction transactionally. |
| `verify` | Check a typed claim with true/false/unknown output. |
| `final` | Submit an answer and terminate the episode. |

Construction failures do not mutate the committed state or Agent trace. Successful constructions are rendered into the next step image, which serves as the auditable sketchpad.

Calculator calls may include supporting `fact_ids`. Returned provenance records valid and missing facts and whether the expression has solver-backed geometry support.

Image-mode length and distance measurement uses the rendered viewport and reports a pixel value. The exact solver value and visual-solver residual remain in the private audit channel.

## Quick Start

Run from the repository parent:

```bash
cd /data/Geosym127K
export PYTHONPATH=/data/Geosym127K
```

List pilot tasks:

```bash
python -m geosym_world_geometry.tools.run_geoclaw_pilot --list-tasks
```

Run the default midpoint episode:

```bash
python -m geosym_world_geometry.tools.run_geoclaw_pilot \
  --task construct_midpoint \
  --output-dir geosym_world_geometry/outputs/geoclaw_demo
```

This writes `trajectory.json`, step PNGs, and any crop images below `outputs/geoclaw_demo/construct_midpoint/`.

Score a stored trajectory:

```bash
python -m geosym_world_geometry.tools.score_geoclaw_trajectory \
  --task construct_midpoint \
  --trajectory geosym_world_geometry/outputs/geoclaw_demo/construct_midpoint/trajectory.json
```

Replay custom JSONL actions:

```json
{"action":"construct","arguments":{"operation":"midpoint","id":"M","params":{"of":["A","B"]}}}
{"action":"verify","arguments":{"claim":{"type":"midpoint","object":"M"}}}
{"action":"final","arguments":{"answer":"4"}}
```

```bash
python -m geosym_world_geometry.tools.run_geoclaw_pilot \
  --task construct_midpoint --actions /path/to/actions.jsonl
```

Programmatic use:

```python
from geosym_world_geometry.geoclaw import GeoClawEnvironment
from geosym_world_geometry.geoclaw.datasets import pilot_task

environment = GeoClawEnvironment(pilot_task("construct_midpoint"))
observation = environment.step({
    "action": "construct",
    "arguments": {
        "operation": "midpoint",
        "id": "M",
        "params": {"of": ["A", "B"]},
    },
})
```

Use `expose_feedback=False` to run the hidden-verifier ablation.

Restrict the environment to an ablation tool set:

```python
environment = GeoClawEnvironment(
    task,
    allowed_tools=("calculate", "verify"),
)
```

`final` is always enabled. Disabled actions receive `unsupported_operation` rather than silently disappearing from the action schema.

CLI example:

```bash
python -m geosym_world_geometry.tools.run_geoclaw_pilot \
  --task construct_midpoint \
  --allowed-tools construct,verify
```

Run all ten pilot tasks and write an aggregate manifest:

```bash
python -m geosym_world_geometry.tools.run_geoclaw_batch \
  --baseline oracle-final \
  --output-dir geosym_world_geometry/outputs/geoclaw_batch
```

The output `manifest.json` records experiment configuration, aggregate metrics, per-task termination, and trajectory paths.

## Metrics

`geoclaw/scoring.py` implements final accuracy, tool-selection precision/recall/F1, schema validity, argument validity, invalid action rate, verified step ratio, useful action rate, tool efficiency, recovery rate, tool counts, aggregation, and feedback gain.

It also reports expression validity, geometry-expression support, calculation masking, zoom grounding IoU, required-region coverage, visual measurement rate, visual-solver residual, measurement reliance error, construction validity/utility, and evidence-strength counts.

Required tools are represented as acceptable sets rather than a unique reference sequence. Recovery is measured over failure events and a default three-step repair window.

## Tests

```bash
cd /data/Geosym127K
PYTHONPATH=/data/Geosym127K pytest -q -p no:cacheprovider geosym_world_geometry/tests
```

## Current Boundaries

- Pilot tasks use an oracle formal geometry state.
- The parser currently resolves object identifiers and conservative aliases; it is not OCR or a learned scene parser.
- Length and distance in image mode are measured through the actual `RenderScene` viewport transform and report both `pixel_value` and recovered world-space value. Other metric types still use a rounded geometry-backed proxy.
- Verification uses typed claims rather than unrestricted natural language.
- Utility uses target-related fact growth, not a complete proof-distance planner.
- Model APIs, training, a frontend, and large-scale generation are intentionally outside v0.
