# Original `scripts/tools` Compatibility

The package `geosym_world.legacy` exposes the original implementations directly and adds a deterministic orchestrator around them. It therefore preserves the complete class method surface and legacy JSON schema while allowing individual stages and record counts to be controlled from one CLI.

## Coverage Matrix

| Original module | Complete compatibility surface | Optimized orchestration |
|---|---|---|
| `template.py` | six base-shape families, all derivation rules, exact legacy export | deterministic per-record seeds and killable process isolation |
| `builder.py` | all four random operations, automatic line/arc intersections, finalization | ordered process workers, configurable rounds, hard timeout isolation |
| `drawer.py` | legacy canvas, bounds, line/arc drawing and single/batch APIs | stable artifact paths and stage reuse |
| `region.py` | connected-region extraction, contour resampling, curvature and arc detection | available directly and through shader stage |
| `shader.py` | point/line/perpendicular annotation, contour-to-primitive matching, region selection | stage control, deterministic NumPy/Python seeds, configurable attempts |
| `shaders.py` | hatch, crosshatch, solid and gradient fills | all strategies available by config |
| `gt.py` | line/arc lengths, mixed-boundary perimeter and area, shadow GT | ordered process workers and stable stage artifacts |
| `qa_template.py` | length, angle, perimeter, entity area, shadow area, both ratios, all-shadow sum | parameterized type set, weights and question count |
| `utils.py` | MathD/SymPy/VERL grading, boxed-answer extraction and normalization | re-exported through the compatibility package |

Runtime evidence is available without generating data:

```bash
cd /data/Geosym127K/Geosym-World
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.run_legacy_synthesis --compatibility-report
```

Small complete pipeline sample:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.run_legacy_synthesis \
  --config configs/legacy/config_hard.json \
  --output-root artifacts/generated/legacy_review \
  --records 1 \
  --rounds-json '{"0":1,"1":1}' \
  --shader-attempts 1 \
  --questions-per-record 2 \
  --workers 1
```

Use `--stages` and `--input-base`, `--input-enhanced`, `--input-shaded`, or `--input-gt` to resume at any stage. No generation occurs on import.

Each generation, GT, and QA item has a 300-second process-level timeout by default. Unlike the original daemon-thread timeout, a timed-out calculation is terminated and cannot continue consuming CPU in the background. Use `--task-timeout 0` for a reusable process pool without hard timeouts. On Python 3.11 or newer, `--max-tasks-per-child` optionally controls worker recycling in that mode; Python 3.10 rejects that option explicitly.

Keep `--workers 1` as the baseline for `config_hard.json`: its SymPy-heavy derivations can be CPU-contention bound, so more workers may reduce rather than improve throughput. Benchmark the target config and machine before increasing concurrency.
