# Geosym-World

Geosym-World 是 GeoSym 工作区中的几何数据与计算基座，负责版本化几何状态契约、确定性 2D/3D 几何、构造、求解、验证、渲染、合成、Parser、数据适配和组件 Benchmark。

GeoClaw 已拆分为独立仓库并单向依赖本包。Geosym-World 不包含 Agent runtime、模型 provider、策略或轨迹实验实现，也不得反向导入 GeoClaw。

## 当前能力

- 基于 SymPy 精确值的 2D 几何状态和可重放构造轨迹；
- 点、平面、多面体、截面和测量组成的最小 3D 内核；
- 事实闭包、target 查询和构造轨迹验证；
- SVG 与 PNG 共用同一个 `RenderScene`；
- 角标、直角、垂足、延长线、根式/分式表达式和标签避让；
- 参数化合成、扰动、分阶段增强、去重、exact-count、QA/GT sidecar、manifest 和进度输出；
- 对原 `scripts/tools` 全部功能的冻结兼容层；
- 输出 confidence 与 provenance hypothesis 的 Parser 原型；
- 数据适配、组件评测和错误诊断基础。

真实题图通用解析、复杂 3D 解析、完整定理证明和 CAD 级渲染仍属于递进开发目标。Parser 候选未经明确验证，不会直接升级为持久事实。

## 目录

```text
src/geosym_world/
  schema/          Geometry State Contract v1
  kernel/          2D/3D 几何内核
  construction/    typed 构造执行
  solver/          事实闭包
  verifier/        轨迹与状态验证
  rendering/       SVG/PNG 共用渲染管线
  synthesis/       采样、增强和导出
  parser/          结构化与视觉 Parser 原型
  datasets/        数据布局、适配和 manifest
  benchmark/       组件评测与 diagnosis
  legacy_tools/    冻结的原 scripts/tools 实现
  legacy/          兼容 facade 与 runner
  tools/           正式命令入口
artifacts/          本地数据和生成结果，不进入 Git
configs/legacy/     旧合成链的维护配置
tests/              聚焦回归测试
```

## 环境与安装

开发和测试统一使用 `geosym` Conda 环境：

```bash
cd /data/Geosym127K/Geosym-World
/data/miniconda3/envs/geosym/bin/python -m pip install -e '.[test]'
/data/miniconda3/envs/geosym/bin/python -m pytest -q
```

默认产物目录为仓库内的 `artifacts/`。可通过 `GEOSYM_WORLD_ARTIFACTS` 覆盖。

## 核心接口

```python
from geosym_world import ConstructionBuilder, FactClosureSolver, TraceVerifier

verification = TraceVerifier().verify_trace(trace)
state = ConstructionBuilder().execute(trace)
facts = FactClosureSolver().solve(state)
```

旧合成工具通过明确的兼容命名空间继续提供：

```python
from geosym_world.legacy import EnhancedDrawer, GeometryCalculator
from geosym_world.legacy import QAGenerator, RandomGeometryBuilder, TemplateGenerator
```

## 常用命令

查看合成 family：

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.synthesize_records --list-families
```

生成小规模审阅集：

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.synthesize_records \
  --output artifacts/generated/review/records.jsonl \
  --max-records 6 --per-family 1 --no-mutation \
  --format png --workers 4 --scene-payload metadata --seed 23
```

执行维护中的 500 条分阶段 preset：

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.synthesize_records \
  --preset large-500 --format png --workers 4 --require-exact-count
```

该 preset 每 25 条打印一次进度，默认写入 `artifacts/generated/synthetic_500/records.jsonl`。

复用已验收的 enhanced specs，跳过重复采样和增强：

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.synthesize_records \
  --input-enhanced artifacts/generated/synthetic_500/json/enhanced/records.enhanced.jsonl \
  --output artifacts/generated/synthetic_500_both/records.jsonl \
  --max-records 500 --format both --workers 4 --require-exact-count
```

渲染与基础检查：

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.smoke_check
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.render_smoke_check
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.render_pilot --format both --max-samples 5
```

运行冻结兼容链：

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.run_legacy_synthesis \
  --config configs/legacy/config_easy.json --records 1 --disable-shader
```

详细契约见 `docs/FORMAL_SCHEMA.md`、`docs/SYNTHESIS_ENGINE.md` 和 `docs/LEGACY_TOOL_COVERAGE.md`。
