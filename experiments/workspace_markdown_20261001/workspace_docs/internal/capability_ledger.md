# 双仓能力迁移账本

## Geosym-World

| 能力 | 重构前来源 | 当前权威位置 | 处置 | 验证证据 |
|---|---|---|---|---|
| Geometry State Contract v1 | `geosym_world_geometry`、根 Parser schema | `src/geosym_world/schema/` | 统一并版本化 | schema round-trip tests |
| 2D 几何内核 | 外层 `geometry.py`、`intersections.py` | `src/geosym_world/kernel/geometry2d/` | 迁移并保留精确值 | intersection tests |
| 基础 3D 内核 | `geosym3d`、外层 3D 原型 | `src/geosym_world/kernel/geometry3d/` | 选择可验证最小实现 | synthesis/contract tests |
| 构造与求解 | 外层 builder、construction、solver | `construction/`、`solver/`、`verifier/` | 按职责拆分 | synthesis and trace tests |
| SVG/PNG 渲染 | 外层 rendering、旧 scripts/tools | `src/geosym_world/rendering/` | 统一 scene、数学文本和标签布局 | rendering tests |
| 批量合成 | 外层 synthesis、旧 scripts/tools | `src/geosym_world/synthesis/`、`tools/` | 保留 exact-count、增强、进度和 enhanced-spec 断点导出 | strict 500/500 records + 500 PNG + 500 SVG |
| Parser | 根 `geosym`、旧 hybrid Parser | `src/geosym_world/parser/` | 迁移为研究原型；输出 hypothesis | parser hypothesis test |
| 数据布局与 adapter | 根数据脚本、旧 export | `src/geosym_world/datasets/` | 默认写入仓内 artifacts | dataset layout tests |
| Benchmark/Eval/Diagnosis | 根 Eval 原型 | `src/geosym_world/benchmark/` | 保留组件评测骨架 | import and full test suite |
| 旧工具兼容 | `/data/Geosym127K/scripts/tools` | `src/geosym_world/legacy_tools/` | 维护兼容入口，不复制旧产物 | legacy compatibility tests |

Geosym-World 当前完整测试结果为 `56 passed`。Parser 候选可用于临时校验和渲染，但持久化 `GeometryRecord.state` 必须为空；候选保存在带 confidence、provenance 和 admission metadata 的 `PROPOSED` hypothesis 中。严格 500 条合成验证覆盖 13 个 family，过滤全部重合点退化，且 PNG/SVG 双格式产物均通过结构校验。Parser v0 fixture 已具备 objects、relations、quantities、target 和 grounding 五组件指标。

## GeoClaw

| 能力 | 重构前来源 | 当前权威位置 | 处置 | 验证证据 |
|---|---|---|---|---|
| Typed action/feedback 协议 | 内层 GeoClaw | `src/geoclaw/schema.py`、`feedback.py` | 迁移并保留 | schema tests |
| Verifier-gated environment | 内层 environment/tools | `src/geoclaw/environment.py`、`tools/` | 事务执行、拒绝不提交 | environment/tool tests |
| Persistent state version/checksum | 旧 state/replay 原型 | `src/geoclaw/environment.py`、`trajectory.py` | 增加确定性 checksum gate | replay tests |
| Privacy projection | 内层 public/private 原型 | `src/geoclaw/` | 保留并测试 oracle 隔离 | privacy tests |
| Model adapters | scripted/OpenAI-compatible 原型 | `src/geoclaw/models.py` | 保留通用 provider 接口，厂商参数通过显式 `extra_body` 配置 | offline protocol + controlled runner integration |
| Direct/Open/Closed-loop runner | 旧实验 runner | `src/geoclaw/runners/` | 删除 fair-v2 绑定，保留通用 runtime | runner imports and CLI |
| Geometry tools | 内层复制工具和几何包 | `src/geoclaw/tools/` + `geosym-world` 依赖 | 删除复制内核，改为单向依赖 | cross-repo import tests |
| Pilot tasks | frozen task 原型 | `src/geoclaw/datasets/` | 压缩为最小 fixture | 10-task CLI listing |

GeoClaw 当前完整测试结果为 `49 passed`。活动源码中不再存在复制的 Geosym-World 包；依赖方向固定为 `GeoClaw -> Geosym-World`。pilot、batch replay 和 model batch 统一写出 `geoclaw.run_manifest.v1`，记录配置、seed 适用性、动作/trajectory checksum、terminal state checksum 和产物路径。

## 已删除或不迁移

| 资产 | 决策 | 原因 |
|---|---|---|
| `GeoClaw/paper/` | 全部删除 | 用户后续重新写论文，不保留旧稿 |
| GeoClaw E1/E2/E3、fair-v2 专用代码和结果 | 删除 | 旧论文实验耦合，不属于新 runtime 主干 |
| GeoClaw 内复制几何包 | 删除 | 消除双实现漂移，改用正式包依赖 |
| 大规模合成图片和 replay 输出 | 不迁移到源码仓 | 可重建产物，文件清单已归档 |
| 一次性 smoke/debug 脚本 | 删除 | 不具备稳定接口或长期维护价值 |

## 后续研究而非本次基础重构承诺

- 开放世界真实题图 Parser 与跨数据集泛化；
- calibrated hypothesis admission 和不确定状态 Agent；
- 完整高级 3D、CAD 一致性和 3D Agent；
- 复杂证明程序、长程规划和大规模模型实验。
