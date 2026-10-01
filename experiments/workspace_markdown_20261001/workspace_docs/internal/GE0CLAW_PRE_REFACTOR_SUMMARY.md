# GeoClaw 仓库总结与重构基线

> 更新时间：2026-08-09  
> 用途：在整体重构前，记录当前仓库的真实能力、核心调用链、实验契约、数据资产和不可破坏行为。  
> 状态：基于当前根目录代码阅读与离线测试；`python -m pytest -q` 为 **106 passed**。

## 1. 一句话定位

GeoClaw 是一个面向多模态几何求解的、由确定性符号几何环境支撑的 Agent 交互原型。模型不直接修改几何状态，而是提交结构化动作；环境负责引用解析、执行、验证、提交或回滚，并把经过隐私投影的反馈返回给模型。

它当前同时承担四类职责：

1. 可执行符号几何内核；
2. 合成几何数据生成与渲染；
3. GeoClaw 工具交互环境及模型实验框架；
4. 论文、冻结协议、结果和研究历史归档。

这四类职责目前共处一个 Python 包和一个仓库，是本次整体重构最需要重新划定的边界。

## 2. 当前权威口径

仓库里存在两套实验历史，重构时不能混为一套：

| 资产 | 当前定位 | 结果 |
|---|---|---|
| `paper/`、`deliverables/` | 当前四页 ECCV Workshop 稿 | 50 题；Direct 74%，Open-loop 10%，Stateful Closed-loop 90% |
| `data/fair_v2/`、`results/fair_v2/`、`docs/` | 较早的冻结 fair-v2 可复现版本 | 100 题；Direct 52%，Open-loop Batch 71%，Closed-loop 84% |
| `new_8page_paper/` | 历史规划、协议、审计、smoke/formal 记录和文献笔记 | 非当前论文权威来源 |
| `geosym_world_geometry/outputs/synthetic_500/` | 已生成的合成记录与图片 | 体积约 50 MB，属于生成产物 |

当前研究结论只支持 workshop-scale 合成/重建任务上的完整闭环效果。它不证明：

- Parser 准确率；
- 真实图片鲁棒性；
- 跨模型泛化；
- SOTA 定理证明能力；
- verifier 单组件的独立因果贡献。

## 3. 仓库地图

```text
GeoClaw/
├── geosym_world_geometry/       # 可执行代码主体
│   ├── geometry.py              # 几何计算原语
│   ├── intersections.py         # 交点算法
│   ├── objects.py               # 几何对象定义
│   ├── construction.py          # 构造操作 schema/归一化
│   ├── builder.py               # 构造轨迹 -> FormalGeometryState
│   ├── state.py                 # 形式几何状态容器
│   ├── fact.py                  # Fact / FactStore
│   ├── solver.py                # 状态 -> 事实闭包
│   ├── verifier.py              # 轨迹和状态验证
│   ├── derivation.py            # 推导相关结构
│   ├── export.py                # 完整记录导出
│   ├── legacy_adapter.py        # 旧 JSON 兼容层
│   ├── rendering/               # scene、布局、样式、PNG/SVG 渲染
│   ├── synthesis/               # 模板、采样、增强、过滤、导出流水线
│   ├── geoclaw/                 # Agent 环境、工具、协议、runner、评测
│   ├── tools/                   # CLI 入口
│   └── examples/                # 小型示例
├── tests/                       # 106 个离线测试
├── data/fair_v2/                # 冻结公开任务、图片、manifest、E3 故障
├── results/fair_v2/             # 冻结聚合结果和配对统计
├── docs/                        # 架构、实验协议、复现文档及设计历史
├── paper/                       # 当前四页论文 LaTeX
├── deliverables/                # 当前 PDF 和 Overleaf 包
├── new_8page_paper/             # 历史研究过程材料
├── literature/                  # 文献证据账本
└── fixtures/                    # 旧格式兼容测试样例
```

## 4. 系统分层

### 4.1 几何内核

核心数据流：

```text
construction trace
    -> ConstructionOp normalization
    -> ConstructionBuilder.execute()
    -> FormalGeometryState
    -> FactClosureSolver.solve()
    -> FactStore
    -> TraceVerifier / export / render / GeoClaw tools
```

主要职责：

- `objects.py` 和 `state.py` 保存点、线、射线、线段、圆、弧、多边形、区域、平面、多面体及 target。
- `construction.py` 把字典形式的 trace 归一化成 `ConstructionOp`。
- `builder.py` 是命令解释器，按顺序执行构造操作并维护对象引用。
- `solver.py` 从坐标与拓扑关系推导长度、角度、面积、周长、平行、垂直、共线、相似、圆关系、空间关系等事实。
- `FactStore` 保存事实值、来源、规则、依赖和 metadata。
- `verifier.py` 在导出或动态构造前检查重复 ID、未知引用、维度错误、退化对象和 target 缺失等问题。

当前内核是坐标/解析几何驱动的确定性系统，不是通用公理证明器。

### 4.2 合成数据流水线

`synthesis/pipeline.py` 的流程为：

```text
template parameter grid
    -> sampler + base mutation
    -> formal trace enhancer
    -> duplicate filter
    -> degeneracy verifier
    -> state/fact construction
    -> deterministic rendering
    -> JSONL + QA + GT + manifest
```

当前内置 8 个模板族：

1. `triangle_area_from_height`
2. `angled_line_intersection`
3. `circle_sector`
4. `parallel_cut_similarity`
5. `rectangle_area_diagonal`
6. `parallelogram_area`
7. `trapezoid_area`
8. `tetrahedron_section`

生成的记录同时包含图像、scene、formal state、事实、验证报告、问题、答案和 lineage metadata。随机性通过 seed 和确定性样式策略控制。

### 4.3 渲染层

```text
FormalGeometryState
    -> rendering.adapter.state_to_render_scene()
    -> RenderScene
    -> layout/style policy
    -> PNG or SVG
```

渲染既服务数据生成，也服务 Agent 构造后的逐步重绘和 zoom。`rendering/scene.py` 是中间表示，`style_policy.py` 负责确定性视觉样式。

### 4.4 GeoClaw 交互层

`GeoClawEnvironment` 是运行时总入口：

```text
GeoClawTask
    -> 初始化 base trace / state / facts / image
    -> observe()
    -> AgentAction.from_dict()
    -> tool policy gate
    -> Tool.execute()
    -> utility classification
    -> optional rerender
    -> ToolFeedback + TrajectoryStep
    -> final scoring / termination
```

当前工具：

| 工具 | 作用 | 是否改变持久状态 |
|---|---|---:|
| `zoom` | 裁剪当前图像并加入 observation | 否 |
| `calculate` | 对受限 SymPy 表达式求精确值/数值 | 否 |
| `measure` | 图像或 oracle 测量 | 否；正式 fair 协议禁用 |
| `construct` | 添加辅助几何对象 | 是，验证通过后提交 |
| `verify` | 检查定性几何 claim | 否 |
| `final` | 提交一次最终答案并终止 | 终止 episode |

`construct` 的事务语义是关键：它先组合 `base_trace + agent_trace + candidate_op`，完整验证并重新构造候选状态；只有候选有效时才替换环境的 `agent_trace/state/facts`。失败动作必须保持持久状态不变。

### 4.5 模型与实验层

`geoclaw/runners/agents.py` 实现：

- `Direct`：一次模型调用，直接回答；
- `Open-loop Batch`：先一次性生成最多 5 个非 final 动作，批量执行后统一看到反馈，再回答；
- `Closed-loop`：每次只生成一个动作，在下一步决策前看到该步反馈；
- legacy `Open-loop`：保留用于重放已失效的开发实验，不是当前 fair 主条件。

正式 fair 配置：

- 最多 5 个非 final 动作；
- episode 总输出预算 4096 tokens；
- `SAFE_TOOL_NAMES = (zoom, calculate, construct, verify, final)`；
- 禁止 `measure`；
- 禁止 oracle measurement 和 target-value verification；
- final scorer 不向模型暴露，不提供第二次猜答案机会。

模型适配层当前包含：

- `ScriptedModelClient`：测试和可复现脚本；
- `OpenAICompatibleClient`：调用 OpenAI-compatible `/chat/completions`，支持图像、JSON schema、usage、重试和 parse-error 记录。

## 5. 公开/私有信息边界

这是重构中优先级最高的契约。

私有记录可以包含完整构造轨迹、坐标、事实闭包、参考答案、certificate 和 oracle 诊断。模型只能看到自包含的 `PublicProblem`：

- question；
- structured givens；
- target 类型和公开对象 ID；
- visible objects；
- annotated image；
- 经投影后的 action feedback；
- action history 和剩余预算。

`privacy.assert_public_payload()` 在模型请求发出前递归拒绝以下内容：

- `formal_state`、`facts`、`trace`、`construction_trace`；
- `answer`、`gt`；
- 坐标、内部 audit、state before/after；
- oracle 数值和 residual；
- 带内部 derivation rule/dependencies/source 的 fact；
- `target_lookup` 等私有规则。

`protocol.project_public_feedback()` 只投影稳定状态、错误码、公开对象、公开计算结果、verify truth、创建对象和 crop 信息，不应序列化内部 facts 或审计信息。

注意：裸 `GeoClawEnvironment` 的默认 `PublicToolPolicy` 是宽松模式，允许 oracle measure 和 target verify；正式 runner 显式传入受限 policy 并使用 safe tool allowlist。重构时应避免让“开发默认值”意外进入正式实验。

## 6. 关键数据模型与外部契约

### 几何域

- `ConstructionOp`：`op + id + params`；trace 的最小命令单元。
- `FormalGeometryState`：所有几何对象、关系、target 和 metadata 的持久状态。
- `Fact` / `FactStore`：求解器输出及其 provenance。
- `VerificationReport`：`static_valid`、`executable`、`state_valid`、issues、metrics。

### Agent 域

- `GeoClawTask`：问题、私有 trace/answer、可选 public problem、图像、预算、容差、难度和评测 metadata。
- `AgentAction`：固定 action 名称和字典 arguments。
- `GeoClawObservation`：当前公开可观察状态。
- `ToolFeedback`：动作状态、utility、result/evidence/errors、预算和图像路径。
- `EpisodeTrajectory` / `TrajectoryStep`：可重放过程、私有审计状态、终止原因和 metrics。

### 文件契约

- 合成记录使用 JSONL；每条记录带 `id`、image、formal state、facts、verification、synthetic lineage 等。
- public task 与 private certificate 分文件保存。
- frozen manifests 保存文件 checksum、采样配置和任务顺序。
- 每个实验 episode 保存 private trajectory 和 projected public trajectory。
- batch manifest 保存 condition、模型信息、usage、复现 metadata 和 episode 列表。
- `RunLedger` 支持已完成任务恢复，基础设施失败会被单独记录。

## 7. 评测逻辑

### E1：可执行完整性

检查加载、公开/私有分离、schema、回滚、重放、评分、checksum 和泄漏控制。它是协议/实现审计，不是模型准确率实验。

### E2：闭环有效性

三种条件接收相同公开题目、图像、目标、模型和评分规则。主要指标是答案准确率；过程指标包括 schema/argument 有效率、invalid/useful action rate、调用次数和 token。

fair-v2 的主要结论是 Open-loop Batch 71% 到 Closed-loop 84%，差值 +13 个百分点，配对 95% CI `[7, 20]`，Holm-adjusted `p=0.000313`。这支持逐步适应的价值，但不是等算力比较。

### E3：受控错误恢复

30 个配对故障：argument grounding、invalid construction、unsupported claim 各 10 个。Generic Failure 恢复 22/30，Structured Feedback 恢复 20/30，差异不显著。只能声称多数错误可恢复，不能声称 structured feedback 更优。

## 8. 当前测试基线

根目录执行：

```bash
python -m pytest -q
```

当前结果：

```text
106 passed in 50.13s
```

覆盖范围：

- 构造轨迹、交点和几何状态；
- 合成模板、增强、过滤、导出；
- PNG/SVG 渲染和样式确定性；
- action schema、工具 allowlist、预算与终止；
- calculator 安全表达式和 public provenance；
- construct 成功提交与失败回滚；
- verify、measure、zoom、final；
- public problem 充分性和 private leakage 防护；
- Direct/Open-loop/Closed-loop 公平性；
- runner resume、usage 和基础设施失败记录；
- E1 审计、E2 配对统计、E3 故障恢复；
- frozen artifact checksum 和 protocol split。

重构期间应持续把这 106 个测试作为行为回归基线；模块移动后先保持测试语义，再讨论删除或重写测试。

## 9. 已观察到的重构压力点

以下是代码现状，不代表必须采用某一种新架构：

1. **职责集中。** `solver.py` 超过 1200 行，`builder.py` 约 600 行，分别集中大量二维、三维、圆、三角形和目标特例。
2. **schema 分散。** construction op、builder 分派、verifier 引用规则、construct allowlist、Agent JSON schema 需要人工同步。
3. **环境职责偏多。** `GeoClawEnvironment` 同时承担状态机、policy gate、工具调度、渲染、副作用、轨迹记录和评分触发。
4. **协议和运行时耦合。** 开发模式允许 `measure`/oracle，fair runner 再通过 allowlist 和 policy 收紧，默认值容易误用。
5. **I/O 与领域逻辑混合。** environment 初始化会创建输出目录；runner 同时处理调用、恢复、trajectory、ledger 和 manifest。
6. **公开模型部分硬编码。** `public_problem.py` 按特定 synthetic family 手写 givens 和 certificate 公式，不是通用公开题转换器。
7. **历史资产较多。** 当前论文、fair-v2 和旧八页研究材料并列，容易造成结果引用和入口选择混乱。
8. **包清单手工维护。** `pyproject.toml` 显式枚举子包，模块重排时容易漏包。
9. **代码风格不统一。** 部分实验 runner 使用高度压缩的一行式代码，降低审查和协议 diff 的可读性。
10. **生成产物进入源码树。** `geosym_world_geometry/outputs/` 约 50 MB，使代码包、fixture 和实验资产边界不清晰。

## 10. 建议的重构边界

可考虑按依赖方向拆成以下逻辑包；名称可调整，关键是保持单向依赖：

```text
geometry-core
  objects / state / operations / exact geometry
        ↓
geometry-engine
  builder / solver rules / verifier
        ↓
geometry-rendering       geometry-synthesis
  scene / layout         templates / mutation / export
        ↓                       ↓
geoclaw-protocol
  public problem / action / observation / feedback / privacy
        ↓
geoclaw-runtime
  environment / tools / transaction / trajectory
        ↓
geoclaw-evaluation
  agents / model clients / runners / faults / scoring / statistics
        ↓
cli-and-artifacts
  commands / manifests / frozen releases / paper support
```

建议优先抽出的稳定接口：

- `GeometryEngine.apply(trace) -> state + facts + verification`；
- `GeometryTransaction.try_apply(op) -> committed/rejected result`；
- `ToolContext`，替代工具直接依赖整个 environment；
- `ProtocolPolicy`，统一工具 allowlist、oracle 权限、反馈投影和预算；
- `ArtifactStore`，统一 trajectory、manifest、ledger 和 checksum I/O；
- operation registry，由同一来源派生 builder handler、参数 schema、引用规则和 public allowlist。

## 11. 推荐重构顺序

### 阶段 A：冻结行为

1. 保存当前测试结果和 frozen artifact checksums。
2. 为核心 JSON/JSONL 增加 golden schema 或 snapshot。
3. 明确当前 public API 和仅内部使用 API。
4. 将论文/实验数字与对应 manifest 建立显式索引。

### 阶段 B：纯机械整理

1. 统一格式化和命名，不改变协议。
2. 把生成产物移出 importable package。
3. 改为自动 package discovery。
4. 将 CLI 参数解析与业务函数分离。

### 阶段 C：拆领域内核

1. 按对象/规则族拆 builder 和 solver。
2. 建立统一 operation registry。
3. 保持 trace 和 exported record 的向后兼容 adapter。
4. 每拆一个规则族就运行内核、合成和 public problem 测试。

### 阶段 D：拆运行时与协议

1. 从 environment 抽出 transaction、policy、renderer 和 recorder。
2. 让工具依赖窄接口，不直接修改环境所有字段。
3. 把 permissive development policy 与 frozen fair policy 命名成显式 profile。
4. 对每次模型 outbound payload 强制执行 privacy guard。

### 阶段 E：整理评测和资产

1. 统一 Direct/Open-loop/Closed-loop runner 模板。
2. 把 legacy runner 与当前 runner 分目录。
3. 分离 current paper、frozen release 和 research archive。
4. 重新生成 manifest，但保留原始 frozen 文件及 checksum 供比对。

## 12. 重构不可破坏清单

- [ ] 相同 trace 构造出等价状态和等价 target facts。
- [ ] 符号值优先保持 exact，不被无意转成 float。
- [ ] 无效构造不会改变 `agent_trace`、state 或 facts。
- [ ] 对象 ID 和结构别名解析行为保持兼容。
- [ ] final 只评分一次，评分信息不回流给 Agent。
- [ ] fair 条件不暴露 `measure`、oracle 或 target verifier。
- [ ] 模型请求不包含 answer、坐标、internal facts、trace 或 audit。
- [ ] public feedback 只包含投影后的稳定字段。
- [ ] Open-loop Batch 在整批动作生成前看不到任何执行结果。
- [ ] Closed-loop 的下一动作确实接收上一动作反馈。
- [ ] 动作预算和 token 总预算保持当前语义。
- [ ] private/public trajectory 继续分离。
- [ ] task 顺序、split、base-spec disjointness 和 checksum 可验证。
- [ ] seed 下的采样和渲染继续可复现。
- [ ] 当前 106 个离线测试持续通过。
- [ ] 当前四页论文结果与 fair-v2 结果仍被明确区分。

## 13. 常用入口

安装和测试：

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

查看付费模型实验参数：

```bash
python -m geosym_world_geometry.tools.run_model_experiment --help
python -m geosym_world_geometry.tools.run_e3_smoke --help
```

数据与环境 smoke check：

```bash
python -m geosym_world_geometry.tools.smoke_check
python -m geosym_world_geometry.tools.render_smoke_check
```

主要阅读入口：

- `README.md`
- `docs/architecture.md`
- `docs/experiment_protocol.md`
- `docs/reproducibility.md`
- `geosym_world_geometry/geoclaw/environment.py`
- `geosym_world_geometry/geoclaw/runners/agents.py`
- `geosym_world_geometry/geoclaw/protocol.py`
- `geosym_world_geometry/geoclaw/privacy.py`
- `geosym_world_geometry/builder.py`
- `geosym_world_geometry/solver.py`
- `geosym_world_geometry/synthesis/pipeline.py`

## 14. 最终判断

当前 repo 已经具备完整的研究原型闭环：形式构造、事实求解、验证回滚、图像渲染、结构化 Agent 工具、隐私投影、公平条件、故障恢复、统计与冻结产物都能离线验证。

整体重构的主要目标不应是改变研究逻辑，而应是把已有逻辑拆成清晰、单向依赖、可独立测试的层，并把实验协议与普通开发默认值隔离。最重要的资产不是某个类名，而是 exact geometry、事务提交、公开/私有分离、公平条件和可复现 manifest 这五组行为契约。
