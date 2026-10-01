# 双仓重构完成审计

## 结论

双仓基础重构、旧资产删除、分仓提交和 fresh-clone 验收已经完成。宿主文件系统仅保留两个活动 Git 根，两仓原工作树和 fresh clone 均保持 clean。

- Geosym-World：`cabb9d1b490caaa2bf38262cc9dd8395cc856c5f`；
- GeoClaw：`9e12737748039f0f3a1dfd42976f3c2a02e5f2e9`。

后期研究阶段（开放世界真实题图 Parser、不确定状态 Agent、高级 3D）按计划不是基础重构完成的前置条件，也未在本审计中宣称完成。

## 完成标准矩阵

| 完成标准 | 状态 | 当前证据 | 剩余动作 |
|---|---|---|---|
| 工作区只有两个活动 Git 仓库 | 已完成 | 宿主文件系统扫描仅返回 `Geosym-World/.git`、`GeoClaw/.git`；根 `.git` 不存在 | 无 |
| 两仓在 `geosym` 中独立安装和测试 | 已完成 | Geosym-World `56 passed`；GeoClaw `49 passed`；`pip check` 通过 | 无 |
| Geosym-World 不依赖 GeoClaw | 已完成 | `src/` 与 tests 的 import 扫描为 0 | 无 |
| GeoClaw 通过包依赖调用 Geosym-World | 已完成 | `pyproject.toml` 声明 `geosym-world>=0.1.0`；runtime 从正式包导入 builder、solver、verifier、rendering 和 schema | 无 |
| GeoClaw 无复制几何源码 | 已完成 | `src/geosym_world/` 与旧 `geosym_world_geometry/` 均不存在 | 无 |
| 合成、Parser、Benchmark、Agent 共用 schema/ID | 已完成 | 合成 record、Parser `GeometryRecord`、`ParserStateEvaluator`、`SyntheticTaskAdapter` 和 Agent runtime 均保留 formal IDs；跨仓 fixture tests 通过 | 无 |
| Parser hypothesis 与 admitted fact 分离 | 已完成 | Real/Hybrid Parser 持久化 `state=null`，候选为 `PROPOSED` hypothesis；contract/fixture tests 通过 | 无 |
| 无效 Agent 动作不污染持久状态 | 已完成 | rejected construct 的 version/checksum/state before-after 一致测试通过 | 无 |
| 数据和运行产物带 manifest、seed、配置和路径 | 已完成 | synthesis manifest；`geoclaw.run_manifest.v1` 覆盖 pilot、batch、model batch；seed 不适用时显式为 `null` | 无 |
| 两仓 artifacts 物理与职责分离 | 已完成 | 两仓独立 `artifacts/README.md`、路径 helper 与 ignore 规则 | 无 |
| 源码仓不保留缓存和大规模运行结果 | 已完成 | 两仓 cache/build/egg-info 为 0；`artifacts/` 均只保留 README；无大规模运行产物被跟踪 | 无 |
| runtime 无硬编码工作区绝对路径 | 已完成 | 两仓 `src/tests/configs/examples` 扫描为 0 | 无 |
| 新 Agent 自动读取 AGENTS.md | 已完成 | 工作区根及两仓各有适用范围明确的 `AGENTS.md` | 无 |
| 测试和最小运行后 Git status clean | 已完成 | 原工作树与 fresh clone 的 `git status --porcelain` 均为空 | 无 |
| 删除项均有迁移/归档/可重建依据 | 已完成 | archive、inventory、capability ledger、最终 deletion manifest 完整 | 无 |

## 阶段 9 验收矩阵

| 验收项 | 状态 | 证据 |
|---|---|---|
| 2D exact-state 回归 | 已完成 | structured intersection、construction、render scene 和 degeneracy tests |
| 500 条 exact-count/唯一性/target/结构审计 | 已完成 | 500/500 valid、500 unique IDs、500 targets、0 coincident-point records |
| SVG/PNG 一致性与视觉抽检 | 已完成 | 500 PNG + 500 SVG；引用、signature、XML 全通过；四类代表图人工抽检 |
| 基础 3D 与最小数值验证 | 已完成 | Plane/Polyhedron contract round-trip、tetrahedron volume/section、hidden-edge rendering tests |
| Parser v0 五组件指标 | 已完成 | `ParserStateEvaluator` 输出 objects、relations、quantities、target、grounding；right-triangle fixture 五项均为 1.0 |
| invalid action state isolation | 已完成 | state version/checksum 不变测试 |
| replay checksum | 已完成 | 单元 gate 与仓外双 replay checksum 均通过 |
| public payload privacy | 已完成 | private state、answer、oracle、fact/certificate leakage tests |
| 真实 model adapter 离线协议与受控集成 | 已完成 | OpenAI-compatible request/response/image/usage test；direct runner integration 与 model manifest test |

## 提交前仓外打包验收

- 两仓从当前工作树构建 wheel，安装到 `/tmp/geosym_refactor_acceptance_20260809/venv`；
- import 路径均位于该 venv 的 `site-packages`，不指向源码工作树；
- `geosym-synthesize` 从 `/tmp` 生成 2/2 valid records，并打印逐条进度；
- `geoclaw-pilot` 生成 `geoclaw.run_manifest.v1`，final accuracy 为 1.0；
- 同一动作序列两次 replay 得到 terminal checksum `eb4b8e1695c18147f5fef1fdd55834133e93e9ba60e17bbd3c723cd7aef595d3`；
- venv `pip check` 通过；wheel 和批量验证 checksum 见 `validation_evidence.json`。

## 提交后 fresh-clone 验收

- 从两个本地 commit 使用 `git clone --no-local` 创建全新 clone；
- fresh clone 测试：Geosym-World `56 passed`，GeoClaw `49 passed`；
- 从 fresh clone 构建并安装 wheel 到源码树外 venv，import 路径位于该 venv 的 `site-packages`；
- wheel SHA256：Geosym-World `d35da25f8654b40fac8325e0673804c179c5a12e14450626e2f75e71b42b7360`，GeoClaw `6fc1aed430aa5efc223d61cfe43006c98da53e6e3bf2d20c33f429858e9607f5`；
- 最小合成生成 2/2 valid、2 个唯一 ID，并逐条打印进度；
- 两次 `construct_midpoint` replay 的 final accuracy 均为 1，terminal checksum 均为 `eb4b8e1695c18147f5fef1fdd55834133e93e9ba60e17bbd3c723cd7aef595d3`；
- fresh venv `pip check` 通过，两个 fresh clone 和两个原工作树均为 clean。

## 最终状态

基础重构没有剩余阻塞。未执行远端 push；Geosym-World 的旧 remote 仍需在首次推送前由用户确认或替换。
