# GeoSym 研究进展复盘：从旧 2D 生成器到 GeoSym-Hard / GeoSym-Debug

更新日期：2026-07-07

## 1. 当前结论

这段时间的探索说明，GeoSym 后续最稳的方向仍然是基于旧 2D GeoSym 做 **GeoSym-Hard / GeoSym-Debug**，但需要把目标收紧：

1. 不应优先押注“真实几何图片全自动转 GeoSym JSON”。
2. 不应让 VLM 直接从真实图像生成可验证数学状态。
3. 不应把当前 canonical builder 当成通用几何重建器。
4. 应先从 GeoSym 自己能表达、能执行、能验证的 formal construction 出发，生成完整数学状态，再由 solver 推出大量 facts，形成可诊断 benchmark。

换句话说，当前更可行的主线不是：

```text
real image -> VLM parse -> GeoSym JSON -> benchmark
```

而是：

```text
GeoSym formal construction
  -> executable SymPy geometry state
  -> solver fact closure
  -> derivation / diagnosis graph
  -> GeoSym-Hard / GeoSym-Debug benchmark
```

真实图片解析仍然有研究价值，但它应该先作为后续扩展或半自动标注工具，而不是当前 benchmark 的第一依赖。

## 2. 原始 repo 的核心能力

旧 2D GeoSym 主要在 `scripts/` 下，是一个几何数据生成管线：

```text
template -> builder -> drawer -> shader -> gt -> qa
```

关键文件包括：

- `scripts/tools/template.py`：生成基础几何模板。
- `scripts/tools/builder.py`：在 SymPy 坐标系统中添加点、线、圆弧、交点和增强结构。
- `scripts/tools/drawer.py`：把几何状态画成图。
- `scripts/tools/shader.py` / `scripts/tools/shaders.py`：生成阴影区域。
- `scripts/tools/gt.py`：计算长度、面积、周长、弧长等 GT。
- `scripts/tools/qa_template.py`：根据几何状态生成 QA。

这个 repo 最有价值的地方是：**生成过程中已经维护真实数学坐标和几何对象结构**。这比真实图片 benchmark 更适合做可验证诊断，因为我们天然知道每个点、线、区域、目标量和答案。

旧管线的不足是：

- 输出 schema 偏生成中间态，不是 benchmark problem state。
- 构造过程没有作为稳定的一等 trace 保存。
- GT 主要服务 final answer，不服务 step-level diagnosis。
- 没有统一 fact store、derivation graph、failure attribution。
- 没有真实数据 adapter。

因此，它不是不能用，而是需要从“数据生成器”升级成“可验证几何推理 benchmark 框架”。

## 3. 已经做过的工程改造

我们在 `geosym/` 下新增了一套不破坏旧 `scripts/` 的框架雏形。

### 3.1 统一 problem state

新增 `geosym/core/schema.py`，定义：

- `GeoSymProblemState`
- `GeoSymSource`
- `GeoSymInput`
- `GeoSymGeometry`
- `GeoSymTarget`
- `GeoSymProof`
- `GeoSymDiagnostic`
- `VerificationReport`

目标是让 synthetic 数据和真实数据最终都能进入同一种结构：

```text
source + input + geometry + target + proof + diagnostic + verification
```

这一步是必要的，因为 GeoSym-Hard / GeoSym-Debug 不能只存图像、题干、答案，还必须存：

- 可解析几何对象；
- 已知条件；
- 构造过程；
- 中间事实；
- 目标指针；
- 验证报告；
- 诊断标签。

### 3.2 Verifier 和诊断接口

新增：

- `geosym/core/verifier.py`
- `geosym/verify/consistency.py`
- `geosym/verify/image_alignment.py`
- `geosym/verify/llm_judge.py`
- `geosym/eval/answer_grader.py`
- `geosym/eval/diagnosis.py`

当前 verifier 能做一些 model-free 检查：

- 点线引用合法；
- 线段不退化；
- `parallel / perpendicular / collinear / length` 等低阶关系是否和坐标一致；
- target answer 是否和 solver 计算一致；
- answer consistency；
- 初版 image alignment；
- 初版 model output diagnosis。

这部分方向是对的，但当前事实库太弱，所以诊断能力还没有真正释放出来。

### 3.3 Synthetic / real adapters

新增：

- `geosym/adapters/synthetic_adapter.py`
- `geosym/adapters/real_adapter.py`
- `geosym/adapters/canonical_builder.py`

`synthetic_adapter` 的目标是把旧 GeoSym 生成结果包装成 `GeoSymProblemState`。

`real_adapter` 的目标是把真实题解析出的 constraint graph 转成 canonical GeoSym state。

`canonical_builder` 当前只支持很薄的 coordinate reconstruction，主要是：

- 点线 scaffold；
- `length`；
- `parallel`；
- `perpendicular`；
- 简单直角三角形模板。

这也是后面 Geometry3K 实验失败的核心原因之一：它不是通用几何重建器，不能承接真实题里的圆、弧、相似、角追踪、面积分解和辅助线。

### 3.4 真实数据目录、CLI、前端

新增了真实数据处理相关模块：

- `geosym/data/layout.py`
- `geosym/data/hf_download.py`
- `geosym/data/records.py`
- `geosym/pipeline/real_parse.py`
- `geosym/pipeline/hybrid_parse.py`
- `geosym/frontend/server.py`

形成了目录约定：

```text
raw data:
  /dev/shm/geosym_datas/<dataset>/

parsed data:
  /data/Geosym127K/datas/<dataset>/
```

也做了一个本地前端，但这个前端本质上只是脚本启动器，用来统一跑：

- 数据下载；
- parse；
- summarize；
- export；
- inspect；
- triage；
- replay。

这个设计是合理的，因为真正可复现的入口应该仍然是脚本。

## 4. 真实图片解析路线的探索

我们尝试过两条真实图片解析路线。

### 4.1 第一版：VLM 直接抽关系 / build instruction

最早的想法是：

```text
image + problem text
  -> VLM extracts objects / relations / target
  -> constraint normalizer
  -> canonical builder
  -> verifier
```

这条路在 Geometry3K 上失败得很明显：

- VLM 经常漏点、漏线、漏标签。
- VLM 会把复杂圆弧、相似、角追踪题压成线段长度题。
- VLM 输出的关系不一定能被当前 GeoSym builder 执行。
- VLM 容易给出看似合理但没有 grounding 的几何关系。
- builder 拿到稀疏约束后会生成退化图，常见结果是点落在一条直线上。
- 即使 overlay 看起来找到了原图结构，canonical state 仍然不是题目的真实数学结构。

这说明问题不只是 prompt，而是任务结构本身不对。

### 4.2 第二版：Script + VLM hybrid parser

后来改成 candidate-first：

```text
raw image + problem text
  -> script visual candidates
  -> script observations
  -> evidence graph
  -> VLM relation pass
  -> instruction task packet
  -> VLM instruction pass
  -> grounding / formalization / builder / verifier
```

相关模块包括：

- `geosym/parsing/vision/primitive_detector.py`
- `geosym/parsing/hybrid/script_observations.py`
- `geosym/parsing/hybrid/evidence_graph.py`
- `geosym/parsing/hybrid/semantic_extractor.py`
- `geosym/parsing/hybrid/instruction_task_packet.py`
- `geosym/parsing/hybrid/instruction_planner.py`
- `geosym/parsing/hybrid/formalizer.py`
- `geosym/parsing/hybrid/extraction_contract.py`

这个结构比第一版更可审计：

- 脚本提供 visual candidates 和 text claims。
- VLM 负责语义绑定、候选确认、关系判断。
- planner 负责判断哪些关系能进入当前 GeoSym builder。
- formalizer 负责区分 coordinate-buildable、theorem-only、unsupported。
- report 记录每一步失败在哪里。

但是这条路当前仍然不适合作为主线，原因是：

1. 传统视觉候选本身很弱，几何图上的标签、圆弧、角标、阴影区域难以稳定提取。
2. VLM 对 named object binding 不稳定，尤其是 `AB -> 某条视觉线段`、`∠ABC -> 两条 incident lines`。
3. 当前 builder capability 太窄，只支持少量低阶关系。
4. 真实题里的核心关系往往是 theorem-level，而不是 coordinate scaffold。
5. 即使 parser 输出了关系，缺少完整 construction kernel 和 theorem solver，也无法形成可验证 derivation。

所以 hybrid parser 的工程价值是：它帮我们看清楚真实图片解析的失败分层。

但它还不能支撑“下个月快速做出一个可靠 benchmark”。

## 5. Geometry3K 100 / 30 条实验暴露的问题

我们围绕 `hiyouga/geometry3k` 做过小规模实验，目标是把真实题转成 GeoSym 可解析状态。

实际暴露出的问题包括：

### 5.1 VLM 关系抽取不稳定

即使用 `qwen/qwen3.7-plus` 做 staged extraction，VLM 也不能稳定输出：

- 完整对象列表；
- 正确标签绑定；
- 当前 GeoSym builder 可执行的 relation；
- 可用 target；
- 可复查的 build instruction。

很多样本会出现：

- `target` 是 unknown；
- relation type 是 theorem-level；
- relation 缺少可执行对象；
- candidate decision 缺失；
- VLM 硬造 builder 当前不支持的 instruction。

### 5.2 Canonical builder 会把不充分约束画成假图

早期版本里，如果只有少量点线约束，builder 会启发式布局，导致大量图变成直线或错误拓扑。

后面虽然加了 scaffold / canonical sketch 区分，但本质问题仍然在：真实题解析出的关系不足以唯一重建数学状态。

这说明：**真实图片转 GeoSym JSON 不应该是 image-to-coordinate，也不能靠稀疏关系硬 build。**

### 5.3 当前 parser 缺少真正的 theorem fact closure

Geometry3K 很多题需要：

- 圆周角；
- 弦切角；
- 相似三角形；
- 全等；
- 角平分线；
- 面积比；
- 辅助线；
- 多步代数方程。

当前 formalizer 能把这些标成 theorem gap，但没有 solver 把它们变成可验证 facts。

因此报告可以告诉我们“这里需要 theorem solver”，但不能真正解决样本。

### 5.4 真实图像解析不应作为当前 benchmark 的硬依赖

真实图像解析是长期价值方向，但短期内它有三个不确定性：

- 视觉识别不稳；
- 语义绑定不稳；
- GeoSym solver 覆盖不够。

如果当前 benchmark 依赖这条链路，就会把论文风险从“模型几何推理诊断”转移成“我们的 parser 能不能解析真实图”。这不是最优路径。

## 6. 当前研究判断

现在最合理的判断是：

### 6.1 GeoSym 的优势不是解析真实图，而是生成可验证几何世界

GeoSym 最强的地方是：

- 可以生成真实数学坐标；
- 可以控制构造过程；
- 可以知道所有对象和中间量；
- 可以渲染图；
- 可以生成题目和答案；
- 可以通过 solver/verifier 自动检查。

这天然适合做：

- hard geometry benchmark；
- diagnostic benchmark；
- counterfactual evaluation；
- solver-in-the-loop repair；
- auxiliary construction evaluation。

### 6.2 GeoSym-Hard 应该先从 synthetic verified data 做起

如果目标是快速做一篇 benchmark，最稳路线是：

```text
old GeoSym generation
  -> formal construction trace
  -> full GeoSymProblemState
  -> solver fact closure
  -> hard target selection
  -> static benchmark export
```

而不是先解决 Geometry3K。

### 6.3 GeoSym-Debug 的核心是 fact-level diagnosis

GeoSym-Debug 不应该只是“模型答错了”。它应该能判断：

- 模型引用了不存在对象；
- 模型说错了拓扑关系；
- 模型用了前提不满足的定理；
- 模型漏了关键辅助构造；
- 模型几何路径对但计算错；
- 模型最终答对但中间推理无效。

要做到这一点，关键不是更强 VLM parser，而是：

```text
complete state + fact store + derivation graph
```

## 7. 新的核心设计：Formal Construction -> Solver Facts

下一步应新增一层 formal construction kernel。

### 7.1 Formal construction trace

每道题先保存 GeoSym 自己能执行的 construction：

```json
{
  "construction_trace": [
    {"op": "add_point", "id": "A", "coord": ["0", "0"]},
    {"op": "add_point", "id": "B", "coord": ["4", "0"]},
    {"op": "add_point", "id": "C", "coord": ["4", "3"]},
    {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
    {"op": "add_segment", "id": "BC", "points": ["B", "C"]},
    {"op": "assert_perpendicular", "lines": ["AB", "BC"]},
    {"op": "set_target", "target": {"type": "length", "object": "AC"}}
  ]
}
```

第一阶段支持的 op 可以很小：

- `add_point`
- `add_segment`
- `add_circle`
- `add_polygon`
- `intersection`
- `midpoint`
- `foot_of_perpendicular`
- `parallel_through`
- `perpendicular_through`
- `assert_length`
- `assert_angle`
- `assert_parallel`
- `assert_perpendicular`
- `mark_shadow_region`
- `set_target`

这层的目标不是兼容所有真实题，而是让 GeoSym 自己生成的数据都有完整、可重放的数学构造。

### 7.2 Executable SymPy state

construction trace 执行后生成：

- `points`
- `lines`
- `arcs`
- `entities`
- `relations`
- `quantities`
- `target`

这些对象全部在真实数学坐标系中，由 SymPy 表达式维护，而不是来自像素坐标。

### 7.3 Solver fact closure

在完整 state 上跑 solver，推出大量 facts：

```json
{
  "id": "fact_023",
  "type": "length",
  "object": "AC",
  "value": "5",
  "source": "derived",
  "rule": "distance_formula",
  "dependencies": ["point:A", "point:C"]
}
```

fact 类型包括：

- object facts：点、线、圆、区域是否存在；
- topology facts：共线、点在线上、相交、平行、垂直；
- metric facts：长度、角度、面积、周长、半径；
- equivalence facts：等长、等角、比例；
- theorem facts：直角三角形、相似、全等、圆相关关系；
- target facts：目标答案。

第一版可以先做 coordinate fact closure，因为坐标已知：

- 任意两点距离；
- 任意三点角度；
- 任意多边形面积；
- 任意两线平行/垂直；
- 点是否在线上；
- 线线交点；
- 长度比；
- 面积比。

这已经足以支撑一批高质量诊断。

### 7.4 Derivation / dependency graph

每个 derived fact 都保存 rule 和 dependencies。target answer 从 fact store 中选出，并反向追踪出 derivation graph。

这会成为 GeoSym-Debug 的核心 GT：

```text
given facts
  -> derived facts
  -> target answer
```

模型输出不再只和 final answer 对比，而是和 fact graph 对比。

## 8. 建议新增模块

建议新增：

```text
geosym/formal/
  construction.py
  state_builder.py
  fact.py
  solver.py
  rules.py
  derivation.py
  export.py
```

职责：

- `construction.py`：定义 formal construction op schema。
- `state_builder.py`：执行 construction trace，生成 GeoSymProblemState。
- `fact.py`：定义 fact / fact store。
- `solver.py`：在完整 state 上推 facts。
- `rules.py`：维护 coordinate rules 和 theorem-lite rules。
- `derivation.py`：为 target 找 dependency graph。
- `export.py`：导出 GeoSym-Hard / GeoSym-Debug JSONL。

这条线可以和旧 `scripts/tools/builder.py` 对接：先从旧 builder 的状态中导出 construction trace 或至少导出 equivalent formal state，然后逐步替代不稳定的真实图 parser。

## 9. Benchmark 论文路线建议

### 9.1 第一阶段：GeoSym-Hard

目标：做静态高难平面几何 benchmark。

重点不是题量最大，而是：

- 非多选；
- exact answer；
- 有完整数学状态；
- 有可验证 facts；
- 有复杂数值；
- 有视觉拓扑干扰；
- 有辅助构造子集；
- 有 difficulty axis。

建议先做：

- 500-1000 道 pilot；
- 5K-10K paper version；
- 覆盖 length、area、angle、ratio、shadow area、construction-required。

### 9.2 第二阶段：GeoSym-Debug

目标：在 GeoSym-Hard 上增加诊断协议。

核心指标：

- Answer Accuracy；
- Grounding Accuracy；
- Fact Validity；
- Step Validity；
- Pseudo-Reasoning Rate；
- Minimal Repair Level；
- Repair@Topology / Repair@Theorem / Repair@Construction。

这比单纯 leaderboard 更有论文价值。

### 9.3 第三阶段：Solver-in-the-loop repair

在 fact graph 稳定后，再做：

- 模型提出辅助线；
- solver 验证 construction 是否 valid；
- solver 判断是否产生 useful facts；
- 只把 useful construction 加回题目；
- 比较 direct / random / model-only / solver-filtered。

这可以成为 GeoSym-Debug 的 method extension。

## 10. 真实图片解析后续怎么放

真实图片解析不应该丢掉，但要重新定位。

合理定位是：

1. 半自动标注工具：帮助人工把真实题转成 GeoSym formal construction。
2. parser 诊断 benchmark：单独评估 VLM 能不能从图中抽出 objects / relations。
3. 外部小样本验证：选择少量真实题，人工构建 formal construction，用来证明 GeoSym-Debug 协议能迁移。
4. 长期 data adapter：等 construction kernel、fact solver、theorem-lite solver 稳定后，再扩大真实图转换。

短期不建议继续把大量时间花在 Geometry3K 全自动解析通过率上。

## 11. 当前代码资产如何保留

已经写的真实解析代码不是完全浪费，它可以保留为：

- parser research sandbox；
- VLM extraction baseline；
- evidence graph / task packet 设计参考；
- future real-data adapter；
- frontend 脚本启动器；
- batch report / triage 工具。

但主线开发应从 `geosym/parsing/hybrid/` 暂时切回：

```text
geosym/formal/
geosym/core/
geosym/eval/
scripts/tools/
```

先把 synthetic verified benchmark 做扎实。

## 12. 下一步最小可行开发计划

### Week 1：formal construction 和 fact store

1. 新增 `geosym/formal/`。
2. 定义 construction op schema。
3. 实现 construction executor。
4. 实现 coordinate fact store。
5. 支持 length、angle、area、parallel、perpendicular、point_on_line。
6. 写 20 个 synthetic smoke cases。

### Week 2：接旧 GeoSym generator

1. 从旧 builder 输出转成 `GeoSymProblemState`。
2. 将已有 points / lines / arcs / entities 统一进新 schema。
3. 对每个样本跑 fact closure。
4. 自动选择 target 和 answer。
5. 导出 500 条 pilot JSONL。

### Week 3：GeoSym-Hard pilot

1. 设计 difficulty axis。
2. 生成 length / area / angle / ratio / shadow subsets。
3. 渲染图像。
4. 跑 3-5 个模型做初步 benchmark。
5. 统计 final accuracy 和 fact-level failure。

### Week 4：GeoSym-Debug

1. 模型输出解析成 object refs、relation claims、calculation claims。
2. 和 fact store 对齐。
3. 输出 failure attribution。
4. 做 counterfactual repair view：给对象、给拓扑、给关键 fact、给辅助构造。

## 13. 最重要的复盘结论

当前失败不是“VLM 不够强”这么简单，而是我们把最难的部分放在了最前面：真实图片到可验证数学状态。

GeoSym 的核心竞争力不在这里。它的核心竞争力是：**自己生成一个完全知道真值的几何世界，并用 solver 把这个世界变成可验证、可诊断、可干预的 benchmark。**

所以接下来应该先把 GeoSym 自己的数据做成真正强的可验证 benchmark，再考虑把真实数据接进来。

更短地说：

```text
先做可靠的 synthetic verified geometry world，
再做 real-image-to-world adapter。
```

