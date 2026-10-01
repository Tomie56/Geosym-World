# Geosym-World 代码诊断与重构计划

日期：2026-09-16  
状态：**只读诊断，等待审阅；不是已经执行的重构。**  
代码根：/data/Geosym127K/Geosym-World  
审查基线：HEAD 2dde581759d54a9374d29417d13cd3ecf1d959a2，加当前工作区内容。

## 1. 结论与审查边界

当前最值得重构的是 **Parser 的多套状态、编排、准入和重建路径**。它们没有形成单一的数据流：一些模块认真做了验证，但下游仍使用另一份对象或另一套 builder；新提示词已经改变输出协议，旧合并器、修补器和审计器还在按照旧协议工作。这比单纯“文件太长”更严重。

不建议整体推倒重写，也不建议按“引用少”批量删除。正式几何内核、构造轨迹、事实作用域、区域计算、合成与渲染已有可复用基础；GeoClaw 也正在依赖其中一部分。应先让这些能力被一条清晰的主流程真正使用，再下线重复实现。

本次执行边界：

- 全量读取并 AST 扫描 src 下 **227 个 Python 文件、79,147 行**，统计定义、导入依赖、长函数、重复函数体和低引用候选。
- 同时静态扫描 tests 的 41 个 Python 文件、scripts 的 1 个脚本、examples 的 5 个文件；查看项目配置、公开入口、关键文档和现存失败产物。
- 对核心调用链和风险模块进行了正文检查；附录列出全部源码文件。**全量静态覆盖不等于逐行验证全部算法，也不等于运行覆盖率。**
- GeoClaw 仅检查对 Geosym-World 的导入边界，没有审查或修改其内部实现。
- 未修改源代码、测试、配置和已有产物，未运行测试、模型调用或生成任务，未删除、提交或推送。唯一新增文件是本报告。
- 审查时已有 **30 个修改文件、11 个未跟踪文件**。结论包含这些尚未提交的内容，不能只以 HEAD 复现。

### 1.1 规模分布

行数包含注释和空行，反映维护体量，不代表质量评分。

| 子系统 | Python 文件 | 行数 | 初步判断 |
|---|---:|---:|---|
| parser | 90 | 33,665 | 占源码约 42.5%，首要收敛对象 |
| legacy + legacy_tools | 13 | 12,301 | 独立旧管线，需隔离并做能力迁移清单 |
| schema | 22 | 5,007 | 有可靠正式契约，也有并行旧契约和薄别名 |
| synthesis | 14 | 4,337 | 主链较清楚，目标驱动规划尚不完整 |
| tools | 21 | 3,840 | 命令入口多，混有业务逻辑与旧审阅工具 |
| solver | 3 | 3,790 | 规则集中、私有接口耦合较高 |
| benchmark | 14 | 3,761 | 有价值，但三种 Parser 评测口径需标明用途 |
| kernel | 15 | 3,426 | 正式 2D/基础 3D 与独立 CAD 流程混放 |
| rendering | 15 | 3,412 | 公共 scene 已有，布局度量与后端仍有分叉 |
| verifier | 5 | 2,738 | 应成为主流程的唯一准入依据 |
| datasets | 9 | 1,545 | 下载与存储布局混入大量 Parser 阶段知识 |
| construction | 4 | 1,243 | 核心保留，复用操作注册和事务边界 |
| 包根与 errors | 2 | 82 | 根包过度聚合导入 |

另有测试约 15,473 行、572 个静态 test 函数；参数化后的实际用例数不同。本轮没有重新执行，不能据此宣称当前工作区测试通过。

## 2. 当前实际流程与主要断点

### 2.1 合成主链

~~~text
templates / sampler
  -> enhancer（选择操作、preflight、逐轮追加）
  -> rejector（执行、验证、求解）
  -> split 分组
  -> exporter（再次执行、验证、求解）
  -> FormalGeometryState / facts / target / derivation
  -> RenderScene -> SVG 与 PNG
  -> record / QA / GT / Parser public inputs + private references
~~~

这条主链具备明确的正式状态。主要问题是重复计算、任务规划与事后审计混用，以及难度、split 名称可能强于实际保证，不是没有合成能力。

### 2.2 Typed / Verified Parser 主链

入口集中在 [workflow.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/workflow.py:61)、[parse_reconstruct_v0.py](/data/Geosym127K/Geosym-World/src/geosym_world/tools/parse_reconstruct_v0.py:171)：

~~~text
ParserInput（image + question）
  -> typed visual / external prediction / fused parser
  -> ParseGraph proposal
  -> PixelTopologyVerifier
  -> ParseGraphAdmission
  -> ParseGraphReconstructor
  -> source redraw / semantic redraw / exact resynthesis
~~~

它的职责边界相对清楚，适合选作收敛骨架；**这不意味着现有检测器、重建器已经足够准确**。视觉融合、对象身份解析和真实图验证仍需要实际样例验收。

### 2.3 Hybrid Parser 主链

入口：[hybrid_parse.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/pipeline/hybrid_parse.py:177)。

~~~text
VLM triage
  -> OpenCV candidates / text focus / overlay
  -> script observations / evidence graph / text claims / evidence board
  -> VLM relation + 条件性 rescue / binding / quantity / instruction
  -> adapters / validation / canonicalization
  -> stage repair / relation draft repair / grounding repair / instruction repair
  -> 再次 validation / repair / coverage / adherence
  -> RelationGrounder -> FormalGraphBuilder
       +-> ParseGraph + semantic ParseGraph
       |    -> PixelTopologyVerifier + ParseGraphAdmission（保存结果）
       |    -> ClauseGraph / ConstraintState / completion plan
       +-> instruction planner
            -> RealDataAdapter -> CanonicalBuilder -> GeoSymProblemState
            -> GeoSymVerifier -> SketchRenderer / scaffold / review
~~~

关键问题：**严格 ParseGraph admission 的结果已经算出，却不是旧 canonical 重建链唯一的输入。** 结果文件丰富，但“到底哪份结构是可信状态”仍不明确。

### 2.4 仍存在的其他路径

- RealParsePipeline：旧 extractor -> normalizer -> GeoSymProblemState -> verifier/sketch。它接受 solution、answer 等字段，不能作为公开 image-only Parser 的等价入口。
- CleanDiagramParserV0：较早的图像 baseline，有独立 CLI 和 evaluator。
- SyntheticParserV0：结构化合成观测/debug 通道，不能冒充真实图片 Parser。
- parser/v1：ObservationGraph、ClauseGraph、ConstraintState、materializer、solver。部分编译/分析已接入 Hybrid；多项 admission/materialization 能力主要由测试和公开 API 使用。
- parser/constraint_completion.py：另有约束补全和 SSS 分支构造路径。
- kernel/geometry3d/pipeline.py：独立 scene/QA/GT/CAD/Blender 流程，不是正式合成主链的统一 3D 后端。

“有多个 provider”是合理扩展；“多个 provider 后面各自再造准入、状态、重建和成功标准”是目前应解决的结构问题。

## 3. 已确认的高优先级问题

以下均以当前代码为依据。严重度 P0 表示可能改变语义或可信边界，P1 表示主要结构债务，P2 表示可后置整理；不是线上故障等级。

### F01 · P0：对象清单在阶段合并时丢失

证据：

- [semantic_extractor.py:1430](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/semantic_extractor.py:1430) 的通用 relation prompt 明确要求 objects。
- [同文件:1127](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/semantic_extractor.py:1127) 的 _merge_staged_outputs 只拷贝 bindings、relations、quantities 等，没有 objects。
- [parse_graph_compiler.py:85](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/parse_graph_compiler.py:85) 主要从关系、数量、target 和绑定解析对象，也没有完整消费新的对象清单。

影响：模型即使识别出全部点和边，孤立于关系描述之外的对象仍可能丢失。之后再靠字母长度推断对象，是在补偿上游协议断裂。

重构要求：先确定唯一 proposal 契约，明确 objects 是正式字段；adapter 只能规范格式，不能默默删对象。协议变更必须带跨阶段 fixture。

### F02 · P0：两套 grounding/admission 并行，可信结果没有统一支配重建

证据：

- [hybrid_parse.py:413](/data/Geosym127K/Geosym-World/src/geosym_world/parser/pipeline/hybrid_parse.py:413) 先用 RelationGrounder/FormalGraphBuilder 创建 formal。
- [同文件:443](/data/Geosym127K/Geosym-World/src/geosym_world/parser/pipeline/hybrid_parse.py:443) 才计算严格 ParseGraphAdmission 并挂入 formal。
- [同文件:1120](/data/Geosym127K/Geosym-World/src/geosym_world/parser/pipeline/hybrid_parse.py:1120) 的绘图状态仍来自旧 adapter/constraint_graph。
- [formalizer.py:31](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/formalizer.py:31) 合并 accepted 与 uncertain 关系用于分类和对象提取。

这不等于所有 uncertain 已被写成持久 fact；但它们可以参与旧 builder 的候选构建，而严格 admission 对该分支不是统一约束。

重构要求：提议图、证据图可以包含不确定信息；只有 admitted 结构能进入可信状态物化。候选布局可以展示，但必须独立标识，不能影响 admitted 状态。

### F03 · P0：视觉 ID 存在与对象身份正确之间仍有缺口

[grounder.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/grounder.py:1) 中，关系通过 binding policy、模型给 grounded=True 且有 visual_ids 时可进入 accepted。

[visual_binding_policy.py:88](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/visual_binding_policy.py:88) 主要检查 ID、类型、数量；新增 parallel 检查仅在两条可用线且方向可取时比较方向差。

仍不能保证：

- 名称 S 对应的是那个候选点；
- PR 对应的是那条候选边；
- S∈PR 的点和线确实是这两个数学对象；
- 两条平行候选边绑定到了关系声明中的同一对边。

重构要求：先做稳定的 name/object -> observation binding，再验证关系。VLM 的 grounded 字段只能是自报信息，不作为 verifier 结论。缺失空间支持要明确标为未验证，而非通过。

### F04 · P0：CanonicalBuilder 含有任意布局，与“原图重建”目标不一致

[canonical_builder.py:156](/data/Geosym127K/Geosym-World/src/geosym_world/parser/adapters/canonical_builder.py:156) 使用图遍历布局：连通分量横移、缺长度默认 1、方向候选选择、孤立点绕圆放置。

这可以用于普通图示意布局，不能证明恢复原图，也不能证明满足完整几何约束。当前它又位于 Hybrid 的 canonical sketch 路径，容易造成“两条水平线也像是重建结果”的语义混淆。

处理：从可信重建主链退出。若保留，只能命名为非验证 diagram layout，并且不得从其坐标提取 source facts。正式物化统一经过约束验证和 FormalGeometryState。

### F05 · P0：能否画图被错误绑定到 theorem solver 状态

[formalizer.py:10](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/formalizer.py:10) 把 triangle、incidence、circle 等也分进 THEOREM_TYPES；[hybrid_parse.py:1132](/data/Geosym127K/Geosym-World/src/geosym_world/parser/pipeline/hybrid_parse.py:1132) 又要求没有 requires_theorem_solver 才画完整 canonical 图。

对于三角形和边上点，结构可以被恢复或给出一个合法 realization，并不需要先解题。反过来，求出了 target 也不能证明图形提取完整。

应分别报告：拓扑提取、约束一致性、realization、解题、source redraw。不要让一个 theorem_only/coordinate_buildable 分类同时决定这五件事。

### F06 · P1：新通用提示词与旧流程契约并不一致

当前第一阶段提示词已经缩短，**不能再把“旧提示词约 4.6 万字符”描述为所有当前首阶段请求的现状**。真正残留的问题是：

- 新首阶段只要求通用对象与关系。
- 后续 instruction task packet、response template、candidate decisions、builder mapping 等仍在工作。
- [hybrid_parse.py:970](/data/Geosym127K/Geosym-World/src/geosym_world/parser/pipeline/hybrid_parse.py:970) 还审计模型是否覆盖脚本候选、构造映射和任务模板。
- [semantic_extractor.py:1625](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/semantic_extractor.py:1625) 的 stage="relation" 上下文裁剪分支已不在当前 _relation_prompt 的调用路径上。

这会产生大量与新任务无关的“缺字段”诊断，推动下一轮修补，形成复杂度循环。

处理：先删去主链对旧 instruction 模板协议的依赖，而不是继续往通用提示词里补旧内部字段。是否支持工具构造动作由确定性 compiler 判断。

### F07 · P1：报告与存储结构主导了运行时设计

- Hybrid 文件 2,674 行。
- _process_input 约 310 行。
- _build_report 从第 744 行开始，约 **738 行**，且含构建、验证、绘图、judge，不只是格式化报告。
- [datasets/layout.py](/data/Geosym127K/Geosym-World/src/geosym_world/datasets/layout.py:1) 定义 evidence board、task packet、semantic raw/repaired、build plan 等大量阶段目录。

问题不是“诊断不需要”，而是诊断模块反过来要求主链维持多份冗余状态。每修一个分支，都要调整多套文件与 counters。

处理：阶段函数返回固定结果对象；编排器不写几十类文件；单独 artifact writer 按 summary/review/debug 级别序列化。报告不能执行几何构建。

### F08 · P0：render gate 有字段读取错误，且数量门槛不是结构完整性证明

[parse_graph_compiler.py:159](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/parse_graph_compiler.py:159) 将 unresolved_objects 放在 compile audit；[hybrid_parse.py:1615](/data/Geosym127K/Geosym-World/src/geosym_world/parser/pipeline/hybrid_parse.py:1615) 却从 parse_graph 顶层读取，因此该检查不能按设计消费编译审计。

此外，formal line 数量、任意 named binding 是否存在，不能证明端点、连通性、闭合区域和标签身份一致。门槛可挡住空图，不能证明正确重建。

处理：统一 audit 字段归属；验证缺失对象、悬空引用、端点关联、局部 incidence、边界闭合和绑定覆盖。数量指标留作诊断，不充当语义证书。

### F09 · P1：运行成功、验证成功、重建成功的口径混用

[hybrid_parse.py:1241](/data/Geosym127K/Geosym-World/src/geosym_world/parser/pipeline/hybrid_parse.py:1241) 在 verification.valid 外又拼接 image alignment、render gate、旧 accepted_relations，得到另一套 acceptance policy。

本次只读取既有产物，得到：

| 已有运行 | 输入数 | records_succeeded | records_failed | 最终 reject_or_repair | 报告指向且实际存在的 canonical sketch |
|---|---:|---:|---:|---:|---:|
| geometry3k_short_general | 6 | 6 | 0 | 6 | 0 |
| synthetic_short_general_v2 | 4 | 3 | 1 | 4 | 0 |

来源：[Geometry3K manifest](/data/Geosym127K/Geosym-World/artifacts/parser/evaluations/pipeline_revision_20260916/geometry3k_parsed_short/geometry3k_short_general/run_manifest.json)、[synthetic manifest](/data/Geosym127K/Geosym-World/artifacts/parser/evaluations/pipeline_revision_20260916/synthetic_parsed_short_v2/synthetic_short_general_v2/run_manifest.json)，以及对应 reports。

这些数字只说明调用完成与当前策略判定，**不是独立真值评测的准确率，也不是整个数据集成功率**。0 张 canonical sketch 不代表没有 overlay/review 图片。

处理：固定分开 request completed、schema valid、proposal coverage、admitted coverage、realization valid、source redraw、target solved；每个指标记录分母。

### F10 · P1：Parser 存在多套模型传输与表达式入口

模型调用散在：

- parser/openrouter.py；
- parser/hybrid/semantic_extractor.py；
- parser/relation_extractor.py；
- parser/verification/llm_judge.py。

超时、重试、模型默认值、JSON 抽取和审计写法各自维护。当前 Hybrid 也有重试/补救分支，不能继续叠加默认恢复逻辑。

表达式方面，严格 ParseGraph 使用受限 MathExpression，而 [problem_normalize.py:45](/data/Geosym127K/Geosym-World/src/geosym_world/schema/problem_normalize.py:45) 仍接受字符串进入 sympify 并宽泛捕获异常。外部模型/文本边界不宜保留这种不一致；本次未做安全利用验证。

处理：一个小型传输实现 + 明确 provider 配置；提取器不兼任传输层。所有外部表达式走受限解析；内部可信 SymPy 对象仍可复用。不要把它扩成通用 Agent 框架。

### F11 · P1：Clause v1 是部分接线，不应再扩成第三条生产链

[parser/v1/analyzer.py:78](/data/Geosym127K/Geosym-World/src/geosym_world/parser/v1/analyzer.py:78) 已被 Hybrid 用来生成 clause/constraint 附件；admit_clauses、model_payload、materializer、solver、facts 在本仓库的非测试业务调用较少，主要是重导出、彼此包装和测试。

此外，realize_exact_points 生成坐标 witness，_classify 在一个 branch 且无 free_parameters 时给 WELL_CONSTRAINED；它明确是结构分析，没有验证所有关系的满足性，也没有证明解的唯一性。

处理：保留“约束与 realization 分离”的设计，收敛为正式 ParseGraph 后的一个 IR/求解组件。若当前没有消费方，先冻结其独立公开模型协议。状态名称必须表达证据级别，不能把“提供了一组坐标”当作“数学唯一解”。

### F12 · P1：约束补全、形式重建、Clause 物化有重叠

涉及：

- parser/reconstruction.py：2,315 行，含精确、像素 witness、交点/派生点/距离分支编译。
- parser/constraint_completion.py：831 行，另有 midpoint、垂足、比例点、SSS 分支等 trace 构建。
- parser/v1/materializer.py：单一 materialize_clause_graph 约 439 行。
- parser/adapters/canonical_builder.py：旧坐标布局。

应有三种不同的**策略结果**，而不是四套重复的对象创建实现：

1. 观测布局恢复：保留原图相对位置，像素不升级为精确数学事实。
2. 约束 realization：找到满足已接纳约束的一组几何状态；可能多解或连续自由。
3. seed completion：显式增加假设或采样参数，生成新题，不声称恢复原题。

共同的对象/关系到 ConstructionOp 编译应只保留一份。策略差异留在选择约束、分支、参数和事实作用域中。

## 4. 合成、内核、渲染等子系统诊断

### F13 · P1：合成依赖图仍偏事后审计，不能等同目标驱动规划

[enhancer.py:62](/data/Geosym127K/Geosym-World/src/geosym_world/synthesis/enhancer.py:62) 逐轮选择、追加操作，再调用 build_task_dependency_graph。

[task_graph.py:23](/data/Geosym127K/Geosym-World/src/geosym_world/synthesis/task_graph.py:23) 的 live roots 除 target 还含 public_given 和 intended_intersection；midpoint、垂足等操作默认可成为 public_given。

因此，“没有 dead_enhancement_steps”仅说明符合这套活跃性定义，**不保证每一步参与目标推导**。把新增操作标为公开条件也不证明其解题必要性。

处理：保留依赖审计；另将规划放在执行前，以 target 的证明/计算依赖为主。区分 target-required、intentional distractor、display-only，不用一个 live 标志混合。

### F14 · P1：筛选、导出重复执行和求解；并行主要位于导出

[filters.py:45](/data/Geosym127K/Geosym-World/src/geosym_world/synthesis/filters.py:45) 与 [exporter.py:65](/data/Geosym127K/Geosym-World/src/geosym_world/synthesis/exporter.py:65) 都执行 trace、验证 state、求 facts；enhancer 在候选步骤中还重建当前 trace。

这是确定存在的重复工作；实际性能占比本次未测。不能直接宣称会加速几倍。

处理：阶段之间传递绑定 spec checksum、solver profile 和 verifier version 的已验证构建结果；输入变更时失效。区分生成/验证 workers 与渲染 workers，避免用户误解 --workers 能并行整个过程。独立复核策略可保留，但不能无意重复。

### F15 · P1：difficulty 与 topology split 的保证要收窄

- [difficulty.py:65](/data/Geosym127K/Geosym-World/src/geosym_world/synthesis/difficulty.py:65) 仍混合人工操作权重、诊断轴、trace 深度和可选 derivation/solution 信息。构造深度不是证明深度。
- [splits.py:23](/data/Geosym127K/Geosym-World/src/geosym_world/synthesis/splits.py:23) 的 topology_key 使用 family、操作名序列、target type、diagnostic axes，不是真正几何图同构规范化。

该 split 能提供一种确定性分组，但可能过度合并不同依赖图，也不能保证跨 family 的相同结构隔离。

处理：先准确命名现有 signature 保证；逐步引入忽略参数与点名的依赖图签名，并把 lineage 与结构分组共同审计。难度分成构造复杂度、有效推导深度、分支歧义，保留单项指标。

### F16 · P1：solver 大单体与增量求解器私有调用耦合

[closure.py](/data/Geosym127K/Geosym-World/src/geosym_world/solver/closure.py:1) 3,385 行；[incremental.py:137](/data/Geosym127K/Geosym-World/src/geosym_world/solver/incremental.py:137) 直接调用 _metric_facts、_construction_relation_facts、_semantic_area_facts 等私有方法。

处理：按度量、拓扑、面积/周长、theorem-lite、target 拆规则组；规则声明输入对象/事实依赖，全量与增量共享规则入口。避免为了拆文件再加一层庞大插件系统。

保留现有 FactScope、provenance、依赖失效与 full-equivalence 检查；这些是可靠性基础。当前代码已有局部重算和明确 full_recompute 分支，不能简单说“增量是假实现”。

### F17 · P1：独立 3D/CAD 管线位于 kernel 内，分层不当

kernel/geometry3d/pipeline.py 包含写盘、QA、benchmark 与渲染调度；renderer.py 包含 Blender 脚本生成；CAD 组件还有独立可选依赖和 scene 字典。

本仓库搜索没有发现该独立 pipeline 被正式 synthesis/tools/tests 主链使用；仍有 package export，因此只能判为**本仓库低集成度**，不能断言没有外部调用。

处理：正式 3D 点、平面、多面体、截面和体积能力保留；独立 CAD/Blender 链路作为显式可选实验组件隔离，迁移有价值模板到统一 state/record 时逐项验收。不要把基础 3D 一并删掉。

### F18 · P1：SVG/PNG 的公共 scene 不等于最终像素一致

PNG 使用 OpenCV/Pillow，SVG 使用自身绘制代码；它们共享 scene 与 position_labels，但：

- _clip_segment 有重复实现。
- PNG 使用 DejaVu 16，SVG 使用 Inter/Arial 15。
- [label_layout.py:25](/data/Geosym127K/Geosym-World/src/geosym_world/rendering/label_layout.py:25) 按固定顺序贪心放置，宽度估算为 10.5×字符数，没有按实际字体测量。
- [regression.py:16](/data/Geosym127K/Geosym-World/src/geosym_world/rendering/regression.py:16) 分别记录 SVG 与 PNG hash，并没有光栅化 SVG 后比较二者。

所以已有视觉回归能力，但“严格一致”“全局碰撞优化”应作为后续目标而非现状保证。

处理：共享一次布局结果、真实文字度量、裁剪与标记几何；确定统一字体/字号。可以继续保留两个后端，是否改为 SVG 转 PNG 要单独评估字体、依赖和性能，不在本次直接决定。

### F19 · P1：旧工具兼容层仍是一整套第二引擎

legacy_tools 约 11,505 行，包含自己的模板、增强、交点、画图、阴影、GT、QA 和答案处理；legacy/pipeline 继续完整调度。

它不是“无用代码”：有 run_legacy_synthesis CLI、文档和兼容测试，而且用户此前明确要求保留原 tools 能力。不能用新引擎有同名能力就认定已实现等价替代。

处理：冻结旧链新增开发，建立“已迁移 / 待迁移 / 不再支持”矩阵；将独有阴影、题型等能力迁入正式实现后，再按条目退役。兼容 import 与真实能力覆盖分开验收。

### F20 · P2：表达层、评测层和包导入边界仍可精简

- GeoSymProblemState 与 FormalGeometryState 都被称为 canonical；前者 dict 风格、后者严格正式状态，职责需要命名区分。
- benchmark/parser_evaluator、parser_v0_evaluator、parse_graph_evaluator 对不同表示进行评测，分数不能直接横向比较。
- 根 __init__.py 聚合 parser、benchmark、synthesis 等；schema/__init__.py 还导入 verifier 和 construction 相关能力，形成包级双向依赖。
- rendering/quality.py 导入 label_layout/layout 私有 helper。
- datasets/layout.py 默认 artifact 根依赖源码相对位置，源码运行合理，但安装到其他位置时产物目录语义应显式指定。

处理：收紧公开 API；契约定义与执行/验证分开；用户文档只给一个推荐 Parser CLI、一个推荐合成 CLI，旧入口明确标 deprecated/debug。不要仅为统一而把所有 evaluator 重写；以 adapter 汇入统一报告字段。

## 5. 模块处置清单

### 5.1 高复杂、重复编排：退出主链或合并职责

以下是下一轮重构对象，**不是本轮删除清单**。先迁移消费方与测试，再删除实现。

| 模块/组 | 处置建议 | 保留内容 |
|---|---|---|
| parser/pipeline/hybrid_parse.py | 用统一 workflow 接管；去掉多套状态并行驱动 | 输入边界、实际失败诊断、必要产物字段 |
| parser/hybrid/semantic_extractor.py | 拆出纯提取 provider 与传输；停止承担内部构造决策 | 通用图像提示词、原始响应、调用预算 |
| parser/hybrid/instruction_task_packet.py | 从默认视觉提取链退出 | 真正需要的构造映射作为确定性 compiler 表 |
| parser/hybrid/evidence_board.py、evidence_graph.py、text_claims.py | 收敛到一份带 evidence refs 的图；报告只做视图 | provenance、定位信息、冲突证据 |
| parser/hybrid/stage_contract_repair.py、relation_draft_repair.py、grounding_decision_repair.py、instruction_repair.py | 逐步退役语义修补链 | 纯格式规范化与明确错误信息 |
| parser/hybrid/stage_contract.py、template_adherence.py、relation_template_adherence.py、task_packet_coverage.py | 移除对旧 response template 的运行时依赖 | 迁移期旧协议检查 fixture |
| parser/hybrid/build_plan_seed_coverage.py、instruction_coverage.py、candidate_usage.py、board_adjudication.py | 必要指标改成纯诊断函数 | 对象/约束覆盖，而非内部模板填满率 |
| parser/hybrid/build_plan_compiler.py、instruction_planner.py、builder_capabilities.py、builder_mapping_consistency.py | 合为确定性构造编译与能力表 | 内核支持范围、unsupported 原因 |
| parser/hybrid/relation_builder_decision.py、relation_engine_metadata.py、engine_assessment.py、extraction_contract.py | 不再让 VLM 判断内部 builder 能否支持 | 由 compiler 返回的结构化能力结果 |
| parser/hybrid/relation_pass_adapter.py、instruction_pass_adapter.py、relation_object_normalizer.py、semantic_role_canonicalizer.py、semantic_validator.py | 一次严格解码 + 一次规范化；逐项淘汰旧变体 | 明确版本迁移，不能静默改语义 |
| parser/hybrid/formalizer.py、grounder.py、stage_classifier.py | 被统一 typed binding/admission/status 替代 | 用户可理解的失败分类 |
| parser/hybrid/script_observations.py | 与正式视觉 evidence 编译去重 | 有用 detector 观测，不保留重复 builder 提示 |
| parser/hybrid/parse_graph_compiler.py | 作为迁移桥，最终直接产出统一 proposal | 对象引用、类型、来源审计 |
| parser/hybrid/quantity_evidence.py、text_theorem_extractor.py、visual_binding_policy.py | 留作候选能力，融入统一提取/绑定/验证 | 外部证据不得变成已验证事实 |
| parser/pipeline/real_parse.py、relation_extractor.py、constraint_normalizer.py | 旧数据迁移通道冻结；不作为推荐真实图片 Parser | 旧记录导入时明确字段来源 |
| parser/adapters/canonical_builder.py、real_adapter.py、visualization/sketch.py | 可信重建/渲染路径退出旧模型 | 必要历史导入；候选示意图显式降级 |
| parser/reconstruction.py、constraint_completion.py、v1/materializer.py | 共用一份 trace compiler，多种 realization 策略 | exact、witness、多解与 seed 的清晰区分 |

不主张为了文件数好看，把全部 hybrid 内容移动到一个新大文件。目标是让重复表示和重复阶段真正消失。

### 5.2 静态低引用、孤立或仅测试消费的候选

| 候选 | 已观察证据 | 建议与限制 |
|---|---|---|
| datasets/records.py | iter_jsonl 未见 src/tests/scripts/examples/docs 调用 | 删除或真正替代重复 JSONL reader；不要作为第三份 reader 保留 |
| datasets/hf_download.py | 未见仓内调用；有 datasets/hub 两种下载分支 | 若不提供正式数据下载命令可移出核心；删除前确认外部用户 |
| schema/relations.py | GeometryRelation 仅是 Mapping 别名，未见消费 | 可移除；不作为“严格 relation schema”宣传 |
| schema/targets.py、schema/traces.py | 薄重导出，未见仓内消费 | 低优先级；公共 import 兼容成本可能高于省下的几行 |
| kernel/geometry3d/kernel.py::_poly_area_3d | 在源码/测试/文档搜索仅发现定义 | 删除候选；正式 analytic 已有 polygon_area_3d |
| tools/seed_synthesis_from_parser_v0.py::_has_evidence | 运行源码无调用，但有测试直接验证 | 测试可能守着已退出运行路径的 helper；先审查现有 evidence 分类的替代关系 |
| parser/v1/admission.py、facts.py、model_payload.py、materializer.py、solver.py | 有测试和公开导出，主流程仅部分接入 v1 analyzer/compiler | 合并或冻结独立协议；不是可直接删除的死代码 |
| kernel/geometry3d 的 CAD/独立 pipeline | 主要内部调用和导出；未见正式主链/测试消费 | 可选实验组件隔离，保留能力迁移清单 |
| tools/smoke_check.py、render_smoke_check.py、export_pilot.py、render_pilot.py；datasets/pilot_exporter.py、samples.py | 历史 pilot/debug 工具，也有公开入口/测试用途 | 归为 examples/debug；复用正式 exporter，不能全部盲删 |
| CleanDiagramParserV0、SyntheticParserV0、synthetic_debug | 有 CLI、测试或诊断消费 | 可冻结老 baseline；保留必要 ablation/debug，不算死模块 |

误判排除：

- ConstructionBuilder 的大量 _op_* 在文本中可能只出现一次，但通过 getattr 动态分发，**不能删除**。
- tools 模块通常没有源码入边，因为通过 python -m / console scripts 调用。
- __init__ 重导出表示公开 API，不等于真实生产使用率，但也不能忽略外部调用可能。
- 不因为“测试文件多”就删除测试；只清理随废弃路径一起退出的实现细节测试。

### 5.3 确认的重复函数体

AST 检查对至少 12 行的函数体做结构比较，发现以下组。它不是完整重复率，未覆盖改变量名后的近似克隆。

| 实现 A | 实现 B | 建议 |
|---|---|---|
| construction/builder.py::_arc_domain | verifier/trace.py::_arc_domain | 共享纯几何域提取；验证策略仍独立 |
| kernel/geometry2d/intersections.py::_compare_real | regions.py::_compare | 统一精确实数比较与 undecidable 策略 |
| parser/grounding.py::_point_segment_distance | quantity_grounding.py 同名函数 | 共享视觉几何 helper |
| hybrid/instruction_repair.py::_same_value | template_adherence.py::_same_value | 随旧协议退役，避免专门保留空泛 util |
| hybrid/relation_draft_repair.py::_refs_from_item | stage_contract_repair.py 同名函数 | 随阶段收敛删除重复 |
| rendering/png_renderer.py::_clip_segment | svg_renderer.py::_clip_segment | 共享已布局 scene 几何 |
| solver/closure.py::_segments_are_adjacent_polygon_edges | verifier/trace.py 同名函数 | 共享基本谓词；不要共享整个证明结论 |
| tools/evaluate_parse_graph_v0.py::_read_jsonl | materialize_parser_references_v0.py 同名函数 | 一个严格 reader，保留路径/行号错误 |
| parser/evidence_fusion.py::_rewrite_relation / _rewrite_quantity / _rewrite_target | 同文件三份相同结构 | 可共享引用重写，但类型边界要保留 |

### 5.4 明确保留的基础能力

- FormalGeometryState、稳定 ID、checksum、provenance、FactStore/FactScope、trace 与 target 契约。
- kernel/geometry2d 的交点分类、精确/数值判定、区域 Boolean 与边界审计。
- ConstructionBuilder 的操作执行与单操作失败回滚；HypothesisAdmission 的候选状态验证。整条 trace 的原子性与单步原子性应在 API 中区别说明。
- ParseGraph、严格外部输入防泄漏、PixelTopologyVerifier、ParseGraphAdmission。
- typed_v0 中真正有效的 detector、OCR、marker、topology、region 提议能力；按阶段拆分，不能因文件长直接废弃。
- evidence_fusion 的身份匹配、冲突与证据兼容检查，替代较弱的平行实现时逐项复用。
- 正式 RenderScene、annotation policy、scene quality 与 regression 能力。
- 合成的去重、lineage、精确计数、失败分类、public/private Parser 数据隔离。
- ParseGraphEvaluator、source manifests、人工 reference/review/adjudication 契约。这些较重，但服务可信评测，不属于为了修单张图而堆出的补丁。

GeoClaw 当前直接依赖 FormalGeometryState、ConstructionBuilder、FactClosureSolver、TraceVerifier、state_to_render_scene 等；这些是重构兼容边界。

## 6. 推荐目标架构

保持项目定位：Geosym-World 负责几何数据、解析、内核和组件评测，GeoClaw 负责 Agent。不要在本轮再建立一套 Agent orchestration。

### 6.1 一条 Parser 主流程，多个可替换提取来源

~~~mermaid
flowchart TD
    I[图片与可选题目文本] --> E[VLM 提取与视觉观测]
    E --> P[统一 ParseGraph proposal]
    P --> B[对象身份绑定与冲突记录]
    B --> A[统一 admission 与证据检查]
    A --> C[约束表示]
    C --> O[观测布局 source redraw]
    C --> R[约束 realization]
    C --> S[显式 seed completion]
    R --> G[FormalGeometryState 与事实作用域]
    S --> G
    G --> K[构造 求解 验证]
    K --> V[RenderScene 与输出]
    A --> D[纯诊断与审阅导出]
    K --> D
~~~

具体约束：

1. VLM 只提取对象、拓扑、显式关系、数量、区域、目标以及置信度/证据，不填写内部 builder 决策表。
2. OpenCV 提供像素观测、stroke/topology 与空间校验；数值不完备不代表这些信息无用，也不允许从像素猜精确长度。
3. triage 可以提供“疑似约束型/有数值信息”等提示，不能代替数学唯一性判断或在提取对象前删除证据。
4. provider 输出只编译一次；所有对象稳定 ID 与名字绑定保留下来。单个对象没有关系也不丢弃。
5. admission 只有一个定义。用于显示的 hypothesis 与可用于推理的 admitted facts 分开存储。
6. 约束结构本身就是有效 Parser 结果；不要求每个结果都能唯一坐标化，也不要求先解 target 才能画图。
7. “完全解析”表示对源信息的覆盖与正确性；“唯一解”表示数学可识别性；二者独立。
8. seed completion 增加的每个条件、分支选择和参数有独立来源；保留但未执行的 source constraint 明确列出，不能算保约束成功。
9. 从观测坐标得到的面积/角度留在 realization scope，不能冒充题目已知。
10. 两种 reconstruction mode 共享对象物化和验证，不共享未经确认的真假结论。

### 6.2 简化目录职责，不先搬家再想接口

建议方向而非立即创建的新目录：

| 责任 | 应承载内容 | 不应承载内容 |
|---|---|---|
| schema | 数据契约、版本、序列化 | builder、网络、写文件 |
| kernel | 纯几何计算与判定 | QA、dataset export、Blender 调度 |
| construction / solver / verifier | 执行、推导、准入验证 | VLM prompt 和审阅排版 |
| parser | 提取、绑定、约束编译、realization 策略、轻量 workflow | 独立第二几何内核 |
| rendering | 共用 scene、布局与输出后端 | 解题和模型置信度判断 |
| synthesis | 任务规划、候选生成、验证后导出 | 在 exporter 重造任务状态 |
| datasets | 数据来源、输入输出格式、manifest | 每种 Parser 内部阶段的专属目录规则 |
| benchmark | 对真实 reference 的组件指标与诊断 | 把运行结束计为解析成功 |
| tools | 参数解析与调用上述接口 | seed 映射、重建和评测业务实现 |
| legacy / 可选实验组件 | 明确冻结的历史能力 | 默认主链隐式调用 |

### 6.3 统一审阅产物

~~~text
artifacts/parser/runs/<run_id>/
  manifest.json
  metrics.json
  results.jsonl
  json/<sample_id>.json
  images/<sample_id>/
    input.png
    source_redraw.png       # 只有实际存在此种结果才写
    realization.png         # 标明不是唯一原图时保留说明
    review.png
  debug/                    # 显式启用才写逐阶段原始输出
~~~

每条结果直接含 proposal、admission、realization、diagnosis 的摘要与引用；默认不复制相同语义树到十多个 JSON。失败样例仍可展示输入、提取 overlay 和具体缺项，但不生成一张假“成功重绘图”。

raw VLM 响应可保留用于重放；限制为一个明确位置，避免默认把大图 base64 与同份 prompt 重复嵌入多个报告。

已有 artifacts 本轮不清理。以后按 run manifest 与保留清单清理：基线、选定失败样例、人工 reference 与有价值原始输出保留；只有确认可重建的重复中间产物进入删除列表。

## 7. 分阶段执行计划与验收

### R0：固定审计基线与行为契约

工作：

- 记录当前已修改/未跟踪文件，人工区分已验证能力、未完成接线和一次性诊断。
- 选定现有离线小样本：三角形内平行线、圆与弦、阴影/孔洞、无数字证明图、符号数量图、一个基础 3D 合成例。
- 固定 public input、原始模型响应、预期对象/关系与失败分类；reference 不进入 Parser 输入。
- 明确需要保留的 CLI/API，检查 GeoClaw 调用边界。
- 以本报告候选表建立迁移清单，不直接批量 rm。

验收：同一输入的旧结果可追溯；每个拟删除模块都有消费方和替代方案。先保证能比较重构前后，而不是一开始改行为又换模型。

### R1：只修协议断点，收敛状态和身份

工作：

- 定义唯一 proposal 对象字段、关系参数、evidence、confidence、status。
- 保留完整 objects；修正 compile audit 消费位置。
- 统一点名/对象 ID/视觉候选三者的映射；禁止以 grounded=True 自证可信。
- 统一 ParseGraphAdmission 作为可信数据入口。
- 区分 process success 与 semantic/reconstruction success。

验收：

- 固定原始模型响应进入新链，不丢点、不丢边、不静默改关系。
- 故意把 S 绑定到 R、PR 绑定到 PQ 时被明确拦截。
- 未接纳候选不能进入 source facts。
- 没有 question/target 的图仍输出结构；没有唯一数值解不被当成提取失败。

这是最优先的一个开发单元，不应与 CAD/legacy 大迁移合成一次提交。

### R2：统一 workflow，拆除旧协议修补链

工作：

- 用现有 VerifiedParseWorkflow 边界承载统一编排，逐项接入 Hybrid 的有效视觉/VLM能力。
- 将网络请求、纯 proposal 解码、绑定、admission、诊断各自独立。
- 默认链移除 instruction task packet、旧 response template、重复 repair/adherence。
- 旧 Hybrid CLI 如需短期保留，只作显式入口迁移，不保留独立准入语义。
- _build_report 变为纯报告组装；产物写入交给一个 writer。

验收：一个样例只有一个可信 proposal/admission 链；没有默认语义修补循环；调用预算、时间和失败阶段可见；审阅只需打开一个 sample JSON 和一张 review 图。

### R3：统一约束物化和三种绘图语义

工作：

- 从 reconstruction / constraint_completion / v1 materializer 提取公共 ConstructionOp 编译。
- 从可信重建路径移除 CanonicalBuilder 的任意布局。
- 保留观测布局、约束 realization、seed completion 三种用途，使用显式类型而非模糊字符串拼接。
- 用约束验证证明 realization 有效；唯一性/多分支/自由参数单独记录。
- 将渲染条件从 requires_theorem_solver 中解耦。

验收：

- 无数字三角形能输出点边与平行/共线关系，并能给观测布局或一个合法 realization。
- SSS 镜像分支不偷偷选一个称为唯一解。
- 不足约束可作为 seed，但补充条件全部可查。
- 无法验证的图保持 partial/unsupported，并定位缺少的对象或谓词。

### R4：收敛渲染与合成执行成本

工作：

- 共用字体、文本度量、一次布局结果、裁剪与符号几何。
- 保留有意义的 scene 回归，增加必要的最终图像检查。
- 候选通过后把 state/facts/verification 传给 exporter，避免无意重新构建。
- 将任务依赖规划前置；明确额外辅助线是目标必需还是刻意干扰。
- 修正文档对 topology signature、difficulty 和 workers 的描述。

验收：

- 同一 scene 两后端标签、边界、虚线与符号语义相同，字体环境固定。
- task-required 操作都有可追溯的 target 依赖；display-only 不计入有效证明深度。
- 重复执行减少有计数证据；性能提升等实际测量后报告。
- 500 条严格计数只在用户安排的批量验收阶段运行，不在每次重构时重复跑。

### R5：隔离旧路径，整理核心大文件与依赖

工作：

- 依据覆盖矩阵迁移 legacy 独有能力，冻结或退出对应旧模块。
- 独立 CAD/Blender 流程移出纯 kernel 责任范围，保持基础 3D API。
- 拆 solver 规则组与 typed_v0 的 topology/marker/region 编译；减少私有方法跨模块调用。
- 清理确定无调用的 helper、重复 JSONL reader、薄别名和失效测试。
- 限制 __init__ 聚合导入；同步公开 API/CLI 文档。

验收：GeoClaw 必需导入保持有效；旧能力有明确保留/迁移/停止支持表；没有仅改目录名的“伪精简”；新增目录不用于安置无消费方的占位 API。

### R6：统一评测，确认重构收益

先离线重放，之后才少量网络样例，再决定是否做 50/100 条比较。

分别报告：

- schema 合法率、对象/名字绑定/拓扑/数量/区域的 precision 与 recall；
- 已接纳候选正确性与 coverage，避免“全拒绝=安全且优秀”；
- 可满足约束 realization 比例与明确未支持比例；
- source redraw 的结构保真，不用存在 PNG 代替正确性；
- image-only 与 image+text 的配对差异；
- 每样例模型调用数、token、耗时、默认产物数与体积；
- 合成合法率、拒绝原因、有效依赖深度与 split 泄漏检查。

验收不承诺一个未经基线支持的统一成功率。固定小样本中，已支持的简单结构应正确保留；未支持案例诚实失败；相比旧基线，成功不能来自 reference 泄漏或放宽 admission。

## 8. Git 与验证安排

本轮仅出计划，不提交当前脏工作区，不覆盖已有改动。

后续实施时：

1. 每个独立阶段保持小范围 diff；代码搬迁与行为变化尽量分开。
2. 先相关契约/回归测试，再按仓库规范运行完整测试；涉及 GeoClaw 公共接口时检查消费侧。
3. 不把已有“609/610 等测试通过”的历史陈述当作本次证明；重构验收使用实际命令和结果。
4. 确定性 fixture 保护业务不变量，不继续仅用私有 helper 的输出去验证同一个 helper。
5. 只有用户要求才提交；每次只提交同一目的且验证过的变更。
6. artifacts、临时脚本和原始模型运行内容不进入正式提交；需要入库的回归材料压缩成最小合法 fixture。

## 9. 尚未证明、不得据此直接删除或宣称完成的事项

- 低引用分析不是线上调用统计；外部 Python 用户、notebook 和历史离线脚本可能仍依赖公开 API。
- 重复执行对实际耗时的占比未知，本轮没有性能 profile。
- 87 处宽泛异常捕获是静态现象，包含必要回滚与 CLI 边界；不能一概替换或当作 bug 数量。
- Preflight 对新增 segment 的重叠检查使用旧 state 查端点；若某个单操作同时引入新端点和 segment，可能出现引用风险。当前主要调用是否触发需专门用例验证，暂不认定为已复现故障。
- 现有 region Boolean、多解交点和高级 solver 规则没有在本轮逐条重算；建议保留，后续通过对应测试保护。
- 当前 VLM 输出质量本身也可能不足；收敛管线能修正信息丢失和准入错误，不能保证换架构后模型就全对。
- 本报告没有删除任何旧运行或测试结果；产物清理必须另列具体路径与保留理由。

## 附录 A：全量源码静态覆盖清单

以下每个文件均已读取并做 AST 扫描。行数用于定位维护体量；正文列出的关键链路另做了语义检查。没有把这份清单宣称为所有分支的执行覆盖。

<details>
<summary>benchmark（14 个文件）</summary>

| 文件 | 行数 |
|---|---:|
| [benchmark/__init__.py](/data/Geosym127K/Geosym-World/src/geosym_world/benchmark/__init__.py:1) | 100 |
| [benchmark/answer_grader.py](/data/Geosym127K/Geosym-World/src/geosym_world/benchmark/answer_grader.py:1) | 30 |
| [benchmark/diagnosis.py](/data/Geosym127K/Geosym-World/src/geosym_world/benchmark/diagnosis.py:1) | 65 |
| [benchmark/geoparsing_grammar.py](/data/Geosym127K/Geosym-World/src/geosym_world/benchmark/geoparsing_grammar.py:1) | 173 |
| [benchmark/output_parser.py](/data/Geosym127K/Geosym-World/src/geosym_world/benchmark/output_parser.py:1) | 52 |
| [benchmark/parse_graph_evaluator.py](/data/Geosym127K/Geosym-World/src/geosym_world/benchmark/parse_graph_evaluator.py:1) | 680 |
| [benchmark/parse_graph_reference.py](/data/Geosym127K/Geosym-World/src/geosym_world/benchmark/parse_graph_reference.py:1) | 876 |
| [benchmark/parser_annotation.py](/data/Geosym127K/Geosym-World/src/geosym_world/benchmark/parser_annotation.py:1) | 873 |
| [benchmark/parser_dataset.py](/data/Geosym127K/Geosym-World/src/geosym_world/benchmark/parser_dataset.py:1) | 168 |
| [benchmark/parser_evaluator.py](/data/Geosym127K/Geosym-World/src/geosym_world/benchmark/parser_evaluator.py:1) | 183 |
| [benchmark/parser_perturbation.py](/data/Geosym127K/Geosym-World/src/geosym_world/benchmark/parser_perturbation.py:1) | 81 |
| [benchmark/parser_slices.py](/data/Geosym127K/Geosym-World/src/geosym_world/benchmark/parser_slices.py:1) | 64 |
| [benchmark/parser_sources.py](/data/Geosym127K/Geosym-World/src/geosym_world/benchmark/parser_sources.py:1) | 321 |
| [benchmark/parser_v0_evaluator.py](/data/Geosym127K/Geosym-World/src/geosym_world/benchmark/parser_v0_evaluator.py:1) | 95 |

</details>

<details>
<summary>construction（4 个文件）</summary>

| 文件 | 行数 |
|---|---:|
| [construction/__init__.py](/data/Geosym127K/Geosym-World/src/geosym_world/construction/__init__.py:1) | 7 |
| [construction/builder.py](/data/Geosym127K/Geosym-World/src/geosym_world/construction/builder.py:1) | 967 |
| [construction/operations.py](/data/Geosym127K/Geosym-World/src/geosym_world/construction/operations.py:1) | 211 |
| [construction/trace.py](/data/Geosym127K/Geosym-World/src/geosym_world/construction/trace.py:1) | 58 |

</details>

<details>
<summary>datasets（9 个文件）</summary>

| 文件 | 行数 |
|---|---:|
| [datasets/__init__.py](/data/Geosym127K/Geosym-World/src/geosym_world/datasets/__init__.py:1) | 6 |
| [datasets/adapters/__init__.py](/data/Geosym127K/Geosym-World/src/geosym_world/datasets/adapters/__init__.py:1) | 5 |
| [datasets/adapters/legacy.py](/data/Geosym127K/Geosym-World/src/geosym_world/datasets/adapters/legacy.py:1) | 481 |
| [datasets/export.py](/data/Geosym127K/Geosym-World/src/geosym_world/datasets/export.py:1) | 33 |
| [datasets/hf_download.py](/data/Geosym127K/Geosym-World/src/geosym_world/datasets/hf_download.py:1) | 82 |
| [datasets/layout.py](/data/Geosym127K/Geosym-World/src/geosym_world/datasets/layout.py:1) | 218 |
| [datasets/pilot_exporter.py](/data/Geosym127K/Geosym-World/src/geosym_world/datasets/pilot_exporter.py:1) | 161 |
| [datasets/records.py](/data/Geosym127K/Geosym-World/src/geosym_world/datasets/records.py:1) | 15 |
| [datasets/samples.py](/data/Geosym127K/Geosym-World/src/geosym_world/datasets/samples.py:1) | 544 |

</details>

<details>
<summary>kernel（15 个文件）</summary>

| 文件 | 行数 |
|---|---:|
| [kernel/__init__.py](/data/Geosym127K/Geosym-World/src/geosym_world/kernel/__init__.py:1) | 1 |
| [kernel/geometry2d/__init__.py](/data/Geosym127K/Geosym-World/src/geosym_world/kernel/geometry2d/__init__.py:1) | 25 |
| [kernel/geometry2d/analytic.py](/data/Geosym127K/Geosym-World/src/geosym_world/kernel/geometry2d/analytic.py:1) | 376 |
| [kernel/geometry2d/intersections.py](/data/Geosym127K/Geosym-World/src/geosym_world/kernel/geometry2d/intersections.py:1) | 709 |
| [kernel/geometry2d/regions.py](/data/Geosym127K/Geosym-World/src/geosym_world/kernel/geometry2d/regions.py:1) | 781 |
| [kernel/geometry3d/__init__.py](/data/Geosym127K/Geosym-World/src/geosym_world/kernel/geometry3d/__init__.py:1) | 12 |
| [kernel/geometry3d/cad/__init__.py](/data/Geosym127K/Geosym-World/src/geosym_world/kernel/geometry3d/cad/__init__.py:1) | 6 |
| [kernel/geometry3d/cad/hlr.py](/data/Geosym127K/Geosym-World/src/geosym_world/kernel/geometry3d/cad/hlr.py:1) | 123 |
| [kernel/geometry3d/cad/validate.py](/data/Geosym127K/Geosym-World/src/geosym_world/kernel/geometry3d/cad/validate.py:1) | 70 |
| [kernel/geometry3d/kernel.py](/data/Geosym127K/Geosym-World/src/geosym_world/kernel/geometry3d/kernel.py:1) | 497 |
| [kernel/geometry3d/pipeline.py](/data/Geosym127K/Geosym-World/src/geosym_world/kernel/geometry3d/pipeline.py:1) | 202 |
| [kernel/geometry3d/qa.py](/data/Geosym127K/Geosym-World/src/geosym_world/kernel/geometry3d/qa.py:1) | 129 |
| [kernel/geometry3d/renderer.py](/data/Geosym127K/Geosym-World/src/geosym_world/kernel/geometry3d/renderer.py:1) | 446 |
| [kernel/geometry3d/symbolic/__init__.py](/data/Geosym127K/Geosym-World/src/geosym_world/kernel/geometry3d/symbolic/__init__.py:1) | 5 |
| [kernel/geometry3d/symbolic/quantities.py](/data/Geosym127K/Geosym-World/src/geosym_world/kernel/geometry3d/symbolic/quantities.py:1) | 44 |

</details>

<details>
<summary>legacy（3 个文件）</summary>

| 文件 | 行数 |
|---|---:|
| [legacy/__init__.py](/data/Geosym127K/Geosym-World/src/geosym_world/legacy/__init__.py:1) | 65 |
| [legacy/pipeline.py](/data/Geosym127K/Geosym-World/src/geosym_world/legacy/pipeline.py:1) | 651 |
| [legacy/registry.py](/data/Geosym127K/Geosym-World/src/geosym_world/legacy/registry.py:1) | 80 |

</details>

<details>
<summary>legacy_tools（10 个文件）</summary>

| 文件 | 行数 |
|---|---:|
| [legacy_tools/__init__.py](/data/Geosym127K/Geosym-World/src/geosym_world/legacy_tools/__init__.py:1) | 5 |
| [legacy_tools/builder.py](/data/Geosym127K/Geosym-World/src/geosym_world/legacy_tools/builder.py:1) | 2259 |
| [legacy_tools/drawer.py](/data/Geosym127K/Geosym-World/src/geosym_world/legacy_tools/drawer.py:1) | 475 |
| [legacy_tools/gt.py](/data/Geosym127K/Geosym-World/src/geosym_world/legacy_tools/gt.py:1) | 1212 |
| [legacy_tools/qa_template.py](/data/Geosym127K/Geosym-World/src/geosym_world/legacy_tools/qa_template.py:1) | 1224 |
| [legacy_tools/region.py](/data/Geosym127K/Geosym-World/src/geosym_world/legacy_tools/region.py:1) | 412 |
| [legacy_tools/shader.py](/data/Geosym127K/Geosym-World/src/geosym_world/legacy_tools/shader.py:1) | 1867 |
| [legacy_tools/shaders.py](/data/Geosym127K/Geosym-World/src/geosym_world/legacy_tools/shaders.py:1) | 162 |
| [legacy_tools/template.py](/data/Geosym127K/Geosym-World/src/geosym_world/legacy_tools/template.py:1) | 3397 |
| [legacy_tools/utils.py](/data/Geosym127K/Geosym-World/src/geosym_world/legacy_tools/utils.py:1) | 492 |

</details>

<details>
<summary>package root（2 个文件）</summary>

| 文件 | 行数 |
|---|---:|
| [__init__.py](/data/Geosym127K/Geosym-World/src/geosym_world/__init__.py:1) | 49 |
| [errors.py](/data/Geosym127K/Geosym-World/src/geosym_world/errors.py:1) | 33 |

</details>

<details>
<summary>parser（90 个文件）</summary>

| 文件 | 行数 |
|---|---:|
| [parser/__init__.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/__init__.py:1) | 216 |
| [parser/adapters/__init__.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/adapters/__init__.py:1) | 8 |
| [parser/adapters/canonical_builder.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/adapters/canonical_builder.py:1) | 337 |
| [parser/adapters/hypothesis_record.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/adapters/hypothesis_record.py:1) | 116 |
| [parser/adapters/real_adapter.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/adapters/real_adapter.py:1) | 108 |
| [parser/adapters/synthetic_adapter.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/adapters/synthetic_adapter.py:1) | 113 |
| [parser/clean_v0.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/clean_v0.py:1) | 229 |
| [parser/constraint_completion.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/constraint_completion.py:1) | 831 |
| [parser/constraint_normalizer.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/constraint_normalizer.py:1) | 132 |
| [parser/contracts.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/contracts.py:1) | 129 |
| [parser/evidence_fusion.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/evidence_fusion.py:1) | 1369 |
| [parser/external_v0.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/external_v0.py:1) | 331 |
| [parser/fused_v0.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/fused_v0.py:1) | 127 |
| [parser/geoparsing.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/geoparsing.py:1) | 845 |
| [parser/geoparsing_contracts.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/geoparsing_contracts.py:1) | 282 |
| [parser/grounding.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/grounding.py:1) | 328 |
| [parser/hybrid/__init__.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/__init__.py:1) | 81 |
| [parser/hybrid/board_adjudication.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/board_adjudication.py:1) | 83 |
| [parser/hybrid/build_plan_compiler.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/build_plan_compiler.py:1) | 197 |
| [parser/hybrid/build_plan_seed_coverage.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/build_plan_seed_coverage.py:1) | 64 |
| [parser/hybrid/builder_capabilities.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/builder_capabilities.py:1) | 214 |
| [parser/hybrid/builder_mapping_consistency.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/builder_mapping_consistency.py:1) | 113 |
| [parser/hybrid/candidate_usage.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/candidate_usage.py:1) | 199 |
| [parser/hybrid/engine_assessment.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/engine_assessment.py:1) | 47 |
| [parser/hybrid/evidence_board.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/evidence_board.py:1) | 192 |
| [parser/hybrid/evidence_graph.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/evidence_graph.py:1) | 591 |
| [parser/hybrid/extraction_contract.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/extraction_contract.py:1) | 179 |
| [parser/hybrid/formalizer.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/formalizer.py:1) | 239 |
| [parser/hybrid/grounder.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/grounder.py:1) | 85 |
| [parser/hybrid/grounding_decision_repair.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/grounding_decision_repair.py:1) | 116 |
| [parser/hybrid/instruction_coverage.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/instruction_coverage.py:1) | 79 |
| [parser/hybrid/instruction_pass_adapter.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/instruction_pass_adapter.py:1) | 236 |
| [parser/hybrid/instruction_planner.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/instruction_planner.py:1) | 290 |
| [parser/hybrid/instruction_repair.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/instruction_repair.py:1) | 415 |
| [parser/hybrid/instruction_task_packet.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/instruction_task_packet.py:1) | 1551 |
| [parser/hybrid/parse_graph_compiler.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/parse_graph_compiler.py:1) | 852 |
| [parser/hybrid/quantity_evidence.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/quantity_evidence.py:1) | 203 |
| [parser/hybrid/relation_builder_decision.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/relation_builder_decision.py:1) | 179 |
| [parser/hybrid/relation_draft_repair.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/relation_draft_repair.py:1) | 268 |
| [parser/hybrid/relation_engine_metadata.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/relation_engine_metadata.py:1) | 125 |
| [parser/hybrid/relation_object_normalizer.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/relation_object_normalizer.py:1) | 203 |
| [parser/hybrid/relation_pass_adapter.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/relation_pass_adapter.py:1) | 390 |
| [parser/hybrid/relation_template_adherence.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/relation_template_adherence.py:1) | 205 |
| [parser/hybrid/script_observations.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/script_observations.py:1) | 1161 |
| [parser/hybrid/semantic_extractor.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/semantic_extractor.py:1) | 1811 |
| [parser/hybrid/semantic_role_canonicalizer.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/semantic_role_canonicalizer.py:1) | 336 |
| [parser/hybrid/semantic_validator.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/semantic_validator.py:1) | 619 |
| [parser/hybrid/stage_classifier.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/stage_classifier.py:1) | 80 |
| [parser/hybrid/stage_contract.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/stage_contract.py:1) | 525 |
| [parser/hybrid/stage_contract_repair.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/stage_contract_repair.py:1) | 208 |
| [parser/hybrid/task_packet_coverage.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/task_packet_coverage.py:1) | 179 |
| [parser/hybrid/template_adherence.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/template_adherence.py:1) | 146 |
| [parser/hybrid/text_claims.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/text_claims.py:1) | 75 |
| [parser/hybrid/text_theorem_extractor.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/text_theorem_extractor.py:1) | 86 |
| [parser/hybrid/visual_binding_policy.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/hybrid/visual_binding_policy.py:1) | 189 |
| [parser/ocr.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/ocr.py:1) | 508 |
| [parser/openrouter.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/openrouter.py:1) | 863 |
| [parser/pipeline/__init__.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/pipeline/__init__.py:1) | 6 |
| [parser/pipeline/hybrid_parse.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/pipeline/hybrid_parse.py:1) | 2674 |
| [parser/pipeline/real_parse.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/pipeline/real_parse.py:1) | 243 |
| [parser/quantity_grounding.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/quantity_grounding.py:1) | 236 |
| [parser/question.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/question.py:1) | 251 |
| [parser/reconstruction.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/reconstruction.py:1) | 2315 |
| [parser/redraw.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/redraw.py:1) | 457 |
| [parser/relation_extractor.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/relation_extractor.py:1) | 167 |
| [parser/synthetic_debug.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/synthetic_debug.py:1) | 51 |
| [parser/synthetic_v0.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/synthetic_v0.py:1) | 185 |
| [parser/typed_v0.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/typed_v0.py:1) | 2923 |
| [parser/v1/__init__.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/v1/__init__.py:1) | 48 |
| [parser/v1/admission.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/v1/admission.py:1) | 74 |
| [parser/v1/analyzer.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/v1/analyzer.py:1) | 115 |
| [parser/v1/compiler.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/v1/compiler.py:1) | 130 |
| [parser/v1/contracts.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/v1/contracts.py:1) | 329 |
| [parser/v1/facts.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/v1/facts.py:1) | 50 |
| [parser/v1/materializer.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/v1/materializer.py:1) | 627 |
| [parser/v1/model_payload.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/v1/model_payload.py:1) | 54 |
| [parser/v1/solver.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/v1/solver.py:1) | 29 |
| [parser/verification/__init__.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/verification/__init__.py:1) | 7 |
| [parser/verification/consistency.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/verification/consistency.py:1) | 21 |
| [parser/verification/image_alignment.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/verification/image_alignment.py:1) | 67 |
| [parser/verification/llm_judge.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/verification/llm_judge.py:1) | 198 |
| [parser/vision/__init__.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/vision/__init__.py:1) | 10 |
| [parser/vision/marker_detector.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/vision/marker_detector.py:1) | 791 |
| [parser/vision/primitive_detector.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/vision/primitive_detector.py:1) | 1144 |
| [parser/vision/text_focus.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/vision/text_focus.py:1) | 213 |
| [parser/visualization/__init__.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/visualization/__init__.py:1) | 7 |
| [parser/visualization/review.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/visualization/review.py:1) | 133 |
| [parser/visualization/sketch.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/visualization/sketch.py:1) | 76 |
| [parser/visualization/visual_overlay.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/visualization/visual_overlay.py:1) | 122 |
| [parser/workflow.py](/data/Geosym127K/Geosym-World/src/geosym_world/parser/workflow.py:1) | 229 |

</details>

<details>
<summary>rendering（15 个文件）</summary>

| 文件 | 行数 |
|---|---:|
| [rendering/__init__.py](/data/Geosym127K/Geosym-World/src/geosym_world/rendering/__init__.py:1) | 44 |
| [rendering/adapter.py](/data/Geosym127K/Geosym-World/src/geosym_world/rendering/adapter.py:1) | 1456 |
| [rendering/annotation_policy.py](/data/Geosym127K/Geosym-World/src/geosym_world/rendering/annotation_policy.py:1) | 33 |
| [rendering/label_layout.py](/data/Geosym127K/Geosym-World/src/geosym_world/rendering/label_layout.py:1) | 220 |
| [rendering/layout.py](/data/Geosym127K/Geosym-World/src/geosym_world/rendering/layout.py:1) | 122 |
| [rendering/marker_geometry.py](/data/Geosym127K/Geosym-World/src/geosym_world/rendering/marker_geometry.py:1) | 124 |
| [rendering/math_text.py](/data/Geosym127K/Geosym-World/src/geosym_world/rendering/math_text.py:1) | 71 |
| [rendering/parse_graph_adapter.py](/data/Geosym127K/Geosym-World/src/geosym_world/rendering/parse_graph_adapter.py:1) | 442 |
| [rendering/png_renderer.py](/data/Geosym127K/Geosym-World/src/geosym_world/rendering/png_renderer.py:1) | 285 |
| [rendering/quality.py](/data/Geosym127K/Geosym-World/src/geosym_world/rendering/quality.py:1) | 124 |
| [rendering/regression.py](/data/Geosym127K/Geosym-World/src/geosym_world/rendering/regression.py:1) | 43 |
| [rendering/relation_text.py](/data/Geosym127K/Geosym-World/src/geosym_world/rendering/relation_text.py:1) | 46 |
| [rendering/scene.py](/data/Geosym127K/Geosym-World/src/geosym_world/rendering/scene.py:1) | 94 |
| [rendering/style_policy.py](/data/Geosym127K/Geosym-World/src/geosym_world/rendering/style_policy.py:1) | 51 |
| [rendering/svg_renderer.py](/data/Geosym127K/Geosym-World/src/geosym_world/rendering/svg_renderer.py:1) | 257 |

</details>

<details>
<summary>schema（22 个文件）</summary>

| 文件 | 行数 |
|---|---:|
| [schema/__init__.py](/data/Geosym127K/Geosym-World/src/geosym_world/schema/__init__.py:1) | 120 |
| [schema/canonical.py](/data/Geosym127K/Geosym-World/src/geosym_world/schema/canonical.py:1) | 48 |
| [schema/derivations.py](/data/Geosym127K/Geosym-World/src/geosym_world/schema/derivations.py:1) | 301 |
| [schema/entities.py](/data/Geosym127K/Geosym-World/src/geosym_world/schema/entities.py:1) | 255 |
| [schema/expected.py](/data/Geosym127K/Geosym-World/src/geosym_world/schema/expected.py:1) | 187 |
| [schema/facts.py](/data/Geosym127K/Geosym-World/src/geosym_world/schema/facts.py:1) | 315 |
| [schema/hypotheses.py](/data/Geosym127K/Geosym-World/src/geosym_world/schema/hypotheses.py:1) | 62 |
| [schema/ids.py](/data/Geosym127K/Geosym-World/src/geosym_world/schema/ids.py:1) | 19 |
| [schema/parse_graph.py](/data/Geosym127K/Geosym-World/src/geosym_world/schema/parse_graph.py:1) | 1195 |
| [schema/problem.py](/data/Geosym127K/Geosym-World/src/geosym_world/schema/problem.py:1) | 343 |
| [schema/problem_normalize.py](/data/Geosym127K/Geosym-World/src/geosym_world/schema/problem_normalize.py:1) | 120 |
| [schema/problem_verifier.py](/data/Geosym127K/Geosym-World/src/geosym_world/schema/problem_verifier.py:1) | 252 |
| [schema/provenance.py](/data/Geosym127K/Geosym-World/src/geosym_world/schema/provenance.py:1) | 55 |
| [schema/quantities.py](/data/Geosym127K/Geosym-World/src/geosym_world/schema/quantities.py:1) | 48 |
| [schema/records.py](/data/Geosym127K/Geosym-World/src/geosym_world/schema/records.py:1) | 57 |
| [schema/relation_normalization.py](/data/Geosym127K/Geosym-World/src/geosym_world/schema/relation_normalization.py:1) | 168 |
| [schema/relations.py](/data/Geosym127K/Geosym-World/src/geosym_world/schema/relations.py:1) | 7 |
| [schema/solutions.py](/data/Geosym127K/Geosym-World/src/geosym_world/schema/solutions.py:1) | 161 |
| [schema/state.py](/data/Geosym127K/Geosym-World/src/geosym_world/schema/state.py:1) | 1198 |
| [schema/targets.py](/data/Geosym127K/Geosym-World/src/geosym_world/schema/targets.py:1) | 5 |
| [schema/traces.py](/data/Geosym127K/Geosym-World/src/geosym_world/schema/traces.py:1) | 5 |
| [schema/transitions.py](/data/Geosym127K/Geosym-World/src/geosym_world/schema/transitions.py:1) | 86 |

</details>

<details>
<summary>solver（3 个文件）</summary>

| 文件 | 行数 |
|---|---:|
| [solver/__init__.py](/data/Geosym127K/Geosym-World/src/geosym_world/solver/__init__.py:1) | 6 |
| [solver/closure.py](/data/Geosym127K/Geosym-World/src/geosym_world/solver/closure.py:1) | 3385 |
| [solver/incremental.py](/data/Geosym127K/Geosym-World/src/geosym_world/solver/incremental.py:1) | 399 |

</details>

<details>
<summary>synthesis（14 个文件）</summary>

| 文件 | 行数 |
|---|---:|
| [synthesis/__init__.py](/data/Geosym127K/Geosym-World/src/geosym_world/synthesis/__init__.py:1) | 47 |
| [synthesis/diagnostics.py](/data/Geosym127K/Geosym-World/src/geosym_world/synthesis/diagnostics.py:1) | 152 |
| [synthesis/difficulty.py](/data/Geosym127K/Geosym-World/src/geosym_world/synthesis/difficulty.py:1) | 156 |
| [synthesis/enhancer.py](/data/Geosym127K/Geosym-World/src/geosym_world/synthesis/enhancer.py:1) | 509 |
| [synthesis/exporter.py](/data/Geosym127K/Geosym-World/src/geosym_world/synthesis/exporter.py:1) | 656 |
| [synthesis/filters.py](/data/Geosym127K/Geosym-World/src/geosym_world/synthesis/filters.py:1) | 177 |
| [synthesis/mutators.py](/data/Geosym127K/Geosym-World/src/geosym_world/synthesis/mutators.py:1) | 31 |
| [synthesis/pipeline.py](/data/Geosym127K/Geosym-World/src/geosym_world/synthesis/pipeline.py:1) | 387 |
| [synthesis/preflight.py](/data/Geosym127K/Geosym-World/src/geosym_world/synthesis/preflight.py:1) | 112 |
| [synthesis/sampler.py](/data/Geosym127K/Geosym-World/src/geosym_world/synthesis/sampler.py:1) | 149 |
| [synthesis/splits.py](/data/Geosym127K/Geosym-World/src/geosym_world/synthesis/splits.py:1) | 126 |
| [synthesis/task_contract.py](/data/Geosym127K/Geosym-World/src/geosym_world/synthesis/task_contract.py:1) | 314 |
| [synthesis/task_graph.py](/data/Geosym127K/Geosym-World/src/geosym_world/synthesis/task_graph.py:1) | 106 |
| [synthesis/templates.py](/data/Geosym127K/Geosym-World/src/geosym_world/synthesis/templates.py:1) | 1415 |

</details>

<details>
<summary>tools（21 个文件）</summary>

| 文件 | 行数 |
|---|---:|
| [tools/__init__.py](/data/Geosym127K/Geosym-World/src/geosym_world/tools/__init__.py:1) | 1 |
| [tools/adapt_legacy_json.py](/data/Geosym127K/Geosym-World/src/geosym_world/tools/adapt_legacy_json.py:1) | 93 |
| [tools/audit_geoparsing_grammar_v0.py](/data/Geosym127K/Geosym-World/src/geosym_world/tools/audit_geoparsing_grammar_v0.py:1) | 105 |
| [tools/benchmark_parser_batch_v0.py](/data/Geosym127K/Geosym-World/src/geosym_world/tools/benchmark_parser_batch_v0.py:1) | 513 |
| [tools/diagnose_parser_v0.py](/data/Geosym127K/Geosym-World/src/geosym_world/tools/diagnose_parser_v0.py:1) | 96 |
| [tools/evaluate_parse_graph_v0.py](/data/Geosym127K/Geosym-World/src/geosym_world/tools/evaluate_parse_graph_v0.py:1) | 167 |
| [tools/export_pilot.py](/data/Geosym127K/Geosym-World/src/geosym_world/tools/export_pilot.py:1) | 28 |
| [tools/materialize_parser_references_v0.py](/data/Geosym127K/Geosym-World/src/geosym_world/tools/materialize_parser_references_v0.py:1) | 258 |
| [tools/parse_images_v0.py](/data/Geosym127K/Geosym-World/src/geosym_world/tools/parse_images_v0.py:1) | 53 |
| [tools/parse_openrouter_v0.py](/data/Geosym127K/Geosym-World/src/geosym_world/tools/parse_openrouter_v0.py:1) | 55 |
| [tools/parse_real_data_hybrid_v0.py](/data/Geosym127K/Geosym-World/src/geosym_world/tools/parse_real_data_hybrid_v0.py:1) | 251 |
| [tools/parse_reconstruct_v0.py](/data/Geosym127K/Geosym-World/src/geosym_world/tools/parse_reconstruct_v0.py:1) | 665 |
| [tools/parse_synthetic_v0.py](/data/Geosym127K/Geosym-World/src/geosym_world/tools/parse_synthetic_v0.py:1) | 67 |
| [tools/prepare_parser_inputs_v0.py](/data/Geosym127K/Geosym-World/src/geosym_world/tools/prepare_parser_inputs_v0.py:1) | 135 |
| [tools/render_pilot.py](/data/Geosym127K/Geosym-World/src/geosym_world/tools/render_pilot.py:1) | 46 |
| [tools/render_smoke_check.py](/data/Geosym127K/Geosym-World/src/geosym_world/tools/render_smoke_check.py:1) | 82 |
| [tools/run_legacy_synthesis.py](/data/Geosym127K/Geosym-World/src/geosym_world/tools/run_legacy_synthesis.py:1) | 82 |
| [tools/seed_synthesis_from_parser_v0.py](/data/Geosym127K/Geosym-World/src/geosym_world/tools/seed_synthesis_from_parser_v0.py:1) | 667 |
| [tools/smoke_check.py](/data/Geosym127K/Geosym-World/src/geosym_world/tools/smoke_check.py:1) | 98 |
| [tools/summarize_parser_run_v0.py](/data/Geosym127K/Geosym-World/src/geosym_world/tools/summarize_parser_run_v0.py:1) | 68 |
| [tools/synthesize_records.py](/data/Geosym127K/Geosym-World/src/geosym_world/tools/synthesize_records.py:1) | 310 |

</details>

<details>
<summary>verifier（5 个文件）</summary>

| 文件 | 行数 |
|---|---:|
| [verifier/__init__.py](/data/Geosym127K/Geosym-World/src/geosym_world/verifier/__init__.py:1) | 30 |
| [verifier/admission.py](/data/Geosym127K/Geosym-World/src/geosym_world/verifier/admission.py:1) | 148 |
| [verifier/parse_graph.py](/data/Geosym127K/Geosym-World/src/geosym_world/verifier/parse_graph.py:1) | 600 |
| [verifier/pixel_topology.py](/data/Geosym127K/Geosym-World/src/geosym_world/verifier/pixel_topology.py:1) | 683 |
| [verifier/trace.py](/data/Geosym127K/Geosym-World/src/geosym_world/verifier/trace.py:1) | 1277 |

</details>

