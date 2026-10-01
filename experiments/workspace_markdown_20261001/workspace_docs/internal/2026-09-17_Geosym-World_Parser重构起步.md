# Parser 重构起步记录（2026-09-17）

依据：`docs/reports/2026-09-16_Geosym-World代码诊断与重构计划.md`。本轮只处理 R1 的协议断点；R0 的离线样本基线尚未冻结。

## 现状和边界

- Geosym-World 是独立仓库；开始时已有约 30 个修改文件和 11 个未跟踪文件，均保留。本轮未提交、未删除任何产物，未调用网络模型。
- 第一轮 VLM 提示词已经要求 `objects.points/segments/circles/arcs`，但分阶段合并器此前没有转交该字段；ParseGraph compiler 此前仅从关系推断对象。
- 严格 ParseGraph admission 已存在，但 Hybrid 同时仍由 legacy formalizer / instruction planner 构造结果；admitted graph 尚非唯一可信构建输入。R2 必须解决，不能以当前绘图成功率宣称已解决。

## 本轮改变

- 分阶段输出与无关系 rescue 路径保留首轮对象清单，校验对象字段的基本形状；无问题文本、无关系和 target 的图仍保留候选对象。
- 编译器按显式类型生成完整 proposal；无可定位证据的命名对象保持未接纳，避免将对象清单误认成持久事实。冲突类型直接报错。
- 修正渲染审计从 `parse_graph_audit` 读取未解析对象，而不是从不存在的 graph 字段读取。
- 命名点绑定只接受单个存在的 point 类型视觉候选；歧义、缺失、错类型写入 rejected binding 审计，不静默接受。此检查**不能**确认同类型候选就是正确的 S，尚需独立标签定位与拓扑证据。
- 新增离线单测覆盖清单传递、类型冲突、准入隔离、审计字段和错误视觉候选。

## 后续验收

1. 冻结现有离线图片及原始 VLM 响应，用相同输入比较前后 proposal / admission，区分缺失对象和错误绑定。
2. 加入标签-候选的实际定位及局部 incidence 检查：故意把 S 绑定到 R、PR 绑定到 PQ 时应明确拒绝，不能仅因像素附近有点就通过。
3. 将 Hybrid 重建与事实入口切至同一 ParseGraphAdmission；`semantic_parse_graph` 只能作待验证约束/合成种子，不得冒充 source facts。
4. 再推进报告 R2-R6；大规模网络评测与产物清理另行安排。

## 2026-09-18 增量：SeedRealizer 第一条垂直切片

- 新增 `geosym_world.parser.SeedRealizer`，输入严格要求 `ParseGraphAdmissionReport`，不接受原始 semantic 或 instruction plan。
- 当前支持点、线段、点在线段上、平行、垂直、共线和像素观测位置；无数值时使用确定性的约束种子布局。
- 输出 `SeedRealization`，显式记录 `realization_mode`、`exact=false`、自由参数、满足/未满足约束、来源 checksum 和 renderer-neutral `RenderScene`。
- `HybridParsePipeline` 现在记录 `formal.seed_realization`，成功时额外输出 `*.seed.png`；exact/legacy 的 `RealDataAdapter` 分支暂未替换。
- 新增平行线三角形 fixture；完整测试从 618 增至 623，当前 `/data/miniconda3/envs/geosym/bin/python -m pytest -q` 结果为 `623 passed`。

仍未完成：真实 VLM 语义对象如何获得足够的局部证据并进入 admitted graph；Hybrid 默认 canonical branch 尚未改为 seed-first；观察布局、seed 布局和 exact state 尚未统一成最终报告协议。

## 2026-09-20 增量：单一主流程和状态物化

依据：`docs/reports/2026-09-20_通用几何Parser开发计划.md`。本轮承接工作树中已经进行的 legacy/Hybrid Parser 删除，不恢复旧模块，不调用网络模型，不生成运行产物。

已完成：

- 新增 `ParserPipeline`，唯一顺序为 extractor -> 可选 pixel topology -> admission -> SeedRealizer；外部 prediction 必须与完整 `ParserInput` checksum 匹配。
- `ParserPipelineResult` 保留 prediction、pixel report、admission、realization、verification 和逐候选生命周期；未准入、拒绝、未支持和违反不再通过缺席方式隐藏。
- `SeedRealizer` v2 将成功及部分成功种子统一物化为 `FormalGeometryState`，renderer 只消费 state；匿名点保留稳定 ID 且不强制显示标签。
- 线、射线、线段域保持区分；加入 `point_on_segment`、`point_on_ray` 和 `ordered_points` 的 state contract、静态引用检查及 verifier。
- 非精确 Parser 数值见证显式携带 `verification_mode=numeric_witness`、容差和残差；仅该模式允许容差，普通精确状态继续严格符号验证，实质错误仍拒绝。
- ParseGraph 对平行/垂直、点-支撑对象、点序和共线关系执行严格角色校验；OpenRouter adapter 与 synthetic reference producer 均改为正式角色。
- 增加四维能力表：representation / extraction / realization / verification。VLM extraction 统一标为 `partial`，避免把协议可表达误报为真实提取已验证。

验证：

- Parser 聚焦测试：`149 passed`。
- 最终全量测试：`353 passed in 67.63s`。
- `git diff --check` 通过；未跟踪 `.env`、运行 artifacts、tmp、缓存或真实 API key。

下一步：

1. 继续 P1 人工 ParseGraph 的二维直线闭环，补中点、比例、交点多解和退化/冲突组合的通用测试。
2. 将现有 `constraint_completion.py` 中整图策略逐步拆成可组合构造规则，删除确认不再需要的形状专属策略。
3. P1 稳定后再进入 P2：用同一 `ParserPipeline` 接 VLM，分别评测 extraction 与 realization，不以重绘成功替代结构正确。
4. 圆、弧、区域和 3D 当前只保留协议能力或部分 verifier，generic realization 仍明确 deferred。
