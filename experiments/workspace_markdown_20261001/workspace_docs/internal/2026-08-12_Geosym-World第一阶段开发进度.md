# Geosym-World 第一阶段开发进度

## 本轮完成

- Geometry State Contract：canonical serialization、schema version、state/record checksum、稳定 fact/relation ID、严格实体边界校验、统一错误类型。
- State admission：observation -> proposed hypothesis -> grounding -> verified admission -> admitted state；构造操作具备原子回滚，所有 transition 带 checksum/provenance。
- 2D kernel：line/ray/segment 实际定义域、结构化交点状态、多解选择、共线 overlap、延长线排除、solver 与 verifier 共享交点语义。
- Directed arc kernel：Arc 端点必须精确位于所属圆上，orientation 与可选 central angle 必须匹配真实 sweep；新增 linear-arc、circle-arc、arc-arc 结构化交点和同圆 overlap。builder、static refs、state relation verifier 与 solver 共用定义域语义，后者生成 `point_on_arc` 和独立复验的 `curve_intersection` facts。
- Solver degeneracy fix：圆周角闭包按坐标排除与弦端点重合的 observer，避免不同 ID 同坐标产生零向量异常。
- Relation contract：关系别名与白名单、对称关系排序、parallel/perpendicular/midpoint/intersection exact audit、equal_length/equal_angle/similar/congruent exact residual。
- Region/area：holes、CompositeRegion、sum/difference area closure 和显式区域依赖。
- Region boundary closure：共享 `resolve_region_geometry_2d`，支持点边界、无序线段闭环、孔洞和两个简单非自交多边形（含凹多边形）的 union/intersection/difference 边界；结果可表达单环、孔洞和 `multi_polygon`，solver 记录 `region_boundary` provenance，SVG/PNG 统一消费 rings。
- Region admission/derivation：verifier 拒绝 2D 自交边界、退化边界和外置孔洞；area/perimeter target derivation 显式依赖 `region_boundary` fact ID。
- Construction supervision：记录级 `construction_steps` 保存每步 typed input references、状态前后 checksum、增量对象/关系、target 变化和 provenance；末步 checksum 与 `formal_state_checksum` 对齐。
- Rendering：SVG/PNG 共享 scene；质量报告覆盖标签越界、点/marker/几何线/区域孔洞遮挡，点名和量值标签具备扩展候选位置；垂足、角标、延长线和复杂根式/分式排版有回归保护。
- Visual regression：新增 `geosym.render_regression.v1` 紧凑签名，绑定 scene fingerprint、SVG 字节、PNG RGB、PNG 前景 mask 与场景计数；PNG 支持不落盘的 `render_png_rgb`，fixture 只保存 JSON 摘要而不提交生成图片。
- Representative render matrix：固定 sector、circular segment、equal-circle lens、double-hole 和 annulus 五个 canonical 场景，逐例要求质量检查零错误并匹配 scene/SVG/PNG/mask 签名；fixture 仅保存紧凑 JSON 签名，不保存生成图片。
- Synthesis：lineage-aware split、multi-step trace export、QA/GT/manifest、进度输出和精确计数路径。
- Candidate diagnosis：manifest 记录 accepted/rejected/skipped、稳定拒绝原因和限量样例；验证失败与 family quota 等策略跳过分开统计，accepted record 保存验证摘要。
- Derivation supervision：目标事实子图升级为版本化拓扑步骤，包含规则、输入 fact/reference、深度、provenance、checksum、integrity 和可脱离 FactStore 运行的结构复验；无/多 target、未解析依赖或不完整推导在候选阶段拒绝。
- Task/split contract：正式 record 提供顶层 public givens 和带 checksum/validation 的 task 复杂度摘要；CLI 默认按 answer-free topology key 分组 train/dev/test，参数、坐标、对象名不同但任务拓扑相同的变体不会跨 split；仍可选择 lineage 模式。resume 保留既有 split 并拒绝同 group 冲突，manifest 二次审计 group 唯一性并聚合 target、构造深度、推导深度/规则和结构签名。
- Area benchmark derivation：三角形、矩形、平行四边形、梯形和风筝的面积 target 绑定公开底高/对角线事实；语义面积先校验平行/垂直条件，再与 shoelace 精确面积做 residual admission，冲突公式和未解析 preferred target 均拒绝导出。
- Region partition supervision：新增严格 `partition` 复合区域，校验子区域位于母区域内、内部不重叠且面积完全覆盖；solver 保留 partition residual admission 证据，并新增“垂足分割三角形 -> 两个子面积 -> 总面积”的多步骤任务族。
- Shaded difference supervision：新增语义 `region_difference` contract，要求减数区域完整包含为单孔；外/内矩形面积由公开长宽推导，差值通过 exact boolean residual admission 后成为阴影面积 target，并保留 PNG 透明孔洞回归。
- Concave boundary-notch supervision：`region_difference` 新增严格 `boundary_notch` 模式，仅接纳轴对齐矩形在一个外角形成的缺口；`concave_composite_area` 由四个公开边长推导外矩形和缺口矩形面积，再经 exact residual admission 得到六边形 L 区域目标。隐藏补全点不进入点标注、视觉尺度或隐式交点去重，SVG/PNG 共用同一物化边界。
- Multi-hole region supervision：新增 `region_hole_difference` contract，孔洞必须严格位于外边界内部且互不接触、重叠或包含；区域反序列化按依赖拓扑加载，canonical 映射顺序不影响严格引用。`double_rectangular_hole_area` 由六个公开尺寸推导外框和两个孔洞面积，经 `area_hole_difference_residual` admission 后得到双孔阴影目标；SVG/PNG 共用双孔 rings，并保留像素透明回归。
- Semantic perimeter supervision：新增独立 `perimeter_formula` contract。矩形周长严格要求两个公开长度来自相邻边且四点构成精确矩形，经 `perimeter_formula_residual` admission；多孔区域周长按外边界与全部内边界求和，经 `perimeter_hole_sum_residual` admission。`double_rectangular_hole_perimeter` 与面积任务共用 formal trace，但导出独立 perimeter target 和五层公开量推导。
- Curved annulus supervision：`annulus_region` 现在严格绑定同心内外圆、公共圆心和有序半径；面积与总边界周长分别通过 `annulus_radius_area`、`annulus_radius_perimeter` 从两条公开半径段推导，并由 `annulus_area_residual`、`annulus_perimeter_residual` 对圆对象真值做 exact admission。新增 `annulus_perimeter` 任务族，与面积任务共用 formal trace；large-500 扩展为 19 个 2D families，PNG 回归验证内圆盘保持透明。
- Curved sector supervision：`sector_region` 现在严格限定 `0 < theta <= pi` 的小扇形/半圆，要求圆心、两端点、圆、半径段和题面角一致。圆心角以 exact radian 标签渲染并进入 public givens；面积、弧长和总周长分别通过 `sector_radius_angle_area`、`sector_radius_angle_arc_length`、`sector_radius_angle_perimeter` 从公开半径与角度推导，四类 residual 对坐标端点真值做 admission。新增 `sector_arc_length`、`sector_perimeter`，large-500 扩展为 21 个 2D families；旧 pilot sector 同步迁移到正式公开量契约。
- Circular-segment supervision：新增严格 `circular_segment_region`，边界必须由同一圆上的 arc 与 chord 构成，并引用对应 sector 与圆心弦三角形。polygon `area_formula` 现已进入 state/construction/incremental 正式引用契约；弦三角形通过 `chord_triangle_radius_angle` 从公开半径和圆心角推导，圆弓形再通过 `circular_segment_difference` 执行 sector-minus-triangle，多级 residual 与独立坐标面积共同 admission。SVG/PNG 均渲染真实弧弦阴影，large-500 扩展为 22 个 2D families。
- Circular-segment perimeter supervision：同一 `circular_segment_region` 现要求独立 `perimeter_formula`。弦长不作为公开量泄漏，而由 `chord_radius_angle_length=2*r*sin(theta/2)` 从公开半径和圆心角推导并以端点距离 residual admission；最终 `circular_segment_arc_chord_perimeter` 只依赖已接纳的弧长与弦长。新增独立周长任务族，large-500 扩展为 23 个 2D families。
- Lens supervision：`lens_region` 先以等圆 contract 建立双弧、交点和公开量闭环，随后扩展到严格两交点不等半径子集。`circle_lens_area` 公开两条半径和圆心距，依次推导两个圆心投影、公共半弦、两侧圆心角、两个扇形面积和两个三角形面积，并以独立坐标闭式面积及 `lens_two_radius_area_check` admission。SVG/PNG 复用双弧闭合路径；large-500 扩展为 25 个 2D families。
- Lens grid audit：不等圆透镜改用显式有效 `(r1, r2, d)` case 网格；只有两圆真交且两侧公共边界均为劣弧的组合进入内置/preset 网格，避免把运行时退化或大弧拓扑留给批量补齐逻辑。
- Lens perimeter：等圆和不等圆透镜新增独立 perimeter formula、公开半径/圆心距推导、坐标弧长校验和双弧复用渲染；`atan2`/`acos` 等价 residual 仅在高精度 `nsimplify` 认证后规范为零。
- Area-ratio supervision：`ratio` target 新增严格 `ratio_kind=area` 引用契约，从两个已 admission 的区域面积事实计算无量纲比值；新增三角形垂足分区面积比任务，并在 scene 中用两种填充区分比较区域。
- Parser v0：synthetic parser closed loop、admission debug、JSONL CLI、进度输出；真实 Parser 统一 proposed-record contract。
- Benchmark：Parser component evaluator 扩展和按 parser/task/difficulty/diagnostic axis 的 `ParserBenchmarkSlicer`。
- Legacy isolation timeout：子进程 spawn/import 启动开销不再计入单任务执行预算；worker ready 后才开始计时，快任务不会被误杀，真实超时任务仍会硬终止并进入 typed failure/metrics。

## 验证证据

使用环境：`/data/miniconda3/envs/geosym/bin/python`。

```text
focused parser/schema tests: 34 passed
focused relation/solver tests: 26 passed
full repository tests: 143 passed（严格 partition、阴影差集、面积比和 target canonicalization 后）
full repository tests: 147 passed（边界缺口差集、凹形任务推导和隐藏点视觉契约后）
full repository tests: 152 passed（严格多孔区域、双孔任务族及依赖有序反序列化后）
full repository tests: 154 passed（矩形语义周长、多孔边界周长及独立 perimeter 任务后）
full repository tests: 157 passed（圆环公开半径面积/总周长、真透明 SVG/PNG 内孔及层级回归后）
full repository tests: 162 passed（扇形公开半径/圆心角、面积/弧长/总周长 admission 和精确角标后）
full repository tests: 165 passed（圆弓形面积、弦三角形推导和真实弧弦 SVG/PNG 阴影后）
full repository tests: 168 passed（圆弓形弦长公开量推导、弧加弦周长和独立 perimeter family 后）
full repository tests: 170 passed（跨后端紧凑视觉签名 API 与固定 fixture 后）
full repository tests: 171 passed（sector/circular-segment/double-hole/annulus 代表性视觉矩阵后）
full repository tests: 180 passed（严格 Arc sweep、三类弧交点、curve facts 和圆周角退化修复后）
full repository tests: 183 passed（严格等圆透镜区域、公开量多步面积推导和双弧视觉回归后）
full repository tests: 187 passed（不等圆透镜、双半径公开量推导、劣弧独立验证和完整网格审计后）
full repository tests: 192 passed（等圆/不等圆透镜 perimeter 规则、公开量周长推导和视觉回归后）
full repository tests: 193 passed（TraceVerifier 独立 lens perimeter admission 与 mutation rejection 后）
focused synthesis tests: 20 passed（candidate diagnostics 增强后）
focused state+synthesis tests: 62 passed（derivation/task/split contract 增强后）
focused state+synthesis tests: 68 passed（公开量面积推导和错误公式拒绝后）
focused state+synthesis+rendering tests: 101 passed（严格 partition、阴影差集、面积比和区域对比渲染后）
focused state+synthesis+rendering tests: 104 passed（边界缺口差集、凹形公开量推导和 PNG 缺口回归后）
focused state+synthesis+rendering tests: 110 passed（严格多孔区域、双孔公开量推导、乱序 round-trip 和 PNG 双孔回归后）
focused state+synthesis tests: 84 passed（矩形语义周长、双孔边界周长和错误对角线引用拒绝后）
focused annulus synthesis/rendering tests: 4 passed（圆环公式引用、语义 target、SVG even-odd 真孔和 PNG 环形 mask 后）
focused sector synthesis/rendering tests: 5 passed（扇形三种 target、错误角度/半径拒绝和 SVG/PNG 圆心角标注后）
focused circular-segment perimeter tests: 7 passed（含面积、周长、错误公式、preset 和同 scene 回归）
focused render regression tests: 3 passed（固定基线、几何变更敏感性和非法 viewport）
focused representative render matrix tests: 3 passed（含四个关键场景，未生成正式图片）
focused arc/schema/state/synthesis/rendering tests: 159 passed（严格 Arc、三类弧交点、回滚、relation verification 和合法扇形/圆弓形回归）
focused equal-circle lens tests: 4 passed（公开量推导、错误边界拒绝、双弧 SVG/PNG 渲染和 preset 覆盖）
focused schema/state/intersection/synthesis/rendering tests: 162 passed（透镜跨层契约回归）
focused lens/arc tests: 17 passed（等圆/不等圆透镜、错误公式、大弧拒绝和双弧渲染）
focused schema/state/intersection/synthesis/rendering tests: 166 passed（不等圆透镜跨层契约回归）
focused lens/perimeter/rendering tests: 20 passed（两类透镜周长 target、错误引用和 canonical 视觉矩阵）
focused schema/state/intersection/synthesis/rendering tests: 171 passed（透镜面积/周长完整跨层契约回归）
focused lens/verifier tests: 14 passed（透镜 perimeter 公式篡改、面积/周长推导和视觉矩阵）
focused schema/state/intersection/synthesis/rendering tests: 172 passed（独立 perimeter verifier 后）
circle lens grid audit: default 12 variants and large preset 22 variants all constructed and admitted with zero `lens_two_radius_area_check` residual
circle lens perimeter grid audit: default and large preset explicit cases reuse the same validated two-minor-arc geometry; canonical equal/unequal variants admit zero perimeter residual
focused state transition tests: 59 passed（简单凹多边形 boolean、幂等边界和 exact signed area 后）
focused state/schema/intersection tests: 98 passed（tangent、line/ray/circle/arc incidence、垂足和分点关系 exact admission 后）
focused rendering tests: 40 passed（复杂分式排版、标签跨线/遮点/遮 marker/遮孔洞检查及更新后的跨后端签名）
full synthesis tests: 48 passed（topology-aware split、冲突拒绝、resume 和 manifest group 审计后）
focused legacy compatibility tests: 8 passed（worker-ready timeout 计时协议和慢任务终止后）
full repository tests: 207 passed（五阶段最终验收）
git diff --check: passed
built-in pilot state validation: 20/20 passed
```

## 2026-08-20 五阶段强化完成情况

### 1. Geometry State Contract 与拓扑判定

- 状态 schema 升级到 `1.1`，正式支持 `value`、`proof`、`constraint` target；constraint 使用严格比较操作和 exact expectation。
- `IntersectionResult` 增加 contact dimension、contact kind 和 exactness；零方向与非正半径显式判为 `degenerate`。
- 多解交点支持相对参考点的 `nearest`/`farthest` 选择。
- 近似共线和近似相切只作为 `PredicateDiagnostic` 输出，不会改变 exact status 或 admission fact。

### 2. 确定性 2D 与 Region Boolean

- polygon ring 会删除闭合重复点、连续重复点和精确共线中间点，并统一方向与 canonical 起点。
- 共线重叠边先切分再做 exact ownership，避免把共享边直接当作 unsupported。
- boolean 输入从单一简单多边形扩展为 resolved ring sets，支持 holes 和 multi-components 的 union/intersection/difference。
- 输出 topology metadata 包含 component/hole 数量、Euler characteristic 和 ring vertex counts。
- 当前边界仍是 polygonal ring sets；一般曲线区域 boolean、自交或无法闭合的边界明确失败，不猜测结果。

### 3. Incremental Closure 与 Provenance

- 新增显式 `FactDependencyIndex`，维护 token -> facts 和 fact -> dependents。
- 新增、修改、删除、relation 和 target 变化均执行传递失效，并按显式 rule agenda 重算受影响 scope。
- report 保存 changed tokens、invalidated/recomputed facts、semantic scope、rule agenda 和前后 checksum。
- `verify_equivalence=True` 可将增量结果与 full solver checksum 做差分验证，发现偏差立即失败。

### 4. 多步骤任务、Proof/Constraint 与解法路径

- solution-path schema `1.0` 将目标 fact 的依赖子图导出为拓扑步骤，逐路径保存 provenance、checksum 和 integrity。
- requested solution rules 必须逐一解析为不同 root fact，缺失、重复、环或 unresolved dependency 均拒绝导出。
- `similarity_dual_proof` 对同一相似三角形目标提供 SSS 与 AA 两条独立路径；2026-08-20 起改为最小公开条件：SSS 依赖统一对应边比例，AA 依赖平行与 incidence facts，不再把六条绝对边长和坐标推断直角铺到图面。
- `circular_segment_area_ratio` 复用扇形面积、弦三角形、sector-minus-triangle 和 region area ratio 的完整 exact chain。
- `large-500` preset 已纳入上述两个 family，当前为 29 个 2D families；未执行大规模生成。

### 5. Difficulty 与失败诊断

- difficulty schema 升级到 `2.0`，分别记录 construction、target、semantic axes、derivation、solution paths 的特征与贡献。
- trace-only 分数作为采样预估；candidate admission 和 exporter 使用 closure 后的 derivation/solution paths 重算权威难度。
- record 保存完整 difficulty policy；manifest 保存 schema version、最低/最高/平均分。
- candidate rejection 保留兼容 reason，并新增稳定 code、stage、component、family 聚合。
- domain exception 可区分 invalid input、unknown reference、duplicate object、degeneracy、numerical instability、verification、unsupported construction 和 unexpected failure。

## Git 与验证

本轮工作只修改 `/data/Geosym127K/Geosym-World`，未修改 `/data/Geosym127K/GeoClaw`，未生成正式 artifacts，未运行大规模合成、真实数据批量解析、模型调用或训练。

按目的拆分的提交：

```text
ea52d9e feat: strengthen geometry state and synthesis foundation
8aead4f feat: formalize exact 2d intersection topology
279f0d6 feat: extend exact region ring-set booleans
a63e208 feat: make fact closure dependency incremental
4e1c47b feat: add proof and constraint target modes
1afcb50 feat: add validated multi-path geometry tasks
4934d49 feat: audit task difficulty and candidate failures
```

新增阶段验证：

```text
focused intersection+state tests: 95 passed
focused state+rendering tests: 113 passed
focused target/schema/export tests: 30 passed
full synthesis tests: 53 passed
full repository tests: 228 passed
git diff --check: passed
```

## 后续边界

1. 一般曲线区域的 union/intersection/difference 仍未进入 exact boolean contract。
2. 当前 alternative solution paths 是已推导 fact 子图，不等于完整自动定理证明或自然语言证明生成。
3. approximate topology 只覆盖共线和 line-circle tangency 诊断，尚未形成全面数值鲁棒策略。
4. 开放世界真实题图 Parser 和高级 3D 仍是后置研究任务，不应由 synthetic closed loop 的结果替代能力证据。
5. 下一轮应优先增加 proof/constraint family 多样性、曲线区域的确定性子集，以及 benchmark diagnosis 的分层统计。

## 2026-08-20 最小公开标注契约

- task contract 升级到 `1.4`，annotation plan `1.2` 保存实际使用的参数化标注 profile，并分别记录可见绝对量、可见关系 marker、可见关系数值和文字关系。
- 标注角色固定为：`public_given` 可进入题面；construction 只定义对象；derived 只进入推导；target 只描述待求或待证内容。
- 新增可回放 `add_relation` 操作和 exact `proportional_lengths` relation verification。
- `point_on_segment_ratio` 不再因为是构造关系就自动进入公开条件；没有图示或文字证据的关系保持内部状态。
- parallel relation 使用成对箭头 marker；`similarity_dual_proof` 禁止自动直角推断，图面只保留 A-E 点名和 BC/DE 平行标记。
- 关系数值由可扩展 formatter 注册表生成，当前覆盖 `proportional_lengths`、`equal_length` 和 `equal_angle`，不依赖具体 task family。
- 新增 `AnnotationPolicy(level, relation_markers)`：`points` 只保留可见点名，`givens` 再增加公开数值和关系公式；`relation_markers` 统一控制直角、平行、等长和角平分符号。
- 同一份 formal spec 可按不同 annotation profile 输出，不再由模板固定最终标注密度。
- SSS solution path 的根依赖公开 `proportional_lengths` fact；AA solution path 的根依赖公开 parallel 与 point-on-line facts。

本轮验证：

```text
focused annotation tests: 7 passed
full repository tests: 232 passed
git diff --check: passed
```

## 2026-09-04 合成质量与 Parser Baseline 增量

### 已完成

- 新增严格 `ParserInput` `1.0`：运行时只允许 `schema_version/id/image/question`，formal state、scene、facts、trace、derivation、target 和 answer 均不能进入 Parser 输入。
- 合成图片导出新增公开 `*.parser-input.jsonl` 与 evaluator-only `*.parser-reference.jsonl`。私有 reference 保存可观察对象投影和完整监督，并用 input/reference checksum 防止错配。
- 新增统一 enhancement preflight：候选操作先在临时 state 执行，精确拒绝重合点、共线重叠线段、重合构造线和构造异常；skip reason 汇总进入 manifest。
- midpoint、ratio point、垂足、连接线、平行线、垂线和角平分线均改为候选枚举后预检，不再只随机抽一个对象组合。
- 新增 task dependency graph `1.0`，记录每一步输入、输出、依赖、消费者、语义角色、有效深度和分支；pipeline 拒绝 dead enhancement operation。
- `connect_points` 与 `point_on_segment_ratio` 产生待消费 frontier，后续步骤必须引用；有其他可行操作时不能作为末步。固定 12 条三轮审计中 dead record 从 6/12 降至 0/12。
- 修复 `point_on_segment_ratio` 未进入 enhancer inventory；修复角平分线用两个不可化简 `atan` 相减造成的假非零 residual，改用 exact squared-cosine equality。
- 修复 `triangle_area_from_height` 模板：删除与目标无关的固定 midpoint；垂足落在 A/B 时复用端点作为高度引用，不创建重合 H。
- 新增 `CleanDiagramParserV0`：只读取图片与题目，分阶段输出 OpenCV 线/圆候选、拓扑、关系 hypothesis、视觉点 grounding 和 target type hypothesis；全部结果保持 proposed 并带 confidence/provenance。
- clean primitive detector 增加共线 Hough 碎片合并、圆周 edge-support 与画布边界过滤；单样本三角形端到端检查由 22 线/20 圆降到 4 线/0 圆。
- 新增可注入 OCR stage。当前环境未提供 OCR 依赖，默认保留 text region 并明确标为 unresolved，不伪造点名。
- 新增 Parser 扰动评测：clean、noise、blur、low contrast、rotate 2 degrees，输出对象/点名/关系/target 分项指标、诊断轴和相对 clean 的 score drop。

### 验证证据

```text
focused Parser contract/clean/CLI/perturbation tests: 11 passed
full repository tests: 251 passed
three-round structural audit: 12/12 valid, 0 rejected, 0 dead enhancement operations
preflight audit: 112 checked, 79 accepted, 33 degenerate operations skipped before candidate validation
git diff --check: passed
```

结构审计产物位于 `/data/Geosym127K/tmp/geosym_structural_audit_20260904_v2/`，属于临时验证结果，不进入 Git。

### 尚未完成

1. 尚未接入实际 OCR backend，点名与数值文本识别仍是 Parser v0 的主要缺口。
2. 当前视觉 primitive 基于 Hough/connected components，尚未完成线段去重、弧检测、marker 分类和 label-to-object grounding。
3. 当前 Parser evaluator 的对象匹配以数量和关系类型为主，尚未实现稳定的拓扑图匹配与坐标无关 ID 对齐。
4. 尚未运行 500 条正式严格计数验收；本轮只有 12 条、3 family、固定三轮的无图片结构审计。
5. 真实开放世界题图、复杂 OCR、VLM 融合和不确定 hypothesis admission 仍为后续阶段，不能用 synthetic clean 结果代替。

## 2026-09-09 Parser marker 真值与评测闭环

### 已完成

- 将几何 Parser 明确为 evidence -> perceptual organization -> topology -> semantic/task 的分层恢复问题；扩展调研已记录 30 项主要来源，见 `docs/reports/2026-09-09_几何Parser概念与路线扩展调研.md` 和对应 literature-search 内部目录。
- `VisualMarkerDetector` 已能从图片像素检测显式直角 L glyph，并绑定两条视觉 support line；`TypedDiagramParserV0` 输出独立 marker hypothesis 和依赖该 marker 的 perpendicular relation，抑制同一线对的弱角度残差重复候选。
- 新增 renderer-neutral `marker_geometry`，SVG、PNG、标签避障和 synthetic marker GT 共用同一套 angle/right-angle/tick/parallel path 几何，不再维护三份坐标实现。
- renderer marker metadata 现在保留 point、support、line、polygon 或 region 来源；synthetic typed reference 将每个可见 glyph 投影为带 tight pixel bbox、canonical subtype、confidence 和 provenance 的 `ParseObjectKind.MARKER`。
- canonical marker subtype 当前为 `right_angle`、`parallel`、`equal_length_tick` 和 `angle_arc`；每个单独绘制的 glyph 是一个 marker object，视觉对象与其支持的语义 relation 分开建模。
- `ParseGraphEvaluator` schema 升级到 `1.2`，增加独立 `markers` 指标和 `marker_mismatch` 诊断；同位置不同 marker subtype 不再误配。原五项 core macro score 与 `core_perfect_parsing` 定义保持不变，完整 Perfect Parsing 从八项扩展为九项。
- CLI metric/manifest envelope 升级到 `geosym.parse_graph_metric.v2` 和 `geosym.parse_graph_eval_manifest.v2`。
- synthetic typed Parser v0 reference 明确只接受 `render_dim=2`。3D 主记录和图片继续正常导出，但不生成不完整的 2D Parser sidecar；manifest 新增 `num_parser_benchmark_records` 和分类 skip reason，当前为 `unsupported_render_dim:3`。
- 更新 marker 相关视觉回归基线；受影响图已目视检查，且 `inspect_render_scene` 保持 valid。

### 验证证据

```text
focused marker/reference/evaluator/rendering tests: 70 passed
Parser regression subset: 147 passed
full repository tests: 394 passed
built-in typed-reference audit: 29/29 2D family first variants round-trip
marker truth in audit: angle_arc=7, parallel=4, right_angle=19
3D exclusion audit: tetrahedron_section -> unsupported_render_dim:3
git diff --check: passed
```

### 下一步

1. 为已实现的 parallel-arrow、equal-length tick、equal-angle arc 跨 glyph grouping 增加更多合成/扰动回归，并校准不同线宽、缩放和低对比度下的拒绝边界。
2. 接入可审计 OCR backend，分别评测 point-name transcription、text-object detection、label binding 和数学表达式识别；OCR 输出仍必须保持 hypothesis。
3. 增加 arc/curve primitive、tangent/contact、shading/region segmentation 的视觉候选和 topology verification。
4. 基于合成 scene 生成 counterfactual supervision，系统扰动 junction、attachment、marker subtype 和 label binding，而不是只做全图 blur/noise。
5. 建立经过独立复核和 adjudication 的真实题图最小集；在此之前 synthetic clean 分数不能作为真实 Parser 能力结论。

## 2026-09-10 Geoparsing / GDP-29K baseline 接入

### 已完成

- 完成公开 Parser 路线补充调研，确认 Geoparsing/GDP-29K 是当前最接近的统一 2D/3D image-to-formal-language baseline；其公开 schema 不包含局部 evidence、confidence/conflict、题干 target、region/shading、可执行坐标或运行时 admission。
- 新增 checksum-bound `GeoparsingPrediction`：绑定公开 ParserInput、图片字节、backend、model revision、prompt version、解码参数和 formal description，拒绝未知字段和 payload 篡改。
- 新增严格 `GeoparsingAdapter`，将公开的 points/lines/circles/planes/structure/semantics 转为现有 `ParseGraph`；支持顺序线、命名线、多字符点名唯一分词、圆上点、平面和基础 3D structure。
- 语义转换当前支持垂直及其显式交点、平行链、线段等长链、线段数值/符号长度、角等量链、角度值和数值弧度量；根式、分式、隐式乘法和混合分数进入安全 MathExpression AST。
- 上游形式语言没有 bbox/坐标，所有候选固定使用 `IMAGE_ONLY` evidence；缺失 confidence 使用 `unscored_external_sequence`，不生成伪概率。转换完成也只生成 proposed hypothesis，默认 admission 全部 abstain。
- unsupported、unknown reference 和 compact-name ambiguity 进入结构化 import diagnostics；严格模式失败，显式 partial 模式才保留可表示候选。
- `parse_reconstruct_v0` 新增 `--geoparsing-predictions` 和 `--allow-partial-geoparsing`；运行前检查完整 ID 集、ParserInput checksum 和图片 hash，manifest 使用 `geosym.parser_run_manifest.v4`。
- 将 Geoparsing 严格预测契约与形式语言编译器拆成独立模块，保持原公共导入不变，并补强 decoding/usage 的 mapping 校验。
- 新增 `ParseGraphEvidenceFusion`：只根据无冲突唯一点名、唯一 segment endpoint、ordered-points support 和 circle center 做对象映射；语义 relation/quantity/target 的引用可落到本地视觉对象，同签名本地候选直接复用。
- 融合不会创建新的 `BOUND` evidence，也不会提高 confidence 或改变外部候选的 `IMAGE_ONLY` 精度；未获得独立局部支持的模型语义继续在 admission 中 abstain，歧义点名保持 unresolved。
- 新增 `FusedParseGraphParser` 和 CLI `--fuse-geoparsing-visual`。该模式可同时使用 checksum-bound OCR 与 Geoparsing sidecar；通用 external ParseGraph 仍与二者互斥。run manifest 升级至 `geosym.parser_run_manifest.v4`，记录融合配置、双图 checksum 和逐候选 mapping decision。
- 新增 image-free `GeoparsingGrammarAuditor` 与 `audit_geoparsing_grammar_v0` CLI。工具读取 GDP-style JSONL 的六类 formal 字段，输出 complete rate、typed candidate 数、diagnostic code/field 和有限失败例；结果显式声明只衡量 adapter grammar coverage，不是图片 Parser accuracy。
- 形式语言编译器新增独立 `compile_geoparsing_formal_description` 入口，grammar audit 不需要伪造图片或绕过 ParserInput 校验；正式 prediction replay 仍由 `GeoparsingAdapter` 执行完整 checksum/image hash 边界。

### 验证证据

```text
focused Geoparsing/fusion/grammar-audit/OCR/hybrid-input tests: 29 passed
Parser regression subset: 184 passed
full repository tests: 461 passed in 99.75s
git diff --check: passed
```

- Hybrid batch 入口已正式迁入 `geosym_world.tools.parse_real_data_hybrid_v0`；每条数据先投影为 `ParserInput`，隐藏的 answer/solution/formal state 不进入 detector、脚本候选层、VLM 或 builder。视觉读取错误默认 fail-fast，只有显式 `--continue-on-error` 才写失败报告并继续；run manifest 记录输入字段、失败计数和进度配置。
- semantic staged/single prompt 已改为在无答案运行时明确声明 answer consistency 只属于 evaluator；新增回归覆盖三类 prompt。正式运行命令和产物位置同步到 `Geosym-World/docs/PARSER_PIPELINE.md`，旧 `temp/` 入口不再作为维护接口。
- `ParseGraphEvidenceFusion` 的 compatibility certificate 已进一步审计 marker relation 和 quantity label：检查 `right_angle`、`parallel`、`equal_length_tick`、`angle_arc` 类型，marker 依赖是否完整，support/marked-segment/ordered-angle 是否覆盖 grounded relation refs，以及视觉数量文本与语义表达式 AST 是否一致；该证书仍只提供审计信息，不提升 evidence precision 或 admission。
- typed parser 新增参数化 `line_name` grounding：靠近线段内部的非点名/非数学表达式文本才进入候选，保留多 support、重复名称和同一 support 多名称冲突；命名直线与视觉 segment 的 label binding 进入 ParseGraph 并按现有 evidence policy admission。
- typed parser 对线圆单点接触输出独立 `tangent` hypothesis；视觉切触仍带 `exact_relation_not_admitted`，只有独立证据才能进入精确关系。
- visual topology 对有正长度的近共线重叠输出 `line_overlaps`，typed parser 编译为 `linear_overlap` hypothesis，保留重叠长度和像素残差，不把重叠线静默合并成一条语义线。
- 题干 proof cue 在唯一显式关系可用时生成关系型 proof target，target 依赖关系候选；pixel-witness 重建会将其写成稳定的 `object_0|object_1` proof key。
- `linear_overlap` 作为对称拓扑关系纳入规范化与 ParseGraph evaluator；评测会同时统计其通用 relation F1 和 topology F1，不再漏报重合候选。
- detector 新增受约束的 open circular-arc contour 拟合：稳定圆拟合、角度跨度和完整圆排除均保留为像素证据；只有唯一 support circle 存在时才生成 typed ARC 对象，否则保持 unresolved evidence。
- 弧拓扑继续向后贯通：线-弧、圆-弧和弧-弧的有限交点按弧段跨度过滤，交点进入共享匿名点观察、线段 ordered_points 和 typed intersection relation；未 grounding 的弧不会生成正式关系。
- PixelTopologyVerifier 现在支持 grounded ARC 路径和弧交点局部残差审计，弧对象按圆心、端点、方向和 sweep 采样；ARC 也纳入 require_pixel_topology admission 门槛。
- 对“完整圆边缘被开口轮廓误识别为近似大弧”的情况增加抑制：已有高 edge-support 完整圆时过滤近大弧候选；缺少完整圆支撑时保留真实大弧候选，避免把圆对象重复为弧。
- marker detector 的 topology 排除点已覆盖弧端点、线-弧、圆-弧和弧-弧交点，避免曲线交点附近重复产生符号候选。
- VisualPrimitiveDetector 新增受约束的 filled point blob 候选：小圆点需有暗色盘面支持，避开文字框、已有端点/交点和圆心；候选作为 point observation 进入 clean/typed Parser，但不自动绑定点名或数学语义。
- 点候选现在记录与有限线段的投影、残差和局部笔画支持；typed parser 将其编译为 bounded `point_on_line` hypothesis，并参与 ordered point topology 与后续 face 分割。
- `PixelTopologyVerifier` 新增 `point_on_line` 审计：检查点到 support 的残差、有限线段边界和点附近像素支持；关系未通过时不会被 pixel-witness admission 接受。
- 新增可选 `TesseractLabelRecognizer`：按已有文字候选逐区域裁剪识别，只有正置信度结果才标为 resolved；`parse_reconstruct_v0 --tesseract-ocr` 已接入，且与 external/recorded OCR 及未启用 fusion 的 Geoparsing 输入互斥。
- ParseGraph source redraw 已支持 grounded ARC 对象和显式 arc/segment 混合边界；formal exact radius 不会被误当像素单位，缺少 `radius_px` 时只从已定位圆心和弧端点恢复审阅半径。
- typed parser 新增唯一 filled arc-chord 区域匹配：弧、有限弦和 fill contour 必须同时匹配；完整弦复用原 segment，延长 support 才创建边界子段，区域仍保持 `shading_status=unknown`。
- typed parser 新增唯一 filled arc-radii 区域匹配：两条不同有限半径、唯一共享交点、拟合圆心、弧和 fill contour 必须一致；多个半径 support 同分时明确 abstain，不靠顺序猜测。
- 弧方向现在显式跨坐标系转换：detector 的 image orientation 保存在 metadata 供像素审计，ParseGraph geometry 使用笛卡尔 orientation 供重建和重绘。

### 本轮验证

```text
point/arc focused tests: 2 passed
OCR/Parser/clean/topology/tools regression: 45 passed
Parser/topology/reconstruction/evaluator/rendering regression: 157 passed
Parser/topology/reconstruction focused regression: 99 passed
full repository tests: 461 passed in 99.75s
git diff --check: passed
```

### 尚未完成

1. 尚未下载或运行 GDP-4B，因此没有真实模型质量、速度、显存或覆盖率数据。
2. 尚未固定 GDP-29K revision 并做全量 grammar coverage audit；公开样例已显示 symbolic angle、链式等式和多种 3D notation，仍可能存在未文档化变体。
3. Geoparsing 引用已能保守映射到本地 primitive/OCR 图，并输出 candidate-level `compatibility_certificate` 审计方向、点圆残差、局部等价候选、引用可达性、marker support、点名 label binding、数量 label AST 和 target presence；证书仍不等同语义验证，也不会提升 evidence precision。数学表达式/数量的真实 OCR 识别、可校准置信度和基于证书的严格 admission policy 仍未完成。
4. 公开语言没有 region、hole、shading 和 question target，这些必须由 Geosym-World 自己的 Parser 分支补齐。
5. 真实图最小 benchmark 与人工独立 adjudication 仍未建立，不能把 synthetic 或 GDP 格式转换成功率称为真实 Parser accuracy。

## 2026-09-10 区域孔洞候选审计

### 已完成

- typed parser 继续保留视觉 fill 的 `interior_gaps` 轮廓，并在 evidence 与绑定 face metadata 中写入 `interior_gap_count`、`interior_gap_status` 和 `hole_semantics`。
- 未经独立闭合 `ParseRegion` 边界支持，gap 不会写入 `ParseRegion.holes`；这避免把文字遮挡、描边断裂或白色背景误判为数学孔洞。
- 当 gap 与唯一的内部闭合 planar face 在包含关系、轮廓和面积上都匹配时，外层 region 才建立 `holes` 链接；重复或近似候选保持 unresolved。
- `face_extraction` 记录 `num_interior_gap_candidates`、`num_hole_region_links`、`num_regions_with_unresolved_gaps` 和 `hole_policy`；存在未决 gap 时 `fill_grounding` 阶段明确返回 `partial`。
- `ParseGraphEvaluator` 按 region 依赖顺序先匹配 hole/component，再匹配父 region，避免嵌套区域被错误计为漏检。

### 验证证据

```text
region/parser/evaluator focused tests: 59 passed
```

### 尚未完成

- 尚未从曲线边界、hatching 或真实题图语义中构造可 admission 的 hole `ParseRegion`。
- 尚未建立真实图阴影、孔洞、hatching 和 curved-region 的人工 adjudication benchmark。

## 2026-09-10 阴影语义假设与目标依赖

### 已完成

- typed parser 在题干明确要求阴影面积且只有一个 `fill_status=observed` 区域时，生成独立的 `shaded_region(region)` 语义假设；不再只把图面填充直接藏在 target metadata 中。
- `shadow_area` target 显式依赖该关系，关系本身标记 `semantic_shading_unverified`。默认 admission 对关系返回 `semantic_shading_unverified`，target 随后因依赖未接纳而 abstain，避免把配色误当数学事实。
- 同时存在多个填充区域时不按 detector 顺序猜测，保持 target unresolved；未解析目标的 admission 诊断优先返回 `target_object_unresolved`。
- typed parser 版本更新为 `0.13`，ParseGraph 新增受约束的 `shaded_region` 候选谓词。

### 验证证据

```text
typed parser/admission/contract focused tests: 61 passed
Parser/reconstruction/evaluator regression: 132 passed
full repository tests: 462 passed in 98.87s
git diff --check: passed
```

### 尚未完成

- 尚无能够独立验证并接纳阴影语义的 hatching、图例、自然语言指代消歧或人工 adjudication 证据。
- 复杂多组件阴影、差集/并集区域和真实题图精度仍待建立。

## 2026-09-10 Clean 阴影复合证据 Admission

### 已完成

- `ParseAdmissionPolicy` 新增默认关闭的 `permit_unique_fill_shading`，不改变严格策略的既有行为。
- 启用时，Verifier 不信任 Parser 的单个 metadata 标志，而是重新核验 `IMAGE_TEXT` 来源、唯一 region 参数、全图唯一 observed-fill region、参数身份、显式中英文阴影面积文本，以及指向同一 region 的 `BOUND` fill evidence。
- 证书通过后分别接纳 `shaded_region` 与依赖它的 `shadow_area` target；多填充或篡改后的 ParseGraph 明确返回 `unique_fill_shading_verification_failed`，不会靠 detector 顺序猜测。
- `parse_reconstruct_v0` 新增 `--permit-unique-fill-shading`；manifest 升级为 `geosym.parser_run_manifest.v5` 并记录完整 admission policy。

### 验证证据

```text
admission/typed-parser/CLI focused tests: 64 passed
Parser/reconstruction/tool regression: 159 passed
full repository tests: 462 passed in 99.14s
CLI help and policy serialization audit: passed
git diff --check: passed
```

### 尚未完成

- 该 policy 只适用于明确题干和唯一彩色填充的 clean subset，不代表 hatching、图例、多组件阴影或开放域语义已经解决。
- 尚未在独立真实标注集上测量该 policy 的 precision、coverage 和校准误差。

## 2026-09-10 SSS 三角形 Exact Gauge

### 已完成

- `ParseGraphReconstructor` 新增受限 `sss_triangle_v1` planner：只处理独立、闭合、三点三边且每边具有一致有限数值长度的 SSS 分量，并排除已有 exact coordinate 或其他 point construction 的候选。
- planner 以确定性 base support 固定平移/旋转 gauge，使用精确 SymPy 公式恢复第三点；两个镜像解必须通过 `first`/`second` runtime branch 明确选择，exact mode 不读取像素猜分支。
- `enumerate_branches()` 现可枚举 SSS 镜像解；每个分支独立执行 trace verification，点坐标标为 `realization_fact`，三条长度继续保留为 `source_assertion`。
- 非正边长、退化或违反三角不等式的 SSS 条件进入 `reconstruction.condition_conflict`；无法判定的代数符号不进入该 planner。
- reconstruction analysis 会把由构造或 gauge 得到的完整坐标准确标为 `constructed_exact_coordinates`，不再误报为输入显式坐标。

### 验证证据

```text
reconstruction focused tests: 55 passed
Parser/reconstruction integration tests: 124 passed
full repository tests: 465 passed in 99.19s
git diff --check: passed
```

### 尚未完成

- 尚未推广到非 2-tree distance graph、角度/平行混合约束或冗余约束的全局一致性求解。
- finite branch set 的 semantic redraw 已由后续批处理实现；branch-specific resynthesis 尚未提供。

## 2026-09-10 区域孔洞候选审计

### 已完成

- typed parser 继续保留视觉 fill 的 `interior_gaps` 轮廓，并在 evidence 与绑定 face metadata 中写入 `interior_gap_count`、`interior_gap_status` 和 `hole_semantics`。
- 未经独立闭合 `ParseRegion` 边界支持，gap 不会写入 `ParseRegion.holes`；这避免把文字遮挡、描边断裂或白色背景误判为数学孔洞。
- `face_extraction` 记录 `num_interior_gap_candidates`、`num_regions_with_unresolved_gaps` 和 `hole_policy`；存在未决 gap 时 `fill_grounding` 阶段明确返回 `partial`。

### 验证证据

```text
region/parser contract tests: 47 passed
```

### 尚未完成

- 尚未从独立闭合线框或区域布尔关系构造可 admission 的 hole `ParseRegion`。
- 尚未建立真实图阴影、孔洞、hatching 和 curved-region 的人工 adjudication benchmark。

## 2026-09-10 有限 Exact Branch Set 批量导出

### 已完成

- `parse_reconstruct_v0` 新增 `--enumerate-branches` 和 `--max-branches`，只允许 exact reconstruction，并在分支数超限时硬失败。
- 每条结果保留未选分支的主 `result.reconstruction`，另写 checksummed `branch_set`；每个分支独立保存 branch choice、trace、state、source/realization facts、verification、analysis 和 checksum，不按列表顺序默认选择一个解。
- manifest 升级为 `geosym.parser_run_manifest.v6`，记录分支配置，并汇总 branch set 数、总分支数、多分支记录数和分支 reconstruction status。
- branch enumeration 与 `--resynthesis-seeds` 明确互斥；在没有证据选择分支前，不生成看似唯一的再合成结果。

### 当前验证

```text
external prediction / CLI / OCR / Geoparsing focused tests: 23 passed
reconstruction / external prediction / CLI focused tests: 63 passed
full repository tests: 467 passed in 100.27s
CLI help audit: passed
git diff --check: passed
```

### 尚未完成

- branch-specific semantic redraw 已由后续批处理实现；resynthesis 尚未提供。
- 一般非线性约束、非 2-tree distance graph、连续分支和高阶有限分支仍未支持。

## 2026-09-10 共享边 2-tree 距离图 Exact Reconstruction

### 已完成

- 将原先硬编码为三点分量的 SSS planner 泛化为严格 2-tree distance planner：连通分量必须具有 `2n-3` 条唯一已知长度边，并能反复消去“度为 2 且两个邻点相连”的顶点。
- planner 确定性选择剩余基三角形和 base segment，固定平移/旋转 gauge；随后按消元逆序，通过两圆精确 trilateration 构造共享边三角网中的每个新点。
- 每个基三角形 apex 和后续 trilateration 点均保留 `first`/`second` 镜像分支；普通 reconstruct 不读像素选边，branch enumeration 对四点双三角形输出 4 个独立 exact realization。
- 每个 realization 点记录 `triangulated_distance_graph_v1`、直接 support points、三条 support segment IDs、对应 quantity IDs 和 runtime branch provenance；源长度仍是 `source_assertion`，坐标仍是 `realization_fact`。
- 局部边长非正、退化或违反三角不等式时拒绝整个 distance gauge；无 spanning triangular scaffold 和连续自由度图仍保持 underconstrained/unsupported，不猜坐标。

### 当前验证

```text
SSS / branch / triangulated-distance focused tests: 10 passed
reconstruction / external prediction / CLI integration tests: 66 passed
full repository tests: 470 passed in 101.18s
py_compile: passed
git diff --check: passed
```

### 尚未完成

- 尚未支持无 spanning triangular scaffold 的一般刚性 distance graph、角度/平行混合约束和连续参数族。
- branch-specific semantic redraw 已由后续批处理实现；resynthesis 仍未形成批量产物。

## 2026-09-10 冗余长度一致性与 Branch Pruning

### 已完成

- distance planner 允许连通图包含超过 `2n-3` 的已知长度边，但必须先找到确定性的 spanning triangular scaffold；额外边不参与坐标选择，只作为独立约束证书。
- 每个候选 realization 完成后，对所有已接纳长度执行 exact squared-distance residual；非零残差标记对应 segment 的 `reconstruction.condition_conflict`，不会被浮点容差吞掉。
- `ReconstructionBranchSet` 升级到 `1.1`：`branches` 只包含 exact realization，所有被裁候选完整保存在 `rejected_branches`；同时记录搜索候选数、接纳数、拒绝数和候选状态分布。
- K4 冗余长度案例从 4 个局部镜像组合中保留 2 个 exact 分支，另 2 个以冲突 segment、trace、state、branch choices 和 checksum 完整留档。
- Parser run manifest 升级到 `geosym.parser_run_manifest.v7`，分别汇总候选分支、exact 分支、拒绝分支、状态和 rejection code；`max_branches` 明确定义为 pruning 前搜索上限。

### 当前验证

```text
distance / SSS / branch focused tests: 15 passed
redundant-distance / external CLI focused tests: 22 passed
reconstruction / CLI / OCR / Geoparsing integration tests: 84 passed
full repository tests: 473 passed in 101.72s
py_compile: passed
git diff --check: passed
```

### 尚未完成

- 尚未支持没有三角 scaffold 的一般刚性 distance graph。
- 尚未把 angle、parallel、perpendicular 与 distance 放入统一代数约束求解。
- branch-specific semantic redraw 已由后续批处理实现；resynthesis 仍未形成批量产物。

## 2026-09-10 Branch-Specific Semantic Redraw

### 已完成

- 新增 `BranchRedrawResult` 契约；每个结果绑定 exact reconstruction checksum、显式 branch choices、semantic `RedrawResult` 和自身 checksum。
- semantic redraw schema 升级为 `1.1`，scene ID 与 lineage 均绑定 reconstruction checksum；不同镜像 realization 不再共享 scene identity。
- `ParseRedrawPipeline.render_branches()` 只消费 `ReconstructionBranchSet.branches` 中已接纳的 exact realization；`rejected_branches` 不进入渲染。
- `parse_reconstruct_v0 --enumerate-branches` 自动生成 `branch_redraws`。启用 SVG/PNG 时，文件写入 `renders/semantic_redraw_branches/`，文件名包含记录 hash、branch index 和 branch checksum。
- Parser run manifest 升级到 `geosym.parser_run_manifest.v8`，新增 branch semantic redraw 数量和 render verification 状态；所有 branch 文件同时进入 artifact counts。

### 当前验证

```text
branch / distance / SSS focused tests: 23 passed
py_compile: passed
git diff --check: passed
```

### 尚未完成

- 尚未支持对每个 exact branch 执行带 seed 的 rigid/similarity resynthesis。
- 尚未完成更广参数变体、条件/答案同步改写及真实外部样例验收。
