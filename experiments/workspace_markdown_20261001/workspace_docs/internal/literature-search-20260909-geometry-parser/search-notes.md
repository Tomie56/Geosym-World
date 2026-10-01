# Search Notes

日期：2026-09-09。此次扩展检索承接上一版工程设计；复用了前轮已读技能、隐私与来源规则，补读 report-template。未检索私有代码或未发表摘要。

## 检索问题与关键词

- geometry diagram parsing task definition; GeoS text diagram interpretation
- geometry diagram parsing PGPSNet structural semantic clauses
- GOLD natural language geometry description
- geometry diagram parsing perception benchmark
- geometry autoformalization AutoGPS
- diagram inverse graphics geometry program
- Learning to Interpret and Reason about Geometry Problems
- geometry diagram not drawn to scale
- diagram parsing uncertainty geometry
- Diagram Understanding in Geometry Questions AAAI
- geometry perception GeoPQA
- geometry diagram parsing perfect parsing rate hierarchical formalization
- multimodal formalization visual scene graph Lean
- Euclidean geometry autoformalization theorem prover SMT
- multimodal theorem proving benchmark geometry

## 候选筛选

题图核心保留 12 个主条目，参数重建增加 5 个，形式化/证明入口增加 5 个，共 22 个主条目，见 papers.md/csv。此前记录的背景/近邻候选继续保留；这是工作条目计数，不是去重原图或下载论文数。

| 候选 | 状态与理由 |
| --- | --- |
| PGDP5K 数据集论文，arXiv 2205.09947 | 上轮已核验；本轮归并至 PGDPNet 数据讨论，不重复作主条目 |
| Towards Geometry Problem Solving in the Large Model Era，arXiv 2506.02690 | 读取 survey 的 parsing 分类用于检索导航；技术结论回到原论文 |
| PGPSNet-v2，arXiv 2407.07327 | 命中摘要，偏融合/推理/验证；未深入，不作性能主张 |
| GeoDANO，arXiv 2502.11360 | 上轮读摘要；偏视觉表示与解题，非本轮结构图核心 |
| Generating Qualitative Descriptions of Diagrams with a Transformer-Based Language Model，Diagrams 2024 | Springer 原始条目，Euler 图的定性描述属于邻近任务；不直接外推几何题 |
| Understanding Natural Language with Diagrams，AAAI 1990 作者页面 | 阅读图/文模型与指代说明，领域偏物理，不作几何 Parser baseline |
| Statistical Visual Language Models for Ink Parsing | 命中高校 PDF，可用于歧义语法背景；未进一步核验出版原始页 |
| From Symbolic Perception to Logical Deduction / ZhongkaoGeo | 命中 OpenReview PDF；版本和正式出处未充分确认，未纳入结论 |
| A Survey of Multimodal Mathematical Reasoning，ACL 2026 | 检索命中正式 PDF，未深入；不以综述摘要替代主文证据 |

“Learning to Interpret and Reason about Geometry Problems”的检索未获得本轮可直接核验的明确条目，不加入列表或编造引用。

## 阅读深度

- PGDPNet：任务定义 3.1、SGG 比较 3.2、数据标注 4.2、评测 6.1。
- GeoS：diagram score/high-confidence literals、不按比例图与求解器边界。
- PGPSNet：第 1 节的高阶关系边界、第 3.2 节的点序/clauses、第 6 节部分对照。
- GOLD：第 3.3 sym2geo/geo2geo 与匿名对象，第 3.4 自然语言转写。
- Inter-GPS：第 5 节 Parser，以及 GT/预测条件比较和失败分析。
- Geoparsing：正式 Findings PDF，第 5 节奖励公式，特别是 ground-truth precision；调用第 6 页截图工具辅助公式检查。
- AutoGPS：第 3.3 节、相关消融、附录 Figure 12 的忠实度失败案例。
- SD-GPS：第 3 节训练输入/反馈，实验消融和 solver coverage 限制。
- Geo-Code：arXiv 摘要和方法结构，未复核全部重绘数值。
- GeoPQA：正式 Findings 摘要、PDF 方法和任务示例。
- Geo-TinyLLaVA：延用已核验 workshop 页面和官方代码，本轮 PDF 被 OpenReview challenge 拦截。
- Geoparsing：补读层级 prompt、GDP-29K 标注流程、Sample Accuracy、Perfect Parsing Rate、3D 虚实线限制。
- AutoGPS：用 ICLR 2026 官方页面核验正式状态；补读 multimodal formalizer、DSR 和 Refine@5。
- GeoX：核验 ICLR 2025 官方页面，用于“统一形式化视觉语言服务解题”的边界。
- MMFormalizer：阅读 recursive grounding、scene graph 和 115 样例 benchmark 范围。
- Autoformalizing Euclidean Geometry：核验 ICML 2024 页面及语义等价/证明器补全边界。
- MATP-BENCH：阅读 benchmark 摘要和 1,056 个多模态定理范围，只作下游压力测试。

## 来源限制

最终来源为官方会议、作者代码和 arXiv 原文。搜索命中的 MDPI 项目按技能规则排除；ResearchGate、Scribd、第三方生成综述、Reddit 和社交帖子未作为结论证据。

OpenReview 浏览器验证不能绕过；相应条目改用已核验官方摘要并降低阅读置信度。论文是否被正式录用，只按核验结果标注，未把 arXiv comment 当录用证据。

## 关键修正

1. 上版对 Geoparsing 的“几何一致性奖励”概括过粗。其公式 5 是 GT 对齐的按类别 precision，已更正设计文档。
2. Geoparsing 附录 B 的三层质量流程并非双人从零独立标注：平面几何由人工修正 GPT-5 初稿，另一学生复核，最后由专家审计并要求一致；3D 才声明从零人工识别。该流程可作为低成本 assisted annotation 先例，但不能直接证明模型初稿没有锚定偏差。
3. Geoparsing 同时报告逐类别 P/R/F1、Sample Accuracy 和整图 Perfect Parsing Rate。后两项能揭示组件分数尚可但整图任一错误即失败的乘法效应，适合加入本项目真实 benchmark 汇总层。
4. 本项目首版可信 GT 因此继续采用 `model_assistance=false`、不同人员独立 ParseGraph、显式 disagreement 和裁决。未来若引入模型预标注，应建立独立 trust grade 和对照集，不能复用 `human_double_annotation_v1` 名义。
5. 明确 diagram parsing 与 multimodal problem formalization 两种协议；是否恢复 target 应随协议决定。
6. 直接 VLM 路线是成本较低的首个对照，不是 literature 支持的绝对最佳方法。
7. 来源引用的存在、局部定位和正确语义绑定分别记状态，避免要求模型一次生成过长且伪精确的输出。
8. solver/prover 反馈属于 refinement 或下游验证信号，不能覆盖 source assertion，也不能作为唯一忠实度指标。

## 检查状态

公开关键词、来源排除、主条目去重、年份/来源/类型、初筛质量及限制、引用定位、文件落盘均完成。未执行论文复现、下载数据/模型、许可证逐文件审计或完整数值一致性审计。

后续交接：先冻结纯图最小输出与 GT，比较通用 VLM、辅助 VLM 和可运行专用 Parser。重点检查点序、匿名交点、量值绑定、阴影边界和没有证据的精确关系；不以解题答案正确代替这些检查。

## 2026-09-10 第二轮扩展

新增查询族：general diagram parse graph、perceptual grouping/connectivity/discourse、visual-language ambiguity、line-drawing junction topology、raster-to-parametric CAD sketch、rendering self-supervision、structured diagram image-to-code benchmark。

新增主条目 P23-P30，重点不是扩大几何解题模型清单，而是核对不同领域对 Parser 中间表示的理解：

- BEATRIX：并行图文 blackboard，中间 hypothesis 与最终 problem model 分离；图输入不是 raster。
- AI2D/AI2D-RST：从单一 DPG 扩展为 grouping、connectivity、discourse 分层图。
- Visual Languages 1999：attachment、segmentation、occlusion 是 diagram parsing 的独立歧义类型。
- Pacific Graphics 2019：junction topology 应先于最终 curve vectorization 恢复。
- DAVINCI：BMVC 2024 官方来源已核验；单阶段联合预测不妨碍系统输出分层审计。
- PICASSO：rendering self-supervision 可训练 primitive 参数化，但不能验证 semantic binding。
- STRUDEL：structured diagram benchmark 将识别、量化、结构分析、图文关联和 image-to-code 分开评价。
- UniSketch：CAD primitive/constraint 图支持更丰富约束，但 design intent 不等于题图中的可见条件。

本轮来源继续限定为官方会议/期刊、作者页和 arXiv；搜索命中的 MDPI 条目未纳入。未下载模型、数据或执行复现。新增结论已经写入面向用户的概念调研报告第 9-11 节。

## 完整目标的补充调研

随后按真实/更广数据到可验证重建与再合成的完整目标，增加 5 个主条目 P13-P17，再增加 P18-P22 覆盖统一形式化视觉语言、递归 grounding、欧氏几何自动形式化和多模态定理证明。2026-09-10 补充 P23-P30；`papers.csv` 当前共 30 项。重建来源和工程库影响见 [reconstruction-supplement.md](reconstruction-supplement.md)；当前主开发计划为 `docs/reports/2026-09-09_可验证几何解析重建与再合成开发计划.md`。
