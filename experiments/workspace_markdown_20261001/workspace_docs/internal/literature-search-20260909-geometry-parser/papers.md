# Geometry Parser 扩展检索条目

日期：2026-09-09。模式：standard 的定向概念调研。用途：回答用户“别人怎样理解 Parser”，为工程设计补证据，不评估本项目投稿概率。

主报告：`docs/reports/2026-09-09_几何Parser概念与路线扩展调研.md`。

后续扩展：本页保留题图方向 P01-P12；参数约束重建方向 P13-P17 见 [补充条目](reconstruction-supplement.md)。形式化/证明入口为 P18-P22。2026-09-10 再补充通用 diagram parsing、视觉歧义、线稿拓扑和 CAD self-supervision P23-P30；`papers.csv` 已汇总 30 项，完整工程主计划为 `docs/reports/2026-09-09_可验证几何解析重建与再合成开发计划.md`。

## 核心条目

分数依次为 insight / completeness / experimental numeric evidence，1–5。它们是本轮阅读范围内的初筛判断，不是复现结果或论文排名；数值证据分衡量与所讨论主张的对应程度。未经全篇实验审计的条目使用低置信度并注明限制。

| ID | 工作 | 年份、来源、类型 | 分数 | 优先级与依据 |
| --- | --- | --- | --- | --- |
| P01 | [Diagram Understanding in Geometry Questions](https://ojs.aaai.org/index.php/AAAI/article/view/9146) | 2014，AAAI，method + benchmark | 4/3/3 | A；明确对象定位与图文指代；旧数据范围有限，中置信度 |
| P02 | [Solving Geometry Problems: Combining Text and Diagram Interpretation](https://aclanthology.org/D15-1171/) | 2015，EMNLP，pure method | 4/4/3 | A；图文命题评分、不按比例绘图边界；旧 SAT 范围，中置信度 |
| P03 | [Inter-GPS](https://aclanthology.org/2021.acl-long.528/) | 2021，ACL，method + benchmark | 4/4/4 | A；图文 formalization 与 GT 替换对照，利于错误隔离，中置信度 |
| P04 | [Plane Geometry Diagram Parsing](https://www.ijcai.org/proceedings/2022/228) | 2022，IJCAI，method + benchmark | 4/4/4 | A；任务定义、细粒度 GT、检测/关系/形式化分层指标，中置信度 |
| P05 | [A Multi-Modal Neural Geometric Solver with Textual Clauses Parsed from Diagram](https://www.ijcai.org/proceedings/2023/376) | 2023，IJCAI，method + benchmark | 4/4/3 | A；结构/语义 clauses 清晰，主要结果仍是解题指标，中置信度 |
| P06 | [GOLD](https://aclanthology.org/2024.findings-naacl.19/) | 2024，Findings NAACL，pure method | 4/4/3 | A；sym2geo/geo2geo 分工及匿名引用；阅读重点为方法，中置信度 |
| P07 | [Plane Geometry Diagram Formalization via Vision-Language Models](https://icml.cc/virtual/2025/52421) | 2025，ICML AI4Math Workshop，method + benchmark | 3/2/2 | B；完整点标注条件、官方合并流程；正文访问受限，低置信度 |
| P08 | [Geoparsing](https://aclanthology.org/2026.findings-acl.1494/) | 2026，Findings ACL，method + benchmark | 4/4/3 | A；统一形式输出和 GT 奖励；版本计数口径待核验，中置信度 |
| P09 | [AutoGPS](https://iclr.cc/virtual/2026/poster/10009666) | 2026，ICLR，pure method | 4/4/3 | A；预形式化、对齐、有界 refinement、失败忠实度案例；中置信度 |
| P10 | [Verifiable Geometry Problem Solving: Solver-Driven Autoformalization and Theorem Proposing](https://arxiv.org/abs/2606.27926) | 2026，arXiv，pure method | 3/3/3 | B；solver-driven formalization，训练答案与执行反馈；未复现，低置信度 |
| P11 | [Geo-Code](https://arxiv.org/abs/2602.07749) | 2026，arXiv，method + benchmark | 3/3/2 | B；逆向编程边界；重绘指标不能直接外推 Parser 忠实度，低置信度 |
| P12 | [GeoPQA](https://aclanthology.org/2025.findings-emnlp.1400/) | 2025，Findings EMNLP，method + benchmark | 4/4/3 | A；感知问答诊断；完整图发现非其唯一主任务，中置信度 |
| P18 | [GeoX](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2722a0ccf6acfe3d144fdbb0dedd80b5-Abstract-Conference.html) | 2025，ICLR，pure method | 3/4/3 | B；统一视觉语言形式化服务解题；不以可审计 ParseGraph 为主要输出，中置信度 |
| P19 | [MMFormalizer](https://arxiv.org/abs/2601.03017) | 2026，arXiv technical report，method + benchmark | 3/3/2 | B；visual scene graph 到命题/Lean 的递归 grounding；仅 115 个跨域精选样例，低置信度 |
| P20 | [Autoformalizing Euclidean Geometry](https://icml.cc/virtual/2024/poster/33614) | 2024，ICML，pure method | 4/4/4 | A；形式语义等价及 prover/SMT 补全；不是 raster 图元解析，中置信度 |
| P21 | [MATP-BENCH](https://arxiv.org/abs/2506.06034) | 2025，arXiv，benchmark | 3/3/3 | B；多模态形式证明下游压力测试；不能代替 Parser 忠实度评价，低置信度 |
| P22 | [A Survey of Deep Learning for Geometry Problem Solving](https://aclanthology.org/2026.acl-long.1829/) | 2026，ACL，survey | 3/4/2 | B；element/relation/formalization/solving 分类导航；技术结论回到原论文，中置信度 |
| P23 | [Understanding Natural Language with Diagrams](https://www.cs.utexas.edu/~novak/aaai90.html) | 1990，AAAI，system/tool | 4/3/2 | B；图文并行 blackboard、hypothesis 与最终 problem model 分层；输入不是 raster，历史架构证据 |
| P24 | [A Diagram Is Worth a Dozen Images](https://arxiv.org/abs/1603.07396) | 2016，ECCV，method + benchmark | 4/4/4 | A；Diagram Parse Graph 与 syntactic/semantic/QA 分工；非几何专用，中置信度 |
| P25 | [AI2D-RST](https://link.springer.com/article/10.1007/s10579-020-09517-1) | 2020，Language Resources and Evaluation，pure benchmark | 5/4/N/A | A；拆分 grouping、connectivity、discourse 三层图；非几何专用，高相关概念证据 |
| P26 | [Ambiguity in Visual Language Theory and Its Role in Diagram Parsing](https://ieeexplore.ieee.org/document/795889/) | 1999，IEEE Visual Languages，theory/proof | 4/3/2 | B；attachment、segmentation、occlusion 歧义分类；假设 vector input，中置信度 |
| P27 | [Deep Line Drawing Vectorization via Line Subdivision and Topology Reconstruction](https://diglib.eg.org/items/447958d5-c239-4214-b341-1cf1ad5b6994) | 2019，Pacific Graphics/CGF，pure method | 4/4/4 | A；先全局/局部 topology 再曲线拟合，直指 junction；非语义 Parser，中置信度 |
| P28 | [PICASSO](https://openaccess.thecvf.com/content/WACV2025/papers/Karadeniz_PICASSO_A_Feed-Forward_Framework_for_Parametric_Inference_of_CAD_Sketches_WACV_2025_paper.pdf) | 2025，WACV，pure method | 4/4/4 | A；rendering self-supervision 学 primitive 参数；不能验证语义或 binding，中置信度 |
| P29 | [STRUDEL](https://aclanthology.org/2026.lrec-1.866/) | 2026，LREC，pure benchmark | 4/4/N/A | A；结构图任务拆成识别、量化、结构、关联和 image-to-code；跨域而非几何专用 |
| P30 | [UniSketch](https://ojs.aaai.org/index.php/AAAI/article/view/39525) | 2026，AAAI，method + benchmark | 4/4/3 | A；7 类 primitive、23 类约束及统一 image-to-sketch/constraint 任务；CAD design intent 与题图证据不同 |

## 归类与后续证据

| 类别 | 已有覆盖 | 本项目应验证的事项 |
| --- | --- | --- |
| primitive graph | P04、P06 | 匿名交点、虚线、点序和符号绑定覆盖；现成 checkpoint 能否直接适配 |
| 图文命题 | P01、P02、P03、P05 | 纯图/联合图文协议；题干增加的条件能否单独追踪 |
| VLM formalization | P07、P08 | 直接模型输出与定位能力分开评价；协议等价性与数据污染 |
| 执行驱动 formalization | P09、P10 | 形式化忠实度与 solver readiness 的差别；有限复查是否引入无依据条件 |
| 图到程序 | P11 | 重绘、拓扑、数学语义、原始构造轨迹非唯一性 |
| 感知诊断 | P12 | 局部问答与完整图匹配的差距 |
| 形式化/证明入口 | P18-P21 | 编译语义等价、证据依赖与 Parser 忠实度分开评价 |
| diagram 分层 | P23-P26、P29 | evidence、grouping、connectivity、semantic/discourse 不压成单层关系图 |
| junction topology | P27 | 交叉、遮挡、相切及 stroke fragmentation 单独标注和评分 |
| renderer/CAD 监督 | P13-P16、P28、P30 | 借用约束图和 rendering loss；不把 CAD design intent 当可见 source assertion |

上述普遍任务已有大量先例，不作为新颖性声明。阴影区域、不确定性和证据式 admission 的组合仍需更细的 schema/代码级比较，不能据本轮阅读称为空白。

## 数据候选

- PGDP5K：primitive 和关系；与 Geometry3K 有原图重叠，按来源图像分组。
- Geometry3K：图/文形式命题及求解；不把 solver 的 GT 输入结果当 Parser 成绩。
- PGPS9K：结构/语义 clauses 和程序；[官方使用条件](https://nlpr.ia.ac.cn/databases/CASIA-PGPS9K/)要求按协议使用，不能把可下载当成可任意再分发。
- GDP-29K：2D/3D formal descriptions；固定数据 revision 后审计真实文件数。
- GeoPQA：局部感知诊断，不能替代完整图真值。

## 必须保留的证据边界

- Geoparsing 第 5 节公式 5 对照 GT 计算按类别 precision；不是纯几何自洽奖励。
- Geoparsing 平面标注是 GPT-5.1 初稿后人工修正、复核和专家审计；不能表述为无模型辅助的双人独立 GT。
- Geoparsing 平面聚合分数 96.4，但整图 Perfect Parsing Rate 为 72.8%；发布 benchmark 时必须同时报告 whole-graph 指标。
- AutoGPS Figure 12 给出自洽且答案正确但形式化不忠实的例子。
- AutoGPS 已有 ICLR 2026 官方接收页面，旧记录中的“venue 未确认”已更正。
- SD-GPS 训练利用答案监督，文中的模型输入定义为图文，不能据此指控测试答案泄漏。
- Geo-Code 的 arXiv comment 写 ICML2026，本轮未找到并核验正式录用条目，按 arXiv 来源报告。
- P07 workshop 正文访问遇到浏览器验证，依据已核验 workshop 摘要与官方代码，只作有限结论。
- DAVINCI 已通过 BMVC 2024 官方 proceedings 核验，`papers.csv` 中来源由 arXiv 更正为 BMVC。
- AI2D-RST 明确指出单一 DPG 会混淆 grouping、connectivity 和 discourse；本项目不能只增加更多 relation predicate 而忽略层级。
- PICASSO 的 rendering self-supervision 只支持外观/参数恢复，不支持用重绘相似度接纳精确语义。
