# 参数约束重建方向补充调研

日期：2026-09-09。目的：将题图 Parser 设计扩展到真实/更广数据的结构提取、可验证重建和再合成。未运行模型或下载大型数据。

## 检索与来源

使用公开关键词：image to constrained CAD sketch、parametric sketch inference、constraint preserving transformations、constraint generation design intent、mathematical diagram constraint layout，以及各已公开论文标题。未向搜索引擎提交用户未发表的项目摘要、私有代码或设计段落。

技术结论使用论文原文、作者项目和官方库文档；MDPI 排除。第三方聚合站、搜索摘要只作发现入口，不作最终技术证据。

本轮新增 5 项主条目，在先前 12 项的基础上扩展到 17 项；不是声称系统穷尽。以下评分沿用 1-5 初筛量表，表示当前阅读下的判断，不是论文正式评分或性能复现。尤其不把 CAD 指标横向换算为教材 Parser 指标。

## 主条目

| ID | 条目与正式状态 | 类型 | Insight / Completeness / Numeric | 阅读和适用边界 |
| --- | --- | --- | --- | --- |
| P13 | [SketchGraphs](https://arxiv.org/abs/2007.08506)，2020，官方引用为 ICML Workshop on Object-Oriented Learning | method + benchmark | 4 / 4 / 3 | 图元和设计约束图；已读摘要、官方仓库及数据文档。规模不等于可再发布权利 |
| P14 | [Vitruvion](https://arxiv.org/abs/2109.14124)，ICLR 2022，作者仓库确认 | pure method | 4 / 4 / 3 | 参数图元序列、引用约束、image-conditional 重建；读原文图像条件实验及噪声讨论。手绘草图不等于教材题图 |
| P15 | [DAVINCI](https://arxiv.org/abs/2410.22857)，2024 预印本，未另行确认正式 venue | method + benchmark | 4 / 4 / 3 | 读第 3 节图元/子引用/约束表示，第 4 节 CPT，第 5 节范围；未逐表复核全部结果 |
| P16 | [Aligning Constraint Generation with Design Intent in Parametric CAD](https://arxiv.org/abs/2504.13178)，ICCV 2025，作者官方发布说明确认 | pure method | 4 / 3 / 3 | 读摘要、引言、问题/方法结构及限制目录；完整训练和人工评价未复核。fully-constrained 不能直接作为数学题目标 |
| P17 | [Penrose](https://penrose.cs.cmu.edu/siggraph20)，SIGGRAPH 2020 | system/tool | 4 / 3 / 2 | 官方项目页与论文的语义/样式/布局设计；不是图片 Parser，对本项目价值是架构借鉴 |

所有新增评分置信度为中等或较低：未运行公开代码，不保证 checkpoint、依赖或数据下载当前可用；较低 completeness 阅读置信度不表示论文实际不完整。

SketchGraphs 的来源核验：[官方仓库](https://github.com/PrincetonLIPS/SketchGraphs)、[文档](https://princetonlips.github.io/SketchGraphs/)。Vitruvion 的 venue 和代码入口：[作者仓库](https://github.com/PrincetonLIPS/vitruvion)。CAD design alignment 的 venue 核验：[Autodesk Research](https://www.research.autodesk.com/blog/ai-alignment-in-cad-design-teaching-machines-to-understand-design-intent-in-autoconstrain/)。

## 工程资源，不计论文主条目

| 来源 | 已核验用途 | 不能据此声称 |
| --- | --- | --- |
| [SolveSpace 技术说明](https://solvespace.com/tech.pl) | 几何方程、数值求解及欠约束处理 | 求得 witness 即精确证明；库可无条件嵌入 |
| [SciPy least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html) | 局部非线性最小二乘候选后端 | 收敛证明全局唯一或所有约束精确成立 |
| [CGAL Boolean 手册](https://doc.cgal.org/latest/Boolean_set_operations_2/index.html) | 多边形/带孔、regularized Boolean、曲线 traits | 低维交点/共享边可由 regularized 面域结果完整保留 |
| [diffvg](https://cseweb.ucsd.edu/~tzli/diffvg/) | 矢量到栅格的可微优化背景 | 像素拟合即可获得正确几何语义 |

diffvg 不纳入首批实现：已有渲染器可用，先解决关系与标注忠实度，比添加可微像素优化更直接。CGAL、SolveSpace、SciPy 是候选，不是已安装依赖；采用前审计许可、版本和绑定。

## 近邻和访问限制

- 检索命中 SketchConcepts / Discovering Design Concepts for CAD Sketches，及 Parametric Primitive Analysis of CAD Sketches with Vision Transformer；未充分细读，不纳入主表性能/方法主张。
- CVF 的 design alignment HTML/PDF 本轮直接打开失败；改读 arXiv 原文，并用作者官方 ICCV 说明核验 venue。没有以非官方镜像作为核心证据。
- DAVINCI 原文有不同段落规模表述不一致的风险，本计划不复述数据总量；未来使用具体 release manifest。
- 版本和许可尚未逐文件审计；不承诺可直接把公开数据集完整复制进公开仓库。

## 对设计产生的实际改变

1. Parser 输出与 realization 分离：不完整关系图可保存，执行/渲染通过独立编译。
2. 把 source redraw、semantic redraw、resynthesis 定为不同交付，不能用重绘相似度替代原图忠实度。
3. 增加设计约束与实例偶然性质区分，防止参数增强固定错误关系。
4. Native import 与 image-only 分轨；扩大数据源不牺牲评测输入公平性。
5. 将区域接触点/共享边与 regularized 面域分开，避免面积为零的几何关系消失。
6. 明确 source_assertion、entailed_fact、realization_fact、generation_parameter 的消费权限；必须有运行时 enforcement，不能只添 metadata。

完整设计见 [主开发计划](../../reports/2026-09-09_可验证几何解析重建与再合成开发计划.md)。本轮没有证据支持“所有这些部件组合就是新方法”或“真实开放域 Parser 已可实现高精度”的结论。
